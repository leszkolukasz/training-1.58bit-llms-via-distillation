from abc import ABC
from typing import Callable

import lightning as L
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.constants import (EPSILON, INITIAL_LR, LAYERS_ZD_TOP_P,
                           MAX_SEQUENCE_LENGTH, QWEN_MODEL_ID, SMOL_MODEL_ID)
from src.layers import ImplementationType, QuantizationType, quantize_model
from src.loss import LossFunctionType, get_loss_function
from src.utils import get_grad_norm

from .mixins import ChatMixin, LogArtifactMixin, Message


class QuantizedModel(ABC, LogArtifactMixin, L.LightningModule, ChatMixin):
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    quantized_layers: list[nn.Linear]
    initial_lr: float

    def __init__(
        self,
        quantization: QuantizationType,
        bitlinear_implementation: ImplementationType,
        loss_function: LossFunctionType,
        model_id: str,
        layers_to_quantize: list[str],
        lr: float = INITIAL_LR,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.initial_lr = lr

        def load(compile=True):
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float32
            )  # there are problems with bfloat16

            if compile:
                model = torch.compile(model, mode="max-autotune", fullgraph=True)
            return model

        base_model = load(compile=False)

        if loss_function != "CrossEntropyWithoutKD":
            self.teacher_model = load(compile=False)
            for param in self.teacher_model.parameters():
                param.requires_grad = False

            self.teacher_model.config.use_cache = False
        else:
            self.teacher_model = None

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, padding_side="left"
        )
        self.model, self.quantized_layers = quantize_model(
            base_model, quantization, bitlinear_implementation, layers_to_quantize
        )

        self.model.gradient_checkpointing_enable()

        # TODO: do wee need this?
        # for full_name, module in self.model.named_modules():
        #     name = full_name.split(".")[-1]
        #     if name not in layers_to_quantize:
        #         for param in module.parameters():
        #             nn.init.normal_(param)

        self.criterion = get_loss_function(loss_function)

        self.previous_weights = [
            layer.weight.data.clone().detach() for layer in self.quantized_layers
        ]

    def training_step(self, batch, _batch_idx):
        input_ids, attention_mask, labels = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )

        student_output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if self.teacher_model is not None:
            self.teacher_model.eval()
            with torch.no_grad():
                teacher_logits = self.teacher_model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
        else:
            teacher_logits = None

        loss = self.criterion(
            student_output.logits, teacher_logits=teacher_logits, labels=labels
        )

        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True
        )

        self.log("step", int(self.global_step), prog_bar=True, logger=False)

        return loss

    def on_before_optimizer_step(self, *_args):
        flip_flop_sum = 0.0
        flip_flop_count = 0

        with torch.no_grad():
            for layer, previous_weight in zip(
                self.quantized_layers, self.previous_weights
            ):
                previous_weight = previous_weight.to(layer.weight.data.device)
                flip_flop_sum += (
                    layer.weight.data.sign() - previous_weight.sign()
                ).abs().sum().item() / 2.0
                flip_flop_count += layer.weight.data.numel()

        flip_flop_ratio = flip_flop_sum / (flip_flop_count + EPSILON)

        self.log(
            "flip_flop",
            flip_flop_ratio,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )

        self.previous_weights = [
            layer.weight.data.clone().detach() for layer in self.quantized_layers
        ]

        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]

        self.log(
            "lr",
            lr,
            prog_bar=True,
            on_step=True,
            logger=True,
        )

        grad_norm = get_grad_norm(self.model)

        self.log(
            "grad_norm",
            grad_norm,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.initial_lr, betas=(0.9, 0.98), weight_decay=0.1
        )

        T_max = next(
            t
            for t in [
                self.trainer.estimated_stepping_batches,
                self.trainer.max_steps,
                10000,
            ]
            if t is not None and t > 0
        )

        print(f"T_max: {T_max}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=1e-7
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    @torch.inference_mode()
    def chat(self, messages: list[Message] = None, prompt: str = None, **kwargs) -> str:
        self.model.eval()

        if messages is not None:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        output = self.model.generate(
            **model_inputs,
            max_new_tokens=MAX_SEQUENCE_LENGTH,
            do_sample=True,
            top_p=0.95,
            temperature=0.1,
            **kwargs,
        )

        return self.tokenizer.decode(
            output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )


class QuantizedQwenModel(QuantizedModel):
    def __init__(
        self,
        quantization: QuantizationType,
        bitlinear_implementation: ImplementationType,
        loss_function: LossFunctionType,
        lr: float = INITIAL_LR,
    ):
        super().__init__(
            quantization=quantization,
            bitlinear_implementation=bitlinear_implementation,
            loss_function=loss_function,
            model_id=QWEN_MODEL_ID,
            lr=lr,
            layers_to_quantize=[
                "o_proj",
                "q_proj",
                "k_proj",
                "v_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )


class QuantizedSmolModel(QuantizedModel):
    def __init__(
        self,
        quantization: QuantizationType,
        bitlinear_implementation: ImplementationType,
        loss_function: LossFunctionType,
        lr: float = INITIAL_LR,
    ):
        # read from file
        layers_by_LIM = []

        with open("data/smol_layers_sorted_by_ZD.txt", "r", encoding="utf-8") as f:
            for line in f:
                layers_by_LIM.append(line.strip())

        layers_by_LIM = list(filter(lambda x: x not in ["lm_head"], layers_by_LIM))

        super().__init__(
            quantization=quantization,
            bitlinear_implementation=bitlinear_implementation,
            loss_function=loss_function,
            model_id=SMOL_MODEL_ID,
            lr=lr,
            layers_to_quantize=layers_by_LIM[
                : int(LAYERS_ZD_TOP_P * len(layers_by_LIM))
            ],
        )


if __name__ == "__main__":
    m = AutoModelForCausalLM.from_pretrained(SMOL_MODEL_ID, torch_dtype=torch.bfloat16)
    print(m)
