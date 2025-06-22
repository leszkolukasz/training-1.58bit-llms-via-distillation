from abc import ABC
from typing import Callable

import lightning as L
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.constants import (EPSILON, MAX_SEQUENCE_LENGTH, QWEN_MODEL_ID,
                           SMOL_MODEL_ID)
from src.layers import ImplementationType, QuantizationType, quantize_model
from src.loss import LossFunctionType, get_loss_function

from .mixins import ChatMixin, LogArtifactMixin, Message


class QuantizedModel(ABC, LogArtifactMixin, L.LightningModule, ChatMixin):
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    quantized_layers: list[nn.Linear]

    def __init__(
        self,
        quantization: QuantizationType,
        bitlinear_implementation: ImplementationType,
        loss_function: LossFunctionType,
        model_id: str,
        layers_to_quantize: list[str],
    ):
        super().__init__()
        self.save_hyperparameters()

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

            self.teacher_model.config.use_cache = True
        else:
            self.teacher_model = None

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, padding_side="left"
        )
        self.model, self.quantized_layers = quantize_model(
            base_model, quantization, bitlinear_implementation, layers_to_quantize
        )
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

        return loss

    def backward(self, loss):
        super().backward(loss)

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
            "flip_flop_ratio",
            flip_flop_ratio,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )

        self.previous_weights = [
            layer.weight.data.clone().detach() for layer in self.quantized_layers
        ]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1
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
    ):
        super().__init__(
            quantization=quantization,
            bitlinear_implementation=bitlinear_implementation,
            loss_function=loss_function,
            model_id=QWEN_MODEL_ID,
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
    ):
        super().__init__(
            quantization=quantization,
            bitlinear_implementation=bitlinear_implementation,
            loss_function=loss_function,
            model_id=SMOL_MODEL_ID,
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


if __name__ == "__main__":
    m = AutoModelForCausalLM.from_pretrained(SMOL_MODEL_ID, torch_dtype=torch.bfloat16)
    print(m)
