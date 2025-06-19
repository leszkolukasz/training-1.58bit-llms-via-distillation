from abc import ABC
from typing import Callable

import lightning as L
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.constants import MODEL_ID
from src.layers import ImplementationType, QuantizationType, quantize_model
from src.loss import LossFunctionType, get_loss_function

from .mixins import GeneratorMixin, Message
from src.constants import EPSILON


class QuantizedQwenModel(ABC, L.LightningModule, GeneratorMixin):
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    quantized_layers: list[nn.Linear]

    def __init__(
        self,
        quantization: QuantizationType,
        bitlinear_implementation: ImplementationType,
        loss_function: LossFunctionType,
    ):
        super().__init__()

        def load(compile=True):
            m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)

            if compile:
                m = torch.compile(m, mode="max-autotune", fullgraph=True)
            return m

        self.teacher_model, base_model = load(), load(False)

        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, use_fast=True, padding_side="left"
        )

        layers_to_quantize = ["o_proj", "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
        self.model, self.quantized_layers = quantize_model(
            base_model, quantization, bitlinear_implementation, layers_to_quantize
        )
        self.criterion = get_loss_function(loss_function)
        self.quantized_layers = layers_to_quantize

        self.save_hyperparameters()

    def training_step(self, batch, _batch_idx):
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]

        student_output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        with torch.no_grad():
            teacher_output = self.teacher_model(
                input_ids=input_ids, attention_mask=attention_mask
            )

        loss = self.criterion(teacher_output.logits, student_output.logits)

        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True
        )

        return loss
    
    def backward(self, loss):
        previous_weights = [
            layer.weight.data.clone().detach() for layer in self.quantized_layers
        ]

        super().backward(loss)

        flip_flop_sum = 0.
        flip_flop_count = 0

        with torch.no_grad():
            for layer, previous_weight in zip(self.quantized_layers, previous_weights):
                flip_flop_sum += (layer.weight.data.sign() - previous_weight.sign()).abs().sum().item() / 2.
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def generate(self, messages: list[Message], **kwargs) -> str:
        self.model.eval()

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        output = self.model.generate(
            **model_inputs,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.95,
            temperature=0.1,
            **kwargs,
        )

        return self.tokenizer.decode(
            output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
