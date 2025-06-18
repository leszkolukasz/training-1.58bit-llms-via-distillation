from abc import ABC

import lightning as L
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.constants import MODEL_ID
from src.layers import ImplementationType, QuantizationType, quantize_model
from src.loss import LossFunctionType, get_loss_function

from .mixins import GeneratorMixin, Message


class QuantizedQwenModel(ABC, L.LightningModule, GeneratorMixin):
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    criterion: any

    def __init__(
        self,
        quantization: QuantizationType,
        bitlinear_implementation: ImplementationType,
        loss_function: LossFunctionType,
    ):
        super().__init__()

        config = AutoConfig.from_pretrained(MODEL_ID)
        base_model = AutoModelForCausalLM.from_config(
            config, torch_dtype=torch.bfloat16
        )
        base_model = torch.compile(base_model, mode="max-autotune", fullgraph=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, use_fast=True, padding_side="left"
        )

        layers_to_quantize = []
        # self.model = quantize_model(
        #     base_model, quantization, bitlinear_implementation, layers_to_quantize
        # )
        self.model = base_model
        self.criterion = get_loss_function(loss_function)

        self.save_hyperparameters()

    def training_step(self, batch, _batch_idx):
        input_ids, target = batch
        output = self.model(input_ids, target)
        loss = self.criterion(output, target)

        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True
        )

        return loss

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
