# pylint: disable=abstract-method

import os
import shutil

import torch
from transformers import (AutoTokenizer, PretrainedConfig, PreTrainedModel)

from src.loss import LossFunctionType
from src.layers import QuantizationType, ImplementationType
from src.models import *
from src.constants import HF_CONVERTED_OUT_DIR, HF_MODEL_CKPT, HF_BITLINEAR_IMPL, HF_QUANTIZATION, HF_LOSS_FUNCTION

# Script for converting a quantized model to a Hugging Face compatible format


class BitConfig42(PretrainedConfig):
    model_type = "quantized_net42"

    quantization: QuantizationType
    bitlinear_implementation: ImplementationType
    loss_function: LossFunctionType

    def __init__(
        self,
        quantization: QuantizationType = HF_QUANTIZATION,
        bitlinear_implementation: ImplementationType = HF_BITLINEAR_IMPL,
        loss_function: LossFunctionType = HF_LOSS_FUNCTION,
        **kwargs,
    ):
        self.quantization = quantization
        self.bitlinear_implementation = bitlinear_implementation
        self.loss_function = loss_function

        super().__init__(**kwargs)


class HFQuantizedSmolModel(PreTrainedModel):
    config_class = BitConfig42

    def __init__(self, config: BitConfig42, **kwargs):
        super().__init__(config, **kwargs)
        self.model = QuantizedSmolModel(
            quantization=config.quantization,
            bitlinear_implementation=config.bitlinear_implementation,
            loss_function=config.loss_function,
        ).model

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

if __name__ == "__main__":
    shutil.rmtree(HF_CONVERTED_OUT_DIR, ignore_errors=True)
    os.makedirs(HF_CONVERTED_OUT_DIR, exist_ok=True)

    config = BitConfig42(
        quantization=HF_QUANTIZATION,
        bitlinear_implementation=HF_BITLINEAR_IMPL,
        loss_function=HF_LOSS_FUNCTION,
    )
    model = HFQuantizedSmolModel(config)
    checkpoint = torch.load(HF_MODEL_CKPT)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()

    print("Saving model...")
    model.save_pretrained(HF_CONVERTED_OUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(SMOL_MODEL_ID)
    tokenizer.save_pretrained(HF_CONVERTED_OUT_DIR)
    print("Model saved")
