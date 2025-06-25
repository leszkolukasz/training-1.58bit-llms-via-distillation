from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel,  PretrainedConfig, AutoModelForCausalLM
from src.models.quantized import QuantizedSmolModel  
from src.models import  *
import torch
import torch.nn as nn
import os
from transformers import PretrainedConfig
from transformers.models.auto import modeling_auto, configuration_auto
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from lm_eval.api.registry import register_model
import os


# Script for converting a quantized model to a Hugging Face compatible format

class BitConfig42(PretrainedConfig):
    model_type = "quantized_net42"
    
    def __init__(
        self,
        quantization="1b_no_shift",
        bitlinear_implementation="FBI",
        loss_function="CrossEntropy",
        **kwargs
    ):
        self.quantization = quantization
        self.bitlinear_implementation = bitlinear_implementation
        self.loss_function = loss_function
        super().__init__(**kwargs)

class HFQuantizedSmolModel(PreTrainedModel):
    config_class = BitConfig42

    def __init__(self, config):
        super().__init__(config)
        self.quantized_model = QuantizedSmolModel(
            quantization=config.quantization,
            bitlinear_implementation=config.bitlinear_implementation,
            loss_function=config.loss_function,
        ).model 

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.quantized_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

# class HFQuantizedModelWrapper(HFLM):
#     def __init__(self, pretrained, device=None, batch_size=None, **kwargs):
#         config = AutoConfig.from_pretrained(pretrained)
#         self.device = device
#         self.batch_size = batch_size
#         self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
#         self.model = HFQuantizedSmolModel.from_pretrained(pretrained, config=config).to(device)

#     def model_call(self, *args, **kwargs):
#         return self.model(*args, **kwargs)

if __name__ == "__main__":
    ckpt_path = "./mlruns/303248108160348311/d40998e3504b46c99bece2b8f3dbf174/checkpoints/epoch=9-step=500.ckpt"
    output_dir = "./benchmarks/models/d40998e3504b46c99bece2b8f3dbf174/checkpoints"
    os.makedirs(output_dir, exist_ok=True)
    
    model = HFQuantizedSmolModel(BitConfig42())
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    
    print("Saving model...")
    model.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(SMOL_MODEL_ID)
    tokenizer.save_pretrained(output_dir)
    print("Model saved")
    
    
