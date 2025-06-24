from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel,  PretrainedConfig, AutoModelForCausalLM
from src.models.quantized import QuantizedSmolModel  
from src.models import  *
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.models.auto import modeling_auto, configuration_auto
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model

ckpt_path = "/root/studia/nlp/training-1.58bit-llms-via-distillation/mlruns/303248108160348311/d40998e3504b46c99bece2b8f3dbf174/checkpoints/epoch=9-step=500.ckpt"
output_dir = "/root/studia/nlp/training-1.58bit-llms-via-distillation/benchmarks/models/d40998e3504b46c99bece2b8f3dbf174/checkpoints"

# Saving model as a HF instance
class BitConfig(PretrainedConfig):
    model_type = "multibitbitnet42"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class HFQuantizedSmolModel(PreTrainedModel):
    config_class = BitConfig

    def __init__(self, config):
        super().__init__(config)
        self.quantized_model = QuantizedSmolModel(
            quantization="1b_no_shift",
            bitlinear_implementation="FBI",
            loss_function="CrossEntropy",
        ).model 

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.quantized_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

class HFQuantizedModelWrapper(HFLM):
    def __init__(self, pretrained, device=None, batch_size=None, **kwargs):
        config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.model = HFQuantizedSmolModel.from_pretrained(pretrained, config=config, trust_remote_code=True).to(device)

    def model_call(self, *args, **kwargs):
        return self.model(*args, **kwargs)

if __name__ == "__main__":
    AutoConfig.register("multibitbitnet42", BitConfig, exist_ok=True)
    AutoModel.register(BitConfig, HFQuantizedSmolModel, exist_ok=True)
    AutoModel.register(BitConfig, AutoTokenizer, exist_ok=True)
    AutoModelForCausalLM.register(BitConfig, HFQuantizedSmolModel, exist_ok=True)
    register_model("mutlibibitnet42", HFQuantizedModelWrapper)
    
    
