from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, CONFIG_MAPPING
from src.models.quantized import QuantizedSmolModel  
from src.hf_model.hf_class import HFQuantizedSmolModel, BitConfig42
from lm_eval.api.registry import register_model
from lm_eval import evaluator
from decouple import Config, RepositoryEnv

DOTENV_FILE = ".env.local"
config = Config(RepositoryEnv(DOTENV_FILE))

# Model registration and evaluation

def register():
    AutoConfig.register("quantized_net42", BitConfig42)
    AutoModel.register(BitConfig42, HFQuantizedSmolModel)
    AutoModelForCausalLM.register(BitConfig42, HFQuantizedSmolModel)
    register_model("quantized_net42", HFQuantizedSmolModel)
    # Add to transformers/models/auto/configuration_auto.py
    CONFIG_MAPPING.update([("quantized_net42", "BitConfig42")])
    
if __name__ == "__main__":
    register()
    MODEL_NAME = "fbi_1b_no_shift_ce_test"
    evaluator.simple_evaluate(
    model="hf",
    model_args=f"pretrained={config("HF_NAME")}/{MODEL_NAME}",
    tasks=["hellaswag"],
    device="cuda:0",
    )