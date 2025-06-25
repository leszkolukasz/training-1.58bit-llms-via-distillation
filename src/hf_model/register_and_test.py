from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, CONFIG_MAPPING
from src.models.quantized import QuantizedSmolModel  
from src.hf_model.hf_class import HFQuantizedSmolModel, BitConfig42
from lm_eval.api.registry import register_model
from lm_eval import evaluator
from lm_eval.utils import setup_logging
from lm_eval.tasks import TaskManager
from decouple import Config, RepositoryEnv
import pickle as pkl
import subprocess

DOTENV_FILE = ".env.local"
config = Config(RepositoryEnv(DOTENV_FILE))
OUTPUT_DIR = "./benchmarks/results/d40998e3504b46c99bece2b8f3dbf174/checkpoints"

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
    HF_USERNAME = config("HF_USERNAME")
    setup_logging("DEBUG")
    task_manager = TaskManager()
    results = evaluator.simple_evaluate(
    model="hf",
    model_args=f"pretrained={HF_USERNAME}/{MODEL_NAME}",
    tasks=["hellaswag"],
    device="cuda:0",
    batch_size=4,
    task_manager=task_manager,
    )
    # cmd = [
    # "lm_eval",
    # "--model", "hf",
    # "--model_args", f'pretrained={HF_USERNAME}/{MODEL_NAME}',
    # "--tasks", "hellaswag",
    # "--device", "cuda:0",
    # "--batch_size", "4",
    # "--output_path", OUTPUT_DIR,
    # ]
    # result = subprocess.run(cmd, text=True, capture_output=True)
    # print(result.stdout)
    # print(result.stderr)
    # Results save
    with open(f"{OUTPUT_DIR}_results.pkl", "wb") as file:
        pkl.dump(results, file)
    