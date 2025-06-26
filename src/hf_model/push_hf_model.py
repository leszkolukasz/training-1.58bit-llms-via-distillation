import os

from decouple import Config, RepositoryEnv
from huggingface_hub import login
from transformers import AutoModel, AutoTokenizer

from src.hf_model.hf_class import HFQuantizedSmolModel

DOTENV_FILE = ".env.local"
config = Config(RepositoryEnv(DOTENV_FILE))
login(token=config("HF_TOKEN"))

HF_USERNAME = config("HF_USERNAME")
MODEL_PATH = "./benchmarks/models/d40998e3504b46c99bece2b8f3dbf174/checkpoints"

# Load
model = HFQuantizedSmolModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# and push
model.push_to_hub(
    f"{HF_USERNAME}/fbi_1b_no_shift_ce_test",
    commit_message="Pushing FBI 1B No Shift CE Test Model",
)
tokenizer.push_to_hub(f"{HF_USERNAME}/fbi_1b_no_shift_ce_test")
