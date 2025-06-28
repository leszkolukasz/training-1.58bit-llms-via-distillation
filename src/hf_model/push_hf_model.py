from huggingface_hub import login
from transformers import AutoTokenizer

from src.hf_model.hf_class import HFQuantizedSmolModel
from src.constants import HF_TOKEN, ORG_NAME, HF_CONVERTED_OUT_DIR, HF_MODEL_NAME

login(token=HF_TOKEN)

# Load
model = HFQuantizedSmolModel.from_pretrained(HF_CONVERTED_OUT_DIR, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(HF_CONVERTED_OUT_DIR)

# and push
model.push_to_hub(f"{ORG_NAME}/{HF_MODEL_NAME}")
tokenizer.push_to_hub(f"{ORG_NAME}/{HF_MODEL_NAME}")
