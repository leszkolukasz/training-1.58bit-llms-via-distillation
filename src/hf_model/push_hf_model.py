from huggingface_hub import login
from transformers import AutoTokenizer

from src.constants import (HF_CONVERTED_OUT_DIR, HF_MODEL_NAME, HF_TOKEN,
                           ORG_NAME)
from src.hf_model.hf_class import HFQuantizedSmolModel

login(token=HF_TOKEN)

# Load
model = HFQuantizedSmolModel.from_pretrained(
    HF_CONVERTED_OUT_DIR, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(HF_CONVERTED_OUT_DIR)

# and push
model.push_to_hub(f"{ORG_NAME}/{HF_MODEL_NAME}", safe_serialization=False)
tokenizer.push_to_hub(f"{ORG_NAME}/{HF_MODEL_NAME}")
