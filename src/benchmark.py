from transformers import AutoTokenizer, AutoModel, PreTrainedModel,  PretrainedConfig
from src.models.quantized import QuantizedSmolModel  
from src.models import  *
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.models.auto import modeling_auto, configuration_auto
from src.hf_class import BitConfig, HFQuantizedSmolModel
import os

path = os.getcwd()

ckpt_path = path + "/mlruns/303248108160348311/d40998e3504b46c99bece2b8f3dbf174/checkpoints/epoch=9-step=500.ckpt"
output_dir = path + "/models/d40998e3504b46c99bece2b8f3dbf174/checkpoints"

model = HFQuantizedSmolModel(BitConfig())
checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint["state_dict"], strict=False)
model.eval()

print("Saving model...")
model.save_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(SMOL_MODEL_ID)
tokenizer.save_pretrained(output_dir)
print("Model saved")





