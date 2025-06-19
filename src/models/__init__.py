from typing import NamedTuple

from .example import *
from .mixins import *
from .quantized import *

Model = NamedTuple(
    "Model",
    [
        ("cls", type),
        ("repo_id", str),
    ],
)

QUANTIZED_MODELS = {
    "QuantizedQwenModel": Model(QuantizedQwenModel, QWEN_MODEL_ID),
    "QuantizedSmolModel": Model(QuantizedSmolModel, SMOL_MODEL_ID),
}
