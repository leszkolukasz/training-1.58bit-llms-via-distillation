from typing import Literal

from torch import nn

QuantizationType = Literal["1b", "1.58b"]
ImplementationType = Literal["FBI", "OneBit", "BitNet"]

BitLinearType = tuple[QuantizationType, ImplementationType]


def quantize_model(
    model: nn.Module,
    quantization: QuantizationType,
    bitlinear_implementation: ImplementationType,
    layers_to_quantize: list[str],
) -> nn.Module:
    return model
