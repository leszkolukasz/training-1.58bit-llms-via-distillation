from typing import Callable, Literal

import torch
from torch import nn

QuantizationType = Literal["1b", "1.58b"]
ImplementationType = Literal["FBI", "OneBit", "BitNet"]
QuantizationFunctionType = Callable[[torch.Tensor], torch.Tensor]

BitLinearType = tuple[QuantizationType, ImplementationType]


def quantize_1b(
    x: torch.Tensor,
) -> torch.Tensor:
    return x


def quantize_1_58b(
    x: torch.Tensor,
) -> torch.Tensor:
    return x


QUANTIZATION_TYPE_TO_FUNCTION: dict[QuantizationType, QuantizationFunctionType] = {
    "1b": quantize_1b,
    "1.58b": quantize_1_58b,
}


class FBIBitLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        quantization_fun: QuantizationFunctionType,
        bias: bool = True,
    ):
        super().__init__(in_features, out_features, bias)
        self.quantization_fun = quantization_fun

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)


class OneBitBitLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        quantization_fun: QuantizationFunctionType,
        bias: bool = True,
    ):
        super().__init__(in_features, out_features, bias)
        self.quantization_fun = quantization_fun

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)


class BitNetBitLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        quantization_fun: QuantizationFunctionType,
        bias: bool = True,
    ):
        super().__init__(in_features, out_features, bias)
        self.quantization_fun = quantization_fun

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)


IMPLEMENTATION_TYPE_TO_CLASS: dict[ImplementationType, type] = {
    "FBI": FBIBitLinear,
    "OneBit": OneBitBitLinear,
    "BitNet": BitNetBitLinear,
}


def build_bitlinear(
    in_features: int,
    out_features: int,
    quantization: QuantizationType,
    bitlinear_implementation: ImplementationType,
    bias: bool = True,
) -> nn.Module:
    return IMPLEMENTATION_TYPE_TO_CLASS[bitlinear_implementation](
        in_features, out_features, QUANTIZATION_TYPE_TO_FUNCTION[quantization], bias
    )


def quantize_model(
    model: nn.Module,
    quantization: QuantizationType,
    bitlinear_implementation: ImplementationType,
    layers_to_quantize: list[str],
) -> nn.Module:

    modules_dict: dict[str, nn.Module] = dict(model.named_modules())

    for full_name, module in model.named_modules():
        name = full_name.split(".")[-1]

        if isinstance(module, nn.Linear) and name in layers_to_quantize:
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None

            new_module = build_bitlinear(
                in_features,
                out_features,
                quantization,
                bitlinear_implementation,
                bias=bias,
            )

            # TODO: how to initialize weights?

            parent_full_name = full_name.rsplit(".", 1)[0]
            parent = modules_dict[parent_full_name]

            setattr(parent, name, new_module)

    return model
