from typing import Callable, Literal

import torch
from torch import nn
import torch.nn.functional as F

QuantizationType = Literal["1b", "1_58b"]
ImplementationType = Literal["FBI", "OneBit", "BitNet"]
QuantizationFunctionType = Callable[[torch.Tensor], torch.Tensor]

BitLinearType = tuple[QuantizationType, ImplementationType]

EPSILON = 1e-6


def quantize_1b(
    w: torch.Tensor,
) -> torch.Tensor:
    pre_sign = w - w.mean() + EPSILON
    return pre_sign + (pre_sign.sign() - pre_sign).detach()


def quantize_1_58b(
    w: torch.Tensor,
) -> torch.Tensor:
    scale = w.abs().mean() + EPSILON
    pre_round = w / scale
    return pre_round + (pre_round.round().clamp(-1, 1) - pre_round).detach(), scale


def quantize_1b_no_shift(w: torch.Tensor) -> torch.Tensor:
    return w + (w.sign() - w).detach() 


QUANTIZATION_TYPE_TO_FUNCTION: dict[QuantizationType, QuantizationFunctionType] = {
    "1b": quantize_1b,
    "1.58b": quantize_1_58b,
    "1b_no_shift": quantize_1b_no_shift,
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
        self.alpha = nn.Parameter(self.weight.mean(dim=0))
        self.beta = nn.Parameter(torch.abs(self.weight - self.alpha).mean(dim=0))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        quantized_weights = self.quantization_fun(self.weight)[0]
        fbi_weights = self.alpha[None, :] * quantized_weights + self.beta[None, :]
        return F.linear(input, fbi_weights, self.bias) # TODO: Do we want self.bias here? self.beta kind of acts like one but is not equivalent to an affine shift 

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
        activation_bits: int=8, # Like in the original BitNet paper
        bias: bool = True,
    ):
        super().__init__(in_features, out_features, bias)
        self.quantization_fun = quantization_fun
        self.activation_bits = activation_bits
        self.beta = nn.Parameter(self.weight.abs().mean())
        
    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        normalized_x = F.layer_norm(x, eps=EPSILON) # LayerNorm here
        gamma = x.abs().max(dim=1, keepdim=True).values.clamp(min=EPSILON)
        Q_b = 2 ** (self.activation_bits - 1)
        return torch.clamp(
            normalized_x * (Q_b / gamma), 
            -Q_b + EPSILON, Q_b - EPSILON, 
        ), gamma
        

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        quantized_weights, beta = self.quantization_fun(self.weight) 
        quantized_activations, gamma = self.quantize_activation(input)
        scaling_factor = beta * gamma / (2 ** (self.activation_bits -1))
        return F.linear(quantized_activations, quantized_weights, self.bias) * scaling_factor


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
