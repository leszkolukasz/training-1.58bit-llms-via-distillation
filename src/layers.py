# pylint: disable=not-callable

from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch import nn

from src.constants import EPSILON

QuantizationType = Literal["1b", "1_58b", "1b_no_shift"]
ImplementationType = Literal["FBI", "OneBit", "BitNet"]
QuantizationFunctionType = Callable[
    [torch.Tensor], tuple[torch.Tensor, torch.Tensor | None]
]

BitLinearType = tuple[QuantizationType, ImplementationType]


def quantize_1b(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pre_sign = w - w.mean() + EPSILON
    return pre_sign + (pre_sign.sign() - pre_sign).detach(), 1.0


def quantize_1_58b(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = w.abs().mean() + EPSILON
    pre_round = w / scale
    return pre_round + (pre_round.round().clamp(-1, 1) - pre_round).detach(), scale


def quantize_1b_no_shift(w: torch.Tensor) -> tuple[torch.Tensor, None]:
    return w + (w.sign() - w).detach(), 1.0


QUANTIZATION_TYPE_TO_FUNCTION: dict[QuantizationType, QuantizationFunctionType] = {
    "1b": quantize_1b,
    "1_58b": quantize_1_58b,
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
        quantized_weights, _ = self.quantization_fun(self.weight)
        fbi_weights = self.alpha[None, :] * quantized_weights + self.beta[None, :]

        return F.linear(
            input, fbi_weights, self.bias
        )  # TODO: Do we want self.bias here? self.beta kind of acts like one but is not equivalent to an affine shift


class OneBitBitLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        quantization_fun: QuantizationFunctionType,
        bias: bool = True,
        # TODO: For distillation they use additionally MSE on token reps.
    ):
        super().__init__(in_features, out_features, bias)
        self.quantization_fun = quantization_fun
        self.layer_norm = nn.LayerNorm(out_features, eps=EPSILON)

        self.SVID_initialization()

    def SVID_initialization(self):
        # TODO: SVID approximation. Bad but it serves as some weight initialization
        abs_weight = torch.abs(self.weight)
        u, sig, v_T = torch.linalg.svd(abs_weight)
        self.g = nn.Parameter(u[:, 0] * torch.sqrt(sig[0]))
        self.h = nn.Parameter(v_T[0, :] * torch.sqrt(sig[0]))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        quantized_weights, _ = self.quantization_fun(self.weight)
        y = torch.mul(
            F.linear(torch.mul(input, self.h), quantized_weights, self.bias), self.g
        )
        return self.layer_norm(y)


class BitNetBitLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        quantization_fun: QuantizationFunctionType,
        activation_bits: int = 8,  # Like in the original BitNet paper
        bias: bool = True,
    ):
        super().__init__(in_features, out_features, bias)
        self.quantization_fun = quantization_fun
        self.activation_bits = activation_bits
        self.layer_norm = nn.LayerNorm(in_features, eps=EPSILON)

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        normalized_x = self.layer_norm(x)
        gamma = x.abs().max(dim=1, keepdim=True).values.clamp(min=EPSILON)
        Q_b = 2 ** (self.activation_bits - 1)
        return (
            torch.clamp(
                normalized_x * (Q_b / gamma),
                -Q_b + EPSILON,
                Q_b - EPSILON,
            ),
            gamma,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        quantized_weights, beta = self.quantization_fun(self.weight)
        quantized_activations, gamma = self.quantize_activation(input)
        scaling_factor = beta * gamma / (2 ** (self.activation_bits - 1))

        return (
            F.linear(quantized_activations, quantized_weights, self.bias)
            * scaling_factor
        )


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

    quantized_layers = []

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

            quantized_layers.append(new_module)

            new_module.weight.data = module.weight.data.clone()
            if bias:
                new_module.bias.data = module.bias.data.clone()

            parent_full_name = full_name.rsplit(".", 1)[0]
            parent = modules_dict[parent_full_name]

            setattr(parent, name, new_module)

    return model, quantized_layers

if __name__ == "__main__":
    
    w = torch.randn(10, 10)
    q_w, _ = quantize_1b(w)
    assert torch.allclose(q_w, -quantize_1b(-w)[0])
    
    w = torch.tensor([1.0, -2.0, 3.0])
    q_w, scale = quantize_1_58b(w)
    assert torch.allclose(q_w * scale, w.sign(), atol=1e-3)
    
    model = nn.Linear(10, 10)
    quantized, _ = quantize_model(model, "1b", "FBI", ["weight"])
    assert isinstance(quantized.weight, FBIBitLinear)