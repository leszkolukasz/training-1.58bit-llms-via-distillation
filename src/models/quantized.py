from abc import ABC

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from src.layers import ImplementationType, QuantizationType, quantize_model
from src.loss import LossFunctionType, get_loss_function


class QuantizedModel(ABC, L.LightningModule):
    def __init__(
        self,
        base_model: nn.Module,
        quantization: QuantizationType,
        bitlinear_implementation: ImplementationType,
        layers_to_quantize: list[str],
        loss_function: LossFunctionType,
    ):
        super().__init__()
        self.model = quantize_model(
            base_model, quantization, bitlinear_implementation, layers_to_quantize
        )
        self.criterion = get_loss_function(loss_function)

        self.save_hyperparameters()

    def training_step(self, batch, _batch_idx):
        input_ids, target = batch
        output = self.model(input_ids, target)
        loss = self.criterion(output, target)

        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
