import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.demos import Transformer

class TestModel(L.LightningModule):
    def __init__(self, vocab_size: int = 33278):
        super().__init__()
        self.save_hyperparameters()
        self.model = Transformer(vocab_size=vocab_size)

    def training_step(self, batch, _batch_idx):
        input_ids, target = batch
        output = self.model(input_ids, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        input_ids, target = batch
        output = self.model(input_ids, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, _batch_idx):
        input_ids, target = batch
        output = self.model(input_ids, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def on_fit_end(self):
        super().on_fit_end()

        model_path = (
            self.trainer.checkpoint_callback.last_model_path
            or self.trainer.checkpoint_callback.best_model_path
        )

        self.logger.experiment.log_artifact(self.logger.run_id, model_path)
