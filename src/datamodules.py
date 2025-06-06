from pathlib import Path

import lightning as L
from lightning.pytorch.demos import WikiText2
from torch.utils.data import DataLoader, random_split


class WikiText2DataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 8, data_dir: str = "./data"):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir

    def prepare_data(self):
        WikiText2(data_dir=Path(self.data_dir), download=True)

    def setup(self, stage: str):
        self.dataset = WikiText2(data_dir=Path(self.data_dir), download=False)

        n = len(self.dataset)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [n - 4000, 2000, 2000]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
