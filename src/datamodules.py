from pathlib import Path

import lightning as L
from datasets import load_dataset
from lightning.pytorch.demos import WikiText2
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from src.constants import AMBER_DATASET_PATH, BATCH_SIZE, MAX_SEQUENCE_LENGTH
from src.models import QUANTIZED_MODELS


class AmberDataModule(L.LightningDataModule):
    # Chunks is a comma-separated string of integers representing the chunk IDs to load.
    def __init__(
        self, model_name: str, batch_size: int = BATCH_SIZE, chunks: str = None
    ):
        super().__init__()

        self.batch_size = batch_size
        self.detokenizer = AutoTokenizer.from_pretrained("LLM360/Amber", use_fast=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            QUANTIZED_MODELS[model_name].repo_id, use_fast=True
        )

        self.chunks = list(map(int, chunks.split(","))) if chunks else None

    def setup(self, stage: str):
        load_args = {}

        if self.chunks is not None:
            load_args["data_files"] = self._find_chunks(self.chunks)
            if len(load_args["data_files"]) == 0:
                raise ValueError(f"No chunks found for IDs: {self.chunks}")
        else:
            load_args["data_dir"] = AMBER_DATASET_PATH

        self.dataset = load_dataset("json", streaming=True, **load_args).with_format(
            "torch"
        )

    def _collate_fn(self, batch: list[dict]) -> dict:
        texts = self.detokenizer.batch_decode(
            [item["token_ids"] for item in batch], skip_special_tokens=True
        )

        tokenized = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors="pt",
        )

        return {
            "input_ids": tokenized["input_ids"][..., :-1],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"][..., 1:],
        }

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
        )

    # Assumes chunks are named following pattern: "(train_)?0*\d+.jsonl"
    def _find_chunks(self, chunks: list[int]) -> list[str]:
        files = [str(file) for file in Path(AMBER_DATASET_PATH).glob("*.jsonl")]

        file_chunk_id = [
            int(file.split("/")[-1].split("_")[-1].split(".")[0]) for file in files
        ]

        return [
            file for file, chunk_id in zip(files, file_chunk_id) if chunk_id in chunks
        ]


class WikiText2DataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = BATCH_SIZE, data_dir: str = "./data"):
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


if __name__ == "__main__":
    dm = AmberDataModule(model_name="QuantizedSmolModel", batch_size=BATCH_SIZE)
    dm.setup("fit")
    dl = dm.train_dataloader()
    for batch in dl:
        print(batch["input_ids"].shape, batch["attention_mask"].shape)
    print("Done!")
