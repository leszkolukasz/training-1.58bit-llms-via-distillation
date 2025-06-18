from typing import TypedDict

from transformers import AutoTokenizer


class Message(TypedDict):
    role: str
    content: str


class GeneratorMixin:
    tokenizer: AutoTokenizer

    def generate(self, messages: list[Message], **kwargs) -> str:
        raise NotImplementedError("Subclasses must implement this method.")
