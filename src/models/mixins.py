from typing import Optional, TypedDict

from transformers import AutoTokenizer


class Message(TypedDict):
    role: str
    content: str


class ChatMixin:
    tokenizer: AutoTokenizer

    def chat(
        self, messages: Optional[list[Message]], prompt: Optional[str], **kwargs
    ) -> str:
        raise NotImplementedError("Subclasses must implement this method.")


# NOTE: should be inherited before LightningModule
class LogArtifactMixin:
    def on_fit_end(self):
        super().on_fit_end()

        model_path = (
            self.trainer.checkpoint_callback.last_model_path
            or self.trainer.checkpoint_callback.best_model_path
        )

        self.logger.experiment.log_artifact(self.logger.run_id, model_path)

        print(
            f"Model artifact logged to MLflow: {model_path} with run ID: {self.logger.run_id}"
        )
