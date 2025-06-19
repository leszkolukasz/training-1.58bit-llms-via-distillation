import argparse
import datetime

from jsonargparse import lazy_instance
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import MLFlowLogger

from src.chat import chat_loop
from src.datamodules import *
from src.layers import ImplementationType, QuantizationType
from src.loss import LossFunctionType
# Required for LightningCLI to detect all models and datamodules
from src.models import *
from src.models.quantized import QuantizedQwenModel

suffix = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

QUANTIZED_MODELS = {
    "QuantizedQwenModel": QuantizedQwenModel,
}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "command", choices=["chat", "fit", "validate", "test", "predict"]
    )
    parser.add_argument("--model", type=str, required=False)
    parser.add_argument("--model.quantization", type=str, required=False)
    parser.add_argument(
        "--model.bitlinear_implementation", type=str, required=False
    )
    parser.add_argument("--model.loss_function", type=str, required=False)
    parser.add_argument("--ckp_path", type=str, required=False)

    args, _ = parser.parse_known_args()

    model_name = args.model
    quantization = vars(args)["model.quantization"]
    bitlinear_implementation = vars(args)["model.bitlinear_implementation"]
    loss_function = vars(args)["model.loss_function"]
    checkpoint_path = args.ckp_path

    if args.command == "chat":
        if model_name not in QUANTIZED_MODELS:
            raise ValueError(
                f"Model {model_name} is not supported for chat command. Supported models: {list(QUANTIZED_MODELS.keys())}"
            )

        if checkpoint_path is None:
            raise ValueError("Checkpoint path must be provided for chat command.")

        # model = QUANTIZED_MODELS[model_name].load_from_checkpoint(checkpoint_path)
        model = None
        chat_loop(model)

        return

    run_name = f"test_run_{suffix}"

    if args.command == "fit" and model_name in QUANTIZED_MODELS:
        if (
            quantization is None
            or bitlinear_implementation is None
            or loss_function is None
        ):
            raise ValueError(
                "For training, quantization, bitlinear implementation, and loss function must be specified."
            )

        run_name = (
            f"quant_{quantization}_impl_{bitlinear_implementation}_loss_{loss_function}"
        )

    LightningCLI(
        seed_everything_default=42,
        save_config_callback=None,
        trainer_defaults={
            "accelerator": "auto",
            "devices": -1,
            "precision": "bf16-mixed",
            "fast_dev_run": False,
            # "gradient_clip_val": 0.5,
            "deterministic": True,
            "logger": lazy_instance(
                MLFlowLogger,
                experiment_name="nlp_project",
                tracking_uri="file:mlruns",
                run_name=run_name,
                log_model=False,
            ),
        },
    )


if __name__ == "__main__":
    main()
