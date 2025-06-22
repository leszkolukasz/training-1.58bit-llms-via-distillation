import argparse
import datetime

import mlflow
import torch
from jsonargparse import lazy_instance
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import MLFlowLogger

from src.chat import chat_loop
from src.datamodules import *
# Required for LightningCLI to detect all models and datamodules
from src.models import *

torch.set_float32_matmul_precision("high")


suffix = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_name = "nlp_project"
tracking_uri = "file:mlruns"

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

client = mlflow.tracking.MlflowClient()


def get_or_create_run(run_name: str):
    runs = client.search_runs(
        experiment_ids=[client.get_experiment_by_name(experiment_name).experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
    )

    for r in runs:
        return r

    return client.create_run(
        experiment_id=client.get_experiment_by_name(experiment_name).experiment_id,
        tags={"mlflow.runName": run_name},
    )


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "model.class_path",
            "data.init_args.model_name",
            compute_fn=lambda a: a.split(".")[-1],
        )


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "command", choices=["chat", "fit", "validate", "test", "predict"]
    )
    parser.add_argument("--simple", action="store_true", default=False)
    parser.add_argument("--model", type=str, required=False)
    parser.add_argument("--model.quantization", type=str, required=False)
    parser.add_argument("--model.bitlinear_implementation", type=str, required=False)
    parser.add_argument("--model.loss_function", type=str, required=False)
    parser.add_argument("--ckpt_path", type=str, required=False)

    args, _ = parser.parse_known_args()

    model_name = args.model
    quantization = vars(args)["model.quantization"]
    bitlinear_implementation = vars(args)["model.bitlinear_implementation"]
    loss_function = vars(args)["model.loss_function"]
    checkpoint_path = args.ckpt_path

    if args.command == "chat":
        if model_name not in QUANTIZED_MODELS:
            raise ValueError(
                f"Model {model_name} is not supported for chat command. Supported models: {list(QUANTIZED_MODELS.keys())}"
            )

        if checkpoint_path is None:
            raise ValueError("Checkpoint path must be provided for chat command.")

        model = QUANTIZED_MODELS[model_name].cls.load_from_checkpoint(checkpoint_path)
        chat_loop(model, args.simple)

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

    run = get_or_create_run(run_name)

    MyLightningCLI(
        seed_everything_default=42,
        # save_config_callback=None
        save_config_kwargs={
            "overwrite": True,
        },
        trainer_defaults={
            "accelerator": "auto",
            "devices": -1,
            "precision": "bf16-mixed",
            "fast_dev_run": False,
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 16 // BATCH_SIZE + 1,
            "deterministic": True,
            "logger": lazy_instance(
                MLFlowLogger,
                experiment_name="nlp_project",
                tracking_uri="file:mlruns",
                run_name=run_name,
                run_id=run.info.run_id,
                log_model=False,
            ),
        },
    )


if __name__ == "__main__":
    main()
