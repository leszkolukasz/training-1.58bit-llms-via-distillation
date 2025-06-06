import datetime

from jsonargparse import lazy_instance
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import MLFlowLogger

from .datamodules import *
from .models import *

suffix = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def main():
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
                run_name=f"test_run_{suffix}",
                log_model=False,  # does not work
            ),
        },
    )


if __name__ == "__main__":
    main()
