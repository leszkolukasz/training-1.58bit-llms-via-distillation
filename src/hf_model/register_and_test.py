import os
import pickle as pkl

from lm_eval import evaluator
from lm_eval.api.registry import register_model
from lm_eval.tasks import TaskManager
from lm_eval.utils import setup_logging
from transformers import (CONFIG_MAPPING, AutoConfig, AutoModel,
                          AutoModelForCausalLM)

from src.constants import (BENCHMARK_OUTPUT_FILE, HARNESS_TASK, HF_MODEL_NAME,
                           ORG_NAME)
from src.hf_model.hf_class import BitConfig42, HFQuantizedSmolModel

# Model registration and evaluation


def register():
    AutoConfig.register("quantized_net42", BitConfig42)
    AutoModel.register(BitConfig42, HFQuantizedSmolModel)
    AutoModelForCausalLM.register(BitConfig42, HFQuantizedSmolModel)
    register_model("quantized_net42", HFQuantizedSmolModel)
    # Add to transformers/models/auto/configuration_auto.py
    CONFIG_MAPPING.update([("quantized_net42", "BitConfig42")])


if __name__ == "__main__":
    register()
    setup_logging("DEBUG")

    task_manager = TaskManager()
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={ORG_NAME}/{HF_MODEL_NAME}",
        tasks=[HARNESS_TASK],
        device="cuda:0",
        batch_size=2,
        task_manager=task_manager,
    )

    os.makedirs(os.path.dirname(BENCHMARK_OUTPUT_FILE), exist_ok=True)

    with open(f"{BENCHMARK_OUTPUT_FILE}", "wb") as file:
        pkl.dump(results, file)
