import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from src.datamodules import *
from src.models import *

# Script for analyzing which layers to quantize in a model


def LIM(model):
    dataset = load_dataset(
        "json", data_dir=AMBER_DATASET_PATH, streaming=True
    ).with_format("torch")

    dataloader = DataLoader(
        dataset["train"],
        batch_size=BATCH_SIZE,
    )

    input_activations = {}
    output_activations = {}

    handles = []

    def make_hooks(layer_idx):
        def input_hook(_module, pos_inp, kwargs_inp):
            inp = kwargs_inp["hidden_states"] if len(pos_inp) == 0 else pos_inp[0]
            input_activations[layer_idx] = inp.detach().cpu().numpy()

        def output_hook(_module, _inp, out):
            out = out[0] if isinstance(out, tuple) else out
            output_activations[layer_idx] = out.detach().cpu().numpy()

        return input_hook, output_hook

    for index, (name, module) in enumerate(model.named_modules()):
        if name.endswith("mlp"):
            input_hook, output_hook = make_hooks(index)
            handles.append(
                module.register_forward_pre_hook(input_hook, with_kwargs=True)
            )
            handles.append(module.register_forward_hook(output_hook))
        elif name.endswith("self_attn"):
            input_hook, output_hook = make_hooks(index)
            handles.append(
                module.register_forward_pre_hook(input_hook, with_kwargs=True)
            )
            handles.append(module.register_forward_hook(output_hook))

    layer_sims = {
        f"{name}": []
        for name, module in model.named_modules()
        if name.endswith("mlp") or name.endswith("self_attn")
    }

    device = "cpu"
    model.to(device)

    for batch in tqdm(dataloader):

        # NOTE: token_ids need to be converted using tokenizer for Smol
        inputs = {
            "input_ids": batch["token_ids"].to(device),
            "attention_mask": (
                batch["attention_mask"].to(device)
                if "attention_mask" in batch
                else None
            ),
        }

        input_activations.clear()
        output_activations.clear()

        with torch.no_grad():
            model(**inputs)

        for layer_idx, (name, module) in enumerate(model.named_modules()):
            if (
                name.endswith("mlp")
                or name.endswith("self_attn")
                and layer_idx in input_activations
            ):
                inp = input_activations[layer_idx]
                out = output_activations[layer_idx]

                batch_sims = []
                for k in range(inp.shape[0]):
                    for b in range(inp.shape[1]):
                        sim = torch.cosine_similarity(
                            torch.tensor(inp[k, b, :]),
                            torch.tensor(out[k, b, :]),
                            dim=0,
                        ).item()
                        batch_sims.append(sim)
                layer_sims[f"{name}"].append(sum(batch_sims) / len(batch_sims))

    for handle in handles:
        handle.remove()

    lim_scores = {layer: -np.mean(sims) for layer, sims in layer_sims.items()}

    return lim_scores


def ZD(model):
    scores = dict()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weights = module.weight
            std = weights.std().item()
            mean = weights.mean().item()
            z_score = (weights - mean) / (std + EPSILON)
            z_score = z_score.flatten()
            greater_than_one = (z_score > 1).sum().item()
            scores[name] = greater_than_one / len(z_score)
    return scores


def analyze_layers_to_quantize(model_name: str, score_function):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    scores = score_function(model)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


if __name__ == "__main__":
    model_name = SMOL_MODEL_ID
    SCORE = ZD

    scores = analyze_layers_to_quantize(model_name, SCORE)

    for layer, score in scores:
        print(f"Layer: {layer}, Score: {score:.4f}")

    # with open(f"layers_{SCORE}.txt", "w", encoding="utf-8") as f:
    #     for layer, _ in scores:
    #         f.write(layer + "\n")

    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # for name, module in model.named_modules():
    #     print(name)
