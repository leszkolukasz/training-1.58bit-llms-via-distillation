from threading import Thread

import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer)

from src.constants import QWEN_MODEL_ID
from src.models.mixins import GeneratorMixin, Message

base_model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_ID, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, use_fast=True, padding_side="left")

base_model.eval()
base_model.to("cuda")


def stream_reponse(model: GeneratorMixin, messages: list[Message]):
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = tokenizer([prompt], return_tensors="pt").to(base_model.device)

    generate_kwargs = {
        # "messages": messages,
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "streamer": streamer,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 1024,
    }

    thread = Thread(target=base_model.generate, kwargs=generate_kwargs)
    thread.start()

    print("Assistant: ", end="", flush=True)
    response = ""
    for token in streamer:
        print(token, end="", flush=True)
        response += token
    print()

    thread.join()

    return response


def chat_loop(model: GeneratorMixin):
    messages = []

    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break

            messages.append({"role": "user", "content": user_input})
            assistant_response = stream_reponse(model, messages)
            messages.append({"role": "assistant", "content": assistant_response})

        except KeyboardInterrupt:
            break
