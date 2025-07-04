from threading import Thread
from typing import Optional

import torch
from transformers import TextIteratorStreamer

from src.models.mixins import ChatMixin, Message

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def stream_reponse(
    model: ChatMixin,
    messages: Optional[list[Message]] = None,
    prompt: Optional[str] = None,
) -> str:
    streamer = TextIteratorStreamer(
        model.tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    chat_kwargs = {
        "messages": messages,
        "prompt": prompt,
        "streamer": streamer,
    }

    thread = Thread(target=model.chat, kwargs=chat_kwargs)
    thread.start()

    print("Assistant: ", end="", flush=True)
    response = ""
    for token in streamer:
        print(token, end="", flush=True)
        response += token
    print()

    thread.join()

    return response


def chat_loop(model: ChatMixin, simple: bool = False):
    messages = []

    model.eval()
    model.to(DEVICE)

    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break

            messages.append({"role": "user", "content": user_input})
            assistant_response = (
                stream_reponse(model, messages=messages)
                if not simple
                else stream_reponse(model, prompt=user_input)
            )
            messages.append({"role": "assistant", "content": assistant_response})

        except KeyboardInterrupt:
            break
