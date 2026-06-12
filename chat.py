"""Interactive chat REPL backed by RL-trained weights pulled from S3.

Downloads a saved-weights prefix, loads it, and loops — keeping the full
conversation history so it's a real multi-turn chat, chat-template formatted.

Usage:
    uv run python chat.py [s3_path]

s3_path is an s3://bucket/prefix URI or a bare prefix (uses $S3_BUCKET).
Defaults to the run below. Commands: 'exit'/'quit' to leave, 'reset' to clear.
"""

import sys

import torch

from checkpoint_s3 import load_weights_from_s3
from policy.model import device, load_model_and_tokenizer

DEFAULT_S3 = (
    "s3://ml7-model-weights/rl_model_weights/policy_gradient/"
    "qwen2.5-coder-1.5b-instruct-20260612-154511/final"
)


def parse_s3(path: str) -> tuple[str | None, str]:
    """Split 's3://bucket/prefix' into (bucket, prefix); bare prefix -> (None, prefix)."""
    if path.startswith("s3://"):
        bucket, _, prefix = path[len("s3://"):].partition("/")
        return bucket, prefix.rstrip("/")
    return None, path.rstrip("/")


def generate_reply(model, tokenizer, messages: list[dict]) -> str:
    """Apply the chat template to the running history and decode one reply."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()  # AWS_* / S3_BUCKET creds for the download

    bucket, prefix = parse_s3(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_S3)
    local = load_weights_from_s3(prefix, bucket=bucket)

    model, tokenizer = load_model_and_tokenizer(local)
    model.eval()

    print("\nloaded. type 'exit' to quit, 'reset' to clear history.\n")
    messages: list[dict] = []
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("exit", "quit"):
            break
        if user == "reset":
            messages = []
            print("(history cleared)\n")
            continue

        messages.append({"role": "user", "content": user})
        reply = generate_reply(model, tokenizer, messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"model> {reply}\n")


if __name__ == "__main__":
    main()
