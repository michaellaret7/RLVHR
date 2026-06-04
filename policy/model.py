"""Base model + tokenizer loading.

From the article's "Base Model Performance" section. We use a small coder model
so the whole loop runs on modest hardware.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """Load the base model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # Many causal LMs have no pad token; reuse EOS so padding works.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
