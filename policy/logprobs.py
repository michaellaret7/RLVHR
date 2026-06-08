"""Per-token log-probabilities.

This is the piece the article walks through in detail ("Step 1: Shifting",
"Step 2: Log Softmax", "Step 3: Gather"). Given a forward pass over a batch of
full sequences, it returns the log-probability the model assigned to each token
that was actually generated. Every algorithm in algorithms/ reuses this.
"""

import torch
import torch.nn.functional as F


def compute_token_log_probs(model, input_ids, attention_mask, completion_mask):
    """Return (token_log_probs, shift_mask).

    token_log_probs: (batch, seq_len-1) log-prob of each actual next token.
    shift_mask:      (batch, seq_len-1) 1.0 on completion tokens, 0.0 elsewhere.
    """
    # Forward pass WITH gradients
    # Model does a forward pass over the input tokens and returns the raw logits per token
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits 

    # Step 1: shift so logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = completion_mask[:, 1:].contiguous()

    # Step 2: log-softmax over the vocab
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Step 3: gather the log-prob of the token that was actually generated
    token_log_probs = log_probs.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)

    return token_log_probs, shift_mask


if __name__ == "__main__":
    """Minimal: run compute_token_log_probs on the REAL model. uv run python scratch_real.py"""
    import torch
    from model import load_model_and_tokenizer, device
    from logprobs import compute_token_log_probs

    model, tok = load_model_and_tokenizer()

    prompt_ids = tok("def add(a, b):", return_tensors="pt").input_ids.to(device)
    plen = prompt_ids.shape[1]
    seq = model.generate(prompt_ids, max_new_tokens=10, do_sample=True, pad_token_id=tok.eos_token_id)

    input_ids = seq                                   # prompt + completion
    attention_mask = torch.ones_like(input_ids)
    completion_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    completion_mask[:, plen:] = 1.0                   # 1 on generated tokens

    log_probs, mask = compute_token_log_probs(model, input_ids, attention_mask, completion_mask)
    print("text       :", tok.decode(input_ids[0]))
    print("input_ids  :", input_ids[0].tolist())
    print("log_probs  :", [round(x, 3) for x in log_probs[0].tolist()])
    print("shift_mask :", mask[0].tolist())