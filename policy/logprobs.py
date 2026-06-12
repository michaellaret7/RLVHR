"""Per-token log-probabilities.

This is the piece the article walks through in detail ("Step 1: Shifting",
"Step 2: Log Softmax", "Step 3: Gather"). Given a forward pass over a batch of
full sequences, it returns the log-probability the model assigned to each token
that was actually generated. Every algorithm in algorithms/ reuses this.
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_token_log_probs(model, input_ids, attention_mask, completion_mask):
    """Return (token_log_probs, shift_mask).

    token_log_probs: (batch, seq_len-1) log-prob of each actual next token.
    shift_mask:      (batch, seq_len-1) 1.0 on completion tokens, 0.0 elsewhere.
    """
    logger.debug(
        "scoring %d sequences of length %d", input_ids.shape[0], input_ids.shape[1]
    )
    # Forward pass WITH gradients (scoring pass, not generation — the tokens
    # are already decided; we need the graph so backprop can reach the weights).
    # logits: (batch, seq_len, vocab) — position t holds 152k raw scores
    # predicting token t+1.
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # Step 1: align predictions with targets. Position t's logits are scored
    # against the token at t+1: last position predicts nothing → dropped;
    # first token is predicted by nothing → dropped as a target.
    predictions = logits[:, :-1, :]        # (batch, seq_len-1, vocab)
    targets = input_ids[:, 1:]             # (batch, seq_len-1)
    shift_mask = completion_mask[:, 1:]    # 1.0 where target is a completion token

    # Step 2: raw scores → log-probabilities over the vocab (a logit is only
    # meaningful relative to the other 152k scores).
    log_probs = F.log_softmax(predictions, dim=-1)

    # Step 3: for each position, look up the log-prob of the token that
    # actually came next. The token id IS the index into the vocab dimension.
    # Collapses (batch, seq_len-1, vocab) → (batch, seq_len-1).
    token_log_probs = log_probs.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    return token_log_probs, shift_mask


if __name__ == "__main__":
    """Minimal: run compute_token_log_probs on the REAL model. uv run python scratch_real.py"""
    import torch
    from config import TrainingConfig
    from policy.model import load_model_and_tokenizer, device

    model, tok = load_model_and_tokenizer(TrainingConfig().model_name)

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