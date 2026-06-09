"""Pad a batch of generated sequences for a forward pass.

After generating G completions, every algorithm in algorithms/ needs them as a
single padded (G, max_len) tensor plus the masks compute_token_log_probs expects.
Shared here so the four training scripts don't each carry their own copy. Pairs
with policy/logprobs.py.
"""

import torch
import torch.nn.functional as F

from policy.model import device
from transformers import AutoTokenizer


def pad_batch(all_sequences, all_prompt_lengths, tokenizer):
    """Pad variable-length sequences and build completion + attention masks."""
    max_len = max(seq.shape[0] for seq in all_sequences)
    padded_ids = []
    completion_masks = []

    for seq, plen in zip(all_sequences, all_prompt_lengths):
        padding_length = max_len - seq.shape[0]

        padded = F.pad(seq, (0, padding_length), value=tokenizer.pad_token_id)
        padded_ids.append(padded)

        mask = torch.zeros(max_len, dtype=torch.float32, device=device)
        mask[plen:seq.shape[0]] = 1.0

        completion_masks.append(mask)

    input_ids = torch.stack(padded_ids)
    completion_mask = torch.stack(completion_masks)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return input_ids, attention_mask, completion_mask


if __name__ == "__main__":
    all_sequences = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    all_prompt_lengths = [2, 1]
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # gpt2 has no pad token; reuse EOS like model.py does
    input_ids, attention_mask, completion_mask = pad_batch(all_sequences, all_prompt_lengths, tokenizer)
    print(input_ids)
    print(attention_mask)
    print(completion_mask)