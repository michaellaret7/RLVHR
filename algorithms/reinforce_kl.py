"""REINFORCE with a KL penalty to a frozen reference policy.

From the article's "KL-divergence" section. Unconstrained reward maximisation
drifts away from the pre-trained distribution (reward hacking, mode collapse). A
KL penalty against a frozen reference model acts as a trust region.

The article gives this loop in prose (no full code), so this is the minimal
faithful implementation of the steps it lists:

  - Generate N completions with the current policy.
  - Score them with the verifiable reward.
  - For each completion, compute its KL divergence from the reference policy.
  - Adjust rewards: adjusted = reward - kl_coef * KL.
  - Compute baseline and advantage from the *adjusted* rewards.
  - loss = -log_prob * advantage; backprop.

KL here is the standard per-token estimate summed over completion tokens:
    KL ~= sum_t (log pi_policy(token_t) - log pi_ref(token_t))

Run:
    uv run python algorithms/reinforce_kl.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TrainingConfig
from environment.data import load_humaneval
from evaluation import SAMPLES_PER_PROBLEM, evaluate_model, print_summary
from policy.logprobs import compute_token_log_probs
from policy.model import device, load_model_and_tokenizer
from environment.executor import HumanEvalExecutor
from environment.reward import HumanEvalReward


def train(model, ref_model, tokenizer, train_problems, train_prompts, reward_fn, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    problem_by_prompt = {p["prompt"]: p for p in train_problems}

    training_stats = {"step": [], "loss": [], "avg_reward": [], "avg_kl": []}
    global_step = 0

    for epoch in range(config.num_epochs):
        print(f"EPOCH {epoch + 1}/{config.num_epochs}")
        epoch_prompts = np.random.choice(
            train_prompts, size=config.prompts_per_epoch, replace=False
        )
        for prompt in epoch_prompts:
            reward_fn.set_problem(problem_by_prompt[prompt])

            # Step 1: Generate completions (no gradients)
            model.eval()
            with torch.no_grad():
                prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=False)
                prompt_ids = prompt_tokens.input_ids.to(device)
                prompt_length = prompt_ids.shape[1]

                # Generate all G completions in one batched call (the prompt is
                # identical across samples), instead of G separate generate calls.
                outputs = model.generate(
                    prompt_ids,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=True,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    num_return_sequences=config.generations_per_prompt,
                    pad_token_id=tokenizer.eos_token_id,
                )
                
                all_sequences = list(outputs)  # G rows, all the same length
                all_completions = tokenizer.batch_decode(
                    outputs[:, prompt_length:], skip_special_tokens=True
                )
                all_prompt_lengths = [prompt_length] * config.generations_per_prompt

            # Step 2: Compute rewards
            prompts_list = [prompt] * config.generations_per_prompt
            rewards = reward_fn(prompts_list, all_completions).to(device)
            rewards = rewards.squeeze(-1)  # Shape: (G,)

            # Build the padded batch once; reused for policy and reference passes.
            input_ids, attention_mask, completion_mask = _pad_batch(
                all_sequences, all_prompt_lengths, tokenizer
            )

            # Step 3: Log probabilities under the policy (with gradients).
            model.train()
            token_log_probs, shift_mask = compute_token_log_probs(
                model, input_ids, attention_mask, completion_mask
            )

            # Reference log probabilities (frozen, no gradients).
            with torch.no_grad():
                ref_log_probs, _ = compute_token_log_probs(
                    ref_model, input_ids, attention_mask, completion_mask
                )

            # Step 4: Per-completion KL, then adjust rewards by the KL penalty.
            # Detach the policy log-probs here: the KL feeds the (scalar) reward,
            # not the policy-gradient term.
            kl_per_token = (token_log_probs.detach() - ref_log_probs) * shift_mask
            kl_per_seq = kl_per_token.sum(dim=1)  # Shape: (G,)
            adjusted_rewards = rewards - config.kl_coef * kl_per_seq

            # Step 5: Baseline + advantage from the adjusted rewards.
            if config.use_baseline:
                advantages = adjusted_rewards - adjusted_rewards.mean()
            else:
                advantages = adjusted_rewards
            advantages = advantages.unsqueeze(-1)  # Shape: (G, 1)

            # Step 6: REINFORCE loss with advantage.
            loss_per_token = (-token_log_probs * advantages) * shift_mask
            loss = loss_per_token.sum() / shift_mask.sum()

            # Step 7: Backprop and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            global_step += 1
            avg_reward = rewards.mean().item()
            avg_kl = kl_per_seq.mean().item()
            training_stats["step"].append(global_step)
            training_stats["loss"].append(loss.item())
            training_stats["avg_reward"].append(avg_reward)
            training_stats["avg_kl"].append(avg_kl)
            if global_step % config.log_every == 0:
                print(
                    f"Step {global_step:3d} | Loss: {loss.item():7.4f} |"
                    f"Avg Reward: {avg_reward:6.3f} |"
                    f"Avg KL: {avg_kl:7.4f}"
                )
                print(f" Example completion: '{all_completions[0][:80]}...'")

    return training_stats


def _pad_batch(all_sequences, all_prompt_lengths, tokenizer):
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
    config = TrainingConfig()
    model, tokenizer = load_model_and_tokenizer()
    # Frozen reference policy = a second copy of the base model.
    ref_model, _ = load_model_and_tokenizer()
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    train_problems, eval_problems, train_prompts = load_humaneval()
    reward_fn = HumanEvalReward(HumanEvalExecutor())

    train(model, ref_model, tokenizer, train_problems, train_prompts, reward_fn, config)

    summary = evaluate_model(
        model, tokenizer, eval_problems, reward_fn,
        num_samples=SAMPLES_PER_PROBLEM, max_new_tokens=256,
        temperature=0.7, top_p=0.9,
    )
    print_summary("REINFORCE + KL RESULTS", summary)
