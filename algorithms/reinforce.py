"""REINFORCE with a mean baseline.

From the article's "REINFORCE" section. Same loop as vanilla policy gradient,
but we subtract a baseline (the batch mean reward) from the rewards to get the
*advantage*, then use:

    loss = -log_prob * advantage

Subtracting a state-dependent baseline reduces the variance of the gradient
estimate, giving more stable learning. This is the only change from
policy_gradient.py (see "Step 3: NEW" below).

Run:
    uv run python algorithms/reinforce.py
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


def train(model, tokenizer, train_problems, train_prompts, reward_fn, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    problem_by_prompt = {p["prompt"]: p for p in train_problems}

    training_stats = {"step": [], "loss": [], "avg_reward": [], "avg_completion_length": []}
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

            # Step 3: NEW — compute advantages with baseline subtraction
            if config.use_baseline:
                baseline_value = rewards.mean()
                advantages = rewards - baseline_value
            else:
                advantages = rewards
                baseline_value = 0.0
            advantages = advantages.unsqueeze(-1)  # Shape: (G, 1)

            # Step 4: Compute log probabilities (with gradients!)
            model.train()
            input_ids, attention_mask, completion_mask = _pad_batch(
                all_sequences, all_prompt_lengths, tokenizer
            )
            token_log_probs, shift_mask = compute_token_log_probs(
                model, input_ids, attention_mask, completion_mask
            )

            # Step 5: REINFORCE loss with advantage
            loss_per_token = (-token_log_probs * advantages) * shift_mask
            loss = loss_per_token.sum() / shift_mask.sum()

            # Step 6: Backprop and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            global_step += 1
            avg_reward = rewards.mean().item()
            avg_length = np.mean([len(c.split()) for c in all_completions])
            training_stats["step"].append(global_step)
            training_stats["loss"].append(loss.item())
            training_stats["avg_reward"].append(avg_reward)
            training_stats["avg_completion_length"].append(avg_length)
            if global_step % config.log_every == 0:
                print(
                    f"Step {global_step:3d} | Loss: {loss.item():7.4f} |"
                    f"Avg Reward: {avg_reward:6.3f} |"
                    f"Avg Length: {avg_length:5.1f} tokens"
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
    train_problems, eval_problems, train_prompts = load_humaneval()
    reward_fn = HumanEvalReward(HumanEvalExecutor())

    train(model, tokenizer, train_problems, train_prompts, reward_fn, config)

    summary = evaluate_model(
        model, tokenizer, eval_problems, reward_fn,
        num_samples=SAMPLES_PER_PROBLEM, max_new_tokens=256,
        temperature=0.7, top_p=0.9,
    )
    print_summary("REINFORCE RESULTS", summary)
