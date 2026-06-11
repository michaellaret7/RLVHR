"""REINFORCE Leave-One-Out (RLOO).

From the article's "REINFORCE Leave-One-Out (RLOO)" section. A variance-reduction
tweak to the baseline: when computing the advantage for completion i, use a
baseline that is the mean of the *other* completions' rewards (leaving i out).
That makes the baseline independent of completion i, reducing variance better
than the plain batch-mean baseline.

Everything else is identical to reinforce.py; only the advantage computation
changes (see compute_rloo_advantages).

Run:
    uv run python algorithms/rloo.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TrainingConfig
from environment.data import load_humaneval
from evaluation import SAMPLES_PER_PROBLEM, evaluate_model
from policy.batching import pad_batch
from policy.logprobs import compute_token_log_probs
from policy.model import device, load_model_and_tokenizer
from environment.executor import HumanEvalExecutor
from environment.reward import HumanEvalReward


def compute_rloo_advantages(rewards, normalize=True):
    """Compute REINFORCE Leave-One-Out advantages.

    Args:
        rewards: Tensor of shape (G,) - rewards for G completions of same prompt
        normalize: Whether to normalize advantages
    Returns:
        advantages: Tensor of shape (G,)
    """
    G = len(rewards)
    if G == 1:
        # Can't do leave-one-out with a single sample, return zero advantage
        return torch.zeros_like(rewards)
    advantages = []
    for i in range(G):
        # Compute baseline excluding i-th reward
        other_rewards = torch.cat([rewards[:i], rewards[i + 1:]])
        baseline_i = other_rewards.mean()
        # Advantage for i-th completion
        advantage_i = rewards[i] - baseline_i
        advantages.append(advantage_i)
    advantages = torch.stack(advantages)
    # Optional: normalize advantages
    if normalize and len(advantages) > 1:
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages


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

                print(all_sequences[0])
                print(all_completions[0])
                print(all_prompt_lengths[0])

            # Step 2: Compute rewards
            prompts_list = [prompt] * config.generations_per_prompt
            rewards = reward_fn(prompts_list, all_completions).to(device)
            rewards = rewards.squeeze(-1)  # Shape: (G,)

            # Step 3: RLOO advantages (the only change vs reinforce.py)
            advantages = compute_rloo_advantages(rewards, normalize=True)
            advantages = advantages.unsqueeze(-1)  # Shape: (G, 1)

            # Step 4: Compute log probabilities (with gradients!)
            model.train()
            input_ids, attention_mask, completion_mask = pad_batch(
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
    print(
        f"RLOO RESULTS — Pass@1: {summary['pass_at_1']:.1%} | "
        f"Pass@{SAMPLES_PER_PROBLEM}: {summary['pass_at_k']:.1%} | "
        f"Avg reward: {summary['avg_reward']:.3f}"
    )
