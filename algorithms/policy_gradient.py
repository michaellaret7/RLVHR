"""Vanilla Policy Gradient.

From the article's "Policy Gradient Optimisation" section. The simplest recipe:

    loss = -log_prob * reward

For each prompt: generate G completions, score them with the verifiable reward,
compute token log-probs, and take a gradient step. No baseline, no KL.

Run:
    uv run python algorithms/policy_gradient.py
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
    # Initialize the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.learning_rate
    )

    # Map prompt -> problem so we can set the reward's current problem.
    problem_by_prompt = {p["prompt"]: p for p in train_problems}

    training_stats = {"step": [], "loss": [], "avg_reward": [], "avg_completion_length": []}
    global_step = 0

    for epoch in range(config.num_epochs):
        print(f"EPOCH {epoch + 1}/{config.num_epochs}")

        # Randomly sample prompts for the epoch
        epoch_prompts = np.random.choice(
            train_prompts, 
            size=config.prompts_per_epoch, 
            replace=False
        )

        # Loop through each prompt in the epoch
        for prompt in epoch_prompts:

            # Set the problem for the reward function class to the current prompt
            reward_fn.set_problem(problem_by_prompt[prompt])

            # Step 1: Generate completions (no gradients)
            all_completions = []
            all_sequences = []
            all_prompt_lengths = []

            model.eval()

            with torch.no_grad():
                # Tokenize the prompt 
                prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=False)
                prompt_ids = prompt_tokens.input_ids.to(device)
                prompt_length = prompt_ids.shape[1]

                for _ in range(config.generations_per_prompt):
                    outputs = model.generate(
                        prompt_ids,
                        max_new_tokens=config.max_new_tokens,
                        do_sample=True,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    
                    full_sequence = outputs[0]
                    completion_only = full_sequence[prompt_length:]
                    completion_text = tokenizer.decode(
                        completion_only, skip_special_tokens=True
                    )
                    all_completions.append(completion_text)
                    all_sequences.append(full_sequence)
                    all_prompt_lengths.append(prompt_length)

            # Step 2: Compute rewards
            prompts_list = [prompt] * config.generations_per_prompt
            rewards = reward_fn(prompts_list, all_completions).to(device)

            # Step 3: Compute log probabilities (with gradients!)
            model.train()
            input_ids, attention_mask, completion_mask = _pad_batch(
                all_sequences, all_prompt_lengths, tokenizer
            )
            token_log_probs, shift_mask = compute_token_log_probs(
                model, input_ids, attention_mask, completion_mask
            )

            # Step 4: REINFORCE loss — negative because PyTorch minimises.
            loss_per_token = (-token_log_probs * rewards) * shift_mask
            loss = loss_per_token.sum() / shift_mask.sum()

            # Step 5: Backprop and update
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
        model, 
        tokenizer, 
        eval_problems,
        reward_fn,
        num_samples=SAMPLES_PER_PROBLEM,
        max_new_tokens=256,
        temperature=0.7, 
        top_p=0.9,
    )
    print_summary("VANILLA POLICY GRADIENT RESULTS", summary)
