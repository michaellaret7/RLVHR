"""Evaluation: pass@1 and pass@k.

Reproduces `evaluate_model` from the article's "Base Model Performance" section.
Run directly to reproduce the BASELINE RESULTS block:

    uv run python evaluation.py
"""

import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from environment.data import load_humaneval
from policy.model import device, load_model_and_tokenizer
from environment.executor import HumanEvalExecutor
from environment.reward import HumanEvalReward

SAMPLES_PER_PROBLEM = 5

def evaluate_model(
    model, 
    tokenizer: AutoTokenizer, 
    problems: list[dict], 
    reward_fn: HumanEvalReward,
    num_samples: int = 5, 
    max_new_tokens: int = 256,
    temperature: float = 0.7, 
    top_p: float = 0.9,
):
    """Evaluate model on problems."""
    results = []

    model.eval()

    for problem in tqdm(problems, desc="Evaluating"):
        reward_fn.set_problem(problem)

        with torch.no_grad():
            inputs = tokenizer(
                problem["prompt"],
                return_tensors="pt",
                padding=False,  # No padding for single sequence
            ).to(device)

            # Generate all samples for this problem in one batched call.
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,  # pass mask explicitly
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_samples,
                pad_token_id=tokenizer.eos_token_id,
            )

            completions = tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

        # Compute rewards
        reward_values = reward_fn([problem["prompt"]] * num_samples, completions)
        rewards = reward_values.squeeze().tolist()

        if not isinstance(rewards, list):
            rewards = [rewards]

        # Compute metrics
        pass_at_1 = 1.0 if rewards[0] == 1.0 else 0.0
        pass_at_k = 1.0 if any(r == 1.0 for r in rewards) else 0.0
        avg_reward = np.mean(rewards)
        results.append({
            "task_id": problem["task_id"],
            "pass_at_1": pass_at_1,
            "pass_at_k": pass_at_k,
            "avg_reward": avg_reward,
            "rewards": rewards,
        })

    # Summary statistics
    summary = {
        "pass_at_1": np.mean([r["pass_at_1"] for r in results]),
        "pass_at_k": np.mean([r["pass_at_k"] for r in results]),
        "avg_reward": np.mean([r["avg_reward"] for r in results]),
        "results": results,
    }
    
    return summary


def print_summary(title, summary, k=SAMPLES_PER_PROBLEM):
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"Pass@1: {summary['pass_at_1'] * 100:.1f}%")
    print(f"Pass@{k}: {summary['pass_at_k'] * 100:.1f}%")
    print(f"Avg Reward: {summary['avg_reward']:.3f}")


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    _, eval_problems, _ = load_humaneval()

    reward_fn = HumanEvalReward(HumanEvalExecutor())

    baseline_results = evaluate_model(
        model, tokenizer, eval_problems, reward_fn,
        num_samples=SAMPLES_PER_PROBLEM,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
    )
    print_summary("BASELINE RESULTS", baseline_results)
