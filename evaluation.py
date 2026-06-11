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
from policy.model import device
from environment.executor import HumanEvalExecutor
from environment.reward import HumanEvalReward

SAMPLES_PER_PROBLEM = 5


def _score_problem(problem: dict, completions: list[str], reward_fn: HumanEvalReward) -> dict:
    """Run the reward on one problem's completions and compute pass@ metrics."""
    reward_fn.set_problem(problem)
    reward_values = reward_fn([problem["prompt"]] * len(completions), completions)
    rewards = reward_values.squeeze().tolist()

    if not isinstance(rewards, list):
        rewards = [rewards]

    return {
        "task_id": problem["task_id"],
        "pass_at_1": 1.0 if rewards[0] == 1.0 else 0.0,
        "pass_at_k": 1.0 if any(r == 1.0 for r in rewards) else 0.0,
        "avg_reward": np.mean(rewards),
        "rewards": rewards,
    }


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
        with torch.no_grad():
            # Tokenize the prompt to pass to the model
            inputs = tokenizer(problem["prompt"], return_tensors="pt", padding=False).to(device)

            # Generate all samples for this problem in one batched call. 
            # Its one problem per item in the batch and for each problem we generate num_samples completions
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

            # Decode the outputs to get the completions in text format
            completions = tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

        # Score the completions using the reward function and compute the pass@ metrics then append the results to the list
        results.append(_score_problem(problem, completions, reward_fn))

    return {
        "pass_at_1": np.mean([r["pass_at_1"] for r in results]),
        "pass_at_k": np.mean([r["pass_at_k"] for r in results]),
        "avg_reward": np.mean([r["avg_reward"] for r in results]),
        "results": results,
    }


def evaluate_model_vllm(
    llm,
    problems: list[dict],
    reward_fn: HumanEvalReward,
    num_samples: int = 5,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Evaluate via vLLM: all problems x samples generated in one batched call."""
    from vllm import SamplingParams  # local import: keep vLLM optional for HF-only runs

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        n=num_samples,
    )
    outputs = llm.generate([p["prompt"] for p in problems], sampling_params)

    results = []
    for problem, output in zip(problems, outputs):
        completions = [o.text for o in output.outputs]
        results.append(_score_problem(problem, completions, reward_fn))

    return {
        "pass_at_1": np.mean([r["pass_at_1"] for r in results]),
        "pass_at_k": np.mean([r["pass_at_k"] for r in results]),
        "avg_reward": np.mean([r["avg_reward"] for r in results]),
        "results": results,
    }


if __name__ == "__main__":
    import os

    # FlashInfer's JIT sampler needs ninja + a CUDA toolchain, absent on this box.
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    # policy.model initializes CUDA at import; forked engine workers can't reuse it.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    from vllm import LLM

    from policy.model import MODEL_NAME

    _, eval_problems, _ = load_humaneval()

    reward_fn = HumanEvalReward(HumanEvalExecutor())

    llm = LLM(model=MODEL_NAME, max_model_len=2048)
    baseline_results = evaluate_model_vllm(
        llm, eval_problems, reward_fn,
        num_samples=SAMPLES_PER_PROBLEM,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
    )
    print(
        f"BASELINE RESULTS — Pass@1: {baseline_results['pass_at_1']:.1%} | "
        f"Pass@{SAMPLES_PER_PROBLEM}: {baseline_results['pass_at_k']:.1%} | "
        f"Avg reward: {baseline_results['avg_reward']:.3f}"
    )
