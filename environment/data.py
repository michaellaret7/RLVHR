"""Dataset loading.

From the article's "Problem Description" section. We use HumanEval (164 problems)
and, for quicker experiments, take 50 problems for training and 30 for evaluation.
"""

from datasets import load_dataset

NUM_TRAIN = 50
NUM_EVAL = 30


def load_humaneval():
    """Return (train_problems, eval_problems, train_prompts).

    Each problem is a HumanEval record with keys: task_id, prompt, test,
    entry_point, canonical_solution.
    """
    dataset = load_dataset("openai/openai_humaneval", split="test")

    train_problems = [dataset[i] for i in range(NUM_TRAIN)]
    eval_problems = [dataset[i] for i in range(NUM_TRAIN, NUM_TRAIN + NUM_EVAL)]

    train_prompts = [p["prompt"] for p in train_problems]
    
    return train_problems, eval_problems, train_prompts


if __name__ == "__main__":
    train_problems, eval_problems, train_prompts = load_humaneval()

    print("-" * 80 + " Prompt " + "-" * 80)
    print(eval_problems[0]["prompt"])
    print("-" * 80 + " Test " + "-" * 80)
    print(eval_problems[0]["test"])
    print("-" * 80 + " Canonical Solution " + "-" * 80)
    print(eval_problems[0]["canonical_solution"])
    print("-" * 80 + " Entry Point " + "-" * 80)
    print(eval_problems[0]["entry_point"])
