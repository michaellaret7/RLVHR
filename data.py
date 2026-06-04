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

    print(f"Train problems: {len(train_problems)}")
    print(f"Eval problems:  {len(eval_problems)}")
    print(f"Train prompts:  {len(train_prompts)}")

    print("\n" + "=" * 80)
    print("FIRST TRAIN PROBLEM")
    print("=" * 80)
    p = train_problems[0]
    print(f"\ntask_id: {p['task_id']}")
    print(f"entry_point: {p['entry_point']}")
    print(f"\n--- prompt ---\n{p['prompt']}")
    print(f"\n--- canonical_solution ---\n{p['canonical_solution']}")
    print(f"\n--- test ---\n{p['test']}")