"""Dataset loading.

From the article's "Problem Description" section. We use HumanEval (164 problems)
and, for quicker experiments, take 50 problems for training and 30 for evaluation.

We load the EvalPlus **HumanEval+** variant: identical problems, prompts, and
canonical solutions as the original HumanEval, but with ~80x more unit tests per
problem. Since the test suite *is* the verifiable reward, the extra tests make
the reward much harder to hack (buggy code that slipped past HumanEval's few
asserts now fails). It's a drop-in: same record keys, just a larger `test`
field. Switch DATASET_ID back to "openai/openai_humaneval" for the original.
"""

from datasets import load_dataset

DATASET_ID = "evalplus/humanevalplus"
# DATASET_ID = "openai/openai_humaneval"

NUM_TRAIN = 50
NUM_EVAL = 30


def load_humaneval():
    """Return (train_problems, eval_problems, train_prompts).

    Each problem is a HumanEval(+) record with keys: task_id, prompt, test,
    entry_point, canonical_solution.
    """
    dataset = load_dataset(DATASET_ID, split="test")

    train_problems = [dataset[i] for i in range(NUM_TRAIN)]
    eval_problems = [dataset[i] for i in range(NUM_TRAIN, NUM_TRAIN + NUM_EVAL)]

    train_prompts = [p["prompt"] for p in train_problems]
    
    return train_problems, eval_problems, train_prompts


if __name__ == "__main__":
    train_problems, eval_problems, train_prompts = load_humaneval()

    index = 15

    print("-" * 50 + " Prompt " + "-" * 50)
    print(eval_problems[index]["prompt"])
    print("-" * 50 + " Test " + "-" * 50)
    print(eval_problems[index]["test"])
    print("-" * 50 + " Canonical Solution " + "-" * 50)
    print(eval_problems[index]["canonical_solution"])
    print("-" * 50 + " Entry Point " + "-" * 50)
    print(eval_problems[index]["entry_point"])

    full_final_code = eval_problems[index]["prompt"] + eval_problems[index]["canonical_solution"]
    print("-" * 50 + " Full Final Code " + "-" * 50)
    print(full_final_code)