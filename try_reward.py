"""Standalone demo of HumanEvalReward — no model or GPU needed.

Run: python try_reward.py

We hand-build a HumanEval-style problem and feed it several fake "completions"
to see exactly when the verifiable reward is 1.0 vs 0.0.
"""

from environment.executor import HumanEvalExecutor
from environment.reward import HumanEvalReward


def main():
    # A HumanEval problem is just a dict with these keys. The `prompt` holds the
    # function signature; `test` is a harness defining check(candidate); the model
    # is supposed to fill in the body.
    problem = {
        "entry_point": "add_one",
        "prompt": "def add_one(x):\n    \"\"\"Return x + 1.\"\"\"\n",
        "test": (
            "def check(candidate):\n"
            "    assert candidate(1) == 2\n"
            "    assert candidate(10) == 11\n"
            "    assert candidate(-5) == -4\n"
        ),
    }

    # Wire up the real executor + reward function.
    reward_fn = HumanEvalReward(executor=HumanEvalExecutor(timeout=5.0))
    reward_fn.set_problem(problem)

    # The reward function receives (prompts, completions) as parallel lists.
    # We reuse the same prompt and vary the completion to hit each code path.
    prompts = [problem["prompt"]] * 4
    completions = [
        # 1) Correct body -> passes tests -> reward 1.0
        "    return x + 1\n",
        # 2) Wrong body -> tests fail -> reward 0.0
        "    return x + 2\n",
        # 3) Markdown-fenced correct answer -> extractor pulls it out -> 1.0
        "```python\ndef add_one(x):\n    return x + 1\n```",
        # 4) No function definition at all -> extraction returns None -> 0.0
        "I'm not sure how to solve this problem.",
    ]

    rewards = reward_fn(prompts, completions)

    labels = ["correct body", "wrong body", "fenced correct", "no code"]
    print("reward tensor shape:", tuple(rewards.shape))  # -> (4, 1)
    print()
    for label, r in zip(labels, rewards.squeeze(-1).tolist()):
        print(f"  {label:16s} -> reward {r}")


# The executor uses multiprocessing (spawn on macOS), which re-imports this
# module in each child process. Without this guard, every child would re-run
# the demo and crash. This is the same reason the real training entry point
# must run under `if __name__ == "__main__":`.
if __name__ == "__main__":
    main()
