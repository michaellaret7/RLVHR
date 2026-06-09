"""Test-based reward function.

Reproduces `HumanEvalReward` from the article's "Base Model Performance" section.
This is the verifiable reward of RLVR: 1.0 if the generated code passes the
problem's unit tests, 0.0 otherwise.
"""

from typing import List

import torch

from environment.extraction import extract_code_with_prompt


class HumanEvalReward:
    """Reward function for HumanEval."""

    def __init__(self, executor):
        self.executor = executor # initializes the code executor object
        self.current_problem = None

    def set_problem(self, problem):
        self.current_problem = problem # sets the current problem

    def __call__(self, prompts: List[str], completions: List[str]):
        if self.current_problem is None:
            raise ValueError("Must set problem first")

        test = self.current_problem["test"]
        entry_point = self.current_problem["entry_point"]

        # Extract the runnable code for every completion up front.
        # None means the llm didn't return any code → reward 0.0.
        codes = [
            extract_code_with_prompt(prompt, completion, entry_point)
            for prompt, completion in zip(prompts, completions)
        ]

        # Run every completion that produced code in one concurrent batch
        # instead of one blocking process per completion.
        runnable = [(code, test, entry_point) for code in codes if code is not None]
        batch_results = iter(self.executor.execute_batch(runnable))

        # Map results back onto the full list, scoring missing code as 0.0.
        rewards = []
        for code in codes:
            if code is None:
                rewards.append(0.0)
            else:
                passed, _error = next(batch_results)
                rewards.append(1.0 if passed else 0.0)

        return torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)


if __name__ == "__main__":
    # Demo: run a few fake "model" completions through the reward function and
    # watch the verifiable reward come out (1.0 = passed the tests, 0.0 = failed).
    from environment.data import load_humaneval
    from environment.executor import HumanEvalExecutor

    # Load the humaneval problems
    _, eval_problems, _ = load_humaneval()
    problems = eval_problems[:30]

    # Set the reward func with the code executor cls
    reward_fn = HumanEvalReward(HumanEvalExecutor())

    # Score each problem's canonical solution; all should pass (reward 1.0).
    for problem in problems:
        reward_fn.set_problem(problem)  # Set the problem for the reward function

        prompt = problem["prompt"]  # The prompt is the half written code the llm recieves
        completion = problem["canonical_solution"]  # The completion is the completed version of the code the llm returns

        # Pass both the prompt and the completion to the reward function
        result = reward_fn([prompt], [completion])

        print(f"{problem['task_id']}: reward value:", result.item())
