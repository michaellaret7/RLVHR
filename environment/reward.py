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
        self.executor = executor
        self.current_problem = None

    def set_problem(self, problem):
        self.current_problem = problem

    def __call__(self, prompts: List[str], completions: List[str]):
        if self.current_problem is None:
            raise ValueError("Must set problem first")

        rewards = []

        for prompt, completion in zip(prompts, completions):
            code = extract_code_with_prompt(
                prompt,
                completion,
                self.current_problem["entry_point"],
            )
            if code is None:
                rewards.append(0.0)
                continue
            # Execute test
            passed, error = self.executor.execute_test(
                code,
                self.current_problem["test"],
                self.current_problem["entry_point"],
            )
            reward = 1.0 if passed else 0.0
            rewards.append(reward)
            
        return torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
