"""Test-based reward function.

Reproduces `HumanEvalReward` from the article's "Base Model Performance" section.
This is the verifiable reward of RLVR: 1.0 if the generated code passes the
problem's unit tests, 0.0 otherwise.
"""

from typing import List

import torch

from extraction import extract_code_with_prompt


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

        # Initialize empty rewards list
        rewards = []

        # Loop through each prompt and completion pair 
        # The prompt is the half written code the llm recieves 
        # The completion is the comlpleted version of the code the llm returns
        for prompt, completion in zip(prompts, completions):
            # Extract the code from the prompt and completion and pass it to the code extractor
            # This forms the full code snippet that the llm returned
            code = extract_code_with_prompt(
                prompt,
                completion,
                self.current_problem["entry_point"],
            )

            # If the code is None, then the llm didnt return any code
            # So we give it a reward of 0.0
            if code is None:
                rewards.append(0.0)
                continue

            # Execute test using the executor
            # This tests the code snippet against the test harness
            # Pass the code returned from the llm and the test func and the name of the function to the test executor
            passed, error = self.executor.execute_test(
                code,
                self.current_problem["test"],
                self.current_problem["entry_point"],
            )

            # If the code passed the test, then we give it a reward of 1.0
            # Otherwise we give it a reward of 0.0
            reward = 1.0 if passed else 0.0

            # Add the reward to the rewards list
            rewards.append(reward)
            
        return torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)


if __name__ == "__main__":
    # Demo: run a few fake "model" completions through the reward function and
    # watch the verifiable reward come out (1.0 = passed the tests, 0.0 = failed).
    from data import load_humaneval
    from executor import HumanEvalExecutor

    # Load the humaneval problems
    _, eval_problems, _ = load_humaneval()
    problem = eval_problems[0]
  
    # Set the reward func with the code executor cls
    reward_fn = HumanEvalReward(HumanEvalExecutor())
    reward_fn.set_problem(problem)  # Set the problem for the reward function

    prompt = problem["prompt"] # The prompt is the half written code the llm recieves
    completion = problem["canonical_solution"]  # The completion is the completed version of the code the llm returns

    # Pass both the prompt and the completion to the reward function
    result = reward_fn([prompt], [completion])

    # Print the result from the reward func
    print(result)                       # tensor([[1.]]) # The reward value is 1.0 because the code passed the test
    print("reward value:", result.item())  # 1.0
