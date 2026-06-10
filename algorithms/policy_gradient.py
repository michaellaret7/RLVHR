"""Vanilla Policy Gradient.

From the article's "Policy Gradient Optimisation" section. The simplest recipe:

    loss = -log_prob * reward

For each prompt: generate G completions, score them with the verifiable reward,
compute token log-probs, and take a gradient step. No baseline, no KL.

Run:
    uv run python algorithms/policy_gradient.py
"""

import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TrainingConfig
from environment.data import load_humaneval
from evaluation import SAMPLES_PER_PROBLEM, evaluate_model, print_summary
from policy.batching import pad_batch
from policy.logprobs import compute_token_log_probs
from policy.model import device, load_model_and_tokenizer
from environment.executor import HumanEvalExecutor
from environment.reward import HumanEvalReward

logger = logging.getLogger(__name__)


def train(model, tokenizer, train_problems, train_prompts, reward_fn, config):
    # Initialize the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.learning_rate
    )

    # Map prompt -> problem so we can set the reward's current problem.
    problem_by_prompt = {p["prompt"]: p for p in train_problems}

    global_step = 0

    # Begin the epoch loop for the number of epochs specified in the config
    for epoch in range(config.num_epochs):
        logger.info("EPOCH %d/%d", epoch + 1, config.num_epochs)

        # Randomly choose a set of prompts for the epoch at hand
        epoch_prompts = np.random.choice(
            train_prompts, 
            size=config.prompts_per_epoch, 
            replace=False
        )

        # Loop through each prompt in the randomly chosen set of prompts for the epoch
        # The model gets the weights bumped once per prompt 
        # So the model creates x amount of answers for each prompts and the model weights are bumped for each answer
        for prompt in epoch_prompts:
            # Set the problem for the reward function class to the current prompt
            problem = problem_by_prompt[prompt]
            reward_fn.set_problem(problem)

            logger.info(
                "step %d | %s | generating %d completions",
                global_step + 1, problem["task_id"], config.generations_per_prompt,
            )

            # Step 1: Generate completions (no gradients)
            # Set the model to evaluation mode
            model.eval()

            with torch.no_grad():
                # Tokenize the prompt
                prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=False)
                prompt_ids = prompt_tokens.input_ids.to(device)
                prompt_mask = prompt_tokens.attention_mask.to(device)
                prompt_length = prompt_ids.shape[1]

                # Generate all G completions in one batched call. The prompt is
                # identical across samples, so we just ask for G sequences —
                # far faster than calling generate G times in a loop.
                outputs = model.generate(
                    prompt_ids,
                    attention_mask=prompt_mask,  # explicit: pad token == EOS, so it can't be inferred
                    max_new_tokens=config.max_new_tokens,
                    do_sample=True, # sample from the model so the model can explore and learn
                    temperature=config.temperature,
                    top_p=config.top_p,
                    num_return_sequences=config.generations_per_prompt, # number of completions to generate for each prompt
                    pad_token_id=tokenizer.eos_token_id,
                )

                # outputs: (G, prompt_length + new_tokens), all rows same length.
                all_sequences = list(outputs)                  # prompt + answer token IDs (to re-run for gradients)
                all_completions = tokenizer.batch_decode(       # answer only, as text (for scoring)
                    outputs[:, prompt_length:], skip_special_tokens=True
                )
                all_prompt_lengths = [prompt_length] * config.generations_per_prompt

            # Step 2: Compute rewards
            prompts_list = [prompt] * config.generations_per_prompt # Create a list of the same prompt for each generation
            rewards = reward_fn(prompts_list, all_completions).to(device) # Compute the rewards for each generation

            # Step 3: Compute log probabilities (with gradients!)
            # Set the model to training mode
            model.train()

            input_ids, attention_mask, completion_mask = pad_batch(
                all_sequences,
                all_prompt_lengths,
                tokenizer
            )

            token_log_probs, shift_mask = compute_token_log_probs(
                model, 
                input_ids, 
                attention_mask, 
                completion_mask
            )
            
            # Step 4: REINFORCE loss — negative because PyTorch minimises.
            loss_per_token = (-token_log_probs * rewards) * shift_mask
            loss = loss_per_token.sum() / shift_mask.sum() # This is the number that is used to backpropagate

            # Step 5: Backprop and update
            optimizer.zero_grad() # wipe the gradients from the previous step
            loss.backward() # compute the gradients
            optimizer.step() # apply the gradients to the model

            logger.info(
                "step %d | passed %d/%d | loss %.4f",
                global_step + 1,
                int(rewards.sum().item()),
                config.generations_per_prompt,
                loss.item(),
            )

            # Logging
            global_step += 1

            if global_step % config.log_every == 0:
                avg_length = (
                    (outputs[:, prompt_length:] != tokenizer.eos_token_id)
                    .sum(dim=1).float().mean().item()
                )
                logger.info("avg completion length: %.1f tokens", avg_length)
                logger.info("example completion: %r", all_completions[0][:80])


if __name__ == "__main__":
    # INFO = the training narrative (2 lines per step). Switch to DEBUG to also
    # see the inner stages (code extraction, test execution, log-prob scoring).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Hugging Face libraries log every HTTP request at INFO; keep them quiet.
    logging.getLogger("httpx").setLevel(logging.WARNING)

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
