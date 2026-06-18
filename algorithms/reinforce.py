"""REINFORCE with a mean baseline.

From the article's "REINFORCE" section. Same loop as vanilla policy gradient,
but we subtract a baseline (the batch mean reward) from the rewards to get the
*advantage*, then use:

    loss = -log_prob * advantage

Subtracting a state-dependent baseline reduces the variance of the gradient
estimate, giving more stable learning. This is the only change from
policy_gradient.py (see "Step 3: NEW" below).

Run:
    uv run python algorithms/reinforce.py
"""

import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TrainingConfig
from environment.data import load_humaneval
from evaluation import SAMPLES_PER_PROBLEM, evaluate_model_vllm
from policy.batching import pad_batch
from policy.logprobs import compute_token_log_probs
from policy.model import device, load_model_and_tokenizer
from policy.vllm_rollout import create_llm, generate_rollouts, sync_weights
from environment.executor import HumanEvalExecutor
from environment.reward import HumanEvalReward
from infra.checkpoint_s3 import save_weights_to_s3
from infra.runpod_stats import log_pod_utilization

logger = logging.getLogger(__name__)


def train(model, llm, tokenizer, train_problems, train_prompts, reward_fn, config):
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
        # These will be the prompts that the model will be trained on for this epoch
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
            problem = problem_by_prompt[prompt] # pull the problem from the problem dict
            reward_fn.set_problem(problem) # set the problem in the rewaerd func class

            logger.info(
                "step %d | %s | generating %d completions",
                global_step + 1, problem["task_id"], config.generations_per_prompt,
            )

            # Step 1: Generate completions with vLLM (no gradients involved).
            # The engine holds its own copy of the policy weights, kept current
            # by the sync_weights call after each optimizer step (Step 7).
            all_sequences, all_completions, prompt_length = generate_rollouts(llm, prompt, config)
            all_prompt_lengths = [prompt_length] * config.generations_per_prompt

            # Free the engine's GPU memory (weights + KV cache) — the gradient
            # pass below needs the room. sync_weights wakes it back up.
            llm.sleep(level=2)

            # Step 2: Compute rewards
            prompts_list = [prompt] * config.generations_per_prompt # Create a list of the same prompt for each generation
            rewards = reward_fn(prompts_list, all_completions).to(device) # Compute the rewards for each generation
            rewards = rewards.squeeze(-1)  # Shape: (G,)

            # Step 3: NEW — compute advantages with baseline subtraction
            if config.use_baseline:
                baseline_value = rewards.mean()
                advantages = rewards - baseline_value
            else:
                advantages = rewards
                baseline_value = 0.0
            advantages = advantages.unsqueeze(-1)  # Shape: (G, 1)

            # Step 4: Compute log probabilities (with gradients!)
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

            # Step 5: REINFORCE loss with advantage — negative because PyTorch minimises.
            loss_per_token = (-token_log_probs * advantages) * shift_mask
            loss = loss_per_token.sum() / shift_mask.sum() # This is the number that is used to backpropagate

            # Step 6: Backprop and update
            optimizer.zero_grad() # wipe the gradients from the previous step
            loss.backward() # compute the gradients
            optimizer.step() # apply the gradients to the model

            # Drop this step's gradients (~3 GB) now, not at the next zero_grad —
            # the engine is about to wake up and needs the memory.
            optimizer.zero_grad(set_to_none=True)

            # Step 7: Wake the engine and push the updated weights into it.
            # optimizer.step() only mutated the training copy; without this the
            # next generation would sample from a stale policy.
            sync_weights(llm, model)

            logger.info(
                "step %d | passed %d/%d | loss %.4f",
                global_step + 1,
                int(rewards.sum().item()),
                config.generations_per_prompt,
                loss.item(),
            )

            # Logging
            global_step += 1

            if global_step % 10 == 0:
                log_pod_utilization(global_step)

            if global_step % config.log_every == 0:
                avg_length = float(np.mean(
                    [seq.shape[0] - prompt_length for seq in all_sequences]
                ))
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

    from dotenv import load_dotenv
    load_dotenv()  # RUNPOD_API_KEY for pod-util logging, S3_* for weight upload\

    config = TrainingConfig()
    model, tokenizer = load_model_and_tokenizer(config.model_name)
    llm = create_llm(config.model_name)  # rollout engine; same base weights
    train_problems, eval_problems, train_prompts = load_humaneval()
    reward_fn = HumanEvalReward(HumanEvalExecutor())

    train(model, llm, tokenizer, train_problems, train_prompts, reward_fn, config)

    # Persist the final policy to S3 — the trained weights die with the pod
    # otherwise. Best-effort: a failed upload shouldn't lose the eval below.
    try:
        save_weights_to_s3(model, tokenizer, algorithm="reinforce")
    except Exception as exc:
        logger.warning("weight upload to S3 failed: %s", exc)

    # The engine was synced after the last optimizer step, so it already holds
    # the final policy — evaluate through it.
    summary = evaluate_model_vllm(
        llm,
        eval_problems,
        reward_fn,
        num_samples=SAMPLES_PER_PROBLEM,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
    )
    print(
        f"REINFORCE RESULTS — Pass@1: {summary['pass_at_1']:.1%} | "
        f"Pass@{SAMPLES_PER_PROBLEM}: {summary['pass_at_k']:.1%} | "
        f"Avg reward: {summary['avg_reward']:.3f}"
    )
