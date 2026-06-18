"""REINFORCE with a KL penalty to a frozen reference policy.

From the article's "KL-divergence" section. Unconstrained reward maximisation
drifts away from the pre-trained distribution (reward hacking, mode collapse). A
KL penalty against a frozen reference model acts as a trust region.

The article gives this loop in prose (no full code), so this is the minimal
faithful implementation of the steps it lists:

  - Generate N completions with the current policy.
  - Score them with the verifiable reward.
  - For each completion, compute its KL divergence from the reference policy.
  - Adjust rewards: adjusted = reward - kl_coef * KL.
  - Compute baseline and advantage from the *adjusted* rewards.
  - loss = -log_prob * advantage; backprop.

KL here is the standard per-token estimate summed over completion tokens:
    KL ~= sum_t (log pi_policy(token_t) - log pi_ref(token_t))

Run:
    uv run python algorithms/reinforce_kl.py
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


def train(model, ref_model, llm, tokenizer, train_problems, train_prompts, reward_fn, config):
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
            # by the sync_weights call after each optimizer step (Step 8).
            all_sequences, all_completions, prompt_length = generate_rollouts(llm, prompt, config)
            all_prompt_lengths = [prompt_length] * config.generations_per_prompt

            # Free the engine's GPU memory (weights + KV cache) — the gradient
            # pass below needs the room. sync_weights wakes it back up.
            llm.sleep(level=2)

            # Step 2: Compute rewards
            prompts_list = [prompt] * config.generations_per_prompt # Create a list of the same prompt for each generation
            rewards = reward_fn(prompts_list, all_completions).to(device) # Compute the rewards for each generation
            rewards = rewards.squeeze(-1)  # Shape: (G,)

            # Build the padded batch once; reused for policy and reference passes.
            input_ids, attention_mask, completion_mask = pad_batch(
                all_sequences,
                all_prompt_lengths,
                tokenizer
            )

            # Step 3: Log probabilities under the policy (with gradients!)
            # Set the model to training mode
            model.train()

            token_log_probs, shift_mask = compute_token_log_probs(
                model,
                input_ids,
                attention_mask,
                completion_mask
            )

            # Reference log probabilities (frozen, no gradients).
            with torch.no_grad():
                ref_log_probs, _ = compute_token_log_probs(
                    ref_model, input_ids, attention_mask, completion_mask
                )

            # Step 4: Per-completion KL, then adjust rewards by the KL penalty.
            # Detach the policy log-probs here: the KL feeds the (scalar) reward,
            # not the policy-gradient term.
            kl_per_token = (token_log_probs.detach() - ref_log_probs) * shift_mask
            kl_per_seq = kl_per_token.sum(dim=1)  # Shape: (G,)
            adjusted_rewards = rewards - config.kl_coef * kl_per_seq

            # Step 5: Baseline + advantage from the adjusted rewards.
            if config.use_baseline:
                advantages = adjusted_rewards - adjusted_rewards.mean()
            else:
                advantages = adjusted_rewards
            advantages = advantages.unsqueeze(-1)  # Shape: (G, 1)

            # Step 6: REINFORCE loss with advantage — negative because PyTorch minimises.
            loss_per_token = (-token_log_probs * advantages) * shift_mask
            loss = loss_per_token.sum() / shift_mask.sum() # This is the number that is used to backpropagate

            # Step 7: Backprop and update
            optimizer.zero_grad() # wipe the gradients from the previous step
            loss.backward() # compute the gradients
            optimizer.step() # apply the gradients to the model

            # Drop this step's gradients (~3 GB) now, not at the next zero_grad —
            # the engine is about to wake up and needs the memory.
            optimizer.zero_grad(set_to_none=True)

            # Step 8: Wake the engine and push the updated weights into it.
            # optimizer.step() only mutated the training copy; without this the
            # next generation would sample from a stale policy.
            sync_weights(llm, model)

            logger.info(
                "step %d | passed %d/%d | loss %.4f | avg KL %.4f",
                global_step + 1,
                int(rewards.sum().item()),
                config.generations_per_prompt,
                loss.item(),
                kl_per_seq.mean().item(),
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
    # Frozen reference policy = a second copy of the base model.
    ref_model, _ = load_model_and_tokenizer(config.model_name)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    llm = create_llm(config.model_name)  # rollout engine; same base weights
    train_problems, eval_problems, train_prompts = load_humaneval()
    reward_fn = HumanEvalReward(HumanEvalExecutor())

    train(model, ref_model, llm, tokenizer, train_problems, train_prompts, reward_fn, config)

    # Persist the final policy to S3 — the trained weights die with the pod
    # otherwise. Best-effort: a failed upload shouldn't lose the eval below.
    try:
        save_weights_to_s3(model, tokenizer, algorithm="reinforce_kl")
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
        f"REINFORCE + KL RESULTS — Pass@1: {summary['pass_at_1']:.1%} | "
        f"Pass@{SAMPLES_PER_PROBLEM}: {summary['pass_at_k']:.1%} | "
        f"Avg reward: {summary['avg_reward']:.3f}"
    )
