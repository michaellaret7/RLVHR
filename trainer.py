"""Unified RLVR trainer: the shared training loop as a class, the algorithm as a plug-in.

The four scripts in algorithms/ carry identical copies of the rollout ->
reward -> gradient loop and differ only in how rewards become *advantages*.
This file factors that split into two pieces:

  - `RLVRTrainer` — the guts. vLLM rollout generation, engine sleep/wake,
    verifiable reward scoring, the gradient pass, the REINFORCE loss, weight
    sync, and logging. Identical for every algorithm.
  - `Algorithm` — the plug-in. One method, `compute_advantages`, receives the
    full step context (a `Rollout`) and returns `(G, 1)` advantages plus any
    extra stats to log. The KL variant needs more than rewards (the policy
    log-probs and the padded batch for the reference pass), which is why the
    hook gets the whole rollout rather than just the reward vector.

The algorithms/ scripts are untouched; this is a parallel implementation for
side-by-side testing.

Run (from the repo root):
    uv run python trainer.py policy_gradient
    uv run python trainer.py reinforce
    uv run python trainer.py reinforce_kl
    uv run python trainer.py rloo
"""

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

from config import TrainingConfig
from environment.data import load_humaneval
from environment.executor import HumanEvalExecutor
from environment.reward import HumanEvalReward
from evaluation import SAMPLES_PER_PROBLEM, evaluate_model_vllm
from infra.checkpoint_s3 import save_weights_to_s3
from infra.runpod_stats import log_pod_utilization
from policy.batching import pad_batch
from policy.logprobs import compute_token_log_probs
from policy.model import device, load_model_and_tokenizer
from policy.vllm_rollout import create_llm, generate_rollouts, sync_weights

# Import vLLM only after policy.vllm_rollout has set its env-var defaults.
from vllm import LLM

logger = logging.getLogger(__name__)


@dataclass
class Rollout:
    """Everything one training step produced before the advantage computation."""

    rewards: torch.Tensor          # (G,) verifiable rewards, one per completion
    token_log_probs: torch.Tensor  # (G, T-1) policy log-probs, carries gradients
    shift_mask: torch.Tensor       # (G, T-1) 1.0 on completion tokens
    input_ids: torch.Tensor        # (G, T) padded prompt+completion rows
    attention_mask: torch.Tensor   # (G, T)
    completion_mask: torch.Tensor  # (G, T)


class Algorithm(ABC):
    """One RL algorithm = one rule for turning a rollout into advantages."""

    name: str

    @abstractmethod
    def compute_advantages(self, rollout: Rollout) -> tuple[torch.Tensor, dict[str, float]]:
        """Return advantages of shape (G, 1) and extra stats for the step log.

        Must not backprop through `rollout.token_log_probs` — anything fed into
        the advantage is a scalar weight on the loss, so detach before use.
        """


class PolicyGradient(Algorithm):
    """Vanilla policy gradient: advantage = raw reward. No baseline, no KL."""

    name = "policy_gradient"

    def compute_advantages(self, rollout: Rollout) -> tuple[torch.Tensor, dict[str, float]]:
        return rollout.rewards.unsqueeze(-1), {}


class Reinforce(Algorithm):
    """REINFORCE: subtract the batch-mean reward as a variance-reducing baseline."""

    name = "reinforce"

    def __init__(self, use_baseline: bool = True) -> None:
        self.use_baseline = use_baseline

    def compute_advantages(self, rollout: Rollout) -> tuple[torch.Tensor, dict[str, float]]:
        rewards = rollout.rewards
        advantages = rewards - rewards.mean() if self.use_baseline else rewards
        return advantages.unsqueeze(-1), {}


class RLOO(Algorithm):
    """REINFORCE Leave-One-Out: completion i's baseline is the mean of the
    *other* completions' rewards, keeping the baseline independent of i."""

    name = "rloo"

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def compute_advantages(self, rollout: Rollout) -> tuple[torch.Tensor, dict[str, float]]:
        rewards = rollout.rewards
        G = rewards.shape[0]
        if G == 1:
            # Can't leave one out of a single sample: zero advantage.
            return torch.zeros_like(rewards).unsqueeze(-1), {}
        # Leave-one-out baseline for i, vectorized: (sum - r_i) / (G - 1).
        baselines = (rewards.sum() - rewards) / (G - 1)
        advantages = rewards - baselines
        if self.normalize and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages.unsqueeze(-1), {}


class ReinforceKL(Algorithm):
    """REINFORCE with a KL penalty to a frozen reference policy (trust region).

    KL ~= sum_t (log pi_policy(token_t) - log pi_ref(token_t)) per completion;
    adjusted_reward = reward - kl_coef * KL, then the mean baseline.
    """

    name = "reinforce_kl"

    def __init__(self, ref_model, kl_coef: float, use_baseline: bool = True) -> None:
        self.ref_model = ref_model
        self.kl_coef = kl_coef
        self.use_baseline = use_baseline

    def compute_advantages(self, rollout: Rollout) -> tuple[torch.Tensor, dict[str, float]]:
        # Reference log probabilities (frozen, no gradients).
        with torch.no_grad():
            ref_log_probs, _ = compute_token_log_probs(
                self.ref_model,
                rollout.input_ids,
                rollout.attention_mask,
                rollout.completion_mask,
            )

        # Detach the policy log-probs: the KL feeds the (scalar) reward,
        # not the policy-gradient term.
        kl_per_token = (rollout.token_log_probs.detach() - ref_log_probs) * rollout.shift_mask
        kl_per_seq = kl_per_token.sum(dim=1)  # Shape: (G,)
        adjusted_rewards = rollout.rewards - self.kl_coef * kl_per_seq

        if self.use_baseline:
            advantages = adjusted_rewards - adjusted_rewards.mean()
        else:
            advantages = adjusted_rewards
        return advantages.unsqueeze(-1), {"avg KL": kl_per_seq.mean().item()}


class RLVRTrainer:
    """The shared loop: generate -> sleep engine -> reward -> gradient pass ->
    advantage (algorithm-specific) -> REINFORCE loss -> update -> sync weights."""

    def __init__(self, model, llm: LLM, tokenizer, algorithm: Algorithm,
                 reward_fn: HumanEvalReward, config: TrainingConfig) -> None:
        self.model = model
        self.llm = llm
        self.tokenizer = tokenizer
        self.algorithm = algorithm
        self.reward_fn = reward_fn
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.global_step = 0

    def train(self, train_problems: list[dict], train_prompts: list[str]) -> None:
        # Map prompt -> problem so we can set the reward's current problem.
        problem_by_prompt = {p["prompt"]: p for p in train_problems}

        for epoch in range(self.config.num_epochs):
            logger.info("EPOCH %d/%d", epoch + 1, self.config.num_epochs)
            epoch_prompts = np.random.choice(
                train_prompts, size=self.config.prompts_per_epoch, replace=False
            )
            for prompt in epoch_prompts:
                self._train_step(prompt, problem_by_prompt[prompt])

    def _train_step(self, prompt: str, problem: dict) -> None:
        config = self.config
        self.reward_fn.set_problem(problem)
        logger.info(
            "step %d | %s | generating %d completions",
            self.global_step + 1, problem["task_id"], config.generations_per_prompt,
        )

        # Step 1: Generate completions with vLLM (no gradients involved). The
        # engine holds its own copy of the policy weights, kept current by the
        # sync_weights call at the end of every step. Then free the engine's
        # GPU memory (weights + KV cache) — the gradient pass needs the room.
        all_sequences, all_completions, prompt_length = generate_rollouts(self.llm, prompt, config)
        all_prompt_lengths = [prompt_length] * config.generations_per_prompt
        self.llm.sleep(level=2)

        # Step 2: Compute rewards.
        prompts_list = [prompt] * config.generations_per_prompt
        rewards = self.reward_fn(prompts_list, all_completions).to(device).squeeze(-1)

        # Step 3: Policy log probabilities (with gradients!).
        self.model.train()
        input_ids, attention_mask, completion_mask = pad_batch(
            all_sequences, all_prompt_lengths, self.tokenizer
        )
        token_log_probs, shift_mask = compute_token_log_probs(
            self.model, input_ids, attention_mask, completion_mask
        )

        # Step 4: Advantages — the only algorithm-specific part.
        advantages, stats = self.algorithm.compute_advantages(Rollout(
            rewards, token_log_probs, shift_mask,
            input_ids, attention_mask, completion_mask,
        ))

        # Step 5: REINFORCE loss — negative because PyTorch minimises.
        loss = ((-token_log_probs * advantages) * shift_mask).sum() / shift_mask.sum()

        # Step 6: Backprop and update. Then drop this step's gradients (~3 GB)
        # now, not at the next zero_grad — the engine is about to wake up.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Step 7: Wake the engine and push the updated weights into it.
        # optimizer.step() only mutated the training copy; without this the
        # next generation would sample from a stale policy.
        sync_weights(self.llm, self.model)

        self.global_step += 1
        self._log_step(rewards, loss, stats, all_sequences, all_completions, prompt_length)

    def _log_step(self, rewards: torch.Tensor, loss: torch.Tensor, stats: dict[str, float],
                  all_sequences: list[torch.Tensor], all_completions: list[str],
                  prompt_length: int) -> None:
        extra = "".join(f" | {k} {v:.4f}" for k, v in stats.items())
        logger.info(
            "step %d | passed %d/%d | loss %.4f%s",
            self.global_step, int(rewards.sum().item()),
            self.config.generations_per_prompt, loss.item(), extra,
        )
        if self.global_step % 10 == 0:
            log_pod_utilization(self.global_step)
        if self.global_step % self.config.log_every == 0:
            avg_length = float(np.mean(
                [seq.shape[0] - prompt_length for seq in all_sequences]
            ))
            logger.info("avg completion length: %.1f tokens", avg_length)
            logger.info("example completion: %r", all_completions[0][:80])


def build_algorithm(name: str, config: TrainingConfig) -> Algorithm:
    """Construct an algorithm by name, loading any extra models it needs."""
    if name == "policy_gradient":
        return PolicyGradient()
    if name == "reinforce":
        return Reinforce(use_baseline=config.use_baseline)
    if name == "rloo":
        return RLOO(normalize=True)
    if name == "reinforce_kl":
        # Frozen reference policy = a second copy of the base model.
        ref_model, _ = load_model_and_tokenizer(config.model_name)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        return ReinforceKL(ref_model, kl_coef=config.kl_coef, use_baseline=config.use_baseline)
    raise ValueError(
        f"unknown algorithm {name!r}; expected one of: "
        "policy_gradient, reinforce, reinforce_kl, rloo"
    )


def main() -> None:
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
    load_dotenv()  # RUNPOD_API_KEY for pod-util logging, S3_* for weight upload

    algo_name = sys.argv[1] if len(sys.argv) > 1 else "policy_gradient"

    config = TrainingConfig()
    model, tokenizer = load_model_and_tokenizer(config.model_name)
    # Build the algorithm (and any reference model) before the engine claims
    # its slice of GPU memory.
    algorithm = build_algorithm(algo_name, config)
    llm = create_llm(config.model_name)  # rollout engine; same base weights
    train_problems, eval_problems, train_prompts = load_humaneval()
    reward_fn = HumanEvalReward(HumanEvalExecutor())

    trainer = RLVRTrainer(model, llm, tokenizer, algorithm, reward_fn, config)
    trainer.train(train_problems, train_prompts)

    # Persist the final policy to S3 — the trained weights die with the pod
    # otherwise. Best-effort: a failed upload shouldn't lose the eval below.
    try:
        save_weights_to_s3(model, tokenizer, algorithm=algorithm.name)
    except Exception as exc:
        logger.warning("weight upload to S3 failed: %s", exc)

    # The engine was synced after the last optimizer step, so it already holds
    # the final policy — evaluate through it.
    summary = evaluate_model_vllm(
        llm, eval_problems, reward_fn,
        num_samples=SAMPLES_PER_PROBLEM, max_new_tokens=256,
        temperature=0.7, top_p=0.9,
    )
    print(
        f"{algorithm.name.upper()} RESULTS — Pass@1: {summary['pass_at_1']:.1%} | "
        f"Pass@{SAMPLES_PER_PROBLEM}: {summary['pass_at_k']:.1%} | "
        f"Avg reward: {summary['avg_reward']:.3f}"
    )


if __name__ == "__main__":
    main()
