# RLVR — Fine-tuning LLMs with REINFORCE

A from-scratch, runnable companion to the article
[*Fine-tuning LLMs using Reinforcement Learning*](rl_article.md) (Janu Verma).
It fine-tunes `Qwen/Qwen2.5-Coder-1.5B-Instruct` to write Python that passes
HumanEval+ unit tests, using **verifiable, test-based rewards** (RLVR).
(HumanEval+ has ~80× more tests per problem than the original HumanEval, which
makes the reward much harder to hack; switch `DATASET_ID` in
`environment/data.py` to use the original.)

Each concept from the article lives in its own file so you can read, run, and
modify them in isolation.

## Setup

Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync           # create the env and install torch / transformers / datasets
```

Run everything from the repo root (it's a flat research layout, not a package).

## What to run

| Command | Reproduces | Article section |
| --- | --- | --- |
| `uv run python evaluation.py`                | Baseline pass@1 / pass@5 | Base Model Performance |
| `uv run python algorithms/policy_gradient.py`| Vanilla policy gradient  | Policy Gradient Optimisation |
| `uv run python algorithms/reinforce.py`      | REINFORCE + baseline     | REINFORCE |
| `uv run python algorithms/reinforce_kl.py`   | REINFORCE + KL penalty   | KL-divergence |
| `uv run python algorithms/rloo.py`           | RLOO                     | REINFORCE Leave-One-Out |

Each training script trains, then evaluates and prints a results block matching
the article's tables.

> The first run downloads the model (~3 GB) and the HumanEval+ dataset. A GPU is
> strongly recommended; the code falls back to CPU but will be slow.

## Layout

Folders mirror the RL agent–environment loop the article describes:

```
config.py            TrainingConfig (generation + training hyperparameters)
evaluation.py        evaluate_model, pass@1 / pass@k

environment/         "the world": the task and how a completion is scored
  data.py            load HumanEval+, 50 train / 30 eval split
  extraction.py      extract_code_with_prompt  (regex code extractor)
  executor.py        HumanEvalExecutor         (sandboxed exec + timeout)
  reward.py          HumanEvalReward           (pass/fail -> reward tensor)

policy/              "the agent": the LLM and how we read it
  model.py           load Qwen base model + tokenizer
  logprobs.py        per-token log-probs (shift -> log_softmax -> gather)
  batching.py        pad_batch: pad generations + build the masks logprobs needs

algorithms/          "the learning rule": one self-contained training loop each
  policy_gradient.py loss = -log_prob * reward
  reinforce.py       + mean baseline -> advantage
  reinforce_kl.py    + KL penalty vs a frozen reference model
  rloo.py            leave-one-out baseline
```

## The algorithm progression

Each file is a small, readable delta on the previous one:

1. **Policy gradient** — `loss = -log_prob * reward`. Simple but high variance.
2. **REINFORCE** — subtract a batch-mean **baseline** to get the *advantage*:
   `loss = -log_prob * advantage`. Lower variance, more stable.
3. **REINFORCE + KL** — add a **KL penalty** to a frozen reference model so the
   policy doesn't drift / reward-hack away from fluent language.
4. **RLOO** — compute each completion's baseline from the *other* completions
   (leave-one-out), reducing variance further.

## Notes on fidelity

The code mirrors the article. A few helpers the article names but doesn't print
in full (`extract_code_with_prompt`, `HumanEvalExecutor`, and the KL training
loop body) are implemented as the minimal version the article describes —
regex extraction, `exec()` in a separate process with a timeout, and a KL
penalty subtracted from the reward. Two deliberate departures: the dataset is
HumanEval+ rather than the original HumanEval (same problems, ~80× more tests,
so the reward is harder to hack), and the `G` completions per prompt are
generated in one batched `model.generate` call instead of a Python loop.
Hyperparameters are intentionally tiny (`config.py`) for fast experimentation,
so absolute numbers will vary from run to run.
