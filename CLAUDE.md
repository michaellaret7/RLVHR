# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch, runnable companion to the article `rl_article.md` (*Fine-tuning LLMs using Reinforcement Learning*, Janu Verma). It fine-tunes `Qwen/Qwen2.5-Coder-1.5B-Instruct` to pass HumanEval+ unit tests using verifiable, test-based rewards (RLVR). 

## Commands

Dependency management is [uv](https://docs.astral.sh/uv/). Everything runs from the repo root.

```bash
uv sync                                         # create env, install torch (CUDA 12.8) / transformers / datasets
uv run python evaluation.py                     # baseline pass@1 / pass@5
uv run python algorithms/policy_gradient.py     # vanilla policy gradient
uv run python algorithms/reinforce.py           # REINFORCE + mean baseline
uv run python algorithms/reinforce_kl.py        # REINFORCE + KL penalty
uv run python algorithms/rloo.py                # RLOO
```

Every module is also runnable directly as a `__main__` smoke/demo of just that piece (e.g. `uv run python environment/reward.py`, `uv run python environment/extraction.py`). Use these to exercise one component in isolation.

There is **no test suite, linter, or formatter** configured — the `__main__` demos are the only verification harness. Hyperparameters in `config.py` are intentionally tiny for fast iteration, so absolute result numbers vary run-to-run.

First run downloads the model (~3 GB) and dataset. GPU strongly recommended; falls back to CPU (slow). `torch` is pinned to PyTorch's CUDA 12.8 wheel index in `pyproject.toml` because the default Windows PyPI wheel is CPU-only.

## Architecture

This is a **flat research layout, not an installable package**. The `algorithms/` scripts add the repo root to `sys.path` at import time (`sys.path.insert(0, ...)`) so they can `from config import ...` etc. — this is why they must be run from the root, and why there's no `pip install -e`.

The directory structure mirrors the RL agent–environment loop:

- **`policy/`** — the agent (the LLM and how we read it)
  - `model.py` — loads Qwen base model + tokenizer (`load_model_and_tokenizer`), exposes the shared `device`. Reuses EOS as pad token.
  - `logprobs.py` — `compute_token_log_probs`: shift → log_softmax → gather. Returns `(token_log_probs, shift_mask)`. **Every algorithm reuses this** as the gradient-carrying core.
- **`environment/`** — the world (the task and how a completion is scored)
  - `data.py` — loads `evalplus/humanevalplus` (HumanEval+; ~80× more tests than original HumanEval, harder to reward-hack), splits 50 train / 30 eval. Records have keys `task_id, prompt, test, entry_point, canonical_solution`.
  - `extraction.py` — `extract_code_with_prompt`: regex extractor pulling a runnable `def entry_point` out of `prompt + completion` (fenced block → stitched prompt+body → `None`).
  - `executor.py` — `HumanEvalExecutor`: runs extracted code + the problem's `test` harness via `exec()` in a **separate process with a timeout** (default 5s) so bad generations can't hang training. Calls `check(candidate)` on the entry-point function.
  - `reward.py` — `HumanEvalReward`: the verifiable reward (1.0 pass / 0.0 fail), returned as a `(G, 1)` tensor.
- **`config.py`** — `TrainingConfig` dataclass: generation + training hyperparameters, baseline/KL toggles.
- **`evaluation.py`** — `evaluate_model` (pass@1 / pass@k), `print_summary`. Imported by every training script for the final results block.

### The training-loop pattern (shared across all four algorithms)

Each `algorithms/*.py` currently carries its own copy of the `train()` loop and `_pad_batch()` helper; they differ only in the advantage step. The common loop per prompt:

1. Generate `G` completions with `model.generate` under `no_grad` (`model.eval()`).
2. Score with `reward_fn` → rewards `(G,)`. **`reward_fn` is stateful — call `reward_fn.set_problem(problem)` before invoking it.**
3. Turn rewards into **advantages** (this is the only real difference between algorithms).
4. Re-run the batch through `compute_token_log_probs` **with** gradients (`model.train()`).
5. `loss = (-token_log_probs * advantages * shift_mask).sum() / shift_mask.sum()`; backprop.

The advantage step is what each file changes:
- `policy_gradient.py` — `advantage = reward` (no baseline).
- `reinforce.py` — subtract batch-mean baseline.
- `reinforce_kl.py` — subtract a per-completion KL penalty vs a **frozen reference model** (second copy of the base model, `requires_grad_(False)`) from the reward, *then* baseline. KL uses `token_log_probs.detach()` so it feeds only the scalar reward, not the policy-gradient term.
- `rloo.py` — `compute_rloo_advantages`: each completion's baseline is the mean of the *other* completions' rewards (leave-one-out), then normalized.

When editing one algorithm, remember the change usually belongs only in the advantage computation; the surrounding loop should stay identical to the others.

## Development Guidelines

### Core Philosophy

- **KISS** — choose straightforward solutions; simple is easier to maintain and debug.
- **YAGNI** — implement only what's needed now, not what might be useful later.
- **DRY** — single source of truth for every piece of knowledge. Search for an existing helper before writing a new one; extract shared logic into pure reusable functions.

#### 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

#### 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
- Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

#### 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

#### 4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

### Design Principles

- **Dependency Inversion** — high-level modules depend on abstractions, not low-level modules.
- **Open/Closed** — open for extension, closed for modification.
- **Single Responsibility** — one clear purpose per function/class/module.
- **Fail Fast** — validate early, raise immediately when something's wrong.
- **Type safety** — type hints and explicit return types are mandatory; the codebase should read as self-documenting.
- **Resource efficiency** — context managers for all I/O; vectorize data-heavy work.

### Code Constraints

- Files: max 500 lines — split into modules if approaching the limit.
- Functions: max 50 lines, single responsibility.
- Classes: max 100 lines, one concept.
- Group code by feature/responsibility.

# Question Answering 

- When answering a question always answer concisely, simply, briefly, and from first principles unless told otherwise
- Provide examples as well
- Your goal is to answer the questions with as few tokens as possible unless told otherwise