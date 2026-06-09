"""Training configuration.

From the article's "Policy Gradient Optimisation" section. Kept small for fast
experimentation on a single GPU / modest hardware.
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Generation settings
    max_new_tokens: int = 40
    temperature: float = 0.8
    top_p: float = 0.9

    # Training settings
    num_epochs: int = 1
    prompts_per_epoch: int = 2      # Small for fast experimentation
    generations_per_prompt: int = 2  # Sample G completions per prompt
    learning_rate: float = 1e-6

    # Baseline (REINFORCE) — used by algorithms that subtract a baseline
    use_baseline: bool = True

    # KL penalty (REINFORCE + KL) — used by reinforce_kl.py
    kl_coef: float = 0.1

    # Logging
    log_every: int = 5
