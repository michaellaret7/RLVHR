"""vLLM rollout generation + weight sync for the training loop.

The training model and the vLLM engine hold two separate copies of the policy
weights: `optimizer.step()` only updates the training copy. After every update
the new weights must be pushed into the engine (`sync_weights`) before the next
rollout, or generation drifts off-policy.

The engine runs inside the training process (no worker subprocess), so the
sync is a direct same-GPU copy — ~15 ms for the 1.5B model.

Both copies plus gradients, Adam moments, and activations don't fit on a 24 GB
card at once, so the engine uses vLLM sleep mode: `llm.sleep(level=2)` discards
its weights and KV cache before the gradient pass, and `sync_weights` wakes it
and reloads the current policy. The cycle per training step:

    generate -> llm.sleep(2) -> backward/optimizer.step() -> sync_weights -> ...
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# FlashInfer's JIT sampler needs ninja + a CUDA toolchain, absent on this box.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# Keep the engine in this process so sync_weights can hand it live GPU tensors.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# The engine sleeps/wakes every training step; at INFO vLLM logs ~6 lines per
# step about it. Keep only warnings.
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import logging

import torch
from vllm import LLM, SamplingParams

# vLLM warns on every first-time Triton kernel JIT compile ("latency spike").
# Irrelevant for a training loop and there's no env var for it; silence the one
# logger it uses.
logging.getLogger("vllm.triton_utils.jit_monitor").setLevel(logging.ERROR)

from config import TrainingConfig
from policy.model import MODEL_NAME, device


def create_llm(gpu_memory_utilization: float = 0.2) -> LLM:
    """Rollout engine, memory-capped to fit beside the training model,
    gradients, optimizer state, and activations on one GPU."""
    return LLM(
        model=MODEL_NAME,
        max_model_len=2048,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,  # no CUDA graphs: saves memory
        enable_sleep_mode=True,  # lets the engine vacate the GPU during the gradient pass
    )


def generate_rollouts(
    llm: LLM, 
    prompt: str, 
    config: TrainingConfig,
) -> tuple[list[torch.Tensor], list[str], int]:
    """Sample G completions for one prompt.

    Returns the same shapes the HF-generate block produced: full
    prompt+completion token rows (for the gradient pass), decoded completion
    texts (for scoring), and the prompt length in tokens.
    """

    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_new_tokens,
        n=config.generations_per_prompt,
    )

    output = llm.generate([prompt], sampling_params, use_tqdm=False)[0]

    prompt_ids = list(output.prompt_token_ids)
    eos_id = llm.get_tokenizer().eos_token_id

    all_sequences = []

    for o in output.outputs:
        completion_ids = list(o.token_ids)
        # vLLM strips the EOS token; restore it (HF generate keeps it) so the
        # policy still gets gradient signal on when to stop.
        if o.finish_reason == "stop":
            completion_ids.append(eos_id)

        all_sequences.append(
            torch.tensor(prompt_ids + completion_ids, device=device)
        )

    all_completions = [o.text for o in output.outputs]

    return all_sequences, all_completions, len(prompt_ids)


def sync_weights(llm: LLM, model) -> None:
    """Wake the sleeping engine and push the training model's current weights.

    Call after `optimizer.step()`. The engine slept through the gradient pass
    (its weights were discarded), so waking it and loading the current policy
    is one and the same step.
    """
    torch.cuda.empty_cache()  # release cached training blocks to the engine
    llm.wake_up()

    weights = list(model.named_parameters())

    llm.apply_model(lambda m: m.load_weights(weights))
    
    llm.reset_prefix_cache()  # KV cache from the old weights must not be reused


if __name__ == "__main__":
    from policy.model import load_model_and_tokenizer

    config = TrainingConfig()
    llm = create_llm()
    model, _ = load_model_and_tokenizer()

    sequences, completions, prompt_length = generate_rollouts(
        llm, "def fibonacci(n):", config
    )
    print(f"prompt_length={prompt_length}, G={len(sequences)}")
    print("first completion:", completions[0][:120])

    # Exercise the full training-step cycle: sleep -> sync (wake+load) -> generate.
    llm.sleep(level=2)
    sync_weights(llm, model)
    _, completions, _ = generate_rollouts(llm, "def is_even(n):", config)
    print("post-sync completion:", completions[0][:80])
    print("sleep/sync/generate cycle ok")
