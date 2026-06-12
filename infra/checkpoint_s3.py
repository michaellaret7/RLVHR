"""Save / load trained model weights to S3 (rl_model_weights/).

The durable home for RL-fine-tuned weights — these don't exist on HuggingFace
and die with the pod, so this is the part of the S3 setup that actually matters.

    save_weights_to_s3(model, tokenizer, "policy_gradient")   # -> rl_model_weights/policy_gradient/<model>-<datetime>/final
    path = load_weights_from_s3("rl_model_weights/policy_gradient/<model>-<datetime>/final")
    model, tok = load_model_and_tokenizer(path)               # reload to resume / eval
"""

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

from infra.s3_utils import download_dir_from_s3, model_leaf, scratch_dir, upload_dir_to_s3

logger = logging.getLogger(__name__)

RL_WEIGHTS_ROOT = "rl_model_weights"
MODELS_DIR = Path("models")


def _resolve_bucket(bucket: str | None) -> str:
    bucket = bucket or os.environ.get("S3_BUCKET")
    if not bucket:
        raise ValueError("no bucket: set S3_BUCKET in .env or pass bucket=...")
    return bucket


def save_weights_to_s3(
    model,
    tokenizer,
    algorithm: str,
    model_name: str | None = None,
    step: int | None = None,
    bucket: str | None = None,
) -> str:
    """Save model + tokenizer with save_pretrained and upload to S3.

    Lands under ``rl_model_weights/<algorithm>/<model>-<datetime>/<step-N | final>``.
    ``model_name`` defaults to the base model the weights came from. Returns the
    S3 prefix written to; the dir is self-contained, so reloading is just
    ``from_pretrained(<download>)``.
    """
    bucket = _resolve_bucket(bucket)
    name = model_leaf(model_name or model.name_or_path)
    run = f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    leaf = f"step-{step}" if step is not None else "final"
    prefix = f"{RL_WEIGHTS_ROOT}/{algorithm}/{run}/{leaf}"

    with tempfile.TemporaryDirectory(dir=scratch_dir()) as tmp:
        logger.info("saving weights -> %s", tmp)
        model.save_pretrained(tmp)
        tokenizer.save_pretrained(tmp)
        upload_dir_to_s3(Path(tmp), bucket, prefix)
        
    return prefix


def _run_name(prefix: str) -> str:
    """Local folder name for a saved-weights prefix: the S3 run directory.

    'rl_model_weights/policy_gradient/<run>/final' -> '<run>'.
    """
    parts = prefix.strip("/").split("/")
    leaf = parts[-1]
    return parts[-2] if leaf == "final" or leaf.startswith("step-") else leaf


def load_weights_from_s3(
    prefix: str,
    dest: str | None = None,
    bucket: str | None = None,
) -> str:
    """Download a saved-weights prefix from S3 into ``models/<run-name>``.

    Skips the download if the weights are already there. Returns the local
    path — hand it straight to ``load_model_and_tokenizer``.
    """
    bucket = _resolve_bucket(bucket)
    dest = Path(dest) if dest else MODELS_DIR / _run_name(prefix)
    if dest.is_dir() and any(dest.iterdir()):
        logger.info("weights already at %s, skipping download", dest)
        return str(dest)
    download_dir_from_s3(bucket, prefix, dest)
    logger.info("weights ready at %s", dest)
    return str(dest)
