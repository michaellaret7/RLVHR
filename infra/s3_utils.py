"""Shared S3 helpers for the model-weight scripts.

Single source of truth for the boto3 client, directory transfer, and the
``stock_models/<name>`` bucket-path convention. Works with AWS S3 or any
S3-compatible store (R2 / B2 / RunPod) via the ``S3_ENDPOINT_URL`` env var.
"""

import logging
import os
from pathlib import Path

import boto3

logger = logging.getLogger(__name__)


def s3_client():
    """boto3 S3 client; honors S3_ENDPOINT_URL for R2 / B2 / RunPod."""
    return boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None)


def scratch_dir() -> str | None:
    """Big-disk scratch location for temp model files.

    "/" is small on RunPod; prefer $SCRATCH_DIR, else the /workspace volume,
    else fall back to the system tmp (None lets tempfile decide).
    """
    return os.environ.get("SCRATCH_DIR") or ("/workspace" if Path("/workspace").is_dir() else None)


def model_leaf(model: str) -> str:
    """Lowercased leaf name of a model id / path / URL.

    "Qwen/Qwen2.5-Coder-1.5B-Instruct" -> "qwen2.5-coder-1.5b-instruct".
    Accepts a repo id, a huggingface.co URL, or a bare name.
    """
    return model.split("huggingface.co/")[-1].strip("/").split("/")[-1].lower()


def upload_dir_to_s3(local_dir: Path, bucket: str, prefix: str) -> None:
    """Upload every file under local_dir to s3://bucket/prefix/."""
    s3 = s3_client()
    files = [p for p in local_dir.rglob("*") if p.is_file()]
    for path in files:
        key = f"{prefix}/{path.relative_to(local_dir)}"
        logger.info("uploading %s -> s3://%s/%s", path.name, bucket, key)
        s3.upload_file(str(path), bucket, key)
    logger.info("uploaded %d files to s3://%s/%s/", len(files), bucket, prefix)


def download_dir_from_s3(bucket: str, prefix: str, local_dir: Path) -> None:
    """Download every object under s3://bucket/prefix/ into local_dir.

    Raises if the prefix is empty so a typo'd path fails fast instead of
    silently producing an empty directory.
    """
    s3 = s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            dest = local_dir / key[len(prefix):].lstrip("/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            logger.info("downloading s3://%s/%s -> %s", bucket, key, dest)
            s3.download_file(bucket, key, str(dest))
            count += 1
    if count == 0:
        raise RuntimeError(f"no objects under s3://{bucket}/{prefix}")
    logger.info("downloaded %d files to %s", count, local_dir)
