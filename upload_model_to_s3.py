"""Download a Hugging Face model and upload its weights to S3.

Usage:
    uv run python upload_model_to_s3.py Qwen/Qwen2.5-Coder-1.5B-Instruct --bucket my-bucket

Accepts either a repo id ("Qwen/Qwen2.5-Coder-1.5B-Instruct") or a full
huggingface.co URL. Uploads to s3://<bucket>/base/<model-name>/ by default.
"""

import argparse
import logging
import tempfile
from pathlib import Path

import boto3
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def parse_repo_id(model: str) -> str:
    """Turn a repo id or huggingface.co URL into a plain repo id."""
    return model.split("huggingface.co/")[-1].strip("/")


def upload_dir_to_s3(local_dir: Path, bucket: str, prefix: str) -> None:
    """Upload every file under local_dir to s3://bucket/prefix/."""
    s3 = boto3.client("s3")
    files = [p for p in local_dir.rglob("*") if p.is_file()]
    for path in files:
        key = f"{prefix}/{path.relative_to(local_dir)}"
        logger.info("uploading %s -> s3://%s/%s", path.name, bucket, key)
        s3.upload_file(str(path), bucket, key)
    logger.info("uploaded %d files to s3://%s/%s/", len(files), bucket, prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="HF repo id or huggingface.co URL")
    parser.add_argument("--bucket", required=True, help="target S3 bucket")
    parser.add_argument("--prefix", default=None, help="S3 key prefix (default: base/<model-name>)")
    args = parser.parse_args()

    repo_id = parse_repo_id(args.model)
    prefix = args.prefix or f"base/{repo_id.split('/')[-1].lower()}"

    with tempfile.TemporaryDirectory() as tmp:
        logger.info("downloading %s", repo_id)
        snapshot_download(repo_id, local_dir=tmp)
        upload_dir_to_s3(Path(tmp), args.bucket, prefix)


if __name__ == "__main__":
    main()
