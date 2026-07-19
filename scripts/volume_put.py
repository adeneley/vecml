"""Resilient upload from the Mac to the RunPod network volume over S3.

RunPod's S3 gateway 524s mid-multipart under load, so: modest 16MB parts,
adaptive retries, abort any stale multipart debris for the key first, and
verify via list_objects_v2 afterwards (HeadObject 403s on RunPod).

boto3 stays out of the project deps (pods never upload from a Mac); run:

  uv run --with boto3 python scripts/volume_put.py <local-file> <key>
  uv run --with boto3 python scripts/volume_put.py data.tar.gz datasets/data.tar.gz
"""

import argparse
import sys
import time
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

ENV_FILE = Path(__file__).resolve().parent.parent / "infra" / "runpod" / ".env"


def load_env() -> dict:
    env = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("local", help="local file to upload")
    ap.add_argument("key", help="volume key, e.g. datasets/train-10k-bg-run.tar.gz")
    ap.add_argument("--attempts", type=int, default=4)
    args = ap.parse_args()

    env = load_env()
    bucket = env["RUNPOD_VOLUME_ID"]
    s3 = boto3.client(
        "s3",
        endpoint_url=env["RUNPOD_S3_ENDPOINT"],
        aws_access_key_id=env["RUNPOD_S3_ACCESS_KEY"],
        aws_secret_access_key=env["RUNPOD_S3_SECRET_KEY"],
        region_name=env.get("RUNPOD_DATACENTER", "EU-RO-1"),
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"},
                      read_timeout=180, connect_timeout=30),
    )

    # Clear stale multipart debris for this key (failed prior attempts).
    try:
        for up in s3.list_multipart_uploads(Bucket=bucket).get("Uploads", []):
            if up["Key"] == args.key:
                s3.abort_multipart_upload(Bucket=bucket, Key=args.key,
                                          UploadId=up["UploadId"])
    except Exception:
        pass

    local = Path(args.local)
    size_mb = local.stat().st_size / 1e6
    cfg = TransferConfig(multipart_chunksize=16 * 1024 * 1024,
                         multipart_threshold=16 * 1024 * 1024,
                         max_concurrency=4)

    for attempt in range(1, args.attempts + 1):
        t0 = time.time()
        try:
            s3.upload_file(str(local), bucket, args.key, Config=cfg)
        except Exception as exc:
            print(f"attempt {attempt} failed after {time.time()-t0:.0f}s: "
                  f"{type(exc).__name__}: {str(exc)[:150]}", flush=True)
            continue
        listed = s3.list_objects_v2(Bucket=bucket, Prefix=args.key)
        remote = next((o for o in listed.get("Contents", []) if o["Key"] == args.key), None)
        if remote and remote["Size"] == local.stat().st_size:
            mins = (time.time() - t0) / 60
            print(f"UPLOADED attempt {attempt}: {size_mb:.0f} MB in {mins:.1f} min -> {args.key}")
            return 0
        print(f"attempt {attempt}: uploaded but size mismatch on verify, retrying", flush=True)
    print("FAILED: all attempts exhausted", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
