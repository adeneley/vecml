"""Read pod consoles and volume artifacts over the S3 gateway. No pod needed.

The volume is the source of truth: every pod writes its console to
runs-logs/<run>/console.log and its checkpoints under checkpoints/. This is
how you watch a job and fetch results from the Mac (the gateway rejects
ranged GETs, so tail re-downloads the whole log - they are small).

  source infra/runpod/.env first (set -a; source infra/runpod/.env; set +a)

  uv run --with boto3 python scripts/pod_log.py runs            # recent run dirs
  uv run --with boto3 python scripts/pod_log.py tail            # follow newest console
  uv run --with boto3 python scripts/pod_log.py tail --after 20260719-1108
  uv run --with boto3 python scripts/pod_log.py ls checkpoints/
  uv run --with boto3 python scripts/pod_log.py get checkpoints/labels-10k-a/best.pt runs/labels-10k-a/best.pt
"""
import argparse
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.config import Config


def client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["RUNPOD_S3_ENDPOINT"],
        aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_KEY"],
        region_name="EU-RO-1",
        config=Config(retries={"mode": "adaptive", "max_attempts": 5}),
    )


def list_keys(s3, bucket, prefix):
    tok, keys = {}, []
    while True:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1000, **tok)
        keys += [(o["Key"], o["Size"]) for o in resp.get("Contents", [])]
        if not resp.get("IsTruncated"):
            return keys
        tok = {"ContinuationToken": resp["NextContinuationToken"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["runs", "tail", "ls", "get"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--after", default=None,
                    help="tail: only consider run dirs after this stamp "
                         "(YYYYMMDD-HHMM[SS]); use the pod's createdAt")
    ap.add_argument("--interval", type=float, default=20.0)
    args = ap.parse_args()
    s3 = client()
    bucket = os.environ["RUNPOD_VOLUME_ID"]

    if args.cmd == "runs":
        names = sorted({k.split("/")[1] for k, _ in list_keys(s3, bucket, "runs-logs/run-")})
        print("\n".join(names[-15:]))
        return

    if args.cmd == "ls":
        prefix = args.args[0] if args.args else ""
        for k, size in list_keys(s3, bucket, prefix):
            print(f"{size / 1e6:10.1f}MB  {k}")
        return

    if args.cmd == "get":
        key, dest = args.args[0], Path(args.args[1] if len(args.args) > 1 else Path(args.args[0]).name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        body = s3.get_object(Bucket=bucket, Key=key)["Body"]
        with open(dest, "wb") as f:
            while chunk := body.read(1 << 24):
                f.write(chunk)
        print(f"{key} -> {dest} ({dest.stat().st_size / 1e6:.1f}MB)")
        return

    # tail: pick the newest run dir (optionally after a stamp), print new
    # lines forever; exits when the log reports the job ended.
    floor = f"run-{args.after}" if args.after else ""
    key, seen = None, 0
    while True:
        if key is None:
            names = sorted({k.split("/")[1] for k, _ in list_keys(s3, bucket, "runs-logs/run-")})
            cands = [n for n in names if n > floor] if floor else names
            if cands:
                key = f"runs-logs/{cands[-1] if not floor else cands[0]}/console.log"
                print(f"[pod_log] watching {key}", file=sys.stderr)
        if key:
            try:
                body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                lines = body.decode(errors="replace").splitlines()
                for ln in lines[seen:]:
                    print(ln, flush=True)
                seen = len(lines)
                if any("job exited rc=" in ln for ln in lines):
                    return
            except Exception:
                pass
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
