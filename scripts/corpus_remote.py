"""Bootstrap the wreckable SVG corpus on a pod: download, gate, cache.

End state: <work>/clean/ holds the sharded clean tier, ready for
sample_src.py --clean. The expensive part (2.8GB HuggingFace download +
gate2 labelling of 2.28M SVGs) runs once per volume: the clean tier is
cached as a tarball in --cache and later runs just extract it.

  uv run python scripts/corpus_remote.py \
      --cache /workspace/corpus-cache/svg-stack-clean.tar.gz --work /tmp/corpus

Local disk use: ~3GB parquet + ~7GB labelled tree while building; the cache
tarball is ~2GB. Nothing here needs a GPU.
"""
import argparse
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

HF_BASE = "https://huggingface.co/datasets/starvector/svg-stack/resolve/main/data"
SHARDS = (
    [f"train-{i:05d}-of-00012.parquet" for i in range(12)]
    + ["val-00000-of-00001.parquet", "test-00000-of-00001.parquet"]
)


def fetch(url, dest, tries=4):
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                while chunk := r.read(1 << 22):
                    f.write(chunk)
            return
        except Exception as exc:
            if attempt == tries:
                raise
            print(f"  retry {attempt}/{tries - 1} after: {exc}", flush=True)
            time.sleep(5 * attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True,
                    help="clean-tier tarball on durable storage (built if absent)")
    ap.add_argument("--work", required=True,
                    help="fast local dir; ends up holding <work>/clean/")
    args = ap.parse_args()
    cache, work = Path(args.cache), Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    if (work / "clean").is_dir() and any((work / "clean").iterdir()):
        print(f"clean tier already at {work / 'clean'}, nothing to do", flush=True)
        return

    if cache.exists():
        print(f"extracting cached clean tier {cache} ({cache.stat().st_size / 1e9:.1f}GB)",
              flush=True)
        with tarfile.open(cache) as tf:
            tf.extractall(work)
        print("done", flush=True)
        return

    raw = work / "raw"
    raw.mkdir(exist_ok=True)
    print(f"cache miss: downloading {len(SHARDS)} parquet shards from HF", flush=True)
    for name in SHARDS:
        dest = raw / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        t0 = time.perf_counter()
        fetch(f"{HF_BASE}/{name}", dest)
        print(f"  {name}: {dest.stat().st_size / 1e6:.0f}MB "
              f"in {time.perf_counter() - t0:.0f}s", flush=True)

    labelled = work / "labelled"
    rc = subprocess.call([
        sys.executable, str(Path(__file__).parent / "label_split.py"),
        "--data", str(raw), "--out", str(labelled),
    ])
    if rc != 0:
        sys.exit(rc)

    # Stage the clean tier where callers expect it, then cache it durably.
    (labelled / "clean").rename(work / "clean")
    (labelled / "manifest.jsonl").rename(work / "clean-manifest.jsonl")
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    print("packing clean tier into the cache tarball", flush=True)
    with tarfile.open(tmp, "w:gz", compresslevel=4) as tf:
        tf.add(work / "clean", arcname="clean")
        tf.add(work / "clean-manifest.jsonl", arcname="clean-manifest.jsonl")
    tmp.rename(cache)  # atomic: readers never see a half-written cache
    print(f"cached {cache} ({cache.stat().st_size / 1e9:.1f}GB)", flush=True)


if __name__ == "__main__":
    main()
