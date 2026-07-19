"""Sample N clean-tier SVGs into a src dir, excluding shas already used elsewhere.

Exclusions come from --exclude dirs (local *-src dirs) and/or --exclude-shas
(a text file of shas, one per line). The shas file is how a pod stays honest
about val/test hygiene without any local state: keep datasets/used-shas.txt
on the network volume, append what you sample, pass it to the next run.

  uv run python scripts/sample_src.py --n 500 --out data/train-1000-src \
      --exclude data/audit-500-src --seed 1000
  uv run python scripts/sample_src.py --clean /tmp/corpus/clean --n 2000 \
      --out /tmp/wk/src --exclude-shas /workspace/datasets/used-shas.txt \
      --record-shas --seed 777
"""

import argparse
import random
import shutil
from pathlib import Path

DEFAULT_CLEAN = "datasets/svg-stack-labelled/clean"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clean", default=DEFAULT_CLEAN,
                    help="sharded clean-tier corpus root")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="src dirs whose shas must not be re-sampled")
    ap.add_argument("--exclude-shas", default=None,
                    help="text file of shas (one per line) to skip")
    ap.add_argument("--record-shas", action="store_true",
                    help="append the sampled shas to the --exclude-shas file")
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    used = set()
    for ex in args.exclude:
        used |= {p.stem for p in Path(ex).glob("*.svg")}
    shas_file = Path(args.exclude_shas) if args.exclude_shas else None
    if shas_file and shas_file.exists():
        used |= {ln.strip() for ln in shas_file.read_text().splitlines() if ln.strip()}
    print(f"excluding {len(used)} already-used shas")

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    picked = 0
    sampled = []
    shards = sorted(Path(args.clean).iterdir())
    while picked < args.n:
        shard = rng.choice(shards)
        files = list(shard.glob("*.svg"))
        if not files:
            continue
        f = rng.choice(files)
        if f.stem in used:
            continue
        used.add(f.stem)
        sampled.append(f.stem)
        shutil.copy2(f, out / f.name)
        picked += 1
        if picked % 100 == 0:
            print(f"{picked}/{args.n}")
    if args.record_shas and shas_file:
        with open(shas_file, "a") as fh:
            fh.write("".join(s + "\n" for s in sampled))
        print(f"recorded {len(sampled)} shas -> {shas_file}")
    print(f"done: {picked} SVGs -> {out}")


if __name__ == "__main__":
    main()
