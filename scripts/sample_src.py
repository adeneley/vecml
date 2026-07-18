"""Sample N clean-tier SVGs into a src dir, excluding shas already used elsewhere.

  uv run python scripts/sample_src.py --n 500 --out data/train-1000-src \
      --exclude data/audit-500-src --seed 1000
"""

import argparse
import random
import shutil
from pathlib import Path

CLEAN = Path("datasets/svg-stack-labelled/clean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    used = set()
    for ex in args.exclude:
        used |= {p.stem for p in Path(ex).glob("*.svg")}

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    picked = 0
    shards = sorted(CLEAN.iterdir())
    while picked < args.n:
        shard = rng.choice(shards)
        files = list(shard.glob("*.svg"))
        if not files:
            continue
        f = rng.choice(files)
        if f.stem in used:
            continue
        used.add(f.stem)
        shutil.copy2(f, out / f.name)
        picked += 1
        if picked % 100 == 0:
            print(f"{picked}/{args.n}")
    print(f"done: {picked} SVGs -> {out}")


if __name__ == "__main__":
    main()
