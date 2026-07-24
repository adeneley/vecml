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


def off_split_shas(manifest: Path, split: str) -> set:
    """Shas whose manifest split label is NOT `split` (the smaller side:
    train is 95% of the corpus, so we hold the ~114k val/test shas)."""
    out = set()
    tag = f'"split": "{split}"'
    with open(manifest) as fh:
        for ln in fh:
            if tag not in ln:
                i = ln.find('"sha": "') + 8
                out.add(ln[i:ln.find('"', i)])
    return out


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
    ap.add_argument("--record-to", default=None,
                    help="write this run's sampled shas to a separate file "
                         "(sharded pods must not race on the shared ledger)")
    ap.add_argument("--shard", default=None,
                    help="'i/k': take slice i of k from the deterministic "
                         "sample. Every pod computes the identical shuffled "
                         "pick list (same seed + same exclusion file), so "
                         "disjoint slices need no coordination. Union of "
                         "shards 0..j-1 of k = a uniform n*j/k sample, which "
                         "is how one 1M build also yields the 500k set.")
    ap.add_argument("--split", default="train",
                    help="manifest split label to sample from (default "
                         "'train'; 'any' disables the filter). The 21 Jul "
                         "leakage audit found ~5%% of samples were val/test "
                         "shas because this filter didn't exist.")
    ap.add_argument("--manifest", default=None,
                    help="manifest.jsonl path (default: sibling of --clean)")
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    used = set()
    if args.split != "any":
        manifest = Path(args.manifest) if args.manifest \
            else Path(args.clean).parent / "manifest.jsonl"
        if not manifest.exists():
            raise SystemExit(f"manifest not found: {manifest} "
                             "(pass --manifest or --split any)")
        off = off_split_shas(manifest, args.split)
        used |= off
        print(f"split filter '{args.split}': excluding {len(off)} off-split shas")
    for ex in args.exclude:
        used |= {p.stem for p in Path(ex).glob("*.svg")}
    shas_file = Path(args.exclude_shas) if args.exclude_shas else None
    if shas_file and shas_file.exists():
        used |= {ln.strip() for ln in shas_file.read_text().splitlines() if ln.strip()}
    print(f"excluding {len(used)} already-used shas")

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Deterministic pick list: sorted corpus walk, filter exclusions, one
    # seeded shuffle, first n. Identical on every machine that sees the same
    # corpus + exclusion file, which is what makes --shard coordination-free.
    pool = [p for p in sorted(Path(args.clean).glob("*/*.svg"))
            if p.stem not in used]
    if args.n > len(pool):
        raise SystemExit(f"corpus exhausted: want {args.n}, {len(pool)} unused")
    rng.shuffle(pool)
    chosen = pool[: args.n]
    if args.shard:
        i, k = (int(x) for x in args.shard.split("/"))
        chunk = (args.n + k - 1) // k
        chosen = chosen[i * chunk: (i + 1) * chunk]
        print(f"shard {i}/{k}: {len(chosen)} of {args.n}")

    sampled = []
    for j, f in enumerate(chosen, 1):
        sampled.append(f.stem)
        shutil.copy2(f, out / f.name)
        if j % 5000 == 0:
            print(f"{j}/{len(chosen)}", flush=True)
    if args.record_to:
        Path(args.record_to).write_text("".join(s + "\n" for s in sampled))
        print(f"recorded {len(sampled)} shas -> {args.record_to}")
    if args.record_shas and shas_file:
        with open(shas_file, "a") as fh:
            fh.write("".join(s + "\n" for s in sampled))
        print(f"recorded {len(sampled)} shas -> {shas_file}")
    print(f"done: {len(sampled)} SVGs -> {out}")


if __name__ == "__main__":
    main()
