"""CLI: wreck every SVG in an input dir into per-svg training-pair subdirs.

Example:
  uv run python scripts/wreck.py --in assets/seed --out data/wrecked \
      --size 256 --variants 4 --difficulty medium --seed 42
"""

import argparse
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

# Allow running as a plain script (python scripts/wreck.py) without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vecml.degrade.pipeline import wreck_svg  # noqa: E402
from vecml.degrade.renderer import backend_name  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Wreck flat-colour SVGs into training pairs.")
    ap.add_argument("--in", dest="in_dir", required=True, help="dir of *.svg files")
    ap.add_argument("--out", dest="out_dir", required=True, help="output root dir")
    ap.add_argument("--size", type=int, default=256, help="square canvas size in px")
    ap.add_argument("--variants", type=int, default=4, help="wrecked variants per svg")
    ap.add_argument(
        "--difficulty",
        default="medium",
        choices=["light", "medium", "brutal"],
        help="degradation tier",
    )
    ap.add_argument("--seed", type=int, default=0, help="base seed")
    ap.add_argument("--bg", default="white", choices=["white", "random"],
                    help="background mode: random = per-pair solid colours")
    ap.add_argument("--curriculum", action="store_true",
                    help="include identity and barely-damaged variants")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--quiet", action="store_true", help="progress line only")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_root = Path(args.out_dir)
    if not in_dir.is_dir():
        ap.error(f"input dir not found: {in_dir}")

    svgs = sorted(in_dir.glob("*.svg"))
    if not svgs:
        ap.error(f"no *.svg files in {in_dir}")

    print(f"backend: {backend_name()}  |  {len(svgs)} svg(s)  |  size {args.size}px  "
          f"|  {args.variants} variant(s)  |  {args.difficulty}")

    jobs = [
        (
            svg,
            out_root / svg.stem,
            args.size,
            args.variants,
            # Per-svg seed so recipes differ across the set (a single shared
            # seed gives every svg the same damage sequence).
            (args.seed + zlib.crc32(svg.stem.encode())) & 0x7FFFFFFF,
            args.difficulty,
            args.bg,
            args.curriculum,
        )
        for svg in svgs
    ]

    total_pairs, failed = 0, 0
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            results = pool.imap_unordered(_one, jobs, chunksize=16)
            for i, summary in enumerate(results, 1):
                if summary is None:
                    failed += 1
                else:
                    total_pairs += summary["n_variants"]
                if i % 250 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} done ({failed} failed)", flush=True)
    else:
        for job in jobs:
            summary = _one(job)
            if summary is None:
                failed += 1
                continue
            total_pairs += summary["n_variants"]
            if not args.quiet:
                print(
                    f"  {Path(summary['svg']).name:32s} -> {summary['out_dir']}  "
                    f"({summary['n_variants']} pairs, {summary['n_palette']} colours)"
                )

    print(f"done: {len(svgs)} svg(s), {total_pairs} training pair(s), "
          f"{failed} failed, written to {out_root}")


def _one(job):
    svg, out_dir, size, variants, seed, difficulty, bg, curriculum = job
    try:
        return wreck_svg(
            svg, out_dir, size=size, n_variants=variants, seed=seed,
            difficulty=difficulty, bg_mode=bg, curriculum=curriculum,
        )
    except Exception:  # one bad svg must not kill a 10k-file sweep
        return None


if __name__ == "__main__":
    main()
