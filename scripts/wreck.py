"""CLI: wreck every SVG in an input dir into per-svg training-pair subdirs.

Example:
  uv run python scripts/wreck.py --in assets/seed --out data/wrecked \
      --size 256 --variants 4 --difficulty medium --seed 42
"""

import argparse
import sys
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

    total_pairs = 0
    for svg in svgs:
        out_dir = out_root / svg.stem
        summary = wreck_svg(
            svg,
            out_dir,
            size=args.size,
            n_variants=args.variants,
            seed=args.seed,
            difficulty=args.difficulty,
        )
        total_pairs += summary["n_variants"]
        print(
            f"  {svg.name:32s} -> {summary['out_dir']}  "
            f"({summary['n_variants']} pairs, {summary['n_palette']} colours)"
        )

    print(f"done: {len(svgs)} svg(s), {total_pairs} training pair(s) written to {out_root}")


if __name__ == "__main__":
    main()
