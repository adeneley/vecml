"""Regenerate the 500-sample audit set with the new idmap pipeline.

Mirrors the original run: size 256, 2 variants, medium difficulty, seed
500 + index (index = position in the sorted source list). Each file is wrapped
in try/except so one bad SVG cannot abort the batch; failures are captured to
_failures.json.

  uv run python scripts/regen_audit500.py --src data/audit-500-src --out data/audit-500-v2
"""

import argparse
import json
import traceback
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vecml.degrade.pipeline import wreck_svg  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Regenerate the 500-sample audit set.")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--difficulty", default="medium")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    svgs = sorted(src.glob("*.svg"))
    failures = []
    methods = {}
    done = 0
    for index, svg in enumerate(svgs):
        try:
            summary = wreck_svg(
                svg,
                out / svg.stem,
                size=args.size,
                n_variants=args.variants,
                seed=500 + index,
                difficulty=args.difficulty,
            )
            methods[summary["label_method"]] = methods.get(summary["label_method"], 0) + 1
            done += 1
        except Exception as exc:  # noqa: BLE001 - batch must survive one bad file
            failures.append(
                {"svg": svg.name, "error": f"{type(exc).__name__}: {exc}",
                 "traceback": traceback.format_exc()}
            )
        if (index + 1) % 50 == 0:
            print(f"  {index + 1}/{len(svgs)} done, {len(failures)} failures")

    (out / "_failures.json").write_text(json.dumps(failures, indent=2))
    print(f"done: {done}/{len(svgs)} ok, {len(failures)} failures")
    print(f"method split: {methods}")


if __name__ == "__main__":
    main()
