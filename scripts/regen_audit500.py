"""Regenerate the 500-sample audit set with the new idmap pipeline.

Mirrors the original run: size 256, 2 variants, medium difficulty, seed
500 + index (index = position in the sorted source list). Each file is wrapped
in try/except so one bad SVG cannot abort the batch; failures are captured to
_failures.json.

The 4x supersampled ownership render makes each file heavier than the old 1x
path, so the batch runs across a process pool (one worker per CPU, each file is
independent) to keep wall-clock in the few-minutes range rather than ~30 min.

  uv run python scripts/regen_audit500.py --src data/audit-500-src --out data/audit-500-v2
"""

import argparse
import json
import os
import traceback
from pathlib import Path
import sys
from multiprocessing import Pool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vecml.degrade.pipeline import wreck_svg  # noqa: E402

_CFG = {}


def _init(size, variants, difficulty, out):
    _CFG["size"] = size
    _CFG["variants"] = variants
    _CFG["difficulty"] = difficulty
    _CFG["out"] = Path(out)


def _one(job):
    """Process a single (index, svg_path) job in a worker. Returns a result dict."""
    index, svg_str = job
    svg = Path(svg_str)
    try:
        summary = wreck_svg(
            svg,
            _CFG["out"] / svg.stem,
            size=_CFG["size"],
            n_variants=_CFG["variants"],
            seed=500 + index,
            difficulty=_CFG["difficulty"],
        )
        return {"ok": True, "method": summary["label_method"], "svg": svg.name}
    except Exception as exc:  # noqa: BLE001 - batch must survive one bad file
        return {
            "ok": False,
            "svg": svg.name,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def main():
    ap = argparse.ArgumentParser(description="Regenerate the 500-sample audit set.")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--difficulty", default="medium")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    svgs = sorted(src.glob("*.svg"))
    jobs = [(i, str(p)) for i, p in enumerate(svgs)]

    failures = []
    methods = {}
    done = 0
    workers = max(1, args.workers)
    print(f"regenerating {len(svgs)} samples across {workers} workers ...")
    with Pool(
        processes=workers,
        initializer=_init,
        initargs=(args.size, args.variants, args.difficulty, out),
    ) as pool:
        for n, res in enumerate(pool.imap_unordered(_one, jobs), start=1):
            if res["ok"]:
                methods[res["method"]] = methods.get(res["method"], 0) + 1
                done += 1
            else:
                failures.append(
                    {"svg": res["svg"], "error": res["error"], "traceback": res["traceback"]}
                )
            if n % 50 == 0:
                print(f"  {n}/{len(svgs)} done, {len(failures)} failures")

    (out / "_failures.json").write_text(json.dumps(failures, indent=2))
    print(f"done: {done}/{len(svgs)} ok, {len(failures)} failures")
    print(f"method split: {methods}")


if __name__ == "__main__":
    main()
