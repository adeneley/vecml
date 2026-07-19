"""Engine ceiling test: trace PERFECT input, measure the error floor.

Feeds each engine the clean ground-truth raster (no damage, no model) and
measures dE(render(engine(clean)), clean). This is the best any relay can
ever do with that engine, so it separates model residual from tracer loss.

  uv run python scripts/ceiling_eval.py --data data/relay-test --out runs/relay/ceiling
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from relay_eval import ENGINES, de, load_rgb, render_svg  # noqa: E402
from vecml.data.pairs import list_sample_dirs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engines", nargs="*", default=["rust", "vtracer"])
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in list_sample_dirs(args.data):
        clean = load_rgb(d / "clean.png", args.size)
        img_dir = out / d.name
        img_dir.mkdir(exist_ok=True)
        row = {"name": d.name}
        for eng in args.engines:
            svg = img_dir / f"{eng}_ceiling.svg"
            try:
                ENGINES[eng](d / "clean.png", svg)
                rendered = render_svg(svg, args.size)
                Image.fromarray((rendered * 255).astype(np.uint8)).save(
                    img_dir / f"{eng}_ceiling.png")
                row[eng] = de(rendered, clean)
            except Exception as exc:  # noqa: BLE001 - one bad file must not kill the sweep
                row[eng] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        rows.append(row)
        print(row["name"][:12],
              " ".join(f"{e}={row[e]['mean']:.2f}" for e in args.engines
                       if "mean" in row.get(e, {})), flush=True)

    def agg(key):
        vals = [r[key]["mean"] for r in rows if "mean" in r.get(key, {})]
        return {"mean": float(np.mean(vals)), "p99_mean": float(np.mean(
            [r[key]["p99"] for r in rows if "p99" in r.get(key, {})])),
            "n": len(vals)} if vals else None

    summary = {"n_images": len(rows)}
    for e in args.engines:
        summary[e] = agg(e)
    (out / "summary.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
