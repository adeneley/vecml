"""CLI: forensic calibration diagnostics for wrecker output.

Compares per-image forensic features (noise sigma, JPEG quality, edge-ringing
energy) of a v2-wrecked output tree against a directory of real damaged images,
reporting per-family synthetic distributions vs the real set. This is the cheap
measurement half of the calibration protocol (docs README section 3, Step 1);
the C2ST/KID sim-to-real harness is a separate future step.

Example:
  uv run python scripts/wreck_calibrate.py \
      --wrecked data/wrecked-v2 --real data/real-damaged --out runs/calib.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vecml.degrade.calibrate import compare  # noqa: E402

_FEATS = ("noise_sigma", "jpeg_quality", "edge_ringing")


def _row(name, summ):
    cells = []
    for f in _FEATS:
        s = summ.get(f, {})
        if s.get("n"):
            cells.append(f"{f}: {s['mean']:.3g} (p10 {s['p10']:.3g} / p90 {s['p90']:.3g})")
        else:
            cells.append(f"{f}: -")
    return f"  {name:16s}  " + " | ".join(cells)


def main():
    ap = argparse.ArgumentParser(description="Calibrate wrecker output against real damaged files.")
    ap.add_argument("--wrecked", required=True, help="wreck_svg output tree (v2, with meta.json)")
    ap.add_argument("--real", required=True, help="dir of real damaged images")
    ap.add_argument("--out", default=None, help="optional path to write the JSON report")
    args = ap.parse_args()

    report = compare(args.wrecked, args.real)

    print(f"REAL  ({report['real']['n_images']} images)")
    print(_row("real", report["real"]["features"]))
    print("\nSYNTHETIC by family")
    for fam, block in report["synthetic"].items():
        print(_row(f"{fam} ({block['n_images']})", block["features"]))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
