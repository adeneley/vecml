"""Finalize a wrecked dataset dir: audit-gate it and write _audit_summary.json.

Flags (excluded from training by PairsDataset):
  - partial dirs (no clean.png, e.g. a wreck_svg failure left debris)
  - blank renders (clean.png with near-zero variance; bad answer keys)

  uv run python scripts/finalize_set.py --root /tmp/out/train-10k
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)

    flagged, partial, blanks, total = set(), 0, 0, 0
    for d in sorted(root.iterdir()):
        if d.name.startswith("_") or not d.is_dir():
            continue
        total += 1
        clean = d / "clean.png"
        if not clean.exists():
            flagged.add(d.name)
            partial += 1
            continue
        arr = np.asarray(Image.open(clean).convert("RGB"))
        if arr.std() < 1.0:
            flagged.add(d.name)
            blanks += 1

    (root / "_audit_summary.json").write_text(
        json.dumps({"flagged_names": sorted(flagged)}, indent=2)
    )
    print(
        f"finalized {root}: {total} dirs, {partial} partial, "
        f"{blanks} blank, usable {total - len(flagged)}"
    )


if __name__ == "__main__":
    main()
