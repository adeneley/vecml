"""Finalize a wrecked dataset dir: audit-gate it and write _audit_summary.json.

Flags (excluded from training by PairsDataset):
  - partial dirs (no clean.png, e.g. a wreck_svg failure left debris)
  - blank renders (clean.png with near-zero variance; bad answer keys)
  - bad answer keys (qc high_reconstruction_error / empty_labels flags):
    harmless to the RGB head but poison for label-head training

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
    ap.add_argument("--prune", action="store_true",
                    help="delete flagged dirs instead of relying on the audit "
                         "file (shard tarballs merge into one dir, so a "
                         "single _audit_summary.json cannot carry the flags)")
    args = ap.parse_args()
    root = Path(args.root)

    flagged, partial, blanks, badkeys, total = set(), 0, 0, 0, 0
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
            continue
        meta = d / "meta.json"
        if meta.exists():
            qc_flags = set(json.loads(meta.read_text())["qc"]["flags"])
            if qc_flags & {"high_reconstruction_error", "empty_labels_nonempty_render"}:
                flagged.add(d.name)
                badkeys += 1

    if args.prune:
        import shutil
        for name in flagged:
            shutil.rmtree(root / name, ignore_errors=True)
        (root / "_audit_summary.json").write_text(
            json.dumps({"flagged_names": [], "pruned": len(flagged)}, indent=2)
        )
    else:
        (root / "_audit_summary.json").write_text(
            json.dumps({"flagged_names": sorted(flagged)}, indent=2)
        )
    print(
        f"finalized {root}: {total} dirs, {partial} partial, {blanks} blank, "
        f"{badkeys} bad answer keys, usable {total - len(flagged)}"
        + (" (flagged pruned)" if args.prune else "")
    )


if __name__ == "__main__":
    main()
