"""Generate a visual contact sheet of the v2 families for eyeballing realism.

One PNG grid per family: rows = corpus source images, columns = severities
0.2 / 0.5 / 0.8, with the clean render in the first column. Writes to
runs/wrecker-v2-preview/ (gitignored).

Example:
  uv run python scripts/wreck_preview.py --src data/audit-500-src --n 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vecml.degrade import wreck_v2 as w2  # noqa: E402
from vecml.degrade.renderer import render_svg  # noqa: E402

SEVERITIES = (0.2, 0.5, 0.8)
PAD = 6
LABEL_H = 18


def _wreck_at(clean, family, s, seed):
    """Build one family recipe at a fixed severity and apply it."""
    sample_rng = np.random.default_rng(seed)
    steps = w2.FAMILY_BUILDERS[family](sample_rng, s, clean.shape[0])
    recipe = {"family": family, "global_severity": s, "ops": steps}
    apply_rng = np.random.default_rng(seed + 1)
    out, _ = w2.apply_recipe_v2(clean, recipe, apply_rng)
    return out


def _grid(cleans, family, size):
    ncol = 1 + len(SEVERITIES)
    nrow = len(cleans)
    cw = size + PAD
    ch = size + PAD
    W = ncol * cw + PAD
    H = nrow * ch + PAD + LABEL_H
    sheet = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    draw.text((PAD, 4), f"family: {family}   [clean | s=0.2 | s=0.5 | s=0.8]", fill=(20, 20, 20))
    for r, clean in enumerate(cleans):
        tiles = [clean] + [_wreck_at(clean, family, s, 100 + r) for s in SEVERITIES]
        for c, tile in enumerate(tiles):
            x = PAD + c * cw
            y = LABEL_H + PAD + r * ch
            sheet.paste(Image.fromarray(tile, mode="RGB"), (x, y))
    return sheet


def main():
    ap = argparse.ArgumentParser(description="Contact sheet of v2 wrecker families.")
    ap.add_argument("--src", default="data/audit-500-src", help="dir of source svgs")
    ap.add_argument("--out", default="runs/wrecker-v2-preview", help="output dir")
    ap.add_argument("--n", type=int, default=3, help="source images per grid")
    ap.add_argument("--size", type=int, default=256, help="tile size")
    args = ap.parse_args()

    svgs = sorted(Path(args.src).glob("*.svg"))[: args.n]
    if not svgs:
        ap.error(f"no *.svg in {args.src}")
    cleans = [render_svg(p, args.size) for p in svgs]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for family in w2.FAMILY_BUILDERS:
        sheet = _grid(cleans, family, args.size)
        path = out / f"{family}.png"
        sheet.save(path)
        print(f"  wrote {path}")
    print(f"done: {len(list(w2.FAMILY_BUILDERS))} family sheets in {out}")


if __name__ == "__main__":
    main()
