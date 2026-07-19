"""Automatic per-sample quality checks for the ground-truth answer keys.

The wrecking pipeline can produce a bad training pair in ways that are invisible
until you look: a label map that dropped a pale region, a palette that does not
reconstruct the clean render, an art file that renders blank. This module scores
one sample against its own clean render and flags the failures by name, so a run
can be split into train-worthy and quarantine buckets without a human eyeballing
500 thumbnails.

The one subtlety is pale-on-white art. A #ececec icon differs from white by only
19 levels, so any "is this pixel ink?" test has to use a LOW threshold or it
counts the whole icon as background and declares the (correct) label map wrong.
We treat a pixel as ink when any channel differs from 255 by more than 6.
"""

import numpy as np

# A pixel is ink if any channel differs from white by more than this.
INK_THRESHOLD = 6

# Coverage: |ink fraction - labelled fraction|. Above this the label map and the
# visible art disagree about how much of the image is painted.
COVERAGE_TOL = 0.08

# Mean absolute reconstruction error (clean vs palette[label]) above this means
# the palette does not explain the render.
RECON_TOL = 12.0

# Below this ink fraction the render is effectively blank: art invisible or the
# file is genuinely empty. Quarantine, do not train.
BLANK_INK_FRAC = 0.002


def ink_mask(clean_rgb, bg=(255, 255, 255)):
    """Boolean HxW mask of pixels that carry ink (differ from the background)."""
    diff = np.abs(clean_rgb.astype(np.int16) - np.asarray(bg, dtype=np.int16))
    return np.any(diff > INK_THRESHOLD, axis=2)


def audit_sample(clean_rgb, label_map, palette, n_declared=None, bg=(255, 255, 255)):
    """Score one sample. Return {"metrics": {...}, "flags": [...]}.

    clean_rgb: HxWx3 uint8 clean render.
    label_map: HxW integer label indices into palette (0 = background).
    palette: (N+1)x3 uint8, row 0 = background.
    n_declared: distinct foreground colours the SVG declared, or None if unknown
      (the pixel-fallback path cannot report it, so palette_match is skipped).
    bg: the colour the render was composited over; ink = "differs from bg",
      so coloured-background sets audit correctly instead of reading the
      entire background as ink.
    """
    palette = np.asarray(palette)
    ink = ink_mask(clean_rgb, bg)
    ink_frac = float(ink.mean())
    label_nonzero_frac = float((label_map != 0).mean())
    coverage_delta = abs(ink_frac - label_nonzero_frac)

    recon = palette[label_map]
    recon_mae = float(np.abs(clean_rgb.astype(np.int16) - recon.astype(np.int16)).mean())

    palette_size = int(len(palette))
    metrics = {
        "ink_frac": round(ink_frac, 5),
        "label_nonzero_frac": round(label_nonzero_frac, 5),
        "coverage_delta": round(coverage_delta, 5),
        "reconstruction_mae": round(recon_mae, 4),
        "palette_size": palette_size,
        "n_declared": n_declared,
        "n_labels_used": int(len(np.unique(label_map))),
    }

    flags = []
    if ink_frac < BLANK_INK_FRAC:
        # Blank render dominates: everything else is moot, quarantine outright.
        flags.append("blank_render")
    else:
        if label_nonzero_frac == 0.0:
            flags.append("empty_labels_nonempty_render")
        if coverage_delta > COVERAGE_TOL:
            flags.append("coverage_mismatch")
    if recon_mae > RECON_TOL:
        flags.append("high_reconstruction_error")
    if n_declared is not None and palette_size != n_declared + 1:
        flags.append("palette_size_mismatch")

    return {"metrics": metrics, "flags": flags}
