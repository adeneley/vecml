"""Derive a paint-by-numbers label map from a clean flat-colour render.

The clean render is our answer key: because we made the SVG flat, its palette
is a small finite set of colours plus the white background. We recover that
palette in two passes:

  1. Count colours on a coarse grid. Colours that own a large share of the
     image are "definite" flats. Colours owning a smaller but non-trivial share
     are "borderline".
  2. A borderline colour is kept only if it cannot be explained as a linear
     blend of two definite flats (plus white). Anti-aliased edges create thin
     bands of exactly such blend colours, so this rejects fringe while keeping
     genuinely small flat regions (which are not blends of other flats).

Every pixel is then labelled by nearest palette colour.

Known caveat (v1): anti-aliased edge pixels blend two region colours, so
nearest-colour assignment puts them on whichever side is closer. Labels are
therefore near-perfect in region interiors but carry roughly one pixel of
ambiguity along edges. v2 may render an explicit region-ID map from the SVG
instead of inferring it from pixels.
"""

import numpy as np

# Coverage tiers as a fraction of total pixels.
_DEFINITE_COVERAGE = 0.02  # >= 2 % is unambiguously a real flat
_MIN_COVERAGE = 0.003  # 0.3 %..2 % is borderline: kept only if not a blend

# Near-duplicate palette candidates within this RGB distance are merged.
_MERGE_DIST = 24.0

# A colour this close to the segment between two definite flats is treated as an
# anti-alias blend and dropped.
_BLEND_TOL = 25.0

# A colour this close to pure white is treated as background (label 0).
_WHITE_DIST = 32.0
_WHITE = np.array([255, 255, 255], dtype=np.float32)


def _quantize_count(rgb):
    # Round to a coarse 8-level grid (step 32) so anti-alias jitter collapses
    # onto flats before counting: coarse enough to group most fringe, fine
    # enough to keep genuinely distinct flats apart.
    q = (rgb.astype(np.uint16) // 32) * 32 + 16
    flat = q.reshape(-1, 3)
    colours, counts = np.unique(flat, axis=0, return_counts=True)
    return colours.astype(np.float32), counts


def _merge_near_duplicates(colours, counts):
    order = np.argsort(-counts)
    colours = colours[order]
    counts = counts[order]
    merged = []
    merged_counts = []
    for c, n in zip(colours, counts):
        placed = False
        for i, m in enumerate(merged):
            if np.linalg.norm(c - m) < _MERGE_DIST:
                # Keep the more populous representative (already sorted, so m).
                merged_counts[i] += n
                placed = True
                break
        if not placed:
            merged.append(c)
            merged_counts.append(int(n))
    return np.array(merged, dtype=np.float32), np.array(merged_counts)


def _is_blend(colour, anchors):
    # True if colour lies within _BLEND_TOL of the segment between any pair of
    # anchor colours (the anchors are the definite flats plus white).
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            a = anchors[i]
            b = anchors[j]
            d = b - a
            denom = float(d @ d)
            if denom == 0:
                continue
            t = float((colour - a) @ d) / denom
            t = min(1.0, max(0.0, t))
            proj = a + t * d
            if np.linalg.norm(colour - proj) < _BLEND_TOL:
                return True
    return False


def derive_labels(clean_rgb: np.ndarray):
    """Return (label_map, palette).

    label_map: uint8 (or uint16 if the palette is large) array, HxW, where each
      value indexes into palette. Label 0 is always the white background.
    palette: Nx3 uint8 array of RGB colours, row 0 = white background.
    """
    h, w = clean_rgb.shape[:2]
    total = h * w

    colours, counts = _quantize_count(clean_rgb)
    keep = counts >= max(1, int(total * _MIN_COVERAGE))
    merged, merged_counts = _merge_near_duplicates(colours[keep], counts[keep])

    # White-ish candidates fold into the background; the rest are foreground.
    white_mask = np.linalg.norm(merged - _WHITE, axis=1) < _WHITE_DIST
    fg = merged[~white_mask]
    fg_counts = merged_counts[~white_mask]

    definite = fg[fg_counts >= int(total * _DEFINITE_COVERAGE)]

    # Blend anchors: the definite flats plus white (fringe is often a flat-over
    # -white blend, so white must be an anchor).
    anchors = np.vstack([_WHITE[None, :], definite]) if len(definite) else _WHITE[None, :]

    kept = list(definite)
    for c, n in zip(fg, fg_counts):
        if n >= int(total * _DEFINITE_COVERAGE):
            continue  # already in definite
        if not _is_blend(c, anchors):
            kept.append(c)

    # Palette row 0 is always pure white background (keeps the label-0-is-
    # background contract stable even if the render had no white).
    palette = np.vstack([_WHITE[None, :], np.array(kept, dtype=np.float32)]) if kept else _WHITE[None, :]

    # Assign every pixel to the nearest palette colour, chunked to bound memory.
    pixels = clean_rgb.reshape(-1, 3).astype(np.float32)
    labels = np.empty(pixels.shape[0], dtype=np.int64)
    chunk = 200_000
    for start in range(0, pixels.shape[0], chunk):
        block = pixels[start : start + chunk]
        d = np.linalg.norm(block[:, None, :] - palette[None, :, :], axis=2)
        labels[start : start + chunk] = np.argmin(d, axis=1)

    label_map = labels.reshape(h, w)
    dtype = np.uint8 if len(palette) <= 255 else np.uint16
    return label_map.astype(dtype), palette.astype(np.uint8)
