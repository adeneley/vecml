"""Turn one clean SVG into supervised training pairs.

For each variant we write, under out_dir:
  clean.png         the reference clean render (shared by all variants)
  wrecked_XX.png    the degraded model input for variant XX
  labels.png        single-channel answer key (shared: derived once from clean)
  labels_view.png   colourised preview of the answer key (palette applied)
  palette.json      palette RGB rows, index 0 = background
  meta.json         recipe, seed, source svg, size, backend, label method, qc

The label map is derived ONCE from the SVG geometry (the idmap path) and is the
ground truth: only the wrecked input varies across variants. If geometry
derivation is impossible (no SVG, a CSS block, a gradient paint), we fall back to
the old pixel-based derivation and record which path produced the labels.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .audit import audit_sample
from .idmap import DerivationError, derive_labels_from_svg
from .labels import derive_labels_from_pixels
from .renderer import backend_name, render_svg
from .wreck import apply_recipe, sample_recipe


def _derive_ground_truth(svg_path, clean, size):
    """Return (label_map, palette, n_declared, method).

    Prefer geometry-derived labels; fall back to pixel derivation if the SVG
    cannot be turned into a clean answer key.
    """
    try:
        label_map, palette, n_declared = derive_labels_from_svg(svg_path, size)
        return label_map, palette, n_declared, "idmap"
    except DerivationError:
        label_map, palette = derive_labels_from_pixels(clean)
        return label_map, palette, None, "pixels_fallback"


def wreck_svg(
    svg_path,
    out_dir,
    size: int = 256,
    n_variants: int = 4,
    seed: int = 0,
    difficulty: str = "medium",
):
    """Render, label, and wreck one SVG into a directory of training pairs.

    Returns a small summary dict (counts and paths) for the caller to log.
    """
    svg_path = Path(svg_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = render_svg(svg_path, size)
    label_map, palette, n_declared, method = _derive_ground_truth(svg_path, clean, size)

    qc = audit_sample(clean, label_map, palette, n_declared)

    # Write the shared, variant-independent artefacts.
    Image.fromarray(clean, mode="RGB").save(out_dir / "clean.png")
    # labels.png stores raw label indices (0..N-1), not a colourised preview.
    label_mode = "L" if label_map.dtype == np.uint8 else "I;16"
    Image.fromarray(label_map, mode=label_mode).save(out_dir / "labels.png")
    # labels_view.png is the palette applied back to the indices, for eyeballing.
    Image.fromarray(palette[label_map], mode="RGB").save(out_dir / "labels_view.png")

    with open(out_dir / "palette.json", "w") as f:
        json.dump(
            {
                "background_index": 0,
                "palette": palette.tolist(),
            },
            f,
            indent=2,
        )

    variants = []
    for i in range(n_variants):
        # Derive a distinct, reproducible sub-seed per variant from the base
        # seed so the whole run is deterministic yet variants differ.
        variant_seed = seed * 100003 + i
        rng = np.random.default_rng(variant_seed)
        recipe = sample_recipe(rng, difficulty)
        wrecked = apply_recipe(clean, recipe, rng)

        name = f"wrecked_{i:02d}.png"
        Image.fromarray(wrecked, mode="RGB").save(out_dir / name)
        variants.append(
            {
                "file": name,
                "seed": int(variant_seed),
                "recipe": [{"op": op, "severity": sev} for op, sev in recipe],
            }
        )

    meta = {
        "source_svg": str(svg_path),
        "size": size,
        "difficulty": difficulty,
        "base_seed": seed,
        "n_variants": n_variants,
        "backend": backend_name(),
        "n_palette": int(len(palette)),
        "n_declared": n_declared,
        "label_method": method,
        "qc": qc,
        "variants": variants,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "svg": str(svg_path),
        "out_dir": str(out_dir),
        "n_variants": n_variants,
        "n_palette": int(len(palette)),
        "size": size,
        "label_method": method,
        "flags": qc["flags"],
    }
