"""Turn one clean SVG into supervised training pairs.

For each variant we write, under out_dir:
  clean.png         the reference clean render (shared by all variants)
  wrecked_XX.png    the degraded model input for variant XX
  labels.png        single-channel answer key (shared: derived once from clean)
  palette.json      palette RGB rows, index 0 = background
  meta.json         recipe (ops + severities), seed, source svg, size, backend

The label map is derived ONCE from the clean render. It is the ground truth and
never changes: only the wrecked input varies across variants.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .labels import derive_labels
from .renderer import backend_name, render_svg
from .wreck import apply_recipe, sample_recipe


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
    label_map, palette = derive_labels(clean)

    # Write the shared, variant-independent artefacts.
    Image.fromarray(clean, mode="RGB").save(out_dir / "clean.png")
    # labels.png stores raw label indices (0..N-1), not a colourised preview.
    label_mode = "L" if label_map.dtype == np.uint8 else "I;16"
    Image.fromarray(label_map, mode=label_mode).save(out_dir / "labels.png")

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
    }
