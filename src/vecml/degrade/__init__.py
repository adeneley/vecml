"""Wrecking pipeline: render clean SVGs, degrade them, emit training pairs."""

from .audit import audit_sample, ink_mask
from .idmap import DerivationError, derive_labels_from_svg
from .labels import derive_labels, derive_labels_from_pixels
from .pipeline import wreck_svg
from .renderer import backend_name, render_svg, render_svg_rgba
from .wreck import apply_recipe, sample_recipe

__all__ = [
    "backend_name",
    "render_svg",
    "render_svg_rgba",
    "derive_labels",
    "derive_labels_from_pixels",
    "derive_labels_from_svg",
    "DerivationError",
    "audit_sample",
    "ink_mask",
    "sample_recipe",
    "apply_recipe",
    "wreck_svg",
]
