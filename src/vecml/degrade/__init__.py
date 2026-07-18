"""Wrecking pipeline: render clean SVGs, degrade them, emit training pairs."""

from .labels import derive_labels
from .pipeline import wreck_svg
from .renderer import backend_name, render_svg, render_svg_rgba
from .wreck import apply_recipe, sample_recipe

__all__ = [
    "backend_name",
    "render_svg",
    "render_svg_rgba",
    "derive_labels",
    "sample_recipe",
    "apply_recipe",
    "wreck_svg",
]
