"""Tests for the wrecking pipeline. No network, no dependence on assets/."""

import json

import numpy as np
import pytest

from vecml.degrade.labels import derive_labels_from_pixels
from vecml.degrade.pipeline import wreck_svg
from vecml.degrade.renderer import render_svg, render_svg_rgba
from vecml.degrade.wreck import OPS, apply_recipe, sample_recipe

SIZE = 128

# Three overlapping flat-colour shapes on a white ground: bg + 3 colours.
TEST_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <rect x="0" y="0" width="200" height="200" fill="#ffffff"/>
  <rect x="20" y="20" width="120" height="120" fill="#d81e1e"/>
  <circle cx="130" cy="90" r="60" fill="#1e6fd8"/>
  <polygon points="60,180 180,180 120,80" fill="#1ea34a"/>
</svg>
"""


@pytest.fixture
def svg_file(tmp_path):
    p = tmp_path / "shapes.svg"
    p.write_text(TEST_SVG)
    return p


def test_render_shape_and_dtype(svg_file):
    rgb = render_svg(svg_file, SIZE)
    assert rgb.shape == (SIZE, SIZE, 3)
    assert rgb.dtype == np.uint8

    rgba = render_svg_rgba(svg_file, SIZE)
    assert rgba.shape == (SIZE, SIZE, 4)
    assert rgba.dtype == np.uint8


def test_derive_labels_palette_and_background(svg_file):
    rgb = render_svg(svg_file, SIZE)
    label_map, palette = derive_labels_from_pixels(rgb)

    assert label_map.shape == (SIZE, SIZE)
    # bg white + 3 flats == 4 palette entries (allow one of slack for AA blends).
    assert 4 <= len(palette) <= 5
    # Label 0 is background and must be white-ish.
    assert np.linalg.norm(palette[0].astype(int) - np.array([255, 255, 255])) < 24
    # Every label indexes a real palette row.
    assert label_map.max() < len(palette)
    # Background label should actually be used (the corners are white).
    assert (label_map == 0).any()


@pytest.mark.parametrize("op_name", list(OPS.keys()))
def test_wreck_ops_preserve_shape_and_change_pixels(svg_file, op_name):
    rgb = render_svg(svg_file, SIZE)
    rng = np.random.default_rng(123)
    out = OPS[op_name](rgb, rng, 0.7)
    assert out.shape == rgb.shape
    assert out.dtype == np.uint8
    # At severity 0.7 every op should actually alter the image.
    assert not np.array_equal(out, rgb), f"{op_name} did not change pixels"


def test_recipe_determinism(svg_file):
    rgb = render_svg(svg_file, SIZE)

    def run(seed):
        rng = np.random.default_rng(seed)
        recipe = sample_recipe(rng, "medium")
        return apply_recipe(rgb, recipe, rng)

    a1 = run(7)
    a2 = run(7)
    b = run(8)
    assert np.array_equal(a1, a2), "same seed must give identical output"
    assert not np.array_equal(a1, b), "different seed should differ"


def test_recipe_tiers_valid():
    rng = np.random.default_rng(0)
    for tier in ("light", "medium", "brutal"):
        recipe = sample_recipe(rng, tier)
        assert 1 <= len(recipe) <= 4
        for op, sev in recipe:
            assert op in OPS
            assert 0.0 <= sev <= 1.0
    with pytest.raises(ValueError):
        sample_recipe(rng, "nope")


def test_pipeline_writes_all_files(svg_file, tmp_path):
    out_dir = tmp_path / "out"
    n = 3
    summary = wreck_svg(svg_file, out_dir, size=SIZE, n_variants=n, seed=42, difficulty="medium")

    assert (out_dir / "clean.png").exists()
    assert (out_dir / "labels.png").exists()
    assert (out_dir / "palette.json").exists()
    assert (out_dir / "meta.json").exists()
    for i in range(n):
        assert (out_dir / f"wrecked_{i:02d}.png").exists()

    palette = json.loads((out_dir / "palette.json").read_text())["palette"]
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["n_variants"] == n
    assert len(meta["variants"]) == n

    # labels.png values must all be valid palette indices.
    from PIL import Image

    labels = np.asarray(Image.open(out_dir / "labels.png"))
    assert labels.max() < len(palette)
    assert summary["n_variants"] == n
