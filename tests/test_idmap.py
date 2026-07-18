"""Tests for geometry-derived ground truth and the audit gate. No network."""

import numpy as np
import pytest

from vecml.degrade.audit import audit_sample
from vecml.degrade.idmap import DerivationError, derive_labels_from_svg
from vecml.degrade.renderer import render_svg

SIZE = 128

# Pale #ececec shape plus two saturated shapes on white. The pale shape is the
# one the old pixel heuristic dropped as "near white".
PALE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <rect x="10" y="10" width="90" height="90" fill="#ececec"/>
  <rect x="110" y="10" width="80" height="80" fill="#d81e1e"/>
  <circle cx="60" cy="150" r="40" fill="#1e6fd8"/>
</svg>
"""

# Two overlapping shapes: the later one (green) wins in the overlap region.
OVERLAP_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <rect x="20" y="20" width="120" height="120" fill="#d81e1e"/>
  <rect x="80" y="80" width="100" height="100" fill="#1ea34a"/>
</svg>
"""

# Same colour declared twice but at different opacity: two distinct visible
# colours over white, so two labels, and codes must not corrupt each other.
OPACITY_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <rect x="10" y="10" width="80" height="180" fill="#0000ff"/>
  <rect x="110" y="10" width="80" height="180" fill="#0000ff" fill-opacity="0.5"/>
</svg>
"""

# Inheritance: fill declared on the group, geometry on the children.
INHERIT_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <g fill="#800080">
    <rect x="10" y="10" width="80" height="80"/>
    <rect x="110" y="110" width="80" height="80"/>
  </g>
</svg>
"""

CSS_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <style>.a { fill: #123456; }</style>
  <rect class="a" x="0" y="0" width="50" height="50"/>
</svg>
"""

# Icon idiom: root fill="none", shapes fill="currentColor" with no colour set.
# currentColor must resolve to the default `color` (black), not to fill="none".
CURRENTCOLOR_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none">
  <rect x="4" y="4" width="16" height="16" fill="currentColor"/>
</svg>
"""


def _write(tmp_path, text, name="in.svg"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_pale_shape_is_labelled_not_background(tmp_path):
    svg = _write(tmp_path, PALE_SVG)
    label_map, palette, n_declared = derive_labels_from_svg(svg, SIZE)

    assert n_declared == 3
    assert len(palette) == 4  # background + 3 flats
    assert np.array_equal(palette[0], [255, 255, 255])

    # The pale #ececec entry must be present in the palette...
    dists = np.linalg.norm(palette.astype(int) - np.array([236, 236, 236]), axis=1)
    pale_label = int(np.argmin(dists))
    assert dists[pale_label] < 6
    assert pale_label != 0
    # ...and it must actually own pixels in the top-left quadrant.
    assert (label_map[: SIZE // 2, : SIZE // 2] == pale_label).sum() > 100


def test_overlap_resolves_by_paint_order(tmp_path):
    svg = _write(tmp_path, OVERLAP_SVG)
    label_map, palette, _ = derive_labels_from_svg(svg, SIZE)

    green = int(np.argmin(np.linalg.norm(palette.astype(int) - np.array([30, 163, 74]), axis=1)))
    red = int(np.argmin(np.linalg.norm(palette.astype(int) - np.array([216, 30, 30]), axis=1)))

    # The centre of the image sits in the overlap: green is painted last, wins.
    assert label_map[SIZE // 2, SIZE // 2] == green
    # Red still owns its top-left corner region.
    assert (label_map == red).sum() > 100


def test_opacity_does_not_corrupt_codes(tmp_path):
    svg = _write(tmp_path, OPACITY_SVG)
    label_map, palette, n_declared = derive_labels_from_svg(svg, SIZE)

    # Solid blue and half-opacity blue are two visible colours, hence two labels.
    assert n_declared == 2
    assert len(np.unique(label_map)) == 3  # bg + 2

    # Half-opacity blue over white is roughly (128, 128, 255): a real palette row.
    half = np.array([128, 128, 255])
    d = np.linalg.norm(palette.astype(int) - half, axis=1)
    assert d.min() < 12
    # Reconstruction matches the clean render (no fake blended codes leaked in).
    clean = render_svg(svg, SIZE)
    recon = palette[label_map]
    # Small residual is edge anti-aliasing in the clean render, not corrupt
    # codes (a leaked fake code would blow this up by tens of levels).
    mae = np.abs(clean.astype(int) - recon.astype(int)).mean()
    assert mae < 5.0


def test_inheritance_from_group(tmp_path):
    svg = _write(tmp_path, INHERIT_SVG)
    label_map, palette, n_declared = derive_labels_from_svg(svg, SIZE)
    assert n_declared == 1
    purple = np.array([128, 0, 128])
    d = np.linalg.norm(palette.astype(int) - purple, axis=1)
    assert d.min() < 6
    assert (label_map != 0).sum() > 200


def test_currentcolor_resolves_to_black(tmp_path):
    svg = _write(tmp_path, CURRENTCOLOR_SVG)
    label_map, palette, n_declared = derive_labels_from_svg(svg, SIZE)
    assert n_declared == 1
    # The shape must be labelled (not dropped as no-paint) and its colour black.
    assert (label_map != 0).sum() > 200
    black = np.array([0, 0, 0])
    assert np.linalg.norm(palette[1].astype(int) - black) < 6


# A solid coloured background with an OPAQUE white icon on top: the white icon is
# paper showing through (label 0), not a fake copy of the background colour.
WHITE_ON_COLOUR_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect x="0" y="0" width="100" height="100" fill="#db2c39"/>
  <circle cx="50" cy="50" r="25" fill="#ffffff"/>
</svg>
"""


def test_opaque_white_on_colour_is_background(tmp_path):
    svg = _write(tmp_path, WHITE_ON_COLOUR_SVG)
    label_map, palette, n_declared = derive_labels_from_svg(svg, SIZE)
    # Only the red is a declared foreground colour; white reads as paper.
    assert n_declared == 1
    # The centre (inside the white circle) must be background, not red.
    assert label_map[SIZE // 2, SIZE // 2] == 0
    # A corner (red) is labelled the red foreground.
    assert label_map[2, 2] != 0
    # Reconstruction is faithful: white circle back to white, red back to red.
    clean = render_svg(svg, SIZE)
    mae = np.abs(clean.astype(int) - palette[label_map].astype(int)).mean()
    assert mae < 3.0


# <use> painting a defs template: the paint lives on the use, the geometry in
# defs. The old walk labelled the template black (its tree-inherited default).
USE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 100 100">
  <defs><rect id="box" x="0" y="0" width="60" height="60"/></defs>
  <use fill="#c81e5a" xlink:href="#box"/>
</svg>
"""

# Percentage rgb() with decimals, which PIL rejects.
RGB_PCT_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect x="0" y="0" width="80" height="80" fill="rgb(14.117647%,53.72549%,79.215686%)"/>
</svg>
"""


def test_use_takes_paint_from_use_element(tmp_path):
    svg = _write(tmp_path, USE_SVG)
    label_map, palette, n_declared = derive_labels_from_svg(svg, SIZE)
    assert n_declared == 1
    # The rect must be labelled its use colour (#c81e5a), not the default black.
    want = np.array([200, 30, 90])
    assert np.linalg.norm(palette[1].astype(int) - want) < 6
    assert (label_map != 0).sum() > 200


def test_percentage_rgb_parses(tmp_path):
    svg = _write(tmp_path, RGB_PCT_SVG)
    label_map, palette, n_declared = derive_labels_from_svg(svg, SIZE)
    assert n_declared == 1
    want = np.array([36, 137, 202])  # 14.12% / 53.73% / 79.22% of 255
    assert np.linalg.norm(palette[1].astype(int) - want) < 3


def test_css_block_raises(tmp_path):
    svg = _write(tmp_path, CSS_SVG)
    with pytest.raises(DerivationError):
        derive_labels_from_svg(svg, SIZE)


def test_palette_matches_declared_fills(tmp_path):
    svg = _write(tmp_path, PALE_SVG)
    label_map, palette, n_declared = derive_labels_from_svg(svg, SIZE)
    qc = audit_sample(render_svg(svg, SIZE), label_map, palette, n_declared)
    assert "palette_size_mismatch" not in qc["flags"]
    assert qc["metrics"]["palette_size"] == n_declared + 1


def test_audit_flags_emptied_label_map(tmp_path):
    svg = _write(tmp_path, PALE_SVG)
    _, palette, n_declared = derive_labels_from_svg(svg, SIZE)
    clean = render_svg(svg, SIZE)
    empty = np.zeros((SIZE, SIZE), dtype=np.uint8)
    qc = audit_sample(clean, empty, palette, n_declared)
    assert "empty_labels_nonempty_render" in qc["flags"]


def test_audit_passes_a_good_sample(tmp_path):
    svg = _write(tmp_path, PALE_SVG)
    label_map, palette, n_declared = derive_labels_from_svg(svg, SIZE)
    qc = audit_sample(render_svg(svg, SIZE), label_map, palette, n_declared)
    assert qc["flags"] == []
