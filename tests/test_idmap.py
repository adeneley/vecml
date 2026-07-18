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


# --- CLASS B: translucent paint keeps its blended colour, is not deleted -------

# Translucent white (opacity .18) over a solid green body: the classic "Python
# logo over a coloured plate" case. Composited over WHITE this is white and used
# to be binned as background and deleted; over the GREEN it is a pale green face.
TRANSLUCENT_WHITE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <rect x="0" y="0" width="200" height="200" fill="#647c64"/>
  <rect x="50" y="50" width="100" height="100" fill="#ffffff" fill-opacity="0.18"/>
</svg>
"""

# Translucent black shadow (opacity .35) over a tan body: a darkened face.
TRANSLUCENT_BLACK_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <rect x="0" y="0" width="200" height="200" fill="#d2b48c"/>
  <rect x="60" y="60" width="80" height="80" fill="#000000" fill-opacity="0.35"/>
</svg>
"""


def test_translucent_white_over_colour_is_a_pale_face(tmp_path):
    svg = _write(tmp_path, TRANSLUCENT_WHITE_SVG)
    label_map, palette, n_declared = derive_labels_from_svg(svg, SIZE)

    # Two faces: the green body and the pale-green overlay. Not deleted.
    assert n_declared == 2
    # The overlay sits at the centre; it must be labelled (not background)...
    centre = int(label_map[SIZE // 2, SIZE // 2])
    assert centre != 0
    # ...and its colour is the .18-white-over-green blend, a pale green, clearly
    # lighter than the body but nowhere near pure white.
    overlay = palette[centre].astype(int)
    body = np.array([100, 124, 100])  # #647c64
    assert (overlay > body).all()  # lightened
    assert not (overlay > 249).all()  # not binned as white background
    # Reconstruction stays faithful.
    clean = render_svg(svg, SIZE)
    mae = np.abs(clean.astype(int) - palette[label_map].astype(int)).mean()
    assert mae < 4.0


def test_translucent_black_over_colour_is_a_darkened_face(tmp_path):
    svg = _write(tmp_path, TRANSLUCENT_BLACK_SVG)
    label_map, palette, n_declared = derive_labels_from_svg(svg, SIZE)

    assert n_declared == 2
    centre = int(label_map[SIZE // 2, SIZE // 2])
    assert centre != 0
    shadow = palette[centre].astype(int)
    body = np.array([210, 180, 140])  # #d2b48c
    assert (shadow < body).all()  # darkened by the black overlay
    clean = render_svg(svg, SIZE)
    mae = np.abs(clean.astype(int) - palette[label_map].astype(int)).mean()
    assert mae < 4.0


# --- CLASS C: overlapping translucent shapes make a distinct third face --------

# Two same-colour translucent rects overlapping: the overlap is a visibly darker
# third region. Topmost-shape ownership would give it to the upper rect (2
# regions); planar-face ownership must see 3.
TRANSLUCENT_OVERLAP_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <g fill="#73c0fb" fill-opacity="0.4">
    <rect x="20" y="20" width="110" height="110"/>
    <rect x="70" y="70" width="110" height="110"/>
  </g>
</svg>
"""


def test_translucent_overlap_makes_three_faces(tmp_path):
    svg = _write(tmp_path, TRANSLUCENT_OVERLAP_SVG)
    label_map, palette, n_declared = derive_labels_from_svg(svg, SIZE)

    assert n_declared == 3  # two lobes plus the darker overlap lens
    assert len(np.unique(label_map)) == 4  # + background

    # The overlap sits at the shared centre; the two lobes at their far corners.
    overlap = int(label_map[SIZE // 2, SIZE // 2])
    lobe_a = int(label_map[int(SIZE * 0.22), int(SIZE * 0.22)])
    lobe_b = int(label_map[int(SIZE * 0.80), int(SIZE * 0.80)])
    assert 0 not in (overlap, lobe_a, lobe_b)
    assert len({overlap, lobe_a, lobe_b}) == 3

    # The overlap colour is darker (lower luminance) than either lobe.
    lum = palette.astype(int).sum(axis=1)
    assert lum[overlap] < lum[lobe_a]
    assert lum[overlap] < lum[lobe_b]

    clean = render_svg(svg, SIZE)
    mae = np.abs(clean.astype(int) - palette[label_map].astype(int)).mean()
    assert mae < 5.0


# --- CLASS A: a sub-pixel stroke stays a connected, covered line ---------------

# stroke-width 1 in a 300-unit viewBox is ~0.85px at 256 and vanishes / dashes
# under a 1x crispEdges render. The 4x ownership render plus coverage rescue must
# keep it as one connected line matching the clean render's inked pixels.
THIN_STROKE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 300">
  <line x1="20" y1="150" x2="280" y2="150" stroke="#000000" stroke-width="1"/>
</svg>
"""


def test_subpixel_stroke_survives_as_connected_line(tmp_path):
    svg = _write(tmp_path, THIN_STROKE_SVG, name="thin.svg")
    label_map, palette, n_declared = derive_labels_from_svg(svg, 256)
    clean = render_svg(svg, 256)

    assert n_declared == 1  # the black line
    line = int(np.argmin(np.linalg.norm(palette.astype(int) - np.array([0, 0, 0]), axis=1)))
    labelled = label_map == line

    # The line must actually exist and span most of the width (not a stub).
    cols_with_line = np.where(labelled.any(axis=0))[0]
    assert cols_with_line.size > 0
    assert cols_with_line.max() - cols_with_line.min() > 256 * 0.8

    # Connected, not dashed: no interior column along the drawn span is empty.
    span = labelled[:, cols_with_line.min() : cols_with_line.max() + 1]
    assert span.any(axis=0).all()

    # Coverage matches the clean render's inked pixels within tolerance (this is
    # the invariant the audit's coverage_match metric checks).
    ink = np.abs(clean.astype(int) - 255).max(axis=2) > 6
    ink_frac = ink.mean()
    label_frac = (label_map != 0).mean()
    assert abs(ink_frac - label_frac) < 0.02


def test_translucent_overlap_passes_audit(tmp_path):
    svg = _write(tmp_path, TRANSLUCENT_OVERLAP_SVG)
    label_map, palette, n_declared = derive_labels_from_svg(svg, SIZE)
    qc = audit_sample(render_svg(svg, SIZE), label_map, palette, n_declared)
    assert qc["flags"] == []
