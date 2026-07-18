"""Ground-truth label maps derived from SVG geometry, not from pixels.

The old path (labels.derive_labels_from_pixels) guessed the palette by counting
colours in the rendered image, then folded near-white colours into the
background. That heuristic silently deleted pale art (a #ececec icon, a cupcake's
#e0eaeb frosting) because "near white" and "pale flat colour" look the same once
you only have pixels to go on.

This module goes back to the source. It reads which colours the SVG actually
paints (fill and stroke, presentation attributes and inline style, resolving
inheritance down the group tree), assigns each distinct visible colour a widely
separated CODE colour, renders a recoloured copy through the same renderer, and
reads the label of every pixel straight off its code colour. Because the code
colours are forced opaque and the root is set to crispEdges, the render carries
no blends: each pixel is exactly one code, so the label map is exact by
construction and aligns pixel-perfect with the normal clean render (same viewBox,
same fit).

A colour that the SVG paints at reduced opacity gets a palette entry equal to
that colour composited over white, which is what the clean render shows where the
shape sits on the white background.

Anything this module cannot turn into a clean flat answer key (a CSS <style>
block, a gradient/pattern paint server) raises DerivationError so the caller can
quarantine the file instead of shipping a wrong key.
"""

import copy
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import ImageColor

from .renderer import render_svg

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

# Tags that carry paint and produce pixels.
_PAINTABLE = {
    "path",
    "rect",
    "circle",
    "ellipse",
    "polygon",
    "polyline",
    "line",
    "text",
    "tspan",
}

# Container tags whose descendants are not painted directly (clip/mask geometry,
# gradient stops, marker/pattern templates). We never collect paint from inside
# these.
_NON_PAINT_CONTAINERS = {
    "clipPath",
    "mask",
    "marker",
    "pattern",
    "linearGradient",
    "radialGradient",
    "filter",
}

# Template containers: not rendered where they sit, only through a <use>. We skip
# them in the direct walk and reach their content by expanding the references.
_TEMPLATE_TAGS = {"defs", "symbol"}

# Animation elements carry a `fill` attribute that means the animation fill mode
# ("freeze"/"remove"), not a paint. Never read paint from them.
_ANIMATION_TAGS = {"animate", "animatecolor", "animatetransform", "animatemotion", "set"}

# A colour whose every channel is this close to 255 reads as the white
# background, not as ink. Kept in step with the audit ink threshold (> 6).
_WHITE_MARGIN = 6

# Above this fraction of non-exact pixels we assume the renderer anti-aliased and
# fall back to 4x supersample + majority vote. With resvg + crispEdges this never
# fires, but a weaker backend might need it.
_NONEXACT_FALLBACK = 0.15


class DerivationError(Exception):
    """Raised when an SVG cannot yield a clean geometry-derived label map."""


def _local(tag):
    """Strip the XML namespace from a tag, returning the bare local name."""
    if isinstance(tag, str) and tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _parse_style(style_text):
    """Split an inline style="a:b;c:d" string into a lowercase-keyed dict."""
    out = {}
    for decl in style_text.split(";"):
        if ":" not in decl:
            continue
        key, _, val = decl.partition(":")
        out[key.strip().lower()] = val.strip()
    return out


def _prop(elem, style, name):
    """Look up a paint property: inline style wins over presentation attribute."""
    if name in style:
        return style[name]
    val = elem.get(name)
    return val.strip() if val is not None else None


def _resolve_colour(value, current_color):
    """Turn a paint value into an (r, g, b) tuple, None (no paint), or 'inherit'.

    Handles named colours, #rgb, #rrggbb and rgb()/hsl() via PIL. currentColor
    resolves to the element's computed `color` property (default black), not to
    the inherited fill: an icon with fill="currentColor" and no colour set paints
    black. url(...) paint servers and unparseable values raise DerivationError so
    the file is quarantined rather than guessed.
    """
    v = value.strip().lower()
    if v in ("none", "transparent"):
        return None
    if v == "inherit":
        return "inherit"
    if v == "currentcolor":
        return current_color
    if v in ("context-fill", "context-stroke", "context-value"):
        # Marker context paints: approximate as inheriting the current paint.
        return "inherit"
    if v.startswith("url("):
        raise DerivationError(f"unsupported paint server: {value!r}")

    rgb = _parse_rgb_func(v)
    if rgb is not None:
        return rgb
    try:
        parsed = ImageColor.getrgb(value)
    except ValueError as exc:
        raise DerivationError(f"unparseable colour: {value!r}") from exc
    return (int(parsed[0]), int(parsed[1]), int(parsed[2]))


_RGB_FUNC = re.compile(r"rgba?\(([^)]*)\)")


def _parse_rgb_func(value):
    """Parse rgb()/rgba() with integer, float, or percentage channels.

    PIL rejects percentages with decimals (rgb(25.88%, ...)), which are common in
    SVGs exported by drawing tools, so we handle the function form ourselves and
    ignore any alpha channel (opacity is tracked separately).
    """
    m = _RGB_FUNC.fullmatch(value.strip())
    if m is None:
        return None
    parts = [p.strip() for p in m.group(1).replace("/", ",").split(",") if p.strip()]
    if len(parts) < 3:
        return None
    chans = []
    for p in parts[:3]:
        try:
            n = float(p[:-1]) / 100.0 * 255.0 if p.endswith("%") else float(p)
        except ValueError:
            return None
        chans.append(max(0, min(255, int(round(n)))))
    return (chans[0], chans[1], chans[2])


def _as_float(value, default):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _composite_over_white(rgb, alpha):
    """Composite an opaque colour over white at the given alpha, as uint8."""
    a = max(0.0, min(1.0, alpha))
    return tuple(int(round(c * a + 255 * (1.0 - a))) for c in rgb)


def _is_background(visible_rgb):
    """True if a visible colour is indistinguishable from the white ground."""
    return all(abs(c - 255) <= _WHITE_MARGIN for c in visible_rgb)


def _lattice_codes():
    """Ordered list of widely separated code colours (farthest-point over a 6^3
    lattice, excluding the near-white corner). Consecutive codes are far apart so
    no blend of two used codes can ever land on a third."""
    levels = [0, 51, 102, 153, 204, 255]
    pts = []
    for r in levels:
        for g in levels:
            for b in levels:
                if r >= 204 and g >= 204 and b >= 204:
                    continue  # keep clear of white
                pts.append((r, g, b))
    pts = np.array(pts, dtype=np.float32)
    # Farthest-point ordering, starting from black (the point farthest from white).
    start = int(np.argmax(np.sum((pts - 255.0) ** 2, axis=1)))
    order = [start]
    dist = np.sum((pts - pts[start]) ** 2, axis=1)
    while len(order) < len(pts):
        nxt = int(np.argmax(dist))
        order.append(nxt)
        dist = np.minimum(dist, np.sum((pts - pts[nxt]) ** 2, axis=1))
    return pts[order].astype(np.uint8)


_CODES = None


def _codes_for(n):
    global _CODES
    if _CODES is None:
        _CODES = _lattice_codes()
    if n > len(_CODES):
        raise DerivationError(f"too many distinct colours ({n}) for the code lattice")
    return _CODES[:n]


class _Ctx:
    """Inherited paint context carried down the element tree."""

    __slots__ = ("fill", "stroke", "fill_op", "stroke_op", "opacity", "color")

    def __init__(self, fill, stroke, fill_op, stroke_op, opacity, color):
        self.fill = fill
        self.stroke = stroke
        self.fill_op = fill_op
        self.stroke_op = stroke_op
        self.opacity = opacity
        self.color = color  # the CSS `color` property, for currentColor

    def child(self, elem):
        """Derive the child context from this element's own paint properties."""
        if _local(elem.tag) in _ANIMATION_TAGS:
            # Animation elements' `fill` is a timing keyword, not paint. Inherit.
            return self
        style = _parse_style(elem.get("style", "")) if elem.get("style") else {}

        # Resolve `color` first: fill/stroke may reference it via currentColor.
        color = self.color
        raw = _prop(elem, style, "color")
        if raw is not None:
            resolved = _resolve_colour(raw, self.color)
            if resolved not in ("inherit", None):
                color = resolved

        fill = self.fill
        raw = _prop(elem, style, "fill")
        if raw is not None:
            resolved = _resolve_colour(raw, color)
            fill = self.fill if resolved == "inherit" else resolved

        stroke = self.stroke
        raw = _prop(elem, style, "stroke")
        if raw is not None:
            resolved = _resolve_colour(raw, color)
            stroke = self.stroke if resolved == "inherit" else resolved

        fill_op = _as_float(_prop(elem, style, "fill-opacity"), self.fill_op)
        stroke_op = _as_float(_prop(elem, style, "stroke-opacity"), self.stroke_op)
        # opacity is not inherited, but its effect compounds down groups.
        opacity = self.opacity * _as_float(_prop(elem, style, "opacity"), 1.0)
        return _Ctx(fill, stroke, fill_op, stroke_op, opacity, color)


def _check_no_css(root):
    for elem in root.iter():
        if _local(elem.tag) == "style" and (elem.text and elem.text.strip()):
            raise DerivationError("SVG carries a CSS <style> block")


def _href_id(elem):
    """Return the referenced id of a <use>, without the leading '#', or None."""
    for key in ("href", f"{{{XLINK_NS}}}href"):
        val = elem.get(key)
        if val and val.startswith("#"):
            return val[1:]
    return None


def _build_id_map(root):
    """Map every id to its element, for resolving <use> references."""
    id_map = {}
    for elem in root.iter():
        ident = elem.get("id")
        if ident and ident not in id_map:
            id_map[ident] = elem
    return id_map


def _collect_and_recolour(root):
    """Walk the tree, resolving <use>/<symbol> references. Return
    (ordered_visible_colours, code_for, true_palette_dict).

    Mutates the tree in place: <use> elements are expanded into groups holding a
    recoloured clone of their target, every element is forced fully opaque, each
    paintable leaf gets its fill/stroke replaced by the CODE colour of its visible
    colour (background colours become transparent so the shape beneath shows, no
    paint stays no paint), and the root is marked crispEdges.
    ordered_visible_colours lists the distinct foreground visible colours in
    document order (label i = index i + 1).
    """
    id_map = _build_id_map(root)

    order = []  # visible rgb tuples, document order
    seen = {}  # visible rgb -> index in order
    true_by_visible = {}  # visible rgb -> composited-over-white palette colour

    def register(visible):
        if visible not in seen:
            seen[visible] = len(order)
            order.append(visible)
            true_by_visible[visible] = visible

    def visible_fill(ctx):
        if ctx.fill is None:
            return None
        return _composite_over_white(ctx.fill, ctx.opacity * ctx.fill_op)

    def visible_stroke(ctx):
        if ctx.stroke is None:
            return None
        return _composite_over_white(ctx.stroke, ctx.opacity * ctx.stroke_op)

    # First pass: discover every visible colour so codes are assigned in order.
    def discover(elem, ctx):
        tag = _local(elem.tag)
        cctx = ctx.child(elem)
        if tag in _PAINTABLE:
            if cctx.fill is not None and not _is_background(visible_fill(cctx)):
                register(visible_fill(cctx))
            if cctx.stroke is not None and not _is_background(visible_stroke(cctx)):
                register(visible_stroke(cctx))
        for child in list(elem):
            ctag = _local(child.tag)
            if ctag == "use":
                target = id_map.get(_href_id(child))
                if target is not None:
                    discover(target, cctx.child(child))
            elif ctag in _TEMPLATE_TAGS or ctag in _NON_PAINT_CONTAINERS:
                continue
            else:
                discover(child, cctx)

    # SVG default fill is black; default `color` (for currentColor) is black.
    root_ctx = _Ctx((0, 0, 0), None, 1.0, 1.0, 1.0, (0, 0, 0))
    discover(root, root_ctx)

    codes = _codes_for(len(order))
    code_for = {vis: tuple(int(x) for x in codes[i]) for vis, i in seen.items()}

    def code_hex(visible):
        c = code_for[visible]
        return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"

    def paint_value(paint_colour, alpha):
        # No paint renders as nothing.
        if paint_colour is None:
            return "none"
        visible = _composite_over_white(paint_colour, alpha)
        if _is_background(visible):
            # An OPAQUE white shape genuinely paints paper: it covers whatever is
            # beneath, so it reads as background (white, label 0). A TRANSLUCENT
            # near-white paint is a tint or highlight, so it defers to the shape
            # beneath (transparent) rather than erasing its colour.
            return "#ffffff" if alpha >= 0.9 else "none"
        return code_hex(visible)

    def expand_use(use_elem, ctx):
        """Turn a <use> into a <g> holding a recoloured clone of its target."""
        uctx = ctx.child(use_elem)
        target = id_map.get(_href_id(use_elem))

        x = use_elem.get("x")
        y = use_elem.get("y")
        transform = use_elem.get("transform", "")
        if x or y:
            transform = f"{transform} translate({x or 0},{y or 0})".strip()

        use_elem.tag = f"{{{SVG_NS}}}g"
        use_elem.attrib.clear()
        if transform:
            use_elem.set("transform", transform)
        use_elem.set("fill-opacity", "1")
        use_elem.set("stroke-opacity", "1")
        use_elem.set("opacity", "1")

        if target is not None:
            clone = copy.deepcopy(target)
            if _local(clone.tag) == "symbol":
                # A bare <symbol> is not rendered; a <g> with the same content is.
                clone.tag = f"{{{SVG_NS}}}g"
            recolour(clone, uctx)
            use_elem.append(clone)

    # Second pass: rewrite paint. Forcing opacity to 1 everywhere kills any blend.
    def recolour(elem, ctx):
        tag = _local(elem.tag)
        cctx = ctx.child(elem)

        style = _parse_style(elem.get("style", "")) if elem.get("style") else {}
        for drop in ("fill", "stroke", "fill-opacity", "stroke-opacity", "opacity"):
            style.pop(drop, None)

        if tag in _PAINTABLE:
            elem.set("fill", paint_value(cctx.fill, cctx.opacity * cctx.fill_op))
            elem.set("stroke", paint_value(cctx.stroke, cctx.opacity * cctx.stroke_op))

        # Force full opacity on every element so nothing blends.
        elem.set("fill-opacity", "1")
        elem.set("stroke-opacity", "1")
        elem.set("opacity", "1")
        if style:
            elem.set("style", ";".join(f"{k}:{v}" for k, v in style.items()))
        elif "style" in elem.attrib:
            del elem.attrib["style"]

        for child in list(elem):
            ctag = _local(child.tag)
            if ctag == "use":
                expand_use(child, cctx)
            elif ctag in _TEMPLATE_TAGS or ctag in _NON_PAINT_CONTAINERS:
                continue  # templates render only through <use>; leave them be
            else:
                recolour(child, cctx)

    recolour(root, root_ctx)
    root.set("shape-rendering", "crispEdges")

    return order, code_for, true_by_visible


def _labels_from_render(idmap_rgb, order, code_for):
    """Map an id-map render to a label array using the used code colours."""
    h, w = idmap_rgb.shape[:2]
    candidates = [(255, 255, 255)] + [code_for[vis] for vis in order]
    cand = np.array(candidates, dtype=np.float32)

    pixels = idmap_rgb.reshape(-1, 3).astype(np.float32)
    labels = np.empty(pixels.shape[0], dtype=np.int64)
    worst = 0.0
    nonexact = 0
    chunk = 200_000
    for start in range(0, pixels.shape[0], chunk):
        block = pixels[start : start + chunk]
        d = np.linalg.norm(block[:, None, :] - cand[None, :, :], axis=2)
        idx = np.argmin(d, axis=1)
        mind = d[np.arange(len(block)), idx]
        labels[start : start + chunk] = idx
        nonexact += int(np.count_nonzero(mind > 1.0))
        worst = max(worst, float(mind.max()))

    nonexact_frac = nonexact / max(1, pixels.shape[0])
    return labels.reshape(h, w), nonexact_frac


def _render_labels(recoloured_svg_text, size, order, code_for, supersample):
    """Render the recoloured SVG (optionally at supersample x) and read labels."""
    box = size * supersample
    with tempfile.NamedTemporaryFile(
        suffix=".svg", mode="w", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(recoloured_svg_text)
        tmp_path = fh.name
    try:
        idmap_rgb = render_svg(tmp_path, box)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    labels, nonexact_frac = _labels_from_render(idmap_rgb, order, code_for)
    if supersample == 1:
        return labels, nonexact_frac

    # Majority vote down each supersample x supersample block.
    n_labels = len(order) + 1
    blocks = labels.reshape(size, supersample, size, supersample)
    best_count = np.zeros((size, size), dtype=np.int32)
    out = np.zeros((size, size), dtype=np.int64)
    for lbl in range(n_labels):
        count = np.sum(blocks == lbl, axis=(1, 3))
        take = count > best_count
        out[take] = lbl
        best_count[take] = count[take]
    return out, nonexact_frac


def derive_labels_from_svg(svg_path, size):
    """Return (label_map, palette, n_declared) derived from SVG geometry.

    label_map: HxW uint8/uint16, label 0 = white background, aligned to the
      clean render produced by renderer.render_svg(svg_path, size).
    palette: (N+1)x3 uint8 of the TRUE colours (row 0 white), reduced-opacity
      colours composited over white.
    n_declared: number of distinct foreground visible colours the SVG declares
      (== len(palette) - 1 by construction; the audit compares them).

    Raises DerivationError for CSS blocks, paint servers, or unparseable SVGs.
    """
    svg_path = Path(svg_path)
    try:
        tree = ET.parse(svg_path)
    except ET.ParseError as exc:
        raise DerivationError(f"unparseable SVG XML: {exc}") from exc
    root = tree.getroot()

    _check_no_css(root)
    order, code_for, true_by_visible = _collect_and_recolour(root)

    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    recoloured = ET.tostring(root, encoding="unicode")

    labels, nonexact_frac = _render_labels(recoloured, size, order, code_for, 1)
    if nonexact_frac > _NONEXACT_FALLBACK:
        labels, _ = _render_labels(recoloured, size, order, code_for, 4)

    palette_rows = [(255, 255, 255)] + [true_by_visible[vis] for vis in order]
    palette = np.array(palette_rows, dtype=np.uint8)

    dtype = np.uint8 if len(palette) <= 255 else np.uint16
    return labels.astype(dtype), palette, len(order)
