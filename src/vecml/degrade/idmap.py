"""Ground-truth label maps derived from SVG geometry, not from pixels.

The old pixel path (labels.derive_labels_from_pixels) guessed the palette by
counting colours in the rendered image, then folded near-white colours into the
background. That heuristic silently deleted pale art (a #ececec icon, a cupcake's
#e0eaeb frosting) because "near white" and "pale flat colour" look the same once
you only have pixels to go on.

This module (idmap-v3) goes back to the source geometry but treats the SVG as a
STACK of planar faces, matching the Rust engine's model, not as a bag of source
shapes. It works in three moves:

  ownership (which face)  -- a crispEdges render assigns every pixel to the flat
    region that is visible there. Opaque paint gets one code colour per distinct
    opaque colour; a pixel's region is the stack of translucent leaves covering
    it plus the first opaque owner (or background) beneath. Two pixels share a
    region iff that whole stack is identical, so overlapping translucent shapes
    (a darker lens) become their own third region.

  coverage (which pixels) -- the ownership render runs at 4x supersample and is
    downsampled with a coverage-aware vote so that sub-pixel strokes, which a 1x
    crispEdges render drops entirely, survive: any target pixel the clean render
    inks is forced to the nearest real region rather than binned as background.

  colour (what shade)     -- the palette is NOT read from SVG paint. Each region
    takes the median RGB of the CLEAN anti-aliased render over the pixels it
    owns (eroded 1px to dodge anti-aliased edges). A translucent overlay simply
    gets its blended colour; a face whose sampled colour is indistinguishable
    from the page is folded back into background.

Anything this module cannot turn into a clean flat answer key (a CSS <style>
block, a gradient/pattern paint server, more translucent leaves than the bitmask
can hold) raises DerivationError so the caller can quarantine the file instead of
shipping a wrong key.
"""

import copy
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import ImageColor

from .audit import ink_mask
from .renderer import render_svg

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

LABEL_METHOD = "idmap-v3"

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

# Effective alpha at or above this counts as opaque: the paint fully covers what
# is beneath it, so it is a solid face, not a translucent overlay. Detected from
# the ORIGINAL resolved context alpha, before opacity is forced to 1 for render.
_OPAQUE_EPS = 0.99

# Supersample factor for the ownership render. 4x means 1024 for a 256 target,
# enough to resolve a sub-pixel stroke that a 1x crispEdges render drops.
_SUPERSAMPLE = 4

# Most files have zero translucent leaves; those that do get one extra mask
# render each and a bit in a fixed-width mask. Cap the count so the region key
# stays inside a uint64 (with room to pack the opaque-owner id alongside).
_TRANSLUCENT_CAP = 32

# Solo mask paint: rendered on an otherwise-blank (white) canvas, so any pixel
# that is not near-white is covered by the leaf under test.
_MASK_PAINT = "#000000"
_MASK_COVERED_BELOW = 200


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


def _composite_over(rgb, alpha, under):
    """Composite an opaque colour over a base colour at the given alpha."""
    a = max(0.0, min(1.0, alpha))
    return tuple(c * a + u * (1.0 - a) for c, u in zip(rgb, under))


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


# --- geometry walk: expand <use>, force full opacity, record painted leaves ----


def _force_opaque_attrs(elem):
    """Strip paint/opacity from an element and force it fully opaque.

    Leaves get their fill/stroke set explicitly per render mode, so nothing here
    relies on inheritance for colour; this only guarantees no element ever blends.
    """
    style = _parse_style(elem.get("style", "")) if elem.get("style") else {}
    for drop in ("fill", "stroke", "fill-opacity", "stroke-opacity", "opacity"):
        style.pop(drop, None)
    elem.set("fill-opacity", "1")
    elem.set("stroke-opacity", "1")
    elem.set("opacity", "1")
    if style:
        elem.set("style", ";".join(f"{k}:{v}" for k, v in style.items()))
    elif "style" in elem.attrib:
        del elem.attrib["style"]


def _expand_use(use_elem, id_map):
    """Turn a <use> into a <g> holding a clone of its target, in place.

    The use's own paint attributes are LEFT ON the <g> so the walk resolves them
    into the clone's context (the use paints the referenced geometry). x/y become
    a translate; href/xlink:href are dropped.
    """
    target = id_map.get(_href_id(use_elem))
    x = use_elem.get("x")
    y = use_elem.get("y")
    transform = use_elem.get("transform", "")
    if x or y:
        transform = f"{transform} translate({x or 0},{y or 0})".strip()

    use_elem.tag = f"{{{SVG_NS}}}g"
    for key in ("x", "y", "href", f"{{{XLINK_NS}}}href"):
        if key in use_elem.attrib:
            del use_elem.attrib[key]
    if transform:
        use_elem.set("transform", transform)

    if target is not None:
        clone = copy.deepcopy(target)
        if _local(clone.tag) == "symbol":
            # A bare <symbol> is not rendered; a <g> with the same content is.
            clone.tag = f"{{{SVG_NS}}}g"
        use_elem.append(clone)


def _walk_collect(root):
    """Walk the tree (expanding <use>), forcing full opacity, recording leaves.

    Returns a list of leaf records; each holds the element reference plus its
    resolved fill/stroke colours and effective alphas, so callers can paint each
    leaf differently per render mode without re-resolving inheritance.
    """
    id_map = _build_id_map(root)
    leaves = []

    def record(elem, ctx):
        leaves.append(
            {
                "elem": elem,
                "fill": ctx.fill,
                "fa": (ctx.opacity * ctx.fill_op) if ctx.fill is not None else 0.0,
                "stroke": ctx.stroke,
                "sa": (ctx.opacity * ctx.stroke_op) if ctx.stroke is not None else 0.0,
            }
        )

    def walk(elem, ctx):
        cctx = ctx.child(elem)
        _force_opaque_attrs(elem)
        if _local(elem.tag) in _PAINTABLE:
            record(elem, cctx)
        for child in list(elem):
            ctag = _local(child.tag)
            if ctag == "use":
                _expand_use(child, id_map)
                walk(child, cctx)  # child is now a <g> carrying the use's paint
            elif ctag in _TEMPLATE_TAGS or ctag in _NON_PAINT_CONTAINERS:
                continue
            else:
                walk(child, cctx)

    # SVG default fill is black; default `color` (for currentColor) is black.
    root_ctx = _Ctx((0, 0, 0), None, 1.0, 1.0, 1.0, (0, 0, 0))
    walk(root, root_ctx)
    root.set("shape-rendering", "crispEdges")
    return leaves


def _classify(leaves):
    """Split leaf paint into opaque colours (faces) and translucent units.

    Each leaf channel (fill, stroke) is one of:
      opaque       -- alpha >= _OPAQUE_EPS, real colour: a solid face colour.
      paper        -- alpha >= _OPAQUE_EPS but near-white: covers as background.
      translucent  -- alpha < _OPAQUE_EPS: a blended overlay, gets its own bit.
    Mutates each leaf record with per-channel "<ch>_kind" (and "<ch>_bit").
    Returns (opaque_order, opaque_index, translucent_units).
    """
    opaque_order = []  # distinct opaque rgb tuples, first-seen order
    opaque_index = {}
    translucent_units = []  # (leaf_idx, channel) in document order == paint order

    def reg_opaque(rgb):
        if rgb not in opaque_index:
            opaque_index[rgb] = len(opaque_order)
            opaque_order.append(rgb)

    for i, lf in enumerate(leaves):
        for ch, akey in (("fill", "fa"), ("stroke", "sa")):
            rgb = lf[ch]
            if rgb is None:
                lf[ch + "_kind"] = None
                continue
            alpha = lf[akey]
            if alpha >= _OPAQUE_EPS:
                if _is_background(rgb):
                    lf[ch + "_kind"] = "paper"
                else:
                    reg_opaque(rgb)
                    lf[ch + "_kind"] = "opaque"
            else:
                lf[ch + "_kind"] = "translucent"
                lf[ch + "_bit"] = len(translucent_units)
                translucent_units.append((i, ch))

    if len(translucent_units) > _TRANSLUCENT_CAP:
        raise DerivationError(
            f"too many translucent leaves ({len(translucent_units)}) for the "
            f"region bitmask (cap {_TRANSLUCENT_CAP})"
        )
    return opaque_order, opaque_index, translucent_units


# --- rendering the ownership stack ---------------------------------------------


def _code_hex(code):
    return f"#{code[0]:02x}{code[1]:02x}{code[2]:02x}"


def _render_current_tree(root, box):
    """Serialise the (already mutated) tree and render it to an RGB array."""
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    text = ET.tostring(root, encoding="unicode")
    with tempfile.NamedTemporaryFile(
        suffix=".svg", mode="w", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(text)
        tmp_path = fh.name
    try:
        return render_svg(tmp_path, box)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _nearest_labels(rgb, candidates):
    """Map an RGB render to nearest-candidate indices (0 == white/background)."""
    h, w = rgb.shape[:2]
    cand = np.asarray(candidates, dtype=np.float32)
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    labels = np.empty(pixels.shape[0], dtype=np.int64)
    chunk = 200_000
    for start in range(0, pixels.shape[0], chunk):
        block = pixels[start : start + chunk]
        d = np.linalg.norm(block[:, None, :] - cand[None, :, :], axis=2)
        labels[start : start + chunk] = np.argmin(d, axis=1)
    return labels.reshape(h, w)


def _render_region_map(root, leaves, opaque_order, opaque_index, translucent_units, box):
    """Render the ownership stack at the given box size, returning a region map.

    Region value 0 == background. Every other value is a distinct planar face.
    For a pixel whose topmost paint is OPAQUE, the face is simply that opaque
    colour (whatever translucent paint lies buried beneath is not visible). For a
    pixel whose topmost paint is TRANSLUCENT, the face is identified by (topmost
    opaque owner beneath, bitmask of translucent leaves covering the pixel), so
    two overlapping translucent shapes form their own third face. Also returns
    region_meta: id -> (opaque_label, bitmask).
    """
    n_opaque = len(opaque_order)
    has_translucent = bool(translucent_units)
    # One extra code (the marker) is needed to flag translucent-topmost pixels.
    all_codes = [tuple(int(x) for x in c) for c in _codes_for(n_opaque + (1 if has_translucent else 0))]
    codes = all_codes[:n_opaque]
    marker = all_codes[n_opaque] if has_translucent else None

    def base_paint(lf, ch):
        kind = lf.get(ch + "_kind")
        if kind == "opaque":
            return _code_hex(codes[opaque_index[lf[ch]]])
        if kind == "paper":
            return "#ffffff"
        return "none"  # translucent (hidden here) or no paint

    # Base render: opaque faces only, translucent leaves hidden. Gives the
    # topmost opaque owner (or background) beneath everything translucent.
    for lf in leaves:
        lf["elem"].set("fill", base_paint(lf, "fill"))
        lf["elem"].set("stroke", base_paint(lf, "stroke"))
    base_rgb = _render_current_tree(root, box)
    opaque_labels = _nearest_labels(base_rgb, [(255, 255, 255)] + codes)  # 0..K

    if not has_translucent:
        # Fast path: no overlays, the base render is the whole answer.
        key = opaque_labels.astype(np.int64)
        return _keys_to_regions(key)

    # Combined render: opaque faces keep their colour, every translucent leaf
    # paints the single marker colour. The topmost paint wins per crispEdges, so
    # a pixel showing an opaque code has an opaque top (translucent buried below),
    # and a pixel showing the marker has a translucent top.
    def combined_paint(lf, ch):
        kind = lf.get(ch + "_kind")
        if kind == "opaque":
            return _code_hex(codes[opaque_index[lf[ch]]])
        if kind == "paper":
            return "#ffffff"
        if kind == "translucent":
            return _code_hex(marker)
        return "none"

    for lf in leaves:
        lf["elem"].set("fill", combined_paint(lf, "fill"))
        lf["elem"].set("stroke", combined_paint(lf, "stroke"))
    combined_rgb = _render_current_tree(root, box)
    combined = _nearest_labels(combined_rgb, [(255, 255, 255)] + codes + [marker])
    marker_label = n_opaque + 1  # index of the marker in the candidate list

    # One solo mask render per translucent unit: where does that leaf paint?
    bitmask = np.zeros(opaque_labels.shape, dtype=np.uint64)
    for bit, (leaf_idx, ch) in enumerate(translucent_units):
        for lf in leaves:
            lf["elem"].set("fill", "none")
            lf["elem"].set("stroke", "none")
        leaves[leaf_idx]["elem"].set(ch, _MASK_PAINT)
        solo_rgb = _render_current_tree(root, box)
        covered = solo_rgb.min(axis=2) < _MASK_COVERED_BELOW
        bitmask[covered] |= np.uint64(1) << np.uint64(bit)

    # Where the top paint is opaque, the face is that opaque colour with an empty
    # overlay stack (bitmask 0). Where the top paint is translucent, the face is
    # (opaque owner beneath, overlay bitmask). Elsewhere it is background.
    opaque_top = (combined >= 1) & (combined <= n_opaque)
    translucent_top = combined == marker_label
    owner = np.where(opaque_top, combined, opaque_labels).astype(np.int64)
    bits = np.where(translucent_top, bitmask.astype(np.int64), np.int64(0))
    key = (bits << np.int64(16)) | owner
    return _keys_to_regions(key)


def _keys_to_regions(key):
    """Turn a per-pixel packed region key into a compact region map + meta."""
    uniq = np.unique(key)
    ids = np.zeros(len(uniq), dtype=np.int64)
    region_meta = {}
    next_id = 1
    for idx, k in enumerate(uniq):
        opaque_label = int(k & 0xFFFF)
        bm = int(k >> np.int64(16))
        if opaque_label == 0 and bm == 0:
            ids[idx] = 0  # background: no owner, no overlay
        else:
            ids[idx] = next_id
            region_meta[next_id] = (opaque_label, bm)
            next_id += 1
    region_map = ids[np.searchsorted(uniq, key)]
    return region_map, region_meta


# --- downsample with coverage rescue (CLASS A) ---------------------------------


def _rescue_nearest(out, need, radius):
    """Fill each 'need' pixel with the nearest non-background label within radius."""
    src = out.copy()
    h, w = out.shape
    ys, xs = np.where(need)
    for y, x in zip(ys.tolist(), xs.tolist()):
        best_lbl = 0
        best_d = None
        for dy in range(-radius, radius + 1):
            ny = y + dy
            if ny < 0 or ny >= h:
                continue
            for dx in range(-radius, radius + 1):
                nx = x + dx
                if nx < 0 or nx >= w:
                    continue
                lbl = src[ny, nx]
                if lbl != 0:
                    d = dy * dy + dx * dx
                    if best_d is None or d < best_d:
                        best_d = d
                        best_lbl = lbl
        if best_lbl:
            out[y, x] = best_lbl


def _downsample(region_map, clean_rgb, size, ss):
    """Downsample a super-sampled region map to `size` with a coverage-aware vote.

    Per target pixel: majority label over the ss x ss block, EXCEPT when the
    majority is background but the clean render inks that pixel, in which case the
    most common non-background label in the block wins (or, if the block holds no
    non-background label at all, the nearest one within ~2px). This keeps the
    label map explaining every inked pixel of the clean render, which is exactly
    what the coverage_match audit metric measures.
    """
    present = np.unique(region_map)
    blocks = region_map.reshape(size, ss, size, ss)
    counts = {}
    out = np.zeros((size, size), dtype=np.int64)
    best_count = np.zeros((size, size), dtype=np.int64)
    for lbl in present.tolist():
        c = np.sum(blocks == lbl, axis=(1, 3))
        counts[lbl] = c
        take = c > best_count
        out[take] = lbl
        best_count[take] = c[take]

    inked = ink_mask(clean_rgb)

    # Most common NON-background label per block, and whether one exists at all.
    has_nonbg = np.zeros((size, size), dtype=bool)
    best_nonbg = np.zeros((size, size), dtype=np.int64)
    best_nonbg_c = np.zeros((size, size), dtype=np.int64)
    for lbl in present.tolist():
        if lbl == 0:
            continue
        c = counts[lbl]
        has_nonbg |= c > 0
        take = c > best_nonbg_c
        best_nonbg[take] = lbl
        best_nonbg_c[take] = c[take]

    override = (out == 0) & inked & has_nonbg
    out[override] = best_nonbg[override]

    rescue = (out == 0) & inked & (~has_nonbg)
    if rescue.any():
        _rescue_nearest(out, rescue, radius=2)
    return out


# --- palette sampling from the clean render (CLASS B) --------------------------


def _erode1(mask):
    """4-connectivity 1px erosion, border treated as outside (erodes)."""
    out = np.zeros_like(mask)
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return out
    out[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    return out


def _analytic_colour(meta, opaque_order, translucent_units, leaves):
    """Composited-paint colour for a region, the last-resort palette fallback.

    Starts from the opaque owner (or white), then composites each covering
    translucent unit over it in paint order. Only used when a region owns no
    clean pixels at all, which the label map makes practically impossible.
    """
    opaque_label, bm = meta
    if opaque_label > 0:
        colour = tuple(float(c) for c in opaque_order[opaque_label - 1])
    else:
        colour = (255.0, 255.0, 255.0)
    for bit, (leaf_idx, ch) in enumerate(translucent_units):
        if bm & (1 << bit):
            lf = leaves[leaf_idx]
            alpha = lf["fa"] if ch == "fill" else lf["sa"]
            colour = _composite_over(lf[ch], alpha, colour)
    return np.array(colour, dtype=np.float32)


def _sample_colour(mask, clean_rgb, meta, opaque_order, translucent_units, leaves):
    """Median clean-render colour a region owns, eroded 1px to dodge AA edges."""
    eroded = _erode1(mask)
    if eroded.any():
        return np.median(clean_rgb[eroded].astype(np.float32), axis=0)
    if mask.any():
        return np.median(clean_rgb[mask].astype(np.float32), axis=0)
    return _analytic_colour(meta, opaque_order, translucent_units, leaves)


def _finalize(label_map, clean_rgb, region_meta, opaque_order, translucent_units, leaves):
    """Sample each region's colour, fold near-white faces into background, and
    renumber the survivors to a compact palette."""
    present = [int(lbl) for lbl in np.unique(label_map) if lbl != 0]
    palette_rows = [(255, 255, 255)]
    remap = {0: 0}
    next_id = 1
    for old in present:
        mask = label_map == old
        colour = _sample_colour(
            mask, clean_rgb, region_meta.get(old), opaque_order, translucent_units, leaves
        )
        rounded = tuple(int(round(c)) for c in colour)
        if _is_background(rounded):
            # Indistinguishable from the page: a genuine-white or invisible face.
            remap[old] = 0
        else:
            remap[old] = next_id
            palette_rows.append(rounded)
            next_id += 1

    out = np.zeros_like(label_map)
    for old, new in remap.items():
        if new != 0:
            out[label_map == old] = new
    palette = np.array(palette_rows, dtype=np.uint8)
    return out, palette, next_id - 1


def derive_labels_from_svg(svg_path, size, clean_rgb=None, warn_sink=None):
    """Return (label_map, palette, n_declared) derived from SVG geometry.

    label_map: HxW uint8/uint16, label 0 = background, aligned to the clean render
      produced by renderer.render_svg(svg_path, size).
    palette: (N+1)x3 uint8 of each region's colour sampled from the clean render
      (row 0 white). Translucent overlays get their blended colour.
    n_declared: number of distinct foreground faces (== len(palette) - 1).

    clean_rgb: the clean render, if the caller already has it (saves a render).
    warn_sink: optional list; non-fatal warnings (e.g. a 4x render fell back to
      1x) are appended for the caller to record in meta.

    Raises DerivationError for CSS blocks, paint servers, unparseable SVGs, or
    more translucent leaves than the region bitmask can hold.
    """
    svg_path = Path(svg_path)
    try:
        tree = ET.parse(svg_path)
    except ET.ParseError as exc:
        raise DerivationError(f"unparseable SVG XML: {exc}") from exc
    root = tree.getroot()

    _check_no_css(root)
    if clean_rgb is None:
        clean_rgb = render_svg(svg_path, size)

    leaves = _walk_collect(root)
    opaque_order, opaque_index, translucent_units = _classify(leaves)

    args = (root, leaves, opaque_order, opaque_index, translucent_units)
    try:
        region_ss, region_meta = _render_region_map(*args, size * _SUPERSAMPLE)
        label_map = _downsample(region_ss, clean_rgb, size, _SUPERSAMPLE)
    except DerivationError:
        raise
    except Exception as exc:  # noqa: BLE001 - a bad 4x render must not abort the file
        if warn_sink is not None:
            warn_sink.append(f"supersample render failed ({exc}); fell back to 1x")
        region_map, region_meta = _render_region_map(*args, size)
        label_map = _downsample(region_map, clean_rgb, size, 1)

    label_map, palette, n_declared = _finalize(
        label_map, clean_rgb, region_meta, opaque_order, translucent_units, leaves
    )

    dtype = np.uint8 if len(palette) <= 255 else np.uint16
    return label_map.astype(dtype), palette, n_declared
