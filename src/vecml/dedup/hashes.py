"""The three hash primitives the dedup gate is built on.

Each primitive answers a progressively looser notion of "the same image":

  exact_hash       identical file bytes            (re-uploads, mirror copies)
  normalized_hash  identical drawing after         (re-exports, editor churn,
                   canonicalising the XML           whitespace/id noise, float
                                                    reformatting)
  render_phash     visually identical raster        (rescales, tiny recolours,
                   at 64px                           optimiser rewrites)

The render step reuses the repo's single rendering backend
(`vecml.degrade.renderer.render_svg`); no second renderer is introduced.
"""

import hashlib
import re
import xml.etree.ElementTree as ET

import numpy as np

from vecml.degrade.renderer import render_svg

# ---------------------------------------------------------------------------
# Layer 1: exact bytes
# ---------------------------------------------------------------------------


def exact_hash(data: bytes) -> str:
    """SHA-256 of the raw file bytes, hex-encoded."""
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Layer 2: normalized-SVG hash
# ---------------------------------------------------------------------------

# Attributes that carry no geometry and only add churn between re-exports.
_DROP_ATTRS = {"id", "class", "style"}
# Namespaces used exclusively by editors (Inkscape, Sodipodi, Illustrator,
# Sketch, Figma, RDF/Dublin-Core metadata). Anything under these is decoration.
_EDITOR_NS_HINTS = (
    "inkscape",
    "sodipodi",
    "adobe",
    "illustrator",
    "sketch",
    "figma",
    "purl.org/dc",
    "creativecommons",
    "www.w3.org/1999/02/22-rdf",
)
# Elements that never affect the drawn pixels.
_DROP_TAGS = {"metadata", "title", "desc"}

_NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_HEX_COLOR_RE = re.compile(r"#[0-9A-Fa-f]{3,8}\b")


def _local(tag: str) -> str:
    """Strip the ``{namespace}`` prefix ElementTree prepends to tags/attrs."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _is_editor_ns(tag: str) -> bool:
    return "}" in tag and any(h in tag.split("}", 1)[0].lower() for h in _EDITOR_NS_HINTS)


def _round_numbers(value: str, precision: int) -> str:
    """Round every number embedded in an attribute value to ``precision`` dp.

    Applies uniformly to path ``d`` strings, ``points``, ``transform`` and the
    plain coordinate attributes, so ``10``, ``10.0`` and ``10.00001`` collapse.
    """

    def repl(m: re.Match) -> str:
        try:
            return f"{round(float(m.group()), precision):g}"
        except (ValueError, OverflowError):
            return m.group()

    return _NUM_RE.sub(repl, value)


def _canon_element(el: ET.Element, precision: int, out: list) -> None:
    """Serialise an element into ``out`` in a byte-stable canonical form."""
    tag = _local(el.tag)
    if tag in _DROP_TAGS:
        return
    attrs = []
    for key, val in el.attrib.items():
        if _is_editor_ns(key):
            continue
        name = _local(key)
        if name in _DROP_ATTRS:
            continue
        val = _round_numbers(val.strip(), precision)
        val = _HEX_COLOR_RE.sub(lambda m: m.group().lower(), val)
        attrs.append((name, val))
    attrs.sort()
    out.append("<" + tag)
    for name, val in attrs:
        out.append(f" {name}={val}")
    out.append(">")
    for child in el:
        if isinstance(child.tag, str) and not _is_editor_ns(child.tag):
            _canon_element(child, precision, out)
    out.append("</" + tag + ">")


def normalize_svg(svg_text: str, precision: int = 1) -> str:
    """Canonicalise an SVG string.

    Drops ids/classes/styles, editor-namespace metadata, comments and
    whitespace; rounds coordinates to ``precision`` decimals; sorts attributes.
    Two files that draw the same thing after a re-export produce the same
    string. Falls back to a regex-only pass for XML that will not parse.
    """
    try:
        root = ET.fromstring(svg_text)  # noqa: S314 - local trusted corpus
    except ET.ParseError:
        return _normalize_text_fallback(svg_text, precision)
    out: list[str] = []
    _canon_element(root, precision, out)
    return "".join(out)


def _normalize_text_fallback(svg_text: str, precision: int) -> str:
    """Best-effort normalisation for SVGs that are not well-formed XML."""
    t = re.sub(r"<!--.*?-->", "", svg_text, flags=re.S)
    t = re.sub(r"<(metadata|title|desc)\b.*?</\1\s*>", "", t, flags=re.S | re.I)
    t = re.sub(r'\s(?:id|class|style)="[^"]*"', "", t)
    t = re.sub(r'\s(?:inkscape|sodipodi|sketch):[\w-]+="[^"]*"', "", t, flags=re.I)
    t = _NUM_RE.sub(lambda m: _round_numbers(m.group(), precision), t)
    t = _HEX_COLOR_RE.sub(lambda m: m.group().lower(), t)
    t = re.sub(r">\s+<", "><", t)
    return re.sub(r"\s+", " ", t).strip()


def normalized_hash(svg_text: str, precision: int = 1) -> str:
    """SHA-256 of the canonical SVG string."""
    return hashlib.sha256(normalize_svg(svg_text, precision).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Layer 3: render pHash (DCT, 64-bit, no external dep)
# ---------------------------------------------------------------------------

_DCT_CACHE: dict[int, np.ndarray] = {}


def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II basis matrix, cached per size."""
    m = _DCT_CACHE.get(n)
    if m is None:
        k = np.arange(n).reshape(-1, 1)
        x = np.arange(n).reshape(1, -1)
        m = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
        m[0] *= 1.0 / np.sqrt(2.0)
        m *= np.sqrt(2.0 / n)
        _DCT_CACHE[n] = m
    return m


def phash_from_gray(gray: np.ndarray, hash_side: int = 8) -> int:
    """DCT perceptual hash of a square grayscale float image -> 64-bit int.

    Takes the top-left ``hash_side`` x ``hash_side`` low-frequency DCT block and
    thresholds each coefficient against the block median (excluding the DC
    term), giving ``hash_side**2`` bits.
    """
    n = gray.shape[0]
    d = _dct_matrix(n)
    coeff = d @ gray @ d.T
    block = coeff[:hash_side, :hash_side]
    flat = block.flatten()
    med = np.median(flat[1:])  # exclude DC term from the threshold
    bits = flat > med
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def render_gray(svg_path, size: int = 64) -> np.ndarray:
    """Render an SVG at ``size`` px via the repo renderer, return grayscale."""
    return render_svg(svg_path, size).astype(np.float32) @ _LUMA


def is_degenerate_render(
    gray: np.ndarray, std_floor: float = 8.0, ink_lo: float = 0.02, ink_hi: float = 0.98
) -> bool:
    """True if a render is too flat to perceptually hash meaningfully.

    Blank (all-white), solid (all-ink) and near-flat renders produce a constant
    image whose DCT is all-DC; the low-frequency block is then floating-point
    noise thresholded against a ~zero median, so its pHash is unstable and
    collides arbitrarily. Such images cannot be perceptually deduplicated and
    are left to the exact and normalised layers instead.
    """
    ink = float((gray < 250).mean())
    return bool(gray.std() < std_floor or ink < ink_lo or ink > ink_hi)


def render_phash(svg_path, size: int = 64, hash_side: int = 8) -> int:
    """Render an SVG at ``size`` px via the repo renderer and pHash it."""
    return phash_from_gray(render_gray(svg_path, size), hash_side)


def hamming(a: int, b: int) -> int:
    """Hamming distance between two hashes stored as ints."""
    return (a ^ b).bit_count()
