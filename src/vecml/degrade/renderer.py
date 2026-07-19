"""SVG rendering behind a single backend-agnostic function.

Backend preference order (best quality first):
  1. resvg-py  (bundled Rust wheel, no system libs, chosen default here)
  2. cairosvg  (needs system cairo)
  3. svglib + reportlab (pure python, weakest quality)

The rest of the codebase only ever calls render_svg / render_svg_rgba and
never learns which backend won.
"""

import io
import re

import numpy as np
from PIL import Image

# Resolve the backend once at import time. _BACKEND names the winner so
# callers can log it, but nothing branches on it outside this module.
_BACKEND = None
_render_bytes = None


def _strip_root_size(svg_text: str) -> str:
    """Remove width/height from the root <svg> tag, keeping viewBox.

    Real-world files often declare physical units (width="2.33in") that some
    backends reject. The viewBox alone is enough to size the render.
    """
    match = re.search(r"<svg\b[^>]*>", svg_text, re.S)
    if match is None:
        return svg_text
    root = match.group(0)
    cleaned = re.sub(r'\s(?:width|height)="[^"]*"', "", root)
    return svg_text[: match.start()] + cleaned + svg_text[match.end() :]


def _try_resvg():
    import resvg_py

    def render(svg_path, box):
        # resvg preserves aspect ratio: passing width=height=box fits the art
        # inside a box*box square without distortion (returns e.g. box x (box/2)).
        try:
            return resvg_py.svg_to_bytes(svg_path=str(svg_path), width=box, height=box)
        except ValueError:
            # Retry with root width/height stripped (physical units such as
            # "in"/"cm"/"pt" make resvg report an invalid size).
            with open(svg_path, encoding="utf-8") as fh:
                svg_text = fh.read()
            return resvg_py.svg_to_bytes(
                svg_string=_strip_root_size(svg_text), width=box, height=box
            )

    return render


def _try_cairosvg():
    import cairosvg

    def render(svg_path, box):
        png = cairosvg.svg2png(
            url=str(svg_path),
            output_width=box,
            output_height=box,
        )
        return png

    return render


def _try_svglib():
    from reportlab.graphics import renderPM
    from svglib.svglib import svg2rlg

    def render(svg_path, box):
        drawing = svg2rlg(str(svg_path))
        # Scale the drawing so its longer side maps to box, preserving aspect.
        w = drawing.width or box
        h = drawing.height or box
        scale = box / max(w, h)
        drawing.scale(scale, scale)
        drawing.width = w * scale
        drawing.height = h * scale
        pil = renderPM.drawToPIL(drawing, dpi=72)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()

    return render


def _select_backend():
    global _BACKEND, _render_bytes
    if _render_bytes is not None:
        return
    for name, factory in (
        ("resvg-py", _try_resvg),
        ("cairosvg", _try_cairosvg),
        ("svglib", _try_svglib),
    ):
        try:
            _render_bytes = factory()
            _BACKEND = name
            return
        except Exception:
            continue
    raise RuntimeError(
        "No SVG rendering backend available. Install one of: resvg-py, "
        "cairosvg (plus system cairo), or svglib."
    )


def backend_name() -> str:
    """Return the name of the rendering backend that was selected."""
    _select_backend()
    return _BACKEND


def render_svg_rgba(svg_path, size: int) -> np.ndarray:
    """Render an SVG to an RGBA uint8 array of shape (size, size, 4).

    The art is fit into a square canvas preserving aspect ratio, centred,
    with transparent padding around it (background compositing happens in
    render_svg).
    """
    _select_backend()
    png_bytes = _render_bytes(svg_path, size)
    art = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

    # Paste the aspect-fit art centred onto a transparent square canvas.
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    off_x = (size - art.width) // 2
    off_y = (size - art.height) // 2
    canvas.paste(art, (off_x, off_y))
    return np.asarray(canvas, dtype=np.uint8)


def composite_rgba(rgba: np.ndarray, colour=(255, 255, 255)) -> np.ndarray:
    """Alpha-composite an RGBA array over a solid colour, returning RGB uint8.

    Compositing (rather than pixel substitution) keeps anti-aliased edges
    physically correct on any background colour: a 60% ink pixel blends with
    the actual background instead of dragging white fringes along.
    """
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.broadcast_to(np.asarray(colour, dtype=np.float32), rgb.shape)
    out = rgb * alpha + bg * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def render_svg(svg_path, size: int) -> np.ndarray:
    """Render an SVG to an RGB uint8 array of shape (size, size, 3).

    Transparency (including the aspect-ratio padding) is composited over a
    solid white background, which matches how flat-colour print artwork sits
    on paper.
    """
    return composite_rgba(render_svg_rgba(svg_path, size))
