"""Composable degradation ops that turn a clean render into realistic junk.

Every op has the signature op(rgb, rng, severity) -> rgb, where rgb is an
(H, W, 3) uint8 array, rng is a numpy Generator, and severity is in [0, 1].
Ops are pure functions of (input, rng draws, severity): given the same seeded
rng they reproduce exactly, which is what makes the whole pipeline
deterministic.

The recipe sampler picks an ordered subset of ops for a difficulty tier and
returns [(op_name, severity), ...]. apply_recipe replays it.
"""

import io

import cv2
import numpy as np
from PIL import Image


def _to_pil(rgb):
    return Image.fromarray(rgb, mode="RGB")


def _from_pil(im):
    return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _lerp(a, b, t):
    return a + (b - a) * t


def jpeg_cycle(rgb, rng, severity):
    """Encode/decode as JPEG. Higher severity -> lower quality, more passes.

    Real customer files almost always die as JPEGs, often re-saved several
    times, so this is the single most important op.
    """
    # quality 92 (light) down to 8 (brutal)
    quality = int(round(_lerp(92, 8, severity)))
    passes = 2 if severity > 0.66 else 1
    im = _to_pil(rgb)
    for _ in range(passes):
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        im = Image.open(buf).convert("RGB")
    return _from_pil(im)


def downscale_upscale(rgb, rng, severity):
    """Shrink then re-enlarge with mixed filters (loss of small detail)."""
    h, w = rgb.shape[:2]
    # scale factor from 0.85 (light) down to ~0.15 (brutal)
    factor = _lerp(0.85, 0.15, severity)
    small_w = max(4, int(round(w * factor)))
    small_h = max(4, int(round(h * factor)))

    down_filters = [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_NEAREST]
    up_filters = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST]
    down = down_filters[int(rng.integers(len(down_filters)))]
    up = up_filters[int(rng.integers(len(up_filters)))]

    small = cv2.resize(rgb, (small_w, small_h), interpolation=down)
    back = cv2.resize(small, (w, h), interpolation=up)
    return np.clip(back, 0, 255).astype(np.uint8)


def gaussian_blur(rgb, rng, severity):
    """Isotropic Gaussian blur (defocus, low-res scaling artefacts)."""
    # sigma 0.4 (light) up to ~3.5 (brutal)
    sigma = _lerp(0.4, 3.5, severity)
    k = int(sigma * 3) | 1  # odd kernel roughly covering 3 sigma
    k = max(3, k)
    out = cv2.GaussianBlur(rgb, (k, k), sigmaX=sigma, sigmaY=sigma)
    return np.clip(out, 0, 255).astype(np.uint8)


def gaussian_noise(rgb, rng, severity):
    """Additive Gaussian sensor/compression noise."""
    # std 3 (light) up to ~40 (brutal)
    std = _lerp(3.0, 40.0, severity)
    noise = rng.normal(0.0, std, size=rgb.shape)
    out = rgb.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def posterize(rgb, rng, severity):
    """Reduce bit depth per channel (banding, cheap GIF/PNG re-quantise)."""
    # 7 bits (light) down to 2 bits (brutal)
    bits = int(round(_lerp(7, 2, severity)))
    bits = max(1, min(8, bits))
    shift = 8 - bits
    q = (rgb >> shift) << shift
    # nudge to the middle of each retained band so it does not skew dark
    q = q + (1 << (shift - 1)) if shift > 0 else q
    return np.clip(q, 0, 255).astype(np.uint8)


def dither_palette_crush(rgb, rng, severity):
    """Crush to a small colour palette with dithering (indexed-PNG/GIF look)."""
    # 64 colours (light) down to 4 (brutal)
    n_colours = int(round(_lerp(64, 4, severity)))
    n_colours = max(2, min(256, n_colours))
    im = _to_pil(rgb)
    # Pillow applies Floyd-Steinberg dithering by default in this mode.
    crushed = im.convert("P", palette=Image.ADAPTIVE, colors=n_colours)
    return _from_pil(crushed)


def unsharp_halo(rgb, rng, severity):
    """Oversharpen to produce ringing halos around edges.

    Extremely common in customer files that were 'enhanced' before upload.
    """
    # amount 0.6 (light) up to 3.0 (brutal)
    amount = _lerp(0.6, 3.0, severity)
    radius = _lerp(1.0, 3.0, severity)
    k = int(radius * 3) | 1
    k = max(3, k)
    base = rgb.astype(np.float32)
    blurred = cv2.GaussianBlur(base, (k, k), sigmaX=radius)
    sharp = base + amount * (base - blurred)
    return np.clip(sharp, 0, 255).astype(np.uint8)


# Registry so recipes can be stored/serialised as plain op names.
OPS = {
    "jpeg_cycle": jpeg_cycle,
    "downscale_upscale": downscale_upscale,
    "gaussian_blur": gaussian_blur,
    "gaussian_noise": gaussian_noise,
    "posterize": posterize,
    "dither_palette_crush": dither_palette_crush,
    "unsharp_halo": unsharp_halo,
}

# Ops that model spatial/structural loss and read best applied before the final
# JPEG. jpeg_cycle is handled specially (usually placed last).
_STRUCTURAL = [
    "downscale_upscale",
    "gaussian_blur",
    "unsharp_halo",
    "posterize",
    "dither_palette_crush",
    "gaussian_noise",
]

_DIFFICULTY = {
    "light": {"n_range": (1, 2), "sev_range": (0.10, 0.35), "jpeg_prob": 0.7},
    "medium": {"n_range": (2, 3), "sev_range": (0.30, 0.65), "jpeg_prob": 0.85},
    "brutal": {"n_range": (3, 4), "sev_range": (0.55, 0.95), "jpeg_prob": 0.95},
}


def sample_recipe(rng, difficulty: str = "medium"):
    """Sample an ordered recipe: a list of (op_name, severity) tuples.

    Picks 1-4 ops for the tier. A JPEG cycle is usually included and placed
    last (occasionally second-to-last), mirroring how real files end life as
    JPEGs after whatever else happened to them.
    """
    if difficulty not in _DIFFICULTY:
        raise ValueError(f"unknown difficulty {difficulty!r}, expected one of {list(_DIFFICULTY)}")
    cfg = _DIFFICULTY[difficulty]

    lo, hi = cfg["n_range"]
    n_total = int(rng.integers(lo, hi + 1))

    use_jpeg = rng.random() < cfg["jpeg_prob"]
    n_structural = n_total - 1 if use_jpeg else n_total
    n_structural = max(0, min(n_structural, len(_STRUCTURAL)))

    chosen = list(rng.choice(_STRUCTURAL, size=n_structural, replace=False)) if n_structural else []

    s_lo, s_hi = cfg["sev_range"]

    def sev():
        return float(round(rng.uniform(s_lo, s_hi), 3))

    recipe = [(name, sev()) for name in chosen]

    if use_jpeg:
        jpeg_step = ("jpeg_cycle", sev())
        # Usually last; sometimes (20%) second to last when other ops exist.
        if recipe and rng.random() < 0.2:
            recipe.insert(len(recipe) - 1 if len(recipe) >= 1 else 0, jpeg_step)
        else:
            recipe.append(jpeg_step)

    return recipe


def apply_recipe(clean_rgb: np.ndarray, recipe, rng) -> np.ndarray:
    """Apply an ordered recipe to a clean RGB image and return the wrecked RGB.

    Deterministic given the numpy Generator passed in (the caller seeds it).
    """
    out = clean_rgb
    for name, severity in recipe:
        out = OPS[name](out, rng, severity)
    return out
