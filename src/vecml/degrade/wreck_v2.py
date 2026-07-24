"""Wrecker v2 — correlated capture-path families.

v1 (`wreck.py`) draws independent photometric ops from a flat pool at one of
three narrow severity tiers. v2 instead draws a *capture-path family* first
(`jpeg_chain`, `web_upscale`, `scan`, `office_roundtrip`, `phone_photo`,
`lowres_pdf`, `mild`), then runs that family's ordered, parameter-correlated op
bundle at a single continuous per-sample global severity ``s in [0, 1]``. Each
family models one way a real customer file was captured and re-saved, so the ops
co-vary the way a real pipeline does (a scan carries paper + skew + halftone +
scanner noise *together*, not one at a time).

Contract, kept compatible with v1:

* Every op is a pure function of ``(rgb, rng, params)`` — a resolved parameter
  dict instead of v1's single ``severity`` scalar. Given the same input, the
  same rng stream, and the same params, an op reproduces byte-for-byte.
* Recipe *sampling* (which ops fire, their concrete parameters) and recipe
  *application* (running the ops, including any per-pixel noise draws) use
  SEPARATE rng streams. This is what makes a v2 sample replayable from its
  logged params alone: reconstruct the recipe dict, re-seed the apply stream,
  and the output is identical regardless of how the recipe was first drawn.
* Geometric ops additionally return the homography they applied so the caller
  can warp the label map by the same transform (the matched-warp label
  invariant, README section 2.4, option A).

The full resolved recipe is logged per sample into ``meta.json`` (README 2.5),
which is the join key for calibration (``calibrate.py``) and the answer key for
the geometric label warp.
"""

import io

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Default family mix (documented, tunable — NOT magic).
#
# The README ships a provisional prior keyed to the print-shop taxonomy. This
# default tilts that prior toward the semantically-hard, differentiating
# families. Fresh competitive probing (runs/SCOREBOARD.md, 21 Jul) shows the
# reference commercial tracer already recovers ~98% of gaussian noise and ~89%
# of blur (table stakes) but only ~38% of downscale/upscale, ~0% of
# posterize/banding (it traces the bands as content), and drops 0.5px hairlines
# entirely. A model differentiates where that pipeline fails, so the mix
# over-represents the families that produce banding / palette crush / aggressive
# downscale / hairline-eroding blur+downscale: web_upscale, office_roundtrip,
# lowres_pdf. Scan and phone_photo (realism/geometry families) are weighted
# down until the corpus tag (README section 3, Step 0) sets the real shares.
# Overwrite this from the corpus tag before the real mint.
# ---------------------------------------------------------------------------
DEFAULT_MIX = {
    "jpeg_chain": 0.20,
    "web_upscale": 0.22,
    "office_roundtrip": 0.18,
    "lowres_pdf": 0.14,
    "scan": 0.12,
    "phone_photo": 0.08,
    "mild": 0.06,
}

RECIPE_VERSION = "wreck-v2"


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------
def _lerp(a, b, t):
    return a + (b - a) * t


def _to_pil(rgb):
    return Image.fromarray(rgb, mode="RGB")


def _from_pil(im):
    return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _clip(x):
    return np.clip(x, 0, 255).astype(np.uint8)


_INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
}


# ---------------------------------------------------------------------------
# codec round-trip (JPEG / WebP) with forced subsampling and grid shift
# ---------------------------------------------------------------------------
def _codec_roundtrip(rgb, codec, quality, subsampling):
    """Encode then decode through JPEG or WebP.

    subsampling: 0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0 (JPEG only; forced rather than
    left to Pillow's default so chroma-edge fringing is deterministic).
    """
    im = _to_pil(rgb)
    buf = io.BytesIO()
    if codec == "WEBP":
        im.save(buf, format="WEBP", quality=int(quality))
    else:
        im.save(buf, format="JPEG", quality=int(quality), subsampling=int(subsampling))
    buf.seek(0)
    return _from_pil(Image.open(buf))


# ---------------------------------------------------------------------------
# param-driven ops. signature: op(rgb, rng, params) -> rgb
# geometric ops:                op(rgb, rng, params) -> (rgb, 3x3 homography)
# ---------------------------------------------------------------------------
def op_jpeg_chain(rgb, rng, p):
    """n-pass re-encode; per-pass independent quality/codec/subsampling and an
    8x8 grid shift between passes (models the misaligned NADQ re-save chain)."""
    arr = rgb
    qs = p["qualities"]
    subs = p["subsampling"]
    codecs = p["codecs"]
    shifts = p["grid_shifts"]
    for q, sub, codec, sh in zip(qs, subs, codecs, shifts):
        if sh:
            arr = np.roll(arr, (sh, sh), axis=(0, 1))
        arr = _codec_roundtrip(arr, codec, q, sub)
        if sh:
            arr = np.roll(arr, (-sh, -sh), axis=(0, 1))
    return arr


def op_jpeg_cycle(rgb, rng, p):
    """Single-codec re-encode, optionally repeated on the aligned grid."""
    arr = rgb
    for _ in range(int(p.get("passes", 1))):
        arr = _codec_roundtrip(arr, p.get("codec", "JPEG"), p["quality"], p.get("subsampling", 2))
    return arr


def op_web_upscale(rgb, rng, p):
    """Favicon path: shrink -> low-Q JPEG AT THE SMALL SIZE -> (palette crush)
    -> enlarge to canvas -> optional final JPEG.

    This ordering is the whole point: the 8x8 DCT block grid is baked at the
    small resolution, so enlarging magnifies it into ~scale*8 px macroblocks.
    v1's atomic downscale_upscale can never produce that (README section 2.2).
    """
    h, w = rgb.shape[:2]
    s = int(p["small_size"])
    small = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_AREA)
    small = _codec_roundtrip(small, p["small_codec"], p["small_quality"], p["small_subsampling"])
    if p.get("posterize_bits"):
        small = op_posterize(small, rng, {"bits": p["posterize_bits"]})
    if p.get("dither_colours"):
        small = op_dither(small, rng, {"n_colours": p["dither_colours"]})
    up = cv2.resize(small, (w, h), interpolation=_INTERP[p["interp"]])
    if p.get("final_quality"):
        up = _codec_roundtrip(up, "JPEG", p["final_quality"], 2)
    return _clip(up)


def op_downscale_upscale(rgb, rng, p):
    h, w = rgb.shape[:2]
    sw = max(4, int(round(w * p["factor"])))
    sh = max(4, int(round(h * p["factor"])))
    small = cv2.resize(rgb, (sw, sh), interpolation=_INTERP[p["down_filter"]])
    back = cv2.resize(small, (w, h), interpolation=_INTERP[p["up_filter"]])
    return _clip(back)


def op_nearest_decimate(rgb, rng, p):
    """Integer-stride nearest downsample (aliasing, no smooth resample trace)
    then nearest enlarge back."""
    h, w = rgb.shape[:2]
    stride = int(p["stride"])
    small = rgb[::stride, ::stride]
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return _clip(back)


def op_gaussian_blur(rgb, rng, p):
    sigma = float(p["sigma"])
    k = max(3, int(sigma * 3) | 1)
    return _clip(cv2.GaussianBlur(rgb, (k, k), sigmaX=sigma, sigmaY=sigma))


def op_aniso_motion_blur(rgb, rng, p):
    """Anisotropic gaussian, motion-line, or defocus-disc blur from a resolved
    kernel (analytic, not learned)."""
    kind = p["kind"]
    if kind == "motion":
        n = int(p["length"]) | 1
        n = max(3, n)
        k = np.zeros((n, n), np.float32)
        k[n // 2, :] = 1.0
        m = cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), float(p["angle"]), 1.0)
        k = cv2.warpAffine(k, m, (n, n))
    elif kind == "defocus":
        r = max(1, int(p["radius"]))
        n = 2 * r + 1
        yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
        k = ((xx * xx + yy * yy) <= r * r).astype(np.float32)
    else:  # anisotropic gaussian
        sx, sy = float(p["sigma_x"]), float(p["sigma_y"])
        r = max(1, int(max(sx, sy) * 3))
        n = 2 * r + 1
        yy, xx = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float32)
        th = np.deg2rad(float(p["angle"]))
        xr = xx * np.cos(th) + yy * np.sin(th)
        yr = -xx * np.sin(th) + yy * np.cos(th)
        k = np.exp(-(xr * xr) / (2 * sx * sx) - (yr * yr) / (2 * sy * sy))
    s = k.sum()
    if s <= 0:
        return rgb
    k = k / s
    return _clip(cv2.filter2D(rgb, -1, k))


def op_sinc_ringing(rgb, rng, p):
    """Ideal circular low-pass in the frequency domain -> Gibbs ringing at hard
    edges (distinct from unsharp overshoot; Real-ESRGAN's final sinc; its
    ablation credits 'text and lines' — our exact regime)."""
    cutoff = float(p["cutoff"])  # fraction of Nyquist radius kept, (0, 1]
    h, w = rgb.shape[:2]
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    radius = np.sqrt(fy * fy + fx * fx)
    mask = (radius <= (0.5 * cutoff)).astype(np.float32)
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        f = np.fft.fft2(rgb[:, :, c].astype(np.float32))
        out[:, :, c] = np.real(np.fft.ifft2(f * mask))
    return _clip(out)


def op_gaussian_noise(rgb, rng, p):
    std = float(p["std"])
    if p.get("gray"):
        noise = rng.normal(0.0, std, size=rgb.shape[:2])[:, :, None]
    else:
        noise = rng.normal(0.0, std, size=rgb.shape)
    return _clip(rgb.astype(np.float32) + noise)


def op_poisson_noise(rgb, rng, p):
    """Signal-dependent shot noise. Lower gain -> more noise (fewer photons)."""
    gain = float(p["gain"])
    base = rgb.astype(np.float32)
    if p.get("gray"):
        lum = base.mean(axis=2, keepdims=True)
        noisy = rng.poisson(np.clip(lum, 0, None) * gain) / gain
        out = base + (noisy - lum)
    else:
        out = rng.poisson(np.clip(base, 0, None) * gain) / gain
    return _clip(out)


def op_unsharp_halo(rgb, rng, p):
    amount = float(p["amount"])
    radius = float(p["radius"])
    k = max(3, int(radius * 3) | 1)
    base = rgb.astype(np.float32)
    blurred = cv2.GaussianBlur(base, (k, k), sigmaX=radius)
    return _clip(base + amount * (base - blurred))


def op_posterize(rgb, rng, p):
    bits = max(1, min(8, int(round(p["bits"]))))
    shift = 8 - bits
    q = (rgb >> shift) << shift
    if shift > 0:
        q = q + (1 << (shift - 1))
    return _clip(q)


def op_dither(rgb, rng, p):
    n = max(2, min(256, int(round(p["n_colours"]))))
    crushed = _to_pil(rgb).convert("P", palette=Image.ADAPTIVE, colors=n)
    return _from_pil(crushed)


def op_paper(rgb, rng, p):
    """Composite the art onto a tinted, fibre-textured paper by multiplication
    (paper under ink: whites take the tint, saturated inks barely move)."""
    h, w = rgb.shape[:2]
    tint = np.asarray(p["tint"], dtype=np.float32)
    paper = np.broadcast_to(tint, (h, w, 3)).copy()
    # low-frequency shade variation
    lf = rng.normal(0.0, 1.0, size=(max(2, h // 32), max(2, w // 32)))
    lf = cv2.resize(lf, (w, h), interpolation=cv2.INTER_CUBIC)
    paper *= 1.0 + float(p["shade"]) * lf[:, :, None]
    # high-frequency fibre grain
    fibre = rng.normal(0.0, float(p["fibre"]), size=(h, w, 1))
    paper += fibre * 255.0
    paper = np.clip(paper, 0, 255)
    out = rgb.astype(np.float32) * paper / 255.0
    return _clip(out)


def op_bleed_through(rgb, rng, p):
    """Faint mirrored ghost of the front content showing through the paper."""
    back = np.fliplr(rgb).astype(np.float32)
    k = max(3, int(float(p["blur"]) * 3) | 1)
    back = cv2.GaussianBlur(back, (k, k), sigmaX=float(p["blur"]))
    back = np.roll(back, (int(p["offset"]), int(p["offset"])), axis=(0, 1))
    back_ink = (255.0 - back) / 255.0  # 0 on paper, ->1 on mirrored ink
    out = rgb.astype(np.float32) * (1.0 - float(p["opacity"]) * back_ink)
    return _clip(out)


def op_ink_bleed(rgb, rng, p):
    """Glyph erosion/thickening via morphology plus Kanungo-style edge flips —
    the typography killer that thins hairlines and fills counters."""
    ksz = max(1, int(p["ksize"]))
    kernel = np.ones((ksz, ksz), np.uint8)
    if p["mode"] == "thicken":
        out = cv2.erode(rgb, kernel)  # min filter spreads dark ink
    else:
        out = cv2.dilate(rgb, kernel)  # max filter thins dark ink
    flip = float(p.get("flip", 0.0))
    if flip > 0:
        gray = out.mean(axis=2)
        edge = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
        edge_mask = np.abs(edge) > 8.0
        r = rng.random(gray.shape) < flip
        hit = edge_mask & r
        out = out.copy()
        out[hit] = 255 - out[hit]
    return _clip(out)


def op_halftone(rgb, rng, p):
    """Simplified descreen: rotated dot screen per channel at a given LPI plus a
    low-pass descreen, producing the scanner rosette / moire of a printed
    original re-digitised."""
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    freq = float(p["lpi"]) / max(h, w)  # cycles per pixel (coarse proxy)
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        ang = np.deg2rad(float(p["angles"][c]))
        u = xx * np.cos(ang) + yy * np.sin(ang)
        v = -xx * np.sin(ang) + yy * np.cos(ang)
        screen = 0.5 * (np.sin(2 * np.pi * freq * u) * np.sin(2 * np.pi * freq * v) + 1.0)
        chan = rgb[:, :, c].astype(np.float32) / 255.0
        dots = (chan > screen).astype(np.float32) * 255.0
        out[:, :, c] = dots
    dsig = float(p["descreen"])
    if dsig > 0:
        k = max(3, int(dsig * 3) | 1)
        out = cv2.GaussianBlur(out, (k, k), sigmaX=dsig)
    # blend the screened result back toward the source by descreen strength
    mix = float(p.get("mix", 0.6))
    out = _lerp(rgb.astype(np.float32), out, mix)
    return _clip(out)


def op_illumination(rgb, rng, p):
    """Low-frequency multiplicative lighting gradient across the frame."""
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = np.deg2rad(float(p["angle"]))
    g = (xx / w) * np.cos(ang) + (yy / h) * np.sin(ang)
    g = (g - g.min()) / (np.ptp(g) + 1e-6)
    gain = 1.0 + float(p["strength"]) * (g - 0.5)
    return _clip(rgb.astype(np.float32) * gain[:, :, None])


def op_cast_shadow(rgb, rng, p):
    """A soft-edged darkened polygon dropped across the frame."""
    h, w = rgb.shape[:2]
    pts = np.asarray(p["poly"], dtype=np.int32)
    mask = np.zeros((h, w), np.float32)
    cv2.fillPoly(mask, [pts], 1.0)
    k = max(3, int(float(p["blur"]) * 3) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), sigmaX=float(p["blur"]))
    gain = 1.0 - float(p["opacity"]) * mask
    return _clip(rgb.astype(np.float32) * gain[:, :, None])


def op_geometric_warp(rgb, rng, p):
    """Skew / rotate / mild perspective. Returns (warped_rgb, 3x3 homography)
    so the caller can warp the label map by the identical transform."""
    h, w = rgb.shape[:2]
    M = np.asarray(p["matrix"], dtype=np.float64)
    out = cv2.warpPerspective(
        rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return _clip(out), M


OPS_V2 = {
    "jpeg_chain": op_jpeg_chain,
    "jpeg_cycle": op_jpeg_cycle,
    "web_upscale": op_web_upscale,
    "downscale_upscale": op_downscale_upscale,
    "nearest_decimate": op_nearest_decimate,
    "gaussian_blur": op_gaussian_blur,
    "aniso_motion_blur": op_aniso_motion_blur,
    "sinc_ringing": op_sinc_ringing,
    "gaussian_noise": op_gaussian_noise,
    "poisson_noise": op_poisson_noise,
    "unsharp_halo": op_unsharp_halo,
    "posterize": op_posterize,
    "dither": op_dither,
    "paper": op_paper,
    "bleed_through": op_bleed_through,
    "ink_bleed": op_ink_bleed,
    "halftone": op_halftone,
    "illumination": op_illumination,
    "cast_shadow": op_cast_shadow,
    "geometric_warp": op_geometric_warp,
}

GEOMETRIC_OPS = {"geometric_warp"}


# ---------------------------------------------------------------------------
# recipe sampling helpers
# ---------------------------------------------------------------------------
def _u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def _scaled(lo, hi, s):
    """Endpoint that moves from lo (s=0) to hi (s=1)."""
    return _lerp(lo, hi, s)


def _scaled_u(rng, floor, top_lo, top_hi, s):
    """Uniform draw between a fixed floor and an s-scaled ceiling."""
    top = _lerp(top_hi, top_lo, s)  # ceiling drops as severity rises
    return _u(rng, min(floor, top), max(floor, top))


def _grid_shifts(rng, passes):
    # first pass aligned, subsequent passes misalign 0-7px (NADQ)
    return [0] + [int(rng.integers(0, 8)) for _ in range(passes - 1)]


def _codec_choice(rng, webp_prob):
    return "WEBP" if rng.random() < webp_prob else "JPEG"


def _warp_matrix(rng, max_deg, perspective, size):
    """Build a 3x3 homography: small rotation/skew plus optional mild
    perspective. Magnitudes kept modest so QC reconstruction still holds."""
    h = w = size
    ang = np.deg2rad(_u(rng, -max_deg, max_deg))
    cx, cy = w / 2.0, h / 2.0
    ca, sa = np.cos(ang), np.sin(ang)
    R = np.array([[ca, -sa, cx - ca * cx + sa * cy],
                  [sa, ca, cy - sa * cx - ca * cy],
                  [0, 0, 1]], dtype=np.float64)
    if perspective > 0:
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        j = perspective * min(w, h)
        dst = src + rng.uniform(-j, j, size=src.shape).astype(np.float32)
        P = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
        return P @ R
    return R


# ---------------------------------------------------------------------------
# family recipe builders. each returns a list of {"op", "params"} steps.
# ---------------------------------------------------------------------------
def _f_jpeg_chain(rng, s, size):
    steps = []
    if rng.random() < 0.3:
        steps.append({"op": "aniso_motion_blur",
                      "params": {"kind": "aniso", "sigma_x": _scaled(0.2, 1.2, s),
                                 "sigma_y": _scaled(0.2, 0.9, s), "angle": _u(rng, 0, 180)}})
    if rng.random() < 0.4:
        f = _scaled(1.0, 0.7, s)
        steps.append({"op": "downscale_upscale",
                      "params": {"factor": f, "down_filter": "area", "up_filter": "bilinear"}})
    passes = int(rng.choice([2, 3, 4], p=_pass_weights(s)))
    steps.append({"op": "jpeg_chain", "params": {
        "passes": passes,
        "qualities": [int(_scaled_u(rng, 30, 55, 95, s)) for _ in range(passes)],
        "subsampling": [2 if rng.random() < 0.7 else 0 for _ in range(passes)],
        "codecs": [_codec_choice(rng, 0.2) for _ in range(passes)],
        "grid_shifts": _grid_shifts(rng, passes),
    }})
    return steps


def _f_web_upscale(rng, s, size):
    small = int(_scaled(96, 16, s))
    interp = str(rng.choice(["nearest", "bicubic", "bilinear"], p=[0.4, 0.4, 0.2]))
    p = {
        "small_size": small,
        "small_codec": _codec_choice(rng, 0.15),
        "small_quality": int(_scaled_u(rng, 30, 55, 80, s)),
        "small_subsampling": 2,
        "interp": interp,
        "final_quality": int(_scaled(92, 70, s)) if rng.random() < 0.5 else None,
    }
    if rng.random() < 0.4:
        if rng.random() < 0.5:
            p["posterize_bits"] = int(round(_scaled(6, 3, s)))
        else:
            p["dither_colours"] = int(round(_scaled(48, 6, s)))
    return [{"op": "web_upscale", "params": p}]


def _f_scan(rng, s, size):
    steps = [{"op": "paper", "params": {
        "tint": _paper_tint(rng), "shade": _scaled(0.02, 0.12, s), "fibre": _scaled(0.003, 0.02, s)}}]
    if rng.random() < 0.3:
        steps.append({"op": "bleed_through", "params": {
            "blur": _u(rng, 1.0, 2.5), "offset": int(rng.integers(-4, 5)),
            "opacity": _scaled(0.05, 0.2, s)}})
    if rng.random() < 0.6:
        steps.append({"op": "ink_bleed", "params": {
            "mode": str(rng.choice(["thicken", "thin"])), "ksize": 3,
            "flip": _scaled(0.0, 0.05, s)}})
    if rng.random() < 0.7:
        base = _u(rng, 0, 90)
        steps.append({"op": "halftone", "params": {
            "lpi": _u(rng, 65, 185), "angles": [base + 15, base + 75, base + 0],
            "descreen": _u(rng, 0.6, 1.6), "mix": _scaled(0.3, 0.7, s)}})
    steps.append({"op": "geometric_warp", "params": {
        "matrix": _warp_matrix(rng, _scaled(0.5, 4.0, s), 0.0, size).tolist()}})
    if rng.random() < 0.5:
        steps.append({"op": "illumination", "params": {
            "angle": _u(rng, 0, 360), "strength": _scaled(0.05, 0.3, s)}})
    steps.append({"op": "gaussian_noise", "params": {"std": _scaled(2, 10, s), "gray": False}})
    steps.append({"op": "poisson_noise", "params": {"gain": _scaled(0.5, 0.08, s), "gray": True}})
    steps.append({"op": "jpeg_cycle", "params": {
        "quality": int(_scaled(92, 60, s)), "subsampling": 2, "codec": "JPEG", "passes": 1}})
    return steps


def _f_office_roundtrip(rng, s, size):
    steps = []
    if rng.random() < 0.5:
        f = _scaled(1.0, 0.6, s)
        steps.append({"op": "downscale_upscale",
                      "params": {"factor": f, "down_filter": "bilinear", "up_filter": "bicubic"}})
    if rng.random() < 0.6:
        if rng.random() < 0.5:
            steps.append({"op": "posterize", "params": {"bits": int(round(_scaled(7, 4, s)))}})
        else:
            steps.append({"op": "dither", "params": {"n_colours": int(round(_scaled(64, 8, s)))}})
    if rng.random() < 0.5:
        steps.append({"op": "jpeg_cycle", "params": {
            "quality": int(_scaled(92, 72, s)), "subsampling": 2, "codec": "WEBP", "passes": 1}})
    if rng.random() < 0.6:
        steps.append({"op": "jpeg_cycle", "params": {
            "quality": int(_scaled(92, 70, s)), "subsampling": 2, "codec": "JPEG", "passes": 1}})
    if not steps:  # never emit an empty office recipe above s=0
        steps.append({"op": "posterize", "params": {"bits": int(round(_scaled(7, 4, s)))}})
    return steps


def _f_phone_photo(rng, s, size):
    steps = [
        {"op": "paper", "params": {
            "tint": _paper_tint(rng), "shade": _scaled(0.03, 0.15, s), "fibre": _scaled(0.002, 0.01, s)}},
        {"op": "geometric_warp", "params": {
            "matrix": _warp_matrix(rng, _scaled(1.0, 4.0, s), _scaled(0.0, 0.03, s), size).tolist()}},
        {"op": "illumination", "params": {"angle": _u(rng, 0, 360), "strength": _scaled(0.08, 0.35, s)}},
    ]
    if rng.random() < 0.6:
        w = size
        poly = [[int(_u(rng, 0, w)), 0], [int(_u(rng, 0, w)), 0],
                [int(_u(rng, 0, w)), w], [int(_u(rng, 0, w)), w]]
        steps.append({"op": "cast_shadow", "params": {
            "poly": poly, "blur": _u(rng, 8, 25), "opacity": _scaled(0.1, 0.4, s)}})
    steps.append({"op": "aniso_motion_blur", "params": {
        "kind": str(rng.choice(["aniso", "defocus"])),
        "sigma_x": _scaled(0.3, 2.5, s), "sigma_y": _scaled(0.3, 1.5, s),
        "angle": _u(rng, 0, 180), "radius": int(_scaled(1, 4, s))}})
    steps.append({"op": "poisson_noise", "params": {
        "gain": _scaled(0.4, 0.06, s), "gray": rng.random() < 0.4}})
    steps.append({"op": "jpeg_cycle", "params": {
        "quality": int(_scaled(90, 55, s)), "subsampling": 2, "codec": "JPEG", "passes": 1}})
    return steps


def _f_lowres_pdf(rng, s, size):
    steps = [
        {"op": "downscale_upscale", "params": {
            "factor": _scaled(0.6, 0.2, s), "down_filter": "area", "up_filter": "bilinear"}},
        {"op": "gaussian_blur", "params": {"sigma": _scaled(0.4, 2.5, s)}},
    ]
    if rng.random() < 0.5:
        steps.append({"op": "sinc_ringing", "params": {"cutoff": _scaled(0.9, 0.35, s)}})
    steps.append({"op": "jpeg_cycle", "params": {
        "quality": int(_scaled(85, 50, s)), "subsampling": 2, "codec": "JPEG", "passes": 1}})
    return steps


def _f_mild(rng, s, size):
    steps = []
    if rng.random() < 0.5:
        steps.append({"op": "gaussian_blur", "params": {"sigma": _scaled(0.3, 1.0, s)}})
    if rng.random() < 0.4:
        steps.append({"op": "gaussian_noise", "params": {"std": _scaled(1, 5, s), "gray": False}})
    steps.append({"op": "jpeg_cycle", "params": {
        "quality": int(_scaled(95, 80, s)), "subsampling": 0, "codec": "JPEG", "passes": 1}})
    return steps


FAMILY_BUILDERS = {
    "jpeg_chain": _f_jpeg_chain,
    "web_upscale": _f_web_upscale,
    "scan": _f_scan,
    "office_roundtrip": _f_office_roundtrip,
    "phone_photo": _f_phone_photo,
    "lowres_pdf": _f_lowres_pdf,
    "mild": _f_mild,
}


def _pass_weights(s):
    """Weight the JPEG pass count toward more passes as severity rises."""
    w2 = _lerp(0.7, 0.15, s)
    w4 = _lerp(0.1, 0.5, s)
    w3 = max(0.0, 1.0 - w2 - w4)
    total = w2 + w3 + w4
    return [w2 / total, w3 / total, w4 / total]


def _paper_tint(rng):
    base = int(rng.integers(238, 253))
    warm = int(rng.integers(0, 8))
    return [base, base, max(220, base - warm)]


# ---------------------------------------------------------------------------
# top-level sampling + application
# ---------------------------------------------------------------------------
def variant_rngs(variant_seed):
    """Two independent, reproducible rng streams derived from one variant seed.

    Stream 0 drives recipe *sampling* (family, severity, which ops fire, every
    parameter). Stream 1 drives recipe *application* (per-pixel noise draws).
    Splitting them is what lets a sample be replayed from its logged params
    alone: rebuild the recipe dict and re-derive stream 1 from the same variant
    seed, and the output is byte-identical without re-running the sampler.
    """
    sample_ss, apply_ss = np.random.SeedSequence(int(variant_seed)).spawn(2)
    return np.random.default_rng(sample_ss), np.random.default_rng(apply_ss)


def sample_family(rng, mix=None):
    mix = mix or DEFAULT_MIX
    names = list(mix.keys())
    weights = np.asarray([mix[n] for n in names], dtype=np.float64)
    weights = weights / weights.sum()
    return str(rng.choice(names, p=weights))


def sample_severity(rng):
    """Continuous per-sample severity: 12% exact identity (s=0), else Beta(2,3)
    over [0,1] (mass low-mid, thin brutal tail)."""
    if rng.random() < 0.12:
        return 0.0
    return float(rng.beta(2, 3))


def sample_recipe_v2(rng, size=256, mix=None):
    """Draw a full v2 recipe. Returns a dict:

        {"family", "global_severity", "ops": [{"op", "severity", "params"}...]}

    Consumes ``rng`` for all structural draws (family, severity, which ops fire,
    every concrete parameter). Pure passthrough at s=0 (empty op list).
    """
    family = sample_family(rng, mix)
    s = sample_severity(rng)
    if s == 0.0:
        return {"family": family, "global_severity": 0.0, "ops": []}

    steps = FAMILY_BUILDERS[family](rng, s, size)

    # cross-family finishing sinc pass (README 2.3): concentrates ringing at the
    # hard edges every family produces. order-swapped with the family's final op.
    if family != "mild" and rng.random() < 0.6:
        sinc = {"op": "sinc_ringing", "params": {
            "cutoff": _scaled(0.9, 0.4, s), "order_swapped": True}}
        insert_at = max(0, len(steps) - 1)
        steps.insert(insert_at, sinc)

    for st in steps:
        st.setdefault("severity", round(s, 4))
    return {"family": family, "global_severity": round(s, 4), "ops": steps}


def apply_recipe_v2(clean_rgb, recipe, rng):
    """Replay a v2 recipe. Returns (wrecked_rgb, homography_3x3).

    Deterministic given ``rng`` and the resolved ``recipe`` params: seed the
    same apply stream and the same recipe dict and the output is byte-identical,
    independent of how the recipe was originally sampled. The returned
    homography is the composed transform of any geometric ops (identity if none)
    so the caller can warp the label map to match.
    """
    out = clean_rgb
    M = np.eye(3, dtype=np.float64)
    for step in recipe["ops"]:
        fn = OPS_V2[step["op"]]
        if step["op"] in GEOMETRIC_OPS:
            out, m = fn(out, rng, step["params"])
            M = np.asarray(m, dtype=np.float64) @ M
        else:
            out = fn(out, rng, step["params"])
    return out, M


def warp_label_map(label_map, M):
    """Apply a homography to an integer label map with nearest-neighbour
    interpolation and background (index 0) fill outside the source."""
    h, w = label_map.shape[:2]
    return cv2.warpPerspective(
        label_map, np.asarray(M, dtype=np.float64), (w, h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def is_identity(M, tol=1e-6):
    return np.allclose(np.asarray(M), np.eye(3), atol=tol)
