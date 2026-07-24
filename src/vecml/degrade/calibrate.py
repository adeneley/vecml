"""Forensic calibration diagnostics for the wrecker.

This is the *measurement* half of README section 3, not the full study: given a
directory of v2-wrecked images (a tree of ``wreck_svg`` outputs, each with a
``meta.json`` naming the family per variant) and a directory of real damaged
images, it computes cheap, mostly training-free per-image forensic features and
reports how the synthetic per-family feature distributions line up against the
real set.

Features (README section 3, Step 1):

* estimated noise sigma — ``skimage.restoration.estimate_sigma`` when installed,
  otherwise a numpy median-absolute-deviation wavelet estimator (same idea);
* JPEG quality — inverted from the luminance quantization table via Pillow
  (exact for standard IJG tables, approximate for Photoshop/Office tables);
* edge-ringing energy — high-pass residual power in the band adjacent to hard
  edges, the direct probe for the sinc/Gibbs artefact families.

The heavier sim-to-real harness (C2ST / proxy-A-distance, KID under a frozen
resize+JPEG path) is deliberately left as a seam: ``distribution_distance`` below
documents the interface and raises, so the cheap diagnostics ship now without
pretending the north-star number is done.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

# Standard JPEG (Annex K) luminance quantization base table, row-major 8x8.
_IJG_LUMA_BASE = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.float64)


def estimate_noise_sigma(gray):
    """Estimate additive-noise sigma. Uses skimage if present, else a numpy
    MAD-of-diagonal-detail estimator (Donoho's robust wavelet sigma)."""
    try:
        from skimage.restoration import estimate_sigma

        return float(estimate_sigma(gray, channel_axis=None))
    except Exception:
        # Diagonal high-frequency detail via a Haar-like difference, then the
        # robust MAD estimator sigma = median(|d|) / 0.6745.
        d = gray[:-1, :-1] - gray[:-1, 1:] - gray[1:, :-1] + gray[1:, 1:]
        d = d / 4.0
        mad = np.median(np.abs(d - np.median(d)))
        return float(mad / 0.6745)


def jpeg_quality(path):
    """Estimate JPEG %Q from the luminance quantization table, or None if the
    file carries no quant tables (PNG/WebP/decoded).

    Inverts the IJG scaling law using the order-invariant mean of the table:
    quant ~= (base * S + 50) / 100, so S ~= (100*mean(quant) - 50) / mean(base),
    then Q from S (Q<50: S=5000/Q; else S=200-2Q)."""
    try:
        im = Image.open(path)
        qt = getattr(im, "quantization", None)
    except Exception:
        return None
    if not qt:
        return None
    luma = np.asarray(qt[0], dtype=np.float64)
    if luma.size != 64:
        return None
    scale = (100.0 * luma.mean() - 50.0) / _IJG_LUMA_BASE.mean()
    scale = max(scale, 1e-6)
    q = (5000.0 / scale) if scale > 100.0 else (200.0 - scale) / 2.0
    return float(np.clip(q, 1.0, 100.0))


def edge_ringing_energy(gray):
    """High-pass residual power in the ring just off hard edges — where sinc /
    Gibbs ringing concentrates. Normalised by the image's overall contrast so
    it is comparable across content."""
    import cv2

    g = gray.astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), sigmaX=1.5)
    highpass = g - blur
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    edges = (lap > np.percentile(lap, 98)).astype(np.uint8)
    if edges.sum() == 0:
        return 0.0
    k = np.ones((5, 5), np.uint8)
    dil = cv2.dilate(edges, k)
    ring = (dil > 0) & (edges == 0)
    if ring.sum() == 0:
        return 0.0
    contrast = g.std() + 1e-6
    return float(np.abs(highpass[ring]).mean() / contrast)


def profile_image(path):
    """Return the per-image forensic feature dict for one image file."""
    path = Path(path)
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    gray = rgb.mean(axis=2)
    return {
        "noise_sigma": estimate_noise_sigma(gray),
        "jpeg_quality": jpeg_quality(path),
        "edge_ringing": edge_ringing_energy(gray),
    }


def profile_wrecked_dir(root):
    """Walk a wreck_svg output tree and profile every wrecked variant, grouped
    by the family recorded in each sample's meta.json. Non-v2 outputs (no
    family stamp) fall under the key '_v1'."""
    root = Path(root)
    by_family = {}
    for meta_path in root.rglob("meta.json"):
        meta = json.loads(meta_path.read_text())
        for v in meta.get("variants", []):
            fam = v.get("family", "_v1")
            img = meta_path.parent / v["file"]
            if img.exists():
                by_family.setdefault(fam, []).append(profile_image(img))
    return by_family


def profile_flat_dir(root, exts=(".png", ".jpg", ".jpeg", ".webp", ".bmp")):
    """Profile every image directly under a directory (the real-damaged set)."""
    root = Path(root)
    feats = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in exts:
            feats.append(profile_image(p))
    return feats


_FEATURES = ("noise_sigma", "jpeg_quality", "edge_ringing")


def _summ(feats):
    out = {}
    for f in _FEATURES:
        vals = np.asarray([x[f] for x in feats if x[f] is not None], dtype=np.float64)
        if vals.size == 0:
            out[f] = {"n": 0}
            continue
        out[f] = {
            "n": int(vals.size),
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
            "p10": round(float(np.percentile(vals, 10)), 4),
            "p50": round(float(np.percentile(vals, 50)), 4),
            "p90": round(float(np.percentile(vals, 90)), 4),
        }
    return out


def compare(wrecked_root, real_root):
    """Build the calibration report: per-family synthetic feature summaries plus
    the real-set summary, so the two distributions can be eyeballed side by
    side. Returns a plain dict (JSON-serialisable)."""
    by_family = profile_wrecked_dir(wrecked_root)
    real = profile_flat_dir(real_root)
    report = {
        "real": {"n_images": len(real), "features": _summ(real)},
        "synthetic": {
            fam: {"n_images": len(feats), "features": _summ(feats)}
            for fam, feats in sorted(by_family.items())
        },
    }
    return report


def distribution_distance(*_args, **_kwargs):
    """Sim-to-real distance harness (C2ST / proxy-A-distance + KID).

    Deliberately unimplemented — this is the documented seam for README
    section 3, Step 2. When built it should train a small CNN to separate real
    from v2 residuals (wrecked - clean) and report proxy-A-distance 2*(1-2*eps)
    with a mandatory positive control, cross-checked by KID (never raw FID)
    under one frozen resize+JPEG path. The cheap per-image diagnostics above
    ship without waiting on it."""
    raise NotImplementedError(
        "C2ST/KID sim-to-real harness is future work (README section 3, Step 2)."
    )
