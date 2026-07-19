"""Checkpoint discovery and single-image inference for the cockpit run tab.

Checkpoints are discovered under the runs/ tree (any */best.pt). The head
type is inferred from the state dict itself (presence of head.label.* keys),
not from checkpoint metadata, so pre-label-era checkpoints load fine.

Inference is CPU: a 1.9M-param UNet at 256px is ~100ms, and CPU sidesteps
MPS thread-safety questions inside the FastAPI worker. One model is kept
warm; switching checkpoints swaps it.
"""

from __future__ import annotations

import base64
import io
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vecml.models.unet import DualHead, UNet
from vecml.train.loop import LABEL_COLOURS


def list_ckpts(root: str | Path = "runs") -> list[dict]:
    """Metadata for every runs/*/best.pt, newest first."""
    out = []
    for p in sorted(Path(root).glob("*/best.pt")):
        try:
            ckpt = torch.load(p, map_location="cpu")
            sd = ckpt["model"]
        except Exception:
            continue
        n_classes = int(sd["head.label.weight"].shape[0]) if "head.label.weight" in sd else 0
        out.append(
            {
                "path": str(p),
                "name": p.parent.name,
                "n_classes": n_classes,
                "loss": ckpt.get("loss"),
                "epoch": ckpt.get("epoch"),
                "step": ckpt.get("step"),
                "params": ckpt.get("params"),
                "mtime": p.stat().st_mtime,
            }
        )
    out.sort(key=lambda c: c["mtime"], reverse=True)
    return out


class _Cache:
    def __init__(self):
        self.lock = threading.Lock()
        self.key: tuple | None = None
        self.model: UNet | None = None
        self.n_classes = 0


_cache = _Cache()


def _load(path: Path) -> tuple[UNet, int]:
    key = (str(path), path.stat().st_mtime)
    with _cache.lock:
        if _cache.key == key:
            return _cache.model, _cache.n_classes
        sd = torch.load(path, map_location="cpu")["model"]
        if "head.label.weight" in sd:
            n_classes = int(sd["head.label.weight"].shape[0])
            model = UNet(head=lambda c: DualHead(c, n_classes))
        else:
            n_classes = 0
            model = UNet()
        model.load_state_dict(sd)
        model.eval()
        _cache.key, _cache.model, _cache.n_classes = key, model, n_classes
        return model, n_classes


def _b64_png(arr_hwc_u8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr_hwc_u8, mode="RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run(ckpt_path: str | Path, img: Image.Image, size: int = 256) -> dict:
    """Run one image through a checkpoint; returns base64 PNGs for the UI.

    Label models additionally return the colourised region map and the
    "flat" image: each predicted region painted with the median colour the
    RGB head gave it. Flat is the crisp, zero-AA raster the engine will eat.
    """
    model, n_classes = _load(Path(ckpt_path))

    rgb = img.convert("RGB").resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x)
    ms = (time.perf_counter() - t0) * 1000

    result: dict = {
        "input": _b64_png(np.asarray(rgb, dtype=np.uint8)),
        "n_classes": n_classes,
        "ms": round(ms, 1),
    }

    pred = out["rgb"] if isinstance(out, dict) else out
    pred_u8 = (pred[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    result["pred"] = _b64_png(pred_u8)

    if isinstance(out, dict):
        lab = out["logits"][0].argmax(0).numpy()
        colours = np.asarray(LABEL_COLOURS)
        result["labels"] = _b64_png(colours[np.clip(lab, 0, len(colours) - 1)])
        flat = np.zeros_like(pred_u8)
        slots = np.unique(lab)
        for s in slots:
            mask = lab == s
            flat[mask] = np.median(pred_u8[mask], axis=0).astype(np.uint8)
        result["flat"] = _b64_png(flat)
        result["n_regions"] = int(len(slots))
    return result
