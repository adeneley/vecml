"""Dataset over the wrecked-pair sample dirs from the degrade pipeline.

Each sample dir (named by SVG sha) holds, per the pipeline contract:
  clean.png       shared clean reference render (the target)
  wrecked_XX.png  degraded model inputs (one per variant)
  labels.png      answer-key indices (unused here; this dataset is RGB->RGB)
  meta.json       recipe/qc metadata

An item is (wrecked_rgb, clean_rgb), both float32 CHW in [0, 1]. With
n_classes set, items grow a third element: the label map as int64 HW, indices
remapped to a deterministic slot order (background = 0, remaining palette
colours darkest to lightest) so the classifier sees consistent numbering
regardless of how the generator happened to order the palette.

Overfit mode (the default here) is deterministic: the first N dirs sorted by
name, one fixed variant each, no augmentation. Dirs flagged in
_audit_summary.json (e.g. blank renders) are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _flagged_names(root: Path) -> set[str]:
    """Sample-dir names flagged in the audit summary, if the summary exists."""
    summary = root / "_audit_summary.json"
    if not summary.exists():
        return set()
    data = json.loads(summary.read_text())
    return set(data.get("flagged_names", []))


def _is_sample_dir(p: Path) -> bool:
    """A usable sample dir has a clean render and at least one wrecked variant."""
    if not p.is_dir():
        return False
    return (p / "clean.png").exists() and any(p.glob("wrecked_*.png"))


def list_sample_dirs(root: str | Path, skip_flagged: bool = True) -> list[Path]:
    """Sorted-by-name sample dirs under root, flagged ones optionally removed."""
    root = Path(root)
    flagged = _flagged_names(root) if skip_flagged else set()
    dirs = [
        p
        for p in sorted(root.iterdir())
        if not p.name.startswith("_") and _is_sample_dir(p) and p.name not in flagged
    ]
    return dirs


def _luminance(rgb: list[int]) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def _label_lut(palette_path: Path, n_classes: int) -> np.ndarray:
    """Old palette index -> deterministic slot (bg=0, then dark->light)."""
    pal = json.loads(palette_path.read_text())
    palette, bg = pal["palette"], pal.get("background_index", 0)
    order = sorted(
        (i for i in range(len(palette)) if i != bg),
        key=lambda i: (_luminance(palette[i]), palette[i]),
    )
    lut = np.zeros(max(len(palette), 256), dtype=np.int64)
    for slot, i in enumerate(order, start=1):
        lut[i] = min(slot, n_classes - 1)
    return lut


def _load_labels(d: Path, size: int | None, n_classes: int) -> torch.Tensor:
    img = Image.open(d / "labels.png")
    if size is not None and img.size != (size, size):
        img = img.resize((size, size), Image.NEAREST)
    arr = np.asarray(img, dtype=np.int64)
    lut = _label_lut(d / "palette.json", n_classes)
    return torch.from_numpy(lut[np.clip(arr, 0, len(lut) - 1)])


def _load_rgb(path: Path, size: int | None) -> torch.Tensor:
    """Load a PNG as float32 CHW tensor in [0, 1], optionally resized square."""
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class PairsDataset(Dataset):
    """Wrecked -> clean RGB pairs.

    Parameters
    ----------
    root:
        Directory of sample dirs (e.g. data/audit-500-v2).
    n:
        Cap on number of sample dirs (overfit mode). None = use all.
    size:
        Square resize applied to both images. None = leave native.
    variant:
        Which wrecked variant index to use per dir (deterministic).
    skip_flagged:
        Drop dirs listed in _audit_summary.json flagged_names.
    n_classes:
        When set, items are (wrecked, clean, labels) and dirs whose palette
        exceeds n_classes colours (or lack label files) are dropped.
    cache_ram:
        Keep decoded uint8 images in RAM after first touch, skipping PNG
        decode on every later epoch. ~2GB per 10k pairs at 256px; with
        num_workers > 0 each worker holds its own copy of what it touches.
    """

    def __init__(
        self,
        root: str | Path,
        n: int | None = None,
        size: int | None = 256,
        variant: int = 0,
        skip_flagged: bool = True,
        n_classes: int | None = None,
        cache_ram: bool = False,
    ):
        self.root = Path(root)
        self.size = size
        self.variant = variant
        self.n_classes = n_classes
        self._cache: dict | None = {} if cache_ram else None
        dirs = list_sample_dirs(self.root, skip_flagged=skip_flagged)
        n_before_label_filter = len(dirs)
        if n_classes is not None:
            dirs = [d for d in dirs if self._labels_ok(d, n_classes)]
        if n is not None:
            dirs = dirs[:n]
        if not dirs:
            # Diagnose loudly: remote crash logs are the only debugger a pod has.
            if not self.root.is_dir():
                raise ValueError(f"no usable sample dirs: root {self.root} does not exist")
            entries = sorted(p.name for p in self.root.iterdir())
            probe = next((p for p in sorted(self.root.iterdir()) if p.is_dir()), None)
            probe_files = sorted(f.name for f in probe.iterdir())[:8] if probe else []
            raise ValueError(
                f"no usable sample dirs under {self.root}: "
                f"{len(entries)} entries, {n_before_label_filter} sample dirs "
                f"before label filter (n_classes={n_classes}); "
                f"first entries {entries[:4]}; probe dir {probe and probe.name}: {probe_files}"
            )
        self.dirs = dirs

    @staticmethod
    def _labels_ok(d: Path, n_classes: int) -> bool:
        pal_path = d / "palette.json"
        if not (d / "labels.png").exists() or not pal_path.exists():
            return False
        try:
            return len(json.loads(pal_path.read_text())["palette"]) <= n_classes
        except (json.JSONDecodeError, KeyError):
            return False

    def _wrecked_path(self, d: Path) -> Path:
        """The chosen wrecked variant, clamped to what the dir actually has."""
        variants = sorted(d.glob("wrecked_*.png"))
        idx = min(self.variant, len(variants) - 1)
        return variants[idx]

    def __len__(self) -> int:
        return len(self.dirs)

    def _rgb(self, path: Path) -> torch.Tensor:
        if self._cache is None:
            return _load_rgb(path, self.size)
        arr = self._cache.get(path)
        if arr is None:
            img = Image.open(path).convert("RGB")
            if self.size is not None and img.size != (self.size, self.size):
                img = img.resize((self.size, self.size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.uint8)
            self._cache[path] = arr
        t = torch.from_numpy(arr.astype(np.float32) / 255.0)
        return t.permute(2, 0, 1).contiguous()

    def _labels(self, d: Path) -> torch.Tensor:
        if self._cache is None:
            return _load_labels(d, self.size, self.n_classes)
        key = d / "labels.png"
        lab = self._cache.get(key)
        if lab is None:
            lab = _load_labels(d, self.size, self.n_classes)
            self._cache[key] = lab
        return lab

    def __getitem__(self, i: int) -> tuple[torch.Tensor, ...]:
        d = self.dirs[i]
        wrecked = self._rgb(self._wrecked_path(d))
        clean = self._rgb(d / "clean.png")
        if self.n_classes is None:
            return wrecked, clean
        return wrecked, clean, self._labels(d)
