"""On-the-fly wreck-v2 dataset: damage is generated per item, per epoch.

Same sample-dir layout as PairsDataset but the pre-baked wrecked_XX.png files
are ignored: each __getitem__ draws a fresh v2 recipe from a stream seeded by
(base seed, epoch, index) and applies it to clean.png. Every epoch therefore
sees new damage on the same images - the E8 plan - and a 100k pre-baked mint
shrinks to clean renders + labels only.

Geometric families warp BOTH supervision targets by the recipe's homography
(labels nearest-neighbour with background fill, clean RGB bilinear with the
palette background colour), so neither head is ever asked to undo geometry
the other head is told to keep - the matched-warp invariant, applied to the
full pair rather than labels alone.

Epoch plumbing: the trainer calls set_epoch(e) before building each epoch's
iterator, and the DataLoader must NOT use persistent_workers (workers fork at
iterator creation and would otherwise keep the old epoch's copy forever).
freeze=True pins the stream to epoch 0 - use it for the val instance so
validation always sees the same damage.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from vecml.degrade.wreck_v2 import (
    apply_recipe_v2,
    is_identity,
    sample_recipe_v2,
    variant_rngs,
    warp_label_map,
)

from .pairs import PairsDataset, _load_labels


class WreckOnTheFly(PairsDataset):
    """PairsDataset variant that synthesizes the wrecked input at load time."""

    def __init__(
        self,
        root: str | Path,
        n: int | None = None,
        size: int | None = 256,
        skip_flagged: bool = True,
        n_classes: int | None = None,
        cache_ram: bool = False,
        seed: int = 0,
        freeze: bool = False,
        family_mix: dict | None = None,
    ):
        super().__init__(
            root, n=n, size=size, variant=0, skip_flagged=skip_flagged,
            n_classes=n_classes, cache_ram=cache_ram,
        )
        self.seed = seed
        self.freeze = freeze
        self.family_mix = family_mix
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = 0 if self.freeze else int(epoch)

    def _clean_u8(self, d: Path) -> np.ndarray:
        img = Image.open(d / "clean.png").convert("RGB")
        if self.size is not None and img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)

    def _bg_colour(self, d: Path) -> tuple:
        try:
            pal = json.loads((d / "palette.json").read_text())
            return tuple(int(c) for c in pal["palette"][pal.get("background_index", 0)])
        except Exception:
            return (255, 255, 255)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, ...]:
        d = self.dirs[i]
        clean = self._clean_u8(d)

        # One deterministic stream per (seed, epoch, item): reproducible runs,
        # fresh damage each epoch, and no cross-worker coordination needed.
        vs = int(np.random.SeedSequence(
            [self.seed, self.epoch, i]).generate_state(1)[0])
        sample_rng, apply_rng = variant_rngs(vs)
        recipe = sample_recipe_v2(
            sample_rng, size=clean.shape[0], mix=self.family_mix)
        wrecked, M = apply_recipe_v2(clean, recipe, apply_rng)

        labels = None
        if self.n_classes is not None:
            labels = _load_labels(d, self.size, self.n_classes)

        if not is_identity(M):
            h, w = clean.shape[:2]
            clean = cv2.warpPerspective(
                clean, np.asarray(M, dtype=np.float64), (w, h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                borderValue=self._bg_colour(d),
            )
            if labels is not None:
                warped = warp_label_map(
                    labels.numpy().astype(np.uint8), M)
                labels = torch.from_numpy(warped.astype(np.int64))

        wrecked_t = torch.from_numpy(
            wrecked.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        clean_t = torch.from_numpy(
            clean.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        if labels is None:
            return wrecked_t, clean_t
        return wrecked_t, clean_t, labels
