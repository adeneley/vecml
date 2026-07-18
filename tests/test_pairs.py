"""Dataset tests against a few real sample dirs from data/audit-500-v2."""

from pathlib import Path

import pytest
import torch

from vecml.data.pairs import PairsDataset, list_sample_dirs

ROOT = Path("data/audit-500-v2")

pytestmark = pytest.mark.skipif(
    not ROOT.exists(), reason="audit-500-v2 sample data not present"
)


def test_lists_and_skips_flagged():
    all_dirs = list_sample_dirs(ROOT, skip_flagged=False)
    kept = list_sample_dirs(ROOT, skip_flagged=True)
    assert len(all_dirs) > len(kept) or len(all_dirs) == len(kept)
    # flagged names must not appear in the kept list
    kept_names = {p.name for p in kept}
    import json

    flagged = set(
        json.loads((ROOT / "_audit_summary.json").read_text()).get("flagged_names", [])
    )
    assert kept_names.isdisjoint(flagged)


def test_item_shapes_and_range():
    ds = PairsDataset(ROOT, n=3, size=128)
    assert len(ds) == 3
    wrecked, clean = ds[0]
    assert wrecked.shape == (3, 128, 128)
    assert clean.shape == (3, 128, 128)
    assert wrecked.dtype == torch.float32
    assert 0.0 <= float(wrecked.min()) and float(wrecked.max()) <= 1.0
    assert 0.0 <= float(clean.min()) and float(clean.max()) <= 1.0


def test_deterministic_ordering():
    a = PairsDataset(ROOT, n=3, size=64).dirs
    b = PairsDataset(ROOT, n=3, size=64).dirs
    assert a == b
