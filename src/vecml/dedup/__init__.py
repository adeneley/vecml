"""Corpus deduplication gate for the vecml mint pipeline.

Public surface:

  hashes  exact_hash / normalize_svg / normalized_hash / render_phash / hamming
  bktree  BKTree, UnionFind
  gate    Record, compute_hashes, cluster_phashes, run_gate, GateReport
"""

from vecml.dedup.gate import (
    GateReport,
    Record,
    cluster_phashes,
    compute_hashes,
    run_gate,
)
from vecml.dedup.hashes import (
    exact_hash,
    hamming,
    normalize_svg,
    normalized_hash,
    render_phash,
)

__all__ = [
    "BKTree",
    "GateReport",
    "Record",
    "UnionFind",
    "cluster_phashes",
    "compute_hashes",
    "exact_hash",
    "hamming",
    "normalize_svg",
    "normalized_hash",
    "render_phash",
    "run_gate",
]

from vecml.dedup.bktree import BKTree, UnionFind  # noqa: E402
