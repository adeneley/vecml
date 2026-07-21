"""The corpus deduplication gate.

Three layers, strictest and cheapest first, each seeing only the survivors of
the previous one:

  1. exact bytes      SHA-256 of the raw file
  2. normalised SVG   hash of the canonicalised drawing
  3. render pHash     64-bit perceptual hash, clustered by Hamming distance

On top of dedup it enforces the split-hygiene rule: any training image that is
an exact, normalised, or perceptual (within the Hamming threshold) match of a
held-out val/test image or a relay-bench image is evicted from training,
because a near-duplicate that straddles the split makes the train/val gap lie.
Eviction covers all three identity notions, not just pHash, so a re-exported
copy of a bench image is caught as surely as a rescaled one.

This gate runs at the very front of the mint pipeline, before the autotrace
detector and the structural quality gate (gate2). Feeding either of those a
corpus that is several percent duplicates wastes their compute and, worse,
lets leaked pairs through into training.
"""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from vecml.dedup.bktree import BKTree, UnionFind
from vecml.dedup.hashes import (
    exact_hash,
    is_degenerate_render,
    normalized_hash,
    phash_from_gray,
    render_gray,
)

# Splits that are held out; a train image near any of these is leakage.
HELD_OUT = ("val", "test")
BENCH = "bench"
POOL_SPLITS = ("train", "corpus")


@dataclass
class Record:
    """One corpus item as it flows through the gate."""

    key: str
    path: Path
    split: str = "corpus"
    exact: str | None = None
    norm: str | None = None
    phash: int | None = None
    render_ok: bool = True
    degenerate: bool = False


def compute_hashes(
    records: list[Record],
    precision: int = 1,
    size: int = 64,
    hash_side: int = 8,
    do_render: bool = True,
    progress: int = 0,
) -> None:
    """Populate exact/norm/phash on each record in place.

    A render failure sets ``render_ok=False`` and leaves ``phash`` None; those
    records still pass through the exact and normalised layers but cannot be
    clustered or matched perceptually.
    """
    for i, r in enumerate(records, 1):
        data = r.path.read_bytes()
        r.exact = exact_hash(data)
        try:
            r.norm = normalized_hash(data.decode("utf-8", "replace"), precision)
        except Exception:
            r.norm = r.exact  # unparseable: fall back to bytes identity
        if do_render:
            try:
                gray = render_gray(r.path, size)
                if is_degenerate_render(gray):
                    r.degenerate = True  # too flat to hash; leave phash None
                else:
                    r.phash = phash_from_gray(gray, hash_side)
            except Exception:
                r.render_ok = False
                r.phash = None
        if progress and i % progress == 0:
            print(f"  hashed {i}/{len(records)}", flush=True)


def _dedup_by_key(records: list[Record], attr: str) -> tuple[list[Record], int, Counter]:
    """Keep the first record per distinct value of ``attr``.

    Returns (survivors, removed_count, cluster_size_histogram).
    """
    seen: dict[str, int] = {}
    survivors = []
    for r in records:
        v = getattr(r, attr)
        if v in seen:
            seen[v] += 1
        else:
            seen[v] = 1
            survivors.append(r)
    removed = len(records) - len(survivors)
    hist = Counter(seen.values())
    return survivors, removed, hist


def cluster_phashes(
    records: list[Record], threshold: int
) -> tuple[list[list[Record]], UnionFind]:
    """Single-linkage cluster records by pHash Hamming distance <= threshold.

    Records are first grouped by their *exact* pHash value, and only the
    distinct hashes are inserted into the BK-tree. This is essential at corpus
    scale: simple icons (a plain circle, a single glyph) render to identical
    hashes in the thousands, and inserting N identical keys into a BK-tree
    degenerates into an O(N^2) distance-0 chain. Collapsing them first keeps
    the tree small and each query cheap, while single-linkage over the distinct
    hashes still merges every near-duplicate.
    """
    by_hash: dict[int, list[int]] = {}
    order: list[int] = []
    for i, r in enumerate(records):
        if r.phash is None:
            continue
        bucket = by_hash.get(r.phash)
        if bucket is None:
            by_hash[r.phash] = [i]
            order.append(r.phash)
        else:
            bucket.append(i)

    uf = UnionFind(len(order))
    tree = BKTree()
    for k, h in enumerate(order):
        for kj, _dist in tree.query(h, threshold):
            uf.union(k, kj)
        tree.add(h, k)  # payload is the position in `order`

    groups: dict[int, list[int]] = {}
    for k in range(len(order)):
        groups.setdefault(uf.find(k), []).append(k)
    clusters = [
        [records[i] for k in members for i in by_hash[order[k]]]
        for members in groups.values()
    ]
    return clusters, uf


def _identity_index(refs: list[Record]):
    """Build an (exact-set, norm-set, phash-BK-tree) index over reference records."""
    exact: set[str] = set()
    norm: set[str] = set()
    tree = BKTree()
    for r in refs:
        if r.exact is not None:
            exact.add(r.exact)
        if r.norm is not None:
            norm.add(r.norm)
        if r.phash is not None:
            tree.add(r.phash, r.key)
    return exact, norm, tree


def _matches(r: Record, index, threshold: int) -> bool:
    exact, norm, tree = index
    if r.exact in exact or r.norm in norm:
        return True
    return r.phash is not None and bool(tree.query(r.phash, threshold))


@dataclass
class GateReport:
    total: int = 0
    pool_total: int = 0
    exact_removed: int = 0
    norm_removed: int = 0
    phash_removed: int = 0
    render_failures: int = 0
    degenerate_renders: int = 0
    survivors: int = 0
    exact_hist: Counter = field(default_factory=Counter)
    norm_hist: Counter = field(default_factory=Counter)
    phash_cluster_hist: Counter = field(default_factory=Counter)
    cross_split_clusters: int = 0
    cross_split_pairs: Counter = field(default_factory=Counter)
    bench_evicted: int = 0
    val_evicted: int = 0
    bench_evicted_keys: list = field(default_factory=list)

    def as_dict(self) -> dict:
        def hist(c: Counter) -> dict:
            return {str(k): v for k, v in sorted(c.items())}

        denom = max(1, self.pool_total)
        dedup_removed = self.exact_removed + self.norm_removed + self.phash_removed
        evicted = self.bench_evicted + self.val_evicted
        return {
            "total_records": self.total,
            "pool_total": self.pool_total,
            "render_failures": self.render_failures,
            "degenerate_renders": self.degenerate_renders,
            "survivors": self.survivors,
            "removed": {
                "exact": self.exact_removed,
                "normalized": self.norm_removed,
                "phash": self.phash_removed,
                "dedup_total": dedup_removed,
                "leakage_evicted": evicted,
            },
            "dupe_rate": {
                "exact": round(self.exact_removed / denom, 5),
                "normalized": round(self.norm_removed / denom, 5),
                "phash": round(self.phash_removed / denom, 5),
                "dedup_combined": round(dedup_removed / denom, 5),
            },
            "leakage": {
                "bench_evicted": self.bench_evicted,
                "val_evicted": self.val_evicted,
                "leakage_rate": round(evicted / denom, 5),
            },
            "exact_cluster_hist": hist(self.exact_hist),
            "norm_cluster_hist": hist(self.norm_hist),
            "phash_cluster_hist": hist(self.phash_cluster_hist),
            "cross_split_clusters": self.cross_split_clusters,
            "cross_split_pairs": {
                "-".join(k): v for k, v in sorted(self.cross_split_pairs.items())
            },
        }


def run_gate(records: list[Record], threshold: int) -> tuple[GateReport, list[Record]]:
    """Run split-hygiene eviction then the three dedup layers.

    Records must already have hashes populated (see ``compute_hashes``).
    Returns the report and the surviving train/corpus records a mint keeps.
    Records with split ``bench``/``val``/``test`` act as references only and are
    never part of the kept corpus.
    """
    rep = GateReport(total=len(records))
    rep.render_failures = sum(1 for r in records if not r.render_ok)
    rep.degenerate_renders = sum(1 for r in records if r.degenerate)

    pool = [r for r in records if r.split in POOL_SPLITS]
    bench_refs = [r for r in records if r.split == BENCH]
    held_refs = [r for r in records if r.split in HELD_OUT]
    rep.pool_total = len(pool)

    # Split-hygiene eviction (exact OR normalised OR perceptual), before dedup
    # so a leaked record is attributed as leakage and never chosen as the
    # surviving representative over the held-out image it copies.
    bench_idx = _identity_index(bench_refs)
    held_idx = _identity_index(held_refs)
    kept_pool: list[Record] = []
    for r in pool:
        if bench_refs and _matches(r, bench_idx, threshold):
            rep.bench_evicted += 1
            rep.bench_evicted_keys.append(r.key)
            continue
        if held_refs and _matches(r, held_idx, threshold):
            rep.val_evicted += 1
            continue
        kept_pool.append(r)

    # Layer 1 + 2: exact then normalised, each on the previous survivors.
    surv, rep.exact_removed, rep.exact_hist = _dedup_by_key(kept_pool, "exact")
    surv, rep.norm_removed, rep.norm_hist = _dedup_by_key(surv, "norm")

    # Layer 3: perceptual clustering, one representative kept per cluster.
    clusters, _uf = cluster_phashes(surv, threshold)
    final: list[Record] = []
    for members in clusters:
        rep.phash_cluster_hist[len(members)] += 1
        final.append(members[0])
    final.extend(r for r in surv if r.phash is None)
    rep.phash_removed = len(surv) - len(final)
    rep.survivors = len(final)

    # Cross-split overlap measurement: cluster ALL records (pool + references)
    # and count clusters that span more than one split. This is the descriptive
    # leakage number; the eviction counts above are the action taken.
    if bench_refs or held_refs:
        all_clusters, _ = cluster_phashes(records, threshold)
        for members in all_clusters:
            splits = sorted({m.split for m in members})
            if len(splits) > 1:
                rep.cross_split_clusters += 1
                for a in range(len(splits)):
                    for b in range(a + 1, len(splits)):
                        rep.cross_split_pairs[(splits[a], splits[b])] += 1

    return rep, final
