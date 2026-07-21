"""Tests for the corpus dedup gate. No network, no dependence on assets/.

Covers the load-bearing behaviours: exact dupe caught, normalised dupe caught,
near-dupe render caught at threshold, a genuinely distinct pair NOT merged, and
the bench-eviction rule firing.
"""

from pathlib import Path

import pytest

from vecml.dedup.gate import Record, compute_hashes, run_gate
from vecml.dedup.hashes import (
    exact_hash,
    hamming,
    normalized_hash,
    render_phash,
)

# A small flat-colour drawing (the "canonical" form).
BASE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <rect x="0" y="0" width="200" height="200" fill="#ffffff"/>
  <rect x="20" y="20" width="120" height="120" fill="#d81e1e"/>
  <circle cx="130" cy="90" r="60" fill="#1e6fd8"/>
</svg>
"""

# Same drawing after a re-export: added ids, an inkscape attr, a comment, a
# <metadata> block, reformatted floats, reordered attributes, extra whitespace.
REEXPORT_SVG = """<svg xmlns="http://www.w3.org/2000/svg"
  xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  viewBox="0 0 200 200">
  <!-- exported by some editor -->
  <metadata id="meta7">junk<rdf/></metadata>
  <rect fill="#ffffff" y="0.0" x="0.00001" width="200" height="200" id="bg"/>
  <rect id="a" inkscape:label="box" x="20" y="20.0" width="120.0" height="120" fill="#D81E1E"/>
  <circle cx="130" cy="90" r="60.00002" fill="#1e6fd8" class="shape"/>
</svg>
"""

# A genuinely different drawing (different shapes, different colours).
DISTINCT_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <rect x="0" y="0" width="200" height="200" fill="#ffffff"/>
  <polygon points="100,10 40,190 190,70 10,70 160,190" fill="#101010"/>
</svg>
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def test_exact_dupe_caught():
    a = BASE_SVG.encode()
    b = BASE_SVG.encode()
    c = DISTINCT_SVG.encode()
    assert exact_hash(a) == exact_hash(b)
    assert exact_hash(a) != exact_hash(c)


def test_normalized_dupe_caught():
    # Byte-different (comments, ids, float noise, attr order) but same drawing.
    assert exact_hash(BASE_SVG.encode()) != exact_hash(REEXPORT_SVG.encode())
    assert normalized_hash(BASE_SVG) == normalized_hash(REEXPORT_SVG)
    # And a real difference still hashes differently.
    assert normalized_hash(BASE_SVG) != normalized_hash(DISTINCT_SVG)


def test_near_dupe_render_caught_at_threshold(tmp_path):
    # Same art rendered from a slightly rescaled/recoloured re-export should sit
    # within a small Hamming radius; the distinct drawing should sit far away.
    base = _write(tmp_path, "base.svg", BASE_SVG)
    reexport = _write(tmp_path, "re.svg", REEXPORT_SVG)
    distinct = _write(tmp_path, "d.svg", DISTINCT_SVG)
    hb = render_phash(base)
    hr = render_phash(reexport)
    hd = render_phash(distinct)
    near = hamming(hb, hr)
    far = hamming(hb, hd)
    assert near <= 6, f"near-dupe distance {near} unexpectedly large"
    assert far > near, f"distinct pair ({far}) not separated from near ({near})"


def test_distinct_pair_not_merged(tmp_path):
    base = _write(tmp_path, "base.svg", BASE_SVG)
    distinct = _write(tmp_path, "d.svg", DISTINCT_SVG)
    recs = [
        Record(key="base", path=base, split="corpus"),
        Record(key="distinct", path=distinct, split="corpus"),
    ]
    compute_hashes(recs)
    rep, survivors = run_gate(recs, threshold=6)
    # Two genuinely different drawings: nothing removed, both survive.
    assert rep.survivors == 2
    assert rep.phash_removed == 0


def test_bench_eviction_rule_fires(tmp_path):
    # A train image that is a near-dupe of a bench image must be evicted, and
    # counted separately as leakage.
    bench = _write(tmp_path, "bench.svg", BASE_SVG)
    leaked = _write(tmp_path, "leaked.svg", REEXPORT_SVG)  # near-dupe of bench
    clean = _write(tmp_path, "clean.svg", DISTINCT_SVG)
    recs = [
        Record(key="bench", path=bench, split="bench"),
        Record(key="leaked", path=leaked, split="train"),
        Record(key="clean", path=clean, split="train"),
    ]
    compute_hashes(recs)
    rep, survivors = run_gate(recs, threshold=6)
    keys = {r.key for r in survivors}
    assert rep.bench_evicted == 1
    assert "leaked" in rep.bench_evicted_keys
    assert "leaked" not in keys
    assert "clean" in keys  # the genuinely distinct train image is kept


def test_full_gate_layer_ordering(tmp_path):
    # Exact dupe + normalised dupe + distinct: exact caught first, then norm.
    a = _write(tmp_path, "a.svg", BASE_SVG)
    a2 = _write(tmp_path, "a2.svg", BASE_SVG)  # exact dupe of a
    re = _write(tmp_path, "re.svg", REEXPORT_SVG)  # normalised dupe of a
    d = _write(tmp_path, "d.svg", DISTINCT_SVG)
    recs = [
        Record(key="a", path=a, split="corpus"),
        Record(key="a2", path=a2, split="corpus"),
        Record(key="re", path=re, split="corpus"),
        Record(key="d", path=d, split="corpus"),
    ]
    compute_hashes(recs)
    rep, survivors = run_gate(recs, threshold=6)
    assert rep.exact_removed == 1  # a2 is a byte-for-byte copy of a
    assert rep.norm_removed == 1  # re canonicalises to a
    assert rep.survivors == 2  # one representative + the distinct drawing


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
