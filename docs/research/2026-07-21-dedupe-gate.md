# Corpus deduplication gate

Date: 2026-07-21

Before minting a 2 to 3M image training set from roughly eleven overlapping
sources (SVGRepo re-aggregates Openclipart, Wikimedia re-hosts Openclipart plus
emoji and flag collections, and so on), the corpus needs a deduplication pass.
Duplicates cost twice. They shrink the effective unique count that the data
scaling curve depends on, and a duplicate that straddles the train/val split
quietly closes the train/val gap we read as the signal for whether more data or
more model is worth paying for. This document describes the gate that removes
them, the evidence behind its one tuning knob, and what it measures on the
corpus we already hold.

## Where it sits

The gate runs at the very front of the mint pipeline, ahead of the autotrace
detector and the structural quality gate (`gate2`):

```
raw SVGs -> [dedup gate] -> [autotrace detector] -> [gate2 quality tiers] -> mint
```

Placing it first means the two downstream passes, both of which render and
parse every survivor, never spend that compute on a duplicate, and no leaked
train/val pair can slip past them into a shard.

## Layer design

Three layers run strictest and cheapest first. Each sees only the survivors of
the layer above, so the expensive perceptual pass runs on the smallest set.

1. **Exact bytes.** SHA-256 of the raw file. Catches re-uploads and mirror
   copies. Effectively free.

2. **Normalized-SVG hash.** Parse the XML, drop ids, classes, inline styles,
   comments, `<metadata>`/`<title>`/`<desc>`, and every editor-namespace
   attribute (Inkscape, Sodipodi, Illustrator, Sketch, Figma, RDF). Round all
   numbers (path `d` strings, `points`, `transform`, plain coordinates) to one
   decimal place, lowercase hex colours, sort attributes, and hash the
   canonical serialization. Catches trivial re-exports where the drawing is
   identical but the bytes are not. A regex fallback handles SVGs that are not
   well-formed XML so the layer never crashes on a corpus file.

3. **Render pHash.** Render each SVG at 64px through the repository's single
   rendering backend (`vecml.degrade.renderer.render_svg`, resvg), convert to
   grayscale, and take a 64-bit DCT perceptual hash (top-left 8x8
   low-frequency block, thresholded against its own median, DC term excluded).
   No second renderer and no external hashing dependency are introduced. Near
   duplicates are clustered by Hamming distance with a BK-tree, which indexes
   the hashes in metric space so a radius query visits only the branches whose
   stored distance overlaps the search band. This is what lets clustering scale
   past a million files where an all-pairs comparison cannot.

Two robustness measures matter at corpus scale:

- **Identical-hash pre-grouping.** Thousands of simple icons (a plain circle, a
  single glyph) render to the exact same pHash. Inserting N identical keys into
  a BK-tree degenerates into an O(N^2) distance-zero chain. Records are grouped
  by exact pHash first, and only distinct hashes enter the tree.

- **Degenerate-render guard.** Blank (all-white), solid (all-ink) and near-flat
  renders produce a constant image whose DCT is all-DC; the low-frequency block
  is then floating-point noise thresholded against a near-zero median, so the
  hash is unstable and collides arbitrarily. Such renders (3.4% of the sample,
  mostly single-colour glyphs and near-empty icons) are excluded from
  perceptual clustering and left to the exact and normalized layers.

## Split-hygiene eviction

On top of deduplication the gate enforces a non-negotiable rule: any training
image that matches a held-out val/test image or one of the 24 relay-bench
images, by exact hash, normalized hash, or pHash within the Hamming threshold,
is evicted from training. Eviction covers all three identity notions, not only
pHash, so a re-exported copy of a bench image is caught as surely as a rescaled
one. The bench-eviction count is reported separately because it is a direct
measurement of leakage against the sacred 24-image relay set.

## Calibrating the Hamming threshold

The threshold was fixed from two experiments, both on the real corpus.

**Pairwise separation.** For several hundred corpus images, known near
duplicates were synthesised by nudging every fill colour by a small per-channel
delta and rendering through the same 64px path the gate uses, and known
distinct pairs were drawn at random. The two distance distributions barely
touch:

| Distribution        | p50 | p90 | p95 | p99 | min | max |
|---------------------|-----|-----|-----|-----|-----|-----|
| Recolour near-dupe  | 0   | 2   | 2   | 8 to 12 | 0 | 40 |
| Distinct pair       | 30  |     |     |     | 14 to 16 | 44 |

Recolour near duplicates sit at or below 2 bits for 95% of pairs; distinct
pairs never fall below 14 over thousands of samples. Any threshold from about 4
to 13 separates them pairwise. (A harsher near-dupe class, re-rendering at a
different raster size then resizing, has a heavier tail. It is reported as a
robustness bound but does not drive the threshold, because the gate always
re-renders from source SVG at a fixed 64px, so two rescaled copies produce the
same raster and the scenario does not arise in production.)

**Single-linkage stability.** Pairwise separation is necessary but not
sufficient. Single-linkage clustering chains: A near B and B near C pulls A, B
and C into one cluster even when A and C are far apart. On a large corpus of
simple clip art the space of low-detail pHashes is dense enough that a
too-generous threshold collapses a third of the corpus into one blob. Sweeping
the threshold over a fixed 30k render set makes the failure obvious:

| Threshold | Largest cluster | Removed |
|-----------|-----------------|---------|
| 4  | 87     | 6.6%  |
| 6  | 400    | 11.7% |
| 8  | 3,240  | 22.2% |
| 10 | 10,045 | 42.3% |

At 10 the largest cluster is a third of the sample, which is over-merging, not
deduplication. **The operating threshold is 4:** it holds the largest cluster
to well under 1% of the corpus while still catching 95% of recolour near
duplicates (their p95 is 2), and sits ten or more bits below the distinct
floor.

Calibration artifact: `runs/dedupe/calibration.json`.

## Measured results

Full three-layer gate over a uniform random 150,000-SVG sample of the labelled
clean tier (`datasets/svg-stack-labelled/clean`, seed 13), with the 24
relay-bench SVGs added as references and train/val/test splits read from the
labelled manifest. Pool of training candidates: 142,594; held-out references in
the sample: 7,014 val, 392 test.

Full report: `runs/dedupe/report-150k.json`.

### Duplicate rate per layer

| Layer            | Removed | Rate of pool |
|------------------|---------|--------------|
| Exact bytes      | 2       | 0.001%       |
| Normalized SVG   | 77      | 0.054%       |
| Render pHash     | 15,759  | 11.05%       |
| Combined dedup   | 15,838  | 11.11%       |

The exact and normalized layers remove almost nothing here, and that is the
expected result, not a failure: svg-stack is already content-addressed by an
upstream hash, so byte-identical and re-export duplicates were collapsed before
we ever saw it. Their value is on the forthcoming multi-source mint, where the
same glyph arrives from Openclipart, SVGRepo and Wikimedia under three
different filenames; they are kept in front of the pHash layer because they are
close to free and will do real work there. The perceptual layer is where the
overlap actually lives on this corpus: **11% of the training pool is a visual
near-duplicate of another image in it.**

Render failures: 730 (0.49%). Degenerate renders held out of perceptual
clustering: 5,033 (3.4%).

### Cluster-size distribution (pHash)

108,874 singletons. Duplicate families fall away fast: 5,543 pairs, 1,181
triples, 423 of size four, then a thin tail. Largest clusters: 265, 213, 157,
141, 103. These large families are genuine (icon-set colour variants, flag and
emoji series), not chaining artifacts; the largest is 0.19% of the pool.

### Leakage

Two distinct leakage channels surfaced.

**Near-duplicate leakage (what the gate evicts).** Training-pool images that
are distinct records but perceptual near duplicates of a held-out image:

- Held-out (val/test) evictions: **4,576** (3.22% of the pool).
- Relay-bench evictions: **13** training images within Hamming 4 of one of the
  24 sacred bench images.

Descriptively, 1,071 perceptual clusters span more than one split: 1,003
train/val, 75 test/train, 14 test/val, 6 bench/train, 1 bench/val. Every one of
these is a train/val gap corrupted by a near-duplicate, and the gate removes
them from training.

**Split-membership leakage (a sampler bug the audit surfaced).** Separately
from near-duplicates, the existing 1.165M staged ledger mixes held-out splits
into training. The clean tier carries a train/val/test label per sha (train
1,391,502, val 69,565, test 3,656; 5.0% held out), but `scripts/sample_src.py`
draws from the whole `clean/` directory without consulting that label, so
held-out images are pulled straight into training src dirs. The already-minted
local sets confirm it: `train-50k-src` is 2,340 val + 117 test (4.9%),
`train-10k-src` is 487 val + 30 test (5.2%), and even `relay-test-src` contains
3 val-split shas. Projecting the 5.0% held-out fraction onto the 1,165,024-sha
ledger implies roughly **58,000 val/test-split images already used as
training**. The near-duplicate gate does not fix this channel on its own; the
fix is to make the sampler split-aware (sample only from `clean/train`), and to
run this gate's eviction pass as the backstop for the near-duplicate remainder.

### Throughput and projection

Single-threaded on an M-series Mac:

| Stage                    | Time (150k) | Rate |
|--------------------------|-------------|------|
| Hash (render + 3 layers) | 129 s       | ~1,160 files/s render-bound |
| Gate (cluster + evict)   | 192 s       |      |
| End to end               | 321 s       | 445 files/s |

A 5M-file mint projects to roughly **3.1 hours single-threaded**. Two caveats.
The hashing stage is embarrassingly parallel and render-bound, so it collapses
by the core count of the CPU mint pod. The clustering stage is not linear: the
BK-tree radius query grows with corpus size, so the gate stage is the piece to
watch and the reason to shard a 5M run (cluster within shards, then a second
pass over shard representatives). The projection above is therefore optimistic
on the gate stage and pessimistic on hashing; both point the same way, which is
that this pass is cheap relative to a training run and belongs in the mint.

## Tests

`tests/test_dedup.py` covers the load-bearing behaviours: an exact duplicate is
caught, a re-export with added ids, editor attributes, comments, reordered
attributes and reformatted floats is caught by the normalized layer, a rendered
near-duplicate sits within the threshold while a genuinely distinct pair does
not merge, and the bench-eviction rule fires and reports the leaked key. All six
pass.

## Limitations and next steps

- Single-linkage clustering is inherently prone to chaining; the conservative
  threshold controls it at the cost of missing near-duplicates beyond 4 bits
  (an estimated low-single-digit percent of recolour cases). If recall matters
  more than diversity on a future source, complete-linkage within candidate
  buckets would tighten cluster diameter, at higher cost.
- The exact and normalized layers are unexercised by svg-stack and should be
  re-measured on the first genuinely multi-source mint, where cross-source
  byte and re-export overlap is the whole point.
- The split-membership leakage is a `sample_src.py` fix, tracked separately
  from this gate.

## Files

- `src/vecml/dedup/hashes.py` : the three hash primitives, DCT pHash, guard.
- `src/vecml/dedup/bktree.py` : BK-tree and union-find.
- `src/vecml/dedup/gate.py` : layer orchestration, eviction, reporting.
- `scripts/dedup_gate.py` : CLI (`calibrate` and `run`).
- `tests/test_dedup.py` : unit tests.
- `runs/dedupe/calibration.json`, `runs/dedupe/report-150k.json` : measurements.
