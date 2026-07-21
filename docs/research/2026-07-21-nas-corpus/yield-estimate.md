# Print-shop PDF corpus: training-tile yield and wreck-pipeline convertibility

Date: 2026-07-21. Read-only survey of a production print-shop artwork archive.
Aggregate figures only.

## Scope and method

- Corpus index: 16,048 print-ready ("-PRINT") files, of which 14,776 are PDF.
- Sample: 40 PDFs drawn at random (seed 43) and copied to local scratch for
  processing. Source archive was treated strictly read-only.
- Part A: each PDF rendered with PyMuPDF at 150 DPI (longest side capped at
  6000 px to bound large-format banners), up to the first 5 pages per document.
  Each page cut into non-overlapping 256 px squares; partial edge tiles dropped.
  79 pages rendered, 2,961 tiles classified.
- Part B: 15 of the most vector-dominant pages from the Part A sample converted
  to SVG via PyMuPDF `page.get_svg_image(text_as_path=True)`, then run through
  the gate2 structural checks and the wreck pipeline's `derive_labels_from_svg`.

Tile classifier (heuristic, per tile): near-white coverage, count of distinct
quantised colours (5 bits/channel), and coverage share of the top colours.
- near-blank: >95% near-white pixels (margins, whitespace)
- flat-colour ("trainable"): top 12 quantised colours cover >=85% of the tile —
  the vector-looking kind: solid regions, text, logos, line art
- photographic: top 12 colours cover <=50% — continuous tone
- other: in between (mixed / anti-aliased transition tiles)

## Corpus health finding

5 of 40 sampled files (12.5%) are 0-byte placeholders on the archive — the
source files themselves are empty, not a copy error. Extrapolated, roughly
1,850 of the 14,776 "-PRINT" PDFs are empty, leaving ~12,900 usable documents.
All yield figures below are stated against the usable set.

## Part A — tile yield

Distribution (79 rendered pages, non-blank edge-partials excluded):

| class          | tiles | share |
|----------------|-------|-------|
| flat (trainable) | 1,418 | 47.9% |
| other          | 773   | 26.1% |
| photographic   | 411   | 13.9% |
| near-blank     | 359   | 12.1% |

- Tiles per page: min 1, median 24, mean 37, max 294 (large-format work skews
  the mean; most jobs are small-format, 1–2 pages).
- Trainable share: 47.9% of all tiles, 54.5% of non-blank tiles.
- Trainable tiles per page: mean ~18, median ~11.

### Extrapolation to the full corpus

Assumptions:
- ~12,900 usable (non-empty) PDFs.
- Page-count profile from the sample: median 2 pages/doc; a long tail of
  multi-page documents exists (two sampled docs had 52 and 396 pages) but was
  truncated at 5 pages each, so deep pages of large documents are not counted.
- ~18 trainable 256 px tiles per rendered page at 150 DPI, non-overlapping.
- Raw tile count, before near-duplicate removal.

Two anchors:
- Per-page: ~12,900 docs x ~2 pages x ~18 trainable tiles ≈ **465k**.
- Per-document (observed rendered totals, ~40 trainable tiles/usable doc) scaled
  across the usable set ≈ **524k**.

Estimate: **on the order of 0.5 million raw trainable 256 px tiles**
(~450k–550k), with upside from the multi-page/large-format tail that the 5-page
render cap excluded. Two caveats pull the *unique-content* number down:
- Heavy near-duplication is expected (repeated house logos, common layouts,
  template stationery, gang-run repeats). Deduplication could remove a large
  fraction; unique trainable tiles are plausibly a few hundred thousand.
- 150 DPI is one detail level. Rendering at multiple resolutions multiplies raw
  tiles but not unique content.

## Part B — SVG convertibility funnel

Converting a rendered page to SVG and pushing it through the existing pipeline,
per vector-dominant page (n = 15):

| stage | pass | notes |
|-------|------|-------|
| PDF page -> valid SVG (`get_svg_image`) | 15/15 (100%) | conversion never failed |
| -> gate2 clean/warn (not reject) | 2/15 (13%) | see failure modes |
| -> wreck derivation runs (no `DerivationError`) | 14/15 (93%) | 1 fail: translucent-leaf cap |
| -> **meaningful ground truth** (label map explains the page) | 7/15 (47%) | see below |

### Failure modes

- **gate2 "huge" reject (12/15).** `get_svg_image` output routinely exceeds
  gate2's 200 KB `huge` cap (sampled SVGs ran 37 KB to 3.7 MB). The bulk is
  embedded base64 raster and PyMuPDF's verbose, repeated clip-path scaffolding,
  not art complexity. gate2 was tuned for compact stock icon/logo SVGs and is
  mis-calibrated for PyMuPDF page dumps; the `huge` short-circuit fires before
  any structural check even runs.
- **Embedded raster (11/15 pages carry base64 PNG/JPEG).** This is the core
  problem. `idmap` does not treat `<image>` as paintable, so it silently
  ignores embedded rasters and derives labels from the vector layer alone.
  Derivation therefore "succeeds" even on a page that is mostly a photo —
  yielding a degenerate 1–3 region label map that does not describe the visible
  page. Measured as the fraction of inked pixels present only in the raster
  layer, raster dominance across the 15 pages ranged from 0 to 0.995; 6 pages
  exceeded 0.19 (up to two pages that are ~99% photo). Those pass derivation but
  would fail the pipeline's own coverage QC.
- **Translucent-leaf cap (1/15).** One clean all-vector page raised
  `DerivationError` — 112 translucent leaves against the bitmask cap of 32.
- **Degenerate single-region output.** One more clean-vector page derived a
  single foreground region (low training value even though it "passes").

Genuinely wreck-consumable — derivation succeeds, no dominant raster
(raster-only ink <=0.1), and >=2 meaningful regions — was **7/15 (47%) of the
vector-dominant subset**. Note this subset was already the most vector-looking
pages in the sample, so 47% is an upper bound on the vector-dominant slice, not
a corpus-wide rate. Across the whole corpus the SVG route's meaningful-yield is
substantially lower, because most pages carry photographic or raster content
that the current `idmap` cannot represent.

## Verdict: which mint route is viable

**Render-then-tile as raster pairs (recommended).**
Render each page with PyMuPDF, tile into 256 px squares, keep the flat-colour
tiles, and apply synthetic damage to the rendered tile to form (clean, wrecked)
pairs. This bypasses SVG conversion and its two failure cliffs (gate2 `huge` and
raster-blind `idmap`) entirely. It taps the full ~0.5 M trainable-tile pool and
is robust to the raster-heavy reality of print artwork.
- Caveat: raster tiling gives no paint-derived region ground truth. It supports
  self-supervised / reconstruction-style or degrade-and-restore objectives on
  rendered pixels, not the SVG-paint region-labelling the wreck pipeline emits.
  If per-region labels are required, they would have to come from a pixel-based
  labeller run on the clean render rather than from SVG geometry.
- Rough engineering cost: **low, ~2–4 days.** Render + tile + classify already
  works (this survey). Remaining work: reuse the existing `wreck` damage recipes
  on rendered tiles, a blank/photo tile filter, and near-duplicate dedup.

**Full SVG conversion through the existing wreck pipeline (not viable as-is).**
The pipeline consumes paint-derivable SVG regions; PyMuPDF page SVGs are
clip-path-and-embedded-image soup for the majority of print pages, and `idmap`
ignores the raster that carries most of their content. Only a minority of
already-vector-dominant pages survive, and gate2 rejects nearly all converted
SVGs on file size before the structural gate runs.
- To make it work would need: raster-aware handling in `idmap` (segment or
  exclude `<image>` regions instead of silently dropping them), a gate2
  recalibration for PyMuPDF output (size cap, clip-path tolerance), lifting or
  reworking the 32-translucent-leaf cap, and a pre-filter that routes only
  low-raster vector pages into the SVG path.
- Rough engineering cost: **high, ~3–6 weeks**, for a pipeline that would still
  only harvest the vector-dominant minority of the corpus.

### Recommendation

Mint raster training pairs by render-then-tile with synthetic damage on rendered
pages. It reaches the whole trainable-tile pool at low cost. Reserve full SVG
conversion for a later, targeted pass over the low-raster vector-dominant subset,
and only after `idmap` is made raster-aware and gate2 is recalibrated.
