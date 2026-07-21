# Source feasibility: starvector/FIGR-SVG

Single-source study, 2026-07-21. Scope: whether `starvector/FIGR-SVG` is worth
minting into the Stage-1 cleanup UNet corpus (currently 1,165,024 uniques drawn
from `starvector/svg-stack`). Method: HF dataset-card + datasets-server API
verification, a 700-row gate2 mini-audit across the full row range, and a
structural trace-signature measurement.

---

## VERDICT: NOT FEASIBLE / SKIP

Estimated net-new usable after honest filtering: **~0**. (Nominal identifier-
disjoint yield is the full ~1.33M, but the corpus fails on provenance and adds
no gap coverage — see below.)

Three independent reasons, any one of which is disqualifying:

1. **Raster-traced ground truth, not born-vector (decisive).** FIGR-SVG is
   derived from FIGR-8, which is definitively a *raster* corpus — 1,548,256
   images at **192×192 grayscale (0–255)** captured from The Nounji App
   (marcdemers/FIGR-8). The FIGR-SVG `Svg` column is uniform 200×200, **100%
   single `<path>`, no fill attribute (default black silhouette)**, 43% curve
   commands, median 40 / p90 109 path commands. That is the exact fingerprint
   of contour-tracing a low-res grayscale silhouette, not a born-vector icon.
   Our entire premise is "wreck a *clean* SVG, learn to restore it." Training on
   traces of 192px rasters teaches the model to reproduce tracing artifacts
   (stair-stepped contours, over-smoothed corners, quantization) as if they were
   ground truth — the DuetSVG "auto-vectorised removal" concern, self-inflicted.
   (The StarVector abstract does not document FIGR-SVG's construction; the
   raster origin is inferred with high confidence from the FIGR-8 source being
   raster-only plus the single-path silhouette structure. Confirm with a visual
   spot-check before trusting any contrary claim — but the burden is on proving
   it born-vector, which is implausible.)

2. **Zero gap coverage (icons are the saturated class).** Sampled content is
   monochrome single-path silhouettes: 0/700 use stroke, 0/700 use gradients,
   0 fill colors (all default-black), uniform 200×200. Labels are Noun-Project
   keyword strings (`mustache/hair/style`, `hospital/healthcare/medical/...`).
   This is *more* of our #1 oversupplied distribution (filled monochrome icons)
   and attacks none of the three measured gaps — typography, fineline strokes,
   boundary-label accuracy. Prior census already ranked it Tier D, row 50
   (2026-07-20-deep-research/fuel.md): "Misleading name — it's icons. Overlaps
   existing bias; skip unless bulking icon volume." The model is data-
   constrained, so volume nominally helps — but not *this* volume, because it is
   both redundant and provenance-poisoned.

3. **License untenable for a commercial vectorizer.** The HF card carries **no
   license field** (verified via datasets-server `/info`: license null; README
   frontmatter empty). The upstream FIGR-8 images are Noun Project /
   Creative-Commons with an explicit term that "reproduction on any material
   intended to be sold or to be made profit from is strictly prohibited" without
   author consent. Minting these into a commercial model's training set is the
   same commercial-restriction risk already flagged for FIGR-8-SVG (fuel.md row
   51).

---

## Verified facts (datasets-server API + card)

| Field | Value | Source |
|---|---|---|
| Rows | 1,331,707 (train 1.30M / test 15k / valid 15k) | `/size`, card |
| Parquet size | 858,317,059 bytes (~858 MB) | `/size` |
| Format | parquet, single `default` config | `/info` |
| Columns | `Id`, `Label`, `Caption`, `len_pix`, `Svg` | `/rows` features |
| License | **none stated** (info.license = null, empty README frontmatter) | `/info`, README |
| Paper | arXiv:2312.11556 (StarVector) | card |
| Upstream | FIGR-8: 17,375 classes / 1,548,256 imgs / 192×192 grayscale / Nounji (Noun Project) | marcdemers/FIGR-8 |

Note the schema differs from svg-stack: FIGR-SVG keys on `Id` (5–11 char), while
svg-stack (and our whole pipeline) keys on `Filename`. This matters for dedup.

## gate2 mini-audit (700 rows, offsets 0 / 260k / 520k / 780k / 1.04M / 1.30M)

- **Tier split: 700 clean / 0 warn / 0 reject (100% clean).** Zero flags of any
  kind. Expected gate2 survival on the full 1.33M is effectively 100%.
- This is a **false pass**, not a good sign. gate2's only trace heuristic
  (`trace-suspect`) requires >800 path commands *and* >70% tiny chords; these
  silhouettes are short (0/700 exceed 800 cmds, p90=109), so every traced icon
  sails through into the clean tier and would silently poison the GT. gate2 as
  written cannot screen this source — a source-level reject is required, not a
  per-file gate.

## Overlap & dedup analysis

- **Identifier dedup is a no-op here.** Our ledger (`datasets/used-shas.txt`,
  consumed by `sample_src.py --exclude-shas`) is a set of svg-stack `Filename`
  shas. FIGR-SVG lives in a disjoint `Id` namespace, so *zero* entries collide —
  the ledger would wave through all ~1.33M as "net new" without checking whether
  the picture is a duplicate.
- **Exact-content dedup (sha256 of normalized SVG) would also mostly miss.**
  svg-stack is web-scraped born-vector SVG; FIGR-SVG is freshly traced from
  192px rasters. Even where both depict the same Noun Project icon, the byte
  streams differ (different path encodings), so content-hash matches would be
  rare. You cannot cheaply dedup the *semantic* overlap.
- **Semantic overlap is nonetheless high** — both are icon corpora, and Noun
  Project art is widely re-hosted, so a real (near-dup) fraction of FIGR-SVG
  depicts subjects already present in svg-stack. Near-dup dedup (perceptual hash
  on renders, or embedding clustering) is not in the pipeline and is not worth
  building for a source that is disqualified on provenance anyway.

Net: the pipeline's dedup would *over*-count FIGR-SVG as net-new; the true
unique-and-usable contribution is ~0 once provenance is honored.

## Content character vs svg-stack

No meaningful distributional difference in the useful direction. svg-stack is
icon-dominant flat few-color born-vector; FIGR-SVG is a *narrower, worse* slice
of the same region — strictly monochrome, strictly single-path, uniform canvas,
and raster-derived. FIGR-8's "fonts+icons for few-shot generation" framing does
not translate to typography value here: there are no glyph outlines in the
`Svg` payloads sampled, just pictogram silhouettes. It removes color and stroke
diversity rather than adding any.

## PLAN

None — do not mint. If icon *volume* is ever wanted despite saturation, prefer
born-vector icon sources already catalogued (Microsoft Fluent System Icons ~18.6k
MIT, Tabler ~6.2k, nyuuzyou/svgfind 3.66M CC-BY) over a raster-traced,
license-unstated corpus.

If someone insists on a test before rejecting: render ~200 FIGR-SVG paths and
overlay against the original FIGR-8 raster; confirm the tracing artifacts by eye
(est. ~1 hour, no GPU). The expected outcome is confirmation of tracing, at
which point the source is dropped.

## RISKS (if minted despite this verdict)

- **GT contamination:** model learns to emit trace-shaped contours; end-to-end
  deltaE on the real-file bench likely *worsens* on clean inputs.
- **Gate blind spot:** these pass gate2 as "clean," so the poison is invisible
  in tier stats — it would only surface as a bench regression after a full
  ~$15–30 training run.
- **Legal:** untenable commercial-use terms on the underlying Noun Project art;
  no license on the HF card to rely on.
- **Corpus-balance drift:** +1.33M monochrome single-path icons would swamp the
  1.165M existing mix, pushing the model further toward the already-saturated
  class and away from the measured gaps.

---
Sources: [starvector/FIGR-SVG card](https://huggingface.co/datasets/starvector/FIGR-SVG) ·
[HF datasets-server /size + /info + /rows](https://datasets-server.huggingface.co/) ·
[marcdemers/FIGR-8](https://github.com/marcdemers/FIGR-8) ·
[StarVector arXiv:2312.11556](https://arxiv.org/abs/2312.11556) ·
prior census docs/research/2026-07-20-deep-research/fuel.md (rows 50–51).
