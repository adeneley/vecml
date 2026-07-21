# Source feasibility: starvector/svg-diagrams

Single-source study for the Stage-1 cleanup UNet corpus. Question: is `starvector/svg-diagrams` worth minting, given the live-text problem, and what does a text-to-path promotion stage buy. Measured against fuel.md (2026-07-20-deep-research) row #12 and open question #5.

## VERDICT

**Feasible, conditional on two things: a text-to-path promotion stage, and license clearance for commercial use.**

Out of the box the source yields **~0 usable rows**: the pipeline mints only the `clean` tier (`corpus_remote.py` stages `labelled/clean/` and drops `warn`/`reject`), and essentially every diagram carries `<text>` (93.6% of a 700-row sample), which gate2 flags `live-text` → `warn`. Nothing reaches the wrecker.

With a `usvg` text-to-path promotion stage bolted onto the `warn` tree, **estimated net usable ≈ 85,000 clean-tier, gradient-free, born-vector diagram SVGs** (range 80k–105k) from the 182,618 total. This is the first source that attacks typography-*in-context* and diagram fineline (arrows, connectors, axes, bonds) at scale, complementing the isolated-glyph font mint in fuel.md Tier A.

## Verified facts

- **Size/format:** 182,618 rows (train 182,144 + test 474), 4 parquet shards, 379 MB download / 1.26 GB on disk. Two columns: `Filename` (sha id), `Svg` (raw SVG string). Same schema as svg-stack, so it drops straight into `label_split.py`. (HF datasets-server `/info` + `/api/datasets`.)
- **License:** **UNSTATED.** The HF `license` field is empty and the dataset card says nothing. This corrects fuel.md open-Q #5 partially: the set is **public and NOT gated** (`gated:false`), unlike fuel.md's "same gating as svg-stack" note, but the license gap is real and is a **hard blocker for a commercial vectorizer until cleared**. Provenance is web-scraped diagrams (arxiv:2312.11556, StarVector); mixed/unknown per-file rights.
- **Content character (from sampled rows):** flowcharts, network diagrams, bar/line/scatter charts, Graphviz-style node graphs, molecular structures, kanji. Confirmed text-dense and thin-stroke, i.e. exactly the gap distribution. Path counts run med-high (typically 4–20 `<path>` plus primitives per file). Autotrace pollution is low: **0/700 trace-suspect**, consistent with these being tool exports (Graphviz/matplotlib/mermaid/D3) rather than raster traces.

## The live-text problem, quantified

gate2 run over a 700-row stratified sample (7 offsets across the shard):

| tier | count | share |
|---|---|---|
| clean | 0 | 0% |
| warn | 490 | 70.0% |
| reject | 210 | 30.0% |

Full flag census (rows, flags co-occur):

- `W:live-text` 655 (93.6%) — near-universal, the reason clean=0
- `R:script` 110 (15.7%) — embedded JS (interactive/D3 exports); hard reject
- `W:editor-junk` 102, `W:css-block` 73, `W:mask/clip` 49, `W:filter` 40, `W:low-occupancy` 9
- `R:parse-error` 45 (6.4%), `R:external-ref` 33, `R:foreignObject` 31, `R:raster` 15, `R:raster-b64` 7, `R:no-size` 2, `R:off-canvas` 1

Why live-text is `warn` not just cosmetic: a `<text>` render is font-dependent, so the "clean SVG" answer key is non-reproducible across machines (resvg substitutes fonts). Converting text to paths makes the ground truth deterministic and machine-independent, which is the actual promotion win — on top of supplying the typography.

## Text-to-path promotion, assessed

**Tooling choice: `usvg` (from the resvg project).** The pipeline already renders with `resvg-py` (`src/vecml/degrade/renderer.py`), and `usvg` is the same project's normalizer. It **converts text to paths by default** and, in the same pass, resolves CSS, strips editor namespaces, and resolves `<use>`. That means one `usvg` call clears three gate2 warn flags at once: `live-text`, `editor-junk`, `css-block`. Output is guaranteed consistent with what the wrecker's resvg renderer would draw.

- rsvg-convert / cairosvg are **renderers** (raster/PDF only) — cannot emit flattened SVG, so not usable for the promotion itself.
- Inkscape CLI (`--export-text-to-path --export-plain-svg`) works but is slow (per-file process spawn, headless GTK) and rasterizes filters. Fallback only.
- fonttools alone would require reimplementing text layout + kerning + font matching — high effort, rejected.

**Font handling:** diagrams reference Arial/Helvetica/Times/sans-serif/monospace. On a Linux pod, install the metric-compatible substitutes (Liberation Sans/Serif/Mono = Arial/Times/Courier clones, DejaVu, Noto) so `usvg` has outlines to convert to. Font *substitution does not break supervision*: `clean.png` is rendered from the same flattened SVG, so the (SVG, wrecked-render) pair stays self-consistent. The only cost is stylistic drift from the original foundry face, which is acceptable.

**Modelled promotion outcome** (700-row sample, treating `{live-text, editor-junk, css-block}` as usvg-clearable, everything in `reject` as unpromotable):

| outcome | share | note |
|---|---|---|
| hard-reject, unpromotable | 30.0% | script / parse-error / external-ref / foreignObject / raster / no-size — text conversion cannot fix these |
| promote → **clean** | 58.3% | usvg clears live-text + editor-junk + css-block together |
| promote → still-warn | 11.7% | residual `mask/clip` (40), `filter` (39), `low-occupancy` (9) — not minted |

Net-usable estimate: 58.3% best case, minus gradient rows (4.7% overall, excluded per the gradients-out constraint), minus ~10–15% conversion attrition (usvg failures, degenerate glyph output, filter rows that usvg rasterizes into a fresh `raster` reject). Lands at **~48% of 182,618 ≈ 85,000** (band 80k–105k). Re-gating after conversion is mandatory so any usvg-introduced rasters/failures drop out automatically.

## PLAN (if license clears)

1. **Ingest + label.** Add svg-diagrams shards to `label_split.py` (schema-identical). Writes `clean/` (≈0), `warn/`, `reject/` + manifest. Cost: CPU-only, minutes.
2. **Promotion stage** (new `scripts/promote_text.py`, ~1 engineer-day). Iterate the `warn/` tree filtered by manifest to rows whose flag set ⊆ `{live-text, editor-junk, css-block}` (skip mask/clip/filter/low-occupancy and all reject). For each: run `usvg --use-fonts-dir <substitute-fonts> in.svg out.svg`, then re-run `gate2.analyze` on the output. Keep only rows that come back `clean` AND `feat['gradient']` is false. Write survivors into `clean/` + append manifest. Embarrassingly parallel; usvg is ~ms/file, so ~130k promotable files run in minutes on a multicore box. No GPU.
3. **Wreck + train as normal.** Folds into the existing `wreck.py` → train flow.
4. **Validate as its own slice.** Hold out a per-source diagram val slice AND report deltaE separately from filled-icon deltaE (fuel.md open-Q #3: tracer ceilings vtracer 1.28 / rust 1.44 were measured on filled icons; hairline diagram art may trace worse). Late-anneal-upsample this slice per fuel.md §2.6 if it under-performs.

Marginal cost: one incremental training run to measure the deltaE delta (~1 GPU-day); promotion itself is CPU-only and cheap.

## Value vs our measured gaps

- **Typography (gap #1):** supplies real labels at real sizes and positions inside layouts — typography-*in-context*, which the isolated-glyph font mint (fuel.md Tier A) does not cover. Complementary, not redundant.
- **Fineline (gap #2):** flowchart connectors/arrows, chart axes/ticks/gridlines, molecular bonds are thin strokes — direct hits. Caveat: value is unproven until the hairline tracer-ceiling question is answered on a diagram eval slice.
- **Complexity:** med-high path counts and multi-region structure add distributional diversity beyond the icon-dominated corpus.

## RISKS

1. **License unstated → commercial blocker.** No license on card or API. Must be cleared before any commercial mint; treat as gating for this source specifically. (Corrects fuel.md: ungated, but license still unknown.)
2. **Font substitution drift.** Converted glyphs use Liberation/DejaVu, not the original faces. Training-valid (self-consistent GT) but the model learns substitute letterforms. Mitigate with metric-compatible clones.
3. **usvg filter/mask handling.** `usvg` may rasterize unsupported filters into an `<image>`, which the re-gate catches as `raster` and drops. Already priced into the ~12% still-warn + attrition; no silent corruption because re-gating is mandatory.
4. **High reject rate is real.** 30% is genuinely lost (script 15.7% is the big one — interactive/D3 exports). Not recoverable; the 85k figure already excludes them.
5. **Tracer ceiling on hairline unknown** (fuel.md open-Q #3). Diagram value could be lower than hoped until measured. Report deltaE per-source before committing to a large mint.
6. **Gradients present (4.7%)** violate the gradients-out constraint; must be filtered in the promotion re-gate (gate2 records `feat['gradient']` — use it as an exclusion here even though it is not a default reject).

## Correction to fuel.md

Row #12 says svg-diagrams has the "same gating as svg-stack." It is in fact **public and ungated**; the license is unstated (empty on card and API), which is the real constraint. Size confirmed at 182,618. Open-Q #5 (verify size/license) is now: size resolved, license still open pending upstream clarification.
