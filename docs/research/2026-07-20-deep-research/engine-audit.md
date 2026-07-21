# Engine audit: replacing inference with source facts

Audit date: 2026-07-20
Engine: `/Users/aden/development/vectorizer` (Rust workspace, `crates/{core,cli,eval}`)
Method: read the pipeline end to end from the CLI entry (`crates/cli/src/main.rs`) through
`vectorize()` (`crates/core/src/lib.rs:87`) and each stage module. Every claim below carries a
`file:line` citation into the actual source. This document supersedes the inferred internals in
`engine.md` sections 3-4 and answers open questions 1-3 there.

Headline: the engine.md ranking assumed we are "DP-based / greedy / grid-snapped" and that we
"reconcile shared edges post-hoc." Both assumptions are wrong. We run **no polyline simplifier at
all** (Schneider cubics are fit directly to the raw pixel staircase), and shared-edge consistency
is **already guaranteed by construction** (the DCEL is the whole engine). The real, cheap, missing
win is the fill colour: regions are painted with the shared quantizer centroid, not their own
pixel mean.

---

## Summary table: question -> answer

| # | Question | Answer (source) |
|---|----------|-----------------|
| 1 | Output topology | **True partition**, not stacked. One planar DCEL; every boundary is one shared edge referenced by both faces (`planar.rs:1-16`, exact-coverage test `tests/planar.rs:44-68`). Emit paints one `<path>` per region back-to-front by area with a self-coloured overfill stroke to hide AA seams (`emit.rs:40-77`) - a partition, lightly stroked, not vtracer-style opaque stacking. |
| 2 | Boundary extraction | **Crack-following on the integer pixel-corner lattice** (`planar.rs:136-167`). Vertices are lattice nodes with crack-degree != 2 (`planar.rs:176-183`); all `pts` are `(i32,i32)` integer coords (`planar.rs:69`). Staircase polygons, **grid-snapped**. No marching squares, no subpixel extraction. |
| 3 | Simplification | **None.** No Douglas-Peucker, no Visvalingam, no penalty simplifier. Schneider cubics are fit directly to the raw staircase; straight runs collapse to one line via a chord-error test (`fit.rs:18-21`, `fit.rs:215-217`). |
| 4 | Curve fitting | **Yes** - Schneider least-squares cubic Beziers, Newton reparameterisation, recursive split on error (`fit.rs:222-258`). Error metric = max squared deviation, `error_sq` default `1.2^2 = 1.44 px^2` (`fit.rs:41`). Corner gate: k-cosine windowed detector, `corner_angle_deg` default **70.0** (not 60), window 4 (`fit.rs:39-45`, `fit.rs:123-181`). |
| 5 | Fill colour | **Shared quantizer centroid**, not per-region pixel mean. Each region copies `q.palette[pidx]` at label time (`label.rs:55`); the palette entry is the k-means/Lab centroid (`quantize.rs:390-396`) or median-cut box mean (`quantize.rs:104-124`). Regions sharing a palette index get identical fills. |
| 6 | Small regions / despeckle | `min_region_area` dissolves sub-threshold regions into the **nearest-colour dominating neighbour** (`label.rs:113-172`), which protects thin strokes. Default is **0 = no merging** (`lib.rs:55`); the fineline risk sits earlier, at the quant seed threshold. |
| 7 | Quantisation before extraction | **Yes**, mandatory. Two modes: `median_cut` (default, k=16, `lib.rs:52-53`) and `flat_quant` (the real one: Lab, edge-aware weighting, dynamic-K leader clustering, weighted k-means) (`quantize.rs:261-417`). |
| 8 | SVG structure | One `<path>` per region, `fill-rule="evenodd"` (holes via even-odd), painted largest-area-first, optional self-coloured stroke `overfill` default 1.1px `stroke-linejoin="round"`; `shape-rendering="crispEdges"` only when overfill=0 (`emit.rs:22-81`). |

---

## Detailed findings

### Q1. Output topology - TRUE PARTITION with a light overfill stroke

The engine builds a single half-edge planar subdivision (DCEL) in `planar.rs`. Boundaries are
enumerated once as unit "cracks" between differently-labelled cells (`planar.rs:142-167`), traced
into maximal edges (`planar.rs:211-275`), and each edge stores `left`/`right` region labels
(`planar.rs:76-78`) with two twin half-edges (`planar.rs:284-304`). An `Edge` is shared by exactly
two regions, so both adjacent faces walk the **same** `segs` chain (forward or reversed):
`emit.rs:107-115` and the module contract at `planar.rs:6-16`.

This is verified, not asserted: `tests/planar.rs:34-68` sums the signed area of every face ring and
requires it to equal `w*h` exactly (donut/hole cases at `tests/planar.rs:80-83`, 2x2 at `:92-93`).
Zero double-counting = a genuine gap-free partition.

Emit is a painter over that partition: `emit.rs:36-41` sorts regions largest-area-first and paints
back-to-front so an inner region's AA edge composites over its neighbour, not the canvas; each path
is optionally stroked with **its own fill colour** at `overfill` width (default 1.1px,
`lib.rs:58`, applied `emit.rs:68-73`) to defeat the "two 50%-coverage AA fills never sum to opaque"
seam. This is *not* vtracer's stacked mode (full opaque silhouettes drawn back-to-front); it is a
true partition plus a ~1px cosmetic stroke. Shared geometry is identical on both sides, so there is
no seam to begin with - the stroke only fixes renderer anti-aliasing at the shared edge.

### Q2. Boundary extraction - integer lattice crack-following, grid-snapped

Cracks live on the `(w+1) x (h+1)` grid of pixel corners (`planar.rs:29-32`, `:131-134`). A
vertical crack separates cells `(x-1,y)`/`(x,y)`; horizontal separates `(x,y-1)`/`(x,y)`
(`planar.rs:140-167`). Vertices are lattice nodes whose crack-degree != 2 (i.e. 3+ regions meet)
(`planar.rs:176-183`); degree-2 nodes are interior polyline points. Every stored boundary point is
an integer `(i32,i32)` (`Edge::pts`, `planar.rs:69`). So boundaries are exact pixel staircases on
the integer grid; there is **no subpixel edge reading** at extraction and **no vertex is ever moved
off an integer coordinate**.

Consequence for engine.md Rec 2 (subpixel vertex adjustment): junction vertices - the endpoints of
every edge - are hard-pinned to integer lattice coords and never adjusted (`fit.rs:6-8`,
`fit.rs:95-98`). Interior shape does get subpixel positions, but only as a side effect of the
Bezier least-squares averaging the staircase (`fit.rs:18-21`), never by an explicit fit-line
intersection. So the "integer-snapped vertices" premise is **confirmed for junctions**.

### Q3. Simplification - there is none (fit runs on the raw staircase)

This is the biggest correction to engine.md. There is **no polyline simplification stage**. `fit.rs`
top comment is explicit: "Fitting through the raw staircase (rather than a pre-simplified polygon)
is deliberate" (`fit.rs:18-21`). `fit_planar` (`fit.rs:49-55`) feeds `Edge::pts` (the integer
staircase) straight into `fit_edge`. The only "simplification" is emergent:

- a smooth run whose points are all within `error_sq` of the straight chord collapses to a single
  `Line` (`fit.rs:213-217`, chord-error `max_line_error_sq` at `fit.rs:265-285`);
- otherwise a Schneider cubic is fit and recursively split.

So engine.md's entire "are we DP or greedy?" framing (sections 2.2, 2.4) is moot: we are neither.
And critically, **the fit is per-edge** (`fit.rs:49-55`), i.e. per shared arc - so the "simplify
each shared arc exactly once, both faces reference it" property (engine.md Rec 3 / E3 shared-arc)
is **already true by construction**, not a missing feature.

### Q4. Curve fitting - Schneider LSQ cubics with a 70-degree corner gate

Full curve fitting is present and is the Graphics Gems Schneider algorithm:
`generate_bezier` (`fit.rs:305-354`, least-squares for the two handle lengths with fixed endpoints
and tangents), `reparameterize`/`newton_root` Newton-Raphson (`fit.rs:374-393`, capped at 4 iters
`fit.rs:202`), and recursive split at the worst point (`fit.rs:253-257`). Error metric is max
squared point-to-curve distance (`max_error`, `fit.rs:357-370`); acceptance threshold `error_sq`
default `1.44 px^2` (`fit.rs:41`, CLI `--tol T` -> `T*T` `eval/src/main.rs:71-74`).

Corner/cusp gating **exists** and runs before smoothing: `detect_corners` (`fit.rs:123-181`) is a
windowed k-cosine measure with non-maximum suppression; the threshold is `corner_angle_deg` default
**70.0**, window 4 (`fit.rs:38-46`). Corners split the edge into arcs each fit independently
(`fit.rs:84-105`); open-edge endpoints are always corners (`fit.rs:95-98`). So engine.md E3's
"add a hard corner gate >= 60 deg" is **already implemented at 70 deg** - the recommendation to add
one is refuted; only the threshold value differs.

### Q5. Fill colour - shared quantizer centroid, NOT per-region pixel mean

`connected_components` sets each region's colour to the palette entry of its quantized index at the
moment the region is born: `region_color.push(q.palette[pidx])` (`label.rs:55`). It never revisits
the source pixels. The palette entry is:

- for `flat_quant`, the weighted Lab k-means centroid mapped back to sRGB (`quantize.rs:390-396`);
- for `median_cut`, the arithmetic mean of the box's pixels (`quantize.rs:104-124`).

Two distinct regions that happen to share a palette index therefore get the **identical** fill, and
a single flat region is painted with the global centroid for its colour, not the mean of its own
member pixels. On the Google-G test this centroid lands at mean dE ~0.50 (per `CLAUDE.md`), whereas
a per-region residue mean would recover the exact flat colour (dE ~0), which is what vtracer does
(engine.md 2.5). This is the cleanest confirmed gap in the whole audit.

### Q6. Small regions, despeckle, finelines

`min_region_area` (Config `min_region_area`, `lib.rs:36-39`) drives `merge_small`
(`label.rs:113-172`). The merge target is the **nearest-colour dominating neighbour** (larger area,
or equal area with smaller label id, to guarantee termination) - deliberately not largest-area, so
a thin gold serif merges into the gold body rather than being carved into the white background
(`label.rs:100-111`, `:144-161`). After merging, labels are compacted (`label.rs:176-191`).

Default `min_region_area` is **0** (no merging, `lib.rs:55`); eval/CLI set it via `--min-area`.
The real fineline exposure is upstream at quantization: a flat colour whose total edge-weighted
weight is below `min_weight_frac` (default 0.002) never seeds its own palette mode and is absorbed
into the nearest one (`quantize.rs:337-340`); `edge_floor` (0.25, `quantize.rs:250`) keeps thin
features partly weighted but does not guarantee survival. This matches the fineline gap flagged in
engine.md 2.6.

### Q7. Quantisation before extraction

Mandatory first stage (`lib.rs:91-94`). `median_cut` is the default (k=16, `lib.rs:52-53`): widest-
channel-range box splitting at the median, palette = box means (`quantize.rs:41-127`). `flat_quant`
is the real one (`quantize.rs:261-417`): composite over white, edge-aware per-pixel weight
`floor + (1-floor)*exp(-(grad/sigma)^2)` (`quantize.rs:282-314`), Lab leader clustering into
dynamic-K modes gated by `merge_de` (default 8.0) and `min_weight_frac` (`quantize.rs:331-340`),
then weighted Lloyd k-means (12 iters, `quantize.rs:347-385`). Note: default pipeline still ships
median-cut; flat is `--quant flat`. Any quant error is baked into every downstream region colour
(see Q5).

### Q8. SVG structure and AA interaction

`emit::to_svg` (`emit.rs:22-81`): one `<path>` per region, `fill-rule="evenodd"` so holes need no
special handling (`emit.rs:71,75`); regions painted largest-first (`emit.rs:40-41`); when
`overfill>0` each path also gets `stroke=<own fill>` `stroke-width=<overfill>`
`stroke-linejoin="round"` (`emit.rs:68-73`); when overfill=0 the whole SVG carries
`shape-rendering="crispEdges"` (`emit.rs:45-49`). Coordinates are emitted at 2-decimal precision
(`emit.rs:148-161`). The overfill stroke is the only piece that interacts with bench-time AA: it is
the current mitigation for the 50%-coverage seam and is functionally a lightweight, per-path version
of stacking.

---

## Mapping onto engine.md's recommendations

| engine.md lever | Finding | Verdict |
|-----------------|---------|---------|
| **E1** stacked-emit flag | We emit a true partition + ~1.1px self-coloured overfill stroke, back-to-front by area (`emit.rs:36-77`). Not opaque-silhouette stacking. Our spine already keeps `holes ~0` (CLAUDE.md), so seams are a renderer-AA cosmetic, not a topology defect. | **Partially present** (overfill is a lighter mitigation). Full stacked mode is a niche add; low expected deltaE on this bench. |
| **E2** Potrace penalty simplifier + both-ends clipping | Premise refuted: we have **no** simplifier to replace; we fit Schneider cubics to the raw staircase (`fit.rs:18-21,49-55`). The real question becomes whether a global-optimal polygon stage *before* Schneider beats direct staircase fitting. | **Premise refuted; open as a redesign, not a swap.** High effort, uncertain marginal gain. |
| **E3a** hard corner gate >= 60 deg | Already implemented at **70 deg** (`fit.rs:38-46,123-181`). | **Present.** Only a threshold sweep remains. |
| **E3b** mean-of-residue fills | Refuted: fills are shared quantizer centroids (`label.rs:55`, `quantize.rs:390-396`), not per-region pixel means. | **Confirmed gap.** Cheap, direct deltaE win. |
| **E3c** shared-arc single simplification | Already true by construction: fit is per-edge, both half-edges share one chain (`fit.rs:49-55`, `planar.rs:6-16`, `tests/planar.rs:44-68`). | **Present.** No ticket - this is the engine's spine. |
| Rec 2 subpixel vertex adjustment | Junction vertices are integer-snapped and never adjusted (`fit.rs:6-8,95-98`, `planar.rs:69,176-183`). | **Confirmed missing** for junctions. |

---

## Ranked ticket list (by expected deltaE-per-effort against the 0.16 perfect-input gap)

**T1 - Per-region residue-mean fills (do this first).**
File: `crates/core/src/label.rs` (region colour assignment `:55`), reading source pixels composited
over white. Change: after labelling, replace each region's `region_color` with the arithmetic mean
of its own member pixels' source RGB (vtracer's `residue_color`), instead of `q.palette[pidx]`.
Optionally down-weight/exclude high-gradient AA pixels (reuse the `grad_at` idea from
`quantize.rs:283-304`) so edge blend pixels do not bias the mean. Decouples fill from quantization;
each flat region recovers its exact colour (dE ~0.5 -> ~0 on flat interiors, and every region
independently). Effort: **~2-4 h**. Highest deltaE-per-effort: it is a localised change to one
assignment and it moves the metric the bench actually rewards (area-dominated mean dE on flat
fills). Confirms E3b.

**T2 - Corner-gate threshold sweep 70 -> 60 deg (and window).**
File: `crates/core/src/fit.rs:41` (`corner_angle_deg` default), already exposed as `--corner-angle`
(`eval/src/main.rs:78-80`). No code change to run the sweep; just a `veval` parameter sweep on
corner-heavy icons, then flip the default if 60 wins. Effort: **~2-3 h** (mostly measurement).
Low risk, small but real if our current 70 rounds genuine 60-70 deg corners. Tunes the already-
present E3a.

**T3 - Subpixel junction vertex adjustment (Potrace-style).**
File: `crates/core/src/fit.rs` (add a pre-fit pass) reading `Edge::pts` and neighbouring edges at
shared vertices; least-squares fit-lines to the two incident arcs and place the junction at their
intersection, clamped to a 1/2-px box, applied once per vertex so both edges stay consistent
(preserves the shared-arc invariant). Effort: **~1-1.5 days**. Medium payoff: mean-dE on icons is
area-dominated and near-blind to half-pixel junction error (engine.md 2.1, caveat), so likely a
small bench move; keep behind a flag and measure. Addresses Rec 2.

**T4 - Global-optimal polygon stage before Schneider (E2 redesign).**
File: new module + hook in `crates/core/src/fit.rs:49-55` (run per-edge on `Edge::pts`, feed the
polygon to the existing cubic fitter). This is the large lever engine.md ranked #1, but its premise
(replacing a bad DP/greedy simplifier) does not hold - we already subpixel-average the staircase via
LSQ, so the marginal gain over direct fitting is unproven. Effort: **~3-5 days**. Gate it behind the
`--mode polygon` ablation engine.md proposes (Rec 1 experiment): run vtracer `--mode polygon` vs our
polyline output first; only build T4 if that ablation shows the gap survives at polygon stage.

**T5 - True stacked-emit mode (E1).**
File: `crates/core/src/emit.rs`. Add an opt-in that emits each region as a full opaque silhouette
back-to-front instead of a partition. Effort: **~1 day**. Lowest priority for the 0.16 gap: our
partition already keeps `holes ~0`, the overfill stroke already handles the AA seam, and stacking
sacrifices the zero-gap invariant that is the engine's reason to exist. Build only if a specific
seam artefact survives overfill on the bench.

**Not a ticket - confirmed already done:** shared-arc single simplification / shared-edge
consistency (E3c, engine.md Rec 3). The DCEL guarantees it and `tests/planar.rs:44-68` proves it.
engine.md open questions 1 and 2 are hereby answered: we do **not** reconcile post-hoc, and we run
**no** DP/greedy/grid-simplifier - so the single highest-value "audit our own tracer" step it asked
for is complete, and it redirects the roadmap from the simplifier (T4, deprioritised) to the fill
colour (T1, promoted).

### Recommended order

T1 (hours, direct win) -> T2 (hours, measurement) -> run the `--mode polygon` ablation -> T3 or T4
gated on what the ablation shows -> T5 only if a seam survives. The 0.16 perfect-input gap is most
plausibly attacked by T1 first (fill exactness on every region) before any geometry work.
