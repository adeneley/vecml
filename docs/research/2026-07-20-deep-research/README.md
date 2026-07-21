# Deep Research Round — Master Synthesis

Date: 2026-07-20. Reader: the project owner allocating ~$50–100 of GPU/engineering budget.
Source theme reports on disk in this folder: [`fuel.md`](./fuel.md) (datasets),
[`recipe.md`](./recipe.md) (training priors), [`engine.md`](./engine.md) (Stage-2 tracer gap).

Cost convention below: GPU dollars assume rented mid-tier GPU at ~$1–1.5/GPU-hr; one full
927k-scale Stage-1 run ≈ $15–30 of compute. Engineering-only tickets carry ~$0 compute.

---

## 1. Verdict per theme

**FUEL (datasets).** We are still data-constrained (val loss falls 0.00782→0.00630 with zero
train/val gap), but the shallow ~0.10 data-scaling exponent means *more icons buys almost nothing*
— the leverage is distribution, not volume. The single highest-value, lowest-risk fuel is
**typography minted from open fonts**: every glyph is a born-vector Bezier extractable with
fontTools/SynthTIGER at zero autotrace-contamination risk, directly filling our #1 measured gap.
Behind it, **Openclipart (178,604 CC0, tag-filterable)** fills the colour/illustration gap, **flat
emoji** (Twemoji/Noto/Fluent, permissive, exclude all 3D/gradient variants) add organic multi-region
colour, and **stroke-icon families** attack fineline. Two hazards the verification round corrected:
the "Apache Google Fonts has Roboto/Open Sans" escape hatch is false (both are OFL now, the clean
`apache/` subset is ~42 novelty faces), and OFL Q1.25 makes font-producing ML output virally OFL — a
legal question for a *commercial* ship, not a blocker for research runs. There is no off-the-shelf
autotrace classifier; build cheap structural heuristics (topology/primitive-absence/path-explosion/
palette-quantisation) and hold out per-source + per-severity val slices.

**RECIPE (training priors).** We sit firmly on the data-dominated branch of the scaling law, so
**do not grow the net** (width 48→64 is predicted to buy little); spend on data and epochs instead.
But the shallow exponent means the payoff comes from *gap-coverage* and *harder degradation*, not raw
count. The 500k divergence is fully explained by a missing gradient clip and/or the bf16 divergence
lottery — both cheap to neutralise, and it is **not** a data-ceiling signal. Do **not** build
multi-task optimizers (tuned fixed-weight scalarization matches GradNorm/PCGrad/MGDA); settle the CE
weight with a cheap 5-point sweep whose load-bearing arm is weight=0. And our 95.97% label accuracy
is hiding the pixels that matter — switch label eval to a boundary F-score band, and prefer
distance-weighted CE over exotic boundary losses. The most important cross-theme number the report
surfaces: **vtracer on perfect input scores 1.28 but on model-cleaned input only 1.52 — a 0.24-deltaE
cleanup headroom that better Stage-1 can still recover.**

**ENGINE (Stage-2 tracer gap).** Our Rust planar-map engine loses to vtracer because of
**curve/polygon-fitting fidelity, not topology and not subpixel-edge reading** — both reference
tracers discard anti-aliasing before fitting, so our loss on *perfect* input (1.44 vs 1.28) is purely
the fitting stage. Three levers, cheapest first, all near-$0 compute: (1) verify the fill is
**residue-mean not palette-snapped** (deltaE~0 on flat regions by construction — free deltaE if we're
snapping); (2) add a **hard corner-angle gate before smoothing** so corners are never rounded; (3)
adopt **Potrace's globally-optimal polygon + subpixel vertex adjustment** (vtracer beats us with only
a *greedy* version of this). The whole ranking rests on an unaudited assumption about what our engine
does today, so **a ~1-day read of our own tracer is the single highest-value next step in this theme.**
Caveat: mean-deltaE on the icon-heavy 24-image bench is area-dominated, so fitting gains read clearly
but future subpixel work will barely move it.

---

## 2. THE PRESCRIPTION — the next three spends

Sequenced by **deltaE-per-dollar**, cheapest and most certain first. The stability kit (grad clip +
warmup + fp32 master/optimizer/softmax accumulation + EMA) is a **$0 prerequisite baked into every
GPU run below** — never spend compute without it, or you risk repeating the 500k divergence.

### Spend 1 — Engine audit + the two free fitting fixes  ·  ~$0 GPU, ~1–2 eng-days

**Do this first because it is the cheapest deltaE on the board and it is diagnostic for everything
else.** Read our own tracer (`engine.md` §4 Q1 is explicit that the whole ranking assumes we're
DP-based/greedy/grid-snapped — inferred from the gap size, not from our code). Then:
- **Ticket E-1 (grep, ~1 hr):** confirm region fill = arithmetic mean of member pixels, not palette-
  snapped/quantised. If snapped, switch to residue-mean. On flat icon regions this is deltaE~0 by
  construction — this is *the whole game* on icon content.
- **Ticket E-2 (~0.5 day):** add a hard corner gate (turn angle ≥ ~60°) before any smoothing; 3-point
  scheme at corners. Our label errors already concentrate at boundaries; uniform smoothing rounds
  exactly those corners.
- **Ticket E-3 ablation (~0.5 day):** run vtracer `--mode polygon` vs our polygon output on the
  24-image bench to localise how much of the 0.16 perfect-input gap is polygon-stage vs Bezier-stage.

**Predicted outcome:** if we palette-snap or lack a corner gate, E-1+E-2 recover a meaningful slice of
the Rust 1.79→1.52 cleaned-input gap for zero compute. E-3 tells us whether the bigger lever (Potrace
global polygon, ~a week of engineering) is worth queuing. **Honest scope note:** the *shipped* bench
(1.520) uses vtracer, so engine work only moves the shipped number if Rust surpasses vtracer — but at
~$0 this is the correct first spend because it's the cheapest deltaE and it de-risks the strategic goal
of owning the full stack.
**Confidence:** High that the fixes are cheap and E-1/E-2 apply; Medium that they fully close the gap.

### Spend 2 — Degradation-realism upgrade + one retrain with stability kit  ·  ~$20–40 GPU, ~1–2 eng-days

**This is the spend most likely to move the *shipped* (vtracer) bench, because the 1.52→1.28 gap is a
cleanup-quality gap, not a tracer gap.** The 24-image bench measures on damaged inputs, and our
degradation wreck is a first-order, fixed-order, JPEG-only model the restoration literature is
unanimous under-covers reality. Port Real-ESRGAN/APISR degradation code and add: second-order chain
(apply blur→resize→noise→compress twice), **sinc ringing/overshoot** (~80% prob — vector icons are
all hard edges, so this is arguably our single most-relevant omitted artifact), randomised operation
order (BSRGAN), modern codecs (WebP/AVIF/H.264/H.265 probability table), and non-aligned double-JPEG
(FBCNN). Retrain one 927k model with the full stability kit and cosine length == planned steps. This
run also recovers the 500k point cleanly as a free by-product.

**Predicted outcome:** vtracer-on-cleaned moves off 1.52 toward the 1.28 perfect-input ceiling on the
icon bench; the sinc/edge-ringing term is the likely largest single contributor. No epoch-1 loss rise,
confirming the divergence was stability not data.
**Confidence:** High on the mechanism (unanimous literature; icons are hard-edge content); Medium on
magnitude on our exact bench.

### Spend 3 — Targeted gap mint (typography + clipart + flat emoji) folded into one run  ·  ~$20–40 GPU, ~2–3 eng-days

Mint a gap-focused shard and fold it into the 927k corpus (re-fit the intercept since the mix changes).
**Mint composition (all near-zero autotrace risk, all emitting our supervision shape = clean render +
per-region label map):**
- **Typography (primary):** ~30–50k single-glyph + short-wordmark pairs from ~200 Apache/OFL fonts via
  fontTools/SynthTIGER, plus a slice of pre-vectorised SVG-Fonts/glyphazzn (~14M glyphs, no render). #1
  gap, zero contamination.
- **Openclipart CC0 (colour gap):** ~50k, with the tag-based autotrace filter (`autotrace`,
  `vectorized`, `upload2openclipart`) and dedup against freesvg/publicdomainvectors.
- **Flat emoji (organic multi-region colour):** the ~11k flat set from Twemoji (CC-BY) + Noto
  (Apache) + Fluent (MIT, Flat + High-Contrast only). **Exclude every 3D/gradient variant** to respect
  the gradients-out constraint.
- **Stroke icons (fineline):** ~20k across Tabler/Lucide/Phosphor-weights/Material-Symbols; report
  their deltaE *separately* from filled icons to expose any hairline tracer penalty.

Build the autotrace heuristic filter (~1 eng-day, no GPU) before minting anything from mixed-provenance
sources. Hold out per-source + per-severity val slices and optimise the mixture toward end-to-end
deltaE, not raw val loss; late-anneal-upsample the typography/fineline sources in the last 10–20% of
training.

**Predicted outcome:** larger end-to-end deltaE improvement per image than an equal-size icon shard,
and — importantly — this run *also produces the higher-edge-density / typography eval set* that
`engine.md` and `fuel.md` both flag as necessary before finer tracer tuning is measurable. It attacks
the product's real distribution gaps even where the current icon bench under-registers them.
**Confidence:** High that fonts/clipart/emoji are clean, mintable, and fill real gaps; Medium that the
deltaE delta on the *current* icon-heavy bench is large (its payoff is partly on content the bench
barely tests — hence build the new eval slice here).

**Budget sequencing.** Do Spend 1 first ($0) — it may reprioritise everything. Then Spend 2 (~$20–40)
as the highest-probability mover of the shipped bench. Spend 3 (~$20–40) if budget remains, or run it
before Spend 2 if the strategic priority is typography/product-coverage over squeezing the icon bench.
Total lands inside $50–100 with the stability kit free-riding on both runs. **Cross-theme punchline:
the engine fixes (E-1/E-2) are near-$0 for real deltaE and beat any GPU run on cost-per-deltaE, so
they go first even though they don't move the vtracer-shipped number — and the biggest shipped-bench
lever is not the tracer at all but the 0.24-deltaE cleanup headroom that Spend 2 targets.**

---

## 3. What we should NOT spend on

- **Growing UNet width 48→64 (or any capacity increase).** We're on the data-dominated branch; the
  model-size term drops out of the scaling law in this regime, and inference is per-user-image so a
  bigger net is also cost-negative. (`recipe.md` §2.1, §2.5)
- **More icon-only data.** Shallow ~0.10 exponent → a 10× icon mint buys ≤~19% val-loss reduction and
  probably less. Volume-only corpora (svgfind 3.6M, FIGR-SVG 1.3M) relieve starvation but don't move
  the gaps. (`fuel.md` §2.2, `recipe.md` §2.2)
- **Multi-task optimizers (GradNorm/PCGrad/MGDA/uncertainty weighting).** Tuned fixed-weight
  scalarization matches or beats them; LR tuning dwarfs the weighting choice 6–7×. Use the cheap CE
  sweep instead. (`recipe.md` §2.8)
- **Exotic boundary losses (Kervadec/Lovász).** On non-pathological data they move mIoU only 1–2.5%,
  add instability, and need region-loss pairing / α-schedules. Distance-weighted CE is the cheap
  drop-in. (`recipe.md` §2.9)
- **Subpixel-from-coverage edge reading in the tracer.** Neither reference tracer does it, and
  mean-deltaE on icons is near-blind to half-pixel error — it won't move the current bench. Park it as
  a later ceiling-raiser tied to swapping the argmax label head for a soft/coverage head. (`engine.md`
  §2.1, Rec 7)
- **Raising batch size.** Batch 16 is already in the quality-optimal band; raising it only buys
  wall-clock, forfeits small-batch regularization, and narrows the stable-LR window (a divergence
  risk). (`recipe.md` §2.7)
- **NC-licensed / trademark-risky corpora as mint feedstock.** FloorPlanCAD/CubiCasa (CC-BY-NC),
  FIGR-8 (Noun Project commercial restriction), DESCAN-18K (NC), Font Awesome Brands / clker
  (autotrace-from-raster). Use NC sets as realism *references* only. (`fuel.md` §2.3–2.4, table)
- **Building a rigorous 4-point scaling law right now.** It's diagnostic, not deltaE-moving; even 3
  points give an exact-interpolation fit with zero degrees of freedom. Recover the 500k point as a
  free by-product of Spend 2, but don't spend a dedicated budget on curve-fitting before the gap mint.
  (`recipe.md` §2.2, Run A)
- **Gradients / 3D-emoji content.** Excluded by design; the new Noto 3D / Fluent Color/3D variants
  violate the constraint. (`fuel.md` §2.2)

---

## 4. Unresolved questions worth a future round

1. **What does our Rust tracer actually do today?** The entire engine ranking assumes DP/greedy/
   grid-snapped, inferred from the gap. Spend 1's audit answers it; until then Recs E-1..E-3 are
   hypotheses. (`engine.md` §4 Q1–Q2)
2. **Does OFL Q1.25 bind a raster-cleanup model trained on rendered OFL glyphs?** Confirmed for
   font-producing systems, but the render-to-raster-then-train case is legally untested. Needs
   counsel, not more research, before any *commercial* ship. (`fuel.md` §5 Q1)
3. **How far do our tracer ceilings (vtracer 1.28 / rust 1.44) degrade on hairline/typography/CAD
   content?** Measured only on filled icons; could flip the value ranking of the fineline fuel. Needs
   the dedicated high-edge-density eval slice that Spend 3 produces. (`fuel.md` §5 Q3, `engine.md` §4 Q5)
4. **The real customer-file degradation distribution.** We're inferring the codec/scan mix from SR
   literature; a small labelled sample of actual print-shop customer uploads would let us weight the wreck
   correctly and later enable AnimeSR-style learned degradation operators. (`fuel.md` §5 Q6)
5. **Where our broken-power-law knee sits and our true exponent/floor.** Depends on our synthetic-
   damage severity (which we control but haven't characterised). Resolvable with the 500k + a 300k
   point once Spend 2 gives clean runs. (`recipe.md` §4 Q1–Q2)
6. **Is the true product bottleneck Stage-1 or Stage-2?** rust 1.79 vs vtracer 1.52 on identical
   cleaned input, and the 1.52→1.28 cleanup headroom, both live — a parallel question is whether a
   week of Potrace-polygon engine work outranks a cleanup run for the shipped bench. (`recipe.md` §4 Q6)
7. **Does the CE aux head earn its 4.3M-param capacity in our zero-gap regime?** The overfitting-
   reduction rationale doesn't apply; the representational benefit is plausible but unmeasured. The
   weight=0 sweep arm settles it against the current mix — but the answer may flip once gap data lands.
   (`recipe.md` §4 Q5)
