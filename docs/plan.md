# Plan of record

Written 18 Jul 2026, from the moat research and the GPU cost analysis.
Plain-language version; CLAUDE.md has the condensed operational form.

## Why this repo exists

vectorizer.ai = AI eyes + classical hands. We already built better classical
hands (the Rust engine: seam-free by construction, at free-tier parity on
clean flat logos). The one durable gap is degraded input: small, blurry,
JPEG-crushed, scanned logos where deciding "what shapes are really here" is
the hard part. That decision is what we train.

## Phases

### Phase 0: foundations (Mac, ~free)
- [ ] Wrecking pipeline v1: render clean SVGs, apply degradation ladder
      (JPEG quality, rescale, blur, noise, dither), emit (input, label-map)
      pairs. Pure CPU, runs anywhere.
- [ ] Corpus v0: a few thousand CC0 SVGs (Openclipart first), rendered.
- [ ] NAS pair survey (read-only): count job folders holding both a junk
      raster and a production vector; sample ~50 for quality. Decides how
      much real validation data exists.
- [ ] Dataloader + tiny U-Net + overfit-on-100-images sanity run on the M5.

### Phase 1: prototype (rented 5090s, ~$200-400 AUD total)
- [ ] Shard 200k pairs at 256px. Pre-wreck offline; never augment on the fly
      on rented GPUs.
- [ ] First real training runs, overnight each (~$15-30 AUD/run).
- [ ] Week-one calibration: measure img/s, re-anchor every cost estimate.
- [ ] 2-3 GPU pods for parallel independent variants (architecture, wreck
      recipe, loss). Questions-per-night is the metric.
- Exit test: model beats flat_quant's label map on wrecked inputs it has
  never seen, measured by full-pipeline veval metrics, not model loss.

### Phase 2: serious (5090 or A100 PCIe, ~$150-250 AUD/run, 8-15 runs)
- [ ] 512px, 1-2M pairs, real ablations.
- [ ] Validate against NAS real pairs every run; synthetic-only wins that
      fail on real pairs mean the wreck recipe is wrong, fix the recipe.
- [ ] Freeze the design here. Nothing moves to Phase 3 unfrozen.

### Phase 3: final (2x H100 NVL DDP, ~$1,000-1,500 AUD, 1-2 runs)
- [ ] 1024px fine-tune of the frozen winner.
- [ ] Export for local inference (the product runs on the Mac / shop
      machines; only training rents GPUs).
- [ ] Wire into the Rust engine as the optional neural front-end and A/B it
      in the gallery against flat_quant on the full regression set.

## Budget guardrails

- Ceiling: BALANCED scenario, ~$2,700 USD total. Lean floor ~$1,300.
- The two named ways to waste money: prototyping on H100/H200-class cards,
  and running a 1024px final before the 512px design is frozen.
- If spot/interruptible pricing is available: take it (40-60% off),
  checkpoint every 15-30 min. Beats every patience play.
- If budget is ever no object: 8x B300 runs the final overnight (~$59/hr,
  ~17-33h). Noted for amusement more than planning.

## Success metric

Not model loss. The metric is the Rust engine's veval numbers (deltaE,
PSNR, holes ~ 0) on DEGRADED inputs with the model in front, in the same
comparison gallery as every classical attempt, plus survival on the NAS
real-pair set. The target distribution is print-shop uploads, not the whole
consumer internet.
