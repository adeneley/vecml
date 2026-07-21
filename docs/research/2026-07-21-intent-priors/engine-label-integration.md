# Model-trust integration: feeding the label map to the Rust engine

Date: 2026-07-21
Engine branch: `label-input` at `/Users/aden/development/vectorizer` (local, unpushed)
Checkpoint: `runs/hippo48-1m/best.pt` (UNet base 48, 16-class label head)
Bench: 24-image relay set at `data/relay-test`, mean deltaE76 vs `clean.png` (lower better)

## What was built

A second entry point into the Rust engine that takes the region structure from an
external per-pixel class map instead of re-deriving it with the quantizer. Stage 1
(the UNet) already segments the image at ~96% per-pixel accuracy; Stage 2 previously
threw that away and re-solved region discovery on the cleaned RGB. The new path
consumes the class map directly.

### Interface

`vectorize <rgb.png> --labels <labels.png> -o out.svg`

- `labels.png` is a **greyscale (L-mode) PNG whose pixel value is the class id** (0..15).
  Chosen over an indexed/paletted PNG because it needs the least glue on both sides:
  the Python producer is `Image.fromarray(labels.astype(uint8), "L")` from the argmax
  map, and the Rust consumer is `image::open(..).to_luma8()` with each pixel value used
  verbatim as the class id. No palette parsing, no colour round-trip.
- The positional `<rgb.png>` supplies the colours the fills read; it should be the
  model's **cleaned** RGB, aligned with the labels the model predicted alongside it.
- Dimensions of labels and rgb must match; the CLI errors otherwise.

### Semantics (unchanged from the design constraints)

- **Regions = connected components of the label map.** 16 classes but many regions can
  share a class, so the map is connected-componented; `class == region` is never assumed.
  This falls out for free: `connected_components` already components the `idx` array, and
  the label path simply sets `idx` to the class map.
- **Fills = residue means over the RGB**, via the existing `assign_residue_fills`
  (interior-pixel mean, AA-boundary pixels excluded, thin-region fallback). Identical to
  the quantized path.
- **Boundary geometry** is the existing DCEL -> Schneider path, untouched.
- `merge_small` still despeckles sub-threshold label-map components; the per-class mean
  colour seeded at construction gives it a nearest-colour signal to merge by.

### Code

- `crates/core/src/quantize.rs` — `from_labels(img, class_of) -> QuantImage`: composites
  RGB over white into `src`, sets `idx` to the class map, seeds `palette` with each
  class's mean colour.
- `crates/core/src/lib.rs` — `vectorize_labels(img, labels, cfg)`; the post-quantize tail
  (label -> planar -> fit -> emit) is factored into `vectorize_quant` and shared with the
  RGB path, so the two differ only in how the `QuantImage` is built. The no-labels path is
  byte-for-byte unchanged.
- `crates/cli/src/main.rs` — `--labels <png>`.
- `crates/core/tests/labels.rs` — disconnected same-class blobs stay separate regions;
  label-map despeckle; end-to-end SVG emission. `cargo test` green (18 tests).

Python glue in `scripts/relay_eval.py`: a `rust_labels` variant exports the argmax map as
an L PNG and invokes the engine with the model's cleaned RGB; a `rust_labels_gt` ceiling
variant feeds the ground-truth `labels.png` plus `clean.png`. New `--min-area` flag
(default 8) controls label despeckle.

## Results

Aggregate over 24 images (mean deltaE, min-area 8):

| variant | mean dE | note |
|---|---|---|
| rust_flat (recorded baseline) | 1.732 | reproduced exactly (1.7318) |
| vtracer_flat | 1.520 | reproduced exactly (1.5201) |
| **rust_labels** (model labels + cleaned RGB) | **2.143** | worse than rust_flat |
| rust_labels_gt (ground-truth labels + clean RGB, ceiling) | 1.605 | |

The label path **loses in aggregate**. min-area is not the cause: sweeping 0/4/8/16 moves
rust_labels only 2.12–2.15 and the ceiling not at all (1.605).

### Why: the loss is gradient art, and it is structural

Per-image, the median rust_labels (1.453) equals rust_flat (1.46); the mean is dragged up
by a few outliers. Those outliers correlate with `reconstruction_mae` in each image's
`meta.json` — the error of painting each ground-truth label region with a single flat
colour, i.e. the flat-fill ceiling for that image. corr(recon_mae, rust_labels_gt) = 0.63.

Stratifying the bench by it:

| subset | rust_flat | vtracer_flat | rust_labels | rust_labels_gt (ceiling) |
|---|---|---|---|---|
| **Flat art** (recon_mae < 1.0, n=10) | 1.566 | 1.270 | **1.478** | **1.133** |
| **Gradient/shaded** (recon_mae ≥ 1.0, n=14) | 1.850 | 1.699 | 2.619 | 1.941 |

On genuinely flat art — the domain the label map can represent — the thesis holds:

- rust_labels (1.478) beats rust_flat (1.566): consuming the label map does close part of
  the noise-robustness gap, exactly as hypothesised.
- The ceiling (1.133) **beats vtracer (1.270)**, so the integration itself is sound; the
  remaining rust_labels − ceiling gap (1.478 − 1.133 = 0.345) is model label/colour error,
  not engine error.

On gradient/shaded art the label path is structurally wrong. A 16-class map assigns one
class per SVG element; the engine then paints each connected component one flat residue
mean. Where a single element carries a gradient or soft shading, the clean render has many
colours (measured ~16 quantized colours under a 2-class map on the worst images) and the
flat fill cannot represent them — the ceiling itself is 1.9 on this subset. The RGB
quantizer, by contrast, splits a gradient into several colour bands, approximating it. So
trusting the coarse label map *removes* the quantizer's one advantage on gradients. 14 of
the 24 bench images are gradient/shaded, which is why the aggregate rust_labels loses.

## Limitations

- The win is conditional on flat-representable art. This bench is 58% gradient/shaded, so
  the aggregate does not reflect the flat-art result.
- The ceiling is bounded by the label map's granularity (16 classes, one class per SVG
  element), not by the engine. `reconstruction_mae` is the hard floor for a given image.
- `rust_labels_gt` uses the dataset's ground-truth `labels.png`; its indices come from the
  source SVG id map, which is not a perfect partition of the anti-aliased render (a second,
  smaller source of ceiling error alongside gradients).
- Fill source is the model's cleaned RGB; a blurrier RGB head raises the residue-mean error
  on thin regions independently of label quality.

## Next steps

1. **Gate on flat-representability.** Route to label mode only when the image is flat
   (e.g. estimate per-region colour variance from the RGB; fall back to the RGB quantizer
   when a region's interior variance is high). This keeps the flat-art win without the
   gradient-art loss and should move the aggregate below rust_flat.
2. **Per-region gradient fills.** Let a label region emit a linear/radial gradient fill
   instead of a flat mean when its interior variance warrants it. Lifts the ceiling on the
   14 gradient images, which is where all the aggregate loss lives.
3. **Hybrid structure.** Use the label map to seed regions but allow the RGB residue to
   re-split a component when it straddles two colours (label error or gradient), recovering
   the quantizer's banding only where the label map is too coarse.
4. **Report the flat-art slice as the headline** when evaluating the thesis; add a
   recon_mae column to the bench summary so the two regimes are never averaged blind.
