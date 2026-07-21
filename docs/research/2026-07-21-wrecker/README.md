# Wrecker v2 — The Mint Prescription

Date: 2026-07-21. Reader: project owner deciding this week's mint recipe.
Component: `src/vecml/degrade/wreck.py` (ops) + `pipeline.py` (orchestration).
Framing (owner's, correct): "We are training a model to reverse our very specific
wrecker. If our wrecker isn't right, nothing is right." The degradation model **is**
the task definition, so it is specified here against a measured target, not authored
by taste.

Source theme reports in this folder, read in full:
[`taxonomy.md`](./taxonomy.md) (what damage walks in),
[`sota.md`](./sota.md) (how the best pipelines are built),
[`calibration.md`](./calibration.md) (measuring against real files),
[`realism-diversity.md`](./realism-diversity.md) (how exact to be).
Interleaves with the prior round's [`../2026-07-20-deep-research/README.md`](../2026-07-20-deep-research/README.md)
Spend 2 (degradation-realism upgrade), which this document supersedes and details.

Current wrecker in one line: seven photometric ops (`jpeg_cycle`,
`downscale_upscale`, `gaussian_blur`, `gaussian_noise`, `posterize`,
`dither_palette_crush`, `unsharp_halo`), 1–4 drawn independently per sample from
one of three narrow per-run severity tiers, JPEG pinned last, first-order (never
repeated), one shared `labels.png` across all variants. No spatial, scan, halftone,
paper, lighting, or sinc op anywhere.

---

## 1. Verdict per theme

**A — Taxonomy (what damage walks in).** The current wrecker is a competent
*photometric-loss* generator; every gap is *structural or spatial*, and the gaps
line up exactly with the two families the model under-serves (fineline, typography).
Four named target families are unmodelled: the upsampled small web logo (its 8×8 DCT
block grid must be baked *at the small size then magnified*, which the atomic
`downscale_upscale` provably never produces); the scanned card/letterhead (halftone
+ descreen + paper + skew, none present); the multi-generation JPEG chain (real files
are 3–4 shifted-grid generations, the wrecker caps at 2 same-grid passes); and the
phone-photographed card (perspective, illumination, shadow — the wrecker has zero
geometric ops). Chroma-edge fringing (4:2:0) and sinc ringing are plausible direct
causes of the fineline weakness and are neither isolated nor present. Prevalence
percentages were unreachable this round; the family weights must come from tagging
the 400 real pairs.

**B — SOTA.** Every surveyed pipeline (Real-ESRGAN, BSRGAN, APISR, AnimeSR, Augraphy)
diverges from the current wrecker in the same directions, verified against primary
sources: high-order (repeated) degradation is universal and is the single biggest
structural gap; a sinc/Gibbs-ringing op distinct from `unsharp_halo`'s amplitude
overshoot is standard and its ablation names "text and lines" — our exact regime; a
mildly sharpened L1 target biases toward crisp edges with no new op and no
architecture change (only the L1-target half transfers, this being a non-GAN model);
order randomization, print/document ops, and beyond-JPEG codecs (WebP/AVIF) round out
the menu. Keep `posterize` and `dither_palette_crush` — they model indexed-PNG/GIF web
logos that photo/anime pipelines lack, a genuine strength. Photo-calibrated parameter
*ranges* do not transfer; the *structure* does, and ranges must be re-fit to print.

**C — Calibration.** The 400 harvested damaged/clean pairs make measurement feasible,
not hypothetical. JPEG quality (`%Q` from quant tables, exact for standard IJG tables),
generation count (pyIFD ADQ/NADQ), noise sigma (`skimage.estimate_sigma`), and resample
periodicity (Popescu–Farid EM) are all off-the-shelf, mostly training-free, and
content-independent for the header-parse cases — they convert each wrecker parameter
from a guess into a measured distribution. The sim-to-real gap gets a single north-star
number: C2ST / proxy-A-distance on *content-controlled residuals* (degrade the same
clean sources whose real counterparts we hold), with a mandatory positive control and
KID (never raw FID) as cross-check. The one firm correction: **KernelGAN is a blur
probe, not a blur oracle** here — it returns one confounded kernel per photo and cannot
separate a PSF from JPEG blocking / halftone / upsample staircase, which is our exact
composite damage; adopt the anisotropy gap it flags, source kernels analytically.

**D — Realism vs diversity.** The literature rejects the binary. The settled position,
directly implementable: **calibrate the distribution's anchor and bounds to real data,
randomize structurally within those bounds, and tune the free parameters against
downstream restoration deltaE on real pairs — never against visual realism.** Evidence:
a diversity floor exists and is counted in distinct parameterizations, not op types
(Tobin); structured/correlated sampling beats uniform and specifically rescues
small/thin features — our failure mode (Prakash SDR); the most task-useful simulator is
not the most realistic (Ruiz); mixed-continuous severity beats fixed tiers (DnCNN-B,
FFDNet); pure AWGN under-covers real noise by 5.63 dB (CBDNet). Broad randomization not
anchored to the real intake wastes the 4.3M UNet's capacity on off-distribution mush
(e.g. QF=8 double-compression the intake may never contain); measure before going brutal.

---

## 2. The v2 pipeline spec

### 2.1 Architecture change: families replace independent op draws

v1 draws ops independently and uniformly, then pins JPEG last. v2 draws a **capture-path
family** first (structured domain randomization, Theme D §2.2), then runs that family's
*correlated, ordered* op bundle at a **per-sample continuous global severity**
`s ∈ [0, 1]` (Theme D §2.4, replacing per-run tiers). Each family is a named recipe whose
ops and parameter ranges co-vary the way a real capture path does (a scan carries
halftone + paper + skew + scanner-noise *together*, not one at a time).

Sampling, per training sample:

1. Draw family `F ~ Categorical(w)`, where `w` are the **family weights** — the single
   highest-value calibration output (§3, set by tagging the 400 pairs). Ship v2 with the
   provisional prior below; overwrite from the corpus tag before the real mint.
2. Draw global severity `s ~ mixture`: with prob 0.12 `s = 0` (identity, the "don't fix
   what isn't broken" lesson, now the lower tail of a smooth distribution rather than a
   discrete branch — Theme D §2.4/Rec 6); otherwise `s ~ Beta(2, 3)` over `[0, 1]` (mass
   in the low-mid, thin brutal tail). `s` scales each op's range endpoints within `F`.
3. Run `F`'s ordered ops; each op resolves its concrete parameters from its range and `s`,
   and **logs the resolved values** (§2.5).

Provisional family weights (PLACEHOLDER — overwrite from the corpus tag, Theme A/C/D
open questions all converge on this being unmeasured):

| Family | Provisional weight | Print-shop taxonomy target |
|---|---|---|
| `jpeg_chain` | 0.24 | emailed / WhatsApped / screenshotted / re-saved logos |
| `web_upscale` | 0.20 | upsampled favicons / website header images |
| `scan` | 0.18 | scanned letterheads, business cards |
| `office_roundtrip` | 0.14 | logos placed in and re-extracted from Word/PowerPoint |
| `phone_photo` | 0.12 | business card photographed on a desk |
| `lowres_pdf` | 0.08 | PDFs flattened at low resolution |
| `mild` | 0.04 | lightly-touched / near-clean intake |

`identity` is not a family; it is `s = 0` under any family (all ops become
near-no-ops), keeping the clean-passthrough lesson smooth.

### 2.2 New ops to add to the registry

Keep all seven v1 ops (they are reused inside families). Add, each with the same
`op(rgb, rng, severity) -> rgb` signature except the geometric op (§2.4):

| New op | Mechanism | Primary source |
|---|---|---|
| `jpeg_chain` | n-pass JPEG, per-pass independent quality, **shifted 8×8 grid** between passes (crop 0–7px, re-pad), forced subsampling choice, optional per-pass small rescale | Real-ESRGAN 2nd-order; NADQ misaligned re-save (Theme A §2.3, C §2.1) |
| `web_upscale` | downscale to small px → low-Q JPEG **at small size** → nearest/bicubic enlarge to canvas → optional light JPEG | block-magnification (Theme A §2.2, decision-critical) |
| `halftone` | descreen-style: downscale → optional binarise → rotated Gaussian dot screen at LPI + angle → upscale → low-pass descreen | Augraphy Faxify (Theme A §2.1) |
| `paper` | non-white tint + fibre texture as the composite background, replacing flat `_sample_bg` fill for scan/photo families | Augraphy PaperFactory / ocrodeg (Theme A §2.6) |
| `bleed_through` | flip L-R, blur, offset, alpha-blend under front | Augraphy BleedThrough (Theme A §2.6) |
| `ink_bleed` | Kanungo edge-flip `p = α₀·exp(-α·d²)+η` + morphological closing (glyph erosion/thickening) | Kanungo / DocCreator (Theme A §2.6) |
| `sinc_ringing` | 2D sinc / bandlimit ripple (distinct from `unsharp_halo` overshoot) | Real-ESRGAN final sinc, ablation names "text and lines" (Theme B §2.2) |
| `geometric_warp` | skew / rotate / mild perspective — **also transforms the label map** (§2.4) | ocrodeg / Augraphy Geometric (Theme A §2.5) |
| `illumination` | additive/multiplicative low-frequency gradient across the frame | Document mosaicing (synthesised, not removed) (Theme A §2.5) |
| `cast_shadow` | random blurred polygon blended at variable opacity | Augraphy ShadowCast (Theme A §2.5) |
| `poisson_noise` | signal-dependent shot noise; gray/luma-only variant with prob | CBDNet 5.63 dB AWGN gap; Real-ESRGAN (Theme B §2.6, D §2.6) |
| `aniso_motion_blur` | independent `sigmaX`/`sigmaY` + angle, plus a motion-line kernel and optional defocus disc | Real-ESRGAN kernel bank, analytic (Theme B §2.4, C §2.1) |
| `nearest_decimate` | pure integer-stride nearest downsample (aliasing, *no* periodic trace) then enlarge | BSRGAN striding path (Theme A §2.7) |

Also: extend `jpeg_cycle` (and `jpeg_chain`) with a **codec choice** — JPEG default,
WebP with prob (Pillow `format="WEBP"`), AVIF/HEIF optional low prob — matching modern
WhatsApp/web exports (Theme B §2.7). And **force the subsampling factor** (2 = 4:2:0 vs
0 = 4:4:4) rather than leaving it to PIL defaults, so chroma-edge fringing is
deterministic (Theme A §2.4).

### 2.3 Family recipes (ordered ops + parameter distributions)

All ranges below are *starting anchors* re-fit for flat art, to be tuned by §3. `U`
= uniform, probabilities are per-op fire chance within the family. `s` = global severity.
Ranges written `[lo→hi]` interpolate lo (at `s=0`) to hi (at `s=1`).

**`jpeg_chain`** (multi-generation email/WhatsApp/screenshot):
1. optional mild `aniso_motion_blur`, p=0.3, sigma `[0.2→1.2]`
2. optional small rescale `[0.7→1.0]` then back, p=0.4 (screenshot downscale)
3. **JPEG loop, n passes** `n ~ {2,3,4}` weighted by `s` (2 at low `s`, 4 at high):
   per pass quality `~ U[floor→95]` (**floor calibrated in §3, do NOT hardcode 8** —
   Theme D §2.6), subsampling 4:2:0 with p=0.7 else 4:4:4, codec {JPEG 0.8, WebP 0.2},
   **8×8 grid shift** of 0–7 px between passes
4. final 8-bit round/clamp
Rationale: this is the second-order chain specialised to re-encode history; grid shift
+ varied quality is what a same-grid 2-pass cannot reproduce (Theme A §2.3, B §2.1).

**`web_upscale`** (favicon/header enlarged past native detail):
1. downscale to small size `S ~ U[16→96] px` (**calibrate to measured factors**, §3)
2. low-Q JPEG **at small size**, quality `~ U[floor→80]` (bakes small blocks)
3. optional `posterize`/`dither_palette_crush`, p=0.4 (indexed web source)
4. enlarge to 256 px, interpolation `{nearest 0.4, bicubic 0.4, bilinear 0.2}`
   (nearest → staircase; bicubic → magnified blocks + overshoot)
5. optional light `jpeg_cycle` at canvas, p=0.5, quality `[70→92]`
Rationale: only this order magnifies the 8×8 grid into ~scale×8 px blocks; the atomic
`downscale_upscale` provably cannot (Theme A §2.2). This is the named dominant intake.

**`scan`** (letterhead / business card scanned):
1. `paper` — warm/grey tint + fibre texture as the background composite
2. optional `bleed_through`, p=0.3
3. `ink_bleed`, p=0.6 (glyph erosion/thickening — the typography killer)
4. `halftone` — LPI `~ U[65→185]`, CMYK screen angles ~30° apart, scanner-DPI resample,
   partial low-pass descreen, p=0.7
5. `geometric_warp` — skew/rotate `± U[0→4]°`, mild perspective (**matched label warp**)
6. `illumination` gradient, p=0.5
7. `gaussian_noise` + `poisson_noise` (scanner sensor), low std
8. `jpeg_cycle`, quality `[60→92]` (scans often saved once, moderate Q)
Rationale: correlated scan cluster (SDR). Gated by corpus prevalence — build fully but
weight to measured share (Theme A §2.1/2.6, C §2.4).

**`office_roundtrip`** (logo through Word/PowerPoint):
1. optional mild downscale/upscale (placement scaling), p=0.5
2. `posterize` `[7→4]` bits and/or `dither_palette_crush`, p=0.6 (WMF/palette crush)
3. codec re-quantise: PNG round-trip or WebP, p=0.5
4. `jpeg_cycle`, quality `[70→92]`, p=0.6
Rationale: Office re-export is palette/codec churn, not heavy spatial loss.

**`phone_photo`** (business card on a desk):
1. `paper` background
2. `geometric_warp` — perspective (**matched label warp**), stronger than scan skew
3. `illumination` gradient + `cast_shadow`, p=0.6
4. `aniso_motion_blur` / defocus, sigma `[0.3→2.5]`
5. `poisson_noise` (sensor), gray-noise variant p=0.4
6. `jpeg_cycle`, quality `[55→90]`, subsampling 4:2:0
Rationale: camera capture cluster; the only family needing full lighting + perspective.

**`lowres_pdf`** (flattened at low resolution):
1. `downscale_upscale` (v1 op), factor `[0.6→0.2]`
2. `gaussian_blur` `[0.4→2.5]`
3. optional `sinc_ringing`, p=0.5
4. `jpeg_cycle`, quality `[50→85]`
Rationale: uniform low-res flatten; the closest family to the v1 recipe.

**`mild`** (lightly-touched intake): one or two ops at low `s`, JPEG `[80→95]`, no
geometric/scan ops. Fills the low-damage region above pure identity.

**Cross-family finishing step (all families except `identity`):** with p=0.6 apply a
`sinc_ringing` pass, order-swapped with the family's final JPEG (Real-ESRGAN
`final_sinc_prob`-style, Theme B §2.2) — the most direct structural lever for the
fineline/typography under-service, applied globally because ringing concentrates at the
hard edges every family produces.

### 2.4 Geometric ops and the label-map invariant (decision required)

`pipeline.py` today writes **one shared `labels.png`** for all variants because every
v1 op is a per-pixel photometric map — the ground-truth geometry never moves. Any
geometric op breaks this: a skewed input needs a label map warped by the *same*
transform, or the supervision is wrong.

Two options; **v2 default = (A)**:

- **(A) Matched warp, per-variant label (recommended default).** Apply the identical
  affine/homography to both the wrecked input and a copy of the label map; write a
  per-variant `labels_XX.png` for geometric variants (non-geometric variants keep the
  shared `labels.png`). Preserves the task definition — the model reproduces the logo as
  the customer sent it. Keep warp magnitudes **modest** (skew ≤4°, gentle perspective)
  so QC reconstruction still holds.
- **(B) Input-only warp → model learns to deskew.** Arguably the *desired* product
  behaviour for a print shop (output the clean straight vector), but it redefines the
  task and the Stage-2 tracer must consume deskewed geometry. Do not ship silently; it
  is an explicit A/B, not a default.

Implication for the mint: geometric families require the per-variant-label plumbing that
pixel-only ops never needed. This is the one non-trivial engineering change in v2.

### 2.5 Per-sample parameter logging (required)

v1 `meta.json` already logs `variants[i].recipe = [{op, severity}]` — the hook exists.
v2 **extends each variant record** so calibration can compare what the wrecker actually
applied against what the forensic profiler measures on real files (§3), and so a DASR
discriminator can be trained per-sample:

```
variants[i] = {
  "file": "wrecked_00.png",
  "seed": <int>,
  "recipe_version": "wreck-v2",     # rollout stamp, §5
  "family": "jpeg_chain",           # the capture-path drawn
  "global_severity": 0.41,          # the per-sample s
  "label_file": "labels.png",       # or "labels_00.png" if geometric-warped
  "ops": [
    {"op": "jpeg_chain", "severity": 0.41,
     "params": {"passes": 3, "qualities": [78, 64, 71],
                "subsampling": [2, 2, 0], "codecs": ["JPEG","WEBP","JPEG"],
                "grid_shifts": [3, 5]}},
    {"op": "sinc_ringing", "severity": 0.41,
     "params": {"cutoff": 1.9, "order_swapped": true}}
  ]
}
```

Every op resolves and records its concrete draws (JPEG qualities, subsampling, sigmaX/
sigmaY/angle, warp matrix, LPI + screen angle, resample factor, kernel type). This log is
the join key for calibration and the answer key for the geometric label warp.

---

## 3. Calibration protocol

The 400 harvested damaged-original / clean-PRINT pairs are the differentiating asset
(Theme D §2.7). Protocol, cheapest first:

**Split first.** Partition the 400 pairs into a **calibration/tune set (~300)** and a
**frozen validation set (~100)**, disjoint, before any tuning — or DE-GAN's
in-distribution-inflation trap reappears in our own metrics (Theme C §2.2, D open q 5).

**Step 0 — Tag the corpus (highest value, ~2–3 hrs, no GPU).** Manually label each of
the 400 pairs by capture-path family (`jpeg_chain` / `web_upscale` / `scan` /
`office_roundtrip` / `phone_photo` / `lowres_pdf` / `mild`). This sets `w`, the family
weights in §2.1, and answers whether scans are a meaningful share (whether the `scan`
family earns its build cost). Every theme's top open question reduces to this tag.

**Step 1 — Per-file forensic profiler (~1 day, no GPU).** Over the corpus, run:
- JPEG quality: Pillow `.quantization` → IJG `%Q` (exact for standard tables; store raw
  64-entry tables for Photoshop/Office non-standard cases). → sets `jpeg_chain`/all-JPEG
  quality histograms and **the floor** (settles whether QF<30 / Q8 is on-distribution —
  Theme D §2.6, do not keep the hardcoded 8 without this).
- Generation count: pyIFD ADQ/NADQ (aligned + misaligned double-JPEG). → sets the pass-
  count distribution `n` and confirms the misaligned-grid re-save case exists.
- Noise sigma: `skimage.restoration.estimate_sigma`, per-channel, on flat regions. → sets
  additive-noise std. (Over-reads on hard-edged art; use as the additive-leg probe only.)
- Resample factor: Popescu–Farid EM/Fourier periodicity (N=2, σ₀=0.0075). → sets
  `web_upscale`'s small-size distribution and `resize` ranges as a *prior* (factor is not
  uniquely recoverable; model the final resample distinctly).

Replace v1's single-`severity`-drives-everything mapping with **sampling from these
measured marginals**.

**Step 2 — Sim-to-real gap, the north-star number (~1 day, 1 GPU).** Degrade the *same*
clean sources whose real-damaged counterparts we hold (content-controlled), then:
- **C2ST / proxy-A-distance** `2(1−2ε)`: train a small CNN to tell real-damaged from
  v2-damaged on **residuals** (wrecked − clean), 50/50 split, held-out accuracy → 0.5 =
  matched. **Mandatory positive control** (feed an obviously-wrong synthetic set, confirm
  accuracy → 1.0) or a "matched" reading is untrustworthy (under-power failure). Track
  this one number across wreck versions.
- **KID (never raw FID)** under one frozen resize+JPEG path (clean-fid style) as feature-
  space cross-check; FID's resize/JPEG sensitivity, ImageNet-Inception bias on line-art,
  and small-sample inflation all hit our exact case. Consider the vectorizer's own encoder
  over ImageNet Inception.
- **High-frequency spectral** cross-check (high-pass residual power spectra) as a cheap,
  content-robust confirm.

**Step 3 — End-to-end train-synthetic / test-real deltaE (the task objective).** Train
Stage-1 on v2 output, then measure bench deltaE on (i) held-out synthetic vs (ii) the
frozen 100 real pairs. The gap between them is the calibration objective — shrinking it
is what "the wrecker is right" means operationally (Theme C §2.3, D §2.3). Score the
unaligned raster-damaged / vector-clean pairs with a **CoBi / contextual loss** (tolerates
40–80 px shift), **not** pixel L1 and **not** a chased affine registration (Theme C §2.3).

**Step 4 — Close the loop (task-optimal, not realism).** Tune family weights `w` and the
free parameter ranges to **minimize real-pair bench deltaE**, not to maximise visual
realism or synthetic-val metric (Ruiz; the proxy trap — Theme D §2.3). Optional, after
cheap fixes land: a **DASR-style patch discriminator** (reuse the Step-2 C2ST classifier)
to down-weight synthetic samples that still read as unreal.

Sim-to-real gap is thus measured three ways, in rising cost and rising trust: C2ST/proxy-A
(distribution), KID + spectral (features), end-to-end real-pair deltaE (task). The last is
the one that decides the recipe.

---

## 4. Realism-vs-diversity policy

**Policy: calibrate the anchor and bounds to real data; randomize structurally within
those bounds; choose every free parameter by downstream real-pair deltaE, never by visual
realism.** Concretely in v2 this means: family weights and per-op range *endpoints* come
from §3 (calibrated); *within* a family, ops fire probabilistically and severity is
continuous per sample (randomized structurally); and §3 Step 4 is the only arbiter of the
free knobs.

Evidence, load-bearing:
- **Diversity floor, counted in parameterizations not ops** — Tobin et al.: sim-to-real
  degrades below a threshold of distinct random parameterizations. v1's 3 narrow per-run
  tiers plausibly sit below the floor. v2's continuous severity + per-op ranges widen the
  distinct-setting count without adding op types for their own sake.
- **Structure beats uniform, rescues thin features** — Prakash SDR: context-aware
  correlated randomization beat uniform DR and specifically rescued small objects (our
  fineline/typography failure). This is why v2 uses correlated capture-path families, not
  independent op draws. (Magnitude is photo-domain; must be re-measured on the 400 pairs.)
- **Task-useful ≠ realistic** — Ruiz Learning to Simulate: tuning the simulator to
  downstream accuracy beats hand-tuned realism. The 400 pairs are the *tuning objective*,
  not a realism eyeball.
- **Mixed-continuous severity beats fixed tiers** — DnCNN-B, FFDNet: one blind model over
  a continuous range matches or beats per-level specialists. Hence §2.1's per-sample `s`,
  with identity as the smooth lower tail.
- **Bounds matter, broad ≠ better** — Real-ESRGAN/BSRGAN floor JPEG at QF∈[30,95]; v1
  drives Q to 8. Unless the corpus shows QF<30 (Step 1), Q8 double-compression teaches
  hallucination from off-distribution mush and wastes the 4.3M UNet's capacity.
- **Coverage gaps are real, not excess breadth** — CBDNet: pure AWGN fails real images by
  5.63 dB, fixed by Poisson + ISP noise. SRMD/DE-GAN: >10 dB / F-score collapse under
  degradation mismatch, confirming operation coverage *is* the task definition.

The synthesis: exactness governs the *bounds and anchor* (calibrate), breadth governs
*within* the bounds (structured randomize), and neither is chosen by realism — both by
real-pair deltaE.

---

## 5. Rollout: versioned recipes, comparability, cost

**Comparability is preserved by versioning the recipe, not mutating it in place.** Runs
minted with v1 must stay comparable to v2 runs, so:

1. **Freeze v1.** Keep the current seven ops and `sample_recipe`/`apply_recipe` importable
   and unchanged as `recipe_version="wreck-v1"`. Do not edit them; v2 is additive.
2. **Stamp every sample.** `meta.json` gains `recipe_version` (§2.5). The bench and any
   training-corpus manifest record which version minted each shard. A model's scorecard
   names its wrecker version; a v1-minted run and a v2-minted run are never compared
   without that tag visible.
3. **Mint v2 as a new shard**, not an overwrite of the existing corpus. This lets the
   first v2 experiment be a clean A/B: same clean sources, v1 vs v2 damage, same Stage-1
   config, compared on the frozen 100 real pairs (Step 3). It also recovers the prior
   round's 500k divergence point cleanly (prior README Spend 2).
4. **Gate v2 behind a flag** on `wreck_svg` (e.g. `recipe="v2"` selecting the family
   dispatcher; default stays `v1` until the A/B clears). No silent behaviour change to
   in-flight mints.
5. **Reproducibility.** v2 keeps v1's deterministic-from-seed property: family draw,
   severity, and every op parameter derive from the per-variant seed, so any sample
   regenerates exactly. Note the RNG-seeding gap (Theme B §2.10): any new cv2 op that
   draws randomness needs `cv2.setRNGSeed` alongside the numpy generator, which v1 never
   required.

**Sequencing.** Do §3 Step 0 (tag, ~half day) *before* minting — it sets the family
weights the whole recipe hangs on. Then A/B one v2 shard vs v1 on the frozen real pairs.
This is the prior round's Spend 2 (~$20–40 GPU, 1–2 eng-days), now detailed; it remains the
highest-probability mover of the shipped (vtracer) bench because the 1.52→1.28 headroom is
a cleanup-quality gap, not a tracer gap.

**Cost.** Second-order chains + geometric warps roughly **1.5–2× the per-sample wreck CPU
cost** vs v1 (the full chain runs 2–4× for `jpeg_chain`, geometric adds an affine on input
+ label). At the current pre-baked `n_variants=4`, wreck is CPU-side and cheap relative to
the ~$15–30 GPU training run, so the increase is immaterial to total mint cost — but if it
throttles bake throughput, gate second-order behind a probability rather than always-on
(Theme A open q). One extra GPU training run for the A/B is the real spend.

---

## 6. What NOT to build

- **Do NOT wire KernelGAN in as a blur source.** It returns one confounded per-image
  kernel, is untested on flat art, and contradicts forward-degradation practice; if run at
  all, treat its output as a candidate anisotropy prior validated on the cleanest
  near-single-resample subset only. Source anisotropic/motion/defocus kernels analytically.
  (Theme C §2.1)
- **Do NOT chase pixel registration** of the raster-damaged / vector-clean pairs. Use a
  CoBi/contextual loss that tolerates 40–80 px shift; published rigs still need manual patch
  rejection and report no residual metric. (Theme C §2.3)
- **Do NOT trust raw FID** for the sim-to-real number. Its resize/JPEG sensitivity,
  ImageNet-Inception bias on line-art, and small-sample inflation all hit this exact case;
  use KID under a frozen path. (Theme C §2.2)
- **Do NOT hardcode JPEG quality to 8** (or any brutal floor) before Step 1 confirms QF<30
  exists in the intake. Off-distribution mush teaches hallucination. (Theme D §2.6)
- **Do NOT optimize the wrecker for visual realism.** The tuning objective is downstream
  real-pair deltaE. A realistic-looking wreck that doesn't improve restoration is a waste.
  (Theme D §2.3)
- **Do NOT over-warp geometry.** Beyond modest skew/perspective it changes the task
  (deskew) and breaks QC reconstruction; keep magnitudes bounded until the label-warp
  decision (§2.4) is made. (Theme A open q)
- **Do NOT build auto-curriculum / adversarial-hard-sampling machinery** for v2. Supervised
  256px restoration is trained wide-and-fixed (how Real-ESRGAN/BSRGAN succeed); auto-
  expansion is RL-regime-bounded. Treat it as a later A/B only if fixed-wide plateaus.
  (Theme D §2.5)
- **Do NOT build AnimeSR-style learned degradation operators yet.** A heavier build than the
  op-level and calibration changes; defer until the cheap fixes and calibration land. (Theme
  B §2.9, C §2.4)
- **Do NOT move to on-the-fly GPU degradation** unless the pre-baked `n_variants=4` diversity
  cap shows up as a train/val gap; bake-time reproducibility is worth more until then. (Theme
  B §2.10)
- **Do NOT remove `posterize` or `dither_palette_crush`.** They model indexed-PNG/GIF web
  logos the photo/anime pipelines lack — a genuine strength. (Theme B §1)
- **Carry-over from the prior round:** do not grow the UNet, do not build multi-task
  optimizers, do not add exotic boundary losses, do not add gradient/3D content.

---

## 7. Open questions

1. **Corpus composition (blocks the family weights).** What fraction of the 400 pairs are
   each capture-path family? Answered by §3 Step 0; sets `w` and whether `scan`/`phone_photo`
   earn their build. Every theme's top open question reduces to this. (Cheap: ~half day tag.)
2. **Does the intake contain QF<30 material?** Decides whether the brutal JPEG tail is on-
   or off-distribution and how low the quality floor goes. Answerable now from Step 1. (Theme
   D open q 1)
3. **Empirical upsampling-factor distribution.** Does the shop's junk cluster at specific
   factors (favicon 16/32/64 → banner)? Sets `web_upscale`'s small-size range via Farid EM.
   (Theme A open q)
4. **Label-warp decision (§2.4): matched warp (reproduce) vs input-only (deskew)?** Matched
   is the safe default; deskew may be the better *product*. Needs an explicit call before
   geometric families mint, and it interacts with the Stage-2 tracer. (Theme A/B open q)
5. **Does 4:2:0 chroma fringing actually explain the fineline weakness,** or is it dominated
   by blur + block magnification? Only an ablation on the fineline/type deltaE sub-metric
   separates them; the chroma hypothesis is source-supported, not confirmed as cause. (Theme
   A §2.4)
6. **Halftone label integrity.** Rescreen + descreen changes colours; does the white-paper
   assumption in `_derive_ground_truth`/`audit_sample` still hold, or do halftone/scan
   samples need a QC exemption? (Theme A open q)
7. **Marginal vs joint calibration.** Profiling gives per-op marginals; real files have
   correlated histories (a WhatsApp chain co-occurs with specific quality + generation
   counts). Does sampling marginals independently under-cover, and is the DASR discriminator
   enough to catch it, or must families encode the joint? (Theme C open q)
8. **Which encoder feature space for KID/C2ST?** ImageNet Inception is a poor fit for
   logos/line-art; is the vectorizer's own encoder available and better? Small ablation.
   (Theme C open q)
9. **Rasterization DPI/filter for the vector-clean reference.** The clean side is a vector
   PDF; rasterizing it introduces its own resample, which affects every paired metric. What
   DPI/filter makes it a fair target? (Theme C open q)
10. **Second-order bake cost.** If 1.5–2× wreck CPU throttles mint throughput, gate second-
    order behind a probability rather than always-on. Only matters if bake becomes the
    bottleneck. (Theme A open q)
