# THEME B — State of the art in synthetic degradation pipelines

Date: 2026-07-21
Scope: how the best synthetic degradation pipelines are built (operations, orders, parameter
distributions) and the measured evidence for what each component buys — written as deltas
against the current wrecker (`src/vecml/degrade/wreck.py`, `pipeline.py`).

---

## 1. Executive summary

The current wrecker is a **single-order** pipeline: `sample_recipe` draws 1–4 ops from a flat
pool (`jpeg_cycle`, `downscale_upscale`, `gaussian_blur`, `gaussian_noise`, `posterize`,
`dither_palette_crush`, `unsharp_halo`), applies each at most once, and usually places JPEG last.
Every surveyed SOTA pipeline diverges from this in the same directions, and the divergences map
cleanly onto the shop's stated failure mode (fineline/typography) and target distribution
(upsampled web logos, multi-generation JPEG chains, scans, phone photos, Office re-exports).

The convergent findings, ranked by decision value:

1. **High-order (repeated) degradation.** Real-ESRGAN, BSRGAN, APISR, AnimeSR all run the full
   blur→resize→noise→compress set **twice** (second stage gated by a skip probability). This is
   the single biggest structural gap and is exactly what models multi-generation JPEG/WhatsApp/
   screenshot chains. The wrecker's `jpeg_cycle passes=2` is a narrow special case of this.
2. **Sinc / bandlimit ringing.** Real-ESRGAN's ablation states that omitting sinc filters leaves
   restored output with ringing/overshoot "especially around the text and lines." The wrecker has
   `unsharp_halo` (amplitude overshoot) but **no sinc/Gibbs-ringing op** — a physically distinct
   artefact that is the dominant edge failure in JPEG'd and downsampled logos.
3. **Sharpened supervision target.** Real-ESRGAN trains L1/perceptual against a USM-sharpened GT;
   APISR line-enhances the GT (unsharp + XDoG + dilation). Both deliberately bias the model toward
   crisper edges. The wrecker's target is the plain clean render. This is a target-side lever for
   the typography gap requiring no new degradation op.
4. **Order randomization.** BSRGAN shuffles ~7 ops per sample so the model never overfits one
   canonical order; the wrecker fixes structural-first / JPEG-last.
5. **Print-specific / document artefacts** (Augraphy, Genalog, DocCreator): ink-vs-paper layer
   separation, halftone/fax, scanner drum streaks, bleed-through, **and any geometric degradation
   at all** (skew, rotation, fold, perspective). The wrecker has zero geometric ops — a direct
   mismatch with scans and phone-photographed cards.
6. **Beyond-JPEG compression** (APISR): WebP/AVIF/HEIF and single-frame video codecs, matching
   modern web/WhatsApp exports that are increasingly not JPEG.
7. **Learned-from-real degradation** (AnimeSR) and **calibration against real pairs** (TextZoom:
   real paired training beats synthetic bicubic by up to **+8.6 pts** text-recognition accuracy).
   This is the strongest argument for calibrating the wrecker against the ~400 harvested
   damaged/clean pairs rather than trusting a hand-authored recipe.

Two current ops — `posterize` and `dither_palette_crush` — are a **genuine strength** that the
photo/anime pipelines lack; they model indexed-PNG/GIF web logos. Keep them.

All decision-critical Real-ESRGAN claims below were independently verified against primary
sources (config YAML, BasicSR model code, paper). Verdicts are noted inline.

---

## 2. Findings by sub-topic

### 2.1 High-order (repeated) degradation — the core structural gap

Real-ESRGAN adopts a **second-order** model: two repeated classical passes, each
`blur → resize → noise → JPEG`, with independently sampled hyperparameters. The paper's rationale:
"an n-order model involves n repeated degradation processes … Empirically, we adopt a second-order
degradation process for a good balance between simplicity and effectiveness," and the first-order
ablation "cannot effectively remove noise on the wall or blur in the wheat field"
([Real-ESRGAN paper](https://ar5iv.labs.arxiv.org/html/2107.10833)). **Verified CONFIRMED**
against the primary source. Caveat from verification: the paper's evidence is photographic
(camera noise, wall texture); adopt the *structure* (≥2 repeated chains, because real files are
multi-generation) but **re-parameterize for flat art** rather than copying photo hyperparameters.

BSRGAN reaches the same place differently: it **randomly shuffles** ~7 ops per sample
(`shuffle_order = random.sample(range(7), 7)`), constraining only that the final downsample stays
last, so the model never sees a fixed order
([BSRGAN code](https://raw.githubusercontent.com/cszn/BSRGAN/main/utils/utils_blindsr.py)).
APISR and AnimeSR both use two-stage degradation as well
([APISR](https://ar5iv.labs.arxiv.org/abs/2403.01598),
[AnimeSR](https://ar5iv.labs.arxiv.org/abs/2206.07038)).

Delta vs wrecker: `sample_recipe` picks n=1–4 ops **once**. `jpeg_cycle` does `passes=2` only when
`severity > 0.66` — a degenerate single-op special case of second-order. To model the real
multi-generation chain the whole set (resize + recompress + optional blur/noise) must repeat, ideally
with a shifting JPEG **block-grid offset between passes** (the raw notes flag that real re-saves
recompress at shifting 8×8 grids and accumulate 4:2:0 chroma bleed, which a same-grid re-encode
does not reproduce).

### 2.2 Sinc / bandlimit ringing

Real-ESRGAN synthesizes ringing/overshoot with a **sinc filter in two places**: as a blur-kernel
option per pass (`sinc_prob: 0.1`, `sinc_prob2: 0.1`) and as a near-mandatory final step
(`final_sinc_prob: 0.8`), where the final sinc and the last JPEG are applied in randomly swapped
order (`if np.random.uniform() < 0.5`)
([config](https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/options/train_realesrgan_x4plus.yml),
[model code](https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/models/realesrgan_model.py)).
**Verified CONFIRMED** (config values matched verbatim; two-place application and random swap
confirmed in code).

The ablation is the payoff: without sinc, "the restored results will amplify the ringing and
overshoot artifacts … especially around the text and lines. In contrast, models trained with sinc
filters can remove those artifacts" ([paper](https://ar5iv.labs.arxiv.org/html/2107.10833)).
**Verified CONFIRMED** (quote verbatim). Verification note: this is one of the rare Real-ESRGAN
components whose relevance *does* transfer out of the photo domain — ringing concentrates at
high-contrast step edges, which in flat art means glyph strokes and thin lines, so the ablation's
own demonstration case (text and lines) is our exact regime.

Correction to a claim in the raw notes: the notes say the wrecker "has no ringing/sinc op (only
unsharp_halo)." Strictly, `unsharp_halo` **is** a ringing op (its docstring: "produce ringing halos
around edges"). The accurate statement is: the wrecker has an **amplitude-overshoot** op but **no
sinc/bandlimit (Gibbs) ringing** op. These are physically distinct — unsharp overshoot comes from a
sharpening kernel; sinc ringing comes from frequency truncation (the oscillating ripples JPEG blocks
and downsampling actually produce). The gap is real; the phrasing needed correcting.

### 2.3 Sharpened / line-enhanced supervision target

Real-ESRGAN trains L1 and perceptual losses against a **USM-sharpened** ground truth
(`l1_gt_usm: True`, `percep_gt_usm: True`) while the GAN loss uses the un-sharpened GT
(`gan_gt_usm: False`)
([config](https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/options/train_realesrgan_x4plus.yml);
mechanism `self.gt_usm = self.usm_sharpener(self.gt)` in
[model code](https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/models/realesrgan_model.py)).
**Verified CONFIRMED**. Verification caveat that matters for us: the `gan_gt_usm=False` split is
specific to a GAN pipeline (unsharpened adversarial target avoids overshoot while L1/percep still
pull toward sharpness). The vectorizer Stage 1 is **L1 + region-CE, no GAN, no perceptual loss**, so
**only the L1-target half transfers**: apply a mild USM to the RGB-L1 target so reconstruction
favors edges crisper than the render. Cheap, well-precedented, no architecture change.

APISR does the target-side fix more aggressively and more on-topic (line art): pseudo-GT prep =
recursive unsharp (n=3) → XDoG sketch extraction → outlier removal → passive dilation → binary-mask
merge of sharpened edges with GT, "teaching the model to output crisp lines rather than reproduce
faint/blurred ones" ([APISR](https://ar5iv.labs.arxiv.org/abs/2403.01598)).

Note (not proven): that this fixes the vectorizer's typography gap is a plausible **extrapolation**,
not established. The gap could equally stem from 256px starving thin strokes, loss weighting, or thin
text coverage in the clean-SVG corpus. Treat sharpened-target as one contributing fix to validate
against real pairs, not a guaranteed cure.

Supporting architectural evidence: DocDiff needs a dedicated high-frequency **residual-diffusion**
refiner on top of a 4.03M coarse predictor (nearly identical size to the 4.3M UNet) because a single
pixel-loss UNet of that size recovers low-frequency content but loses edges/text
([DocDiff](https://ar5iv.labs.arxiv.org/abs/2305.03892)). This is consistent evidence that a lone
~4M L1 UNet structurally under-serves high frequency — corroborating the reported weakness, though
DocDiff's remedy (add a refiner) is a heavier lever than target sharpening.

### 2.4 Richer blur than single isotropic Gaussian

Real-ESRGAN samples **6 kernel types** over a 21px kernel: iso 0.45, aniso 0.25, generalized_iso
0.12, generalized_aniso 0.03, plateau_iso 0.12, plateau_aniso 0.03; sigma [0.2,3] (pass2 [0.2,1.5]),
betag [0.5,4], betap [1,2] — plus the sinc kernel discussed above
([config](https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/options/train_realesrgan_x4plus.yml)).
**Verified CONFIRMED** (every number matched verbatim). Verification caveat: sigma/beta ranges are
photo-calibrated; the *direction* (multi-kernel + anisotropic, not single iso) transfers, the exact
ranges should be re-fit to the print target.

Delta vs wrecker: `gaussian_blur` is a single isotropic Gaussian, sigma 0.4–3.5. Albumentations
provides drop-in alternatives with documented ranges as calibration anchors: `MotionBlur (3,7)`,
`Defocus radius (3,10)`, `GlassBlur`, `ZoomBlur`, and crucially `RingingOvershoot(blur_limit, cutoff)`
which is a ready-made sinc op ([Albumentations blur transforms](https://github.com/albumentations-team/albumentations/blob/main/albumentations/augmentations/blur/transforms.py)).

### 2.5 Resize: genuine upsampling past native detail

Real-ESRGAN resize is a standalone rescale with up/down/keep probabilities `[0.2, 0.7, 0.1]`
(pass1) and `[0.3, 0.4, 0.3]` (pass2), scale [0.15,1.5] / [0.3,1.2], random area/bilinear/bicubic
interpolation. APISR explicitly includes **upscaling to 1.2x** with probability mass on enlargement,
"directly modelling the small-web-logo-enlarged case rather than only shrink-then-restore"
([APISR](https://ar5iv.labs.arxiv.org/abs/2403.01598)).

Delta vs wrecker: `downscale_upscale` **always shrinks then re-enlarges back to the original size** —
it never models a genuinely low-native-resolution original blown up *past* its real detail, which is
the shop's dominant input (favicons/headers upsampled with nearest/bicubic). It does already include
`INTER_NEAREST` in both directions, which is the right interpolation for that case; the gap is the
missing "start smaller than native and stay big" path.

### 2.6 Noise: Gaussian-or-Poisson, gray-noise, quantization

Real-ESRGAN noise is a Gaussian-**or**-Poisson choice (`gaussian_noise_prob: 0.5`), applied as
**gray** (luma-only) noise 40% of the time (`gray_noise_prob: 0.4`), Gaussian sigma [1,30], Poisson
scale [0.05,3]. It also rounds/clamps to 8-bit at the end
(`torch.clamp((out*255).round(),0,255)/255`) as a cheap re-quantization step
([config + model code](https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/options/train_realesrgan_x4plus.yml)).
BSRGAN adds Poisson/speckle and a reverse-ISP camera-noise path (`isp_prob`).

Delta vs wrecker: `gaussian_noise` is additive Gaussian only, always per-channel color, sigma up to
40. Missing: Poisson (signal-dependent, models real sensor/scan noise), gray/luma-only noise, and the
final 8-bit re-quantization. These are minor buys individually.

### 2.7 Compression beyond JPEG

APISR treats **WebP and AVIF/HEIF** as first-class degradations (JPEG q[20,95], WebP q[20,95], HEIF
q[30,100], AVIF q[30,100]) and even single-frame video codecs (H.264 CRF[23,38], H.265 CRF[28,42])
([APISR opt.py](https://raw.githubusercontent.com/Kiteretsu77/APISR/main/opt.py), **decision-critical**).
Modern website/WhatsApp exports are frequently WebP, and newer-OS screenshots can be HEIF/AVIF.

Delta vs wrecker: `jpeg_cycle` models JPEG only. Adding WebP (Pillow supports it natively via
`im.save(buf, format="WEBP", quality=q)`) is nearly free and matches a real and growing slice of the
target distribution. Video codecs are lower priority for still logos.

### 2.8 Print / document-specific degradation (the biggest distribution gap)

Augraphy structures degradation as an **ordered three-phase** pipeline — ink layer → paper layer →
merge + physical post — not a flat op subset. Ink effects hit an extracted ink/toner layer; paper
effects hit a separate paper layer; the two merge, then physical post effects (folds, shadows,
borders, skew) apply last ([Augraphy](https://github.com/sparkfish/augraphy),
[default pipeline](https://raw.githubusercontent.com/sparkfish/augraphy/master/augraphy/default/pipeline.py),
**decision-critical**). Copyable parameter ranges for print-relevant effects the wrecker lacks:

- **BadPhotoCopy** (mask-multiply, not additive): `noise_type ∈ {blobs,gaussian,perlin,worley,rectangular}`, `noise_value (128,196)`, `noise_sparsity (0.3,0.6)`, `noise_concentration (0.1,0.6)`.
- **Faxify** (halftone/fax): `scale_range (0.3,0.6)`, monochrome, halftone toggle, `half_kernel_size ∈ {(1,1),(2,2)}`, `angle (0,360)`, `sigma (1,3)`.
- **DirtyDrum** (scanner streaks): `line_width_range (1,6)`, `line_concentration ~U(0.05,0.15)`, `noise_intensity ~U(0.6,0.95)`.
- **InkBleed**: `intensity_range (0.1,0.2)`, `kernel ∈ {(3,3),(5,5),(7,7)}`, `severity (0.4,0.6)`.
- **BleedThrough** (reverse-page ghost): `intensity_range (0.1,0.3)`, `color_range (32,224)`, `offsets (10,20)`.
- **Geometric** (skew/rotate): `rotate_range (-5,5)`, `scale (0.75,1.25)`, `translation (-10,10)`, flips.
- **Folding**: `fold_count 2–8`, `fold_angle_range (-360,360)`; plus ShadowCast/LightingGradient for uneven illumination.

Each default op fires at **p=0.2**, wrapped in `OneOf` blocks, so each document gets a sparse random
subset at random severity — an alternative composition philosophy to guaranteed second-order chains,
and a closer match to real documents that carry a few artefacts, not all at once.

Delta vs wrecker: the wrecker applies its 7 ops over an **already-composited RGB render**, so it
structurally cannot produce ink-vs-paper artefacts, and it has **no geometric/spatial degradation at
all** — no skew, rotation, perspective, fold, or shadow. This is the most direct mismatch with the
shop's real inputs (skewed scans, phone-photographed business cards, low-res flattened PDFs).

Genalog and DocCreator round out the classical menu: morphological ink erode/dilate with directional
kernels (e.g. `(9,1)`), bleed-through, and 3D paper (mesh) deformation
([Genalog](https://raw.githubusercontent.com/microsoft/genalog/main/genalog/degradation/README.md),
[DocCreator](https://doc-creator.labri.fr/)).

Caution: Augraphy's own validating benchmark (**ShabbyPages**) is itself synthetic — 6,000+
born-digital images degraded *by Augraphy* — so published Augraphy denoiser results do not prove
real-world generalization ([ShabbyPages](https://arxiv.org/abs/2303.09339)). This reinforces §2.9.

### 2.9 Calibration against real pairs beats hand-authored recipes

TextZoom: same architecture trained on **real** degraded-text pairs vs **synthetic bicubic**
downsampling — recognition accuracy improves SRResNet 48.9→51.3 (+2.4), LapSRN 48.7→53.0 (+4.3),
**TSRN 49.7→58.3 (+8.6)** ([TextZoom](https://ar5iv.labs.arxiv.org/abs/2005.03341),
**decision-critical**). It also adds a **Gradient Profile Loss** that sharpens character boundaries
instead of smoothing them.

DE-GAN quantifies the "if the wrecker is wrong, nothing is right" failure: near-perfect in-
distribution (DIBCO2013 F=99.5) collapses out-of-distribution (H-DIBCO2018 F=77.59 vs 88.34 winner)
([DE-GAN](https://ar5iv.labs.arxiv.org/abs/2010.08764), **decision-critical**). A restoration model
only fixes what its degradation distribution simulated.

AnimeSR **learns** degradation operators (small 2–3-conv-layer nets fit to real low-quality footage)
and mixes them into the synthetic pipeline; measured MANIQA: hand-crafted basic-ops 0.3554 →
one learned operator 0.3763 → 3-operator pool **0.3832**
([AnimeSR](https://ar5iv.labs.arxiv.org/abs/2206.07038)). The enabler is an input-rescaling trick
that works *because* flat lines/colours survive rescaling (it fails on natural texture) — which
suits our flat-art regime specifically. This is a concrete sim-to-real recipe for the ~400 real
pairs, though a larger lift than the target-side and op-level changes.

### 2.10 Engineering idioms worth adopting

- **On-the-fly vs pre-baked** is an explicit, documented tradeoff. Real-ESRGAN runs degradation
  on-GPU inside the training loop for per-epoch diversity, and also ships a paired-from-disk loader
  ([realesrgan_model.py](https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/models/realesrgan_model.py)).
  The vectorizer's `pipeline.py` **pre-bakes** `n_variants=4`, capping per-sample diversity at bake
  time. Kornia offers GPU-batched differentiable versions (`RandomJPEG`, `RandomGaussianNoise`,
  `RandomGaussianBlur`, `RandomMotionBlur`) if moving on-the-fly ([Kornia](https://kornia.readthedocs.io/en/latest/augmentation.html)).
- **Per-sample param logging / answer-key lockstep.** Albumentations `ReplayCompose` serializes
  applied params and replays the identical transform on another image
  ([composition.py](https://github.com/albumentations-team/albumentations/blob/main/albumentations/core/composition.py)).
  The wrecker already does the equivalent — `meta.json` records `[{op, severity}]` per variant — so
  this need is met; note only that any **geometric** op added in §2.8 must propagate the identical
  transform to `labels.png`, which the current pixel-only ops never had to.
- **RNG seeding.** Augraphy seeds `random`, `numpy`, and `cv2.setRNGSeed`; the wrecker seeds numpy's
  `default_rng` only. Low risk today because its cv2 ops (`resize`, `GaussianBlur`) are deterministic,
  but any cv2 op that draws randomness later would need `cv2.setRNGSeed`.

### 2.11 Bounding severity for flat art (a caution, not a gap)

Real-CUGAN authors note Real-ESRGAN's "sharpening strength is the largest; the painting style may be
changed; the lines may be incorrectly reconstructed," and expose an alpha knob (0.75–1.3) to trade
blur vs sharpen ([Real-CUGAN](https://github.com/bilibili/ailab/blob/main/Real-CUGAN/README_EN.md)).
Relevant to the curriculum's identity/low-damage pairs (`pipeline.py` `curriculum` branch teaching
"don't fix what isn't broken"): a sharpened target (§2.3) must be **mild** or it will teach the model
to alter clean input.

---

## 3. Recommendations

| # | Recommendation | Confidence + why | Cheapest confirming experiment (cost) |
|---|---|---|---|
| 1 | Add second-order (repeated) degradation: wrap the whole recipe in 2 passes, second gated by a skip prob (~0.5–0.8), instead of only `jpeg_cycle passes=2`. | High — universal across Real-ESRGAN/BSRGAN/APISR/AnimeSR; verified CONFIRMED; directly models multi-gen chains. | Bake a second-order variant set, retrain Stage 1, compare mean deltaE on the ~400 real pairs vs current (~1 training run, hours). |
| 2 | Add a sinc/bandlimit-ringing op (final-step high prob + per-pass low prob), distinct from `unsharp_halo`; e.g. Albumentations `RingingOvershoot` or a circular-lowpass kernel. | High — verified CONFIRMED; ablation names "text and lines"; ringing physics is domain-independent and hits glyph edges. | Add op, retrain, eval typography crops (deltaE + visual) on real pairs (~1 run). |
| 3 | Apply a mild USM to the RGB-L1 target only (not GAN/percep — none exist here) to bias toward crisper edges. | Medium-high — verified CONFIRMED technique; but only the L1-target half maps to this L1+CE model, and the typography-cure link is extrapolated. | Toggle USM on the L1 target, retrain, compare fineline deltaE; no new degradation op (~1 run). |
| 4 | Add genuine upsampling: a resize path that starts below native resolution and stays enlarged (APISR ≤1.2x), not only shrink-then-restore-to-original. | High — APISR decision-critical; exactly the shop's dominant favicon/header case. | Add path to `downscale_upscale`, bake, eval on the real upsampled-logo subset (~half day). |
| 5 | Add geometric degradation (skew/rotate/perspective ±5°, mild fold/shadow) AND propagate the identical transform to `labels.png`. | High for distribution match (scans/phone photos have none modeled); medium for lift size — needs label-transform plumbing the current pixel ops never required. | Add rotate+perspective with matched label warp, bake a small set, visually verify label alignment then eval on scan/photo real pairs (~1 day). |
| 6 | Add WebP (and optionally AVIF/HEIF) to `jpeg_cycle`'s codec choice. | Medium-high — APISR decision-critical; Pillow WebP is a near-free add; matches modern WhatsApp/web exports. | Add `format="WEBP"` branch, bake, eval on WhatsApp-sourced real pairs (~2 hours). |
| 7 | Enrich blur/noise: anisotropic + motion/defocus kernels; Poisson + gray-noise option; final 8-bit re-quantization. | Medium — verified CONFIRMED but photo-calibrated ranges; individually small buys, re-fit ranges to print. | Swap in Albumentations blur/noise transforms, retrain, ablate one at a time (~1–2 runs). |
| 8 | Calibrate wrecker parameters against the ~400 real damaged/clean pairs (distribution match), and consider learning 1–3 AnimeSR-style operators from them. | High for calibration (TextZoom +8.6 pts, DE-GAN collapse); medium for learned ops (larger build). | Measure a distribution distance (e.g. artefact-feature histogram / FID-like) between wrecked outputs and the real damaged set; tune ranges to close it (~1 day, no retrain). |

Keep as-is: `posterize` and `dither_palette_crush` (model indexed-PNG/GIF web logos that photo/anime
pipelines omit) — a genuine strength.

---

## 4. Open questions

1. **Composition philosophy:** guaranteed second-order chains (Real-ESRGAN) vs sparse per-op p=0.2
   `OneOf` (Augraphy)? The two are different priors. The shop's distribution is arguably bimodal —
   heavy multi-gen chains *and* single-artefact scans — which may argue for a mixture rather than one.
2. **How much geometric is right?** Logos are often re-centered/cropped clean before print, so
   aggressive skew/fold may over-model. Needs measurement on the real pairs, not assumption.
3. **Does sharpened-target actually close the typography gap,** or is the bottleneck 256px
   resolution / loss weighting / thin-text corpus coverage? R3 experiment isolates the target lever;
   confounds remain.
4. **On-the-fly vs pre-baked:** is the current `n_variants=4` diversity cap materially limiting, or
   is bake-time reproducibility worth more? Only worth resolving if diversity shows up as a
   train/val gap.
5. **Real-pair split integrity:** the ~400 pairs must be partitioned so calibration tuning and final
   evaluation use disjoint sets, or DE-GAN's in-distribution-inflation trap reappears in our own
   metrics.
6. **Chroma-subsampling / block-grid offset:** worth modeling explicitly (shifting 8×8 grid + 4:2:0
   between JPEG passes), or does naive repeated re-encode suffice? Unresolved; no measured evidence
   found for the marginal value on flat art specifically.
