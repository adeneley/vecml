# Theme A — What damage actually walks into a print shop

A print/prepress-specific degradation taxonomy for the wreck pipeline, written as
deltas against the current implementation.

**Scope of "current wrecker" in this document:** the degradation ops live in
`src/vecml/degrade/wreck.py`; `pipeline.py` only orchestrates (render → derive
labels → composite over background → `apply_recipe`). The registry `OPS` holds
exactly seven ops, all photometric per-pixel:
`jpeg_cycle, downscale_upscale, gaussian_blur, gaussian_noise, posterize,
dither_palette_crush, unsharp_halo`. `sample_recipe` picks 1–4 ops for a
difficulty tier, orders the structural ops by `rng.choice` order, and appends
`jpeg_cycle` last (80%) or second-to-last (20%). There is no spatial-warp,
lighting, halftone, paper, or bleed op anywhere.

---

## 1. Executive summary

The current wrecker is a competent **photometric-loss** generator: multi-pass
JPEG as the anchor, mixed-filter shrink/enlarge, blur, additive noise, posterise,
adaptive-palette dither, and unsharp ringing. Against the shop's stated target
distribution, the gaps are **structural and spatial**, and they line up exactly
with the two families the model under-serves (fineline, typography):

1. **No print-screen path.** Scanned business cards and letterheads — a named
   target family — carry a halftone dot grid whose interference with the scanner
   sampling grid produces moire, and are killed further by binarisation and
   ink-bleed at glyph edges. None of this is modelled.
2. **No geometric or lighting distortion at all.** Skew, perspective, illumination
   gradient, cast shadow, paper texture, and non-white paper tint define the scan
   and phone-photo families; the wrecker has only photometric ops.
3. **Ordering is wrong for two dominant families.** The "upsampled small web logo"
   case needs JPEG baked *at the small resolution before enlargement* so the 8×8
   DCT block grid is magnified into large visible blocks; `jpeg_cycle` always runs
   at full 256 px canvas, producing 8 px blocks that no real upscaled favicon has.
   Multi-generation email/WhatsApp/screenshot chains need 3–4 varied-quality
   passes on shifting grids, not the current 2-pass single-quality cap.
4. **Chroma-edge fringing is not isolated.** JPEG 4:2:0 halves colour resolution
   in both axes, bleeding colour across the sharp coloured edges of type and thin
   lines — a plausible specific cause of the fineline/typography weakness — and the
   pipeline leaves subsampling to PIL defaults rather than forcing it.
5. **Missing edge-ringing mechanism.** The only ringing op is `unsharp_halo`
   (oversharpen); there is no resize-induced sinc ringing/overshoot, which is the
   artefact production blind-SR recipes model explicitly.

The corrective blueprint is well-attested: Real-ESRGAN's **second-order** chain
(run blur→resize→noise→JPEG twice, then a final sinc), BSRGAN's **shuffled-order**
argument plus a nearest-neighbour striding downsample path, and Augraphy/ocrodeg/
Kanungo for the **print/scan-specific** ops (halftone, bleed-through, ink-bleed,
shadow, paper). Calibration is feasible, not hypothetical: Farid's EM/Fourier
resampling detector can be run on the ~400 real damaged originals to measure the
actual customer upsampling-factor distribution and set `resize_range` to it.

**Evidence caveat that applies to the whole theme:** session web search was
disabled and Reddit / prepress forums returned 403/404, so hard *prevalence
percentages* (how often each family walks in) were unreachable. Prevalence below
rests on the shop's own stated target distribution and on which families the
mature document-degradation libraries prioritise, not on quantified polls. Treat
all prevalence tags as qualitative until measured against the harvested archive.

---

## 2. Findings by sub-topic

### 2.1 Halftone screening and descreen moire (scan-of-print)

Scanned printed matter carries a halftone dot screen at a known ruling; when that
screen is re-sampled by the scanner's own grid it produces moire, which scanner
software partially removes with a low-pass "descreen" blur. This is the defining
signature of the scanned-business-card / letterhead family and is entirely absent
from the wrecker.

Print-realistic screen rulings to sample from (all verified against the Wikipedia
[Halftone](https://en.wikipedia.org/wiki/Halftone) table): screen printing 45–65
lpi; 300 dpi laser ~65 lpi; 600 dpi laser 85–105 lpi; offset newsprint 85 lpi;
offset coated 85–185 lpi. Business cards / letterheads are usually coated/uncoated
offset (~150–175 lpi) or office laser (65–105 lpi), so a v2 halftone op should
draw LPI from roughly **65–185** and CMYK screen angles ~30° apart.

**Correction (screen angles + descreen wording).** The verified detail differs from
the raw claim on two points. (a) The verbatim phrase describing "an optional
filter, called a descreen filter … produced when scanning printed halftone images"
is **not** on the cited Halftone page — the substance is standard scanner-software
behaviour but the quote is misattributed; do not cite it as Wikipedia text. (b) The
specific angle set `15/45/75/90` is a valid offset convention but is not what the
article states; its own figure lists typical angles as 90/105/165, and conventions
vary (e.g. C15/M75/Y90/K45). Use "~30° apart to avoid same-colour moire" as the
load-bearing rule, not a fixed quadruple. "Removable *only* by a descreen filter"
is also slightly absolute — descreen/low-pass is standard but not uniquely
exhaustive.

Concrete cheap recipe (Augraphy's
[Faxify](https://raw.githubusercontent.com/sparkfish/augraphy/dev/augraphy/augmentations/faxify.py)):
downscale → optional binarise (Otsu / Sauvola) → rotated Gaussian-filtered dot
pattern at a random angle → upscale. Defaults: `scale_range (1.0,1.25)`,
`half_kernel_size (1,1)`, `angle (0,360)`, screen Gaussian `sigma (1,3)`.
Binarisation + halftone + resize is precisely the thin-line / type killer.
Augraphy's own validation reports OCR ~52% average accuracy drop on its augmented
docs ([paper](https://ar5iv.labs.arxiv.org/html/2208.14558)), evidence the
distribution is both hard and high-impact.

### 2.2 Placed-logo effective-PPI mismatch, and the upsampled-web-logo ordering bug

Two related resolution families, both mis-modelled by the atomic `downscale_upscale`
op (which shrinks to factor 0.85→0.15 then re-enlarges back to the **same** 256 px
canvas, with no re-encode in between).

**Placed-logo effective-PPI mismatch.** Prepress preflight
([FlightCheck](https://markzware.com/products/flightcheck/)) treats **Effective PPI**
— native PPI divided by placement scaling — as the load-bearing resolution metric,
tracked as a field distinct from native PPI. A 300 ppi logo dropped into a layout
at 200% is 150 effective ppi. The formula itself is definitional prepress arithmetic,
not verbatim on the product page (noted for citation honesty), but the field's
existence and primacy are confirmed by the source. The modelling consequence: real
placed-logo degradation is a resolution **mismatch at placement scale**, so the
JPEG/noise chain should be applied *at the low native pixel budget* and then the
frame enlarged, rather than compressing at full canvas after re-enlarging.

**Upsampled small web logo (ordering bug, decision-critical).** JPEG uses 8×8 DCT
blocks; quantisation differs per block, producing "discontinuities at the block
boundaries" most visible "in flat areas," plus ringing around sharp
text/colour edges
([Compression artifact](https://en.wikipedia.org/wiki/Compression_artifact),
[JPEG](https://en.wikipedia.org/wiki/JPEG): "the compressed 8×8 squares are visible
in the scaled-up picture"). A 100 px web JPEG enlarged 2.56× to 256 px magnifies its
8 px blocks into ~20 px blocks — the actual signature of an upscaled favicon. The
current pipeline **never** produces this: `jpeg_cycle` always encodes at 256 px
(literal 8 px blocks), and because `downscale_upscale` is atomic, the image is never
left small between ops, so JPEG is never nested inside the downscale/upscale bracket.
Even the 20%-second-to-last branch, if followed by `downscale_upscale`, would
shrink-then-enlarge the 256 px blocks and destroy the grid, not magnify it.

Fix: a dedicated family that does **downscale-to-small → JPEG (low quality) at small
size → enlarge (nearest/bicubic) → optional light second JPEG at canvas**. The
existing `downscale_upscale` op is the scaffold; it needs an intermediate JPEG pass.

*Citation-precision note:* the raw detail's line about bilinear/bicubic "reduces
contrast (sharp edges) … undesirable for line art" is from Wikipedia's
[Image scaling](https://en.wikipedia.org/wiki/Image_scaling) article, not the
Compression-artifact page; the load-bearing block-magnification facts are correctly
sourced.

### 2.3 Multi-generation JPEG chains

Email → WhatsApp → screenshot → re-save compounds artefacts: each hop re-quantises
on a **shifted** 8×8 grid at a different quality, and social re-encoders typically
force 4:2:0 at quality ~70–80 and downscale to a max dimension. The current
`jpeg_cycle` caps at 2 passes at a **single** quality (and only reaches 2 passes when
`severity > 0.66`), which understates generation loss. A realistic chain is
3–4 passes at varied quality with small intervening rescales, so the block grid
shifts between passes. Wikipedia notes successive lossy compressions compound
information loss ([Compression artifact](https://en.wikipedia.org/wiki/Compression_artifact)).

This is the same conclusion the blind-SR literature reaches structurally: adopt a
**second-order** degradation — run the whole blur→resize→noise→JPEG chain twice with
independent params — which Real-ESRGAN found the best balance for real degradations
([paper](https://ar5iv.labs.arxiv.org/abs/2107.10833);
[train yml](https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/options/train_realesrgan_x4plus.yml),
`second_blur_prob 0.8`, stage-2 `noise_range2 [1,25]`, `jpeg_range2 [30,95]`).

### 2.4 Chroma subsampling 4:2:0 — coloured-edge fringing on type

JPEG 4:2:0 (the most common JPEG/JFIF default) subsamples Cb and Cr by a factor of 2
**both horizontally and vertically**, halving colour resolution in each dimension. Per
Wikipedia [Chroma subsampling](https://en.wikipedia.org/wiki/Chroma_subsampling),
chroma then "bleeds" into luma and "a loss of luminance occurs at the border" where a
saturated colour meets an unsaturated/complementary one — which includes saturated
logo/text colours against white. The article's illustrative examples are synthetic
flat colour patches, so this transfers to graphic art better than a typical
photo-domain result. This is a **plausible (source-supported, not source-proven)**
contributor to the model under-serving fineline and typography.

**Correction.** The raw detail presents "colored edges appear fuzzy rather than
crisp" as a Wikipedia quote; that phrase is **not in the article** — treat it as an
uncited paraphrase, not a source quote. Also, pure spatial fringing from halved
chroma resolution is a distinct effect from the gamma luminance-loss-at-borders the
article emphasises; both are real.

**Practical caveat (decision-relevant):** libjpeg/PIL often switch to 4:4:4 at high
quality (~Q90+), so 4:2:0's relevance depends on the quality factors the pipeline
samples. The current `jpeg_cycle` reaches down to quality 8 and leaves subsampling to
PIL defaults, so heavy passes very likely already hit 4:2:0 — but it is not
guaranteed or isolated. A v2 that **forces 4:2:0** (and/or adds an explicit
chroma-blur-at-edges op) makes the fringing lesson deterministic.

### 2.5 Geometric distortion: skew, perspective, illumination, shadow

Confirmed against the code: the wrecker has **no** rotation, skew/affine,
homography/perspective, or illumination-gradient op — all seven ops are photometric.
Yet the scanned-letterhead/business-card and phone-photo-on-a-desk families are
*defined* by these distortions. Document-imaging references model scanned pages with
skew in the ±20° range, plane-to-plane perspective projectivity, and
illumination/page-coloration variation
([Document mosaicing](https://en.wikipedia.org/wiki/Document_mosaicing); note the
source *removes* illumination via a Sobel gradient as normalisation — a wrecker must
*synthesise* the same gradient, opposite direction, same phenomenon).

Camera-captured documents additionally need cast shadows and directional lighting,
not just noise/blur. Augraphy's
[ShadowCast](https://raw.githubusercontent.com/sparkfish/augraphy/dev/augraphy/augmentations/shadowcast.py)
models a random polygon, strongly blurred, blended at variable opacity
(`shadow_opacity_range (0.2,0.9)`, `shadow_blur_kernel_range (101,301)`).
[ocrodeg](https://github.com/NVlabs/ocrodeg) gives published ranges for the scan
path: page skew ±1–2°, elastic distortion sigma [1,2,5,20], ruled-surface warp
magnitude [5,20,100,200].

### 2.6 Scanner and paper realism (paper tint/texture, streaks, bleed-through, ink-bleed)

The current `gaussian_noise` is spatially-white i.i.d. additive noise and `_sample_bg`
picks only flat solid colours — neither reproduces the structured background of a real
scan. Missing signatures:

- **Non-white paper + fibre texture.** A flatbed scan's "white" is warm/grey, not
  255,255,255, over a fibrous texture. Model with paper texture + tint rather than a
  flat fill (Augraphy PaperFactory; ocrodeg `make_fibrous_image`).
- **Descreen low-pass blur** applied by scanner software (ties to §2.1).
- **Bleed-through / show-through** (reverse-side ink through paper): flip
  left-right, blur, offset, alpha-blend under the front image. Augraphy
  [BleedThrough](https://raw.githubusercontent.com/sparkfish/augraphy/dev/augraphy/augmentations/bleedthrough.py)
  (`alpha 0.2`, `offsets (20,20)`, `ksize (17,17)`).
- **Ink bleed into fibres**, which thickens/erodes glyph edges — a primary
  typography degradation. Augraphy
  [InkBleed](https://raw.githubusercontent.com/sparkfish/augraphy/dev/augraphy/augmentations/inkbleed.py)
  (Sobel edges → dilate/erode ink → blur → blend). The principled canonical model is
  **Kanungo**: flip foreground/background pixels with probability decaying with
  distance from the ink boundary (`p = α₀·exp(-α·d²)+η`), then morphological closing
  with a small disk — the mechanism behind thin-stroke erosion, shipped in
  [DocCreator](https://doc-creator.labri.fr/).
- **Dirty-drum / photocopier streaks** (directional noise stripes): Augraphy
  [DirtyDrum](https://raw.githubusercontent.com/sparkfish/augraphy/dev/augraphy/augmentations/dirtydrum.py)
  (`line_width_range (1,4)`, random H/V direction).

### 2.7 Resampling forensics — the upsampling signature and its fragility

Enlargement leaves a learnable signature: **periodic inter-pixel correlations at
period = the resampling factor**, detectable from as little as ~1% upsampling
(Popescu–Farid, [TSP05](https://hfarid.org/downloads/publications/tsp05.pdf)). The
interpolation kernel determines the ringing the model must invert: bicubic ×2
converges to a **signed** kernel (−0.25, 0.5, −0.25) whose negative lobes are the
mathematical origin of edge overshoot/ringing; bilinear gives averaging weights;
nearest is exact pixel replication (staircasing, no new correlation trace).

**Fragility (decision-critical for this shop's junk):** JPEG at Q≤97, and especially
Q≤90, largely destroys the clean periodic trace, and JPEG's own 8 px block grid
*coincides with and masks* the resample period (e.g. a 60% upsample vs the 8 px
block cell). So for heavily-JPEG'd inputs the model cannot rely on the periodic
signature and must recover coarse structure — which reinforces that **op order is
the central lever**. Pure integer-factor **nearest** decimation leaves *no* periodic
correlation, only aliasing — a distinct mode that teaches aliasing recovery, so it
should be a separate degradation branch, not folded into interpolated resize.

**Calibration is directly implementable:** run Farid's EM/Fourier-peak detector
(neighbourhood N=2, σ₀=0.0075) on the ~400 real damaged originals to estimate the
empirical **distribution of upsampling factors** customers actually use, then set
`resize_range` to match. The 3/4-vs-5/4 ambiguity is acceptable because only the
factor distribution is needed, not exact recovery.

### 2.8 Missing sinc ringing/overshoot, and op-ordering structure

Real-ESRGAN models resize-induced **ringing and overshoot with a 2D sinc filter**,
applied as the final step at `final_sinc_prob 0.8`, with the order of the final sinc
and JPEG randomly swapped ([paper](https://ar5iv.labs.arxiv.org/abs/2107.10833)).
The wrecker has no such op — `unsharp_halo` is oversharpen, a different mechanism —
so a sinc ringing op is the most direct structural addition for the
fineline/typography under-service.

BSRGAN argues the **order** of degradations should be randomly shuffled rather than
fixed, because real degradation order varies, and includes a pure nearest striding
downsample path (`img[0::sf, 0::sf]`)
([utils_blindsr](https://raw.githubusercontent.com/cszn/BSRGAN/main/utils/utils_blindsr.py)).
The current `sample_recipe` uses a fixed structural-then-JPEG ordering; JPEG is never
mid-chain. Concrete parameter centres to port (production-validated,
[Real-ESRGAN yml](https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/options/train_realesrgan_x4plus.yml)):
resize `[0.15,1.5]` / `[0.3,1.2]` with up/down/keep probs `[0.2,0.7,0.1]`/`[0.3,0.4,0.3]`;
blur sigma `[0.2,3]`/`[0.2,1.5]`; kernel size `{7,9,…,21}`; JPEG `[30,95]`; Gaussian
noise `[1,30]`. Print-shop junk is worse than SR benchmarks, so keeping the wrecker's
lower JPEG floor (~8) and stronger enlargement is justified — these ranges give the
realistic **centre**, not the tail.

### 2.9 What the current wrecker already covers (do not rebuild)

Confirmed adequate and better than a "simple recipe": multi-pass JPEG as the anchor,
mixed-filter shrink/enlarge (including nearest), Gaussian blur, additive noise,
posterise/banding, adaptive-palette Floyd-Steinberg dither (GIF/PNG re-quantise), and
unsharp ringing. The competitor
[vectorizer.ai](https://vectorizer.ai/) validates the priority targets —
sub-pixel/anti-aliased edge recovery, "sensible guesses when the pixels are a mess,"
adaptive simplification of "faint and indistinct boundaries," scans/photos — none of
which contradicts the gap list above. The gaps are structural/spatial, not the
photometric basics.

---

## 3. Recommendations

Confidence reflects both source strength and regime fit (flat graphic art / placed
logos / scanned paper, not photo-restoration).

| # | Recommendation | Confidence + why | Cheapest confirming experiment (cost) |
|---|----------------|------------------|----------------------------------------|
| 1 | Add a **JPEG-at-small-then-enlarge** family (downscale → low-Q JPEG at small size → nearest/bicubic upscale → optional light JPEG), separate from the atomic `downscale_upscale`. | High — mechanism confirmed against source and against the code (block grid never magnified today); directly targets a named family. | Generate 20 samples both ways; eyeball that block cells are ~scale×8 px, not 8 px (~30 min, no training). |
| 2 | Add a **halftone/descreen op** (Faxify-style: rescreen at LPI 65–185, CMYK angles ~30° apart, scanner-DPI resample, partial low-pass descreen). | High — halftone physics and LPI table verified; scanned cards are a named family. | Run the op on 10 clean cards, overlay a real scanned card, compare moire/FFT peaks (~1 hr). |
| 3 | Adopt **second-order degradation** (run blur→resize→noise→JPEG twice) and raise JPEG chains to 3–4 varied-quality passes on shifting grids. | High — Real-ESRGAN production-validated; matches email/WhatsApp chains; current 2-pass single-quality cap understates it. | Ablate second-order vs single-pass on the ~400 real pairs; compare bench mean ΔE (1 train run). |
| 4 | Add **geometric + lighting** ops (skew ±≤5–20°, mild perspective, illumination gradient, cast shadow, paper tint+texture) for the scan/phone-photo families. | Medium-high — sources solid; magnitude ranges need calibration to the harvested pairs to avoid over-warping vector art. | Sample 30 wrecked variants, human-rate "looks like a real scan/photo" against real pairs (~1 hr). |
| 5 | Add a **sinc ringing/overshoot** op (`final_sinc_prob`-style, order-swapped with JPEG); keep `unsharp_halo` as distinct. | Medium-high — Real-ESRGAN-standard; plausibly the most direct lever for fineline, but "most direct" is inference not measured. | Add op, train one model, compare fineline/type ΔE sub-metric (1 train run). |
| 6 | Add **ink-bleed / Kanungo edge-flip + morphological closing** for glyph-edge erosion/thickening. | Medium-high — canonical bilevel model; the principled mechanism behind typography damage. | Apply to 10 type-heavy logos, diff stroke width vs clean, sanity-check erosion direction (~1 hr). |
| 7 | **Force 4:2:0** in `jpeg_cycle` (and/or an explicit chroma-edge-blur op) so coloured-edge fringing on type is deterministic. | Medium — mechanism source-supported but *not proven* to be the cause; 4:4:4-at-high-Q caveat means it may already partly fire. | Toggle `subsampling=2` in PIL save, diff chroma at type edges on 10 logos (~20 min). |
| 8 | **Shuffle op order** and add a pure **nearest striding-decimation** aliasing branch (distinct from interpolated resize). | Medium — BSRGAN-argued; benefit over the current fixed order is real but likely second-order vs #1–#3. | A/B shuffled vs fixed order on the ~400 pairs, one train run each. |

---

## 4. Open questions

- **Prevalence is unmeasured.** Web/forum prevalence data was unreachable this round.
  What is the *actual* mix across the ~400 harvested damaged/clean pairs — what
  fraction are scans vs upscaled web JPEGs vs Office re-exports vs phone photos? This
  should set the recipe-family sampling weights and is the single highest-value
  calibration step. (Cheap: manually tag the 400 pairs, ~2–3 hrs.)
- **Empirical upsampling-factor distribution.** Run Farid's EM/Fourier detector on
  the real pairs to set `resize_range` to the measured distribution rather than the
  SR-benchmark default `[0.15,1.5]`. Does the shop's junk cluster at specific
  factors (e.g. favicon 16/32/64 → banner)?
- **Does 4:2:0 fringing actually explain the fineline weakness,** or is it dominated
  by blur + block magnification? Only an ablation on the fineline/type ΔE sub-metric
  can separate these; the chroma hypothesis is source-supported but not confirmed as
  the cause.
- **Geometric magnitude ceiling.** How much skew/perspective is realistic before it
  changes the *task* (the tracer must output the deskewed logo, or the warped one)?
  This interacts with the label pipeline — a spatial warp on the input without a
  matching warp on the label map would teach the model to deskew, which may or may
  not be intended. Needs a decision before implementing §2.5.
- **Halftone label integrity.** Rescreen + descreen changes colours; does the
  white-paper assumption in `_derive_ground_truth` / `audit_sample` still hold, or do
  halftone samples need a QC exemption? (`bg_mode="random"` already flags
  reconstruction error, but halftone is a stronger perturbation.)
- **Second-order cost.** Running the full chain twice per sample roughly doubles
  wreck compute; is that acceptable at the current dataset-generation throughput, or
  should second-order be sampled with a probability rather than always-on?
