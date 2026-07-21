# Theme D — Realism vs Diversity: How Exact Must the Wrecker Be

Research synthesis, 2026-07-21. Scope: whether `src/vecml/degrade/pipeline.py` +
`wreck.py` should mimic the shop's real customer-file distribution precisely or
randomize broadly, and the concrete sampling strategy that follows.

All recommendations are deltas against the current implementation:
- **7 ops** (`jpeg_cycle`, `downscale_upscale`, `gaussian_blur`, `gaussian_noise`,
  `posterize`, `dither_palette_crush`, `unsharp_halo`).
- **3 disjoint difficulty tiers** (`light`/`medium`/`brutal`), each a uniform
  severity band, selected once per `wreck_svg` run via the `difficulty` arg.
- **Independent, uniform op draws** in `sample_recipe`; JPEG pinned last
  (20% second-to-last).
- **Bimodal curriculum**: 12% identity, 18% severity×0.3.
- First-order chain (never repeated); AWGN-only noise; isotropic blur only;
  in-place JPEG re-encode (grid stays aligned); JPEG quality driven to 8.
- Bench reports a single pooled 1.520 mean deltaE.

---

## 1. Executive summary

The literature does not support a binary choice between "mimic exactly" and
"randomize broadly." The evidence converges on a third position that is directly
implementable here: **calibrate the distribution's anchor and bounds to measured
real data, then randomize structurally within those bounds, and tune the
free parameters against downstream restoration error on real pairs — not against
visual realism.**

Four load-bearing results:

1. **Diversity has a floor, and it is counted in distinct *parameterizations*,
   not ops.** Below a threshold, sim-to-real transfer degrades sharply
   ([Tobin et al.](https://ar5iv.labs.arxiv.org/abs/1703.06907)). The current
   7-op / 3-narrow-tier design plausibly sits below the effective floor because
   each run locks to one tier.

2. **Structure beats uniform sampling, and specifically rescues small/thin
   features** — the exact failure mode (fineline, typography) the model shows
   today ([Prakash et al. SDR](https://ar5iv.labs.arxiv.org/abs/1810.10093)).
   Independent uniform op draws lack the conditional structure of real capture
   paths (scan, screenshot-chain, photo-of-card).

3. **The most *task-useful* wrecker is not the most *realistic* one.** Tuning a
   generator to maximize downstream accuracy beats both random and hand-tuned
   realism ([Ruiz et al. Learning to Simulate](https://ar5iv.labs.arxiv.org/abs/1810.02513)).
   The ~400 real damaged/clean pairs should be the *tuning objective* (minimize
   real-pair bench deltaE), not merely a realism eyeball.

4. **Mixed continuous severity dominates fixed-tier training** for blind
   restoration ([DnCNN-B](https://arxiv.org/abs/1608.03981),
   [FFDNet](https://arxiv.org/abs/1710.04026)). Draw severity continuously
   per-sample from 0 (clean) through brutal, rather than per-run tiers.

The synthesis answer to Theme D: **exactness matters for the *bounds and anchor*
of the distribution (calibrate these to real data); breadth matters *within*
those bounds (randomize structurally); and neither is chosen by realism — both
are chosen by downstream deltaE on real pairs.** Broad randomization that is not
anchored to the real intake will waste the small 4.3M UNet's capacity on
off-distribution mush (e.g. QF=8 double-compression the real intake may never
contain); precise mimicry that is too narrow will fall below the diversity floor
and fail to generalize to capture paths not in the 400-pair sample.

---

## 2. Findings by sub-topic

### 2.1 Is there a diversity floor, and where does the current design sit?

Yes, there is a floor, and it is measured in **distinct parameterizations**.
[Tobin et al. (2017)](https://ar5iv.labs.arxiv.org/abs/1703.06907) ran a texture
ablation (Fig. 5) showing sim-to-real localization "degrades significantly when
fewer than 1,000 [unique random texturizations] are used," where those are
*instances* sampled from only 3 procedural categories. The binding constraint
was diversity of parameterizations, not raw image count: 10k images with 1k
textures performed comparably to 1k images.

**Verification caveats (CONFIRMED with scope limits):** this is robotics RGB
object localization, not flat-graphic-art restoration. The specific number 1,000
does **not** transfer; the *principle* does. The source reports "degrades
significantly," not a hard collapse to zero — "collapse" overstates the
sharpness.

**Read for the wrecker:** the count of distinct degradation parameterizations
sampled per op governs generalization. Because `wreck_svg` locks one
`difficulty` tier per run and each tier is a narrow uniform band, the *effective*
diversity per training corpus is lower than the op count suggests. This argues
for continuous per-sample severity (see 2.4) and for widening the number of
distinct settings each op can take — not merely adding more op types.

### 2.2 Uniform vs structured sampling — the fineline/typography link

[Prakash et al. (Structured Domain Randomization, arXiv:1810.10093)](https://ar5iv.labs.arxiv.org/abs/1810.10093)
show, at identical 25k-image budgets on KITTI car detection, that context-aware
structured randomization beats uniform DR: 56.7/38.8/24.0 → 77.3/65.6/52.2 AP
(easy/mod/hard). The Hard-case gain is ~2.18× and SDR's Hard 52.2 even exceeds
real BDD100K's 45.6. The paper explicitly attributes uniform DR's failure to
missing conditional dependence p(o_j | c_i), and ties the small-vehicle failure
to lack of context. Ablations: removing context and removing saturation/texture
structure were the dominant degraders.

**Verification caveats (CONFIRMED with scope limits):** autonomous-driving PHOTO
domain + object detection, where "structure" is *spatial scene composition*. The
mapping to correlated degradation *operators* (scan ⇒ halftone + paper-texture +
skew + scanner-noise together; screenshot ⇒ resample + multi-JPEG together) is a
defensible analogy of the same principle, but the 2× magnitude and small-object
rescue are **not** measured for print-artifact restoration and must be
re-measured on the 400 real pairs. Note transcription: the paper's Hard SDR is
52.2, not 52.5.

**Read for the wrecker:** `sample_recipe` currently draws ops independently and
uniformly (`rng.choice(_STRUCTURAL, ...)`). Real degradation is *not*
independent — it arrives in capture-path clusters. The fineline/typography
under-service is the small-object failure mode this paper describes. v2 should
model **capture-path recipes** (named correlated op bundles) rather than
independent draws. This is the single most direct lever on the model's known
weakness, though its magnitude is unproven here.

### 2.3 Realism vs task-optimality — what to optimize the wrecker against

The most task-useful simulator is not the most realistic one.
[Ruiz et al. (Learning to Simulate, arXiv:1810.02513)](https://ar5iv.labs.arxiv.org/abs/1810.02513)
tune simulator parameters via policy gradient to maximize the *downstream
model's* validation accuracy, explicitly not to mimic real data: KITTI
segmentation IoU 48.0 → 57.9 (real-data ceiling 77.8). Verbatim: "mimicking real
data [may not be] the best use of simulation, since a different distribution
might be optimal for maximizing a test-time metric."

This is reinforced from the restoration side.
[RandAugment (arXiv:1909.13719)](https://ar5iv.labs.arxiv.org/abs/1909.13719)
confirms "the optimal magnitude of augmentation depends on the size of the model
and the training set," so severity ranges borrowed from big-model literature are
systematically mis-sized for the small 4.3M UNet and bounded ~12k-PDF corpus.

**Verification caveats (both CONFIRMED with a critical tightening):** both are
natural-photo regimes; the IoU/accuracy magnitudes are directional, not
predicted for this domain. More important — RandAugment optimizes augmentation
as a *regularizer* against a fixed test distribution, whereas the wrecker
*defines* the inference-input distribution. Therefore a parameter search that
maximizes validation metric on **synthetic-degraded** data just re-creates the
proxy mistake the paper warns against. The search must be scored against
**held-out real damaged/clean pairs**. Also, "a separate proxy search is
inherently suboptimal" should be softened to "may be suboptimal" per the paper's
own hedged wording.

**Read for the wrecker:** the ~400 pairs are a *tuning objective*, not a realism
check. Minimize end-to-end bench deltaE on real pairs when setting severity
ranges, op probabilities, and correlation structure. Do not optimize visual
similarity of wrecked renders.

### 2.4 Severity distribution — mixed continuous beats fixed tiers

[DnCNN-B (arXiv:1608.03981)](https://arxiv.org/abs/1608.03981): a single model
trained blind over σ∈[0,55] matches or beats specialists trained at each fixed
level (BSD68: 31.61 vs 31.42 at σ=15; 29.16 vs 28.92 at σ=25; 26.23 vs 25.97 at
σ=50). [FFDNet (arXiv:1710.04026)](https://arxiv.org/abs/1710.04026) trains one
network over σ∈[0,75] and reaches the clean end by making 0 the lower bound of a
continuous range, plus a test-time strength control — it does **not** allocate a
named clean-pair fraction.

**Read for the wrecker:** two concrete deltas.
- `difficulty` currently selects one disjoint band per run (`medium` →
  sev_range (0.30,0.65) for the whole run). Replace per-run tiers with a
  **continuous per-sample severity draw** spanning 0 → brutal.
- The curriculum branch's **bimodal 12% identity + 18% (severity×0.3)** spike is
  a discrete approximation of a smooth lower tail. A continuous distribution
  reaching 0 removes the need to hand-tune the 12/18 split and matches how
  DnCNN/FFDNet include the clean end. Keep identity mass, but as the lower tail
  of a smooth distribution, not a spike.

### 2.5 Curriculum ordering — low priority

[When Do Curricula Work? (arXiv:2012.03107)](https://arxiv.org/abs/2012.03107):
easy-to-hard ordering gives "only marginal benefits, and randomly ordered
samples perform as well or better," except under a limited training budget or
noisy labels. Restoration here has perfect paired labels and (presumably) a full
budget, so an epoch-scheduled mild→severe curriculum is low-value. Caveat: this
evidence is from classification, so it is suggestive, not conclusive for dense
restoration.

The auto-curriculum result — [OpenAI ADR (arXiv:1910.07113)](https://ar5iv.labs.arxiv.org/abs/1910.07113),
start narrow and widen a boundary only when performance clears a threshold — is
**CONFIRMED but strongly regime-bounded**: it is long-horizon RL control where
wide-from-scratch training was *infeasible*. Supervised per-pixel restoration on
256px flat art is routinely trained wide-and-fixed (this is exactly how
Real-ESRGAN/BSRGAN succeed). So "replace tiers with a per-op auto-curriculum" is
a *hypothesis to A/B test*, not an established win. The transferable part is the
**calibration principle** (initialize the distribution to measured real stats),
not the auto-expansion machinery.

**Read for the wrecker:** invest in per-sample mixed-severity sampling (2.4), not
in scheduled ordering. Treat auto-curriculum as optional and only if a
well-tuned fixed-wide recipe underperforms.

### 2.6 Op-coverage realism gaps that bound how "broad" broad can be

Broad randomization only helps if it stays on the real manifold. Two findings
bound the current op set (fuller treatment belongs to Theme B on the degradation
chain; summarized here because they constrain the realism/diversity trade-off):

- **JPEG floor.** Both [Real-ESRGAN](https://arxiv.org/abs/2107.10833) and
  [BSRGAN](https://arxiv.org/abs/2103.14006) cap JPEG at QF∈[30,95]. The current
  `jpeg_cycle` drives quality to 8. Quality-8 double-compressed content is far
  more destroyed than either reference permits; unless the real intake genuinely
  contains QF<30 material (verifiable from the 400 pairs), this teaches the model
  to hallucinate from off-distribution mush. **Measure the real intake's
  effective QF distribution and clamp to match** — this is the clearest case
  where broad ≠ better.
- **AWGN over-fitting.** [CBDNet (arXiv:1807.04686)](https://arxiv.org/abs/1807.04686):
  denoisers trained on pure additive Gaussian noise fail on real images by
  5.63 dB, fixed only by adding signal-dependent Poisson noise + an ISP/JPEG
  pipeline (+4.88 dB). The current `gaussian_noise` is pure AWGN. This is a
  realism *gap*, not excess breadth — the real manifold includes noise the
  wrecker cannot currently produce.

The general backbone: [SRMD (arXiv:1712.06116)](https://arxiv.org/abs/1712.06116)
shows >10 dB collapse under degradation mismatch, confirming the owner's framing
that operation coverage *is* the task definition.

### 2.7 The literature's own gap — which makes the 400 pairs the key asset

Neither Real-ESRGAN nor BSRGAN quantitatively ablated their diversity/order
choices; validation was qualitative on real images
([BSRGAN publishes no shuffle-vs-fixed number](https://arxiv.org/abs/2103.14006)).
And [RealSR (arXiv:1904.00523)](https://arxiv.org/abs/1904.00523) shows real
captured pairs beat synthetic by ~0.6 dB — modest, so real pairs are a
calibration/validation asset more than a full replacement for synthesis. Because
no published number tells you "how much diversity is enough," the shop's ~400
real pairs are the differentiating asset: they close the loop the papers left
open. Build a **fixed real-pair validation set** and measure the wrecker's
quality by restoration deltaE on it.

---

## 3. Recommendations

| # | Recommendation | Confidence + why | Cheapest confirming experiment (cost) |
|---|----------------|------------------|----------------------------------------|
| 1 | Score wrecker changes by end-to-end bench deltaE on a frozen held-out real-pair validation set, not by visual realism. | High — directly supported by Learning to Simulate + RealSR; the 400 pairs make it feasible. | Split the 400 pairs 300/100; freeze 100 as val; wire deltaE-on-val into the bench (~1 day, no training). |
| 2 | Break the single 1.520 mean deltaE into severity-stratified bands + a clean-passthrough slice. | High — DnCNN/FFDNet standard practice; without it, fineline regressions hide in the mean. | Add per-band aggregation to the existing bench script (~half day). |
| 3 | Replace per-run `difficulty` tiers with a continuous per-sample severity draw from 0 → brutal. | High — DnCNN-B/FFDNet show mixed-continuous ≥ fixed-tier for blind restoration. | Add a `continuous` sampling mode to `sample_recipe`; train one run; compare stratified deltaE vs tiered baseline (~1 train cycle). |
| 4 | Replace independent uniform op draws with named capture-path recipes (scan-cluster, screenshot-chain, photo-of-card, office-doc-roundtrip) that correlate ops. | Medium — SDR principle is strong but magnitude unproven off-photo-domain; the fineline link is the reason to try. | Author 4-5 correlated recipes; A/B one training run vs uniform draws on stratified real-pair deltaE (~1-2 train cycles). |
| 5 | Measure the real intake's effective QF distribution and clamp `jpeg_cycle` quality to match (likely raise the floor above 8). | Medium-High — both SOTA recipes floor at QF30; risk is teaching hallucination from off-distribution mush. | Run a JPEG-quality estimator over the 400 damaged originals; histogram; set the floor (~half day, no training). |
| 6 | Fold the bimodal 12%/18% curriculum spike into the smooth lower tail of the continuous severity distribution (identity = lower bound, not a discrete branch). | Medium-High — FFDNet reaches clean via range lower bound, not a named fraction. | Ships with #3; compare clean-passthrough deltaE slice against the current bimodal branch. |
| 7 | Add a signal-dependent (Poisson) + gray-channel noise op alongside AWGN. | Medium-High — CBDNet's 5.63 dB AWGN gap is a clear realism hole; magnitude for flat art unverified. | Add op; single training run; compare deltaE on the noisiest real-pair band (~1 train cycle). |
| 8 | Treat auto-curriculum / adversarial-hard-sampling as a later A/B against a well-tuned fixed-wide recipe, not a v2 default. | Medium — ADR/Volpi confirmed only in RL/classification regimes; fixed-wide is how Real-ESRGAN succeeds. | Defer until #1-7 land; then one A/B run if fixed-wide plateaus. |

---

## 4. Open questions

1. **Does the real intake contain QF<30 material?** Decides whether driving JPEG
   quality to 8 is on- or off-distribution. Answerable now from the 400 pairs
   (Rec 5); blocks how brutal the severe tail should go.
2. **How much does op *correlation* actually lift fineline/typography deltaE
   here?** The 2× SDR magnitude is photo-domain; the print-domain number is
   unknown and must be measured (Rec 4). If the lift is small, the effort is
   better spent on op-coverage gaps (2.6).
3. **What is the effective diversity floor for *this* task?** Tobin's 1,000 is
   regime-specific. Unknown whether the current parameterization count is above
   or below the floor for a 4.3M UNet on flat art — testable only by sweeping
   distinct-parameterization count against real-pair deltaE.
4. **Do the 400 pairs cover the capture paths the shop actually sees, or are
   they biased toward whatever was easy to harvest?** If capture-path coverage
   is skewed, tuning to the 400-pair val set (Rec 1) inherits that skew. Needs a
   capture-path audit of the pair set before it is trusted as the sole objective.
5. **Blind vs degradation-aware model.** [DASR (arXiv:2104.00416)](https://arxiv.org/abs/2104.00416)
   shows conditioning on a degradation representation prevents blind-model
   collapse under mismatch. Out of scope for the wrecker itself, but if broad
   randomization can't span the full real space, a degradation-conditioned
   cleanup head is the alternative — a model-architecture question for a later
   theme.
