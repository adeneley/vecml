# Theme A — Semantic Conditioning of Restoration Networks

Scope: what is known about injecting recognition signals into cleanup/restoration
nets — methods, measured gains, costs, failure modes — and what transfers to a
4.3M-param dual-head UNet operating at 256px on graphic art, under a hard
print-fidelity constraint (sharpen what is there, never confidently invent what
is not).

---

## 1. Executive summary

The owner's vision — "recognize it's a star, feed that into cleanup; recognize
type, apply typography knowledge" — is not novel. It is a solved architectural
pattern in the restoration literature, and the pipeline already holds the input
it needs (a 16-class per-pixel region-label head).

Four load-bearing conclusions, each survivable after adversarial verification:

1. **The mechanism exists and is cheap.** Spatial Feature Transform (SFT-GAN,
   CVPR 2018) conditions a restoration/super-resolution net on segmentation
   *probability* maps and emits spatially-varying affine parameters
   `SFT(F) = γ⊙F + β` that modulate a few intermediate feature layers. FiLM is
   the simpler global special case (SFT is its spatial generalization). Cost is
   ~2 parameters per modulated feature map, independent of resolution. Because
   the UNet's region head is per-pixel, it feeds the *spatial* SFT variant
   naturally — which is exactly what lets the star region get a different prior
   than the type region within one tile.

2. **The universal failure mode is wrong-recognition → wrong-restoration**, and
   it is the print-fidelity risk stated in mechanism form. But its severity is
   coupled to *generative capacity*. All the dramatic hallucination evidence
   comes from billion-parameter diffusion priors (SeeSR/SD2, SUPIR/SDXL). A
   4.3M-param UNet trained with an L1 RGB-repair objective is structurally
   anchored to the input pixels by that loss and is a far lower-risk rung. The
   risk climbs monotonically as you climb toward generative priors.

3. **Payoff shrinks as the base restorer nears ceiling.** Frozen-feature priors
   deliver large gains only where the baseline is weak (+4 dB on weak low-light)
   and near-zero where it is strong (+0.25 dB on a strong Gaussian denoiser).
   The engine already ceilings near ΔE 1.2 on clean input and the model shows
   zero train/val gap — a mild-damage, near-ceiling regime. Denoising-on-a-strong-
   baseline is the fair analog, and it argues for small payoff.

4. **Published wins are perceptual and often come WITH a fidelity drop** — so the
   literature's headline numbers (FID/LPIPS/user-preference) may not transfer to
   a ΔE objective. This drop is tied to *generative / perceptual-loss*
   optimization, not to semantic conditioning per se: DA-CLIP is a discriminative
   counter-example that led FID while holding PSNR/SSIM. Conditioning must be
   evaluated on ΔE directly, never on a perceptual proxy.

Net recommendation for the roadmap: the lowest rung (spatial SFT modulation of the
UNet from the existing soft region head) is cheap, transfers architecturally, and
is the fidelity-safe entry point — but expect a small ΔE gain in this regime, and
gate every rung on ΔE, not perception. The generative rungs (diffusion priors)
carry the real hallucination risk and only pay off in perceptual terms.

---

## 2. Findings by sub-topic

### 2.1 The injection mechanism: modulate features, do not concatenate channels

The single strongest architectural verdict in this literature: *how* you feed a
predicted label map in matters more than *whether* you do.

- **Feature modulation beats input-channel concatenation by large margins.**
  SPADE modulates decoder activations with per-pixel affine γ/β derived from the
  seg map; the concat baseline (pix2pixHD) is beaten on every benchmark:
  ADE20K mIoU 38.5 vs 20.3, FID 33.9 vs 81.8; COCO-Stuff mIoU 37.4 vs 14.6, FID
  22.6 vs 111.5; Cityscapes mIoU 62.3 vs 58.3.
  [SPADE](https://arxiv.org/abs/1903.07291)

- **Concatenation is actively destroyed by normalization on uniform-label
  patches — the exact situation inside a flat vector fill.** Verbatim: "after we
  apply InstanceNorm to the output, the normalized activation will become all
  zeros no matter what the input semantic label is given. Therefore, semantic
  information is totally lost." This happens precisely when a segment carries a
  single uniform label — i.e. the flat fills the tracer cares about most.
  [SPADE](https://ar5iv.labs.arxiv.org/abs/1903.07291)

- **SFT-GAN is the restoration-domain instance and the right analog for Stage 1.**
  It conditions a super-resolution net on segmentation *probability* maps (soft,
  not hard labels) over 7 outdoor categories + background, emitting
  spatially-varying `γ⊙F + β` on a few intermediate layers.
  [SFT-GAN](https://ar5iv.labs.arxiv.org/html/1804.02815)

- **FiLM is the general/cheap form; SFT is its spatial generalization.** FiLM
  (`γ_{i,c}·F_{i,c} + β_{i,c}`) uses 2 params per feature map, "cost does not
  scale with image resolution," reached 97.7% on CLEVR (halving error 4.5%→2.3%).
  Note the direction: FiLM applies *one* class-vector per tile (global); SFT
  applies *per-location* modulation. The 16-class region head is per-pixel, so it
  maps onto the spatial SFT variant — which is what enables different priors for
  the star vs the type in the same tile. [FiLM](https://ar5iv.labs.arxiv.org/html/1804.02815)

**Transfer note (verified caveat):** SFT-GAN's empirical texture-prior gains are
on outdoor *photos* (grass, water). The mechanism is domain-agnostic and
transfers; the specific learned priors do not. This validates the
architecture/cost rung, not any graphic-art result. No paper in this set reports
SFT-style conditioning measured on vector/graphic art — that gap is ours to close.

### 2.2 The failure mode: wrong-recognition → wrong-restoration

This is the print-fidelity hazard expressed as a mechanism, and it is real and
source-documented — with an important magnitude caveat.

- **Documented as a stated limitation.** SeeSR verbatim: "DAPE may predict
  incorrect tags for heavily degraded images, resulting in wrongly restored
  objects," and tag-region alignment "can be inaccurate in cases of severe
  degradation." SFT-GAN: incorrect categorical classification makes "the network
  apply unsuitable texture patterns." [SeeSR](https://arxiv.org/html/2311.16518)

- **Corrected framing (verification):** the claim that this is the "dominant,
  universal" failure mode is *overstated*. SeeSR lists it as one of three
  unquantified limitations; "universal" is editorial, not a source finding. More
  importantly, **severity is coupled to generative capacity.** Every dramatic
  exemplar is a billion-param generative prior (SeeSR/SD2, SUPIR/SDXL, GAN texture
  synthesis) whose job *is* to synthesize plausible content. A 4.3M-param UNet
  with an L1 RGB-repair loss is anchored to the actual input pixels and is
  structurally limited in its ability to "paint the wrong content" even under a
  misclassified region label. So: directionally true, but the *magnitude* does not
  transfer — risk is low for frozen-feature injection into an L1 UNet, high for
  diffusion priors.

- **Analogous mechanism from the feedback-loop family.** Auto-context (the 2010
  origin of feeding predictions back to yourself): "error propagation can occur
  when early mistakes become embedded in context features, potentially amplifying
  uncertainties"; "classifier confidence may become artificially inflated through
  feedback loops." This is the precise mechanism by which a mislabeled asymmetric
  star could be "corrected." Any self-conditioning loop needs the robustness trick
  in 2.4 and must feed soft probabilities + raw image, never hard labels alone.
  [Auto-context](https://pages.ucsd.edu/~ztu/publication/pami_autocontext.pdf)

### 2.3 Payoff shrinks near the ceiling

- **Frozen-feature priors help most where the baseline is weak.** PTG-RM
  (<1M-param plugin over frozen CLIP/BLIP2/SD/restoration features): low-light
  LOL-real SNR 21.48→25.50 (+4.02 dB) and URetinex 21.16→24.70 (+3.54 dB) on
  weak baselines; but Gaussian denoise BSD68 σ=25 DRUNet 29.48→29.73 (**only
  +0.25 dB**) on a strong baseline. [PTG-RM](https://arxiv.org/html/2403.06793)

- **Corrected framing (verification):** two figures in the raw finding were wrong
  and are dropped here — the "Rain100H +0.31 SSIM" is actually ~+0.009 SSIM, and
  two "author quotes" ("highly degraded images," "improvements are smaller for
  already-effective baselines") do **not** appear in the paper. What the paper
  *does* say (Limitations): the variance "correlate[s] with the capacity of the
  target network **and** the difficulty/complexity of the target task" — two
  coupled factors, not baseline-strength alone. The read-across to graphic art is
  an analogy, but the denoising-on-strong-baseline result (+0.25 dB) is the fair
  analog and it supports limited payoff in a mild-damage, near-ceiling regime.
  The engine ceilings at ΔE 1.219 on clean input; the model is data-constrained
  with zero train/val gap. Both point to "the base restorer is already good,"
  which is exactly where these priors add least.

### 2.4 Perceptual wins do not equal fidelity wins

- **The perception-distortion tradeoff is a proven, unavoidable limit.** For ANY
  distortion measure (not just PSNR/SSIM), lowering mean distortion necessarily
  increases the distance between the restored-image distribution and the natural-
  image distribution. The P(D) function is monotonically non-increasing and
  convex. [Blau & Michaeli](https://arxiv.org/abs/1711.06077)

- **L1/MMSE estimators sit at the safe, non-inventing extreme of that curve.** The
  current Stage-1 L1 RGB head is, by construction, on the maximally-faithful
  (blurry) end. Adding a GAN/diffusion perceptual prior to "crisp it up" provably
  trades measurable fidelity for realism — the exact axis where a deliberately-
  asymmetric star gets corrected to a generic one. [Blau & Michaeli](https://arxiv.org/abs/1711.06077)

- **Published wins are perceptual and come WITH a fidelity drop — by the authors'
  own admission.** SUPIR: "our results have better visual effects, but they do not
  have an advantage in these [PSNR/SSIM] metrics ... there is a need to reconsider
  the reference values of existing metrics"; ~75%-preferred in a user study
  despite lower PSNR/SSIM. SeeSR/CoSeR headline on FID/LPIPS/DISTS/MUSIQ, not
  PSNR. [SUPIR](https://arxiv.org/html/2401.13627)

- **But the drop is tied to generative/perceptual-loss optimization, not to
  semantic conditioning itself.** DA-CLIP (discriminative restoration) led FID
  (34.89 vs AirNet 64.86) while *holding* competitive PSNR 27.01 / SSIM 0.794 —
  a case where semantic conditioning did not cost pixel fidelity.
  [DA-CLIP](https://arxiv.org/html/2401.13627)
  (Numbers secondary and not independently re-verified — web-search budget was
  exhausted; treat DA-CLIP figures as UNVERIFIED but directionally load-bearing.)

- **No-reference perceptual metrics cannot detect hallucination.** The PIRM
  Perceptual Index (`PI = ½·((10 − Ma) + NIQE)`) rewards natural-looking sharpness
  and would score a plausibly-invented shape well. Only full-reference metrics
  against ground truth (RMSE, and by extension mean ΔE) penalize invented-but-
  plausible content. This validates ΔE as the primary print-fidelity gate.
  [PIRM 2018](https://ar5iv.labs.arxiv.org/abs/1809.07517)

### 2.5 Fidelity-anchoring is a documented, necessary countermeasure — for the generative rung

- **Range/null-space decomposition lets a prior fill only what is unobserved.**
  DDNM pins the range-space term to the measurement (Ax=y exactly) and generates
  only null-space content, giving an analytic guarantee the output still matches
  the input. If intent ever lives as a generative prior in Stage 1, wrap it in a
  consistency/projection layer so it can only *add* null-space detail, never
  overwrite present pixels. [DDNM](https://arxiv.org/abs/2212.00490)

- **Consistency wrappers can bolt onto any SR net.** Explorable SR analytically
  guarantees outputs match the low-res input when downsampled, and treats
  restoration as ill-posed with many valid explanations rather than committing to
  one. [Explorable SR](https://arxiv.org/abs/1912.01839)

- **Consistency alone is necessary but NOT sufficient.** PULSE satisfies a
  downscaling-consistency loss yet still hallucinates identity — many distinct
  plausible high-res images satisfy the same constraint, so the prior picks one.
  A consistency layer must be paired with a fidelity metric (ΔE) and a
  conservative non-generative default. [PULSE](https://arxiv.org/abs/2003.03808)

- **SUPIR's in-model anchors, as a concrete pattern.** Restoration-guided sampling
  interpolates the prediction toward the LQ latent each step with weight
  `k = (σ_t/σ_T)^τ_r` (τ_r=4), prioritizing fidelity early/low-frequency;
  negative-quality prompts trained on 100K synthetic LQ images are needed or "CFG
  introduces artifacts." CoSeR adds LR-attention anchoring to the low-res input.
  [SUPIR](https://arxiv.org/html/2401.13627)

- **Verification caveat on necessity:** these anchors are *necessary* specifically
  for the strong-generative-prior regime (a 2.6B T2I model that free-runs unless
  constrained). They are not established as necessary for a 4.3M-param L1-regression
  UNet, which is already anchored by its pixel loss. A residual/identity path there
  is defensible design, not a literature requirement.

### 2.6 Fidelity-safe design properties worth building in

- **Graceful degradation to the unconditioned baseline.** SFT-GAN: "When facing
  ... the absence of segmentation probability maps, our model degenerates itself
  as SRGAN." Design the conditioning so a low-confidence recognition falls back to
  pure classical/low-prior cleanup rather than forcing a wrong prior.
  [SFT-GAN](https://ar5iv.labs.arxiv.org/html/1804.02815)

- **Soft embeddings are more robust than hard tags under degradation.** SeeSR uses
  a Degradation-Aware Prompt Extractor producing soft representation embeddings
  because caption/tag models "proved susceptible to degradation artifacts." Derive
  the intent signal from a degradation-robust soft embedding, not a brittle
  explicit classifier decision. [SeeSR](https://arxiv.org/html/2311.16518)

- **Boundaries are untrustworthy; interiors are reliable — refine boundaries FROM
  interiors.** SegFix (model-agnostic post-process) replaces each boundary pixel's
  label with a nearby interior pixel's label via a learned direction map:
  Cityscapes DeepLabv3 mIoU 79.5→82.6, boundary-F (1px) 56.6→68.6 (+12.0). This
  operationalizes the print-fidelity principle — refine geometry without inventing
  content — and is a candidate Stage-1/Stage-2 bridge for the region map.
  [SegFix](https://ar5iv.labs.arxiv.org/abs/2007.04269)

- **Semantic (stuff-only) maps merge same-class adjacent regions.** Two same-labeled
  shapes that abut (e.g. two black marks on a business card) cannot be separated by
  a 16-class semantic map; panoptic/instance-aware conditioning is needed to keep
  distinct primitives distinct at their touching boundary. Relevant if the tracer
  must not fuse adjacent same-color primitives.
  [Panoptic conditioning](https://arxiv.org/abs/2004.10289)

### 2.7 Adjacent evidence: discrete priors and self-conditioning

These are from the broader Theme-A sweep (face-restoration and iterative-refinement
literatures). They inform the *upper* rungs and are included for completeness; they
are less directly a "conditioning the UNet" result.

- **Discrete codebook priors bound the output space and preserve identity better
  than continuous GAN-latent priors.** CodeFormer reframes restoration as
  classification into a fixed 1024-entry dictionary (`d=256`, compression `r=32`,
  so 512² → 16×16 tokens), "significantly attenuating" LQ→HQ mapping uncertainty.
  On CelebA-Test it beats continuous-latent methods on identity (IDS 0.60 vs GPEN
  0.54 vs GFPGAN 0.42). **Caveat:** the bound is on the *output space* (the trained
  domain's dictionary), not on faithfulness to the specific input — a codebook
  learned on generic graphic tiles would snap a bespoke mark to the nearest atom,
  i.e. dictionary-quantized invention, not the intent-preservation the print
  constraint needs. [CodeFormer](https://ar5iv.labs.arxiv.org/html/2206.11253)

- **A runtime fidelity knob is the most transferable single idea from that line.**
  CodeFormer's CFT module exposes `w ∈ [0,1]` at inference — low w = prior-driven
  quality, high w = input-faithful fidelity — with no retraining. An analogous
  per-inference dial (conservative for fidelity-critical print jobs) is worth
  designing in regardless of which prior rung is chosen.
  [CodeFormer](https://ar5iv.labs.arxiv.org/html/2206.11253)

- **A frozen continuous prior drifts.** GPEN's un-fine-tuned GAN variant produced
  clean but wrong-identity faces; joint end-to-end fine-tuning was required. This
  is a caution against the naive "frozen recognizer features injected into the
  UNet" rung *if* the prior is a generative decoder — a frozen generative prior
  snaps outputs toward its own mode. (Less applicable to frozen *recognizer
  features* used only as SFT conditioning, which do not generate pixels.)
  [GPEN](https://ar5iv.labs.arxiv.org/html/2105.06070)

- **The published anti-drift fix is a heavily-weighted consistency loss.** GFPGAN
  weights an ArcFace identity loss at λ_id=10 — 100× its L1 term — telling the net
  "do not change who this is." The portable analog is a strong geometry/shape-
  consistency loss for graphic art (the star's real geometry must dominate the
  class prior). [GFPGAN](https://ar5iv.labs.arxiv.org/html/2101.04061)

- **Self-conditioning is a near-free quality upgrade for iterative refinement.**
  Feeding the model its own prior output estimate, zeroed ~50% of steps for
  cold-start robustness, with no backprop through it, costs <25% extra training
  time and "greatly improves" quality; it is load-bearing in recurrent refinement
  (RIN degrades sharply at self-conditioning rate 0). If Stage 1 ever becomes
  iterative, this is cheap — but see the auto-context error-amplification warning
  in 2.2. [Self-conditioning](https://ar5iv.labs.arxiv.org/abs/2208.04202),
  [RIN](https://ar5iv.labs.arxiv.org/abs/2212.11972)

### 2.8 Invention-intolerant precedents (why the constraint is not academic)

- **Deep-learning reconstruction silently invents/erases critical structure.**
  Antun et al. (PNAS 2020): tiny perturbations cause severe artifacts, and small
  structures (a tumor) can fail to appear — higher-performing networks are *less*
  stable. Argues for stability/consistency guarantees over raw perceptual
  performance. [Antun et al.](https://arxiv.org/abs/1902.05300)

- **The Xerox JBIG2 bug** substituted stored patterns for similar image segments in
  a prepress/document pipeline (2005–2013), silently changing e.g. 6→8 in scanned
  construction-plan dimensions, with no visual indication. The exact failure mode
  print fidelity must avoid: plausible, invisible substitution of real content.
  [dkriesel](http://www.dkriesel.com/en/blog/2013/0802_xerox-workcentres_are_switching_written_numbers_when_scanning)

- **Competitive positioning:** vectorizer.ai openly performs interpretive inference
  — "teases out features less than a pixel wide," "makes sensible guesses when the
  pixels are a mess." A print-fidelity-first stance (guaranteed non-invention,
  measured by full-reference ΔE) is a genuine differentiator, but only if it is
  actually measured that way. [vectorizer.ai](https://vectorizer.ai/)

---

## 3. Recommendations

| # | Recommendation | Confidence + why | Cheapest confirming experiment (cost) |
|---|---|---|---|
| 1 | Inject the existing 16-class region head into the UNet via **spatial SFT/SPADE-style feature modulation** on a few decoder layers, never as a concatenated input channel. | **High** — SPADE/SFT show large, consistent gains over concat, and concat is provably zeroed on uniform-label fills (the flat regions the tracer needs). | Add SFT layers fed by the soft region map; retrain; compare end-to-end ΔE vs the current dual-head baseline on the 24 held-out images (~1 training run). |
| 2 | **Gate every conditioning experiment on mean ΔE (full-reference), never on FID/LPIPS/user-preference.** | **High** — perception-distortion tradeoff is proven; no-reference metrics provably cannot detect hallucination. | Zero new cost — it is a change to which metric decides go/no-go. |
| 3 | Feed the conditioning as **soft probabilities + raw image**, and design **graceful fallback to unconditioned cleanup** when region confidence is low. | **High** — SFT-GAN degrades to baseline by design; soft prompts are degradation-robust; hard labels amplify misclassification. | Ablate soft-vs-hard and confidence-gated fallback in the same run as #1 (marginal cost). |
| 4 | **Temper expectations for the low rung:** budget a small ΔE gain (order tenths, not a rung change) in this near-ceiling, mild-damage regime. | **Medium** — the fair analog (denoise on a strong baseline) gave +0.25 dB; read-across from photos to graphic art is an analogy, not a measured result. | Covered by #1's ΔE delta; the experiment *is* the expectation check. |
| 5 | **Do not adopt a diffusion/generative prior in Stage 1** without an explicit data-consistency wrapper (DDNM null-space or Explorable-SR projection) AND ΔE gating. | **High** — generative priors are where hallucination severity lives; consistency-alone (PULSE) is insufficient; wins there are perceptual and drop PSNR. | Defer; if piloted, wrap a small diffusion prior in a null-space projection and measure ΔE on the 24-image bench before any broader use (~1 focused pilot). |
| 6 | Prefer a **discrete/dictionary prior over a frozen continuous generative prior** if a stronger prior is ever needed, and prefer **recognizer-feature SFT over a generative decoder**. | **Medium** — CodeFormer beats continuous-latent on identity; frozen generative decoders (GPEN) drift; but discrete priors still quantize-invent, so not automatically fidelity-safe. | Only if #1 underdelivers; compare a codebook-style prior vs continuous on identity/ΔE (research spike, ~2–3 runs). |
| 7 | Build in a **runtime fidelity dial** (CodeFormer-CFT analog) and a **strong geometry/shape-consistency loss** (ArcFace-identity-loss analog) so print jobs can be run conservative. | **Medium** — both are documented anti-drift mechanisms; the shape-consistency-loss form for graphic art is unproven and needs design. | Add a `w`-style blend on the SFT residual and a shape-consistency term; A/B on ΔE + a hand-picked "deliberate asymmetry" test set (~1 run + a curated eval set). |
| 8 | Consider **SegFix-style boundary-from-interior refinement** as the Stage-1→Stage-2 bridge for the region map, since boundaries are the untrustworthy part. | **Medium** — SegFix gives large boundary gains and is model-agnostic, but the graphic-art transfer is unmeasured. | Run SegFix post-process on predicted region maps; measure boundary-F and downstream tracer ΔE (~post-process only, no retrain). |

---

## 4. Open questions

1. **Does spatial SFT conditioning actually move ΔE on graphic art?** No paper in
   this set measures SFT/SPADE conditioning on vector/graphic-art restoration with
   a full-reference fidelity metric. Every gain cited is either photo-domain or
   perceptual. This is the primary unknown and #1 in the recommendations is the
   experiment that resolves it.

2. **Can a "deliberate asymmetry" test set be built to catch intent-invention?**
   ΔE against ground truth catches invention *in aggregate*, but a curated set of
   deliberately-idiosyncratic marks (asymmetric stars, hand-drawn wobble, bespoke
   letterforms) is needed to measure whether conditioning specifically "corrects"
   intent. No such benchmark exists in the literature; it must be constructed from
   the NAS corpus.

3. **Where does the recognition signal come from at inference?** The current head
   is trained on synthetic renders. On real print artwork (typography, fineline,
   gradients — the known content gaps), will the region head classify well enough
   to condition safely, or will it be the misclassification source the failure-mode
   literature warns about? Untested on the real target distribution.

4. **Is the panoptic (instance) distinction load-bearing for the tracer?** Whether
   two abutting same-color primitives must be kept separate depends on Stage-2
   behavior. If it does, a 16-class semantic map is insufficient and instance-aware
   conditioning is required — a larger change than SFT modulation.

5. **DA-CLIP's fidelity-preserving result is unverified.** The one counter-example
   showing semantic conditioning need not cost PSNR/SSIM (DA-CLIP) could not be
   independently re-verified (search budget exhausted). If the roadmap leans on
   "conditioning can be fidelity-neutral," that claim should be re-checked against
   the DA-CLIP primary source before it carries weight.

6. **Does the residual/identity anchor help or is it redundant at 4.3M params?**
   The L1 loss already anchors the UNet; the literature establishes anchoring as
   necessary only for generative priors. Whether an explicit residual path adds
   measurable fidelity on the SFT rung is untested.
