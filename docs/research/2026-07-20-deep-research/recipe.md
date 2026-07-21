# RECIPE: ML-engineering priors for our next training spends

**Theme 2 synthesis — 2026-07-20**

Scope: a concrete prescription for the next 2–3 Stage-1 (cleanup UNet) training runs — exact config
deltas and predicted outcomes — grounded in cited scaling, stability, and loss literature. This
document is self-contained; the project owner should not need to read any source to act on it. Claims
that failed adversarial verification appear here only in corrected form, with the correction noted.
Claims that were not independently verified are flagged as such.

Reminder of where we stand (measured ground truth): 4.3M-param dual-head UNet (L1 RGB repair +
0.1×CE over 16 region labels), 256px, bf16 autocast, AdamW-style, lr≈4.2e-4, batch 16, cosine decay
to lr/20. Data scaling at **fixed** capacity: val loss 0.00782 (100k) → 0.00630 (927k), **zero
train/val gap at both scales**. One 500k run diverged (train loss rose from epoch 1); the
identical-config 1M run was clean. End-to-end bench (mean ΔE, lower better): no-cleanup 1.81, best
model + vtracer 1.52. Ceilings on perfect input: vtracer 1.28, our rust engine 1.44. Stage-2 is the
current quality ceiling, not Stage-1.

---

## 1. Executive summary

The single most important prior is that **we are on the data-dominated branch of the scaling law and
should spend the next runs on data, not capacity.** The closest published analog to our Stage-1 net —
a tunable UNet (32.9k–6.1M params, our 4.3M sits inside the grid) fit to a Chinchilla-style loss that
extrapolates test loss with Spearman 0.95 — shows explicitly that in the data-limited regime the
model-size term drops out of the loss
([González et al. 2026](https://arxiv.org/pdf/2606.06725)). Our own signature (zero train/val gap
**and** val loss still falling from data alone at fixed capacity) is the textbook definition of that
regime. Growing base width 48→64 (~2× params) is therefore predicted to buy little until data grows.

The second prior is a warning about **diminishing returns from more of the same data.** Our two clean
points imply a naive data-scaling exponent of ≈0.10, which is very shallow. Verification showed this
is even shallower than the shallow end of the published vision/language range, so a further 10× of
*icon* data would cut val loss by at most ~19% and probably less. The higher-ROI move is to mint
*diverse* data into our known gaps (typography/lettering, fine-line strokes, region boundaries) rather
than more icons, and possibly to prune toward hard/informative examples, which can beat power-law
scaling ([Sorscher et al. 2022](https://arxiv.org/pdf/2206.14486)).

The third prior concerns **stability.** The 500k divergence has two independently documented, non-
exclusive causes: a missing gradient clip, and the bf16 divergence "lottery" (a controlled study saw
~10% of bf16 runs diverge on identical config differing only by seed, vs 0% for TF32/FP32,
[Zhang et al. 2024](https://arxiv.org/html/2405.18710v2)). Both are cheap to neutralise. Add global-
norm gradient clipping, keep fp32 master weights + optimizer state + loss/softmax/norm accumulation,
add a short LR warmup, and consider running this small 4.3M net in TF32/FP32 outright to remove the
lottery. **Do not** treat the diverged run as a data-ceiling signal.

The fourth prior: on our two-task loss, **do not invest in fancy multi-task optimizers.** Two rigorous
studies show fixed-weight scalarization with a tuned weight matches or beats GradNorm/PCGrad/MGDA/
uncertainty weighting ([Xin et al. 2022](https://arxiv.org/pdf/2209.11379);
[Kurin et al. 2022](https://arxiv.org/abs/2201.04122)). The correct spend is a cheap 5-point sweep of
the single CE weight at 100k, with **weight = 0** as the load-bearing ablation.

Finally, **our headline metrics are hiding the thing that matters.** Global 95.97% label accuracy is
dominated by easy interior pixels; the boundary pixels that decide vectorizer curve placement are
where our errors concentrate. Switch evaluation to a boundary F-score band, and prefer cheap boundary
interventions (distance-weighted CE, or SegFix-style boundary→interior snapping at trace time) over
exotic boundary losses.

The concrete prescription for the next three runs is the boxed "Run plan" after the recommendations
table in §3.

---

## 2. Findings by sub-topic

### 2.1 Are we still data-constrained? (Yes — but read the slope, not the gap)

Zero train/val gap by itself does **not** prove more data helps. A zero gap with high, flat loss means
capacity-constrained; a zero gap with loss that keeps falling as unique data grows is the definition of
data-constrained ([IBM overfitting/underfitting](https://www.ibm.com/think/topics/overfitting-vs-underfitting)).
The load-bearing signal is the **negative slope of val loss vs unique-data count**, which we have
already measured (0.00782→0.00630 over 100k→927k). So the diagnosis "still data-constrained" is
sound, but it rests on the slope, not the gap.

This is corroborated by the closest structural analog to our net. González et al. fit a tunable UNet
(depth L × width W, 32.9k–6.1M params) to the Chinchilla-style loss
`L(N,D) = [ (N_c/N)^(α_N/α_D) + D_c/D ]^α_D`, extrapolating test loss with Spearman ρ=0.950 and a
global data exponent α_D=0.249; **verification confirmed every number against the primary PDF.** Their
decisive quote: *"For models with a limited size, trained in large datasets, the second [model-size]
term of the equation can be removed."* Our 4.3M sits between their 3.9M and 6.1M grid points, and their
small nets "operate near the performance saturation point ... additional parameters yield diminishing
returns in this data-limited regime" ([González et al. 2026](https://arxiv.org/pdf/2606.06725)).
Caveat surfaced in verification: their validated data range is 500–10,000 images extrapolating to
10k, so our 100k–1M range is 10–100× beyond their demonstrated evidence, and their loss is pure BCE
segmentation vs our L1-dominant multitask loss. The recipe is loss- and scale-agnostic in principle,
but treat the transfer as strong-by-analogy, not proven at our scale.

The standard compute-optimal vision shape recipes do **not** apply here, because they assume unlimited
data. SoViT states verbatim: *"We also assume that data is unlimited so that there is no risk of
overfitting"* ([Alabdulmohsin et al. 2023](https://arxiv.org/pdf/2305.13035)). Using its width/depth
exponents to justify growing our net would be a category error given our data bottleneck.

### 2.2 What the scaling exponent is, and how to fit a law responsibly

Two robust, cross-domain priors:

- **The data-scaling exponent is a property of the task/data distribution, not the architecture.**
  Changing architecture or model size shifts the power-law *offset* but not the *slope*
  ([Hestness et al. 2017](https://arxiv.org/abs/1712.00409), verified — image classification with CNNs
  is one of their four in-scope domains). Two corrections from verification: (a) architecture moves the
  power-law **coefficient**, not the irreducible term E (E is fixed by the data distribution); (b) the
  finding is a hedged empirical regularity with known later counterexamples where architecture *does*
  change the exponent (Tay et al. 2022), not a proven law. Implication for us: changing UNet width
  should move our loss offset and (probably) leave the fitted data exponent unchanged. The further
  implication that **swapping the Stage-2 tracer** (vtracer vs rust) leaves the exponent unchanged is
  an extrapolation beyond the source — the tracer is a post-hoc classical stage, not the trained model
  being scaled — so treat it as plausible-by-analogy only.

- **Model an irreducible-loss term before estimating an exponent.** Fitting a pure power law to total
  loss (which has a non-zero asymptote) biases the apparent exponent downward.

  > **[REFUTED — corrected form]** The raw claim was: "you MUST include E (Chinchilla form
  > L=E+A/N^α+B/D^β); a pure power law underestimates the exponent by >2×; with only 3 datapoints you
  > cannot fit all 3 parameters (underdetermined)," cited to the Epoch AI Chinchilla replication. The
  > cited page supports **only** the refitted equation L=1.8172+482.01/N^0.3478+2085.43/D^0.3658 (a
  > billion-param LM measuring cross-entropy in nats). It does **not** state the "must include E,"
  > ">2× underestimate," or "underdetermined" claims — those are from the original Hoffmann et al.
  > paper and practitioner folklore, and the numbers do not transfer to our 4.3M UNet with L1 loss
  > (val ≈0.006). The defensible, regime-agnostic kernel: (1) model an irreducible term before
  > estimating a power-law exponent, and (2) fitting a 3-parameter nonlinear model from 2–3 noisy
  > scale points is unreliable — prefer fixing/borrowing the exponent and fitting the remaining terms,
  > or collect more points. Note "3 params from 3 points" is exactly-determined (zero residual DoF),
  > i.e. exact interpolation, not a validated fit.
  > ([Epoch AI](https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt))

  > **[REFUTED — corrected form]** The raw claim characterised our naive ≈0.10 exponent as
  > "LM-shallow" and inferred a "contradiction" implying a masked steeper reducible exponent, cited to
  > [Rosenfeld et al. 2020](https://arxiv.org/abs/1909.12673). Verification against that paper: its
  > fitted **data** exponents are ~0.74–1.01 (language, cross-entropy) and ~0.40–1.10 (image
  > classification, top-1 error). Our 0.097 is **far shallower than every exponent in the paper**, not
  > merely "LM-shallow." The "0.28–0.37" band the original claim invoked corresponds to that paper's
  > **model-size** exponents β, a different axis. Also, our L1 pixel-restoration loss is not directly
  > comparable to their error-rate/CE metrics (exponents are metric-dependent). A very shallow observed
  > slope is consistent with sitting near a non-zero floor that flattens the total-error curve, but
  > does **not** by itself imply a masked steeper reducible exponent; zero train/val gap only means
  > data- rather than capacity-limited.

The directly applicable fitting recipe for a fixed-model data-scaling study like ours comes from the
medical-segmentation UNet work: a **two-step strategy** — estimate the data-scaling exponent α_D
globally (as a task property), hold it fixed, then fit the remaining two parameters (A, D_c) from test
loss at several data points ([González et al. 2026](https://arxiv.org/pdf/2606.06725), verified quote).
One verification correction: **that paper's own loss (Eq. 1) has no irreducible floor term** — it
asymptotes to zero; the floor is a feature of a different cited paper, not this one. So if we want a
floor E in our fit we must add it deliberately.

On our naive number itself: 0.097 = ln(0.00782/0.00630)/ln(9.27) is arithmetically correct but
unreliable — it assumes E=0, uses only 2 points, and mixes L1+0.1×CE. **The diverged 500k run is our
missing third point.** Even after a clipped re-run, three points fit a 3-parameter law with zero
degrees of freedom (exact interpolation, not a validated fit), so a genuinely trustworthy law needs
≥4 converged scale points. Adding an intermediate size (e.g. 300k) is the cheapest way to begin
separating the floor from the reducible term.

Two hopeful counterweights to the shallow number, both **not adversarially verified** (treat as
directional):
- Dense/regression vision tasks can scale far steeper than language (MSE-loss data exponents ~1.2–2.3),
  so a restoration head is not *fated* to shallow returns — the exponent is worth measuring, not
  assuming ([Cadez & Kim 2025](https://arxiv.org/abs/2509.10000)). However, a separate finding places
  those steep exponents only on smooth physical-science targets and puts hard-perception/icon tasks at
  ~0.1–0.35 ([Bahri et al.](https://arxiv.org/pdf/2209.06640)) — our measured 0.10 sits in the hard
  regime, favouring the pessimistic read for *more-of-same* data.

### 2.3 Restoration-specific scaling: do we saturate early?

The closest task analog is Klug & Heckel, the only paper measuring scaling laws for CNN image
restoration (denoising/SR/MRI). **Verified:** restoration follows a **broken/two-regime power law** —
an initially steep slope that abruptly flattens at moderate data (UNet denoising coefficient 0.0048 up
to ~6k images, then 0.0019 beyond; "~100 images is already sufficient to train a decent image
denoiser"). Interpolating their laws suggests even millions of images would not much improve
reconstruction ([Klug & Heckel 2023](https://arxiv.org/abs/2209.13435)). Two verification caveats:
SwinIR (their second model) is a transformer not a CNN, and the quoted coefficients are denoising-
specific.

Crucially, **why this may not doom us:**

> **[REFUTED — corrected form]** The raw claim said their saturation is driven by an irreducible
> *measurement-noise* floor that our noise-free SVG targets lack, so "the floor is absent." Verification
> refuted the mechanism: their floor is set by the **corrupted/ill-posed input** (y=Ax+noise; for
> MRI/SR the forward operator is non-invertible so the floor is non-zero even at zero noise), **not**
> by target noise — their targets are clean too. Our cleanup task feeds the UNet lossy, information-
> destroying inputs (JPEG quantization, blur, added noise), so by their own theory it **also has an
> error floor of the same kind**; noise-free targets do not remove it. The likeliest reason we still
> gain at 927k with zero gap is that **we remain in the steep power-law regime** (signal-model error
> still large relative to the floor) because our signal/task differs and/or our synthetic damage is
> milder — not that the floor is absent. Zero train/val gap is a variance (no-overfitting) result,
> orthogonal to the asymptotic floor argument. ([Klug & Heckel 2023](https://arxiv.org/abs/2209.13435))

The actionable read: we are probably still on the steep part of a broken curve, but a flattening
likely exists ahead of us, and its onset depends on our damage severity (which we control). This is
another reason to (a) measure the curve with more points before a big mint, and (b) prefer diversifying
the data distribution over deepening more of the same.

### 2.4 Data: repeat vs mint, and composition

**Repeat is nearly free for a few epochs.** Muennighoff et al. (400+ runs, 10M–9B params, up to 1500
epochs) find repeating data up to ~4 epochs costs negligible loss vs fresh unique data; returns decay
with a half-life-style constant R_D*≈15 and are essentially dead by ~16–40 epochs; excess parameters
decay ~3× faster than repeated data ([Muennighoff et al. 2023](https://arxiv.org/abs/2305.16264)).
**Not adversarially verified**, and it is a language-model result. Two adjustments in our favour, both
directional: (1) vision routinely tolerates many more epochs than the LM 4-epoch heuristic, bounded by
**memorization**, whose onset scales *up* with dataset size — at 927k unique images with a small 4.3M
net we are far from that danger; (2) our damaged-render corpus can be **re-augmented** (fresh
JPEG/noise/blur draws per epoch), which raises the effective value of each pass above the static-text
figure. Net: running more epochs over the current 927k (with fresh augmentation each pass) is a safe,
cheap lever, and near-free up to ~4 nominal epochs.

**Composition likely beats count.** Data pruning/selection can beat power-law scaling toward
exponential error decay: when data is abundant keep the *hard/informative* examples; when scarce keep
the *easy/prototypical* ones ([Sorscher et al. 2022](https://arxiv.org/pdf/2206.14486), not
adversarially verified). Our residual errors concentrate at region boundaries and in known gaps
(typography, fine-line, gradients). So minting for **coverage of gaps** and/or a hard-example selection
metric is predicted to steepen our curve more than minting additional icons — consistent with the
shallow ≈0.10 exponent on our current icon-heavy tier. When the corpus mix changes, **re-fit the
intercept**: scaling gradients transfer across data distributions but absolute loss does not
([González et al. 2026](https://arxiv.org/pdf/2606.06725), the CAMUS→CEUS transfer preserved slope but
had a significant intercept offset).

### 2.5 Capacity: width vs data (grow data first)

Converging evidence that our 4.3M net is **not** the binding constraint:
- In the data-limited regime the model-size term vanishes from the UNet scaling law (§2.1, verified).
- Compute-optimal shape work reached SOTA-matching dense-prediction quality with 10×–240× fewer params
  (a 1.5M-param UNet outperformed a 14.7M SOTA) ([González et al. 2026](https://arxiv.org/pdf/2606.06725));
  SoViT matched a 2.5×-larger ViT at half inference cost
  ([Alabdulmohsin et al. 2023](https://arxiv.org/pdf/2305.13035)).
- If we *do* ever grow capacity, pure width is the lever to grow **last**: SoViT ordering is
  MLP/hidden-dim > depth > width (s_MLP≈0.6 > s_depth≈0.45 > s_width≈0.22), and input **resolution** is
  a co-lever for a 256px dense net (EfficientNet: single-dimension scaling saturates early; compound
  scaling keeps improving). All **not adversarially verified**; note SoViT and older CNN evidence
  partially conflict on width-for-small-nets, so a width doubling is at best ambiguous.
- The product is **inference-heavy** (runs per user image) and Stage-2 tracing is the real quality
  ceiling (rust 1.79 vs vtracer 1.52 on identical cleaned input), so a smaller-net/more-data posture is
  also inference-cost-optimal ([Beyond-Chinchilla](https://arxiv.org/html/2401.00448v2), not verified).

**Guardrail:** if we ever fit our own L(N,D) to choose width-vs-data, sparse/IsoFLOP fits
systematically **underestimate** the optimal size and report misleadingly narrow confidence intervals;
use several runs, model the noise, and don't trust a single parabola
([sparse-grid fitting bias](https://arxiv.org/pdf/2603.22339), not verified). Our one diverged run is
exactly the kind of noisy point that wrecks such a fit.

### 2.6 Stability: the 500k divergence, clipping, precision

The divergence (train loss rising from epoch 1, never recovering) has two documented, non-exclusive
causes:

1. **The bf16 divergence lottery.** A controlled nanoGPT study saw 18/188 (~10%) bf16 runs diverge on
   identical config differing only by seed, vs 0/70 for TF32
   ([Zhang et al. 2024](https://arxiv.org/html/2405.18710v2), not adversarially verified). Different
   data scale ⇒ different data order ⇒ different float-accumulation order ⇒ different susceptibility.
   This alone explains 500k diverging while identical-config 1M was clean, **without** a config bug.
2. **Missing gradient clip.** The standard, cheap guard.

> **[REFUTED — corrected form]** The raw claim asserted "global-norm ~1.0 is the standard fix,"
> "healthy runs clip 5–20% of batches," and "force fp32 for softmax and final loss," cited to
> [AdaGC (arXiv:2502.11034)](https://arxiv.org/html/2502.11034v3). Verification: that paper studies
> 1.3B–10B LLM transformers (not ~4M CNNs); it argues fixed global-norm-1.0 is **insufficient** and
> proposes adaptive per-tensor clipping instead of endorsing 1.0 as the fix; it contains **no**
> "5–20% clipped in healthy runs" statistic (and frames instability as driven by **rare** spikes, the
> opposite of routine clipping); and its precision recommendation is **fp32 RMSNorm/normalization**,
> not fp32 softmax/loss. Also our failure is persistent divergence, not the transient spike-then-
> recover it studies. **Corrected takeaway:** gradient clipping is sensible cheap general practice and
> worth adding, but do not cite this paper for a specific clip threshold, a clip-rate, or an fp32-
> softmax rule, and do not assume its billion-param conclusions transfer.

What the broader literature **does** support for a small vision net (mostly not adversarially verified,
flagged where so):
- **Global-norm gradient clipping** at ~1.0 (common vision default; a data-driven pick is the 90–95th
  percentile of the unclipped grad-norm trace). Clip-by-norm preserves update direction; log per-module
  norms so an imbalanced head (our CE tower) firing a spike is visible.
- **Keep fp32 master weights + fp32 optimizer state (m,v) + fp32 loss/softmax/norm accumulation** under
  bf16 autocast, and **upcast the CE-head logits before log-softmax**; an in-place bf16 softmax
  silently underflows and a run can look fine for thousands of steps then NaN
  ([mixed-precision practice](https://zeroentropy.dev/concepts/mixed-precision-training/), not verified).
- **Short LR warmup (1–5% of steps).** Adam's bias-corrected second moment is unreliable for the first
  steps (v̂₁=v₁/(1−β₂) inflates the effective step), so a no-warmup start at peak LR can "catapult" the
  loss from epoch 1 — exactly our symptom
  ([warmup](https://mbrenndoerfer.com/writing/learning-rate-warmup-linear-duration-large-batch-training);
  [loss catapult](https://arxiv.org/html/2406.09405v1), neither adversarially verified).
- **StableAdamW** (AdamW + AdaFactor-style update clipping) empirically beats plain norm-1 clipping on
  both spike removal *and* final accuracy at zero extra memory, on large vision models
  ([StableAdamW](https://ar5iv.labs.arxiv.org/html/2304.13013), not verified) — a low-cost optimizer
  swap worth trying once basic clipping is in.
- **Raise AdamW ε 1e-8 → 1e-6** as cheap insurance (matters mainly if optimizer state is stored in
  bf16; largely neutralised if m,v are fp32).
- **Run the whole 4.3M net in TF32/FP32.** For a net this small on modern GPUs this is cheap and
  removes the bf16 lottery outright — the most decisive stability fix if reproducibility matters more
  than a small speed gain.
- **Weight EMA (decay 0.9998–0.9999)** is a rare *data-free* quality+stability lever: it damps weight
  oscillations toward flatter minima and can allow higher LR. Given we are data-constrained with zero
  gap, it is close to a free quality gain ([EMA](https://arxiv.org/html/2411.18704v1), not verified).

### 2.7 Optimizer, LR, batch, schedule

- **LR-vs-batch is the SQUARE-ROOT rule for AdamW, not linear.** If we raise batch 16→64 (4×) for
  speed, LR should go ≈4.2e-4 → ≈8.4e-4 (×2), not ×4. Adam's optimal LR is also **non-monotonic** in
  batch (the "surge": rises to a peak at a critical batch B_noise then falls), so blindly scaling LR up
  overshoots once batch approaches B_noise ([Surge](https://arxiv.org/html/2405.14578v2), not verified).
- **Batch 16 is already in the quality-optimal small-batch band** (best test performance and widest
  stable-LR window at m=2–32; [Masters & Luschi 2018](https://arxiv.org/abs/1804.07612), not verified),
  and well below the critical batch size for vision. Raising batch buys **wall-clock only**, forfeits
  small-batch regularization, and **narrows** the stable-LR window (a divergence risk). Raise batch
  only if training time is the bottleneck.
- **Cosine is horizon-dependent.** A cosine tuned for a long horizon and stopped early is substantially
  worse than one tuned to the actual step budget, on identical data
  ([Kempner anytime-pretraining](https://kempnerinstitute.harvard.edu/research/deeper-learning/anytime-pretraining-horizon-free-learning-rate-schedules-with-weight-averaging/),
  not verified). **Set cosine length == planned steps every run.** If we expect to retrain repeatedly as
  the corpus grows, a Warmup-Stable-Decay or constant-LR + weight-averaging schedule lets one stable
  checkpoint be branched with a short decay to match a full cosine without pre-committing the horizon
  ([WSD](https://arxiv.org/abs/2410.05192), not verified).

### 2.8 The two-task loss (CE weight and the auxiliary head)

- **Do not build a multi-task optimizer.** Fixed-weight scalarization with a tuned weight matches or
  beats GradNorm/PCGrad/MGDA/uncertainty weighting, which merely land on the scalarization Pareto front
  ([Xin et al. 2022](https://arxiv.org/pdf/2209.11379);
  [Kurin et al. 2022](https://arxiv.org/abs/2201.04122)). Both **not adversarially verified**, but
  mutually corroborating.
- **LR tuning dwarfs the weighting choice.** Xin et al.: variance from a sparse LR grid is 6–7× the
  random-seed variance and orders of magnitude larger than any MTO effect. So any weight ablation must
  hold LR/regularization tuning fair.
- **Uncertainty weighting is not a safe default** — it overfits and has ~¼-of-training "update inertia";
  if adaptivity is wanted, UW-SO (one temperature) is better. Kendall drop-in formulas if ever needed:
  CE → (1/σ²)L+log σ; L1 → (1/σ)L+log σ (the log term blocks the trivial σ→∞ collapse).
- **The aux head's usual justification barely applies to us.** Auxiliary tasks help most by reducing
  *overfitting* in the low-data regime — but we have zero train/val gap, so that mechanism has little to
  bite ([NeurIPS 2020 aux-task](https://papers.neurips.cc/paper_files/paper/2020/file/4f87658ef0de194413056248a00ce009-Paper.pdf)).
  In our regime the CE head's value is more likely *representational* (boundary/region-aware features),
  and it may instead compete for our 4.3M-param capacity. A free per-step diagnostic is **gradient
  cosine similarity** between the L1 and CE gradients on the shared encoder: if negative, the aux task
  is fighting restoration that step ([Du et al. 2018](https://arxiv.org/abs/1812.02224)).
- **Cheap correct recipe for the CE weight:** a 5-point sweep {0, 0.03, 0.1, 0.3, 1.0} at the smallest
  (100k) scale, selected on val **restoration** loss (L1/ΔE), not joint loss. **weight=0 is the load-
  bearing ablation** — it directly answers whether the segmentation head helps the primary task at all.
  Scalarization theory says a good fixed weight is the ceiling, so this small sweep is sufficient; no
  per-step optimizer needed.

### 2.9 Boundary quality and the label head (fix the metric first)

- **Our headline metric hides the important pixels.** Global 95.97% label accuracy is dominated by easy
  interior pixels; interior predictions are more reliable than boundary predictions
  ([SegFix, Yuan et al. 2020](https://arxiv.org/abs/2007.04269)). Boundary pixels decide vectorizer
  curve placement. **Switch evaluation to a Boundary F-score in a narrow band around region edges**, and
  judge any loss/head change on that, not global loss.
- **Cheapest high-fit intervention: the original UNet distance-weighted CE** (per-pixel weight map
  w(x)=w_c(x)+w0·exp(−(d1+d2)²/2σ²), w0=10, σ≈5px) — a drop-in multiplier on our existing CE, one-time
  offline distance-map precompute, no architecture change, no new stability failure mode
  ([Ronneberger et al. 2015](https://www.sfu.ca/~kabhishe/posts/posts/summary_miccai_unet_2015/)).
- **Exotic boundary losses buy little for us.** On natural-image (non-pathological) data, loss choice
  moves mIoU only 1–2.55%; Kervadec boundary loss's headline +8% is specific to highly-imbalanced
  binary medical tasks, needs pairing with a region loss + an α schedule, and risks exploding gradients
  in multi-class; Lovász-softmax needs CE-pretrain-then-finetune and its dataset-mIoU is batch-size/
  class-count dependent (a real caveat at batch 16 / 16 classes)
  ([loss survey](https://arxiv.org/html/2312.05391v1)). Keep CE in the mix — standalone boundary losses
  underperform.
- **Alternative that changes no training at all: SegFix-style snapping** — replace each boundary pixel's
  label with a learned-offset interior pixel's label, bakeable into the Rust tracer at trace time
  ([SegFix](https://arxiv.org/abs/2007.04269)). This directly targets our measured failure mode.

*(One search result — a NAFNet channel-attention finding tagged summary "test" — is off-topic to
Stage-1 cleanup training and is excluded.)*

---

## 3. Recommendations

Confidence reflects both source strength and how well the source's regime matches ours. Costs assume a
"unit" = one full 927k-scale run at current settings.

| # | Recommendation | Confidence | Cheapest confirming experiment (rough cost) |
|---|---|---|---|
| 1 | **Add stability hardening to every future run before any scaling spend:** global-norm grad clip (~1.0, or 90–95th pct of unclipped trace), fp32 master weights + optimizer state + loss/softmax/norm accumulation, upcast CE logits before log-softmax, short LR warmup (1–5% steps). | **High** — multiple independent sources; the diverged run is the textbook symptom; changes are near-free. | Re-run the 500k config with clip + warmup; success = no epoch-1 loss rise. ~0.5 unit. |
| 2 | **Prioritise data over width for the next spend.** Do not grow base width 48→64; instead add epochs (fresh per-epoch augmentation) and/or mint diverse data. | **High** — the closest UNet-scaling analog (verified) plus our own zero-gap-with-falling-loss signature both put us on the data-dominated branch. | Already have it: our 100k→927k slope at fixed 4.3M is the confirmation. A 2× width run at 100k as a null check = ~0.3 unit. |
| 3 | **Mint for gap coverage (typography, fine-line, boundaries), not more icons; consider hard-example pruning.** | **Medium** — pruning-beats-power-law and composition results are directional (not adversarially verified); but our shallow ≈0.10 icon exponent strongly implies more-of-same is low ROI. | Mint a 100k gap-focused shard, fold into the 927k corpus, compare end-to-end ΔE delta vs a 100k icon shard. ~0.4 unit. |
| 4 | **Recover the 500k point and add a 300k point to build a real data-scaling curve (≥4 converged scale points before any large mint).** | **Medium** — two-step fitting recipe is verified for the analog task, but validated only to ~10k images / BCE loss; extrapolation to our range is unproven. | Run 300k + clipped 500k; fit L(D) with a floor term; check the fit predicts 927k within a few %. ~1.3 units. |
| 5 | **Sweep the single CE weight {0, 0.03, 0.1, 0.3, 1.0} at 100k, select on val restoration loss; weight=0 is the ablation that tells us if the aux head earns its capacity.** | **High** — scalarization-beats-MTO is well-supported; the sweep is cheap and decisive. | The sweep itself: 5 × 100k runs ≈ 1.5 units, fewer if warm-started. |
| 6 | **Switch label-head evaluation to a Boundary F-score band; if boundaries need work, try distance-weighted CE (w0=10, σ≈5px) before any exotic loss.** | **High** for the metric switch (global accuracy provably hides boundaries); **Medium** for the exact loss (natural-image loss gains are small). | Compute BF-score on the current val set (no training). ~0. Distance-weighted CE at 100k = ~0.3 unit. |
| 7 | **Add weight EMA (0.9998–0.9999) — a data-free quality/stability lever.** | **Medium** — well-established in vision practice but not adversarially verified here. | Enable EMA on the next planned run; compare EMA vs raw checkpoint on end-to-end ΔE. ~0 marginal. |
| 8 | **Keep batch at 16; if training time forces a raise, use the AdamW square-root LR rule (16→64 ⇒ LR ×2, not ×4) and set cosine length == planned steps.** | **Medium** — square-root rule and small-batch-optimality are well-argued but not adversarially verified. | If batch is ever raised: one batch-64 run at LR ×2 vs ×4, compare val loss. ~0.6 unit. |
| 9 | **Consider running the whole 4.3M net in TF32/FP32 to eliminate the bf16 divergence lottery outright.** | **Medium** — the ~10% bf16-divergence figure is a single (unverified) controlled study, but the net is small enough that the speed cost is minor. | One FP32 run at a previously-diverging config; success = reproducible convergence across seeds. ~0.5 unit. |

### Boxed run plan — the next three runs

**Run A (stability + curve recovery, ~1.8 units).** Re-run 500k **and** a new 300k, both with the full
stability kit from Rec 1 (clip + warmup + fp32 accumulation + EMA), cosine length == planned steps.
*Predicted outcome:* both converge cleanly (no epoch-1 rise); combined with 100k/927k we get four
converged scale points. Fit L(D)=A/D^b+E with the two-step method. *Predicted:* b in the 0.10–0.25
band, non-trivial floor E, extrapolation says a 10× icon mint buys ≤~19% val-loss reduction — which
motivates Run C over a naive big mint.

**Run B (CE-weight sweep + metric fix, ~1.5 units).** At 100k, sweep CE weight {0, 0.03, 0.1, 0.3,
1.0}, evaluate on val **restoration** loss and the new **Boundary F-score** band. *Predicted outcome:*
weight=0 is within seed-noise of 0.1 on global loss (aux head's gain is boundary-local, not global), so
the decision hinges on the BF-score — where a small positive CE weight (≈0.1) should show a modest
boundary-F improvement. If not, drop the head and reclaim capacity/compute.

**Run C (targeted mint, ~1× the mint + one 927k+shard run).** Mint a gap-focused shard (typography,
fine-line, boundary-rich) and either fold it in or up-weight it; re-fit the intercept since the mix
changed. *Predicted outcome:* larger end-to-end ΔE improvement per image than an equal-size icon shard,
and steeper local scaling on the gap categories. This is the run most likely to move the 1.52 bench,
because Stage-1's remaining error is concentrated exactly where the current icon corpus is thin.

---

## 4. Open questions the round could not settle

1. **Our true data-scaling exponent and floor.** Two points (one loss-metric, mixed L1+CE) cannot
   identify (E, coefficient, exponent). Even three points give an exact-interpolation fit with zero
   degrees of freedom. Resolvable only by Run A plus at least one more converged scale point.
2. **Where our broken-power-law knee sits.** Klug & Heckel guarantee restoration curves eventually
   flatten and show a floor set by the *corrupted input*, but their knee (~6k images) and damage
   distribution differ from ours; we do not know how far ahead our flattening is, and it depends on our
   synthetic-damage severity — which we can tune but have not characterised.
3. **Whether dense-regression steep-scaling (up to ~2.3) applies to us or only to smooth targets.** The
   two unverified sources disagree; our measured 0.10 currently favours the pessimistic (hard-
   perception) read, but this was not resolved against a matched-regime source.
4. **Actual bf16-divergence rate at our scale.** The ~10% figure is from one small-LM study; we have
   n=1 divergence. We cannot yet attribute the 500k failure to the bf16 lottery vs the missing clip vs
   both. Run A (clip only) vs a FP32 control would separate them.
5. **Does the CE aux head earn its capacity in our zero-gap regime?** The overfitting-reduction rationale
   does not apply; the representational benefit is plausible but unmeasured. Run B's weight=0 arm settles
   it, but only against the current corpus mix — the answer may flip once gap data (Run C) is added.
6. **Is the true quality bottleneck even in Stage-1?** On identical cleaned input, rust 1.79 vs vtracer
   1.52 and perfect-input ceilings (vtracer 1.28) suggest Stage-2 tracing, not cleanup, may cap the
   product. None of these Stage-1 training spends address that; a parallel question is whether tracer
   work outranks any cleanup run for moving the end-to-end bench.
7. **Transferability of the LM repeat-vs-mint constants** (4-epoch free, R_D*≈15) to our re-augmentable
   vision corpus. Directionally we can run more epochs than the LM heuristic, but the exact safe-epoch
   count for our data is unmeasured.
