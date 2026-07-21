# Intent Priors — Master Synthesis

Date: 2026-07-21. Reader: the project owner sequencing the next month.
Question answered: **where should "intent" live in this pipeline, and in what build order?**
"Intent" = the owner's vision that the system recognizes *what a thing is* (a star, type, a
business card) and feeds that knowledge into cleanup — spanning implicit (scale/data), explicit
conditioning (recognizer features into the UNet), generative priors (diffusion), and Stage-2
geometric intent (primitive/symmetry fitting in the tracer).

Source theme reports on disk in this folder — read for detail:
[`conditioning.md`](./conditioning.md) (A: semantic conditioning of the UNet),
[`primitives.md`](./primitives.md) (B: parametric primitive/shape fitting in Stage 2),
[`document-intent.md`](./document-intent.md) (C: document/layout-archetype priors),
[`grounding.md`](./grounding.md) (D: what fits OUR scale + product constraint).

This round assumes the prior round's prescription
([`../2026-07-20-deep-research/README.md`](../2026-07-20-deep-research/README.md): composition
mint, CE sweep, engine noise-robustness) and interleaves with it in §4.

Cost convention (unchanged from the prior round): rented mid-tier GPU ~$1–1.5/GPU-hr; one full
927k-scale Stage-1 run ≈ $15–30. Engineering-only tickets carry ~$0 compute.

---

## 1. Verdict per theme

**A — Semantic conditioning of the UNet (`conditioning.md`).** The owner's "recognize the star,
feed it into cleanup" is a solved architectural pattern, and the pipeline already holds the signal
it needs: the 16-class per-pixel region head. The mechanism is **spatial feature modulation**
(SFT-GAN / SPADE: emit per-location `γ⊙F + β` from the soft region map onto a few decoder layers),
which beats input-channel concatenation by large margins — and concatenation is *provably zeroed*
by normalization on uniform-label patches, i.e. the exact flat fills the tracer cares about most.
It is cheap (~2 params per modulated map, resolution-independent), transfers architecturally, and
carries a built-in graceful-degradation property (unknown region → neutral γ/β → unconditioned
baseline). Two sobering caveats: **payoff shrinks near the ceiling** (the fair analog, denoising on
a strong baseline, gave +0.25 dB — and our engine already ceilings near ΔE 1.2 with zero
train/val gap), and **published wins are perceptual and often come *with* a fidelity drop**, so
every rung must be gated on full-reference ΔE, never on FID/LPIPS/user-preference (no-reference
metrics provably cannot detect hallucination). The real hallucination severity lives in the
*generative* rungs, not in an L1-anchored 4.3M UNet.

**B — Parametric primitive & shape fitting in Stage 2 (`primitives.md`).** Geometric intent belongs
in Stage 2 as **error-gated primitive substitution, never unconditional snapping**. Fitting is the
easy part (use Taubin or Hyper circle fits, never biased Kåsa); **acceptance is the fidelity-critical
part**. The a-contrario / Number-of-False-Alarms (NFA) framework is the cleanest gate for
"never invent structure" — it accepts a primitive only when it is statistically unexplainable as
noise, is provably insensitive to its threshold, and its guarantee is specifically about false
*detections*. (MDL is a *secondary* gate only, and only with a conservative σ²: its simplicity bias
actively *licenses* correcting an asymmetric star, so used naively it is a shape-regularizer, not a
guard.) The route is edge/arc-grouping → algebraic fit → NFA gate (working parameterless templates
exist: ELSD, EDCircles), not Hough voting (does not scale past circles). The strongest practitioner
signal in the whole corpus: **vectorizer.ai does exactly this fitting but ships it OFF by default**
(output flattens to ordinary curves). Our baselines (vtracer, potrace) do *no* primitive
substitution at all — so this is a genuine, unoccupied lever, cheap to prototype (scikit-image ships
RANSAC `CircleModel`/`EllipseModel` today), with the entire risk in the acceptance policy.

**C — Document/layout-archetype priors (`document-intent.md`).** The literature points **away** from
bolting a "this is a business card → clean up better" classifier onto the UNet. **No published
pipeline shows archetype recognition improving downstream raster cleanup** — the datasets that model
our exact distribution (Crello, PosterLayout, CanvasVAE) use archetype knowledge only for
*generation*, never for restoration; this build is research, not adoption. Explicit layout priors
also carry the exact forbidden failure mode (PosterLayout's designer-imitation objective pushes
toward "generic-correct"), and off-the-shelf detectors are too coarse (page-region labels at
~70–89 mAP tell you "a figure is here," not "a five-pointed star"). The transferable lever is
**in-domain self-supervised pretraining + data diversity** (DiT, DocLayNet), i.e. mine the NAS —
not an archetype head. If document-level context is ever added, add it *architecturally* (wider
receptive field / global-local coherence à la Iizuka: enforces consistency, invents nothing), not
semantically. Falsify the archetype rung cheaply (a Donut-style tile-type token) before investing.

**D — What fits OUR scale + product constraint (`grounding.md`).** Explicit conditioning
(spatial-FiLM) is **proven at our exact ~1–20M-param scale and is cheap**; the real cost is the
*knowledge source*, not the injection (off-the-shelf frozen CLIP *fails* and the working controller
is ~125M params — dwarfing a 4.3M net), which is precisely why our own in-domain 16-class head is
the right condition signal and self-conditioning is the cleanest fit. Gains are strongly
degradation-specific (large where degradation destroys structure, near-zero for mild noise) and none
of the evidence is in-domain, so magnitude does not transfer — only an ablation settles which bucket
our pixelated-vector-render cleanup falls in. Generative priors *structurally* invent (perception-
distortion tradeoff), are not a small-net technique, and are usable only clamped per-region.
Bounding invention is a **gate/policy problem layered on the model** (Efficient-RANSAC acceptance,
SelectiveNet abstention, uncertainty pass-through), reinforced by the prepress norm our customers
already trust (PitStop: report-then-approve, auditable). Recommended build order:
**region-head hardening → spatial-FiLM self-conditioning → Stage-2 gated primitive fitting →
per-region abstention → (only if justified) clamped generative refinement.**

**Convergent conclusion.** All four themes agree: intent lives at **two safe homes** — implicit
conditioning off our *own* region head inside Stage 1 (cheap, low-risk, possibly low-payoff), and
gated geometric fitting in Stage 2 (the unoccupied lever, all risk in acceptance policy) — with a
shared guardrail spine (soft signals, graceful fallback, ΔE gating, a deliberate-asymmetry
regression set, report-then-approve surfacing). The document-archetype and generative rungs are
where hallucination risk concentrates and evidence thins; both are deferred/falsified, not built.

---

## 2. THE INTENT LADDER

Sequenced cheapest/safest → most powerful. Each rung is gated: you do not climb until the rung
below has *paid on ΔE* on the 24-image bench **and** passed the deliberate-asymmetry regression set.
The guardrail spine (Rung 0) is a $0 prerequisite baked into every rung above it.

### Rung 0 — Guardrail spine + region-head hardening  ·  ~$0 GPU, ~1–2 eng-days  ·  PREREQUISITE

- **What it is, concretely.** Three pieces of infrastructure that make every rung above measurable
  and safe: (a) build a **deliberate-asymmetry regression set** — 20–40 curated marks with
  intentional idiosyncrasy (asymmetric stars, hand-drawn wobble, bespoke letterforms, off-center
  logos) mined from the NAS, held out as a standing check that no rung "corrects" intent; (b) switch
  region-head eval from the misleading 95.97% pixel accuracy to a **boundary F-score band** (from the
  prior round's recipe theme) so the pixels that matter are what we score; (c) settle the **CE
  weight=0 sweep** (also from the prior round) — because conditioning in Rung 1 is off the region
  head, and if the head does not earn its capacity, conditioning off it is moot.
- **Expected benefit.** No ΔE by itself — it is the measurement and safety layer. Its value is that
  it *gates* every rung above and turns "did we invent anything?" from a vibe into a number.
- **Hallucination risk / guardrail.** This *is* the guardrail. The asymmetry set is the single
  artifact that catches intent-invention (aggregate ΔE catches it only diffusely).
- **Validating experiment.** The asymmetry set + boundary-F eval + weight=0 arm are themselves the
  experiment; the go/no-go for Rung 1 is "does the region head measurably help (weight>0 beats
  weight=0)?" If the head is dead weight, stop here and skip to Rung 2 (Stage-2 geometry).

### Rung 1 — Spatial-FiLM self-conditioning of the UNet  ·  ~$15–30 GPU (one retrain), ~2–3 eng-days

- **What it is, concretely.** Feed the existing 16-class region-label *logits* (soft, not argmax)
  back into the RGB-repair decoder as **spatial SFT/SPADE modulation** — a 4-layer 1×1-conv condition
  net emitting per-location `γ⊙F + β` on a few decoder layers. This is the literal implementation of
  "recognize it's a star → feed that into cleanup." Never concatenate the map as an input channel
  (provably zeroed on flat fills). Fold this into a retrain that is *already happening* (the gap-mint
  run, prior-round Spend 3), not a standalone run.
- **Expected benefit.** **Small — budget order-tenths of ΔE, not a rung change.** We are in the
  near-ceiling, mild-damage regime where frozen-feature priors add least (fair analog: +0.25 dB). The
  honest framing is that this rung is *also a diagnostic*: if region-conditioning barely moves ΔE, we
  are in the low-gain bucket and should redirect effort to Stage-2 geometry (Rung 2).
- **Cost.** Near-free in parameters; the cost is one training run, which piggybacks on an already-
  planned retrain. ~2–3 eng-days to wire the SFT layers and the soft/confidence-gated path.
- **Hallucination risk / guardrail.** **Low** for an L1-anchored 4.3M net (the pixel loss structurally
  limits its ability to paint wrong content), but non-zero via wrong-recognition → wrong-restoration.
  Guardrails: feed **soft probabilities + raw image** (degradation-robust, not brittle hard tags);
  **confidence-gated graceful fallback** to unconditioned cleanup on low-confidence regions;
  **keep the reconstruction loss, add no GAN/adversarial term** (perception-distortion tradeoff);
  **gate on ΔE, never perception.**
- **Validating experiment.** Add SFT layers, retrain on the current corpus, compare end-to-end ΔE on
  the 24-image bench vs the dual-head baseline, *and* run the Rung-0 asymmetry set to confirm no
  intent-correction. Ablate soft-vs-hard conditioning and the confidence fallback in the same run
  (marginal cost). Climb to Rung 2 regardless of Rung-1 outcome (they are independent homes); a null
  Rung-1 result *redirects* budget toward Rung 2, it does not block it.

### Rung 2 — Stage-2 gated primitive fitting  ·  ~$0 GPU, ~1–2 eng-weeks (Rust)

- **What it is, concretely.** In the classical tracer, per planar-map region, fit candidate
  primitives (start with the circle: **Taubin/Hyper**, never Kåsa; then rounded-rect; defer star) and
  substitute the freeform Schneider bezier path **only when an acceptance gate fires**. Edits must be
  **topology-aware** — operate on shared planar-map edges, not per-region, or they reintroduce
  gaps/overlaps (the pitfall vectorizer.ai explicitly names; our engine is already a planar map).
  Ship **OFF by default**, opt-in per job, mirroring the market leader.
- **Expected benefit.** The genuinely **unoccupied lever** relative to our baselines (vtracer/potrace
  do zero primitive substitution). On flat-fill logo/wordmark content a correctly-fitted circle/rect
  is ΔE≈0 by construction on that region, and consistency gains compound across a logo. Prototype-
  cheap: scikit-image RANSAC `CircleModel`/`EllipseModel` proves the concept before any Rust.
- **Cost.** ~$0 compute; the cost is Rust engineering — days for a scikit-image prototype and the
  first primitive+gate, ~1–2 weeks for topology-aware substitution and per-primitive NFA null models.
- **Hallucination risk / guardrail.** **This is the highest-risk safe rung** — it is precisely the
  move the competitor gates OFF, because snapping a deliberately-asymmetric star to a regular one is
  the exact forbidden failure. Guardrails, layered: **NFA acceptance gate** (ε=1, ≤1 false detection
  per image — the guarantee is about non-invention); a **two-sided test** (primitive residual/support
  must beat the freeform Schneider fit by a margin *and* clear a confidence floor, ~99%, à la
  Efficient-RANSAC); **conservative σ²** if MDL is used at all (its simplicity bias otherwise licenses
  correction); a **unit test on the Rung-0 asymmetry set asserting no snap occurs**; and **surface
  substitutions as reviewable/undoable diffs** (PitStop report-then-approve norm), never silent auto-
  correction.
- **Validating experiment.** Prototype: scikit-image RANSAC per region + a hand-derived NFA gate; run
  ELSD/EDCircles on the bench to see how many valid vs spurious primitives emerge parameter-free.
  Measure ΔE on the 24-image bench with substitution on vs off, and assert the asymmetry set produces
  zero snaps. Do NOT climb to symmetry (Rung 4) until per-primitive NFA null models are validated for
  the primitives that fire most on our distribution.

### Rung 3 — Per-region abstention / restore-vs-leave-alone head  ·  ~$15–30 GPU (rides a retrain), ~2–3 eng-days

- **What it is, concretely.** A **SelectiveNet-style** selection head on the UNet: a per-region "clean
  this vs pass it through untouched" gate with an owner-tunable coverage dial, percentile-calibrated
  on validation. Route the known content gaps (typography, fineline, gradients — where the model is
  weakest and most likely to invent) to **pass-through by default**, using epistemic uncertainty as
  the OOD flag.
- **Expected benefit.** Converts our *known weaknesses* from a hallucination liability into an
  explicit, safe fallback — the model declines to "improve" content it has not learned, rather than
  confidently mangling it. Complements both Rung 1 (which region gets conditioned) and Rung 2 (which
  region is eligible for a snap).
- **Cost.** One selection head + calibration, folded into a retrain; ~$0 marginal compute if it rides
  Rung 1's or the gap-mint run.
- **Hallucination risk / guardrail.** This rung *is* a guardrail (it reduces invention). Its own risk
  is mis-calibration — abstaining too much (no cleanup) or too little (invents anyway). Guardrail:
  calibrate the dial against a **labeled OOD probe set** and confirm abstained regions correlate with
  known-hard content, not benign hard edges.
- **Validating experiment.** Add the selection head, calibrate the threshold to a target coverage,
  and check that abstained regions align with typography/gradient/fineline (the known gaps) on a
  held-out probe. Ship the coverage dial conservative for print jobs.

### Rung 4 — Symmetry detect-as-voting + minimal constrained snapping  ·  ~$0 GPU, ~1–2 eng-weeks (HIGHER RISK — gate hard)

- **What it is, concretely.** Detect mirror/rotational symmetry as a **peak in transformation space**
  (Loy-Eklundh feature-vote, or Symmetrization's SVD reflection), use cluster tightness as a
  confidence score, then apply **minimal constrained snapping** (GlobFit-style: enforce the fewest
  non-conflicting relations, balancing data-fidelity against regularization) only when supported.
- **Expected benefit.** Consistency gains on genuinely-symmetric marks (many logos), matching the
  leader's symmetry modelling. Marginal over Rung 2 and materially riskier.
- **Hallucination risk / guardrail.** **High and explicit** — every method in this lineage warns it
  can erase deliberate asymmetry (Symmetrization loses the gecko's toes; "does not respect the
  semantics of the shape"). Guardrail: gate on **cluster-tightness confidence AND a data-fidelity
  term**, prefer the fewest relations (Occam), asymmetry-set regression, and surface as a diff.
- **Validating experiment.** Loy-Eklundh axis voting on ~5 wordmarks; verify a deliberately-asymmetric
  mark scores *low* confidence and is left untouched. Do not enable auto-commit; propose candidates.

### Rung 5 — Clamped generative refinement  ·  significant GPU, DEFER

- **What it is, concretely.** A DiffBIR-style fidelity-stage-then-optional-generative-refinement on a
  per-region dial, defaulted near-zero for logo/type, wrapped in a **DDNM null-space / Explorable-SR
  consistency projection** so the prior can only add unobserved detail, never overwrite present pixels.
- **Expected benefit.** Perceptual "cleanness" on severely-degraded content — but perceptual, not ΔE,
  and against our print-fidelity constraint for the core distribution.
- **Hallucination risk / guardrail.** **This is where the dramatic hallucination evidence lives** (all
  of it from billion-param diffusion priors). Consistency alone is necessary-but-insufficient (PULSE
  hallucinates identity while satisfying downscaling consistency). Not a small-net technique; scale-
  mismatched for us now.
- **Validating experiment.** **Defer.** Revisit only if Rungs 1–4 plateau above the ΔE target; if ever
  piloted, wrap a small prior in null-space projection and measure ΔE + the asymmetry set on the bench
  before any broader use.

---

## 3. What NOT to build (and why)

- **A supervised document-archetype classifier feeding the UNet.** The "this is a business card →
  clean up better" link is *entirely absent* from the literature; the datasets that model our
  distribution use archetype knowledge only for generation, and layout priors push toward generic-
  correct (the forbidden failure). Falsify cheaply first — a Donut-style 4–8-way tile-type token
  (labels auto-derived from NAS folder structure, ~1 GPU-day) — but do not *build* the rung unless
  that falsification test shows ΔE actually moves. (`document-intent.md` §2.1, §2.7; Rec 1, 4)

- **An off-the-shelf foundation encoder (CLIP/SAM) at inference.** Frozen CLIP *fails* to improve
  restoration (26.7% vs 99.2% with a trained controller), and the controller that works (~125M) dwarfs
  a 4.3M net. If a stronger prior is ever wanted, **distill at training time only** (SAM-prior style,
  identical inference cost). Our own region head is the free in-domain signal. (`grounding.md` §2.2)

- **A GAN / adversarial / perceptual-loss term in Stage-1 cleanup.** The perception-distortion tradeoff
  makes fidelity and perceptual-cleanness provably exclusive past the bound; an adversarial loss moves
  us off the fidelity side — the exact axis where a wonky star gets idealized. Keep cleanup distortion-
  anchored (L1 + region-CE). (`conditioning.md` §2.4; `grounding.md` §2.3)

- **Input-channel concatenation of the label map.** Provably zeroed by normalization on uniform-label
  patches — i.e. destroyed exactly on the flat fills the tracer needs. Use feature modulation (SFT),
  not concat. (`conditioning.md` §2.1)

- **Off-the-shelf layout detectors as an "intent" source.** Their labels (page-region archetypes at
  ~70–89 mAP) are structurally too coarse for object-level geometry; they say "a figure is here," not
  "a five-pointed star." (`document-intent.md` §2.4)

- **MDL as a standalone fidelity safeguard.** Its simplicity bias *licenses* substituting a regular
  primitive for an asymmetric one, and a mis-set σ² codes intentional asymmetry as noise. Use **NFA as
  the primary gate**; MDL only secondary, only with a conservative σ². (`primitives.md` §2.2, Rec 6)

- **Hough voting for primitive detection.** Does not scale past lines/circles (`O(A^(m−2))`; a 4-param
  ellipse needs >230 billion accumulator cells). Use edge/arc-grouping → algebraic fit → NFA gate.
  (`primitives.md` §2.3)

- **Unconditional primitive snapping or blind symmetrization.** Even the market leader ships snapping
  OFF by default; blind symmetrization erases deliberate asymmetry. All geometric intent must be
  error-gated and surfaced as a reviewable diff. (`primitives.md` §2.4–2.5; `grounding.md` §2.4)

- **Diffusion as the cleanup default.** Structurally inventive, perceptual-not-ΔE, not a small-net
  technique — wrong default and wrong scale. Deferred to Rung 5, clamped, only if lower rungs plateau.
  (`grounding.md` §2.3; `conditioning.md` §2.5)

---

## 4. How this interleaves with the existing prescription

The prior round ([`../2026-07-20-deep-research/README.md`](../2026-07-20-deep-research/README.md))
prescribed three spends: **Spend 1** (engine audit + free fitting fixes, ~$0), **Spend 2**
(degradation-realism upgrade + retrain, ~$20–40), **Spend 3** (typography/clipart/emoji gap mint +
retrain, ~$20–40). The intent ladder does **not** replace that sequence — it rides on it. Concretely:

- **Rung 0 folds into the prior round's own work.** The boundary-F-score eval switch and the CE
  weight=0 sweep are *already* prior-round recipe items; the only net-new piece is the deliberate-
  asymmetry regression set, which overlaps with the high-edge-density/typography eval slice that
  Spend 3 was already going to produce. Build them together. **Do this alongside Spend 1** (both are
  ~$0 diagnostics that reprioritise everything downstream).

- **Rung 1 (spatial-FiLM) is a retrain-change, not a new run — bundle it into Spend 2 or Spend 3.**
  It changes the architecture of a run that is already happening; it should not consume a dedicated
  GPU budget. **Hard dependency:** the CE weight=0 sweep (Rung 0 / prior-round recipe) must land
  *first* — if the region head does not earn its capacity, conditioning off it is pointless. So the
  order inside the training track is: CE sweep → confirm head helps → add SFT modulation to the next
  scheduled retrain.

- **Rung 2 (primitive fitting) is the natural successor to Spend 1's engine work.** Spend 1 (E-1
  residue-mean check, E-2 corner gate, E-3 polygon-stage ablation) must land first — you cannot add
  gated primitive substitution to a tracer you have not yet audited. Primitive fitting is the *next*
  engine lever after those free fixes. **Shared caveat carried over from Spend 1:** the shipped bench
  (1.520) uses vtracer, so Stage-2 intent work only moves the *shipped* number if Rust surpasses
  vtracer — but at ~$0 compute it is still correct to prototype, and it de-risks owning the full stack.

- **Rung 3 (abstention head) rides whichever retrain carries Rung 1** — same run, marginal cost. Its
  pass-through-the-known-gaps behavior directly protects the typography/fineline/gradient content that
  Spend 3's mint is expanding into.

- **Rungs 4–5 are out of scope for the next month** and gated behind measured plateaus.

**Net monthly sequencing.** Spend 1 + Rung 0 first (both ~$0, both diagnostic). Then the training
track (Spend 2 and/or Spend 3) carries Rung 1 + Rung 3 as architecture changes on runs already
budgeted. In parallel, the engine track carries E-1/E-2/E-3 then Rung 2 (Rust, ~$0 compute). This
keeps the whole intent ladder's next-month footprint **inside the prior round's $50–100 envelope**,
because the only genuinely new GPU cost is that Rung 1/3 make an already-planned retrain slightly
larger, and all Stage-2 geometry is compute-free engineering.

---

## 5. Open questions

1. **Which degradation bucket are we in?** Semantic-conditioning payoff ranges from large (+1.28 dB,
   structured degradation) to near-zero (+0.11–0.25 dB, mild noise). Our pixelated-vector-render
   cleanup could pattern either way, and no cited result is in-domain. Rung 1's ablation is the
   direct, cheap resolution: if region-conditioning barely moves ΔE, we are low-gain and should
   redirect to Stage-2 geometry. (`grounding.md` Q1–Q2; `conditioning.md` Q1)

2. **Does the region head classify *real* print artwork well enough to condition safely?** The head is
   trained on synthetic renders; on real typography/fineline/gradient content it may be the
   misclassification source the failure-mode literature warns about. Untested on the real target
   distribution — and it gates whether Rung 1 is safe at inference. (`conditioning.md` Q3)

3. **Does the CE aux head even earn its 4.3M-param capacity in our zero-gap regime?** Carried over from
   the prior round — the weight=0 sweep settles it, and it is a hard gate on Rung 1 (no useful head →
   nothing to condition off). The answer may flip once gap data lands. (prior round Q7)

4. **Per-primitive NFA null models.** NFA is proven only for line segments; deriving the correct H0
   background model and `N_test` for circles, rounded-rects, and a star family is the real Rung-2
   engineering, and a mis-specified H0 silently breaks the ≤1-false-detection guarantee. Which
   primitive families are worth the derivation given our distribution (logos, wordmarks, cards)?
   (`primitives.md` Q1)

5. **What σ² keeps deliberate asymmetry out of the "code as noise → correct it" regime?** The single
   load-bearing knob for print fidelity in any residual-based Stage-2 gate, currently unquantified. A
   customer's intentional asymmetry is *signal*; the gate must not read it as noise. (`primitives.md`
   Q2; `grounding.md` Q5)

6. **Is a "star" fittable, and how?** A regular n-pointed star is not a canonical algebraic manifold;
   its parametric family (center, n, inner/outer radius, rotation, corner rounding) and its NFA/algebraic
   fit must be hand-defined. vectorizer.ai supports stars but publishes no method. (`primitives.md` Q3)

7. **Topology-aware substitution mechanics.** How does a circle/rect substitution propagate to shared
   planar-map edges without gaps/overlaps? Our engine has the planar map; the edit algebra is
   unspecified. (`primitives.md` Q4)

8. **Does *any* coarse tile-type conditioning move cleanup ΔE?** The Donut-token falsification test
   (Rung "do not build") closes the document-archetype rung permanently if the answer is "no
   movement." Cheap and worth running once before the rung is dismissed for good. (`document-intent.md`
   Q1–Q2)

9. **Does Stage-1 discard sub-pixel/anti-aliasing information Stage-2 needs?** vectorizer.ai places
   boundaries from anti-aliasing pixel values; our 256px UNet may throw this away before the tracer
   sees it. Unaddressed by any cited restoration paper; needs an internal probe. (`grounding.md` Q3)

10. **Where does the fidelity gate's threshold sit for *real* customer intent?** The right values that
    snap a sloppy-scanned circle while sparing a deliberately-asymmetric brand star are an empirical,
    per-customer question — likely needs the report-then-approve UX to gather ground truth rather than
    a fixed constant. (`grounding.md` Q5; `primitives.md` Q7)
