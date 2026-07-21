# Theme B — Parametric Primitive & Shape Fitting in Vectorization

**Question this document answers:** where can *geometric* intent live in Stage 2 (the classical tracer), and how do we fit circles/rects/stars/polygons/symmetry to traced art without violating the print-fidelity constraint (sharpen what is there, never invent what is not)?

**Scope:** Stage-2 geometric intent only. Neural/generative rungs (diffvg, LIVE, CLIPasso) are covered here only where they bear on the fit-vs-accept decision; they are the subject of a separate synthesis. All claims below were independently verified against primary sources; verdicts are noted inline.

---

## 1. Executive summary

The literature gives a firm, buildable answer. Geometric intent belongs in Stage 2 as **error-gated primitive substitution**, not unconditional snapping. Three components:

1. **Fitting has a proven accuracy ladder.** For circles, plain algebraic Kåsa fits are biased (they underestimate radius on short/partial arcs — exactly the tracer's case); Pratt and Taubin sharply reduce that bias; the Hyper fit is unbiased to leading order. **Use Taubin or Hyper, never plain Kåsa.** (CONFIRMED.)

2. **The hard part is acceptance, not fitting** — deciding whether a contour is genuinely a circle/star or a freeform blob. Two rigorous, near-parameterless frameworks exist: **a-contrario / NFA validation** (accept only when the Number of False Alarms ≤ 1, giving ≤1 false detection per image, provably insensitive to the threshold) and **description-length / geometric-MDL model selection** (swap in a primitive only when its parameter saving beats its residual penalty). NFA is the cleaner fit for the "never invent structure" constraint. MDL works but must be tuned conservatively — used naively it is a shape-*regularizer* that erases intentional asymmetry (see §4, this claim was REFUTED as originally stated).

3. **The route is edge/arc grouping → algebraic fit → NFA/MDL gate**, not high-dimensional Hough voting (which does not scale past lines/circles). Working parameterless templates already exist: **ELSD** and **EDCircles**.

Strategic context: the tracers we benchmark against (vtracer, potrace) do **no** primitive substitution at all — they emit only Béziers/polygons. The market leader, **vectorizer.ai**, does exactly this fitting in a post-AI classical stage, supports parameterized circles/ellipses/rounded-rects/stars plus symmetry modelling — but ships it **OFF by default** (output flattens to ordinary curves). That default is the strongest practitioner signal in the corpus: aggressive shape-snapping is dangerous enough that even the leader gates it. Primitive fitting is therefore a genuine, unoccupied lever against our baselines and is cheap to prototype (scikit-image ships RANSAC `CircleModel`/`EllipseModel` today).

---

## 2. Findings by sub-topic

### 2.1 Fitting accuracy — which estimator to use

There is a rigorous, published accuracy hierarchy for circle fitting (Al-Sharadqah & Chernov, *Electronic Journal of Statistics* 3, 2009):

- The algebraic **Kåsa** fit is the fastest but is "heavily biased toward small circles" on incomplete/partial arcs and under noise — it underestimates radius exactly where a tracer sees short contour fragments.
- **Pratt** and **Taubin** constraints cancel the leading bias terms; Taubin's essential bias is half of Pratt's, making it "statistically more accurate than Pratt."
- The **Hyper** fit (H = 2·Taubin − Pratt) makes the essential bias vanish entirely to leading order and "outperforms all the existing methods, including the (previously regarded as unbeatable) geometric fit."

**Directive:** for Stage-2 circle/arc substitution use **Taubin or Hyper, never plain Kåsa**. This is pure classical point-set geometry — domain-agnostic, transfers directly to the tracer with no data-scale caveat. [source](https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-3/issue-none/Error-analysis-for-circle-fitting-algorithms/10.1214/09-EJS419.full) *(CONFIRMED verbatim.)*

Two minor nuances: (a) "Kåsa/Coope" in the raw claim conflates two distinct methods — the bias result is about Kåsa specifically; Coope is not analysed in that paper. (b) On clean, full, well-sampled, low-noise circles all four algebraic fits are close; Kåsa's failure is specifically on short arcs and high noise — which is precisely the Stage-2 regime, so the directive holds where it matters.

For fitting a single primitive amid contour noise/outliers, **RANSAC** is the standard robust engine, with a closed-form iteration budget `k = log(1−p)/log(1−wⁿ)`. Documented failure modes bear directly on our use: it needs >50% inliers, requires a per-problem residual threshold, and cannot fit multiple models simultaneously (a logo contains many primitives). Implication: pair per-region RANSAC with an NFA/MDL acceptance test rather than trusting inlier count alone. [source](https://en.wikipedia.org/wiki/Random_sample_consensus) *(decision-critical: no; not independently re-verified — treat iteration formula and limitations as standard textbook material.)*

### 2.2 Acceptance — the actual print-fidelity gate

Fitting is easy; deciding *whether to substitute at all* is the fidelity-critical step. Two rigorous frameworks:

**a-contrario / Number of False Alarms (NFA).** A candidate is accepted only when `NFA(r,i) = (NM)^(5/2)·γ·B(n,k,p) ≤ ε`, with ε fixed at **1** (at most one false detection per image on average). Theorem 1 of the LSD paper proves `E_H0[#{NFA ≤ ε}] ≤ ε` under the noise null model H0 (the Helmholtz principle). Crucially, detection is **provably insensitive** to ε — the detection limit varies like `√(−log ε)`, so "setting ε to any reasonable value would produce very similar results." This is the principled, threshold-free gate for "only substitute a primitive when it is statistically meaningful, not noise." [source](https://www.ipol.im/pub/art/2012/gjmr-lsd/article.pdf) *(CONFIRMED verbatim.)*

Two scope caveats that bound the transfer (they do not refute the framework):
- The LSD paper proves NFA only for **line-segment** detection. Using it for circles/ellipses/star templates is a well-precedented extension (the a-contrario literature covers circles, ellipses, vanishing points, contours), **but each primitive family needs its own correct H0 null model and its own number-of-tests count** — you cannot reuse LSD's `(NM)^(5/2)·γ`. Getting `N_test` and the background model right per primitive is the real engineering work; a mis-specified H0 breaks the `E[#false] ≤ ε` guarantee.
- "Parameterless" applies to the **significance threshold** only. NFA removes the free error threshold but LSD still carries fixed internals (angle tolerance τ=22.5°→p=1/8, gradient quantization, γ tested precisions, analysis scale). Frame it as "no free significance threshold," not "no parameters."

**Description-length / geometric-MDL model selection.** MDL's two-part code `L(H)+L(D|H)` (or Kanatani's Geometric AIC / Geometric MDL) chooses a primitive only when its parameter-cost saving outweighs its residual penalty. This is real and regime-relevant to a classical tracer — **but it is NOT a clean, self-acting safeguard against over-correction, and the original raw claim to that effect was REFUTED.** Corrected statement:

> MDL/geometric-MDL is a usable substitution gate: adopt a regular primitive only when its parameter-cost saving (shorter `L(H)`) outweighs its residual penalty (longer `L(D|H)`), with Kanatani penalizing by manifold dimension d, data count N, parameter count p and noise variance σ². **However** (a) MDL's inherent bias favors the *simpler* model — here the regular primitive — so the same criterion actively *licenses* substitution; it is as much the engine that "corrects" an asymmetric star as a guard against it. (b) The result hinges entirely on the assumed noise variance σ²: a deliberately-asymmetric star's deviation is **signal, not noise**, so if that asymmetry is small relative to σ², MDL codes it as noise and replaces the star with a regular primitive — precisely the fidelity failure to avoid. (c) A "star" is not one of Kanatani's canonical algebraic manifolds (lines/conics/quadrics); the parametric-star family, the freeform-contour code, and their dimensions must all be hand-defined, and MDL is highly sensitive to that encoding.
>
> **Net:** to serve as a fidelity guard, MDL must be tuned to a conservative σ² (or given an asymmetry-preserving prior) so it errs toward keeping the freeform outline. Used naively it becomes a shape-regularizer that erases intentional asymmetry.

Sourcing note: the cited Wikipedia MDL page supports only the generic two-part code and contains **no** mention of Kanatani, Geometric AIC, or Geometric MDL. The Kanatani specialization is factually accurate but must be sourced from Kanatani's own papers (IJCV 1998 / PAMI 2004), not this URL — flagged as UNVERIFIED against a primary source here. [source](https://en.wikipedia.org/wiki/Minimum_description_length)

**Practitioner reading:** prefer **NFA** as the primary gate (its guarantee is about false *detections* — non-invention — which is exactly the constraint), and treat MDL as a secondary/complementary criterion only with a conservative σ². Both share the same protective logic: a contour that fits a regular model poorly is left as a freeform outline.

### 2.3 The route: arc-grouping + algebraic fit + NFA — not Hough voting

**Brute-force Hough does not scale** to the primitives we care about. Complexity grows as `O(A^(m−2))` with m parameters; the cited source warns Hough "must be used with great care to detect anything other than lines or circles." An axis-aligned (4-parameter) ellipse on an 800×600 image already needs "more than 230 billion" accumulator cells. [source](https://en.wikipedia.org/wiki/Hough_transform) *(CONFIRMED verbatim, with one correction.)*

> Correction: the source counts the ellipse as **4 parameters** (center x, y + two radii) and its 230-billion figure is that 4-param case. A fully general **5-parameter oriented** ellipse (adding orientation) is worse by roughly two orders of magnitude (tens of trillions of cells). The raw claim's "5-parameter → 230 billion" misattributes the number, but errs conservatively — the real 5-DOF cost is much higher, so the scaling argument holds *a fortiori*. Line = 2 params (r,θ), circle = 3 (x,y,r).

This pushes decisively toward **edge/arc grouping → algebraic fit → NFA gate**. Working, parameterless templates for exactly the "which primitive (if any) fits this contour" decision already exist:

- **ELSD** (Pătrăucean, Gurdjos, Grompone von Gioi, ECCV 2012) reuses LSD's line-hypothesis + NFA machinery and adds circular- and elliptical-arc hypotheses, all validated by shared NFA computation. On the sample `stars.pgm` it returns 16 elliptical arcs, 322 circular arcs, 165 line segments with no tuning. [source](https://github.com/viorik/ELSD) *(CONFIRMED — output counts and authorship verified.)*
- **EDCircles / EDLines / EDPF** (Akinlar & Topal, `ED_Lib`) are likewise parameter-free detectors that fit least-squares circles/ellipses and validate via the Helmholtz principle to remove false positives. *(CONFIRMED via ED_Lib.)*

Two nuances: (a) "parameterless" again means threshold-free acceptance — an implicit ε=1 plus LSD-inherited fixed internals. (b) These are whole-image detectors emitting many validated primitives from an edge map, not a single accept/reject on one pre-isolated contour — but adapting the same NFA machinery to a per-region decision (which is our Stage-2 setting, where the planar map already gives us isolated regions) is direct.

### 2.4 What commercial tools and baselines actually do

**vectorizer.ai (the tool to beat).** Its own docs reveal an architecture strikingly parallel to ours: a deep-learning vectorizer followed by a classical cleanup stage that "fit[s] whole geometric shapes, clean[s] up corners, tangent matching, curve fairing." Geometric intent lives in **Stage 2, downstream of the model.** Verified specifics:

- **Full shape fitting:** "fully parameterized circles, ellipses, rounded rectangles, and stars, all with optionally rounded corners and arbitrary rotation angles." [source](https://vectorizer.ai/)
- **Symmetry modelling:** "We detect and model mirror and rotational symmetries… to produce more accurate and more consistent results" — but scoped to "where possible" and "sensible guesses," i.e. hypothesis-under-tolerance, not forced correction.
- **OFF by default:** the API "Parameterized Shapes" option defaults to **Flatten** (all parameterized shapes flattened to ordinary curves); native primitive output is opt-in. **Even the market leader ships aggressive shape-snapping disabled by default.** [source](https://vectorizer.ai/api/outputOptions)
- **Topology-aware edits:** a shared "Vector Graph" (planar map) "allows us to make these changes while keeping neighboring shapes perfectly aligned, a common pitfall for other image vectorizers." **Implication for our Rust planar-map engine: geometric-intent edits must operate on shared edges, not per-region, or they reintroduce gaps/overlaps.**
- **Tunable fit hierarchy:** allowed curve types {Line, Quadratic, Cubic, Circular Arc, Elliptical Arc} with defined fallback; line-fit tolerance Coarse 0.30px → Super Fine 0.01px. Intent is a tunable accuracy knob, not a fixed representation.

*(All four decision-critical vectorizer.ai claims CONFIRMED from primary docs.)*

**Baselines (vtracer, potrace) do NO primitive substitution.** vtracer output modes are exactly pixel / polygon / spline — no parametric-primitive mode; "Potrace uses an O(n²) fitting algorithm, whereas vtracer is entirely O(n)." Potrace is a polygon-then-Bézier contour tracer ("polygon" = boundary approximation, not regular-polygon snapping); its only geometric-intent knob is a single global `alphamax` corner threshold (default 1) plus `opttolerance` (default 0.2). Neither detects/substitutes circles, rectangles, or regular polygons. **Stage-2 geometric intent is therefore a genuine, unoccupied lever relative to our benchmarked tools** — novel relative-to-baseline, not novel in the absolute field. [source](https://github.com/visioncortex/vtracer), [source](https://potrace.sourceforge.net/potrace.1.html) *(CONFIRMED; note the string "It does not fit geometric primitives like circles or rectangles" is a paraphrase, not a literal vtracer README quote — the fact is correct via the mode list, but do not cite it as a verbatim quote.)*

**Low-cost prototype path.** scikit-image ships algebraic `CircleModel`/`EllipseModel` wrapped in RANSAC out of the box (`from_estimate` on a min-sample subset, `residuals` to classify inliers below `residual_threshold`, `max_trials` iterations), plus circular/elliptical Hough. The classical primitive-fitting rung is cheap to prototype and benchmark against current Rust/vtracer Stage-2 before committing to heavier machinery. [source](https://scikit-image.org/docs/stable/auto_examples/transform/plot_ransac.html)

### 2.5 Symmetry: detect-as-voting, then minimal constrained snapping

The symmetry/beautification lineage converges on a two-move recipe that maps onto Stage-2 intent:

**Move 1 — detection as voting.** Approximate symmetry (or any global relation) shows up as a **peak/cluster in a transformation space**, and the tightness of that cluster is a usable confidence score. Loy & Eklundh (ECCV 2006) match local features against their mirror reflections and accumulate votes for symmetry axes, giving an explicit per-axis confidence downstream code can threshold before snapping. [source](https://link.springer.com/chapter/10.1007/11744047_39) Mitra/Guibas/Pauly's **Symmetrization** (SIGGRAPH 2007) parametrizes a reflection as a point in a 2D transformation space; "deviation from perfect symmetry can be observed as variance of the cluster," and the optimal reflection has a closed-form SVD solution. [source](https://graphics.stanford.edu/~niloy/research/symmetrization/symmetrization_sig_07.html) *(both decision-critical, CONFIRMED as sourced.)*

**Move 2 — minimal constrained snapping.** Symmetrization deforms a shape toward exact symmetry while "minimally altering its shape," alternating detection and constrained deformation until cluster variance drops. **GlobFit** (Li et al., SIGGRAPH 2011) is the canonical relation-guided template: fit primitives locally (RANSAC), then discover a **minimal, conflict-free** set of global relations (parallel, orthogonal, equal-angle, equal-length/radius, symmetry) and enforce them via constrained optimization that "balances between the data-driven fitting error and the regularization effects of the inferred [relations]." Its Occam preference for the fewest non-conflicting relations is the concrete **fidelity dial** — enforce a symmetry only when the data supports it and it does not conflict with a higher-confidence relation. [source](https://graphics.stanford.edu/~niloy/research/globFit/paper_docs/globFit_sigg11.pdf)

**The recurring warning — direct print-fidelity relevance.** Symmetrization's own stated failure mode: the deformation "does not respect the semantics of the shape"; small-scale features are "sometimes ignored" or washed out (the gecko's toes are lost). **Blind symmetrization can erase deliberate asymmetry.** The paper's only mitigation is user control + a few detection thresholds — *not* automatic semantic protection. Therefore intent-snapping must be gated on a confidence score **and** a data-fidelity term, and should prefer the fewest relations (Occam) — exactly the print-fidelity constraint that a customer's deliberately-asymmetric star must survive.

**Igarashi's Interactive Beautification (Pegasus, UIST 1997)** offers a fidelity design pattern: global symmetry emerges from a library of cheap **local** pairwise constraints (flipped-congruence + alignment + connection) with no global symmetry solver, and ambiguity is handled by **proposing multiple candidate beautifications for the user to pick** rather than auto-committing — a template for "sharpen, never invent." Caveat: only linear constraints; measured on the original hardware ~80% of beautifications finished <100ms but 17% produced >20 candidates. [source](https://www-ui.is.s.u-tokyo.ac.jp/~takeo/papers/uist97.pdf) *(decision-critical for the multi-candidate pattern; not independently re-verified but internally consistent.)*

### 2.6 Learned-propose / classical-refine hybrids (bridge to Theme A)

If pure classical proposal proves insufficient, the established hybrid is **learned init + classical refine**, not full replacement:

- **Deep Vectorization of Technical Drawings** (Egiazarian et al., arXiv:2003.05471): learned cleaning net → transformer-based primitive estimation → classical optimization to snap primitives to the raster. Directly parallels our Stage-1 UNet → Stage-2 fitter split, adding a learned primitive-proposal head feeding a classical optimizer. [source](https://arxiv.org/abs/2003.05471)
- **SketchGraphs** (Seff et al., arXiv:2007.08506) shows the maximal rung — 15M CAD sketches as primitive+constraint graphs; inferring designer intent (symmetry/parallelism) from raw geometry is learnable at scale, but far heavier than needed here. [source](https://arxiv.org/abs/2007.08506)

These are the natural escalation if §2.3's classical route hits a ceiling; they are covered in depth by the neural-intent synthesis.

---

## 3. Recommendations

| # | Recommendation | Confidence + why | Cheapest confirming experiment (cost) |
|---|---|---|---|
| 1 | Build Stage-2 geometric intent as **error-gated primitive substitution**, gated by an **a-contrario NFA test**, never unconditional snapping. | **High** — the leader ships snapping OFF by default; NFA's guarantee is specifically about false *detections* (non-invention). | Wire scikit-image RANSAC `CircleModel`/`EllipseModel` per planar-map region + a hand-derived NFA gate; measure deltaE on the 24-image bench (~1 day). |
| 2 | For circle/arc fits use **Taubin or Hyper**, never plain Kåsa. | **High** — published, verbatim-confirmed accuracy hierarchy; domain-agnostic classical geometry. | Fit both to synthetic short-arc contours, compare radius bias vs ground truth (~2 hours). |
| 3 | Prototype with **ELSD / EDCircles** (parameterless arc-grouping + NFA) before writing a bespoke detector. | **High** — working, confirmed, parameterless templates for exactly the line-vs-circle-vs-ellipse decision. | Run ELSD on 24-image bench renders, count valid vs spurious primitives by eye (~half day). |
| 4 | Make all primitive edits **topology-aware** (operate on shared planar-map edges), matching vectorizer.ai's Vector Graph. | **High** — leader names per-region edits as the "common pitfall"; our engine is already a planar map. | Substitute one circle in a two-region logo, check for gap/overlap at the shared boundary (~2 hours). |
| 5 | Ship primitive substitution **OFF by default**, opt-in per job — mirror the leader. | **Medium-high** — strong practitioner signal, but our target distribution (logos/word-marks) may tolerate more snapping than general images. | A/B the 24-image bench with substitution on vs off; inspect for any "corrected" intentional asymmetry (~half day). |
| 6 | Use **NFA as the primary gate; treat MDL/geometric-MDL as secondary only with a conservative σ².** | **Medium** — MDL's simplicity bias *licenses* over-correction; σ² mis-set erases intentional asymmetry (claim REFUTED as a standalone safeguard). | Feed a deliberately-asymmetric star through an MDL gate at two σ² values, confirm it "corrects" at large σ² (~half day). |
| 7 | For symmetry, adopt **detect-as-voting + minimal constrained snapping (GlobFit-style), gated on cluster-tightness confidence + a data-fidelity term.** | **Medium** — well-supported recipe, but every method warns it can erase deliberate asymmetry; needs the fidelity term to be safe. | Implement Loy-Eklundh axis voting on 5 word-marks, threshold confidence, verify an asymmetric mark scores low (~1–2 days). |
| 8 | Defer learned primitive proposal (Egiazarian-style) until the classical route hits a measured ceiling. | **Medium** — hybrid is proven but heavier; classical route is unoccupied and cheap, so exhaust it first. | None yet — decision gate is: does classical Stage-2 close the 1.520→~1.28 gap on the bench? |

---

## 4. Open questions

1. **Per-primitive null models.** NFA is proven only for line segments. Deriving the correct H0 background model and `N_test` count for circles, rounded rectangles, and a parametric-star family is the real engineering work — and a mis-specified H0 silently breaks the ≤1-false-detection guarantee. Which primitive families are worth the derivation, given our target distribution (logos, word-marks, business cards)?

2. **σ² for the fidelity constraint.** Both MDL and any residual-based gate hinge on a noise-variance estimate. A customer's intentional asymmetry is *signal*, not noise. What σ² (or asymmetry-preserving prior) reliably keeps deliberate deviation out of the "code as noise → correct it" regime? This is the single load-bearing knob for print fidelity and is currently unquantified.

3. **Star as a fittable primitive.** A regular n-pointed star is not a canonical algebraic manifold (unlike line/conic). What is the right parametric family (center, n, inner/outer radius, rotation, corner rounding), and what is its correct algebraic/NFA fit? vectorizer.ai supports stars but publishes no method.

4. **Topology-aware substitution mechanics.** Concretely, how does a circle/rect substitution propagate to shared planar-map edges without introducing gaps/overlaps or disturbing neighboring regions' fits? Our engine has the planar map; the edit algebra is unspecified.

5. **Acceptable snapping strength for our distribution.** The leader gates snapping for *general* images. Print-ready logos/word-marks are more primitive-heavy and may tolerate (or benefit from) more aggressive substitution. Where is our distribution's break-even between consistency gain and fidelity risk?

6. **Multi-primitive scenes.** RANSAC and single-primitive fits do not handle a logo with many primitives simultaneously; GlobFit's relation graph is NP-hard to reduce exactly. What per-region decomposition (leaning on the planar map) makes per-region fitting tractable without losing cross-region relations (equal radius, shared symmetry axis)?

7. **Interaction affordance.** Igarashi's multi-candidate "propose, don't auto-commit" pattern fits the fidelity constraint but assumes a human in the loop. Does the shop's workflow have a review step where a small set of candidate beautifications could be surfaced, or must Stage 2 auto-commit?

---

*Sourcing hygiene: Kanatani Geometric AIC/MDL specifics are accurate but UNVERIFIED against a primary source (cited Wikipedia page does not mention them — use Kanatani IJCV 1998 / PAMI 2004). RANSAC iteration formula/limitations and Igarashi timings are internally consistent but not independently re-verified. All vectorizer.ai, vtracer, potrace, ELSD/ED_Lib, LSD-NFA, Hough, and circle-fit accuracy claims were CONFIRMED against primary sources. The MDL-as-standalone-fidelity-safeguard claim was REFUTED and appears above only in corrected form.*
