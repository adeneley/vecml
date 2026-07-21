# Theme C — Document/Layout-Level Priors ("this is a business card")

Research synthesis, 2026-07-21. Scope: does archetype-level understanding of a whole document/tile ("this is a business card", "this region is body text") measurably help lower-level raster cleanup, and is any of it worth building at our scale (4.3M-param Stage-1 UNet, 256px tiles, ~0.5M minable NAS tiles, print-fidelity constraint).

Reader note: this document is self-contained for roadmap decisions. Every decision-critical claim carries a verification verdict. REFUTED claims appear only in corrected form with the correction noted inline. Where a source is applied outside its regime, that is flagged.

---

## 1. Executive summary

The document-AI literature is mature but points **away** from bolting a document-archetype classifier onto our cleanup UNet. Three convergent findings:

1. **The specific rung we are asked to evaluate — "classify the document type, then clean up better" — is unproven.** No published pipeline demonstrates that document-archetype recognition (business card / flyer / logo) measurably improves downstream raster restoration or binarization. The design-document datasets that model our exact target distribution (Crello, PosterLayout, CanvasVAE) use archetype knowledge exclusively for *generation and layout synthesis*, never for cleaning damaged raster input. This link is absent from the literature, so any build here is research, not adoption.

2. **The transferable lever is in-domain self-supervised pretraining and data diversity, not explicit type labels.** DiT shows one self-supervised document-domain backbone lifts both classification and layout detection (though via per-task fine-tuning, not a single frozen encoder — see correction below). DocLayNet shows layout-diverse in-domain data beats homogeneous out-of-domain sources. Both argue for mining the shop's own NAS tiles over adopting an off-the-shelf archetype prior trained on scientific papers.

3. **Explicit archetype/layout priors carry the exact print-fidelity failure mode we forbid.** Layout-generation models (PosterLayout) learn canonical designer arrangements and push outputs toward "generic-correct" — the mechanism that would "correct" a deliberately-asymmetric mark. Off-the-shelf detectors are also too coarse (page-region labels: Text/Title/Figure at ~70–89 mAP) to encode object-level geometric intent.

Net: the *cheap and safe* version of "document-level context" is architectural (wider receptive field / global-local conditioning that enforces coherence without inventing semantics) plus in-domain data. The *expensive and speculative* version is a supervised archetype head feeding the UNet — no evidence it helps cleanup, and it introduces a hallucination vector. Recommend a small ablation on the cheap end and a shelving of the archetype-classifier rung pending evidence.

---

## 2. Findings by sub-topic

### 2.1 Does document-archetype classification improve downstream cleanup? (the core question)

**No published evidence that it does.** Across the document-AI corpus surveyed, no pipeline demonstrates that classifying a document's archetype measurably improves downstream restoration, binarization, or vectorization. The "this is a business card → better cleanup" rung is unproven. Archetype/task conditioning is proven only for *extraction* tasks (parsing, key-value), where [Donut](https://arxiv.org/abs/2111.15664) conditions on document task/type via a single decoder start-prompt token over a shared visual encoder — a cheap mechanism *if* archetype knowledge is ever wired in, but demonstrated for understanding, not pixel repair.

Implication: treat any archetype→cleanup link as a hypothesis to test, not a technique to adopt.

### 2.2 Scale/data vs explicit structural priors (the implicit rung)

**[RVL-CDIP, Harley et al. 2015](https://arxiv.org/abs/1502.07058) — CONFIRMED, with scope caveats.** For whole-image document *classification* (400k images, 16 categories), a holistic ImageNet-transferred CNN beats hand-engineered region-partitioned features given enough data ("enforcing region-specific feature-learning is unnecessary given sufficient training data"). This supports the implicit/scale rung **as one data point, for classification only**. Verification flagged three limits that must ride with the citation:
- **Task mismatch.** "Region-specific" here means the old practice of partitioning a page into header/body zones and learning per-zone classifiers — *not* geometric priors for pixel-level reconstruction. It says nothing about repair, region-label segmentation, or hallucination risk.
- **The transfer finding cuts the other way.** "CNNs trained on non-document images transfer well" was shown by fine-tuning ImageNet-pretrained nets — evidence *for* injecting pretrained features (the explicit-conditioning rung), not against it.
- **Regime gap.** Scanned grayscale office docs, 2015-era CNNs, 400k images — not color graphic art or a 4.3M-param dual-head repair UNet.

Use as supporting evidence for the implicit rung in classification, not as proof that geometric intent comes "for free" from scale in a fidelity-constrained reconstructor.

### 2.3 Self-supervised in-domain pretraining as the shared lever

**[DiT, 2022](https://arxiv.org/abs/2203.02378) — original claim REFUTED as applied; corrected form:** Self-supervised document-domain pretraining (BEiT-style masked-image modeling on unlabeled document images) produces a single backbone whose weights, **when fine-tuned per task**, lift both high-level classification (RVL-CDIP 91.11→92.69) and detection/layout tasks (PubLayNet 91.0→94.9 mAP, ICDAR table 94.23→96.55, OCR text detection 93.07→94.29). **Correction:** DiT does *not* use a single frozen encoder feeding multiple heads — each downstream task fine-tunes its own copy of the backbone. So this is evidence for a transferable self-supervised *initialization*, not for the "frozen recognizer features injected into the UNet" mechanism specifically. Also note the backbones are 86–304M params (20–70× our UNet) and the layout tasks are object/instance detection (boxes+masks), not dense per-pixel repair.

Takeaway: the durable lesson is that **in-domain self-supervised pretraining transfers across document tasks**. For us that argues for SSL pretraining on unlabeled NAS tiles as a warm-start for Stage 1, not for a frozen-feature injection scheme (which this paper does not validate).

### 2.4 Off-the-shelf layout detectors are too coarse for geometric intent

**[LayoutParser model zoo](https://layout-parser.readthedocs.io/en/latest/notes/modelzoo.html) — supporting, not decision-critical.** Available detectors output only page-region archetypes at modest ceilings: PubLayNet (Text/Title/List/Table/Figure) best 88.98 mAP; PrimaLayout 69.35 mAP; Math Formula 79.68 mAP; Newspaper Navigator (Photograph/Illustration/Map/Headline/Advertisement). These tell you "there is a figure here", not "this is a five-pointed star." They cannot encode the object-level shape identity our intent vision needs, so they are not a shortcut to the "recognizes it is a star" capability.

### 2.5 In-domain layout diversity beats homogeneous sources (mine the NAS)

**[DocLayNet, 2022](https://arxiv.org/abs/2206.01062) — CONFIRMED.** Built specifically because prior datasets "severely lack in layout variability since they are sourced from scientific article repositories such as PubMed and arXiv only" (80,863 manually annotated pages, 11 classes). DocLayNet-trained models are "more robust and thus the preferred choice for general-purpose document-layout analysis"; models land ~10% behind human inter-annotator agreement. Verification caveat: the win is framed as cross-domain *robustness/generalization*, not a blanket accuracy victory (in-domain, a PubLayNet-trained model still wins on PubLayNet-like pages). This strengthens our case — our target is diverse real artwork, not homogeneous templates. Regime note: these are ~tens-of-millions-param object detectors, not a 256px repair UNet, so this is an analogical data-curation principle (diverse in-domain > homogeneous out-of-domain), not a same-task result.

### 2.6 Design-archetype datasets model our exact distribution — but only for generation

**[Crello](https://huggingface.co/datasets/cyberagent/crello) / [PosterLayout](https://arxiv.org/abs/2303.15937) / CanvasVAE — supporting.** Crello: 23,300 design templates across 22 categories (Instagram AD 1080×1080, Presentation Wide, etc.), each element stored as vector attributes (type, position, size, rotation, opacity, RGBA, font/size/bold/italic/alignment). PosterLayout/PKU: 9,974 poster-layout pairs from 905 posters. CanvasVAE models a document as a multi-modal set of attributes plus a sequence of visual elements. **All published usage is generation / layout synthesis; none targets cleaning up damaged raster input.** So the archetype knowledge for flyers/cards/banners exists in structured form, but the bridge to restoration has not been built by anyone.

### 2.7 Layout priors push toward "generic-correct" — the fidelity failure mode

**[PosterLayout Design Sequence Formation, 2023](https://arxiv.org/abs/2303.15937) — supporting.** Reorganizes elements "to imitate the design processes of human designers" via a discriminator, enforcing "realistic design workflows" and "visual coherence." This is exactly the mechanism that would "correct" a deliberately-asymmetric mark toward a canonical one. Any explicit archetype/layout prior tends toward plausible-generic outputs, which the print-fidelity constraint forbids. This is the strongest literature-grounded caution against wiring archetype priors into geometry.

### 2.8 Cross-theme corroboration (borrowed from Theme B / curriculum results)

Two results from adjacent themes bear directly on how document-level context should be injected safely:

- **[Globally and Locally Consistent Image Completion, Iizuka et al. SIGGRAPH 2017](https://iizuka.cs.tsukuba.ac.jp/projects/completion/en/) — supporting.** The canonical way to condition local processing on whole-scene context: dual discriminators (global over the full 256×256, local over a 128×128 crop) plus dilated convolutions to widen receptive field. Crucially, it *enforces consistency, it does not hallucinate semantics* — the completion network is a single FCN. This is the print-fidelity-safe template for "document context conditions local cleanup": widen the receptive field and enforce coherence, do not add a semantic generator.

- **[Blau & Michaeli perception-distortion tradeoff, 2018](https://arxiv.org/abs/1711.06077) — original claim REFUTED as applied; corrected form:** The theorem is real and distortion-metric-agnostic: the perception-distortion function P(D) (perception = divergence between the *distribution* of reconstructions and natural-image statistics) is monotonically non-increasing and convex. **Correction to the alarmist reading:** (a) the tradeoff binds only on the optimal frontier in its low-distortion region — sub-optimal models can improve fidelity and realism together, and flat regions exist; (b) when degradation is near-invertible (mild vector-render cleanup — ground truth largely recoverable from input), the frontier gap is negligible; (c) "hallucination" is necessary only if you *deliberately* operate at the perceptual/distribution-matching end — it is a selectable operating point, not an unavoidable bug. Our L1 RGB-repair loss sits at the distortion end and regresses toward the posterior mean (it blurs, it does not invent). Confident invention is a property of adversarial/diffusion objectives, not of sharpness per se. For a print-fidelity constraint the correct reading is prescriptive: stay at the low-distortion operating point, and reserve generative priors for content you are willing to have distribution-matched.

---

## 3. Recommendations

| # | Recommendation | Confidence + why | Cheapest confirming experiment (cost) |
|---|---|---|---|
| 1 | **Do not build a supervised document-archetype classifier feeding the UNet** as a near-term roadmap item; the archetype→cleanup link is entirely absent from the literature. | High — no published pipeline demonstrates the link; the datasets that model our distribution use it only for generation. | Skip the build; if tempted, run rec #4 first as the falsification test (~1 GPU-day). |
| 2 | **If any document-level context is added, add it architecturally (wider receptive field / global-local coherence) not semantically**, following the Iizuka global-local template. | High — this is print-fidelity-safe (enforces consistency, does not invent) and is proven for restoration. | Ablate a dilated-conv / larger-context variant of the current UNet on the 24-image bench (~1–2 GPU-days). |
| 3 | **Prefer in-domain SSL pretraining on unlabeled NAS tiles over any off-the-shelf archetype prior**; the transferable lever is in-domain pretraining + data diversity (DiT, DocLayNet). | Medium-high — strongly evidenced for document tasks, but via per-task fine-tuning and at 20–70× our param scale (analogical, not same-regime). | Pretrain the UNet encoder with masked-image modeling on ~0.5M NAS tiles, fine-tune, compare deltaE vs from-scratch (~3–5 GPU-days). |
| 4 | **Falsify the archetype rung cheaply before investing**: condition the existing UNet on a coarse tile-type token (logo / word-mark / card / flyer) à la Donut and measure whether cleanup deltaE moves at all. | Medium — Donut proves the conditioning *mechanism* is cheap; whether it helps *cleanup* is exactly the open question. | Add a 4–8 way tile-type embedding token (labels auto-derived from NAS folder structure), retrain, measure (~1 GPU-day). |
| 5 | **Do not use off-the-shelf layout detectors to supply "intent"** — their outputs (page-region labels at ~70–89 mAP) are too coarse for object-level geometry. | High — the label granularity gap is structural, not a tuning issue. | None needed; inspect LayoutParser label sets to confirm (minutes). |
| 6 | **Treat any layout/archetype prior as a fidelity hazard** and gate it behind an asymmetry-preservation test (e.g. the deliberately-asymmetric star); generative layout priors push toward generic-correct. | High — directly evidenced by PosterLayout's designer-imitation objective. | Add 3–5 adversarial "deliberately irregular" marks to the held-out bench as a standing regression check (hours). |
| 7 | **Keep Stage 1 at the low-distortion (L1/regression) operating point**; the perception-distortion tradeoff is a selectable knob, and our near-invertible cleanup regime has a small frontier gap. | High — corrected reading of Blau-Michaeli; our own zero train/val gap indicates data-, not objective-, limitation. | No new experiment; hold the line and only revisit if a perceptual loss is proposed. |

---

## 4. Open questions

1. **Does *any* coarse tile-type conditioning move cleanup deltaE?** The literature is silent; rec #4 is the direct test. If the answer is "no measurable movement," the entire document-archetype rung can be closed permanently.
2. **Is our NAS folder structure a usable source of free archetype labels?** Auto-deriving tile-type tokens from job-folder naming would make rec #4 nearly free — needs a quick audit of how consistently NAS folders encode product type.
3. **Would SSL pretraining help given we are data-constrained, not architecture-constrained?** DiT-scale results come from 86–304M-param models; the transfer to a 4.3M UNet at ~0.5M tiles is untested. Rec #3 answers it but the param-scale gap makes the prior uncertain.
4. **Where does "document-level" even apply for us — full page or 256px tile?** Our pipeline operates on tiles; a "business card" archetype is a page-level concept that may not survive tiling. Does global context need to be reintroduced at a coarser resolution before per-tile cleanup?
5. **Can the Iizuka global-local coherence mechanism help on graphic art at all**, given it was validated on natural-image inpainting? The receptive-field argument transfers; the discriminator-based realism objective may reintroduce fidelity risk and needs the asymmetry gate (rec #6).
6. **Is there unpublished/industrial evidence for archetype→restoration** (e.g. inside Adobe/Canva pipelines) that the academic corpus does not surface? Not found in this search; flagged as an evidence gap rather than a confirmed negative.
