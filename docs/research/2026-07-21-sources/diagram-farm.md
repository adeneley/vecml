# Source feasibility: synthetic diagram render farm

Single-source study for the Stage-1 cleanup UNet corpus. Question: is a *generator* (programmatically mint unlimited flowcharts / org-charts / graphs as SVG via graphviz, d2, mermaid, plantuml, railroad) worth building, versus just minting the already-scraped diagram set covered in `svg-diagrams.md`? All SVG-quality and throughput numbers below are measured this session on a local M-series Mac (graphviz 15.1.0, d2 0.7.1, mermaid-cli current), not recalled.

## VERDICT

**Feasible but LOW PRIORITY. Build it only if the mined `svg-diagrams` source (85k real diagrams, `svg-diagrams.md`) proves unusable on the license blocker, and even then cap it hard.**

The technical path is real and cheap: **graphviz is the one generator that emits clean-tier SVG**, generation is effectively free (measured ~11,800 graphs/sec in graphviz batch mode), and the text-to-path problem is solved by the same `usvg` promotion stage `svg-diagrams.md` already specifies. But two things sink the priority:

1. **Redundancy.** The mined `starvector/svg-diagrams` set already yields ~85k born-vector, gradient-free diagram SVGs spanning graphviz, matplotlib, mermaid, and D3 idioms. A farm would produce a *narrower* aesthetic (mostly one tool's rendering idiom) than the real scraped set, for the same primitive families. The farm's only genuine advantage over mining is that we own the license outright.
2. **Content mismatch.** Diagrams are not print-shop customer artwork. They share only *low-level primitives* (thin connector lines, arrowheads, text-in-boxes, ruled structure) with the shop's stated target families (logos, business cards, letterheads, menus, forms). As *content* this is cheap-but-irrelevant volume.

**Useful-sample ceiling: low, ~10k-30k.** The rendered visual vocabulary (a handful of node shapes, edge styles, arrowheads, layout engines, times N themes/fonts/palettes) collapses into permutation soup fast, regardless of how much topological entropy the graph sampler injects. Beyond the ceiling the farm just reinforces one synthetic aesthetic and risks teaching graphviz's *specific* spline/arrowhead idiom rather than what real degraded thin lines look like.

If the structured-layout and fineline gaps are the actual target, a **purpose-built ruled-form / table generator** (ruled cells, dashed leaders, thin rules, text set in a grid) plus the font-glyph typography mint from `fuel.md` Tier A hit those gaps more honestly than generic flowcharts do.

## Which generators emit clean SVG (measured)

The pipeline's two gates are gate2 (`scripts/gate2.py`) and, downstream, idmap ground-truth derivation (`src/vecml/degrade/idmap.py`). idmap raises `DerivationError` on any CSS `<style>` block (`_check_no_css`), any `url(...)` paint server (gradient/pattern), and >lattice-cap distinct colours; on `DerivationError` the sample falls back to lower-quality `pixels_fallback` labels. gate2 **rejects** `foreignObject` outright and **warns** on `live-text`, `css-block`, `mask/clip`.

Test diagram: a 6-node LR flowchart with one dashed edge and two labelled edges, rendered by each tool.

| generator | `<text>` | `foreignObject` | `<style>` block | gradient | `<mask>` | colour delivery | gate2 tier | idmap outcome |
|---|---|---|---|---|---|---|---|---|
| **graphviz** `-Tsvg` | 8 (live) | 0 | 0 | 0 | 0 | inline `fill=` attrs | **warn** (live-text only) | **clean** after text-to-path |
| **d2** | 8 (live) | 0 | **2** | 0 | 1 | CSS classes (`fill-B1`) | warn (css-block + mask/clip + live-text) | `DerivationError` -> pixel fallback |
| **mermaid** (default) | 0 | **22** | 1 | 0 | 0 | CSS classes + HTML spans | **REJECT** (foreignObject) | unusable |
| **mermaid** (`htmlLabels:false`) | 5 | **12** | 1 | 0 | 0 | CSS classes | **REJECT** (residual foreignObject) | unusable |
| **plantuml** | live (not rendered here) | 0 | inline theme style | 0 (default) | 0 | mostly inline | warn (untested) | likely clean after text-to-path |

Findings:

- **graphviz is the clear winner.** No `<style>` block, no gradients, colours as inline presentation attributes, geometry as `<polygon>` + `<path>`, text as live `<text>` with `font-family`. Its only gate2 flag is `live-text`, which the `usvg` promotion stage removes. It is the one generator whose output idmap can turn into a geometry-derived answer key.
- **d2 is CSS-dependent.** All fills arrive via classes (`fill-B1`, `fill-N7`) defined in two `<style>` blocks, plus a `<mask>` and base64-embedded web fonts. It renders correctly in a full renderer, but idmap's `_check_no_css` rejects it, so every d2 sample drops to pixel-fallback labels. Usable only if a normalizer (usvg) first inlines the CSS into presentation attributes; even then the mask and embedded font add weight for no benefit.
- **mermaid is disqualified.** Default output puts every label in a `<foreignObject>` with HTML `<span>` (22 of them here) which gate2 rejects. `htmlLabels:false` still leaves 12 residual `foreignObject` (edge labels) plus the `<style>` block. On top of that it is the slowest by three orders of magnitude (below). Not worth the fight.
- **plantuml** was not rendered (JVM/jar not installed this session; honest gap). By reputation and format it emits live `<text>` with inline-styled shapes and no CSS-class dependence, so it would sit near graphviz after text-to-path, but this is unverified here and carries JVM startup cost.
- **railroad diagrams** (railroad-diagram JS): clean geometry but typically ship a `<style>` block (css-dependent, same problem as d2) and cover a very narrow content niche (grammar syntax). Not worth it.

## Text-to-path conversion (the shared cost, already planned)

Every generator that matters emits live `<text>`, which is `warn`-tier and, more importantly, gives idmap no path geometry for the glyphs (`text`/`tspan` are in `_PAINTABLE` but carry no `d`). The fix is the exact stage `svg-diagrams.md` already specifies: **`usvg` promotion**. `usvg` (Rust, from the resvg project the renderer already bundles as `resvg-py`) flattens text to paths given fonts, resolves CSS `<style>` into presentation attributes, and normalizes the document. For graphviz that yields an all-paths, CSS-free, gradient-free SVG that idmap eats as clean-tier. Cost is negligible (usvg processes thousands/sec in Rust). Note `usvg` also *drops* `foreignObject` content entirely, which is another reason mermaid's labels cannot be salvaged this way.

`rsvg-convert` is present on the Mac but is raster-only (SVG->PNG/PDF), so it is not a text-to-path tool; `usvg` (`cargo install resvg`) or an equivalent is the addition needed, and it is the same addition the mined-diagram source needs, so there is no farm-specific tooling cost.

## Throughput and cost per 100k (measured)

| generator | mode | rate | note |
|---|---|---|---|
| graphviz | subprocess per render | 24 /sec | cold `dot` process each call |
| graphviz | single-process batch stream | **~11,800 /sec** | many graphs piped through one `dot -Tsvg` |
| d2 | subprocess per render | ~77 /sec | fast warm binary |
| mermaid | mmdc per render | **~1 /sec** | launches headless Chromium per call |

Generation cost is not the constraint for graphviz or d2. 100k graphviz SVGs is ~10 sec of CPU in batch mode, or ~70 min single-core via subprocess; on a $0.30/hr CPU pod that is single-digit cents. The render-to-PNG and wreck cost is shared with every synthetic source and is not diagram-specific. Mermaid's ~1/sec (Chromium per call) makes 100k a ~28-hour serial job before any other objection, confirming it is not viable at scale without a persistent-browser server, which is not worth building for reject-tier output.

**Cost per 100k (graphviz + usvg): effectively free** on the generation side; dominated entirely by the shared render/wreck pass.

## Randomization strategy (if built)

To resist permutation soup for as long as possible:

- **Topology sampling.** Draw from several graph families rather than one: layered DAGs (flowchart), rooted trees (org-chart / mindmap), random sparse digraphs (network), and sequence/state machines. Vary node count (5-40), branching factor, edge density, and rank direction (LR/TB/RL/BT). graphviz's `rankdir`, `nodesep`, `ranksep`, and engine choice (`dot`/`neato`/`fdp`/`circo`/`twopi`) give real layout diversity from one tool.
- **Label sampling.** Draw labels from a broad word/phrase corpus with varied lengths (single word to short phrase), mixed case, occasional numerals/punctuation, and multiple fonts. This is where typography-in-context actually enters the sample; after text-to-path it trains glyph-edge cleanup in a laid-out context, which is the one honestly gap-relevant contribution.
- **Style sampling.** Vary node shapes (box, rounded, ellipse, diamond, plaintext), edge styles (solid/dashed/dotted, straight/spline/ortho), arrowhead types, stroke widths (bias toward thin, since fineline is the target gap), fill palettes (sample many, including monochrome and near-white to stress the wrecker's background logic), and fonts.

Even with all of this, the *rendered* entropy saturates well before the topological entropy does, because the model sees pixels, not graph structure. Two flowcharts with different topologies but the same shape/edge/font/palette vocabulary look nearly identical to a 256px cleanup UNet. That is the ceiling.

## Information-content ceiling

The visual vocabulary is small and closed: roughly {5 node shapes} x {3 edge line styles} x {handful of arrowheads} x {5 layout engines} x {N palettes} x {M fonts}. Topological variety inflates the *count* of distinct files without inflating the *distribution* of local pixel neighbourhoods the model learns from. Genuine incremental value is estimated at **~10k-30k samples**, concentrated on thin-line/arrowhead/text-in-box primitives; beyond that it is permutation soup that (a) adds no new gap coverage and (b) skews the corpus toward one synthetic aesthetic, against the `fuel.md` / moat guidance to prefer gap-filling fuel over volume. There is also an overfitting risk specific to a farm: the model can learn graphviz's *exact* spline curvature and arrowhead polygon, which is not what a real scanned thin line looks like, so a farm-heavy corpus can teach a rendering idiom rather than a degradation-inversion skill.

## Print-shop relevance (honest check)

Against the shop's stated target distribution (`taxonomy.md`, `nas-corpus`): the families that walk in are logos, business cards, letterheads, menus, and forms, degraded by JPEG/scan/halftone/photo. **Diagrams are not in that distribution.** A print shop rarely vectorizes a flowchart, org-chart, or network graph.

The one honest connection is at the *primitive* level: diagrams are dense in thin connector lines, arrowheads, text set inside boxes, and ruled structured layout, and two of the model's three measured gaps are fineline and structured-layout. So a diagram farm is not content-relevant but is weakly primitive-relevant. However the specific print-shop artefact that shares the most with a diagram is a **ruled form or a menu with rule lines**, and a dedicated ruled-form/table generator (ruled cells, dashed leaders, thin rules, gridded text) would produce those exact primitives *in the shop's own layout idiom* far more relevantly than a flowchart does, for the same near-zero generation cost.

## RISKS

- **Redundant with `svg-diagrams.md`.** The mined 85k real-diagram set covers the same primitives with more idiom diversity and no generation code to maintain. The farm only wins if the mined set's license blocker (unstated license) proves fatal for a commercial vectorizer. Decide the mined-source license question first; the farm is a fallback, not a parallel effort.
- **Aesthetic monotony / idiom overfitting.** One-tool output risks teaching the model graphviz's rendering idiom rather than degradation inversion. Mitigate by capping volume low and mixing generators, but mixing pulls in d2/plantuml which need extra normalization.
- **Distribution skew.** Flooding the corpus with diagrams dilutes the gap-filling fuel that actually matters and biases the model away from the print-shop distribution the whole project targets.
- **Content irrelevance.** The farm produces volume, not target-distribution coverage. Treat any sample count from it as volume-only fuel, weighted down accordingly in the mix.
- **plantuml/railroad unverified.** SVG-quality claims for these two rest on reputation, not this session's measurement. Verify before counting on them.

## Recommendation

Do not build the render farm now. Resolve the `svg-diagrams.md` license question first, since it delivers the same primitives at larger scale for no code. If a farm is still wanted after that, build **graphviz-only + usvg text-to-path**, cap it at ~10k-30k, bias generation toward thin strokes and structured/ruled layout, and prefer redirecting the same effort into a purpose-built ruled-form/table generator that hits the structured-layout gap in the shop's own idiom.
