# Source Study — Layout Compositor (synthetic multi-element artwork)

Date: 2026-07-21. Scope: feasibility of one synthetic training-data generator
that programmatically assembles existing clean SVG assets, synthetic word-marks,
text blocks, rule lines and solid/spot backgrounds into synthetic business
cards, flyers, labels and stickers as single flat SVGs, then feeds them to the
existing wreck pipeline unchanged.

Motivation: the current corpus is single-asset icons; the real target
distribution (audited in `../2026-07-21-nas-corpus/content-audit.md`) is
LAYOUTS. By file share the shop's work is 27% small-format cards/labels/stickers
and 35% flyers/leaflets, composed of a logo plus text blocks plus rules plus
backgrounds on a page. The model is trained to clean up wrecked renders of clean
SVGs; it has never seen a composed layout as a clean target, so there is a
distribution gap between what it trains on (one icon, centred) and what it will
meet (a laid-out page).

---

## Verdict

**Feasible, and it is the correct way to close the single-asset-to-layout gap in
the LABELED synthetic tier.** The decisive finding is architectural: the ground
truth machinery (`src/vecml/degrade/idmap.py`) does not analyse geometry
analytically. It mutates the SVG tree, re-renders it through resvg with code
colours, and reads the labels back from pixels (`_render_region_map` ->
`_nearest_labels`). Because the renderer does the geometry, idmap is already
robust to arbitrary nesting, transforms, nested viewports and clip-paths. A
composed layout is just a bigger stack of planar faces, which is exactly the
model idmap was built around. No idmap change is needed to LABEL composed
layouts, provided the compositor emits SVG that avoids idmap's four hard
`DerivationError` triggers (below). The mint driver already consumes a directory
of `*.svg` and calls `wreck_svg` per file (`scripts/wreck.py`), so the
compositor's only integration contract is "write flat SVG files to a directory."

**Useful-sample ceiling.** High as a LABELED data source, because it produces
the one thing the raster-tile route cannot: perfect per-region ground truth on
layout-structured content at scale. It is complementary to, not a competitor of,
the render-then-tile raster route that `../2026-07-21-nas-corpus/yield-estimate.md`
recommends (that route gives real content with NO paint-derived labels;
self-supervised only). The compositor's ceiling is the same ceiling the current
corpus already has: it is synthetic-clean-then-wrecked, so it teaches the model
what a clean layout looks like and how our own degradations damage it, but it
does not contain real customer degradation. Real degraded artwork still has to
come from the ~400 NAS pairs, which stay the validation anchor per
`../2026-07-21-wrecker/realism-diversity.md`. Net: it removes a real and
currently unaddressed structural gap (composition) in the labeled tier; it does
not remove the synthetic-to-real gap. Treat its diversity as bounded by distinct
slot PARAMETERIZATIONS, not by archetype count (the Tobin diversity-floor lesson
carried in the wrecker doc: 10k images from 1k parameterizations beat 10k from
few).

---

## Prior art worth taking from

- **SynthDoG (Donut, Kim et al. 2021).** The canonical synthetic-document
  generator: composite a background, a text layer with rendered fonts, and a
  paper/photo render. The COMPOSITOR LOOP (sample background, place elements,
  render, pair with labels) is the pattern to copy. What NOT to copy: it targets
  OCR, so it optimises photographic realism and renders to raster with box/text
  labels, not to clean flat SVG with per-region paint labels. We want the
  opposite output (clean vector, region labels), so take the loop, not the
  renderer.
- **DocSynth (Biswas et al. 2021), layout-guided document image synthesis.**
  GAN-based generation of document images from a layout of bounding boxes.
  Relevant as evidence that layout-conditioned synthesis is a solved research
  area; not directly usable because it emits raster images, not editable vector.
- **Crello / CanvasVAE (CyberAgent) and PosterLayout / LayoutDM / LayoutTransformer.**
  These model our exact distribution (design templates as per-element vector
  attributes: type, position, size, rotation, opacity, RGBA, font). They are the
  learned-layout route: fit a generative layout model, SAMPLE element boxes, then
  fill the slots with our assets. This is the "fancy version." Caveat carried
  from `../2026-07-21-intent-priors/document-intent.md` §2.6-2.7: every published
  use of these is GENERATION, and their designer-imitation objective pushes toward
  "generic-correct." For CONDITIONING cleanup that is a fidelity hazard; for
  GENERATING TRAINING DATA, generic-correct layouts are acceptable and even a
  reasonable anchor, so the caution does not block using them here. They are still
  not worth the MVP effort (see plan).
- **DocLayNet / PubLayNet.** Real layout datasets. Not a generation source; use
  them only as a reference for realistic element-mix statistics if the synthetic
  distribution needs calibration later.
- **HTML/CSS-to-SVG template route.** Viable in principle but rejected: HTML->SVG
  (headless browser, or a converter) emits `foreignObject`/CSS soup that idmap
  refuses (see triggers). Author SVG DIRECTLY via templating (lxml / svgwrite /
  Jinja SVG templates). Direct authoring is the only route that produces the
  flat-presentation-attribute SVG idmap needs, and it keeps us in control of every
  trigger.

---

## Layout-realism strategy

Three options were considered.

1. **Encode grid/margin/alignment as explicit constraints** (a small constraint
   sampler: page size -> safe margins -> a grid -> slots -> alignment rules).
2. **Sample from a learned layout model** (fit CanvasVAE/LayoutDM on Crello,
   sample element boxes).
3. **Simple heuristic templates** (5-10 hand-authored archetypes with randomised
   slot content).

**Recommended MVP route: heuristic templates plus a light constraint sampler
(a blend of 1 and 3).** Reasons: (a) we do not need designer-realistic layouts,
we need STRUCTURALLY VARIED ones that exercise multi-element cleanup (logo + text
+ rule + background separation, occlusion, mixed spot colours); (b) diversity
that matters is distinct slot parameterizations, which a constraint sampler
delivers cheaply by jittering slot geometry, colour, scale, count and asset
choice; (c) the learned-layout route adds a training pipeline and a dataset
dependency for a realism gain whose value to a data GENERATOR is unproven. Defer
option 2 to the fancy version and only if a template distribution measurably
underserves the model on the NAS-pair validation set.

Concretely for the MVP: a handful of print-real archetypes (business card
landscape, business card portrait, address/shipping label, product sticker,
A6/A5 flyer, DL leaflet), each defined as a page trim + bleed + safe margins +
2-4 named slots (logo, headline word-mark, body text block, contact rule-block,
background field). Fill slots by sampling: an asset for the logo slot, an
outlined word-mark for the headline, outlined lorem for body, 0-2 rule lines,
and a background that is white (bias, matching the audit) or a spot fill kept a
minimum distance from foreground (reuse `_sample_bg`'s logic). Randomise within
grid and margin conventions rather than freely, so the layouts stay plausible.

---

## Asset supply integration

The 1.165M SVG ledger is the clip-art bin for the logo/icon slot. Technical route
to nest existing SVGs safely for our renderer AND for idmap:

- **Placement.** Wrap each asset instance in `<g transform="translate(tx,ty)
  scale(s)">` computed from the asset's own `viewBox` mapped to its target slot,
  or use a nested `<svg x y width height viewBox>` (resvg honours nested
  viewports and clips to them, and idmap re-renders through resvg so it inherits
  that behaviour). Either works; nested `<svg>` is simpler because it
  auto-scales and clips to the slot.
- **THE critical gotcha: id collisions.** Composing several existing SVGs into
  one document collides their internal ids. Two assets that both define
  `#clip1` or `#grad1`, and reference them via `clip-path="url(#clip1)"` or
  `<use href="#...">`, will cross-wire after composition. idmap's `_build_id_map`
  keeps the FIRST-seen id, so a duplicate silently resolves to the wrong element
  and produces a wrong label map with no error. The compositor MUST rewrite every
  id (and every intra-asset reference to it) to an instance-unique namespace
  before composing. This is the single most important correctness step.
- **idmap's four hard `DerivationError` triggers, all under our control because
  we generate the SVG:**
  1. A CSS `<style>` block with content (`_check_no_css`). Inline all paint as
     presentation attributes; never emit `<style>`.
  2. `url(...)` paint servers, i.e. gradients and patterns (`_resolve_colour`).
     Use flat fills only.
  3. More than 32 translucent leaves (`_TRANSLUCENT_CAP`). The yield survey
     already saw one real page raise this at 112 leaves. Keep translucency rare
     in composed layouts; a stack of many semi-transparent assets will blow it and
     lose the whole layout.
  4. More than 208 distinct opaque colours (the code lattice, `_lattice_codes`
     yields 6^3 minus the near-white corner). Unlikely for real cards but reachable
     if many multi-colour icons are stacked; keep the spot-colour budget modest.
- **Reuse the already-filtered corpus.** The assets that feed the wrecker today
  are exactly the idmap-passable subset (CSS/gradient/complex files already
  quarantine via `DerivationError` and get skipped by `wreck.py`'s try/except).
  The compositor should draw its logo/icon slot from that same pre-filtered
  subset (the `svg-stack-simple` / labelled datasets), so asset supply is
  effectively free and every asset is known to survive idmap in isolation. Note
  the caveat: an asset that passes ALONE can still push a COMPOSED layout over the
  translucent-leaf or colour caps when combined; the compositor must budget these
  across the whole page, not per asset.
- **Bypass gate2.** gate2 rejects files over 200 KB ("huge") and warns on CSS
  blocks; a multi-asset composed file can exceed 200 KB from inlined path data.
  gate2 is a corpus-intake filter, not part of the mint path (`wreck.py` does not
  call it), so route compositor output straight to `wreck_svg` and do not run it
  through gate2.

---

## Label-map derivation for composed layouts

The existing idmap machinery handles nested composed SVGs without modification,
subject to the triggers above. Verified against the implementation:

- `_walk_collect` expands `<use>`/`<symbol>`, recurses through `<g>` and nested
  `<svg>`, carries inherited paint context down, and records painted leaves. It
  does not need to understand transforms or viewports analytically because
  `_render_current_tree` serialises the mutated tree and lets resvg apply them.
- Occlusion is handled correctly: the face stack takes the topmost opaque owner,
  so a text block placed over a busy icon yields the right regions. Overlapping
  translucent shapes still form their own third face.
- clip-paths survive: `clipPath` is a non-paint container (kept in the tree, not
  collected as paint), so resvg clips and the readback picks up only visible
  pixels.
- The palette is sampled from the clean render per region, and near-white faces
  fold into background, so a white card body correctly reads as background.

**Two interface notes for the downstream model, not compositor blockers:**

- idmap emits up to 208 foreground faces, but the UNet region head is
  `n_classes=16` (`src/vecml/models/unet.py`). A busy layout can exceed 16
  distinct faces. Quantisation to the head's capacity is existing downstream
  machinery, but the compositor should keep the realistic spot-colour budget low
  (roughly <= 8-12 distinct foreground colours per card) both to match real print
  work and to sit inside the head.
- `<image>` is not paintable in idmap. The compositor must not place embedded
  raster; it composes vector only. (This is the same blindness that sinks the
  full-SVG NAS route in the yield survey; here it is a constraint we simply
  respect.)

---

## Typography supply (dependency, not owned here)

Text is the compositor's hardest external dependency. Two sub-routes:

1. **Emit `<text>` and rely on resvg font resolution.** Fragile: resvg needs the
   font available in its fontdb; a missing or mismatched font renders nothing or
   a fallback, and idmap collects the `text`/`tspan` leaf but the render disagrees
   with it, producing wrong labels silently. Not recommended.
2. **Pre-outline glyphs to `<path>` at generation time** (freetype/fonttools glyph
   -> path `d`). Deterministic, font-free at render, and idmap sees ordinary flat
   path faces. Recommended.

Route 2 means the layout compositor DEPENDS ON a separate word-mark / text
generator that owns a font collection (e.g. an OFL Google Fonts set) and the
glyph-to-path step. That generator is out of scope for this study but is a hard
prerequisite for realistic type. The layout compositor can ship an MVP before it
exists by using placeholder text (outlined lorem blocks, or grey rule-bars
standing in for text runs), then swap in real outlined word-marks once the text
generator lands. Flag this dependency explicitly in sequencing: realistic
typographic layouts are gated on the word-mark compositor.

---

## Diversity ceiling

- **Asset variety: effectively unbounded** for the logo/icon slot (1.165M ledger,
  idmap-passable subset in the hundreds of thousands).
- **Layout variety: bounded by distinct slot parameterizations, not archetype
  count.** 5-10 archetypes is not the diversity number that matters; the number
  that matters is how many distinct slot geometries, colour schemes, scales,
  counts and asset choices the sampler produces. Randomise within grid/margin
  conventions to push this up cheaply.
- **Typographic variety: gated on the word-mark generator** (font count, size,
  weight, tracking). Until it lands, type diversity is a placeholder.
- **Realism ceiling: synthetic-clean.** These are structurally plausible layouts,
  not real jobs. The wrecker doc's rule applies unchanged: score the trained model
  on the frozen real NAS-pair validation set, not on a synthetic bench, or the
  synthetic-layout distribution's biases go unmeasured.

---

## Plan (if feasible)

MVP, template-based, 5-10 archetypes. Output contract: write flat idmap-clean
SVG files to a directory; the existing `wreck.py` + `wreck_svg` + idmap path
consumes them unchanged.

1. **SVG composer core (~2 days).** Asset loader that parses viewBox, rewrites
   ids to an instance-unique namespace (the critical correctness step), and
   places an asset into a slot via nested `<svg>` or a transform group. Inlines
   all paint as presentation attributes; asserts no `<style>`, no `url()` paint,
   translucent-leaf and distinct-colour budgets tracked across the whole page.
2. **Archetype templates + constraint sampler (~2 days).** 5-10 print-real
   archetypes (card landscape/portrait, shipping label, product sticker, A6/A5
   flyer, DL leaflet), each a trim+bleed+margins+grid with named slots. Sampler
   fills slots with jittered geometry, spot/white backgrounds (reuse
   `_sample_bg` distance logic), 0-2 rule lines.
3. **Text/word-mark stub (~1-2 days).** Placeholder outlined text (lorem blocks,
   rule-bars) so the MVP runs before the real word-mark generator exists. Real
   outlined word-marks are a later swap-in and a documented dependency.
4. **Integration + QC (~1 day).** Emit ~200 sample layouts, run through
   `wreck_svg`, confirm idmap derives meaningful multi-region ground truth (the
   audit's `coverage_match` / QC flags), and eyeball `labels_view.png`. Confirm no
   id-collision mislabels and no cap `DerivationError`s at layout scale.

**MVP effort: ~5-8 engineering days**, ~$0 compute (generation is CPU; the mint
already runs on the CPU data-factory pod).

**Fancy version (defer):** fit a CanvasVAE/LayoutDM layout model on Crello and
sample element boxes; real font pairing and typographic hierarchy; richer asset
semantics (slot-appropriate asset selection). **+2-4 weeks**, and the ROI is
unproven for a data generator given that generic-correct layouts are acceptable
here. Do not build it until a template distribution is shown to underserve the
model on the real-pair validation set.

Asset dependencies: (a) the idmap-passable subset of the 1.165M ledger (already
exists, feeds the wrecker today); (b) a word-mark / text generator emitting
outlined-glyph paths (does NOT yet exist; hard prerequisite for realistic type;
MVP runs on placeholder text without it).

---

## Risks

1. **id collisions across composed assets -> silent mislabelling.** idmap keeps
   the first-seen id; duplicate ids from different assets cross-wire with no
   error. Mitigation: instance-unique id rewriting, and a QC check that composed
   ground truth matches the visible render (the audit's coverage metric already
   catches gross mismatches).
2. **Cap `DerivationError`s at layout scale.** Translucent-leaf cap (32) and
   distinct-colour cap (208) are per-DOCUMENT; assets that pass alone can blow
   them when stacked, losing the whole layout. Mitigation: budget translucency
   and colours across the page; keep translucency rare.
3. **Font dependency.** Realistic type needs the word-mark generator; resvg
   `<text>` is fragile and can desync render from labels. Mitigation: outline
   glyphs to paths; ship MVP on placeholder text.
4. **Synthetic-layout realism gap.** Layouts are plausible, not real; biases go
   unmeasured if scored on a synthetic bench. Mitigation: validate on the frozen
   real NAS-pair set (`../2026-07-21-wrecker/realism-diversity.md` Rec 1).
5. **Diversity ceiling from too few parameterizations.** Adding archetypes is not
   the lever; distinct slot parameterizations is. Mitigation: randomise slot
   geometry/colour/scale/count/asset, not just template identity.
6. **16-class head vs busy palettes.** Layouts can exceed the region head's
   capacity. Mitigation: cap the realistic spot-colour budget (~8-12) at
   generation; downstream quantisation handles the rest.
7. **Embedded-raster temptation.** idmap is raster-blind; any placed `<image>`
   would be silently dropped from labels. Mitigation: compose vector only, hard
   assert no `<image>` in output.
