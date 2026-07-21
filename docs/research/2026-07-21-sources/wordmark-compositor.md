# Source study: synthetic word-mark compositor

Feasibility study for one synthetic training-data generator: a word-mark
compositor that programmatically composes open-font glyphs into short
word-marks/lockups (2-6 glyphs, kerned rows, varied size/weight/colour, occasional
arcs/outlines/flat shadows), emits ordinary flat-fill SVG, and feeds the existing
wreck/mint pipeline unchanged. Written 2026-07-21. Companion to
`../2026-07-20-deep-research/fuel.md` (Theme 1, section 2.1), which established
that typography is the #1 measured content gap and that open-font glyphs are
born-vector with zero autotrace risk.

---

## VERDICT

**Feasible, and the highest-leverage single generator available.** Glyph outlines
are born-vector Bezier data extracted directly from font tables (no raster->vector
step, so zero autotrace-contamination risk), and the compositor's only output
contract is "emit a flat-fill SVG file", which the existing `wreck_svg` pipeline
already consumes for every other source. No pipeline change is required.

The generator uniquely earns its keep on **multi-glyph adjacency and same-colour
region separation**, which no existing corpus supplies. The two ready-made glyph
corpora (`starvector/svg-fonts` ~1.93M rows, verified single-character glyphs like
`E_upper.svg`/`j_lower.svg`, license unstated; and Magenta glyphazzn/SVG-Fonts
~14M glyphs) are **single glyphs only**. Single-glyph coverage is therefore already
cheap and oversupplied; the compositor should not re-mint it. What it adds is
kerned pairs, tight/touching tracking, mixed case, curved baselines, and 2-6
region colour lockups: the exact boundary situations the segmentation head is weak
on.

**Useful-sample ceiling estimate: order 1-3 million distinct-carrying word-marks;
practical sweet spot 200k-500k; cheap first experiment 50k.** Beyond ~1M the
generator is in permutation soup (rearranging the same glyph atoms into new
strings adds pixels, near-zero new supervision). Derivation below.

---

## Useful-sample ceiling (information-content honesty)

The nominal permutation space is effectively unbounded (any string x any font x
any style), so counting permutations is meaningless. The question is how many
samples remain **visually distinct AND carry new supervision** before diversity
per sample decays to zero.

Decompose the independent visual axes the cleanup + segmentation task can actually
learn from:

| Axis | Effective distinct values | Note |
|---|---|---|
| Fonts | ~1,900 families, ~1,000 after skeleton dedup | many families are weight/optical variants of one letterform skeleton |
| Glyph inventory | ~70 usable Latin glyphs (A-Z a-z 0-9 + a little punctuation) | the atomic shapes |
| Scale buckets | ~4 | interacts with the 256/512px render + damage |
| Weight buckets | ~3 | thin/regular/bold, or variable-font axis samples |
| Layout | ~3 | straight row / letter-spaced / arc-curved baseline |
| Tracking regime | ~3 | loose / normal / tight-to-touching |
| Colour regime | ~3 | mono, two-tone, per-glyph multi |

**The single-glyph content is ~1,000 fonts x ~70 glyphs = ~70k (glyph, font)
atoms**, already covered many times over by the two single-glyph corpora above.
The compositor must not spend its budget there.

**The compositor-specific new content is the adjacency/separation signal**: an
ordered glyph pair, in a font, at a tracking regime, in a layout. Ordered Latin
pairs = 70^2 ~ 4,900, but only a few hundred are visually distinct "adjacency
situations" per font once bucketed (round/round, straight/round, ascender
collision, kern-pair overlap, digit runs). So the compositor-new cell count is
roughly:

```
~1,000 fonts x ~300 salient adjacency situations x 3 tracking x 3 layout x 3 weight
  ~ 8 million nominal cells, heavily redundant across fonts sharing skeletons
```

After the redundancy discount (same-skeleton fonts, colour being nearly free to a
deltaE-scored cleanup task, most adjacency situations repeating across words), the
**useful ceiling lands around 1-3M samples**. Past that, each new 4-letter string
re-presents glyph atoms and adjacency regimes the model has already seen a few
times, and marginal deltaE gain is unmeasurable.

**Recommended minting schedule**, consistent with fuel.md's "data-constrained but
the typography gap is orthogonal to raw volume" finding:

- **50k** for the first gap-fill experiment (add to an existing ~927k run, measure
  bench deltaE delta on a held-out typography slice, ~1 GPU-day).
- **200k-500k** for the production gap-fill once the 50k experiment shows a
  positive deltaE delta. This is where the measurable gap should mostly close.
- **Do not exceed ~1M** without evidence the deltaE curve is still moving; enforce
  a coverage sampler (below) so the budget spans fonts x glyph x style cells
  rather than random-spamming strings into permutation soup.

The design lever that protects this ceiling is a **coverage sampler**: iterate the
(font, style, tracking, layout) grid and draw strings to fill under-covered cells,
rather than sampling strings i.i.d. This keeps every minted sample near the
frontier of new supervision instead of piling redundant mass on common letters.

---

## PLAN (if feasible)

### Tooling route (recommended: HarfBuzz shaping + fontTools outline extraction)

Three candidate routes were considered:

1. **fontTools outline extraction only.** `TTFont.getGlyphSet()` +
   `fontTools.pens.svgPathPen.SVGPathPen` gives each glyph's outline as an SVG path
   string, placed by nominal advance width. Pure Python, no extra dependency, and
   it is how the glyphazzn/starvector single-glyph corpora were produced. Weakness:
   no kerning, no ligatures, no complex-script shaping. You would reimplement
   spacing from raw `hmtx` advances and get typographically wrong lockups.

2. **HarfBuzz shaping + fontTools outlines (recommended).** `uharfbuzz` (maintained
   Python binding to the industry-standard shaper used by Chrome/Android/Firefox)
   shapes a string into a glyph run with per-glyph `codepoint`, `x_advance`,
   `x_offset`, `y_offset`, applying GPOS kerning, GSUB ligatures, and contextual
   alternates for free. Feed each shaped glyph id to `fontTools` `getGlyphSet()` +
   `SVGPathPen` to get the outline, then translate it by the shaped pen position.
   Marginal cost over route 1 is one mature dependency; it buys real kerning,
   ligatures (ff/fi become single distinct glyph shapes), and correct spacing.

3. **Render `<text>`/`<tspan>` and convert to paths.** Emit live SVG text and let
   the renderer shape it. Rejected: it hands shaping and per-glyph identity to the
   renderer (you cannot introspect glyph boundaries for labelling), and it couples
   the label-derivation step to the exact font being installed and resolved
   identically at derive time. `idmap.py` does list `text`/`tspan` in `_PAINTABLE`,
   so live text would technically label, but the coupling is fragile at small
   sizes and across renderer versions.

**Recommendation: route 2.** Shape with `uharfbuzz`, extract outlines with
`fontTools`, and **emit each glyph as a pre-outlined `<path>` with a flat `fill`
attribute**. Outlined paths make the rendered geometry identical to the labelled
geometry by construction (render == idmap crispEdges render), remove all
font-availability coupling from the mint step, and pass straight through
`idmap-v3`. Evidence for the choice: route 3 loses per-glyph control and adds
render/derive coupling; route 1 gets spacing wrong; route 2 is what production text
synthesizers (SynthTIGER and similar) use, at the cost of one dependency.

Note on training value honesty: real kerning matters more for *visual
plausibility* (the model sees realistic lockups) than for the cleanup task itself
(a few font-units of spacing error barely changes a degraded letterform). It is
worth taking because HarfBuzz makes it nearly free, not because perfect metrics are
load-bearing. The high-value lever is tracking *extremes* (touching glyphs), which
is a post-shaping scale on `x_advance`, not the base kerning.

### Font supply

- **Bulk source: `git clone https://github.com/google/fonts`.** Practical: a few GB,
  ~1,900 families, each with `METADATA.pb` + a license file. Enumerate `.ttf`/`.otf`,
  parse `METADATA.pb` for family/style, read cmap coverage with fontTools to confirm
  the glyphs you need exist before compositing.
- **License split (per fuel.md, corrected):** `ofl/` ~1,700+ families (SIL OFL,
  the flagship text faces including Roboto and Open Sans, both relicensed from
  Apache), `apache/` ~42 mostly novelty/display faces (Apache-2.0, no viral clause),
  `ufl/` Ubuntu family (UFL). Fontsource's JSON API adds per-font Apache/MIT filtering
  if a strictly-permissive feedstock is wanted, but that subset is small and
  stylistically narrow.
- **Recommended policy for the generator:** use the full OFL set for data
  generation (our output is cleaned rasters / label maps, not fonts, so OFL-FAQ
  Q1.25 is narrower than our use; see RISKS), and **record the per-font license in
  each sample's `meta.json`** so a later commercial decision can filter to
  Apache/UFL/MIT feedstock without re-minting. Add a `--license-allow` switch that
  can restrict to the permissive subset for a clean-feedstock ablation.

### Generator design sketch

A single module (e.g. `src/vecml/degrade/wordmark.py`) plus a minting script; it
touches nothing in `pipeline.py`.

```
for each requested sample (driven by the coverage sampler):
  pick font           # from the covered (font,style) grid cell
  pick string         # 2-6 glyphs, mixed case, weighted to under-covered cells
  shape string        # uharfbuzz -> glyph run with positions
  for each shaped glyph:
    outline = fontTools SVGPathPen(glyph)     # born-vector path
    place at (pen_x + x_offset, y_offset), optionally rotate (arc layout)
    assign fill        # mono / two-tone / per-glyph, <= 15 distinct colours
  apply layout         # straight row | letter-spaced | arc (position+rotate along path)
  apply tracking       # scale advances: loose / normal / tight-to-touching
  optional decoration  # flat solid outline stroke, or flat offset drop-shadow path
  emit SVG             # flat fills only, no <style>, no gradient, no filter,
                       # transparent/white page, viewBox framing the lockup + margin
  write foo.svg to the mint input dir
```

Then the **existing** remote data factory mints pairs from that directory exactly
as it does for Openclipart or icons: `wreck_svg(svg, out_dir, ...)` renders,
derives `idmap-v3` labels, and writes the wrecked variants. No new pipeline code.

### Realism levers, value, and implementation cost

| Lever | Training value | Implementation cost |
|---|---|---|
| Real kerning (GPOS) | low for cleanup, high for plausibility | ~0 (comes with HarfBuzz) |
| Ligatures (GSUB) | medium (distinct glyph shapes) | ~0 (comes with HarfBuzz) |
| Mixed case | medium (shape variety) | ~0 (choose the string) |
| Tight tracking / touching glyphs | **high** (forces boundary separation of same-colour adjacent glyphs, the head's weak spot) | ~0 (scale x_advance) |
| Letter-spacing extremes (loose) | medium (isolated-glyph vs run) | ~0 (scale x_advance) |
| Varied size buckets | medium (scale-dependent damage) | low (viewBox/scale) |
| Varied weights | medium | low (more font files / VF axis) |
| Per-glyph / two-tone colour | high (drives 2-6 region multi-class supervision for the head) | low (fill attribute; bound colour count) |
| Curved baseline / arc | medium-high (rotated letterforms under-represented) | moderate ~0.5 day (per-glyph tangent rotation of the path) |
| Flat outline stroke | medium (fineline coverage bonus) | low (stroke attr; idmap coverage-votes it) |
| Flat drop-shadow (offset solid) | low | low (offset duplicate path) |
| Blur/bevel/gradient shadow | negative (breaks idmap AND violates gradients-out) | do not implement |

### Effort estimate

| Task | Days |
|---|---|
| uharfbuzz shape + fontTools outline -> SVG path, one font/one word | 0.5 |
| Google Fonts clone, font enumeration, METADATA/license parse, cmap coverage check | 0.5 |
| Compositor core: straight rows, tracking, sizes, weights, mono/two-tone/per-glyph colour | 1.0 |
| Arc/curved baseline + per-glyph rotation | 0.5 |
| Flat outline + flat drop-shadow decorators | 0.5 |
| Coverage sampler + config + uniqueness/dedup guard | 0.5 |
| Emit clean SVG, wire into mint dir, validate idmap passes on ~200 samples (no DerivationError, region counts sane) | 1.0 |
| **Total to minting-ready generator** | **~4.5 days** |

First experiment after that (mint 50k, add to a run, measure deltaE) is ~1 GPU-day
per the fuel.md recommendation.

---

## Labelling for the 16-class region head

The region head (`DualHead`, `n_classes=16` in `src/vecml/models/unet.py`) is a
**colour-region classifier**, not an instance segmenter. In the existing pipeline,
`idmap-v3` defines a region as a distinct opaque colour in the planar-face stack
(`src/vecml/degrade/idmap.py`), and `data/pairs.py` remaps palette colours to
deterministic slots (background 0, then darkest-to-lightest). This matches the Rust
engine's model: same-colour touching fills are one face, which is what the tracer
will draw.

Consequences for word-marks, stated honestly:

1. **Same-colour letters are ONE region, not one-region-per-letter.** A mono-colour
   word-mark is a single foreground class (plus background). Two touching
   same-colour letters are the *same* class and the head will not split them into
   different classes: that separation is a geometry/tracer job downstream, not the
   head's. Do not label per-letter instances or expect the head to learn them. The
   phrasing "letters as regions" holds only when letters carry distinct colours.

2. **Multi-colour lockups produce the multi-class supervision.** A two-tone or
   per-glyph-colour word-mark yields 2-6 foreground classes + background, naturally
   within the 16-class budget. Real logos routinely use two-tone lettering, so this
   is realistic. Keep distinct colours per sample <= 15 foreground so it fits the
   head (2-6 glyphs with occasional per-letter colour keeps this automatic).

3. **Labels come free from the existing derivation.** Because the compositor emits
   ordinary flat-fill SVG, feed it straight through `idmap-v3`; do not write a
   bespoke label path. Each distinct fill = one region/class, coverage-aware
   supersampling already preserves the thin strokes and hairline serifs that make
   typography the hard boundary case.

The genuine value to the head is not instance labels; it is the abundance of
**small, thin, closely-spaced same-colour regions** whose boundaries against
background (and against adjacent different-colour glyphs) must be located to
sub-pixel accuracy, which is precisely the boundary-label accuracy weakness
typography exposes.

---

## Integration with the wreck/mint pipeline (verified against pipeline.py)

`wreck_svg(svg_path, out_dir, ...)` in `src/vecml/degrade/pipeline.py` takes a
single SVG path and does everything: `render_svg_rgba` -> `_derive_ground_truth`
(idmap-v3, pixel fallback on `DerivationError`) -> palette/label/clean writes ->
per-variant wreck. **The compositor's entire contract is therefore: produce a
valid SVG file.** "SVG in = compatible by construction" holds, subject to these
verified assumptions the compositor must satisfy:

1. **Renderable by the resvg backend** (`renderer.py`). Outlined `<path>` glyphs
   are trivially renderable.
2. **No `<style>` blocks, no gradient/pattern paint servers, no filters.** `idmap-v3`
   raises `DerivationError` on CSS blocks, gradients, or too many translucent
   leaves, which drops the sample to the *pixel* fallback and produces worse labels
   (`_derive_ground_truth`). Use flat `fill="#rrggbb"` attributes and solid paths so
   every sample takes the clean idmap path. This is also why shadows must be flat
   offset solids, never blur filters, and doubles as compliance with the
   gradients-out constraint.
3. **Bounded distinct opaque colours** (<= 15 foreground for the head; also within
   the idmap bitmask's translucent-leaf cap, which is not exercised if you avoid
   translucency).
4. **Do not bake a background.** The pipeline composites over white or a sampled
   random background itself (`_sample_bg`, `bg_mode`) and rewrites palette row 0.
   Emit a transparent page (or a white one) and let the pipeline own the background;
   `bg_mode="random"` then gives free background-colour diversity.
5. **viewBox frames the lockup with margin.** The renderer scales to `size`
   (256 default, 512 serious). A tight-with-margin viewBox keeps glyph stroke widths
   in a realistic pixel range at the training resolution.

Net: add `src/vecml/degrade/wordmark.py` + a minting script that writes SVGs into a
directory; the existing remote data factory mints pairs from that directory with
zero pipeline changes. Validate on ~200 samples that `label_method == "idmap-v3"`
(not `pixels_fallback`) and region counts match the intended colour count before
scaling up.

---

## RISKS

1. **OFL Q1.25 viral-output caveat (restated).** SIL's OFL-FAQ Q1.25 treats a *font*
   produced by an ML system trained on OFL source as a derivative that must stay
   OFL. Our model emits cleaned rasters / label maps, not fonts, so the letter is
   narrower than our use, but it is SIL guidance untested in court and the
   render-to-raster-then-train case is a grey area it does not address. Mitigation:
   record per-font license in `meta.json`; keep a `--license-allow` switch for an
   Apache/UFL/MIT-only feedstock ablation; get legal sign-off before any commercial
   ship. Not a blocker for research minting.
2. **Permutation soup / wasted GPU.** Diversity per sample decays fast; over-minting
   past ~1M burns compute for no deltaE gain. Mitigation: coverage sampler over
   fonts x glyph x style cells, and a uniqueness guard; stop when the held-out
   typography deltaE curve flattens.
3. **Instance-separation expectation mismatch.** The 16-class head cannot separate
   two same-colour touching glyphs into different classes; that is downstream
   geometry. Do not label or evaluate as if it should.
4. **Live-text fragility.** Emitting `<text>` couples labelling to font availability
   and renderer version at derive time. Mitigation: emit pre-outlined `<path>` so
   render geometry == label geometry by construction.
5. **Decoration temptation.** Any blur filter, bevel, or gradient shadow breaks
   idmap (forces the pixel fallback) and violates gradients-out. Keep all
   decoration flat and solid.
6. **Redundancy with single-glyph corpora.** starvector/svg-fonts (~1.93M) and
   glyphazzn (~14M) already oversupply single glyphs; if the compositor mints short
   or 1-glyph marks it duplicates them. Mitigation: weight strongly toward 2-6 glyph
   lockups with real adjacency, which is the compositor's unique contribution.
7. **Synthetic-vs-real distribution gap.** Perfectly-shaped synthetic lockups may
   not match real degraded customer logos (kerning too clean, backgrounds too
   uniform). The wreck pipeline + random background cover part of it; validate the
   closed gap against real NAS pairs (see `../2026-07-21-nas-corpus/`) before
   declaring the typography gap solved.
8. **License stated as unknown for the ready-made glyph corpora.** starvector/svg-fonts
   ships no license on its card and glyphazzn inherits per-font font licenses;
   generating our own glyphs from Google Fonts (license known per family) is
   cleaner than mining those corpora, which is an additional reason to build the
   compositor rather than lean on the pre-vectorised sets.

---

## One-paragraph summary

Building a word-mark compositor is feasible and high-leverage: glyph outlines are
born-vector (zero autotrace risk), the generator's only contract is "emit flat-fill
SVG" which the existing `wreck_svg` pipeline consumes unchanged, and it uniquely
supplies the multi-glyph adjacency and same-colour separation signal that no
existing corpus (the single-glyph starvector ~1.93M and glyphazzn ~14M sets
included) provides. Recommended tooling is `uharfbuzz` shaping (real kerning,
ligatures, tracking) plus `fontTools` outline extraction emitting pre-outlined
paths, fed by a `git clone` of Google Fonts with per-font license recorded. The
useful-sample ceiling is order 1-3M distinct-carrying marks with a 200k-500k sweet
spot and a 50k first experiment, beyond which it is permutation soup; labelling is
free via idmap-v3 (colour-region, not per-letter instance), and the build is about
4.5 days.
