# Small-but-clean SVG collections: feasibility for the cleanup-UNet corpus

Date: 2026-07-21. Scope: five bundles of small, hand-authored, permissively
licensed born-vector collections assessed as feeder data for the Stage-1
cleanup UNet. Counts pulled live from the GitHub trees API on 2026-07-21;
licenses and prior context from `docs/research/2026-07-20-deep-research/fuel.md`
(adversarially verified 2026-07-20). Register: factual changelog.

## Frame

- Ledger in use: **1.165 M unique clean-tier SVGs** already minted from
  StarVector `svg-stack` (gate2 clean pool is 1.46 M of 2.28 M rows).
- Measured gaps: colour-rich multi-region art, typography, fineline strokes.
  Gradients are out of scope (excluded by design; flag gradient-heavy subsets).
- Pipeline: `scripts/gate2.py`. Relevant gate behaviour for these sets:
  - **reject** on `huge` (>200 KB), `raster`/`raster-b64`, `script`,
    `external-ref`, `no-drawables`, `no-size`, `off-canvas`.
  - **warn** (still usable as GT) on `editor-junk` (inkscape/sodipodi ns),
    `css-block` (`<style>`), `filter`, `mask/clip`, `live-text`, `node-heavy`,
    `trace-suspect`. Only **reject** removes a sample.
  - Gradients are feature-flagged, not rejected by gate2 itself; the
    gradients-out constraint is enforced by **excluding gradient variants at
    fetch time**, not by the gate.
- Spot-checked files (twemoji 450 B, tabler outline 449 B, flag-icons us.svg
  648 B, fluent flat 1.3 KB): all tiny, single-purpose, zero `<style>` / mask /
  gradient / editor namespace / embedded raster. This is the cleanest class of
  input the gate sees. Expected reject rate across all five bundles is near
  zero; survival is dominated by the pre-fetch variant filter, not the gate.
- **Dominant dedup axis is the ledger, not cross-set overlap.** The same glyph
  (a "heart", a "grinning face") drawn by Tabler vs Phosphor vs Twemoji vs
  Fluent is a *different drawing* = different pixels = legitimately distinct
  training samples. Style diversity is the point. The real overlap risk is
  against `svg-stack`, whose composition includes icon and emoji subsets that
  already contain canonical copies of several of these exact sets.

---

## 1. Emoji sets (flat variants only) — colour-rich multi-region gap

| Set | Count (SVG) | License | Flat vs gradient | Fetch | Gate survival | Net-new est. |
|---|---|---|---|---|---|---|
| Noto Emoji (legacy flat) | 3,731 | Apache-2.0 (images) | flat solid multicolor | `git clone googlefonts/noto-emoji` → `svg/` | ~99% clean/warn | low (likely partly in svg-stack emoji subset) |
| Twemoji (jdecked fork) | 4,009 | CC-BY 4.0 (art), MIT (code) | flat solid multicolor | `git clone jdecked/twemoji` → `assets/svg/` | ~99% clean | low (canonical; likely in svg-stack) |
| OpenMoji (color) | 4,495 | **CC-BY-SA 4.0 (share-alike)** | flat multicolor + outline variant | `git clone hfg-gmuend/openmoji` → `color/svg/` | ~99% clean | med style is distinct; **copyleft risk** |
| FxEmoji | ~1,000 usable flat (11,789 files gross) | Apache-2.0 | flat solid multicolor | `git clone mozilla/fxemoji` (gh-pages) | high, but gross tree carries many non-emoji/size variants | low-med (distinct older style) |
| Blobmoji | 3,750 | Apache-2.0 (images), OFL (font) | flat multicolor (blob) | `git clone C1710/blobmoji` → `svg/` | ~99% clean | med (Noto-derived shapes, distinct blob style) |
| MS Fluent Emoji (Flat only) | 3,145 | MIT | Flat=solid (**keep**); Color=gradient, 3D=raster (**exclude**) | `git clone microsoft/fluentui-emoji`, filter `/Flat/` | ~99% clean; excl. Color/3D pre-fetch | med-high (distinct style, unlikely in svg-stack) |

Emoji subtotal (flat, gross): **~20,100 SVG**. All flat solid-fill
organic-curve multi-region shapes with no typography — a direct hit on the
colour-rich gap and compatible with gradients-out **once Color/3D variants are
dropped**. FxEmoji is the messy one: its 11,789-file tree bundles OS/UI glyphs
and size variants; the genuinely useful flat-emoji slice is ~1 k, so treat its
gross count with caution.

---

## 2. Major open icon sets — mono/stroke, fineline gap

| Set | Count (SVG) | License | Char | Fetch | Gate survival | Dedup risk vs ledger |
|---|---|---|---|---|---|---|
| Tabler | 6,166 (5,112 outline + filled) | MIT | 2px stroke, 24px grid, adjustable width | `git clone tabler/tabler-icons` → `icons/` | ~99% clean | med (large, some in svg-stack) |
| Phosphor | 9,072 (~1,500 base × 6 weights) | MIT | thin→filled across 6 weights | `git clone phosphor-icons/core` → `assets/` | ~99% clean | low-med (6-weight span is fresh fineline material) |
| Material Symbols | ~3,300 base (×3 styles ×fills/weights = ~10 k files) | Apache-2.0 | mono, fineline at low weight | `git clone google/material-design-icons` or npm `material-symbols` | ~99% clean | **high** (canonical; near-certainly in svg-stack) |
| Font Awesome Free | 2,883 (solid+regular+brands) | CC-BY 4.0 (art), OFL (font) | mono solid+outline; **Brands=trademarks** | `git clone FortAwesome/Font-Awesome` (7.x) → `svgs/` | ~99% clean | **high** (canonical); drop Brands (~1.9 k left) |
| Lucide | 1,748 | ISC | 2px stroke, 24px | `git clone lucide-icons/lucide` → `icons/` | ~99% clean | med (active Feather fork) |
| Iconoir | 1,671 | MIT | thin stroke | `git clone iconoir-icons/iconoir` → `icons/` | ~99% clean | low-med |
| Feather | 287 | MIT | 2px stroke, 24px | `git clone feathericons/feather` → `icons/` | ~99% clean | **high** overlap: Lucide is its superset — mostly redundant |

Icon subtotal (gross): **~25,100 SVG** (~23 k after dropping FA Brands and
Feather-in-Lucide). Note on the gate: stroke icons that set `stroke`/`fill` on
the root `<svg>` (Feather/Lucide style) rather than per-path still pass — gate2
does not treat a bare `<path>` with no local fill/stroke as hidden (the hidden
test requires `fill=="none"`, and an absent attribute is `None`), so these
count as visible drawables. Survival confirmed clean on spot-check.

**Value caveat:** mono icons are the domain the corpus *already* over-covers
(fuel.md Tiers B/D). Their marginal value is not volume — it is the **fineline
slice** (Tabler/Lucide/Iconoir/Feather 2px strokes; Phosphor thin weight;
Material low weight), which attacks a measured gap. Font Awesome and Material
are near-certain to be partly present in svg-stack already.

**Fluent System Icons (not requested but the biggest clean icon supply):**
fuel.md flags `microsoft/fluentui-system-icons` at ~18,600 mono + ~857 color,
MIT, almost certainly *not* in svg-stack. If icon volume is wanted, that single
repo dwarfs this whole bundle and is worth a look before the smaller sets.

---

## 3. Flags

| Set | Count | License | Fetch | Gate survival | Net-new |
|---|---|---|---|---|---|
| lipis/flag-icons | 542 (271 countries × 1×1 + 4×3) | MIT | `git clone lipis/flag-icons` → `flags/` | ~99% clean (SVGO-cleaned, tiny) | ~500 |
| Wikimedia flag sets (country/historical/subnational) | thousands | per-file (mostly PD, some CC-BY-SA for arms) | MediaWiki API (`allimages`, category members) | high, but per-file license read + arms can be `node-heavy`/warn | large but license-fragmented |

Flags are clean flat colour-region born-vector but **reinforce existing
strength** (flat multi-region) rather than a gap. lipis is a trivial clone;
Wikimedia is a larger but per-file-license, API-fetch effort. Arms-bearing and
subnational flags add welcome complexity but drag in CC-BY-SA and heavier paths.

---

## 4. US-government public-domain vector art

| Source | Count / form | License / access | Fetch | Gate survival | Net-new |
|---|---|---|---|---|---|
| NPS symbol library | 2,611 SVG files (~600 unique symbols × formats/sizes) | PD-adjacent (US gov work) | `git clone nationalparkservice/symbol-library` (gh-pages) | ~99% clean; mono pictograms | ~600 unique, redundant (mono icon domain) |
| AIGA / DOT transportation symbols | 50 | Public domain | AIGA site download / mirrors | ~99% clean | 50, trivial |
| NASA insignia / brand marks | handful of vector marks (meatball, worm, program logos) | PD (US gov) | NASA brand/identity pages; per-asset | insignia flat; some renderings carry gradients (exclude those) | tens |
| CDC PHIL | photographic image library | PD | web/API | **rejects at gate — raster, not vector** | ~0 |

Gov PD art is small and **mostly redundant mono pictograms** (NPS, AIGA) or
not vector at all (CDC PHIL is a photo library and would be rejected as
`raster`). NASA yields a handful of insignia; drop any gradient renderings.
Net gov contribution is low value and low count; include NPS/AIGA opportunistically.

---

## 5. Kenney.nl CC0 game asset packs

| Aspect | Finding |
|---|---|
| Overall library | ~60 k+ assets, CC0-1.0 |
| Vector fraction | **PNG-first**; only a few packs ship SVG (Board Game Icons ~250+ vector, a subset of UI/shape packs) |
| Count (vector) | low hundreds |
| Fetch | per-pack ZIP from kenney.nl (no bulk API); vector packs must be hand-picked |
| Gate survival | high on the SVG subset (flat) |
| Net-new | few hundred, low priority |

Kenney is a PNG ecosystem; the born-vector slice is small and the fetch is
manual per-pack. Not worth special handling beyond grabbing the couple of
CC0 vector packs if convenient.

---

## Aggregate

| Bundle | Gross SVG | Gate-survivable | Net-new (post svg-stack dedup) | Gap hit |
|---|---|---|---|---|
| Emoji (flat) | ~20,100 | ~99% | ~13,000 | **colour-rich multi-region (primary)** |
| Icon sets | ~25,100 (~23 k usable) | ~99% | ~12,000 | fineline strokes (secondary); rest redundant |
| Flags (lipis) | 542 | ~99% | ~500 | reinforces strength |
| Flags (Wikimedia) | thousands | high | large, license-mixed | reinforces strength |
| US gov PD | ~2,700 | ~99% (CDC excluded) | ~700 | redundant mono |
| Kenney (vector) | ~few hundred | high | ~few hundred | minor colour |
| **Total (core, ex-Wikimedia)** | **~48,700** | **~99% (~48 k)** | **~26,000** | colour + fineline |

### VERDICT

Feasible, cheap, and clean, but **small and value-concentrated**. Total core
yield is **~48 k gate-survivable SVGs, of which ~26 k are net-new** after
discounting svg-stack overlap and intra-bundle redundancy — roughly a **2–2.5%
volume bump on the 1.165 M ledger**. Volume is not the reason to do this. Two
slices carry the value:

1. **Flat emoji (~13 k net-new)** — the strongest item in the bundle. Flat
   solid-fill organic multi-region shapes with no typography, spanning six
   distinct art styles (Noto, Twemoji, OpenMoji, FxEmoji, Blobmoji, Fluent).
   Directly fills the colour-rich multi-region gap and is compatible with
   gradients-out once Color/3D variants are excluded at fetch.
2. **Stroke icons (~12 k net-new)** — Tabler/Lucide/Iconoir/Feather 2px strokes
   plus Phosphor's thin weight attack the fineline gap. Lower confidence: the
   tracer ceilings were measured on filled icons and are untested on hairline
   art, so validate on a hairline-only slice before weighting them up.

Neither bundle touches **typography** — that gap is served by the fonts bundle
(fuel.md Tier A), not by any collection here.

### PLAN

Effort is **low** — almost entirely `git clone` + directory glob + gate2 +
hash-dedup. Estimated **~1 engineer-day** for the core bundle, no GPU.

Fetch order (value-first):

1. **Flat emoji** — clone the six repos, keep flat variants only
   (`svg/`, `assets/svg/`, `color/svg/`, `/Flat/`); explicitly drop Fluent
   Color/3D and any Noto 3D. Highest gap value.
2. **Stroke icons** — clone Tabler, Lucide, Iconoir, Phosphor, Feather; take
   outline/thin variants first for the fineline slice. Skip Feather if Lucide
   is taken (superset).
3. **Filled/mono icons** — Material Symbols, Font Awesome (drop Brands) — only
   if bulking icon volume; heavy svg-stack overlap expected, dedup hard.
4. **Flags** — lipis clone (trivial); defer Wikimedia unless subnational
   variety is specifically wanted (API + per-file license work).
5. **Gov PD + Kenney** — opportunistic: NPS/AIGA clone, a couple of Kenney
   vector packs. Skip CDC PHIL (raster).

Then: single hash-based dedup pass against the existing ledger, run the whole
merged set through gate2 for the reject sweep, tag each source for per-source
validation slices (mixture-ratio finding from fuel.md §2.6).

### RISKS

- **svg-stack overlap unmeasured.** Font Awesome, Material Symbols, Noto and
  Twemoji are canonical and near-certainly partly present in the ledger. The
  ~26 k net-new figure assumes moderate overlap on those; a content hash
  against the ledger before minting will pin it down. Emoji *style* diversity
  (six renderings of the same codepoints) is real net-new even where codepoints
  repeat.
- **License fragmentation for a commercial model.** OpenMoji is **CC-BY-SA 4.0
  (share-alike)** — a copyleft contamination risk; Twemoji, Font Awesome art,
  and lipis-plus-Wikimedia-arms are **CC-BY** (attribution). Noto Emoji,
  Blobmoji images, FxEmoji, Fluent, Tabler, Iconoir, Feather, NPS, Kenney are
  permissive (Apache/MIT/ISC/CC0/PD). Font Awesome **Brands are trademarks** —
  exclude them regardless of license.
- **Gradient/3D variants** must be filtered at fetch, not trusted to the gate —
  gate2 flags gradients as a feature but does not reject them. Fluent Color,
  Fluent 3D, and Noto 3D violate gradients-out.
- **Fineline tracer ceiling untested.** The stroke-icon value depends on the
  Rust/vtracer back half handling hairlines; measure on a hairline slice before
  upsampling (open question 3 in fuel.md).
- **FxEmoji gross count is inflated** by non-emoji/size-variant files; real
  flat-emoji yield is ~1 k, not 11.8 k.
- **CDC PHIL is raster** and contributes nothing; do not budget for it.
</content>
</invoke>
