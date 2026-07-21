# Source Feasibility: Wikimedia Commons SVG files

Single-source study for the cleanup-UNet / tracer fuel programme. Scope: can Wikimedia Commons'
SVG holdings supply gap-filling training data (typography, fineline, colour, heraldry-like complex
art), at what net yield, and at what cost. Figures are grounded against Commons' own statistics
where possible; estimates are flagged as estimates. Companion to `2026-07-20-deep-research/fuel.md`
(row 21, 29, 50, 53 there cover Commons in passing; this doc supersedes them for planning).

Data pulled 2026-07-21 from Commons `Special:MediaStatistics` (refreshed 2026-07-19), category
pages, `Commons:Dumps_and_backups`, and `Commons:Licensing`.

---

## VERDICT

**Feasible but second-order and hard-blocked.** Commons is the largest single SVG pool in
existence, and it is the only source that attacks two of our gaps that nothing else covers well:
heraldry-grade complex hand-built art (fineline + high path complexity) and typography-in-context
(labelled maps and diagrams). But its value is almost entirely locked behind the one dependency we
have not built: a structural autotrace detector. Commons' most valuable buckets (portraits, logos,
vectorised emblems, detailed maps) are also its most autotrace-polluted, and the on-wiki trace tag
is too sparsely applied to filter on. Bulk access is also awkward: no media dumps since 2013, the
MIME-search API is disabled, so millions of files can only be reached by category traversal, a paid
enterprise feed, or a slow polite crawl.

- **Do not run a bulk 5M scrape.** Most of that volume is icons / flags / simple clipart that is
  redundant with the existing corpus (fuel.md Tier D) and would dilute, not fill, the gaps.
- **Net usable, targeted:** an estimated **~0.3M-0.8M** gap-relevant born-vector files, and that
  range is *conditional on the autotrace detector existing*. Without the detector the honest net is
  near zero, because minting untraced-looking-but-actually-traced complex art teaches the tracer the
  wrong target, which is the exact failure fuel.md flags.
- **Correct posture:** defer to Phase 2, behind the detector, and harvest narrowly by category
  (heraldry + labelled maps/diagrams), not by MIME sweep.

The licensing question (BY attribution + SA share-alike obligations at dataset scale for a
commercial model) is a legal question, recorded below, not a feasibility blocker: Commons carries no
non-commercial content at all, so nothing here is commercially forbidden outright.

---

## 1. Actual SVG file count

From `Special:MediaStatistics` (refreshed 2026-07-19):

| Metric | Value |
|---|---|
| **SVG files (`image/svg+xml`)** | **5,228,764** |
| Share of all Commons files | 3.61% |
| Combined SVG size | 1.19 TB |
| Total Commons files (all types) | 144,684,232 |

For context the raster formats dwarf it (JPEG 116.9M / 80.8%, PNG 5.48M, TIFF 4.85M), so Commons is
photo-dominated; SVG is a 3.6% slice but in absolute terms 5.2M is larger than every packaged
born-vector corpus in fuel.md except nyuuzyou/svgfind (3.66M) and the FIGR-derived icon sets. It is
the single largest SVG pool named in the whole census.

---

## 2. Bulk access route

**No SVG-specific dump, and no media dumps of any kind since ~2013** (`Commons:Dumps_and_backups`,
confirmed 2026-07-21). There is an open request to resume media dumps; treat it as not-available.
What does exist:

- **XML *metadata* dumps** (`dumps.wikimedia.org/commonswiki/`): page text, revision history,
  categories, templates. Useful to *enumerate and pre-filter by license template / category / trace
  tag offline before ever touching a file*, but they contain no image bytes.
- **MediaWiki API** is the practical byte route, but `list=allimages&aimime=image/svg+xml` is
  **disabled in Miser Mode** (`mimesearchdisabled`), so you cannot page all SVGs by MIME. Reachable
  instead via:
  - `list=categorymembers` traversal of the SVG category tree (deep, diffused into thousands of
    subcats; needs recursive enumeration).
  - `list=search` with `filetype:svg` / `haswbstatement` structured-data queries.
  - PetScan / Quarry (Wikimedia's own category-intersection and SQL tools) to produce target file
    lists offline, then fetch bytes via `imageinfo` URLs.
- **Wikimedia Enterprise** is the sanctioned high-volume commercial feed (snapshots + realtime API;
  free tier / on-wiki account access exists). This is the polite route if a large pull is ever
  justified; a raw crawl of millions of files against the public API is not polite.
- **comload (GPLv3)** and WikiLovesDownloads are existing category-scoped downloader tools; fine for
  a targeted tens-of-thousands harvest.

**Politeness:** public API etiquette is serial or low-concurrency (~1 req/s, `maxlag`, a real
`User-Agent` with contact). At that rate a 5M-file sweep is a multi-week crawl; a targeted
~150k-file category harvest is roughly 2-4 days. This alone argues against bulk.

---

## 3. License mix and the ML-training implication

**Commons accepts only free content; commercial use must be permitted** (`Commons:Licensing`). NC
licenses (CC-BY-NC / -NC-SA / -NC-ND) are explicitly rejected. So unlike arXiv figures, FloorPlanCAD,
CubiCasa, or DESCAN-18K in fuel.md, **there is no non-commercial-blocked content here** and no
per-file commercial veto to screen for.

The mix, per-file, is CC0, CC-BY, CC-BY-SA, PD (expired / government / ineligible), GFDL, GPL, and
Free Art License. Commons does not publish a license histogram for the SVG slice, so fractions below
are estimates from known composition:

| Bucket | Est. fraction of SVGs | Basis |
|---|---|---|
| PD + CC0 (attribution-free) | ~30-45% | flags, many coats of arms, PD-old maps, PD-ineligible simple shapes, PD signatures (the `PD signature:SVG` category alone is 9,202 F) |
| CC-BY (attribution only) | ~15-25% | self-published icons/diagrams electing BY |
| CC-BY-SA (attribution + share-alike) | ~30-45% | the default self-license; dominant for user-made diagrams/logos/maps |
| GFDL / GPL / FAL | small tail | legacy uploads |

**ML-training implication (legal question, not a verdict):**

- **PD + CC0** carries no downstream obligation. This is the clean tier and it maps onto the exact
  buckets we want (heraldry, flags, PD maps).
- **CC-BY** requires attribution. At dataset scale this means retaining per-file provenance
  (author + license + source URL) and reproducing it in a dataset NOTICE; mechanically tractable, we
  already track source IDs through `gate2`.
- **CC-BY-SA** adds share-alike. Whether a model trained on SA-licensed renders, and its weights, is
  a derivative that must itself be SA is unsettled and identical in shape to the OFL Q1.25 question
  fuel.md already routed to counsel. For a commercially-sold vectorizer this is a lawyer decision.
  The cheap engineering hedge is to **filter to PD+CC0 (+optionally CC-BY) and drop SA entirely**,
  which the metadata dumps let us do offline for free before harvesting bytes.

---

## 4. Content mix by category

Commons diffuses SVGs into a deep category tree; top-level category pages show only directly-filed
files, not tree totals, so exact per-bucket counts require PetScan enumeration. Structure observed
2026-07-21:

- **Icons** — `SVG icons` 5,239 F directly + 18 subcats. Redundant with existing corpus.
- **Flags** — `SVG flags`: 206 country subcats + by-subject/color/shape/aspect. Thousands of files.
  Flat, born-vector, clean, but reinforces existing strength (fuel.md Tier D).
- **Coats of arms** — `SVG coats of arms`: `by location` (8 C deep), `by subject`, elements (22 C),
  hatched (24 F), 3D. This is the **standout gap-filler**: dense hand-built paths, fineline charges,
  high complexity, large PD/CC0 subset. This is the bucket no other census source supplies.
- **Maps** — `SVG maps`: `by language` 132 C, `by region` 17 C, `by theme` 21 C, plus
  `SVG maps:Translation possible` **7,280 F** and `SVG maps:Path text` **1,554 F**. These two
  subcats are direct evidence of heavy **live-text** content, i.e. typography-in-context, our #2 gap.
- **Diagrams** — `SVG diagrams` 926 F + long tail; life-science `DBCLS` 1,638 F. Small but clean,
  text-dense, label-rich.
- **Logos** — large, but trademark-laden and heavily autotraced from raster; low value / high risk.
- **Signatures** — `PD signature:SVG` 9,202 F: PD, but thin traced strokes of uneven provenance.

**Estimated gap-relevant fraction:** the typography-rich (labelled maps + diagrams) and
illustration/heraldry-rich buckets that actually attack our gaps are on the order of **~10-20% of
the 5.2M** by count. The remaining ~80% is icons / flags / simple clipart / logos that are either
redundant (Tier D) or high-risk (logos). Maps and flags do *not* numerically dominate the 5.2M the
way intuition suggests, because the icon and clipart long tail is enormous, but maps do dominate the
*text-bearing* subset.

---

## 5. Autotrace pollution — the blocking dependency

This is the crux. Commons is a heavy autotrace-upload venue: portraits, logos, and many "clipart"
SVGs are Potrace / Illustrator Image-Trace / vectorizer output from raster, which is precisely the
input distribution our tracer must *not* learn as ground truth.

- **The on-wiki tag is not a usable filter.** `Category:Created with Potrace` is sparsely populated
  (main category showed 12 F + 2 subcats on 2026-07-21) against an obviously far larger real
  trace population. Uploaders rarely apply the template. Tag-based filtering, which works for
  Openclipart's `filter autotrace` tags (fuel.md §2.3), **does not transfer to Commons.**
- **fuel.md §2.5 stands: no off-the-shelf autotrace classifier exists.** Detection must be
  structural (topology: filled even-odd paths with zero strokes; primitive absence: 100% path-Bezier
  with no `<circle>`/`<rect>`; path/command explosion >8 paths / >32 cmd / >100 Bezier; palette
  quantisation; editor-namespace fingerprints).
- **Our current `gate2.py` heuristic is tuned for the icon corpus, not for Commons.** Its only trace
  signal is `trace-suspect` (>800 cmds and >70% tiny chords), and it fires only as a WARN. Commons'
  legitimately-complex hand-built art (detailed heraldry, dense maps) trips the same command-explosion
  thresholds as autotraced portraits, so the icon-tuned filter cannot separate "complex because
  hand-drawn" from "complex because traced" on this distribution. That collision is the whole
  problem: the gap-filling value (complexity) and the contamination signal (complexity) look alike.

**Consequence:** Commons cannot be minted safely until a structural autotrace detector is built and
*validated specifically on Commons complex art* (heraldry vs traced portrait is the hard case). This
is the prerequisite, and it is more work than the ~1 engineer-day fuel.md estimated for icon-corpus
heuristics, because the Commons distribution is the adversarial case the heuristics are weakest on.

---

## 6. Live-text prevalence

High, and this is a feature not a bug. Maps and diagrams carry `<text>` elements (map subcats
`Translation possible` 7,280 F and `Path text` 1,554 F are direct evidence; diagrams are labelled by
construction). `gate2.py` flags `<text>` as a WARN (`live-text`), not a reject, which is correct:
live text is our #2 gap, so we *want* these, but they need rendering with the referenced fonts
present (missing-font substitution corrupts the ground truth) or the text converted to outlines
before rendering. A meaningful fraction of the text-bearing files will also reference fonts not
installed in the render environment, which is an additional per-file yield loss unless we pre-flatten
text to paths (fontTools can, and doing so also removes the live-text render risk).

---

## 7. Expected funnel

Two tracks. The bulk track is shown only to justify rejecting it.

### Track A — naive bulk MIME sweep (NOT recommended)

| Stage | Count | Note |
|---|---|---|
| Raw SVGs | 5,228,764 | all of Commons SVG |
| License-ok (commercial) | 5,228,764 | 100%; Commons is free-only |
| License-clean (drop SA, keep PD/CC0/BY) | ~2.6M-3.9M | est. 50-75% |
| Structural gate pass (born-vector, <200KB, not raster-hybrid, parseable) | ~0.8M-1.4M | est.; Commons skews large/complex, so reject rate far exceeds the icon corpus (huge-file, raster-embed, external-ref, off-canvas) |
| Autotrace-clean | **unknown, currently unfilterable** | no working detector; complex art collides with the trace signal |
| Net usable | ~0.9M nominal, but **mostly redundant icons/flags** | volume, not gaps |

### Track B — targeted category harvest (recommended, gated on detector)

| Stage | Count | Note |
|---|---|---|
| Target categories (heraldry + labelled maps + diagrams, PD/CC0/BY only) | ~200k-400k | PetScan-enumerated; the gap-relevant buckets |
| License-clean | ~120k-300k | drop SA |
| Structural gate pass | ~80k-200k | complex art trips warns; live-text kept and pre-flattened |
| Autotrace-clean (requires detector) | ~40k-120k | heraldry/maps are mostly hand-built, so pass rate here is *higher* than the bulk track once the detector can tell hand-built from traced |
| **Net usable** | **~0.3M-0.8M across a full sweep of gap buckets; ~40k-120k for a first heraldry+maps harvest** | fills fineline/complexity + typography-in-context gaps that no other source covers |

The Track B net-usable range is the headline number and is explicitly conditional on the autotrace
detector. Realistic *first cut* is the 40k-120k figure (heraldry + labelled maps); the wider
0.3M-0.8M is the ceiling if all gap-relevant subcats are eventually swept.

---

## 8. PLAN (if pursued, Phase 2)

**Prerequisite (hard gate):** build and validate the structural autotrace detector on Commons
complex art. Success criterion: on a hand-labelled 300-file Commons set (heraldry, detailed maps,
traced portraits, traced logos), high precision on "traced" with acceptable recall, and near-zero
false-positive on hand-built heraldry. Tune for precision (drop obvious traces, accept false
negatives) per fuel.md §2.5. Effort: ~3-5 engineer-days (harder than the icon-corpus heuristic
because heraldry/traced-portrait is the adversarial collision case).

**Filter cascade order (cheapest / most-eliminating first, all pre-byte where possible):**

1. **Offline, on XML metadata dumps:** license template filter → keep PD/CC0/(optionally BY), drop
   SA/GFDL/GPL. Eliminates the largest chunk for free, no crawling.
2. **Offline, category filter (PetScan):** restrict to gap buckets (coats of arms, labelled maps,
   diagrams, life-science). Drops icons/flags/logos/clipart redundancy.
3. **Fetch bytes** for the surviving list via `imageinfo` URLs, polite (~1 req/s, `maxlag`,
   contact UA) or via Wikimedia Enterprise if volume justifies.
4. **`gate2.py` structural pass:** reject raster-hybrid / huge / external-ref / off-canvas /
   unparseable; keep live-text as WARN.
5. **Autotrace detector** (the prerequisite): drop traced; this is where heraldry/map value survives
   and traced portraits/logos die.
6. **Pre-flatten live text to paths** (fontTools) for the surviving text-bearing files, to remove
   missing-font render risk and preserve the typography supervision.
7. **Dedup** against the existing corpus and against Openclipart/freesvg (Commons re-hosts and is
   re-hosted; fuel.md §2.3 dedup trap).

**Effort:** detector 3-5 days; harvest + filter pipeline 3-5 days; first heraldry+maps harvest crawl
~2-4 days wall-clock at polite rates. Call it **~2 engineer-weeks** to a first validated
gap-slice, most of it the detector.

**Storage / compute:** full SVG pool is 1.19 TB (do not download it). A targeted 150k-file harvest
is ~20-60 GB of SVG. Rendering + wreck + train cost is the same per-image as existing sources; the
marginal experiment (mint ~40k heraldry+map pairs, add to a 927k run, measure per-source deltaE on a
held-out heraldry/text slice) is ~1 GPU-day, matching fuel.md's other confirming experiments.

---

## 9. RISKS

1. **Autotrace contamination is the dominant risk and is currently unmitigated.** Minting Commons
   before the detector exists actively harms the tracer target. This is a hard blocker, not a
   caveat.
2. **The icon-tuned `gate2` trace heuristic mis-scores Commons complex art** (complexity signal ==
   contamination signal), so it cannot be relied on as the detector.
3. **Access friction:** no dumps, MIME API disabled, deep category diffusion. A bulk pull is a
   multi-week polite crawl or a paid Enterprise feed; underestimating this sinks the schedule.
4. **Share-alike licensing at scale** is an open legal question for a commercial model, same shape as
   OFL Q1.25. Mitigated engineering-side by filtering to PD/CC0, at some yield cost.
5. **Redundancy:** ~80% of the 5.2M is icons/flags/clipart/logos already covered; an untargeted
   harvest buys volume the model is not starved for and dilutes the gap sources.
6. **Live-text render fidelity:** missing referenced fonts corrupt ground truth unless text is
   pre-flattened; an extra pipeline step and a yield loss on the exact files we most want.
7. **Trademark exposure in the logos bucket** (as with Font Awesome Brands in fuel.md); avoid logos.
8. **Provenance/attribution bookkeeping** for any BY-licensed retained files, required at
   redistribution.

---

## Bottom line

Commons is the largest SVG pool anywhere (5,228,764 files) and the unique supplier for the
heraldry/complex-art and typography-in-context gaps, but it is a Phase-2 source: its value is gated
on an autotrace detector we have not built, its most valuable buckets are its most trace-polluted,
and it must be harvested narrowly by category rather than swept in bulk. Recommended action is to
build the detector first, then run a targeted ~40k-120k heraldry+labelled-map harvest (PD/CC0/BY),
not to scrape the 5.2M.
