# Source feasibility: SVGRepo (svgrepo.com)

Date: 2026-07-21. Single-source feasibility study for Stage-1 cleanup-UNet fuel.
Compares against the existing corpus: 1.165M unique SVGs already mined from
StarVector `svg-stack` (content-SHA deduplicated; svg-stack itself scraped much
of the open icon web off GitHub). Prior census: `../2026-07-20-deep-research/fuel.md`
(SVGRepo appears there as row 45, D-tier).

## VERDICT

**Not worth prioritising. Feasible but low-yield.** SVGRepo is an aggregator/search
engine over the same curated open-source icon packs that svg-stack already scraped
from GitHub, re-hosted and SVGO-optimised. It is the icon/symbol domain the corpus
is already saturated in, so it adds **volume, not gap coverage** — and the measured
gaps are typography, fineline strokes, and boundary-label accuracy, none of which
more clipart icons touch. Realistic net-new *usable* after license filtering and
dedup is on the order of **low tens of thousands of same-domain icons** (see yield
math below), for effectively zero compute. If icon volume is wanted anyway, take
the pre-scraped HuggingFace snapshot, not the live site. Otherwise skip in favour
of the gap-filling sources (open-font typography minting, StarVector `svg-diagrams`,
Openclipart CC0 for the illustration gap).

## What SVGRepo actually is

- A **search/aggregator site**, not an original vector house. Content is organised
  into named **data packs** (the HF snapshot exposes a `data pack` field, e.g.
  Carbon Design Pictograms, Feather-style sets). These packs are curated
  open-source icon sets — the exact material svg-stack pulled from GitHub.
- The site markets ~500k vectors today. The best available bulk snapshot,
  `nyuuzyou/svgrepo` on HuggingFace, captured **~218,000** icons across 12 license
  splits — so the site has grown since that scrape, but the additional material is
  more of the same pack-sourced icons.
- Icons are re-processed on ingest (normalised / SVGO-optimised), so a SVGRepo copy
  of an icon is **byte-different** from the GitHub original even when visually
  identical. This matters for dedup (below).
- **Net-new vs re-hosted:** predominantly re-hosted. The uploads/net-new fraction
  is small relative to the pack-sourced bulk, and the packs overlap svg-stack.

## License labelling reliability

Per the `nyuuzyou/svgrepo` split (~218k total):

| license | count | usable for derivative training? |
|---|---|---|
| CC Attribution (CC-BY) | 65.5k | yes, with attribution bookkeeping |
| MIT | 62.3k | yes |
| Public Domain | 42.3k | yes |
| Apache | 17.8k | yes |
| GPL | 11.0k | **no** — copyleft, avoid for model fuel |
| CC0 | 11.4k | yes |
| Logo | 5.51k | **no** — trademark/brand risk |
| OFL | 471 | **no** — font license, restricts standalone glyph use |
| MPL | 288 | marginal — treat as exclude |
| BSD | 267 | yes |
| CC NC Attribution | 135 | **no** — non-commercial |
| MLP | 678 | unknown label — exclude |

- Labels are assigned **per data-pack**, heuristically, not verified per file. Treat
  as **unreliable at the file level** and subset conservatively.
- Cleanest-provenance of the four nyuuzyou clipart corpora (svgrepo / svgfind /
  clker / openclipart), but the most **license-fragmented** — 12 buckets including
  copyleft (GPL) and NC that must be dropped.
- Conservative "safe" subset (CC0 + PD + MIT + Apache + BSD) ≈ **134k**. Adding
  CC-BY (attribution tracked) reaches ≈ **200k**. This is the pre-dedup ceiling.

## Bulk access

- **No public API.** No official bulk/collection download.
- **Live site is behind a Vercel anti-bot "Security Checkpoint."** Confirmed: even
  `https://www.svgrepo.com/robots.txt` returns the JS challenge page (interstitial
  `Vercel Security Checkpoint`), not a robots file. Direct crawling is actively
  fought; WebFetch returned HTTP 429. **Do not build a scraper against the live
  site** — high effort, fragile, adversarial.
- **The realistic access path is the pre-scraped snapshot:** `nyuuzyou/svgrepo`
  (HuggingFace, Parquet, auto-converted from JSON). Fields include id, title,
  data pack, tags (1–12), license type, license owner, download URL, and raw SVG
  content inline. Same org and access pattern as the svg-stack shards the pipeline
  already pulls (`scripts/corpus_remote.py`). Zero scraping required.

## Net-new yield after dedup

The pipeline dedups by **content SHA** (`scripts/label_split.py` writes
`clean/<sha[:2]>/<sha>.svg`). Two dedup layers behave very differently here:

1. **Exact SHA vs the 1.165M svg-stack shas.** SVGRepo re-optimised its copies, so
   byte content diverges from the GitHub originals in svg-stack. SHA collision rate
   will therefore be **low** — meaning naive SHA dedup keeps most files and
   **overstates** net-new.
2. **True visual/semantic duplication.** The packs are the same icon sets already in
   svg-stack. Real novelty is small; the SHA-survivors are mostly re-encodings of
   icons the model has already seen.

Honest yield chain (approximate):
- 218k snapshot -> license subset (safe + CC-BY, attribution tracked) ≈ **200k**.
- Minus true overlap with svg-stack's icon packs (large, but SHA-invisible) — the
  genuinely novel, non-redundant icons are plausibly **~20k–60k**.
- All of it is the **already-saturated icon/symbol domain**, so even the novel slice
  relieves nothing the corpus is short on. Value ≈ volume-only, near-zero gap value.

## Expected gate survival

- **High structural survival** is expected. `gate2` is explicitly "tuned for compact
  stock icon/logo SVGs" — SVGRepo's exact shape — so unlike the NAS PDF corpus
  (which mostly tripped the 200KB `huge` reject) most SVGRepo files should land
  clean/warn.
- **Low autotrace-contamination risk.** These are curated born-vector packs, not
  raster-traced output (contrast `clker`, which is autotrace-from-raster and high
  risk). Little `path>8 / cmd>32` trace signature expected.
- **One check before minting:** verify the file-size distribution against gate2's
  200KB `huge` cap. Detailed multi-colour / pictogram packs can be larger than a
  typical single-path icon; a minority may trip `huge`. Sample and confirm.

## PLAN (if pursued anyway)

Low effort, ~0 compute — sensible only as opportunistic icon top-up, not a priority.

1. `pip`-pull `nyuuzyou/svgrepo` Parquet shards (same pattern as
   `scripts/corpus_remote.py`). No live-site scraping.
2. Filter `license type` to {CC0, PublicDomain, MIT, Apache, BSD} (+ CC-BY only if
   attribution bookkeeping is acceptable). Drop GPL, Logo, OFL, NC, MPL, MLP.
3. Content-SHA dedup within the set, then against the 1.165M corpus shas. Expect
   SHA dedup to under-remove (re-optimised bytes); optionally add a lightweight
   visual/near-dup pass (normalised-path or rendered-pHash) to catch same-icon
   re-encodings, or simply accept the redundancy as cheap augmentation.
4. Run `scripts/gate2.py` on the survivors; keep clean+warn. Sample-audit the
   `huge` reject rate first.
5. Fold the clean shard into the corpus and re-fit the intercept (mix changes).

Effort: ~half a day, mostly a filter/dedup script. Cost: ~$0 (no scraping, no GPU
beyond the existing training). Yield: a modest same-domain icon shard.

## RISKS

- **Low marginal value.** Same domain as the corpus's saturation zone; does not move
  the typography / fineline / label gaps that the deliverable metric (mean deltaE)
  actually rewards. Opportunity cost vs gap-filling sources is the real risk.
- **License fragmentation + unreliable pack-level labels.** Must subset
  conservatively; CC-BY's attribution obligation is bookkeeping overhead; GPL/Logo/
  NC contamination if the label filter is trusted blindly.
- **Dedup blind spot.** SHA dedup misses SVGRepo's re-optimised duplicates of icons
  already in svg-stack — without a visual near-dup pass, the corpus silently
  double-counts the same icons.
- **Live-site scraping is a trap.** Vercel anti-bot checkpoint makes any home-grown
  crawler fragile and adversarial; rely on the HF snapshot or not at all. The HF
  snapshot also lags the live catalogue.
- **Stale/partial snapshot.** ~218k vs the site's claimed ~500k means the snapshot
  is old; refreshing it means re-scraping through the checkpoint — not worth it.

## Bottom line

Feasible and cheap via the HuggingFace snapshot, but it is more of what the corpus
already has too much of. Net-new usable is low tens of thousands of same-domain
icons with near-zero gap value. Take it only as a zero-cost opportunistic top-up
after the gap-filling sources are exhausted; do not scrape the live site.
