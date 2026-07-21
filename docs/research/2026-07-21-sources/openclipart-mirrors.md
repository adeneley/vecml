# Source Feasibility: Openclipart + Public-Domain Mirror Ecosystem

Single-source feasibility study for the vecml cleanup-UNet training corpus. Scope: openclipart.org and its public-domain mirror/derivative ecosystem (freesvg.org, publicdomainvectors.org, svgsilh.com). Question the study answers: how much *net-new, gate-surviving, license-clean* SVG can this ecosystem realistically contribute beyond the 1.165M icon uniques already in training, and at what effort.

Register: factual. All counts are dated snapshots; treat as of 2026-07-21.

---

## Verdict

**FEASIBLE — but the "ecosystem" collapses to one source plus a lower-value tail.**

- **Net-new usable estimate: ~140k–200k SVGs (best estimate ~160k).**
- **~100k–130k of that is Openclipart alone**, high confidence, near-zero effort/cost, acquired from an existing HuggingFace mirror with no scraping required.
- The mirrors add only a modest **~40k–70k net-new tail** (freesvg/publicdomainvectors-native uploads) at disproportionate effort and higher autotrace pollution. **svgsilh.com contributes ≈ 0 net-new** (100% Openclipart-derived, degraded to monochrome silhouettes). **Direct publicdomainvectors scraping contributes ≈ 0 net-new SVG beyond freesvg** (shared administrators; freesvg is that pipeline's SVG republication front).

Domain fit: this is full-colour illustration/clipart, so it attacks the **colour-rich illustration gap** (net-new vs the icon-dominated existing corpus) and carries **tag-minable lettering/word-art** for the typography gap. It does **not** attack the fineline-stroke gap (clipart is fill-dominated, near-zero centerline strokes).

Recommendation: ingest Openclipart now via the HF mirror; treat freesvg as an optional dedup-gated top-up only if the illustration gap still needs volume after Openclipart; skip svgsilh and skip direct publicdomainvectors scraping.

---

## Evidence: the ecosystem is one superset and three dependents

| Site | Count (snapshot) | License | Provenance | Net-new vs Openclipart |
|---|---|---|---|---|
| **openclipart.org** | 185,666 (homepage) / 193,352 (Wikipedia, Sep-2025) / **178,604 in HF mirror** | CC0 1.0 (100% public domain) | Canonical community upload site, since 2004; gifted to Fabricatorz Foundation 2019 | — (the base) |
| **freesvg.org** | 172,266 total | CC0 | **Same administrators as publicdomainvectors** (Vedran & Boris). Explicitly re-hosts "55k clipart images ... that were part of openclipart.org" | ~55k are Openclipart re-hosts; remaining ~117k are freesvg/pdv-native |
| **publicdomainvectors.org** | Unconfirmed (tens of thousands to low 100k); site is Cloudflare-gated and 403s automated fetch | Claimed PD/CC0 (site ToS) | Same admins as freesvg; freesvg is described as "the new home" for this content. Serves EPS + SVG, stock-style vectors | ≈ 0 net-new *SVG* beyond freesvg |
| **svgsilh.com** | Low tens of thousands | CC0 | Openclipart images converted to monochrome silhouettes with a recolor tool | ≈ 0 (derivative + degraded) |

Load-bearing confirmations gathered this session:
- Openclipart robots.txt is **fully permissive**: only a `Sitemap:` directive, no `Disallow`, no `Crawl-delay`. Scraping is tolerated, but unnecessary (see below).
- Openclipart API v2 is in **beta only**; the on-site developer page documents no bulk-dump/torrent endpoint. Contact is support@openclipart.org.
- The canonical acquisition path already exists: **`nyuuzyou/openclipart` on HuggingFace — 178,604 entries, 22.3 GB, CC0**, each row carrying `svg_content` (raw XML, minified with tdewolff/minify), plus `title`, `description`, uploader, timestamp, and a `tags` string array. Autotrace provenance is present in tags (`filter autotrace`, `vectorized`).
- freesvg's own about-us page confirms shared administration with publicdomainvectors and the 55k Openclipart re-host batch.
- freesvg robots.txt **disallows `/download/*`** (and `/converts/*`, `/tag/*?page*`) — the actual SVG download path is off-limits to crawlers.
- publicdomainvectors.org returns **HTTP 403 to automated fetch** (Cloudflare bot protection) — it actively resists scraping; its robots.txt only disallows `/*/search/*` and `/*/long/*`.

---

## Plan (if feasible)

### Acquisition, per site

1. **Openclipart — primary. Effort ~0.5 day, cost ~$0.**
   Pull `nyuuzyou/openclipart` from HuggingFace (22.3 GB). No scraping, no API. Raw SVG is in the `svg_content` column; tags and metadata travel with it. This is the entire recommended intake for phase 1.

2. **freesvg.org — optional top-up. Effort ~2–4 days, cost ~$0, lower confidence.**
   No prepackaged mirror confirmed. The homepage catalogue is crawlable but `/download/*` is robots-disallowed, so respect it: use the sitemap + per-item pages, throttle hard, and only fetch items that survive dedup (below). Realistic reachable net-new pool ≈ 172,266 − 55,000 Openclipart re-hosts ≈ **~117k gross freesvg/pdv-native**, before dedup and gate.

3. **publicdomainvectors.org — skip.** Shared-admin duplicate of the freesvg SVG catalogue, Cloudflare-gated, EPS-heavy, and higher autotrace/stock pollution. No net-new SVG worth the scrape.

4. **svgsilh.com — skip.** 100% Openclipart-derived and degraded to single-fill silhouettes: no colour, no fineline, no typography value.

### Dedup strategy (Openclipart as the base superset)

Ingest Openclipart first, hash everything, then admit a mirror file only if all three novelty checks pass:
1. **Exact-content hash** — SHA256 over a normalized SVG (strip comments/editor namespaces/whitespace, canonicalize path number precision). Catches verbatim and re-minified re-hosts.
2. **Render-perceptual hash** — pHash/dHash of a fixed-size raster render. Catches recolors, silhouette conversions (svgsilh), and re-exports that survive content-hash changes.
3. **Title/filename slug match** — cheap prefilter; Openclipart titles propagate through freesvg re-hosts.

Expected dedup kill on freesvg: the 55k Openclipart batch drops immediately, plus additional near-dups, leaving ~40k–70k genuinely novel freesvg items.

### Pollution filter (before the gate)

Autotrace is the dominant quality risk in this ecosystem. Two-stage filter:
- **Tag filter (Openclipart only, free):** drop items tagged `filter autotrace`, `vectorized`, `traced`, `silhouette`, `photo`, and bulk-import markers (`upload2openclipart`). Estimated ~15–25% flagged.
- **Structural heuristics (all sources):** the tells already catalogued in `fuel.md` §2.5 — grep the literal potrace comment (`Created by potrace` / `Created with Potrace`, near-perfect precision), 100% path-Bezier with zero `<circle>`/`<rect>`/`<line>`/arc primitives, closed even-odd filled paths with zero strokes, path/command explosion, and palette-quantization signatures. Mirrors carry no reliable trace tags, so they lean entirely on structure.

### Expected gate survival and net-new yield

Run survivors through `scripts/gate2.py` (reject / warn / clean tiers).

- **Openclipart:** 178,604 base × ~0.80 (tag filter) × ~0.75 (gate-clean after reject/warn) ≈ **~100k–130k usable**. High confidence.
- **freesvg net-new:** ~40k–70k post-dedup gross × higher pollution and gate loss ≈ **~20k–45k usable**. Medium-low confidence.
- **publicdomainvectors / svgsilh:** ≈ 0 net-new.
- **Ecosystem total: ~140k–200k net-new usable**, ~100k–130k of it high-confidence Openclipart.

Total cost: **~$0** (HF bandwidth only; scraping optional). Total effort: **~0.5 day** for the Openclipart path that captures most of the value; **+2–4 days** for the freesvg tail at modest, lower-confidence yield.

---

## Risks

- **License (low for Openclipart, medium for mirrors).** Openclipart is uniformly CC0 1.0 — clean for any training/redistribution use. Mirror CC0 claims are site-ToS assertions over re-hosted and re-uploaded content; provenance per file is weaker. Retain per-file source and tags so a later license question is traceable, and prefer the Openclipart-native copy of any duplicate.
- **Scrape etiquette (medium).** Openclipart tolerates crawling but the HF mirror makes it moot. freesvg **robots-disallows the `/download/*` path** — honour it; do not bulk-hammer download URLs. publicdomainvectors runs **Cloudflare bot protection (403 on automated fetch)** — treat as an explicit do-not-scrape signal; skip it.
- **Autotrace pollution (medium-high, but manageable).** This ecosystem is a known autotrace reservoir. On Openclipart it is tag-detectable and therefore filterable rather than fatal. On the mirrors there are no reliable trace tags, so pollution rides entirely on the structural heuristics — the freesvg tail is the pollution-exposed portion of the estimate and the reason its yield confidence is lower. Feeding un-filtered traces to the cleanup UNet teaches it the wrong reconstruction target, so the filter is not optional for the mirror tail.
- **Gap-coverage risk (design, not license).** All four sites are fill-dominated illustration/clipart. They fill the colour-rich-illustration gap and offer some tag-minable lettering, but contribute near-nothing to the **fineline-stroke gap** (clipart has almost no centerline strokes). Do not treat this ecosystem as the fineline answer.
- **Duplication trap (handled by the plan).** Without cross-source dedup, freesvg re-inflates the corpus with ~55k Openclipart copies plus near-dups, and svgsilh would inject degraded recolors of images already present. The render-pHash pass is the hard requirement that prevents this.

---

## Sources

- openclipart.org homepage, /developers, /robots.txt (accessed 2026-07-21)
- freesvg.org homepage, /pages/about-us, /robots.txt (accessed 2026-07-21)
- publicdomainvectors.org /robots.txt (homepage 403 under Cloudflare) (accessed 2026-07-21)
- HuggingFace dataset `nyuuzyou/openclipart` — 178,604 / 22.3 GB / CC0, `svg_content` + tags (accessed 2026-07-21)
- Wikipedia, "Openclipart" — history, CC0, Sep-2025 count 193,352 (accessed 2026-07-21)
- Prior census: `docs/research/2026-07-20-deep-research/fuel.md` (Openclipart entry, autotrace heuristics §2.5, freesvg/publicdomainvectors dedup note)
