# Source feasibility: born-vector figures from arXiv at scale

Date: 2026-07-21. Feasibility study for one candidate training-data source for the
Stage-1 cleanup UNet and Stage-2 tracer. Scope: extract born-vector figures
(matplotlib / pgfplots / TikZ plots, technical diagrams, text-dense scientific
figures) from arXiv at bulk scale, convert to gate-clean SVG, and estimate net
usable yield. Factual register; estimates flagged as such.

Target gaps this source attacks (from `fuel.md`): thin/hairline strokes,
text-heavy figures, plots, and technical diagrams. These are exactly what
scientific figures are made of, so gap-relevance of the surviving figures is
high — this is gap-fill, not volume-only fuel.

---

## Verdict

**FEASIBLE, with license as the dominant gate.**

The technical pipeline is sound and cheaper than the NAS PDF route, because it
converts *single born-vector figure files* (not whole print pages), which
sidesteps both cliffs that sank the NAS SVG route: PyMuPDF page-scaffolding
bloat and raster-blind `idmap`. gate2's existing `raster`/`raster-b64` reject
doubles as a free vector-vs-raster classifier.

Net usable, born-vector, gate-clean-or-warn figures:

- **Commercially clean (CC-BY / CC-BY-SA / CC0 only): ~80k–150k** from a ~1M-paper
  plot-heavy category slice; **~200k–350k** if the full ~2.7M-paper corpus is
  processed. License is the largest single cull (~85–90% of the technical yield
  is removed here).
- **If research / non-commercial use is acceptable** (add CC-BY-NC-* and treat
  arXiv redistribution as a deferred legal question): **~600k–1M+** from the full
  corpus. This is a legal decision, not a technical one.

Ranking: this is the highest-value *harvested* born-vector source for the plot /
technical-diagram / thin-line gaps (minted typography from open fonts remains the
higher-value gap-fill overall, per `fuel.md`, because it carries zero license and
zero autotrace risk). arXiv figures are the best available real-world plot and
scientific-diagram distribution; nothing synthetic reproduces genuine
pgfplots/matplotlib layout diversity as cheaply.

---

## Access route

arXiv publishes the full corpus as **AWS S3 requester-pays** buckets
([info.arxiv.org/help/bulk_data_s3](https://info.arxiv.org/help/bulk_data_s3.html)),
confirmed current:

- Bucket `arxiv`, requester-pays. Two sets: `arXiv_pdf_[yymm]_[seq].tar` and
  `arXiv_src_[yymm]_[seq].tar`, each ~500 MB, monthly-grouped, with XML manifests
  (`arXiv_src_manifest.xml`) carrying checksums, file counts and ID ranges.
- Sizes: PDF ~2.7 TB, **source ~2.9 TB** (Mar 2023 figure; ~9.2 TB combined as of
  Apr 2025; growing ~100 GB/month). Mid-2026 source set is on the order of
  ~3.5–4 TB across ~2.7M papers.
- Cost model: requester pays Amazon for retrieval. **In-region is near-free** —
  run extraction on EC2 in `us-east-1` (same region as the bucket) and S3→EC2
  transfer is $0, leaving only GET-request cost (~$0.0004 / 1000 requests, a few
  dollars for ~6–7k tar objects). **Egress to the public internet is ~$0.09/GB**:
  ~$260 for the whole source set, ~$45 for a 500 GB category/date slice. The
  smart play is process-in-region, download-nothing.

**Kaggle mirror is metadata-only.** The Cornell `arXiv` Kaggle dataset is a
single JSON of per-paper metadata (id, authors, title, abstract, categories,
`license`, versions, DOI, dates), ~4 GB, all 2.4M+ papers, refreshed regularly.
It contains **no PDFs, source, or figures.** Its value here is the `license`
field — join it against the harvest to apply the license filter without parsing
every tarball. So the route is: **Kaggle metadata for the license/category join +
S3 source tarballs for the figure bytes.** There is no bulk figure mirror; S3
source is the only route to the figure files.

Recommended slice for a first meaningful harvest: plot-heavy categories
(`cs.*`, `stat.*`, `eess.*`, `physics`, `astro-ph`, `cond-mat`, `hep-*`, `math`,
`q-fin`, `econ`), which is where matplotlib/pgfplots/TikZ density is highest.
That is ~1M+ papers.

---

## Extraction route: LaTeX source, not page-parsing

Two candidate routes; **the LaTeX-source route wins decisively** for a
born-vector harvest.

**Chosen: LaTeX source tarball → extract included figure files directly.**
arXiv source tarballs contain the author's original figure files —
`figure.pdf`, `.eps`, `.ps`, plus `.png`/`.jpg` — alongside the `.tex`. Grabbing
the vector ones (`.pdf`/`.eps`/`.ps`) gives the *original born-vector artwork*,
not a re-rasterized page crop. Rationale:

- Cleanest possible input: no page-region cropping, no risk of pulling in
  surrounding body text or neighbouring figures, no compiled-PDF scaffolding.
- Trivially cheap: untar + glob image extensions (optionally cross-referenced to
  `\includegraphics{}` targets). No PDF compilation, no ML region detector.
- Matches `fuel.md` item #27's guidance ("LaTeX source route ... cleaner than
  page-parsing").

Its one real cost: **inline TikZ/PGF** figures are LaTeX code, not standalone
files, so they need compilation (extract `tikzpicture`, wrap in `standalone`,
`pdflatex`) to render. Defer that — many TikZ/pgfplots users also commit a
compiled `.pdf`, and pgfplots exports are frequently shipped as `.pdf` anyway, so
the bulk of technical-diagram material is reachable without a compile step. Add a
TikZ-compile pass later if the diagram slice needs boosting.

**Rejected for the primary pass: compiled-PDF page-region route (pdffigures2).**
pdffigures2 (AllenAI, Apache-2.0) is excellent at *localising* figures and
associating captions, but it rasterizes crops by default; keeping vector means
vector-cropping the PDF region (mutool/gs), which is fiddly and can slice through
overlapping text. It is JVM/Scala, heavier to run at scale, and designed for
raster figure harvesting. Reserve it for the ~13% of submissions that are
**PDF-only (no source)**, where the source route yields nothing.

---

## Vector-vs-raster fraction

Not every figure file is born-vector; photos, microscopy, and heatmaps ship as
PNG/JPG (or as a `.pdf` that wraps a single embedded raster).

- `fuel.md` cites **~11%** vector among sampled figures — but that datapoint is
  *page-parsing across all fields*, the pessimistic case.
- The **source route on plot-heavy fields is much higher**: estimated
  **30–50%** of figure *files* in `cs/stat/physics/math` source tarballs are
  vector `.pdf`/`.eps`. All-field average via the source route is ~15–25%.
- Critical free filter: a `.pdf` "figure" that is really one embedded raster
  survives file-extension selection but is caught downstream — gate2 rejects it
  as `raster`/`raster-b64`. **The gate is the vector detector**; no separate
  born-vector classifier is needed. This is the single biggest reason this route
  is low-effort.

This fraction is the largest *technical* uncertainty and is cheap to measure
directly (see Risks / de-risking).

---

## Conversion funnel PDF/EPS → SVG → gate-clean

The NAS study found PyMuPDF `get_svg_image` output routinely blew past gate2's
200 KB `huge` cap because of embedded base64 raster and repeated clip-path
scaffolding in *full print pages*. **Single born-vector figures do not have that
problem** — a matplotlib line plot or pgfplots chart is paths + text, no embedded
raster, no page scaffolding, and converts to a compact SVG (typically ~5–80 KB,
comfortably under 200 KB). The bloat cliff was a property of full print pages,
not of vector figures.

Tooling (all mature, zero-cost):

- **dvisvgm** (`--pdf`, `--eps`) — best choice. Native font-to-path (kills the
  `live-text` warn), `--optimize`, `--exact-bbox`, tight path output. Handles
  both PDF and EPS figure files.
- **pdftocairo -svg** (poppler) and **mutool convert** — solid fallbacks; keep
  text as text unless post-processed to paths.
- EPS-only inputs: dvisvgm `--eps` directly, or `ps2pdf` then convert.

Known failure modes carried over and their mitigations:

- **`huge` (>200 KB) on dense plots** — large scatter (10k+ markers) or
  vectorized contour/heatmap PDFs blow up. Mitigate: dvisvgm `--optimize`; a
  point/command pre-count filter; and accept that a slice of dense plots is lost
  (they are also the least clean training targets). A modest gate2 `huge`-cap
  bump for this source (e.g. 300–400 KB) recovers some, but keep it source-scoped.
- **`raster` / `raster-b64` reject** — desirable here; it culls raster-wrapped
  "vector" files for free.
- **`live-text` warn** — avoided by converting text-as-path with dvisvgm; without
  it, these land in warn (still usable per gate2's own definition), not clean.
- **`trace-suspect` / `node-heavy`** — matplotlib line paths are genuine curves
  with low tiny-chord fraction and rarely trip trace-suspect; dense marker
  scatter can. Low incidence.

Funnel for a **~1M-paper plot-heavy slice** (estimates; ranges reflect the
vector-fraction uncertainty):

| Stage | Survival | Count | Note |
|-------|----------|-------|------|
| Papers in slice | — | 1,000,000 | plot-heavy categories |
| Source available (not PDF-only) | ~87% | 870,000 | ~13% PDF-only excluded (reserve for pdffigures2) |
| Figure files extracted | ~5/paper | ~4,350,000 | globbed `.pdf/.eps/.ps` + raster, then vector-only kept |
| Vector figure files (`.pdf/.eps/.ps`) | ~40% | ~1,740,000 | plot-heavy field rate; 30–50% band |
| Convert to SVG (dvisvgm/pdftocairo) | ~90% | ~1,570,000 | timeouts/parse failures on pathological files |
| gate2 not-reject (clean **or** warn) | ~55–65% | ~940,000 | raster-wrapped culled, huge dense plots, off-canvas, parse |
| Dedup + gap-relevance (drop template/author repeats, simple-diagram overlap) | ~75% | ~700,000 | near-dup hash + drop icon-overlapping simple diagrams |
| **License filter — commercial (CC-BY/SA/CC0 ~11%)** | ~11% | **~80k–150k** | dominant cull; join Kaggle `license` field |
| *(alt: research/NC-inclusive use)* | *~85%* | *~600k* | *legal call, not technical* |

Whole-corpus (~2.7M papers) processing scales the commercial-clean line to
~**200k–350k**; the strictly-`clean`-tier-only (no warn) count is roughly half of
the clean-or-warn figures.

---

## Effort, storage, compute

**Engineering: ~10–14 days** for a production harvest.

- S3 requester-pays pull + manifest-driven untar harness (boto3, in-region EC2): 1–2 d
- Figure-file extraction (extension glob + optional `\includegraphics` cross-ref): 1 d
- Conversion harness (dvisvgm/pdftocairo, parallel, per-file timeouts for pathological inputs): 2–3 d
- gate2 integration + source-scoped size-cap / `--optimize` tuning pass: 1–2 d
- License + category join against Kaggle metadata: 1 d
- Near-duplicate dedup (structural/perceptual hash) + gap-relevance filter: 1–2 d
- End-to-end plumbing, per-source validation slice, QC: 2–3 d

Proof-of-concept (measure vector fraction + gate survival on ~2–5k papers before
committing): **~2–3 days.**

**Compute:** extraction + conversion is CPU-bound and embarrassingly parallel.
One large EC2 instance or a short spot fleet in `us-east-1` processes the ~1M
slice in a day or two. dvisvgm can be slow on complex files — mandatory per-file
timeouts. **Cost: ~$50–200 EC2 spot + a few dollars of S3 requests if in-region;
add ~$45 (500 GB slice) to ~$260 (full source) only if downloading out of AWS.**

**Storage:** raw source slice a few hundred GB to ~3–4 TB (full); extracted
vector figure files a few hundred GB; final gate-clean SVG corpus is small (tens
of GB). Do not retain raw tarballs after extraction.

---

## Risks

1. **License is the dominant cull and a genuine legal question.** The arXiv
   default (Perpetual Non-Exclusive License 1.0) does *not* grant redistribution
   or reuse; CC-BY-NC-* excludes commercial use; only CC-BY / CC-BY-SA / CC0
   (~11% of the corpus, estimate) are commercially clean. ~85–90% of the technical
   yield is legally gated for a commercially-sold model. This is a lawyer
   decision, mirroring the OFL Q1.25 open question in `fuel.md`. Note also the
   figure-level nuance: the paper license governs the paper, and third-party
   figures reused within a paper may carry their own terms — another legal flag.

2. **Vector fraction is the biggest technical unknown** — 11% (all-field
   page-parse, pessimistic) vs 30–50% (plot-heavy source-route, estimated).
   The whole yield scales with it. De-risk first: sample 500 figure files from
   the target categories, run the convert→gate2 funnel, measure the born-vector
   and gate-survival rates directly (~2–3 days, no commitment).

3. **Dense vector plots trip `huge`/`node-heavy`.** Scatter/contour/vectorized-
   heatmap PDFs are large; `--optimize` and a source-scoped cap recover some, but
   a slice is lost. These are also lower-value training targets, so acceptable.

4. **Inline TikZ/PGF needs compilation** — the primary pass skips it, losing some
   of the cleanest technical-diagram material. Recoverable with a later
   `standalone`+`pdflatex` compile pass if the diagram slice is thin.

5. **`live-text` / warn-tier share.** Without text-to-path conversion, figures
   land in warn not clean. dvisvgm's font-to-path fixes this; budget the tuning.

6. **Raster-wrapped "vector" PDFs** are common (a `.pdf` that is one embedded
   image). Not fatal — gate2's `raster` reject culls them for free — but they
   depress the vector-fraction survival, already priced into the funnel.

7. **Domain overlap / genuine gap-fill.** Some arXiv figures are simple diagrams
   that overlap the existing icon corpus. The dedup + gap-relevance filter must
   keep the plots / text-dense / hairline figures (the gaps) and drop the
   redundant simple shapes, so this stays gap-fill rather than volume-only.

8. **Autotrace pollution is LOW** (born-vector originals, not raster-traced), the
   opposite of the clipart corpora — a genuine advantage of this source.

---

## Recommendation

Run the ~2–3 day proof-of-concept on a 2–5k-paper plot-heavy sample first: it
resolves risk #2 (vector fraction) and risk #3 (gate survival) with a real
number before any large pull. If the measured commercial-clean, gate-clean yield
clears ~50k figures for the full corpus, build the ~2-week harvest. Regardless,
resolve the license question (risk #1) with counsel before minting anything from
this source into a commercially-sold model; if only research use is intended, the
pool is ~5–8× larger and the pipeline is unchanged.
