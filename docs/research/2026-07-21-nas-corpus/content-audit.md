# Print-ready PDF corpus: content-character audit

Date: 2026-07-21

## Purpose

Size a print shop's print-ready PDF corpus as candidate SVG training data.
The unit that matters for minting is the **vector-dominant page** (drawable
vector art with minimal raster). This audit measures the page-class mix, text
handling, colour-space signals, and content archetypes, then extrapolates to
the full corpus.

## Method

- Frame: 16,048 files suffixed `-PRINT`; 14,776 are PDFs.
- Sample: stratified random draw of 157 PDFs (seed 42), allocated
  proportionally across top-level shares so every letter/segment is represented.
- Each sampled file was copied to a local workdir and analysed with PyMuPDF
  1.28. Per page: page count, page size, vector drawing-object count, placed
  raster-image coverage (grid union of image bounding boxes as % of page area),
  live text vs outlined/no text, and CMYK vs RGB content signals.
- Page classifier:
  - **effectively-empty**: image coverage < 2%, < 3 vector objects, < 5 chars.
  - **raster-dominant**: image coverage >= 50%.
  - **vector-dominant**: image coverage < 10% and >= 5 vector objects.
  - **mixed**: everything else (includes text-forward pages and pages with
    10-50% raster coverage).
- Vector-class pages were calibrated: median 64 drawing objects per page,
  108 of 134 with >= 30 objects, so the class captures genuine vector art
  rather than a few rules or borders.

## Sample disposition

- 157 sampled; 26 (16.6%) were 0-byte and unreadable; 1 opened but held no
  pages; 130 readable files yielded 665 analysed pages.
- **All 26 zero-byte files came from the recycle-bin share.** A follow-up
  metadata-only check (400 recycle vs 400 live, independent draw) found
  **65.8% of recycle-share PDFs are 0-byte** versus **0% of the live corpus**.
- The recycle share is 3,684 PDFs (24.9% of the corpus) and the temp share is
  1,604 PDFs (10.9%); the two overlap. Netting out recycle-bin stubs, the
  effective usable corpus is roughly **12,350 PDFs**, not 14,776.

## Page-class distribution (page-weighted, n = 665)

| Class              | Share |
|--------------------|-------|
| Vector-dominant    | 20.2% |
| Mixed              | 64.2% |
| Raster-dominant    | 13.8% |
| Effectively-empty  |  1.8% |

Sensitivity: a single 250-page ganged card-imposition file supplied 37.6% of
all sampled pages (all classed mixed). Excluding that one outlier, the
distribution shifts to **vector 32.3% / mixed 42.7% / raster 22.2% / empty
2.9%**. The true corpus value sits between these two views; page-level totals
below carry correspondingly wide uncertainty.

## Pages per file

- 50.8% of files are single-page; 30.0% are 2-page; 81% are <= 2 pages.
- Median 1 page. Mean is skewed by a thin tail: ~3% of files hold 21+ pages
  (booklets, multi-up impositions, variable-data runs) and carry most of the
  total page inventory.

## Colour space

- CMYK signal present on 74.9% of pages and 71.5% of files (DeviceCMYK / spot
  `Separation` / DeviceN operators or CMYK image colour spaces).
- The remaining ~28% are RGB-only content, notable in a print-ready set.

## Text handling

- 80.0% of pages carry live text, 10.7% are outlined art (text converted to
  paths, few/no live glyphs), 9.3% carry neither (raster/empty).
- Within vector-dominant pages specifically: 67% retain live text, 27% are
  outlined.
- Both are usable for minting but differ: live text converts to SVG `<text>`
  or to clean glyph outlines; outlined text is already path geometry.

## Content archetypes (geometry-based, bleed-tolerant)

Classification is by trim geometry only (dimensions include print bleed), so
these are approximate families rather than exact product types.

| Archetype                              | Page share | File share |
|----------------------------------------|-----------|-----------|
| Small-format card (business card/label/sticker) | 47.2% | 26.9% |
| Flyer / leaflet (A6-A4 and near sizes) | 37.3% | 35.4% |
| Poster (A3-A2 and near sizes)          |  6.9% | 16.9% |
| Banner / long-format strip             |  5.4% |  6.9% |
| Large-format (A1+)                     |  3.2% | 13.8% |

The file-share column is the more representative view; the small-format page
share is inflated by the single 250-page card-imposition file.

## Extrapolation to the 14,776-PDF corpus

Estimator: sampled count per file scaled to 14,776 files (the 0-byte stubs are
retained in the denominator, so page totals are already net of them).

- Estimated total pages: **~62,600** (wide range; a re-draw without a high-page
  imposition file lands nearer ~39,000).
- **Vector-dominant pages: ~12,600** (point estimate; plausible range roughly
  10,000-15,000 given the concentration of vector pages in a minority of files).
  56 of the 130 readable sampled files contributed at least one
  vector-dominant page.
- Live-text pages: ~50,000. Outlined-art pages: ~6,700.

## File size

Non-empty sample: median 0.73 MB, 25th pct 0.11 MB, 75th pct 2.76 MB, max
199 MB. The long tail is driven by embedded high-resolution raster imagery.

## Takeaways

1. The realistic yield of vector-dominant pages is **~12,600** across the
   corpus, concentrated in a minority of files.
2. Most pages (~80%) retain live text, so glyph handling for minting is mostly
   a live-text-to-SVG path, not an all-outlines path.
3. CMYK dominates (~75%), so the pipeline must treat CMYK as the default colour
   model, not an edge case.
4. The recycle-bin share is largely dead weight: ~66% of it is 0-byte stubs.
   Corpus sizing should work from the ~12,350 usable PDFs, not the nominal
   14,776.
