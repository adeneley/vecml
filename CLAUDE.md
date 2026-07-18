# CLAUDE.md — ai/vectorizer

The ML half of the vectorizer project. The classical engine lives at
`~/development/vectorizer` (Rust, shared-edge planar DCEL, read its CLAUDE.md
before touching anything that feeds it). This repo builds the "smart eyes"
that sit in FRONT of that engine:

```
ugly image -> [trained model: clean + segment] -> [Rust engine: trace + fit] -> SVG
```

The model's job is to turn degraded input (blurry JPEG logos, scans, noisy
phone photos) into a clean label map (which pixel belongs to which region,
how many colours, where the true edge sits). The Rust engine already draws
perfectly when fed clean regions; on clean images it is at free-tier parity
today. This repo exists to close the degraded-input gap, which the moat
research identified as vectorizer.ai's one durable technical advantage.

## Context you should not re-derive

- Full moat research: `~/Documents/docuprint/reports/vectorizer-moat-quantified.html`
  (18 Jul 2026, adversarially verified) and its June companion in the same dir.
- vectorizer.ai is a hybrid: neural front-end for perception, classical
  geometry for the drawing. NOT an image-to-SVG transformer. We mirror that
  shape because we already own a superior classical back half.
- Their data moat is big but UNLABELED (they purge pixels ~24h). Ours is
  small but PAIRED. Do not chase their corpus size.

## The data strategy (three sources, three jobs)

1. **Synthetic (scale + labels).** Rasterize open-licensed SVGs (Openclipart
   CC0, Wikimedia, font glyphs, icon sets), then programmatically wreck them:
   JPEG quality ladder, downscale/upscale, blur, noise, dithering, simulated
   print-and-scan. Because we made the degradation, ground truth is perfect
   by construction. This trains the model.
2. **NAS pairs (realism).** The ARTWORK NAS (192.168.0.205) holds years of
   job folders where a customer's junk raster AND the staff-redrawn
   production vector coexist. Mining those gives real degraded-input pairs
   with human-quality labels. This VALIDATES the model (and later fine-tunes
   it). Survey not yet run; volume unknown (guess: hundreds to low
   thousands of usable pairs). `~/Documents/artworksearch` has the tooling.
3. **Shop stream (currency).** Ongoing customer uploads at Docuprint. Every
   real input the engine fails on becomes a named regression case in the
   Rust repo's veval gallery. Low volume, highest value per image.

Target: robustness on the PRINT-SHOP input distribution, not on
vectorizer.ai's whole consumer distribution. Smaller, reachable, and it is
the segment the engine's structural edges (local, free, DXF) already serve.

## Model class and scale

U-Net / SegFormer-style segmentation + cleanup, 30-100M params, BF16/FP16.
Small by modern standards; VRAM is never the constraint, iteration speed is.
Resolutions: 128px sanity, 256px prototype, 512px serious, 1024px final
fine-tune only after the design is frozen at 512px.

## GPU options (analysed 18 Jul 2026, prices USD, provider list in docs/)

**STATUS: these are costed IDEAS from the analysis, not decisions. Aden has
not committed to any rental, tier, or budget. NOTHING here authorizes
spending money. Every GPU rental, of any size, requires Aden's explicit
go-ahead at the point of usage: name the card, hours, and expected cost,
and wait for his yes before renting.**

- **Default candidate: RTX 5090 @ $0.99/hr.** Best $/work (~$2.06 per H200-hour-equiv).
  Prototype run (256px, 200k imgs) = overnight, $10-21. Use 2-3 per pod for
  parallel independent experiments, NOT DDP, during exploration.
- **Sanity tier: RTX A5000/A4500 @ ~$0.26/hr.** Cheapest work on the list.
- **Dataloader-bound escape hatch: A100 PCIe @ $1.39/hr** (31 vCPU pods).
- **Finals only: 2x H100 NVL DDP @ $3.19/ea**, ~1 week per 1024px run.
- **Traps: H100 PCIe (worst value), B200/B300/H200 for prototyping, the
  48-96GB VRAM-tax tier (L40/L40S/6000 Ada), starved multi-GPU pods.**
- Budget scenarios (illustrative only, not approved): LEAN ~$1,300 /
  ~3 months; BALANCED ~$2,700 / ~2-2.5 months.
  Spot pricing (if offered) beats the patience play: checkpoint every
  15-30 min and take 40-60% off.
- **Pre-shard the dataset before renting.** Wreck images offline into
  ready-to-train shards (WebDataset). On-the-fly augmentation starves the
  cheap pods and is the main way to waste GPU dollars here.
- First paid hour: measure real img/s on a 5090 and re-anchor all estimates.
  The "serious run = 50-150 H200-hours" figure is the softest number.

## Mac usage

M5 MacBooks are for code development, dataloader debugging, and
overfit-on-100-images sanity checks only. A serious run takes months on
Apple silicon. Known trap from the starter-model project: the MPS
`non_blocking` data-corruption bug; if a Mac run mysteriously will not
learn, inspect the tensors actually reaching the model first.

## VTracer's role (baseline + stopgap, never the engine)

The `baselines` extra installs VTracer (open-source tracer, pip package
`vtracer`). It has exactly two jobs:
1. Baseline in every eval: our full pipeline vs VTracer on the same wrecked
   inputs. If we do not beat the free tool, stop and rethink.
2. Stopgap back half during early training, before the Rust engine grows its
   external label-map door (`--quant neural`): trace the model's cleaned
   output with VTracer as a quick proxy for end-to-end results.
The engine of record is and stays the Rust engine. Do not build on VTracer.

## Interop with the Rust engine

The model's output contract is a label map the Rust engine can ingest
(likely PNG-encoded label indices + a palette, format TBD when the first
model exists). Success metric is not model loss: it is the veval metrics
(deltaE, PSNR, holes) of the FULL pipeline on degraded inputs, compared in
the same gallery as every other attempt. Wire it as an optional front-end
(`--quant neural` or a preprocessing step) so it can be A/B'd against
`flat_quant` honestly.

## Conventions

- No em-dashes anywhere (user house rule), including comments and chat.
- Python, uv-managed. `uv sync` then `uv run ...`.
- data/, checkpoints/, runs/ are gitignored: regenerable or heavy.
- Commit at meaningful checkpoints, same style as the Rust repo.
