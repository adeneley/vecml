# ai/vectorizer

Neural front-end for the raster-to-vector engine at `~/development/vectorizer`.

The Rust engine traces flat-colour art perfectly when the input is clean.
This repo trains the model that makes dirty input clean: degraded raster in,
label map out, which the engine then traces into seam-free SVG.

```
ugly image -> model (this repo) -> label map -> Rust engine -> SVG
```

## Layout

```
src/vecml/
  degrade/     the wrecking pipeline: turns clean renders into realistic junk
  data/        corpus building, sharding, dataloaders
  models/      segmentation / cleanup model definitions
  train/       training loops, checkpointing, configs
  evaluate/    metrics + handoff to the Rust engine's veval gallery
scripts/       one-off runnable entry points
configs/       run configs (yaml)
docs/          plans and analyses (see docs/plan.md first)
data/          (gitignored) corpora and shards
checkpoints/   (gitignored) model weights
runs/          (gitignored) training logs
```

## Setup

```bash
uv sync
uv run python -c "import torch; print(torch.__version__)"
```

See `CLAUDE.md` for the full strategy, data sources, and GPU cost analysis.
