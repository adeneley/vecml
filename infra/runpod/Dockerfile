# vecml job image. Two build targets in one file:
#
#   cpu  (default) -> python:3.11-slim base. For CPU-only work: sharding,
#                     wrecking, dataset prep, eval, inference. Small, no CUDA.
#   gpu             -> RunPod PyTorch base with CUDA. For training on 5090s etc.
#
# Build (from repo root, so the build context includes pyproject + uv.lock):
#   docker build -f infra/runpod/Dockerfile --target cpu -t vecml:cpu .
#   docker build -f infra/runpod/Dockerfile --target gpu -t vecml:gpu .
#
# Recommendation: use :cpu for generation/sharding/eval jobs (no GPU rental at
# all, or a cheap CPU pod). Use :gpu only when a job actually trains on a card.
#
# The repo is NOT baked into the image. It is cloned/pulled at container start by
# startup.sh from REPO_REF, so a new commit does not need a rebuild. Only a
# dependency change (pyproject/uv.lock) needs an image rebuild. Deps are
# preinstalled here into a warm uv cache so pod startup is a fast delta sync.

# ---------------------------------------------------------------------------
# Shared dependency layer, parameterized by base image.
# ---------------------------------------------------------------------------
ARG CPU_BASE=python:3.11-slim
# RunPod's maintained PyTorch base (CUDA + cuDNN + py3.11). Pin a concrete tag
# for reproducibility; bump deliberately. Verify the tag exists before a GPU run.
ARG GPU_BASE=runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# ===== CPU target =====
FROM ${CPU_BASE} AS cpu
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:/root/.local/bin:$PATH
# git for repo pull at start; build-essential for any wheels that need it;
# libgl/libglib for opencv-python-headless runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git ca-certificates build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# uv, via the official installer.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
WORKDIR /app
# Warm the dependency layer: copy only the manifests so this layer is cached
# until deps actually change. This is the "warm cache" startup.sh delta-syncs
# against. Install the main + dev + baselines groups so both eval and VTracer
# baseline are present.
COPY pyproject.toml uv.lock* ./
RUN uv sync --extra dev --extra baselines --no-install-project \
    || uv sync --extra dev --extra baselines --no-install-project --no-frozen
# Bootstrap script must live in the image (the repo it pulls does not exist yet
# at container start). Copied last so a script edit is a cheap top layer.
COPY infra/runpod/startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh
ENTRYPOINT ["/app/startup.sh"]

# ===== GPU target =====
FROM ${GPU_BASE} AS gpu
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:/root/.local/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
WORKDIR /app
COPY pyproject.toml uv.lock* ./
# On the GPU base, torch/torchvision already ship CUDA-enabled. We still let uv
# resolve the rest of the tree; if uv would replace the base torch with a CPU
# wheel, prefer installing everything-except-torch (adjust the extra/markers in
# pyproject before a real GPU build). Kept simple here as scaffolding.
RUN uv sync --extra dev --extra baselines --no-install-project \
    || uv sync --extra dev --extra baselines --no-install-project --no-frozen
COPY infra/runpod/startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh
ENTRYPOINT ["/app/startup.sh"]
