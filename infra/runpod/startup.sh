#!/usr/bin/env bash
# vecml pod entrypoint. Runs inside the container built from infra/runpod/Dockerfile.
#
# What it does, in order:
#   1. Mount check: the network volume is expected at $VOL_ROOT (default /workspace).
#   2. Clone or update the repo at REPO_REF (branch, tag, or commit sha).
#   3. uv sync against the warm dependency cache baked into the image (fast delta).
#   4. exec JOB_CMD, streaming stdout+stderr to a per-run log on the volume.
#
# It is robust to re-runs: a re-deployed pod re-uses the same volume, fetches the
# latest ref, and re-syncs. Nothing here is destructive to the volume's data.
#
# Env contract (set by deploy.sh via the pod's env):
#   REPO_URL    git remote (default: the public/known vectorizer-ml remote)
#   REPO_REF    branch/tag/sha to run (default: main)
#   JOB_CMD     the command to execute, e.g. "uv run python -m vecml.train ..."
#   VOL_ROOT    network volume mount point (default: /workspace)
#   RUN_NAME    label for this run's log dir (default: derived from timestamp)
set -Eeuo pipefail

REPO_URL="${REPO_URL:-https://github.com/adeneley/vecml.git}"
REPO_REF="${REPO_REF:-main}"
VOL_ROOT="${VOL_ROOT:-/workspace}"
RUN_NAME="${RUN_NAME:-run-$(date -u +%Y%m%d-%H%M%S)}"
JOB_CMD="${JOB_CMD:-}"

REPO_DIR="${VOL_ROOT}/repo"          # repo checkout lives on the volume, survives re-runs
LOG_DIR="${VOL_ROOT}/runs-logs/${RUN_NAME}"
LOG_FILE="${LOG_DIR}/console.log"

log() { printf '[startup %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# 1. Volume present?
if [ ! -d "${VOL_ROOT}" ]; then
  echo "FATAL: volume root ${VOL_ROOT} not found. Is the network volume attached?" >&2
  exit 2
fi
mkdir -p "${LOG_DIR}" "${VOL_ROOT}/shards" "${VOL_ROOT}/checkpoints" \
         "${VOL_ROOT}/runs-logs" "${VOL_ROOT}/corpus-cache"

# Tee everything from here on into the run log on the volume.
exec > >(tee -a "${LOG_FILE}") 2>&1

log "run=${RUN_NAME} ref=${REPO_REF} vol=${VOL_ROOT}"

# 2. Clone or update the repo (on the volume, so it persists across re-runs).
if [ -d "${REPO_DIR}/.git" ]; then
  log "updating existing checkout at ${REPO_DIR}"
  git -C "${REPO_DIR}" remote set-url origin "${REPO_URL}"
  git -C "${REPO_DIR}" fetch --tags --force --prune origin
else
  log "cloning ${REPO_URL} into ${REPO_DIR}"
  rm -rf "${REPO_DIR}"
  git clone "${REPO_URL}" "${REPO_DIR}"
fi
git -C "${REPO_DIR}" checkout --force "${REPO_REF}"
# If REPO_REF is a branch, fast-forward to its latest tip; ignore for tags/shas.
git -C "${REPO_DIR}" pull --ff-only origin "${REPO_REF}" 2>/dev/null || true
log "HEAD is $(git -C "${REPO_DIR}" rev-parse --short HEAD)"

cd "${REPO_DIR}"

# 3. Sync deps against the warm image cache. --frozen uses the locked versions;
#    fall back to a normal sync if the lock is absent or drifted.
log "uv sync (warm cache at ${UV_CACHE_DIR:-default})"
uv sync --extra dev --extra baselines --frozen \
  || uv sync --extra dev --extra baselines

# 4. Run the job. Point job outputs at the volume by convention.
export VECML_VOL_ROOT="${VOL_ROOT}"
export VECML_SHARDS="${VOL_ROOT}/shards"
export VECML_CHECKPOINTS="${VOL_ROOT}/checkpoints"
export VECML_CORPUS_CACHE="${VOL_ROOT}/corpus-cache"

# Safe tarball extraction (Mac-made tarballs carry foreign uids + ._* junk that
# break GNU tar on the network volume). Jobs should use this instead of raw tar.
vecml_extract() { tar --no-same-owner --exclude='._*' -xzf "$1" -C "$2"; }
export -f vecml_extract

# CUDA gate: if this pod has a GPU, refuse to train blind on a CPU fallback
# (burned us once: cu130 torch vs 12.8 driver ran a 5090 pod on CPU).
# Set ALLOW_CPU=1 to bypass deliberately for CPU-only jobs on GPU pods.
if command -v nvidia-smi >/dev/null 2>&1 && [ -z "${ALLOW_CPU:-}" ]; then
  log "CUDA gate: verifying torch can reach the GPU"
  if ! uv run python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    uv run python -c 'import torch; print("torch", torch.__version__, "built-cuda", torch.version.cuda, "available", torch.cuda.is_available())' || true
    nvidia-smi --query-gpu=name,driver_version --format=csv || true
    echo "FATAL: GPU present but torch.cuda.is_available() is False. Refusing to run on CPU (set ALLOW_CPU=1 to override)." >&2
    exit 3
  fi
  log "CUDA gate: passed"
fi

if [ -z "${JOB_CMD}" ]; then
  log "no JOB_CMD set. Dependencies are synced; dropping to an idle shell."
  log "Set JOB_CMD in the pod env to run a job automatically."
  exec sleep infinity
fi

log "exec: ${JOB_CMD}"
set +e
bash -lc "${JOB_CMD}"
JOB_RC=$?
set -e
log "job exited rc=${JOB_RC}"

# Leave a machine-readable marker for the cockpit / teardown to notice.
printf '{"run":"%s","ref":"%s","rc":%d,"ended":"%s"}\n' \
  "${RUN_NAME}" "$(git rev-parse --short HEAD)" "${JOB_RC}" "$(date -u +%FT%TZ)" \
  > "${LOG_DIR}/result.json"

exit "${JOB_RC}"
