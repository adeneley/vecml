#!/usr/bin/env bash
# vecml pod deploy SKELETON. Creates ONE RunPod pod via the REST API.
#
# SAFETY CONTRACT (do not weaken):
#   - Without --yes this is a DRY RUN. It prints the cost-relevant summary and
#     the exact request body, and touches no network. Creates nothing.
#   - With --yes it POSTs one pod to the RunPod REST API and prints its id.
#   - It never reads or writes a key into the repo. RUNPOD_API_KEY comes from the
#     environment (source your gitignored env file first).
#   - This is scaffolding. Prices below are verified 18 Jul 2026 but re-check the
#     live rate before a real run; the CPU per-vCPU rate is UNVERIFIED.
#
# Usage:
#   source ~/path/to/runpod.env           # sets RUNPOD_API_KEY, RUNPOD_VOLUME_ID, ...
#   ./deploy.sh --gpu RTX5090 --minutes 30 --job "uv run python -m vecml.train ..."
#   ./deploy.sh --cpu --vcpu 32 --minutes 5 --job "uv run python -m vecml.shard ..."
#   add --yes to actually deploy; add --spot for interruptible pricing.
set -Eeuo pipefail

# ---- verified rate card (USD/hr, Secure Cloud, 18 Jul 2026) ----------------
# Kept as case lookups (not associative arrays) so this runs on macOS bash 3.2.
# gpu_rate/gpu_id: verified $/hr; RunPod REST gpuTypeIds are UNVERIFIED exact
# strings, confirm against GET /gpus before a real run.
gpu_rate() { case "$1" in
    RTX5090) echo 0.99;;   # RTX 5090 32GB   - default training card
    A5000)   echo 0.27;;   # RTX A5000 24GB  - sanity tier
    A100)    echo 1.39;;   # A100 PCIe 80GB  - dataloader-bound escape hatch
    H100NVL) echo 3.19;;   # H100 NVL 94GB   - finals only
    *) return 1;; esac; }
gpu_id() { case "$1" in
    RTX5090) echo "NVIDIA GeForce RTX 5090";;   # UNVERIFIED exact string
    A5000)   echo "NVIDIA RTX A5000";;          # UNVERIFIED exact string
    A100)    echo "NVIDIA A100 80GB PCIe";;     # UNVERIFIED exact string
    H100NVL) echo "NVIDIA H100 NVL";;           # UNVERIFIED exact string
    *) return 1;; esac; }
CPU_VCPU_RATE=0.034   # UNVERIFIED, console-only. Assumed compute-optimized rate.

# ---- defaults --------------------------------------------------------------
MODE="gpu"; GPU="RTX5090"; VCPU=32; MINUTES=30; SPOT="false"; YES="false"
IMAGE="${VECML_IMAGE:-}"                       # public GHCR image, e.g. ghcr.io/adeneley/vecml-gpu:latest (or -cpu)
JOB_CMD=""; REPO_REF="${REPO_REF:-main}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-50}"
VOL_ROOT="${VOL_ROOT:-/workspace}"
API="https://rest.runpod.io/v1"

usage() { grep '^# ' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# Default image tracks the mode unless the caller overrides.
default_image() {
  if [ "${MODE}" = "gpu" ]; then echo "ghcr.io/adeneley/vecml-gpu:latest";
  else echo "ghcr.io/adeneley/vecml-cpu:latest"; fi
}

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu)     MODE="gpu"; GPU="$2"; shift 2;;
    --cpu)     MODE="cpu"; shift;;
    --vcpu)    VCPU="$2"; shift 2;;
    --minutes) MINUTES="$2"; shift 2;;
    --job)     JOB_CMD="$2"; shift 2;;
    --ref)     REPO_REF="$2"; shift 2;;
    --image)   IMAGE="$2"; shift 2;;
    --spot)    SPOT="true"; shift;;
    --yes)     YES="true"; shift;;
    -h|--help) usage 0;;
    *) echo "unknown flag: $1" >&2; usage 1;;
  esac
done

# ---- derive spec + cost ----------------------------------------------------
IMAGE="${IMAGE:-$(default_image)}"
HOURS=$(awk "BEGIN{printf \"%.4f\", ${MINUTES}/60}")
if [ "${MODE}" = "gpu" ]; then
  RATE=$(gpu_rate "${GPU}") || { echo "unknown --gpu '${GPU}'. one of: RTX5090 A5000 A100 H100NVL" >&2; exit 1; }
  GID=$(gpu_id "${GPU}")
  SPEC="1x ${GPU} (${GID})"
  COMPUTE_TYPE="GPU"
else
  RATE=$(awk "BEGIN{printf \"%.4f\", ${VCPU}*${CPU_VCPU_RATE}}")
  SPEC="CPU pod, ${VCPU} vCPU"; COMPUTE_TYPE="CPU"
fi
# Spot is advertised as 40-60% cheaper; show a conservative 50% for the estimate.
EFF_RATE="${RATE}"
[ "${SPOT}" = "true" ] && EFF_RATE=$(awk "BEGIN{printf \"%.4f\", ${RATE}*0.5}")
EST_COST=$(awk "BEGIN{printf \"%.4f\", ${EFF_RATE}*${HOURS}}")

# ---- summary ---------------------------------------------------------------
cat <<SUMMARY

=========================== vecml deploy summary ===========================
  mode           : ${COMPUTE_TYPE}
  spec           : ${SPEC}
  pricing        : \$${RATE}/hr on-demand$( [ "${SPOT}" = true ] && echo "  ->  \$${EFF_RATE}/hr spot (est, 50% off)" )
  est. duration  : ${MINUTES} min  (${HOURS} hr)
  est. compute   : \$${EST_COST}   ( ${EFF_RATE} x ${HOURS} hr )
  container disk : ${CONTAINER_DISK_GB} GB (ephemeral, ~\$0.10/GB/mo, negligible per-run)
  network volume : ${RUNPOD_VOLUME_ID:-<UNSET RUNPOD_VOLUME_ID>}  (billed monthly, not per-run)
  image          : ${IMAGE:-<UNSET: pass --image or set VECML_IMAGE>}
  repo ref       : ${REPO_REF}
  job            : ${JOB_CMD:-<none: pod will idle with deps synced>}
  spot           : ${SPOT}
============================================================================
  NOTE: network volume rent (~\$0.07/GB/mo) applies whether or not this runs.
        Estimate above is COMPUTE only. Terminate with ./teardown.sh when done.
SUMMARY

# ---- build request body ----------------------------------------------------
# env passed to the pod; startup.sh reads REPO_REF / JOB_CMD / VOL_ROOT.
CT_LOWER=$(printf '%s' "${COMPUTE_TYPE}" | tr '[:upper:]' '[:lower:]')
read -r -d '' BODY <<JSON || true
{
  "name": "vecml-${CT_LOWER}-$(date -u +%H%M%S)",
  "imageName": "${IMAGE}",
  "computeType": "${COMPUTE_TYPE}",
  $( [ "${MODE}" = "gpu" ] && echo "\"gpuTypeIds\": [\"${GID}\"], \"gpuCount\": 1," || echo "\"vcpuCount\": ${VCPU}," )
  "cloudType": "SECURE",
  $( [ -n "${RUNPOD_REGISTRY_AUTH_ID:-}" ] && echo "\"containerRegistryAuthId\": \"${RUNPOD_REGISTRY_AUTH_ID}\"," )
  "interruptible": ${SPOT},
  "containerDiskInGb": ${CONTAINER_DISK_GB},
  "networkVolumeId": "${RUNPOD_VOLUME_ID:-}",
  "ports": ["8000/http", "7300/http", "22/tcp"],
  "env": {
    "REPO_URL": "https://github.com/adeneley/vecml.git",
    "REPO_REF": "${REPO_REF}",
    "JOB_CMD": "${JOB_CMD}",
    "VOL_ROOT": "${VOL_ROOT}",
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "safe.directory",
    "GIT_CONFIG_VALUE_0": "*"
  }
}
JSON

if [ "${YES}" != "true" ]; then
  echo
  echo "DRY RUN (no --yes). Request body that WOULD be sent to POST ${API}/pods:"
  echo "${BODY}"
  echo
  echo "Re-run with --yes to deploy. This dry run created nothing and made no network call."
  exit 0
fi

# ---- live path (requires --yes) --------------------------------------------
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY in your environment (gitignored env file); never commit it}"
: "${IMAGE:?set --image or VECML_IMAGE}"
: "${RUNPOD_VOLUME_ID:?set RUNPOD_VOLUME_ID (the network volume id from the console)}"

echo "Deploying (POST ${API}/pods) ..."
RESP=$(curl -sS -X POST "${API}/pods" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "${BODY}")
echo "${RESP}"
POD_ID=$(printf '%s' "${RESP}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null || true)
if [ -n "${POD_ID}" ]; then
  echo
  echo "Pod created: ${POD_ID}"
  echo "Tear down with: ./teardown.sh ${POD_ID}"
else
  echo "No pod id parsed from response. Check the error above before assuming nothing was created." >&2
  exit 1
fi
