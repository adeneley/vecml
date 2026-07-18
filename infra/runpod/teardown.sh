#!/usr/bin/env bash
# vecml pod teardown. Stops then terminates a pod. Leaves the network volume
# (and all its shards/checkpoints/logs) INTACT.
#
# Usage:
#   source ~/path/to/runpod.env      # RUNPOD_API_KEY
#   ./teardown.sh <pod-id>
#   ./teardown.sh <pod-id> --stop-only   # stop but do not terminate (still bills volume disk!)
#
# Terminate stops per-second compute billing immediately. The network volume
# keeps billing monthly by design (it holds your data); delete it in the console
# only when the project is over.
set -Eeuo pipefail

API="https://rest.runpod.io/v1"
POD_ID="${1:-}"; MODE="${2:-terminate}"
[ -n "${POD_ID}" ] || { echo "usage: ./teardown.sh <pod-id> [--stop-only]" >&2; exit 1; }
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY in your environment (gitignored env file)}"

auth=(-H "Authorization: Bearer ${RUNPOD_API_KEY}")

# 1. Stop (best-effort; a running pod should be stopped before terminate).
#    UNVERIFIED exact stop path; confirm against the live API reference.
echo "Stopping pod ${POD_ID} ..."
curl -sS -X POST "${API}/pods/${POD_ID}/stop" "${auth[@]}" || true
echo

if [ "${MODE}" = "--stop-only" ]; then
  echo "Stopped only. NOTE: a stopped pod still bills volume disk at \$0.20/GB/mo."
  echo "Run './teardown.sh ${POD_ID}' again to terminate and stop that fee."
  exit 0
fi

# 2. Terminate (DELETE). This is what actually ends billing and frees the pod.
echo "Terminating pod ${POD_ID} ..."
curl -sS -X DELETE "${API}/pods/${POD_ID}" "${auth[@]}"
echo
echo "Terminated. Network volume left intact (still billed monthly until you"
echo "delete it in the console)."
