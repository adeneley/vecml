#!/usr/bin/env bash
# Pod-side dead-man switch. deploy.sh prepends this (backgrounded) to every
# JOB_CMD: the pod terminates ITSELF via the RunPod API when the TTL expires,
# so a dead Mac-side watcher, a dropped home connection, or a job that idles
# forever (flight.py's deliberate end-of-plan idle) can no longer buy hours
# of nothing. The 21 Jul CE-sweep crash idled ~6h/$6 because the only kill
# switch lived on the Mac.
#
# Needs RUNPOD_API_KEY + TTL_MINUTES in the pod env (deploy.sh passes both)
# and RUNPOD_POD_ID (RunPod injects it on every pod). Disarms loudly but
# harmlessly if any are missing, so old JOB_CMDs and local runs are safe.
set -u

TTL_MINUTES="${TTL_MINUTES:-480}"
if [ -z "${RUNPOD_API_KEY:-}" ] || [ -z "${RUNPOD_POD_ID:-}" ]; then
  echo "[ttl] RUNPOD_API_KEY or RUNPOD_POD_ID missing; watchdog DISARMED" >&2
  exit 0
fi

echo "[ttl] armed: pod ${RUNPOD_POD_ID} self-terminates in ${TTL_MINUTES} min"
sleep $(( TTL_MINUTES * 60 ))

echo "[ttl] TTL of ${TTL_MINUTES} min reached; terminating pod ${RUNPOD_POD_ID}"
# Stop first (halts billing even if the delete hiccups), then delete.
curl -sS -X POST "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" || true
curl -sS -X DELETE "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}"
