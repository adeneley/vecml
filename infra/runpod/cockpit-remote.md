# Cockpit over remote runs (design note)

The local training cockpit already renders a run from a stream of JSON events
(loss, img/s, sample previews, veval metrics). Nothing about those events is
local-only: the trainer running on a RunPod pod emits the **same JSON event
schema**. The only question is transport from pod to Mac. Two options, one
recommended.

## v1 (recommended): SSH tunnel to the pod's event port

The trainer already exposes its events over a small HTTP/WebSocket server (the
repo ships FastAPI + uvicorn); `deploy.sh` opens `8000/http` and `22/tcp` on the
pod. To watch a remote run:

```
ssh -N -L 8000:localhost:8000 root@<pod-ip> -p <ssh-port>
# then point the local cockpit at http://localhost:8000 as if the run were local
```

Why this first: zero new code. The cockpit connects to `localhost:8000` exactly
as it does for a Mac run; the tunnel makes the pod's port look local. Live,
low-latency, and it dies cleanly when the pod is torn down. RunPod gives every
pod SSH access, so no extra infrastructure.

Cost of v1: the tunnel is free (it rides the SSH connection); it only works while
the pod is up, which is fine because that is exactly when you want to watch.

## v2 (later): tail a JSONL on the network volume

The trainer also appends every event to `/workspace/runs-logs/<run>/events.jsonl`
(persisted, survives teardown). A local watcher periodically `rsync`s or reads
that file and feeds the cockpit. This gives **post-hoc replay** of a finished run
and survives a dropped connection, at the cost of polling latency and a sync
mechanism to build. Good as a durability layer under v1, not a replacement.

## Recommendation

Ship **v1** for live monitoring now (no implementation beyond opening the port,
which deploy.sh already does). Keep the `events.jsonl`-on-volume write in the
trainer regardless, because it costs nothing and unlocks v2 replay for free when
a "review last night's three parallel runs" workflow becomes worth building. No
implementation in this task; this note fixes the contract so the trainer emits
events the same way local and remote.
