# Playbooks: the proven end-to-end recipes (19 Jul 2026)

Copy-paste runbooks for every remote operation, exactly as last executed
successfully. Read `lessons.md` before improvising; every flag below exists
because something burned without it. General setup (images, rates, API) is
in `README.md`; volume layout in `volume-layout.md`.

Every shell below assumes:

```bash
cd ~/development/ai/vectorizer
set -a; source infra/runpod/.env; set +a     # plain source does NOT export
```

**Discipline:** every deploy prints a pod id — tear it down the moment the
job's `rc=` line appears (`bash infra/runpod/teardown.sh <pod-id>`). The
network volume survives teardown; checkpoints/logs/datasets all live there.
Pods have no SSH: the console log on the volume is the only debugger, so
jobs must print loudly.

---

## 1. Mint a dataset (CPU pod, ~$0.30, ~12 min at n=2000)

The corpus never leaves the datacenter: raw parquet comes from HuggingFace,
the gated clean tier (1,464,723 SVGs) is cached at
`corpus-cache/svg-stack-clean.tar.gz` on the volume, and the sha ledger
`datasets/used-shas.txt` keeps every sample disjoint from all previous
train/val/test sets. On a cache hit the corpus steps are skipped entirely.

```bash
N=2000; NAME=train-2k-remote; SEED=777
bash infra/runpod/deploy.sh --cpu --vcpu 32 --minutes 40 --job "\
export PATH=/root/.local/bin:/opt/venv/bin:\$PATH; \
uv run python scripts/corpus_remote.py --cache /workspace/corpus-cache/svg-stack-clean.tar.gz --work /tmp/corpus && \
uv run python scripts/sample_src.py --clean /tmp/corpus/clean --n ${N} --out /tmp/wk/src --exclude-shas /workspace/datasets/used-shas.txt --record-shas --seed ${SEED} && \
uv run python scripts/wreck.py --in /tmp/wk/src --out /tmp/wk/${NAME} --bg random --curriculum --variants 2 --jobs 30 --quiet --seed ${SEED} && \
uv run python scripts/finalize_set.py --root /tmp/wk/${NAME} && \
tar -czf /workspace/datasets/${NAME}-run.tar.gz -C /tmp/wk ${NAME} && \
ls -la /workspace/datasets/" --yes
```

The `export PATH` prefix is required until the images are rebuilt (lesson
14). Measured timings (32 vCPU): HF download 4.5 min (first time only),
gate 4 min (first time only), cache pack 2 min (first time only),
sample+wreck+finalize+tar ~2.5 min at n=2000 (wreck scales ~linearly).
Verification: the gate's tier counts must reproduce
`{clean: 1464723, warn: 723800, reject: 95352}` on a cache rebuild.

## 2. Training flight (RTX 5090 pod, $0.99/hr)

One pod runs a sequence of training runs from a flight plan
(`infra/flightplans/*.json`), each with a spectator cockpit on port 7300.
The perf recipe is applied via `--from-bench`; use the committed winner
`infra/recipes/perflab-5090.json` (880 img/s: batch 16, bf16, sync 50,
fused Adam, channels_last, compile reduce-overhead) instead of re-running
bench unless the model changed.

```bash
bash infra/runpod/deploy.sh --gpu RTX5090 --minutes 120 --job "\
mkdir -p /tmp/data && \
vecml_extract /workspace/datasets/train-2k-remote-run.tar.gz /tmp/data && \
ls /tmp/data/*/ | head -3 && ls \$(ls -d /tmp/data/*/ | head -1)\$(ls /tmp/data/*/ | head -1) 2>/dev/null; \
uv run python scripts/flight.py --plan infra/flightplans/<plan>.json --from-bench infra/recipes/perflab-5090.json --host 0.0.0.0 --port 7300" --yes
```

Cockpit: `https://<pod-id>-7300.proxy.runpod.net` (readonly; run tab hidden;
header names the active run). Preflight `ls` of one sample dir stays in
every job: a tarball missing labels.png crashes 7 seconds in otherwise
(lesson 11). Checkpoints land in `checkpoints/<run-name>/` on the volume:
`best.pt` (best held-out epoch mean), `last.pt` (newest), `events.jsonl`
(telemetry, samples excluded).

## 3. Perf lab (only when the model/architecture changes)

```bash
bash infra/runpod/deploy.sh --gpu RTX5090 --minutes 45 --job "\
mkdir -p /tmp/data && vecml_extract /workspace/datasets/train-10k-run.tar.gz /tmp/data && \
vecml_extract /workspace/datasets/train-10k-labels.tar.gz /tmp/data && \
uv run python scripts/perflab.py --data /tmp/data/train-10k --classes 16 --emit runs/perflab/perflab.json" --yes
```

~4 min of GPU time. Copy the emitted winner over
`infra/recipes/perflab-5090.json` if it beats the committed one. Known
results: duo (two trainers per card) LOSES; larger batches only win under
compile; bench numbers without a per-step loss read overstate the trainer.

## 4. Watch a pod / fetch results (no pod access needed)

```bash
uv run --with boto3 python scripts/pod_log.py runs                  # recent run dirs
uv run --with boto3 python scripts/pod_log.py tail --after 20260719-1108   # follow console (stamp = pod createdAt)
uv run --with boto3 python scripts/pod_log.py ls checkpoints/
uv run --with boto3 python scripts/pod_log.py get checkpoints/<run>/best.pt runs/<run>/best.pt
```

`--after` matters: crash loops leave a run dir every ~18s, so "newest"
without a floor can be a corpse (lesson: 19 Jul). Agent pattern: wrap
`tail` in a Monitor that greps for progress + `rc=|Traceback|FATAL`, act on
the notification, never poll with sleeps.

## 5. Score a checkpoint locally

```bash
uv run --with boto3 python scripts/pod_log.py get checkpoints/<run>/best.pt runs/<run>/best.pt
uv run python scripts/relay_eval.py --data data/relay-test --ckpt runs/<run>/best.pt --out runs/relay/<run>
```

DualHead checkpoints automatically get the label-flatten variant (`*_flat`).
Reference numbers to beat (24-image relay-test, dE mean): damage 2.22;
RGB-only baseline cleanup 0.55, vtracer_relay 1.81, rust_relay 2.09;
engine ceiling on perfect input: vtracer 1.28, rust 1.44.

## 6. Rarely: push a big file from the Mac

Last resort only (the data factory exists so you don't do this):
`uv run --with boto3 python scripts/volume_put.py <file> datasets/<name>` —
single-stream 8MB parts; concurrency 524s the gateway (lesson: 19 Jul).
