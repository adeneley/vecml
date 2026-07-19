# Burn ledger: issues found during the first PoC flights (18 Jul 2026)

Each entry: what happened, the fix, and where the fix lives. Rule: a lesson
counts as learned only when its fix is committed somewhere a future run
inherits automatically.

## 1. PyPI torch is a cu130 build; RunPod drivers are CUDA 12.8
Torch imported fine, saw the GPU (`device_count()=1`) and still returned
`cuda.is_available()=False` ("driver too old"), so training silently fell back
to CPU: 6 img/s on a 5090 vs 30 on the M-series Mac.
**Fix (durable):** `[tool.uv.sources]` in pyproject pins Linux to the
`download.pytorch.org/whl/cu128` index (torch 2.11+cu128); macOS keeps PyPI
torch for MPS. Re-check this pin whenever RunPod bumps host drivers.

## 2. Never gate nothing: assert CUDA before training
The CPU fallback burned a full pod because nothing refused to run slow.
**Fix (durable):** startup.sh runs a CUDA assertion for GPU jobs before
JOB_CMD; set `ALLOW_CPU=1` in the pod env to bypass deliberately.

## 3. Mac-made tarballs poison Linux extraction
bsdtar on macOS records uid 501 ownership and AppleDouble (`._*`) files.
On the network volume, chown fails per-file and GNU tar exits non-zero,
killing any `&&` chain; the `._*` junk pollutes dataset dirs.
**Fix (durable):** extract with `--no-same-owner --exclude='._*'` (baked into
startup.sh's `vecml_extract` helper). Better: create tarballs with
`COPYFILE_DISABLE=1 tar --no-xattrs ...` on the Mac side.

## 4. Small-file work on the network volume is slow
Extracting ~7k small files / syncing a 6GB venv onto the volume takes minutes;
the same work on container-local disk takes seconds.
**Rule:** volume = artifacts (shards, checkpoints, logs, tarballs); local disk
= working set (extracted datasets, venv). Extract to /tmp, checkpoint to the
volume.
**Deferred fix:** bake the ready venv into the image (startup sync should be a
no-op delta) and stop syncing onto the volume checkout.

## 5. Boot is minutes, not seconds
Measured: ~5 min first image pull per machine (16GB image), ~5 min venv sync on
the volume, seconds for clone/extract. Same-machine relaunch skips the pull.
**Deferred fix:** slim the GPU image (redundant torch layer) and pre-bake the
venv; target ~2 min cold boot. Do not promise "seconds" for custom images.

## 6. Crash-loop is a feature
On failure the container restarts, re-clones main, re-syncs deps. A pushed fix
heals a looping pod with no redeploy (proved live with the cu128 pin). Leave
the loop running while fixing; each lap costs cents.

## 7. Diagnose from the volume, not the pod
No SSH key is configured on pods; the boot log tee'd to
`runs-logs/<run>/console.log` on the volume (readable over S3 from anywhere)
was enough to solve every issue above. Keep the tee.

## 8. Telemetry must persist server-side, not listener-side
The cockpit's event history lives in server memory and dies with the pod; the
PoC's full telemetry survived only because an external script was capturing
the SSE stream. Weights were safe (volume) but the loss curve almost wasn't.
**Deferred fix:** the trainer should append every metric/status event to a
jsonl on the volume (`runs-logs/<run>/events.jsonl`) natively, so artifacts
survive with no observer. Until then: always attach a capture script to
remote runs.

## 9. Anonymous GHCR pulls get throttled from datacenter IPs
Two CPU pods in a row hit `toomanyrequests` pulling vecml-cpu; one crawled at
~5MB/s, the next fail-looped. RunPod hosts share egress IPs, so GHCR's
anonymous rate limit is effectively already spent when your pull starts. GPU
pulls only succeeded earlier by landing on luckier hosts.
**Fixes:** (a) DONE 19 Jul 2026: GitHub token (gh auth refresh -s
read:packages) registered as RunPod registry credential "ghcr-adeneley";
deploy.sh injects containerRegistryAuthId from RUNPOD_REGISTRY_AUTH_ID in the
gitignored .env - authenticated pulls skip the anonymous limit. (b) still
open: slim the images - the CPU image carries the cu128 CUDA torch (~8.3GB
layer) it can never use; pin CPU-only torch for the cpu target (~1.5GB total).

## 10. Verify live state before destructive ops
A pod was torn down based on a pasted log snippet that was actually stale
scrollback; the pull it showed failing had just completed. Cost: the whole
pull, redone. Rule: before teardown/delete/restart, fetch the resource's
CURRENT state (API call, fresh log tail) in the same minute you act on it. A
screenshot, a paste, or "the last thing I saw" is history, not state.

## 11. Slimmed tarballs lie about the pair contract
train-10k-run.tar.gz was slimmed overnight to cut upload time (clean.png +
wrecked only) and later reused for a label-head flight; every sample dir then
failed the labels filter and the flight crashed 7s in with "no usable sample
dirs". The extraction succeeded, the data looked fine locally, and only loud
dataset diagnostics (entry counts + a probe dir listing baked into the
exception) made the cause visible from a crash log.
**Fixes:** (a) DONE: pairs.py raises with entry counts, pre-filter counts and
a probe-dir file listing, so a pod's crash log names the missing files.
(b) Rule: a dataset pack must carry the FULL pair contract (clean, wrecked,
labels.png, palette.json, meta) or its name must say what it dropped
(-rgb-only); the JOB_CMD preflight `ls` of one sample dir before training
stays in every flight job.

## 12. Trainer speed levers, measured (perflab, 5090, 19 Jul 2026)
bench.py's 609 img/s was a benchmark artifact: it never paid the per-step
loss.item() the real trainer paid, whose faithful baseline was 433 img/s.
scripts/perflab.py measured the remaining levers solo and stacked:
sync-every-50 +27%, fused Adam +4%, channels_last -14% ALONE (helps under
compile), torch.compile reduce-overhead +29% and the star of the stack.
Full stack 857 @ batch 8; batch re-sweep with the stack on flips the batch
answer to 16 (880 img/s, 2.03x). Two processes sharing the card aggregate to
788 - WORSE than one process; duo is dead for this model size. Compile tax is
12-18s once per shape.
**Fix (durable):** TrainConfig grew sync_every / fused_adam / channels_last /
compile_mode; flight.py and watch.py --from-bench apply them from a
perflab.json winner. Re-run perflab when the model grows: every one of these
numbers is model-size dependent.

## 13. Cross-image pods trip git's dubious-ownership check on the volume repo
The CPU image's uid differs from the GPU image's that created /workspace/repo;
every git command then dies with "detected dubious ownership" and startup
crash-loops before the job runs (first hit: the corpus data pod, 19 Jul).
**Fixes (durable):** startup.sh marks REPO_DIR safe before git ops (needs an
image rebuild to land), and deploy.sh injects GIT_CONFIG_COUNT/KEY_0=
safe.directory/VALUE_0=* into the pod env - git reads config from env, so
already-pulled images are fixed with zero rebuild.

## 14. bash -lc clobbers PATH on the CPU base image (rc=127, no other clue)
startup.sh verified uv, synced deps, then ran JOB_CMD via `bash -lc` - and
the job died instantly with rc=127. The CPU base's /etc/profile resets PATH
in login shells, dropping /root/.local/bin where uv lives; the GPU base's
profile doesn't, which is why every GPU pod worked. rc=127 with zero output
= "command not found before anything could log".
**Fixes (durable):** startup.sh re-exports its own verified PATH inside the
login shell (next image build); until then JOB_CMDs on the CPU image are
prefixed with `export PATH=/root/.local/bin:$PATH;`.
