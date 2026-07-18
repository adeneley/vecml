# RunPod runbook (vecml)

Everything needed so that, once you supply a RunPod API key, spinning a pod for
even a 5-minute job is one signed-off command. This directory is scaffolding
only. It creates no pods, no volumes, and no billable resource on its own.

**Hard rule carried in from CLAUDE.md / docs/plan.md:** no automated step here
may create a pod, a volume, or any billable RunPod resource. Every actual deploy
is a per-instance decision you make and confirm with `--yes` (see `deploy.sh`).
Phase 0 is free; every rental needs your explicit go-ahead at the point of use.

Pricing and API facts below were verified against runpod.io/pricing and
docs.runpod.io on 18 Jul 2026 (USD, Secure Cloud). Anything I could not confirm
from public docs is tagged **UNVERIFIED** and should be re-checked in the console
before you rely on it.

---

## 0. Container images

Two images are built on x86 by GitHub Actions (`.github/workflows/docker.yml`)
from `Dockerfile` and published to GHCR as **public** packages, so a pod pulls
them with no registry credentials:

- `ghcr.io/adeneley/vecml-cpu:latest` — CPU-only work (sharding, wrecking,
  dataset prep, eval, inference).
- `ghcr.io/adeneley/vecml-gpu:latest` — CUDA base for training on GPU cards.

Each push is also tagged with the commit sha. The workflow rebuilds only when
`pyproject.toml`, `uv.lock`, or anything under `infra/runpod/` changes (or on
manual dispatch); the repo source itself is pulled at pod start, so a code
commit needs no rebuild. Point `deploy.sh` at one of these via `--image` or
`VECML_IMAGE`, e.g. `--image ghcr.io/adeneley/vecml-gpu:latest`.

---

## 1. One-time setup (only you can do these)

These are console + account actions. The scaffolding cannot and must not do them.

### 1a. Create a restricted API key
1. Console -> Settings -> API Keys: https://www.runpod.io/console/user/settings
2. Create a key. Give it the **minimum scope** the workflow needs:
   - **Write/Manage access to Pods** (create, stop, terminate).
   - **Read access to Network Volumes** (so deploy can reference the volume id).
   - Read access to account/billing is optional and can stay off.
   - Do **not** grant Serverless or billing-mutation scopes; we do not use them.
   - RunPod's key scopes are a coarse read/restricted/all selector rather than
     per-resource ACLs (**UNVERIFIED** exact granularity in the current console);
     pick the narrowest option that still allows pod create + terminate.
3. Store the key in the Mac Keychain or an env file that is **gitignored**. It
   is never committed and never pasted into any file in this repo.

### 1b. Set a monthly spend cap
1. Console -> Billing. Enable a **spending limit / auto-pay cap** at a number you
   are comfortable losing in a bad month (e.g. the LEAN reference of a few
   hundred USD, not the whole budget).
2. Keep auto-reload off or capped so a runaway loop cannot refill itself.
   (**UNVERIFIED**: the exact label RunPod uses for the hard cap changes between
   console revisions; confirm it is a *hard* cap, not just a low-balance alert.)

### 1c. Create the network volume
The volume is the one persistent, always-billed resource. Create it once, reuse
it for every run, and leave it intact on teardown.

1. Console -> Storage -> Network Volumes -> New Network Volume.
2. **Datacenter:** pick a region that lists **both CPU pods and RTX 5090
   availability**, because a network volume is pinned to its datacenter and a
   pod can only mount a volume in the same DC. As of 18 Jul 2026 EU-RO-1 and
   EU-SE-1 have historically carried both; **verify live availability in the
   deploy screen before committing** (availability shifts daily).
3. **Size:** 50 GB to start (shards + a few checkpoints). Bump toward 100 GB
   once a 512px shard set plus rolling checkpoints are living on it. See
   `volume-layout.md`. Resizing up later is allowed; you pay for provisioned GB.
4. **Name:** `vecml-vol` (used as a label; the deploy script takes the numeric
   **volume id**, copy it from the volume's detail page).
5. Record the **volume id** and **datacenter id** into your gitignored env file:
   ```
   RUNPOD_API_KEY=...
   RUNPOD_VOLUME_ID=...
   RUNPOD_DATACENTER_ID=EU-RO-1   # must match the volume's DC
   ```

### 1d. (Optional) Wire up the MCP server for conversational control
Official package: **`@runpod/mcp-server`** (repo github.com/runpod/runpod-mcp,
bin `runpod-mcp`, latest 2.0.0 as of 18 Jul 2026, needs Node 18+). Verified on
the npm registry and the official README.

- **Recommended (hosted, OAuth, no key stored on disk):**
  ```bash
  claude mcp add --transport http runpod -s user https://mcp.getrunpod.io/
  ```
  First connect launches the "Sign in with Runpod" OAuth flow.
- **Local with your own key (key lands in the client config):**
  ```bash
  claude mcp add runpod -s user \
    -e RUNPOD_API_KEY=YOUR_API_KEY \
    -- npx -y @runpod/mcp-server@latest
  ```
- **Guided installer (detects installed agents, writes config for you):**
  ```bash
  npx @runpod/mcp-server@latest add     # 'remove' to undo
  ```
Verify with `claude mcp list`; reconnect in-session with `/mcp`.

The MCP server is a convenience layer. It can create pods, so treat it with the
same discipline as `deploy.sh`: nothing spins up without your explicit say-so.
The scripts here deliberately do **not** depend on it.

---

## 2. Per-run flow

Once 1a-1c are done, a run is three commands:

```bash
# 1. Dry-run: prints spec, $/hr, est. duration and est. cost. Creates nothing.
./deploy.sh --gpu RTX5090 --job "uv run python -m vecml.train ..." --minutes 30

# 2. Same command + --yes: actually creates the pod. This is the sign-off gate.
./deploy.sh --gpu RTX5090 --job "uv run python -m vecml.train ..." --minutes 30 --yes

# 3. Cockpit lights up (see cockpit-remote.md): the pod runs startup.sh, which
#    syncs the repo, warms the uv cache, and execs JOB_CMD, streaming logs to
#    the network volume under /runs-logs. Watch locally over an SSH tunnel.

# 4. Teardown when the job is done. Stops + terminates the pod. Volume untouched.
./teardown.sh <pod-id>
```

`deploy.sh` without `--yes` never touches the API. With `--yes` it creates
exactly one pod and prints its id. Nothing auto-terminates a running job; you
call `teardown.sh` when you have what you need. A terminated pod stops billing
compute immediately (per-second); the network volume keeps billing monthly until
you delete it in the console, which is intentional (it holds your shards).

---

## 3. Billing model

Verified 18 Jul 2026 from runpod.io/pricing and docs.runpod.io (USD, Secure Cloud):

| Resource | Rate | Notes |
|---|---|---|
| Compute (pods) | **per second** while running | Stop/terminate to stop compute billing |
| RTX 5090 (32GB) | **$0.99/hr** | our default training card |
| RTX A5000 (24GB) | **$0.27/hr** | sanity tier |
| A100 PCIe (80GB) | **$1.39/hr** | dataloader-bound escape hatch |
| H100 NVL (94GB) | **$3.19/hr** | finals only |
| CPU pod, per vCPU | **UNVERIFIED** (console-only) | see worked example; assume ~$0.03-0.04/vCPU/hr |
| Network volume < 1TB | **$0.07 / GB / mo** | billed continuously, running or not |
| Network volume > 1TB | **$0.05 / GB / mo** | |
| Network volume (high-perf tier) | **$0.14 / GB / mo** | not needed for us |
| Container disk (running) | **$0.10 / GB / mo** | ephemeral, wiped on restart |
| Volume disk on a **stopped** pod | **$0.20 / GB / mo** | the "stopped-pod disk fee" |

Two cost traps to keep in mind:
- **A stopped (not terminated) pod still bills its volume disk at $0.20/GB/mo.**
  For us this is a non-issue because we terminate rather than stop, and our
  persistence lives on the *network* volume, not the pod's volume disk.
- **The network volume bills every month whether or not a pod is attached.**
  A 50 GB volume is ~$3.50/mo; a 100 GB volume ~$7/mo. That is the standing cost
  of keeping shards warm in the datacenter. Delete the volume in the console to
  stop it.

### Worked example: 5-minute job on a 32-vCPU compute-optimized pod

The per-vCPU CPU rate is shown only in the console deploy screen and is
**UNVERIFIED** from public docs. Using an assumed **$0.034 / vCPU / hr**
(compute-optimized tier, in line with RunPod's advertised low-single-cent-per-
vCPU CPU pricing), the arithmetic is:

```
32 vCPU x $0.034 / vCPU / hr        = $1.088 / hr
5 min = 5 / 60 hr                   = 0.0833 hr
compute                            = $1.088 x 0.0833  = $0.0907
container disk (50 GB, 5 min)      = 50 x $0.10/mo x (5/43800) = ~$0.0006  (negligible)
--------------------------------------------------------------
job cost                           ~= $0.09
```

So **~$0.09 for a 5-minute 32-vCPU job**, plus the standing network-volume rent
(~$3.50/mo for 50 GB) that exists regardless of whether the job runs. Re-measure
the real per-vCPU rate on the first paid run and correct this number; if the
console shows a different rate, scale linearly (cost = vCPU x rate x hours).

For reference, a GPU comparison on the same duration: 5 min on an RTX 5090 at
$0.99/hr = $0.99 x 0.0833 = **~$0.083**, i.e. a 5090 minute costs about the same
as a 32-vCPU CPU minute. CPU pods win on longer CPU-only jobs (sharding,
wrecking) where a GPU would sit idle; GPU pods win the moment training starts.

---

## 4. Tooling reference (verified 18 Jul 2026)

### runpodctl
```bash
brew install runpod/runpodctl/runpodctl        # macOS
# or: wget -qO- cli.runpod.net | sudo bash
runpodctl config --apiKey=YOUR_API_KEY
runpodctl pod list
runpodctl pod get <id>
runpodctl pod create --image=<img> --gpu-id=<gpu>   # limited flags, see note
runpodctl pod start <id>
runpodctl pod stop <id>
runpodctl pod delete <id>
```
**Note / UNVERIFIED:** the public README documents only `--image` and `--gpu-id`
on `pod create`. It does not document flags for vCPU count, CPU pods, network
volume attach, ports, env injection, or spot/interruptible. Because our runs need
those (network volume + env-driven JOB_CMD, and CPU pods for sharding),
`deploy.sh` targets the **REST API** for full control and treats runpodctl as a
convenience for list/stop/delete only.

### REST API
- Base URL: **`https://rest.runpod.io/v1`**
- Auth: header `Authorization: Bearer YOUR_API_KEY`
- Create pod: **`POST /pods`**. Body fields used here:
  `imageName`, `gpuTypeIds` (array, priority order), `gpuCount`,
  `computeType` (`GPU`|`CPU`), `vcpuCount` (CPU pods, default 2),
  `containerDiskInGb` (default 50), `volumeInGb` (default 20),
  `networkVolumeId`, `env` (object), `ports` (array, `"port/protocol"`),
  `cloudType` (`SECURE`|`COMMUNITY`), `interruptible` (bool, = spot).
- Terminate pod: **`DELETE /pods/{podId}`**.
- Stop pod: `POST /pods/{podId}/stop` (**UNVERIFIED** exact path; confirm in the
  live API reference before relying on stop-vs-terminate semantics).

---

## 5. What only you (Aden) can do

Scaffolding cannot do any of these; they need your account, your money, or your
sign-off:

1. **Create the restricted API key** (1a) with scopes: **Pods: write/manage**
   (create + stop + terminate) and **Network Volumes: read**. Nothing else.
2. **Set the monthly hard spend cap** (1b).
3. **Create the 50-100 GB network volume** in a DC that has both CPU pods and
   5090s, and record its **volume id** + **datacenter id** (1c).
4. **(Optional) Add the MCP server** (1d) if you want conversational control.
5. **Authorize each deploy** by re-running the `deploy.sh` command with `--yes`.
   Each `--yes` is a fresh, per-instance financial decision.
6. **Delete the network volume** in the console when the project ends (the one
   billable resource that outlives teardown).
