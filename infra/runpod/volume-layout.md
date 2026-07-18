# Network-volume layout (vecml)

One network volume, mounted at `/workspace` (`VOL_ROOT`) in every pod, reused
across runs. It is the only persistent, always-billed resource (see README
billing section). Size 50 GB to start, 100 GB once 512px shards + rolling
checkpoints live on it.

```
/workspace
├── repo/           # git checkout, updated in place each run by startup.sh
├── shards/         # WebDataset .tar shards (wrecked (input,label) pairs)
│   ├── 256/            train-000000.tar ... (prototype resolution)
│   └── 512/            train-000000.tar ... (serious resolution)
├── checkpoints/    # model .pt/.ckpt, one subdir per run
│   └── <run-name>/     epoch-*.pt, last.pt, best.pt
├── runs-logs/      # per-run console + JSONL event logs (see cockpit-remote.md)
│   └── <run-name>/     console.log, events.jsonl, result.json
└── corpus-cache/   # HF datasets cache + rendered clean SVGs (pre-wreck source)
```

Rationale:
- **`shards/` is the point of the whole volume.** Pre-wrecked WebDataset tars
  are built offline (never augment on the fly on a rented GPU, per CLAUDE.md) and
  staged here once, then streamed by every training run in the datacenter at
  local-disk speed instead of over the network from the Mac.
- **`checkpoints/` and `runs-logs/` survive teardown**, so a terminated pod loses
  nothing. A re-deployed pod resumes from `last.pt` and appends to the same log
  tree.
- **`corpus-cache/` holds the HF pull.** svg-stack and the other CC0 corpora are
  fetched **from Hugging Face directly into this volume at datacenter bandwidth**,
  not uploaded from the Mac. The Mac only ever holds seeds and code; the heavy
  raw corpus lands here on the first CPU pod that runs the fetch/render job.
- `repo/` lives on the volume too so `uv sync` and the checkout persist between
  runs; only a dependency change forces an image rebuild, not a re-clone.

Sizing sketch (refine after the first real shard build):
- 256px shards, 200k pairs: low tens of GB depending on encoding.
- A handful of checkpoints for a 30-100M param model in BF16: ~0.06-0.2 GB each.
- Logs are trivial. So 50 GB comfortably holds a prototype cycle; go to 100 GB
  when 512px shards (1-2M pairs) arrive.
