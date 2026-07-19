"""Launch the training cockpit.

    uv run python scripts/watch.py --data data/audit-500-v2 --n 100 --size 256 --port 7300

Starts the FastAPI server, prints the URL, opens it in the browser. Press Start
in the page to kick the overfit-on-N run and watch it live.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
import webbrowser

import uvicorn

from vecml.watch.server import create_app


def main() -> None:
    ap = argparse.ArgumentParser(description="vecml training cockpit")
    ap.add_argument("--data", default="data/audit-500-v2", help="sample-dir root")
    ap.add_argument("--n", type=int, default=100, help="overfit sample count")
    ap.add_argument("--size", type=int, default=256, help="square resolution")
    ap.add_argument("--variant", type=int, default=0, help="wrecked variant index")
    ap.add_argument("--batch", type=int, default=8, help="batch size")
    ap.add_argument("--epochs", type=int, default=400, help="max epochs")
    ap.add_argument("--val", type=int, default=0,
                    help="hold out the last N pairs for per-epoch validation")
    ap.add_argument("--classes", type=int, default=0,
                    help="label-map head with K classes (0 = RGB-only)")
    ap.add_argument("--workers", type=int, default=0, help="dataloader workers")
    ap.add_argument("--cache", action="store_true",
                    help="cache decoded images in RAM (skip per-epoch PNG decode)")
    ap.add_argument("--no-amp", action="store_true",
                    help="disable bf16 autocast on CUDA")
    ap.add_argument("--from-bench", default=None,
                    help="JSON from bench.py --emit; overrides batch/workers/amp "
                         "and sqrt-scales LR with the batch size")
    ap.add_argument("--port", type=int, default=7300)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (0.0.0.0 for remote pods behind a proxy)")
    ap.add_argument("--no-open", action="store_true", help="do not open a browser")
    ap.add_argument("--autostart", action="store_true",
                    help="start the run immediately (remote pods)")
    ap.add_argument("--readonly", action="store_true",
                    help="spectator mode: hide run controls, refuse /start & /stop")
    args = ap.parse_args()

    defaults = {
        "data_root": args.data,
        "n": args.n,
        "size": args.size,
        "variant": args.variant,
        "batch_size": args.batch,
        "max_epochs": args.epochs,
        "ckpt_dir": f"runs/overfit{args.n}" + ("-lab" if args.classes else ""),
        "val_n": args.val,
        "n_classes": args.classes,
        "num_workers": args.workers,
        "cache_ram": args.cache,
        "amp": not args.no_amp,
    }
    if args.from_bench:
        import json
        import math

        winner = json.loads(open(args.from_bench).read())
        winner = winner.get("winner", winner)  # perflab.json nests it
        defaults["batch_size"] = winner["batch_size"]
        defaults["num_workers"] = winner["num_workers"]
        defaults["amp"] = winner["amp"]
        for key in ("sync_every", "fused_adam", "channels_last", "compile_mode"):
            if key in winner:
                defaults[key] = winner[key]
        # sqrt scaling (gentler than linear; safer for Adam) vs the batch-8
        # baseline every historical run used.
        defaults["lr"] = 3e-4 * math.sqrt(winner["batch_size"] / 8)
        print(f"bench winner applied: batch {winner['batch_size']}, "
              f"workers {winner['num_workers']}, amp {winner['amp']}, "
              f"compile {defaults.get('compile_mode')}, sync {defaults.get('sync_every', 1)}, "
              f"lr {defaults['lr']:.2e} ({winner.get('img_s', '?')} img/s measured)")
    from pathlib import Path as _P

    title = (f"{_P(defaults['ckpt_dir']).name} · {_P(args.data).name} "
             f"n={args.n} K={args.classes} batch={defaults.get('batch_size', 8)}")
    app = create_app(defaults, readonly=args.readonly, autostart=args.autostart,
                     title=title)

    url = f"http://127.0.0.1:{args.port}"
    print(f"cockpit: {url}  (data={args.data} n={args.n} size={args.size})")
    print("open the page and press Start.")

    if not args.no_open:
        def _open():
            time.sleep(1.2)
            try:
                subprocess.run(["open", url], check=False)
            except Exception:
                webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    sys.exit(main())
