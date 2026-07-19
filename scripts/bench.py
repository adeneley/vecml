"""Throughput bench: find the training config that saturates the card.

Sweeps batch size x dataloader workers (x bf16 autocast on CUDA), timing
img/s over a fixed number of optimizer steps after warmup. Run this for ~3
minutes at pod boot before any real training run, then train with the
winner instead of folklore.

  uv run python scripts/bench.py --data data/train-10k
  uv run python scripts/bench.py --data data/train-10k --classes 16 --quick
"""

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vecml.data.pairs import PairsDataset  # noqa: E402
from vecml.models.unet import DualHead, UNet  # noqa: E402
from vecml.train.loop import pick_device  # noqa: E402


def bench_one(dataset, device, batch, workers, amp, n_classes, steps, warmup):
    loader = DataLoader(
        dataset, batch_size=batch, shuffle=True, num_workers=workers,
        pin_memory=(device == "cuda"), persistent_workers=workers > 0,
        drop_last=True,
    )
    if len(loader) < steps + warmup:
        return None  # not enough batches at this size to measure honestly

    if n_classes:
        model = UNet(head=lambda c: DualHead(c, n_classes)).to(device)
    else:
        model = UNet().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    l1, ce = nn.L1Loss(), nn.CrossEntropyLoss()
    amp_ctx = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if amp
        else (lambda: torch.autocast("cpu", enabled=False))
    )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    done, imgs, t0 = 0, 0, None
    for batch_data in loader:
        x = batch_data[0].to(device)
        y = batch_data[1].to(device)
        lab = batch_data[2].to(device) if n_classes else None
        with amp_ctx():
            out = model(x)
            if n_classes:
                loss = l1(out["rgb"], y) + 0.25 * ce(out["logits"], lab)
            else:
                loss = l1(out, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        done += 1
        if done == warmup:
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        elif done > warmup:
            imgs += x.shape[0]
        if done >= warmup + steps:
            break
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak_gb = (
        torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None
    )
    return {"img_s": imgs / dt, "peak_gb": peak_gb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=4000,
                    help="pairs to draw from (kept small; this measures speed, not loss)")
    ap.add_argument("--classes", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="smaller sweep grid")
    ap.add_argument("--emit", default=None,
                    help="write the winning config as JSON here (for watch.py --from-bench)")
    args = ap.parse_args()

    device = pick_device()
    cores = os.cpu_count() or 4
    batches = [8, 32, 64] if args.quick else [8, 16, 32, 64, 128]
    workers = [0, min(4, cores)] if args.quick else sorted({0, 2, min(4, cores), min(8, cores)})
    amps = [False, True] if device == "cuda" else [False]

    print(f"device={device}  cores={cores}  pairs={args.n}  "
          f"steps={args.steps}(+{args.warmup} warmup)  classes={args.classes}")
    dataset = PairsDataset(args.data, n=args.n, n_classes=args.classes or None,
                           cache_ram=True)
    # Prime the RAM cache so decode cost doesn't pollute the first config.
    for i in range(len(dataset)):
        dataset[i]
    print(f"cache primed: {len(dataset)} pairs\n")
    print(f"{'batch':>5} {'workers':>7} {'amp':>4} {'img/s':>8} {'peak GB':>8}")

    rows = []
    for b, w, a in itertools.product(batches, workers, amps):
        try:
            r = bench_one(dataset, device, b, w, a, args.classes,
                          args.steps, args.warmup)
        except RuntimeError as exc:  # OOM at big batches is a result, not a crash
            print(f"{b:>5} {w:>7} {str(a):>4}    failed: {str(exc)[:60]}")
            continue
        if r is None:
            print(f"{b:>5} {w:>7} {str(a):>4}    skipped: dataset too small")
            continue
        rows.append((b, w, a, r))
        peak = f"{r['peak_gb']:.1f}" if r["peak_gb"] is not None else "-"
        print(f"{b:>5} {w:>7} {str(a):>4} {r['img_s']:>8.1f} {peak:>8}")

    if rows:
        base = next((r for r in rows if r[:3] == (8, 0, False)), rows[0])
        best = max(rows, key=lambda r: r[3]["img_s"])
        speedup = best[3]["img_s"] / base[3]["img_s"]
        print(f"\nbaseline (batch 8, workers 0, no amp): {base[3]['img_s']:.1f} img/s")
        print(f"winner: batch {best[0]}, workers {best[1]}, amp {best[2]} "
              f"-> {best[3]['img_s']:.1f} img/s ({speedup:.2f}x)")
        print("reminder: LR is scaled with batch automatically by --from-bench (sqrt rule).")
        if args.emit:
            import json

            winner = {
                "batch_size": best[0],
                "num_workers": best[1],
                "amp": best[2],
                "img_s": round(best[3]["img_s"], 1),
                "speedup": round(speedup, 2),
                "device": device,
            }
            Path(args.emit).parent.mkdir(parents=True, exist_ok=True)
            Path(args.emit).write_text(json.dumps(winner, indent=2))
            print(f"winner written to {args.emit}")


if __name__ == "__main__":
    main()
