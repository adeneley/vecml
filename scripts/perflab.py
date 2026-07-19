"""Performance lab: measure each remaining throughput lever, solo and stacked.

bench.py answered batch/workers/amp (winner: 8/0/bf16 -> 609 img/s on a 5090).
This measures what is left, one variable at a time, against a trainer-faithful
baseline (loss.item() every step, plain Adam), then re-asks the batch-size
question with the full stack on -- batch 8 only won because the loop was
overhead-bound, and removing the overhead may flip that answer.

Levers:
  sync   read the loss every 50 steps instead of every step (no per-step
         GPU->CPU stall)
  fused  fused Adam (one kernel per step instead of per-tensor launches)
  cl     channels_last memory format on model + image tensors
  comp   torch.compile mode="reduce-overhead" (CUDA-graphs the step)
  duo    two independent trainer processes sharing the card, aggregate img/s

  uv run python scripts/perflab.py --data /tmp/data/train-10k --classes 16 \
      --emit runs/perflab/perflab.json
  uv run python scripts/perflab.py --data data/train-10k --smoke   # local check
"""

import argparse
import json
import os
import subprocess
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


def measure(dataset, device, cfg, min_steps, min_seconds, warmup):
    """One config's steady-state img/s.

    cfg keys: batch, sync, fused, cl, comp; optional push levers:
    nb (non_blocking H2D from pinned memory, CUDA only), cudnn
    (cudnn.benchmark autotuner), sync_n (loss-read window), mode
    (torch.compile mode).
    """
    batch = cfg["batch"]
    n_classes = cfg["n_classes"]
    nb = bool(cfg.get("nb")) and device == "cuda"
    if device == "cuda":
        torch.backends.cudnn.benchmark = bool(cfg.get("cudnn"))
    loader = DataLoader(
        dataset, batch_size=batch, shuffle=True, num_workers=0,
        pin_memory=(device == "cuda"), drop_last=True,
    )
    if len(loader) < 2:
        return None

    if n_classes:
        model = UNet(head=lambda c: DualHead(c, n_classes)).to(device)
    else:
        model = UNet().to(device)
    if cfg["cl"]:
        model = model.to(memory_format=torch.channels_last)
    fused = cfg["fused"] and device == "cuda"
    optim = torch.optim.Adam(model.parameters(), lr=3e-4, fused=fused)
    l1, ce = nn.L1Loss(), nn.CrossEntropyLoss()
    amp_ctx = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if device == "cuda"
        else (lambda: torch.autocast("cpu", enabled=False))
    )
    compile_s = None
    if cfg["comp"]:
        t_c = time.perf_counter()
        model = torch.compile(model, mode=cfg.get("mode", "reduce-overhead"))
    item_every = cfg.get("sync_n", 50) if cfg["sync"] else 1

    def batches():
        while True:
            yield from loader

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    done, imgs, t0 = 0, 0, None
    pending = []
    for batch_data in batches():
        if cfg["cl"]:
            x = batch_data[0].to(device, memory_format=torch.channels_last,
                                 non_blocking=nb)
            y = batch_data[1].to(device, memory_format=torch.channels_last,
                                 non_blocking=nb)
        else:
            x = batch_data[0].to(device, non_blocking=nb)
            y = batch_data[1].to(device, non_blocking=nb)
        lab = batch_data[2].to(device, non_blocking=nb) if n_classes else None
        with amp_ctx():
            out = model(x)
            if n_classes:
                loss = l1(out["rgb"], y) + 0.25 * ce(out["logits"], lab)
            else:
                loss = l1(out, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        # Trainer-faithful loss reads: the baseline pays the sync every step.
        pending.append(loss.detach())
        if len(pending) >= item_every:
            _ = torch.stack(pending).mean().item()
            pending = []
        done += 1
        if done == 1 and cfg["comp"]:
            compile_s = time.perf_counter() - t_c
        if done == warmup:
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        elif done > warmup:
            imgs += x.shape[0]
            if imgs >= min_steps * batch and time.perf_counter() - t0 >= min_seconds:
                break
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak_gb = (
        torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None
    )
    return {"img_s": imgs / dt, "peak_gb": peak_gb, "compile_s": compile_s}


def base_cfg(batch, n_classes):
    return {"batch": batch, "n_classes": n_classes,
            "sync": False, "fused": False, "cl": False, "comp": False}


def duo_worker(args):
    """One of two concurrent processes; filesystem barrier syncs the start."""
    dataset = PairsDataset(args.data, n=args.n, n_classes=args.classes or None,
                           cache_ram=True)
    for i in range(len(dataset)):
        dataset[i]
    cfg = {**base_cfg(args.duo_batch, args.classes or None),
           **json.loads(args.duo_levers)}
    barrier = Path(args.duo_barrier)
    (barrier / f"ready-{args.duo_id}").touch()
    deadline = time.time() + 600
    while len(list(barrier.glob("ready-*"))) < 2:
        if time.time() > deadline:
            raise SystemExit("duo barrier timed out waiting for sibling")
        time.sleep(0.5)
    r = measure(dataset, "cuda", cfg, args.steps, args.seconds, args.warmup)
    Path(args.duo_out).write_text(json.dumps(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--classes", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100,
                    help="min optimizer steps per measurement")
    ap.add_argument("--seconds", type=float, default=4.0,
                    help="min measured seconds per config (SNR floor)")
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--smoke", action="store_true",
                    help="local plumbing check: tiny, skips CUDA-only levers")
    ap.add_argument("--skip-duo", action="store_true",
                    help="skip the two-process test (measured a loser 19 Jul)")
    ap.add_argument("--emit", default=None)
    # internal duo-worker plumbing
    ap.add_argument("--duo-worker", action="store_true")
    ap.add_argument("--duo-id", default="0")
    ap.add_argument("--duo-batch", type=int, default=8)
    ap.add_argument("--duo-levers", default="{}")
    ap.add_argument("--duo-barrier", default="")
    ap.add_argument("--duo-out", default="")
    args = ap.parse_args()

    if args.duo_worker:
        duo_worker(args)
        return

    device = pick_device()
    cuda = device == "cuda"
    if args.smoke:
        args.n, args.steps, args.seconds, args.warmup = 200, 10, 1.0, 3

    print(f"device={device}  pairs={args.n}  classes={args.classes}  "
          f"min {args.steps} steps / {args.seconds}s per config", flush=True)
    dataset = PairsDataset(args.data, n=args.n, n_classes=args.classes or None,
                           cache_ram=True)
    for i in range(len(dataset)):
        dataset[i]
    print(f"cache primed: {len(dataset)} pairs\n", flush=True)

    nc = args.classes or None
    b0 = 4 if args.smoke else 8
    ladder = [
        ("baseline (item/step, plain Adam)", {}),
        ("solo: sync (item/50)", {"sync": True}),
        ("solo: fused Adam", {"fused": True}),
        ("solo: channels_last", {"cl": True}),
        ("solo: compile r-o", {"comp": True}),
        ("stack: sync+fused", {"sync": True, "fused": True}),
        ("stack: +channels_last", {"sync": True, "fused": True, "cl": True}),
        ("stack: compile, no cl", {"sync": True, "fused": True, "comp": True}),
        ("stack: full (+compile)", {"sync": True, "fused": True, "cl": True,
                                    "comp": True}),
    ]
    if not cuda:  # fused Adam and reduce-overhead compile are CUDA-only
        ladder = [(n, lv) for n, lv in ladder
                  if not lv.get("fused") and not lv.get("comp")]

    rows = []
    print(f"{'config':<34} {'batch':>5} {'img/s':>8} {'peak GB':>8} {'compile s':>9}")
    for name, levers in ladder:
        cfg = {**base_cfg(b0, nc), **levers}
        try:
            r = measure(dataset, device, cfg, args.steps, args.seconds,
                        args.warmup)
        except Exception as exc:  # a lever failing is a result, not a crash
            print(f"{name:<34} {b0:>5}    failed: {str(exc)[:80]}", flush=True)
            rows.append({"name": name, "cfg": cfg, "error": str(exc)[:200]})
            continue
        rows.append({"name": name, "cfg": cfg, **r})
        cs = f"{r['compile_s']:.0f}" if r["compile_s"] else "-"
        pk = f"{r['peak_gb']:.1f}" if r["peak_gb"] is not None else "-"
        print(f"{name:<34} {b0:>5} {r['img_s']:>8.1f} {pk:>8} {cs:>9}", flush=True)

    # Batch re-sweep with the full stack: does compile change the batch answer?
    full = {"sync": True, "fused": True, "cl": True, "comp": cuda}
    sweep_batches = [8] if args.smoke else [16, 32, 64]
    for b in sweep_batches:
        cfg = {**base_cfg(b, nc), **full}
        name = f"full stack @ batch {b}"
        try:
            r = measure(dataset, device, cfg, args.steps, args.seconds,
                        args.warmup)
        except Exception as exc:
            print(f"{name:<34} {b:>5}    failed: {str(exc)[:80]}", flush=True)
            rows.append({"name": name, "cfg": cfg, "error": str(exc)[:200]})
            continue
        rows.append({"name": name, "cfg": cfg, **r})
        cs = f"{r['compile_s']:.0f}" if r["compile_s"] else "-"
        pk = f"{r['peak_gb']:.1f}" if r["peak_gb"] is not None else "-"
        print(f"{name:<34} {b:>5} {r['img_s']:>8.1f} {pk:>8} {cs:>9}", flush=True)

    ok = [r for r in rows if "img_s" in r]
    base = next((r for r in ok if r["name"].startswith("baseline")), None)
    best = max(ok, key=lambda r: r["img_s"]) if ok else None

    # Push phase: unexplored levers stacked on the winner so far, chasing
    # the gap between the measured ~880 and the ~3,300 roofline.
    if cuda and not args.smoke and best is not None:
        bb = best["cfg"]["batch"]
        stack = {k: best["cfg"][k] for k in ("sync", "fused", "cl", "comp")}
        push = [
            ("push: +nb", {"nb": True}),
            ("push: +nb +cudnn.benchmark", {"nb": True, "cudnn": True}),
            ("push: +nb, sync 200", {"nb": True, "sync_n": 200}),
            ("push: +nb, max-autotune", {"nb": True, "mode": "max-autotune"}),
            ("push: +nb, no cl", {"nb": True, "cl": False}),
        ]
        print("\npush phase (stacked on winner so far)", flush=True)
        for name, levers in push:
            cfg = {**base_cfg(bb, nc), **stack, **levers}
            try:
                r = measure(dataset, device, cfg, args.steps, args.seconds,
                            args.warmup)
            except Exception as exc:
                print(f"{name:<34} {bb:>5}    failed: {str(exc)[:80]}",
                      flush=True)
                rows.append({"name": name, "cfg": cfg, "error": str(exc)[:200]})
                continue
            rows.append({"name": name, "cfg": cfg, **r})
            cs = f"{r['compile_s']:.0f}" if r["compile_s"] else "-"
            pk = f"{r['peak_gb']:.1f}" if r["peak_gb"] is not None else "-"
            print(f"{name:<34} {bb:>5} {r['img_s']:>8.1f} {pk:>8} {cs:>9}",
                  flush=True)

        # Fine batch sweep around the winner with the best push levers.
        ok = [r for r in rows if "img_s" in r]
        best = max(ok, key=lambda r: r["img_s"])
        for b in (12, 24):
            cfg = {**best["cfg"], "batch": b}
            name = f"push winner @ batch {b}"
            try:
                r = measure(dataset, device, cfg, args.steps, args.seconds,
                            args.warmup)
            except Exception as exc:
                rows.append({"name": name, "cfg": cfg, "error": str(exc)[:200]})
                continue
            rows.append({"name": name, "cfg": cfg, **r})
            print(f"{name:<34} {b:>5} {r['img_s']:>8.1f}", flush=True)

    ok = [r for r in rows if "img_s" in r]
    best = max(ok, key=lambda r: r["img_s"]) if ok else None

    duo = None
    if cuda and not args.smoke and not args.skip_duo and best is not None:
        import tempfile

        print("\nduo: two processes, best solo config each", flush=True)
        with tempfile.TemporaryDirectory() as td:
            levers = {k: best["cfg"][k] for k in ("sync", "fused", "cl", "comp")}
            outs = [Path(td) / f"out-{i}.json" for i in (0, 1)]
            procs = [
                subprocess.Popen([
                    sys.executable, __file__, "--duo-worker",
                    "--data", args.data, "--n", str(args.n),
                    "--classes", str(args.classes),
                    "--steps", str(args.steps), "--seconds", str(args.seconds),
                    "--warmup", str(args.warmup),
                    "--duo-id", str(i), "--duo-batch", str(best["cfg"]["batch"]),
                    "--duo-levers", json.dumps(levers),
                    "--duo-barrier", td, "--duo-out", str(outs[i]),
                ])
                for i in (0, 1)
            ]
            rcs = [p.wait(timeout=1200) for p in procs]
            if all(rc == 0 for rc in rcs):
                parts = [json.loads(o.read_text())["img_s"] for o in outs]
                duo = {"per_proc": [round(p, 1) for p in parts],
                       "aggregate": round(sum(parts), 1)}
                print(f"duo aggregate: {duo['aggregate']:.1f} img/s "
                      f"({parts[0]:.1f} + {parts[1]:.1f})", flush=True)
            else:
                print(f"duo failed: worker rcs {rcs}", flush=True)

    if base and best:
        print(f"\nbaseline: {base['img_s']:.1f} img/s   "
              f"best single: {best['name']} -> {best['img_s']:.1f} img/s "
              f"({best['img_s'] / base['img_s']:.2f}x)", flush=True)

    if args.emit and best:
        winner = {
            "batch_size": best["cfg"]["batch"],
            "num_workers": 0,
            "amp": cuda,
            "sync_every": best["cfg"].get("sync_n", 50) if best["cfg"]["sync"] else 1,
            "fused_adam": best["cfg"]["fused"],
            "channels_last": best["cfg"]["cl"],
            "compile_mode": best["cfg"].get("mode", "reduce-overhead")
            if best["cfg"]["comp"] else None,
            "non_blocking": bool(best["cfg"].get("nb")),
            "cudnn_benchmark": bool(best["cfg"].get("cudnn")),
            "img_s": round(best["img_s"], 1),
            "speedup_vs_faithful_baseline": round(best["img_s"] / base["img_s"], 2)
            if base else None,
            "device": device,
        }
        out = {"rows": [
            {k: v for k, v in r.items() if k != "cfg"} | {"cfg": r["cfg"]}
            for r in rows
        ], "duo": duo, "winner": winner}
        Path(args.emit).parent.mkdir(parents=True, exist_ok=True)
        Path(args.emit).write_text(json.dumps(out, indent=2))
        print(f"results written to {args.emit}", flush=True)


if __name__ == "__main__":
    main()
