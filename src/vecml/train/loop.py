"""Training loop for the cleanup model, wired for the cockpit dashboard.

Design notes:
- Device auto-pick is mps -> cpu. On this machine MPS `non_blocking=True`
  silently corrupts tensors, so every transfer is a plain `.to(device)`.
  Do not "optimise" that back to non_blocking.
- The trainer never raises out of its worker: every exception in the loop is
  caught and emitted as an {"type": "error"} event, so the dashboard shows the
  traceback instead of the thread dying silently.
- Events are plain JSON-able dicts handed to a sink callable. The contract is
  generic (status / metric / sample / error) so the same trainer can later
  drive a different model without the UI changing.
"""

from __future__ import annotations

import base64
import io
import subprocess
import threading
import time
import traceback
from contextlib import nullcontext
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from vecml.data.pairs import PairsDataset
from vecml.models.unet import DualHead, UNet, count_params

# Fixed distinct colours for label-map thumbnails (slot 0 = background).
# Deliberately NOT the image's own palette: slot-splitting mistakes are
# invisible if two slots happen to render in similar colours.
LABEL_COLOURS = np.array(
    [
        [255, 255, 255], [17, 17, 17], [214, 39, 40], [44, 160, 44],
        [31, 119, 180], [255, 127, 14], [148, 103, 189], [23, 190, 207],
        [227, 119, 194], [188, 189, 34], [140, 86, 75], [127, 127, 127],
        [174, 199, 232], [255, 152, 150], [152, 223, 138], [197, 176, 213],
    ],
    dtype=np.uint8,
)

EventSink = Callable[[dict], None]


class GpuMonitor(threading.Thread):
    """Background nvidia-smi sampler; latest util%/VRAM exposed as attrs.

    Subprocess polling (not pynvml) to stay dependency-free; at one sample
    every couple of seconds the fork cost is irrelevant.
    """

    def __init__(self, interval_s: float = 2.0):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.util: float | None = None
        self.mem_gb: float | None = None
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip().splitlines()[0]
                util, mem = out.split(",")
                self.util = float(util)
                self.mem_gb = round(float(mem) / 1024, 1)
            except Exception:
                self.util = self.mem_gb = None
            self._stop.wait(self.interval_s)


def pick_device() -> str:
    """cuda -> mps -> cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _tensor_to_b64_png(chw: torch.Tensor, size: int) -> str:
    """Encode a CHW float[0,1] tensor as a base64 PNG thumbnail (size x size)."""
    arr = (chw.clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy())
    arr = np.transpose(arr, (1, 2, 0))  # HWC
    img = Image.fromarray(arr, mode="RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _labels_to_b64_png(idx_hw: torch.Tensor, size: int) -> str:
    """Encode an HW int label map as a base64 PNG via the fixed colour table."""
    arr = idx_hw.cpu().numpy().clip(0, len(LABEL_COLOURS) - 1)
    img = Image.fromarray(LABEL_COLOURS[arr], mode="RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@dataclass
class TrainConfig:
    """Overfit-on-N defaults; every field is plumbed from the CLI/cockpit."""

    data_root: str
    n: int = 100
    size: int = 256
    variant: int = 0
    batch_size: int = 8
    lr: float = 3e-4
    max_epochs: int = 400
    target_loss: float = 0.003
    # Display cadence is wall-time based so it never depends on training speed
    # (a fast local run and a slow remote one should feel the same in the UI).
    metric_interval_s: float = 0.25
    sample_interval_s: float = 1.0
    sample_size: int = 192
    num_workers: int = 0
    ckpt_dir: str = "runs/overfit100"
    seed: int = 0
    # Held-out validation: the LAST val_n of the selected dirs never train;
    # a val pass runs each epoch and val_mean rides the epoch_end metric.
    val_n: int = 0
    # Label-map head: 0 = RGB-only (v0 behaviour). When set, the model grows
    # a K-way classification head, pairs with >K palette colours are dropped,
    # and the loss becomes L1(rgb) + label_loss_w * CrossEntropy(labels).
    # The chart/target-loss "loss" stays the L1 component so runs remain
    # comparable across modes.
    n_classes: int = 0
    label_loss_w: float = 0.25
    # Throughput levers (see scripts/bench.py for finding the right values):
    # bf16 autocast on CUDA, pinned host memory, in-RAM decoded-image cache.
    amp: bool = True
    pin_memory: bool = True
    cache_ram: bool = False
    extra: dict = field(default_factory=dict)


class Trainer:
    """Runs an overfit/training loop, emitting cockpit events as it goes."""

    def __init__(self, cfg: TrainConfig, sink: EventSink):
        self.cfg = cfg
        self.sink = sink
        self._stop = threading.Event()
        self.device = pick_device()
        self.best_loss = float("inf")
        self.params = 0

    # -- external control -------------------------------------------------
    def stop(self) -> None:
        """Request the loop to halt at the next step boundary."""
        self._stop.set()

    def _emit(self, event: dict) -> None:
        try:
            self.sink(event)
        except Exception:  # a broken sink must never kill training
            pass

    # -- the loop ---------------------------------------------------------
    def run(self) -> None:
        """Entry point for the worker thread. Catches everything."""
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all
            self._emit(
                {
                    "type": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            self._emit({"type": "status", "state": "error", "detail": str(exc)})

    def _run(self) -> None:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)

        self._emit(
            {
                "type": "status",
                "state": "running",
                "detail": f"device={self.device}",
            }
        )

        dataset = PairsDataset(
            cfg.data_root,
            n=cfg.n,
            size=cfg.size,
            variant=cfg.variant,
            n_classes=cfg.n_classes or None,
            cache_ram=cfg.cache_ram,
        )
        val_n = min(cfg.val_n, max(len(dataset) - 1, 0))
        if val_n > 0:
            from torch.utils.data import Subset

            train_set = Subset(dataset, range(len(dataset) - val_n))
            val_set = Subset(dataset, range(len(dataset) - val_n, len(dataset)))
            val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
        else:
            train_set, val_set, val_loader = dataset, None, None
        loader = DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and self.device == "cuda",
            persistent_workers=cfg.num_workers > 0,
            drop_last=False,
        )
        use_amp = cfg.amp and self.device == "cuda"
        # bf16 needs no GradScaler; MPS stays fp32 (past burns earn caution).
        amp_ctx = (
            (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
            if use_amp
            else nullcontext
        )
        gpu_mon = None
        if self.device == "cuda":
            gpu_mon = GpuMonitor()
            gpu_mon.start()

        if cfg.n_classes:
            model = UNet(head=lambda c: DualHead(c, cfg.n_classes)).to(self.device)
        else:
            model = UNet().to(self.device)
        self.params = count_params(model)
        optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        # Cosine decay from lr down to ~lr/20 over the whole run, stepped once
        # per epoch. Fixes the constant-LR bounce observed near the loss floor.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max(cfg.max_epochs, 1), eta_min=cfg.lr / 20.0
        )
        loss_fn = nn.L1Loss()
        ce_fn = nn.CrossEntropyLoss() if cfg.n_classes else None

        # Fixed items for the live sample views: drawn from the held-out split
        # when one exists (so the 4-up shows images the model never trains on).
        pool = val_set if val_set is not None else dataset
        n_show = min(4, len(pool))
        idxs = sorted({round(i * (len(pool) - 1) / max(n_show - 1, 1)) for i in range(n_show)})
        val_items = [pool[i] for i in idxs]

        ckpt_dir = Path(cfg.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        window_start = start
        window_imgs = 0
        global_step = 0
        stopped = False
        # Wall-clock gates for display cadence (independent of step count).
        last_metric_t = 0.0
        last_sample_t = 0.0

        for epoch in range(cfg.max_epochs):
            epoch_loss_sum = 0.0
            epoch_items = 0
            model.train()

            for batch in loader:
                if self._stop.is_set():
                    stopped = True
                    break

                x = batch[0].to(self.device)  # plain transfer, never non_blocking
                y = batch[1].to(self.device)
                lab = batch[2].to(self.device) if ce_fn is not None else None

                ce_val = None
                with amp_ctx():
                    out = model(x)
                    if ce_fn is not None:
                        l1 = loss_fn(out["rgb"], y)
                        ce = ce_fn(out["logits"], lab)
                        loss = l1 + cfg.label_loss_w * ce
                        ce_val = float(ce.detach().to("cpu").item())
                    else:
                        l1 = loss = loss_fn(out, y)

                optim.zero_grad()
                loss.backward()
                optim.step()

                global_step += 1
                bs = x.shape[0]
                window_imgs += bs
                # The chart/target "loss" is always the L1 component so label
                # and RGB-only runs stay comparable on the same axis.
                loss_val = float(l1.detach().to("cpu").item())
                epoch_loss_sum += loss_val * bs
                epoch_items += bs

                if loss_val < self.best_loss:
                    self.best_loss = loss_val
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "step": global_step,
                            "epoch": epoch,
                            "loss": loss_val,
                            "params": self.params,
                            "n_classes": cfg.n_classes,
                        },
                        ckpt_dir / "best.pt",
                    )

                now = time.perf_counter()
                first = global_step == 1  # wake the UI on the very first step
                if first or (now - last_metric_t) >= cfg.metric_interval_s:
                    dt = max(now - window_start, 1e-9)
                    self._emit(
                        {
                            "type": "metric",
                            "step": global_step,
                            "epoch": epoch,
                            "loss": loss_val,
                            "epoch_mean": epoch_loss_sum / max(epoch_items, 1),
                            "loss_ce": ce_val,
                            "lr": optim.param_groups[0]["lr"],
                            "img_per_s": window_imgs / dt,
                            "elapsed_s": now - start,
                            "gpu_util": gpu_mon.util if gpu_mon else None,
                            "gpu_mem_gb": gpu_mon.mem_gb if gpu_mon else None,
                        }
                    )
                    window_start = now
                    window_imgs = 0
                    last_metric_t = now

                if first or (now - last_sample_t) >= cfg.sample_interval_s:
                    self._emit_sample(model, val_items, global_step)
                    last_sample_t = now

            if stopped:
                break

            epoch_mean = epoch_loss_sum / max(epoch_items, 1)
            val_mean = None
            val_acc = None
            if val_loader is not None:
                model.eval()
                v_sum, v_items = 0.0, 0
                v_correct, v_pix = 0, 0
                with torch.no_grad():
                    for vbatch in val_loader:
                        vx = vbatch[0].to(self.device)  # plain transfer
                        vy = vbatch[1].to(self.device)
                        vout = model(vx)
                        if ce_fn is not None:
                            v_loss = loss_fn(vout["rgb"], vy)
                            vlab = vbatch[2].to(self.device)
                            v_correct += int((vout["logits"].argmax(1) == vlab).sum().item())
                            v_pix += int(vlab.numel())
                        else:
                            v_loss = loss_fn(vout, vy)
                        v_sum += float(v_loss.item()) * vx.shape[0]
                        v_items += vx.shape[0]
                model.train()
                val_mean = v_sum / max(v_items, 1)
                val_acc = v_correct / v_pix if v_pix else None
            # Final per-epoch mean point (the exit-criterion line on the chart).
            self._emit(
                {
                    "type": "metric",
                    "step": global_step,
                    "epoch": epoch,
                    "loss": loss_val,
                    "epoch_mean": epoch_mean,
                    "val_mean": val_mean,
                    "val_acc": val_acc,
                    "epoch_end": True,
                    "lr": optim.param_groups[0]["lr"],
                    "elapsed_s": time.perf_counter() - start,
                }
            )
            scheduler.step()  # cosine LR decay, once per epoch
            if epoch_mean < cfg.target_loss:
                self._emit(
                    {
                        "type": "status",
                        "state": "running",
                        "detail": f"target loss reached at epoch {epoch}"
                        f" (mean {epoch_mean:.5f})",
                    }
                )
                break

        if gpu_mon is not None:
            gpu_mon.stop()
        if stopped:
            self._emit({"type": "status", "state": "stopped", "detail": "user stop"})
        else:
            self._emit(
                {
                    "type": "status",
                    "state": "finished",
                    "detail": f"best loss {self.best_loss:.5f}, "
                    f"{global_step} steps",
                }
            )

    def _emit_sample(
        self,
        model: nn.Module,
        val_items: list[tuple[torch.Tensor, ...]],
        step: int,
    ) -> None:
        """Run the fixed val items (one batched forward) and emit thumbnails.

        Event stays additive: input/pred/target are item 0 (the original
        single-triptych contract); `items` carries every val item for the
        multi-sample view; label-mode items additionally carry
        pred_labels/target_labels colourised via the fixed table.
        """
        was_training = model.training
        model.eval()
        with torch.no_grad():
            xs = torch.stack([it[0] for it in val_items]).to(self.device)  # plain transfer
            out = model(xs)
            if isinstance(out, dict):
                preds = out["rgb"].to("cpu")  # plain transfer back
                pred_labs = out["logits"].argmax(1).to("cpu")
            else:
                preds = out.to("cpu")  # plain transfer back
                pred_labs = None
        if was_training:
            model.train()
        size = self.cfg.sample_size
        items = []
        for i, it in enumerate(val_items):
            entry = {
                "input": _tensor_to_b64_png(it[0], size),
                "pred": _tensor_to_b64_png(preds[i], size),
                "target": _tensor_to_b64_png(it[1], size),
            }
            if pred_labs is not None:
                entry["pred_labels"] = _labels_to_b64_png(pred_labs[i], size)
                entry["target_labels"] = _labels_to_b64_png(it[2], size)
            items.append(entry)
        self._emit({"type": "sample", "step": step, **items[0], "items": items})
