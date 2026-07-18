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
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from vecml.data.pairs import PairsDataset
from vecml.models.unet import UNet, count_params

EventSink = Callable[[dict], None]


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
            cfg.data_root, n=cfg.n, size=cfg.size, variant=cfg.variant
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=False,
        )

        model = UNet().to(self.device)
        self.params = count_params(model)
        optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        # Cosine decay from lr down to ~lr/20 over the whole run, stepped once
        # per epoch. Fixes the constant-LR bounce observed near the loss floor.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max(cfg.max_epochs, 1), eta_min=cfg.lr / 20.0
        )
        loss_fn = nn.L1Loss()

        # Fixed validation items (evenly spaced across the set) for the live
        # sample views. Kept on CPU and moved per-sample with plain .to(device).
        n_val = min(4, len(dataset))
        idxs = sorted({round(i * (len(dataset) - 1) / max(n_val - 1, 1)) for i in range(n_val)})
        val_items = [dataset[i] for i in idxs]

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

            for x, y in loader:
                if self._stop.is_set():
                    stopped = True
                    break

                x = x.to(self.device)  # plain transfer, never non_blocking
                y = y.to(self.device)

                pred = model(x)
                loss = loss_fn(pred, y)

                optim.zero_grad()
                loss.backward()
                optim.step()

                global_step += 1
                bs = x.shape[0]
                window_imgs += bs
                loss_val = float(loss.detach().to("cpu").item())
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
                            "lr": optim.param_groups[0]["lr"],
                            "img_per_s": window_imgs / dt,
                            "elapsed_s": now - start,
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
            # Final per-epoch mean point (the exit-criterion line on the chart).
            self._emit(
                {
                    "type": "metric",
                    "step": global_step,
                    "epoch": epoch,
                    "loss": loss_val,
                    "epoch_mean": epoch_mean,
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
        val_items: list[tuple[torch.Tensor, torch.Tensor]],
        step: int,
    ) -> None:
        """Run the fixed val items (one batched forward) and emit thumbnails.

        Event stays additive: input/pred/target are item 0 (the original
        single-triptych contract); `items` carries every val item for the
        multi-sample view.
        """
        was_training = model.training
        model.eval()
        with torch.no_grad():
            xs = torch.stack([x for x, _ in val_items]).to(self.device)  # plain transfer
            preds = model(xs).to("cpu")  # plain transfer back
        if was_training:
            model.train()
        size = self.cfg.sample_size
        items = [
            {
                "input": _tensor_to_b64_png(x, size),
                "pred": _tensor_to_b64_png(preds[i], size),
                "target": _tensor_to_b64_png(y, size),
            }
            for i, (x, y) in enumerate(val_items)
        ]
        self._emit({"type": "sample", "step": step, **items[0], "items": items})
