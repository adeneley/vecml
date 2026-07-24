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
import random
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


def _label_boundaries(lab: torch.Tensor) -> torch.Tensor:
    """Bool (B,H,W): pixels adjacent to any label change (both sides)."""
    b = torch.zeros_like(lab, dtype=torch.bool)
    d = lab[:, 1:, :] != lab[:, :-1, :]
    b[:, 1:, :] |= d
    b[:, :-1, :] |= d
    d = lab[:, :, 1:] != lab[:, :, :-1]
    b[:, :, 1:] |= d
    b[:, :, :-1] |= d
    return b


def _dilate3(b: torch.Tensor) -> torch.Tensor:
    """One 3x3 binary dilation (B,H,W bool)."""
    f = torch.nn.functional.max_pool2d(b.float().unsqueeze(1), 3, 1, 1)
    return f.squeeze(1) > 0


def _boundary_f1_parts(pred: torch.Tensor, gt: torch.Tensor):
    """Boundary-F1 components for int label maps (B,H,W), 1px tolerance.

    Overall pixel accuracy saturates in the mid-90s while nearly all label
    errors sit ON region boundaries, so accuracy cannot rank label heads.
    Boundary F1 can: precision = predicted-boundary pixels within 1px of a
    true boundary, recall = true-boundary pixels within 1px of a predicted
    one. Returns raw sums so batches accumulate exactly.
    """
    bmask, dilate = _label_boundaries, _dilate3
    pb, gb = bmask(pred), bmask(gt)
    return (
        int((pb & dilate(gb)).sum().item()), int(pb.sum().item()),
        int((gb & dilate(pb)).sum().item()), int(gb.sum().item()),
    )


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
    # E2: boundary-weighted CE. When >1, cross-entropy switches to a per-pixel
    # weighted mean where pixels within 1px of a GT label boundary count this
    # many times. Motivation: label errors concentrate ON boundaries (the
    # val_bf story) while plain CE weights the ~95% interior pixels equally.
    boundary_w: float = 1.0
    # E3 stabilizers. ema_decay > 0 keeps an exponential moving average of the
    # weights (0.999 typical); val + checkpoints then use the EMA weights, so
    # best.pt ships what we'd deploy. warmup_epochs prepends a linear LR ramp
    # (start 0.1x) before the cosine decay - the 500k divergence started in
    # epoch 1 at full LR under bf16.
    ema_decay: float = 0.0
    warmup_epochs: int = 0
    # E4/E8: "" = pre-baked wrecked_XX.png pairs (PairsDataset, v0 behaviour);
    # "v2-otf" = on-the-fly wreck-v2 damage synthesized per item per epoch
    # from clean.png (WreckOnTheFly). The val slice gets its own frozen
    # instance so val_mean stays comparable across epochs.
    wreck: str = ""
    # UNet width: channel count of the first encoder stage (32 -> ~1.9M params,
    # 48 -> ~4.2M, 64 -> ~7.5M). Params scale ~quadratically with base.
    base: int = 32
    # Throughput levers (see scripts/bench.py for finding the right values):
    # bf16 autocast on CUDA, pinned host memory, in-RAM decoded-image cache.
    amp: bool = True
    pin_memory: bool = True
    cache_ram: bool = False
    # Second-round levers, measured by scripts/perflab.py (19 Jul 2026, 5090:
    # trainer-faithful baseline 433 img/s -> full stack 880). All default off;
    # CUDA-only ones are ignored elsewhere.
    #   sync_every    read losses off the GPU every N steps, not every step
    #                 (the per-step .item() stall alone cost ~27%)
    #   fused_adam    single-kernel Adam update
    #   channels_last NHWC memory format (hurts eager, helps under compile)
    #   compile_mode  torch.compile mode for the train step, e.g.
    #                 "reduce-overhead" (~12-18s one-off compile at start)
    sync_every: int = 1
    fused_adam: bool = False
    channels_last: bool = False
    compile_mode: str | None = None
    # Both CUDA-only, ignored elsewhere. non_blocking must NEVER be enabled
    # on MPS: it silently corrupts transferred tensors there (starter-model
    # lesson); on CUDA with pinned memory it overlaps copy with compute.
    non_blocking: bool = False
    cudnn_benchmark: bool = False
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
                # Exit conditions, so the cockpit can show how/when the run
                # will stop (first one hit wins).
                "max_epochs": cfg.max_epochs,
                "target_loss": cfg.target_loss,
            }
        )

        otf = cfg.wreck == "v2-otf"
        if otf:
            from vecml.data.wreck_otf import WreckOnTheFly

            dataset = WreckOnTheFly(
                cfg.data_root,
                n=cfg.n,
                size=cfg.size,
                n_classes=cfg.n_classes or None,
                cache_ram=cfg.cache_ram,
                seed=cfg.seed,
            )
            # Frozen twin over the same dirs: the val slice always sees
            # epoch-0 damage, so val_mean is comparable across epochs.
            val_source = WreckOnTheFly(
                cfg.data_root,
                n=cfg.n,
                size=cfg.size,
                n_classes=cfg.n_classes or None,
                cache_ram=cfg.cache_ram,
                seed=cfg.seed,
                freeze=True,
            )
        else:
            dataset = PairsDataset(
                cfg.data_root,
                n=cfg.n,
                size=cfg.size,
                variant=cfg.variant,
                n_classes=cfg.n_classes or None,
                cache_ram=cfg.cache_ram,
            )
            val_source = dataset
        val_n = min(cfg.val_n, max(len(dataset) - 1, 0))
        if val_n > 0:
            from torch.utils.data import Subset

            train_set = Subset(dataset, range(len(dataset) - val_n))
            val_set = Subset(val_source, range(len(dataset) - val_n, len(dataset)))
            val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
        else:
            train_set, val_set, val_loader = dataset, None, None
        loader = DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and self.device == "cuda",
            # OTF workers must re-fork each epoch to pick up set_epoch;
            # persistent workers would keep epoch-0 damage forever.
            persistent_workers=cfg.num_workers > 0 and not otf,
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
            model = UNet(base=cfg.base, head=lambda c: DualHead(c, cfg.n_classes)).to(self.device)
        else:
            model = UNet(base=cfg.base).to(self.device)
        cuda = self.device == "cuda"
        use_cl = cfg.channels_last and cuda
        nb = cfg.non_blocking and cuda
        if cuda and cfg.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        if use_cl:
            model = model.to(memory_format=torch.channels_last)
        # `model` stays the plain module for checkpoints, val and samples;
        # `net` (maybe compiled) is used for the train step only, so eval-time
        # shape changes never touch the CUDA-graph cache.
        net = model
        if cfg.compile_mode and cuda:
            net = torch.compile(model, mode=cfg.compile_mode)
        self.params = count_params(model)
        optim = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, fused=cfg.fused_adam and cuda
        )
        # Cosine decay from lr down to ~lr/20 over the whole run, stepped once
        # per epoch. Fixes the constant-LR bounce observed near the loss floor.
        # warmup_epochs (E3) prepends a linear 0.1x -> 1x ramp.
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max(cfg.max_epochs - cfg.warmup_epochs, 1),
            eta_min=cfg.lr / 20.0,
        )
        if cfg.warmup_epochs > 0:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optim,
                [torch.optim.lr_scheduler.LinearLR(
                    optim, start_factor=0.1, total_iters=cfg.warmup_epochs),
                 cosine],
                milestones=[cfg.warmup_epochs],
            )
        else:
            scheduler = cosine
        loss_fn = nn.L1Loss()
        # reduction="none" in boundary mode: the weighted mean is taken by
        # hand in the step so boundary pixels can count boundary_w times.
        ce_fn = None
        if cfg.n_classes:
            ce_fn = nn.CrossEntropyLoss(
                reduction="none" if cfg.boundary_w > 1.0 else "mean")

        # E3: EMA shadow of every state tensor. Pairs are hoisted once -
        # state_dict() tensor refs stay valid because the optimizer and the
        # val-time load_state_dict both mutate in place.
        ema_sd = None
        ema_pairs = None
        if cfg.ema_decay > 0.0:
            ema_sd = {k: v.detach().clone()
                      for k, v in model.state_dict().items()}
            ema_pairs = [(ema_sd[k], v)
                         for k, v in model.state_dict().items()]

        # Live sample views, drawn from the held-out split when one exists
        # (so the 4-up shows images the model never trains on). Item 0 is
        # FIXED for the whole run - the single triptych tracks one image's
        # progress - while the other three are re-drawn at random from the
        # pool on every broadcast, so the 4-up rotates through the val set.
        pool = val_set if val_set is not None else dataset
        anchor = pool[0]

        def draw_sample_items() -> list:
            k = min(3, max(len(pool) - 1, 0))
            others = random.sample(range(1, len(pool)), k) if k else []
            return [anchor] + [pool[i] for i in others]

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

        # Loss reads are batched: sums stay on the GPU and are synced to
        # Python every cfg.sync_every steps (and at epoch end). loss_val /
        # ce_val hold the latest synced window means between flushes; the
        # best-checkpoint check rides the same cadence, which also throttles
        # early-run save churn. With sync_every=1 this is exactly the old
        # per-step behaviour.
        epoch_loss_sum = 0.0
        epoch_items = 0
        pend_l1 = None
        pend_ce = None
        pend_items = 0
        pend_steps = 0
        loss_val = float("inf")
        ce_val = None

        def flush_pending() -> None:
            nonlocal pend_l1, pend_ce, pend_items, pend_steps
            nonlocal loss_val, ce_val, epoch_loss_sum, epoch_items
            if pend_steps == 0:
                return
            l1_sum = float(pend_l1.item())
            epoch_loss_sum += l1_sum
            epoch_items += pend_items
            # The chart/target "loss" is always the L1 component so label
            # and RGB-only runs stay comparable on the same axis.
            loss_val = l1_sum / pend_items
            if pend_ce is not None:
                ce_val = float(pend_ce.item()) / pend_items
            pend_l1, pend_ce, pend_items, pend_steps = None, None, 0, 0

        for epoch in range(cfg.max_epochs):
            if otf:
                dataset.set_epoch(epoch)  # fresh damage; workers fork after this
            epoch_loss_sum = 0.0
            epoch_items = 0
            model.train()

            for batch in loader:
                if self._stop.is_set():
                    stopped = True
                    break

                x = batch[0].to(self.device, non_blocking=nb)
                y = batch[1].to(self.device, non_blocking=nb)
                if use_cl:
                    x = x.contiguous(memory_format=torch.channels_last)
                    y = y.contiguous(memory_format=torch.channels_last)
                lab = (batch[2].to(self.device, non_blocking=nb)
                       if ce_fn is not None else None)

                with amp_ctx():
                    out = net(x)
                    if ce_fn is not None:
                        l1 = loss_fn(out["rgb"], y)
                        if cfg.boundary_w > 1.0:
                            # E2: GT-boundary band (1px dilated) weighted
                            # boundary_w:1 against the interior. loss_ce on
                            # the chart is this weighted mean - comparable
                            # within an arm, not across boundary_w values.
                            ce_map = ce_fn(out["logits"], lab)
                            w = torch.where(
                                _dilate3(_label_boundaries(lab)),
                                cfg.boundary_w, 1.0)
                            ce = (ce_map * w).sum() / w.sum()
                        else:
                            ce = ce_fn(out["logits"], lab)
                        loss = l1 + cfg.label_loss_w * ce
                    else:
                        l1 = loss = loss_fn(out, y)

                optim.zero_grad()
                loss.backward()
                # Seatbelt against one-off divergence (hippo48-500k, 19 Jul):
                # a single pathological step under bf16 can kick the weights
                # somewhere training never recovers from. Norm-clipping bounds
                # that step; at 1.0 it is inert on healthy gradients.
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optim.step()
                if ema_pairs is not None:
                    with torch.no_grad():
                        for e, v in ema_pairs:
                            if torch.is_floating_point(e):
                                e.lerp_(v, 1.0 - cfg.ema_decay)
                            else:
                                e.copy_(v)

                global_step += 1
                bs = x.shape[0]
                window_imgs += bs
                # Accumulate on-GPU immediately: under reduce-overhead the
                # loss tensor's buffer is reused by the next graph replay, so
                # the add must be enqueued now (same stream = safe ordering).
                d1 = l1.detach() * bs
                pend_l1 = d1 if pend_l1 is None else pend_l1 + d1
                if ce_fn is not None:
                    dc = ce.detach() * bs
                    pend_ce = dc if pend_ce is None else pend_ce + dc
                pend_items += bs
                pend_steps += 1

                if pend_steps >= cfg.sync_every:
                    flush_pending()

                now = time.perf_counter()
                first = global_step == 1  # wake the UI on the very first step
                if first:
                    flush_pending()  # never show the UI a pre-sync sentinel
                if first or (now - last_metric_t) >= cfg.metric_interval_s:
                    dt = max(now - window_start, 1e-9)
                    self._emit(
                        {
                            "type": "metric",
                            "step": global_step,
                            "epoch": epoch,
                            "loss": loss_val,
                            # Early in an epoch nothing has synced yet; show the
                            # last window mean instead of a bogus 0.0.
                            "epoch_mean": (epoch_loss_sum / epoch_items) if epoch_items else loss_val,
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
                    self._emit_sample(model, draw_sample_items(), global_step)
                    last_sample_t = now

            if stopped:
                break

            flush_pending()  # drain the tail window into the epoch sums
            epoch_mean = epoch_loss_sum / max(epoch_items, 1)
            # E3: val + checkpoints judge the EMA weights (what we'd ship);
            # raw weights are restored after saving so training continues
            # unperturbed. In-place load_state_dict keeps tensor storages
            # stable for the compiled train step and the hoisted ema_pairs.
            raw_sd = None
            if ema_sd is not None:
                raw_sd = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
                model.load_state_dict(ema_sd)
            val_mean = None
            val_acc = None
            val_bf = None
            if val_loader is not None:
                model.eval()
                v_sum, v_items = 0.0, 0
                v_correct, v_pix = 0, 0
                v_bf = [0, 0, 0, 0]  # tp_precision, n_pred_boundary, tp_recall, n_gt_boundary
                with torch.no_grad():
                    for vbatch in val_loader:
                        vx = vbatch[0].to(self.device)  # plain transfer
                        vy = vbatch[1].to(self.device)
                        vout = model(vx)
                        if ce_fn is not None:
                            v_loss = loss_fn(vout["rgb"], vy)
                            vlab = vbatch[2].to(self.device)
                            vpred = vout["logits"].argmax(1)
                            v_correct += int((vpred == vlab).sum().item())
                            v_pix += int(vlab.numel())
                            for j, part in enumerate(_boundary_f1_parts(vpred, vlab)):
                                v_bf[j] += part
                        else:
                            v_loss = loss_fn(vout, vy)
                        v_sum += float(v_loss.item()) * vx.shape[0]
                        v_items += vx.shape[0]
                model.train()
                val_mean = v_sum / max(v_items, 1)
                val_acc = v_correct / v_pix if v_pix else None
                if v_bf[1] and v_bf[3]:
                    v_p, v_r = v_bf[0] / v_bf[1], v_bf[2] / v_bf[3]
                    val_bf = 2 * v_p * v_r / (v_p + v_r) if (v_p + v_r) else 0.0
                else:
                    val_bf = None
            # Checkpoint policy: best is judged once per epoch on the held-out
            # mean (epoch mean when no val split), never on a lucky single
            # batch - a per-step minimum once froze best.pt at epoch 162 of a
            # 300-epoch run. last.pt always tracks the newest weights.
            criterion = val_mean if val_mean is not None else epoch_mean
            payload = {
                "model": model.state_dict(),
                "step": global_step,
                "epoch": epoch,
                "loss": criterion,
                "val_mean": val_mean,
                "val_acc": val_acc,
                "val_bf": val_bf,
                "params": self.params,
                "n_classes": cfg.n_classes,
                "boundary_w": cfg.boundary_w,
                "ema_decay": cfg.ema_decay,
            }
            torch.save(payload, ckpt_dir / "last.pt")
            if criterion < self.best_loss:
                self.best_loss = criterion
                torch.save(payload, ckpt_dir / "best.pt")
            if raw_sd is not None:
                model.load_state_dict(raw_sd)
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
                    "val_bf": val_bf,
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
