"""FastAPI cockpit server: Start a run, watch it live over SSE.

The trainer runs in a background thread and publishes JSON events onto an
EventBus. The /events endpoint streams a small history replay then live events
to each connected browser. Everything the trainer emits is generic, so this
server does not know or care what model is training.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from collections import deque
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from vecml.train.loop import TrainConfig, Trainer
from vecml.watch import infer as infer_mod

STATIC = Path(__file__).parent / "static"


class EventBus:
    """Thread-safe fan-out of JSON events with typed replay retention.

    Late joiners must be able to reconstruct the whole run, so retention is
    by event type rather than a single rolling window:
      - status / error / epoch-end metrics: pinned for the run's lifetime
        (bounded by epoch count, tiny)
      - step metrics: large rolling window (small dicts, cheap)
      - samples: only the latest (they carry big base64 PNGs)
    """

    def __init__(self, detail: int = 50_000, pinned_cap: int = 5_000):
        self._lock = threading.Lock()
        self._seq = 0
        self._pinned: deque[dict] = deque(maxlen=pinned_cap)
        self._detail: deque[dict] = deque(maxlen=detail)
        self._last_sample: dict | None = None
        self._subs: list[queue.Queue] = []

    def publish(self, event: dict) -> None:
        with self._lock:
            event = {**event, "_seq": self._seq}
            self._seq += 1
            kind = event.get("type")
            if kind == "sample":
                self._last_sample = event
            elif kind == "metric" and not event.get("epoch_end"):
                self._detail.append(event)
            else:  # status, error, epoch_end metrics
                self._pinned.append(event)
            subs = list(self._subs)
        for q in subs:
            q.put(event)

    def subscribe(self) -> tuple[queue.Queue, list[dict]]:
        q: queue.Queue = queue.Queue()
        with self._lock:
            hist = [*self._pinned, *self._detail]
            if self._last_sample is not None:
                hist.append(self._last_sample)
            hist.sort(key=lambda e: e.get("_seq", 0))
            self._subs.append(q)
        return q, hist

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def clear_history(self) -> None:
        with self._lock:
            self._pinned.clear()
            self._detail.clear()
            self._last_sample = None


class Runner:
    """Owns the single active trainer thread."""

    def __init__(self, bus: EventBus):
        self.bus = bus
        self._lock = threading.Lock()
        self._trainer: Trainer | None = None
        self._thread: threading.Thread | None = None
        self._starts = 0

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, cfg: TrainConfig) -> bool:
        with self._lock:
            if self.is_running():
                return False
            self.bus.clear_history()
            # Stamp every event with a per-run tag so clients can tell a
            # reconnect replay (same tag, seen seqs -> skip) from a genuinely
            # new run (new tag -> reset the chart).
            self._starts += 1
            run_tag = f"{cfg.ckpt_dir}#{self._starts}"
            publish = self.bus.publish
            trainer = Trainer(cfg, lambda e: publish({**e, "_run": run_tag}))
            thread = threading.Thread(target=trainer.run, daemon=True)
            self._trainer = trainer
            self._thread = thread
            thread.start()
            return True

    def stop(self) -> None:
        with self._lock:
            if self._trainer is not None:
                self._trainer.stop()


def create_app(
    defaults: dict,
    readonly: bool = False,
    autostart: bool = False,
    event_log: str | Path | None = None,
    title: str | None = None,
) -> FastAPI:
    """Build the app around a base config dict (data_root, n, size, ...).

    readonly: spectator mode; the UI hides run controls and the control
    endpoints refuse. autostart: kick off a run with the default config as
    soon as the server starts (remote pods; no Start press needed).
    event_log: append every event as a JSON line to this path. Telemetry
    persists server-side (on the pod that means the volume) instead of
    depending on someone's browser or a Mac-side capture being attached.
    """
    app = FastAPI()
    bus = EventBus()
    if event_log is not None:
        log_path = Path(event_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a", buffering=1)
        _publish = bus.publish

        def publish_and_log(event: dict) -> None:
            _publish(event)
            try:
                log_file.write(json.dumps(event) + "\n")
            except Exception:  # telemetry loss must never kill training
                pass

        bus.publish = publish_and_log
    runner = Runner(bus)
    app.state.defaults = defaults
    app.state.runner = runner

    if autostart:
        @app.on_event("startup")
        def _autostart() -> None:
            runner.start(TrainConfig(**defaults))

    @app.get("/")
    def index() -> HTMLResponse:
        import html as html_mod

        html = (STATIC / "index.html").read_text()
        if readonly:
            html = html.replace(
                'name="vecml-readonly" content="0"',
                'name="vecml-readonly" content="1"',
            )
        if title:
            html = html.replace(
                'name="vecml-title" content=""',
                f'name="vecml-title" content="{html_mod.escape(title, quote=True)}"',
            )
        return HTMLResponse(html)

    @app.get("/events")
    async def events(request: Request) -> StreamingResponse:
        q, hist = bus.subscribe()

        async def gen():
            try:
                for event in hist:
                    yield _sse(event)
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.to_thread(q.get, True, 10)
                    except queue.Empty:
                        yield ": keepalive\n\n"
                        continue
                    yield _sse(event)
            finally:
                bus.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/start")
    async def start(request: Request) -> JSONResponse:
        if readonly:
            return JSONResponse({"error": "read-only cockpit"}, status_code=403)
        try:
            overrides = await request.json()
        except Exception:
            overrides = {}
        cfg_dict = {**defaults, **(overrides or {})}
        cfg = TrainConfig(**cfg_dict)
        if not runner.start(cfg):
            return JSONResponse({"error": "already running"}, status_code=409)
        return JSONResponse({"ok": True, "config": cfg_dict})

    @app.post("/stop")
    def stop() -> JSONResponse:
        if readonly:
            return JSONResponse({"error": "read-only cockpit"}, status_code=403)
        runner.stop()
        return JSONResponse({"ok": True})

    # --- run tab: checkpoint listing + single-image inference -------------
    # Deliberately allowed in readonly mode: inference mutates nothing.

    @app.get("/ckpts")
    def ckpts() -> JSONResponse:
        return JSONResponse(infer_mod.list_ckpts())

    @app.post("/infer")
    async def do_infer(request: Request) -> JSONResponse:
        import base64 as b64mod
        import io

        from PIL import Image

        try:
            body = await request.json()
            ckpt = Path(body["ckpt"]).resolve()
            runs_root = Path("runs").resolve()
            if not ckpt.is_relative_to(runs_root) or ckpt.suffix != ".pt":
                return JSONResponse({"error": "checkpoint outside runs/"}, status_code=400)
            raw = body["image"].split(",", 1)[-1]  # accept raw b64 or data URL
            img = Image.open(io.BytesIO(b64mod.b64decode(raw)))
        except Exception as exc:  # noqa: BLE001 - surface bad uploads to the UI
            return JSONResponse({"error": f"bad request: {exc}"}, status_code=400)
        try:
            result = await asyncio.to_thread(infer_mod.run, ckpt, img)
        except Exception as exc:  # noqa: BLE001 - surface inference failures too
            return JSONResponse({"error": f"inference failed: {exc}"}, status_code=500)
        return JSONResponse(result)

    return app


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"
