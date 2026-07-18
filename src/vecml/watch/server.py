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

STATIC = Path(__file__).parent / "static"


class EventBus:
    """Thread-safe fan-out of JSON events with a bounded replay history."""

    def __init__(self, history: int = 800):
        self._lock = threading.Lock()
        self._history: deque[dict] = deque(maxlen=history)
        self._subs: list[queue.Queue] = []

    def publish(self, event: dict) -> None:
        with self._lock:
            self._history.append(event)
            subs = list(self._subs)
        for q in subs:
            q.put(event)

    def subscribe(self) -> tuple[queue.Queue, list[dict]]:
        q: queue.Queue = queue.Queue()
        with self._lock:
            hist = list(self._history)
            self._subs.append(q)
        return q, hist

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()


class Runner:
    """Owns the single active trainer thread."""

    def __init__(self, bus: EventBus):
        self.bus = bus
        self._lock = threading.Lock()
        self._trainer: Trainer | None = None
        self._thread: threading.Thread | None = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, cfg: TrainConfig) -> bool:
        with self._lock:
            if self.is_running():
                return False
            self.bus.clear_history()
            trainer = Trainer(cfg, self.bus.publish)
            thread = threading.Thread(target=trainer.run, daemon=True)
            self._trainer = trainer
            self._thread = thread
            thread.start()
            return True

    def stop(self) -> None:
        with self._lock:
            if self._trainer is not None:
                self._trainer.stop()


def create_app(defaults: dict) -> FastAPI:
    """Build the app around a base config dict (data_root, n, size, ...)."""
    app = FastAPI()
    bus = EventBus()
    runner = Runner(bus)
    app.state.defaults = defaults
    app.state.runner = runner

    @app.get("/")
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC / "index.html").read_text())

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
        runner.stop()
        return JSONResponse({"ok": True})

    return app


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"
