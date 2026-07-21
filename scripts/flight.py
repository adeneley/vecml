"""Run a sequence of training runs on one pod, cockpit visible throughout.

Each run in the plan gets a fresh cockpit (readonly + autostart) on the same
port; when its trainer finishes, the server is cycled and the next run
begins. The browser's SSE auto-reconnect picks up the new run's replay, so a
spectator sees run A finish and run B start without touching anything.

Every run's full event stream is appended to runs/<name>/events.jsonl next
to its checkpoint (on a pod both live on the volume via the repo checkout).

  uv run python scripts/flight.py --plan infra/flightplans/labels-ab.json \
      --from-bench /tmp/bench.json --host 0.0.0.0 --port 7300
"""

import argparse
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vecml.watch.server import create_app  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="JSON flight plan: {runs: [TrainConfig dicts + name]}")
    ap.add_argument("--from-bench", default=None, help="bench.py --emit JSON; applied to every run")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7300)
    ap.add_argument("--single", type=int, default=None,
                    help="internal: train only run #N (1-based) then exit")
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    winner = None
    if args.from_bench:
        winner = json.loads(Path(args.from_bench).read_text())
        winner = winner.get("winner", winner)  # perflab.json nests it

    # Sequencer mode: one fresh process per run. torch.compile's cudagraph
    # trees live in thread-local storage; a second trainer thread in the same
    # process hits the compiled cache and dies on the TLS assertion (the
    # 21 Jul CE-sweep failure). A process per run also returns all GPU memory
    # between arms.
    if args.single is None:
        for i, run in enumerate(plan["runs"], 1):
            cmd = [sys.executable, str(Path(__file__).resolve()),
                   "--plan", args.plan, "--host", args.host,
                   "--port", str(args.port), "--single", str(i)]
            if args.from_bench:
                cmd += ["--from-bench", args.from_bench]
            print(f"[flight] run {i}/{len(plan['runs'])}: "
                  f"{run['name']} (fresh process)", flush=True)
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                print(f"[flight] WARNING: {run['name']} exited rc={rc}; "
                      f"continuing with next run", flush=True)
        # Idle rather than exit: an exiting job makes RunPod restart the
        # container and replay the plan (see the skip guard). Teardown ends
        # the pod.
        print("[flight] all runs complete; idling for teardown", flush=True)
        while True:
            time.sleep(60)

    for i, run in enumerate(plan["runs"], 1):
        if i != args.single:
            continue
        run = dict(run)
        name = run.pop("name")
        run.setdefault("ckpt_dir", f"runs/{name}")
        # RunPod restarts the container when the job process exits, and startup
        # re-runs the whole plan. A finished run must never train again: its
        # first checkpoint save would overwrite the completed best.pt. The
        # event log is the durable record - a "finished" status means done.
        ev_path = Path(run["ckpt_dir"]) / "events.jsonl"
        if ev_path.exists():
            with open(ev_path, errors="replace") as f:
                if any('"state": "finished"' in ln for ln in f):
                    print(f"[flight] {name}: already finished per events.jsonl; "
                          f"skipping (delete the events file to force a re-run)",
                          flush=True)
                    continue
        if winner:
            # The recipe supplies defaults; anything the flightplan sets
            # explicitly wins (e.g. big datasets that cannot cache_ram need
            # their own num_workers).
            run.setdefault("batch_size", winner["batch_size"])
            run.setdefault("num_workers", winner["num_workers"])
            run.setdefault("amp", winner["amp"])
            # perflab levers, when the recipe carries them (older bench.json
            # winners don't; TrainConfig defaults are the safe eager path).
            for key in ("sync_every", "fused_adam", "channels_last", "compile_mode",
                        "non_blocking", "cudnn_benchmark"):
                if key in winner:
                    run.setdefault(key, winner[key])
            # sqrt LR scaling vs the batch-8 baseline all prior runs used.
            run.setdefault("lr", 3e-4 * math.sqrt(run["batch_size"] / 8))
        print(f"[flight] run {i}/{len(plan['runs'])}: {name} "
              f"batch={run.get('batch_size', 8)} workers={run.get('num_workers', 0)} "
              f"amp={run.get('amp', True)} lr={run.get('lr', 3e-4):.2e} "
              f"compile={run.get('compile_mode')} sync={run.get('sync_every', 1)} "
              f"fused={run.get('fused_adam', False)} cl={run.get('channels_last', False)}",
              flush=True)

        title = (f"{name} · run {i}/{len(plan['runs'])} · "
                 f"{Path(run['data_root']).name} n={run.get('n', '?')} "
                 f"K={run.get('n_classes', 0)} batch={run.get('batch_size', 8)}"
                 + (" · compiled" if run.get("compile_mode") else ""))
        app = create_app(
            defaults=run,
            readonly=True,
            autostart=True,
            event_log=Path(run["ckpt_dir"]) / "events.jsonl",
            title=title,
        )
        server = uvicorn.Server(uvicorn.Config(
            app, host=args.host, port=args.port, log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        runner = app.state.runner
        time.sleep(20)  # let startup + autostart spin the trainer up
        if not runner.is_running():
            print(f"[flight] WARNING: {name} trainer not running 20s after "
                  f"autostart; check events", flush=True)
        while runner.is_running():
            time.sleep(5)
        print(f"[flight] {name} trainer finished; cycling server", flush=True)
        time.sleep(10)  # let spectators drain the final events
        server.should_exit = True
        thread.join(timeout=30)
        time.sleep(3)  # free the port before the sequencer's next bind

    return 0


if __name__ == "__main__":
    sys.exit(main())
