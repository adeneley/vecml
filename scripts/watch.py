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
    ap.add_argument("--port", type=int, default=7300)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (0.0.0.0 for remote pods behind a proxy)")
    ap.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = ap.parse_args()

    defaults = {
        "data_root": args.data,
        "n": args.n,
        "size": args.size,
        "variant": args.variant,
        "batch_size": args.batch,
        "max_epochs": args.epochs,
        "ckpt_dir": f"runs/overfit{args.n}",
    }
    app = create_app(defaults)

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
