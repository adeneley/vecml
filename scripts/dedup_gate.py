"""CLI for the corpus dedup gate.

Two modes:

  calibrate  Pick the pHash Hamming threshold empirically. Renders a few
             hundred corpus SVGs, synthesises known near-dupes (rescale + tiny
             recolour) and pairs them against known-distinct images, and prints
             the two distance distributions so the separating threshold is
             obvious.

  run        Run the full three-layer gate over a sample of the corpus, with
             the relay-bench images and the val split wired in as held-out
             references, and write a JSON report.

Splits come from the labelled manifest (datasets/svg-stack-labelled/manifest.jsonl):
each clean-tier sha carries a train/val/test tag. The bench images are the 24
SVGs in data/relay-test-src.

  uv run python scripts/dedup_gate.py calibrate --n 600
  uv run python scripts/dedup_gate.py run --n 100000 --threshold 4 \
      --out runs/dedupe/report.json
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

from vecml.dedup.gate import Record, compute_hashes, run_gate
from vecml.dedup.hashes import hamming, phash_from_gray, render_phash
from vecml.degrade.renderer import render_svg

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CLEAN = ROOT / "datasets/svg-stack-labelled/clean"
DEFAULT_MANIFEST = ROOT / "datasets/svg-stack-labelled/manifest.jsonl"
DEFAULT_BENCH = ROOT / "data/relay-test-src"


_MANIFEST_RE = re.compile(r'"sha":\s*"([0-9a-f]+)".*?"split":\s*"(\w+)"')


def load_split_map(manifest: Path, wanted: set[str] | None = None) -> dict[str, str]:
    """sha -> split, scanned from the manifest.

    Uses a regex slice rather than json.loads per line (2.28M lines) and, when
    ``wanted`` is given, keeps only shas in the sample, so the whole scan is a
    few seconds and a small dict instead of a 1.5M-entry one.
    """
    m: dict[str, str] = {}
    with open(manifest) as f:
        for line in f:
            mo = _MANIFEST_RE.search(line)
            if mo is None:
                continue
            sha, split = mo.group(1), mo.group(2)
            if wanted is None or sha in wanted:
                m[sha] = split
    return m


def sample_corpus_paths(clean: Path, n: int, seed: int) -> list[Path]:
    """Randomly sample ``n`` SVG paths from the sharded clean tier.

    Collects paths with os.scandir over the two-hex shard dirs (no global sort
    of ~1.5M Path objects) and takes a seeded random sample.
    """
    files: list[str] = []
    with os.scandir(clean) as shards:
        for shard in shards:
            if not shard.is_dir():
                continue
            with os.scandir(shard.path) as entries:
                files.extend(e.path for e in entries if e.name.endswith(".svg"))
    rng = random.Random(seed)
    if n < len(files):
        files = rng.sample(files, n)
    return [Path(p) for p in files]


def _gray_render(path: Path, size: int) -> np.ndarray:
    rgb = render_svg(path, size).astype(np.float32)
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


_HEX6 = __import__("re").compile(r"#([0-9A-Fa-f]{6})\b")


def _recolor_svg(svg: str, delta: int) -> str:
    """Nudge every 6-digit hex fill by +delta per channel (a tiny re-palette)."""

    def shift(m):
        v = int(m.group(1), 16)
        r = min(255, ((v >> 16) & 255) + delta)
        g = min(255, ((v >> 8) & 255) + delta)
        b = min(255, (v & 255) + delta)
        return f"#{r:02x}{g:02x}{b:02x}"

    return _HEX6.sub(shift, svg)


def _phash_from_svg_text(svg: str, size: int):
    """Render an SVG string at ``size`` px through the repo renderer and pHash."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=True) as fh:
        fh.write(svg)
        fh.flush()
        try:
            return render_phash(fh.name, size)
        except Exception:
            return None


def calibrate(args):
    """Measure near-dupe vs distinct pHash distance distributions."""
    sample = sample_corpus_paths(Path(args.clean), args.n, args.seed)
    rng = random.Random(args.seed)

    # Base pHash at the canonical 64px render.
    base = {}
    for f in sample:
        try:
            base[f] = render_phash(f, args.size)
        except Exception:
            continue
    ok = list(base)
    print(f"rendered {len(ok)}/{len(sample)} base images")

    # Known near-dupes, generated at the SVG level and rendered through the
    # exact 64px path the gate uses (apples-to-apples). Two realistic classes:
    #   recolour  every fill nudged by a small per-channel delta (a re-palette)
    #   resample  the same art rendered at a different size then resized to 64
    #             (a pre-rendered rescale; harsher, reported separately)
    near_recolor = []
    near_resample = []
    for f in ok:
        try:
            svg = f.read_text(errors="replace")
            for delta in (6, 12, 20):
                variant = _recolor_svg(svg, delta)
                h = _phash_from_svg_text(variant, args.size)
                if h is not None:
                    near_recolor.append(hamming(base[f], h))
            g_alt = _gray_render(f, args.alt_size)
            g_alt = np.asarray(
                Image.fromarray(g_alt.astype(np.uint8)).resize((args.size, args.size))
            ).astype(np.float32)
            near_resample.append(hamming(base[f], phash_from_gray(g_alt)))
        except Exception:
            continue

    # Known-distinct: pHash distance between different corpus images.
    distinct = []
    for _ in range(max(len(near_recolor), len(near_resample))):
        a, b = rng.sample(ok, 2)
        distinct.append(hamming(base[a], base[b]))

    near = np.array(near_recolor)
    resample = np.array(near_resample)
    distinct = np.array(distinct)

    def pct(a, ps):
        return {str(p): int(np.percentile(a, p)) for p in ps}

    def dist_block(a):
        return {
            "min": int(a.min()),
            "mean": round(float(a.mean()), 2),
            "max": int(a.max()),
            "pctile": pct(a, [50, 90, 95, 99, 100]),
        }

    report = {
        "n_base": len(ok),
        "n_recolor_pairs": len(near),
        "n_resample_pairs": len(resample),
        "n_distinct_pairs": len(distinct),
        "near_recolor": dist_block(near),
        "near_resample": dist_block(resample),
        "distinct": {
            "min": int(distinct.min()),
            "mean": round(float(distinct.mean()), 2),
            "max": int(distinct.max()),
            "pctile": pct(distinct, [0, 1, 5, 10, 50]),
        },
    }
    # Threshold is chosen on the realistic recolour class (the gate re-renders
    # every SVG at a fixed 64px, so the resample class never arises in
    # production; it is reported only as a robustness bound). Split the gap
    # between the recolour p99 and the distinct p1.
    near_hi = float(np.percentile(near, 99))
    dist_lo = float(np.percentile(distinct, 1))
    report["recolor_p99"] = near_hi
    report["distinct_p1"] = dist_lo
    report["suggested_threshold"] = int(round((near_hi + dist_lo) / 2))
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print("wrote", args.out)


def build_records(args) -> list[Record]:
    t = time.time()
    sample = sample_corpus_paths(Path(args.clean), args.n, args.seed)
    print(f"sampled {len(sample)} paths in {time.time() - t:.1f}s", flush=True)

    split_map: dict[str, str] = {}
    if args.manifest:
        t = time.time()
        split_map = load_split_map(Path(args.manifest), {p.stem for p in sample})
        print(f"loaded splits for {len(split_map)} shas in {time.time() - t:.1f}s", flush=True)

    recs = [
        Record(key=f.stem, path=f, split=split_map.get(f.stem, "corpus"))
        for f in sample
    ]
    if args.with_bench:
        for f in sorted(Path(args.bench).glob("*.svg")):
            recs.append(Record(key=f.stem, path=f, split="bench"))
    return recs


def run(args):
    recs = build_records(args)
    n_bench = sum(1 for r in recs if r.split == "bench")
    n_val = sum(1 for r in recs if r.split == "val")
    n_test = sum(1 for r in recs if r.split == "test")
    print(
        f"records: {len(recs)} (bench={n_bench} val={n_val} test={n_test})",
        flush=True,
    )

    t0 = time.time()
    compute_hashes(recs, precision=args.precision, size=args.size, progress=20000)
    t_hash = time.time() - t0
    t1 = time.time()
    rep, survivors = run_gate(recs, threshold=args.threshold)
    t_gate = time.time() - t1

    d = rep.as_dict()
    n = rep.pool_total
    throughput = n / max(1e-9, t_hash + t_gate)
    d["timing"] = {
        "hash_seconds": round(t_hash, 2),
        "gate_seconds": round(t_gate, 2),
        "files_per_sec": round(throughput, 1),
        "projected_5M_hours": round(5_000_000 / throughput / 3600, 2),
    }
    d["config"] = {
        "n": args.n,
        "threshold": args.threshold,
        "precision": args.precision,
        "size": args.size,
        "seed": args.seed,
    }
    print(json.dumps(d, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(d, indent=2))
        print("wrote", args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate")
    c.add_argument("--n", type=int, default=400)
    c.add_argument("--clean", default=str(DEFAULT_CLEAN))
    c.add_argument("--size", type=int, default=64)
    c.add_argument("--alt-size", type=int, default=96)
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--out", default=None)
    c.set_defaults(func=calibrate)

    r = sub.add_parser("run")
    r.add_argument("--n", type=int, default=100000)
    r.add_argument("--clean", default=str(DEFAULT_CLEAN))
    r.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    r.add_argument("--bench", default=str(DEFAULT_BENCH))
    r.add_argument("--with-bench", action="store_true", default=True)
    r.add_argument("--no-bench", dest="with_bench", action="store_false")
    r.add_argument("--threshold", type=int, default=4)
    r.add_argument("--precision", type=int, default=1)
    r.add_argument("--size", type=int, default=64)
    r.add_argument("--seed", type=int, default=13)
    r.add_argument("--out", default=None)
    r.set_defaults(func=run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
