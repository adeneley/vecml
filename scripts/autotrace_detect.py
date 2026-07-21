"""Autotrace detector: build a labelled set, measure feature separation, fit a
logistic classifier, and report an operating threshold for the corpus gate.

Positives (traced) are self-minted: a sample of born-vector SVGs is rendered to
raster and re-traced with vtracer (colour) and potrace (silhouette), which
reproduces the exact structural damage Wikimedia Image-Trace uploads carry.
Negatives (born-vector) are the source SVGs themselves. Because each positive is
the trace of a specific negative, the two classes are matched on subject matter,
so any separation is provenance, not content.

Usage:
    uv run python scripts/autotrace_detect.py \
        --src data/audit-500-src --n 300 --out docs/research/autotrace-eval

Outputs a JSON of per-file feature rows and prints the separation table,
classifier metrics, and the recommended threshold. With --svgo it also runs the
adversarial post-processing check (needs `npx svgo`).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vecml.gate.features import FEATURE_NAMES, extract_features, feature_vector  # noqa: E402


# --- tracing backends ------------------------------------------------------

def _render_png_bytes(svg_path: Path, size: int) -> bytes:
    """Render an SVG to a white-background RGB PNG (bytes)."""
    from vecml.degrade.renderer import composite_rgba, render_svg_rgba

    rgba = render_svg_rgba(svg_path, size)
    rgb = composite_rgba(rgba, colour=(255, 255, 255))
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def trace_vtracer(png_bytes: bytes, workdir: Path, stem: str) -> str | None:
    import vtracer

    inp = workdir / f"{stem}.png"
    out = workdir / f"{stem}.vt.svg"
    inp.write_bytes(png_bytes)
    try:
        vtracer.convert_image_to_svg_py(str(inp), str(out), colormode="color", filter_speckle=4)
        return out.read_text(encoding="utf-8")
    except Exception:
        return None


def trace_potrace(png_bytes: bytes, workdir: Path, stem: str) -> str | None:
    inp = workdir / f"{stem}.bmp"
    out = workdir / f"{stem}.pt.svg"
    Image.open(io.BytesIO(png_bytes)).convert("L").save(inp, format="BMP")
    try:
        subprocess.run(
            ["potrace", str(inp), "-s", "-o", str(out)],
            check=True, capture_output=True, timeout=60,
        )
        return out.read_text(encoding="utf-8")
    except Exception:
        return None


# --- tiny numpy logistic regression (no sklearn dependency) ----------------

def _standardize(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd, mu, sd


def fit_logreg(X, y, iters=4000, lr=0.2, l2=1e-3):
    Xs, mu, sd = _standardize(X)
    n, d = Xs.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        w -= lr * (Xs.T @ g / n + l2 * w)
        b -= lr * g.mean()
    return {"w": w, "b": b, "mu": mu, "sd": sd}


def predict_proba(model, X):
    Xs = (X - model["mu"]) / model["sd"]
    return 1.0 / (1.0 + np.exp(-(Xs @ model["w"] + model["b"])))


def roc_auc(y, p):
    order = np.argsort(-p)
    y = y[order]
    P = y.sum()
    N = len(y) - P
    if P == 0 or N == 0:
        return float("nan")
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / P
    fpr = fp / N
    fpr = np.concatenate([[0], fpr])
    tpr = np.concatenate([[0], tpr])
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(tpr, fpr))


# --- dataset build ---------------------------------------------------------

def build_dataset(src: Path, n: int, size: int, seed: int, workdir: Path, trace_dir: Path | None = None):
    files = sorted(src.glob("*.svg"))
    random.Random(seed).shuffle(files)
    rows = []
    minted = skipped = 0
    for f in files:
        if minted >= n:
            break
        try:
            svg = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        neg = extract_features(svg)
        if neg.get("parse_error"):
            skipped += 1
            continue
        try:
            png = _render_png_bytes(f, size)
        except Exception:
            skipped += 1
            continue
        vt = trace_vtracer(png, workdir, f.stem)
        pt = trace_potrace(png, workdir, f.stem)
        if vt is None and pt is None:
            skipped += 1
            continue
        rows.append({"file": f.name, "label": 0, "tracer": "born", **neg})
        for tag, txt in (("vtracer", vt), ("potrace", pt)):
            if txt is None:
                continue
            rows.append({"file": f.name, "label": 1, "tracer": tag, **extract_features(txt)})
            if trace_dir is not None:
                (trace_dir / f"{f.stem}.{tag}.svg").write_text(txt, encoding="utf-8")
        minted += 1
        if minted % 25 == 0:
            print(f"  minted {minted}/{n} (skipped {skipped})", file=sys.stderr)
    return rows, minted, skipped


def svgo_augment_rows(trace_dir: Path, opt_dir: Path):
    """Batch-optimize every minted trace with svgo (one npx startup) and return
    svgo-processed positive rows. The `file`/`tracer` keys mirror the source so
    the train/test split-by-file keeps svgo variants on their parent's side.
    """
    opt_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["npx", "--yes", "svgo", "-f", str(trace_dir), "-o", str(opt_dir), "-q"],
            check=True, capture_output=True, timeout=600,
        )
    except Exception as e:
        print(f"  svgo batch failed ({e}); skipping augmentation", file=sys.stderr)
        return []
    rows = []
    for f in sorted(opt_dir.glob("*.svg")):
        stem, tag = f.stem.rsplit(".", 1) if "." in f.stem else (f.stem, "svgo")
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rows.append({"file": stem + ".svg", "label": 1, "tracer": tag + "_svgo",
                     **extract_features(txt)})
    return rows


def _matrix(rows):
    X = np.array([feature_vector(r) for r in rows], dtype=float)
    y = np.array([r["label"] for r in rows], dtype=float)
    return X, y


def separation_table(rows):
    neg = [r for r in rows if r["label"] == 0]
    pos = [r for r in rows if r["label"] == 1]
    out = []
    for k in FEATURE_NAMES:
        nv = np.array([r[k] for r in neg])
        pv = np.array([r[k] for r in pos])
        nm, pm = nv.mean(), pv.mean()
        ns, ps = nv.std(), pv.std()
        pooled = math.sqrt((ns**2 + ps**2) / 2) or 1e-9
        d = abs(pm - nm) / pooled  # Cohen's d
        out.append((k, nm, pm, d))
    out.sort(key=lambda r: -r[3])
    return out


def threshold_for_poison(y, p, max_fpr=0.01):
    """Highest keep-rate of clean files while admitting <= max_fpr of traces.

    A 'keep clean' decision is p < threshold. FPR here = fraction of positives
    (traces) that slip through as clean. Returns (threshold, keep_clean, poison).
    """
    best = None
    for thr in np.unique(p):
        keep_clean = np.mean(p[y == 0] < thr)   # clean files retained
        poison = np.mean(p[y == 1] < thr)       # traces wrongly kept
        if poison <= max_fpr:
            if best is None or keep_clean > best[1]:
                best = (float(thr), float(keep_clean), float(poison))
    if best is None:
        thr = float(p.min())
        return thr, float(np.mean(p[y == 0] < thr)), float(np.mean(p[y == 1] < thr))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/audit-500-src"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("docs/research/autotrace-eval"))
    ap.add_argument("--svgo", action="store_true", help="run adversarial svgo post-processing check")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        trace_dir = workdir / "traces"
        trace_dir.mkdir()
        base_rows, minted, skipped = build_dataset(
            args.src, args.n, args.size, args.seed, workdir, trace_dir=trace_dir)
        print(f"\nminted {minted} source images, {skipped} skipped; "
              f"{len(base_rows)} feature rows", file=sys.stderr)

        svgo_rows = []
        if args.svgo:
            print("running svgo over minted traces (adversarial)...", file=sys.stderr)
            svgo_rows = svgo_augment_rows(trace_dir, workdir / "svgo")
            print(f"  svgo produced {len(svgo_rows)} optimized positive rows", file=sys.stderr)

        rows = base_rows + svgo_rows
        (args.out / "features.json").write_text(json.dumps(rows, indent=1))

        # --- separation (born vs raw trace) ---
        table = separation_table(base_rows)
        print("\n=== per-feature separation, born vs raw trace (Cohen's d) ===")
        print(f"{'feature':>18}  {'born_mean':>10}  {'trace_mean':>10}  {'cohen_d':>8}")
        for k, nm, pm, d in table:
            print(f"{k:>18}  {nm:>10.3f}  {pm:>10.3f}  {d:>8.2f}")

        # split by SOURCE FILE so a trace and its svgo variant never straddle
        src_files = sorted({r["file"] for r in rows})
        random.Random(args.seed + 1).shuffle(src_files)
        train_files = set(src_files[:int(0.7 * len(src_files))])

        def split(subset):
            tr = np.array([r["file"] in train_files for r in subset])
            X = np.array([feature_vector(r) for r in subset], dtype=float)
            y = np.array([r["label"] for r in subset], dtype=float)
            return X, y, tr

        # baseline model: trained on born + raw trace only
        Xb, yb, trb = split(base_rows)
        base_model = fit_logreg(Xb[trb], yb[trb])
        auc_b = roc_auc(yb[~trb], predict_proba(base_model, Xb[~trb]))
        thr_b, keep_b, pois_b = threshold_for_poison(
            yb[~trb], predict_proba(base_model, Xb[~trb]), 0.01)
        print("\n=== baseline model (born + raw trace), held-out ===")
        print(f"ROC-AUC: {auc_b:.4f}")
        print(f"op-point (<=1% poison): keep p<{thr_b:.4f} -> "
              f"clean kept {keep_b*100:.1f}%, traces admitted {pois_b*100:.2f}%")

        print("\n=== single-feature threshold AUCs ===")
        for k, nm, pm, d in table[:5]:
            xi = Xb[:, FEATURE_NAMES.index(k)]
            print(f"{k:>18}: AUC {roc_auc(yb, xi if pm > nm else -xi):.4f}")

        # baseline model's blind spot: score held-out svgo-optimized traces
        if svgo_rows:
            held_svgo = [r for r in svgo_rows if r["file"] not in train_files]
            Xs = np.array([feature_vector(r) for r in held_svgo], dtype=float)
            ps = predict_proba(base_model, Xs)
            print("\n=== adversarial: svgo-optimized traces vs BASELINE model ===")
            print(f"{len(held_svgo)} held-out svgo traces, caught (p>={thr_b:.3f}): "
                  f"{np.mean(ps >= thr_b)*100:.1f}%, mean p={ps.mean():.3f}")

            # augmented model: train on born + raw trace + svgo trace
            Xa, ya, tra = split(rows)
            aug_model = fit_logreg(Xa[tra], ya[tra])
            pa = predict_proba(aug_model, Xa[~tra])
            auc_a = roc_auc(ya[~tra], pa)
            thr_a, keep_a, pois_a = threshold_for_poison(ya[~tra], pa, 0.01)
            # its catch rate specifically on held-out svgo traces
            ps2 = predict_proba(aug_model, Xs)
            print("\n=== augmented model (born + raw trace + svgo trace), held-out ===")
            print(f"ROC-AUC (all positive types): {auc_a:.4f}")
            print(f"op-point (<=1% poison): keep p<{thr_a:.4f} -> "
                  f"clean kept {keep_a*100:.1f}%, traces admitted {pois_a*100:.2f}%")
            print(f"held-out svgo traces caught (p>={thr_a:.3f}): {np.mean(ps2 >= thr_a)*100:.1f}%, "
                  f"mean p={ps2.mean():.3f}")
            final = aug_model
            final_thr, final_auc = thr_a, auc_a
        else:
            final = base_model
            final_thr, final_auc = thr_b, auc_b

        (args.out / "model.json").write_text(json.dumps({
            "feature_names": FEATURE_NAMES,
            "w": final["w"].tolist(), "b": float(final["b"]),
            "mu": final["mu"].tolist(), "sd": final["sd"].tolist(),
            "threshold_1pct": final_thr, "auc": final_auc,
            "trained_with_svgo": bool(svgo_rows),
        }, indent=1))

    print("\nfeatures.json + model.json ->", args.out)


if __name__ == "__main__":
    main()
