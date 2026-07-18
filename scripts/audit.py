"""Audit a directory of wrecked samples: aggregate QC flags and build a report.

Walks an output dir whose subdirs each hold one sample (clean.png, labels.png,
palette.json, meta.json). For every sample it reads the QC block the pipeline
wrote, or recomputes it from the files if absent, then writes two things at the
root of the dir:

  _audit_summary.json   counts per flag, worst offenders, method split
  _audit_flagged.html   only the flagged samples, three columns
                        (clean | colourised labels | wrecked_00) with metrics,
                        lazy-loading images, relative paths, no external requests

Example:
  uv run python scripts/audit.py --dir data/audit-500-v2
"""

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vecml.degrade.audit import audit_sample  # noqa: E402


def _load_sample_qc(sample_dir):
    """Return (meta, qc) for one sample, recomputing qc if the meta lacks it."""
    meta_path = sample_dir / "meta.json"
    if not meta_path.exists():
        return None, None
    meta = json.loads(meta_path.read_text())
    qc = meta.get("qc")
    if qc is not None:
        return meta, qc

    # Recompute from files (older samples, or samples written without qc).
    clean_path = sample_dir / "clean.png"
    labels_path = sample_dir / "labels.png"
    palette_path = sample_dir / "palette.json"
    if not (clean_path.exists() and labels_path.exists() and palette_path.exists()):
        return meta, None
    clean = np.asarray(Image.open(clean_path).convert("RGB"))
    label_map = np.asarray(Image.open(labels_path))
    palette = np.array(json.loads(palette_path.read_text())["palette"], dtype=np.uint8)
    qc = audit_sample(clean, label_map, palette, meta.get("n_declared"))
    return meta, qc


def _render_html(out_dir, flagged):
    """Build the self-contained flagged-samples report."""
    rows = []
    for item in flagged:
        name = item["name"]
        m = item["qc"]["metrics"]
        flags = ", ".join(item["qc"]["flags"])
        method = item.get("label_method", "?")
        metric_line = (
            f"ink {m['ink_frac']:.3f} &middot; labels {m['label_nonzero_frac']:.3f} "
            f"&middot; cov&Delta; {m['coverage_delta']:.3f} &middot; "
            f"recon {m['reconstruction_mae']:.2f} &middot; "
            f"palette {m['palette_size']} (declared {m['n_declared']})"
        )
        rows.append(
            f"""    <div class="card">
      <div class="head">
        <span class="sha">{html.escape(name)}</span>
        <span class="method">{html.escape(method)}</span>
        <span class="flags">{html.escape(flags)}</span>
      </div>
      <div class="metrics">{metric_line}</div>
      <div class="imgs">
        <figure><img loading="lazy" src="{html.escape(name)}/clean.png" alt="clean"><figcaption>clean</figcaption></figure>
        <figure><img loading="lazy" src="{html.escape(name)}/labels_view.png" alt="labels"><figcaption>labels</figcaption></figure>
        <figure><img loading="lazy" src="{html.escape(name)}/wrecked_00.png" alt="wrecked"><figcaption>wrecked_00</figcaption></figure>
      </div>
    </div>"""
        )

    body = "\n".join(rows) if rows else '    <p class="none">No flagged samples.</p>'
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Audit: flagged samples</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0; padding: 24px; background: #12141a; color: #e6e8ee;
         font: 14px/1.5 -apple-system, Segoe UI, Roboto, sans-serif; }}
  h1 {{ font-size: 18px; margin: 0 0 4px; }}
  .sub {{ color: #99a; margin: 0 0 20px; }}
  .card {{ background: #1b1e27; border: 1px solid #2a2e3a; border-radius: 10px;
          padding: 14px; margin: 0 0 16px; }}
  .head {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: baseline; }}
  .sha {{ font-family: ui-monospace, monospace; font-size: 12px; color: #8fb7ff; }}
  .method {{ font-size: 11px; color: #9aa; border: 1px solid #333; border-radius: 4px;
            padding: 1px 6px; }}
  .flags {{ color: #ffb4a2; font-weight: 600; }}
  .metrics {{ color: #aab; font-size: 12px; margin: 6px 0 10px;
             font-family: ui-monospace, monospace; }}
  .imgs {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  figure {{ margin: 0; }}
  figcaption {{ text-align: center; color: #889; font-size: 11px; margin-top: 4px; }}
  img {{ width: 220px; height: 220px; object-fit: contain; background:
        repeating-conic-gradient(#20232c 0% 25%, #262a35 0% 50%) 50% / 20px 20px;
        border: 1px solid #2a2e3a; border-radius: 6px; }}
  .none {{ color: #7a7; }}
</style>
</head>
<body>
  <h1>Flagged samples: {len(flagged)}</h1>
  <p class="sub">{html.escape(str(out_dir))}</p>
{body}
</body>
</html>
"""
    (out_dir / "_audit_flagged.html").write_text(doc, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Audit a dir of wrecked samples.")
    ap.add_argument("--dir", required=True, help="output dir of per-sample subdirs")
    ap.add_argument("--worst", type=int, default=25, help="worst offenders to list")
    args = ap.parse_args()

    out_dir = Path(args.dir)
    if not out_dir.is_dir():
        ap.error(f"not a directory: {out_dir}")

    sample_dirs = sorted(p for p in out_dir.iterdir() if p.is_dir())
    total = 0
    scored = 0
    flag_counts = {}
    method_counts = {}
    flagged = []
    all_scores = []

    for sd in sample_dirs:
        meta, qc = _load_sample_qc(sd)
        if meta is None:
            continue
        total += 1
        method_counts[meta.get("label_method", "unknown")] = (
            method_counts.get(meta.get("label_method", "unknown"), 0) + 1
        )
        if qc is None:
            flag_counts["qc_uncomputable"] = flag_counts.get("qc_uncomputable", 0) + 1
            flagged.append({"name": sd.name, "qc": {"metrics": {}, "flags": ["qc_uncomputable"]}})
            continue
        scored += 1
        for flag in qc["flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        all_scores.append((sd.name, qc["metrics"]))
        if qc["flags"]:
            flagged.append(
                {"name": sd.name, "qc": qc, "label_method": meta.get("label_method", "?")}
            )

    def worst_by(metric):
        ranked = sorted(all_scores, key=lambda x: x[1].get(metric, 0), reverse=True)
        return [
            {"name": n, metric: m.get(metric)} for n, m in ranked[: args.worst]
        ]

    summary = {
        "dir": str(out_dir),
        "n_samples": total,
        "n_scored": scored,
        "n_flagged": len(flagged),
        "flag_counts": dict(sorted(flag_counts.items(), key=lambda x: -x[1])),
        "method_counts": method_counts,
        "worst_reconstruction": worst_by("reconstruction_mae"),
        "worst_coverage": worst_by("coverage_delta"),
        "flagged_names": [f["name"] for f in flagged],
    }
    (out_dir / "_audit_summary.json").write_text(json.dumps(summary, indent=2))
    _render_html(out_dir, flagged)

    print(f"samples: {total}  scored: {scored}  flagged: {len(flagged)}")
    print("flag counts:")
    for flag, n in summary["flag_counts"].items():
        print(f"  {flag:32s} {n}")
    print("method split:")
    for method, n in method_counts.items():
        print(f"  {method:32s} {n}")
    print(f"wrote {out_dir/'_audit_summary.json'} and {out_dir/'_audit_flagged.html'}")


if __name__ == "__main__":
    main()
