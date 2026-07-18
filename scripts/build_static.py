"""Rebuild the browsable static.html gallery for an audit set.

One row per sample: sha, palette-size ("N colours"), any red audit flag badges,
the wreck recipes, and four thumbnails (clean | label map | wrecked_00 |
wrecked_01). Mirrors the hand-built v2 gallery so the report stays consistent.

  uv run python scripts/build_static.py --dir data/audit-500-v2
"""

import argparse
import html
import json
from pathlib import Path

_STYLE = """  :root { color-scheme: light dark; }
  body { font: 15px/1.5 -apple-system, system-ui, sans-serif; margin: 0; background: #fafafa; color: #1a1a1a; }
  @media (prefers-color-scheme: dark) { body { background: #161616; color: #e8e8e8; } header { background: #161616 !important; border-color: #333 !important; } .row { border-color: #2c2c2c !important; } }
  header { position: sticky; top: 0; background: #fafafa; border-bottom: 1px solid #ddd; padding: 12px 20px; z-index: 2; }
  header h1 { font-size: 17px; margin: 0 0 2px; }
  header p { margin: 0; font-size: 13px; opacity: .75; }
  .legend { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; max-width: 1100px; margin: 6px 0 0; font-size: 12px; font-weight: 600; opacity: .8; }
  .row { border-bottom: 1px solid #e4e4e4; padding: 10px 20px 14px; }
  .cap { font-size: 12px; margin-bottom: 6px; opacity: .85; }
  .cap .rec { float: right; text-align: right; opacity: .7; font-size: 11px; }
  .flag { background: #c0392b; color: #fff; border-radius: 3px; padding: 1px 6px; font-size: 11px; font-weight: 700; margin-left: 6px; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; max-width: 1100px; }
  .c img { width: 100%; background:
    repeating-conic-gradient(#00000014 0 25%, transparent 0 50%) 0 0 / 16px 16px; border: 1px solid #00000022; border-radius: 3px; }"""


def _bucket(n):
    if n <= 2:
        return "1-2"
    if n <= 4:
        return "3-4"
    if n <= 9:
        return "5-9"
    return "10+"


def _recipe_text(variants):
    lines = []
    for i, v in enumerate(variants):
        ops = " + ".join(f"{r['op']}({r['severity']:.2f})" for r in v["recipe"])
        lines.append(f"{i:02d}: {html.escape(ops)}")
    return "<br>".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Build static.html for an audit set.")
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.dir)
    sample_dirs = sorted(p for p in out_dir.iterdir() if p.is_dir())

    rows = []
    buckets = {"1-2": 0, "3-4": 0, "5-9": 0, "10+": 0}
    flagged = 0
    total = 0
    for sd in sample_dirs:
        meta_path = sd / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        total += 1
        n_pal = int(meta.get("n_palette", meta.get("qc", {}).get("metrics", {}).get("palette_size", 0)))
        buckets[_bucket(n_pal)] += 1
        flags = meta.get("qc", {}).get("flags", [])
        if flags:
            flagged += 1
        sha = sd.name
        flag_html = (
            f'<span class="flag">{html.escape(", ".join(flags))}</span>' if flags else ""
        )
        rec = _recipe_text(meta.get("variants", []))
        imgs = (
            f'<div class="c"><img loading="lazy" src="{sha}/clean.png" alt="clean"></div>'
            f'<div class="c"><img loading="lazy" src="{sha}/labels_view.png" alt="labels"></div>'
            f'<div class="c"><img loading="lazy" src="{sha}/wrecked_00.png" alt="wrecked 00"></div>'
            f'<div class="c"><img loading="lazy" src="{sha}/wrecked_01.png" alt="wrecked 01"></div>'
        )
        rows.append(
            f'<div class="row"><div class="cap"><code>{html.escape(sha[:12])}</code> '
            f'&middot; {n_pal} colours {flag_html}<span class="rec">{rec}</span></div>'
            f'<div class="grid">{imgs}</div></div>'
        )

    bucket_str = " &nbsp; ".join(f"{k}: {v}" for k, v in buckets.items())
    header_p = (
        f"Labels now read from SVG source geometry (idmap-v3, planar faces), all "
        f"{total} via the new path &middot; auto-audit flagged {flagged} of {total} "
        f"(red badges) &middot; palette sizes: {bucket_str}"
    )
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wreck audit v2: geometry-derived answer keys</title>
<style>
{_STYLE}
</style></head><body>
<header>
  <h1>Wreck audit v2: geometry-derived answer keys, same {total} files</h1>
  <p>{header_p}</p>
  <div class="legend"><span>clean render</span><span>label map (answer key)</span><span>wrecked 00</span><span>wrecked 01</span></div>
</header>
{''.join(rows)}
</body></html>
"""
    (out_dir / "static.html").write_text(doc, encoding="utf-8")
    print(f"wrote {out_dir/'static.html'}: {total} rows, {flagged} flagged")


if __name__ == "__main__":
    main()
