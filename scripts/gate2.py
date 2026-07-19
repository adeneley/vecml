"""Gate v2: structural SVG quality analysis via XML + path-geometry parsing.

Three tiers instead of binary:
  reject — unusable as ground truth (raster, blank, unparseable, external refs)
  warn   — usable after preprocessing or human call (live text, trace suspect, editor junk)
  clean  — no findings

Usage: gate2.py <parquet> <out.html> <title>
"""
import base64
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "svg-stack-full-test.parquet"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "svg_gate2_audit.html"
TITLE = sys.argv[3] if len(sys.argv) > 3 else "SVG-Stack (Full)"
SEED = 42
N = 500
EMBED_CAP = 300_000

NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
DTOK_RE = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|([-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?)")
PARAMS = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}
DRAWABLE = {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "text", "use", "image"}
EDITOR_NS = ("inkscape", "sodipodi", "adobe", "sketch", "figma", "xmlns:dc", "xmlns:cc", "xmlns:rdf")


def tag_of(el):
    return el.tag.split("}")[-1].lower() if isinstance(el.tag, str) else ""


def parse_paths(d):
    """Parse a d attribute. Returns (n_cmds, points, chord_lengths, n_curves, n_lines)."""
    toks = DTOK_RE.findall(d)
    cx = cy = sx = sy = 0.0
    pts, chords = [], []
    n_cmds = n_curves = n_lines = 0
    cmd = None
    nums = []
    stream = []
    for c, num in toks:
        if c:
            stream.append(("cmd", c))
        else:
            # clamp: coords beyond this are data corruption, and 1e300**2 overflows
            stream.append(("num", max(-1e9, min(1e9, float(num)))))
    i = 0
    while i < len(stream):
        kind, val = stream[i]
        if kind == "cmd":
            cmd = val
            i += 1
            if cmd.upper() == "Z":
                chords.append(abs(cx - sx) + abs(cy - sy))
                cx, cy = sx, sy
                n_cmds += 1
                continue
        elif cmd is None:
            i += 1
            continue
        need = PARAMS[cmd.upper()]
        args = []
        while len(args) < need and i < len(stream) and stream[i][0] == "num":
            args.append(stream[i][1])
            i += 1
        if len(args) < need:
            break
        rel = cmd.islower()
        u = cmd.upper()
        px, py = cx, cy
        if u == "H":
            cx = cx + args[0] if rel else args[0]
        elif u == "V":
            cy = cy + args[0] if rel else args[0]
        elif u == "A":
            cx = cx + args[5] if rel else args[5]
            cy = cy + args[6] if rel else args[6]
        else:
            coords = [(args[j] + (px if rel and j % 2 == 0 else py if rel else 0), ) for j in range(0)]
            # generic endpoint + control points
            for j in range(0, need, 2):
                x = args[j] + (cx if rel else 0)
                y = args[j + 1] + (cy if rel else 0)
                pts.append((x, y))
            cx = args[need - 2] + (cx if rel else 0)
            cy = args[need - 1] + (cy if rel else 0)
        pts.append((cx, cy))
        chords.append(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5)
        n_cmds += 1
        if u in ("C", "S", "Q", "T", "A"):
            n_curves += 1
        elif u in ("L", "H", "V"):
            n_lines += 1
        if u == "M":
            sx, sy = cx, cy
            cmd = "l" if rel else "L"  # implicit repeats of M are lineto
    return n_cmds, pts, chords, n_curves, n_lines


def style_get(el, prop):
    v = el.get(prop)
    if v is not None:
        return v.strip()
    style = el.get("style", "")
    m = re.search(rf"{prop}\s*:\s*([^;]+)", style)
    return m.group(1).strip() if m else None


def analyze(svg):
    n = len(svg)
    flags = {"reject": [], "warn": []}
    feat = {"len": n, "paths": 0, "cmds": 0, "colors": set(), "gradient": False,
            "stroke": False, "curve_frac": 0.0}

    if n > 200_000:
        flags["reject"].append("huge")
        return flags, feat
    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        flags["reject"].append("parse-error")
        return flags, feat
    if tag_of(root) != "svg":
        flags["reject"].append("not-svg")
        return flags, feat

    lower = svg.lower()
    if "<script" in lower or re.search(r"\bon(load|click|mouseover)\s*=", lower):
        flags["reject"].append("script")
    if re.search(r"href\s*=\s*[\"']https?://", lower):
        flags["reject"].append("external-ref")
    if "base64" in lower and ("image/png" in lower or "image/jpeg" in lower or "image/jpg" in lower):
        flags["reject"].append("raster-b64")
    if any(ns in lower for ns in EDITOR_NS):
        flags["warn"].append("editor-junk")
    if "<style" in lower:
        flags["warn"].append("css-block")
    if "<filter" in lower:
        flags["warn"].append("filter")
    if "<mask" in lower or "clip-path" in lower:
        flags["warn"].append("mask/clip")

    # viewBox / size
    vb = root.get("viewBox")
    W = H = None
    if vb:
        vals = NUM_RE.findall(vb)
        if len(vals) == 4:
            vx, vy, W, H = map(float, vals)
    if W is None:
        try:
            W = float(NUM_RE.search(root.get("width", ""))[0])
            H = float(NUM_RE.search(root.get("height", ""))[0])
            vx = vy = 0.0
        except (TypeError, IndexError):
            W = H = None
            flags["reject"].append("no-size")
    if W is not None and (W <= 0 or H <= 0):
        flags["reject"].append("zero-size")
        W = None

    # walk tree
    drawables, visible = 0, 0
    has_transform = False
    all_pts, all_chords = [], []
    total_cmds = total_curves = total_lines = 0
    for el in root.iter():
        t = tag_of(el)
        if el.get("transform"):
            has_transform = True
        if t == "image":
            flags["reject"].append("raster")
        if t == "foreignobject":
            flags["reject"].append("foreignObject")
        if t == "text":
            if "live-text" not in flags["warn"]:
                flags["warn"].append("live-text")
        if t == "svg" and el is not root:
            flags["warn"].append("nested-svg")
        if t not in DRAWABLE:
            continue
        drawables += 1
        fill = style_get(el, "fill")
        stroke = style_get(el, "stroke")
        opacity = style_get(el, "opacity")
        display = style_get(el, "display")
        hidden = (display == "none" or (opacity is not None and float(NUM_RE.search(opacity)[0] if NUM_RE.search(opacity) else 1) == 0)
                  or (fill == "none" and stroke in (None, "none") and t == "path"))
        if not hidden:
            visible += 1
        if fill and fill != "none":
            feat["colors"].add(fill)
        if stroke and stroke != "none":
            feat["stroke"] = True
        if t == "path":
            feat["paths"] += 1
            d = el.get("d", "")
            c, pts, chords, ncv, nln = parse_paths(d)
            total_cmds += c
            total_curves += ncv
            total_lines += nln
            all_pts.extend(pts)
            all_chords.extend(chords)

    feat["cmds"] = total_cmds
    feat["gradient"] = "gradient" in lower
    if total_curves + total_lines:
        feat["curve_frac"] = total_curves / (total_curves + total_lines)

    if drawables == 0:
        flags["reject"].append("no-drawables")
    elif visible == 0:
        flags["reject"].append("all-invisible")

    # geometry vs canvas (skip when transforms could move things)
    if W is not None and all_pts and not has_transform:
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        bx0, bx1, by0, by1 = min(xs), max(xs), min(ys), max(ys)
        if bx1 < vx or bx0 > vx + W or by1 < vy or by0 > vy + H:
            flags["reject"].append("off-canvas")
        else:
            ix = max(0, min(bx1, vx + W) - max(bx0, vx))
            iy = max(0, min(by1, vy + H) - max(by0, vy))
            if W * H > 0 and (ix * iy) / (W * H) < 0.01:
                flags["warn"].append("low-occupancy")

    # auto-trace signature: many commands, dominated by tiny straight chords
    if W is not None and all_chords and total_cmds > 800:
        diag = (W * W + H * H) ** 0.5
        tiny = sum(1 for c in all_chords if c < diag * 0.005)
        if tiny / len(all_chords) > 0.7:
            flags["warn"].append("trace-suspect")
    if total_cmds > 5000:
        flags["warn"].append("node-heavy")
    if feat["paths"] > 500:
        flags["warn"].append("many-paths")

    flags["reject"] = list(dict.fromkeys(flags["reject"]))
    flags["warn"] = list(dict.fromkeys(flags["warn"]))
    return flags, feat


def tier_of(flags):
    if flags["reject"]:
        return "reject"
    if flags["warn"]:
        return "warn"
    return "clean"


def main():
    rows = pq.read_table(SRC).to_pylist()
    print(f"shard rows: {len(rows)}")

    # full-shard stats
    from collections import Counter
    tiers = Counter()
    flagc = Counter()
    for r in rows:
        fl, _ = analyze(r["Svg"])
        tiers[tier_of(fl)] += 1
        for f in fl["reject"]:
            flagc["R:" + f] += 1
        for f in fl["warn"]:
            flagc["W:" + f] += 1
    print("full shard:", dict(tiers))
    print(json.dumps(dict(flagc.most_common()), indent=2))

    # gallery over the same seeded 500
    random.seed(SEED)
    sample = random.sample(rows, N)
    cards = []
    stiers = Counter()
    for i, row in enumerate(sample):
        svg = row["Svg"]
        fl, ft = analyze(svg)
        tier = tier_of(fl)
        stiers[tier] += 1
        if len(svg) <= EMBED_CAP:
            b64 = base64.b64encode(svg.encode()).decode()
            img = f'<img loading="lazy" src="data:image/svg+xml;base64,{b64}" alt="">'
        else:
            img = '<div class="toobig">too large to embed</div>'
        badges = "".join(f'<span class="badge br">{f}</span>' for f in fl["reject"]) + \
                 "".join(f'<span class="badge bw">{f}</span>' for f in fl["warn"])
        kb = ft["len"] / 1024
        feats = []
        if ft["colors"]:
            feats.append(f'{len(ft["colors"])}col')
        if ft["gradient"]:
            feats.append("grad")
        if ft["stroke"]:
            feats.append("stroke")
        if ft["cmds"]:
            feats.append(f'{ft["curve_frac"]:.0%}curve')
        cards.append(
            f'<figure class="card t-{tier}">'
            f'<div class="stage">{img}</div>'
            f'<figcaption><span class="idx">#{i+1}</span> {kb:.1f}KB · {ft["paths"]}p/{ft["cmds"]}c'
            f'{" · " + " ".join(feats) if feats else ""}{badges}</figcaption></figure>'
        )

    pct = {k: f"{v} ({v/N:.0%})" for k, v in stiers.items()}
    top_flags = " · ".join(f"{k} {v}" for k, v in flagc.most_common(10))
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE} — gate v2 audit</title>
<style>
  :root {{ --bg:#f6f5f2; --panel:#fff; --ink:#1e1d23; --dim:#75737e; --accent:#b07a10; --grid:210px; }}
  * {{ box-sizing:border-box; margin:0; }}
  body {{ background:var(--bg); color:var(--ink); font:14px/1.5 "SF Mono",ui-monospace,Menlo,monospace; padding:28px; }}
  header {{ margin-bottom:20px; max-width:1050px; }}
  h1 {{ font-size:20px; font-weight:600; }} h1 span {{ color:var(--accent); }}
  .sub {{ color:var(--dim); margin-top:6px; font-size:13px; }}
  .statline {{ margin-top:14px; display:flex; gap:26px; flex-wrap:wrap; }}
  .stat b {{ display:block; font-size:22px; font-weight:600; }}
  .stat span {{ color:var(--dim); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }}
  .s-clean b {{ color:#2e8540; }} .s-warn b {{ color:#b07a10; }} .s-reject b {{ color:#c03434; }}
  .controls {{ margin:18px 0 22px; display:flex; gap:8px; }}
  .controls button {{ background:var(--panel); color:var(--ink); border:1px solid #ddd9d0; padding:6px 14px;
    border-radius:6px; font:inherit; font-size:12px; cursor:pointer; }}
  .controls button.on {{ border-color:var(--accent); color:var(--accent); }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(var(--grid),1fr)); gap:14px; }}
  .card {{ background:var(--panel); border:1px solid #e4e1da; border-radius:10px; overflow:hidden;
    display:flex; flex-direction:column; box-shadow:0 1px 3px rgba(30,29,35,.05); }}
  .card.t-warn {{ border-color:#e3c98a; }} .card.t-reject {{ border-color:#dfa0a0; }}
  .stage {{ aspect-ratio:1; display:flex; align-items:center; justify-content:center; padding:14px;
    background:repeating-conic-gradient(#ecebe7 0% 25%, #f9f8f6 0% 50%) 0 0/20px 20px; }}
  .stage img {{ max-width:100%; max-height:100%; }}
  .toobig {{ color:#c03434; font-size:12px; }}
  figcaption {{ padding:8px 10px; font-size:11px; color:var(--dim); border-top:1px solid #eeece7; }}
  .idx {{ color:var(--ink); }}
  .badge {{ display:inline-block; margin-left:6px; padding:1px 7px; border-radius:999px; font-size:10px; }}
  .badge.br {{ background:#f6dede; color:#a03636; }} .badge.bw {{ background:#f2ecd4; color:#8a7a1e; }}
  .hidden {{ display:none; }}
</style></head><body>
<header>
  <h1>{TITLE} <span>· gate v2 — structural analysis</span></h1>
  <div class="sub">XML tree + path-geometry parsing: visibility, canvas occupancy, trace signatures ·
  seed {SEED}, {N} of {len(rows):,} rows · full-shard tiers: clean {tiers["clean"]:,} · warn {tiers["warn"]:,} · reject {tiers["reject"]:,}</div>
  <div class="statline">
    <div class="stat s-clean"><b>{stiers["clean"]}</b><span>clean</span></div>
    <div class="stat s-warn"><b>{stiers["warn"]}</b><span>warn</span></div>
    <div class="stat s-reject"><b>{stiers["reject"]}</b><span>reject</span></div>
  </div>
  <div class="sub">top flags (full shard): {top_flags}</div>
</header>
<div class="controls">
  <button class="on" data-t="all">all {N}</button>
  <button data-t="clean">clean</button>
  <button data-t="warn">warn</button>
  <button data-t="reject">reject</button>
</div>
<div class="grid">{"".join(cards)}</div>
<script>
  document.querySelectorAll(".controls button").forEach(b => b.addEventListener("click", () => {{
    document.querySelectorAll(".controls button").forEach(x => x.classList.remove("on"));
    b.classList.add("on");
    const t = b.dataset.t;
    document.querySelectorAll(".card").forEach(c =>
      c.classList.toggle("hidden", t !== "all" && !c.classList.contains("t-" + t)));
  }}));
</script></body></html>"""
    OUT.write_text(doc)
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB) — sample tiers: {dict(stiers)}")


if __name__ == "__main__":
    main()
