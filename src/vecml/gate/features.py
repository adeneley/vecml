"""Structural feature extraction for autotrace detection.

A machine trace (potrace / autotrace / vtracer / Adobe Image Trace) turns a
raster into vector paths by walking pixel boundaries. That process leaves a
consistent structural fingerprint that born-vector art does not share:

  * everything is a cubic-Bezier `<path>`; no `<rect>`/`<circle>`/`<line>`,
    no elliptical-arc (`A`) commands, no stroked centre-lines;
  * coordinates are full-precision floats straight off the tracer, never the
    round / grid-snapped numbers a human or a design tool emits;
  * contours are long chains of short, similar-length segments that turn by
    small angles (a smooth curve approximated by many pieces), rather than a
    few long segments meeting at sharp corners;
  * no editor namespaces, named ids, `<defs>`/`<use>`, or live `<text>`.

`extract_features` parses one SVG string and returns a flat float dict. The
path tokenizer is a hardened copy of the one in `scripts/gate2.py` extended to
record per-segment turning angles and raw coordinate precision, which gate2
never looked at. Parsing is pure-stdlib (ElementTree + regex): no svgelements,
lxml, or cairo dependency, so it runs at corpus scale.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET

# --- tokenizers ------------------------------------------------------------

_NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_DTOK_RE = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|([-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?)")
_PARAMS = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}

_DRAWABLE = {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "text", "use", "image"}
_PRIMITIVE = {"rect", "circle", "ellipse", "line", "polyline", "polygon"}
_EDITOR_NS = ("inkscape", "sodipodi", "adobe", "illustrator", "sketch", "figma", "affinity", "corel")
# generic ids a tool auto-emits carry no born-vector signal; anything else does
_GENERIC_ID_RE = re.compile(r"^(layer|path|g|svg|shape|clip|mask|use|_|xmlid|surface|glyph)[-_]?\d*$", re.I)


def _tag_of(el) -> str:
    return el.tag.split("}")[-1].lower() if isinstance(el.tag, str) else ""


def _decimals(tok: str) -> int:
    """Count fractional digits in a raw numeric token (2 for '1.25', 0 for '1')."""
    if "e" in tok or "E" in tok:
        return 6  # scientific notation is inherently full-precision
    dot = tok.find(".")
    return 0 if dot < 0 else len(tok) - dot - 1


def _parse_d(d: str):
    """Parse a path `d`. Returns per-path geometry stats accumulated into lists.

    Yields, appended onto the caller's accumulators:
      chords    chord length of every drawn segment
      turns     absolute turning angle (rad) between consecutive segments
      decimals  fractional-digit count of every coordinate token
    plus command tallies. Mirrors gate2's clamp on runaway coords.
    """
    toks = _DTOK_RE.findall(d)
    stream = []
    for c, num in toks:
        if c:
            stream.append(("cmd", c))
        else:
            stream.append(("num", num))

    cx = cy = sx = sy = 0.0
    chords, turns, decimals = [], [], []
    n_cmds = n_curves = n_lines = n_arcs = 0
    axis_lines = 0
    cmd = None
    prev_dx = prev_dy = None  # heading of previous segment, for turning angle
    i = 0
    while i < len(stream):
        kind, val = stream[i]
        if kind == "cmd":
            cmd = val
            i += 1
            if cmd.upper() == "Z":
                dx, dy = sx - cx, sy - cy
                chords.append(abs(dx) + abs(dy))
                cx, cy = sx, sy
                n_cmds += 1
                prev_dx = prev_dy = None
                continue
        elif cmd is None:
            i += 1
            continue
        need = _PARAMS[cmd.upper()]
        args = []
        raw = []
        while len(args) < need and i < len(stream) and stream[i][0] == "num":
            raw.append(stream[i][1])
            args.append(max(-1e9, min(1e9, float(stream[i][1]))))
            i += 1
        if len(args) < need:
            break
        for t in raw:
            decimals.append(_decimals(t))
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
            cx = args[need - 2] + (cx if rel else 0)
            cy = args[need - 1] + (cy if rel else 0)
        dx, dy = cx - px, cy - py
        seglen = math.hypot(dx, dy)
        chords.append(seglen)
        if seglen > 1e-9:
            if prev_dx is not None:
                pl = math.hypot(prev_dx, prev_dy)
                if pl > 1e-9:
                    cosv = (dx * prev_dx + dy * prev_dy) / (seglen * pl)
                    turns.append(math.acos(max(-1.0, min(1.0, cosv))))
            prev_dx, prev_dy = dx, dy
        n_cmds += 1
        if u in ("C", "S", "Q", "T"):
            n_curves += 1
        elif u == "A":
            n_arcs += 1
            n_curves += 1
        elif u in ("L", "H", "V"):
            n_lines += 1
            if u in ("H", "V") or abs(dx) < 1e-6 or abs(dy) < 1e-6:
                axis_lines += 1
        if u == "M":
            sx, sy = cx, cy
            cmd = "l" if rel else "L"  # implicit repeats of M draw lines
    return {
        "chords": chords, "turns": turns, "decimals": decimals,
        "n_cmds": n_cmds, "n_curves": n_curves, "n_lines": n_lines,
        "n_arcs": n_arcs, "axis_lines": axis_lines,
    }


def _stats(xs):
    if not xs:
        return 0.0, 0.0
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / n
    return m, math.sqrt(var)


# --- feature order (stable; classifier relies on it) -----------------------

FEATURE_NAMES = [
    "path_frac",           # paths / drawables (traced ~1.0)
    "prim_frac",           # primitive shapes / drawables (born-vector > 0)
    "has_arc",             # any elliptical-arc command present
    "curve_frac",          # curve cmds / (curve+line) cmds
    "axis_frac",           # axis-aligned lines / line cmds
    "cmds_per_path",       # mean commands per path
    "n_paths",
    "frac_tiny_chords",    # chords < 0.5% of canvas diagonal
    "chord_cv",            # coeff. of variation of chord length (uniformity)
    "mean_turn",           # mean |turning angle| between segments (rad)
    "frac_small_turn",     # fraction of turns < 10 degrees (contour jitter)
    "mean_decimals",       # mean fractional-digit count of coordinates
    "frac_lowprec",        # fraction of coords with <=1 decimal place
    "distinct_fills",
    "has_stroke",
    "has_gradient",
    "has_editor_ns",
    "has_defs_use",
    "has_text",
    "has_named_id",
]


def extract_features(svg: str) -> dict:
    """Parse an SVG string into the flat feature dict keyed by FEATURE_NAMES.

    Never raises on malformed input: an unparseable file yields the zero vector
    plus ``parse_error=1`` so callers can route it however they like.
    """
    feat = {k: 0.0 for k in FEATURE_NAMES}
    feat["parse_error"] = 0.0
    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        feat["parse_error"] = 1.0
        return feat

    lower = svg.lower()
    feat["has_gradient"] = 1.0 if "gradient" in lower else 0.0
    feat["has_editor_ns"] = 1.0 if any(ns in lower for ns in _EDITOR_NS) else 0.0
    feat["has_defs_use"] = 1.0 if ("<defs" in lower or "<use" in lower or "<symbol" in lower) else 0.0

    drawables = paths = primitives = 0
    stroked = 0
    fills = set()
    has_text = False
    has_named_id = False
    all_chords, all_turns, all_decimals = [], [], []
    tot_cmds = tot_curves = tot_lines = tot_arcs = tot_axis = 0

    for el in root.iter():
        t = _tag_of(el)
        idv = el.get("id")
        if idv and not _GENERIC_ID_RE.match(idv.strip()):
            has_named_id = True
        if t == "text":
            has_text = True
        if t not in _DRAWABLE:
            continue
        drawables += 1
        if t in _PRIMITIVE:
            primitives += 1
        fill = el.get("fill")
        if fill is None:
            m = re.search(r"fill\s*:\s*([^;]+)", el.get("style", ""))
            fill = m.group(1).strip() if m else None
        stroke = el.get("stroke")
        if stroke is None:
            m = re.search(r"stroke\s*:\s*([^;]+)", el.get("style", ""))
            stroke = m.group(1).strip() if m else None
        if fill and fill != "none":
            fills.add(fill)
        if stroke and stroke != "none":
            stroked += 1
        if t == "path":
            paths += 1
            g = _parse_d(el.get("d", ""))
            all_chords.extend(g["chords"])
            all_turns.extend(g["turns"])
            all_decimals.extend(g["decimals"])
            tot_cmds += g["n_cmds"]
            tot_curves += g["n_curves"]
            tot_lines += g["n_lines"]
            tot_arcs += g["n_arcs"]
            tot_axis += g["axis_lines"]

    # canvas diagonal for chord normalization
    vb = root.get("viewBox")
    W = H = None
    if vb:
        v = _NUM_RE.findall(vb)
        if len(v) == 4:
            W, H = float(v[2]), float(v[3])
    if W is None:
        try:
            W = float(_NUM_RE.search(root.get("width", "") or "")[0])
            H = float(_NUM_RE.search(root.get("height", "") or "")[0])
        except (TypeError, IndexError):
            W = H = None
    diag = math.hypot(W, H) if (W and H and W > 0 and H > 0) else None

    feat["n_paths"] = float(paths)
    feat["path_frac"] = paths / drawables if drawables else 0.0
    feat["prim_frac"] = primitives / drawables if drawables else 0.0
    feat["has_arc"] = 1.0 if tot_arcs else 0.0
    feat["curve_frac"] = tot_curves / (tot_curves + tot_lines) if (tot_curves + tot_lines) else 0.0
    feat["axis_frac"] = tot_axis / tot_lines if tot_lines else 0.0
    feat["cmds_per_path"] = tot_cmds / paths if paths else 0.0
    feat["distinct_fills"] = float(len(fills))
    feat["has_stroke"] = 1.0 if stroked else 0.0
    feat["has_text"] = 1.0 if has_text else 0.0
    feat["has_named_id"] = 1.0 if has_named_id else 0.0

    if diag and all_chords:
        feat["frac_tiny_chords"] = sum(1 for c in all_chords if c < diag * 0.005) / len(all_chords)
    m, s = _stats(all_chords)
    feat["chord_cv"] = (s / m) if m > 1e-9 else 0.0
    tm, _ = _stats(all_turns)
    feat["mean_turn"] = tm
    if all_turns:
        thr = math.radians(10)
        feat["frac_small_turn"] = sum(1 for a in all_turns if a < thr) / len(all_turns)
    dm, _ = _stats(all_decimals)
    feat["mean_decimals"] = dm
    if all_decimals:
        feat["frac_lowprec"] = sum(1 for d in all_decimals if d <= 1) / len(all_decimals)

    return feat


def feature_vector(feat: dict):
    return [float(feat.get(k, 0.0)) for k in FEATURE_NAMES]
