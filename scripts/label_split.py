"""Label the svg-stack parquet shards with gate v2, split into tier folders.

Promoted from datasets/svg-stack-labelled/tools/ (where it produced the
1,464,723-SVG clean tier from 2,283,875 rows) and parameterised so a pod can
run it against a freshly downloaded corpus. Pure CPU, scales with cores.

Output layout:
  <out>/clean/<sha[:2]>/<sha>.svg    (likewise warn/, reject/)
  <out>/manifest.jsonl               one line per SVG: tier, flags, features, split

  uv run python scripts/label_split.py --data /tmp/corpus/raw --out /tmp/corpus/labelled
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gate2 import analyze, tier_of  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402

TIERS = ("clean", "warn", "reject")
_OUT = None  # set per-process via initializer (ProcessPoolExecutor pickling)


def _init(out_str):
    global _OUT
    _OUT = Path(out_str)


def process_shard(shard_path_str):
    shard = Path(shard_path_str)
    split = shard.name.split("-")[0]  # train / val / test
    rows = pq.read_table(shard).to_pylist()
    counts = {t: 0 for t in TIERS}
    manifest_path = _OUT / "manifests" / f"{shard.stem}.jsonl"
    with open(manifest_path, "w") as mf:
        for row in rows:
            sha = row["Filename"].removesuffix(".svg")
            svg = row["Svg"]
            try:
                flags, feat = analyze(svg)
            except Exception:
                flags = {"reject": ["analyzer-error"], "warn": []}
                feat = {"len": len(svg), "paths": 0, "cmds": 0, "colors": set(),
                        "gradient": False, "stroke": False, "curve_frac": 0.0}
            tier = tier_of(flags)
            counts[tier] += 1
            sub = _OUT / tier / sha[:2]
            sub.mkdir(parents=True, exist_ok=True)
            (sub / f"{sha}.svg").write_text(svg)
            mf.write(json.dumps({
                "sha": sha, "tier": tier, "split": split,
                "reject": flags["reject"], "warn": flags["warn"],
                "len": feat["len"], "paths": feat["paths"], "cmds": feat["cmds"],
                "colors": len(feat["colors"]), "gradient": feat["gradient"],
                "stroke": feat["stroke"], "curve_frac": round(feat["curve_frac"], 3),
            }) + "\n")
    return shard.name, len(rows), counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir holding the *.parquet shards")
    ap.add_argument("--out", required=True, help="labelled output root")
    ap.add_argument("--expect-shards", type=int, default=14,
                    help="refuse to run on a partial download (0 disables)")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)

    by_name = {}
    for p in sorted(data.rglob("*.parquet")):
        if re.match(r"(train|val|test)-\d{5}-of-\d{5}\.parquet$", p.name):
            by_name.setdefault(p.name, p)
    shards = sorted(by_name.values())
    if args.expect_shards and len(shards) != args.expect_shards:
        sys.exit(f"expected {args.expect_shards} shards under {data}, "
                 f"found {len(shards)}: {sorted(by_name)}")
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    for t in TIERS:
        (out / t).mkdir(exist_ok=True)

    total = {t: 0 for t in TIERS}
    n_rows = 0
    workers = min(len(shards), max(2, (os.cpu_count() or 4) - 2))
    print(f"{len(shards)} shards, {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(str(out),)) as ex:
        futs = {ex.submit(process_shard, str(s)): s for s in shards}
        for fut in as_completed(futs):
            name, rows, counts = fut.result()
            n_rows += rows
            for t in TIERS:
                total[t] += counts[t]
            print(f"done {name}: {rows} rows {counts}", flush=True)

    with open(out / "manifest.jsonl", "w") as merged:
        for m in sorted((out / "manifests").glob("*.jsonl")):
            merged.write(m.read_text())

    print(f"\nTOTAL {n_rows} rows: {total}")
    (out / "summary.json").write_text(json.dumps({"rows": n_rows, "tiers": total}, indent=2))


if __name__ == "__main__":
    main()
