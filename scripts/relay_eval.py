"""The relay experiment: does neural cleanup improve the final vectorization?

For each never-seen test pair (wrecked, clean):
  damage    dE(wrecked, clean)                      how bad the input is
  cleanup   dE(model(wrecked), clean)               what the model recovers
  raw path  dE(render(engine(wrecked)), clean)      vectorize the damage
  relay     dE(render(engine(model(wrecked))), clean)  the thesis

Run against one or more checkpoints and one or more engines (rust | vtracer).
Writes per-image artifacts + summary.json under --out.

  uv run python scripts/relay_eval.py --data data/relay-test \
      --ckpt runs/overfit1000/best.pt --out runs/relay/mac1000
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vecml.data.pairs import list_sample_dirs  # noqa: E402
from vecml.models.unet import DualHead, UNet  # noqa: E402

RUST = "/Users/aden/development/vectorizer/target/release/vectorize"


def load_rgb(path, size=256):
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def srgb_to_lab(rgb):
    """Vectorized sRGB [0,1] -> CIELAB (D65)."""
    r = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = r @ m.T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def de(a, b):
    """Per-pixel deltaE76 stats between two RGB [0,1] images."""
    d = np.linalg.norm(srgb_to_lab(a) - srgb_to_lab(b), axis=-1)
    return {"mean": float(d.mean()), "p99": float(np.percentile(d, 99))}


def vec_rust(png, svg):
    subprocess.run([RUST, str(png), "-o", str(svg)], check=True,
                   capture_output=True, timeout=120)


def vec_rust_labels(rgb_png, labels_png, svg, min_area):
    """Label-input mode: hand the engine the class map directly instead of
    letting it quantize. `rgb_png` supplies the colours the residue-mean fills
    read; `labels_png` is a greyscale PNG whose pixel value is the class id."""
    cmd = [RUST, str(rgb_png), "--labels", str(labels_png), "-o", str(svg)]
    if min_area > 0:
        cmd += ["--min-area", str(min_area)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)


def vec_vtracer(png, svg):
    import vtracer
    vtracer.convert_image_to_svg_py(str(png), str(svg))


def render_svg(svg, size=256):
    from vecml.degrade.renderer import render_svg as _render
    return np.asarray(_render(svg, size), dtype=np.float32) / 255.0


ENGINES = {"rust": vec_rust, "vtracer": vec_vtracer}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engines", nargs="*", default=["rust", "vtracer"])
    ap.add_argument("--size", type=int, default=256)
    # Despeckle threshold for the rust label-input variants. Model label maps
    # still carry a little speckle; dissolving sub-threshold components keeps it
    # from spawning spurious regions. 0 disables.
    ap.add_argument("--min-area", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    state = torch.load(args.ckpt, map_location="cpu")
    sd = state["model"]
    n_classes = state.get("n_classes", 0)
    # Width is not in the payload; the first encoder conv's out-channels is
    # the UNet base (32 default, 48+ for the wider variants).
    base = sd["enc1.body.0.weight"].shape[0]
    has_labels = any(k.startswith("head.label") for k in sd)
    if has_labels:
        model = UNet(base=base, head=lambda c: DualHead(c, n_classes))
    else:
        model = UNet()
    model.load_state_dict(sd)
    model.eval()

    rows = []
    for d in list_sample_dirs(args.data):
        clean = load_rgb(d / "clean.png", args.size)
        wrecked = load_rgb(d / "wrecked_00.png", args.size)

        with torch.no_grad():
            x = torch.from_numpy(wrecked).permute(2, 0, 1)[None]
            outp = model(x)
        flat = None
        labels = None
        if has_labels:
            pred = outp["rgb"][0].permute(1, 2, 0).clamp(0, 1).numpy()
            labels = outp["logits"][0].argmax(0).numpy()
            # Flatten: paint each label region with the median colour the RGB
            # head predicted for it. Zero anti-aliasing, hard edges - the
            # input format the engines can't get from a raster any other way.
            flat = np.empty_like(pred)
            for slot in np.unique(labels):
                mask = labels == slot
                flat[mask] = np.median(pred[mask], axis=0)
        else:
            pred = outp[0].permute(1, 2, 0).clamp(0, 1).numpy()

        img_dir = out / d.name
        img_dir.mkdir(exist_ok=True)
        Image.fromarray((pred * 255).astype(np.uint8)).save(img_dir / "cleaned.png")
        if flat is not None:
            Image.fromarray((flat * 255).astype(np.uint8)).save(img_dir / "flat.png")

        row = {"name": d.name,
               "damage": de(wrecked, clean),
               "cleanup": de(pred, clean)}
        if flat is not None:
            row["cleanup_flat"] = de(flat, clean)

        variants = [("raw", d / "wrecked_00.png"),
                    ("relay", img_dir / "cleaned.png")]
        if flat is not None:
            variants.append(("flat", img_dir / "flat.png"))
        for eng, fn in ((e, ENGINES[e]) for e in args.engines):
            for label, src in variants:
                svg = img_dir / f"{eng}_{label}.svg"
                try:
                    fn(src, svg)
                    rendered = render_svg(svg, args.size)
                    Image.fromarray((rendered * 255).astype(np.uint8)).save(
                        img_dir / f"{eng}_{label}.png")
                    row[f"{eng}_{label}"] = de(rendered, clean)
                except Exception as exc:  # noqa: BLE001 - one bad file must not kill the sweep
                    row[f"{eng}_{label}"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}

        # Label-input mode (the thesis under test): hand the rust engine the
        # model's predicted class map directly instead of letting it quantize.
        #   rust_labels     model labels    + model-cleaned RGB fills
        #   rust_labels_gt  ground-truth    + clean-render RGB fills  (ceiling)
        # The ceiling feeds perfect labels and perfect colours, isolating the
        # engine's integration error from the model's label/colour error.
        if "rust" in args.engines and labels is not None:
            ids_png = img_dir / "label_ids.png"
            Image.fromarray(labels.astype(np.uint8), mode="L").save(ids_png)
            label_runs = [
                ("rust_labels", img_dir / "cleaned.png", ids_png),
                ("rust_labels_gt", d / "clean.png", d / "labels.png"),
            ]
            for tag, rgb_src, lbl_src in label_runs:
                svg = img_dir / f"{tag}.svg"
                try:
                    vec_rust_labels(rgb_src, lbl_src, svg, args.min_area)
                    rendered = render_svg(svg, args.size)
                    Image.fromarray((rendered * 255).astype(np.uint8)).save(
                        img_dir / f"{tag}.png")
                    row[tag] = de(rendered, clean)
                except Exception as exc:  # noqa: BLE001
                    row[tag] = {"error": f"{type(exc).__name__}: {exc}"[:200]}

        rows.append(row)
        print(f"{d.name[:12]} damage={row['damage']['mean']:.2f} "
              f"cleanup={row['cleanup']['mean']:.2f}", flush=True)

    def agg(key):
        vals = [r[key]["mean"] for r in rows if "mean" in r.get(key, {})]
        return {"mean": float(np.mean(vals)), "n": len(vals)} if vals else None

    summary = {"ckpt": args.ckpt, "n_images": len(rows),
               "damage": agg("damage"), "cleanup": agg("cleanup"),
               "cleanup_flat": agg("cleanup_flat")}
    for e in args.engines:
        summary[f"{e}_raw"] = agg(f"{e}_raw")
        summary[f"{e}_relay"] = agg(f"{e}_relay")
        summary[f"{e}_flat"] = agg(f"{e}_flat")
    if "rust" in args.engines:
        summary["rust_labels"] = agg("rust_labels")
        summary["rust_labels_gt"] = agg("rust_labels_gt")
    (out / "summary.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
