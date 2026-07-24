# Wrecker v2 — implementation notes

Companion to [`README.md`](./README.md). The README specifies the v2 synthesis;
this records what shipped in code, the default parameters, and how to operate it.

Code:

- `src/vecml/degrade/wreck_v2.py` — families, ops, recipe sampling, replay.
- `src/vecml/degrade/pipeline.py` — `wreck_svg(..., wreck="v1"|"v2")`.
- `src/vecml/degrade/calibrate.py` + `scripts/wreck_calibrate.py` — forensic diagnostics.
- `scripts/wreck.py --wrecker {v1,v2}` — the training-folder CLI.
- `scripts/wreck_preview.py` — per-family contact sheets for visual sanity.
- `tests/test_wreck_v2.py` — replay, v1 regression, monotonicity, favicon fix.

## Architecture

v1 draws independent photometric ops from a flat pool at one of three narrow
severity tiers. v2 draws a **capture-path family** first, then runs that family's
ordered, parameter-correlated op bundle at a single **continuous per-sample
global severity** `s in [0, 1]`. `s` is drawn as: 12% exact identity (`s=0`,
pure passthrough — the "don't fix what isn't broken" lesson), otherwise
`Beta(2, 3)` (mass in the low-mid, thin brutal tail). Ranges written `[lo→hi]`
interpolate `lo` at `s=0` to `hi` at `s=1`.

### Families

| Family | Capture path | Ordered ops (probabilistic) |
|---|---|---|
| `jpeg_chain` | emailed / WhatsApped / re-saved logos | optional blur/rescale → n-pass JPEG (2–4, shifted 8×8 grid, per-pass quality/codec/subsampling) |
| `web_upscale` | upsampled favicons / header images | shrink → low-Q JPEG **at the small size** → optional palette crush → enlarge → optional final JPEG |
| `scan` | scanned letterheads / cards | paper → optional bleed-through → ink-bleed → halftone/descreen → geometric skew → illumination → sensor noise → JPEG |
| `office_roundtrip` | logo through Word / PowerPoint | optional rescale → posterize/dither → WebP/PNG re-quantise → JPEG |
| `phone_photo` | business card on a desk | paper → perspective warp → illumination + cast shadow → motion/defocus blur → shot noise → 4:2:0 JPEG |
| `lowres_pdf` | flattened at low resolution | downscale/upscale → gaussian blur → optional sinc ringing → JPEG |
| `mild` | lightly-touched intake | one or two light ops → high-quality JPEG |

A cross-family finishing **sinc-ringing** pass fires with p=0.6 on every family
except `mild`, order-swapped with the family's final op — ringing concentrates
at the hard edges every family produces, the most direct lever for the
fineline/typography weakness.

`geometric_warp` (in `scan` and `phone_photo`) returns the homography it applied
so the pipeline warps the label map by the identical transform and writes a
per-variant `labels_XX.png` (matched-warp label invariant, README 2.4 option A).
Warp magnitudes are kept modest (skew ≤4°, gentle perspective) so QC
reconstruction still holds.

## Default family mix and the competitive rationale

```
jpeg_chain 0.20  web_upscale 0.22  office_roundtrip 0.18  lowres_pdf 0.14
scan 0.12  phone_photo 0.08  mild 0.06
```

Defined in `wreck_v2.DEFAULT_MIX` — documented and tunable, not hard-coded magic.
The README ships a provisional prior keyed to the print-shop taxonomy; this
default tilts it toward the semantically-hard, differentiating families. Fresh
competitive probing of the reference commercial tracer (runs/SCOREBOARD.md, 21
Jul) shows it already recovers ~98% of gaussian noise and ~89% of blur (table
stakes) but only ~38% of downscale/upscale, ~0% of posterize/banding (it traces
the bands as content), and drops 0.5px hairlines entirely — so a trained model
differentiates precisely where that pipeline fails. The mix therefore
over-represents the families that produce banding / palette crush / aggressive
downscale / hairline-eroding blur+downscale (`web_upscale`, `office_roundtrip`,
`lowres_pdf`) and weights down the realism/geometry families (`scan`,
`phone_photo`) until the corpus tag sets the true shares. Overwrite the mix from
the corpus tag (README 3, Step 0) before the real mint; the whole recipe hangs
on it.

## Parameter ranges (starting anchors, re-fit by calibration)

- JPEG quality: floor ~55 (per-pass, s-scaled ceiling to 95). **Not** hard-driven
  to 8 — README 6 forbids a brutal floor until Step 1 confirms QF<30 in the
  intake. Adjust the `_scaled_u` floors in the family builders once measured.
- JPEG passes: 2–4, weighted toward more passes as `s` rises; misaligned 8×8
  grid shift 0–7px between passes.
- `web_upscale` small size: `[96→16]` px (calibrate to measured upsample factors).
- Blur: gaussian `[0.4→2.5]`; anisotropic/motion/defocus for photo/chain.
- Noise: additive gaussian std `[2→10]` (scan), Poisson shot noise for sensors.
- Geometric: skew `±[0.5→4]°`, perspective jitter up to 3% of the frame.

## Per-sample logging

Each v2 variant records in `meta.json`: `recipe_version` (`wreck-v2`), `family`,
`global_severity`, `label_file`, and `ops` — every op with its fully resolved
concrete parameters (JPEG qualities, subsampling, codecs, grid shifts, kernel
sizes, warp matrix, LPI + screen angles). This log is the join key for
calibration and the answer key for the geometric label warp.

## Determinism and replay

v2 keeps v1's deterministic-from-seed property and adds byte-exact replay from
logged params. Each variant seed spawns **two** rng streams (`variant_rngs`):
stream 0 drives recipe sampling, stream 1 drives per-pixel application (noise
draws). Splitting them means a sample replays from its logged params alone —
rebuild the recipe dict, re-derive stream 1 from the same variant seed, and the
output is byte-identical without re-running the sampler (tested in
`test_replay_from_logged_params_identical_bytes`). All randomness goes through
numpy; no op uses the cv2 global RNG, so no `cv2.setRNGSeed` coupling is needed.

## Calibration

`scripts/wreck_calibrate.py --wrecked <v2 tree> --real <dir>` computes per-image
forensic features — estimated noise sigma (`skimage.restoration.estimate_sigma`,
with a numpy MAD fallback if scikit-image is absent), JPEG quality inverted from
the luminance quant table, edge-ringing energy — and reports per-family
synthetic distributions against the real set. JPEG-quality forensics read the
file's own quant tables, so they populate on real `.jpg` intake (and on
synthetic only if variants are saved as JPEG rather than the default PNG). The
heavier C2ST/proxy-A-distance + KID sim-to-real harness is a documented seam
(`calibrate.distribution_distance`, raises `NotImplementedError`).

## Operating it

```
# preview (contact sheets, gitignored)
uv run python scripts/wreck_preview.py --src data/audit-500-src --n 3

# mint a v2 shard (default is still v1)
uv run python scripts/wreck.py --in <svgs> --out data/wrecked-v2 --wrecker v2

# calibrate against real damaged files
uv run python scripts/wreck_calibrate.py --wrecked data/wrecked-v2 --real <dir>
```

### Flipping the default to v2

The CLI (`--wrecker`) and `wreck_svg(wreck=...)` both default to `v1` so no
in-flight mint changes silently. To flip after the A/B clears: change the
`default="v1"` on the `--wrecker` argument in `scripts/wreck.py` and the
`wreck: str = "v1"` default in `wreck_svg`. v1 stays frozen and importable
(`sample_recipe`/`apply_recipe`) so existing training sets remain replayable and
comparable; a run's scorecard should always name its wrecker version.

## Consciously deferred

- **Corpus tag / weight fit** — the mix and range endpoints are provisional
  until the 400 real pairs are tagged (README 3, Step 0/4). This is calibration
  input, not a code gap.
- **C2ST/KID sim-to-real harness** — seam left in `calibrate.py`; per-image
  diagnostics ship now.
- **AVIF/HEIF codec** — JPEG + WebP implemented; AVIF omitted to avoid a heavy
  optional decoder dependency (README named it low-prob/optional).
- **Per-variant warped clean target** — geometric families warp the input and
  the label map; the cleanup *target* stays the shared unwarped `clean.png`. The
  matched-vs-deskew task decision (README open q 4) is unresolved, so this waits
  on that call. Warp magnitudes are held modest meanwhile.
