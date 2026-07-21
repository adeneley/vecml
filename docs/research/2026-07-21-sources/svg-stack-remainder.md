# Source feasibility: the unminted remainder of svg-stack's clean tier

Date: 2026-07-21. Scope: one training-data source — the clean-tier SVGs from
`starvector/svg-stack` that our sha ledger has not yet sampled. Question: how many
net-new usable SVGs remain, at what quality, and what marginal value they buy the
Stage-1 cleanup UNet.

---

## VERDICT

**Feasible but low value. Net-new usable SVGs for training: ~226,478 (gross clean
remainder ~299,699, of which ~73,221 are held-out val/test-split clean that must not
be minted into training data).**

The remainder is technically trivial to mint — the full clean tier is already on disk,
the ledger already tracks what is used, and `sample_src.py` already excludes it — but it
is *volume-only, more-of-the-same icon distribution* fuel. Our measured data-scaling
exponent on this exact distribution is ~0.10 (val loss 0.00782 at 100k → 0.00630 at 927k),
and this remainder grows the unique-count by only ~19% (1.165M → 1.391M). Predicted
val-loss reduction is **<2%**, and it touches none of the three measured gaps
(typography, fine-line strokes, region boundaries) that actually cap the end-to-end bench.
It does not warrant a dedicated mint. Fold it in for free only if a full-clean-tier retrain
is being run anyway.

## The census, verified

| Quantity | Value | Source |
|---|---|---|
| svg-stack total rows | **2,283,875** | HF dataset card (verified 2026-07-21) == local `manifest.jsonl` line count == `summary.json` |
| — train / val / test rows | 2.17M / 108k / 5,710 | HF card (splits) |
| Gate-v2 clean tier | **1,464,723** (64.1% survival) | `datasets/svg-stack-labelled/summary.json`, confirmed by full manifest tally |
| — clean train | 1,391,502 | manifest by-split tally |
| — clean val | 69,565 | manifest by-split tally (held out) |
| — clean test | 3,656 | manifest by-split tally (held out) |
| Gate warn / reject | 723,800 / 95,352 | summary.json |
| Unique SVGs already minted (ledger) | **1,165,024** | `used-shas` ledger, deduped line count |
| **Gross clean remainder** | **299,699** | 1,464,723 − 1,165,024 |
| Less held-out val+test clean | −73,221 | 69,565 + 3,656 |
| **Net training-usable remainder** | **~226,478** | == clean/train (1,391,502) − used (1,165,024) |

Notes on the numbers:
- The HF card total (2,283,875) matches our local download exactly, so **we already hold the
  entire dataset** — there is no un-downloaded portion of svg-stack to fetch. "Remainder"
  means un-*sampled*, not un-*downloaded*.
- The two figures resolve consistently: net training-usable remainder computed as
  (gross clean − held-out val/test clean) equals (clean/train − used ledger), which holds
  only if the ledger is train-split-only. See RISKS.
- fuel.md's older "~2M / mine ~927k clean tier" line predates this build; the current
  labelled build is the authority (clean = 1,464,723 from 2,283,875).

## Did our sampling skim the cream?

No. `sample_src.py` builds its pick list by a single seeded shuffle of the sorted clean-tier
walk, filtered only by the exclusion ledger — a **uniform random sample with no quality
ranking**. The remainder is therefore statistically identical in distribution and quality to
the 1.165M already used. Two consequences: (1) quality is comparable, not degraded — no cream
was skimmed; (2) equally, there is **no untapped high-value subset hiding in the remainder** —
it is the same icon-dominant, flat-few-colour distribution, weak on exactly the gaps
(typography, hairline strokes) that fuel.md and recipe.md identify as the real bottleneck.

## Marginal value

Grounded in `../2026-07-20-deep-research/recipe.md` §2.2/§2.4:

- The fitted icon-data exponent is ~0.097 (recipe.md §2.2). Growing unique count 1.165M →
  1.391M is a factor of 1.194, giving a predicted loss multiplier of 1.194^−0.097 ≈ 0.983 —
  a **~1.7% val-loss reduction**, and the end-to-end ΔE gain is smaller still because the added
  data sits on a distribution orthogonal to the measured gaps.
- recipe.md's own framing: even a *10×* icon mint buys ≤~19% val-loss reduction on this shallow
  branch; a 0.19× mint is deep in diminishing returns.
- recipe.md Rec 3 is explicit: "Mint for gap coverage (typography, fine-line, boundaries),
  **not more icons**." This remainder is definitionally more icons.

## PLAN (if minted — near-zero effort, low priority)

Only justified as a free add-on to an already-planned full-clean-tier retrain, not as a
standalone spend.

1. Sample the remainder with existing tooling (no new code):
   `sample_src.py --clean <clean-tier> --n 226478 --exclude-shas <ledger> --record-shas`.
   With the val/test-split clean SVGs excluded (see RISKS) this drains the train-split clean
   tier to zero unminted. Effort: minutes; pure CPU file copy.
2. Wreck the pairs with `wreck.py`. At the measured M5 rate (10k pairs / 9 min from the
   overnight flight report), ~226k pairs ≈ **~3.4 h local**. Fresh JPEG/noise/blur draws per
   epoch (recipe.md §2.4) already make re-augmentation cheap.
3. Fold into the next Stage-1 run, taking the corpus from ~1.165M → ~1.391M unique (one full
   epoch of the entire train-split clean tier). Re-fit the loss intercept since the mix shifts
   slightly (recipe.md §2.4).

**Effort:** <0.5 engineer-day (all existing scripts). **Compute cost:** wreck ~3–4 h local
CPU/MPS (≈$0); a ~1.39M-scale GPU run ≈ **$20–35** (scaling the $15–30/927k-run figure from
`../2026-07-21-intent-priors/README.md`) — but that cost is the retrain itself, not this
remainder; the remainder's *marginal* cost folded into a planned retrain is effectively the
wreck time only.

## RISKS

1. **Opportunity cost, not compute cost.** The real risk is spending a retrain slot on
   volume-only fuel instead of gap fuel. recipe.md Run C (typography/fine-line/boundary mint)
   is predicted to move the 1.52 bench; this remainder is not. Treat it as ballast folded into
   a run that exists for another reason, never as the reason for a run.
2. **Val/test hygiene.** The staged clean tier mixes all three splits into one sha-sharded
   tree, and `sample_src.py` globs across it. If the 1.165M ledger was ever sampled from that
   mixed tree without excluding the 73,221 val/test-split clean shas, some eval SVGs may
   already have leaked into training — which would also mean the "net-new = 226,478" figure is
   slightly off. Before any further mint, intersect the ledger against the manifest's
   val/test-split shas to confirm zero overlap, and add a split filter to the sample step.
   (This is a pre-existing pipeline concern surfaced here, not introduced by minting the
   remainder.)
3. **Distribution drift is nil, which is the point.** Because the remainder is a uniform sample
   of the same tier, it will not shift the corpus mix toward the gaps; do not expect per-source
   val slices to move. If anything it slightly *deepens* the existing icon over-representation
   the mixture-ratio findings (fuel.md §2.6) warn against.
4. **Warn tier is not a rescue.** If more volume from svg-stack is ever wanted, the 723,800
   warn-tier SVGs dwarf this remainder — but they carry live-text/trace-suspect/editor-junk
   flags and would need per-flag reclamation, a different and larger project than this.
