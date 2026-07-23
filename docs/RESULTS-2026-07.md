# Results — July 2026 three-way run

Three models, identical items and seeds, one overnight session on a single
4x 96 GB Blackwell node (sovereign/self-hosted; temp 0, concurrency 1,
single-stream). Serving: GLM-5.2 UD-Q3_K_XL (343 GB, llama.cpp, layer-split,
max reasoning effort) · DeepSeek V4-Flash (W4A16-class, vLLM-family engine,
TP=4, fp8 KV, speculative decoding, prefix caching) · Laguna S 2.1 FP8
(121 GB, stock vLLM 0.25.1 day-one, TP=4).

## v2 (80 adversarially-verified items)

| | math_hard | grounded_v2 | deepctx | agentic | OVERALL | med tok/s | wall |
|---|---|---|---|---|---|---|---|
| GLM-5.2 Q3 @ max | **18/20** | **30/30** | 7/15 | 12/15 | **67/80** | 40 | 5.5 h |
| DeepSeek V4-Flash | 17/20 | 23/30 | **10/15** | **14/15** | 64/80 | **166** | **22 min** |
| Laguna S 2.1 | 12/20 | 24/30 | 3/15 | 12/15 | 51/80 | 109 | ~1.8 h |

**Exact two-sided McNemar (discordant items):**

- GLM vs V4 — overall 13-10, p = 0.678 (tie). **grounded_v2: 7-0, p = 0.0156**
  — V4 fabricated on seven plausible-but-absent traps where GLM refused;
  never the reverse. The first statistically significant quality separation
  either suite has produced between these two.
- GLM vs Laguna — overall 18-2, **p = 0.0004** (corrected run). Category-level:
  grounded_v2 5-0, deepctx 5-0, math_hard 7-1 — each individually marginal
  (p = 0.06-0.07); the overall separation is what survives.
- V4 vs Laguna — deepctx 9-2 (p = 0.065); grounded_v2 dead even (6-7).

**deepctx accuracy by needle depth (start / middle / end of a 128K haystack):**
GLM 3/5 · 2/5 · 2/5 — V4 3/5 · **4/5** · 3/5 — Laguna 2/5 · 0/5 · 1/5.

Failure anatomy worth quoting: GLM's deepctx misses are recency failures — in
one dissected case it executed all five reference hops with correct record IDs
but trusted a superseded record mid-chain, landing on the decoy path the
generator plants for exactly this diagnosis. Laguna's misses are calibration —
10 of 12 were full-budget reasoning truncations, and a labeled double-budget
sensitivity arm recovered only one item: at 128K it searches without
converging. V4's grounded misses are fabrications under trap pressure.

## v1 (99 paired items — the saturated suite)

V4 96/99 · GLM 92/99 · Laguna 92/99 (corrected) — all pairwise p > 0.2. Three very
different architectures, statistically indistinguishable: v1 is a regression
floor, not a discriminator. This is why v2 exists.

## Correction (2026-07-23, same day)

A transport bug was found by a fourth-model run on another node: some vLLM
builds emit the reasoning field as `reasoning`, not `reasoning_content`, so
truncated items from those engines were silently scored as empty. Fixed in
commit `d280f16`; of the models above only Laguna was affected. Full re-run
under the fixed harness: **v2 unchanged (51/80)** — its truncations cut off
mid-search with no salvageable answer; **v1 +3 (105 -> 108)** — shallow
truncations held rescuable answers. Tables above show corrected numbers.
The affected model's score could only have been *under*-counted by this bug,
never inflated.

## Serving envelope highlights (same node)

- **Laguna depth curve is nearly flat**: 129 tok/s at 7K → **84.5 at 983K**
  (mixed sliding-window/global attention; ~29 KiB/token effective KV). V4 over
  the same span: 250 → 31. GLM (262K max tested): 44 → 13.
- **Laguna serves 1,048,576 tokens on TWO GPUs** (1.9M-token KV pool, 68 tok/s
  at 983K) after applying the publisher's own BF16-repo 1M rope parameters to
  the FP8 checkpoint; a planted-fact gate at 900K depth passed with exact
  retrieval plus spontaneous correlation to a warning planted at 5% depth.
- **Prefix caching at 1M: cold 261 s → warm 1.52 s (172x)** on stock defaults.
- TP=2 vs TP=4 (Laguna): identical quality by design, near-identical
  concurrency curves at ≤16K context; TP=4 pays off only beyond ~250K
  (prefill 1.5x, decode +24% at 1M).
- Board power (mean/peak): GLM pipeline 694/728 W · V4 TP=4 1,090/2,020 W ·
  Laguna TP=4 991/2,060 W. All GPU temps ≤ 63 °C on air.

## Read

The three models partition along orthogonal axes: **V4** is the engine
(speed, depth, agent loops — but fabricates under trap pressure), **GLM Q3**
is the conscience (zero fabrications across both suites at max effort — but
slow, and relentless past the point of usefulness at depth), **Laguna** is the
economist (giant-class shallow competence and a peerless serving profile at
8B-active cost — not yet a reasoning peer at depth, on day one). Model
selection is a routing problem, not a leaderboard problem.
