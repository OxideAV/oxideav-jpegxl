# Round-24 d1 LfCoefficients per-cluster D[] byte-trace + per-call alias-mapping invariant audit (Auditor mode)

**Date**: 2026-05-10
**Fixture**: `crates/oxideav-jpegxl/tests/fixtures/vardct_256x256_d1.jxl`
**Round-23 hypotheses pursued**: paths (1) per-cluster ANS distribution
byte-trace for clusters 0+1, and (2) per-call alias-mapping invariant
check.

## TL;DR

Round 24 closes paths (1) and (2) cleanly: **no bug found in either
surface**. Both clusters 0 and 1 (the only two LfCoefficients touches
per r23) decode their 64-entry `D[]` arrays summing to exactly 4096;
the alias table built from each `D[]` routes probability mass to
symbols identically to the declared `D[]` (per-symbol routed-mass
divergence = 0); and across the FULL 3072-call ANS trace the
spec C.3.2 alias-mapping invariant
`(symbol, offset) = AliasMapping(state & 0xFFF)` holds bit-for-bit
when checked against either cluster 0's or cluster 1's alias table
(0 hard violations across 3072 calls, 288 ambiguous calls where
both clusters yield the same `(symbol, offset, prob)`).

So the d1 ANS final-state delta of `0x21914271 - 0x00130000 ≈ 562M`
is **NOT** caused by:

* per-cluster D[] shape mismatch (D[] sums to 4096, alias routing
  consistent),
* alias-table self-map / Vose-pump bugs (every call's alias lookup
  reproduces against the spec C.3.2 procedure),
* state-arithmetic per-call (the
  `state = D[symbol] * (state >> 12) + offset` update was checked
  inline against the trace and matches in every call),
* cluster mis-routing (only clusters 0 and 1 are ever touched, and
  both `prob = D[symbol]` and `(symbol, offset)` consistently
  identify which cluster).

The bug must be EITHER:

* the D[] tables we DECODE are SELF-CONSISTENT but DIFFERENT from what
  the cjxl encoder WROTE (a bug in `read_distribution` against
  C.2.5 that produces a valid-looking distribution but with a few
  entries offset from the encoder's intent), OR
* the leaf-pick / cluster routing is wrong at some sample DOWNSTREAM
  of sample 22 (round 23 verified the leaf-pick at sample 22 only
  and audited the Y' channel).

## Path (1) — per-cluster D[] full byte-trace

### Cluster 0 (`split_exp=4 msb=1 lsb=2 split=16`)

```text
sum(D)=4096 (must equal 4096) ✓

Full D[] (64 entries, only non-zeros listed):
  [ 0]= 384  [ 1]= 384  [ 2]= 256  [ 3]= 640  [ 4]= 192  [ 7]= 660
  [ 8]= 128  [11]= 512  [12]=  48  [15]= 320  [16]=  48  [19]= 320
  [20]=  16  [23]= 128  [24]=   8  [27]=  32  [28]=   2  [31]=  16
  [35]=   2

alias-routed total = 4096 ✓ (must equal bucket_size * table_size = 64*64 = 4096)
per-symbol routed-mass vs declared-D divergence: NONE.
```

The cluster-0 alias table reconciles cleanly: every bucket either
self-maps (`cut=0, sym=i` for buckets 0/8/11/15/19/23 — these are
where `D[i] == bucket_size = 64` exactly OR where the Vose pump
ended with `cutoffs[i] == bucket_size` and the trailing
reconciliation reset to self-map), redirects fully (`cut=0,
sym!=i`), or splits at `cut`.

For example bucket 0 (`D[0] = 384 = 6 * bucket_size`): self-map after
Vose pump because the overfull bucket gave away its excess mass to
six underfull peers, leaving `cutoffs[0] = bucket_size`, which the
trailing loop resets to `(sym=0, off=0, cut=0)`. The mass routed to
symbol 0 by the alias table totals `64 (self) + 64 (bucket 1's
redirect tail) + 64 (bucket 5) + 64 (bucket 6) + 64 (bucket 9) +
64 (bucket 10) = 384 = D[0]`. Round-trip exact.

### Cluster 1 (`split_exp=4 msb=1 lsb=2 split=16`)

```text
sum(D)=4096 (must equal 4096) ✓

Full D[] (64 entries, only non-zeros listed):
  [ 0]= 384  [ 1]= 224  [ 2]= 320  [ 3]= 192  [ 4]= 576  [ 7]= 160
  [ 8]= 644  [11]=  48  [12]= 448  [15]=  32  [16]= 448  [17]=   2
  [19]=  32  [20]= 320  [23]=   8  [24]= 192  [27]=   8  [28]=  16
  [31]=   2  [32]=  32  [36]=   4  [40]=   2  [60]=   2

alias-routed total = 4096 ✓
per-symbol routed-mass vs declared-D divergence: NONE.
```

Same bit-for-bit reconciliation. Bucket 0: self-map, mass routed to
symbol 0 totals exactly 384.

### Verdict

Path (1) **falsifies** the round-23-proposed per-cluster D[] shape
mismatch hypothesis. The internal consistency of D[] ↔ alias is
exact for both clusters.

## Path (2) — per-call alias-mapping invariant audit

Captured the FULL 3072-call ANS state trace from the d1
LfCoefficients sub-bitstream decode. For each call `(state_pre, slot,
symbol_obs, offset_obs, prob, state_post, refill_bits)`, recompute the
expected `(symbol_re, offset_re)` from the cluster's alias table per
the spec C.3.2 procedure:

```text
i   = slot >> log_bucket_size  (= slot >> 6 for the d1 6-bit alphabet)
pos = slot & (bucket_size - 1) (= slot & 63)
in_redirect = pos >= cutoffs[i]
symbol_re   = in_redirect ? symbols[i] : i
offset_re   = in_redirect ? offsets[i] + pos : pos
```

Cluster identification: cross-validate by trying BOTH cluster 0 AND
cluster 1 alias tables; declare a "hard violation" only if NEITHER
candidate cluster's `(symbol_re, offset_re, prob_re)` matches the
trace. (This avoids false-positives from cluster mis-attribution
when D0[sym] == D1[sym] for some symbol.)

### First 30 ANS reads — full table

```text
   row  0: pre=0x18c81f01 slot=3841 c=1 sym=60 off= 1  → ✓
   row  1: pre=0x00031903 slot=2307 c=1 sym=36 off= 3  → ✓
   row  2: pre=0x00c7270d slot=1805 c=1 sym=28 off=13  → ✓
   row  3: pre=0xc72d2478 slot=1144 c=1 sym= 2 off=70  → ✓
   row  4: pre=0x0f8f86c6 slot=1734 c=1 sym=27 off= 6  → ✓
   row  5: pre=0x0007c7c6 slot=1990 c=0 sym=31 off= 6  → ✓
   row  6: pre=0x07c6d7ca slot=1994 c=0 sym=31 off=10  → ✓
   row  7: pre=0x0007c6da slot=1754 c=0 sym=27 off=26  → ✓
   row  8: pre=0x0f9af6bf slot=1727 c=0 sym= 3 off=135 → ✓
   row  9: pre=0x02703607 slot=1543 c=0 sym=24 off= 7  → ✓
   row 10: pre=0x0001381f slot=2079 c=1 sym=32 off=31  → ✓
   row 11: pre=0x027fe810 slot=2064 c=1 sym=32 off=16  → ✓
   row 12: pre=0x0004ffd0 slot=4048 c=1 sym=24 off=144 → ✓
   row 13: pre=0x3bd0fd52 slot=3410 c=1 sym=16 off=148 → ✓
   row 14: pre=0x068adad4 slot=2772 c=1 sym= 8 off=474 → ✓
   row 15: pre=0x0107550e slot=1294 c=1 sym=16 off=398 → ✓
   row 16: pre=0x001cce4e slot=3662 c=1 sym=20 off=16  → ✓
   row 17: pre=0x00023f10 slot=3856 c=1 sym=20 off=208 → ✓
   row 18: pre=0x2c907534 slot=1332 c=1 sym=16 off=436 → ✓
   row 19: pre=0x04dfcdf4 slot=3572 c=1 sym=16 off=310 → ✓
   row 20: pre=0x00887a36 slot=2614 c=1 sym= 8 off=316 → ✓
   row 21: pre=0x001574d8 slot=1240 c=1 sym=19 off=24  → ✓
   row 22: pre=0x2af8c7c0 slot=1984 c=0 sym=31 off= 0  → ✓
   row 23: pre=0x002af8c0 slot=2240 c=0 sym=35 off= 0  → ✓
   row 24: pre=0x055e66c0 slot=1728 c=0 sym=27 off= 0  → ✓
   row 25: pre=0x000abcc0 slot=3264 c=0 sym=11 off=256 → ✓
   row 26: pre=0x00015700 slot=1792 c=0 sym=28 off= 0  → ✓
   row 27: pre=0x002aa808 slot=2056 c=1 sym=32 off= 8  → ✓
   row 28: pre=0x55481a01 slot=2561 c=1 sym=40 off= 1  → ✓
   row 29: pre=0x000aa903 slot=2307 c=1 sym=36 off= 3  → ✓
```

**0/30 alias-invariant violations**, **0/30 state-update arithmetic
mismatches**. `state = prob * (state_pre >> 12) + offset` reproduces
the trace's pre-refill new_state value at every call.

### Full 3072-call audit

```text
hard violations: 0
ambiguous (both c0 & c1 reproduce identically): 288
cluster usage (by `prob`-match disambiguation):
  c0 = 1755 calls
  c1 = 1317 calls
  unknown = 0
```

The ambiguous count of 288 is expected — clusters 0 and 1 share a
similar HybridUintConfig and several symbols where `D0[sym] ==
D1[sym]` (e.g. both have `D[31] = 16` for cluster 0, `D[31] = 2` for
cluster 1, but other symbols overlap).

### Verdict

Path (2) **falsifies** the round-23-proposed alias-mapping invariant
violation hypothesis. The decoder's alias lookup is bit-correct for
every call against the alias table the decoder built from the
prelude.

## What round 24 falsifies

* **Per-cluster D[] shape is not internally inconsistent**. Both
  clusters 0 and 1 sum to 4096 with alias-routed mass exactly
  matching D[].
* **Alias-mapping invariant holds for every call**. The
  `(symbol, offset) = AliasMapping(state & 0xFFF)` reproduction is
  bit-for-bit across 3072 calls.
* **Per-call state-arithmetic is correct**. `state = prob *
  (state >> 12) + offset` matches the trace exactly.
* **Only clusters 0 and 1 are touched**. No leakage into clusters
  2/3/4 (which hold HFMetadata).

## What round 24 confirms

* Cluster 0 has 19 nonzero entries; cluster 1 has 23. Both use
  `HybridUintConfig(split_exp=4, msb=1, lsb=2)`.
* Cluster 0 routes 1755 calls; cluster 1 routes 1317 calls (totals
  3072, matching the per-channel sample count of 3 × 32 × 32 = 3072
  for the 256×256 d1 fixture).
* The d1 final ANS state remains at `0x21914271`, off the sentinel
  by ~562M; this delta is real and tied to a SURFACE NOT YET
  BISECTED by rounds 17..24.

## Round-25 candidates

Given rounds 17..24 have ruled out: per-token hybrid-uint accounting,
extra-bits, cluster_map uniformity, ANS state init, prelude bit
consumption, "267-bit overshoot" (illusory), per-cluster distribution
decode shape, alias-table self-map / Vose-pump, alias-mapping
invariant lookup, per-call state-arithmetic, WP rounding bias,
leaf-pick at sample 22, WP y=0 / NE boundary, property derivation:

The ONLY remaining surface for the d1 bug is:

1. **D[] from cjxl reference comparison** — somehow obtain the EXACT
   D[] values cjxl 0.11.1 wrote into the prelude (e.g. via cjxl
   `--debug` if it surfaces histogram counts, or by re-encoding the
   same source pixel data with a controllable encoder and observing
   the ANS distribution side-channel). Compare against our
   `[0]=384 ... [35]=2` for cluster 0 and `[0]=384 ... [60]=2` for
   cluster 1. A single mismatched count would be the smoking gun.

2. **Leaf-pick + cluster-routing audit AT samples beyond sample 22**.
   r23 only verified sample 22's leaf-pick. Round 25 should walk the
   first ~80 samples (up to sample 79 where r23's first ctx-flip
   between bias=3 and bias=4 was observed) and verify each sample's
   leaf-pick + cluster route via the per-sample log already added in
   r23.

3. **HFMetadata stream cross-talk**. The ANS state IS shared across
   `(LfCoefficients, HFMetadata)` substreams — the global tree
   has 16 contexts, of which only 2 are LfCoefficients (cluster 0/1)
   and 14 are HFMetadata (clusters 2/3/4 via the cluster_map). If
   our LfCoefficients decode somehow LEAKS state-init or stream
   boundaries from HFMetadata (i.e. the call counts split into two
   separate streams aren't right), we'd see the observed delta.
   Cross-check: instrument `decode_channels_at_stream` to log the
   per-stream `(stream_idx, ctx_used, cluster_used)` for the first
   ~80 calls to verify the LfCoefficients sub-stream truly only
   touches contexts 0+1.

The order above prioritises (1) since it's the most surgical: a
direct D[]-vs-encoder-intent comparison would either pinpoint the
bug or definitively rule out the distribution-shape surface.

## Sentinels

* All 5 small lossless fixtures stay regression-free.
* Round-3..23 sentinel tests stay green.
* Round-10 synth_320 drift bisect stays green.
* Test count: 343 → 345 (+2: per-cluster D[] byte trace,
  per-call alias-mapping invariant audit).
