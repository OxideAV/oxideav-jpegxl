# Round-23 d1 LfCoefficients leaf-pick property dump + WP y=0 boundary audit (Auditor mode)

**Date**: 2026-05-10
**Fixture**: `crates/oxideav-jpegxl/tests/fixtures/vardct_256x256_d1.jxl`
**Round-22 hypothesis pursued**: paths (2) leaf-pick property dump at
Y' sample 22 + (3) WP edge-case audit at the y=0 / NE boundary.

## TL;DR

Round 22 hypothesised that the bias=3 → bias=4 sample divergence at
Y' sample 22 (row 0, col 22) was caused by a **leaf-pick flip** —
the WP rounding bias toggle changing `prop[15] = max_error` enough
to push a tree decision over its threshold. Round 23 falsifies this
hypothesis: the leaf-pick at sample 22 is **bit-identical** between
bias=3 and bias=4. Both runs walk the same MA-tree path, end at the
same leaf (`ctx=0, predictor=6, multiplier=1, offset=0`), and decode
the same ANS token. The single-bit sample difference (Y'=500 vs 501)
comes purely from `(wp_pred8 + bias) >> 3` rounding the same `pred8 =
4212` differently — a one-LSB delta in the RECONSTRUCTED sample
value, not in the leaf or token.

The first leaf-pick that actually flips between bias=3 and bias=4
is at sample (channel=0, x=15, y=2), at log index 79 — far downstream
of sample 22. By that point the bias delta has rippled through 79
samples of WP `true_err` and the `max_error` property at that sample
crosses the `> 0` threshold differently (bias=3: 36 → ctx=0 left;
bias=4: -34 → ctx=1 right). This downstream flip is an EXPECTED
consequence of the bias delta, not a NEW bug.

The WP y=0 / NE-boundary audit is clean: at Y' sample 22 (x=22, y=0)
the `te_n`, `te_nw`, `te_ne` values are all 0 as expected (the row
above doesn't exist; te_ne_raw correctly falls back to te_n which is
also 0). `max_error` correctly equals `te_w = 96` per Listing E.4.

## Path (2) — Leaf-pick property dump at Y' sample 22

New diagnostic infra added to `src/modular_fdis.rs`:

* `LEAF_PICK_TRACE_TARGET: AtomicU64` — packed `(channel, x, y)` target.
* `LEAF_PICK_TRACE_BUF` — thread-local per-decision step recorder.
* `LEAF_PICK_TRACE_PROPS` — thread-local property vector at the target.
* `LEAF_PICK_TRACE_WP` — thread-local WP intermediates at the target.
* `LEAF_PICK_TRACE_LEAF` — thread-local final-leaf recorder.
* `LEAF_PICK_LOG_ENABLED` + `LEAF_PICK_LOG` — per-sample leaf log
  across the whole decode (for first-flip bisect).
* `evaluate_tree_inner(record_trace: bool)` — variant of `evaluate_tree`
  that pushes each interior decision when tracing is on.

The single per-sample loop in `decode_channels_at_stream` checks the
target on each iteration (cheap atomic compare) and populates the
trace only for the matched sample. Default (`u64::MAX` target) is
zero-overhead.

### Sample 22 (channel=0, x=22, y=0) — bias=3 (spec) vs bias=4 (auditor)

```text
Property vector — IDENTICAL between bias=3 and bias=4:

  prop[ 0] channel(c)         =   0
  prop[ 1] stream_index       =   1
  prop[ 2] y                  =   0
  prop[ 3] x                  =  22
  prop[ 4] abs(N)             = 529
  prop[ 5] abs(W)             = 529
  prop[ 6] N                  = 529
  prop[ 7] W                  = 529
  prop[ 8] prop8(W-grad@x-1)  = -12
  prop[ 9] grad(W+N-NW)       = 529
  prop[10] W-NW               =   0
  prop[11] NW-N               =   0
  prop[12] N-NE               =   0
  prop[13] N-NN               =   0
  prop[14] W-WW               = -12
  prop[15] max_error          =  96

WP intermediates — IDENTICAL:
  te_w=96  te_n=0  te_nw=0  te_ne=0
  w8=4232 n8=4232 nw8=4232 ne8=4232
  wp_pred8=4212 max_error=96

MA-tree decision steps — IDENTICAL:
  step[0] node=0  prop=1  (stream_index) value=2 pv=1   → RIGHT (stream_index <= 2 → LfCoefficients branch)
  step[1] node=2  prop=15 (max_error)    value=0 pv=96  → LEFT  (max_error > 0)

Final leaf — IDENTICAL:
  ctx=0 predictor=6 offset=0 multiplier=1
```

So at sample 22, the bias toggle has NO effect on the property
vector, the tree walk, or the leaf chosen. The leaf-pick hypothesis
from round 22 is **falsified**.

What DOES differ between bias=3 and bias=4 at sample 22:

* The reconstructed sample value: `(4212 + bias) >> 3`. With
  bias=3: `4215 >> 3 = 526`. With bias=4: `4216 >> 3 = 527`. Add the
  same decoded `diff = -26` and we get bias=3: 500, bias=4: 501.

This bias-delta-of-1 in the reconstructed sample value then
propagates into sample 23's `te_w` (which differs by 8 in 8x scale),
and so on.

## Path (2) follow-up — first leaf-pick flip bisect

Added `d1_first_leaf_flip_bisect_round_23` test which captures the
per-sample leaf-pick log under both biases and finds the FIRST log
index at which `(channel, x, y)` agree but the picked leaf differs.

```text
First property[15] diff: log_idx=23 (channel=0 x=23 y=0)
  bias=3 ctx=0 max_error=212
  bias=4 ctx=0 max_error=204
  (max_error differs by 8 — exactly the 8x-scale delta of the +1
  bias rounding at sample 22 — but both still > 0, so ctx unchanged)

First leaf-ctx flip: log_idx=79 (channel=0 x=15 y=2)
  bias=3 ctx=0 max_error= 36   →  prop[15] > 0  → LEFT (ctx=0)
  bias=4 ctx=1 max_error=-34   →  prop[15] ≤ 0  → RIGHT (ctx=1)

ctx_flip count over 3072 samples: 1512
p15_diff count: 2972

bias=3 leaf-ctx histogram (3072 samples total):
  ctx 0:  1615 (max_error > 0  — LfCoeff predictor-6 leaf)
  ctx 1:  1457 (max_error <= 0 — LfCoeff predictor-6 leaf)
  ctx 2..15: 0 (these contexts only appear in the OTHER sub-bitstreams
              that share the global tree — e.g. HFMetadata)
```

The first flip at sample (0, 15, 2) is **expected**: the bias
delta has had 79 samples to ripple, and at this sample the
accumulated drift in `te_w` pushes max_error across the threshold.

## Path (3) — WP edge-case audit at the y=0 / NE boundary

At Y' sample 22 (x=22, y=0) the WP intermediates collected by the
trace are:

```text
te_w = 96   (true_err carried from sample 21, where wp_pred8=4328 −
             v=541·8=4232 = 96)
te_n  = 0   (no row above — `state.true_err_at(22, -1)` returns 0)
te_nw = 0   (no row above — `state.true_err_at(21, -1)` returns 0)
te_ne = 0   (te_ne_raw branch: `(x+1) < state.width && y > 0` is FALSE
             at y=0; te_ne falls back to te_n = 0, per Listing H.5.2
             "if NW or NE does not exist, the value of N is used")
```

This matches the spec's H.5.2 prose for top-row samples: any
true_err read above the first row returns 0 (the storage was zero-
initialised and never written for y < 0), and `te_ne_raw` correctly
falls back to `te_n` because the y > 0 condition is false. The
`max_error = te_w = 96` per Listing E.4 (the abs-comparison loop
keeps `te_w` because the other three are zero).

The neighbour-value fallbacks for `nb.n / nb.nw / nb.ne` (in
`Neighbours::at`) at y=0 also collapse correctly: all three reduce
to `nb.w = 529`, so `n8 = w8 = nw8 = ne8 = 4232`.

The WP y=0 boundary audit is **clean** — no bug here.

## Sentinels

* All five small lossless fixtures stay regression-free.
* Round-3..22 sentinel tests stay green.
* Round-10 synth_320 drift bisect stays green (untouched).
* Test count: 338 → 343 (+5: tree-topology, sample-0 baseline,
  sample-21 boundary, sample-22 main audit, first-flip bisect).

## What round 23 falsifies

* **Leaf-pick at sample 22 is NOT the bug source**. Identical
  property vector, identical tree walk, identical leaf, identical
  token decode under bias=3 and bias=4.
* **WP y=0 / NE-boundary handling is correct**. te_n/te_nw/te_ne
  collapse to 0 as required; max_error = te_w; the te_ne_raw
  fallback to te_n triggers correctly when y=0.
* **Property derivation is correct at sample 22** under both biases
  (verified by side-by-side identity).

## What round 23 confirms

* The d1 LfCoefficients global MA tree has 31 nodes, 16 leaves
  (`num_ctx=16`), and uses only **2 of those 16 contexts** for
  LfCoefficients itself (ctx=0 and ctx=1, both predictor=6
  multiplier=1 offset=0, picked by the prop[15] > 0 split at node 2).
  The remaining 14 contexts are reserved for HFMetadata.
* EntropyStream uses `use_prefix_code=false` (ANS), 5 clusters,
  cluster_map = `[0, 1, 2, 3, 4, 3, 3, 3, 1, 3, 1, 1, 2, 1, 1, 2]`
  (so LfCoefficients ctx 0 → cluster 0, ctx 1 → cluster 1).
* Per-cluster HybridUintConfig:
  - cluster 0: split_exponent=4 split=16 msb=1 lsb=2
  - cluster 1: split_exponent=4 split=16 msb=1 lsb=2
  - cluster 2: split_exponent=0 split=1  msb=0 lsb=0  (degenerate)
  - cluster 3: split_exponent=4 split=16 msb=2 lsb=0
  - cluster 4: split_exponent=4 split=16 msb=2 lsb=0
* The d1 final ANS state (`0x21914271`) after 3072 calls is off the
  sentinel `0x00130000` by ~562M. Since leaf-pick + WP boundary +
  property derivation are clean, the bug is in one of the
  surfaces NOT yet bisected.

## Round-24 candidates (in priority order)

The remaining surface area for the d1 bug, given rounds 17..23
have ruled out: per-token hybrid-uint accounting, extra-bits,
cluster_map uniformity, ANS state init, prelude bit consumption,
"267-bit overshoot" (illusory), per-cluster distribution decode,
alias-table self-map branch, WP rounding bias, leaf-pick at sample
22, WP y=0 / NE boundary, property derivation:

1. **Per-cluster ANS distribution byte-trace** (round 21 path b
   deferred). Capture the post-renorm `D` array bytes for clusters
   0 and 1 (the two LfCoefficients clusters) and compare against a
   re-derived array using FDIS Annex D.3.4 directly. If our cluster
   distribution shape is off by one entry or scaled by the wrong
   `log_alphabet_size`, every ANS call returns symbols off-by-one
   from the encoder's intent — and the cumulative error compounds
   across 3072 calls into the observed 562M state delta.

2. **Per-call alias-mapping invariant check**. For each ANS call,
   verify `cutoff[i] >= 0`, `(symbol, offset)` reconstructable to
   `state & 0xFFF`, and `D[symbol] >= 1`. If our alias table
   construction has a bug that randomly mis-routes a few state
   indices, those ~1 in 1000 mis-decodes accumulate to the observed
   final-state offset.

3. **Hybrid uint extra-bit count audit at LfCoefficients ctx
   0 / ctx 1**. cluster 0 / cluster 1 share `(split_exp=4,
   msb_in=1, lsb_in=2)`. For a token > 16 (split), the spec reads
   `lsb_in_token + msb_in_token + (n - split_exp)` = 1+2+(n-4)
   extra bits, where n is `floor(log2(token))`. Verify this matches
   our `HybridUintState::decode` against a small worked example.

4. **Mid-stream ANS state cross-check** (originally round 23 path
   1, deferred). With the round-23 per-sample leaf log and the
   first-divergence point now known (sample 79 for ctx flip,
   sample 23 for property[15] flip), capture `(pre_state, post_state,
   refill_bits)` for samples 21, 22, 23 specifically and hand-walk
   them against FDIS D.3.3 Listing D.3 to spot-check the ANS update.
   Either matches or pinpoints a one-call bug.

The order above prioritises (1) since the round-19 cluster_map
audit confirmed clustering ITSELF is uniform but did NOT byte-trace
the cluster-0 / cluster-1 distribution arrays — a per-cluster D[]
shape mismatch would explain a stable per-call bias accumulating to
the observed state offset.
