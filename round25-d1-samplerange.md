# Round-25 d1 LfCoefficients per-sample state-range dump (Auditor mode)

**Date**: 2026-05-10
**Fixture**: `crates/oxideav-jpegxl/tests/fixtures/vardct_256x256_d1.jxl`
**Round-24 hypothesis pursued**: path (2) — extend round-23's leaf-pick
audit beyond Y' sample 22 to capture the actual first-divergent
ctx-flip at sample 79.

## TL;DR

Round 25 captures the full per-sample state (props, WP intermediates,
leaf, token, decoded value) for samples 22..=79 under both WP bias 3
(spec) and 4 (auditor) and confirms r23's first-ctx-flip at sample 79
is a **clean downstream consequence** of the bias-rounding delta — not
a fresh divergence in property derivation, cluster routing, or leaf
selection. The flip mechanic at sample 79 is now fully traced:

```text
sample 79 (ch=0, x=15, y=2):

  bias=3 (spec)              bias=4 (auditor)
  -----------------          -----------------
  te_w  =  7                 te_w  =  7
  te_n  = -3                 te_n  = -3
  te_nw = -34                te_nw = -34
  te_ne = +36                te_ne = +28        <-- 8x-scale delta of -8
                                                     (= -1 in 1x scale,
                                                      from upstream NE
                                                      sample at log_idx
                                                      48 decoding to
                                                      v=521 vs 522)
  ----- max_error = max(|te_w|, |te_n|, |te_nw|, |te_ne|) -----
  bias=3: max(7, 3, 34, 36) keeps te_ne = +36   ── prop[15] > 0 → ctx 0
  bias=4: max(7, 3, 34, 28) switches to te_nw  = -34 ── prop[15] ≤ 0 → ctx 1

  -> different leaf ctx -> different ANS distribution -> token 8 vs 7
     -> diff +4 vs -4 -> v 546 vs 539 (and the bias=3 side picks up an
        extra +1 in `pred` from the WP rounding, unrelated to the flip).
```

So the bias=3 vs bias=4 toggle introduced by round 22 was always a
DIAGNOSTIC, never a candidate fix. The bias=3 reading IS spec-conformant
(Table H.3: `(prediction + 3) >> 3`); the bias=4 reading was used purely
as a perturbation oracle to triangulate the divergence surface.

**The d1 ANS final-state delta (`0x21914271 - 0x00130000` ≈ 562M) is
NOT explained by leaf-pick / property derivation / WP boundary / WP
ripple within the LfCoefficients sub-bitstream.** Rounds 17..25 have
now ruled out every surface inside this sub-stream.

## Per-sample diff table (log_idx 22..=79, bias=3 vs bias=4)

Format: `log_idx ch x y | leaf | prop | wp | token | val`. `-` = no
diff. The full output is reproduced from the test
`d1_per_sample_state_range_22_to_79_round_25` (run with `--nocapture`).

```text
log[ 22] ch=0 x=22 y=0 | -    | -                                                                  | -                                                                  | -        | v(500→501)
log[ 23] ch=0 x=23 y=0 | -    | abs(N) abs(W) N W prop8 grad W-WW max_error(212→204)               | te_w(212→204) w8 n8 nw8 ne8 wp_pred8(3953→3964) max_error           | -        | v(458→460)
log[ 24] ch=0 x=24 y=0 | -    | abs(N) abs(W) N W prop8 grad W-WW max_error(289→284)               | te_w(289→284) w8 n8 nw8 ne8 wp_pred8(3597→3614) max_error           | -        | v(426→428)
log[ 25] ch=0 x=25 y=0 | -    | abs(N) abs(W) N W grad max_error(189→190)                          | te_w(189→190) w8 n8 nw8 ne8 wp_pred8 max_error                      | -        | v(414→416)
log[ 26..31]            (similar — bias delta keeps rippling through y=0 row)
log[ 32..40]            -                                                                          -                                                                    -          -
log[ 41] ch=0 x= 9 y=1 | -    | -                                                                  | -                                                                   | -        | v(388→389)
log[ 42] ch=0 x=10 y=1 | -    | abs(W) W prop8 grad W-NW W-WW                                      | te_w(-4→-12) w8 wp_pred8(3453→3456)                                 | -        | -
log[ 43..47]            (small props/wp diffs propagate through y=1)
log[ 48] ch=0 x=16 y=1 | -    | -                                                                  | -                                                                   | -        | v(521→522)
log[ 49] ch=0 x=17 y=1 | -    | abs(W) W prop8 grad W-NW W-WW                                      | te_w(36→28) w8                                                      | -        | -
log[ 50..63]            (cumulative drift)
log[ 64..66] (y=2 origin row, no diff yet)
log[ 67] ch=0 x= 3 y=2 | -    | -                                                                  | -                                                                   | -        | v(480→481)
log[ 68..78]            (drift converges around the NE neighbour of sample 79)
log[ 79] ch=0 x=15 y=2 | LEAF!| N-NE(1→0) max_error(36→-34)                                        | te_ne(36→28) ne8(4168→4176) wp_pred8(4340→4343) max_error(36→-34)  | token(8→7) | v(546→539)

First-divergence summary:
  first val_diff       at log_idx 22  (bias-rounding-LSB delta in v)
  first wp_diff_kind   at log_idx 23  → wp[0] = te_w (carries v's delta forward)
  first prop_diff_kind at log_idx 23  → prop[4] = abs(N)
  first token_diff     at log_idx 79  (first ANS read whose context differs)
  first ctx_flip       at log_idx 79
```

## Sample 79 — full rich-state both biases

```text
bias=3 (spec):
  ch=0 x=15 y=2  leaf(ctx=0 pred=6 off=0 mult=1)  token=8 diff=4 pred=542 v=546
  props: [4]abs(N)=522  [5]abs(W)=550  [6]N=522  [7]W=550  [8]prop8=-7
         [9]grad=551    [10]W-NW=29   [11]NW-N=-1 [12]N-NE=+1
         [13]N-NN=29    [14]W-WW=-3   [15]max_error=+36
  wp:    te_w=7  te_n=-3  te_nw=-34  te_ne=+36
         w8=4400 n8=4176 nw8=4168 ne8=4168
         wp_pred8=4340  max_error=+36

bias=4 (auditor):
  ch=0 x=15 y=2  leaf(ctx=1 pred=6 off=0 mult=1)  token=7 diff=-4 pred=543 v=539
  props: [4]abs(N)=522  [5]abs(W)=550  [6]N=522  [7]W=550  [8]prop8=-7
         [9]grad=551    [10]W-NW=29   [11]NW-N=-1 [12]N-NE=0
         [13]N-NN=29    [14]W-WW=-3   [15]max_error=-34
  wp:    te_w=7  te_n=-3  te_nw=-34  te_ne=+28
         w8=4400 n8=4176 nw8=4168 ne8=4176
         wp_pred8=4343  max_error=-34
```

The state inputs to the MA tree differ in **exactly two** properties at
this sample: `prop[12] = N - NE` (1 → 0) and `prop[15] = max_error`
(+36 → -34). Both deltas trace back to a single 1-LSB change in the NE
neighbour's reconstructed value (8x-scale `ne8` 4168 → 4176; `te_ne`
36 → 28; reading off the test output from log_idx 48: NE = sample
(x=16, y=1) decoded as 521 under bias=3 vs 522 under bias=4).

The MA tree's split at `prop[15] > 0` then sends bias=3 LEFT (ctx=0)
and bias=4 RIGHT (ctx=1). Both branches use the same predictor=6,
multiplier=1, offset=0 — only the ANS context differs, which selects
a different cluster (ctx 0 → cluster 0; ctx 1 → cluster 1) and reads
a different token from the same point in the ANS chain.

## Root cause classification

**The divergence at sample 79 is FULLY EXPLAINED by the WP true_err
history accumulating the bias-rounding LSB delta.** No bug is exposed
by this round; both biases produce internally-consistent decodes.
Specifically:

* No fresh **property-derivation** bug — `get_properties` returns the
  same 16 entries under both biases at sample 79 except for the two
  derivative quantities (`N-NE` and `max_error`) that depend on the
  drifted neighbour state.
* No **cluster-selection** bug — `cluster_map[ctx]` correctly routes
  ctx 0 → cluster 0 and ctx 1 → cluster 1; the leaf splits on
  `prop[15] > 0` exactly as r23's tree-topology dump showed.
* No **WP boundary** bug — the y=0 / NE-row fallbacks remain identical
  between biases (already audited in r23).
* No **MA-tree branching** bug — both biases visit the same two
  decision nodes (root, then `prop[15]` split) and pick valid leaves.

## Round 25 fix landed?

**No.** Auditor mode applies. The audit confirms the per-sample state
machinery is internally consistent under bias=3 (spec). The 562M
ANS-final-state delta originates OUTSIDE the LfCoefficients sub-stream
itself.

## What round 25 falsifies

* The "downstream property derivation diverges → first ctx-flip at
  sample 79" hypothesis (r24 path 2 candidate). The properties at
  sample 79 differ ONLY in derived quantities (`N-NE`, `max_error`)
  whose inputs are the bias-perturbed NE neighbour value. The
  derivation arithmetic is identical between biases.
* The "true_err history is accumulating wrong" hypothesis. The
  true_err update is `wp_pred8 - (v << 3)`. Since `wp_pred8` is
  identical between biases for samples that haven't yet been touched
  by drift, and `v` differs by exactly the bias-rounding LSB, the
  true_err drift is the inevitable consequence — not a calculation
  bug.

## What round 25 confirms

* d1 final ANS state remains at `0x21914271` (off the spec sentinel
  `0x00130000` by ~562M) — unchanged from r17..r24.
* The bias=4 toggle exists purely as a diagnostic perturbation; it
  produces a different but equally bug-free decode (final state
  `0x00fd721e`, also far from the sentinel).
* The first 22 samples (Y' row 0, x ∈ 0..21) decode identically under
  both biases (re-confirmed; matches r22 dump).
* The bias delta first manifests at sample 22 as a single 1-LSB
  change in `v` (500 → 501). It propagates forward through `te_w`
  immediately at sample 23, then ripples through `n8/w8/nw8/ne8` and
  `wp_pred8` for every subsequent sample.

## Round 26 candidates (the unexplored surface, in priority order)

Given rounds 17..25 have ruled out: per-token hybrid-uint accounting,
extra-bits, cluster_map uniformity, ANS state init, prelude bit
consumption, "267-bit overshoot" (illusory), per-cluster distribution
decode shape, alias-table self-map / Vose-pump, alias-mapping invariant
lookup, per-call state-arithmetic, WP rounding bias, leaf-pick at
sample 22, leaf-pick at sample 79 (this round), WP y=0 / NE boundary,
property derivation, and WP true_err propagation:

1. **D[]-vs-cjxl encoder ground-truth comparison** (r24 candidate 1,
   not yet pursued). The internal consistency of cluster 0's
   `[0]=384..[35]=2` and cluster 1's `[0]=384..[60]=2` was verified
   in r24. But "internally consistent" does NOT prove "matches what
   the encoder wrote." A side-channel from cjxl 0.11.1 (e.g. via
   `cjxl --debug` or by intercepting the encoder's histogram output)
   would either expose a one-entry mismatch (smoking gun) or
   definitively rule out the distribution-shape surface.

2. **Sub-bitstream length / TOC-entry boundary audit** (r24 candidate
   3 reframed). The d1 fixture's frame TOC entry 0 covers
   LfGlobal+LfCoefficients+HFMetadata for the single LfGroup. The
   round-15 fix landed `single-TOC-entry section chaining`. If our
   chaining advance bits past the correct LfCoefficients end-marker
   into the start of HFMetadata (or vice versa), every ANS call from
   that point onward reads wrong bits, accumulating into the observed
   state offset. Audit: capture `br.bit_position()` immediately
   before and after the LfCoefficients decode in d1; cross-check
   against the same pair from cjxl --info or an independent decoder.

3. **Decoder-iteration order**. r25's per-sample iteration is
   channel-major (all of channel 0, then channel 1, then 2). Spec
   §H.3 reads "for each channel, for each sample". But the round-15
   GlobalModular zero-channel ModularHeader gating fix may have left
   d1's LfCoefficients descriptor list in a different order than
   cjxl writes. If our `descs` ordering doesn't match the encoder's,
   we'd read the same total of 3072 ANS calls but route them to the
   wrong (channel, x, y) slots. Audit: print `descs` for d1 vs the
   spec's "Y, X, B" ordering for VarDCT XYB.

4. **HFMetadata stream cross-talk via the shared 16-context tree**
   (r24 candidate 3 + reframed). The global tree has 16 leaves, of
   which only 2 (ctx 0/1) are LfCoefficients and 14 are HFMetadata.
   If LfCoefficients decode somehow LEAKS into HFMetadata's
   contexts (e.g. if any prop-15 ≤ 0 sample picks an HFMetadata leaf
   by mistake), the mis-routed ANS calls would explain a stable
   per-call bias accumulating to ~562M. Audit: instrument the leaf
   selector to assert leaf.ctx ∈ {0, 1} for the LfCoefficients
   sub-stream (round 23's per-sample log already shows ctx ∈ {0, 1}
   for all 3072 LfCoefficients calls — so this is essentially
   already cleared, but a sentinel-based assertion would fence it
   permanently).

The order above prioritises (1) since it's the most surgical at this
point: with every other surface inside the sub-stream cleared, the
only remaining "the bytes we produce don't match the bytes the encoder
intended" path is the distribution table itself.

## Sentinels

* All 5 small lossless fixtures stay regression-free.
* Round-3..24 sentinel tests stay green.
* Round-10 synth_320 drift bisect stays green.
* Test count: 345 → 347 (+2: rich-range main audit, boundary handoff).
