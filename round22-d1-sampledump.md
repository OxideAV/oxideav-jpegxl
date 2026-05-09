# Round-22 d1 LfCoefficients sample-dump + WP rounding toggle (Auditor mode)

**Date**: 2026-05-10
**Fixture**: `crates/oxideav-jpegxl/tests/fixtures/vardct_256x256_d1.jxl`
**Round-21 candidates pursued**: (a) `lf_quant` first-256 dump per channel
+ (c) WP `(p+3)>>3` rounding toggle. Paths (b) and (d) deferred.

## Path (a) — `lf_quant` first-256 dump per channel

The d1 LfCoefficients sub-bitstream decodes to three 32×32 channels of
`lf_quant` (1024 samples each, 3072 total — matches the round-19 ANS
call count). Spec-default (`(prediction + 3) >> 3`) decode produces:

```text
ch=0 (Y'): mean=467.96, min=326, max=644, all 256 samples nonzero
  [  0.. 16]  424  476  502  503  479  443  404  374  366  383  419  453  471  479  483  493
  [ 16.. 32]  505  517  527  537  541  529  500  458  426  414  436  482  546  602  638  640
  [ 32.. 48]  458  498  511  497  461  416  375  352  356  388  436  478  505  517  521  522
  …

ch=1 (X'): mean=14.27, min=-125, max=135
  [  0.. 16]    6   12   13   19   17    3   -6  -16  -10    4   14   16   36   35   35   36
  [ 16.. 32]   46   44   42   32   26   26   48   58   54   52   48   41   36   27   29   30
  …

ch=2 (B'): mean=41.33, min=-49, max=123, 253/256 nonzero
  [  0.. 16]  -18  -20  -10   -4    0   22   20   30   40   40   39   41   45   44   42   40
  [ 16.. 32]   31   25   26   28   28   38   94  104  104   98   87   79   75   80   84   86
  …
```

These look like a smooth low-frequency image: Y' is a high-positive
luma DC of ~470 with gentle modulation; X' (luma-correlated chroma)
is small with sign changes; B' is moderate positive with gradient.

cjxl's `djxl --debug` against this fixture decodes a clean 256×256
RGB image whose 8×8-block per-channel means span ranges that are
qualitatively consistent with these `lf_quant` magnitudes after
the `dX = mXDC * qX / (1 << extra_precision)` (I.5.2) dequant + LF
adaptive-smoothing + LLF + IDCT + colour-convert pipeline. We do
NOT have an in-tree pre-IDCT oracle for byte-exact comparison, so
this is shape-validation only — but the shape is plausible enough
that the per-sample decode loop is NOT producing garbage.

## Path (c) — WP rounding toggle

Added a runtime atomic `WP_ROUND_BIAS` (default 3) to
`modular_fdis.rs`. The two `(pred + 3) >> 3` sites now read the
atomic. Toggling it re-runs the LfCoefficients decode with a
different rounding bias and reports the post-decode ANS state.

| bias | final state  | calls | `|state − 0x130000|` |
|------|--------------|-------|-----------------------|
|  +0  | 0x0042cd42   | 3072  |  3,132,738            |
|  +3  | 0x21914271   | 3072  | 561,922,673  (spec)   |
|  +4  | 0x00fd721e   | 3072  |     15,364,638        |
|  +7  | 0x001214ac   | 3072  |         60,244        |

### What this falsifies

Both ISO/IEC 18181-1:2024 Table H.3 (predictor 6) and the FDIS-2021
Listing C.16 unambiguously say `(prediction + 3) >> 3`. The +3 bias
is normative.

The +4 result being closer to the sentinel than +3 is **NOT evidence
that Table H.3 should read +4**: bias=+7 is closer still (60 244 vs
15 M), and bias=+0 is closer than +3 too (3.1 M vs 561 M). All four
biases miss the sentinel by between 60 k and 562 M — none hit it.
The variation is essentially random ANS-state noise propagated
through 3072 decode calls under different leaf-selection chains.

Bug class **falsified**: WP rounding bias is not the bug.

### What we learn anyway

The per-sample chain divergence between bias=3 and bias=4 starts at:

* ch=0 (Y'): sample 22 (row 0, col 22). Samples 0..21 are bit-
  identical. This means the leaf picked for samples 0..21 does NOT
  depend on `property[15]` (= `max_error`), OR `max_error` happens
  to land in the same leaf bucket for both biases.
* ch=1 (X'): sample 0 (`6` vs `10`). Diverges immediately because
  the channel-1 decode runs AFTER channel-0, and the previous-
  channel properties (H.4 properties 16+) read X' from the just-
  decoded Y' channel (which already diverged at sample 22).
* ch=2 (B'): sample 0 (`-18` vs `0`). Same reason — previous-channel
  properties read both Y' and X'.

So a single per-sample divergence at Y' sample 22 propagates to
every subsequent sample in all three channels. This is consistent
with the round-19/20/21 picture that the bug is one specific bit
in one specific sample's decode (or in the WP state evolution
fed into MA-tree property[15]) that triggers a leaf-selection
flip from then on.

## Sentinels

* All five small lossless fixtures stay regression-free (atomic
  defaults to +3 = spec-conformant; lossless fixtures don't use
  predictor 6 in their MA trees, so the toggle is inert anyway).
* Round-3..21 sentinel tests stay green.
* Round-10 synth_320 drift bisect stays green (untouched).
* Test count: 337 → 338 (+1).
* New test: `round22_d1_sample_dump`.

## Round-23 candidates

With WP rounding bias falsified as the LfCoefficients bug source,
the remaining candidates from round-21 are:

1. **Mid-stream ANS state cross-check** (round-21 path 2) — capture
   `(pre_state, idx, sym, prob, new_state)` for every Nth call (say
   every 64th) over the 3072-call chain and compare against an
   encoder-side oracle. The first divergence call N would localise
   the bug to one specific sample. Easier now that we know the
   first-divergent sample under bias-flip is Y' sample 22 — the
   ANS-call-count to reach that sample is 22.

2. **Leaf-pick trace at Y' sample 22** — dump the MA-tree property
   array (16 base + variable previous-channel) at (x=22, y=0, c=0)
   for the d1 fixture. If `prop[15]` = max_error is 0, the leaf
   picked ought to be deterministic from the other 15 properties.
   If `prop[15]` is nonzero, then the WP state's `last_max_error`
   from sample 21's WP-predict call is in question — and since
   sample 21 has neighbours W=505, N=505 (from the dump under
   bias=3, but identical under bias=4), the WP predictor and its
   error history at sample 21 should be tractable to hand-walk.

3. **WP edge-case audit at the ne_raw boundary** — `wp_predict`
   reads `te_ne_raw = state.true_err_at(x+1, y-1)` only when both
   `(x+1) < width` AND `y > 0`. Otherwise it falls back to `te_n`.
   This is the H.5.2 spec "if NW or NE does not exist, the value
   of N is used instead" rule, but our impl gates on `state.width`
   (the WP state's width = channel width = 32 for d1). At Y' sample
   22 (x=22, y=0), we have y=0 so the y>0 branch is FALSE — `te_ne`
   correctly falls back to `te_n` regardless. Verify this matches
   the spec's "rightmost border" worked example
   (true_err_NE → true_err_N at the rightmost border) which only
   covers the x = width-1 case, not the y = 0 case.

4. **`true_err` storage + read order audit** — H.5.1 says "After
   decoding a sample, the decoder then computes `true_err` and
   `err[i]`". Verify our implementation stores `true_err` BEFORE
   the next sample's `wp_predict` call reads it. (Round 19 audit
   noted this was correct, but at sample 22 specifically there's
   a NE neighbour at (23, -1) that doesn't exist; te_ne_raw falls
   back to te_n at (22, -1) which also doesn't exist → te_n=0 →
   te_ne=0. So the WP at (22,0) sees all-zero te_*. Why does the
   max_error compute differ from sample 21 then?)

The order above prioritises (2) since the first-divergent sample
location is now known and (2) is the cheapest in-tree diagnostic
to localise the bug to a specific (sample, property, leaf) tuple.
