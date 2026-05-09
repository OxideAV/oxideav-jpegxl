# Round-20 d1 DC_GROUP boundary recount + ANS-final-state oracle (Auditor mode)

**Date**: 2026-05-09  
**Fixture**: `crates/oxideav-jpegxl/tests/fixtures/vardct_256x256_d1.jxl`  
**Round-19 hypothesis**: 267-bit overshoot in `LfCoefficients` per-token decode.  
**Round-20 finding**: the 267-bit overshoot is an artefact of misreading
the cjxl `JXL_TRACE` output. The real bug is upstream of the per-token
loop — the ANS state at the end of `LfCoefficients` is `0x21914271`,
not the spec's `0x00130000` end-of-stream sentinel.

## Path (b) per round-19 dispatch — confirmed

cjxl's `JXL_TRACE`-derived `docs/image/jpegxl/fixtures/vardct-256x256-d1/trace.txt`
emits (excerpt):

```text
DC_GLOBAL_END   bits_consumed=1026
DC_GROUP_END    bits_consumed=12754
AC_GLOBAL_END   bits_consumed=307
```

Round 17 / 18 / 19 read `bits_consumed` as a *cumulative file
position*, in which case `DC_GROUP` would span `12754 − 1026 = 11728`
bits. Under that reading our `LfCoefficients` (alone consuming 11995
bits) was 267 bits over the entire `LfGroup` budget — supposedly the
defining symptom blocking d1.

**That reading is wrong.** Inside the same trace,
`AC_GLOBAL_END num_histograms=1 bits_consumed=307`. Since
`307 < 1026 = DC_GLOBAL_END`, the value cannot be a cumulative file
position (the section starts at the end of `DC_GROUP`, which is
already deeper than `307`). So `bits_consumed` is **section-local**:
it counts the bits inside that one named section.

Under the corrected reading:

| Section          | Section-local bits | Absolute bit range          |
|------------------|--------------------|-----------------------------|
| `DC_GLOBAL`      | 1026               | `[0,     1026)`             |
| `DC_GROUP`       | **12754**          | `[1026,  13780)`            |
| `AC_GLOBAL`      | 307                | `[13780, 14087)`            |
| `AC_GROUP`       | (≤ 2569)           | `[14087, ≤16656)`           |

So the `LfGroup` (= LfCoefficients + ModularLfGroup + HfMetadata)
budget is **12754 bits**, not 11728. With our LfCoefficients consuming
11995 bits and `ModularLfGroup` empty (no qualifying channels for
d1), HfMetadata's implied budget is **`12754 − 11995 = 759` bits**,
and the alleged 267-bit overshoot vanishes.

This is empirically confirmed by
`tests/round20_d1_dc_group_recount.rs::d1_dc_group_recount_round_20`,
which now asserts `LfCoeff_end ≤ DC_GROUP_end`.

## ANS final-state — NEW oracle for round 21

Per FDIS D.3.3 last sentence (also Listing M.1's encoder side):

> *After the decoder reads the last symbol in a given stream, `state`
> is `0x130000`.*

Our `AnsDecoder::final_state` exposes this check, but production code
never invokes it. Round 20 wires a thread-local "latest state" sink
via [`LATEST_ANS_STATE`] / [`LATEST_ANS_CALL_COUNT`] (in
`src/ans/symbol.rs`) so a test can read the post-decode state without
holding the per-stream `MaTreeFdis` clone alive.

Result on d1 (test
`tests/round20_d1_ans_final_state.rs::d1_lfcoefficients_ans_final_state_round_20`):

```text
LfCoefficients final ANS state = 0x21914271 after 3072 decode_symbol calls
ANS_FINAL_STATE (D.3.3 sentinel) = 0x00130000
final_state == ANS_FINAL_STATE? false
```

**`0x21914271 ≠ 0x00130000`.** Our LfCoefficients per-sample loop is
*decoding the wrong number of samples*, OR *one of the per-cluster
distributions / alias tables is wrong*, OR *the alias-mapping (sym,
offset) lookup is producing a different result than cjxl's encoder*.
In any of these cases the state evolution diverges and end-of-stream
verification fails.

The state-walk test (`tests/round20_d1_ans_state_walk.rs`)
confirms: in 3072 calls the state never reaches `0x00130000`. So
truncating the per-sample loop early (at any `K < 3072`) won't
recover the sentinel either — the divergence is structural, not a
sample-count off-by-one.

This is THE r21 starting point: a precise, side-effect-free oracle.

## HfMetadata bit-position drift — distinct symptom

With LfCoefficients's cursor landing at section-local bit 13021
(file bit 13021 + leading 72 = 13093 absolute), HfMetadata's first
233 bits parse as:

```text
nb_blocks_minus_1 = u(10) = 609 → nb_blocks = 610
inner_use_global_tree = false
WPHeader (default_wp = false) → 52 bits of explicit WP fields
nb_transforms = 1
transform[0] = Squeeze, num_sq = 11
SP[0] horizontal=true in_place=true begin_c=39 num_c=2 → ERROR (begin_c+num_c>4)
```

Both `inner_use_global_tree=false` AND `default_wp=false` are
extremely atypical for a cjxl-encoded VarDCT HfMetadata (the global
tree is shared and saves bits; default WP is the standard choice).
Most cjxl-encoded HfMetadata sub-bitstreams parse as
`iugt=true default_wp=true nb_transforms=0` (4 bits total) followed
by per-channel decode.

Combined with the ANS final-state mismatch, the conclusion is that
**LfCoefficients consumes a different bit budget than cjxl's
encoder*** — which subsequently mis-aligns HfMetadata. We don't yet
know whether LfCoefficients is OVER- or UNDER-running, since the ANS
state oracle only tells us "wrong", not "off by N samples".

The r20-alignment-scan test (`tests/round20_d1_hfmeta_alignment.rs`)
sweeps `[lfc_end - 270, lfc_end + 5)` looking for a starting position
that yields `iugt=true, default_wp=true, nb_transforms ∈ {0,1}`. It
finds dozens of candidates (any 1-bit `iugt`/`dwp` patterns trigger
high scores), so this scan alone is inconclusive without the ANS
oracle pinning the true endpoint.

## Sentinels

* All five small lossless fixtures stay regression-free.
* All round-3..19 sentinel tests stay green.
* Round-10 synth_320 drift bisect stays green (untouched).
* New round-20 tests (5 files): `d1_dc_group_recount`,
  `d1_lfcoefficients_ans_final_state`, `d1_lfcoefficients_full_state_walk`,
  `d1_hfmeta_field_walk`, `d1_hfmeta_alignment_scan`. None assert
  divergence as a hard failure — Auditor mode.
* Test count: 331 → 336 (+5).

## Round-21 candidates

1. **Bisect the per-cluster distribution decode.** Each cluster's
   probability table is decoded by `EntropyStream::read`; instrument
   the decoder to log every distribution alongside the symbol stream
   it backs, and diff against the cjxl trace's
   `ENTROPY num_contexts=16 num_histograms=5 ... bits=602` prelude.
   The 602-bit total matches but the per-cluster *split* into 5
   distributions could differ.

2. **Bisect alias-table construction.** `AliasTable::build` was
   touched in round 3 (the "self-map case `pos < cutoffs[i] →
   offset = pos`" branch). Spec D.3.5 (alias mapping) is dense; a
   subtle bucket-fill divergence would silently mis-decode without
   raising an error. Cross-check by hand-building the cluster-3
   alias table (D = `[0]=2026, [14]=2070`, log_alpha=6 → 64 slots,
   bucket_size = 4096/64 = 64) and printing every (state, sym, off)
   row alongside the spec's reference sequence.

3. **Cross-validate `read_uint` formula on cluster-2.** Cluster 2
   uses `split_exp=0 msb=0 lsb=0 split=1` — every token is
   "above split" so the formula `n = split_exp - msb - lsb +
   ((token - 1) >> 0) = 0 + (token - 1)`. For tokens 5/6/7
   (probability 512/2560/256 = 82% of cluster 2 mass) this gives
   `n = 4/5/6` extra bits per call — a sizable budget. If the
   round-4 "− msb − lsb" correction is wrong here, the divergence
   would be concentrated in cluster-2 calls. (NOTE: cluster 2 is
   only reachable via the L-subtree of the global tree, used by
   HfMetadata, NOT LfCoefficients. So this is a HfMetadata-side
   concern.)

4. **Run an end-to-end pixel-level XYB → sRGB pipeline on the
   already-decoded LfQuant** (`first8=[424, 476, …]`) and visually
   inspect. If the `lf_quant` array is sensible (smooth gradients
   matching the source), then the per-sample decode is OK and the
   ANS final state divergence is *purely* a bookkeeping issue —
   maybe an extra ANS read somewhere we don't track.

5. **Path (a) revisited**: build cjxl from source with `JXL_TRACE`
   enabled (the shipped `cjxl v0.11.1` doesn't have the trace points
   compiled in — verified empirically with `JXL_TRACE=1 djxl input.jxl
   /tmp/out.png` producing zero trace lines). Once a per-call bit
   position trace is available, diff against ours.

The order above prioritises (1)-(3) since they don't require building
cjxl from source.
