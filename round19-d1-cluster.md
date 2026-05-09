# Round-19 d1 cluster + ANS-state evolution audit (Auditor mode)

**Date**: 2026-05-09
**Fixture**: `crates/oxideav-jpegxl/tests/fixtures/vardct_256x256_d1.jxl`
**Round-18 hypothesis**: ANS state evolution / cluster_map resolution.

## Round-18 figures (recap)

* cjxl `DC_GROUP` total: **11 728 bits** (DC_GROUP_END − DC_GROUP_END).
* Our `LfCoefficients` alone: **11 995 bits** (267 over).
* 3072 `read_uint` calls; 821 extra-bit reads, 694 ANS refills (round-18 count).

## Round-19 instrumentation

Round 19 extends the trace ring (`TRACE_ENABLED`, `with_trace_records`)
with three new fields on each `TraceRecord`:

* `ctx` — leaf context the call resolved to (forwarded by
  `decode_uint_in_with_dist` via `push_trace_extras`).
* `cluster` — cluster index the context resolved through `cluster_map`.
* `ans_refill_bits` — bits consumed by ANS renormalisation alone for
  this call's symbol decode (16 if a refill fired, 0 otherwise — courtesy
  of new `AnsDecoder::decode_symbol_with_refill`).

A new diagnostic eprintln in `MaTreeFdis::read` reports the leaf-stream
`EntropyStream::read` prelude bit count; a new state-trace ring
(`STATE_TRACE_BUF`) records the first 30 ANS state transitions for
spot-checking against the raw codestream. The new test
`tests/round19_d1_cluster.rs` drives the d1 LfCoefficients sub-bitstream
under all three traces and emits per-cluster / per-ctx histograms.

## Round-19 captured data

### Prelude bit count — EXACT MATCH with cjxl

```
[r19-prelude] num_ctx=16 leaf-stream EntropyStream::read consumed 602 bits
              (cjxl trace says 602 for 16/5/6)
```

cjxl trace (`docs/image/jpegxl/fixtures/vardct-256x256-d1/trace.txt`)
reports `ENTROPY num_contexts=16 num_histograms=5 lz77_enabled=0
use_prefix_code=0 log_alpha_size=6 bits=602`. Our prelude matches **bit
for bit**. The 2024-spec C.2.1 reading (`use_prefix_code=0 →
log_alphabet_size=5+u(2)`) is corroborated against this fixture.

### Cluster map — EXACTLY 16→5 as cjxl expects

```
ctx[0] -> cluster 0     ctx[8]  -> cluster 1
ctx[1] -> cluster 1     ctx[9]  -> cluster 3
ctx[2] -> cluster 2     ctx[10] -> cluster 1
ctx[3] -> cluster 3     ctx[11] -> cluster 1
ctx[4] -> cluster 4     ctx[12] -> cluster 2
ctx[5] -> cluster 3     ctx[13] -> cluster 1
ctx[6] -> cluster 3     ctx[14] -> cluster 1
ctx[7] -> cluster 3     ctx[15] -> cluster 2
```

`max(cluster_map) + 1 == 5`, matching cjxl's `num_histograms=5`. Per
cluster `HybridUintConfig`:

```
cfg[0] split_exp=4 msb=1 lsb=2 split=16
cfg[1] split_exp=4 msb=1 lsb=2 split=16
cfg[2] split_exp=0 msb=0 lsb=0 split=1
cfg[3] split_exp=4 msb=2 lsb=0 split=16
cfg[4] split_exp=4 msb=2 lsb=0 split=16
```

Per-cluster distribution sums all equal 4096 (alias-build invariant).
D[cluster 0] has 19 nonzero entries / 64; D[cluster 1] has 23 nonzero;
D[cluster 2..4] are extremely concentrated (2-5 nonzero entries).

### Tree walk audit — REACHABLE LEAVES = ctx 0 + ctx 1 ONLY

Tree root tests `prop[1] > 2`. `prop[1]` is `stream_index`. For
LfCoefficients lf_group_idx=0, `stream_index = 1 + 0 = 1`, NOT > 2 → R
subtree → node #2 (`prop[15] > 0`) → leaf #5 (ctx=0) or #6 (ctx=1),
both with predictor 6. The other 14 contexts (ctx 2..15) live under
the L subtree and are reached **only** by HfMetadata (stream_index=3,
> 2). This is by design.

Per-cluster usage:

```
cluster 0: 1615 calls,  294 extra-bits, 5872 refill-bits (=367 refills), ctxs=1
cluster 1: 1457 calls,  527 extra-bits, 5264 refill-bits (=329 refills), ctxs=1
TOTAL:    3072 calls,  821 extra-bits, 11136 refill-bits (=696 refills)
```

(Per-ctx counts: ctx 0 = 1615, ctx 1 = 1457. Matches cluster split
because cluster_map[0] = 0 and cluster_map[1] = 1.)

### Sample-loop bit reconstruction

```
6 (LfCoeff hdr) + 32 (state init) + 821 (extra) + 11136 (refill) = 11995
```

Matches the observed consumed count exactly. **Every bit is accounted
for**. The 267-bit overshoot is concentrated in the **ANS refill
budget**: 696 refills at 16 bits each = 11136 bits.

### State-evolution sanity check

First 5 ANS state transitions verified bit-for-bit against the raw
codestream:

```
state init u(32) at codestream bit 1104 = 0x18c81f01    ✓ matches raw bits
call #0  pre=0x18c81f01 idx=0xf01 sym=60 prob=2  off=1   new=0x00031903 refill=0
call #0  read_uint extra u(6)  at bit 1136 = 20 (0b010100)   value=848   ✓
call #1  refill u(16)         at bit 1142 = 0x270d            ✓
call #1  pre=0x00c7270d idx=0x903 sym=36 prob=4  off=3   new=0x000000c7 refill=16
call #1  read_uint extra u(3)  at bit 1158 = 2               value=104  ✓
```

The state evolution IS bit-faithful to the codestream. Our alias build,
alias lookup (with the conditional `pos < cutoffs[i] → offset = pos`
branch retained), and `read_uint` formula (the round-3 `n =
split_exp - msb - lsb + n_extra` correction) all produce the values
the encoder wrote.

## Round-19 candidates falsified

* **Prelude bit count** → matches cjxl exactly (602 bits).
* **Cluster map decode** → 16 contexts collapse to 5 distinct clusters
  exactly as cjxl signals.
* **`read_uint` extra-bits** → token #0's `n_extra=6, value=848` matches
  the raw codestream extra bits (`u(6) = 20`); literal-spec
  `n = split_exp + n_extra` (= 9) would be impossible for value=848.
* **Alias lookup `offset = pos`** → reverting to spec-literal `offsets[i]
  + pos` for the self-map case yields offsets in `[bucket_start,
  bucket_end)` outside `D[i]`'s probability range; that breaks
  gray-64x64 (round-18 confirmed) and is mathematically incoherent.
* **WP `s = (sum_weights >> 1) - 1`** → reverted to spec-literal
  `s = sum_weights >> 1` shifts the round-10 synth_320 first-drift
  y-coord from 24 to 8, so the round-3 `- 1` IS the right reading
  empirically (regardless of FDIS Listing E.3's literal text).
* **`MaTreeFdis::cloned_with_fresh_state`** — clones the tree's nodes,
  cluster_map, configs, and entropies; resets `ans_state = None` so a
  fresh `u(32)` is read at the next `read_ans_state_init` call;
  re-allocates the 1 MiB hybrid-uint sliding window. **No leak found.**

## Round-19 fix

**No fix landed.** All five round-18 hypotheses are now ruled out:
prelude is bit-exact, cluster_map is bit-exact, configs are bit-exact,
state transitions are bit-faithful to the raw codestream, and
cloned_with_fresh_state is correct.

The 267-bit overshoot must therefore lie in either:

1. **Sample-count arithmetic** — we decode 3072 LF samples (3 × 32 × 32);
   if cjxl decodes fewer (e.g. only 2 channels per LF group for some
   XYB-encoded paths), our overshoot is purely "extra channel(s) we
   shouldn't be reading". §C.5.3 explicitly says LfQuant has **three
   channels** so this is a long shot.
2. **HfBlockContext reads + 4-bit stuffing inside d1's
   `nb_block_ctx=3` custom path** — our trace shows HfBlockContext
   consumes 87 bits matching the cjxl `DC_GLOBAL_END=1026` checkpoint,
   so this is also a long shot.
3. **A different cjxl interpretation of the LfGroup section's *total*
   bits**: cjxl trace's `DC_GROUP_END bits_consumed=12754` may include
   some byte-padding / section-boundary bits that we count separately
   (e.g. the next-section's TOC-driven byte alignment).
4. **An unaccounted-for read inside `decode_channels_at_stream`'s pre-
   loop fields** — e.g. a missing `pu0()` byte alignment, or a stray
   read between the inner `read_ans_state_init` and the per-pixel
   loop. `decode_channels_at_stream` reads the ANS state init then
   immediately enters the per-channel y/x loop, so the pre-loop side
   is bit-tight; there's no obvious gap.

### r20 candidate

Ship a side-by-side bit-position trace via cjxl `--debug` (or libjxl
`JXL_TRACE`) that reports **per-call bit positions**, then diff
against our `[r19]` first-30 trace. The first call where cjxl's
post-decode bit-position diverges from ours is where the bug lives.
Without per-call cjxl positions we cannot localise further from the
oxideav side alone.

If the divergence is at call #N for some specific N, then:

* **N = 0 (state init)** → state-init bit position is wrong
  (unlikely — we verified call #0 against raw bits).
* **N small (1..50)** → an early refill / extra-bits read is
  consuming the wrong number of bits.
* **N near 3072 (end of LF loop)** → an end-of-stream / final-state
  check is missing or extra.
* **A different N for the cjxl trace's reported `DC_GROUP_END` field
  meaning** — possibly cjxl's "11728 bits for DC_GROUP" actually counts
  through ModularLfGroup + HfMetadata too, and our LfCoefficients-only
  measurement is the right number; in which case there is no actual
  drift, and the round-16 HfMetadata `Squeeze begin_c=39` blocker is
  caused by a different upstream discrepancy.

The latter possibility is intriguing: the round-17/round-18 reasoning
assumed `LfCoefficients << 11728` (because cjxl's split was estimated
at ~5000 + ~6800), but if cjxl's `DC_GROUP` counts only LfCoefficients
(plus zero-byte ModularLfGroup), then **our 11995 over their 11728 is
a 267-bit overshoot in the LF channel decode itself** — far smaller
relative drift, and the HfMetadata "begin_c=39" symptom is then a
different bug (likely a missing channel-list normalisation between
LfCoefficients and HfMetadata).

## Sentinels

Five small lossless fixtures stay green. Round-3..18 sentinels stay
green. Round-10 synth_320 drift bisect stays green (verified after
the WP `- 1` revert experiment was rolled back). New
`d1_cluster_and_refill_trace_round_19` test passes.
