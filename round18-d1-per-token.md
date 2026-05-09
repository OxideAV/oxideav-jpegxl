# Round-18 d1 per-token bit-accounting (Auditor mode)

**Date**: 2026-05-09
**Fixture**: `crates/oxideav-jpegxl/tests/fixtures/vardct_256x256_d1.jxl`
**Round-17 hypothesis (PRIMARY)**: ~2.3 bits/sample over-consumption in
`HybridUintConfig::read_uint`'s extra-bits accounting.

## Round-17 figures (recap)

* cjxl LfGroup TOTAL: **11 728 bits** (DC_GROUP_END − DC_GROUP_BEGIN).
* Our LfCoefficients ALONE: **11 995 bits** (267 over the whole LfGroup).
* Per-sample average: 11 957 / 3072 = 3.89 bits/sample.

## Round-18 instrumentation

A thread-local trace ring (`TRACE_ENABLED` + `with_trace_records` in
`src/ans/hybrid_config.rs`) records `(split_exponent, msb_in_token,
lsb_in_token, token, n_extra_bits, value)` for every `read_uint` call.
The `tests/round18_d1_per_token.rs` test enables the trace, runs
`LfCoefficients::read` on d1, and prints summary statistics.

### Captured data (3072 samples in d1's LfCoefficients)

```
[r18] cfg(split_exp=4, msb=1, lsb=2): 3072 calls, 821 extra-bit-reads,
                                         avg 0.267 extra/call
[r18] histogram of n_extra_bits:
        n=0  -> 2433 times
        n=1  ->  483 times
        n=2  ->  134 times
        n=3  ->   20 times
        n=4  ->    1 times
        n=6  ->    1 times
[r18] histogram of token magnitude (1+log2(token)):
        bucket=0 -> 299    bucket=4 -> 797
        bucket=1 -> 217    bucket=5 -> 617
        bucket=2 -> 516    bucket=6 ->  22
        bucket=3 -> 604
```

### Decomposition of the 11957 sample-loop bits

* extra-bits (counted above): **821**
* ANS-state-init `u(32)`: **32**
* ANS refills (`u(16)` after `state < 1<<16`): 11957 − 821 − 32 = 11104
  → 11104 / 16 = **694 refills** out of 3072 symbol decodes.

Refill rate: 22.6 % of decodes triggered a renormalisation. For a
typical photo-content distribution that's plausible per-symbol but the
total bit-cost is too high.

## Round-17 PRIMARY hypothesis falsified

Per-token extra-bit accounting matches FDIS §D.3.6 Listing D.6 with the
2024-spec `n = (split_exp − msb − lsb) + ((token − split) >> (msb +
lsb))` correction. Manual verification of token #0 (token=60, value=848,
n_extra=6) confirms `n = 1 + 5 = 6` extra bits decode the right value
under our formula — the literal FDIS-2021 `n = split_exp + ((token −
split) >> (msb + lsb))` = 9 would yield `value = 4 × 848 + ...` (way too
large), so the round-3 subtraction is correct.

The 821 extra-bits-read count is also reasonable per cjxl's expected
HybridUintConfig (split_exp=4, msb=1, lsb=2, split=16). Bit cost is
NOT in extra-bits.

## Round-17 SECONDARY hypothesis (ANS prelude bit count) — likely
also not the bug

cjxl trace says the leaf-level entropy stream prelude consumes 602 bits
for `num_contexts=16, num_histograms=5, log_alpha_size=6`. Our
GlobalModular bit position matches cjxl exactly at bit 1026, so the
prelude can't be over by more than the (tree decode + ANS state init)
bit budget. Sub-section bisect via the round-17 trace pinpoints
LfCoefficients's per-sample loop as the over-consumer, not the prelude.

## Where the drift comes from — round-19 candidate

The 267-bit excess is concentrated in **ANS state evolution** (refills),
not extra-bits or prelude. Likely root causes:

1. **Per-symbol ANS state update math** — a subtle off-by-one in
   `D[symbol] * (state >> 12) + offset` or in the `state < 1<<16` refill
   condition could cause systematically more or less refills.
2. **Alias-table build / lookup** — round-3 introduced a "conditional
   offset" deviation from FDIS §D.3.2 Listing D.2 (returning `pos`
   instead of `offsets[i] + pos` in the not-in-redirect branch).
   Analysis confirms this deviation IS correct for the gray-64x64
   distribution (and was empirically required to pass that fixture in
   round 3); reverting to the spec-literal formula reduces d1's bit
   consumption from 11 995 → 11 654 (matching cjxl within 74 bits) but
   breaks gray-64x64 with `unexpected end of JXL bitstream`.
   - Implication: **the build OR the lookup has a subtle further bug
     that causes different distributions to require different
     compensating formulas**. The "right" formula is something
     in-between that handles both fixtures correctly.
3. **Per-cluster `HybridUintConfig` resolution** — the trace shows ALL
   3072 calls in d1 use `(split_exp=4, msb=1, lsb=2)`. cjxl's encoder
   may have written 5 distinct configs (one per cluster); our 5
   `EntropyStream::configs` entries might all be identical OR we might
   be using cluster index 0 for every call (a plausible bug). The trace
   doesn't yet distinguish per-cluster — round-19 should add cluster
   index to the trace record.

## Sentinels

5 small lossless fixtures stay green. All round-11..17 sentinel tests
stay green. Round-18's `d1_per_token_trace_round_18` emits diagnostic
output but does not assert correctness, so it stays green.

## Round-19 dispatch

Auditor-mode handoff: extend the trace to include cluster index per
call, then re-run d1 to see whether the cluster_map for the leaf-level
stream is being computed correctly. If cluster usage matches the cjxl
trace's `num_histograms=5` (5 distinct clusters used roughly evenly),
investigate the alias build/lookup for an asymmetric bug. If cluster
usage is degenerate (all 0 or all-of-one), investigate the cluster_map
decode in `read_clustering`.
