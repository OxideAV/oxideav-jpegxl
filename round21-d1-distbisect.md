# Round-21 d1 per-cluster distribution + alias-table self-map audit (Auditor mode)

**Date**: 2026-05-09
**Fixture**: `crates/oxideav-jpegxl/tests/fixtures/vardct_256x256_d1.jxl`
**Round-20 candidates pursued**: (1) per-cluster distribution decode bisect
+ (2) alias-table self-map branch audit. Path (3)/(4)/(5) deferred.

## Per-cluster distribution dump (path 1)

The d1 LfGlobal entropy stream prelude (602 bits, `num_contexts=16
num_histograms=5 use_prefix_code=0 log_alpha_size=6` per cjxl trace)
defines five clusters. New diagnostic test
`tests/round21_d1_dist_alias_dump.rs` decodes the prelude and dumps
each cluster's `(cfg, D, alias_first_30)`. Confirmed:

```
table_size = 64, bucket_size = 64, log_bucket = 6

cluster 0: cfg(split_exp=4 msb=1 lsb=2 split=16)
  D sum=4096 nonzero=19/64 max=660
  bucket-stats: above=11 at=0 below_nz=8

cluster 1: cfg(split_exp=4 msb=1 lsb=2 split=16)
  D sum=4096 nonzero=23/64 max=644
  bucket-stats: above=11 at=0 below_nz=12

cluster 2: cfg(split_exp=0 msb=0 lsb=0 split=1)
  D sum=4096 nonzero=5/64 max=2560
  bucket-stats: above=5 at=0 below_nz=0

cluster 3: cfg(split_exp=4 msb=2 lsb=0 split=16)
  D sum=4096 nonzero=2/64 max=2070
  bucket-stats: above=2 at=0 below_nz=0

cluster 4: cfg(split_exp=4 msb=2 lsb=0 split=16)
  D sum=4096 nonzero=2/64 max=3279
  bucket-stats: above=2 at=0 below_nz=0
```

* Every cluster's distribution sums to 4096 (alias-build invariant).
* No cluster has any `D[i] == bucket_size` entry (`at=0` across all
  five). The alias-table self-map branch from round-3 (which only
  triggers when an entry equals exactly bucket_size) is **never
  exercised in d1**.

## Alias-table self-map branch audit (path 2)

Spec C.2.6 (2024 edition, p. 17) reads:

```pseudo
for (i = 0; i < alphabet_size; i++) {
  cutoffs.push_back(D[i]); symbols.push_back(i);
  if (cutoffs[i] > bucket_size) overfull.push_back(i);
  else if (cutoffs[i] < bucket_size) underfull.push_back(i);
}
```

Our `AliasTable::build` writes `if d[i] > bucket_size { overfull } else
{ underfull }`. The spec's `else if (cutoffs[i] < bucket_size)` skips
the equality case (puts neither queue); ours sweeps it into underfull.

**Empirical effect on d1**: zero. None of the five clusters has a
`D[i] == bucket_size` entry. Even if it did, hand-tracing the divergent
branch shows: when a balanced bucket enters underfull, the Vose pump
pops it, computes `by = bucket_size - cutoffs[u] = 0`, sets
`symbols[u] = o, offsets[u] = cutoffs[o]`, and leaves cutoffs[o]
unchanged. The outer overfull bucket is left in its original state and
the trailing reconciliation loop subsequently self-maps both. Net
result is identical to the spec-compliant skip. Documented as a
strict-spec divergence with no observable effect; deferred fix-up
because it would add complexity without changing any fixture's
behaviour.

## Cluster-1 alias-table 64-entry full dump

The d1 LfCoefficients per-token trace (round 19) confirms calls #0/#1
both land in cluster 1 (idx 0xf01 → slot 60, idx 0x903 → slot 36).
Cluster 1's full alias table:

```text
slot[ 0]: sym=  0 off=    0 cut=    0   slot[32]: sym=  4 off=  364 cut=   32
slot[ 1]: sym=  0 off=  320 cut=   16   slot[33]: sym=  4 off=  428 cut=    0
slot[ 2]: sym=  1 off=  160 cut=   16   slot[34]: sym=  4 off=  492 cut=    0
slot[ 3]: sym=  2 off=  256 cut=   46   slot[35]: sym=  7 off=   44 cut=    0
slot[ 4]: sym=  3 off=  128 cut=   38   slot[36]: sym=  8 off=    8 cut=    4
slot[ 5]: sym=  0 off=   64 cut=    0   slot[37]: sym=  8 off=   72 cut=    0
slot[ 6]: sym=  0 off=  128 cut=    0   slot[38]: sym=  8 off=  136 cut=    0
slot[ 7]: sym=  4 off=  512 cut=   44   slot[39]: sym=  8 off=  200 cut=    0
slot[ 8]: sym=  7 off=   96 cut=   12   slot[40]: sym=  8 off=  262 cut=    2
slot[ 9]: sym=  0 off=  192 cut=    0   slot[41]: sym=  8 off=  326 cut=    0
slot[10]: sym=  0 off=  256 cut=    0   slot[42]: sym=  8 off=  390 cut=    0
slot[11]: sym=  0 off=  272 cut=   48   slot[43]: sym=  8 off=  454 cut=    0
slot[12]: sym=  8 off=  580 cut=    2   slot[44]: sym=  8 off=  518 cut=    0
slot[13]: sym=  1 off=   16 cut=    0   slot[45]: sym= 12 off=    2 cut=    0
slot[14]: sym=  1 off=   80 cut=    0   slot[46]: sym= 12 off=   66 cut=    0
slot[15]: sym=  1 off=  112 cut=   32   slot[47]: sym= 12 off=  130 cut=    0
slot[16]: sym= 12 off=  384 cut=    2   slot[48]: sym= 12 off=  194 cut=    0
slot[17]: sym=  2 off=   14 cut=    2   slot[49]: sym= 12 off=  258 cut=    0
slot[18]: sym=  2 off=   78 cut=    0   slot[50]: sym= 12 off=  322 cut=    0
slot[19]: sym=  2 off=  110 cut=   32   slot[51]: sym= 16 off=    2 cut=    0
slot[20]: sym= 16 off=  384 cut=    2   slot[52]: sym= 16 off=   66 cut=    0
slot[21]: sym=  2 off=  174 cut=    0   slot[53]: sym= 16 off=  130 cut=    0
slot[22]: sym=  2 off=  238 cut=    0   slot[54]: sym= 16 off=  194 cut=    0
slot[23]: sym=  3 off=   38 cut=    8   slot[55]: sym= 16 off=  258 cut=    0
slot[24]: sym= 24 off=    0 cut=    0   slot[56]: sym= 16 off=  322 cut=    0
slot[25]: sym=  3 off=  102 cut=    0   slot[57]: sym= 20 off=    2 cut=    0
slot[26]: sym=  4 off=   38 cut=    0   slot[58]: sym= 20 off=   66 cut=    0
slot[27]: sym=  4 off=   94 cut=    8   slot[59]: sym= 20 off=  130 cut=    0
slot[28]: sym=  4 off=  142 cut=   16   slot[60]: sym= 20 off=  192 cut=    2
slot[29]: sym=  4 off=  206 cut=    0   slot[61]: sym= 20 off=  256 cut=    0
slot[30]: sym=  4 off=  270 cut=    0   slot[62]: sym= 24 off=   64 cut=    0
slot[31]: sym=  4 off=  332 cut=    2   slot[63]: sym= 24 off=  128 cut=    0
```

Self-map slots: only **slot[0]** (sym=0 off=0 cut=0; original D[0]=384
overfull, gets fully consumed by Vose) and **slot[24]** (sym=24 off=0
cut=0; D[24]=192, also overfull and self-mapped after Vose redirects).
Both arise from the trailing-loop's `cutoffs[i] == bucket_size` branch
firing AFTER the Vose pump leaves them at exactly bucket_size — no
spec-divergence trigger.

## Round-19 trace cross-check

Calls #0 and #1 (verified bit-faithful in round 19):

```text
call #0: pre=0x18c81f01 idx=0xf01 → i=60,  pos=1
         cluster-1 slot[60]={sym=20, off=192, cut=2}; pos<cut → sym=60, off=1 ✓
         prob=D[1][60]=2, new = 2*(0x18c81f01>>12)+1 = 0x00031903 ✓

call #1: pre=0x00c7270d idx=0x903 → i=36,  pos=3
         cluster-1 slot[36]={sym=8,  off=8,  cut=4}; pos<cut → sym=36, off=3 ✓
         prob=D[1][36]=4, new = 4*(0x00c7270d>>12)+3 = 0x000000c7 ✓
```

Both calls land in **cluster 1** (the leaf with ctx=1 reached when
`prop[15] = wp_max_error > 0`). With the round-19 cluster-map dump
showing ctx[1] → cluster 1, this is consistent.

## What r21 falsified

* **Per-cluster distribution decode** — all 5 distributions sum to 4096
  with no malformed entries. The 602-bit prelude split is bit-exact.
* **Alias-table self-map branch (round-3 fix)** — d1 has zero
  bucket_size-equal entries, so the spec-divergent `else` (vs `else if
  < bucket_size`) is never triggered. Cluster 1 self-maps only at
  slot[0] and slot[24], both via the standard Vose-then-trailing-loop
  path.
* **Cluster-1 alias entries** — 64/64 entries dumped; all Vose
  invariants hold (sum of cutoffs[i] over redirect contributors equals
  bucket_size for each contributor; slots' (sym, off) pairs reconcile
  with the lookup formula).

## Spec divergence noted (no fix landed)

Our `AliasTable::build` initial-bucket-classification loop reads:

```rust
if d[i] as u32 > bucket_size { overfull.push(i); } else { underfull.push(i); }
```

while the 2024-edition spec at C.2.6 reads:

```pseudo
if (cutoffs[i] > bucket_size) overfull.push_back(i);
else if (cutoffs[i] < bucket_size) underfull.push_back(i);
```

(Spec skips the equality case entirely.) For d1 this is observationally
inert (no equal-bucket entries exist), and the hand-trace of the
hypothetical equal-bucket flow shows our impl converges to the same
output. Logged here as a strict-conformance fix candidate when a
fixture exhibits an equal-bucket distribution and r22+ wants to
eliminate the divergence.

## Sentinels

* All five small lossless fixtures stay regression-free.
* Round-3..20 sentinel tests stay green.
* Round-10 synth_320 drift bisect stays green (untouched).
* New round-21 test (1 file): `round21_d1_dist_alias_dump`.
* Test count: 336 → 337 (+1).

## Round-22 candidates

With the per-cluster distribution and alias-table fully audited and
the 602-bit prelude + 5 distributions all bit-exact and well-formed,
the LfCoefficients ANS final-state divergence (`0x21914271 ≠
0x130000`) cannot be attributed to the prelude or alias-mapping paths.
The remaining surface area:

1. **Sample arithmetic eyeball-validation (r20 path 4)** —
   `lfc.lf_quant[c]` first-256 dump per channel; if the values look
   like a smooth gradient with cjxl-plausible local statistics, the
   per-sample decode is OK and the final-state divergence is
   bookkeeping (extra ANS state init somewhere, missing end-of-stream
   read). If they look like garbage, the divergence is in the decoded
   sample stream itself.

2. **Mid-stream ANS state cross-check** — capture every 256th
   `(pre_state, idx, sym, prob, new_state)` tuple for our 3072 calls
   and compare against a hand-encoded oracle constructed by re-running
   our forward (encoder-side) Vose alias-inverse. If our state diverges
   from the oracle at call N, the bug is in the per-symbol arithmetic;
   if it tracks the oracle but ends in the wrong final-state, the
   encoder must have intended a different call count.

3. **WP `(wp_pred8 + 3) >> 3` rounding cross-check** — the round-3
   `(p + 3) >> 3` (vs spec-literal `(p + 4) >> 3`) for predictor 6 was
   debated; if the predictor's quantised output diverges by 1 LSB at
   any sample, the corresponding `prop[15]` (max_error) shifts and the
   tree may pick the wrong leaf for the next sample — once. Add a test
   that decodes d1 LfCoefficients with `(p + 4) >> 3` and reports the
   final ANS state delta.

4. **cjxl source build with JXL_TRACE on (r20 path 5)** — heavy
   external dep but the only way to get a per-call cjxl bit-position
   reference. Defer until at least one of (1)-(3) gives a concrete
   suspect.

The order above prioritises (1) since it's the cheapest in-tree
diagnostic and the result strongly bisects between bookkeeping and
sample-decode bugs.
