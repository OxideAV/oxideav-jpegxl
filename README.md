# oxideav-jpegxl

Pure-Rust **JPEG XL** (ISO/IEC 18181-1:2024) decoder. Resumed
2026-05-08 against the final published 2024 core spec after the
trace-doc-driven rounds 7-11 + encoder rounds 1-6 were retired
(see "Why retired (history)" below).

**Round 238 (2026-06-05)** lands the
`hf_coeff_histogram_size::HfCoefficientHistogramSize` typed sizing
primitive for the §C.7.2 HF coefficient histogram block. The spec
line "Let `nb_block_ctx` be equal to `max(block_ctx_map)+1`. The
decoder reads a histogram with `495 × num_hf_presets × nb_block_ctx`
clustered distributions D from the codestream as specified in D.3"
now has a single typed home: `HfCoefficientHistogramSize::new(
num_hf_presets, nb_block_ctx)` direct constructor + `from_block_ctx_map(
map, num_hf_presets)` deriving `nb_block_ctx` from the §C.7.2 line-1
`max(block_ctx_map) + 1` rule. Accessors: `per_preset()`
(`495 × nb_block_ctx` — single-preset distributions),
`num_distributions()` (`495 × num_hf_presets × nb_block_ctx` — the
§C.7.2 total), and `offset_for_hfp(hfp)` (`495 × nb_block_ctx × hfp`
— the §C.8.3 per-pass routing offset, range-checked on `hfp <
num_hf_presets`). Spec constant published as
`pub const PER_PRESET_PER_BLOCK_CTX: u64 = 495`. The duplicated
`495u64 * num_hf_presets * nb_block_ctx` and
`495u64 * nb_block_ctx * hfp` arithmetic in `hf_pass::HfPass::read`
and `pass_group_hf::PassGroupHfHeader::read` is now routed through
the primitive so the spec constant has one home and the per-pass
offset derivation shares its `nb_block_ctx` factor with the §C.7.2
read-size. Defensive zero-input guards reject `num_hf_presets == 0`,
`nb_block_ctx == 0`, and an empty `block_ctx_map`. 5 unit + 6
integration (`round238_hf_coeff_histogram_size`) tests pin:
default-shape 39-entry `block_ctx_map` deriving `nb_block_ctx = 15`
and `num_distributions == 7425`; multi-preset multi-context
arithmetic (`num_hf_presets ∈ {1, 2, 4, 8}` × `nb_block_ctx ∈ {1, 7,
15}`); `offset_for_hfp(hfp)` stepping uniformly by `per_preset()`
across `hfp ∈ [0, num_hf_presets)`; out-of-range `hfp` rejected;
zero-input rejection on every constructor; `HfPass::read` and
`PassGroupHfHeader::read` post-refactor producing bit-identical
sizes/offsets to the primitive's direct computation across the
worked-example matrix. Lib tests 705 → 710 (+5). Pure-control-flow
sizing primitive — no bit reads, no spec re-derivation, no
histogram materialisation, no ANS state setup. The actual §C.7.2
read of the clustered-distributions block (the
`EntropyStream::read(br, num_distributions)` call against the
read-size the primitive now computes) remains a deferred next step.

**Round 232 (2026-06-04)** lands the per-LfGroup multi-pass HF-header
+ per-pass `histogram_offset` routing driver
`decode_multi_pass_with_hf_headers` — the §C.8.3 first-paragraph
`hfp = u(ceil(log2(num_hf_presets)))` + derived
`histogram_offset = 495 × nb_block_ctx × hfp` wiring above the
round-228 per-LfGroup multi-pass three-channel driver. New
`multi_pass_hf_header` module exposes the typed `PerPassHfHeaders`
container (one [`pass_group_hf::PassGroupHfHeader`] per pass) with
`PerPassHfHeaders::read(br, num_passes, num_hf_presets,
nb_block_ctx)` consuming the per-pass header sequence from a
[`bitreader::BitReader`] and `PerPassHfHeaders::from_headers(vec)`
constructing from a pre-built `Vec`. Accessors expose per-pass
`hfp` + `histogram_offset` + a `PassHfDigest` snapshot for callers
performing the per-pass [`hf_pass::HfPass`] coefficient-order lookup
or routing the offset into a per-pass entropy stream. The driver
itself wraps
[`multi_pass_decode::decode_multi_pass_three_channels_with_resolver`]
with two augmented closure shapes
`read_non_zeros(p, channel, predicted, histogram_offset)` /
`decode_symbol(p, channel, coeff_ctx, histogram_offset)`. The
per-pass offset is pre-resolved once per pass before the inner walk
so the closure body sees a constant offset across each pass's
per-channel calls. The companion
`read_and_decode_multi_pass_with_hf_headers(br, ...)` reads the
per-pass header sequence inline from a `BitReader` and invokes the
driver in one call — the entry-point a future round wiring §C.7.2
entropy histograms (#799 DOCS-GAP) into per-pass `EntropyStream`s
will use. 16 unit + 12 integration
(`round232_multi_pass_hf_header`) tests pin: per-pass header read
with `num_hf_presets ∈ {1, 2, 4, 8}` (single-preset zero-bit fast
path, two-preset one-bit-per-pass, four-preset two-bits-per-pass,
eight-preset three-bits-per-pass with 15 bits across 5 passes);
digest round-trip through bits LSB-first; `hfp = 0` always yielding
`histogram_offset = 0` regardless of `nb_block_ctx`; offset scaling
with `nb_block_ctx`; out-of-range errors; zero-passes degenerate
empty container; `num_hf_presets == 0` rejection propagating
through `PerPassHfHeaders::read`; per-pass offset uniformly routed
across the three channels (X / Y / B) within a pass; both
`read_non_zeros` and `decode_symbol` closures receiving the
matching per-pass offset (378 = 2 × 3 × 63 decode_symbol calls
covering the full DCT8×8 `k ∈ [num_blocks, size)` sweep); per-pass
error propagation; `num_passes` mismatch rejected pre-walk;
pass-distinct `qdc_at` invocation preserving the round-228 per-pass
`qdc[3]` propagation; mixed transform `DCT16×8 + 2 DCT8×8` layout
consistency across passes with distinct per-pass offsets; inline
read+decode end-to-end (header bits consumed exactly, decode walk
runs, output shape matches); inline-read error path (empty
`BitReader` yields a proper `Error::InvalidData` from `read_bit`).
Lib tests 689 → 705 (+16). Pure-control-flow primitive in the same
shape as round-89 [`dct_quant_weights`], round-95 [`hf_dequant`],
round-121 [`llf_from_lf`], round-138 [`chroma_from_luma`],
round-141 [`gaborish`], round-144 [`epf`], round-147
[`afv::afv_idct`], round-159 / 164 [`pass_group_hf`], round-177
[`non_zeros_grid`], round-183 [`per_channel_non_zeros`], round-190
[`per_pass_non_zeros`], round-208 [`varblock_walk`], round-214
[`block_context_resolver`], round-221's three-channel driver, and
round-228's multi-pass driver — no bit reads beyond the spec-line
`hfp` u-read, no spec re-derivation, no histogram materialisation,
no ANS state setup.

**Round 228 (2026-06-04)** lands the per-LfGroup **multi-pass**
three-channel varblock decode driver
`decode_multi_pass_three_channels_with_resolver` — the outer
per-pass loop that wraps the round-221 inner three-channel driver
and walks `p ∈ [0, num_passes)` against the FDIS §C.8.3 + Table
C.6 `Passes` prose. New `multi_pass_decode` module reads
`num_passes` off [`per_pass_non_zeros::PerPassNonZerosGrids::num_passes`]
(round-190's container is the authoritative pass-count source),
walks the [`dct_select::DctSelectGrid`] once per pass, invokes
the caller's `qdc_at(p, &vb)` closure once per varblock per pass
(so the closure may read from a per-pass quantised-LF buffer if
the upstream signal evolves between passes), and threads each
`(p, c)` call through
[`per_pass_non_zeros::PerPassNonZerosGrids::decode_block_at_for_pass_channel`].
Return shape is the typed
[`MultiPassThreeChannelOutput = Vec<Vec<ThreeChannelVarblock>>`]
where `out[p][i]` is the `i`-th varblock (raster order) decoded
in pass `p`. The per-pass per-channel `NonZeros(x, y)`
bookkeeping is already isolated by `p` (round-190 invariant), so
the caller does not have to clear state between passes. The
`read_non_zeros(p, channel, predicted)` / `decode_symbol(p,
channel, coeff_ctx)` closures take the pass index as their first
argument so the caller can route each call to the matching
per-pass per-channel histogram without rebinding closures for
each pass. The new `count_decoded_blocks(grid, num_passes)`
helper returns `num_passes × count_varblocks(grid)` for callers
that need to size a downstream coefficient buffer (defensive u64
overflow guard on the multiplication). 14 unit + 12 integration
(`round228_multi_pass_decode`) tests pin: single-pass single-
DCT8×8 parity with the round-221 inner driver; 4×4 DCT8×8 grid
(16 varblocks) preserving raster order in a single pass; two-
pass 2×2 raster-order per-pass walk; per-pass `qdc` closure
invocation count (3 passes × 4 varblocks = 12 calls, not 36);
three-pass per-channel routing isolation with pass-distinct
raw_non_zeros values landing on per-pass writeback cells without
cross-pass leakage; pass error aborts remaining passes; pass-0
inner error aborts before pass-1 starts (pass-1 closure never
called); per-pass predicted invariant
(`PredictedNonZeros(0, 0) = 32` across every pass + channel);
per-pass `qdc[3]` value propagation through the outer loop;
mixed-transform (`DCT16×8 + 2 DCT8×8`) layout consistency across
passes; pass-1 channel routing read from pass-1 histogram;
`count_decoded_blocks` helper; DCT16×16 single-block single-pass
pass-through; integration coverage of pass-index threading
through both `read_non_zeros` and `decode_symbol` closures; mid-
varblock inner-driver error (pass 1, X-channel decode_symbol
failure) propagating through the outer loop. Lib tests 675 →
689 (+14). Pure-control-flow primitive in the same shape as
round-89 `dct_quant_weights`, round-95 `hf_dequant`, round-121
`llf_from_lf`, round-138 `chroma_from_luma`, round-141
`gaborish`, round-144 `epf`, round-147 `afv_idct`, round-159 /
164 `pass_group_hf`, round-177 `non_zeros_grid`, round-183
`per_channel_non_zeros`, round-190 `per_pass_non_zeros`,
round-208 `varblock_walk`, round-214 `block_context_resolver`,
and round-221's inner three-channel driver — no bit reads, no
spec re-derivation, no histogram materialisation. A future
round wiring §C.7.2 entropy histograms (#799 DOCS-GAP) +
per-pass `hfp` selection from the [`hf_pass::HfPass`] array +
per-channel quantised-LF buffers can drop this driver in as the
per-LfGroup multi-pass control-flow layer without re-deriving
any §C.8.3 raster-walk or per-pass loop geometry.

**Round 221 (2026-06-03)** lands the three-channel per-LfGroup
varblock decode driver
`decode_varblocks_three_channels_with_resolver` — the canonical
§C.8.3 prose ordering (outer varblock-raster loop, inner X / Y / B
channel sweep) wired directly above the round-214
[`BlockContextResolver`]. The new driver walks the
[`dct_select::DctSelectGrid`] once, computes the shared `qdc[3]`
triple once per varblock (via the caller's `qdc_at` closure), then
invokes [`BlockContextResolver::resolve`] three times against that
shared `qdc` (channel order 0 = X → 1 = Y → 2 = B) and routes each
`(p, c)` call through
[`per_pass_non_zeros::PerPassNonZerosGrids::decode_block_at_for_pass_channel`].
Return shape is the typed
[`ThreeChannelVarblock = (Varblock, [DecodedHfBlock; 3], [u32; 3])`]
in raster order — per-channel decoded blocks + per-channel raw
`non_zeros` indexed 0 = X / 1 = Y / 2 = B. The per-channel ANS
closures `read_non_zeros(channel, predicted)` /
`decode_symbol(channel, coeff_ctx)` take the channel as their first
argument so the caller routes per-channel histograms inside one
closure pair instead of binding three. Eliminates the three
separate grid walks (and the three independent `qdc[3]`
derivations) that the naive composition of three round-214
single-channel walks would incur. 11 unit + 12 integration
(`round221_three_channel_resolver`) tests pin: single-DCT8×8
yielding 3 per-channel decodes per varblock; 4×4 DCT8×8 grid (16
varblocks) preserving raster order; single DCT16×16 (1 varblock);
qdc closure invoked exactly once per varblock (= 4 calls for 4
varblocks, NOT 12) — shared across all three channels; channel
order strictly X / Y / B at every read_non_zeros + decode_symbol
call site; per-channel non_zeros writeback at `(0, c, 0, 0)` with
distinct per-channel raw counts (10 / 20 / 30); per-pass routing
(pass = 1 isolated from pass = 0); qdc error aborts before any
per-channel reads; X-channel error aborts before Y + B reads;
mixed-transform `DCT16×8 + 2 DCT8×8` placement preserved; custom
[`HfBlockContext`] (qf_threshold = 5) round-trip; DCT16×16
`num_blocks = 4` per-channel non_zeros = 4 → 4 decode_symbol calls
+ `(4 + 3) / 4 = 1` stored. Lib tests 664 → 675 (+11). Pure
control-flow primitive in the same shape as round-89
`dct_quant_weights`, round-95 `hf_dequant`, round-121
`llf_from_lf`, round-138 `chroma_from_luma`, round-141 `gaborish`,
round-144 `epf`, round-147 `afv_idct`, round-159 / 164
`pass_group_hf`, round-177 `non_zeros_grid`, round-183
`per_channel_non_zeros`, round-190 `per_pass_non_zeros`, round-208
`varblock_walk`, and round-214 `block_context_resolver` — no bit
reads, no spec re-derivation, no histogram materialisation.

**Round 214 (2026-06-03)** lands the typed per-LfGroup
`BlockContext()` resolver — the "per-LfGroup `HfBlockContext`
parameter sweep" the round-208 module notes explicitly named as
follow-up. New `block_context_resolver` module exposes the
borrow-based [`BlockContextResolver`] (wraps a
`&lf_global::HfBlockContext` and offers a per-varblock
`resolve(channel, &Varblock, qdc) -> Result<u32>` lookup matching
the §C.8.3 / Listing C.13 `BlockContext()` signature) plus the
convenience driver [`decode_varblocks_with_resolver`] that walks
a [`dct_select::DctSelectGrid`] via the round-208
[`varblock_walk::VarblockWalk`] iterator and threads each
varblock through
[`per_pass_non_zeros::PerPassNonZerosGrids::decode_block_at_for_pass_channel`]
with the resolver-supplied `block_ctx`. The resolver eliminates
the four-argument `(qf_thresholds, lf_thresholds, block_ctx_map,
nb_block_ctx)` boilerplate at every per-varblock callsite — the
LfGlobal §I.2.2 bundle is captured once and the
`order_id_for_transform` mapping for `s` (Table I.1 OrderId) is
applied internally so callers thread only `(channel, Varblock,
qdc)`. 14 unit + 12 integration (`round214_block_context_resolver`)
tests pin: borrow accessor + `nb_block_ctx` pass-through (default
15); default-branch `(c=0, s=0)` → `map[13] = 7`, `(c=1, s=0)` →
`map[0] = 0`, `(c=2, s=0)` → `map[26] = 7`; DCT16×16 → OrderId 2
→ `map[15] = 9`; DCT32×32 → OrderId 3 → `map[16] = 9`; DCT16×8 +
DCT8×16 share OrderId 4 → both `map[17] = 10`; Hornuss → OrderId
1 → `map[14] = 8`; default-branch invariance to `qdc` and
`hf_mul` (empty thresholds collapse those knobs);
custom-branch `qf_threshold` perturbation grows `idx` exactly as
the underlying `block_context` formula does; driver pass-through
on single-DCT8×8 / raster-order 2×2 DCT8×8 / single-DCT16×16
grids; `qdc_at` closure called once per varblock in walk order;
closure-error propagation through the driver. Lib tests 650 →
664 (+14). Pure-control-flow primitive in the same shape as
round-89 `dct_quant_weights`, round-95 `hf_dequant`, round-121
`llf_from_lf`, round-138 `chroma_from_luma`, round-141
`gaborish`, round-144 `epf`, round-147 `afv_idct`, round-159 /
164 `pass_group_hf`, round-177 `non_zeros_grid`, round-183
`per_channel_non_zeros`, round-190 `per_pass_non_zeros`, and
round-208 `varblock_walk` — no bit reads, no spec re-derivation,
no histogram materialisation. A future round wiring §C.7.2
entropy histograms (#799 DOCS-GAP) + per-channel quantised-LF
buffers (so the `qdc_at` closure can read off a per-LfGroup
buffer instead of returning a synthetic `[0; 3]`) can drop this
resolver in as the per-varblock `block_ctx` source without
re-deriving any §C.8.3 lookup geometry.

**Round 208 (2026-06-02)** lands the per-LfGroup varblock-walk
driver — the "varblock-shape grid" the round-177 / 183 / 190
module notes repeatedly deferred to. New `varblock_walk` module
exposes the [`Varblock`] descriptor
(`{x, y, transform, hf_mul}`), the borrow-based
[`VarblockWalk`] raster-order iterator over a
[`dct_select::DctSelectGrid`] that yields one entry per top-left
cell (Continuation cells are skipped; a residual Empty cell errors
cleanly), the [`count_varblocks`] cell-scan helper, and the typed
per-pass per-channel driver
[`decode_varblocks_for_pass_channel`] that walks the grid +
invokes the caller's `block_ctx_for_varblock` closure (a
Listing C.13 `BlockContext()` lookup, encapsulating the
`block_ctx_map` + `qf_thresholds` + `lf_thresholds` ladder the
walker does not own) + threads each varblock's `(p, c, x, y, t,
block_ctx)` through
[`per_pass_non_zeros::PerPassNonZerosGrids::decode_block_at_for_pass_channel`].
Returns the in-raster-order `Vec<(Varblock, DecodedHfBlock,
raw_non_zeros)>` triple so callers that need to write per-block
coefficients into a per-channel buffer get a deterministic layout.

26 new tests (14 unit + 12 integration
`round208_varblock_walk`) pin: single-DCT8×8 walk yielding one
varblock; raster-order 4×4 DCT8×8 grid (16 varblocks); single
DCT16×16 covering 2×2 cells (1 TopLeft + 3 Continuation);
mixed-transform `DCT16×8 + DCT8×8 + DCT8×8` placement order;
mixed-transform `DCT8×16 + DCT8×8 + DCT8×8`; `count_varblocks`
== walk-collect-length on every shape; residual-Empty-cell
defensive error; all-Continuation tolerated as zero varblocks;
`hf_mul` read from the top-left cell; typed driver routes per-pass
per-channel (pass 1 / channel 2 mutation isolated from the other
five (pass, channel) pairs); closure-error propagation; DCT16×16
single-block typed-driver pass-through; multi-varblock distinct
`hf_mul` (4, 8 from mul-1 = 3, 7) reaching the closure unchanged.
Lib tests 636 → 650 (+14). Pure-control-flow primitive in the
same shape as round-89 `dct_quant_weights`, round-95 `hf_dequant`,
round-121 `llf_from_lf`, round-138 `chroma_from_luma`, round-141
`gaborish`, round-144 `epf`, round-147 `afv_idct`, round-159 / 164
`pass_group_hf`, round-177 `non_zeros_grid`, round-183
`per_channel_non_zeros`, and round-190 `per_pass_non_zeros` — no
bit reads, no spec re-derivation, no histogram materialisation.
A future round wiring §C.7.2 entropy histograms (#799 DOCS-GAP)
+ the per-LfGroup `HfBlockContext` parameter sweep + per-channel
`BlockContext()` history threading can drop this walker in as the
per-LfGroup driver layer without re-deriving any §C.5.4 placement
geometry or §C.8.3 raster-walk recurrence.

**Round 202 (2026-06-01)** widens the round-191 / round-195
weighted-predictor diagnostic from a one-sample pin into a
full-row chain by validating the production WP state across the
entire row `y = 3` window of the `noise-64x64-lossless` fixture
(samples 192..=200) against the trace doc's surrounding-sample
context table (`wp-trace-sample-194.md` lines 130-168). New
`tests/r202_wp_row3_chain.rs` (7 tests) captures `wp_pred8` +
`te_*` at each row-3 sample via the existing
`LEAF_PICK_TRACE_WP` hook and:

- Pins the per-sample `wp_pred8` / `stored true_err` delta
  profile across row 3. New finding: the divergence is
  **already large at sample 192** (`Δ pred8 = -50`,
  `Δ stored = -50`), before the round-191-pinned `Δ pred8 = +8`
  at sample 194. Profile (production vs spec):

  ```text
       s    pred8  spec   stored  spec    Δpred8 Δstored
     192    388   +438    -804   -754       -50    -50
     193    917   +896     317   +296       +21    +21
     194    717   +709     437   +437        +8     +0
     195    305   +302    -399   +222        +3   -621
     196    570   +151     178   +119      +419   +59
     197    769   +626    -839   -622      +143  -217
     198   1368  +1171    -568   -669      +197  +101
     199    810   +766     -14   +390       +44  -404
     200    754   +413       0  -1323     +341     —
  ```

  The `+421` jump in `Δ pred8` at sample 196 (and 197..200
  cascades) signals the WP state corruption flips an MA-tree
  leaf-pick decision past sample 195 (production decodes
  `v(195) = 88` vs spec `10` — an off-by-78, not the off-by-1 a
  pure rounding defect would produce).
- Validates the in-row read chain `te_w(s+1) == pred8(s) -
  v_prod(s)*8` across samples 192..=194 (chain-correct in this
  window). Sample 194's production decoded value `v_prod = 35`
  (vs spec 34) is pinned in its own test.
- Validates the cross-row chain `te_n(s+1) == te_ne(s)` and
  `te_nw(s+1) == te_n(s)` across all 8 row-3 boundaries
  (samples 192..=200) — both walk the row-2 stored-`true_err`
  shadow as observed at row-3 read time; chain is
  structurally-consistent throughout.
- Pins sample 192's left-border zeroing (`te_w = te_nw = 0`
  because `x = 0`), guarding the `WpState::at` border policy.
- Pins sample 194's cross-row reads against the trace doc's
  explicit table: `te_n@194 = -456` (matches spec at sample 130
  — known good cell), `te_nw@194 = 716` (spec 737, Δ = -21 —
  the round-191 smoking gun at sample 129), `te_ne@194 = -160`
  (spec -165, Δ = +5 — at sample 131).

This widens the bisect roadmap from one sample to nine and
shows the upstream defect's footprint is asymmetric: sample 130
matches spec exactly while its left-neighbour 129 is off by -21,
ruling out a uniform row-2 shift and pointing at a per-sample
state-evolution glitch.

**Round 191 (2026-05-30)** lands a Weighted-Predictor oracle
test driven by the newly-staged clean-room behavioural trace at
`docs/image/jpegxl/fixtures/noise-64x64-lossless/wp-trace-sample-194.md`
(provenance: `wp-trace-provenance.md`). The trace records the
FDIS-conformant per-listing intermediates a reference decoder
produces at the `(channel 0, x=2, y=3)` divergence point bisected
in rounds 31..126. A new `pub fn modular_fdis::wp_predict_pub`
test wrapper exposes the production weighted-predictor as a pure
function. `tests/r191_wp_trace_oracle.rs` (5 tests) drives it
with the trace's `WpState` / `Neighbours` inputs and confirms our
Annex E.2 Listings E.1 / E.2 / E.3 / E.4 arithmetic **is** spec-
correct: the production `wp_predict` returns the trace's
`subpred = [1248, 747, 420, 559]`, final `prediction = 709`, and
`max_error = 737` exactly when fed the spec-conformant state.
This isolates the still-unfixed sample-194 `wp_pred8 = 717` vs
trace `709` off-by-8 divergence (= +1 in un-shifted pixel space,
matching `r126_first_divergence_scan` dec=35 / exp=34) to
**upstream WP state evolution** (the `set_true_err` /
`set_sub_err` calls fired across samples 0..193), excluding the
predictor itself from suspicion. A companion test pins the
production-vs-trace delta as a roadmap for the next round's
bisect: `Δ te_w = +21`, `Δ te_nw = -21` (symmetric pair → likely
a single upstream defect), `Δ err_sum_0 = 0` (sub-predictor 0
state evolution already correct), `Δ wp_pred8 = +8`. The
`error2weight` cross-check also documents a minor FDIS-literal
(inner Idiv first) vs production (multiplication-first) reading
discrepancy that is a no-op at sample 194 — both readings give
identical `[3, 4, 3, 6]` shifted weights after the Listing E.3
`>> sh` step.

**Round 190 (2026-05-30)** lifts the round-183 per-channel
[`per_channel_non_zeros::PerChannelNonZerosGrids`] into a typed
per-pass container
[`per_pass_non_zeros::PerPassNonZerosGrids`] that owns one
per-channel container per pass index `p ∈ [0, num_passes)`. A
frame's VarDCT path is decoded in `num_passes` ordered passes
(declared in the [`frame_header::FrameHeader`]'s
[`frame_header::Passes`] field); each pass scans every
`PassGroup` once and §C.8.3 specifies that within a pass each
channel of each varblock maintains its own `NonZeros(x, y)`
state. Between passes the per-channel `NonZeros(x, y)`
bookkeeping is reset because the per-pass histogram is selected
by `hfp` from the per-pass `HfPass` array — a different pass
uses a different histogram and the prediction recurrence is
keyed against the current pass's own coefficient counts.

The new module is the routing primitive layered above round
183's per-channel container:

- [`per_pass_non_zeros::PerPassNonZerosGrids::new`] takes a
  `&[&[(u32, u32)]]` slice (one per-channel `(width, height)`
  list per pass) so a callsite that already knows the per-pass
  per-channel shapes can construct the container in one call
  — validated entry-by-entry against
  [`per_channel_non_zeros::PerChannelNonZerosGrids::new`].
- [`per_pass_non_zeros::PerPassNonZerosGrids::new_uniform`]
  builds the unsubsampled-and-uniform-across-passes container
  in one line.
- [`per_pass_non_zeros::PerPassNonZerosGrids::{num_passes,
  pass, pass_mut, predicted, get, set, update_after_block,
  update_after_block_for_transform}`] route to the matching
  per-pass container; out-of-range `p` errors cleanly.
- [`per_pass_non_zeros::PerPassNonZerosGrids::decode_block_at_for_pass_channel`]
  — typed per-pass per-channel driver wrapping
  [`per_channel_non_zeros::PerChannelNonZerosGrids::decode_block_at_for_channel`]
  with pass routing. The caller passes `block_ctx` computed via
  [`pass_group_hf::block_context`] with the matching `c`; the
  container is a pure storage + routing primitive and does not
  re-derive [`pass_group_hf::block_context`] nor materialise the
  per-pass histogram.
- Per-pass per-channel shapes are independent — ragged per-pass
  channel counts are tolerated (e.g. a DC-only preview pass
  with one channel followed by a three-channel main pass) so
  the container does not encode a semantic choice that belongs
  to the per-LfGroup driver.

41 new tests (28 unit + 13 integration
`round190_per_pass_non_zeros`) pin: empty-pass-list /
zero-channel-pass / zero-dim rejection; two-pass
chroma-subsampled construction at
`[(16, 16), (8, 8), (8, 8)]` shapes; `new_uniform`
convenience; out-of-range pass index errors on every accessor
(`pass`, `pass_mut`, `predicted`, `get`, `set`,
`update_after_block`, `update_after_block_for_transform`,
`decode_block_at_for_pass_channel`);
`PredictedNonZeros(0, 0) = 32` on every (pass, channel) pair;
per-pass write isolation (`set(0, 1, 0, 0, 42)` does not leak
into pass 1 or channel 0/2 of pass 0); per-pass `predicted`
propagation (pass 1's `predicted(1, 1, 0)` reads back pass 1's
own `(0, 0)`, not pass 0's); per-pass
`update_after_block_for_transform` dispatch (`DCT8×8 /
DCT16×16 / DCT32×32` reduces raw `non_zeros = 17` to
`{17, 5, 2}` on three independent passes);
`decode_block_at_for_pass_channel` routes the round-183 typed
driver per pass and per channel; a two-pass three-channel
raster walk at `(0, 0)` / `(1, 0)` with distinct per-pass
per-channel `raw_non_zeros` sequences `[4, 8, 12]` /
`[3, 6, 9]` preserves cross-pass isolation; ragged per-pass
channel counts; `u32::MAX` no-panic saturating-add chain
through the per-pass route. Lib tests 608 → 636 (+28).

Pure-control-flow primitive in the same shape as round-89
`dct_quant_weights`, round-95 `hf_dequant`, round-121
`llf_from_lf`, round-138 `chroma_from_luma`, round-141
`gaborish`, round-144 `epf`, round-147 `afv_idct`, round-159 /
164 `pass_group_hf`, round-177 `non_zeros_grid`, and round-183
`per_channel_non_zeros` — no bit reads, no spec
re-derivation. A future round wiring the §C.7.2 entropy
histograms (#799 DOCS-GAP) + the per-LfGroup varblock-shape
grid + the per-pass `hfp` selection from the
[`hf_pass::HfPass`] array can drop these helpers in as the
per-pass routing step without re-deriving any Listing C.13 /
C.14 formulae.

**Round 183 (2026-05-29)** lifts the round-177 single-channel
[`non_zeros_grid::NonZerosGrid`] into a typed per-channel container
[`per_channel_non_zeros::PerChannelNonZerosGrids`] that owns one
grid per channel (YCbCr / XYB: Y / Y', Cb / X, Cr / B). Listing
C.13's `BlockContext()` factors the channel index `c` into
`(c < 2 ? c ^ 1 : 2) × 13 + s`, and the `NonZeros(x, y)`
bookkeeping is keyed per-channel because chroma subsampling +
`TransformType` heterogeneity means each channel's varblock-grid
shape can differ. The new module is the routing primitive layered
above round 177's per-position storage primitive:

- [`per_channel_non_zeros::PerChannelNonZerosGrids::new`] takes a
  `&[(width, height)]` slice (one pair per channel) so the caller
  can construct an asymmetric Y / Cb / Cr container in one call —
  validated entry-by-entry against [`NonZerosGrid::new`] (zero or
  `> 65535` dims rejected).
- [`per_channel_non_zeros::PerChannelNonZerosGrids::new_uniform`]
  builds the unsubsampled 4:4:4-style container in one line.
- [`per_channel_non_zeros::PerChannelNonZerosGrids::{grid, grid_mut,
  predicted, get, set, update_after_block,
  update_after_block_for_transform}`] route to the matching
  per-channel grid; out-of-range `c` errors cleanly.
- [`per_channel_non_zeros::PerChannelNonZerosGrids::decode_block_at_for_channel`]
  — typed per-channel driver wrapping
  [`non_zeros_grid::decode_block_at`] with channel routing. The
  caller passes `block_ctx` computed via
  [`pass_group_hf::block_context`] with the matching `c`; the
  container is a pure storage + routing primitive and does not
  re-derive [`pass_group_hf::block_context`].
- [`per_channel_non_zeros::DEFAULT_NUM_CHANNELS`] = 3 — the
  canonical YCbCr / XYB channel count.

36 new tests (24 unit + 12 integration
`round183_per_channel_non_zeros`) pin: empty-channel-list / zero-
dim / oversize-dim rejection; three-channel construction at
chroma-subsampled `[(16, 16), (8, 8), (8, 8)]` shapes;
[`PerChannelNonZerosGrids::new_uniform`] convenience builder;
out-of-range channel index errors on every accessor (`grid`,
`grid_mut`, `predicted`, `get`, `set`, `update_after_block`,
`update_after_block_for_transform`, `decode_block_at_for_channel`);
`PredictedNonZeros(0, 0) = 32` on every channel; per-channel
write isolation (`set(0, 1, 1, 99)` does not leak into channel
1 or 2); per-channel `predicted` horizontal chain on a seeded
channel-1 grid with channel 0 / channel 2 still at default;
`update_after_block_for_transform` dispatch reduces a raw
`non_zeros = 17` to `{17, 5, 2}` at DCT8×8 / DCT16×16 / DCT32×32
on three independent channels; `decode_block_at_for_channel`
routes the round-177 typed driver per channel (channel 2's
`raw_non_zeros = 11` updates only channel 2's `(0, 0)` cell);
the typed driver's post-update cell feeds the next-position
predicted value back; OOB `(x, y)` past the per-channel grid
errors cleanly; a full two-step three-channel raster walk at
`(0, 0)` and `(1, 0)` with distinct `[4, 12, 20]` /
`[6, 18, 30]` per-channel raw_non_zeros sequences preserves
cross-channel isolation. Lib tests 584 → 608 (+24).

Pure-control-flow primitive in the same shape as round-89
`dct_quant_weights`, round-95 `hf_dequant`, round-121
`llf_from_lf`, round-138 `chroma_from_luma`, round-141
`gaborish`, round-144 `epf`, round-147 `afv_idct`, round-159 /
164 `pass_group_hf`, and round-177 `non_zeros_grid` — no bit
reads, no spec re-derivation. A future round wiring the §C.7.2
entropy histograms (#799 DOCS-GAP) + the per-LfGroup
varblock-shape grid + per-channel `BlockContext()` history can
drop these helpers in as the per-channel step without
re-deriving any Listing C.13 / C.14 formulae.

**Round 177 (2026-05-29)** lands the typed per-pass / per-channel
`NonZeros(x, y)` grid bookkeeping that bridges round 159's
[`pass_group_hf::predicted_non_zeros`] (the four-branch
`PredictedNonZeros(x, y)` recurrence in FDIS Listing C.13 prelude)
with round 164's
[`pass_group_hf::read_non_zeros_and_decode_block_for_transform`]
(the [`TransformType`]-driven per-block coefficient loop) — the
storage layer the FDIS §C.8.3 prose after Listing C.14 describes:

> NonZeros(x, y) is then `(non_zeros + num_blocks − 1) Idiv num_blocks`.

New `non_zeros_grid` module:

- [`non_zeros_grid::NonZerosGrid`] — rectangular `width × height`
  varblock-grid storage of `NonZeros(x, y)` cells (one cell per
  varblock origin, indexed in 8-sample units). `new`, `get`, `set`,
  `width`, `height`, `cells` accessors + the two spec-driven
  primitives:
  - `predicted(x, y) -> u32` — delegates to
    [`pass_group_hf::predicted_non_zeros`] against
    `|xx, yy| self.get(xx, yy).unwrap_or(0)` so the four-branch
    recurrence (`(0,0) → 32`, top-row → left-neighbour, left-col →
    above-neighbour, interior → `(above + left + 1) >> 1`) is the
    single source of truth.
  - `update_after_block(x, y, non_zeros, num_blocks) -> u32` —
    Listing C.14 post-prose formula `(non_zeros + num_blocks − 1)
    Idiv num_blocks` (ceiling-divide identity, defensively
    `saturating_add` to avoid panic at `u32::MAX`).
  - `update_after_block_for_transform(x, y, non_zeros, t)` —
    `num_blocks` derived from
    [`pass_group_hf::transform_block_params`].
- [`non_zeros_grid::decode_block_at`] — typed per-varblock driver
  that threads
  [`pass_group_hf::read_non_zeros_and_decode_block_for_transform`]
  through the grid: computes `predicted = grid.predicted(x, y)`,
  invokes the round-164 read-then-decode entry point with the
  caller's two ANS closures, then calls
  `grid.update_after_block_for_transform(x, y, raw_non_zeros, t)`
  before returning the
  [`pass_group_hf::DecodedHfBlock`] + raw `non_zeros` pair.

35 new tests (23 unit + 12 integration
`round177_non_zeros_grid`) pin: defensive rejection of zero /
oversize (`> 65535`) dimensions and out-of-range `(x, y)`;
zero-init cell semantics; `PredictedNonZeros(0, 0) = 32` across
a sweep of grid shapes (1×1 through 32×32); the y == 0 / x == 0
border-recurrence branches via horizontal / vertical raster
chains; the interior `(above + left + 1) >> 1` average (odd-sum
rounding); the `predicted_non_zeros` helper agreement
byte-for-byte across an arbitrary seeded 3×3 grid; the
post-Listing-C.14 ceiling-divide at `num_blocks ∈ {1, 4, 16}`
(DCT8×8 / DCT16×16 / DCT32×32); the [`TransformType`] dispatch
via `update_after_block_for_transform` reduces a raw
`non_zeros = 17` to `{17, 5, 2}` at the three shapes; the
typed driver's `predicted = 32` at the origin routes through
the `predicted >= 8` `NonZerosContext` branch (`ctx = block_ctx
+ nb_block_ctx × (4 + 32 Idiv 2)`); `decode_block_at` reads
back `(0, 0)`'s post-update cell when invoked at `(1, 0)`; OOB
positions error cleanly; per-channel independence (two grids of
the same shape evolve independently); and pathological
`u32::MAX` does not panic. Lib tests 561 → 584 (+23).

Pure-control-flow primitive in the same shape as round-89
`dct_quant_weights`, round-95 `hf_dequant`, round-121
`llf_from_lf`, round-138 `chroma_from_luma`, round-141
`gaborish`, round-144 `epf`, round-147 `afv_idct`, and
round-159 / 164 `pass_group_hf` — no bit reads, no spec
re-derivation. A future round wiring the §C.7.2 entropy
histograms (#799 DOCS-GAP) + the per-LfGroup varblock-shape
grid + per-channel `BlockContext()` history can drop these
helpers in as the per-varblock-position step without
re-deriving any Listing C.13 / C.14 formulae.

**Round 164 (2026-05-27)** lifts the round-159 raw-`(num_blocks,
size, natural_order)` per-block coefficient loop into a typed,
[`TransformType`]-driven entry point and pins it at DCT16×16 /
DCT16×8 / DCT32×32 dimensions end-to-end. New public API in
`pass_group_hf`:

- `transform_block_params(t: TransformType) -> (num_blocks, size)`
  — §I.2.4 opening paragraph + Listing C.14 derivation:
  `num_blocks = (bwidth / 8) × (bheight / 8)` (the LLF prefix cell
  count of the varblock) and `size = bwidth × bheight`. Every
  Table C.16 transform satisfies `num_blocks * 64 == size`
  (verified across all 27 entries).
- `decode_block_coefficients_for_transform(t, initial_non_zeros,
  block_ctx, nb_block_ctx, decode_symbol)` — typed wrapper that
  derives `(num_blocks, size, natural_order)` from `t` (via
  `coeff_order::order_id_for_transform` +
  `coeff_order::natural_coeff_order`) and reduces to the round-159
  `decode_block_coefficients` after defensive validation
  (`initial_non_zeros > size - num_blocks` is rejected; for
  DCT16×16 that's > 252).
- `read_non_zeros_and_decode_block_for_transform(t, predicted,
  block_ctx, nb_block_ctx, read_non_zeros, decode_symbol)` — the
  analogous typed wrapper around the round-159
  `read_non_zeros_and_decode_block` convenience entry point.

20 new tests (8 unit + 12 integration
`round164_dct16x16_block_coefficient_loop`) cover: every Table
C.16 transform's `(num_blocks, size)` invariant; DCT16×16's
`prev`-context threshold at `non_zeros == 17` (= size/16 + 1);
typed-vs-raw byte-for-byte equivalence at DCT8×8 and DCT16×16 (on
a mixed `[2, 0, 4, 0, 0, 6]` symbol sequence — 6 reads, 3
non-zeros decrement non_zeros to 0); DCT16×16 all-zero / first-
non-zero / three-consecutive-non-zeros / full-density (252 reads)
shapes with coefficients landing at
`natural_coeff_order(Id2)[4..]`; the rectangular DCT16×8 / DCT8×16
pair collapsing to the same per-block outcome (they share
`OrderId::Id4`); and one DCT32×32 smoke-test at `(num_blocks=16,
size=1024)`. Lib tests 553 → 561 (+8). Pure-typed wrapper layer:
no new bit reads, no spec re-derivation — the round-159 module
note that the primitive is "shape-agnostic and ready for the
larger variable-block sizes once their parameterisation lands" is
now exercised from the caller-facing API.

**Round 159 (2026-05-27)** lands the §C.8.3 per-block HF coefficient
decode loop scaffolding (FDIS Listing C.13 + Listing C.14 +
surrounding prose). New public API in `pass_group_hf`:

- `prev_for_context(k, num_blocks, size, non_zeros, prev_nonzero)` —
  Listing C.14 verbatim. At `k == num_blocks` returns `1` iff
  `non_zeros > size / 16` else `0`; for `k > num_blocks` returns `1`
  iff the previously-decoded coefficient (queried via the `prev_nonzero`
  closure on `k - 1`) was non-zero.
- `decode_block_coefficients(natural_order, num_blocks, size,
  initial_non_zeros, block_ctx, nb_block_ctx, decode_symbol)` —
  Listing C.14's per-block raster-order loop. Walks `k` from
  `num_blocks` to `size`, computes `CoefficientContext` per Listing
  C.13, calls the caller-supplied `decode_symbol(ctx)` closure to
  read each `ucoeff`, applies `UnpackSigned`, places at
  `natural_order[k]`, decrements `non_zeros` when `ucoeff != 0`,
  and stops as soon as `non_zeros` reaches 0 ("If non_zeros reaches
  0, the decoder stops decoding further coefficients" — §C.8.3
  prose). Returns a `DecodedHfBlock` bundle (`coeffs: Vec<i32>` in
  natural-order **index space** + `remaining_non_zeros` +
  `coeffs_read` symbol count).
- `read_non_zeros_and_decode_block(.., predicted, .., read_non_zeros,
  decode_symbol)` — convenience wrapper that issues the
  `D[NonZerosContext(predicted) + offset]` read via the first
  closure and drives `decode_block_coefficients` with the result.
  Returns `(DecodedHfBlock, non_zeros)` for NonZeros-grid
  bookkeeping per `NonZeros(x, y) = (non_zeros + num_blocks - 1)
  Idiv num_blocks`.

The closure-based symbol-source interface keeps the primitive
independent of the (still un-landed) §C.7.2 histogram array (#799
remains DOCS-GAP-blocked). A future round wiring §C.7.2 into a real
`EntropyStream` + `HybridUintState` closure can drop this primitive
in as the per-block loop body without re-deriving any C.13 / C.14
formulae. 11 new unit tests (`pass_group_hf::tests::*`) + 11
integration tests (`round159_block_coefficient_loop`) cover:
all-zero block (no symbol reads); single non-zero at the first HF
slot (one read, `UnpackSigned(1) = -1` at `natural_order[1]`); three
consecutive non-zeros (loop stops after three reads); full-density
block (63 reads for DCT8×8, LLF cell untouched); the size/16
threshold for `prev` (crossover at `non_zeros == 5` for DCT8×8); the
"previous coefficient is zero / non-zero" flag tracking through the
loop's history; defensive rejection of malformed natural-order
vectors, zero `num_blocks`, and over-large `initial_non_zeros`;
end-to-end smoke through `read_non_zeros_and_decode_block`. Lib
tests 538 → 553 (+15); the existing `DCT8×8 alone` bounded scope
(`num_blocks = 1`, `size = 64`, `OrderId::Id0`) is the simplest
shape that exercises the full state machine — the primitive itself
is shape-agnostic and ready for the larger variable-block sizes
once their parameterisation lands. The §C.7.2 entropy histogram
decode, the per-channel (Y / X / B) `non_zeros` read in the
varblock driver above this primitive, the per-pass NonZeros-grid
update, and the per-varblock `BlockContext()` derivation remain
follow-up work for subsequent rounds.

**Round 150 (2026-05-26)** wires the round-147 `AFV_IDCT` primitive
into `idct::idct_for_transform` via §I.2.3.8 / Listing I.13 (Inverse
AFV transform). New `idct::idct_afv(coefficients: &[f32], t:
TransformType) -> Result<Vec<f32>>` composes one `afv_idct` call (the
AFV 4×4 sub-block, Listing I.6) with two `idct_2d` calls (one at
4×4, one at 4×8) and the per-variant `flip_x = n & 1` / `flip_y = n
>> 1` axes — yielding the full 8×8 sample buffer for
`TransformType::Afv0..Afv3`. The dispatcher now routes all four AFV
variants to `idct_afv` instead of returning `Err(Unsupported)`,
finishing the non-DCT IDCT family (Hornuss / DCT2×2 / DCT4×4 /
DCT8×4 / DCT4×8 + AFV0..AFV3 are all pure-math-complete). Seven new
property-style tests cover length / non-AFV rejection, all-zero
pass-through, DC-only → constant 1.0 (the three DC patches
`(c00+c01+c10)×4`, `c00-c01+c10`, `c00-c01` reduce to `4`, `1`, `1`
and each sub-block IDCT maps a DC-only cell to a constant
sub-block), dense-AC coverage, the AFV0↔AFV1 x-axis flip, the
AFV0↔AFV2 y-axis flip, and full-block linearity. Lib tests 531 →
538. One additional FDIS typo documented in the module doc:
Listing I.13's final source line reads `samples_4×4(ix, iy)` but
`ix` iterates `0..8` while `samples_4×4` only has columns `0..3`,
and the immediately preceding line computes `samples_4×8 =
IDCT_2D(coeffs_4×8)`; implementation reads from `samples_4×8` per
context. The full VarDCT IDCT dispatch (`idct_for_transform`) is
now usable for every Table I.4 transform.

**Round 147 (2026-05-26)** lands the Annex I.2.2 AFV basis +
`AFV_IDCT` pure-math primitive (FDIS Listings I.5 + I.6, page 76).
New `afv` module exposes the orthonormal `AFV_BASIS: [[f32; 16];
16]` table (Listing I.5, 256 floats transcribed verbatim from the
FDIS PDF), the §I.2.2 cell length constant `AFV_CELL_LEN = 16`
(the 4×4-as-flat-16 mapping `index = 4×y + x`), and
`afv_idct(coefficients: &[f32]) -> Result<[f32; 16]>` (Listing I.6
`samples[i] = sum_j coefficients[j] × AFVBasis[j][i]`). 10 new
unit tests + 9 integration tests (`round147_afv_idct`)
independently verify the transcription at the table level: row 0
is identically 0.25 in every column (Listing I.5 line 1), row 4
has only two non-zero entries at columns 1 and 4 both at
`±1/sqrt(2)` with zero elsewhere (Listing I.5 line 5), every row
has unit L2 norm (orthonormality diagonal), every distinct pair
of rows has zero inner product (orthonormality off-diagonal),
`afv_idct` is linear and L2-energy-conserving, and one-hot
coefficient input recovers `AFV_BASIS[j]` row-for-row — so a
single transcription typo in any of the 256 entries would fail at
least one orthonormality sum. Lib tests 521 → 531. Pure-math
primitive in the same shape as round-89 `dct_quant_weights`,
round-95 `hf_dequant`, round-121 `llf_from_lf`, round-138
`chroma_from_luma`, round-141 `gaborish`, and round-144 `epf` — a
future round wiring §I.2.3.8 Listing I.13 (Inverse AFV transform)
into `idct_for_transform` can drop this helper in without
re-deriving any I.5 or I.6 cells. The Listing I.13 composition
(the `coeffs_afv` corner-load, the two `IDCT_2D` 4×4 / 4×8
sub-blocks, the `flip_x` / `flip_y` AFVn flip) remains follow-up
work; `idct_for_transform(Afv0..Afv3, ..)` continues to return
`Err(Unsupported)` until that wiring lands.

**Round 144 (2026-05-26)** lands the Annex J.3 "Edge-preserving
filter" pure-math primitive (pages 85–87). New `epf` module
exposes Listing J.1's `distance_step_0_and_1` (the five-pixel
cross-shape three-channel L1 distance with per-channel
`epf_channel_scale` weighting) and `distance_step_2` (the
single-sample three-channel L1 distance for step 2 — under the
literal §J.3.2 reading where `(ix, iy) == (0, 0)`); Listing
J.2's `weight(distance, inv_sigma, position_multiplier,
zeroflush)` decreasing-function-of-distance kernel with the
`v <= zeroflush` cutoff; Listing J.3's
`vardct_sigma_from_listing_j3(quantization_width, sharpness,
&rf)` per-varblock derivation with the `max(1e-4, sigma)`
clamp; `inv_sigma_for_pass(step_multiplier, sigma)` for the
Listing J.2 `step_multiplier × 4 × (sqrt(0.5) - 1) / sigma`
factor; `is_border_position(x, y)` for the "either coord is 0
or 7 IMod 8" `epf_border_sad_mul` predicate; Listing J.4's
`apply_step_5tap(pass, ..)` (passes 1 and 2 with the 5-tap
cross kernel) and `apply_step_13tap` (pass 0 with the 13-tap
diamond kernel covering all neighbours with `|cx|+|cy| <= 2`).
All three passes consume three-channel f32 planes and write
into three-channel f32 outputs; §6.5 Mirror1D boundary handling
is reused verbatim from round-141 `gaborish::mirror1d`. 36 new
unit tests + 12 integration tests (`round144_epf`) pin
self-distance-is-zero on constant planes, per-channel scale
linearity, offset symmetry, `DistanceStep2` hand-derived
spatially-varying-plane case (`x:1×40 + y:2×5 + b:0×3.5 =
50`), `Weight()` zero-distance returns 1.0 / zeroflush cutoff /
position-multiplier scaling, Listing J.3 sigma at default `rf`
sharpness 0 → 1e-4 clamp and sharpness 7 → full quant, the
`is_border_position` 8×8 grid layout, constant-plane invariance
across all three passes, and the zero-channel-scale collapse to
the uniform mean on a centre impulse. Lib tests 485 → 521.
Pure-math primitive in the same shape as round-89
`dct_quant_weights`, round-95 `hf_dequant`, round-121
`llf_from_lf`, round-138 `chroma_from_luma`, and round-141
`gaborish` — a future round wiring §J.3 into the per-frame
restoration-filter pipeline can drop these helpers in without
re-deriving any of the J.1/J.2/J.3/J.4 listings. Per-frame
loop (calling each pass for each varblock under the right
`epf_iters` / per-block sigma / position-multiplier conditions
with output of pass `i` feeding pass `i+1`), the `sigma < 0.3`
skip-the-block path, and the `epf_iters > 0` skip remain caller
responsibilities (deferred to follow-up rounds). DOCS-GAP
observed in Listing J.1 (`DistanceStep2` free `ix`/`iy`
variables) and Listing J.2 (`step_multiplier` array missing
comma); both surfaced in the module-level rustdoc with the
adopted reading and rationale.

**Round 141 (2026-05-26)** lands the Annex J.2 "Gabor-like
transform" pure-math primitive (page 85). New `gaborish` module
exposes the §6.5 `Mirror1D` boundary helper, `gab_kernel(w1, w2)
-> [f32; 9]` (the normalised 3×3 symmetric kernel `(centre = 1,
edges = w1, corners = w2)` rescaled so its nine entries sum to 1),
`apply_channel` / `apply_channel_in_place` (per-channel
convolution with an interior fast path plus mirror-extended edge
fallback), and the three-channel
`apply_xyb_planes_in_place(x, y, b, w, h, &RestorationFilter)`
convenience that dispatches the per-channel `gab_x_*` /
`gab_y_*` / `gab_b_*` weight pair. 23 new unit tests + 10
integration tests (`round141_gaborish`) pin Mirror1D's identity,
first-reflection, and single-row collapse cases, the default-
weight kernel sum-to-one (`≈ 1.7057 → 1.0`) and centre tap
(`≈ 0.586`), kernel symmetry, identity-kernel pass-through,
constant-plane invariance, the impulse response on a 3×3 plane,
linearity of the convolution operator, and the per-channel
dispatch through `apply_xyb_planes_in_place`. Lib tests 462 →
485. Pure-math primitive in the same shape as round-89
`dct_quant_weights`, round-95 `hf_dequant`, round-121
`llf_from_lf`, and round-138 `chroma_from_luma`. §J.3
edge-preserving filter and the `rf.gab` boolean skip remain
caller responsibilities (deferred to follow-up rounds).

**Round 138 (2026-05-26)** lands the Annex G "Chroma from luma"
pure-math primitive (Listing G.1). New `chroma_from_luma` module
exposes `kx_kb_lf` / `kx_kb_hf` (per-tile `(kX, kB)` from a
parsed `LfChannelCorrelation` + optional 64×64-tile factor pair),
`apply_sample` / `apply_lf_sample` / `apply_hf_sample` for the
per-sample reconstruction `Y = dY`, `X = dX + kX × Y`,
`B = dB + kB × Y`, and the plane-level
`apply_lf_plane_inplace` (constant per-frame multipliers) +
`apply_hf_plane_inplace(.., w, h, x_from_y, b_from_y, cfl)`
(per-`tile_x=x/64`/`tile_y=y/64` lookup with a per-tile cache).
20 new unit tests + 11 integration tests
(`round138_chroma_from_luma`); lib tests 442 → 462. This is a
pure-math primitive in the same shape as round-89
`dct_quant_weights`, round-95 `hf_dequant`, and round-121
`llf_from_lf` — a future round wiring §F.3 + Annex G into the
per-LfGroup VarDCT pipeline can drop these helpers in without
re-deriving any G.1 formulae. Subsampled chroma is out of scope
(Annex G explicitly skips that case).

**Round 129 (2026-05-25)** lands the per-varblock LF→LLF
composition glue: `vardct::extract_lf_subblock`,
`compose_lf_to_llf_block`, and `compose_lf_to_llf_block_3ch`
drive the round-121 [`llf_from_lf::llf_from_lf`] pure-math step
from a single channel's dequantised LF samples for a single
varblock placement (§I.2.5). 24 new tests; lib tests 422 → 437.
This is the geometry glue between rounds 12/13 (per-LfGroup LF
dequant + smoothing) and rounds 91+/95 (HF coefficient ANS
decode + HF dequantisation) — a future round wiring §F.x into
`decode_codestream` can drop these helpers in as the per-varblock
loop body without re-deriving any LF→LLF geometry. The
`noise-64x64-lossless` sample-194 wp_pred8 = 717 vs spec
divergence remains DOCS-GAP-blocked per `project_jpegxl_pixel_
blocked`.

This crate currently ships:

- Round-1..3 baseline (pre-retire): signature + container detection,
  `SizeHeader` + full `ImageMetadata` (FDIS A.6 form), FrameHeader +
  TOC, the Annex C entropy stack (ANS + prefix codes + hybrid uint +
  LZ77 + clustering), LfGlobal + GlobalModular tree-prelude.
- **Rounds 1..5 (2024-spec)**: end-to-end Modular pixel decode for
  single-group, single-pass frames. Multi-leaf MA tree per Annex
  H.4.1 (16 base properties of Table H.4 + per-previous-channel
  properties), Table H.3 predictors 0..13, full H.5 self-correcting
  WP predictor, RCT / Palette / Squeeze inverse transforms (H.6),
  Grey + RGB output at 8 bpp. **Round 29** lifts the fixture count
  from 5 to 6 by adding `alpha-64x64` (4-channel RGBA, extra-channel
  path per FDIS G.1.3 colour-then-extras + ExtraChannelInfo of type
  Alpha) and fixing the ISOBMFF jxlc payload `FF 0A` strip in
  `decode_one_frame`. **Round 30** lifts the fixture count from 6
  to 7 by adding `bit-depth-16` (3-channel RGB lossless Modular at
  `bits_per_sample = 16`) and adopts the LE-pack plane convention
  documented under "Plane byte layout" below. **Round 31** applies
  FDIS §F.3's section zero-pad rule uniformly to the
  single-TOC-entry LfGlobal fast path, so the
  `noise-64x64-lossless` fixture (`cjxl -d 0 -e 7`, high-entropy
  64×64 RGB lossless Modular, MA tree `leaves=84`) now
  decode-completes (vs hard-EOF pre-r31). **Round 32** bisects
  the residual pixel-divergence on that fixture to the
  Self-correcting weighted predictor at the first
  `predictor == 6` sample whose WP path uses `WW` and `NN` both
  as in-image values (i.e. `x >= 2 && y >= 2`); fix deferred
  pending a docs-collaborator WP behavioural trace at the
  divergence point. **Round 126** extends the WP diagnostics
  with `WP_DEEP_TRACE` (20-entry capture of `subpred[0..4]`,
  `err_sum[0..4]`, post-shift weights, `sum_weights_pre/post`,
  `log_weight`, `sh`, `nn8`, `ww8`, `pred_pre_clamp`,
  `clamped_flag`) and pins the full sample-194 intermediates
  (`wp_pred8 = 717`, `subpred = [1248, 734, 420, 563]`, weights
  `= [3, 4, 3, 5]`). The hand-derivation against FDIS Listings
  E.1/E.2/E.3 in `tests/r126_wp_intermediates_at_194.rs`'s
  docstring proves NEITHER the `subpred[3]` sign knob NOR the
  `s_init - 1` knob can recover the spec-correct prediction
  window `[709..716]` from the captured neighbour state; the
  divergence is in either `sub_err` history evolution or a
  `WpHeader` parameter mismatch. The FDIS-literal `sub_err`
  formula (line 6832) was tried and reverted (regresses
  `synth_320` first-drift from `(y=24, x=14)` to
  `(y=11, x=104)`). Seven committed fixtures still decode
  pixel-correct vs `expected.png` (PNG-decoder-backed
  byte-for-byte comparison): `pixel-1x1`, `gray-64x64`,
  `gradient-64x64-lossless`, `palette-32x32`,
  `grey_8x8_lossless`, `alpha-64x64`, **`bit-depth-16`**.
- **Round 89 (2024-spec)** materialises the §I.2.4 / §I.2.5 +
  Table I.6 default dequantization-matrix set. New
  [`dct_quant_weights`] module transcribes the 2024 spec listing
  for `Mult`, `Interpolate`, `GetDCTQuantWeights`, the per-mode
  weights-derivation rules (DCT, DCT4, DCT2, Hornuss, DCT4x8,
  AFV) and the AFV Listing C.11 freqs/bands ladder. Public API
  exposes `materialise_default_dequant_set()` → the full 17-slot
  × 3-channel set (Table I.4 dims, element-wise reciprocal of
  the weights matrix). 26 new tests (15 unit + 11 integration);
  every cell of every channel of every slot is positive-finite
  per the spec's §I.2.4 last-paragraph invariant. Documented
  spec-listing typo notes (FDIS 2021 bands/weights nested-loop
  bug, corrected in 2024 published edition) and a SPECGAP for
  the DCT2 `(0, 0)` cell (not specified by the spec text;
  filled with `params(c, 0)` to keep the dequant reciprocal
  finite). Unblocks downstream HF coefficient dequantisation
  (§F.3) on the `u(1) == 1` HfGlobal default-encoding fast path.
- **Round 90 (2021-FDIS / 2024-spec) — HfPass + PassGroup HF
  structural parsers.** Three new modules surface the §C.7.1 /
  §C.7.2 HfPass bundle and the §C.8.3 PassGroup HF entry-points:
  * `coeff_order` — §I.2.4 natural coefficient ordering for every
    `OrderId` 0..=12 (Table I.1). Builds `LLF` prefix sorted by
    `y × bwidth + x`, then `HF` tail sorted by `(key1, key2)`
    per Listing I.14. Exposes `natural_coeff_order(OrderId)`,
    `varblock_size_for_order`, `coefficient_count`, and the
    `TransformType → OrderId` table.
  * `hf_pass` — §C.7.1 Listing C.12 parser. The `used_orders ==
    0` fast path materialises all 13 natural orders directly;
    `used_orders != 0` returns `Error::Unsupported` (the
    permutation reads need the shared 8-cluster ANS stream that
    §C.7.2 histograms also feed — round 91 work). Exposes
    `num_histogram_distributions = 495 × num_hf_presets ×
    nb_block_ctx` so the next round knows the §C.7.2 read
    count up-front.
  * `pass_group_hf` — §C.8.3 first line + Listing C.13. Reads
    `hfp = u(ceil(log2(num_hf_presets)))` and computes
    `histogram_offset = 495 × nb_block_ctx × hfp`. Verbatim
    transcriptions of `BlockContext`, `NonZerosContext`,
    `CoefficientContext`, `PredictedNonZeros`, plus the two
    64-element `CoeffFreqContext` /
    `CoeffNumNonzeroContext` ladder tables.

  49 new tests: 12 (`coeff_order`) + 7 (`hf_pass`) + 18
  (`pass_group_hf`) + 12 (integration
  `round34_hf_pass_pass_group_hf`). Unblocks downstream per-
  block coefficient decode loop (the `used_orders == 0` typed
  surface is now usable end-to-end).
- **Round 133 (2021-FDIS / 2024-spec) — §C.7.1 `DecodePermutation()`
  for `used_orders != 0`.** Wires Listing C.12's non-natural
  coefficient-order path. The shared "8 clustered distributions D"
  are read once into a `modular_fdis::EntropyStream` (`num_dist = 8`)
  with its ANS state initialised; each set `used_orders` bit then
  runs the §C.3.2 Lehmer-code permutation against that same stream.
  New `coeff_order::decode_permutation_from_stream(br, entropy,
  hybrid, size, skip)` factors the §C.3.2 procedure (`GetContext`,
  Lehmer read against `D[prev_elem]`, `temp`-shuffle) generically;
  §C.7.1 supplies `size = coefficient_count(order)` and
  `skip = size / 64`. The final order is
  `order[i] = natural_coeff_order[nat_ord_perm[i]]`. `HfPass::read`
  no longer returns `Unsupported` for `used_orders != 0`. 8 new
  tests (`get_context`, four `lehmer_to_permutation` cases, two
  `hf_pass` stream-path cases). Still lacks §C.7.2 histogram decode
  + the per-block coefficient loop + CfL / Gaborish / EPF.
- **Round 121 (2021-FDIS / 2024-spec) — §I.2.5 LLF-from-LF
  pure-math step (Listings I.15 + I.16).** New `src/llf_from_lf.rs`
  lands the bridge from §F.2's dequantised+smoothed LF samples
  into the top-left LLF coefficient block of each HF varblock —
  the step the trailing prose of §F.2 hands off to §I.2.7
  (renumbered §I.2.5 in the 2021 FDIS). Public API: Listing I.15
  closed-form helpers (`scale_i8`, `scale_d8`, `scale_i`,
  `scale_d`, `scale_c`, `scale_f`); §I.2.1 / §I.2.2 forward DCT
  (`dct_1d`, `dct_2d`) — the algorithmic inverse of the round-12
  `idct::idct_2d`; `llf_dims(t) -> (u32, u32)` LF-block dims per
  `TransformType`; `llf_from_lf(input, t)` (Listing I.16
  verbatim, with non-DCT pass-through for Hornuss / DCT2×2 /
  DCT4×4 / DCT4×8 / DCT8×4 / AFV0..3). 44 new tests (28 unit + 16
  integration `round121_llf_from_lf`) pin the closed-form scale
  identities (`ScaleF(1, 8, 0) = 1.0` exactly for DCT8×8 corner),
  the §I.2.1 1-D DCT formula at the unit-impulse, byte-exact LLF
  blocks for DCT16×16 (`out[y·2+x] = 0.25 · SF(2,16,y) ·
  SF(2,16,x)` from a [1,0,0,0] LF impulse), rectangular
  DCT16×8 / DCT8×16 paths, and a `dct_2d ↔ idct_2d` 4×4
  round-trip to f32 epsilon. Per-LfGroup wiring that drives the
  per-varblock invocation from the `pass_group_hf` coefficient
  buffer is still ahead of this step; round 121 lands the
  bit-exact arithmetic so a future round can wire it in without
  re-deriving any I.15/I.16 formulae.
- **Round 95 (2021-FDIS / 2024-spec) — §F.3 HF dequantisation
  pure-math step.** New `src/hf_dequant.rs` glues the round-89
  `dct_quant_weights` 17-slot default dequant set to the
  round-90 `hf_pass` / `pass_group_hf` structural parsers via
  the FDIS Listing F.2 bias-adjust + per-block `HfMul`
  multiplier + `0.8^(qm_scale - 2)` per-channel factor.
  Public API: `bias_adjust(quant, channel, oim) -> f32`
  (Listing F.2 verbatim — `*= quant_bias[c]` for `|q| <= 1`
  branch, `-= quant_bias_numerator / quant` otherwise);
  `QmScaleFactors::for_frame(&FrameHeader)` (precompute the
  per-frame X / B factors once, Y is implicitly 1.0);
  `dequant_hf_coefficient(quant, channel, hf_mul,
  dequant_matrix_entry, oim, qm) -> f32` (full FDIS p. 72
  pipeline: bias-adjust → × `HfMul` → × qm-factor → × matrix
  entry); `dequant_hf_pre_matrix(...)` (partial product
  without the matrix entry, for callers that want to apply
  the dequant-matrix multiplication in a vectorised pass).
  23 new tests (13 unit + 10 integration
  `round35_hf_dequant`); cross-module composition pins the
  pipeline against `materialise_default_dequant_set()` for X
  and Y channels at the DCT8×8 corner cell, the FDIS default
  `quant_bias_numerator = 0.145` is fixed-point pinned at
  `quant = 2 → 1.9275`, and the `0.8^(scale - 2)` formula is
  swept over all 8 legal `u(3)` values for positive-finite
  output. The per-block ANS coefficient decode + indexing
  glue is still ahead of this step; round 95 lands the
  bit-exact arithmetic so a future round can drop the integer
  ANS reader on top without re-deriving any F.3 formulae.
- **Round 77 (2024-spec)** lands an audit-grade SPECDIFF harness
  for `docs/image/jpegxl/fixtures/animation-3frame/input.jxl` (3
  Regular Modular frames, `have_animation = 1`, encoded by cjxl
  0.12.0 against the 2024 final core spec). The probe-level path
  is correct (`probe_fdis` recovers SizeHeader + ImageMetadata
  with `have_animation = true` + AnimationHeader). The decode
  path remains blocked on a 2-bit format split between ISO/IEC
  18181-1:2021 FDIS Table C.9 (no leading `all_default` in
  RestorationFilter — what our `RestorationFilter::read`
  follows) and ISO/IEC 18181-1:2024 final Table J.1 (which
  prepends `all_default Bool()` + adds a `u(32) (ignored)`
  field). The seven small lossless fixtures were encoded by cjxl
  0.11.1 against the 2021 layout, so a uniform 2024-spec patch
  would break them; the audit doc-side recommendation is to
  re-encode those fixtures with cjxl 0.12.0+ before flipping.
  See `tests/r77_animation_3frame_specdiff.rs` module docs for
  the byte-level bit-trace bisect that pins the discrepancy
  down to a single byte boundary on the codestream's
  FrameHeader → TOC junction.
- **Round 7 (2024-spec)**: four-piece refactor wiring multi-group
  decode infrastructure (Annex G.1.3 last paragraph + G.4.2):
  `GlobalModular::read` honours the "stops decoding at channels
  exceeding `group_dim`" rule; new
  `decode_channels_at_stream(br, descs, tree, wp, stream_index)`
  threads Table H.4 property[1]; `pass_group::decode_modular_group_into`
  decodes per-PassGroup modular sub-bitstreams and copies samples back
  into the parent image; post-PassGroup inverse transforms run AFTER
  all groups complete (driven by `decode_codestream`). The committed
  `synth_320_grey/` multi-group fixture (320×320 grey lossless,
  `cjxl 0.11.1 -d 0 -m 1 -e 1 -g 0 -R 0` → 9 groups) is left
  unconsumed by tests pending a SPECGAP clarification: cjxl emits
  per-cluster ANS distributions with `alphabet_size > table_size`
  (33 > 32 at log_alpha=5), which the 2024 spec text in C.2.5 implies
  should be rejected. Round-8 lands the SPECGAP fix.

- **Round 6 (2024-spec)**: Annex E.4 ICC profile decode +
  LfGroup / PassGroup type scaffolding.
  * `src/icc.rs` — full Annex E.4 ICC decoder. Reads `enc_size =
    U64()`, decodes 41 pre-clustered distributions + `enc_size`
    bytes via `DecodeHybridVarLenUint` and the
    `IccContext(i, prev_byte, prev_prev_byte)` 41-context function;
    walks the resulting encoded stream through E.4.3 (header with
    predicted-byte ladder) + E.4.4 (tag list) + E.4.5 (main content
    + Nth-order predictor at orders 0/1/2). When `want_icc=true` is
    set in `ColourEncoding` the decoder no longer fails outright —
    the bit reader is correctly advanced past the ICC stream and a
    minimal "acsp" magic check at offset 36 validates the result.
    The decoded ICC bytes are not yet propagated to `VideoFrame`
    (`oxideav_core::VideoFrame` has no ICC slot in 0.1.x).
  * `src/lf_group.rs` (G.2) and `src/pass_group.rs` (G.4) — typed
    bundles + per-group-rectangle geometry + per-pass `(minshift,
    maxshift)` recurrence. Per-LfGroup / per-PassGroup decode
    itself is round-7 work, gated on a coordinated four-piece
    refactor (GlobalModular `nb_meta_channels`-aware partial
    decode + `stream_index` threading + TOC permutation awareness
    + post-PassGroup inverse-transform application). Multi-LfGroup
    / multi-group / multi-pass / VarDCT frames now fail with
    precise round-7-targeting error messages.

Black-box validation against `cjxl` / `djxl` is available as
opaque-binary tests (the binaries are treated as opaque processes
— we never read libjxl source).

## Why retired (history)

`OxideAV/docs` retired `image/jpegxl/libjxl-trace-reverse-engineering.md`
(the 792-line behavioural-trace writeup that previously drove rounds
7-11) on 2026-05-06 (commit `d732002`) under fruits-of-poisonous-tree:
even when no libjxl source is literally quoted, an agent that read
libjxl source while authoring the writeup carries structural narrative
across. Decoder rounds 7-11 + encoder rounds 1-6 were authored within
that session window and have been reset off master with the trace doc.
See `CHANGELOG.md [Unreleased]` for the full retired-commits list. The
pre-retirement history is preserved on the `old` branch.

## Forward path

Decoder rounds resumed 2026-05-08 against the published 2024 core
spec PDF + the 18181-3 conformance corpus + the small lossless
fixtures already committed under `docs/image/jpegxl/fixtures/`. The
contaminated trace docs (`libjxl-trace-reverse-engineering.md`,
`jpegxl-fixtures-and-traces.md`, `round9_python_redecoder.py`) and
the `old` branch are universally off-limits per
`feedback_no_external_libs.md` workspace policy.

Round 8+ candidates (in priority order):

1. **Resolve C.2.5 alphabet_size SPECGAP** — round-8 partial
   resolution; round-9 lifted the synth_320 fixture to ~21k
   pixel-correct samples; remaining drift parked for an
   Auditor-mode bisect (see round-10 CHANGELOG).
2. **VarDCT decode** (Annex I) — round-8 lands the IDCT-8x8
   primitive + structural recognition; **round-11** wires the
   LfGlobal VarDCT bundles (Quantizer + HfBlockContext default
   table + LfChannelCorrelation) and the LfCoefficients
   sub-bitstream (per-LfGroup `extra_precision` + 3-channel
   modular decode at `ceil(group_dim/8)` resolution); **round-12**
   lands the spec-conformant 1-D + 2-D IDCT dispatch
   (`idct::idct_for_transform`) covering the 18 plain-DCT block
   sizes from Table C.16 (DCT8x8 through DCT256x256) per FDIS
   I.2.1 + I.2.2 Listing I.4; **round-13** extends the dispatch to
   the non-DCT IDCT helpers per Listings I.9.3..I.9.7 — `Hornuss`,
   `DCT2×2`, `DCT4×4`, `DCT8×4`, `DCT4×8` — via new public
   functions `aux_idct_2x2`, `idct_dct2x2`, `idct_dct4x4`,
   `idct_hornuss`, `idct_dct8x4`, `idct_dct4x8`. The four AFVn
   variants (Listing I.9.8) continue to return `Err(Unsupported)`
   pending an independently verified 256-entry `AFVBasis` table.
   Round 14+: PassGroup HF coefficient ANS decode + F.3
   dequantisation + AFV completion + Chroma-from-Luma + Gaborish +
   EPF.
3. **XYB inverse colour transform** (§L.2) — **landed round 11**.
   `xyb::inverse_xyb_to_rgb` and `xyb::inverse_ycbcr_to_rgb`
   transcribe FDIS Annex L.2.2 + L.3 verbatim; the modular output
   stage in `decode_codestream` now branches on
   `metadata.xyb_encoded` / `frame_header.do_ycbcr` and applies the
   inverse colour transform before mapping to `VideoFrame`. 9 unit
   tests + 6 integration tests including a forward-→-inverse
   round-trip oracle. Output gamma is left linear (downstream
   colour-management's job per §L.2.2 NOTE).
4. **ICC bytes propagation** — coordinate with `oxideav-core` to
   add `VideoFrame::icc_profile`.

Encoder rounds will be re-authored on top of those decoder
milestones; encoder is still retired pending decoder forward
progress.

Zero C dependencies, zero FFI, zero `*-sys` (carried over from the
round-1..3 design).

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-jpegxl = "0.0"
```

## What this crate does today

JPEG XL files come in two wrappings:

- **Raw codestream** — starts with `FF 0A` (little-endian `0x0AFF`).
- **ISOBMFF-wrapped** — starts with the 12-byte signature box
  `00 00 00 0C 4A 58 4C 20 0D 0A 87 0A`, followed by standard MP4-style
  boxes. The codestream lives in a `jxlc` box or is split across `jxlp`
  partial-codestream boxes.

Both are detected; the codestream is extracted transparently before the
codestream preamble is parsed.

The codestream preamble is parsed with an LSB-first bit reader
(`bitreader::BitReader`) that matches the reference behavioural-trace
bit packing, including the JXL `U32` 2-bit-selector encoding. On top of it:

- **`SizeHeader`** — width + height, covering all four encodings the spec
  allows: the 5-bit "small (≤256, multiple of 8)" form, the 2-bit
  selector large form, implicit aspect ratio via the 3-bit `ratio` field
  (the full seven-entry `FIXED_ASPECT_RATIOS` table), and explicit xsize.
- **`ImageMetadata`** — the bundle's `all_default` shortcut, and when
  clear: `extra_fields` with orientation + `have_intrinsic_size` +
  preview/animation presence flags, the `BitDepth` sub-bundle
  (integer 1..=31 and IEEE-float variants with range checking), the
  `modular_16_bit_buffer_sufficient` flag, and `num_extra_channels`.

`ColorEncoding`, `ToneMapping`, `ExtraChannelInfo`, `PreviewHeader`,
`AnimationHeader` and the `FrameHeader` TOC are **not** decoded yet; the
parser stops cleanly before them. Presence of a preview or animation
bundle surfaces as `Error::Unsupported("jxl: preview/animation header
parsing not yet implemented")` rather than silent misparse.

## What this crate does **not** do

- No pixel decoding. Neither the Modular path (Weighted + Gradient
  predictor, MA-tree range coder) nor the VarDCT path (variable-size
  DCT, LF/HF subbands, Chroma-from-Luma, Gaborish, EPF) is implemented.
  `registry.make_decoder(&params)` returns
  `Error::Unsupported("jxl decode not yet implemented")`.
- No encoder. Not registered; `make_encoder` rejects any call.
- No animation, preview, or intrinsic-size sub-bundle decoding (parsing
  stops at the `have_*` flags).

### Why pixel decode is blocked

Pixel-decoder work is gated on having the normative ISO/IEC 18181-1
(JPEG XL Core Coding System) text in `docs/image/jxl/`. As of this
release the workspace does not carry the spec — it is listed in the
project-wide `docs/README.md` "Known gaps — ISO/IEC (paid)" section.
Workspace policy forbids consulting third-party source (libjxl,
jxlatte, jxl-rs, FUIF, brunsli) as a substitute. See
[`SPEC_BLOCKED.md`](SPEC_BLOCKED.md) for the audit, the documents
checked, and the unblock procedure + planned work-order for when the
ISO PDF lands.

## Usage

```rust
use oxideav_jpegxl::{probe, Signature};

let bytes = std::fs::read("input.jxl")?;
let headers = probe(&bytes)?;

match headers.signature {
    Signature::RawCodestream => println!("raw .jxl codestream"),
    Signature::Isobmff => println!("ISOBMFF-wrapped .jxl"),
}
println!("{}x{}", headers.size.width, headers.size.height);
println!("{} bits/sample, float={}",
    headers.metadata.bit_depth.bits_per_sample,
    headers.metadata.bit_depth.floating_point);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Registering the codec stub in a registry also works, but
`make_decoder` will reject with `Error::Unsupported`:

```rust
use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters};

let mut reg = CodecRegistry::new();
oxideav_jpegxl::register(&mut reg);

let params = CodecParameters::video(CodecId::new("jpegxl"));
assert!(reg.make_decoder(&params).is_err());
```

### Codec / container IDs

- Codec: `"jpegxl"` — decoder slot registered (returns
  `Error::Unsupported` on instantiation); no encoder slot.
- No demuxer is registered: this crate treats a JXL file as a single
  codestream buffer fed directly to `probe(...)`.

## Plane byte layout

`oxideav_core::VideoPlane` carries `(stride, data)` only — there is no
per-plane bit-depth field in core 0.1.x. The decoder therefore packs
samples into `data: Vec<u8>` according to the codestream's
`metadata.bit_depth.bits_per_sample` (FDIS Annex A.6 + Table A.22):

| `bits_per_sample` (`bps`) | Bytes / sample | Plane stride | Layout                              |
|---------------------------|----------------|--------------|-------------------------------------|
| `1 ..= 8`                 | 1              | `width`      | sample clamped to `[0, 2^bps - 1]`  |
| `9 ..= 16`                | 2              | `width × 2`  | **little-endian** `u16` per sample  |

Round 30 (2026-05) introduced the 16-bit row; the 8-bit row is the
pre-round-30 default. Floating-point samples (`bit_depth.float_sample
== true`) and `bps > 16` are not yet supported and surface as
`Error::Unsupported`.

The XYB (`metadata.xyb_encoded == true`) and YCbCr
(`frame_header.do_ycbcr == true`) inverse-colour-transform paths still
hard-require `bps == 8` because their dequantisation lattice is
calibrated against the 8-bit output range; high-bit-depth XYB / YCbCr
is round-31+.

A downstream consumer that wants to recover native `u16` samples from
a 16-bit plane does:

```rust
let samples: Vec<u16> = plane
    .data
    .chunks_exact(2)
    .map(|c| u16::from_le_bytes([c[0], c[1]]))
    .collect();
```

The convention deliberately mismatches PNG (RFC 2083 §2.1 specifies
big-endian 16-bit samples) so that on a little-endian host
`bytemuck::cast_slice::<u8, u16>(&plane.data)` is a zero-cost view.

## License

MIT — see [LICENSE](LICENSE).
