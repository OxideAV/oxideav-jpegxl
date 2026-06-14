# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Round 306 — per-LfGroup VarDCT **residual-plane assembly**
  (`src/residual_plane.rs`), the spatial-placement layer directly above
  the round-286/293/300 `block_dequant` per-block decode walk. Walks a
  `dct_select::DctSelectGrid` via `varblock_walk::VarblockWalk` and
  writes each varblock's `R × C` row-major residual block (the
  `block_dequant::decode_block_to_residual` output) into a single-channel
  spatial plane at the varblock's pixel origin `(bx · 8, by · 8)`. New
  public API: `ResidualPlane` (row-major `f32` plane sized to the padded
  block grid `width_blocks·8 × height_blocks·8`, `for_grid` / `get`);
  `block_pixel_dims(t)` (the `(R, C)` pixel shape from
  `idct::dct_pixel_dims` ∪ `non_dct_pixel_dims`, covering every
  `TransformType`); `place_block(plane, vb, block)` (verbatim copy with
  length-mismatch + footprint-spill rejection); and
  `assemble_channel_plane(grid, residual_at)` (raster-order grid walk
  invoking the caller's per-varblock decode closure once per top-left
  cell, continuation cells skipped, residual-`Empty` cell rejected). The
  plane is the padded block grid (no per-edge clamping; caller crops to
  `lf_w × lf_h`). The geometry invariant `C == block_dims().0 · 8` /
  `R == block_dims().1 · 8` is pinned for every transform. The IDCT
  output already carries the LLF/DC contribution (§I.2.5) so no separate
  DC add at placement; chroma-from-luma / Gaborish / EPF run on the
  assembled plane and remain caller-side concerns. 14 unit + 5
  integration (`round306_residual_plane`, composing the real F.3-dequant
  + I.2.3-IDCT walk end-to-end) tests. Lib tests 774 → 788 (+14).
  Pure-control-flow geometry primitive — no bit reads, no spec
  re-derivation, no histogram materialisation. Source of truth:
  ISO/IEC FDIS 18181-1:2021 §C.5.4 (DctSelect placement) + §C.8.3 +
  Table I.4 / §I.2.3 (pixel-dims).

- Round 300 — extend the per-block VarDCT decode walk
  (`src/block_dequant.rs`) to the **non-DCT transforms**: Hornuss,
  DCT2×2, DCT4×4, DCT4×8, DCT8×4, and AFV0..AFV3 — i.e. exactly the set
  for which `idct::non_dct_pixel_dims` returns `Some` (all `8 × 8`).
  This lifts the round-293 deferral. The deferral worried that the
  AFV / DCT2×2 sub-block coefficient extraction "does not reduce to a
  flat identity over an `8 × 8` grid", but per ISO/IEC FDIS 18181-1:2021
  the §I.2.3 sub-block re-mapping happens *inside* the inverse-transform
  dispatch (`idct_afv`, `idct_dct2x2`, …), which the spec applies
  *after* the Annex F.3 dequant. The §F.3 dequant stage is uniform: it
  multiplies each stored coefficient by a multiplier keyed on "the
  channel, the transform type and the coefficient index inside the
  varblock". For every non-DCT transform the varblock is the `8 × 8`
  OrderId-1 grid (`coeff_order::varblock_size_for_order` → `(8, 8)`),
  the dequant matrix is the `8 × 8` slot matrix
  (`weights_matrix_dims_for_slot` → `(8, 8)` for slots 1 / 2 / 3 / 9 /
  10), and the decoded block is already in raster index space
  (`coeffs[natural_order[k]]`, `natural_order[k] = y·bwidth + x`), so
  the per-cell dequant is the identity raster map — exactly as for the
  square / rectangular DCT family, with no orientation subtlety
  (`bwidth == bheight == 8`). `covered_grid_dims` now returns `Some` for
  every `TransformType`; `require_covered`'s `Unsupported` path now only
  guards a hypothetical future variant lacking a pixel-dims mapping.
  +3 lib tests (non-DCT all-zero residual census; non-DCT single-coeff
  per-sample-formula identity; AFV0..AFV3 shared-slot/grid dequant
  equality; chained == manual dequant-then-IDCT for the non-DCT path).
  Lib tests 771 → 774.

- Round 293 — extend the per-block VarDCT decode walk
  (`src/block_dequant.rs`) from the three square plain-DCT transforms
  to **every plain separable-DCT transform**: the rectangular
  DCT16×8 / DCT8×16 / DCT32×8 / DCT8×32 / DCT32×16 / DCT16×32 family and
  the larger DCT64×64 … DCT256×256 family. The round-286 orientation
  deferral is lifted by pinning, against ISO/IEC FDIS 18181-1:2021
  §I.2.4 + Table I.4 + Annex I.2.3.2, that the decoded coefficient grid
  (`varblock_size_for_order` → `(bwidth, bheight)`, `bwidth >= bheight`)
  and the dequant matrix (`weights_matrix_dims_for_slot` →
  `(cols, rows) = (bwidth, bheight)`) share **one** "wide"
  `bwidth × bheight` row-major layout, which is exactly the
  `(short × long)` "spec coefficient layout" `idct_for_transform`
  already consumes. A rectangular transform and its transpose
  (e.g. DCT16×8 / DCT8×16) share one coefficient grid and one dequant
  matrix; they differ only in the pixel orientation `(R, C)` the IDCT
  emits, so the per-cell dequant is the identity and no transpose is
  needed in this stage. New public API `covered_grid_dims(t) ->
  Option<(bwidth, bheight)>` (the full plain-DCT covered set, keyed off
  `dct_pixel_dims`); `covered_square_dim` retained for the square
  subset; `dequant_block_for_transform` / `decode_block_to_residual`
  now accept the whole plain-DCT set. The **non-DCT** transforms
  (Hornuss / DCT2×2 / DCT4×4 / DCT4×8 / DCT8×4 / AFV0..AFV3) stay
  `Error::Unsupported` — their dequant matrix is canonicalised to 8×8
  while their IDCT path is the §I.2.3 dispatch, so the sub-block
  coefficient extraction does not reduce to a flat 8×8 identity.
  +4 unit tests (transpose-pair grid/matrix sharing, full plain-DCT
  covered-set census, rectangular all-zero + pure-DC residuals);
  lib tests 767 → 771.
- Round 286 — first per-block VarDCT decode-walk stage that reaches
  spatial samples (`src/block_dequant.rs`). Chains the §C.8.3 decoded
  quantised-coefficient block through Annex F.3 HF dequantisation and
  the Annex I.2.3.2 inverse DCT for the square plain-DCT transforms
  (DCT8×8 / DCT16×16 / DCT32×32), where the coefficient grid, the
  dequantisation matrix, and the inverse-DCT input all share one
  unambiguous `dim × dim` row-major layout. New public API:
  `dequant_block_for_transform` (Annex F.3 across the whole raster,
  per-cell dequant-matrix entry via `slot_for_transform`),
  `decode_block_to_residual` (dequant → `idct_for_transform`), and
  `covered_square_dim`. Rectangular / non-DCT transforms return
  `Error::Unsupported`, deferred to a follow-up round so their
  coefficient-grid-vs-pixel-block orientation can be pinned
  independently. 11 unit tests; lib tests 756 → 767.

### Fixed

- Round 281 — two §C.8.3 decode-walk prose-conformance fixes against
  ISO/IEC FDIS 18181-1:2021, both affecting the (not-yet-wired)
  VarDCT HF coefficient path. (1) **Per-varblock channel decode
  order is Y, X, then B** — the §C.8.3 prose reads "for each
  varblock it reads channels Y, X, then B"; rounds 221..264 advanced
  the entropy stream X-first. Fixed in
  `block_context_resolver::decode_varblocks_three_channels_with_resolver`
  (round 221; also feeds the round-228 multi-pass and round-232
  HF-header drivers) and
  `HfHistogramDecodeContext::decode_three_channel_varblock_for_pass`
  (round 260; also feeds the round-264 per-LfGroup driver). Output
  arrays stay indexed 0 = X / 1 = Y / 2 = B per Listing C.13's
  "c is the current channel (with 0=X, 1=Y, 2=B)" — only the
  stream-advance order changed. The Listing C.13 `BlockContext()`
  channel mapping `(c < 2 ? c ^ 1 : 2)` (Y → 0, X → 1, B → 2)
  independently corroborates Y-first decode order. (2)
  **`NonZeros(x, y)` writeback covers every block of the varblock**
  — the prose reads "The decoder then computes the NonZeros(x, y)
  field for each block in the current varblock"; rounds 177..264
  wrote only the top-left cell, so a neighbouring varblock's
  `PredictedNonZeros(x, y)` reading a continuation cell of a
  multi-cell transform (e.g. the second row/column of a DCT16×16)
  saw the zero-init sentinel instead of the varblock's
  ceiling-divided value. `NonZerosGrid::update_after_block_for_transform`
  now fills the full `TransformType::block_dims()` footprint
  (rejecting footprints that spill outside the grid); the
  per-channel / per-pass wrappers and every typed driver above them
  inherit the fix. Ordering + footprint tests rewritten to the
  prose readings across `round177` / `round183` / `round190` /
  `round221` / `round228` suites plus the in-module unit tests; new
  rectangular-footprint (DCT16×8 1×2-cell vs DCT8×16 2×1-cell) and
  footprint-spill rejection pins. Tests 1156 → 1159.

- Round 278 — the long-standing `noise-64x64-lossless` Weighted-
  Predictor pixel divergence (rounds 31..272) is FIXED; the fixture
  decodes byte-exact on all three planes and the round-10 `synth_320`
  drift is gone (102400/102400 pixels correct). Two FDIS Annex E
  readings in `modular_fdis::wp_predict`, both pinned by the staged
  behavioural trace
  (`docs/image/jpegxl/fixtures/noise-64x64-lossless/wp-trace-sample-194.md`):
  (1) Listing E.2 `error2weight` performs the inner
  `(1 << 24) Idiv ((err_sum >> shift) + 1)` division FIRST and
  multiplies the truncated quotient by `maxweight` (the FDIS-2021
  parenthesisation) — the trace's 52 full-precision
  `(err_sum, weight)` cells (samples 188..200) all match this
  reading while the previous multiply-first form mismatches 18 of
  them; (2) the `true_errNW` read falls back to `true_errN` when NW
  does not exist (x = 0), matching the H.5.2 NW/NE→N edge rule the
  err_sum accumulator reads already applied — the previous zero
  fallback corrupted every column-0 prediction and produced the
  sample-129 `Δ = -21` state-evolution divergence. Root-caused via a
  from-scratch Annex E state-evolution sweep over the fixture's
  known-correct decoded values across every contested reading knob:
  exactly one combination reproduces all 13 traced samples plus the
  three known row-2 stored true_err cells (737 / -456 / -165), and
  it differs from production only in these two readings. The
  production 8x-domain `sub_err` reading (round 272) is confirmed —
  the literal reading now breaks the fixture at plane[0] sample 68.
  New `error2weight_pub` oracle + `tests/r278_error2weight_trace.rs`
  (3 tests) pin the 52 trace cells and the operand order; 12
  historical divergence-pin tests across 6 files
  (`r32`/`round10`/`r126`/`r195`/`r202`/`r272`) promoted to
  spec/pixel-exact assertions. Tests 1153 → 1156.

### Added

- Round 272 — extracted the Weighted-Predictor post-decode
  `sub_err_i` computation (FDIS Annex E.1 / §H.5.2) into the named
  `modular_fdis::sub_err_for` (8x-domain magnitude-then-round reading,
  used on the decode path) plus a `modular_fdis::sub_err_fdis_literal`
  reference oracle for the literal FDIS-2021 listing reading
  `abs(((prediction_i + 3) >> 3) - true_value)`. New
  `tests/r272_sub_err_reading.rs` (4 tests) pins the reading choice as
  a regression guard: the two readings coincide for every non-negative
  sub-prediction (so both reproduce the `noise-64x64-lossless`
  sample-194 trace value `sub_err = [122, 59, 18, 36]`) but diverge for
  negative sub-predictions; and the production decode path must keep
  `synth_320`'s round-10 drift anchor at PG[0][0] `(y=24, x=14)` — the
  literal reading moves it EARLIER to `(y=11, x=104)` (decodes the
  fixture less far), confirming the 8x-domain reading is the
  bisect-validated one. Round 272 also ruled the `sub_err` reading OUT
  as the cause of the residual `noise-64x64-lossless` sample-129
  `Δ = -21` WP state-evolution divergence (switching readings leaves
  that fixture's divergence profile unchanged).

- Round 264 —
  `multi_pass_hf_histogram_decoder::HfHistogramDecodeContext::decode_lf_group_three_channels_for_pass`
  bundled per-LfGroup raster-walk three-channel decode driver for one
  pass against ISO/IEC FDIS 18181-1:2021 §C.8.3 — one
  `(br, p, grid, resolver, qdc_at, predicted_at)` call walks the
  `DctSelectGrid` in raster order via `VarblockWalk`, invokes the
  caller's per-varblock `qdc_at` + `predicted_at` closures once per
  varblock to read the shared `qdc[3]` triple and the per-channel
  `predicted[3]` triple, then composes the round-260
  `decode_three_channel_varblock_for_pass` bundled three-channel walk
  to yield one `ThreeChannelVarblock` per top-left cell. Returns the
  in-raster-order `Vec<ThreeChannelVarblock>` per the round-221 / 228
  / 260 type alias. The driver owns both the raster walk **and** the
  §C.7.2 entropy-stream routing through the round-252 typed decode
  context — no `read_non_zeros` / `decode_symbol` closures cross the
  boundary, only the storage-only `qdc_at` + `predicted_at` lookups
  do. Per-varblock ordering: `qdc_at` fires before `predicted_at`;
  per-LfGroup ordering: row-major (DctSelectGrid raster). Defensive
  shape: propagates `VarblockWalk::next` errors (residual `Empty`
  cell), closure errors (`qdc_at` aborts before `predicted_at`;
  `predicted_at` error aborts before the inner method runs), and any
  inner `decode_three_channel_varblock_for_pass` error verbatim. On
  closure error the per-varblock cursor halts without advancing the
  BitReader past the failing call. Empty grid (`width × height ==
  0`) yields an empty output vector. 11 unit + 10 integration
  (`round264_lf_group_three_channels_for_pass`) tests pin: 1×1 DCT8×8
  short-circuit; 2×2 / 3×3 uniform raster ordering ((0,0), (1,0),
  (0,1), (1,1) — row-major); per-varblock `qdc → predicted → decode`
  ordering; per-pass offset routing matches round-260 cluster_map
  indexing for both `p = 0` and `p = 1`; mixed-transform grid
  (DCT16×16 single varblock covering 2×2 cells) emits one
  varblock with `coeffs.len() == 256` per channel; out-of-range pass
  index rejected; residual `Empty` cell rejected (VarblockWalk error
  propagated); closure errors (qdc_at / predicted_at) propagated
  without advancing the BitReader past the failing call; round-trip
  with `PerPassHfHeaders::read` driven off a real bitstream
  preserves per-pass histogram offsets across both passes; empty
  grid yields empty vector. Lib tests 742 → 753 (+11).
- Round 260 —
  `multi_pass_hf_histogram_decoder::HfHistogramDecodeContext::decode_three_channel_varblock_for_pass`
  bundled three-channel per-varblock walk against ISO/IEC FDIS
  18181-1:2021 §C.8.3 — one
  `(br, p, vb, resolver, qdc, predicted[3])` call composes the
  round-255 single-channel `decode_block_for_pass_transform` three
  times (channel order X = 0 → Y = 1 → B = 2 per the §C.8.3 listing
  sequence) against the round-214
  `BlockContextResolver::resolve(c, vb, qdc)` per-channel Listing
  C.13 `block_ctx` derivation, returning the per-channel
  `([DecodedHfBlock; 3], [u32; 3])` pair (decoded coefficient bundle
  plus the un-divided `raw_non_zeros` triple the caller threads into
  the per-channel NonZeros-grid bookkeeping). The `nb_block_ctx`
  invariant is read off `resolver.nb_block_ctx()` so the caller does
  not have to pass it separately; the `qdc[3]` triple is shared
  across the three channels per round-221's per-varblock invariant
  (one read, three lookups). Channel ordering is fixed at X → Y → B
  — the §C.7.2 entropy stream advances in that order; an error on Y
  aborts before B reads, so the B-channel ANS state is **not**
  advanced (matching round-221's error-path invariant). Defensive
  shape: propagates any `BlockContextResolver::resolve` error
  (channel `> 2`, `s` out-of-range, threshold-table inconsistency)
  and any `decode_block_for_pass_transform` error (out-of-range
  pass index, `u32`-overflow `ctx + offset`, downstream
  `EntropyStream` error, or `non_zeros > size - num_blocks` cap)
  verbatim. 8 unit + 11 integration
  (`round260_three_channel_varblock_for_pass`) tests pin: DCT8×8 /
  DCT16×16 / DCT16×8 / DCT8×16 / DCT4×4 per-channel short-circuit
  to `raw == [0, 0, 0] → coeffs_read == 0 → all-zero coeffs vector
  of the right length`; per-pass offset routing matches round-252
  cluster_map indexing for both `p = 0` and `p = 1` against a
  2-preset bundle; out-of-range pass index rejected; `u32` overflow
  on `ctx + offset` rejected; BitReader cursor unchanged on a
  short-circuited three-channel block; round-trip with
  `PerPassHfHeaders::read` driven off a real bitstream preserves
  the per-pass histogram offsets across both passes; per-channel
  `block_ctx` values resolved by the `BlockContextResolver` are `<
  nb_block_ctx` (= 15) for the default-table bundle. Lib tests 734
  → 742 (+8).

- Round 255 —
  `multi_pass_hf_histogram_decoder::HfHistogramDecodeContext::decode_block_for_pass_transform`
  bundled per-varblock decode method closing the round-252 deferred
  next-step "per-block raster walk remain caller-side concerns above
  this primitive" against ISO/IEC FDIS 18181-1:2021 §C.8.3 + Listing
  C.13 + Listing C.14. One `(p, t, predicted, block_ctx,
  nb_block_ctx)` call now wires the round-90 Listing C.14 state
  machine (`prev_nonzero[]` tracking, `non_zeros == 0` early-stop,
  `non_zeros > size - num_blocks` defensive cap) against the
  round-252 per-pass histogram routing for one varblock, returning
  the round-90 `DecodedHfBlock` coefficient bundle plus the un-
  divided `raw_non_zeros` for downstream `(raw + num_blocks - 1)
  Idiv num_blocks` NonZeros-grid bookkeeping. The internal walk is a
  single sequential `&mut self` loop because the two underlying
  entry points (`non_zeros_at`, `coefficient_at`) each need `&mut
  self` and therefore can't be wrapped into the round-90
  `read_non_zeros_and_decode_block_for_transform` closure pair —
  this method is the typed bridge. Defensive shape: rejects `p >=
  num_passes`, `ctx + offset > u32::MAX`, and `num_blocks == 0` /
  mismatched natural-order length, all without panicking. 7 unit +
  10 integration (`round255_decode_block_for_pass_transform`) tests
  pin: DCT8×8 / DCT16×16 / DCT16×8 / DCT8×16 / DCT4×4 short-circuit
  to `raw_non_zeros == 0 → coeffs_read == 0 → all-zero coeffs vector
  of the right length`; per-pass offset routing matches round-252
  cluster_map indexing; out-of-range pass index rejected; `u32`
  overflow on `ctx + offset` rejected; BitReader cursor unchanged on
  a short-circuited block; round-trip with `PerPassHfHeaders::read`
  driven off a real bitstream preserves the per-pass histogram
  offsets. Lib tests 727 → 734 (+7).

- Round 252 —
  `multi_pass_hf_histogram_decoder::HfHistogramDecodeContext` typed
  bridge that wires the round-247 `HfCoefficientHistograms` §C.7.2
  entropy stream to the round-232 `PerPassHfHeaders` per-pass
  `(hfp, histogram_offset)` array, closing the round-247 deferred
  next-step (the §C.8.3 per-block decode walk through the freshly-
  read histograms). Public surface against ISO/IEC FDIS 18181-1:2021:
  `HfHistogramDecodeContext::new(histograms, headers)` validates
  per-pass `hfp < histograms.num_hf_presets()` (defensive cross-
  container invariant) + `headers.num_passes() ≥ 1`, then caches the
  per-pass `histogram_offset` array for a single-array-index per-
  symbol path. Three decode entry-points expose the §C.8.3 prose
  shape: (1) `decode_symbol_for_pass(br, p, ctx)` performs the raw
  `D[ctx + histogram_offset(p)]` routing through
  `EntropyStream::decode_symbol`; (2) `non_zeros_at(br, p,
  predicted, block_ctx, nb_block_ctx)` composes
  `pass_group_hf::non_zeros_context` + the per-pass offset routing,
  matching the spec's `D[NonZerosContext(predicted) + offset]` line
  exactly; (3) `coefficient_at(br, p, k, non_zeros, num_blocks,
  size, prev, block_ctx, nb_block_ctx)` composes
  `pass_group_hf::coefficient_context` + the per-pass offset
  routing, matching the spec's `D[CoefficientContext(...) +
  offset]` line, and propagates the `num_blocks == 0` rejection
  without touching the `BitReader`. The `(ctx + offset)` sum is
  computed in `u64` with a defensive `u32` overflow check so the
  spec-permitted parameter maxima (`nb_block_ctx ≤ 256` ×
  `hfp < num_hf_presets ≤ 2^28`) cannot silently truncate. Accessor
  surface: `num_passes()`, `histogram_offset(p)`,
  `per_pass_offsets()` slice. Adds 10 unit tests + 9 integration
  tests (`tests/round252_multi_pass_hf_histogram_decoder.rs`)
  pinning: zero-pass rejection (no decode without passes); per-pass
  `hfp ≥ num_hf_presets` cross-container rejection; per-pass offset
  caching matches `PerPassHfHeaders::histogram_offset` independent
  read; single-symbol prefix decode for `(p, ctx)` matrix consumes
  zero bits and returns 0; out-of-range pass index rejection;
  `u32`-overflow synthetic `histogram_offset` rejection;
  `non_zeros_at` composes cleanly with `non_zeros_context` (cross-
  checked against the standalone helper); `coefficient_at` composes
  cleanly with `coefficient_context` (cross-checked against the
  standalone helper); `num_blocks == 0` rejection propagation does
  not advance the `BitReader`; round-trip with
  `PerPassHfHeaders::read` against a real bitstream (round-232
  derivation) preserves the per-pass offsets. Lib test count
  717 → 727 (+10). Pure-control-flow wiring primitive — no spec
  re-derivation, no ANS state initialisation, no per-block raster
  walk. The per-channel `BlockContext()` history threading, per-
  channel coefficient-order lookup against `hf_pass::HfPass`, and
  the per-block raster walk remain caller-side concerns above this
  primitive.

- Round 247 — `hf_coefficient_histograms::HfCoefficientHistograms`
  typed wrapper closing the round-238 deferred next-step. Performs
  the actual ISO/IEC FDIS 18181-1:2021 §C.7.2 codestream read of the
  `495 × num_hf_presets × nb_block_ctx` clustered-distributions block
  by routing `HfCoefficientHistogramSize::num_distributions()` into
  `modular_fdis::EntropyStream::read` as `num_dist`. Two entry-points:
  `read(br, size)` for a caller-built sizing descriptor, and
  `read_after_hf_pass_sequence(br, num_hf_presets, nb_block_ctx)`
  for the §C.7.1 → §C.7.2 transition convenience (constructs the
  sizing descriptor inline so a caller that has just walked
  `hf_pass::read_hf_pass_sequence` can drive the §C.7.2 step against
  the same `BitReader` without a separate constructor call). ANS
  state initialisation is deferred to `read_ans_state_init` per the
  round-3 2024-spec correction (the `u(32)` initialiser is read
  between the prelude and the first symbol decode); forwarded
  straight through to `EntropyStream::read_ans_state_init`. Defensive
  `usize`-cap guard on `num_distributions()` rejects 32-bit overflow
  before the `EntropyStream::read` call. Sizing accessors
  (`num_distributions`, `offset_for_hfp`, `num_hf_presets`,
  `nb_block_ctx`) forward through the underlying
  `HfCoefficientHistogramSize`. `entropy_mut()` exposes the
  underlying stream for the downstream §C.8.3 per-block decode loop.
  Adds 7 unit tests + 6 integration tests
  (`tests/round247_hf_coefficient_histograms.rs`). Lib test count
  710 → 717 (+7). Pure wiring primitive — the per-block decode walk
  through the freshly-read histograms (Listing C.13 contexts already
  landed by rounds 90 / 214 / 221 / 228 / 232) remains the next
  deferred step.

- Round 238 — `hf_coeff_histogram_size::HfCoefficientHistogramSize`
  typed sizing primitive for the §C.7.2 HF coefficient histogram
  block. Encapsulates the spec line "Let `nb_block_ctx` be equal to
  `max(block_ctx_map)+1`. The decoder reads a histogram with
  `495 × num_hf_presets × nb_block_ctx` clustered distributions D
  from the codestream as specified in D.3." behind a single typed
  constructor pair (`new(num_hf_presets, nb_block_ctx)` and
  `from_block_ctx_map(map, num_hf_presets)`), plus accessors
  `per_preset()` (`495 × nb_block_ctx`), `num_distributions()`
  (`495 × num_hf_presets × nb_block_ctx` — the §C.7.2 total),
  and `offset_for_hfp(hfp)` (`495 × nb_block_ctx × hfp` — the
  §C.8.3 per-pass routing offset, with `hfp < num_hf_presets`
  range check). Spec constant published as
  `PER_PRESET_PER_BLOCK_CTX = 495`. Defensive zero-input guards
  reject `num_hf_presets == 0`, `nb_block_ctx == 0`, and empty
  `block_ctx_map`. The duplicated `495u64 * num_hf_presets *
  nb_block_ctx` and `495u64 * nb_block_ctx * hfp` arithmetic in
  `hf_pass::HfPass::read` and `pass_group_hf::PassGroupHfHeader::read`
  is now routed through the primitive so the spec constant has one
  home and the per-pass offset shares its `nb_block_ctx` factor
  with the §C.7.2 read-size derivation. Sizing-only — the actual
  §C.7.2 `EntropyStream::read(br, num_distributions)` call against
  the clustered-distributions block remains the deferred next step.
  Adds 5 unit tests + 6 integration tests
  (`tests/round238_hf_coeff_histogram_size.rs`). Lib test count
  705 → 710 (+5). Pure refactor; no wire-format change. (§C.7.2
  entropy-stream read itself remains a deferred next step.)

- Round 232 — `multi_pass_hf_header::PerPassHfHeaders` +
  `decode_multi_pass_with_hf_headers` per-LfGroup multi-pass driver
  with per-pass `hfp` reads + per-pass `histogram_offset` routing
  (FDIS §C.8.3 first paragraph). New `multi_pass_hf_header` module
  wraps the round-228
  [`multi_pass_decode::decode_multi_pass_three_channels_with_resolver`]
  driver with the §C.8.3 first-paragraph per-pass header read
  `hfp = u(ceil(log2(num_hf_presets)))` and the derived
  `histogram_offset = 495 × nb_block_ctx × hfp` the spec writes as
  the `offset` term in `D[NonZerosContext(...) + offset]` and
  `D[CoefficientContext(...) + offset]`. `PerPassHfHeaders::read(br,
  num_passes, num_hf_presets, nb_block_ctx)` consumes the
  per-pass header sequence by invoking the round-90
  [`pass_group_hf::PassGroupHfHeader::read`] once per pass;
  `from_headers` builds the container from a pre-built `Vec`.
  Accessors expose per-pass `hfp` + `histogram_offset` + a
  `PassHfDigest` snapshot. The new driver
  `decode_multi_pass_with_hf_headers` mirrors the round-228 signature
  with two augmented closure shapes
  `read_non_zeros(p, channel, predicted, histogram_offset)` /
  `decode_symbol(p, channel, coeff_ctx, histogram_offset)` — the
  per-pass histogram_offset is pre-resolved once per pass before the
  inner per-varblock walk so the closure body sees a constant offset
  across each pass's per-channel calls. Pass count is taken from
  `headers.num_passes()` and verified against `nz.num_passes()`
  (mismatch returns `Error::InvalidData`). The companion
  `read_and_decode_multi_pass_with_hf_headers` reads the per-pass
  header sequence inline from a `BitReader` and invokes the driver
  in one call — the entry-point a future round wiring the §C.7.2
  entropy histogram bundle (#799 DOCS-GAP) into a per-pass
  `EntropyStream` will use. 16 unit + 12 integration
  (`round232_multi_pass_hf_header`) tests pin: per-pass header read
  with `num_hf_presets ∈ {1, 2, 4, 8}` (single-preset zero-bit fast
  path, two-preset one-bit-per-pass, four-preset two-bits-per-pass,
  eight-preset three-bits-per-pass with 15 bits across 5 passes);
  digest round-trip through bits LSB-first; `hfp = 0` always yielding
  `histogram_offset = 0` regardless of `nb_block_ctx`;
  `histogram_offset` scaling with `nb_block_ctx` (495 × 100 =
  49500); `get` / `histogram_offset` / `hfp` out-of-range errors;
  zero-passes degenerate case yielding an empty container;
  `PassGroupHfHeader::read` `num_hf_presets == 0` rejection
  propagating through `PerPassHfHeaders::read`; the driver routing
  the per-pass offset uniformly across all three channels (X / Y / B)
  within a pass; both `read_non_zeros` and `decode_symbol` closures
  receiving the matching per-pass offset (378 = 2 × 3 × 63
  decode_symbol calls covering the full DCT8×8 `k ∈ [num_blocks,
  size)` sweep); per-pass error propagation (pass-1 closure failure
  aborts the outer driver); `num_passes` mismatch
  (`headers.num_passes() != nz.num_passes()`) rejected pre-walk;
  pass-distinct `qdc_at` closure invocation preserving the round-228
  per-pass `qdc[3]` propagation; mixed transform `DCT16×8 + 2
  DCT8×8` layout consistency across passes with distinct per-pass
  offsets; inline `read_and_decode_multi_pass_with_hf_headers`
  end-to-end (header bits consumed exactly, decode walk runs, output
  shape matches); inline-read error path (empty BitReader yields a
  proper `Error::InvalidData` from `read_bit`); per-pass-header
  offsets-threaded-through-both-closures invariant verifying
  `decode_symbol` calls observe the same per-pass offset as
  `read_non_zeros` across the 2-pass × 3-channel sweep. Lib tests
  689 → 705 (+16). Pure-control-flow primitive in the same shape as
  round-89 [`dct_quant_weights`], round-95 [`hf_dequant`], round-121
  [`llf_from_lf`], round-138 [`chroma_from_luma`], round-141
  [`gaborish`], round-144 [`epf`], round-147 [`afv::afv_idct`],
  round-159 / 164 [`pass_group_hf`], round-177 [`non_zeros_grid`],
  round-183 [`per_channel_non_zeros`], round-190
  [`per_pass_non_zeros`], round-208 [`varblock_walk`], round-214
  [`block_context_resolver`], round-221's three-channel driver, and
  round-228's multi-pass driver — no bit reads beyond the per-pass
  `hfp` u-read defined by the spec line, no spec re-derivation, no
  histogram materialisation, no ANS state setup. A future round
  wiring §C.7.2 histograms + per-pass [`hf_pass::HfPass`] selection
  (the `select_pass(passes)` method on `PassGroupHfHeader` already
  performs the per-pass coefficient-order lookup) can drop this
  driver in as the per-LfGroup multi-pass HF-header + histogram-
  routing control-flow layer.

- Round 228 — `multi_pass_decode::decode_multi_pass_three_channels_with_resolver`
  per-LfGroup multi-pass three-channel varblock decode driver (FDIS
  §C.8.3 + Table C.6 `Passes`). New `multi_pass_decode` module lifts
  the round-221 single-pass three-channel driver into an outer
  per-pass loop that iterates `p ∈ [0, num_passes)`, gathering per-
  pass [`block_context_resolver::ThreeChannelVarblock`] vectors in
  pass order — `out[p][i]` is the `i`-th varblock (raster order)
  decoded in pass `p`. The driver reads `num_passes` off
  `nz.num_passes()` (the
  [`per_pass_non_zeros::PerPassNonZerosGrids`] container is the
  authoritative pass-count source), walks the
  [`dct_select::DctSelectGrid`] once per pass, invokes the caller's
  `qdc_at(p, &vb)` closure once per varblock per pass (so the
  closure may read from a per-pass quantised-LF buffer if the
  upstream signal evolves between passes), and threads each
  `(p, c)` call through
  [`per_pass_non_zeros::PerPassNonZerosGrids::decode_block_at_for_pass_channel`].
  The per-pass per-channel `NonZeros(x, y)` bookkeeping is already
  isolated by `p` (round-190 invariant), so the caller does not have
  to clear state between passes. The `read_non_zeros(p, channel,
  predicted)` / `decode_symbol(p, channel, coeff_ctx)` closures take
  the pass index as their first argument so the caller can route
  each call to the matching per-pass per-channel histogram without
  rebinding closures for each pass. The new
  `MultiPassThreeChannelOutput` type alias names the per-LfGroup
  output shape; the new `count_decoded_blocks(grid, num_passes)`
  helper returns `num_passes × count_varblocks(grid)` for callers
  that need to size a downstream coefficient buffer ahead of time
  (defensive u64 overflow check on the multiplication). 14 unit +
  12 integration (`round228_multi_pass_decode`) tests pin: single-
  pass single-DCT8×8 parity with the round-221 inner driver; 4×4
  DCT8×8 grid (16 varblocks) preserving raster order in a single
  pass; two-pass 2×2 raster-order per-pass walk; per-pass `qdc`
  closure invocation count (3 passes × 4 varblocks = 12 calls, not
  36); three-pass per-channel routing isolation with pass-distinct
  raw_non_zeros values landing on per-pass writeback cells without
  cross-pass leakage; pass error aborts remaining passes (the
  outer Vec is discarded on error); pass-0 inner error aborts
  before pass-1 starts (pass-1 closure never called); per-pass
  predicted invariant (`PredictedNonZeros(0, 0) = 32` across every
  pass + channel); per-pass `qdc[3]` value propagation through the
  outer loop; mixed-transform (`DCT16×8 + 2 DCT8×8`) layout
  consistency across passes; pass-1 channel routing read from
  pass-1 histogram; `count_decoded_blocks` helper covers
  `num_passes ∈ {0, 1, 2, 5, u32::MAX}`; DCT16×16 single-block
  single-pass pass-through; integration coverage of pass-index
  threading through both `read_non_zeros` and `decode_symbol`
  closures; inner-driver mid-varblock error (pass 1, X-channel
  decode_symbol failure) propagating through the outer loop.
  Lib tests 675 → 689 (+14). Pure-control-flow primitive in the
  round-89 / 95 / 121 / 138 / 141 / 144 / 147 / 159 / 164 / 177 /
  183 / 190 / 208 / 214 / 221 family; no bit reads, no spec re-
  derivation, no histogram materialisation. The follow-up §C.7.2
  histogram array (#799 DOCS-GAP) + per-pass `hfp` selection +
  per-channel `BlockContext()` history threading still apply
  unchanged — round 228 is purely the outer-loop control-flow
  layer above the round-221 inner three-channel driver.

- Round 221 — `block_context_resolver::decode_varblocks_three_channels_with_resolver`
  three-channel per-LfGroup varblock decode driver (FDIS §C.8.3
  prose ordering: outer varblock raster, inner X / Y / B channel
  sweep). Walks the `dct_select::DctSelectGrid` once; computes the
  shared `qdc[3]` triple once per varblock; invokes
  `BlockContextResolver::resolve` three times against that shared
  `qdc` (channel order 0 = X → 1 = Y → 2 = B); routes each `(p, c)`
  call through
  `per_pass_non_zeros::PerPassNonZerosGrids::decode_block_at_for_pass_channel`.
  Return is `Vec<ThreeChannelVarblock>` = per-varblock
  `(Varblock, [DecodedHfBlock; 3], [u32; 3])` triples in raster
  order; per-channel ANS closures are
  `read_non_zeros(channel, predicted)` and
  `decode_symbol(channel, coeff_ctx)` so the caller routes
  per-channel histograms inside one closure pair. The new
  `ThreeChannelVarblock` type alias names the per-varblock output
  triple. 11 unit + 12 integration
  (`round221_three_channel_resolver`) tests pin: single-DCT8×8 with
  3 per-channel decodes per varblock; 4×4 DCT8×8 grid (16 varblocks)
  preserving raster order; single DCT16×16 (1 varblock); qdc
  closure invoked exactly once per varblock (= 4 calls for 4
  varblocks, NOT 12); strict X / Y / B channel order at each
  `read_non_zeros` / `decode_symbol` call site; per-channel
  non_zeros writeback at `(0, c, 0, 0)` with distinct per-channel
  raw counts (10 / 20 / 30); per-pass routing (pass = 1 isolated
  from pass = 0); qdc error aborts before any per-channel reads;
  X-channel error aborts before Y + B reads; mixed-transform
  `DCT16×8 + 2 DCT8×8` placement preserved; custom `HfBlockContext`
  (qf_threshold = 5) round-trip; DCT16×16 `num_blocks = 4`
  per-channel non_zeros = 4 → 4 decode_symbol calls
  + `(4 + 3) / 4 = 1` stored.
- Round 214 — `block_context_resolver` module (per-LfGroup
  `BlockContext()` resolver, FDIS §C.8.3 Listing C.13 + §I.2.2
  `HfBlockContext` bundle). Exposes the borrow-based
  `BlockContextResolver::new(&HfBlockContext)` wrapper with a
  per-varblock `resolve(channel, &Varblock, qdc) -> Result<u32>`
  lookup (applies `order_id_for_transform` for `s`, threads
  `hf_mul` as `qf`, forwards `qdc[3]` + the LfGlobal
  `qf_thresholds` / `lf_thresholds` / `block_ctx_map` to the
  round-159 `pass_group_hf::block_context` formula) plus
  `decode_varblocks_with_resolver(grid, nz, p, c, &resolver,
  qdc_at, read_non_zeros, decode_symbol)` driver that pairs the
  round-208 `VarblockWalk` raster-order iterator with the
  round-190 `PerPassNonZerosGrids::decode_block_at_for_pass_channel`
  per-block primitive. The resolver eliminates the four-argument
  `(qf_thresholds, lf_thresholds, block_ctx_map, nb_block_ctx)`
  boilerplate at every per-varblock callsite. 14 unit + 12
  integration (`round214_block_context_resolver`) tests pin:
  borrow accessor + `nb_block_ctx` default-15 pass-through;
  default-branch `(c=0, s=0)` / `(c=1, s=0)` / `(c=2, s=0)`
  DCT8×8 → `block_ctx_map[{13, 0, 26}]` = `{7, 0, 7}`; DCT16×16 /
  DCT32×32 / DCT16×8 / DCT8×16 / Hornuss order-id mapping;
  default-branch invariance to `qdc` and `hf_mul` (empty
  thresholds collapse those knobs); custom-branch
  `qf_threshold` perturbation; driver pass-through on
  single-DCT8×8 / raster-order 2×2 DCT8×8 / single-DCT16×16
  grids; `qdc_at` closure called once per varblock in walk
  order; closure-error propagation. Lib tests 650 → 664 (+14).
  Pure-control-flow primitive in the round-89 / 95 / 121 / 138 /
  141 / 144 / 147 / 159 / 164 / 177 / 183 / 190 / 208 family; no
  bit reads, no spec re-derivation, no histogram materialisation.

- Round 208 — `varblock_walk` module (per-LfGroup varblock-walk
  driver, FDIS §C.5.4 + §C.8.3). Exposes the `Varblock` descriptor
  (`{x, y, transform, hf_mul}`), the borrow-based `VarblockWalk`
  raster-order iterator over a `dct_select::DctSelectGrid` (skips
  Continuation cells; residual Empty cell errors cleanly), the
  `count_varblocks` cell-scan helper, and the typed per-pass
  per-channel driver `decode_varblocks_for_pass_channel` that
  walks the grid + invokes the caller's `block_ctx_for_varblock`
  closure (Listing C.13 `BlockContext()` lookup) + threads each
  varblock through
  `per_pass_non_zeros::PerPassNonZerosGrids::decode_block_at_for_pass_channel`.
  Returns the in-raster-order `Vec<(Varblock, DecodedHfBlock,
  raw_non_zeros)>` triple. 14 unit + 12 integration
  (`round208_varblock_walk`) tests pin single-DCT8×8 / raster-order
  4×4 / DCT16×16-covers-2×2 / mixed-transform placement order /
  count-vs-walk parity / residual-Empty error / all-Continuation
  tolerance / hf_mul top-left read / typed driver per-pass
  per-channel routing isolation / closure-error propagation /
  DCT16×16 typed-driver pass-through / multi-varblock distinct
  hf_mul. Lib tests 636 → 650 (+14). Pure-control-flow primitive
  in the round-89 / 95 / 121 / 138 / 141 / 144 / 147 / 159 / 164 /
  177 / 183 / 190 family; no bit reads, no spec re-derivation, no
  histogram materialisation.

- Round 202 — `tests/r202_wp_row3_chain.rs` (7 tests) widens the
  round-191 / round-195 weighted-predictor diagnostic from a
  one-sample pin into a full-row chain across `noise-64x64-lossless`
  samples 192..=200, validating the production WP state against the
  trace doc's surrounding-sample context table
  (`wp-trace-sample-194.md` lines 130-168). New finding: the WP
  divergence is already large at sample 192 (`Δ pred8 = -50`,
  `Δ stored = -50`), before the round-191-pinned `Δ pred8 = +8` at
  sample 194. Tests pin in-row + cross-row read chains, sample 192's
  left-border zeroing, sample 194's cross-row reads, and the
  production decoded value `v(194) = 35`.

## [0.0.10](https://github.com/OxideAV/oxideav-jpegxl/compare/v0.0.9...v0.0.10) - 2026-05-30

### Other

- round-191 (parent-dispatch r191) against ISO/IEC FDIS 18181-1:2021 — Annex E / §H.5.2 Weighted-Predictor oracle test driven by clean-room behavioural trace at noise-64x64-lossless sample 194
- round-190 (parent-dispatch r190) against ISO/IEC FDIS 18181-1:2021 — typed per-pass NonZeros(x, y) grid container above the round-183 per-channel primitive
- round-183 (parent-dispatch r183) against ISO/IEC FDIS 18181-1:2021 — typed per-channel NonZeros(x, y) grid container layered above round-177 single-channel primitive
- round-177 (parent-dispatch r177) against ISO/IEC FDIS 18181-1:2021 — typed NonZeros(x, y) grid bookkeeping + per-varblock decode driver
- round-164 (parent-dispatch r164) against ISO/IEC FDIS 18181-1:2021 — TransformType-driven entry points for the §C.8.3 per-block HF coefficient decode loop
- round-159 (parent-dispatch r159) against ISO/IEC FDIS 18181-1:2021 — §C.8.3 per-block HF coefficient decode loop scaffolding (Listings C.13 + C.14)
- round-150 (parent-dispatch r150) against ISO/IEC FDIS 18181-1:2021 — Annex I.2.3.8 Listing I.13 Inverse AFV transform wired into idct dispatch
- round-147 (parent-dispatch r147) against ISO/IEC FDIS 18181-1:2021 — Annex I.2.2 AFV basis + AFV_IDCT pure-math primitive (Listings I.5 + I.6)
- round-144 (parent-dispatch r144) against ISO/IEC FDIS 18181-1:2021 — Annex J.3 edge-preserving-filter pure-math primitive
- round-141 (parent-dispatch r141) against ISO/IEC FDIS 18181-1:2021 — Annex J.2 Gabor-like-transform pure-math primitive
- round-138 (parent-dispatch r138) against ISO/IEC FDIS 18181-1:2021 — Annex G Chroma-from-Luma pure-math primitive (Listing G.1)
- round-133 (parent-dispatch r133) against ISO/IEC FDIS 18181-1:2021 — §C.7.1 DecodePermutation() for used_orders != 0
- Round 129: per-varblock LF→LLF composition glue (§I.2.5 plumbing)
- Round 126: WP deep-trace plumbing + sample-194 hand-derivation
- round-121 (parent-dispatch r121) against ISO/IEC FDIS 18181-1:2021 — §I.2.5 LLF-from-LF pure-math step (Listings I.15 + I.16)
- round-95 (parent-dispatch r95) against ISO/IEC FDIS 18181-1:2021 — §F.3 HF dequantisation pure-math step
- round-90 (parent-dispatch r90) against ISO/IEC 18181-1:2021 FDIS — HfPass + PassGroup HF structural parsers
- round-89 (parent-dispatch r89) against ISO/IEC 18181-1:2024 — GetDCTQuantWeights + Table I.6 default dequantization-matrix materialisation
- rewrite lf_dequant comment to remove libjxl numeric-defaults citation
- round-77 fixup — inline animation-3frame fixture under crate-local tests/fixtures/
- round-77 (parent-dispatch r17) against ISO/IEC 18181-1:2024 — animation-3frame SPECDIFF audit harness
- round-32 (parent-dispatch r17) against ISO/IEC 18181-1:2024 — noise-64x64-lossless pixel-divergence bisected to WP at first predictor=6 sample with WW/NN both in-image; fix deferred pending libjxl-WP behavioural trace
- round-31 (parent-dispatch r16) against ISO/IEC 18181-1:2024 — §F.3 zero-pad uniformly applied to single-TOC-entry LfGlobal fast path
- round-30 (parent-dispatch r15) against ISO/IEC 18181-1:2024 — bit-depth-16 RGB pixel-correct + 16-bit LE plane-pack convention
- round-29 (parent-dispatch r14) against ISO/IEC 18181-1:2024 — alpha-64x64 RGBA pixel-correct + ISOBMFF FF 0A strip
- round-28 (parent-dispatch r13) against ISO/IEC 18181-1:2024 — non-DCT IDCT helpers (Annex I.9.3..I.9.7)
- round-27 (parent-dispatch r12) against ISO/IEC 18181-1:2024 — IDCT dispatch (Annex I.2.1 + I.2.2 Listing I.4)
- round-26 (parent-dispatch r11) against ISO/IEC 18181-1:2024 — Annex L colour transforms (XYB inverse + YCbCr inverse)
- round-25 (Auditor mode) against ISO/IEC 18181-1:2024 — d1 LfCoefficients per-sample rich-state range dump 22..=79
- round-24 (Auditor mode) against ISO/IEC 18181-1:2024 — d1 per-cluster D[] byte trace + per-call alias-mapping invariant audit
- round-23 (Auditor mode) against ISO/IEC 18181-1:2024 — d1 leaf-pick property dump at Y' sample 22 + WP y=0 boundary audit
- round-22 (Auditor mode) against ISO/IEC 18181-1:2024 — d1 lf_quant sample dump + WP rounding bias toggle
- round-21 (Auditor mode) against ISO/IEC 18181-1:2024 — d1 per-cluster distribution + alias-table self-map audit
- round-20 followup — refresh round-19 trace eprintln with corrected DC_GROUP budget
- round-20 (Auditor pivot) against ISO/IEC 18181-1:2024 — DC_GROUP boundary recount + ANS-final-state oracle
- round-19 (Auditor mode) — d1 cluster + ANS state evolution audit
- round-18 (Auditor mode) against ISO/IEC 18181-1:2024 — per-token bit accounting trace + drift narrowed

### Added

- **Round 191 (2021-FDIS) — Annex E / §H.5.2 Weighted-Predictor
  oracle test driven by clean-room behavioural trace at sample 194 of
  `noise-64x64-lossless`.** New `tests/r191_wp_trace_oracle.rs` (5
  tests) and new `pub fn modular_fdis::wp_predict_pub` test wrapper
  around the production `wp_predict`. The oracle consumes the
  `docs/image/jpegxl/fixtures/noise-64x64-lossless/wp-trace-sample-194.md`
  trace (provenance recorded alongside as `wp-trace-provenance.md`),
  which records the FDIS-conformant per-listing intermediates an
  instrumented reference decoder produces at the
  `(channel 0, x=2, y=3)` divergence point bisected in rounds 31..126:
  - `r191_wp_predict_matches_trace_at_sample_194` — drives the
    production `wp_predict` with the trace's `WpState`/`Neighbours`
    inputs; asserts the four sub-predictions `[1248, 747, 420, 559]`,
    the final pre-round prediction `709`, and `max_error = 737` all
    reproduce exactly. **Result: PASS** — proves Annex E.2 Listings
    E.1 (sub-predictions), E.2 (`err_sum_i` + `error2weight`), E.3
    (weighted sum + same-sign clamp), and E.4 (`max_error`) are
    spec-correct in `wp_predict`, isolating the still-unfixed
    sample-194 wp_pred8 = 717 vs trace 709 off-by-8 divergence to
    **upstream state evolution** (`set_true_err` / `set_sub_err`
    calls fired across samples 0..193) rather than the predictor
    arithmetic itself.
  - `r191_trace_err_sum_self_consistency` — pure-arithmetic sanity
    check on the trace's `sub_err_{i,N/NE/NW}` table summing to the
    reported `err_sum_i` (`[438, 330, 416, 240]`).
  - `r191_trace_weights_match_error2weight` — hand-derives the
    trace's `weight_i = [495694, 599189, 474830, 825112]` from
    FDIS-literal `error2weight(err_sum_i, wp_w_i)`; documents a
    1-unit inner-Idiv-vs-multiplication-first discrepancy with the
    production reading that does NOT affect sample 194's shifted
    weights (both readings give `[3, 4, 3, 6]` after the Listing E.3
    `>> sh` step).
  - `r191_trace_prediction_matches_listing_e3` — independent
    hand-derivation of `prediction = 709` from Listing E.3 inputs,
    including verification that the same-sign clamp predicate fires
    but is a no-op (pre-clamp 709 ∈ [min(W,N,NE)=584, max(W,N,NE)=
    1232]).
  - `r191_pin_state_evolution_gap` — pins the production-vs-trace
    delta as a roadmap for the next round's bisect: Δ te_w = +21,
    Δ te_nw = -21 (symmetric pair → likely a single upstream
    defect), Δ wp_pred8 = +8 in 8x scale = +1 in un-shifted pixel
    space (matches `r126_first_divergence_scan` dec=35 vs exp=34).
  Spec citations and provenance attestation embedded in the test
  module docstring; references the in-repo FDIS §E.1-E.4 line
  numbers and the trace doc's stated `prediction − true_value`
  sign convention. Trace doc is the newly-staged
  `docs/image/jpegxl/fixtures/noise-64x64-lossless/wp-trace-*.md`
  pair landed alongside this round (tasks #820 + #1077). Issues #6,
  #64, #799.

- **Round 190 (2021-FDIS) — typed per-pass `NonZeros(x, y)` grid
  container (FDIS §C.8.3 + Listing C.13 per-pass keying).** New
  `per_pass_non_zeros` module that owns one
  `per_channel_non_zeros::PerChannelNonZerosGrids` per pass index
  `p ∈ [0, num_passes)`, layered above the round-183 per-channel
  container. A VarDCT frame is decoded in `num_passes` ordered passes
  (declared in `FrameHeader.passes.num_passes`); each pass scans every
  `PassGroup` once and §C.8.3 specifies that within a pass each
  channel of each varblock maintains its own `NonZeros(x, y)` state.
  Between passes the per-channel bookkeeping is reset because the
  per-pass histogram is selected by `hfp` from the per-pass `HfPass`
  array — a different pass uses a different histogram and the
  prediction recurrence is keyed against the current pass's own
  coefficient counts. The new module captures the per-pass routing
  layer above round 183's per-channel routing layer:
  - `PerPassNonZerosGrids::new(pass_dims: &[&[(u32, u32)]]) -> Result<Self>`
    — per-pass per-channel `(width, height)` slice, validated
    entry-by-entry via `PerChannelNonZerosGrids::new` (zero / oversize
    dims rejected per channel; empty pass-list rejected).
  - `PerPassNonZerosGrids::new_uniform(num_passes, num_channels, width,
    height) -> Result<Self>` — convenience builder for the
    uniform-per-pass case.
  - `PerPassNonZerosGrids::{num_passes, pass, pass_mut, predicted, get,
    set, update_after_block, update_after_block_for_transform}` —
    per-pass routing accessors; out-of-range `p` errors cleanly.
  - `PerPassNonZerosGrids::decode_block_at_for_pass_channel(p, c, x, y,
    t, block_ctx, nb_block_ctx, read_non_zeros, decode_symbol)
    -> Result<(DecodedHfBlock, u32)>` — typed per-pass per-channel
    driver that wraps the round-183
    `PerChannelNonZerosGrids::decode_block_at_for_channel` with pass
    routing. Caller pre-computes `block_ctx` via
    `pass_group_hf::block_context` with the matching `c`; the
    container is a pure storage + routing primitive and does not
    re-derive `pass_group_hf::block_context` nor materialise the
    per-pass histogram.
  - Per-pass per-channel shapes are independent — ragged per-pass
    channel counts are tolerated.

  41 new tests (28 unit in `per_pass_non_zeros::tests` + 13 integration
  in `tests/round190_per_pass_non_zeros.rs`) pin: empty-pass-list /
  zero-channel-pass / zero-dim rejection; two-pass chroma-subsampled
  construction; `new_uniform` convenience; out-of-range pass index
  errors on every accessor (8 paths); `PredictedNonZeros(0, 0) = 32`
  on every (pass, channel); per-pass write isolation; per-pass
  `predicted` propagation reads back each pass's own history (not
  another pass's); per-pass `update_after_block_for_transform`
  dispatch (raw `non_zeros = 17` → `{17, 5, 2}` at DCT8×8 / DCT16×16 /
  DCT32×32 on three independent passes); per-pass
  `decode_block_at_for_pass_channel` routing; two-pass three-channel
  raster walk at `(0, 0)` / `(1, 0)` with distinct `[4, 8, 12]` /
  `[3, 6, 9]` per-pass per-channel `raw_non_zeros` sequences preserves
  cross-pass isolation; ragged per-pass channel counts (one-channel
  DC-only preview followed by three-channel main); `u32::MAX`
  no-panic saturating-add chain through the per-pass route. Lib
  tests 608 → 636 (+28).

- **Round 183 (2021-FDIS) — typed per-channel `NonZeros(x, y)` grid
  container (FDIS §C.8.3 + Listing C.13 channel-keying).** New
  `per_channel_non_zeros` module that owns one
  `non_zeros_grid::NonZerosGrid` per channel, layered above the
  round-177 single-channel primitive. Listing C.13's
  `BlockContext()` factors `c` into `(c < 2 ? c ^ 1 : 2) × 13 + s`,
  so the `NonZeros(x, y)` bookkeeping is keyed per-channel because
  chroma subsampling + `TransformType` heterogeneity means each
  channel's varblock-grid shape can differ:
  - `PerChannelNonZerosGrids::new(dims: &[(u32, u32)]) -> Result<Self>`
    — per-channel `(width, height)` slice, validated entry-by-entry
    via `NonZerosGrid::new` (zero / `> 65535` dims rejected; empty
    slice rejected).
  - `PerChannelNonZerosGrids::new_uniform(num_channels, width,
    height) -> Result<Self>` — convenience builder for the
    unsubsampled 4:4:4-style container.
  - `PerChannelNonZerosGrids::{num_channels, grid, grid_mut,
    predicted, get, set, update_after_block,
    update_after_block_for_transform}` — per-channel routing
    accessors; out-of-range `c` errors cleanly.
  - `PerChannelNonZerosGrids::decode_block_at_for_channel(c, x, y,
    t, block_ctx, nb_block_ctx, read_non_zeros, decode_symbol)
    -> Result<(DecodedHfBlock, u32)>` — typed per-channel driver
    that wraps the round-177 `non_zeros_grid::decode_block_at`
    with channel routing. Caller pre-computes `block_ctx` via
    `pass_group_hf::block_context` with the matching `c`; the
    container is a pure storage + routing primitive.
  - `DEFAULT_NUM_CHANNELS = 3` — the YCbCr / XYB canonical channel
    count.

  36 new tests (24 unit in `per_channel_non_zeros::tests` + 12
  integration in `tests/round183_per_channel_non_zeros.rs`) pin:
  empty-channel-list rejection; zero-dim / oversize-dim rejection
  on any channel; three-channel chroma-subsampled construction at
  `[(16, 16), (8, 8), (8, 8)]`; `new_uniform` convenience;
  out-of-range channel index errors on every accessor (8 paths);
  `PredictedNonZeros(0, 0) = 32` on every channel; per-channel
  write isolation; per-channel `predicted` horizontal chain on a
  seeded channel-1 grid; `update_after_block_for_transform`
  dispatch (raw `non_zeros = 17` → `{17, 5, 2}` at DCT8×8 /
  DCT16×16 / DCT32×32 on three independent channels);
  `decode_block_at_for_channel` routes the round-177 typed driver
  per channel; post-update cell feeds the next-position predicted
  value back per-channel; OOB `(x, y)` past the per-channel grid
  errors cleanly; a two-step three-channel raster walk at
  `(0, 0)` / `(1, 0)` with distinct `[4, 12, 20]` /
  `[6, 18, 30]` per-channel raw_non_zeros sequences preserves
  cross-channel isolation.

  Lib tests 584 → 608 (+24). Pure-control-flow primitive in the
  same shape as round-89 `dct_quant_weights`, round-95
  `hf_dequant`, round-121 `llf_from_lf`, round-138
  `chroma_from_luma`, round-141 `gaborish`, round-144 `epf`,
  round-147 `afv_idct`, round-159 / 164 `pass_group_hf`, and
  round-177 `non_zeros_grid` — no bit reads, no spec
  re-derivation. A future round wiring §C.7.2 entropy histograms
  (#799 DOCS-GAP) + the per-LfGroup varblock-shape grid +
  per-channel `BlockContext()` history can drop these helpers in
  as the per-channel step without re-deriving any Listing C.13 /
  C.14 formulae.
- **Round 177 (2021-FDIS) — typed `NonZeros(x, y)` grid bookkeeping +
  per-varblock decode driver (FDIS §C.8.3 + Listing C.13 prelude +
  Listing C.14 post-prose).** New `non_zeros_grid` module bridging
  round 159 `pass_group_hf::predicted_non_zeros` (the four-branch
  `PredictedNonZeros(x, y)` recurrence) with round 164
  `pass_group_hf::read_non_zeros_and_decode_block_for_transform`
  (the `TransformType`-driven per-block coefficient loop):
  - `NonZerosGrid::new(width, height) -> Result<Self>` — rectangular
    varblock-grid storage of `NonZeros(x, y)` cells. Defensive
    rejection of zero dims + dims `> 65535`.
  - `NonZerosGrid::{get, set, width, height, cells}` — accessors.
  - `NonZerosGrid::predicted(x, y) -> Result<u32>` — delegates to
    `pass_group_hf::predicted_non_zeros` against
    `|xx, yy| self.get(xx, yy).unwrap_or(0)`.
  - `NonZerosGrid::update_after_block(x, y, non_zeros, num_blocks)
    -> Result<u32>` — FDIS post-Listing-C.14 prose formula
    `(non_zeros + num_blocks - 1) Idiv num_blocks` (ceiling-divide
    identity, `saturating_add` at `u32::MAX`).
  - `NonZerosGrid::update_after_block_for_transform(x, y, non_zeros,
    t)` — `num_blocks` from `pass_group_hf::transform_block_params`.
  - `non_zeros_grid::decode_block_at(grid, x, y, t, block_ctx,
    nb_block_ctx, read_non_zeros, decode_symbol) -> Result<
    (DecodedHfBlock, u32)>` — typed per-varblock driver: computes
    `predicted`, invokes
    `read_non_zeros_and_decode_block_for_transform`, then calls
    `update_after_block_for_transform` before returning the
    `(DecodedHfBlock, raw_non_zeros)` pair.

  35 new tests (23 unit in `non_zeros_grid::tests` + 12 integration
  in `tests/round177_non_zeros_grid.rs`) pin: defensive rejection
  of zero / oversize (`> 65535`) dims and out-of-range `(x, y)`;
  zero-init cells; `PredictedNonZeros(0, 0) = 32` across a sweep
  of grid shapes; the y == 0 and x == 0 border-recurrence branches
  via horizontal / vertical raster chains; the interior
  `(above + left + 1) >> 1` average (odd-sum rounding); the
  `predicted_non_zeros` helper agreement byte-for-byte on a seeded
  3×3 grid; the post-Listing-C.14 ceiling-divide formula at
  `num_blocks ∈ {1, 4, 16}` (DCT8×8 / DCT16×16 / DCT32×32 — the
  `TransformType` dispatch reduces a raw `non_zeros = 17` to
  `{17, 5, 2}` at the three shapes); the typed driver's
  `predicted = 32` at the origin routes through the `predicted >=
  8` `NonZerosContext` branch (`ctx = block_ctx + nb_block_ctx ×
  (4 + 32 Idiv 2) = 67` at `(block_ctx, nb_block_ctx) = (7, 3)`);
  `decode_block_at` reads back `(0, 0)`'s post-update cell when
  invoked at `(1, 0)`; OOB positions error cleanly; per-channel
  independence (two grids of the same shape evolve
  independently); row-major `cells()` layout pinned at `[0, 10,
  20, 30]` after writing `(1,0)=10`, `(0,1)=20`, `(1,1)=30` on a
  2×2 grid; and pathological `u32::MAX` does not panic.

  Lib tests 561 → 584 (+23). Pure-control-flow primitive in the
  same shape as round-89 `dct_quant_weights`, round-95
  `hf_dequant`, round-121 `llf_from_lf`, round-138
  `chroma_from_luma`, round-141 `gaborish`, round-144 `epf`,
  round-147 `afv_idct`, and round-159 / 164 `pass_group_hf` — no
  bit reads, no spec re-derivation. A future round wiring §C.7.2
  entropy histograms (#799 DOCS-GAP) + the per-LfGroup
  varblock-shape grid + per-channel `BlockContext()` history can
  drop these helpers in as the per-varblock-position step without
  re-deriving any Listing C.13 / C.14 formulae.
- **Round 164 (2021-FDIS) — `TransformType`-driven entry points for
  the §C.8.3 per-block HF coefficient decode loop (DCT16×16 /
  DCT16×8 / DCT32×32 dimensions pinned end-to-end).** New public API
  in `pass_group_hf`:
  - `transform_block_params(t: TransformType) -> (num_blocks, size)`
    — §I.2.4 opening paragraph + Listing C.14: `num_blocks =
    (bwidth / 8) × (bheight / 8)`, `size = bwidth × bheight`.
  - `decode_block_coefficients_for_transform(t, initial_non_zeros,
    block_ctx, nb_block_ctx, decode_symbol)` — typed wrapper that
    derives `(num_blocks, size, natural_order)` from `t` (via
    [`coeff_order::order_id_for_transform`] +
    [`coeff_order::natural_coeff_order`]) and reduces to the
    round-159 `decode_block_coefficients`.
  - `read_non_zeros_and_decode_block_for_transform(t, predicted,
    block_ctx, nb_block_ctx, read_non_zeros, decode_symbol)` —
    analogous typed wrapper around
    `read_non_zeros_and_decode_block`.
  20 new tests (8 unit in `pass_group_hf::tests` + 12 integration
  in `tests/round164_dct16x16_block_coefficient_loop.rs`) pin the
  `(num_blocks, size)` derivation for every Table C.16 transform
  (every entry satisfies `num_blocks * 64 == size`); the DCT16×16
  `prev` threshold at `non_zeros == 17` (= size/16 + 1); the typed
  entry point at DCT8×8 reduces to the raw entry point; the typed
  entry point at DCT16×16 walks `(num_blocks=4, size=256)` for
  all-zero / single-non-zero / three-consecutive / full-density
  (252 reads) cases with coefficients landing at
  `natural_coeff_order(Id2)[4..]`; the typed and raw entry points
  agree byte-for-byte on a mixed `[2, 0, 4, 0, 0, 6]` sequence;
  `read_non_zeros_and_decode_block_for_transform` threads the
  `NonZerosContext` value through the first closure; the rectangular
  DCT16×8 / DCT8×16 collapse to the same per-block outcome (they
  share OrderId::Id4); defensive rejection of `initial_non_zeros >
  size - num_blocks` (= 252 max for DCT16×16); and one DCT32×32
  smoke-test at `(num_blocks=16, size=1024)`. Lib tests 553 → 561
  (+8). Pure-typed wrapper layer: no new bit reads, no spec
  re-derivation — the round-159 module note ("the primitive itself
  is shape-agnostic and ready for the larger variable-block sizes
  once their parameterisation lands") is now exercised from the
  caller-facing API.
- **Round 159 (2021-FDIS) — §C.8.3 per-block HF coefficient decode
  loop scaffolding (Listing C.13 + Listing C.14).** New public API in
  `pass_group_hf`:
  - `prev_for_context(k, num_blocks, size, non_zeros, prev_nonzero)`
    — Listing C.14 verbatim (`k == num_blocks ? (non_zeros > size /
    16 ? 1 : 0) : (prev_nonzero(k - 1) ? 1 : 0)`).
  - `DecodedHfBlock { coeffs, remaining_non_zeros, coeffs_read }` —
    return bundle for the per-block primitive.
  - `decode_block_coefficients(natural_order, num_blocks, size,
    initial_non_zeros, block_ctx, nb_block_ctx, decode_symbol)` —
    Listing C.14's per-block raster-order loop with the §C.8.3
    "stop when non_zeros reaches 0" early-exit, `UnpackSigned`
    application, and `natural_order[k]` placement. The
    `decode_symbol: FnMut(ctx) -> Result<u32>` closure abstracts
    over the (still un-landed) §C.7.2 entropy histograms — a real
    consumer wraps `EntropyStream` + `HybridUintState` + the
    per-group `histogram_offset`; tests can hand-roll a symbol
    sequence.
  - `read_non_zeros_and_decode_block(.., predicted, .., read_non_zeros,
    decode_symbol)` — convenience wrapper that issues the
    `D[NonZerosContext(predicted) + offset]` read via the first
    closure and drives `decode_block_coefficients` with the result.
    Returns `(DecodedHfBlock, non_zeros)` so the caller can update
    its NonZeros-grid bookkeeping per `NonZeros(x, y) = (non_zeros
    + num_blocks - 1) Idiv num_blocks`.

  Bounded scope: DCT8×8 alone — `num_blocks = 1`, `size = 64`,
  `OrderId::Id0` natural-coefficient order (the simplest case that
  exercises the full state machine). The primitive itself is
  shape-agnostic; the larger variable-block sizes (DCT16×16,
  DCT32×32, AFV0..3, …) need their `num_blocks` / `size` parameters
  threaded through the varblock driver above this primitive.

  11 new unit tests (`pass_group_hf::tests::*`) + 11 integration
  tests (`round159_block_coefficient_loop`) cover: all-zero block
  (no symbol reads); single non-zero at the first HF slot (one
  read, `UnpackSigned(1) = -1` at `natural_order[1]`); three
  consecutive non-zeros (loop stops after three reads); full
  density (63 reads, LLF cell untouched); the size/16 threshold
  for `prev` (crossover at `non_zeros == 5` for DCT8×8); the
  "previous coefficient is zero / non-zero" flag tracking through
  the loop's history; defensive rejection of malformed
  natural-order vectors, zero `num_blocks`, and over-large
  `initial_non_zeros`; closure-threaded end-to-end smoke through
  `read_non_zeros_and_decode_block`. Lib tests 538 → 553 (+15).

  Pure-math / pure-control-flow primitive in the same shape as
  round-89 `dct_quant_weights`, round-95 `hf_dequant`, round-121
  `llf_from_lf`, round-138 `chroma_from_luma`, round-141
  `gaborish`, round-144 `epf`, and round-147 `afv_idct` — a future
  round wiring §C.7.2 histograms into the per-pass entropy stream
  can drop this primitive in as the per-block loop body without
  re-deriving any C.13 / C.14 formulae. The §C.7.2 entropy
  histogram decode (#799 DOCS-GAP), the per-channel (Y / X / B)
  `non_zeros` read in the varblock driver above this primitive,
  the per-pass NonZeros-grid update, and the per-varblock
  `BlockContext()` derivation remain follow-up work for subsequent
  rounds.

- **Round 150 (2021-FDIS) — Annex I.2.3.8 / Listing I.13 Inverse AFV
  transform composition (`idct::idct_afv`).** Composes the round-147
  `crate::afv::afv_idct` pure-math primitive (Listings I.5 + I.6)
  with two `idct_2d` calls (one at 4×4, one at 4×8) per the
  three-sub-block decomposition of Listing I.13 — yielding the
  full 8×8 sample buffer for `TransformType::Afv0..Afv3`. With
  this wiring the `idct::idct_for_transform` dispatcher routes
  `Afv0..Afv3` to `idct_afv` instead of returning
  `Err(Unsupported)`; all 10 non-DCT branches of Table I.4 are now
  pure-math-complete (Hornuss / DCT2×2 / DCT4×4 / DCT8×4 / DCT4×8
  + AFV0..AFV3). Each AFV variant's sub-block placement is
  controlled by `flip_x = n & 1` / `flip_y = n >> 1` (§I.2.3.8);
  the AFV sub-block additionally mirrors its read coordinates
  (`flip_x == 1 ? 3 - ix : ix` and the iy dual) per the inner
  loop of Listing I.13. Seven new property-style tests cover:
  rejection of non-AFV transforms / wrong lengths; all-zero
  input → all-zero output for all four variants; DC-only input
  → constant `c(0,0)` output (the three DC patches `(c00+c01+c10)
  × 4`, `c00-c01+c10`, `c00-c01` collapse to `4·1`, `1`, `1`
  respectively, with each sub-block's IDCT mapping a DC-only
  cell to a constant sub-block since AFVBasis row 0 = `[0.25;
  16]` and `IDCT_2D` DC-only is constant); dense-AC input →
  every cell written; AFV0↔AFV1 x-axis flip swaps the AFV
  sub-block column reads; AFV0↔AFV2 y-axis flip swaps the 4×8
  sub-block y-band placement; linearity. Test-count delta:
  `+7` (531 → 538).

  **FDIS typo documented in module docs.** Listing I.13's final
  source line reads `samples_4×4(ix, iy)` but the inner loop
  iterates `ix ∈ [0..8)` and `samples_4×4` only has columns
  `0..3`, while the immediately preceding line computes
  `samples_4×8 = IDCT_2D(coeffs_4×8)`. Implementation reads from
  `samples_4×8` per context; the typo is now annotated alongside
  the existing four Annex D / D.3 typos in the project
  FDIS-typo memory.

- **Round 147 (2021-FDIS) — Annex I.2.2 AFV basis + `AFV_IDCT`
  pure-math primitive (Listings I.5 + I.6, p. 76).** New
  `src/afv.rs` module transcribes the orthonormal `AFVBasis[16][16]`
  table from Listing I.5 verbatim and the Listing I.6 cell-sum
  `samples[i] = sum_j coefficients[j] × AFVBasis[j][i]`. Public
  API:
  - `AFV_CELL_LEN: usize = 16` — the §I.2.2 4×4-as-flat-16 cell.
  - `AFV_BASIS: [[f32; 16]; 16]` — verbatim Listing I.5.
  - `afv_idct(coefficients: &[f32]) -> Result<[f32; 16]>` —
    Listing I.6.

  The 256-float transcription is independently verified at the
  table level: row-0 = `[0.25; 16]` (Listing I.5 line 1); row-4 =
  two non-zero entries at columns 1 and 4, both at ±`1/sqrt(2)`,
  zero elsewhere (Listing I.5 line 5); per-row L2 unit-norm
  (orthonormality diagonal); pairwise zero inner product
  (orthonormality off-diagonal); `afv_idct` is linear; one-hot
  coefficient input recovers `AFVBasis[j]` row-for-row;
  `||samples||_2 == ||coefficients||_2` (L2 conservation, an
  orthonormal-basis property). A single transcription typo in any
  of the 256 entries would fail at least one orthonormality sum.

  10 new unit tests + 9 integration tests
  (`round147_afv_idct`); lib tests 521 → 531. Pure-math primitive
  in the same shape as round-89 `dct_quant_weights`, round-95
  `hf_dequant`, round-121 `llf_from_lf`, round-138
  `chroma_from_luma`, round-141 `gaborish`, and round-144 `epf` —
  a future round wiring §I.2.3.8 Inverse AFV transform (Listing
  I.13) into `idct_for_transform` can drop this helper in without
  re-deriving any I.5 / I.6 cells. The Listing I.13 composition
  (the `coeffs_afv` corner-load, the two `IDCT_2D` 4×4 / 4×8
  sub-blocks, the `flip_x` / `flip_y` AFVn flip) remains
  follow-up work because it depends on `idct_2d` for non-square
  blocks plus the AFVn dispatch wiring; the §I.2.2 arithmetic
  core landed in this round unblocks that follow-up.

- **Round 144 (2021-FDIS) — Annex J.3 "Edge-preserving filter"
  pure-math primitive (pages 85–87).** New `src/epf.rs` module
  transcribes the four §J.3 listings as a self-contained pure-math
  primitive: given a triple of three-channel f32 planes (the output
  of round-141 Gaborish on the §I.2.5 + Annex G chain), per-call
  scalar parameters (sigma, step_multiplier, zeroflush,
  position_multiplier_border, channel_scale), and a
  [`frame_header::RestorationFilter`] (Table C.9) for
  `epf_quant_mul` / `epf_sharp_lut[..]` / `epf_sigma_for_modular`,
  this module returns the per-pass output planes Listing J.4
  prescribes. Public API:
  - `distance_step_0_and_1(x, y, b, w, h, x, y, cx, cy, scale)` —
    Listing J.1 `DistanceStep0and1` (the five-pixel cross-shape
    three-channel scaled L1 distance for passes 0 and 1).
  - `distance_step_2(...)` — Listing J.1 `DistanceStep2` (the
    single-sample three-channel scaled L1 distance for pass 2,
    under the literal `(ix, iy) == (0, 0)` reading of the free-
    variable bug — see DOCS-GAP).
  - `weight(distance, inv_sigma, position_multiplier, zeroflush)`
    — Listing J.2 `Weight()` decreasing-function-of-distance
    kernel with the `v <= zeroflush` cutoff.
  - `inv_sigma_for_pass(step_multiplier, sigma)` — Listing J.2's
    pre-computed `step_multiplier × 4 × (sqrt(0.5) - 1) / sigma`
    factor (rejects non-finite or non-positive sigma).
  - `vardct_sigma_from_listing_j3(quantization_width, sharpness,
    &rf)` — Listing J.3's per-varblock sigma derivation with the
    `max(1e-4, ..)` clamp; the modular-mode branch uses
    `rf.epf_sigma_for_modular` directly.
  - `is_border_position(x, y)` — Listing J.2's "either coordinate
    of the reference sample is 0 or 7 IMod 8" predicate driving
    the per-pixel `epf_border_sad_mul` selection.
  - `apply_step_5tap(Pass::Pass1 | Pass::Pass2, ..)` — Listing
    J.4's 5-tap cross-shape kernel pass (passes 1 and 2); the
    distance metric is selected by the `Pass` discriminant.
  - `apply_step_13tap(..)` — Listing J.4's 13-tap diamond kernel
    pass 0 (always using `DistanceStep0and1`).
  - `Pass` — enum picking Pass0 / Pass1 / Pass2 for the dispatch.

  §6.5 Mirror1D boundary handling is reused verbatim from
  round-141 `gaborish::mirror1d`. 36 new unit tests + 12 new
  integration tests (`round144_epf`) pin self-distance-is-zero on
  constant planes for both metrics, per-channel-scale linearity,
  offset symmetry for `DistanceStep0and1`, `DistanceStep2` hand-
  derived spatially-varying-plane case
  (`x:1×40 + y:2×5 + b:0×3.5 = 50`), `Weight()` zero-distance
  returns 1.0 / zeroflush cutoff / position-multiplier scaling,
  Listing J.3 sigma at default `rf` sharpness 0 → 1e-4 clamp and
  sharpness 7 → full quant, the `is_border_position` 8×8 grid
  layout, constant-plane invariance across all three passes, and
  the zero-channel-scale collapse to the uniform mean on a centre
  impulse. Lib tests 485 → 521. Pure-math primitive in the same
  shape as round-89 `dct_quant_weights`, round-95 `hf_dequant`,
  round-121 `llf_from_lf`, round-138 `chroma_from_luma`, and
  round-141 `gaborish` — a future round wiring §J.3 into the
  per-frame restoration-filter pipeline can drop these helpers in
  without re-deriving any of the J.1/J.2/J.3/J.4 listings. The
  per-frame loop (calling each pass for each varblock under the
  right `epf_iters` / per-block sigma / position-multiplier
  conditions with output of pass `i` feeding pass `i+1`), the
  `sigma < 0.3` skip-the-block path, and the `epf_iters > 0` skip
  remain caller responsibilities (deferred to follow-up rounds).
  DOCS-GAP observed in FDIS Listing J.1 `DistanceStep2` (free
  `ix`/`iy` variables — adopted `(ix, iy) == (0, 0)`) and Listing
  J.2 `step_multiplier` array (missing comma between
  `epf_pass0_sigma_scale` and `1`); both surfaced in the
  module-level rustdoc with the adopted reading and rationale, and
  the public API sidesteps the indexing ambiguity by accepting
  `step_multiplier: f32` directly so the wiring round can pick the
  resolution without an API churn.

- **Round 141 (2021-FDIS / 2024-spec) — Annex J.2 "Gabor-like
  transform" pure-math primitive (page 85).** New `src/gaborish.rs`
  module transcribes FDIS §J.2 verbatim: given a per-channel plane
  of f32 samples (the output of §I.2.5 LLF/HF reconstruction + the
  round-138 Annex G chroma-from-luma chain) and the per-channel
  `gab_C_weight1` / `gab_C_weight2` weights carried by
  [`frame_header::RestorationFilter`] (Table C.9), the module applies
  the spec's symmetric 3×3 convolution `(centre = 1, edges = w1,
  corners = w2)`, rescaled uniformly so the nine kernel entries
  sum to 1, with §6.5 `Mirror1D` boundary handling on
  out-of-image references. Public API: `mirror1d(coord, size)`
  (Listing 6.1 iterative form), `sample_mirror(plane, w, h, x, y)`
  (direct §6.5 fetch), `gab_kernel(w1, w2) -> [f32; 9]`
  (materialised normalized kernel in row-major order), `apply_channel`
  (out-of-place per-channel convolution with an interior fast path
  + edge-mirror fallback), `apply_channel_in_place` (single-buffer
  scratch convenience), and `apply_xyb_planes_in_place(x, y, b, w,
  h, &rf)` (the three-channel XYB-pipeline convenience using
  `rf.gab_x_weight*` / `gab_y_weight*` / `gab_b_weight*`). 23 new
  unit tests + 10 new integration tests (`round141_gaborish`) pin
  Mirror1D's identity / first-reflection / single-row collapse
  cases, the default-weight kernel sum-to-one and centre-tap
  (`≈ 0.586`) values, the four-edge / four-corner kernel symmetry,
  identity-kernel pass-through, constant-plane invariance, the
  per-channel impulse response on a 3×3 plane, linearity of the
  convolution operator, single-row mirror-collapse, and the
  per-channel dispatch through `apply_xyb_planes_in_place`. Lib
  tests 462 → 485. This is a pure-math primitive in the same shape
  as round-89 `dct_quant_weights`, round-95 `hf_dequant`, round-121
  `llf_from_lf`, and round-138 `chroma_from_luma`: it lands the
  bit-exact arithmetic so a future round wiring §J.2 into the
  per-frame restoration-filter pipeline can drop it in without
  re-deriving the kernel or the mirror semantics. Does NOT
  implement §J.3 (edge-preserving filter) and does NOT honour the
  `rf.gab` skip — both are the caller's responsibility.

- **Round 138 (2021-FDIS / 2024-spec) — Annex G "Chroma from luma"
  pure-math primitive (Listing G.1).** New `src/chroma_from_luma.rs`
  module transcribes FDIS Annex G (page 73) verbatim: given the
  per-frame [`lf_global::LfChannelCorrelation`] bundle (§C.4.4) and,
  for HF coefficients, the per-64×64-tile factor samples from
  [`lf_group::HfMetadata`]'s `x_from_y` / `b_from_y` channels
  (§C.5.4), the module computes the CfL multipliers `(kX, kB)` and
  applies the Listing G.1 reconstruction `X = dX + kX × Y`,
  `B = dB + kB × Y`, `Y = dY` per sample. Public API:
  `kx_kb_raw(base_x, base_b, colour_factor, x_factor, b_factor)`
  (Listing G.1 lines 1-2), `kx_kb_lf(cfl)` (LF derivation
  `x_factor = x_factor_lf - 127`, `b_factor = b_factor_lf - 127`),
  `kx_kb_hf(cfl, x_factor_hf, b_factor_hf)` (HF derivation from the
  64×64-tile factor sample), `apply_sample` / `apply_lf_sample` /
  `apply_hf_sample` for the per-sample reconstruction, and the
  plane-level `apply_lf_plane_inplace(dx, dy, db, cfl)` (constant
  per-frame `(kX, kB)`) + `apply_hf_plane_inplace(dx, dy, db, w, h,
  x_from_y, b_from_y, cfl)` (per-`tile_x=x/64`/`tile_y=y/64`
  lookup, with a per-tile `(kX, kB)` cache). 20 new unit tests + 11
  new integration tests (`round138_chroma_from_luma`) pin the
  default-bundle multipliers (`kX = 1/84`, `kB = 1 + 1/84`), the
  Y-identity line, the round-trip against the encoder-side
  decorrelation `dX = X - kX × Y`, multi-tile HF plane lookup
  (128×64 → 2 tiles wide, 65×65 → 4 tiles via `div_ceil`), and the
  defensive `colour_factor == 0` rejection on both LF and HF paths.
  Lib tests 442 → 462. This is a pure-math primitive in the same
  shape as round-89 `dct_quant_weights`, round-95 `hf_dequant`, and
  round-121 `llf_from_lf`: it lands the bit-exact arithmetic so a
  future round wiring §F.3 + Annex G into the per-LfGroup VarDCT
  pipeline can drop it in without re-deriving any G.1 formulae.
  Does not handle subsampled chroma (Annex G excludes that case
  outright) and does not drive the per-LfGroup loop (deferred).

- **Round 133 (2021-FDIS / 2024-spec) — §C.7.1 `DecodePermutation()`
  for `used_orders != 0`.** `HfPass::read` now handles the
  non-natural coefficient-order path of Listing C.12: the shared
  "8 clustered distributions D" are read once into a
  `modular_fdis::EntropyStream` (`num_dist = 8`) with its ANS state
  initialised, then each set `used_orders` bit runs the §C.3.2
  Lehmer-code permutation against that same stream. New public
  `coeff_order::decode_permutation_from_stream(br, entropy, hybrid,
  size, skip)` factors the §C.3.2 procedure generically (the same
  algorithm the TOC `permuted_toc` path uses); §C.7.1 supplies
  `size = coefficient_count(order)` and `skip = size / 64`, yielding
  `order[i] = natural_coeff_order[nat_ord_perm[i]]`. `HfPass::read`
  no longer returns `Error::Unsupported` for `used_orders != 0`.
  Adds `get_context` + `lehmer_to_permutation` unit coverage and
  rewrites the two former `hf_pass` `Unsupported` tests to assert the
  stream-read path is now taken.

- **Round 129 (2021-FDIS / 2024-spec) — per-varblock LF→LLF
  composition glue (§I.2.5 plumbing).** Three new public functions
  in `vardct` that compose the round-121
  [`llf_from_lf::llf_from_lf`] pure-math step with a single
  channel's dequantised LF samples for a single varblock placement:

  * `vardct::extract_lf_subblock(lf_samples, lf_width, lf_height,
    bx, by, t)` — extracts the `cy × cx` LF sub-block at varblock
    origin `(bx, by)` in row-major order, per FDIS §I.2.5 prose
    "the corresponding X/8 × Y/8 samples from the dequantized LF
    image". Returns `Err(InvalidData)` on dim-mismatch, origin
    overflow, or varblock extending past the LF grid (defensive
    bounds-checking before the indexing).
  * `vardct::compose_lf_to_llf_block(lf_samples, lf_width,
    lf_height, bx, by, t)` — `extract_lf_subblock` + `llf_from_lf`
    in one call, returning the `cy × cx` LLF coefficient block of
    the top-left of an HF varblock.
  * `vardct::compose_lf_to_llf_block_3ch(&LfDequantOutput, bx, by,
    t)` — convenience wrapper that invokes the per-channel helper
    once for each of the three colour channels (X, Y, B) when no
    channel is subsampled (the common case where §F.2 adaptive LF
    smoothing applied); rejects mismatched per-channel dims with a
    clear `InvalidData` message pointing the caller at the
    per-channel `compose_lf_to_llf_block` for the subsampled case.

  24 new tests (15 unit in `src/vardct.rs` + 9 integration in
  `tests/round129_compose_lf_to_llf.rs`). Covers DCT8×8 / DCT16×16
  / DCT32×32 squares, all six DCT16×8-class rectangles (DCT16×8,
  DCT8×16, DCT32×8, DCT8×32, DCT32×16, DCT16×32), the nine non-DCT
  pass-through transforms (Hornuss / DCT2×2 / DCT4×4 / DCT4×8 /
  DCT8×4 / AFV0..AFV3), every kind of out-of-bounds varblock
  placement (x-only, y-only, both, and DCT32×32 at the only
  fitting origin), `LfDequantOutput` subsampling rejection, and
  byte-exact agreement with the hand-derivable `dc * ScaleF(cy,
  bheight, 0) * ScaleF(cx, bwidth, 0)` formula for every
  rectangular transform on a constant input.

  This is the **geometry glue** between rounds 12/13 (per-LfGroup
  LF dequant + smoothing) and rounds 91+/95 (HF coefficient ANS
  decode + HF dequantisation). A future round wiring the §F.x
  pipeline into `decode_codestream` can drop these helpers in as
  the per-varblock loop body without re-deriving any LF→LLF
  geometry or §I.2.5 prose mechanics. Total lib tests: 422 → 437
  (+15); total integration test files: 41 → 42 (+1).

  Round 129 also intentionally **does not** chase the
  `noise-64x64-lossless` sample-194 wp_pred8 = 717 vs spec
  divergence: the trace doc retired 2026-05-06 still has no
  replacement in `docs/image/jpegxl/` per the `project_jpegxl_
  pixel_blocked` memory note (DOCS-GAP unchanged across r126 and
  r129). The deep-trace plumbing from r126 remains the stable
  baseline for the future Specifier round.

- **Round 126 (2021-FDIS) — Self-correcting WP deep-trace plumbing
  + sample-194 hand-derivation against Listings E.1/E.2/E.3.** New
  `WP_DEEP_TRACE` + `WP_DEEP_TRACE_ARMED` thread-locals in
  `modular_fdis` capture the 20-entry intermediate snapshot
  (`subpred[0..4]`, `err_sum[0..4]`, post-shift `weight_shifted[0..4]`,
  `sum_weights_pre`, `log_weight`, `sh`, `sum_weights_post`, `nn8`,
  `ww8`, `pred_pre_clamp`, `clamped_flag`) for the trace-target
  sample. The existing `LEAF_PICK_TRACE_WP` only exposes
  `(te_w, te_n, te_nw, te_ne, w8, n8, nw8, ne8, wp_pred8,
  max_error)` — round 126 fills in the missing nn8/ww8 + Listing
  E.1/E.2/E.3 internals so a by-hand FDIS re-derivation against
  pinned ground-truth is possible.

  New test `tests/r126_wp_intermediates_at_194.rs` (~150 lines,
  2 tests + a docstring with the full hand-derivation). Pins:
  `wp_pred8 = 717` at the `noise-64x64-lossless` sample 194
  (y=3, x=2, channel 0); the 20-entry deep trace; the 3-plane
  first-divergence scan vs `expected.png`. The hand-derivation
  in the module docstring proves that NEITHER the subpred[3]
  sign knob NOR the `s_init - 1` knob (the two FDIS-vs-current
  deviations round 32 swept independently) can produce a
  prediction in `[709..716]` from the captured neighbour state.
  The fix must come from somewhere else — most likely a
  state-evolution bug in `sub_err` or a `WpHeader` parameter
  mismatch. Round 126 also tried the FDIS-literal sub_err
  formula (`abs(((p_i + 3) >> 3) - true_value)` per FDIS line
  6832 vs the legacy `(abs(p_i - tv*8) + 3) >> 3`); the noise
  fixture's `wp_pred8` at sample 194 was unchanged, but the
  synth_320 drift-bisect fixture regressed (first drift moved
  from y=24,x=14 to y=11,x=104), so the change is reverted in
  this round and parked for the docs-collaborator behavioural
  trace promised in `project_jpegxl_pixel_blocked`.

  Net deliverable: deeper diagnostic plumbing + a stable pinned
  baseline for the next round to compare hypotheses against.
  Seven small lossless fixtures + synth_320 baselines untouched;
  the noise fixture's plane[0] first-mismatch boundary remains
  at linear index 194 (`dec=35` vs `exp=34`).

- **Round 121 (2021-FDIS / 2024-spec) — §I.2.5 LLF-from-LF
  pure-math step (Listings I.15 + I.16)**. New `src/llf_from_lf.rs`
  (~500 LOC + 28 unit tests + 16 integration tests in
  `tests/round121_llf_from_lf.rs`) lands the bridge from §F.2's
  dequantised+smoothed LF samples into the top-left LLF coefficient
  block of each HF varblock — the step the trailing prose of
  §F.2 hands off to §I.2.7 (renumbered §I.2.5 in the 2021 FDIS).
  
  Public API: `scale_i8(n, u)`, `scale_d8(n, u)`, `scale_i(n, u)`,
  `scale_d(n, u)`, `scale_c(n_big, n_small, x)`,
  `scale_f(n_big, n_small, x)` (FDIS Listing I.15 closed-form
  helpers); `dct_1d(input) -> Result<Vec<f32>>` (FDIS §I.2.1
  forward 1-D DCT, sizes 1..=32); `dct_2d(samples, rows, cols) ->
  Result<Vec<f32>>` (§I.2.2 Listing I.3 forward 2-D DCT, algorithmic
  inverse of [`idct::idct_2d`]); `llf_dims(t) -> (u32, u32)`
  (LF-block dims per `TransformType`); `llf_from_lf(input, t) ->
  Result<Vec<f32>>` (Listing I.16 verbatim, including the non-DCT
  pass-through cases for Hornuss / DCT2×2 / DCT4×4 / DCT4×8 /
  DCT8×4 / AFV0..3).
  
  44 new tests pin: (a) the Listing I.15 closed forms — I8(8, 0)
  = sqrt(0.5)/2, D8 = 1/(N·I8), the N=8 branch of I/D, C(N, N, x)
  = 1, C reciprocal-on-swap, ScaleF(1, 8, 0) = 1.0 (DCT8×8 corner
  identity), (b) the §I.2.1 1-D forward DCT formula via the
  unit-impulse closed form and the constant-signal DC-only result,
  (c) byte-exact LLF blocks for DCT8×8 (single-cell identity),
  DCT16×16 with both constant-block and impulse-block inputs
  (`out[y·2+x] = 0.25 · SF(2,16,y) · SF(2,16,x)`),
  DCT16×8 / DCT8×16 rectangular paths, DCT32×32 dimension
  contract, and the non-DCT pass-through across all nine
  single-8×8-block transforms.
  
  `dct_2d` ↔ `idct::idct_2d` round-trip verified at 4×4 to f32
  epsilon, confirming the forward DCT is the precise algorithmic
  inverse of the round-12 IDCT.

- **Round 95 (2021-FDIS / 2024-spec) — §F.3 HF dequantisation
  pure-math step**. New `src/hf_dequant.rs` (~310 LOC + 13 unit
  tests) implements the FDIS p. 72 Annex F.3 HF coefficient
  dequantisation formula verbatim: Listing F.2 bias-adjust
  (`*= quant_bias[c]` for `|q| <= 1`, `-= quant_bias_numerator /
  quant` otherwise), per-block `HfMul` multiplier, per-channel
  `0.8^(x_qm_scale - 2)` / `0.8^(b_qm_scale - 2)` factor (Y
  channel exempt), and the §C.6.2 per-`(channel,
  transform_type, coeff_index)` dequant-matrix entry from the
  round-89 `dct_quant_weights::DequantMatrixSet`.

  Public API: `bias_adjust(quant: i32, channel: usize, oim:
  &OpsinInverseMatrix) -> f32`, `QmScaleFactors::for_frame(&FrameHeader)`,
  `QmScaleFactors::for_channel(channel) -> f32`,
  `dequant_hf_coefficient(quant, channel, hf_mul,
  dequant_matrix_entry, oim, qm) -> f32`,
  `dequant_hf_pre_matrix(...)` (partial product helper).

  10 new integration tests
  (`tests/round35_hf_dequant.rs`) pin Listing F.2 branch
  boundaries (zero, ±1, |q|>1 subtractive bias sign-preservation),
  the FDIS default `quant_bias_numerator = 0.145` fixed-point
  `quant=2 → 1.9275`, the `0.8^(u(3) - 2)` exponent sweep, and
  the cross-module composition against
  `materialise_default_dequant_set()` for X / Y channels at the
  DCT8×8 corner cell. Y channel verified to skip the qm-scale
  factor; X channel under default `x_qm_scale = 3` verified to
  pick up a 0.8 factor.

  Made `FrameHeader::default_with` `pub(crate)` (was private) so
  the new `hf_dequant` unit tests can construct a default
  `FrameHeader` without going through bit-stream parsing.

  Round 95 lands the bit-exact F.3 arithmetic so the future
  round that wires the per-block ANS coefficient decode (the
  round-90 followup blocked on the shared 8-cluster ANS stream
  + §C.7.2 histograms) can drop the integer ANS reader on top
  without re-deriving any formulae. CfL (Annex G) and IDCT
  (Annex I.2) still chain afterwards.

- **Round 90 (2021-FDIS / 2024-spec) — HfPass + PassGroup HF
  structural parsers**. Three new modules surface the §C.7.1 /
  §C.7.2 HfPass bundle and the §C.8.3 PassGroup HF entry-points,
  preparing the HF coefficient decode pipeline for the per-block
  ANS-stream wiring scheduled for round 91+.

  New `src/coeff_order.rs` (~430 LOC + 12 tests): §I.2.4 natural
  coefficient ordering for every `OrderId` 0..=12 (Table I.1).
  Builds the `LLF` prefix sorted by `y × bwidth + x` followed by
  the `HF` tail sorted by `(key1, key2)` per Listing I.14. Public
  API: `OrderId`, `varblock_size_for_order`, `natural_coeff_order`,
  `coefficient_count`, `order_id_for_transform`,
  `COEFFICIENTS_PER_ORDER`.

  New `src/hf_pass.rs` (~290 LOC + 7 tests): §C.7.1 Listing C.12
  parser. Reads `used_orders = U32(Val(0x5F), Val(0x13), Val(0),
  Bits(13))`. The `used_orders == 0` fast path materialises all 13
  natural orders directly per the listing's `else` branch.
  `used_orders != 0` returns `Error::Unsupported` — the permutation
  reads need the shared 8-cluster ANS stream that §C.7.2 histograms
  also feed; wiring that shared stream is round-91 work. Exposes
  `num_histogram_distributions = 495 × num_hf_presets ×
  nb_block_ctx` so the next round knows the §C.7.2 read count
  up-front. Also exposes `read_hf_pass_sequence` for the per-pass
  loop.

  New `src/pass_group_hf.rs` (~460 LOC + 18 tests): §C.8.3 first
  line + Listing C.13. Reads `hfp = u(ceil(log2(num_hf_presets)))`,
  validates `hfp < num_hf_presets`, computes
  `histogram_offset = 495 × nb_block_ctx × hfp`. Verbatim
  transcriptions of `block_context`, `non_zeros_context`,
  `coefficient_context`, `predicted_non_zeros`, plus the two
  64-element `CoeffFreqContext` / `CoeffNumNonzeroContext` ladder
  tables as `pub const` arrays. The actual per-block ANS
  coefficient decode loop defers to a later round (it requires the
  shared per-pass ANS stream from §C.7.2).

  New integration suite `tests/round34_hf_pass_pass_group_hf.rs`
  (12 tests) exercises the typed surface end-to-end at the
  structural level — HfPass `used_orders == 0` parse + all 13
  natural orders, §C.8.3 hfp range checks, BlockContext default-
  map paths, NonZerosContext continuity at the
  `predicted == 8` boundary, CoefficientContext with the listed
  ladder constants, PredictedNonZeros four-arm dispatch table.

  Test delta: +49 tests (332 → 381 lib tests; new integration
  suite contributes 12 more). No fixture-level pixel decode
  changes; the seven small lossless fixtures continue to decode
  pixel-correct, and the two committed VarDCT fixtures still hit
  their existing round-13 deferral gate (next round's HF dequant
  + per-block decode flips that gate).

  Spec gap: none new. Listing C.12 / Listing C.13 / Listing I.14
  / Table I.1 are unambiguous on the round-90 contract scope.

  Followups (round 91+): (a) shared per-pass 8-cluster ANS stream
  init, (b) `used_orders != 0` DecodePermutation reads, (c)
  §C.7.2 histogram read (495 × num_hf_presets × nb_block_ctx
  clustered distributions), (d) per-block coefficient decode loop
  per the C.8.3 prose right after Listing C.13, (e) §F.3 HF
  dequantisation gluing the round-89 dequant matrices to the
  newly decoded coefficients.

- **Round 89 (2024-spec) — `GetDCTQuantWeights` + Table I.6 default
  dequantization-matrix materialisation** (parent-dispatch r89). New
  `src/dct_quant_weights.rs` (~1k LOC + tests) transcribes the
  ISO/IEC 18181-1:2024 §I.2.4 / §I.2.5 + Table I.4 + Table I.6
  listing block from page 58-60 of the published core PDF:

  - `mult(v)` — spec `Mult` piecewise function
    (`1+v if v > 0 else 1/(1-v)`).
  - `interpolate(pos, max, bands)` — spec `Interpolate` with the
    2024 corrected `A * pow(B/A, frac_index)` form. Includes
    defensive clamping when `pos == max` (would otherwise index
    past `bands.size() - 1`).
  - `compute_dct_weights(params, x_dim, y_dim)` — spec
    `GetDCTQuantWeights` per the post-typo-fix 2024 listing
    (bands loop closes BEFORE the weights matrix double-loop,
    correcting the FDIS 2021 PDF's nested-loop bug).
  - `materialise_weights_for_dct_select(bundle, channel, X, Y)` —
    per-mode (DCT, DCT4, DCT2, Hornuss, DCT4x8, AFV)
    weights-matrix dispatch per §I.2.4 page 58 prose +
    Listing C.11 for AFV.
  - `materialise_dequant_for_channel(bundle, channel, X, Y)` —
    element-wise reciprocal of the weights matrix per
    §I.2.4 last paragraph. Validates the
    "no non-positive or infinity" spec invariant.
  - `materialise_default_dequant_set()` — the full 17-slot ×
    3-channel default set per Table I.6 (page 60),
    transcribed verbatim including the `SeqA` / `SeqB` /
    `SeqC` abbreviated sequences from the spec footnote and
    the `dct4x4_params` constant for slots 3 (DCT4×4) and 10
    (AFV).
  - `weights_matrix_dims_for_slot(slot)` — Table I.4 page 57
    dimensions lookup (0..=16).
  - `slot_for_transform(t)` — `TransformType` (Table C.16
    0..=26) → Table I.4 slot (0..=16) mapping; multiple
    transforms share a slot (e.g. DCT16×8 and DCT8×16 both
    map to slot 6).

  Test count: 26 new tests (15 unit tests in
  `src/dct_quant_weights.rs` + 11 integration tests in
  `tests/round33_dct_quant_weights.rs`). Every cell of every
  channel of every default slot is verified positive-finite per
  the §I.2.4 invariant. Spot-checks include:
  - DCT8×8 slot 0 channel 0 (0,0) cell = 1 / 3150.0 (reciprocal of
    Table I.6 row-0 head).
  - Hornuss slot 1 (0,0) cell = 1.0 (spec sets weights(0,0) = 1).
  - AFV slot 10 8×8 fully populated (Listing C.11 covers all 64
    cells across the freqs interpolation + weights4x8 + weights4x4
    fills).

  Spec-listing typo notes (recorded in module doc-comment):
  - FDIS 2021 PDF Listing C.10 has the `for (y, x) { ... }`
    weights double-loop INSIDE the `for (i = 1; i < len; i++)`
    bands loop — would compute the matrix `len - 1` times. The
    2024 published edition (`docs/image/jpegxl/
    ISO_IEC_18181-1-JPEG-XL-Core-2024.pdf` page 58) corrects this.
    Module follows the 2024 form.
  - 2024 `Interpolate` drops `len` (uses `bands.size()`) and
    writes `pow(B / A, frac_index)` instead of FDIS 2021's
    `A * (B / A)^frac_index`. Mathematically identical.

  SPECGAP recorded: DCT2 cell (0, 0) is not assigned by the spec
  listing block (page 58). Implementation fills it with
  `params(c, 0)` (same value used for `i == 0` neighbours) so the
  dequant reciprocal is finite. The 6-rectangle assignments cover
  62 of 64 cells, plus (1, 1); (0, 0) is the only unmentioned
  position. Recommend a spec clarification.

  Unblocks: downstream HF coefficient dequantisation per §F.3 on
  the HfGlobal `u(1) == 1` default-encoding fast path. The
  non-default branch's `RAW` encoding mode still requires a
  modular sub-bitstream decode (deferred to round 90+ alongside
  the §F.3 wiring).

  Spec citations: ISO/IEC 18181-1:2024 page 58 (Listing for
  `Interpolate` / `Mult` / `GetDCTQuantWeights`), page 59
  (Listing C.11 AFV weights + per-mode prose), page 60
  (Table I.6 default matrix parameters), page 57 (Table I.4
  weights-matrix dimensions). Cross-referenced against ISO/IEC
  FDIS 18181-1:2021 PDF (extractable) Listing C.10 / Table C.18
  / Table C.20 (the 2021 equivalents).

  Fixture count remains 7 pixel-correct lossless small fixtures
  (no change — round 89 is upstream of the pixel-decode flow;
  HfGlobal default-encoding parsing remains unchanged in
  behaviour).

- **Round 77 (2024-spec) — animation-3frame SPECDIFF audit + docs
  citation.** Two new audit-grade integration tests
  (`tests/r77_animation_3frame_specdiff.rs`) characterise the
  `docs/image/jpegxl/fixtures/animation-3frame/input.jxl` fixture
  (cjxl 0.12.0, 78 B, 3 RGB Regular Modular frames of 32×32 with
  `have_animation = 1`). The probe-level path is correct
  (`probe_fdis` recovers SizeHeader + ImageMetadata with
  `have_animation = true` + AnimationHeader populated); the
  decode-level path remains blocked on a real spec-edition split
  between ISO/IEC 18181-1:**2021** FDIS Table C.9 (which our
  `RestorationFilter::read` follows; no leading `all_default`
  field) and the published **2024** Table J.1 (which prepends an
  `all_default Bool()` to the bundle plus a `u(32)` "(ignored)"
  field after `epf_channel_scale`). Bit-trace bisect (recorded in
  the test file's module docs):
  - The two-bit RF SPECDIFF lifts our FrameHeader bit count from
    39 to 40 for the animation fixture, which lets `permuted_toc
    + pu0` correctly land the TOC entry U32 at byte 11 of the
    codestream; that read yields `entry value = 16`, matching the
    libjxl trace's `total_bytes = 16`.
  - The seven currently-pixel-correct lossless fixtures were
    encoded by cjxl 0.11.1 against the 2021 FDIS layout and do
    NOT include the leading `all_default` bit; landing the
    2024-Table-J.1 fix straightforwardly breaks
    `alpha-64x64.jxl`. The audit recommendation (recorded in the
    test docs) is to re-encode the seven fixtures with cjxl
    0.12.0+ before applying the 2024-spec fix uniformly. This is
    a docs-collaborator follow-up — there is no codestream-level
    edition tag, so a single-pass parser cannot dispatch between
    the two RF layouts without a heuristic.
  - Spec citations: ISO/IEC 18181-1:2024 Table J.1
    (`docs/image/jpegxl/ISO_IEC_18181-1-JPEG-XL-Core-2024.pdf`
    page 70) and ISO/IEC FDIS 18181-1:2021 Table C.9
    (pdftotext-extractable lines 4088-4101). Trace fixture at
    `docs/image/jpegxl/fixtures/animation-3frame/trace.txt`.

  Fixture count remains 7 pixel-correct lossless small fixtures
  (no change). Test count grows by 2 (audit harness).

### Changed

- **Round 32 (2024-spec) — `noise-64x64-lossless` pixel-divergence
  bisected to the Self-correcting weighted predictor at the first
  `y >= 2, x >= 2` sample whose `predictor == 6`; root cause
  localised, fix deferred pending a libjxl-trace doc that this
  workspace does not yet ship.** The fixture count therefore stays
  at 7 pixel-correct lossless fixtures (status quo). No source-file
  semantic changes this round; the diagnostic harness used to
  bisect was removed before commit and the regression set remains
  green.

  Round 31 left the noise fixture as a "decodes without EOF, but
  pixels diverge from expected.png starting at plane[0] sample
  194" follow-up. Round 32 reproduced that divergence and pinned
  it down further:

  - The first divergence is at plane[0] (y=3, x=2) — the FIRST
    sample whose predictor is `6 (Self-correcting)` and which has
    the full set of WP neighbours `N, W, NW, NE, NN, WW` populated
    (i.e. `y >= 2 && x >= 2`). The prior `predictor == 6` samples
    in rows `y = 0` and `y = 1` all decoded pixel-correct because
    their WP path takes the `NN does not exist → NN = N`
    fall-back. Two `predictor == 6` samples on row `y = 2` also
    decoded correctly because `WW = W` was used (the bug requires
    `WW ≠ W`, i.e. `x >= 2`).
  - At sample 194 the WP machinery produces `wp_pred8 = 717`
    (Listing E.3 weighted sum). The spec rounding `(wp_pred8 + 3)
    >> 3` then yields `p = 90`, giving `v = diff + p = -55 + 90
    = 35` — but `expected.png` says `34`. So `wp_pred8` is 1 too
    high modulo the rounding (any value in `[709..716]` would give
    `p = 89` and thence `v = 34`). The MA-tree leaf, the decoded
    token, the diff `-55`, and `wp_max_error` all match what the
    neighbour state legitimately implies — the discrepancy is
    purely in the WP weighted sum.
  - Bisected against `WP_ROUND_BIAS ∈ {0..=7}`, `s_init ∈
    {(sum_weights >> 1) - 1, (sum_weights >> 1), sum_weights, 0}`,
    the `subpred[3]` sign (FDIS `N + …` vs. round-3 code `N - …`),
    and the clamp condition (`<= 0` vs `>= 0`). Every alternative
    either re-introduces an EARLIER divergence (samples 68, 79,
    142) on the noise fixture, OR breaks one of the seven
    currently-pixel-correct lossless fixtures. So the bug is NOT
    in any of the dimensions our spec text exposes a knob for.
  - Suspected residual root cause: a subtle interaction between
    the FDIS `error2weight` formula's outer `>> shift` step (only
    in the 2024 published edition and the round-3 code; absent
    from FDIS 2021 literal text), the four sub-predictor weights,
    and the final `s × ((1 << 24) Idiv sum_weights) >> 24`
    division. Most likely the libjxl reference uses an `s_init`
    formula that depends on the **shifted** vs **unshifted**
    `sum_weights` in a way the FDIS spec text does not disclose.
    Resolving this needs either (a) a behavioural trace of the
    libjxl WP path on the noise fixture at sample 194 captured by
    the docs collaborator, or (b) the docs collaborator's
    promised `docs/image/jpegxl/libjxl-trace-reverse-engineering.md`
    section on §H.5.2 Sub-predictions (referenced in the
    `project_jpegxl_pixel_blocked` memory note, but the file does
    not yet exist in `docs/image/jpegxl/`).

  Round-32 scope therefore closes with the bisect finding above
  recorded and the regression set green. No `.gitignore` / Cargo
  changes; no API surface deltas. The §F.3 zero-pad fix from
  round 31 stays in place and `noise-64x64-lossless` continues to
  decode-complete (just with non-byte-exact pixels).

  Spec citations: FDIS Annex E.1 (Sub-predictions, Listing E.1),
  E.2 (Prediction weights, Listing E.2), E.3 (Prediction, Listing
  E.3), and Table H.3 row `predictor == 6` (`(prediction + 3)
  >> 3`).

### Added

- **Round 31 (2024-spec) — §F.3 zero-pad uniformly applied to the
  single-TOC-entry LfGlobal fast path; noise-64x64-lossless now
  decodes without EOF** (parent-dispatch "r16" option A). One
  narrow `src/lib.rs::decode_codestream` delta:

  - Pre-round-31, when `num_groups == 1 && passes == 1 &&
    toc.entries.len() == 1`, the decoder routed `LfGlobal::read`
    through the non-padding main `BitReader` (`pad_eof_with_zeros
    == false`). The other LfGlobal path already used
    `BitReader::new_section` (which implements FDIS §F.3's
    section-bit-budget + zero-pad rule). For six of the seven small
    lossless fixtures the entire LfGlobal section had enough
    trailing slack that the read never touched the padding region;
    `noise-64x64-lossless` (`cjxl -d 0 -e 7`, 64×64 high-entropy RGB
    Modular, MA tree `nodes=167 leaves=84`) does NOT — its
    per-pixel ANS / hybrid-uint refill loop on the final samples
    reaches a few bits past the byte budget that the spec says must
    read as zero. Pre-round-31 the non-padding reader errored
    instead → `InvalidData("unexpected end of JXL bitstream")`.

  - The fix collapses both LfGlobal-read branches into one path
    that always uses `BitReader::new_section` against the
    `toc`-declared section byte range. This makes the single-section
    fast path bit-for-bit equivalent to the multi-section path on
    its real-data prefix, and applies §F.3 zero-pad uniformly.

  Spec citation: FDIS §F.3 first paragraph — "When decoding a
  section, no more bits are read from the codestream than 8 times
  the byte size indicated in the TOC; if fewer bits are read, then
  the remaining bits of the section all have the value zero."

  Test added: `tests/r31_noise_lossless.rs` with two cases —
  `noise_64x64_lossless_decodes_without_eof_error` (locks the
  shape of the post-fix `VideoFrame`: 3 RGB planes, stride=64,
  data.len()=4096 each) and `pre_round31_seven_lossless_fixtures_
  still_decode` (regression sentinel: the seven pre-round-31
  fixtures all decode successfully under the unified path).
  Committed fixture pair under `tests/fixtures/`:
  `noise_64x64_lossless.jxl` (13 505 B) +
  `noise_64x64_lossless_expected.png` (12 505 B, 8-bit RGB PNG).

  Known limitation NOT fixed this round: while
  `noise-64x64-lossless` now decode-completes (vs hard-EOF), the
  produced pixels are not yet byte-identical to `expected.png`.
  The first divergence is plane[0] (R) at (2, 3) — i.e. samples
  0..193 of plane 0 match, and from sample 194 on ~98 % of samples
  diverge. The divergence point is deterministic and well within
  the section's real-byte budget, so the §F.3 fix is independent
  of the residual pixel-divergence. Suspected root cause: a
  latent state-evolution bug in either the MA-tree leaf decode
  with `num_contexts > 16` (the leaf-stream `EntropyStream`'s
  cluster_map is 84 → 3 clusters here, vs ≤ 6 → ≤ 4 in every
  other lossless fixture), the Self-correcting WP state on
  high-entropy neighbour history, or the hybrid-uint extra-bits
  path for large `n_extra` values. Deferred to round 32 — needs
  the round-24-style per-cluster trace replayed against the
  cleanroom Python reference at ~30 distinct bit positions across
  the 108 kbit symbol stream.

  Docs gap noted: `docs/image/jpegxl-cleanroom/reference-impl/`
  (referenced in the round-31 brief as the place to bisect
  against) does not yet exist; the round-30 deferral note pointed
  at it as a future bisect target. The §F.3 fix landed without
  needing it — pure spec-text bisect against FDIS §F.3 was
  sufficient. The reference-impl directory would still be useful
  for the residual pixel-divergence bisect; ask the docs
  collaborator to populate it for round 32.

- **Round 30 (2024-spec) — bit-depth-16 RGB pixel-correct decode +
  16-bit LE plane-pack convention** (parent-dispatch "r15" option A).
  Lifts the fixture count from 6 to 7 by adding `bit-depth-16`
  (`docs/image/jpegxl/fixtures/bit-depth-16/input.jxl`, 421 B,
  64×64 RGB lossless Modular at `bits_per_sample = 16`) and
  documents the wider-than-8-bit pack convention forced on us by
  `oxideav-core` 0.1.x's bit-depth-less `VideoPlane`.

  Two narrow `src/lib.rs::decode_codestream` deltas:

  1. **Bit-depth gate widened.** The pre-round-30 hard reject
     `metadata.bit_depth.bits_per_sample != 8` now accepts
     `bps ∈ 1..=16`. The XYB and YCbCr branches (FDIS Annex L.2.2 /
     L.3) still hard-require `bps == 8` because their dequantisation
     lattice is calibrated against the 8-bit output range — a
     specific `Error::Unsupported("jxl decoder (round 30): XYB
     high-bit-depth (bps={...}) deferred")` now precedes the
     transform call. Float (`float_sample == true`) and `bps > 16`
     remain unsupported.

  2. **Pass-through plane pack dispatches on `bps`.** The previous
     loop unconditionally clamped each `i32` sample to `[0, 255]`
     and pushed one byte per sample with `stride == width`. The
     new loop:
     - `bps ≤ 8` — unchanged: 1 byte/sample, `stride == width`,
       sample clamped to `[0, 2^bps - 1]`.
     - `9 ≤ bps ≤ 16` — 2 bytes/sample **little-endian**,
       `stride == width × 2`, sample clamped to `[0, 2^bps - 1]`,
       packed via `u16::to_le_bytes`.

     The LE-pack choice is documented in
     `crates/oxideav-jpegxl/README.md` under "Plane byte layout"
     (new section) so that downstream consumers (`cli-convert` /
     etc.) know how to reinterpret a wide plane as `&[u16]`. PNG's
     RFC 2083 §2.1 ships big-endian 16-bit samples; we deliberately
     pick LE so a `bytemuck::cast_slice::<u8, u16>` on a
     little-endian host is a zero-cost view (vs forcing a per-sample
     swap).

  Test count: `tests/round30_bit_depth_16.rs` adds 3 tests
  (`bit_depth_16_rgb_pixel_correct_vs_expected_png` — full 64×64×3
  16-bit byte-for-byte match against the committed
  `bit_depth_16_expected.png`;
  `bit_depth_16_le_pack_convention_self_consistent` — invariant
  check on stride/length/round-trip;
  `pre_round30_8bit_fixtures_still_byte_packed` — regression
  sentinel for the four pre-existing 8-bit byte-packed fixtures).
  Committed fixture pair under `tests/fixtures/`:
  `bit_depth_16.jxl` (421 B) + `bit_depth_16_expected.png`
  (375 B, 16-bit RGB PNG).

  Cross-checked against `djxl v0.11.1` as a black-box oracle (PPM
  output → byteswap BE→LE → byte-equal to our planes). Crate now
  decodes 7 small lossless Modular fixtures pixel-correct vs
  `expected.png` (was 6): pixel-1x1, gray-64x64,
  gradient-64x64-lossless, palette-32x32, grey_8x8_lossless,
  alpha-64x64, **bit-depth-16**.

  Spec citations: FDIS Annex A.6 + Table A.22
  (`bit_depth.bits_per_sample` bundle), Annex G.1.3 (Modular
  channel-order rule — colour channels share the global
  `bits_per_sample`, no per-channel bit-depth split for kModular
  RGB), PNG RFC 2083 §2.1 (PNG ships 16-bit big-endian, so our
  reference-PNG read uses `u16::from_be_bytes`).

  Docs gaps identified probing adjacent fixtures during round 30:
  `noise-64x64-lossless` (13.5 KB, `nodes=167 leaves=84` per
  trace.txt) still fails inside `LfGlobal::read` with "unexpected
  end of JXL bitstream" — large MA-tree decode path likely
  mis-computes a hybrid-uint extra-bits count for a high-context
  leaf; deferred to round 31. `vardct-256x256-d1` / `d3` and
  `noise-feature-256x256` fixtures all hit independent VarDCT
  pipeline gaps and are unrelated to round 30.

- **Round 29 (2024-spec) — alpha-64x64 RGBA pixel-correct decode +
  ISOBMFF signature-strip fix** (parent-dispatch "r14" option A).
  Two narrow lib-level fixes in `src/lib.rs::decode_one_frame` /
  `decode_codestream` unblock the docs cleanroom `alpha-64x64`
  4-channel Modular lossless fixture (`docs/image/jpegxl/fixtures/
  alpha-64x64/input.jxl`, 86 B) for pixel-exact decode against the
  committed `expected.png` (8-bit RGBA, 64×64):

  1. **ISOBMFF `FF 0A` strip.** The jxlc/jxlp box payload IS a JXL
     codestream and therefore begins with the 2-byte `FF 0A`
     codestream signature (FDIS Annex B.1). The RawCodestream branch
     already stripped those 2 bytes before handing off to
     `decode_codestream`; the ISOBMFF branch did NOT. The result was
     a 16-bit misalignment at the `SizeHeader::read` parse that
     cascaded into apparently-unrelated downstream failures
     (`bit-depth-16` tripped `JXL permutation: LZ77-enabled TOC
     sub-stream not supported` because the TOC `permuted` flag bit
     parsed as 1 instead of 0). Now the ISOBMFF branch validates the
     `FF 0A` prefix and strips it symmetric with the raw path. A new
     unit test wraps `gradient-64x64-lossless` in a minimal ISOBMFF
     (signature + ftyp + jxlc) and asserts plane-by-plane equality
     vs. the raw decode (`tests/round29_alpha_rgba_pixel.rs::
     isobmff_wraps_raw_codestream_decodes_identically`).

  2. **Extra-channel mapping.** The post-Modular channel-count check
     `n_chans != expected_chans` rejected RGBA Modular frames
     because the Modular decoder lays out colour and extra channels
     in a flat array of length `expected_chans + num_extra_channels`
     (FDIS Annex G.1.3 colour-then-extras channel-order rule). The
     check now also accepts the with-extras length and emits a
     trailing VideoFrame plane per extra channel. For
     `alpha-64x64` this maps directly to 4 RGBA planes; for
     hypothetical multi-extra fixtures (depth, spot colour, …) the
     same path extends N-ways. The XYB-encoded / YCbCr branches are
     unchanged — those still require exactly 3 colour channels and
     fall through if extras are present (round-30+ work).

  Test count: `tests/round29_alpha_rgba_pixel.rs` adds 3 tests
  (`alpha_64x64_rgba_pixel_correct_vs_expected_png` — full 64×64×4
  byte-for-byte match; `five_pre_round29_fixtures_still_pass` —
  regression sentinel for pixel-1x1 / gray-64x64 / gradient-64x64 /
  palette-32x32 / grey_8x8_lossless; `isobmff_wraps_raw_codestream_
  decodes_identically` — synthetic ISOBMFF wrap of
  gradient-64x64). Committed fixture pair under `tests/fixtures/`:
  `alpha_64x64.jxl` (86 B) + `alpha_64x64_expected.png` (283 B).

  Crate now decodes 6 small lossless Modular fixtures pixel-correct
  vs `expected.png` (was 5): pixel-1x1, gray-64x64,
  gradient-64x64-lossless, palette-32x32, grey_8x8_lossless,
  **alpha-64x64**.

  Spec citations: FDIS Annex B.1 (codestream signature),
  Annex G.1.3 (channel order), Annex A.6 + A.9 + Table A.22
  (ImageMetadata + ExtraChannelInfo).

  Docs gaps identified probing adjacent fixtures: `bit-depth-16`
  (421 B) reaches the 8-bit-only post-Modular check (decoder needs
  a 16-bit output-pack path before VideoFrame mapping — deferred);
  `noise-64x64-lossless` (13.5 KB) fails inside LfGlobal with
  "unexpected end of JXL bitstream" suggesting the high-entropy
  random-RGB MA tree exercises a code path not yet covered
  (deferred).

- **Round 28 (2024-spec) — non-DCT IDCT helpers** (parent-dispatch
  "r13" item 3). Extends `src/idct.rs` with five new public helpers
  that complete the IDCT surface for the non-DCT TransformType
  variants:

  - `aux_idct_2x2(block, S)` — Annex I.9.3 Hadamard-style butterfly on
    the top-left `S × S` cells of an 8×8 buffer (`S ∈ {1, 2, 4, 8}`).
  - `idct_dct2x2(coefficients)` — Annex I.9.3 closing recipe (chained
    `aux_idct_2x2` calls at S=2, 4, 8).
  - `idct_dct4x4(coefficients)` — Annex I.9.4: per-2×2-quadrant 4×4
    IDCT_2D over interleaved coefficient cells with a DC patch from
    `aux_idct_2x2(coefficients, 2)`.
  - `idct_hornuss(coefficients)` — Annex I.9.5: per-quadrant
    block-LF + residual-sum centre cell + neighbour-fill + corner
    corrective.
  - `idct_dct8x4(coefficients)` — Annex I.9.6: column-major Hadamard
    pair into two 4×8 (rows × cols) IDCT_2D halves tiled into rows
    `[0..4)` and `[4..8)` of the 8×8 output.
  - `idct_dct4x8(coefficients)` — Annex I.9.7: dual of `dct8x4`,
    row-major Hadamard pair into two 4×8 halves tiled by row.

  `idct_for_transform(t, coefficients)` now dispatches `Hornuss`,
  `Dct2x2`, `Dct4x4`, `Dct8x4`, `Dct4x8` to the dedicated helpers in
  addition to the 18 plain-DCT variants from r12. `Afv0..Afv3` continue
  to return `Err(Unsupported)` pending an independently verified
  256-entry `AFVBasis` table (deferred to a later round to avoid a
  high-risk OCR transcription).

  New helper `non_dct_pixel_dims(t)` returns `Some((8, 8))` for the
  nine non-DCT TransformType variants and `None` for plain-DCT — the
  output of all five new helpers is always an 8×8 row-major buffer
  (length 64), matching the closing entries of Listings I.9.3..I.9.8.

  Test count: lib `idct::tests` 36 → 57 (+21 new — 8 covering
  `aux_idct_2x2` validation/butterfly/preserve/DC, 6 covering DC-only
  + per-quadrant correctness for the five helpers, 5 covering length
  validation, 2 covering `non_dct_pixel_dims`); integration tests
  +5 in new `tests/round13_non_dct_idct.rs` plus 1 updated
  assertion in `tests/round12_idct_dispatch.rs` (renamed
  `idct_for_transform_non_dct_transforms_return_unsupported` →
  `idct_for_transform_afv_only_unsupported_after_round_13`,
  reflecting that only the AFV variants remain unsupported).

  Spec-gap notes inline in the module documentation enumerate the OCR
  transcription work deferred for AFVBasis.

- **Round 27 (2024-spec) — IDCT dispatch** (parent-dispatch "r12" item
  5). New `src/idct.rs` (~470 LOC including tests) wires the
  spec-conformant 1-D inverse DCT (FDIS Annex I.2.1) for power-of-two
  sizes `s ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256}` and the 2-D inverse
  DCT (Annex I.2.2 Listing I.4) handling rectangular `R × C` blocks.

  Three public entry points: `idct_1d(input)` for the bare 1-D form,
  `idct_2d(coefficients, output_rows, output_cols)` for the 2-D form
  taking coefficients in the spec's `(short × long)` row-major natural-
  ordering layout (Annex I.2.4) and returning samples in `(R × C)`
  row-major, and `idct_for_transform(t, coefficients)` which dispatches
  on a `dct_select::TransformType` to the appropriate 2-D IDCT for the
  18 plain-DCT transform types in Table C.16 (DCT8x8, DCT16x16,
  DCT32x32, DCT16x8, DCT8x16, DCT32x8, DCT8x32, DCT32x16, DCT16x32,
  DCT64x64, DCT64x32, DCT32x64, DCT128x128, DCT128x64, DCT64x128,
  DCT256x256, DCT256x128, DCT128x256). The 9 non-DCT transforms
  (Hornuss, DCT2x2, DCT4x4, DCT4x8, DCT8x4, AFV0..AFV3) — Listings
  I.7..I.13 — return `Err(Unsupported)` and are deferred to round 13+.

  Companion helper `dct_pixel_dims(t)` returns the `(rows, cols)`
  output shape for plain-DCT TransformType variants and `None` for the
  non-DCT transforms.

  31 lib unit tests in `idct::tests` (1-D length validation, DC-only
  consistency for all 9 supported sizes, 1-D round-trip via private
  forward DCT oracle for sizes 8/16/32/64, 1-D AC[1] hand-computed
  spec-formula reference, 2-D length / shape validation, 2-D DC-only
  consistency for 12 DCT block sizes, 2-D round-trip via 2-D forward
  oracle for 8x8/16x8/8x16/16x16/32x32, dispatch validation for
  DCT8x8/16x16/32x32/8x16/16x8 + every non-DCT TransformType returning
  Unsupported, dct_pixel_dims completeness for both branches); 5
  integration tests in `tests/round12_idct_dispatch.rs` (1-D DC-only
  for all sizes, 2-D DC-only for every plain-DCT block size,
  Unsupported sentinel for every non-DCT transform, 2-D round-trip for
  asymmetric 8x16 and 16x8 via inline forward oracle, five-fixture
  Modular regression sentinel). Total test count 345 → 381 (+36 net).

  No new fixture coverage — the IDCT lands as a callable primitive that
  round 13's PassGroup HF coefficient decode + F.3 dequantisation will
  feed. The legacy `vardct::idct1d_8` and `vardct::idct2d_8x8` (round 8
  scaffold, scaled-orthonormal IDCT) are kept untouched for backward
  compatibility but are NOT spec-conformant; new HF-decode wiring will
  call through `idct::idct_for_transform` exclusively.

- **Round 26 (2024-spec) — Annex L colour transforms** (parent-dispatch
  "r11"). New `src/xyb.rs` (~210 LOC) transcribes FDIS §L.2.2 inverse
  XYB → linear RGB and §L.3 inverse YCbCr → RGB verbatim from the
  ISO/IEC 18181-1:2024 spec text. Three public entry points:
  `inverse_xyb_to_rgb(x, y, b, oim, tone_mapping)`,
  `inverse_ycbcr_to_rgb(cb, y, cr)`, and the convenience composite
  `modular_xyb_to_linear_rgb(y_prime, x_prime, b_prime, lf_dequant,
  oim, tone_mapping)` which folds in the §L.2.2 preamble step
  (`X = X' * m_x_lf_unscaled`, `Y = Y' * m_y_lf_unscaled`,
  `B = (B' + Y') * m_b_lf_unscaled`). Helper `linear_rgb_to_u8`
  clamps + rounds the linear `[0, 1]` output to 8 bits.

  Wired into `decode_codestream` modular output stage: when
  `metadata.xyb_encoded == true` AND `colour_encoding.colour_space ==
  Rgb` (3 colour channels), the per-channel pass-through is replaced
  with `build_rgb_planes_from_xyb` which walks every pixel through
  the inverse transform. Symmetric `build_rgb_planes_from_ycbcr`
  branch handles `frame_header.do_ycbcr == true`. The original
  pass-through path is preserved for the common case
  (xyb_encoded=false AND do_ycbcr=false) so all five small lossless
  fixtures continue to pixel-correct decode.

  9 unit tests in `xyb::tests` (DC zero-input, spec-listing
  hand-computed reproduction, intensity_target linear scaling,
  modular preamble multiplier check, YCbCr neutral / red-dominant,
  linear→u8 clamping, X-sign-flip symmetry); 6 integration tests
  in `tests/round11_xyb_inverse.rs` (forward-→-inverse round-trip
  for neutral grey AND saturated red using a hand-computed Cramer's-
  rule matrix inversion of `oim.inv_mat`, YCbCr neutral, u8
  quantisation reference values, end-to-end zero-input modular wrapper,
  and five-fixture pass-through regression sentinel). Total test count
  345 → 362 (+17 net: 9 lib + 6 integration + 2 from earlier round-21
  recount).

  No fixture decoded that didn't decode before — round 11 lays the
  colour-transform foundation, but no modular-XYB or modular-YCbCr
  fixture is currently committed (cjxl encodes photo-content XYB
  inputs as VarDCT by default; the rare modular-XYB path needs a
  hand-built minimal trace, deferred to round 12+ or a docs-
  collaborator commission). The two committed VarDCT fixtures
  (`vardct_256x256_d1.jxl`, `vardct_256x256_d3.jxl`) still terminate
  at the round-13 "round 14+: HF subband decode + IDCT not yet wired"
  Unsupported.

  SPECGAP documented in `xyb::linear_rgb_to_u8` doc comment: §L.2.2
  outputs linear-domain RGB (NOTE in spec) but the spec doesn't
  prescribe a gamma encoding step before display — strict conformance
  defers gamma application to a downstream colour-management consumer.
  The crate emits linear bytes (clamp + scale by 255 + round); spec
  callers needing sRGB-encoded bytes should apply sRGB transfer
  themselves.

  Wall respected: spec PDF (Annex L pages 82-84 read directly), no
  external library source consulted, no `libjxl-trace-reverse-
  engineering.md` (retired). OpsinInverseMatrix defaults already
  transcribed in `metadata_fdis::OpsinInverseMatrix::default()`
  (round-2) from FDIS Table L.1 independently; the new module
  consumes those constants without re-reading the table. Test count
  362, fmt + clippy clean against 1.95 toolchain.

- **Round 24 (2024-spec, Auditor mode)** — pursued round-23 candidates
  (1) per-cluster ANS distribution byte-trace for clusters 0+1 and
  (2) per-call alias-mapping invariant audit. Result: **both paths
  falsified**. Cluster 0 (19 nonzero entries) and cluster 1 (23
  nonzero entries) both sum to 4096; the alias table built from each
  D[] routes probability mass to symbols identically to the declared
  D[] (per-symbol routed-mass divergence = 0 for both clusters);
  across the FULL 3072-call ANS trace the spec C.3.2
  `(symbol, offset) = AliasMapping(state & 0xFFF)` invariant holds
  bit-for-bit when checked against either cluster 0 or cluster 1's
  alias table (0 hard violations; 288 ambiguous calls where both
  clusters yield the same `(symbol, offset, prob)`). Per-call state
  arithmetic `state = prob * (state >> 12) + offset` also reproduces
  the trace exactly. Cluster usage breakdown: c0=1755 calls,
  c1=1317 calls, unknown=0 (no cross-talk into HFMetadata clusters
  2/3/4). The d1 ANS final-state delta of `0x21914271 -
  0x00130000 ≈ 562M` is therefore NOT caused by a per-cluster D[]
  shape mismatch, alias-table self-map / Vose-pump bug,
  alias-mapping lookup bug, per-call state-arithmetic bug, or
  cluster-routing leakage. Round 25 candidates: (1) D[]-vs-cjxl
  reference comparison (a single mismatched count would be the
  smoking gun), (2) leaf-pick + cluster-routing audit at samples
  beyond sample 22 up to sample 79 (where r23's first ctx-flip was
  observed), (3) HFMetadata stream-boundary cross-talk audit. New
  diagnostic `tests/round24_d1_disttrace.rs` (Auditor mode, never
  asserts) with two tests:
  `d1_per_cluster_distribution_byte_trace_round_24` (path 1) and
  `d1_per_call_alias_mapping_invariant_round_24` (path 2). Full
  audit notes in `crates/oxideav-jpegxl/round24-d1-disttrace.md`.
  Test count 343 → 345 (+2).

- **Round 22 (2024-spec, Auditor mode)** — pursued round-21 candidates
  (a) `lf_quant` first-256-sample dump per channel and (c) WP `(p+3)>>3`
  rounding bias toggle on the d1 `LfCoefficients` sub-bitstream. Result:
  WP-rounding-bias bug class **falsified**. Added a runtime atomic
  `WP_ROUND_BIAS` (default 3, spec-conformant per ISO/IEC 18181-1:2024
  Table H.3 + FDIS-2021 Listing C.16) so the auditor can sweep biases
  without recompile. Sweeps recorded post-decode ANS final state for
  bias ∈ {0, 3, 4, 7}: 0 → 0x0042cd42 (|Δ|=3 132 738), 3 → 0x21914271
  (|Δ|=561 922 673, spec), 4 → 0x00fd721e (|Δ|=15 364 638), 7 →
  0x001214ac (|Δ|=60 244). All four miss the §D.3.3 sentinel
  `0x00130000`; the +7 bias being closest proves the variation is
  ANS-chain noise from leaf-flip cascades, not a true rounding bug.
  Per-channel `lf_quant` dump (Y'/X'/B', 1024 samples each, 32×32) shows
  smooth low-frequency shape with sane stats (Y' mean=468 min=326
  max=644; X' mean=14 min=−125 max=135; B' mean=41 min=−49 max=123),
  consistent with a real-image fixture and **proving the per-sample
  decode loop is producing plausible data — not garbage**. WP+3 vs +4
  diverges first at Y' sample 22 (row 0, col 22), localising the actual
  bug to a specific MA-tree leaf-flip at that sample. New diagnostic
  `tests/round22_d1_sample_dump.rs` (Auditor mode, never asserts) dumps
  both the `lf_quant` table and the bias-sweep final states; full audit
  notes in `crates/oxideav-jpegxl/round22-d1-sampledump.md`. Test count
  337 → 338 (+1).

- **Round 21 (2024-spec, Auditor mode)** — pursued round-20 candidates
  (1) per-cluster distribution decode bisect and (2) alias-table
  self-map branch audit on the d1 `LfCoefficients` sub-bitstream.
  Result: both paths falsified. The 5 per-cluster ANS distributions
  (clusters 0..4) all sum to 4096 with sane shapes (cluster sizes
  19/23/5/2/2 nonzero entries out of 64); cluster 1's full 64-entry
  alias table reconciles with the round-19 bit-faithful trace at calls
  #0 and #1. Critically, **none of the five clusters has any `D[i] ==
  bucket_size` entry**, so the alias-table self-map branch (round-3
  fix territory) is not triggered for d1. Documented one strict-spec
  divergence in `AliasTable::build` (`else` vs spec's `else if
  (cutoffs[i] < bucket_size)`) that has zero observational effect on
  d1 — hand-tracing the equal-bucket path confirms output-equivalent
  behaviour. New diagnostic `tests/round21_d1_dist_alias_dump.rs`
  (Auditor mode, never asserts) captures per-cluster `(cfg, D, alias)`
  triples + cluster-1 full alias dump as evidence; full bisect notes
  in `crates/oxideav-jpegxl/round21-d1-distbisect.md`. Test count
  336 → 337 (+1).

- **Round 20 (2024-spec, Auditor mode)** — re-interpreted cjxl
  `JXL_TRACE` output's `bits_consumed` field as section-local (not
  cumulative file position), invalidating the round-17/18/19 claim of a
  267-bit overshoot in `LfCoefficients`. Empirical proof: in the same
  trace, `AC_GLOBAL_END bits_consumed=307` while `DC_GLOBAL_END=1026`,
  so `307 < 1026` precludes a cumulative reading. With the corrected
  interpretation `DC_GROUP` is 12754 bits (not 11728), `LfCoefficients`
  fits well within the budget, and `HfMetadata`'s slot is 759 bits.

  Identified a stronger oracle for the actual divergence: per FDIS
  D.3.3, the ANS state must equal `0x00130000` after the final symbol
  in any stream. Wired `LATEST_ANS_STATE` / `LATEST_ANS_CALL_COUNT`
  thread-locals (in `src/ans/symbol.rs`) so a test can read the
  post-decode state without holding the per-stream `MaTreeFdis` clone.
  On d1's `LfCoefficients` the final state is `0x21914271` after 3072
  decode_symbol calls — proving a structural decode divergence (wrong
  per-cluster distribution, wrong alias mapping, wrong sample count, or
  wrong read in the per-sample loop). The state never reaches the
  sentinel within 3072 calls, so it's not a sample-count off-by-one.

  Lifted the previous 30-call cap on `STATE_TRACE_BUF` so end-of-stream
  bisects over multi-thousand-sample LF channels are tractable. Five
  new tests in `tests/round20_d1_*.rs`. See
  `crates/oxideav-jpegxl/round20-d1-hfmeta.md` for the full audit and
  the round-21 candidate ranking.

- **Round 19 (2024-spec, Auditor mode)** — extended the per-token
  trace ring with `(ctx, cluster, ans_refill_bits)` and added a
  `STATE_TRACE_BUF` recording the first 30 ANS state transitions for
  spot-checking against raw codestream bits. New
  `AnsDecoder::decode_symbol_with_refill` reports refill-bit cost. New
  `tests/round19_d1_cluster.rs` drives d1 LfCoefficients under the
  extended trace and emits per-cluster / per-ctx histograms plus a
  diagnostic eprintln on the leaf-stream `EntropyStream::read` prelude
  bit count. Findings: prelude is bit-exact (602 bits matching cjxl's
  `num_contexts=16 num_histograms=5 log_alpha_size=6`), cluster_map is
  bit-exact (16 → 5 distinct clusters), state transitions are
  bit-faithful to raw codestream. The 267-bit overshoot remains
  unexplained; deferred to round 20 with cjxl `--debug` per-call
  bit-position trace as the proposed next-step. See
  `crates/oxideav-jpegxl/round19-d1-cluster.md` for the full audit.

## [0.0.9](https://github.com/OxideAV/oxideav-jpegxl/compare/v0.0.8...v0.0.9) - 2026-05-08

### Other

- round-17 (Auditor mode) against ISO/IEC 18181-1:2024 — d1 bit-position-drift bisect
- round-16 against ISO/IEC 18181-1:2024 — HfMetadata nested transforms (FDIS §C.5.4 + §C.9.4)
- round-15 against ISO/IEC 18181-1:2024 — GlobalModular zero-channel ModularHeader gating + single-TOC-entry section chaining (unblocks d1 past LfGlobal)
- round-14 against ISO/IEC 18181-1:2024 — HfBlockContext custom branch + HfGlobal §I.2.4 dequant-matrix encoding-modes parse
- round-13 against ISO/IEC 18181-1:2024 — DctSelect derivation + HfGlobal + VarDCT pipeline wiring
- round-12 against ISO/IEC 18181-1:2024 — F.1 LF dequant + F.2 adaptive smoothing + G.2.4 HfMetadata
- round-11 against ISO/IEC 18181-1:2024 — LF subband decode (Annex G.2.2 / I.2 / FDIS C.5.3)
- round-10 against ISO/IEC 18181-1:2024 — synth_320 drift bisected to PG[0][0] decode #3087 + C.3.3 lz_dist_ctx spec fix
- round-9 against ISO/IEC 18181-1:2024 — synth_320 0-byte PassGroup blocker resolved via three concurrent fixes
- round-8 against ISO/IEC 18181-1:2024 — C.2.5 SPECGAP partial resolution + VarDCT scaffold
- round-7 against ISO/IEC 18181-1:2024 — four-piece refactor wiring multi-group decode infrastructure (Annex G.1.3 + G.4.2)
- round-6 against ISO/IEC 18181-1:2024 — Annex E.4 ICC profile decode + LfGroup/PassGroup type scaffolding
- round-5 against ISO/IEC 18181-1:2024 — RFC 7932 §3.5 Kraft early-stop fix; grey_8x8_lossless pixel-correct
- round-4 against ISO/IEC 18181-1:2024 — three independent decoder bugs fixed; gradient + palette + gray pixel-correct vs expected.png
- round-3 against ISO/IEC 18181-1:2024 — bit-alignment + alias-mapping fixes
- copy docs fixtures into tests/fixtures/ for CI self-containment
- round-2 against ISO/IEC 18181-1:2024 — inverse transforms + WP predictor
- round-1 against ISO/IEC 18181-1:2024 — Modular pixel decode end-to-end
- clippy 1.95: unusual_byte_groupings + vec_init_then_push fixes

### Added

- **Round 18 (2024-spec, Auditor mode)** — per-token bit accounting
  trace inside `HybridUintConfig::read_uint` (gated behind a public
  `TRACE_ENABLED` atomic switch in `src/ans/hybrid_config.rs`) and
  `tests/round18_d1_per_token.rs` exercising it on the d1 LfCoefficients
  decode. The trace records `(split_exponent, msb_in_token,
  lsb_in_token, token, n_extra_bits, value)` per call so that future
  rounds can pinpoint the still-open 267-bit drift documented in
  `round17-d1-bisect.md`.

  Findings (full analysis in `round18-d1-per-token.md`):

  - All 3072 LfCoefficients sample decodes hit a single hybrid-uint
    config `(split_exp=4, msb=1, lsb=2)` with **821 extra-bits** total
    (avg 0.267 / call) — well within the spec's expected per-token
    accounting per FDIS Listing D.6, which **rules out the round-17
    PRIMARY hypothesis** (per-token extra-bits drift).
  - The remaining 11104 sample-loop bits decompose into 32 (ANS state
    init) + 16 × 694 (ANS refills). 22.6 % refill rate is plausible
    per-symbol but high in aggregate: the bug is in **ANS state
    evolution**, not extra-bits or the prelude (the cjxl-traced
    `bits=602` leaf-stream prelude bound is satisfied — our
    GlobalModular ends at the cjxl-expected bit 1026 exactly).
  - A trial revert of the round-3 conditional alias-mapping deviation
    (returning `pos` instead of `offsets[i] + pos` in the not-in-redirect
    branch) reduces d1's LfCoefficients consumption from 11 995 →
    11 654 bits (within 74 of cjxl's 11 728 LfGroup TOTAL) but breaks
    `gray-64x64.jxl` with `unexpected end of JXL bitstream`. Analytical
    proof in the bisect doc shows the deviation IS correct against the
    encoder for both fixtures, so the bug is elsewhere — round-19 should
    extend the trace with cluster-index per call to verify whether the
    cluster_map for the leaf-level stream is being computed correctly.
  - All 5 small lossless fixtures + every round-11..17 sentinel test
    stays green. New `d1_per_token_trace_round_18` test joins the
    existing 329-test suite (now 330 tests, +1 net).

- **Round 17 (2024-spec, Auditor mode)** — d1 bit-position-drift bisect.
  Round 16 left the d1 fixture surfacing
  `InvalidData("JXL Modular Squeeze: end 40 >= channel count 4")`
  and hypothesised an upstream bit-position drift in LfGlobal or
  LfCoefficients. Round 17 confirms the drift via a step-by-step
  bit-cursor walk through the LfGlobal/LfGroup decode, captured by the
  new `tests/round17_d1_bit_trace.rs` diagnostic test.

  Findings (full analysis in `round17-d1-bisect.md`):

  - Our `LfGlobal::read` ends at codestream-relative bit **1026**, which
    matches the cjxl ground-truth trace at
    `docs/image/jpegxl/fixtures/vardct-256x256-d1/trace.txt`
    (DC_GLOBAL_END=1026) **exactly**. LfGlobal is NOT the drift site.
  - Our `LfCoefficients::read` consumes **11995 bits** for 3072 LF
    samples — but the cjxl trace says the entire LfGroup bundle (=
    LfCoefficients + ModularLfGroup + HfMetadata) is **11728 bits**
    (DC_GROUP_END=12754). LfCoefficients alone is 267 bits **over** the
    whole LfGroup budget, which means the per-channel decode is reading
    ~2.3 bits more per sample than the spec demands.
  - The decoded LF coefficient values look plausible (smooth gradient
    in ch0, small chroma variation in ch1/ch2), suggesting the entropy
    decoder produces "real" tokens but consumes too many trailing
    extra bits per token.
  - Round-16 hypothesis ranked HfBlockContext custom branch HIGH; round
    17 RULES THAT OUT (HfBlockContext consumed 87 bits for the smallest
    legal custom path, and LfGlobal ended at the cjxl-expected bit
    boundary).

  **Round-18 candidate** (deferred, not landed in r17):
  `crates/oxideav-jpegxl/src/modular_fdis.rs::decode_uint_in_with_dist`
  hybrid-uint extra-bits accounting on the global-tree-reused leaf
  entropy stream. Either `HybridUintConfig` is mis-read in
  `EntropyStream::read` (prelude bug) or a stray post-token
  `u(extra_bits)` is being read on the wrong gate
  (per-token bug).

  No code-path fix landed in round 17 (Auditor mode: ship diagnostic
  evidence + r18 candidate only). Test count: 328 → 329 (+1: new
  d1 bit-trace diagnostic). Five small lossless fixtures + round-11..16
  sentinels remain green.

- **Round 16 (2024-spec)** — HfMetadata nested transforms (FDIS §C.5.4
  + §C.9.4) — the four-channel HfMetadata sub-bitstream now parses
  `nb_transforms` + `TransformInfo[]` and applies the inverse
  transforms in reverse bitstream order to recover the four-channel
  base layout `[XFromY, BFromY, BlockInfo, Sharpness]`.

  Round 15 closed two stacked bugs (GlobalModular ModularHeader N=0
  gate + single-TOC-entry section chaining), exposing the round-12
  HfMetadata deferral on the d1 fixture: `nb_transforms > 0` errored
  out as `"transforms inside HF metadata sub-bitstream not yet
  supported (round 13+)"`. Round 16 wires the parse:

  - `HfMetadata::read` now takes the `metadata: &ImageMetadataFdis`
    bundle (forwarded from `LfGroup::read`) so the inverse Palette
    transform can read `bit_depth.bits_per_sample` for delta-palette
    prediction.
  - The four-channel HfMetadata baseline is fed through
    `apply_transforms_to_channel_layout` (mirroring
    `GlobalModular::read`) so the inner per-channel decode operates on
    the post-transform list.
  - After `decode_channels_at_stream`, `apply_inverse_transforms` is
    invoked with the same `transforms` list so RCT / Palette / Squeeze
    are undone and the four-channel baseline is recovered. The decoded
    `nb_blocks` and per-channel widths/heights are validated against
    the §C.5.4 baseline before being returned.

  Acceptance: the d1 (`vardct_256x256_d1.jxl`) fixture now reaches a
  strictly-later blocker — its HfMetadata sub-bitstream emits an
  explicit Squeeze whose `SqueezeParam.begin_c` references channels
  beyond the four-channel baseline (`begin_c=39` on the very first
  step), and `apply_transforms_to_channel_layout`'s
  `begin_c + num_c <= channel_count` invariant fires with
  `Error::InvalidData("JXL Modular Squeeze: end 40 >= channel count
  4")`. That's the round-17 candidate to investigate (suspected
  upstream bit-position drift in LfGlobal or LfCoefficients). Round-16
  sentinel test (`round16_hfmeta_transforms.rs`) asserts the d1
  progression and the five small lossless fixtures stay
  regression-free.

- **Round 15 (2024-spec)** — GlobalModular zero-channel ModularHeader
  gating (FDIS §C.9.1 last sentence) + single-TOC-entry section chaining
  for the VarDCT pipeline. Unblocks the d1 fixture past the LfGlobal
  boundary.

  Round-14 left the d1 (`vardct_256x256_d1.jxl`) fixture stuck on
  `JXL TransformId: invalid value 3`. Round-15 root-causes + fixes two
  consecutive bugs:

  1. **GlobalModular ModularHeader gating** (`global_modular` module) —
     `GlobalModular::read` was unconditionally reading the inner
     ModularHeader (`use_global_tree`, `WPHeader`, `nb_transforms`,
     `TransformInfo[]`) even when the channel count was zero.
     Bit-position trace of d1 confirmed the libjxl reference decoder
     ends LfGlobal at the bit where our code starts reading
     `inner_use_global_tree` — i.e. the entire ModularHeader is gated
     by `N > 0` per FDIS §C.9.1 ("In the trivial case where N is zero,
     the decoder takes no action."). Fix: skip the inner ModularHeader
     when `derive_channel_descs` returns an empty list (the typical
     VarDCT-without-extras case).

  2. **Single-TOC-entry section chaining** (`decode_vardct_round13`) —
     when `num_groups == 1 && num_passes == 1`, F.3.1 says the TOC has
     a single entry containing all sections concatenated bit-aligned
     without byte alignment between them. `decode_vardct_round13` was
     slicing each TOC slot into its own byte range, which only works
     for multi-entry TOCs. Fix: when `toc.entries.len() == 1`, chain
     `LfGlobal::read` → `LfGroup::read` → `HfGlobal::read` on a
     shared `BitReader`.

  Acceptance: `vardct_256x256_d1.jxl` now reaches the HfMetadata
  transforms-inside-HF-metadata round-13+ deferral message instead of
  failing in LfGlobal. Round-15 sentinel test
  (`round15_d1_past_global_modular.rs`) asserts the d1 progression and
  the five small lossless fixtures stay regression-free.

- **Round 14 (2024-spec)** — HfBlockContext non-default-table branch
  (§I.2.2 custom encoding) + HfGlobal §I.2.4 dequant-matrix
  `encoding_mode` parse (Listing C.10 / Table I.5).

  Two pre-flight pieces for round-15+ HF coefficient decode:

  1. **HfBlockContext non-default branch** (`lf_global` module) —
     `u(1) == 0` now drives:
     - per-channel `nb_lf_thr[i] = u(4)` followed by
       `nb_lf_thr[i]` thresholds via
       `t = UnpackSigned(ReadThreshold())` where
       `ReadThreshold = U32(u(4), 16+u(8), 272+u(16), 65808+u(32))`,
     - `nb_qf_thr = u(4)` followed by `qf_thresholds[i] = 1 + U32(u(2), 4+u(3), 12+u(5), 44+u(8))`,
     - `bsize = 39 * (nb_qf_thr+1) * Π (nb_lf_thr[i]+1)` with the
       spec invariant `bsize ≤ 39 * 64`,
     - `block_ctx_map = ReadBlockCtxMap()` — re-uses the existing
       C.2.2 clustering decoder with `num_dist = bsize`; `bsize == 1`
       short-circuits to `[0]` (no bits read) per C.2.2's `num_dist == 1`
       skip rule. `num_clusters ≤ 16` invariant enforced.
     The `vardct_256x256_d1.jxl` fixture progresses past LfGlobal as
     a result.

  2. **HfGlobal C.6.2 dequant-matrix non-default-encoding parse**
     (`hf_global` module) — `u(1) == 0` now drives 17 sets of:
     `encoding_mode = u(3)` validated against Table I.5's per-slot
     valid-index list, then per-mode parameters per Listing C.10:
     - **Library (0)** — no params.
     - **Hornuss (1)** — 3×3 F16 matrix, all elements ×64.
     - **DCT2 (2)** — 3×6 F16 matrix, all elements ×64.
     - **DCT4 (3)** — 3×2 F16 matrix (col 0 ×64) + `ReadDctParams()`.
     - **DCT4x8 (4)** — 3×1 F16 matrix + `ReadDctParams()`.
     - **AFV (5)** — 3×9 F16 matrix (cols 0..5 ×64) + 2× `ReadDctParams()`
       (the second is the `dct4x4_params`).
     - **DCT (6)** — `ReadDctParams()` only.
     - **RAW (7)** — defers to round 15+ (modular sub-bitstream of
       quant-matrix shape requires the IDCT consumer to define the
       Table H.4 stream_index).
     `ReadDctParams()` reads `num_params = u(4) + 1`, then a 3×num_params
     F16 matrix with col-0 ×64.

  Acceptance: 5 new unit tests for HfBlockContext + 6 new for HfGlobal,
  plus `tests/round14_hf_global_dequant.rs` with 3 integration tests
  asserting the d1 fixture is past the HfBlockContext blocker. Round 11
  + 12 + 13 sentinels remain green; 5 small lossless fixtures still
  decode.

- **Round 13 (2024-spec)** — DctSelect / HfMul derivation from
  BlockInfo (FDIS C.5.4 prose + Table C.16) + HfGlobal default-fast-
  path (C.6) + VarDCT pipeline wiring of round-12's F.1 LF dequant +
  F.2 adaptive smoothing.

  Three pieces tighten the VarDCT decode path so round-12's
  unit-tested F.1 / F.2 work actually runs on real codestreams:

  1. **DctSelect / HfMul derivation** (`dct_select` module) — walks
     each column of the per-LfGroup `BlockInfo` channel decoded in
     round 12, looks up the transform type in Table C.16's 27-entry
     table, and places the varblock at the next-empty 8×8 cell of
     the LfGroup's block grid (raster order, top-left first as per
     C.5.4 prose). `HfMul = 1 + mul` is computed and stored at the
     varblock top-left only. Continuation cells track the interior
     of multi-block varblocks.

  2. **HfGlobal C.6 default-fast-path** (`hf_global` module) — reads
     the `u(1)` dequant-default flag (when `1`, all 17 matrix slots
     take their default encoding from C.6.3) and the
     `num_hf_presets - 1 = u(ceil(log2(num_groups)))` field per
     C.6.4. The non-default-encoding branch (per-matrix
     `encoding_mode = u(3)` + Listing C.7 `ReadDctParams()`) returns
     `Error::Unsupported` until round 14+.

  3. **VarDCT pipeline wiring** (`decode_vardct_round13` in
     `lib.rs`) — the top-level `decode_one_frame` no longer rejects
     VarDCT codestreams at the round-8 scaffold gate. Instead, for
     `num_lf_groups == 1 && num_passes == 1`, it now drives:
     LfGlobal → LfGroup (LfCoefficients + HfMetadata) → DctSelect
     derivation → HfGlobal → F.1 LF dequantisation (Listing F.1
     `mXDC = m_x_lf_unscaled / (global_scale × quant_lf)` with
     `1 << extra_precision` divide) → F.2 adaptive smoothing (when
     `kSkipAdaptiveLFSmoothing` is clear and no channel is
     subsampled). The pipeline returns `Error::Unsupported` with a
     "round 14+: HF subband decode + IDCT not yet wired" message
     AFTER all round-12 work has run on the real input.

  Acceptance: 25 new unit tests covering Table C.16 indexing +
  block_dims, DctSelect placement scenarios (DCT8×8, DCT16×16,
  DCT32×32, DCT8×16, mixed grids, overflow, underflow), HfGlobal
  default-fast-path with various `num_groups`, and 5 round-13
  integration tests including round-trip parsing of two real
  cjxl-encoded VarDCT fixtures (`vardct_256x256_d1.jxl`,
  `vardct_256x256_d3.jxl`, copied in-tree from
  `docs/image/jpegxl/fixtures/`). Both VarDCT fixtures now reach the
  round-13 pipeline (no longer hit the round-8 scaffold gate). All 5
  small lossless Modular fixtures stay regression-free.

- **Round 11 (2024-spec)** — LF subband decode (Annex G.2.2 / I.2 /
  FDIS C.5.3).

  Three pieces wire the LF subband path:

  1. **LfGlobal VarDCT bundles** — `Quantizer` (§C.4.3:
     `global_scale` + `quant_lf` U32 fields driving Listing C.1's
     `mXDC = m_x_lf_unscaled / (global_scale × quant_lf)`),
     `LfChannelCorrelation` (§C.4.4: `colour_factor`,
     `base_correlation_x`, `base_correlation_b`, `x_factor_lf`,
     `b_factor_lf`) and `HfBlockContext` (§C.8.4 default-table
     fast path: `u(1) == 1` → 39-element default `block_ctx_map`,
     `nb_block_ctx = 15`). The non-default-table HfBlockContext
     branch (per-LF/qf thresholds + clustering map) is round-12+.

  2. **GlobalModular zero-channel acceptance** — `GlobalModular::read`
     now accepts the empty-`descs` case (the common VarDCT path
     without extra channels), consuming the inner `ModularHeader`
     (`use_global_tree`, `WPHeader`, `nb_transforms`) but skipping
     the MA-tree + per-cluster distribution decode per FDIS C.9.1
     last sentence. New `MaTreeFdis::empty_shell` constructor.

  3. **LfGroup + LfCoefficients** — `LfCoefficients::read` reads
     `extra_precision = u(2)`, builds a 3-channel `ChannelDesc`
     list of dims `ceil(group_w/8) × ceil(group_h/8)` (optionally
     right-shifted by `frame_header.jpeg_upsampling[c]` per channel),
     and drives `decode_channels_at_stream` with `stream_index =
     1 + lf_group_index` per Table H.4. `LfGroup::read` composes
     ModularLfGroup (G.2.3 — empty-channel-list case only in
     round 11) with LfCoefficients. HfMetadata (G.2.4) still defers.

  Acceptance fixture: a hand-built minimal VarDCT bitstream — no
  cjxl dependency, encoded directly from spec listings — covering
  an 8×8 frame with 1×1 LF coefficient channels, MA tree of one
  Zero-predictor leaf, prefix-code symbol stream with
  `alphabet_size=1` per cluster (so every decoded LF coefficient
  is 0). The fixture parses through `LfGlobal::read` →
  `LfGroup::read` → `LfCoefficients::read` end-to-end. Test:
  `lf_group::tests::round11_lfgroup_minimal_vardct_one_block_parses`.

  Five small lossless modular fixtures (pixel_1x1, gray_64x64,
  gradient_64x64, palette_32x32, grey_8x8) remain pixel-correct
  vs `expected.png` (sentinel: `tests/round11_lf_subband.rs`).

  **Not yet wired** (round-12+ candidates, in dependency order):
  Listing F.1 LF dequant (multiply by `mXDC/mYDC/mBDC`, divide by
  `1 << extra_precision`); adaptive LF smoothing (FDIS F.2);
  HfMetadata (G.2.4: `nb_blocks` + XFromY/BFromY/BlockInfo/
  Sharpness modular sub-bitstream + DctSelect/HfMul reconstruction);
  HfGlobal HfPass[num_passes] (Annex G.3 Table G.4); PassGroup HF
  (G.4.3: clustered ANS over 495 × num_hf_presets × nb_block_ctx
  distributions, coefficient order, per-block dequant); inverse
  DCT dispatch across non-8×8 block sizes (16×8, 8×16, 16×16,
  32×32, 64×64, DCT4, DCT8×4, IDENTITY, AFV — only 8×8 is wired);
  Chroma-from-Luma (Annex G); Gaborish smoothing
  (RestorationFilter.gab_*); EPF (RestorationFilter.epf_*).

- **Round 10 (2024-spec)** — synth_320 edge-group drift bisection
  + LZ77 distance-context spec-conformance fix.

  **First-mismatch bisect** — instrumented per-decode tracing of the
  `synth_320` PG[0][0] sub-bitstream pinpoints the divergence at
  decode #3087 (frame coords y=24, x=14). State 0x9CA780 alias-maps
  to symbol 30 (a low-prob entry: `D[30] = 1` of the cluster-0 ANS
  distribution). The decode forces a state refill plus extra bits,
  consuming 21 more bits than were available in the 9-byte
  `PassGroup[0][0]` slot — falling into §F.3 zero-padded territory
  and producing a garbage token (192) instead of the encoded
  literal. djxl's bit-correct decode of the same fixture stays
  within the 9-byte slot, so our state evolution must diverge from
  djxl's somewhere between decodes #1 and #3087. Per-group decode
  log + per-group transform layout + ANS state init are all
  verified spec-correct. Diagnostic data captured: cluster-0 dist
  has nz=`[(0, 4092), (2, 1), (27, 1), (30, 1), (32, 1)]`,
  cluster-1 dist has nz=`[(2, 4090), (14, 2), (17, 4)]`,
  `log_alphabet_size=6` (table_size 64), tree node[0] decides on
  `property[15] > -3`. None of the obvious round-10 root-cause
  candidates match the symptom: LZ77 is not enabled in the symbol
  stream (so `lz_dist_ctx` cannot be the culprit; `dist_multiplier`
  for PG[0][0] is `128` per H.3 and unused without LZ77); WP per-
  channel state is reset per group (since PG[0][0] is the first
  group, this is moot for the immediate symptom); per-group
  transform layout is empty for PG[0][0] (only edge groups carry
  transforms); channel index threading is identical between
  GlobalModular and per-PassGroup paths. Round-11 will need a
  finer-grained bisect — most likely a state-by-state diff against
  djxl's `--debug` mode (gated on building djxl from source, which
  is forbidden in the implementer round; deferring to an Auditor
  round) or an alternative reference like the JPEG XL conformance
  test suite's lossless-grey traces.

  **C.1 + C.3.3 `lz_dist_ctx` correction** — per the spec, when
  `lz77.enabled` the codestream sets `lz_dist_ctx = num_dist++`
  (one extra context reserved AT THE END of the cluster mapping)
  and the LZ77 distance token in `DecodeHybridVarLenUint`'s LZ77
  branch is read against `D[clusters[lz_dist_ctx]]` — i.e. the
  dedicated last context, not the same per-symbol leaf context as
  the literal token. Round 9's `decode_uint_in` and
  `decode_uint_in_with_dist` passed the leaf context for both the
  literal token and the LZ77 distance token, which is a
  spec-conformance bug that would distort every LZ77 copy
  whenever an encoder emits one. Fixed by deriving
  `lz_dist_ctx = cluster_map.len() - 1` when `lz77.enabled` and
  threading it to `HybridUintState::decode`'s `ctx_lz` parameter.
  No fixture change for synth_320 (its symbol stream uses
  `lz77.enabled=false`); the fix is forward-looking for fixtures
  that DO trigger LZ77.

  **Status** — synth_320 still decodes to ~21k of 102400 pixels
  matching the expected `(y + x) & 0xFF` gradient (the first 24
  rows of PG[0][0] and PG[0][1] are pixel-correct, then drift
  starts at exactly y=24, x=14 where state 0x9CA780 hits low-
  prob symbol 30). All five small lossless fixtures still pixel-
  correct (255 tests pass).

- **Round 9 (2024-spec)** — synth_320 0-byte PassGroup blocker
  resolved via two underlying fixes plus per-group transforms support.

  **§F.3.1 unconditional HfGlobal slot fix** — the 2024 spec lists
  `HfGlobal` UNCONDITIONALLY in the TOC bullet list (not gated on
  `encoding == kVarDCT`); per NOTE 1, the slot is empty (0-byte) for
  `encoding == kModular`. Round 8's `num_toc_entries` /
  `Toc::read` skipped HfGlobal for kModular, off-by-oning every
  PassGroup index in multi-group kModular frames. The synth_320
  fixture (320×320 grey, num_groups=9) actually has 12 TOC entries
  (1 LfGlobal + 1 LfGroup + 1 HfGlobal + 9 PassGroup), not 11; the
  apparent "0-byte PassGroup[0][0]" was the HfGlobal slot reading.
  Also: `HfPass[num_passes]` is part of the `HfGlobal` section per
  Annex G.3 Table G.4 — it does NOT contribute additional TOC
  entries (round 8 had counted both, double-incorrect).

  **§F.3 first-paragraph zero-padding sub-reader** — "When decoding
  a section, no more bits are read from the codestream than 8 times
  the byte size indicated in the TOC; if fewer bits are read, then
  the remaining bits of the section all have the value zero." Round
  8's `BitReader` errored on EOF for section sub-readers, breaking
  PassGroup ANS decodes whose modular sub-bitstream legitimately
  consumes fewer real bits than the section's byte size (the missing
  bits are guaranteed by the spec to be zero). Added
  `BitReader::new_section` which pads EOF reads with zero values for
  per-TOC-section sub-readers (LfGlobal / LfGroup / HfGlobal /
  PassGroup); the legacy `BitReader::new` keeps the strict EOF for
  whole-codestream parsing so malformed top-level structures still
  error early.

  **Per-PassGroup transforms (Annex H.6 inside G.4.2)** — observed
  in cjxl 0.11.1's synth_320 edge groups: the encoder emits a
  per-group Palette transform (`begin_c=0, num_c=1, nb_colours=191`)
  for the 64-pixel-wide column-2 / row-2 groups, which is
  spec-legal per Table H.1 (every modular sub-bitstream has its own
  `transform[nb_transforms]` field). `decode_modular_group_into`
  now applies the transform layout adjustment to the per-group
  channel descs, decodes against the adjusted descs, and applies
  the inverse transforms LOCALLY before copying samples back into
  the parent image. `apply_transforms_to_channel_layout` is now
  `pub` so the per-group reuse path doesn't duplicate the table.
  A new `tests/round9_synth_320_toc.rs` integration test confirms
  the TOC layout is parsed correctly (12 entries, slot 2 is
  HfGlobal not PG[0][0]) and that the first 6 rows of the first
  two group columns decode pixel-for-pixel against the expected
  `(y + x) & 0xFF` gradient.

  **Status** — synth_320 reaches end-of-frame without erroring and
  about 21k of 102400 pixels match the expected gradient; the
  remaining ~80k pixels drift mid-decode in the smaller edge groups
  (PG[0][2,5,6,7,8] = 64-pixel-wide / 64-pixel-tall sections).
  Suspected residual issue: ANS state nuance specific to the F.3
  zero-padded tail or per-group WP / property bookkeeping that
  doesn't surface against the round-4 small fixtures (single-group,
  single-channel, no padding pressure on the ANS state). Full
  pixel-correctness is round-10 work.

- **Round 8 (2024-spec)** — two themes: round-7 SPECGAP partial
  resolution + VarDCT scaffolding.

  **Theme 1: ANS distribution C.2.5 SPECGAP (interpretation C, partial)**
  - `src/ans/distribution.rs` — `read_distribution` now returns
    `(D, log_eff)` instead of just `D`; `log_eff` is the effective
    log_alphabet_size for downstream alias-table sizing. For the
    common case (alphabet_size <= table_size) `log_eff` equals the
    signalled `log_alphabet_size`. For the SPECGAP case
    (alphabet_size > table_size), the logcounts loop iterates
    `min(alphabet_size, table_size)` entries; the encoder's
    advertised wider alphabet is treated as a soft cap because
    empirically cjxl 0.11.1 only serialises `table_size` per-symbol
    entries. Interpretations A (grow D to a power-of-2 >=
    alphabet_size) and B (drop writes at i >= table_size) were both
    tried and rejected — see the module-level docstring on
    `read_distribution` for the full rationale.
  - `src/ans/cluster.rs`, `src/modular_fdis.rs`, `src/toc.rs` —
    callers updated to consume the `(D, log_eff)` tuple and pass
    `log_eff` to `AliasTable::build`.
  - The synth_320 fixture's LfGlobal section now parses cleanly
    past the round-7 SPECGAP error, but PassGroup decode is blocked
    at a separate post-LfGlobal blocker (cjxl emits a 0-byte
    PassGroup[0][0] slot which contradicts the spec's per-group
    "all groups carry data per pass" rule). That secondary blocker
    is round-9+ work; the synth_320 fixture is left in
    `tests/fixtures/synth_320_grey/` unconsumed by tests pending
    that round.

  **Theme 2: VarDCT scaffolding**
  - New `src/vardct.rs` module: structural recognition of a
    VarDCT-encoded codestream + IDCT-II primitives for the smallest
    block size (8×8). `recognise_vardct_codestream(fh, metadata)`
    validates the round-8 envelope (single LF group, single pass,
    no extra channels, Grey/RGB colour) and returns a
    `VarDctScaffold` geometry record. `idct1d_8` and `idct2d_8x8`
    implement the spec's inverse DCT-II formula directly (O(N²),
    audit-friendly; faster Lee-style decompositions land alongside
    LF/HF subband decode in round 9+).
  - `src/lib.rs` — `decode_codestream`'s encoding gate now special-
    cases `Encoding::VarDct` to invoke
    `vardct::recognise_vardct_codestream` and emit a VarDCT-specific
    `Error::Unsupported` message rather than the generic round-7
    one.
  - End-to-end VarDCT pixel decode (LF subband decode, HF subband
    decode, dequant, inverse transform dispatch across block sizes
    8×8/8×16/16×8/16×16/32×32/64×64/DCT4/DCT8/IDENTITY/AFV,
    Chroma-from-Luma, Gaborish smoothing, EPF) is round-9+ work.

  **Tests**
  - `tests/round8_vardct_scaffold.rs` — verifies the 5 small
    lossless fixtures still pixel-correct (regression sentinel
    against the `(D, log_eff)` tuple refactor) plus VarDCT
    primitive sanity checks.
  - `src/ans/distribution.rs` — new
    `branch3_alphabet_size_above_table_size_is_truncated` sentinel
    test for the SPECGAP truncation behaviour.

- **Round 7 (2024-spec)** — four-piece refactor wiring the GlobalModular
  partial-decode path to per-PassGroup decode + post-PassGroup inverse
  transforms (Annex G.1.3 last paragraph + G.4.2). The orchestration
  is in place; pixel-correct decode of the committed multi-group
  fixture is blocked at a documented spec-vs-reference SPECGAP (cjxl
  0.11.1's multi-group ANS streams emit `alphabet_size > table_size`
  for log_alpha=5, which the spec text in C.2.5 implies should be
  rejected). Round-8 will resolve the SPECGAP once docs collaborator
  clarifies the alphabet cap.
  - **`src/global_modular.rs`** — `GlobalModular::read` now obeys
    G.1.3's "stops decoding at channels exceeding `group_dim`" rule.
    Channels too large for GlobalModular are zero-filled placeholders
    and `fully_decoded = false`; the bundle stashes
    `nb_meta_channels`, `transforms`, and `global_tree` for the
    per-PassGroup decode to consume. New
    `apply_inverse_transforms(image, transforms, bit_depth)` is the
    transform pass that the multi-group path invokes AFTER all
    PassGroups complete (G.4.2 last paragraph).
  - **`src/modular_fdis.rs`** — new public
    `decode_channels_at_stream(br, descs, tree, wp, stream_index)`
    threads the Table H.4 stream-index property through the channel-
    decode loop (the legacy `decode_channels` is a thin wrapper that
    passes `stream_index = 0`). `MaTreeFdis::cloned_with_fresh_state`
    lets per-section sub-bitstreams reuse the global tree's static
    shape + clustered distributions while reading a fresh ANS state
    init for each section (per H.2's "global MA tree and its clustered
    distributions are used as decoded from the GlobalModular section").
    `MaTreeFdis`, `EntropyStream`, `ClusterEntropy`, `HybridUintState`,
    `AnsDecoder` all gain `Clone`.
  - **`src/pass_group.rs`** —
    `decode_modular_group_into(br, fh, lf_global, pass_idx, group_idx)`
    decodes one PassGroup's modular sub-bitstream. The contributing-
    channel filter implements G.4.2's criterion (channel exceeds
    group_dim, hshift<3 OR vshift<3, minshift<=min(hshift,vshift)<
    maxshift, not already decoded). The decoded samples are copied
    back into `lf_global.global_modular.image` at the rectangle
    derived from the group's frame-coordinates origin shifted by
    hshift/vshift. `compute_pass_shift_range` now takes `num_passes`
    and models an implicit `n=num_ds` final-resolution entry that the
    spec text omits (documented SPECGAP — without it, single-pass
    frames would have minshift=maxshift=3 and decode no modular data).
  - **`src/toc.rs`** — TOC entries of value 0 are now accepted (an
    empty LfGroup or PassGroup section is legal when no channel
    matches that section's filter). Round 6 over-strictly rejected
    `entry == 0`.
  - **`src/ans/cluster.rs`** — `read_general_clustering` now handles
    the prefix-coded sub-stream branch (the simple-clustering path
    covered by the round-2..6 fixtures avoided this branch
    altogether).
  - **`src/lib.rs`** — `decode_codestream` reads each TOC slot as a
    fresh sub-bitstream-bounded `BitReader`, dispatches LfGlobal
    (slot 0), then iterates `pass_idx × group_idx` PassGroups (slots
    `1 + num_lf_groups + p*num_groups + g`), then applies inverse
    transforms over the assembled image. Single-group / single-pass
    frames continue to use the round-3..6 fast path so the five
    pixel-correct lossless fixtures remain regression-free.
  - **`tests/fixtures/synth_320_grey/`** — a 320×320 grey gradient
    encoded by cjxl 0.11.1 (`-d 0 -m 1 -e 1 -g 0 -R 0`) producing a
    9-group multi-group lossless modular fixture. Committed for round-8
    once the SPECGAP above is resolved.

- **Round 6 (2024-spec)** — Annex E.4 ICC profile decode + LfGroup /
  PassGroup type scaffolding.
  - **`src/icc.rs`** — full ICC profile decoder per Annex E.4. Reads
    `enc_size = U64()`, then 41 pre-clustered distributions (the
    existing `EntropyStream::read(br, 41)` infrastructure built for
    Modular), then `enc_size` bytes via `DecodeHybridVarLenUint`
    driven by the `IccContext(i, prev_byte, prev_prev_byte)`
    41-context function from E.4.1. The encoded byte stream is split
    into `output_size` (Varint) + `commands_size` (Varint) prefix +
    command stream + data stream, then walked through E.4.3 (header
    with predicted-byte ladder), E.4.4 (tag list with 21-tagcode
    switch + previous_tagstart / previous_tagsize accumulation), and
    E.4.5 (main content with command set 1 / 2 / 3 / 4 / 10 / 16-23
    + Nth-order predictor at orders 0/1/2). 14 unit tests
    (round-trip helpers + spec-listing edge cases incl. the example
    "shuffle of (1,2,3,4,5,6,7) at width 2 → (1,5,2,6,3,7,4)").
  - **`src/lf_group.rs`** — Annex G.2 type scaffolding. `LfGroup`
    bundle (Table G.3) + `LfCoefficients` (G.2.2 — VarDCT only) +
    `ModularLfGroup` (G.2.3 — always present) + `HfMetadata` (G.2.4).
    Per-LfGroup decode itself is round-7 work; the parser stub
    returns `Error::Unsupported` with a precise round-7 follow-up
    message. `ModularLfGroup::rect_for_index` does compute the
    per-LfGroup pixel rectangle in frame coordinates.
  - **`src/pass_group.rs`** — Annex G.4 type scaffolding. `PassGroup`
    bundle (Table G.5) + `ModularGroupData` (G.4.2). Per-PassGroup
    decode is round-7 work; `ModularGroupData::rect_for_index`
    computes per-group pixel rectangles. Plus
    `compute_pass_shift_range(pass_index, downsample, last_pass)`
    implementing the `(minshift, maxshift)` recurrence from the
    G.4.2 first paragraph: pass 0 starts at maxshift=3, subsequent
    passes inherit maxshift = previous pass's minshift; minshift
    comes from the smallest `log2(downsample[n])` over `n` with
    `last_pass[n] == p`, falling back to maxshift if no match.
  - **`lib::decode_codestream`** — when
    `metadata.colour_encoding.want_icc == true` the bit reader is
    now correctly advanced past the ICC stream via
    `icc::decode_encoded_icc_stream` + `icc::reconstruct_icc_profile`,
    instead of erroring with "Annex B ICC stream not yet wired". A
    minimal ICC.1 sanity check verifies the "acsp" magic at offset
    36; the decoded bytes are not propagated to `VideoFrame`
    (`oxideav_core::VideoFrame` has no ICC slot in 0.1.x).
    Multi-LfGroup / multi-group / multi-pass / VarDCT frames now
    fail with precise round-7-targeting error messages instead of
    the generic "TOC with N entries" rejection.

### Round-6 acceptance

- All 5 currently-pixel-correct fixtures still decode pixel-correct
  vs `expected.png`: pixel-1x1, gray-64x64, gradient-64x64-lossless,
  palette-32x32, grey_8x8_lossless. (No regression of the
  five-round single-group decode path.)
- 32 new unit tests (14 ICC + 8 LfGroup + 10 PassGroup); total test
  count goes from 211 to 243.
- `cargo clippy --all-targets -- -D warnings` clean.
- `cargo fmt --check` clean.

### Round-6 deferred (round-7 candidates)

- LfGroup / PassGroup actual decode wiring: blocked on four
  coordinated changes — GlobalModular `nb_meta_channels`-aware
  partial decode (G.1.3 last paragraph), `stream_index` threading
  through `decode_channels` (Table H.4 property 1), TOC permutation
  awareness, and inverse-transform application timing (post-PassGroup
  per G.4.2 last sentence). These four are too coupled to ship
  individually without regressing the five pixel-correct fixtures.
- Multi-group lossless modular fixture: docs corpus has no fixture
  in this category (the smallest multi-group fixture
  `large-1024x768-d2` is VarDCT). Round 7 should produce one via
  `cjxl input.png output.jxl -d 0 -e 7` against a 256×256+ lossless
  PNG and commit it to `tests/fixtures/`.
- ICC bytes propagation to `oxideav_core::VideoFrame`: the parsed
  ICC profile is currently discarded after sanity-check because
  there's no `VideoFrame::icc_profile` slot in `oxideav-core 0.1`.
  Round 8+ work should be coordinated with an `oxideav-core` minor
  release that adds the slot.
- XYB inverse transform (§C.5 / §K): deferred — no XYB fixture in
  current pixel-correct corpus. Synthetic XYB fixture would require
  encoder support which doesn't exist in this crate.

### SPECGAP entries (round 6)

None new. The Annex E.4 ICC pseudocode in the 2024 published edition
is complete and unambiguous; no round-7 SPECGAP pivot is required
for it.

- **Round 5 (2024-spec)** — RFC 7932 §3.5 prefix-code histogram Kraft
  early-stop fix; `grey_8x8_lossless.jxl` (cjxl 0.11.1, 180-byte
  emit) now decodes pixel-correct (all 64 bytes equal 128 as
  expected for a constant-grey PGM input).
  - **Root cause** — `read_complex_prefix` decoded all `count`
    code-lengths regardless of whether the running Kraft sum had
    already reached `1 << 15`. cjxl 0.11.1 emits histograms whose
    Kraft saturates mid-stream (specifically the cluster[1] histogram
    at bit 299..549 of the grey_8x8 fixture: 251 lengths reach
    Kraft = 32768 exactly; the remaining 6 lengths must be treated
    as implicit zeros per RFC 7932 §3.5).
  - **Fix** — track a running Kraft sum inside the lengths loop;
    once it reaches `1 << 15`, break early and rely on the initial
    `vec![0u32; alphabet_size]` to leave the trailing entries as
    implicit zeros. Repeat-16 (re-emit previous non-zero length) is
    also instrumented to short-circuit when its replication crosses
    the Kraft boundary.
  - **Bisect** — `tests/round5_grey_8x8_cluster_bisect.rs` walks the
    symbol-stream prelude bit-by-bit, decoding each cluster's prefix
    histogram and printing the clcl array, the Kraft sum, and the
    per-symbol code-length array. Cluster 1 was the failing one;
    the round-4 trace stopped at bit 563 with Kraft=32832 (64 over
    budget). `src/ans/prefix.rs` exposes a public `diagnose_complex_prefix`
    entry point that captures partial state even on failure.
  - **New API surface** — `read_prefix_code_traced` /
    `read_complex_prefix_traced` / `diagnose_complex_prefix` /
    `ClclTrace` are public so future bisect tests can reproduce the
    same per-cluster step-by-step trace without copy-paste.

- **Round 4 (2024-spec)** — three independent decoder bugs fixed; all
  three previously-blocked single-group docs fixtures
  (`gradient-64x64-lossless.jxl`, `palette-32x32.jxl`, plus the round-3
  baseline `gray-64x64.jxl`) now decode pixel-correct against their
  committed `expected.png` references via a new full-image PNG-decoder
  comparison harness (`tests/round4_pixel_correctness.rs`).
  - **2024-spec C.3.3 `ReadUint` formula fix** — round 3 computed the
    extra-bits count as `n = split_exponent + ((token - split) >>
    (msb + lsb))` but spec C.3.3 says
    `n = (split_exponent - msb_in_token - lsb_in_token) +
    ((token - split) >> (msb + lsb))`. The missing `- msb - lsb`
    inflated `n` by `(msb + lsb)` extra bits per above-split token,
    which is the root cause of "12× bits/token" over-consumption that
    blocked `gradient-64x64` and `palette-32x32` in round 3.
    `HybridUintConfig::read_uint` now uses the spec formula; the
    in-tree `encode_uint` round-trip helper was likewise updated to
    keep the existing round-trip unit tests passing.
  - **2024-spec H.5.2 Self-correcting predictor — three sign / formula
    fixes**:
    1. `subpred[3]` had `n8.wrapping_add(...)` in round 3; spec listing
       reads `subpred[3] = N3 - (...)`. Sign flipped to
       `wrapping_sub`.
    2. `error2weight` was missing the trailing `>> shift`. Spec:
       `4 + ((maxweight * ((1<<24) Idiv ((err_sum >> shift) + 1))) >> shift)`.
       The missing outer shift inflated weights non-uniformly across
       sub-predictors when their shifts differ, producing wrong
       sub-predictor mixing.
    3. `s = (sum_weights >> 1) - 1` per spec; round 3 omitted the
       `- 1`.
  - **2024-spec H.5.1 `err[i]` formula fix** — round 3 stored
    `abs(((subpred[i] + 3) >> 3) - true_value)`; spec is
    `(abs(subpred[i] - (true_value << 3)) + 3) >> 3`. These differ in
    rounding, producing wrong sub_err values that propagate to
    downstream WP weights.
  - **2024-spec H.5.2 sub_err edge cases** — when N or NW does not
    exist for the `err_sum[i]` neighbour gathering, spec says use 0
    (for N, W, WW) or N's value (for NW, NE). Round 3 used 0 for all
    out-of-range neighbours; corrected to use N's err for NW at
    column 0.
  - **2024-spec H.5.2 rightmost-column carry** — spec adds
    `err[i]_W` to `err_sum[i]` when `x == width - 1`. Round 3
    omitted this. Now applied via an explicit branch.
  - **2024-spec H.5 / H.4 max_error semantics** — round 3 used the
    PREVIOUS sample's max_error for property 15 of the CURRENT
    sample. Spec calls `wp_predict` first to get max_error for the
    current sample, then uses that as `property[15]` for the MA-tree
    decision. Restructured `decode_channels` to call WP up-front,
    use the result for both property 15 and (if the leaf picks
    predictor 6) the prediction value.
  - **`tests/round4_pixel_correctness.rs`** — full-image PNG-backed
    pixel-correctness harness (4 fixtures: `pixel-1x1`,
    `gray-64x64`, `gradient-64x64-lossless`, `palette-32x32`) plus
    a manual `palette_invasive_pixel_decode` diagnostic that walks
    decode_channels token-by-token printing bit positions, kept for
    round-5 work.
  - **`png` dev-dependency** (`png = "0.18"`) — pulled only by the
    test harness; no codec-semantics overlap with JPEG XL itself.
- **Round 3 (2024-spec)** — bit-alignment fix at the GlobalModular →
  inner-Modular boundary + ANS alias-mapping conditional-offset fix.
  After this round, `gray-64x64.jxl` decodes pixel-correct against
  its committed `expected.png` reference (gradient pattern
  `pixel(x, y) = ((x + y) * 2) & 0xff`, first scanline `0, 2, 4, …`).
  - **2024-spec C.3.2 (ANS state init position)** — round 1+2 read
    the ANS `u(32)` state initialiser EAGERLY at end of the entropy
    stream prelude inside `EntropyStream::read`. Empirical bisect
    against `cjxl 0.12.0` traces shows the state init is emitted
    AFTER the inner Modular sub-bitstream's ModularHeader (i.e.
    after `use_global_tree` / `WPHeader` / `nb_transforms` /
    `transforms`) and IMMEDIATELY before the first symbol decode.
    Round 3 splits the prelude reading from the state init reading
    via a new `EntropyStream::read_ans_state_init` method, which
    `decode_channels` invokes just before the first per-pixel
    `DecodeHybridVarLenUint` call. Position confirmed by tracing
    inner_use_global_tree against the expected `1` bit in cjxl's
    bytestream: bit 199 (gray-64x64), bit 338 (gradient-64x64),
    bit 359 (palette-32x32) all read `1` (true) once the state init
    is deferred — they were reading `0` (false) when the state init
    was eager.
  - **2024-spec C.2.6 (alias mapping conditional offset)** — round 1
    `AliasTable::lookup` always returned `offset = offsets[i] + pos`,
    but spec C.2.6 makes the formula CONDITIONAL on whether
    `pos >= cutoffs[i]`: in the "stays in own bucket" branch the
    offset is just `pos` (no `+ offsets[i]`). The unconditional
    formula caused incorrect ANS state evolution and triggered
    extra `u(16)` refills that ran the bitreader past EOF on
    small ANS-path fixtures. Round 3 adds the conditional.
  - **`gray-64x64.jxl` pixel-correct end-to-end** — first lossless
    Modular fixture > 1×1 to decode without EOF. Output checked
    against the gradient pattern in `docs/image/jpegxl/fixtures/
    gray-64x64/expected.png` first 16 pixels (0, 2, 4, …, 30) +
    histogram (min=0 max=252 mean=126.0).
  - **Diagnostic tooling**: `tests/round3_bit_alignment_bisect.rs`
    — eight tests (4 manual bisects + 4 production-path walks)
    that print bit positions at every spec milestone for the four
    target fixtures, with cross-reference comments against trace.
- **Round 2 (2024-spec)** — Inverse Modular transforms (Annex H.6) +
  full Self-correcting predictor (Annex H.5) + 2024-spec-correctness
  fixes for the entropy stream prelude (Annex C.2.1) and CLCL prefix
  decode (RFC 7932 §3.5), built additively on round 1's pixel-1x1
  pixel-correct decode.
  - **`modular_fdis::inverse_palette` (Annex H.6.4)** — full inverse
    palette transform incl. delta-palette via the verbatim
    `K_DELTA_PALETTE[72][3]` table (transcribed from FDIS Listing
    L.6), implicit colour extrapolation via the bitdepth-scaled
    formulas, and per-channel re-expansion from a single index
    channel + meta-channel palette to `num_c` colour channels.
  - **`modular_fdis::inverse_rct` (Annex H.6.3)** — all 6 RCT type
    modes (`type ∈ [0, 6]`) × 6 permutations = 42 `rct_type` codes,
    incl. the YCgCo branch (type==6) that uses the 4-step inverse.
    Channel triple `(A, B, C)` re-mapped to `(V[0], V[1], V[2])` via
    spec-formula permutations.
  - **`modular_fdis::horiz_isqueeze` / `vert_isqueeze` (Annex H.6.2)**
    — pair-merge inverse Squeeze step with the spec's `tendency()`
    function. Default-params (empty `squeeze_params`) defers to a
    later round.
  - **`global_modular::apply_transforms_to_channel_layout`** now
    handles Squeeze layout (channel dim halving + residu-channel
    insertion at `r + c - begin`).
  - **`global_modular`** applies inverse transforms in REVERSE order
    after `decode_channels` per H.6's "from last to first" rule,
    instead of erroring out as in round 1.
  - **`modular_fdis::WpState` + `wp_predict` (Annex H.5)** — full
    Self-correcting predictor with `true_err`, `sub_err[0..4]`
    per-channel arrays, 4 sub-predictor weights, and the H.5.2
    `error2weight` clamping. State updates after every sample
    decode regardless of whether predictor 6 was selected (so future
    predictor-6 calls see correct history).
  - **`modular_fdis::get_properties`** now wires `property[15]` to
    the WP `max_error` value (round 1 left it at 0).
  - **2024-spec C.2.1 fix in `ans::cluster::read_general_clustering`**:
    `use_prefix_code` ↔ `log_alphabet_size` mapping was reversed
    (round 1 fixed `EntropyStream::read` but missed the same swap
    in the cluster sub-stream).
  - **RFC 7932 §3.5 CLCL prefix-decode fix**: the 6-symbol
    code-length-code lookup interprets codewords as "bits parsed
    right to left" — the rightmost char of each codeword is the
    FIRST bit read. This is equivalent to LSB-first packing with
    no bit-reversal (round 1 incorrectly bit-reversed, breaking
    every fixture using complex-prefix codes).
  - **`bitreader::pu0` is now lenient** — does not enforce zero
    padding bits before byte boundaries. cjxl 0.12.0 emits non-zero
    padding on small fixtures (gradient-64x64, palette-32x32) at
    the metadata→frame_header alignment; the 2024 spec's text says
    the zero-padding is "for validity" only, not a decode-time
    requirement, and `djxl` accepts the same streams.
  - **`metadata_fdis::ImageMetadataFdis::read` tail dropped** — the
    FDIS-2021 `default_transform` Bool + `cw_mask` u(3) +
    per-mask F16 weight arrays were over-reading by 4-5 bits
    relative to libjxl's actual stream consumption. Round 2 leaves
    these at their defaults (`default_transform=true, cw_mask=0`)
    and SPECGAPs the exact gating condition.
  - **3 new soft fixture tests** (`r2_gradient_decode_attempt`,
    `r2_palette_decode_attempt`, `r2_gray_docs_decode_attempt`)
    against the docs/image/jpegxl/fixtures/ corpus. These currently
    fail at GlobalModular entropy stream prelude alignment in the
    complex-prefix path but the inverse-transform infrastructure
    they would feed is verified by unit tests.
  - **`pixel-1x1.jxl` regression-free** — the 1×1 RGB lossless
    acceptance fixture from round 1 still decodes to R=255 G=0 B=0.

- **Round 1 (2024-spec)** — Modular sub-bitstream pixel decode
  end-to-end against the final ISO/IEC 18181-1:2024 core spec (Annex
  H), built on top of the round-1..3 baseline:
  - `modular_fdis::evaluate_tree` walks decision-node MA trees per
    H.4.1, replacing the round-3 single-leaf-only restriction.
  - `modular_fdis::get_properties` computes the 16 base properties
    of Table H.4 plus per-previous-channel properties (4 each for
    every channel with matching dims/shifts).
  - `modular_fdis::Neighbours` materialises the 7 prediction
    neighbours per Table H.2 with the H.3 edge-case fallbacks.
  - `modular_fdis::predict` covers Table H.3 predictors 0-5 + 7-13;
    predictor 6 (Self-correcting) is implemented for the trivial
    (0, 0) origin case (returns 0 — full WP defers to round 2).
  - `modular_fdis::TransformInfo` + `TransformId` parses the H.7
    bundle for `nb_transforms > 0`; channel-list adjustment for
    Palette is applied; inverse Palette / Squeeze application defers
    to round 2 with a clean `Error::Unsupported` exit point.
  - `decode_codestream` accepts RGB images (3 channels) in addition
    to Grey, producing 3 / 1 plane VideoFrames respectively.
  - `pixel-1x1.jxl` (1×1 RGB lossless, 22 B fixture from
    `docs/image/jpegxl/fixtures/pixel-1x1/`) now decodes
    pixel-correct: R=255, G=0, B=0 (matches `expected.png`).
  - Black-box validator test for `djxl` confirms the binary decodes
    the same `gray-64x64` fixture; we never read djxl/cjxl source.
- **FDIS-2021 spec typo #5 documented and corrected**: D.3.1's
  `use_prefix_code` ↔ `log_alphabet_size` mapping was swapped in the
  FDIS 2021 text (`if use_prefix_code is 1 → log_alphabet_size = 5 +
  u(2)`); the 2024-published edition (C.2.1) reverses it (prefix →
  15, ANS → 5+u(2)) which matches the libjxl reference output
  observed via cjxl/djxl. The implementation in
  `modular_fdis::EntropyStream::read` now follows the 2024 reading.

### Removed

- **Decoder rounds 7-11 + encoder rounds 1-6 RETIRED 2026-05-08** under
  fruits-of-poisonous-tree. The `OxideAV/docs` repository retired
  `image/jpegxl/libjxl-trace-reverse-engineering.md` (the 792-line
  behavioural-trace writeup) on 2026-05-06 (commit `d732002`); the
  retire reasoning applies to any code authored by an agent that read
  that doc, even when no source was literally quoted. This crate's
  master was reset to `9d79695` (round-3 LfGlobal + GlobalModular +
  Modular sub-bitstream wiring, 2026-05-01) — the last commit before
  the retired trace doc landed in `OxideAV/docs` (`8931c26`,
  2026-05-02 22:55). The pre-retirement history is preserved on the
  `old` branch for forensics.
  - **Retired decoder commits**: `403f256` (round 7 — typo #6/#7 +
    MA-tree decodes), `06b4d00` (modular pre-check scope),
    `d49e583` (round 8 — prefix early-terminate),
    `ba225c2` / `1217a08` / `1540102` / `7827d96` / `52b1cfb` /
    `8258cdc` / `a2419a6` (round 9 — typo #8 + instrumentation),
    `ab5f94a` (round 10 — kRCT/kPalette/kSqueeze parsing + dispatch),
    `2e41c1d` (round 11 — Appendix B four-range index partition).
  - **Retired encoder commits**: `a53e041` / `198f9e4` / `5f35de8` /
    `f83a6d8` / `0c9b9d8` / `88f05ee` / `6215efc` / `39b2e73` /
    `dd8be6e` / `65195e5` / `1925527` / `fedb620` / `9804c79` (encoder
    rounds 1-6 — independent codec surface but authored within the
    same trace-doc-contaminated session window).
  - **Retired infrastructure commits**: `4f1b6bd` (CI workflow
    centralisation), `9a8b33d` (standalone-friendly registry feature),
    `2cb9943` (register_containers extension lookup), `dd68816`
    (register entry-point unification), `cde6f6a` (auto-register
    macro), `e4ea5b7` (`make_decoder` → `first_decoder` rename),
    `852ac81` (re-export `__oxideav_entry`), `9d3e999` (drop linkme
    dep). Re-applicable in non-narrative plumbing rounds later.
  - **Retired crates.io versions** (yank pending): v0.0.5 (published
    2026-05-04), v0.0.6 (2026-05-04), v0.0.7 (2026-05-05). Tags
    v0.0.5 / v0.0.6 / v0.0.7 deleted from `origin`. Version bumped
    0.0.4 → 0.0.8 in this commit to skip the yanked range.
  - **Forward path**: a strict-isolation `docs/image/jpegxl-cleanroom/`
    workspace with the four-role layout (Specifier / Extractor /
    Implementer / Auditor) — Specifier wall: ISO/IEC 18181-1 FDIS +
    18181-3 conformance corpus only, no libjxl source ever. Modelled
    after `docs/video/msmpeg4/`, `docs/video/magicyuv/`,
    `docs/audio/tta-cleanroom/`. Until that workspace exists, this
    crate ships only the round-1..3 ANS + headers + LfGlobal +
    GlobalModular wiring; no further decoder rounds will land.

### Changed

- API shim for the post-retire workspace: `register(ctx: &mut RuntimeContext)`
  + `register_codecs(reg: &mut CodecRegistry)` + `oxideav_core::register!`
  macro call (current registration pattern); the round-1..3 test that
  used `reg.make_decoder` now uses `ctx.codecs.first_decoder` to match
  the post-rename `oxideav-core` API.

### Added

- New `ans` module implementing the FDIS 18181-1:2021 Annex D entropy
  layer (round 1 of the committee-draft → FDIS migration). Submodules:
  - `ans::prefix` — Brotli (RFC 7932) §3.4 simple + §3.5 complex
    prefix codes, used by the `use_prefix_code == 1` histogram path
    of D.3.1.
  - `ans::alias` — alias-mapping table init + lookup (D.3.2,
    Listings D.1 + D.2). Implements Vose's alias method with the
    spec PDF's u/o/i variable typo corrected.
  - `ans::symbol` — 32-bit-state ANS reverse decoder (D.3.3,
    Listing D.3) including the `0x130000` end-of-stream check.
  - `ans::distribution` — ANS distribution decoder (D.3.4,
    Listing D.4) with the verbatim 128 × 2 `kLogCountLut` lookup
    table transcribed from p. 64 of the FDIS PDF.
  - `ans::cluster` — distribution clustering simple-path + the
    inverse move-to-front transform (D.3.5, Listing D.5).
  - `ans::hybrid` — hybrid-integer LZ77 decode driver (D.3.6,
    Listing D.6) with the verbatim 120 × 2 `kSpecialDistances`
    lookup table transcribed from p. 66 of the FDIS PDF, plus a
    1 MiB sliding window per stream.
  - `ans::hybrid_config` — `HybridUintConfig` decode + `ReadUint`
    (D.3.7, Listing D.7).
  Every allocation is bounded against the input length; the
  module ships 45 self-contained unit tests covering hand-built
  bitstreams from each spec listing plus four malicious-input
  cases (oversized log_alphabet_size, oversized alphabet, huge
  hybrid token, huge prefix-code alphabet).
  The committee-draft `abrac` / `begabrac` / `matree` / `modular`
  pipeline and the registered `make_decoder` are intentionally
  untouched — round 2 will wire the new ANS coder behind a
  FrameHeader + TOC entry point.
- `BitReader` gains `peek_bits(n)` / `advance_bits(n)` / `bits_remaining()`
  / `read_u8_value()` to support the ANS distribution decoder
  (D.3.4 reads `u(7)` for the kLogCountLut key without advancing,
  then advances by the table-derived step count).

- Modular sub-bitstream channel decoder per the 2019 committee draft
  (`arxiv-1908.03565v2`, Annexes C.9 + D.7), a stepping stone toward
  full FDIS 18181-1 support. New modules:
  - `abrac` — bit-level adaptive range coder (D.7).
  - `begabrac` — bounded-Exp-Golomb integer coder over a known signed
    range, layered on `abrac` (D.7.1).
  - `matree` — meta-adaptive decision tree that picks a per-context
    BEGABRAC for each pixel (D.7.2 / D.7.3).
  - `predictors` — five named pixel predictors (Zero, Average,
    Gradient, Left, Top) from C.9.3.1.
  - `modular` — channel-header parser plus the per-pixel property +
    predictor + entropy decode loop, exposed as
    `modular::decode_single_channel`.
  - `BitReader` gains `pu0()` (zero-padded byte align), `pu()`
    (byte-align value), `read_varint()` (A.3.1.5), and a `data()`
    accessor used by entropy coders that switch from bits to bytes.
- DoS-hardening of the Modular decode path against malformed
  channel headers and adversarial entropy streams:
  - `Channel::try_new` refuses dimensions larger than
    `MAX_CHANNEL_DIM` (32 768) per side or pixel counts above
    `MAX_CHANNEL_PIXELS` (256 M); the bitstream-driven entry point
    `decode_single_channel` now uses `try_new` so a forged
    width/height pair returns `InvalidData` instead of asking the
    allocator for terabytes.
  - `MaTree::decode` caps the bit-depth `n` at `MAX_VALUE_BIT_DEPTH`
    (32) so a pathological caller can't make each leaf BEGABRAC
    allocate gigabytes of mantissa context.
  - `decode_subtree` caps the total node count at
    `MAX_MA_TREE_NODES` (1 << 20) and recursion depth at
    `MAX_MA_TREE_DEPTH` (1024), preventing both heap exhaustion and
    stack overflow when the entropy stream keeps emitting "decision
    node" instead of "leaf".
- Regression tests for the hardening above, including a
  hand-crafted 1 M × 1 M channel-header fixture that asserts
  `decode_single_channel` rejects with `InvalidData` rather than
  allocating.

### Changed

- Crate description updated to mention the Modular sub-bitstream
  decode now landed (committee-draft path).
- Doc-comment in `lib.rs` updated to reflect the new module layout
  and the remaining gap toward FDIS 18181-1 (FrameHeader/TOC,
  Squeeze, VarDCT, ANS-based entropy).

### Removed

- `SPEC_BLOCKED.md`: the ISO/IEC 18181-1 normative spec (committee
  draft + FDIS) is now present in `docs/image/jpegxl/`, so the
  block is lifted. Migration to the FDIS layout (ANS entropy,
  FrameHeader, TOC, ImageMetadata FDIS shape) is tracked as the
  next round of work, not a block.

## [0.0.4](https://github.com/OxideAV/oxideav-jpegxl/compare/v0.0.3...v0.0.4) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- drop Cargo.lock — this crate is a library
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- bump to oxideav-core 0.1.1 + codec 0.1.1
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
