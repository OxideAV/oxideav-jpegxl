# oxideav-jpegxl

Pure-Rust **JPEG XL** (ISO/IEC 18181-1:2024) decoder. Resumed
2026-05-08 against the final published 2024 core spec after the
trace-doc-driven rounds 7-11 + encoder rounds 1-6 were retired
(see "Why retired (history)" below).

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
  pending a docs-collaborator libjxl-WP behavioural trace at the
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
(`bitreader::BitReader`) that matches the reference libjxl bit packing,
including the JXL `U32` 2-bit-selector encoding. On top of it:

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
