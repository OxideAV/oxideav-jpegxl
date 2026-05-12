# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
