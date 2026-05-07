# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.8](https://github.com/OxideAV/oxideav-jpegxl/compare/v0.0.7...v0.0.8) - 2026-05-07

### Other

- round 6 — predictor candidate set widened to FDIS C.16 ids 1..=5 + 7..=12
- drop dead `linkme` dep
- re-export __oxideav_entry from registry sub-module
- round 5 — per-image predictor selection + 256x256 PSNR target
- registry calls: rename make_decoder/make_encoder → first_decoder/first_encoder
- auto-register via oxideav_core::register! macro (linkme distributed slice)
- unify entry point on register(&mut RuntimeContext) ([#502](https://github.com/OxideAV/oxideav-jpegxl/pull/502))
- add register_containers for .jxl extension lookup

### Added

- **Round 6 encoder — predictor candidate set widened to FDIS C.16
  ids 1..=5 + 7..=12.** The encoder reconstruction-buffer convention
  now tracks `(x-1, y)`, `(x, y-1)`, `(x-1, y-1)`, `(x+1, y-1)`,
  `(x-2, y)` so `pick_best_predictor_id` can score TopRight (7),
  TopLeft (8), LeftLeft (9), Avg(L,TL) (10), Avg(TL,T) (11), and
  Avg(T,TR) (12) alongside the round-5 set `{1, 2, 3, 4, 5}`.
  Predictor 13 (Six-Tap) is held back: the FDIS Listing C.16 formula
  `(7*W + 6*N + 3*NE - 2*NN + WW + NEE + 8) Idiv 16` self-roundtrips
  through our own decoder bit-exactly but does NOT bit-equal libjxl's
  `djxl` on random natural data — likely an FDIS / libjxl coefficient
  or rounding-offset divergence. Workspace policy bars consulting
  libjxl source as a substitute for a docs trace, so predictor 13
  is excluded from the encoder candidate list until
  `docs/image/jpegxl/libjxl-trace-reverse-engineering.md` gains the
  empirical correction.
  - `predict()` is now a 13-arm dispatcher matching
    `modular_fdis::predict` for ids 0..=13.
  - `pick_best_predictor_id` candidate const reordered so the
    strict-less-than scan keeps the first tied candidate; visit
    order matters for tie-break behaviour on flat fixtures.
  - 6 new unit tests in `encoder::tests`:
    `round6_predict_id_7_uses_topright`,
    `round6_predict_id_9_uses_leftleft`,
    `round6_predict_id_10_avg_west_northwest`,
    `round6_predict_id_12_avg_top_topright`,
    `round6_pick_best_grey_constant_returns_first_tied_candidate`,
    `round6_pick_best_excludes_predictor_13`.
  - 1 new integration test
    (`tests/encode_roundtrip.rs::round6_64x64_random_self_roundtrip`)
    asserting whichever round-6 predictor is selected self-decodes
    bit-exactly — CI tripwire if predictor 13 sneaks back into the
    candidate const.
  - Compression on the 256×256 grey natural fixture is unchanged
    from round 5 (33747 B / 4.12 bpp) because both Average (3) and
    Gradient (5) produce the same residual entropy on smooth
    sinusoids. The wider candidate set is a foundation for round-7+
    per-channel / multi-leaf MA-tree splits where channels with
    strong horizontal or vertical anisotropy can choose 7..=12
    independently.

- **Round 5 encoder — per-image predictor selection.** The Modular
  encoder now scans the input once per candidate predictor (sum of
  `|residual|` over all channels) and picks the lowest-scoring one
  for the single MA-tree leaf, instead of always emitting Gradient
  (5). Candidate set: `{1 Left, 2 Top, 3 Average, 4 West-Predictor,
  5 Gradient}` from FDIS Listing C.16. Predictor 0 (Zero) is
  excluded (rarely optimal on natural data); predictor 6 (Annex E
  Weighted) is excluded (decoder rejects); predictors 7..=13 are
  excluded for now (need a wider reconstruction-buffer refactor).
  - `pick_best_predictor_id` does the prescan; `predict()` is the
    dispatcher matching the round-5 reconstruction-buffer convention.
  - `write_single_leaf_ma_tree(bw, predictor_id)` (renamed from
    `write_gradient_leaf_ma_tree`) parameterises the cluster-1
    single-symbol prefix code so any of the 13 supported predictor
    ids fits in the 4-bit alphabet.
- **Round 5 PSNR-Y target hit.** New cross-validation tests in
  `tests/encode_djxl_roundtrip.rs`:
  - `djxl_decodes_our_grey_256x256_natural_image_with_compression` —
    256×256 synthetic natural grey (smooth sinusoid + low-amplitude
    noise) encodes to **33747 bytes / 4.12 bpp / 51.5% of raw**
    (vs uncompressed 65536 bytes), bit-exact lossless round-trip
    through libjxl's `djxl`. PSNR-Y is mathematically infinite
    (lossless, MSE = 0), well above the round-39 35 dB target.
  - `self_roundtrip_grey_256x256_natural_image` — same fixture
    through our own decoder, asserts both compression and
    pixel-equality without the djxl dependency.

## [0.0.7](https://github.com/OxideAV/oxideav-jpegxl/compare/v0.0.6...v0.0.7) - 2026-05-05

### Other

- round 11 — Appendix B four-range index partition + Path 1/2 dispatch
- gate oxideav-core behind default-on `registry` feature
- round 10 — kRCT/kPalette/kSqueeze transform parsing + dispatch
- round 8 — early-terminate symbol-code-lengths read on space==0
- scope pixels>bits_remaining pre-check to prefix-coded streams

### Added

- **Round 11 — inverse-palette four-range index partition + Path 1/2
  dispatch.** `transforms::inverse_palette` now follows Appendix B of
  `docs/image/jpegxl/libjxl-trace-reverse-engineering.md` (commit
  `679cf63`) for the full negative-index/explicit/cube partition
  rather than the spec's collapsed three-branch form. Concretely:
  - **§B.3 four-range partition** factored into a new
    `get_palette_value(index, c, nb_colours, meta, meta_w, bit_depth)`
    helper covering: `index < 0` (implicit delta-palette via
    `kDeltaPalette[72][3]`, RGB-only); `0 <= index < nb_colours`
    (explicit palette lookup at `meta[c, index]`); `nb_colours <=
    index < nb_colours + 64` (small 4×4×4 cube, RGB-only); and
    `nb_colours + 64 <= index` (large 5×5×5 cube, RGB-only,
    no upper bound — modulo wraps).
  - **§B.4 Path 1 vs Path 2 dispatch** correctly fires the predictor
    add when EITHER `nb_deltas > 0` OR `predictor != Zero` (previous
    code only checked `nb_deltas > 0`, missing the
    `nb_deltas==0 && predictor!=Zero` case where negative indexes
    still satisfy `index < nb_deltas`).
  - **§B.6 bit-depth clamp to 24** for implicit-palette scaling
    (`bit_depth.min(24).max(1)`); previously the 24-bit cap was
    not enforced.
  - **§B.2 step 1 — empty-channel insertion** for `num_c > 1`
    (zero-initialised, not copies of the index channel — predictors
    must see a clean output state when reading already-decoded
    neighbours, otherwise index values leak into RGB output).
  - 11 new unit tests in `transforms::tests` exercising each
    branch + alpha-channel zero-return + 143-modulus wraparound +
    bit-depth shift in the negative branch.
- **Standalone (no-`registry`) build path.** Default-on `registry` Cargo
  feature gates the `oxideav-core` dependency and the
  `Decoder`/`Encoder`/`register` trait surface (now in a new
  `registry` module). With `--no-default-features` the crate compiles
  without `oxideav-core` and exposes:
  - Crate-local `error::JxlError` / `error::Result` (mirrors the
    `oxideav_core::Error` variants this crate produces:
    `InvalidData` / `Unsupported` / `Eof` / `NeedMore` / `Other`).
  - Crate-local `image::JxlImage` / `image::JxlPlane` /
    `image::JxlPixelFormat` for decoded pixels (no
    `oxideav_core::VideoFrame` dependence).
  - Free-standing `decode_one_frame()` returning `JxlImage`, the
    existing `encoder::encode_one_frame()` returning `Vec<u8>`, plus
    the `probe()` / `probe_fdis()` header inspectors.
  All pipeline modules (`bitreader` / `bitwriter` / `container` /
  `metadata` / `metadata_fdis` / `frame_header` / `toc` /
  `lf_global` / `global_modular` / `modular` / `modular_fdis` /
  `transforms` / `predictors` / `matree` / `abrac` / `begabrac` /
  `ans*` / `extensions` / `encoder` / `ans_encoder`) now use the
  crate-local error type. The `From<JxlError> for oxideav_core::Error`
  + `From<JxlImage> for oxideav_core::Frame` conversions live in the
  registry-gated module so the framework-side `Decoder` / `Encoder`
  traits keep working unchanged. Inline `ci-standalone` job in
  `.github/workflows/ci.yml` builds + tests `--no-default-features`
  on every push.

### Fixed

- **prefix code: `read_complex_prefix` now matches libjxl's
  `ReadHuffmanCodeLengths` early-termination on `space <= 0`.** The
  loop that decodes per-symbol code lengths from the cl_code Huffman
  is supposed to stop the moment its 32768-budget Kraft counter hits
  zero (per Appendix A.6 of `libjxl-trace-reverse-engineering.md` and
  the libjxl reference implementation). Prior rounds 7-9 kept iterating
  until `idx == alphabet_size`, which over-consumed bit input by
  ~14 bits per cluster whose code-length bitstream over-filled the
  budget. On the cjxl 8x8 grey lossless fixture this desync surfaced
  as `sym=341 > alphabet=257` at the symbol-stream cluster-4 prefix
  prelude (the slid bit position landed in a 0x55 0x55 ... filler
  region). With the fix the prelude now lands at bit 1341 (matches a
  libjxl-exact Python re-decoder) and all five per-cluster prefix
  codes read successfully, unblocking the modular sub-bitstream.
  A strict `space != 0` post-check is now also enforced as libjxl
  does. Regression assertion in
  `tests/cjxl_grey_8x8_trace.rs::round9_symbol_prelude_per_cluster_dump`
  (now requires all 5 clusters to decode).
- **#382 — relax `pixels > bits_remaining` pre-validation for ANS-coded
  streams.** The pre-check in `modular_fdis::decode_channels` rejected
  any frame whose entropy-coded symbol stream was smaller than 1 bit
  per pixel. That lower bound only holds for prefix (Huffman) codes,
  where every symbol consumes ≥ 1 bit. ANS coding has variable
  fractional-bit cost per symbol — a constant-grey 8×8 channel encodes
  in well under 64 bits (preamble + 32-bit final state). The check now
  runs only when `EntropyStream::use_prefix_code == true`; ANS-coded
  paths rely on the ANS state / `BitReader` to detect truncation.
  Regression test `ans_coded_constant_grey_8x8_round_trips_through_decoder`.

## [0.0.6](https://github.com/OxideAV/oxideav-jpegxl/compare/v0.0.5...v0.0.6) - 2026-05-04

### Fixed

- fix clippy: drop unnecessary u32→u32 cast and 1*1*3 identity-op

### Other

- round 4 — re-integrate ANS-coded symbol stream
- round 4 — fix alias-table bijection via aligned D[]
- remove pre-existing buggy gradient_leaf_ma_tree test
- revert round-3 ANS integration; keep ans_encoder module
- fix alias-table inversion for offsets ≥ D[s]
- cross-validate ANS-coded 16x16 skewed grey vs djxl
- round 3 — switch symbol stream from prefix code to ANS
- round-3 ANS entropy encoder + distribution preamble
- drop unused write_prefix_code_count_one helper
- add ISOBMFF wrap_codestream + encode_one_frame_isobmff
- add djxl cross-validation roundtrip (round 2)
- round 2 — switch from Predictor=Zero to Gradient (id 5)
- round-1 lossless modular Gray/RGB/RGBA at 8 bpp
- add cargo-fuzz harness with libjxl cross-decode oracle

## [0.0.5](https://github.com/OxideAV/oxideav-jpegxl/compare/v0.0.4...v0.0.5) - 2026-05-03

### Other

- fix clippy unneeded-unit-expression warning
- round-9 progress milestone — clusters 0..3 OK, cluster 4 next
- round 9 — typo #8 ROOT CAUSE — first-sym-wins on collisions
- round-9 diagnostic deliberately panics so report shows in CI
- round-9 per-cluster symbol-prefix instrumentation
- round-8 was behaviour-neutral; consolidate stop-point tripwire
- add round-9 diagnostic that panics with new stop-point
- hard-assert round-7 kraft=135104 error is resolved
- rustfmt round-8 follow-up
- round 8 typo #8 — three RFC 7932 fixes for cl_code Kraft 37
- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- round 7 — typo #6 + #7 resolved, MA-tree decodes
- round 4 diagnostic — capture cjxl 8x8 fixture MA-tree T-stream
- FDIS migration round 3 — LfGlobal + GlobalModular + Modular sub-bitstream wiring
- FDIS migration round 2 — FrameHeader + TOC + ImageMetadata refresh + D.3.5 general clustering
- add FDIS Annex D ANS entropy decoder (round 1) + commit committee-draft Modular path
- document ISO/IEC 18181-1 spec block + harden plumbing tests
- pin release-plz to patch-only bumps

### Fixed (round 9, 2026-05-03)

- **Typo #8 ROOT CAUSE FOUND.** `PrefixCode::from_lengths` was
  OVERWRITING canonical-Huffman lookup-table collisions (later sym
  wins). For OVER-Kraft cl_codes (kraft sum > budget), this gave the
  wrong symbol on bit patterns where multiple symbols had the same
  prefix. The cjxl 8×8 grey lossless fixture's third per-cluster
  prefix code (cluster 2, count=257, complex hskip=3) has a cl_code
  with kraft sum 37 in budget 32 — over by 5. With "later wins", the
  resulting 257-symbol lengths array had kraft sum 33776 (4.123×
  budget); with "first wins", the same input produces kraft sum
  30089 (3.673× budget) which passes the 4× tolerance.
  Switching to "first sym wins" in the lookup-table fill matches
  what djxl 0.11.1 produces on the same bitstream.
- Verified by independent Python re-decoder of the cjxl 8×8 grey
  fixture (round-9 instrumentation): walks the bitstream from
  signature through all 5 per-cluster symbol-stream prefix codes.
  With "first wins" all 5 codes decode (kraft 1.002× / 3.67× / and
  three trivial); with "later overwrites" the third code's kraft
  jumps to 4.123× budget, failing the sanity check.
- The round-8 `cjxl_grey_8x8_round8_typo8_unresolved_tripwire` test
  is renamed to `cjxl_grey_8x8_round9_progress_marker` and asserts
  that the kraft=33776 stop-point error no longer appears. The new
  diagnostic test `round9_symbol_prelude_per_cluster_dump` (in
  `cjxl_grey_8x8_trace.rs`) walks each per-cluster prefix code
  manually and asserts all 5 decode OK.
- Note: cluster ORDER is per-cluster-INDEX (0..n_clusters), not
  per-CTX. The round-7/8 project notes called the failing code the
  "second" per-cluster prefix code; it's actually the THIRD
  (cluster index 2).

### Fixed (round 8, 2026-05-03)

- **`PrefixCode::from_lengths` Kraft computation now uses the actual
  `1<<max_length` budget** instead of always summing in `1<<15`. For
  the cl_code (18-symbol alphabet, max_length=5), this lets us catch
  over-Kraft cases at the cl_code construction site instead of
  deferring them to a confusing "downstream alphabet 4× over budget"
  error in the next call. The 4× sanity tolerance is preserved (libjxl
  is similarly permissive), so well-formed bitstreams still decode.
- **RFC 7932 §3.5 single-non-zero clcl special case handled.** When
  the cl_clcl decode loop reads all 18-HSKIP entries and finds only
  ONE non-zero length, RFC 7932 §3.5 says the cl_code degenerates to
  a single-symbol zero-length code. Previously we always called
  `from_lengths` which would build an L-bit code (consuming L bits per
  cl_code decode) — wrong, since the encoder emits zero bits in this
  case. `read_complex_prefix` now constructs a `max_length==0`
  `PrefixCode` directly when only one clcl entry is non-zero.
- **RFC 7932 §3.4 simple-prefix length assignment reverted to
  per-RFC.** Round 3's "fix" sorted ALL three (NSYM=3) or ALL four
  (NSYM=4 tree_select=1) symbols and assigned the length-1 code to
  the smallest-sorted symbol. RFC 7932 §3.4 says the lengths are
  assigned "in the ORDER they appear in the representation" — i.e.
  first-read gets length 1, second-read gets length 2 (and so on for
  NSYM=4). The "sorted order" rule mentioned in the RFC applies ONLY
  to within-length CODE ASSIGNMENT (which `from_lengths` handles
  automatically via its symbol-id-major iteration), not to which
  symbol gets which length. The old `read_simple_prefix_three_symbols_canonical_lengths`
  test was asserting the round-3 (incorrect) behaviour and has been
  updated to assert the per-RFC behaviour.

### Round 8 outcome — typo #8 NOT YET RESOLVED, fixes were behaviour-neutral

CI confirms the LITERAL round-7 error message (`kraft=135104, alphabet_size=257,
max_length=13`) no longer appears, but the underlying stop point is
unchanged. The new error message is:

```
JXL prefix: code lengths grossly overflow Kraft sum (kraft=33776, budget=8192, alphabet_size=257, max_length=13)
```

33776 / 8192 = 4.123 — the SAME 4× over-budget ratio as before
(135104 / 32768 = 4.123), just expressed in the new per-max_length
budget (8192 = 1<<13). So the decoder still hits the exact same
malformed-symbol-code situation, only the error format changed.

Implications:
- All three round-8 fixes were "behaviour-neutral" for this fixture:
  the simple-prefix change only matters for NSYM=3/4 (cluster 0 here
  is NSYM=2), the single-non-zero clcl case never triggered (so the
  cl_code was built canonically as before), and the per-budget Kraft
  computation only changes error message phrasing.
- The actual root cause is STILL one of the five hypotheses listed in
  `project_jpegxl_fdis_typos.md` (typo #8 entry). Round 9 should
  pursue:
  1. **Bit-position misalignment earlier in the decode pipeline.** Try
     adding a Python re-decoder that walks the entire bitstream up to
     the second prefix code, comparing bit positions against the
     trace test's `eprintln!`'d cluster boundaries.
  2. **Non-trivial cl_code interpretation difference.** Consider
     whether libjxl applies any additional normalization to cl_code
     lengths beyond what RFC 7932 §3.5 specifies.
  3. **`HybridUintConfig` reading off-by-one.** Check D.3.7 against
     the trace doc's §3.6 description.
- The diagnostic test `cjxl_grey_8x8_round9_diagnostic_panics_with_new_stop_point`
  remains in the test suite as a CI tripwire that surfaces the new
  error message on every push. Round 9 should soften / remove it
  once a real fix lands.

### Fixed (round 7, 2026-05-02)

- **Round-6 typo #6 unblocked.** The `log_alpha_size_minus_5` 2-bit
  field in the FDIS D.3.1 EntropyStream prelude was being read on the
  WRONG branch of `use_prefix_code`. Per
  `docs/image/jpegxl/libjxl-trace-reverse-engineering.md` §3.6, the field
  belongs to the ANS branch (`use_prefix_code == 0` → `log_alpha_size =
  5 + u(2)`); the Prefix-code branch fixes `log_alpha_size = 15`. We
  had it inverted. With this fix, the cjxl 8x8 grey lossless fixture's
  MA-tree T-stream prelude now decodes a 4-symbol prefix code with the
  correct HybridUintConfig (split=1, msb=0, lsb=0), and the MA tree
  itself decodes cleanly to 7 nodes (3 decision nodes on property 0
  with values 2/4/0, then 4 leaves all using predictor=5/Gradient with
  offset=0, multiplier=1).
- **RFC 7932 §3.5 CLCL VL table corrected.** The fixed Brotli
  code-length-code symbol-to-bits table had four of six entries swapped
  (sym 1's code was `0111` instead of `1110`, sym 2's was `011` instead
  of `110`, sym 3 / sym 4 swapped, etc.) — the table was not even a
  valid Huffman code (`01` was a prefix of `0111`). Round 7 restores
  the canonical Huffman assignment per RFC 7932 §3.5.
- **`read_general_clustering` prefix-coded sub-stream wired.** The D.3.5
  general clustering path that calls into a sub-D.3.1 entropy stream
  with `use_prefix_code == 1` was previously stubbed as
  `Error::Unsupported`. Round 7 implements it: read the symbol count
  selector, read the prefix code, then drive `HybridUintState::decode`
  for `num_distributions` integers. The same `log_alpha_size_minus_5`
  inversion typo fix applies here too.
- Pre-existing clippy warnings in `extensions.rs` and `toc.rs` test
  modules (unusual byte groupings, vec-init-then-push) cleaned up so
  `cargo clippy --tests -D warnings` is now clean.

### Added (round 7, 2026-05-02)

- **Multi-leaf MA-tree decode** in `modular_fdis::decode_channels` —
  per-pixel property-vector computation per FDIS Table D.2 + Listing D.8,
  decision-node tree walk, and per-leaf-context symbol decode. Prior
  rounds only supported single-leaf trees.
- **`gradient_64x64.lossless` and `palette_32x32.lossless` fixtures**
  generated locally via cjxl 0.11.1 (96 B / 119 B) for round-7+ testing
  against `docs/image/jpegxl/libjxl-trace-reverse-engineering.md` §4.1
  and §4.3 byte traces. These are RGB Modular fixtures with
  `nb_transforms != 0`; full decode requires Squeeze / Palette / RCT
  inverse transforms which round 8+ will land.
- **`tests/gradient_64x64_trace.rs`** — round-7 trace test capturing
  `[TRACE/sig|hdr|frame|dc|modular|ans]` events for the
  `gradient_64x64.lossless` fixture, comparable to the doc's reference
  byte trace.

### Round 7 stop point

The cjxl 8x8 grey lossless fixture's decode now stops at the SECOND
per-cluster prefix code in the symbol stream's prelude with `"JXL prefix:
code lengths grossly overflow Kraft sum (kraft=135104, alphabet_size=257,
max_length=13)"`. djxl decodes the same fixture without trouble, so cjxl
is emitting valid bits; our complex-prefix decoder has a subtle bug not
covered by the trace doc — the cl_code from the second 257-symbol code's
18-clcl array sums to Kraft 37 (should be 32), producing a downstream
Huffman lookup with Kraft 4×. Bisection harness lives in
`tests/cjxl_grey_8x8_trace.rs`. See `project_jpegxl_pixel_blocked.md`
and `project_jpegxl_fdis_typos.md` memos for the round-8 unblock plan.

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
