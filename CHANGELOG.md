# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
