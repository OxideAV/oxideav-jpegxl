# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
