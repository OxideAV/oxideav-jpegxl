# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `SPEC_BLOCKED.md` records that the planned Modular-path pixel decoder
  (frame header / TOC / GroupHeader for Modular frames, MA-tree §H.5,
  JXL ANS §D, named predictors §H.4, squeeze §H.6) is blocked: the
  ISO/IEC 18181-1 normative spec is not present in `docs/image/jxl/`
  (the directory does not exist), the standard is paid-only via ISO /
  ANSI / accuristech, and workspace policy forbids consulting
  third-party source (libjxl, jxlatte, jxl-rs, FUIF, brunsli). The
  unblock procedure + planned work-order are documented in the file.
- 11 additional unit tests (20 → 31) covering pure-plumbing edge cases
  that do not require the spec:
  - `BitReader::read_bits(0)` is a no-op.
  - `BitReader::read_bits(33)` is rejected.
  - `BitReader::read_bits(32)` round-trips a 32-bit LE value.
  - `BitReader` returns `Error::InvalidData` on EOF.
  - `BitReader::bytes_consumed` correctly tracks partial-byte progress
    across `read_bits(4) + read_bits(4) + read_bit()`.
  - ISOBMFF `jxlp` partial-codestream box: 4-byte index prefix is
    stripped, payload survives.
  - Two `jxlp` boxes back-to-back concatenate in file order.
  - `jxlp` payload shorter than 4 bytes is rejected (no index room).
  - ISOBMFF 64-bit large-size box header (size32=1 → size64) decodes.
  - Truncated large-size box header is rejected.
  - Box claiming size > file length is rejected.

### Changed

- Scrubbed the libjxl source-file reference from `metadata.rs` module
  doc comment (clean-room hygiene). Replaced with a pointer to
  `SPEC_BLOCKED.md` and the normative ISO/IEC 18181-1 text.

## [0.0.4](https://github.com/OxideAV/oxideav-jpegxl/compare/v0.0.3...v0.0.4) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- drop Cargo.lock — this crate is a library
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- bump to oxideav-core 0.1.1 + codec 0.1.1
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
