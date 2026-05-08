# oxideav-jpegxl

Pure-Rust **JPEG XL** (ISO/IEC 18181-1:2024) decoder. Resumed
2026-05-08 against the final published 2024 core spec after the
trace-doc-driven rounds 7-11 + encoder rounds 1-6 were retired
(see "Why retired (history)" below). This crate currently ships:

- Round-1..3 baseline (pre-retire): signature + container detection,
  `SizeHeader` + full `ImageMetadata` (FDIS A.6 form), FrameHeader +
  TOC, the Annex C entropy stack (ANS + prefix codes + hybrid uint +
  LZ77 + clustering), LfGlobal + GlobalModular tree-prelude.
- **Rounds 1..5 (2024-spec)**: end-to-end Modular pixel decode for
  single-group, single-pass frames. Multi-leaf MA tree per Annex
  H.4.1 (16 base properties of Table H.4 + per-previous-channel
  properties), Table H.3 predictors 0..13, full H.5 self-correcting
  WP predictor, RCT / Palette / Squeeze inverse transforms (H.6),
  Grey + RGB output at 8 bpp. Five committed fixtures decode
  pixel-correct vs `expected.png` (PNG-decoder-backed byte-for-byte
  comparison): `pixel-1x1`, `gray-64x64`, `gradient-64x64-lossless`,
  `palette-32x32`, `grey_8x8_lossless`.
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

1. **Resolve C.2.5 alphabet_size SPECGAP** — close the round-7
   `alphabet_size > table_size` blocker so the committed
   `synth_320_grey/` fixture decodes pixel-correct.
2. **VarDCT decode** (Annex I) — start with `vardct-256x256-d1`
   then move to `vardct-256x256-d3` and `large-1024x768-d2`.
3. **XYB inverse colour transform** (§K) — needed for VarDCT
   downstream, plus some Modular images that use XYB.
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

## License

MIT — see [LICENSE](LICENSE).
