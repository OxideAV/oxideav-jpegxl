# oxideav-jpegxl

Pure-Rust **JPEG XL** (ISO/IEC 18181) codec — currently
**RETIRED 2026-05-08 pending strict-isolation cleanroom workspace**.
The crate ships round-1..3 wiring only: signature + container
detection, `SizeHeader` + partial `ImageMetadata` parsing, the FDIS
Annex D ANS entropy module, and the LfGlobal + GlobalModular
tree-prelude. Pixel decoding past the tree-prelude returns
`Error::Unsupported`; there is no encoder.

## Why retired

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

A strict-isolation `docs/image/jpegxl-cleanroom/` workspace with the
four-role layout (Specifier / Extractor / Implementer / Auditor) —
Specifier wall: ISO/IEC 18181-1 FDIS + 18181-3 conformance corpus only,
no libjxl source ever — modelled after `docs/video/msmpeg4/`,
`docs/video/magicyuv/`, `docs/audio/tta-cleanroom/`. Until that
workspace exists, no further decoder rounds will land. Encoder rounds
will be re-authored on top of the cleanroom workspace once decoder
forward progress resumes.

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
