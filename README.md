# oxideav-jpegxl

Pure-Rust **JPEG XL** (JXL, ISO/IEC 18181-1) decoder for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework. Built
clean-room from the published core specification and the conformance /
behavioural-trace fixtures committed under `docs/image/jpegxl/` only —
no external codec source is consulted. Zero C dependencies, zero FFI,
zero `*-sys`.

## Status

This crate is a **decoder under active construction**. The integrated,
registry-driven decoder is **not yet wired end to end**: a registered
`make_decoder` returns `Error::Unsupported` because the codestream
framing (FrameHeader + TOC + frame-byte alignment) that ties the
per-stage machinery together is still in progress. Programs that only
need probe-level information should call `probe(...)` directly.

What is implemented and tested today:

- **Containers + signature detection** — both JXL wrappings: raw
  codestream (`FF 0A`) and the ISOBMFF box form
  (`00 00 00 0C 4A 58 4C 20 0D 0A 87 0A`), including extraction of the
  codestream from `jxlc` / `jxlp` boxes.
- **Codestream preamble** — an LSB-first bit reader (with the JXL `U32`
  selector encoding), `SizeHeader` (all four dimension encodings), and
  `ImageMetadata` up to `num_extra_channels` (bit depth, orientation,
  preview / animation presence flags).
- **Modular path primitives** — the adaptive range coder, the
  bounded-Exp-Golomb integer coder, the meta-adaptive decision tree, the
  named pixel predictors (including the Weighted predictor), and the
  per-channel decode loop. `modular::decode_single_channel` drives a
  single channel against a hand-built fixture, and individual stages
  decode pixel- / byte-exact against the staged behavioural traces.
- **VarDCT path primitives** — the LfGlobal bundles (Quantizer,
  HfBlockContext, LfChannelCorrelation), the LfCoefficients
  sub-bitstream, the spec-conformant 1-D / 2-D IDCT dispatch across the
  plain-DCT block sizes and the non-DCT helpers, per-block dequant +
  residual assembly, the per-LfGroup three-channel residual-plane
  reconstruction, and the inverse XYB / YCbCr colour transforms.

### Not yet implemented

- The integrated frame decode loop (FrameHeader + TOC + frame framing
  wiring the stages together); the registered decoder rejects until it
  lands.
- ColorEncoding / ToneMapping fuller decode, preview / animation /
  intrinsic-size sub-bundles (parsing stops cleanly at the `have_*`
  flags).
- The AFV non-DCT IDCT variants, the §C.7.2 entropy-histogram wiring,
  Gaborish + EPF integration into the registered path.
- Floating-point samples and `bps > 16`; high-bit-depth XYB / YCbCr.
- The encoder (not registered).

Unsupported inputs surface as `Error::Unsupported` rather than a silent
misparse.

### History

Earlier decoder and encoder work was reset off `master` in 2026-05 when
the behavioural-trace document it had been authored against was
withdrawn from `docs/` under fruits-of-the-poisonous-tree (the writeup
could not be guaranteed free of structural narrative carried from a
third-party implementation). Decoder work resumed against the published
core specification PDF, the conformance corpus, and the small lossless
fixtures committed under `docs/image/jpegxl/fixtures/`. Workspace policy
forbids consulting any third-party implementation source as a
substitute.

## Installation

```toml
[dependencies]
oxideav-core   = "0.1"
oxideav-codec  = "0.1"
oxideav-jpegxl = "0.0"
```

## Usage

```rust
use oxideav_jpegxl::{probe, Signature};

let bytes = std::fs::read("input.jxl")?;
let headers = probe(&bytes)?;

match headers.signature {
    Signature::RawCodestream => println!("raw .jxl codestream"),
    Signature::Isobmff       => println!("ISOBMFF-wrapped .jxl"),
}
println!("{}x{}", headers.size.width, headers.size.height);
println!("{} bits/sample, float={}",
    headers.metadata.bit_depth.bits_per_sample,
    headers.metadata.bit_depth.floating_point);
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Codec / container IDs

- Codec `"jpegxl"` — decoder slot registered (returns
  `Error::Unsupported` on instantiation until the integrated decode loop
  lands); no encoder slot.
- No demuxer is registered: a JXL file is treated as a single
  codestream buffer fed directly to `probe(...)`.

## Plane byte layout

`oxideav_core::VideoPlane` carries `(stride, data)` only — there is no
per-plane bit-depth field in core 0.1.x. The decoder packs samples into
`data: Vec<u8>` according to the codestream's `bits_per_sample`
(Annex A.6 + Table A.22):

| `bits_per_sample` (`bps`) | Bytes / sample | Plane stride | Layout                              |
|---------------------------|----------------|--------------|-------------------------------------|
| `1 ..= 8`                 | 1              | `width`      | sample clamped to `[0, 2^bps - 1]`  |
| `9 ..= 16`                | 2              | `width × 2`  | **little-endian** `u16` per sample  |

Floating-point samples and `bps > 16` are not yet supported and surface
as `Error::Unsupported`. The little-endian 16-bit convention lets a
little-endian host take a zero-cost `u16` view of the plane:

```rust
let samples: Vec<u16> = plane
    .data
    .chunks_exact(2)
    .map(|c| u16::from_le_bytes([c[0], c[1]]))
    .collect();
```

## License

MIT — see [LICENSE](LICENSE).
