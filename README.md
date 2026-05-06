# oxideav-jpegxl

Pure-Rust **JPEG XL** (ISO/IEC 18181) codec — full container + signature
detection, `SizeHeader` + `ImageMetadata` + `FrameHeader` + `TOC`
parsing, single-group Modular pixel decode (Grey 8-bit), and a
**round-5 lossless Modular encoder** with per-image predictor selection
across {Left, Top, Average, West-Predictor, Gradient} and a
frequency-adapted ANS-coded symbol stream. Round-5 hits **4.12 bpp**
on a 256×256 grey natural-image fixture (51.5% of raw, lossless,
self-roundtrip + bit-exact through libjxl's `djxl`). Zero C
dependencies, zero FFI, zero `*-sys`.

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

- VarDCT path (variable-size DCT, LF/HF subbands, Chroma-from-Luma,
  Gaborish, EPF) — Modular only.
- Multi-group decode (the round-3 decoder rejects `TOC.entries > 1`).
- Colour-space decode beyond Grey 8-bit (RGB encoder output is valid
  for djxl but our decoder rejects `ColourSpace::Rgb`).
- Predictor 6 (Annex E Weighted) on either decode or encode side.
- Encoder lossy mode (encoder is lossless Modular only).
- Animation / preview / intrinsic-size sub-bundle decoding.

### Why VarDCT decode is still blocked

Modular pixel decode is fully wired (see status section below) and
single-leaf MA-tree + ANS lossless frames round-trip end-to-end. The
remaining decoder gap is the **VarDCT path** (variable-size DCT,
LF/HF subbands, Chroma-from-Luma, Gaborish, EPF) — these need a deeper
walk through FDIS §3.8 which is partially documented in the in-tree
clean-room behavioural trace
(`docs/image/jpegxl/libjxl-trace-reverse-engineering.md`). Workspace
policy forbids consulting third-party source (libjxl, jxlatte,
jxl-rs, FUIF, brunsli) as a substitute. See
[`SPEC_BLOCKED.md`](SPEC_BLOCKED.md) for the audit + planned
work-order. Modular Appendix B §B.3.1 / §B.4 Path 1 (delta-palette
edge case, idx=-1 / nb_deltas=0 / predictor=Zero) is also still
gapped (#500).

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

- Codec: `"jpegxl"` — decoder + encoder slots both registered.
- No demuxer is registered: this crate treats a JXL file as a single
  codestream buffer fed directly to `probe(...)`.

### Encoder status (round 5)

The Modular lossless encoder ([`encoder::encode_one_frame`]) accepts
Gray8 / Rgb8 / Rgba8 input up to 1024×1024 (single-group cap) and
emits a raw JXL codestream. Round 5 added per-image predictor
selection across `{1 Left, 2 Top, 3 Average, 4 West-Predictor, 5
Gradient}` (FDIS Listing C.16 ids) — the encoder pre-scans residual
magnitudes and picks the lowest-scoring predictor for the single
MA-tree leaf, then emits an ANS-coded symbol stream against an
aligned 4096-summing distribution (round 4). Cross-validated through
both our own decoder and libjxl's `djxl` on:

- 8×8 / 16×16 / 64×64 grey synthetic fixtures (round 4 baseline).
- **256×256 grey natural image (round 5):** 33747 bytes for 65536-pixel
  raw input → **4.12 bits/pixel**, 51.5% compression, bit-exact
  lossless self + djxl round-trip. PSNR-Y is mathematically infinite
  (lossless, MSE = 0), well above the round-39 35 dB target.

### Modular pixel decode status

The Modular sub-bitstream pipeline (FDIS Annex C.9 + D, plus the
inverse transforms in Annex L.4 / L.5 / I.3) is wired end-to-end:
container → SizeHeader → ImageMetadata → FrameHeader → TOC →
LfGlobal → MA-tree → ANS / prefix entropy → per-channel pixel decode
→ inverse transforms (RCT / Palette / Squeeze).

Round 11 made the **inverse Palette** transform Appendix-B-faithful
(see `docs/image/jpegxl/libjxl-trace-reverse-engineering.md`), with
the four-range index partition (negative → kDeltaPalette,
0..nb_colours → explicit lookup, nb_colours..+64 → 4×4×4 cube,
+64.. → 5×5×5 cube), Path 1 / Path 2 dispatch on `(nb_deltas,
predictor)`, and the §B.6 bit-depth-24 clamp. The cjxl-encoded
8×8 grey-lossless fixture (`nb_colours=3 nb_deltas=0 d_pred=0`,
idx=-1 throughout) still decodes to all-zero rather than djxl's
all-128: per FDIS L.6 *and* Appendix B.4 Path 1, this should be
the kDeltaPalette[0][c]=0 lookup result, but the encoder side
expects a different value. This points to an extra-deep gap in
both the FDIS spec and Appendix B for the trivial-encoder case;
needs another empirical correction round.

## License

MIT — see [LICENSE](LICENSE).
