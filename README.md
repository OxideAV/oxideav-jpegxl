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
  residual assembly (including the §I.2.4 LLF-coefficient placement that
  folds the LF-derived DC block into the natural-order low-frequency
  prefix before the §I.2.3.2 inverse DCT), the per-LfGroup three-channel
  residual-plane reconstruction, the §6.2 right/bottom crop that turns
  the padded block-grid reconstruction into the logical channel extent
  (`ResidualPlane::crop_to` / `ChannelResidualPlanes::crop_to`), and the
  inverse XYB / YCbCr colour transforms. The **non-square** transform
  families (DCT8×16 / DCT16×8 / DCT32×8 / DCT8×32 / DCT32×16 / DCT16×32
  and their larger relatives) reconstruct to spatial samples through the
  same walk — the IDCT carries the Listing I.4 pre/post-transpose for
  `R != C`, the LLF extraction reads a `cy × cx` sub-block, and the
  dequant matrix is the wide `bwidth × bheight` layout.
- **§C.8.3 cross-pass HF accumulation** — the multi-pass coefficient
  stack the per-pass decode driver yields (`out[p][i]`) is folded into a
  single accumulated quantised grid per varblock (`cross_pass`): each
  pass's HF coefficients are left-shifted by the Table C.6 `shift[i]`
  (last pass behaves as shift 0) and summed cell-wise, uniform across
  every transform family. `vardct_reconstruct::reconstruct_lf_group_cross_pass`
  is the one-call per-LfGroup driver tying it together — cross-pass
  accumulate → LF→LLF seed (Listing I.16) → F.3 dequant → §I.2.4 LLF
  merge → §I.2.3.2 IDCT → §C.5.4 placement → Annex G CfL — driving any
  mix of square / non-square / non-DCT varblocks (single- or multi-pass)
  to the three XYB residual planes.
- **§J.3 restoration filters** — the Gabor-like 3×3 convolution
  (`gaborish::apply_xyb_planes_in_place`) and the edge-preserving
  filter, both as pure XYB-plane math. The §J.3.1 three-step EPF
  iteration driver (`epf::apply_epf_iterations`) composes the
  up-to-three passes per `epf_iters`, feeding each step's output into
  the next (§J.3.4), for the constant-sigma (Modular,
  `epf_sigma_for_modular`) case. The §J.3.3 **VarDCT per-block-sigma**
  driver (`epf::apply_epf_iterations_per_block_sigma`) generalises it:
  each 8×8 block carries its own Listing J.3 sigma (packed into
  `epf::SigmaGrid`, looked up per reference pixel) and the
  `sigma < 0.3` block-skip (`epf::EPF_SKIP_SIGMA`) passes a block's
  pixels through unchanged. A uniform grid reduces bit-exactly to the
  constant-sigma path.

### Not yet implemented

- The integrated frame decode loop (FrameHeader + TOC + frame framing
  wiring the stages together); the registered decoder rejects until it
  lands. The per-LfGroup VarDCT reconstruction is now a single call
  (`vardct_reconstruct::reconstruct_lf_group_cross_pass`, covering
  square / non-square / non-DCT transforms and the §C.8.3 cross-pass
  accumulation); what remains is feeding it the live per-pass
  [`DecodedHfBlock`] stack from the §C.7.2 entropy stream rather than a
  caller-supplied one.
- ColorEncoding / ToneMapping fuller decode, preview / animation /
  intrinsic-size sub-bundles (parsing stops cleanly at the `have_*`
  flags).
- The AFV non-DCT IDCT variants, the §C.7.2 entropy-histogram wiring,
  Gaborish + EPF integration into the registered path. The VarDCT
  per-block EPF sigma (Listing J.3 from HfMul / Sharpness) and the
  `sigma < 0.3` block-skip now have a dedicated driver
  (`apply_epf_iterations_per_block_sigma` + `SigmaGrid`); deriving the
  per-block `HfMul`/`Sharpness` grids from the §C.5.4 HF pipeline and
  feeding them into that driver in the registered path is the
  remaining wiring step.
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
