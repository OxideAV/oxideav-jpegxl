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
  Grey + RGB output at 8 bpp. **Round 29** lifts the fixture count
  from 5 to 6 by adding `alpha-64x64` (4-channel RGBA, extra-channel
  path per FDIS G.1.3 colour-then-extras + ExtraChannelInfo of type
  Alpha) and fixing the ISOBMFF jxlc payload `FF 0A` strip in
  `decode_one_frame`. **Round 30** lifts the fixture count from 6
  to 7 by adding `bit-depth-16` (3-channel RGB lossless Modular at
  `bits_per_sample = 16`) and adopts the LE-pack plane convention
  documented under "Plane byte layout" below. **Round 31** applies
  FDIS §F.3's section zero-pad rule uniformly to the
  single-TOC-entry LfGlobal fast path, so the
  `noise-64x64-lossless` fixture (`cjxl -d 0 -e 7`, high-entropy
  64×64 RGB lossless Modular, MA tree `leaves=84`) now
  decode-completes (vs hard-EOF pre-r31). **Round 32** bisects
  the residual pixel-divergence on that fixture to the
  Self-correcting weighted predictor at the first
  `predictor == 6` sample whose WP path uses `WW` and `NN` both
  as in-image values (i.e. `x >= 2 && y >= 2`); fix deferred
  pending a docs-collaborator libjxl-WP behavioural trace at the
  divergence point. Seven committed fixtures still decode
  pixel-correct vs `expected.png` (PNG-decoder-backed
  byte-for-byte comparison): `pixel-1x1`, `gray-64x64`,
  `gradient-64x64-lossless`, `palette-32x32`,
  `grey_8x8_lossless`, `alpha-64x64`, **`bit-depth-16`**.
- **Round 89 (2024-spec)** materialises the §I.2.4 / §I.2.5 +
  Table I.6 default dequantization-matrix set. New
  [`dct_quant_weights`] module transcribes the 2024 spec listing
  for `Mult`, `Interpolate`, `GetDCTQuantWeights`, the per-mode
  weights-derivation rules (DCT, DCT4, DCT2, Hornuss, DCT4x8,
  AFV) and the AFV Listing C.11 freqs/bands ladder. Public API
  exposes `materialise_default_dequant_set()` → the full 17-slot
  × 3-channel set (Table I.4 dims, element-wise reciprocal of
  the weights matrix). 26 new tests (15 unit + 11 integration);
  every cell of every channel of every slot is positive-finite
  per the spec's §I.2.4 last-paragraph invariant. Documented
  spec-listing typo notes (FDIS 2021 bands/weights nested-loop
  bug, corrected in 2024 published edition) and a SPECGAP for
  the DCT2 `(0, 0)` cell (not specified by the spec text;
  filled with `params(c, 0)` to keep the dequant reciprocal
  finite). Unblocks downstream HF coefficient dequantisation
  (§F.3) on the `u(1) == 1` HfGlobal default-encoding fast path.
- **Round 90 (2021-FDIS / 2024-spec) — HfPass + PassGroup HF
  structural parsers.** Three new modules surface the §C.7.1 /
  §C.7.2 HfPass bundle and the §C.8.3 PassGroup HF entry-points:
  * `coeff_order` — §I.2.4 natural coefficient ordering for every
    `OrderId` 0..=12 (Table I.1). Builds `LLF` prefix sorted by
    `y × bwidth + x`, then `HF` tail sorted by `(key1, key2)`
    per Listing I.14. Exposes `natural_coeff_order(OrderId)`,
    `varblock_size_for_order`, `coefficient_count`, and the
    `TransformType → OrderId` table.
  * `hf_pass` — §C.7.1 Listing C.12 parser. The `used_orders ==
    0` fast path materialises all 13 natural orders directly;
    `used_orders != 0` returns `Error::Unsupported` (the
    permutation reads need the shared 8-cluster ANS stream that
    §C.7.2 histograms also feed — round 91 work). Exposes
    `num_histogram_distributions = 495 × num_hf_presets ×
    nb_block_ctx` so the next round knows the §C.7.2 read
    count up-front.
  * `pass_group_hf` — §C.8.3 first line + Listing C.13. Reads
    `hfp = u(ceil(log2(num_hf_presets)))` and computes
    `histogram_offset = 495 × nb_block_ctx × hfp`. Verbatim
    transcriptions of `BlockContext`, `NonZerosContext`,
    `CoefficientContext`, `PredictedNonZeros`, plus the two
    64-element `CoeffFreqContext` /
    `CoeffNumNonzeroContext` ladder tables.

  49 new tests: 12 (`coeff_order`) + 7 (`hf_pass`) + 18
  (`pass_group_hf`) + 12 (integration
  `round34_hf_pass_pass_group_hf`). Unblocks downstream per-
  block coefficient decode loop (the `used_orders == 0` typed
  surface is now usable end-to-end; the `used_orders != 0`
  branch + shared-ANS-stream wiring is the round-91 task).
- **Round 95 (2021-FDIS / 2024-spec) — §F.3 HF dequantisation
  pure-math step.** New `src/hf_dequant.rs` glues the round-89
  `dct_quant_weights` 17-slot default dequant set to the
  round-90 `hf_pass` / `pass_group_hf` structural parsers via
  the FDIS Listing F.2 bias-adjust + per-block `HfMul`
  multiplier + `0.8^(qm_scale - 2)` per-channel factor.
  Public API: `bias_adjust(quant, channel, oim) -> f32`
  (Listing F.2 verbatim — `*= quant_bias[c]` for `|q| <= 1`
  branch, `-= quant_bias_numerator / quant` otherwise);
  `QmScaleFactors::for_frame(&FrameHeader)` (precompute the
  per-frame X / B factors once, Y is implicitly 1.0);
  `dequant_hf_coefficient(quant, channel, hf_mul,
  dequant_matrix_entry, oim, qm) -> f32` (full FDIS p. 72
  pipeline: bias-adjust → × `HfMul` → × qm-factor → × matrix
  entry); `dequant_hf_pre_matrix(...)` (partial product
  without the matrix entry, for callers that want to apply
  the dequant-matrix multiplication in a vectorised pass).
  23 new tests (13 unit + 10 integration
  `round35_hf_dequant`); cross-module composition pins the
  pipeline against `materialise_default_dequant_set()` for X
  and Y channels at the DCT8×8 corner cell, the FDIS default
  `quant_bias_numerator = 0.145` is fixed-point pinned at
  `quant = 2 → 1.9275`, and the `0.8^(scale - 2)` formula is
  swept over all 8 legal `u(3)` values for positive-finite
  output. The per-block ANS coefficient decode + indexing
  glue is still ahead of this step; round 95 lands the
  bit-exact arithmetic so a future round can drop the integer
  ANS reader on top without re-deriving any F.3 formulae.
- **Round 77 (2024-spec)** lands an audit-grade SPECDIFF harness
  for `docs/image/jpegxl/fixtures/animation-3frame/input.jxl` (3
  Regular Modular frames, `have_animation = 1`, encoded by cjxl
  0.12.0 against the 2024 final core spec). The probe-level path
  is correct (`probe_fdis` recovers SizeHeader + ImageMetadata
  with `have_animation = true` + AnimationHeader). The decode
  path remains blocked on a 2-bit format split between ISO/IEC
  18181-1:2021 FDIS Table C.9 (no leading `all_default` in
  RestorationFilter — what our `RestorationFilter::read`
  follows) and ISO/IEC 18181-1:2024 final Table J.1 (which
  prepends `all_default Bool()` + adds a `u(32) (ignored)`
  field). The seven small lossless fixtures were encoded by cjxl
  0.11.1 against the 2021 layout, so a uniform 2024-spec patch
  would break them; the audit doc-side recommendation is to
  re-encode those fixtures with cjxl 0.12.0+ before flipping.
  See `tests/r77_animation_3frame_specdiff.rs` module docs for
  the byte-level bit-trace bisect that pins the discrepancy
  down to a single byte boundary on the codestream's
  FrameHeader → TOC junction.
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

1. **Resolve C.2.5 alphabet_size SPECGAP** — round-8 partial
   resolution; round-9 lifted the synth_320 fixture to ~21k
   pixel-correct samples; remaining drift parked for an
   Auditor-mode bisect (see round-10 CHANGELOG).
2. **VarDCT decode** (Annex I) — round-8 lands the IDCT-8x8
   primitive + structural recognition; **round-11** wires the
   LfGlobal VarDCT bundles (Quantizer + HfBlockContext default
   table + LfChannelCorrelation) and the LfCoefficients
   sub-bitstream (per-LfGroup `extra_precision` + 3-channel
   modular decode at `ceil(group_dim/8)` resolution); **round-12**
   lands the spec-conformant 1-D + 2-D IDCT dispatch
   (`idct::idct_for_transform`) covering the 18 plain-DCT block
   sizes from Table C.16 (DCT8x8 through DCT256x256) per FDIS
   I.2.1 + I.2.2 Listing I.4; **round-13** extends the dispatch to
   the non-DCT IDCT helpers per Listings I.9.3..I.9.7 — `Hornuss`,
   `DCT2×2`, `DCT4×4`, `DCT8×4`, `DCT4×8` — via new public
   functions `aux_idct_2x2`, `idct_dct2x2`, `idct_dct4x4`,
   `idct_hornuss`, `idct_dct8x4`, `idct_dct4x8`. The four AFVn
   variants (Listing I.9.8) continue to return `Err(Unsupported)`
   pending an independently verified 256-entry `AFVBasis` table.
   Round 14+: PassGroup HF coefficient ANS decode + F.3
   dequantisation + AFV completion + Chroma-from-Luma + Gaborish +
   EPF.
3. **XYB inverse colour transform** (§L.2) — **landed round 11**.
   `xyb::inverse_xyb_to_rgb` and `xyb::inverse_ycbcr_to_rgb`
   transcribe FDIS Annex L.2.2 + L.3 verbatim; the modular output
   stage in `decode_codestream` now branches on
   `metadata.xyb_encoded` / `frame_header.do_ycbcr` and applies the
   inverse colour transform before mapping to `VideoFrame`. 9 unit
   tests + 6 integration tests including a forward-→-inverse
   round-trip oracle. Output gamma is left linear (downstream
   colour-management's job per §L.2.2 NOTE).
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

## Plane byte layout

`oxideav_core::VideoPlane` carries `(stride, data)` only — there is no
per-plane bit-depth field in core 0.1.x. The decoder therefore packs
samples into `data: Vec<u8>` according to the codestream's
`metadata.bit_depth.bits_per_sample` (FDIS Annex A.6 + Table A.22):

| `bits_per_sample` (`bps`) | Bytes / sample | Plane stride | Layout                              |
|---------------------------|----------------|--------------|-------------------------------------|
| `1 ..= 8`                 | 1              | `width`      | sample clamped to `[0, 2^bps - 1]`  |
| `9 ..= 16`                | 2              | `width × 2`  | **little-endian** `u16` per sample  |

Round 30 (2026-05) introduced the 16-bit row; the 8-bit row is the
pre-round-30 default. Floating-point samples (`bit_depth.float_sample
== true`) and `bps > 16` are not yet supported and surface as
`Error::Unsupported`.

The XYB (`metadata.xyb_encoded == true`) and YCbCr
(`frame_header.do_ycbcr == true`) inverse-colour-transform paths still
hard-require `bps == 8` because their dequantisation lattice is
calibrated against the 8-bit output range; high-bit-depth XYB / YCbCr
is round-31+.

A downstream consumer that wants to recover native `u16` samples from
a 16-bit plane does:

```rust
let samples: Vec<u16> = plane
    .data
    .chunks_exact(2)
    .map(|c| u16::from_le_bytes([c[0], c[1]]))
    .collect();
```

The convention deliberately mismatches PNG (RFC 2083 §2.1 specifies
big-endian 16-bit samples) so that on a little-endian host
`bytemuck::cast_slice::<u8, u16>(&plane.data)` is a zero-cost view.

## License

MIT — see [LICENSE](LICENSE).
