# oxideav-jpegxl

Pure-Rust **JPEG XL** (JXL, ISO/IEC 18181-1) decoder for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework. Built
clean-room from the published core specification and the conformance /
behavioural-trace fixtures committed under `docs/image/jpegxl/` only —
no external codec source is consulted. Zero C dependencies, zero FFI,
zero `*-sys`.

## Status

This crate is a **decoder under active construction**. The Modular path
decodes end to end (grey / RGB / RGBA, 1–16-bit integer, XYB / YCbCr
inverse colour) for the small lossless fixtures; the **VarDCT** path
decodes **on the public path** (round 389): the full chain — §C.8.3
per-(pass, group) HF-entropy decode → F.3 dequant → Annex G
coefficient-domain chroma-from-luma → §I.2.3.2 IDCT → §C.2 group
assembly → §6.2 crop → §J restoration filters → §L.2.2 XYB→RGB →
Table A.10 transfer encoding — is validated by direct sRGB byte
comparison against three reference decodes:
`large-1024x768-d2` (12-group multi-group frame) per-channel MAD
0.55 / 0.49 / 0.33, `vardct-256x256-d3` 0.89 / 0.70 / 0.95, and
`vardct-256x256-d1` ≈ 3.4 (the strongest-HF fixture; the residual HF
tail is the one open accuracy item, ratcheted by
`round362_vardct_d1_reference_divergence`). Single-LfGroup frames of
any group count decode; multi-LfGroup (> 2048 px) framing is the
remaining structural gap and surfaces a precise `Error::Unsupported`.
Multi-frame codestreams compose per §C.2 (Reference slots +
Table C.8 blending) in `decode_all_frames`. Programs that only need
probe-level information should call `probe(...)` directly.

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
  `vardct_reconstruct::reconstruct_lf_group_from_entropy` fuses that
  reconstruction with the **live** §C.8.3 multi-pass entropy decode
  (`multi_pass_decode::decode_multi_pass_three_channels_with_resolver`) in
  a single per-LfGroup call: it walks the DctSelect grid once per pass
  against the caller's entropy closures
  (`qdc_at` / `read_non_zeros` / `decode_symbol`), producing the per-pass
  `DecodedHfBlock` stack from the stream itself, then runs the cross-pass
  reconstruction on it — closing the "feed the live per-pass stack rather
  than a caller-supplied one" wiring step. It is bit-for-bit identical to
  the explicit decode-then-reconstruct two-call path. The
  **histogram-backed** sibling
  (`vardct_reconstruct::reconstruct_lf_group_from_histogram` over
  `HfHistogramDecodeContext::decode_lf_group_multi_pass_three_channels`)
  goes one step further: it owns the §C.7.2 entropy-stream routing
  itself — the per-pass `histogram_offset` selection, the per-pass
  per-channel `PredictedNonZeros` read + `NonZeros(x, y)` writeback — so
  the only entropy input the caller supplies is the storage-only `qdc_at`
  quantised-LF lookup (no `read_non_zeros` / `decode_symbol` closures).
  It is bit-for-bit identical to the closure path wired to the same
  histogram context over the same stream.
- **§C.7 HfGlobal-section assembly** — `hf_global_section::HfGlobalSection`
  reads the HfGlobal TOC slot as the three contiguous pieces the spec
  lays out on one bit cursor with no byte alignment between them:
  §I.2.4 dequant matrices + §I.2.6 `num_hf_presets`
  (`HfGlobal::read`) → §C.7.1 `num_hf_presets` `HfPass` coefficient-order
  bundles (`read_hf_pass_sequence`) → §C.7.2
  `495 × num_hf_presets × nb_block_ctx` HF-coefficient histograms
  (`HfCoefficientHistograms`) + the §C.3.2 ANS-state init (`u(32)`,
  a no-op for prefix streams). `nb_block_ctx` is threaded in from the
  LfGlobal `HfBlockContext` (§I.2.2). `HfGlobalSection::decode_context`
  binds the parsed §C.7.2 histograms to a per-frame §C.8.3
  `PerPassHfHeaders` to produce the
  `HfHistogramDecodeContext` (cross-validating every per-pass `hfp`
  against the section's authoritative `num_hf_presets`) — the bridge
  the per-LfGroup `reconstruct_lf_group_from_histogram` decode walks
  against. The integrated VarDCT decode path now parses through this
  full §C.7 section on a real codestream (`vardct_256x256_d1.jxl`).
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
- **§C.4.6 + §K.3 Splines image feature** — the self-contained `splines`
  module decodes and renders centripetal Catmull-Rom splines. The §C.4.6
  parse (`decode_splines` / `decode_splines_with`) reads Listing C.3
  (num_splines, `quant_adjust`, delta-coded start coords) + per-spline
  control points (Listing C.4 `DecodeDoubleDelta`) and 4×32 DCT
  coefficients over the §D.3 six-distribution ANS stream, dequantizes
  (`dequant_dct32`, `kChannelWeight`) and recorrelates (`recorrelate_xb`,
  `Y × base_correlation_{x,b}`). The §K.3 render (`Spline::render` /
  `render_splines`) upsamples control points (`upsample_control_points`),
  resamples by unit arc length (`resample_by_arclength`), and additively
  splats an `erf`-based Gaussian brush (`s2s = √2·σ`,
  `maximum_distance = -2·ln(0.1)·σ²`) onto the XYB planes, evaluating each
  channel via `continuous_idct`. A suspected FDIS typo in the Listing K.1
  arc parameter is corrected (see below). 25 unit tests + 3 end-to-end
  integration tests. **Not yet wired into the registered decode path** —
  that needs an f32-XYB plane hand-off at the §C.4 LfGlobal splines
  section plus a spline conformance fixture; the `lf_global.rs` splines
  rejection is the integration hook.

### Round 389 — multi-group / multi-pass framing, sRGB output, public exposure

- **Multi-group VarDCT framing** (§C.3.1 / §C.8.1): one PassGroup
  section per `(pass, group)` off the pass-major TOC slot map; per
  group the §C.8.1 group-local views (`group_rect` module: sub-grid
  slice with the §C.5.4 no-straddle invariant, LF rect, 64×64-aligned
  CfL tiles), the section's own `hfp` header + D.3.3 ANS state
  re-init, group-local NonZeros grids, pasted at the group offset.
  Landed reference-exact on first measure: per-pixel XYB MAD 7e-5 /
  1.4e-3 / 9e-4 on the 12-group fixture.
- **Multi-pass framing** (Table C.1 `hf_pass[num_passes]`): the
  HfGlobal slot reads one §C.7.1-orders + §C.7.2-histograms slice per
  pass; each `(pass, group)` section decodes as its own entropy
  stream and the per-pass stacks fold through the §C.8.3 cross-pass
  accumulator. (No multi-pass fixture is staged yet — unit-pinned.)
- **Table A.10 transfer encoding** (`xyb::TransferEncoder`): the
  §L.2.2 linear RGB is encoded with the signalled transfer function
  (sRGB / BT.709 / gamma / linear; PQ/DCI/HLG rejected precisely)
  before 8-bit quantisation — the rounds-11–385 linear-bytes SPECGAP
  left every XYB fixture uniformly dark (MAD ≈ 70/255).
- **FrameHeader `save_before_ct` presence** shares
  `save_as_reference`'s `!is_last` gate (fixture-measured against the
  2021 FDIS text; unblocked `vardct-256x256-d3`).
- **§C.2 frame composition** (`frame_compose`): Reference[1..=3]
  recording, `Reference[source]` lookup (zeros when unstored),
  crop-rect blending (kReplace / kAdd / kMul; alpha modes +
  pre-CT recording surface precisely), zero-duration frames composed
  but not presented.
- **D.3.5 clustering fix**: dropped the `num_distributions ≤
  bits_remaining` heuristic (ANS cluster indices cost ≪ 1 bit
  amortised; 2475 contexts in a 33-byte section are valid).
- **Real §C.8.3 `qdc[3]`**: the Listing C.13 `lf_thresholds` ladder
  reads the actual quantised-LF samples; the non-empty-`lf_thresholds`
  reject gate is gone.

### Not yet implemented

- **VarDCT HF accuracy tail on d1-quality streams.** Round 385 root-caused
  and fixed the long-pinned reference divergence with four
  fixture-measured FDIS-reading corrections (each recorded as an
  erratum candidate in the corresponding module doc): (1) **Listing
  C.1** — `mXDC = 65536 / (m_x_lf_unscaled × global_scale × quant_lf)`
  (`global_scale` is 16.16 fixed-point; the `m_*_lf_unscaled` F16
  values are divisors — the literal formula was off by `m²/65536` per
  channel: X 256×, Y 4×, B 1×); (2) **Annex G / Figure 2** — CfL is a
  coefficient-domain step with distinct branches (frame-global LF
  factors on the dequantised LF planes before Listing I.16; per-64×64
  `XFromY`/`BFromY` on the F.3-dequantised HF grids before the IDCT),
  plus the LF factor bias is `x_factor_lf - 128`, not `- 127`;
  (3) **Listing I.16** — the LLF block is the plain §I.2.1-normalised
  forward DCT of the LF block (the literal `× ScaleF` reading left
  every LLF AC cell off by exactly `ScaleF(8,64,u)` per axis);
  (4) **F.2 adaptive smoothing** — the factor ramp is
  `clamp(4·gap − 3, 0, 1)` (the literal ramp smooths real content
  hardest and preserves only quantisation noise). With those fixes plus
  the §J filters wired into the integrated path (Gaborish + per-block
  Listing J.3 EPF sigma from HfMul/Sharpness) the `vardct-256x256-d1`
  reconstruction matches the reference decode to per-channel sRGB MAD
  ≈ 3.3 / 1.9 / 2.1 (from ~105–129 railed at round 362), zero railed
  pixels, XYB frame-means equal to ~4 decimals
  (`round362_vardct_d1_reference_divergence` +
  `round385_vardct_xyb_accuracy` ratchets; internal XYB planes
  observable via the `VARDCT_XYB_CAPTURE` per-thread hook). Round 389
  narrowed the residual: d2-quality streams land at sRGB MAD < 1, and
  the remaining d1 divergence (post-filter XYB MAD ≈ 0.005 on Y,
  scaling with HF energy) sits in the strong-HF decode/filter tail —
  isolating it needs the still-pending per-coefficient trace (#168),
  since the reference PNG includes the §J filters. The §C.7.1
  signalled coefficient-order permutations are routed (all staged
  fixtures signal natural orders). Multi-LfGroup framing (frames
  wider/taller than 2048 px) is still pending, as are the alpha blend
  modes + `save_before_ct` reference recording in the §C.2 composer,
  and a progressive-AC (true multi-pass) fixture to pin the
  round-389 multi-pass framing end-to-end.
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

- Codec `"jpegxl"` — decoder slot registered; no encoder slot. The
  registered decoder handles the Modular path (grey / RGB / RGBA, 1–16-bit
  integer) and the VarDCT path (single-LfGroup frames of any group
  count, reference-validated; see Status).
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
