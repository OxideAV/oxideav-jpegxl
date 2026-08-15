# oxideav-jpegxl

[![CI](https://github.com/OxideAV/oxideav-jpegxl/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-jpegxl/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-jpegxl.svg)](https://crates.io/crates/oxideav-jpegxl) [![docs.rs](https://docs.rs/oxideav-jpegxl/badge.svg)](https://docs.rs/oxideav-jpegxl) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

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
comparison against four reference decodes. Round 393's **§F.3 HfMul
erratum fix** (HfMul is the per-varblock quantisation-precision
multiplier and *divides* on dequant, arbitrated externally on the
purpose-built `flat-content-lf-smoothing` fixture) collapsed every
VarDCT baseline: `vardct-256x256-d1` per-channel sRGB MAD
0.66 / 0.47 / 0.61 (was ≈ 3.4 — the round-385 "d1 HF accuracy tail"
is closed), `vardct-256x256-d3` 0.76 / 0.51 / 0.81, `large-1024x768-d2`
(12-group) 0.44 / 0.37 / 0.32, flat-content 0.20 / 0.20 / 0.20. The
same fixture resolved **§F.2 erratum candidate 4**: the corrected
`clamp(4·gap − 3, 0, 1)` smoothing ramp is the conformant reading
(CI-gated arbitration). Single-LfGroup frames of any group count
decode; §C.5 multi-LfGroup (> 2048 px) framing + LZ77 TOC permutations
landed round 393 and are pinned on `large-3072x2048-multigroup`.
Round 437 resolved the §C.7.1 `used_orders != 0` boundary for
single-preset single-pass frames (the Listing C.12 per-channel
permutation layout erratum — see below); multi-preset / multi-pass
§C.7 slices still refuse loudly. Multi-frame
codestreams compose per §C.2 (Reference slots + Table C.8 blending,
incl. round-393 kBlend / kAlphaWeightedAdd alpha modes) in
`decode_all_frames`. Programs that only need probe-level information
should call `probe(...)` directly.

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
  parse (`decode_splines` / `decode_splines_raw`) reads Listing C.3
  (num_splines, delta-coded start coords, then `quant_adjust` — the
  round-441 field-order erratum, see below) + per-spline control points
  (Listing C.4 `DecodeDoubleDelta`) and 4×32 DCT coefficients over the
  §D.3 six-distribution entropy stream, dequantizes (`dequant_dct32`,
  `kChannelWeight`) and recorrelates (`recorrelate_xb`,
  `Y × base_correlation_{x,b}`). The §K.3 render (`Spline::render` /
  `render_splines`) upsamples control points (`upsample_control_points`),
  resamples by unit arc length (`resample_by_arclength`), and additively
  splats an `erf`-based Gaussian brush (`s2s = √2·σ`,
  `maximum_distance = -2·ln(0.1)·σ²`) onto the XYB planes, evaluating each
  channel via `continuous_idct`. A suspected FDIS typo in the Listing K.1
  arc parameter is corrected (see below). **Wired into the registered
  decode path since round 441** (VarDCT and Modular-XYB frames, drawn
  after patches and before noise per §K.1) and pinned wire-level on a
  hand-assembled 43-byte codestream the reference decoder accepts —
  our render lands within ±1/255 of the black-box reference decode.

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

### Round 393 — flat-content fixture arbitration, alpha blending, multi-LfGroup

- **§F.3 HfMul erratum (the "d1 HF accuracy tail" closed).** The FDIS
  prose says the bias-adjusted quant "is then multiplied by … the
  value of HfMul" — but HfMul is the per-varblock
  quantisation-precision multiplier (§C.8.3 `qf`), so the decoder must
  DIVIDE. Arbitrated externally on the `flat-content-lf-smoothing`
  fixture (uniform HfMul = 13, near-empty HF band: the literal multiply
  produced ±30-code low-frequency garbage, MAD 2.67 → 0.20 with the
  division) and confirmed on every staged VarDCT fixture (d1
  3.42/1.99/2.10 → 0.66/0.47/0.61). Same divisor-vs-multiplier shape
  as the round-385 Listing C.1 erratum. Ratchets tightened
  (`round362…` 4.5 → 1.0/255; new flat-content 0.35/255 bound). The
  earlier round-385 corrections (Listing C.1 multipliers, Annex G CfL
  branch split + `-128` bias, Listing I.16 LLF normalisation) stand.
- **§F.2 erratum candidate 4 RESOLVED.** On the flat fixture (674/900
  interior LF samples at the `gap = 0.5` floor, where the two candidate
  ramps take opposite values) the corrected `clamp(4·gap − 3, 0, 1)`
  ramp beats the literal `max(0, 3 − 4·gap)` on every channel and
  matches ~740 more reference pixels exactly. CI-gated; the literal
  ramp stays reachable only through the per-thread arbitration hook.
- **Crate-side instrumentation** (the #168 fixture-notes deliverables):
  per-sample §F.2 LF trace (`lf_dequant::LF_SMOOTH_TRACE` — pre/post
  planes + per-sample gap/factor), the literal-ramp override, and the
  per-varblock decoded quantised HF-coefficient capture
  (`VARDCT_HF_COEFF_CAPTURE`).
- **§C.2 alpha blending**: kBlend (premultiplied + straight branches,
  0-alpha guard) and kAlphaWeightedAdd (post-blend alpha per the §C.2
  definitions paragraph) land in the composer; the alpha plane itself
  blends `oa + na·(1 − oa)`; Reference slots store the full plane
  stack; the multi-frame walk threads `alpha_plane` /
  `alpha_associated` / `ec_blending_info`.
- **§C.5 multi-LfGroup framing + §C.3.2 LZ77 TOC permutations**:
  permuted TOCs decode over the shared full-D.3 reader (cjxl
  large-image TOC permutations are LZ77-enabled) with §C.3.3
  permutation-aware offsets; §D.3.5 clustering accepts LZ77 nested
  sub-streams (depth-capped); per-LfGroup structures assemble into
  frame-level canvases and §F.2 smoothing runs frame-level. Pinned on
  `large-3072x2048-multigroup` (2×1 LF groups, 96 groups, permuted
  100-entry TOC) up to the §C.7.1 boundary below.

### Round 406 — ISO/IEC 18181-3 conformance corpus (Modular blending / layering)

Four of the six committed Part 3 conformance streams now decode
end-to-end, validated against black-box reference decodes
(`round406_conformance_composition`):

- **`alpha_nonpremultiplied`** (12-bit) and **`alpha_triangles`**
  (9-bit): **bit-exact** on all four channels.
- **`blendmodes`** — a five-frame `Reference[1]` chain exercising every
  Table C.8 blend mode (kReplace → kBlend → kAdd → kMul →
  kAlphaWeightedAdd) — within ±1/4095 (alpha exact at native depth).
- **`sunset_logo`** (RCT, 10-bit, orientation 7, two kBlend layers with
  out-of-canvas signed crops): **bit-exact** on all four channels at
  the correct transposed 924×1386 extent.

The fixes behind them (each an FDIS-text divergence pinned on the
official corpus): float-domain **unclamped** §C.2 composition
(`DecodedFrame::raw_f32` carries out-of-range Modular samples;
quantisation only at presentation), 9–16-bit plane composition,
kAlphaWeightedAdd weighting by the frame's own alpha with the alpha
channel left unchanged, signed (`UnpackSigned`) crop offsets with
§3.5.1 clipping, Table C.7 `alpha_channel`/`clamp` presence gated on
the blend mode alone (not `multi_extra`), §A.6 Table A.17 orientation
(all eight transforms, new `orientation` module), and the §5.2 **Idiv**
semantics (round toward zero) in the Listing C.16 averaging predictors
and the Listing I.21 Squeeze tendency function.

### Round 408 — ImageMetadata tail, ICC decode, §C.7.1 half-resolution, Squeeze + multi-LfGroup

- **Squeeze decodes end-to-end** (second block): the Listing I.19
  default-parameter sequence (derived at transform-application time;
  the printed `count` formula has a sign typo), the Listing I.21
  tendency erratum (refined round 420, below), and the Listing D.8
  `rleft = 0` column-0 rule. Single-group Squeeze is **bit-exact**
  (`round408_squeeze_multilf`).
- **§C.5.2 ModularLfGroup**: multi-LfGroup Modular frames decode — the
  `grayscale_public_university` conformance stream (2880×1620, 2 LF
  groups, Squeeze) went from hard-`Unsupported` to a full decode.

### Round 420 — the multi-group Squeeze tail CLOSED, restoration filters on Modular

- **Coded-domain forward-Squeeze oracle**
  (`round420_squeeze_residual_oracle`): the inverse Squeeze is a
  bijection, so forward-transforming a reference decode reconstructs
  the exact coded channel pyramid; comparing it sample-by-sample
  against the decoder's assembled pre-inverse Modular image pins every
  GlobalModular / per-LfGroup / per-PassGroup residual slice exactly.
- **Listing I.21 tendency half-tie erratum** (refines the round-408
  floor reading): the division rounds **half-away-from-zero** —
  `x = sign(m) × ((|m| + 6) Idiv 12)` for `m = 4A - 3C - B`. The two
  readings differ ONLY on exact negative half-ties (`m ≡ 6 mod 12`,
  ascending), which is precisely where every multi-group Squeeze
  stream diverged. **`sq_512` is now bit-exact** (was MAD 0.27); the
  round-408 "sporadic multi-group residual tail" was never a
  group-boundary issue.
- **Multi-LfGroup + weighted-predictor Squeeze pinned bit-exact**: new
  fixture `sq_2880x320_wp` (2 LfGroups, 12 PassGroups, > 1024-node
  all-predictor-6 global tree) decodes coded-domain and output
  bit-exact — WP state, group-seam borders and the §C.5.2 walk all
  verified in the coded domain. The D.4.2 tree-size cap now matches
  the spec bound (`(1 << 26)`; the old 1024 working cap rejected this
  real encoder output).
- **Listing I.18 in-place inverse-Squeeze pairing fix**: `r` stays
  constant through the c-loop (each merge removes `channel[r]`).
  The old `r + (c - begin)` mis-paired every in-place step with
  `num_c > 1` — invisible on grey pyramids, a hard error on the
  3-channel XYB default sequence. Lossy-modular XYB streams now run
  the full inverse (their RGB output mapping for Grey colour-space
  frames is still pending).
- **§J restoration filters wired into the registered Modular path**:
  Gabor-like transform + EPF now run on Modular frames that signal
  them. The `grayscale_public_university` stream (lossy Squeeze,
  gab=1, epf_iters=3) drops from MAD 1.68 to **1.00**; its Modular
  pyramid decode is verified fully in sync (all 87 modular
  sub-bitstreams end on the D.3.3 ANS final-state invariant `0x130000`
  — the residual is filter accuracy, not entropy or Squeeze).
- **§J.2 EPF weight-sign erratum**: the printed
  `4 × (sqrt(0.5) - 1) / sigma` is negative, making `Weight()`
  INCREASE with distance (the most dissimilar neighbours would get
  the largest weights) — contradicting J.3.1's normative "decreasing
  function" prose. The magnitude `4 × (1 - sqrt(0.5))` is the
  conformant reading (literal sign: MAD 6.3 on the same stream).
- **Docs-gap (filed)**: the §J.3 EPF sample-domain / channel-scale
  semantics for kModular non-XYB frames are underdetermined — under
  the literal signalled `epf_sigma_for_modular = 20` the EPF is a
  near-identity on 8-bit-scaled samples, while an (unjustified)
  sigma ≈ ×32 would minimise the residual at ≈ 0.42. The literal
  reading is shipped pending a behavioural trace.

- **The ImageMetadata-tail SPECGAP is resolved** (Table A.16): the
  `default_transform` Bool() is unconditional — present even when
  `all_default` — and its printed gating is inverted (bit set =
  "defaults, nothing follows"; bit clear reads `opsin_inverse_matrix`
  / `cw_mask` / the custom upsampling-weight arrays). A per-field
  metadata-tail gating trace (`metadata_fdis::METADATA_TAIL_TRACE`)
  pins the layout on every staged fixture.
- **ICC-bearing streams decode.** `enc_size = U64()` follows the end
  of ImageMetadata at the very next bit (no ZeroPadToByte(), despite
  the §B.2 "byte aligned" opener); Listing B.1's `IccContext` gained
  its missing ASCII-digit class; and the D.2.1 simple prefix code
  assigns the short code to the FIRST symbol as transmitted (only
  equal-length symbols sort). The 18181-3 `grayscale` stream's
  embedded 912-byte GRAY/ADBE profile decodes with every header field
  matching the reference tooling, and a synthesised digit-heavy
  profile embedded by a real encoder round-trips **byte-exactly**
  (`round408_icc_grayscale`).
- **§C.7.1 half-resolved**: the §C.3.2 `end` field is a Lehmer-entry
  count, not an endpoint (pinned: the grayscale stream codes
  `end = 0, skip = 4`, impossible under the endpoint reading). The
  decode advances past `DecodePermutation()` and now stops loudly in
  §C.7.2 — the permutation stream's exact end position is still
  underdetermined (the ANS final-state invariant fails on locally
  generated `used_orders` streams), so the grayscale frame itself
  remains refused, one boundary later than round 393.

### Round 437 — used_orders custom coefficient orders, kNoise, kModular EPF posture, multi-pass gate

- **§C.7.1 `used_orders != 0` streams DECODE — the Listing C.12
  per-channel permutation layout erratum.** The printed listing reads
  ONE `DecodePermutation()` per set `used_orders` bit; the wire
  carries THREE — one per colour channel, in the §C.8.3 decode
  sequence Y, X, B. Pinned by two independent oracles: the staged
  `patches-256x256` clean-room decode trace (under one-per-bit the
  §C.7.2 read starts 281 bits early and misparses; under
  one-per-channel it begins at the recorded position, parses to the
  recorded shape and the section lands on the trace's `AC_GLOBAL_END`
  to the bit) and the D.3.3 ANS final-state closure on ANS-coded
  specimens (fails under one-per-bit for every fdis-errata.md Part 8.3
  context/count grid combination — the prescribed six-way bisection
  was run to exhaustion first — and closes under one-per-channel).
  The 18181-3 `grayscale` conformance stream decodes past its
  round-393 boundary, and the frame-level **multi-pass gate is
  lifted** (pass-major §C.3.1 TOC walk + §C.8.3 cross-pass
  accumulation run end to end). Multi-preset / multi-pass §C.7 slices
  (the staged `progressive-ac-multipass` fixture, 3 passes × 2
  presets) still refuse loudly one boundary later.
- **Known limitation (ratcheted):** synthetic-edge content decodes
  structurally exactly (closure invariant; flat saturated regions
  byte-exact) but carries a high-detail VarDCT accuracy deficiency
  (MAD ≈ 20 on the `custom_orders_t256_e1` fixture vs sub-1 on photo
  content) that is INDEPENDENT of the permutation content — an open
  follow-up, bounded by `round437_custom_orders_decode`.
- **§K.4 kNoise decodes end to end** (`noise` module): §C.4.7 LUT
  parse in LfGlobal + per-group XorShift128Plus/SplitMix64
  pseudorandom channels + frame-level 5×5 convolution (§6.5
  mirroring) + Listing K.5 injection. Staged `noise-feature-256x256`
  fixture: sRGB MAD 0.92 / 0.79 / 0.88, max ≤ 7 — the same sub-1/255
  band as the noise-free VarDCT fixtures.
- **The §J.3-for-kModular SPECGAP is RESOLVED** by the in-crate grid
  bisection fdis-errata.md Part 9 prescribes: samples normalised to
  `[0, 1]` (reading N1) with a 1-channel Grey frame replicated into
  all three Annex J planes. `grayscale_public_university` MAD
  **1.00 → 0.2909** (max 21 → 8); the previously reported
  "sigma ≈ ×32 best fit at ≈ 0.42" is reproduced exactly by the
  normalised-domain grey-in-c0 grid cell — that fit was the missing
  domain normalisation. CI-gated arbitration
  (`round437_modular_epf_posture`).

### Round 441 — Patches + Splines wired; two new FDIS errata (§L.2 /128, Listing C.3 order)

- **§C.4.5 + §K.2 kPatches decodes and renders end to end** (`patches`
  module): the Listing C.2 dictionary parse (10-distribution §D.3
  stream, D.3.3 final-state guard) and the Table K.1 blending
  (kNone / kReplace / kAdd / kMul; alpha modes and extra-channel
  blending refuse precisely — no specimen exercises them). The §C.2
  plumbing that feeds it: Table C.3 **kReferenceOnly frames** decode
  and are skipped by the multi-frame walk ("not itself part of the
  image"), and `save_before_ct` recordings land in a walk-level
  **pre-CT `Reference[0..4]` store** (float-XYB for xyb frames,
  normalised samples for integer Modular frames; a pre-CT slot named
  as a §C.2 *blending* source refuses precisely). Pinned on three
  locally generated fixtures: two lossless Modular patch streams
  (60 single-position dot patches; 9×5-dict multi-position non-square
  glyph patches) decode **bit-exact** against black-box reference
  decodes, and a VarDCT+XYB sibling (Modular-XYB dictionary consumed
  in the pre-CT float-XYB domain) sits at MAD 1.7/0.8/0.7 — its
  residual is the round-437 impulse deficiency below, not patch error.
  The Listing C.2 `mode`-context question (printed ctx 5 vs the unused
  ctx 6) is **unarbitrable on available wire evidence** — every
  specimen's cluster map merges contexts 5 and 6 — so the printed
  reading ships, with a CI equivalence pin and a per-thread override.
- **§C.4.6 + §K.3 kSplines wired and wire-validated.** No encoder
  emits spline streams, so round 441 hand-assembles one bit-by-bit
  from the FDIS bundle tables (43 bytes; the builder lives in the
  round-441 test and must reproduce the committed fixture
  byte-for-byte). The reference decoder accepts it, and arbitrated a
  **NEW FDIS erratum — the Listing C.3 field order**: on the wire
  `quant_adjust` follows the starting-coordinate loop, not
  `num_splines` as printed. Under the printed order the reference
  decode places the spline at `y = sp_x`, starts x at 0, and scales
  the brush by exactly `1 + sp_y/8` (our second token consumed as a
  start coordinate, our fourth as `quant_adjust`); three independent
  geometry/σ probes all fit the corrected order. With it, our §K.3
  render (Catmull-Rom upsampling → arc-length resampling → erf brush,
  incl. the Part 3 K.1 arc-parameter correction) matches the
  reference decode to **max ±1/255**.
- **§L.2 kModular XYB rescale erratum — the ×`m` product divides by
  128.** The FDIS prose reads `X = X' × m_x_lf_unscaled` with no
  further scale; on real streams the literal reading saturates every
  sample (≈128× too large). Per-channel linear regression of the wire
  integers against black-box reference decodes of three independent
  lossy-Modular-XYB streams fits slope `m / 128` on every channel
  (±0.02 %, zero intercept). The Modular-XYB output path — previously
  never pixel-validated — now lands **max ±1/255** on all three
  fixtures; the same /128 makes the VarDCT patches fixture's XYB
  dictionary land correctly.
- **The round-437 "synthetic-content VarDCT accuracy deficiency" is
  sharply characterised** (not yet fixed): isolated impulse content
  (single-pixel dots) vanishes entirely on VarDCT frames with or
  without features. The dot blocks are Hornuss / DCT2×2 varblocks
  whose **declared NonZeros exceeds the decoded nonzero count** (e.g.
  raw 20, decoded 9, `remaining_non_zeros = 11` after the full k-walk
  — a silent D.3.3-class violation the §C.8.3 loop currently does not
  reject). The dedicated fixture generated this round reproduces it
  standalone; follow-up round material.

### Round 444 — the §C.8.3 entropy layer root-caused: impulse deficiency FIXED, two new FDIS errata (§F.3 2^16/global_scale, Listing I.4 orientation)

Round 444 took the round-441 impulse reproducer (Hornuss / DCT2×2
varblocks decoding fewer nonzeros than declared, `remaining_non_zeros
> 0` silently accepted) to root cause and found SIX distinct defects
stacked across the §C.8.3 entropy layer and the reconstruction chain,
each arbitrated black-box on purpose-built single-basis-function /
impulse probe streams (committed as the `r444_*` fixtures with their
reference decodes):

- **§C.8.3 reads are D.3.6 hybrid-integer reads** — the
  histogram-backed path returned raw entropy tokens, truncating every
  value ≥ the cluster's `split` and skipping its raw completion bits.
  Invisible while photo-content coefficients stayed below the split;
  a desyncing misparse on impulse content. This was the round-437/441
  deficiency's primary cause.
- **Per-section entropy-stream lifecycle (D.3.3)** — each PassGroup
  section is its own stream: the per-section `u(32)` ANS init was
  silently skipped for sections after the first (idempotency guard),
  and no terminal-state check existed. Sections now re-init, tear
  down, and check `state == 0x130000`, with two public per-thread
  diagnostics (`hf_coefficient_histograms::section_closure_failures`,
  `pass_group_hf::walk_underruns`) that CI pins per fixture — the
  desync states that rounds 437/441 accepted invisibly can never go
  silent again.
- **Listing I.4 IDCT orientation (FDIS erratum)** — the inverse-DCT
  pre-transpose belongs to the `C > R` branch only; running it for
  every shape (rounds 12..441) transposed the coefficient
  interpretation of square and tall blocks. Masked inside the photo
  sub-1/255 band, fatal on basis/impulse content: the reference
  decoder reproduces the encoded orientation, the pre-transposed
  reading its transpose. The forward `DCT_2D` helpers were re-derived
  to Listing I.3 literally.
- **§F.3 / §C.6.2 omit the global quantization scale (NEW FDIS
  erratum #5)** — the "final multiplier defined by the channel, the
  transform type and the coefficient index" also carries
  `2^16 / global_scale` (the §C.4.3 Quantizer field; the LF sibling
  is explicit in Listing C.1). Fit on five independently generated
  probe streams spanning `global_scale` 1022..10223 and HfMul 7/11:
  reference amplitude ratio ≡ `65536 / global_scale` on every stream
  (−2..−9 %, the sign and size of the Listing F.2 bias adjustment);
  independent of `quant_lf` (varied 15..23) and of HfMul beyond the
  round-393 §F.3 division, which the same data re-confirms.
- **Listing I.16 LLF normalisation (round-385 erratum refined)** —
  over the §I.2.1-normalised forward DCT, each LLF axis carries
  exactly the Listing I.15 `C(c, 8c, u)` boundary term (measured
  0.7871 / 0.9018 at u = 3 / 2 on Dct32x32 probes — the cosine
  products to four decimals). The round-385 "no factor at all"
  measurement was taken atop the transposed IDCT and the missing
  global scale, which masked it.
- **§C.7.1 per-channel permutation assignment is channel-index order
  X, Y, B (round-437 erratum refined)** — the assignment is invisible
  to every bit-position oracle (the three permutations are decoded
  back-to-back either way), so round 437's Y-X-B reading was never
  actually arbitrated; a 171-byte custom-orders impulse specimen
  (`r444_minidots`) decides it: under Y-first the Y-channel Hornuss
  corner coefficient lands on the wrong cell and the dots vanish,
  under index order the decode is reference-band exact.

Measured: the round-441 standalone impulse reproducer class decodes
at **max ±1/255** (`r444_onedot`, `r444_minidots`, `r444_basis32`,
`r444_basis64` — including a Dct64x64 walk with 569 declared nonzeros
and |q| ≈ 500), `flat-content-lf-smoothing` tightens to **max 1**,
`vardct-256x256-d3` 0.47/0.31/0.55 (was 0.76/0.51/0.81),
`large-1024x768-d2` 0.39/0.33/0.30, `noise-feature` 0.69/0.67/0.70
max 4 (was max 7), `patches_vardct` MAD 1.91/0.85/0.91.

### Not yet implemented

- **Multi-preset / multi-pass §C.7 slices with `used_orders != 0`**
  (the round-437 residual): after preset 0's per-channel Listing C.12
  bundles, the next preset's fields still misparse on the staged
  `progressive-ac-multipass` fixture — the per-preset repetition or
  per-pass slice layout hides one more wire divergence
  (`round437_custom_orders_boundary` pins the loud refusal).
- **A residual §C.8.3 entropy desync class** (round 444, replacing
  the fixed round-437/441 impulse deficiency): streams whose §C.7.2
  histograms carry near-uniform NON-DYADIC distributions (spectral
  leakage of non-bin-aligned content; also the r437
  `custom_orders_t256_e1` synthetic-edge stream and the committed
  photo `vardct-256x256-d1`) decode with a D.3.3 terminal-state miss
  and a bounded residual. The desync is now DIAGNOSED loudly (public
  `section_closure_failures` / `walk_underruns` counters, per-fixture
  CI pins on `r444_wave64` + `custom_orders_t256_e1`), never silent.
  Ruled out on the wire this round: `prev` semantics variants, `s`
  as the Table C.18 index, the Listing D.1 alias-pump stack/equality
  variants (all break other bit-exact streams), the NonZeros value,
  and the §C.8.3 writeback formula (the dangling `cur` in the FDIS
  prose is dead text — the printed uniform ceiling is
  wire-confirmed). The first divergent symbol on the minimal
  reproducer is a `prev = 1`-context read one symbol after a correct
  read; suspicion now rests on the large §C.7.2 cluster-map /
  histogram-prelude decode for these distribution shapes.
- The residual sub-1/255 VarDCT accuracy tail (float rounding + §J
  filter differences) and `save_before_ct` pre-CT reference recording
  in the §C.2 composer. (The staged `progressive-ac-multipass`
  fixture now exists and reaches the multi-preset §C.7 boundary
  above; the end-to-end multi-pass pixel pin lands when that boundary
  closes.)
- ColorEncoding / ToneMapping fuller decode, preview / animation /
  intrinsic-size sub-bundles (parsing stops cleanly at the `have_*`
  flags).
- The AFV non-DCT IDCT variants (parsed and dispatched; accuracy
  unvalidated — no staged fixture reaches them with a pixel oracle).
- Floating-point samples and `bps > 16`; high-bit-depth XYB / YCbCr.
- Surfacing the decoded ICC profile to callers (the Annex B decode
  runs and validates, but `oxideav_core::VideoFrame` has no ICC slot)
  and applying an embedded profile's transfer curve to the decoded
  samples (the `grayscale` stream's image output currently uses the
  signalled/default transfer, sRGB, rather than the profile's
  gamma-2.2-class `kTRC` curve).

- Output mapping for xyb_encoded Modular frames whose colour space is
  Grey (3 XYB channels → 1 grey plane): the Modular + inverse-Squeeze
  walk completes since round 420, the final XYB→grey hand-off is
  unwired and errors loudly.
- Patch alpha blend modes (kBlendAbove/Below, kAlphaWeightedAdd
  Above/Below) and extra-channel patch blending — parsed exactly,
  refused precisely at render (no specimen exercises them; the
  decode paths carry no extra planes there yet). Splines on non-XYB
  Modular frames (the §K.3 coefficients are XYB-domain quantities;
  domain undetermined) and kNoise on Modular frames stay refused.
- JPEG reconstruction, and the LfFrame (`lf_level > 0`) dimension
  scaling `progressive-dc` needs.
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
