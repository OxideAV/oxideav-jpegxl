//! JPEG XL (JXL) codec — decoder-side header parsing.
//!
//! JPEG XL is ISO/IEC 18181 (final specification 2022). It supersedes
//! classic JPEG with a modal design that separates a "VarDCT" path
//! (variable-size DCT + LF/HF subbands, quality-competitive with AVIF
//! and modern JPEG) from a "Modular" path (grid-of-pixels predictor +
//! MA-tree range coder, strong at lossless + non-photo material).
//!
//! This crate currently ships:
//!
//! * Container + signature detection for both JXL wrappings:
//!   raw codestream (`FF 0A`) and ISOBMFF-wrapped
//!   (`00 00 00 0C 4A 58 4C 20 0D 0A 87 0A`), including extraction of
//!   the codestream from `jxlc` / `jxlp` boxes.
//! * An LSB-first [`bitreader::BitReader`] matching the reference
//!   bit packing used by the codestream.
//! * Parsing of the codestream preamble: [`metadata::SizeHeader`] and the
//!   [`metadata::ImageMetadata`] fields up to `num_extra_channels`
//!   (bit depth, orientation, preview/animation flags). Fuller
//!   ColorEncoding + ToneMapping decoding is deferred.
//! * Modular sub-bitstream pixel decode (per the 2019 committee draft,
//!   Annexes C.9 + D.7), made of:
//!   - [`abrac::Abrac`] — the bit-level adaptive range coder (D.7);
//!   - [`begabrac::Begabrac`] — bounded-Exp-Golomb integer coder over a
//!     known signed range (D.7.1);
//!   - [`matree::MaTree`] — the meta-adaptive decision tree that picks
//!     a per-context BEGABRAC for each pixel (D.7.2 / D.7.3);
//!   - [`predictors`] — the five named pixel predictors (Zero, Average,
//!     Gradient, Left, Top) from C.9.3.1;
//!   - [`modular`] — the channel header parser plus the per-pixel
//!     property + predictor + entropy decode loop.
//!
//! The integrated registered decoder is not yet wired: the registered
//! `make_decoder` reports [`Error::Unsupported`] because the surrounding
//! codestream framing (FrameHeader + TOC + frame-byte alignment) is not
//! yet wired to the per-channel path. Programs that only need
//! probe-level information (dimensions, bit depth) should call
//! [`probe`] directly; programs that want to drive the per-channel
//! Modular decode end-to-end should instantiate
//! [`modular::decode_single_channel`] against a hand-built fixture
//! (unit tests in `modular` show the expected wire format).
//!
//! Follow-up work (tracked for the eventual landing PR):
//!
//! * GlobalModular wiring (C.4.8) so the FDIS path can actually drive
//!   the Modular sub-bitstream end-to-end.
//! * Squeeze inverse transform (I.3) for multi-resolution Modular
//!   images.
//! * VarDCT-path decoder (variable-size DCT + LF/HF, Chroma-from-Luma,
//!   Gaborish smoothing, EPF) — out of scope for this round.
//! * MABrotli / MAANS entropy coders (the 2019 committee draft's
//!   `entropy_coder` ∈ {1, 2}); only MABEGABRAC (`entropy_coder == 0`)
//!   is implemented today.
//!
//! ## FDIS 18181-1:2021 layer
//!
//! In addition to the committee-draft pipeline above, the FDIS layer
//! is being built up additively across rounds:
//!
//! * Round 1: [`ans`] — FDIS Annex D entropy decoder (prefix codes,
//!   ANS, distribution clustering, hybrid integer coding).
//! * Round 2: [`extensions`] — A.5 Extensions; [`metadata_fdis`] —
//!   full A.6 ImageMetadata refresh including ColorEncoding,
//!   ToneMapping, ExtraChannelInfo, AnimationHeader, OpsinInverseMatrix,
//!   PreviewHeader; [`frame_header`] — C.2 FrameHeader bundle including
//!   Passes, BlendingInfo, RestorationFilter; [`toc`] — C.3 TOC with
//!   Lehmer-code permutation decoder driven by the round-1 ANS layer;
//!   [`ans::cluster::read_general_clustering`] — D.3.5 general path.
//! * Round 3 (planned): GlobalModular wiring + cjxl fixture decode.
//!
//! ## Round-1 (2024-spec) status (this commit)
//!
//! `make_decoder` returns a live decoder ([`JxlDecoder`]) that handles
//! the simplest end-to-end Modular bitstreams:
//!
//! * Raw codestream OR ISOBMFF wrapping.
//! * Grey (1 plane) OR RGB (3 planes), 8 bits per sample (integer).
//! * Single-group, single-pass frame (`num_groups == 1 &&
//!   num_passes == 1`).
//! * `nb_transforms` arbitrary at the *parse* level (TransformInfo
//!   bundles per H.7 are decoded for any nb_transforms > 0); inverse
//!   application of Palette / Squeeze defers to round 2 with a clean
//!   `Error::Unsupported` exit point. RCT (no channel-list change)
//!   passes through the layout step.
//! * Multi-leaf MA tree evaluated end-to-end (decision-node
//!   `property[k] > value` traversal per H.4.1).
//! * `use_global_tree` is honoured.
//! * No Patches / Splines / NoiseParameters — those are LfGlobal
//!   features round 2 will land alongside the VarDCT path.
//! * No ICC profile (Annex E.4).
//! * Predictor 6 (Annex H.5 Self-correcting) only resolved at the
//!   (0, 0) origin; full WP defers to round 2.
//!
//! The acceptance fixture for round 1 is `pixel-1x1.jxl` (1×1 RGB
//! lossless, 22 B): decodes to R=255 G=0 B=0 matching its
//! `expected.png`.
//!
//! Anything outside this envelope returns
//! [`Error::Unsupported`](oxideav_core::Error::Unsupported) at the
//! relevant gate point. Wider coverage (VarDCT, Squeeze inverse,
//! Palette inverse, ICC, full WP predictor 6) lands in round 2+.
//!
//! ## Round-6 (2024-spec) additions
//!
//! * **Annex E.4 ICC profile decode** ([`icc`]): the 7-state-equivalent
//!   entropy-coded ICC byte stream (41 pre-clustered distributions +
//!   `IccContext(i, b1, b2)` 41-context function) is decoded into the
//!   final ICC profile bytes per E.4.3 (header), E.4.4 (tag list) and
//!   E.4.5 (main content). When `metadata.colour_encoding.want_icc ==
//!   true` the bit-position is now correctly advanced past the ICC
//!   stream rather than failing with `Error::Unsupported` outright;
//!   the decoded bytes are validated for the "acsp" magic at offset 36
//!   but are not yet propagated to `oxideav_core::VideoFrame` (which
//!   has no ICC slot in 0.1.x).
//! * **G.2 LfGroup / G.4 PassGroup type scaffolding** ([`lf_group`],
//!   [`pass_group`]): typed bundles + per-group rectangle geometry +
//!   `(minshift, maxshift)` computation per pass. Per-LfGroup and
//!   per-PassGroup decode itself is not yet wired (round-7 follow-up
//!   gated on the GlobalModular `nb_meta_channels`-aware refactor —
//!   see `lf_group` crate-level docs).
//! * Multi-LfGroup / multi-group / multi-pass / VarDCT frames fail
//!   with precise round-7-targeting error messages instead of the
//!   round-3 generic "TOC with N entries" rejection.
//!
//! ## Round-7 (2024-spec) additions
//!
//! Four-piece refactor coordinating the GlobalModular partial-decode
//! path with per-PassGroup decode + post-PassGroup transforms (Annex
//! G.1.3 last paragraph + G.4.2):
//!
//! * **Partial GlobalModular** — [`global_modular::GlobalModular::read`]
//!   stops decoding at any non-meta channel exceeding `group_dim`
//!   (G.1.3 last paragraph). Such channels are zero-filled placeholders
//!   in `image.channels` until per-PassGroup decode fills them.
//! * **`stream_index` threading** —
//!   [`modular_fdis::decode_channels_at_stream`] takes the stream index
//!   from Table H.4: `0` for GlobalModular,
//!   `1 + 3*num_lf_groups + 17 + num_groups * pass_idx + group_idx` for
//!   ModularGroup. Threaded through `get_properties` so the MA tree's
//!   `property[1] > value` decisions select the correct per-section
//!   leaf.
//! * **TOC layout + empty entries** — [`toc::Toc::read`] now accepts
//!   zero-size entries (e.g. an empty LfGroup or PassGroup section is
//!   legal when no channel matches that section's filter). The
//!   `decode_codestream` consumer addresses sections by their TOC
//!   offsets (computed from the entry running sum), with permutation
//!   already handled in the round-2 TOC reader.
//! * **Post-PassGroup transforms** —
//!   [`global_modular::apply_inverse_transforms`] is invoked AFTER all
//!   PassGroups complete (G.4.2 last paragraph), not inside
//!   `GlobalModular::read`, so the inverse transform sees the
//!   fully-assembled image rather than a half-decoded one.
//!
//! Per-PassGroup decode is in
//! [`pass_group::decode_modular_group_into`]; the
//! `(minshift, maxshift)` computation in [`pass_group::compute_pass_shift_range`]
//! models an implicit `n=num_ds` final-resolution entry that the
//! printed spec text omits but whose absence would make single-pass
//! frames decode no modular data (documented SPECGAP).
//!
//! **Round-7 SPECGAP** — cjxl 0.11.1 emits multi-group lossless modular
//! fixtures where the per-cluster ANS distribution's `alphabet_size`
//! exceeds `1 << log_alphabet_size` (specifically: alphabet_size=33
//! against table_size=32 when `log_alphabet_size = 5 + u(2) = 5`). The
//! 2024 spec text in C.2.5 is silent on the cap (the introductory
//! paragraph describes D as a `1 << log_alphabet_size`-element array
//! but the listing's alphabet_size-iterating loop can exceed it).
//!
//! ## Round-8 (2024-spec) additions
//!
//! Two themes:
//!
//! 1. **C.2.5 SPECGAP partial resolution** ([`ans::distribution`]):
//!    [`ans::distribution::read_distribution`] now returns
//!    `(D, log_eff)` where `log_eff` is the effective log_alphabet_size
//!    for downstream alias-table sizing. Round 8 picks
//!    "interpretation C": iterate the logcounts loop for
//!    `min(alphabet_size, table_size)` entries, treating the
//!    bitstream's signalled `alphabet_size > table_size` as a
//!    soft cap (the encoder advertises a wider alphabet but only
//!    serialises `table_size` per-symbol entries). Empirically
//!    validated by parsing the LfGlobal section of
//!    `tests/fixtures/synth_320_grey/synth_320.jxl` cleanly past
//!    the round-7 SPECGAP error. Interpretations A (grow D to
//!    accommodate alphabet_size) and B (drop writes at i >=
//!    table_size, accumulate total_count only over stored entries)
//!    were both tried and rejected — see [`ans::distribution`]
//!    crate docs for the comparison. The synth_320 fixture is
//!    still NOT decoded end-to-end: a separate post-LfGlobal blocker
//!    appears (cjxl emits a 0-byte PassGroup[0][0] slot which
//!    contradicts the spec's "all groups carry data per pass"
//!    rule); that is round-9+ work.
//!
//! 2. **VarDCT scaffold** ([`vardct`]): the FrameHeader's
//!    `encoding == kVarDCT` path is now structurally recognised
//!    rather than rejected with a generic `Error::Unsupported`.
//!    The module exposes
//!    [`vardct::recognise_vardct_codestream`] which validates the
//!    round-8 envelope (single LF group, single pass, no extra
//!    channels, Grey or RGB colour space) and returns a
//!    [`vardct::VarDctScaffold`] geometry record. The IDCT-II
//!    primitive for the 8x8 block size ([`vardct::idct1d_8`] +
//!    [`vardct::idct2d_8x8`]) is also wired with unit tests. End-
//!    to-end VarDCT pixel decode (LF subband, HF subband, dequant,
//!    inverse transform dispatch across block sizes 8x8 / 8x16 /
//!    16x8 / 16x16 / 32x32 / 64x64 / DCT4 / IDENTITY / AFV,
//!    Chroma-from-Luma, Gaborish smoothing, EPF) is round-9+
//!    work.
//!
//! ## Round-9 (2024-spec) additions
//!
//! Three concurrent fixes that together unblock the synth_320 fixture
//! (multi-group lossless grey, 320×320, num_groups=9):
//!
//! 1. **§F.3.1 HfGlobal slot is unconditional** — the 2024 spec
//!    bullets list `HfGlobal` for every TOC, with NOTE 1 calling out
//!    that the slot is 0-byte for `encoding == kModular`. Round 8's
//!    `num_toc_entries` / [`toc::Toc::read`] gated HfGlobal on
//!    `encoding == kVarDCT`, off-by-oning every PassGroup index in
//!    multi-group kModular frames. Also: `HfPass[num_passes]` is part
//!    of the `HfGlobal` section per Annex G.3 Table G.4 — round 8 had
//!    incorrectly counted it as separate TOC entries. With both off-
//!    by-ones fixed, synth_320's TOC reads as 12 entries
//!    `[33, 0, 0, 9, 20, 7, 20, 9, 24, 7, 23, 7]` (slot 2 is the 0-
//!    byte HfGlobal, not PG[0][0]).
//!
//! 2. **§F.3 first-paragraph zero-padding** — "When decoding a
//!    section, no more bits are read from the codestream than 8 times
//!    the byte size indicated in the TOC; if fewer bits are read,
//!    then the remaining bits of the section all have the value
//!    zero." Round 8's [`bitreader::BitReader`] errored on EOF for
//!    section sub-readers, breaking PassGroup ANS decodes whose
//!    modular sub-bitstreams consumed fewer real bits than the
//!    TOC-stated section size. Round 9 adds
//!    [`bitreader::BitReader::new_section`] which returns 0 for any
//!    read past the end of the section data; the legacy
//!    [`bitreader::BitReader::new`] preserves strict EOF for whole-
//!    codestream parsing.
//!
//! 3. **Per-PassGroup transforms (Annex H.6 inside G.4.2)** —
//!    observed in cjxl 0.11.1's synth_320 edge groups: the encoder
//!    emits a per-group Palette transform (`begin_c=0, num_c=1,
//!    nb_colours=191`) for the 64-pixel-wide column-2 / row-2 groups.
//!    [`pass_group::decode_modular_group_into`] now applies the
//!    transform layout adjustment to the per-group channel descs,
//!    decodes against the adjusted descs, and applies the inverse
//!    transforms LOCALLY before copying samples back into the parent
//!    image. [`global_modular::apply_transforms_to_channel_layout`]
//!    is now `pub` so the per-group reuse path doesn't duplicate the
//!    table.
//!
//! **Round-9 status** — synth_320 reaches end-of-frame without
//! erroring and ~21k of 102400 pixels match the expected
//! `(y + x) & 0xFF` gradient (the first 6 rows across the first two
//! group columns); the remaining pixels drift mid-decode in the
//! smaller edge groups. Full pixel-correctness is round-10 work
//! (suspected residual: ANS state nuance specific to F.3 zero-
//! padded tail OR per-group WP bookkeeping). All five small
//! lossless fixtures still pixel-correct vs round 4's
//! `expected.png`.
//!
//! ## Round-10 (2024-spec) additions
//!
//! Two themes:
//!
//! 1. **C.1 + C.3.3 `lz_dist_ctx` spec-conformance fix** —
//!    [`modular_fdis::decode_uint_in`] and `decode_uint_in_with_dist`
//!    previously passed the per-symbol leaf context for both the
//!    literal token AND the LZ77 distance token, which contradicts
//!    the spec's "the LZ77 distance token is read using
//!    `D[clusters[lz_dist_ctx]]`" with `lz_dist_ctx = num_dist`
//!    (the dedicated extra context the codestream reserves whenever
//!    `lz77.enabled`). When LZ77 fires, that bug would distort
//!    every copy. Fixed: derive `lz_dist_ctx = cluster_map.len() -
//!    1` from the post-prelude state of the `EntropyStream` and
//!    thread it to `HybridUintState::decode`'s `ctx_lz` argument.
//!    No-op for fixtures whose symbol stream has `lz77.enabled =
//!    false` (synth_320 included).
//!
//! 2. **synth_320 edge-group drift bisect** — instrumented per-
//!    decode tracing pinpoints the first mismatch at PG[0][0]
//!    decode #3087 (frame coords y=24, x=14). State 0x9CA780
//!    alias-maps to symbol 30 (cluster 0's low-prob `D[30] = 1`
//!    entry), forcing an ANS refill plus extra bits that
//!    over-consume 21 bits beyond the 9-byte slot. Bisect ruled
//!    out: per-PassGroup transform layout (PG[0][0] carries no
//!    transforms; only edge groups do); LZ77 path (off in the
//!    symbol stream); per-channel WP state reset (PG[0][0] is the
//!    first group); cluster_map / `dist_multiplier` derivation
//!    (matches H.3). Round-11+ work will need a finer state-by-
//!    state diff against djxl `--debug` (deferred to an Auditor
//!    round) since the implementer wall bars djxl source / the
//!    reference-decoder-trace doc.
//!
//! **Round-10 status** — synth_320 still decodes ~21k of 102400
//! pixels matching the gradient (first 24 rows of PG[0][0] and
//! PG[0][1] are pixel-correct; drift begins at y=24, x=14). All
//! five small lossless fixtures still pixel-correct.
//!
//! ## Round-11 (2024-spec) additions
//!
//! Three pieces wire LF subband decode (Annex G.2.2 / I.2):
//!
//! 1. **LfGlobal VarDCT bundles** ([`lf_global`]):
//!    [`lf_global::Quantizer`] (FDIS C.4.3 — `global_scale` +
//!    `quant_lf`) drives LF dequant per Listing C.1.
//!    [`lf_global::LfChannelCorrelation`] (C.4.4) carries the CfL
//!    factors used by Annex G to reconstruct X/B from dY (default
//!    `colour_factor=84`, `base_correlation_x=0.0`,
//!    `base_correlation_b=1.0`). [`lf_global::HfBlockContext`]
//!    (C.8.4) implements only the `u(1)==1` default-table fast path
//!    in round 11; the per-LF-threshold / qf-threshold / clustering-
//!    map branch returns `Error::Unsupported`. With these bundles
//!    wired, `LfGlobal::read` advances correctly past the VarDCT-only
//!    region of the LfGlobal slot rather than rejecting outright.
//!
//! 2. **GlobalModular zero-channel acceptance**
//!    ([`global_modular`]): `GlobalModular::read` previously rejected
//!    any frame where `derive_channel_descs` returned 0 channels (the
//!    common VarDCT-without-extras case). Round 11 accepts the empty
//!    case: the inner ModularHeader (`use_global_tree`, `WPHeader`,
//!    `nb_transforms`) is still consumed, but the MA-tree and per-
//!    cluster distribution decode are skipped per FDIS C.9.1 last
//!    sentence ("In the trivial case where N is zero, the decoder
//!    takes no action.").
//!
//! 3. **LfGroup + LfCoefficients** ([`lf_group`]):
//!    [`lf_group::LfCoefficients::read`] reads `extra_precision = u(2)`
//!    per FDIS C.5.3, builds the per-LfGroup channel layout (3 LF
//!    channels of `ceil(group_w/8) × ceil(group_h/8)` samples,
//!    optionally further right-shifted by `frame_header.jpeg_upsampling`
//!    on chroma planes), then drives a Modular sub-bitstream with
//!    `stream_index = 1 + lf_group_index` per Table H.4.
//!    [`lf_group::LfGroup::read`] composes ModularLfGroup (G.2.3 —
//!    round-11 only handles the empty-channel-list case for now)
//!    with LfCoefficients. HfMetadata (G.2.4) is round-12+ work.
//!
//! Acceptance fixture: hand-built minimal VarDCT bitstream — no cjxl
//! dependency, encoded directly from spec listings — covering an
//! 8×8 frame with 1×1 LF coefficient channels, MA tree of one
//! Zero-predictor leaf, and prefix-code symbol stream with
//! alphabet_size=1 (so every decoded LF coefficient is 0). The
//! fixture parses through `LfGlobal::read` → `LfGroup::read` →
//! `LfCoefficients::read` end-to-end. Test:
//! `lf_group::tests::round11_lfgroup_minimal_vardct_one_block_parses`.
//!
//! All five small lossless fixtures stay pixel-correct (see
//! `tests/round11_lf_subband.rs`).
//!
//! ## Round-13 (2024-spec) additions
//!
//! Three pieces tighten the VarDCT pipeline so round-12's unit-tested
//! F.1 / F.2 work actually runs on real codestreams:
//!
//! 1. **DctSelect / HfMul derivation from BlockInfo** ([`dct_select`]):
//!    walks each column of the per-LfGroup BlockInfo channel decoded
//!    in round 12, looks up the transform type in Table C.16, and
//!    places the varblock at the next-empty 8×8 cell of the LfGroup's
//!    block grid (raster order). HfMul is computed as `1 + mul`. The
//!    27-entry transform-type table is committed verbatim with
//!    per-entry `(block_cols, block_rows)` from the FDIS spec.
//!
//! 2. **HfGlobal C.6 default-fast-path** ([`hf_global`]): reads the
//!    `u(1)` dequant-default flag and the `num_hf_presets - 1 =
//!    u(ceil(log2(num_groups)))` field. The non-default-encoding
//!    branch (per-matrix `encoding_mode = u(3)` + Listing C.7
//!    `ReadDctParams()`) returns `Error::Unsupported` until round 14+
//!    wires the full table.
//!
//! 3. **VarDCT pipeline wiring** ([`decode_vardct_round13`]): the
//!    top-level `decode_one_frame` no longer rejects VarDCT
//!    codestreams at the round-8 scaffold gate. Instead, for
//!    `num_lf_groups == 1 && num_passes == 1`, it now drives:
//!    LfGlobal → LfGroup (LfCoefficients + HfMetadata) → DctSelect
//!    derivation → HfGlobal → F.1 LF dequantisation → F.2 adaptive
//!    smoothing (when not skipped). The round-13 pipeline returns
//!    `Error::Unsupported` with a "round 14+: HF subband decode +
//!    IDCT not yet wired" message AFTER all round-12 work has run on
//!    the real input.
//!
//! Round-13 status — five small lossless Modular fixtures stay
//! pixel-correct; both VarDCT fixtures (`vardct_256x256_d1.jxl` and
//! `vardct_256x256_d3.jxl`, copied from `docs/image/jpegxl/fixtures/`)
//! reach the round-13 pipeline (no longer hit the round-8 scaffold
//! gate).
//!
//! Round-14 candidates (in dependency order):
//!
//! * HfBlockContext non-default-table branch (per-LF thresholds + qf
//!   thresholds + clustering map), required for any cjxl-encoded VarDCT
//!   fixture that doesn't take the `u(1)=1` default-table fast path.
//! * HfGlobal C.6.2 dequant-matrix encoding modes (Listing C.7) +
//!   Listing C.10 `GetDCTQuantWeights` for per-DctSelect dequant
//!   matrices.
//! * HfPass C.7.1 coefficient orders (`used_orders` 13-bit mask,
//!   `DecodePermutation`) + C.7.2 histograms (495 × num_hf_presets ×
//!   nb_block_ctx clustered distributions).
//! * PassGroup HF coefficients C.8.3: per-block `hfp =
//!   u(ceil(log2(num_hf_presets)))` + clustered ANS coeff decode +
//!   F.3 HF dequantisation (Listing F.2 + per-channel scale +
//!   per-DctSelect dequant matrix multiply).
//! * Inverse DCT dispatch across block sizes (8×8 IDCT wired round 8;
//!   8×16 / 16×8 / 16×16 / 32×32 / 64×64 / DCT4 / DCT4×8 / DCT8×4 /
//!   IDENTITY / AFV remain).
//! * Listing I.5 LLF-from-downsampled-LF composition (the bridge from
//!   F.2-smoothed LF samples to varblock LF coefficients) — pure-math
//!   step landed round 121 as [`llf_from_lf`] (FDIS Listings I.15 +
//!   I.16). Still pending: per-LfGroup wiring that drives the
//!   per-varblock invocation from the [`pass_group_hf`] coefficient
//!   buffer.
//! * Chroma-from-Luma (Annex G), Gaborish (Annex J?), EPF.
//!
//! ## Round-16 (2024-spec) additions
//!
//! [`lf_group::HfMetadata::read`] now wires nested transforms (FDIS
//! §C.5.4 + §C.9.4): the four-channel HfMetadata sub-bitstream parses
//! `nb_transforms` + `TransformInfo[]` exactly like the GlobalModular
//! section, applies the transform-rewritten channel layout via
//! [`global_modular::apply_transforms_to_channel_layout`] before the
//! inner per-channel decode, then walks
//! [`global_modular::apply_inverse_transforms`] in reverse bitstream
//! order to recover the four-channel base layout
//! `[XFromY, BFromY, BlockInfo, Sharpness]`.
//!
//! Round-15 left the d1 fixture stuck on the round-12 deferral inside
//! `HfMetadata::read` (`nb_transforms > 0` errored with "transforms
//! inside HF metadata sub-bitstream not yet supported"). With round 16
//! the parse succeeds; the d1 fixture surfaces a strictly-later
//! blocker — its HfMetadata Squeeze step references channels beyond
//! the four-channel baseline (`begin_c=39` on step 0), tripping the
//! `apply_transforms_to_channel_layout` channel-count invariant.
//! That's the round-17 candidate (suspected upstream bit-position
//! drift in LfGlobal or LfCoefficients).
//!
//! `HfMetadata::read` and `LfGroup::read` now both take a
//! `&ImageMetadataFdis` argument so the inverse Palette transform can
//! read `bit_depth.bits_per_sample` for delta-palette prediction.
//!
//! ## Round-26 (2024-spec) — Annex L colour transforms
//!
//! Parent-dispatch "r11". New [`xyb`] module transcribes FDIS §L.2.2
//! inverse XYB → linear RGB and §L.3 inverse YCbCr → RGB verbatim
//! from the ISO/IEC 18181-1:2024 PDF. Three public entry points:
//! [`xyb::inverse_xyb_to_rgb`], [`xyb::inverse_ycbcr_to_rgb`], and
//! the convenience composite [`xyb::modular_xyb_to_linear_rgb`]
//! (§L.2.2 preamble + inverse XYB in one call).
//!
//! Wired into [`decode_codestream`] modular output stage: when
//! `metadata.xyb_encoded == true` and `colour_encoding.colour_space`
//! is `Rgb`, the per-channel pass-through is replaced with
//! [`build_rgb_planes_from_xyb`]; symmetric branch for
//! `frame_header.do_ycbcr == true`. Pre-round-26 pass-through path
//! preserved for `xyb_encoded == false && do_ycbcr == false` modular
//! frames so the five small lossless fixtures stay pixel-correct.
//!
//! Round-26 SPECGAP: §L.2.2 NOTE describes the output as
//! linear-domain RGB but doesn't prescribe a gamma encoding step
//! before display. [`xyb::linear_rgb_to_u8`] emits linear bytes
//! (clamp + scale by 255 + round); callers that need sRGB-encoded
//! bytes apply the sRGB transfer function downstream.
//!
//! ## Round-27 (2024-spec) — IDCT dispatch
//!
//! Parent-dispatch "r12" item (5). New [`idct`] module wires the
//! spec-conformant 1-D inverse DCT for power-of-two sizes
//! `s ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256}` (FDIS Annex I.2.1) and
//! the 2-D inverse DCT (Annex I.2.2 Listing I.4) for rectangular
//! `R × C` blocks. Three public entry points: [`idct::idct_1d`],
//! [`idct::idct_2d`] (taking coefficients in spec `(short × long)`
//! row-major natural-ordering layout per Annex I.2.4 and returning
//! `(R × C)` row-major samples), and [`idct::idct_for_transform`]
//! which dispatches on a [`dct_select::TransformType`] to the 2-D
//! IDCT for the 18 plain-DCT block sizes in Table C.16.
//!
//! The 9 non-DCT transforms (Hornuss, DCT2x2, DCT4x4, DCT4x8,
//! DCT8x4, AFV0..AFV3) — Listings I.7..I.13 — return
//! `Err(Unsupported)` from [`idct::idct_for_transform`] and are
//! deferred to round 28+ alongside HF coefficient decode + F.3
//! dequantisation. The legacy [`vardct::idct1d_8`] /
//! [`vardct::idct2d_8x8`] (round-8 scaffold, scaled-orthonormal
//! IDCT) are retained for backward compatibility but are NOT
//! spec-conformant; new HF-decode wiring will call through
//! [`idct::idct_for_transform`] exclusively.

pub mod abrac;
pub mod afv;
pub mod ans;
pub mod begabrac;
pub mod bitreader;
pub mod block_context_resolver;
pub mod block_dequant;
pub mod chroma_from_luma;
pub mod coeff_order;
pub mod container;
pub mod cross_pass;
pub mod dct_quant_weights;
pub mod dct_select;
pub mod epf;
pub mod extensions;
pub mod frame_header;
pub mod gaborish;
pub mod global_modular;
pub mod hf_coeff_histogram_size;
pub mod hf_coefficient_histograms;
pub mod hf_dequant;
pub mod hf_global;
pub mod hf_global_section;
pub mod hf_pass;
pub mod icc;
pub mod idct;
pub mod lf_dequant;
pub mod lf_global;
pub mod lf_group;
pub mod llf_from_lf;
pub mod matree;
pub mod metadata;
pub mod metadata_fdis;
pub mod modular;
pub mod modular_fdis;
pub mod multi_pass_decode;
pub mod multi_pass_hf_header;
pub mod multi_pass_hf_histogram_decoder;
pub mod non_zeros_grid;
pub mod pass_group;
pub mod pass_group_hf;
pub mod per_channel_non_zeros;
pub mod per_pass_non_zeros;
pub mod predictors;
pub mod residual_plane;
pub mod toc;
pub mod varblock_walk;
pub mod vardct;
pub mod vardct_reconstruct;
pub mod xyb;

pub use container::{detect, extract_codestream, Signature};
pub use metadata::{parse_headers, BitDepth, Headers, ImageMetadata, SizeHeader};

use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Error, Result};
use oxideav_core::{
    CodecInfo, CodecRegistry, Decoder, Encoder, Frame, Packet, RuntimeContext, VideoFrame,
    VideoPlane,
};

use crate::bitreader::BitReader;
use crate::frame_header::{FrameDecodeParams, FrameHeader};
use crate::lf_global::LfGlobal;
use crate::metadata_fdis::{ColourSpace, ImageMetadataFdis, SizeHeaderFdis};
use crate::toc::Toc;

/// Public codec id string. Matches the aggregator feature name `jpegxl`.
pub const CODEC_ID_STR: &str = "jpegxl";

/// Register the JPEG XL decoder stub into the supplied
/// [`CodecRegistry`]. The encoder slot is intentionally left
/// unregistered: the crate is decoder-side only and currently
/// retired-pending-cleanroom (see crate-level docs).
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("jpegxl_headers_only")
        .with_lossy(true)
        .with_intra_only(true);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder),
    );
}

/// Unified entry point: install the JPEG XL codec into a
/// [`RuntimeContext`].
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
}

oxideav_core::register!("jpegxl", register);

fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let codec_id = params.codec_id.clone();
    Ok(Box::new(JxlDecoder {
        codec_id,
        pending: None,
        eof: false,
    }))
}

/// Round-1 (2024-spec) JXL decoder. Drives `decode_one_frame` per packet.
///
/// Limitations (round 1):
/// * Only Modular-encoded frames (kModular, not kVarDCT).
/// * Grey (1ch) OR RGB (3ch) only — XYB / YCbCr defer.
/// * Single-group, single-pass frames.
/// * Inverse Palette / Squeeze transforms defer (parsing + RCT
///   layout pass-through is wired).
/// * Predictor 6 (Self-correcting) only at (0, 0) origin.
/// * No Patches / Splines / Noise / ICC profile.
///
/// Anything outside this envelope returns `Error::Unsupported` from a
/// well-defined point in the bitstream rather than panicking.
struct JxlDecoder {
    codec_id: CodecId,
    pending: Option<Packet>,
    eof: bool,
}

impl Decoder for JxlDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "jxl decoder: receive_frame must be called before sending another packet",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            };
        };
        let vf = decode_one_frame(&pkt.data, pkt.pts)?;
        Ok(Frame::Video(vf))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

/// Decode the ICC stream (Annex E.4) at the current bit position and
/// return the resulting ICC profile bytes.
///
/// The caller has already verified that
/// `metadata.colour_encoding.want_icc == true`. Round 6 wires the
/// decode end-to-end; the returned bytes are valid per E.4.3..E.4.5 if
/// `Ok`. The function also performs a minimal ICC.1 sanity check —
/// for outputs >= 40 bytes the magic "acsp" must be at offset 36 —
/// because the predicted-header rule in E.4.3 forces those bytes when
/// the encoded delta is zero, but a malformed delta could shift them.
fn decode_icc_stream_at(br: &mut BitReader<'_>) -> Result<Vec<u8>> {
    let encoded = icc::decode_encoded_icc_stream(br)?;
    let profile = icc::reconstruct_icc_profile(&encoded)?;
    if profile.len() >= 40 && &profile[36..40] != b"acsp" {
        return Err(Error::InvalidData(format!(
            "JXL ICC: decoded profile lacks 'acsp' magic at offset 36 (got {:02X?})",
            &profile[36..40]
        )));
    }
    Ok(profile)
}

/// Decode the entire JXL packet (raw codestream OR ISOBMFF-wrapped) and
/// return the first frame as a [`VideoFrame`]. Round-3 envelope.
pub fn decode_one_frame(input: &[u8], pts: Option<i64>) -> Result<VideoFrame> {
    decode_first_frame(input, pts, false)
}

/// Decode the first frame, returning the integrated VarDCT
/// reconstruction's pixels **without** the public path's pixel-withhold
/// gate. Exposed for the crate's integration tests and offline tooling
/// that want to exercise the full §C.8.3 → §L.2.2 VarDCT chain on a real
/// codestream and inspect its output.
///
/// For non-VarDCT (Modular) frames this is identical to
/// [`decode_one_frame`]. For VarDCT frames it returns the reconstructed
/// RGB frame instead of the "pixels not yet validated" sentinel error
/// that [`decode_one_frame`] returns — see [`decode_vardct_frame`] for
/// the pixel-validation caveat.
pub fn decode_vardct_frame_from_codestream(input: &[u8], pts: Option<i64>) -> Result<VideoFrame> {
    decode_first_frame(input, pts, true)
}

/// Shared container-strip + codestream dispatch for [`decode_one_frame`]
/// and [`decode_vardct_frame_from_codestream`]. `return_vardct_pixels`
/// selects whether a successful VarDCT reconstruction returns its
/// (not-yet-pixel-validated) frame or the public-path withhold sentinel.
fn decode_first_frame(
    input: &[u8],
    pts: Option<i64>,
    return_vardct_pixels: bool,
) -> Result<VideoFrame> {
    let sig = container::detect(input)
        .ok_or_else(|| Error::InvalidData("jxl decoder: no JXL signature".into()))?;
    match sig {
        container::Signature::RawCodestream => {
            decode_codestream(&input[2..], pts, return_vardct_pixels)
        }
        container::Signature::Isobmff => {
            // The jxlc/jxlp box payload concatenation is itself a JXL
            // codestream and therefore begins with the 2-byte `FF 0A`
            // codestream signature (FDIS Annex B.1). Skip those 2 bytes
            // before handing off to `decode_codestream` (which expects
            // bits *after* the signature, matching the raw-codestream
            // entry point above). Without this strip the SizeHeader
            // parse below would misalign by 16 bits and cascade into
            // corrupted FrameHeader/TOC reads.
            let codestream_owned = container::extract_codestream(input)?;
            let cs: &[u8] = &codestream_owned;
            if cs.len() < 2 || cs[0] != 0xFF || cs[1] != 0x0A {
                return Err(Error::InvalidData(
                    "JXL ISOBMFF: jxlc/jxlp payload missing FF 0A codestream signature".into(),
                ));
            }
            decode_codestream(&cs[2..], pts, return_vardct_pixels)
        }
    }
}

fn decode_codestream(
    codestream: &[u8],
    pts: Option<i64>,
    return_vardct_pixels: bool,
) -> Result<VideoFrame> {
    let mut br = BitReader::new(codestream);

    // 1. SizeHeader (FDIS A.3).
    let size = SizeHeaderFdis::read(&mut br)?;

    // 2. ImageMetadata (FDIS A.6).
    let metadata = ImageMetadataFdis::read(&mut br)?;

    // 3. ICC profile (Annex E.4) — round-6 lands the decoder. The
    //    decoded ICC bytes are validated (must contain "acsp" magic at
    //    offset 36 if length >= 40) but not currently propagated to
    //    `VideoFrame` because `oxideav_core::VideoFrame` has no ICC
    //    slot. The decode is still run because (a) it advances the
    //    bit reader past the ICC stream so subsequent FrameHeader /
    //    TOC parsing finds the right bit offset, and (b) it gives a
    //    direct `Error::InvalidData` if the codestream's ICC stream
    //    is malformed.
    if metadata.colour_encoding.want_icc {
        let _icc_bytes = decode_icc_stream_at(&mut br)?;
    }

    // 4. Byte-align before frame data per FDIS 6.3.
    br.pu0()?;

    // 5. FrameHeader (FDIS C.2).
    let fh_params = FrameDecodeParams {
        xyb_encoded: metadata.xyb_encoded,
        num_extra_channels: metadata.num_extra_channels,
        have_animation: metadata.have_animation,
        have_animation_timecodes: metadata
            .animation
            .map(|a| a.have_timecodes)
            .unwrap_or(false),
        image_width: size.width,
        image_height: size.height,
    };
    let fh = FrameHeader::read(&mut br, &fh_params)?;

    // 6. TOC (FDIS C.3) — entries byte-aligned per spec.
    let toc = Toc::read(&mut br, &fh)?;

    // 7. Single-group frames have a single TOC entry containing all
    //    frame data. Round 6 only handled that case; round 7 wires
    //    multi-group via per-section bit readers, with inverse
    //    transforms applied AFTER all PassGroups complete (G.4.2).
    let num_groups = fh.num_groups();
    let num_lf_groups = fh.num_lf_groups();
    if num_lf_groups > 1 {
        return Err(crate::lf_group::unsupported_multi_lf_group_error(
            num_lf_groups,
            fh.encoding,
        ));
    }
    // Diagnostic on unhandled features. Round 13 wires LfGlobal +
    // LfGroup (incl. LfCoefficients + HfMetadata) + HfGlobal + F.1 LF
    // dequant + F.2 adaptive smoothing into the VarDCT pipeline. End-
    // to-end pixel decode (HF coefficient subband + IDCT dispatch +
    // CfL + restoration filters) is round-14+ work — the fast path
    // below errors with a precise round-14 message AFTER consuming
    // the LfGlobal/LfGroup/HfGlobal sections + computing the
    // dequantised LF samples per Listing F.1 + applying F.2 smoothing
    // when `kSkipAdaptiveLFSmoothing == 0`.
    if fh.encoding == crate::frame_header::Encoding::VarDct {
        let scaffold = crate::vardct::recognise_vardct_codestream(&fh, &metadata)?;
        // The integrated VarDCT decode (`decode_vardct_frame`) now runs
        // the whole §C.8.3 HF-entropy → F.3 dequant → IDCT → CfL →
        // §6.2 crop → §L.2.2 XYB→RGB chain end-to-end on a real
        // codestream. Genuine parse errors (malformed sections, an
        // unhandled sub-case) propagate verbatim. A *successful*
        // reconstruction is NOT yet surfaced from the public decode
        // path: the per-block HF coefficient scaling has not been
        // validated bit-exact against a reference decode, so returning
        // its pixels would be a silent-misparse risk (the very thing
        // this crate's "no silent misparse" contract forbids). The
        // reconstruction is instead exercised structurally by the
        // crate's integration tests against `decode_vardct_frame`
        // directly. Once the per-block scaling is pixel-validated this
        // branch returns the frame.
        let frame = decode_vardct_frame(&fh, &metadata, &toc, &mut br, scaffold, pts)?;
        if return_vardct_pixels {
            return Ok(frame);
        }
        return Err(Error::Unsupported(
            "jxl VarDCT decoder: the integrated reconstruction (HF entropy decode + \
             F.3 dequant + IDCT + CfL + §6.2 crop + §L.2.2 XYB→RGB) runs end-to-end on \
             this codestream, but per-block HF coefficient scaling is not yet \
             pixel-validated against a reference decode — the public path withholds \
             unvalidated pixels rather than risk a silent misparse"
                .into(),
        ));
    }
    if fh.encoding != crate::frame_header::Encoding::Modular {
        return Err(Error::Unsupported(format!(
            "jxl decoder: encoding {:?} not supported",
            fh.encoding
        )));
    }
    if fh.width == 0 || fh.height == 0 {
        return Err(Error::InvalidData("jxl decoder: zero-dim frame".into()));
    }

    // Map TOC entries to byte ranges (post-permutation order). Each
    // section starts byte-aligned and runs `entries[i]` bytes. The
    // bit reader is currently aligned to a byte (TOC consumed); the
    // first section begins at the current byte offset.
    let frame_data_start = br.bytes_consumed();
    let codestream_data = br.data();
    if frame_data_start > codestream_data.len() {
        return Err(Error::InvalidData(
            "JXL decoder: frame data start past codestream end".into(),
        ));
    }
    let frame_bytes = &codestream_data[frame_data_start..];
    // Validate total length against TOC sum.
    let total_frame_len: u64 = toc.entries.iter().map(|&e| e as u64).sum();
    if total_frame_len > frame_bytes.len() as u64 {
        return Err(Error::InvalidData(format!(
            "JXL decoder: TOC declares {total_frame_len} frame bytes but only {} remaining",
            frame_bytes.len()
        )));
    }
    // Compute per-section start offsets in the *bitstream* order from
    // the running sum. The TOC permutation has already been applied to
    // `entries` and `group_offsets` so they're in the order the spec
    // says the sections appear on the wire (LfGlobal first, etc.).
    let mut section_starts: Vec<usize> = Vec::with_capacity(toc.entries.len());
    let mut acc: u64 = 0;
    for &e in &toc.entries {
        section_starts.push(acc as usize);
        acc = acc.saturating_add(e as u64);
    }
    let section_byte_range = |idx: usize| -> Result<&[u8]> {
        let start = section_starts[idx];
        let len = toc.entries[idx] as usize;
        let end = start + len;
        if end > frame_bytes.len() {
            return Err(Error::InvalidData(format!(
                "JXL decoder: section {idx} byte range [{start}..{end}) exceeds frame bytes ({})",
                frame_bytes.len()
            )));
        }
        Ok(&frame_bytes[start..end])
    };

    // Slot index helpers per ISO/IEC 18181-1:2024 §F.3.1 TOC layout
    // (round-9 fix: HfGlobal slot is unconditional, 0-byte for
    // kModular; HfPass is part of HfGlobal, NOT separate slots):
    //   slot 0       — LfGlobal
    //   slots 1..1+num_lf_groups — LfGroup[*]
    //   slot 1+num_lf_groups — HfGlobal (0-byte for kModular)
    //   slots 2+num_lf_groups + p*num_groups + g — PassGroup[p][g]
    let lf_global_slot = 0usize;
    let lf_group_slot = |lf_group_idx: u64| -> usize { 1 + lf_group_idx as usize };
    let hf_global_slot = 1 + num_lf_groups as usize;
    let pass_group_slot = |pass_idx: u32, group_idx: u32| -> usize {
        2 + num_lf_groups as usize + (pass_idx as u64 * num_groups + group_idx as u64) as usize
    };

    // 8. LfGlobal (slot 0) — read the GlobalModular prelude. For images
    //    where every channel fits in group_dim, this fully populates
    //    `lf_global.global_modular.image`. Otherwise the larger
    //    channels are zero-filled placeholders that PassGroups fill.
    //
    // Round-31 (parent-dispatch r16, noise-64x64-lossless unblock):
    //   The single-TOC-entry case still constitutes a section per
    //   FDIS §F.3: "no more bits are read from the codestream than 8
    //   times the byte size indicated in the TOC; if fewer bits are
    //   read, then the remaining bits of the section all have the
    //   value zero." High-entropy modular fixtures (e.g. `cjxl -e 7`
    //   noise) can leave the ANS / hybrid-uint refill loop trying to
    //   read past the last byte by a few bits — those reads must
    //   return zero, not error. Pre-round-31 the fast path used the
    //   non-padding main reader and rejected the `cjxl -e 7` noise
    //   fixture with `unexpected end of JXL bitstream` mid-pixel
    //   decode. Now every LfGlobal read goes through a section reader
    //   sliced to the TOC-declared length so the §F.3 zero-pad rule
    //   applies uniformly to single-TOC-entry frames and multi-TOC
    //   frames alike.
    let lf_global_bytes = section_byte_range(lf_global_slot)?;
    let mut lf_global = {
        let mut lf_br = BitReader::new_section(lf_global_bytes);
        LfGlobal::read(&mut lf_br, &fh, &metadata)?
    };

    // 8b. LfGroups (slots 1..1+num_lf_groups) — round 7 only handles
    //     num_lf_groups <= 1 (gated above). For num_lf_groups == 1 with
    //     a fully-decoded GlobalModular image (small-image case), the
    //     LfGroup section is empty (no channel has hshift>=3, vshift>=3
    //     by default for round-7 lossless fixtures). We still consume
    //     the slot bytes by reading the empty ModularLfGroup
    //     sub-bitstream — for round 7 the slot is allowed to be
    //     ignored when no channel matches the LfGroup criterion.

    // 8c. PassGroups (slots 1+num_lf_groups + p*num_groups + g) —
    //     decode each per-pass per-group modular sub-bitstream and
    //     copy samples back into `lf_global.global_modular.image`.
    if !lf_global.global_modular.fully_decoded || num_groups > 1 || fh.passes.num_passes > 1 {
        for pass_idx in 0..fh.passes.num_passes {
            for group_idx in 0..(num_groups as u32) {
                let slot = pass_group_slot(pass_idx, group_idx);
                let pg_bytes = section_byte_range(slot)?;
                let mut pg_br = BitReader::new_section(pg_bytes);
                crate::pass_group::decode_modular_group_into(
                    &mut pg_br,
                    &fh,
                    &mut lf_global,
                    pass_idx,
                    group_idx,
                )?;
            }
        }
        // After all PassGroups complete, apply inverse transforms over
        // the now fully-assembled GlobalModular image (G.4.2 last
        // paragraph).
        let bit_depth = metadata.bit_depth.bits_per_sample.max(1);
        let transforms = lf_global.global_modular.transforms.clone();
        crate::global_modular::apply_inverse_transforms(
            &mut lf_global.global_modular.image,
            &transforms,
            bit_depth,
        )?;
    }
    let _ = lf_group_slot; // currently only used by round-8 multi-LfGroup
    let _ = hf_global_slot; // round-10+ VarDCT consumer; for kModular the slot is 0-byte

    // 9. Map the decoded modular image to a VideoFrame.
    //
    // Round-1 (2024-spec) supports:
    //   - Grey colour_space (single channel, 1 plane)
    //   - RGB colour_space (3 channels → 3 planes in R/G/B order)
    //   - 8-bit integer bit depth
    //
    // Round-11 (2024-spec) adds: kModular + `metadata.xyb_encoded == true`
    // path through Annex L.2.2 inverse XYB → linear RGB; and the
    // `frame_header.do_ycbcr == true` path through Annex L.3 inverse
    // YCbCr → RGB. Both paths land 3-channel RGB output (Grey colour
    // encoding remains a 1-channel pass-through).
    //
    // Other colour spaces (CMYK, etc.) and float bit depths fall in
    // later rounds.
    if metadata.bit_depth.float_sample {
        return Err(Error::Unsupported(
            "jxl decoder (round 1): float bit depth not supported".into(),
        ));
    }
    // Round 30 (2024-spec) — accept 1..=16 integer samples for the
    // pass-through path. The 8-bit and 16-bit cases each have their
    // own pack loop further down; other widths in 1..=16 emit byte
    // planes whose samples are clamped into the integer range
    // `[0, 2^bps - 1]` (1 byte/sample for `bps <= 8`, 2 bytes/sample
    // little-endian for `9 <= bps <= 16`).
    //
    // FDIS Annex A.6 + Table A.22 (`bit_depth.bits_per_sample`).
    // The XYB / YCbCr branches further down still hard-require 8-bit
    // because their dequantisation lattice is calibrated against the
    // 8-bit output range; high-bit-depth XYB / YCbCr is round-31+.
    if metadata.bit_depth.bits_per_sample == 0 || metadata.bit_depth.bits_per_sample > 16 {
        return Err(Error::Unsupported(format!(
            "jxl decoder (round 30): bits_per_sample {} not supported (1..=16 only)",
            metadata.bit_depth.bits_per_sample
        )));
    }
    let img = lf_global.global_modular.image;
    let n_chans = img.channels.len();
    let expected_chans = match metadata.colour_encoding.colour_space {
        ColourSpace::Grey => 1,
        ColourSpace::Rgb => 3,
        _ => {
            return Err(Error::Unsupported(format!(
                "jxl decoder (round 1): colour_space {:?} not supported (Grey/RGB only)",
                metadata.colour_encoding.colour_space
            )));
        }
    };
    // Round 29 (parent-dispatch r14) extends the channel-count contract:
    // a kModular frame may carry `expected_chans` colour channels plus
    // `metadata.num_extra_channels` extra channels (alpha, depth, …).
    // The Modular decoder produces them as a flat channel array in
    // colour-then-extras order (FDIS Annex G.1.3 "channel order" rule).
    let n_extra = metadata.num_extra_channels as usize;
    let expected_with_extras = expected_chans + n_extra;
    if n_chans != expected_chans && n_chans != expected_with_extras {
        return Err(Error::Unsupported(format!(
            "jxl decoder (round 29): {} channels but colour_space wants {} (with {} extra channels = {})",
            n_chans, expected_chans, n_extra, expected_with_extras
        )));
    }

    // Round-11 inverse colour transform decision. The decoded modular
    // image's first three channels are reinterpreted per Annex L:
    //   * `metadata.xyb_encoded` true → §L.2.2 inverse XYB → linear RGB
    //     (channel order on input: Y', X', B').
    //   * `frame_header.do_ycbcr` true (xyb_encoded must be false per
    //     §L.1) → §L.3 inverse YCbCr → RGB (channel order: Cb, Y, Cr).
    //   * else → channels are already in colour_encoding's space; pass
    //     through (round-1 behaviour).
    if expected_chans == 3 && metadata.xyb_encoded {
        if metadata.bit_depth.bits_per_sample != 8 {
            return Err(Error::Unsupported(format!(
                "jxl decoder (round 30): XYB high-bit-depth (bps={}) deferred",
                metadata.bit_depth.bits_per_sample
            )));
        }
        let planes = build_rgb_planes_from_xyb(&img, &lf_global.lf_dequant, &metadata)?;
        return Ok(VideoFrame { pts, planes });
    }
    if expected_chans == 3 && fh.do_ycbcr {
        if metadata.bit_depth.bits_per_sample != 8 {
            return Err(Error::Unsupported(format!(
                "jxl decoder (round 30): YCbCr high-bit-depth (bps={}) deferred",
                metadata.bit_depth.bits_per_sample
            )));
        }
        let planes = build_rgb_planes_from_ycbcr(&img)?;
        return Ok(VideoFrame { pts, planes });
    }

    // Pass-through path: each channel becomes a plane (no colour
    // conversion). Pre-round-11 behaviour, retained for the five
    // small lossless fixtures and any non-XYB / non-YCbCr modular
    // image. Round 30 (2024-spec) extends the per-sample pack rule:
    //
    //   bps  ≤ 8 → 1 byte/sample, plane stride == width;
    //   9 ≤ bps ≤ 16 → 2 bytes/sample, little-endian, plane stride
    //                  == width × 2.
    //
    // Choice of LE pack: PNG ships its 16-bit samples big-endian
    // (RFC 2083 §2.1) whereas the JXL ImageMetadata bit-depth field
    // is endian-agnostic; we therefore pick LE so a downstream
    // consumer can treat each plane as a `&[u16]` after a
    // `bytemuck::cast_slice` or `<u16>::from_le_bytes` step on a
    // little-endian host without a swap. The convention is
    // documented in this crate's README under "Plane byte layout".
    let bps = metadata.bit_depth.bits_per_sample;
    let max_sample: i32 = (1i32 << bps) - 1;
    let mut planes: Vec<VideoPlane> = Vec::with_capacity(n_chans);
    for (i, ch_data) in img.channels.iter().enumerate() {
        let desc = img.descs[i];
        let w = desc.width as usize;
        let h = desc.height as usize;
        let plane = if bps <= 8 {
            let mut bytes = Vec::with_capacity(w * h);
            for &v in ch_data.iter() {
                bytes.push(v.clamp(0, max_sample) as u8);
            }
            VideoPlane {
                stride: w,
                data: bytes,
            }
        } else {
            let mut bytes = Vec::with_capacity(w * h * 2);
            for &v in ch_data.iter() {
                let s = v.clamp(0, max_sample) as u16;
                bytes.extend_from_slice(&s.to_le_bytes());
            }
            VideoPlane {
                stride: w * 2,
                data: bytes,
            }
        };
        planes.push(plane);
        // Sanity check height while we're here.
        let expected_len = if bps <= 8 { w * h } else { w * h * 2 };
        debug_assert_eq!(planes[i].data.len(), expected_len);
    }
    Ok(VideoFrame { pts, planes })
}

/// Convert a 3-channel decoded modular image whose channels carry
/// `(Y', X', B')` XYB-domain integer samples into an `R G B` plane
/// triple (per §L.2.2). All three channels must share the same
/// dimensions; the output planes are byte-stride packed at the same
/// width × height.
fn build_rgb_planes_from_xyb(
    img: &crate::modular_fdis::ModularImage,
    lf_dequant: &crate::lf_global::LfChannelDequantization,
    metadata: &ImageMetadataFdis,
) -> Result<Vec<VideoPlane>> {
    if img.channels.len() != 3 {
        return Err(Error::InvalidData(format!(
            "JXL XYB inverse: expected 3 channels (Y', X', B'), got {}",
            img.channels.len()
        )));
    }
    let desc0 = img.descs[0];
    for (i, d) in img.descs.iter().enumerate().take(3) {
        if d.width != desc0.width || d.height != desc0.height {
            return Err(Error::InvalidData(format!(
                "JXL XYB inverse: channel {i} dims {}x{} differ from channel 0 {}x{} \
                 — chroma subsampling not supported in modular XYB output",
                d.width, d.height, desc0.width, desc0.height
            )));
        }
    }
    let w = desc0.width as usize;
    let h = desc0.height as usize;
    let n = w * h;
    if img.channels[0].len() < n || img.channels[1].len() < n || img.channels[2].len() < n {
        return Err(Error::InvalidData(format!(
            "JXL XYB inverse: channel sample count short of {}x{}={n}",
            w, h
        )));
    }
    let mut r_bytes = Vec::with_capacity(n);
    let mut g_bytes = Vec::with_capacity(n);
    let mut b_bytes = Vec::with_capacity(n);
    let oim = &metadata.opsin_inverse_matrix;
    let tm = &metadata.tone_mapping;
    for idx in 0..n {
        // Channel order on input is `(Y', X', B')` per FDIS §L.2.2
        // first paragraph.
        let y_prime = img.channels[0][idx];
        let x_prime = img.channels[1][idx];
        let b_prime = img.channels[2][idx];
        let (r_lin, g_lin, b_lin) =
            crate::xyb::modular_xyb_to_linear_rgb(y_prime, x_prime, b_prime, lf_dequant, oim, tm);
        r_bytes.push(crate::xyb::linear_rgb_to_u8(r_lin));
        g_bytes.push(crate::xyb::linear_rgb_to_u8(g_lin));
        b_bytes.push(crate::xyb::linear_rgb_to_u8(b_lin));
    }
    Ok(vec![
        VideoPlane {
            stride: w,
            data: r_bytes,
        },
        VideoPlane {
            stride: w,
            data: g_bytes,
        },
        VideoPlane {
            stride: w,
            data: b_bytes,
        },
    ])
}

/// Convert a 3-channel decoded modular image whose channels carry
/// `(Cb, Y, Cr)` samples (YCbCr-encoded modular path) into an
/// `R G B` plane triple per §L.3. Outputs 8-bit bytes; the spec
/// formula treats inputs as floats in the [0, 1] interval, so we
/// rescale `[0..=255]` integer samples by `1/255` first then re-
/// quantise the RGB outputs by 255.
fn build_rgb_planes_from_ycbcr(img: &crate::modular_fdis::ModularImage) -> Result<Vec<VideoPlane>> {
    if img.channels.len() != 3 {
        return Err(Error::InvalidData(format!(
            "JXL YCbCr inverse: expected 3 channels (Cb, Y, Cr), got {}",
            img.channels.len()
        )));
    }
    let desc0 = img.descs[0];
    for (i, d) in img.descs.iter().enumerate().take(3) {
        if d.width != desc0.width || d.height != desc0.height {
            return Err(Error::Unsupported(format!(
                "JXL YCbCr inverse: channel {i} dims {}x{} differ from channel 0 {}x{} \
                 — chroma subsampling not yet supported in YCbCr modular output",
                d.width, d.height, desc0.width, desc0.height
            )));
        }
    }
    let w = desc0.width as usize;
    let h = desc0.height as usize;
    let n = w * h;
    let mut r_bytes = Vec::with_capacity(n);
    let mut g_bytes = Vec::with_capacity(n);
    let mut b_bytes = Vec::with_capacity(n);
    for idx in 0..n {
        // Spec §L.3 channel order: (Cb, Y, Cr).
        let cb = img.channels[0][idx] as f32 / 255.0;
        let y = img.channels[1][idx] as f32 / 255.0;
        let cr = img.channels[2][idx] as f32 / 255.0;
        let (r_lin, g_lin, b_lin) = crate::xyb::inverse_ycbcr_to_rgb(cb, y, cr);
        r_bytes.push(crate::xyb::linear_rgb_to_u8(r_lin));
        g_bytes.push(crate::xyb::linear_rgb_to_u8(g_lin));
        b_bytes.push(crate::xyb::linear_rgb_to_u8(b_lin));
    }
    Ok(vec![
        VideoPlane {
            stride: w,
            data: r_bytes,
        },
        VideoPlane {
            stride: w,
            data: g_bytes,
        },
        VideoPlane {
            stride: w,
            data: b_bytes,
        },
    ])
}

/// Inputs the integrated single-pass VarDCT finish step (`finish_vardct_decode`)
/// pulls together once the LfGlobal / LfGroup / HfGlobal sections have
/// been parsed and the LF image dequantised. Grouped into one struct so
/// the finisher's arity stays manageable (Clippy `too_many_arguments`).
struct VarDctFinishInputs<'a> {
    fh: &'a FrameHeader,
    metadata: &'a ImageMetadataFdis,
    /// Dequantised + (optionally) smoothed per-LfGroup LF image (Listing
    /// F.1). Channel order `[X, Y, B]`.
    lf: crate::lf_dequant::LfDequantOutput,
    /// The §C.5.4 per-LfGroup DctSelect + HfMul grid.
    grid: crate::dct_select::DctSelectGrid,
    /// Per-64×64-tile chroma-from-luma factor channels (HfMetadata).
    x_from_y: &'a [i32],
    b_from_y: &'a [i32],
    /// LfGlobal §I.2.7 colour-correlation base/colour factors.
    cfl: crate::lf_global::LfChannelCorrelation,
    /// LfGlobal §I.2.2 block-context bundle (drives the resolver).
    hf_block_context: crate::lf_global::HfBlockContext,
    /// Logical frame extent the padded block grid is cropped to (§6.2).
    frame_width: u32,
    frame_height: u32,
}

/// Finish an integrated single-LfGroup single-pass VarDCT decode.
///
/// `br` is positioned at the start of the frame's PassGroup payload (the
/// §C.8.3 per-pass HF header `hfp` followed by the HF-coefficient
/// entropy stream). `hf_section` is the parsed §C.7 HfGlobal section
/// (borrowed mutably so its §C.7.2 histograms drive the ANS decode
/// state). The function reads the per-pass HF header, builds the
/// histogram-decode context, runs
/// [`crate::vardct_reconstruct::reconstruct_lf_group_from_histogram`] to
/// the three XYB residual planes, crops them to the logical frame
/// extent (§6.2), and converts XYB → 8-bit RGB (§L.2.2).
///
/// Restoration filters (Gaborish §J.2, EPF §J.3) are NOT applied here —
/// the default-encoding fixtures this lands against carry `gab` / `epf`
/// enabled, so the output is "IDCT-exact, pre-filter". The pre-filter
/// XYB planes are the input the §J pipeline consumes; wiring those two
/// filters into this path is the documented follow-up.
fn finish_vardct_decode(
    inputs: VarDctFinishInputs<'_>,
    hf_section: &mut crate::hf_global_section::HfGlobalSection,
    br: &mut BitReader<'_>,
    pts: Option<i64>,
) -> Result<VideoFrame> {
    use crate::block_context_resolver::BlockContextResolver;
    use crate::hf_dequant::QmScaleFactors;
    use crate::multi_pass_hf_header::PerPassHfHeaders;
    use crate::per_pass_non_zeros::PerPassNonZerosGrids;
    use crate::vardct_reconstruct::{reconstruct_lf_group_from_histogram, DequantContext};

    let VarDctFinishInputs {
        fh,
        metadata,
        lf,
        grid,
        x_from_y,
        b_from_y,
        cfl,
        hf_block_context,
        frame_width,
        frame_height,
    } = inputs;

    // Single-pass only: the §C.8.3 cross-pass accumulation is exercised
    // by the reconstruction driver's unit tests, but the integrated
    // multi-pass PassGroup framing (per-group section slicing for
    // num_groups > 1) is a separate wiring step. Reject here so a
    // multi-pass / multi-group frame surfaces precisely rather than
    // mis-reading the single-section layout.
    if fh.passes.num_passes != 1 {
        return Err(Error::Unsupported(format!(
            "jxl VarDCT integrated decode: num_passes={} — only single-pass frames \
             are wired end-to-end (multi-pass cross-pass framing is the next step)",
            fh.passes.num_passes
        )));
    }

    let num_hf_presets = hf_section.num_hf_presets();
    let nb_block_ctx = hf_block_context.nb_block_ctx;

    // §C.8.3 per-pass HF header (`hfp` selector + derived
    // histogram_offset). For a single-pass frame this is one header read
    // at the head of the PassGroup payload.
    let headers = PerPassHfHeaders::read(br, fh.passes.num_passes, num_hf_presets, nb_block_ctx)?;

    // Bind the §C.7.2 histograms to the per-pass headers → the
    // histogram-backed decode context.
    let mut ctx = hf_section.decode_context(&headers)?;

    // Per-pass per-channel NonZeros grids, sized to the DctSelect grid's
    // block dimensions (the non-subsampled VarDCT case: every channel
    // shares the LfGroup's block geometry).
    let mut nz = PerPassNonZerosGrids::new_uniform(
        fh.passes.num_passes,
        3,
        grid.width_blocks,
        grid.height_blocks,
    )?;

    let resolver = BlockContextResolver::new(&hf_block_context);

    // F.3 dequant context: default dequant-matrix set + opsin-inverse
    // bias + per-channel 0.8^(qm_scale - 2) factors.
    let set = crate::dct_quant_weights::materialise_default_dequant_set()?;
    let qm = QmScaleFactors::for_frame(fh);
    let dq = DequantContext {
        set: &set,
        oim: &metadata.opsin_inverse_matrix,
        qm: &qm,
    };

    // qdc_at: the quantised LF DC triple at each varblock's top-left 8×8
    // cell. The §I.2.2 *default* HfBlockContext (the only case this path
    // accepts — see caller gate) has empty `lf_thresholds`, so the
    // resolver never consults `qdc`; a zero triple is therefore exact
    // for the default-context fixtures. Non-default contexts are gated
    // out by the caller, so this never silently mis-derives a block
    // context.
    let qdc_at =
        |_p: u32, _vb: &crate::varblock_walk::Varblock| -> Result<[i32; 3]> { Ok([0, 0, 0]) };

    let planes_xyb = reconstruct_lf_group_from_histogram(
        &fh.passes, &grid, &mut nz, &resolver, &mut ctx, br, &lf, &dq, x_from_y, b_from_y, &cfl,
        qdc_at,
    )?;

    // §6.2 crop the padded block-grid reconstruction to the logical
    // frame extent.
    let cropped = planes_xyb.crop_to(frame_width as usize, frame_height as usize)?;

    // §L.2.2 inverse XYB → linear RGB → 8-bit. The reconstructed XYB
    // samples come straight out of the IDCT (no §L.2.2 kModular rescale
    // preamble — that is the modular path's step).
    let oim = &metadata.opsin_inverse_matrix;
    let tone = &metadata.tone_mapping;
    let w = frame_width as usize;
    let h = frame_height as usize;
    let x_plane = &cropped.planes[0];
    let y_plane = &cropped.planes[1];
    let b_plane = &cropped.planes[2];
    let mut r_bytes = Vec::with_capacity(w * h);
    let mut g_bytes = Vec::with_capacity(w * h);
    let mut b_bytes = Vec::with_capacity(w * h);
    for i in 0..(w * h) {
        let (r, g, bb) = crate::xyb::inverse_xyb_to_rgb(
            x_plane.samples[i],
            y_plane.samples[i],
            b_plane.samples[i],
            oim,
            tone,
        );
        r_bytes.push(crate::xyb::linear_rgb_to_u8(r));
        g_bytes.push(crate::xyb::linear_rgb_to_u8(g));
        b_bytes.push(crate::xyb::linear_rgb_to_u8(bb));
    }
    Ok(VideoFrame {
        pts,
        planes: vec![
            VideoPlane {
                stride: w,
                data: r_bytes,
            },
            VideoPlane {
                stride: w,
                data: g_bytes,
            },
            VideoPlane {
                stride: w,
                data: b_bytes,
            },
        ],
    })
}

/// Integrated single-LfGroup VarDCT decode. Reads LfGlobal + LfGroup +
/// HfGlobal off the TOC, computes the per-channel LF multipliers
/// (Listing C.1 / F.1), runs Listing F.1 dequant + F.2 adaptive
/// smoothing on the LfCoefficients, then — for a single-pass frame whose
/// HfBlockContext carries empty `lf_thresholds` — feeds the dequantised
/// LF image into the integrated HF-entropy decode + F.3 dequant + §I.2.4
/// LLF merge + §I.2.3.2 IDCT + Annex G CfL finish step
/// (`finish_vardct_decode`), crops to the logical frame extent (§6.2),
/// and converts XYB → 8-bit RGB (§L.2.2). Sub-cases outside that
/// envelope (multi-pass, non-empty `lf_thresholds`, …) surface a precise
/// `Error::Unsupported`.
///
/// **Pixel-validation status.** The whole chain executes on a real
/// codestream, but the per-block HF coefficient scaling is not yet
/// validated bit-exact against a reference decode. The public
/// [`decode_one_frame`] path therefore withholds the reconstructed
/// pixels (see `decode_codestream`); this function is exposed so the
/// crate's integration tests can drive the pipeline end-to-end and pin
/// its structural invariants (plane count, dimensions, that every stage
/// runs without aborting). Restoration filters (Gaborish §J.2, EPF
/// §J.3) are likewise not applied here yet.
pub fn decode_vardct_frame(
    fh: &FrameHeader,
    metadata: &ImageMetadataFdis,
    toc: &Toc,
    br: &mut BitReader<'_>,
    scaffold: crate::vardct::VarDctScaffold,
    pts: Option<i64>,
) -> Result<VideoFrame> {
    let num_groups = fh.num_groups();
    let num_lf_groups = fh.num_lf_groups();
    if num_lf_groups != 1 || fh.passes.num_passes != 1 {
        return Err(Error::Unsupported(format!(
            "jxl VarDCT decoder (round 13): num_lf_groups={num_lf_groups} num_passes={} \
             — multi-LfGroup / multi-pass VarDCT defers to round 14+",
            fh.passes.num_passes
        )));
    }

    let frame_data_start = br.bytes_consumed();
    let codestream_data = br.data();
    if frame_data_start > codestream_data.len() {
        return Err(Error::InvalidData(
            "JXL VarDCT round 13: frame data start past codestream end".into(),
        ));
    }
    let frame_bytes = &codestream_data[frame_data_start..];
    let total_frame_len: u64 = toc.entries.iter().map(|&e| e as u64).sum();
    if total_frame_len > frame_bytes.len() as u64 {
        return Err(Error::InvalidData(format!(
            "JXL VarDCT round 13: TOC declares {total_frame_len} frame bytes but only {} \
             remaining",
            frame_bytes.len()
        )));
    }
    let mut section_starts: Vec<usize> = Vec::with_capacity(toc.entries.len());
    let mut acc: u64 = 0;
    for &e in &toc.entries {
        section_starts.push(acc as usize);
        acc = acc.saturating_add(e as u64);
    }
    let section_byte_range = |idx: usize| -> Result<&[u8]> {
        if idx >= toc.entries.len() {
            return Err(Error::InvalidData(format!(
                "JXL VarDCT round 13: TOC slot {idx} out of range (entries={})",
                toc.entries.len()
            )));
        }
        let start = section_starts[idx];
        let len = toc.entries[idx] as usize;
        let end = start + len;
        if end > frame_bytes.len() {
            return Err(Error::InvalidData(format!(
                "JXL VarDCT round 13: section {idx} byte range [{start}..{end}) exceeds frame bytes ({})",
                frame_bytes.len()
            )));
        }
        Ok(&frame_bytes[start..end])
    };

    // Slot indexing per F.3.1 (round-9 fix: HfGlobal slot is unconditional):
    //   slot 0 — LfGlobal
    //   slots 1..1+num_lf_groups — LfGroup[*]
    //   slot 1+num_lf_groups — HfGlobal (contains HfPass for kVarDCT)
    let lf_global_slot = 0usize;
    let lf_group_slot = |lf_group_idx: u64| -> usize { 1 + lf_group_idx as usize };
    let hf_global_slot = 1 + num_lf_groups as usize;

    // Round 15: F.3.1 says when `num_groups == 1 && num_passes == 1`
    // the TOC has a SINGLE entry containing all section bytes
    // concatenated WITHOUT byte alignment between sections. Each section
    // continues from the previous section's bit cursor.  When the TOC
    // has multiple entries, each section is sliced into its own byte
    // range and read against a fresh BitReader.
    let single_toc = toc.entries.len() == 1
        && num_groups == 1
        && fh.passes.num_passes == 1
        && num_lf_groups == 1;

    // The PassGroup slot (the §C.8.3 per-pass header + HF-coefficient
    // entropy stream) is decoded by the integrated finish step below.
    // For single-pass single-group VarDCT it is slot
    // `2 + num_lf_groups` (one PassGroup, pass 0, group 0). In the
    // single-TOC layout the PassGroup continues on the *same* bit cursor
    // immediately after HfGlobal (no byte alignment); in the multi-TOC
    // layout it is its own byte-aligned section.
    let pass_group_slot_0 = 2 + num_lf_groups as usize;

    // The continuation bit reader positioned at the PassGroup start.
    // Single-TOC: a reader sharing the whole frame buffer, advanced past
    // HfGlobal. Multi-TOC: a fresh section reader on the PassGroup slot.
    let (lf_global, lf_group, mut hf_global_section, mut pass_group_br) = if single_toc {
        // Single-TOC-entry path: chain section reads on the same bit
        // reader, no byte-aligned slicing between sections.
        let lf_global_bytes = section_byte_range(lf_global_slot)?;
        let mut shared_br = BitReader::new_section(lf_global_bytes);
        let lf_global = LfGlobal::read(&mut shared_br, fh, metadata)?;
        let _quantizer = lf_global
            .quantizer
            .ok_or_else(|| Error::InvalidData("JXL VarDCT round 13: Quantizer missing".into()))?;
        let _ = lf_global.hf_block_context.as_ref().ok_or_else(|| {
            Error::InvalidData("JXL VarDCT round 13: HfBlockContext missing".into())
        })?;
        let _ = lf_global.lf_channel_correlation.ok_or_else(|| {
            Error::InvalidData("JXL VarDCT round 13: LfChannelCorrelation missing".into())
        })?;

        let lf_group = crate::lf_group::LfGroup::read(&mut shared_br, fh, &lf_global, metadata, 0)?;

        // §C.7: the HfGlobal TOC slot is three contiguous pieces on the
        // same bit cursor — HfGlobal (§I.2.4 + §I.2.6) → HfPass sequence
        // (§C.7.1) → HF-coefficient histograms (§C.7.2) + ANS-state init.
        // `nb_block_ctx` is the LfGlobal HfBlockContext invariant (§I.2.2).
        let nb_block_ctx = lf_global
            .hf_block_context
            .as_ref()
            .expect("HfBlockContext presence checked above")
            .nb_block_ctx;
        let hf_global_section = crate::hf_global_section::HfGlobalSection::read(
            &mut shared_br,
            num_groups,
            nb_block_ctx,
        )?;
        // PassGroup continues on the same cursor (no byte alignment).
        (lf_global, lf_group, hf_global_section, shared_br)
    } else {
        // Multi-TOC-entry path: slice each section into its own byte
        // range and read against a fresh BitReader.
        let lf_global_bytes = section_byte_range(lf_global_slot)?;
        let mut lf_br = BitReader::new_section(lf_global_bytes);
        let lf_global = LfGlobal::read(&mut lf_br, fh, metadata)?;
        let _quantizer = lf_global
            .quantizer
            .ok_or_else(|| Error::InvalidData("JXL VarDCT round 13: Quantizer missing".into()))?;
        let _ = lf_global.hf_block_context.as_ref().ok_or_else(|| {
            Error::InvalidData("JXL VarDCT round 13: HfBlockContext missing".into())
        })?;
        let _ = lf_global.lf_channel_correlation.ok_or_else(|| {
            Error::InvalidData("JXL VarDCT round 13: LfChannelCorrelation missing".into())
        })?;

        let lf_group_bytes = section_byte_range(lf_group_slot(0))?;
        let mut lg_br = BitReader::new_section(lf_group_bytes);
        let lf_group = crate::lf_group::LfGroup::read(&mut lg_br, fh, &lf_global, metadata, 0)?;

        // §C.7 full HfGlobal section (see single-TOC branch comment).
        let nb_block_ctx = lf_global
            .hf_block_context
            .as_ref()
            .expect("HfBlockContext presence checked above")
            .nb_block_ctx;
        let hf_global_bytes = section_byte_range(hf_global_slot)?;
        let mut hg_br = BitReader::new_section(hf_global_bytes);
        let hf_global_section =
            crate::hf_global_section::HfGlobalSection::read(&mut hg_br, num_groups, nb_block_ctx)?;
        // PassGroup is its own byte-aligned section slot.
        let pg_bytes = section_byte_range(pass_group_slot_0)?;
        let pg_br = BitReader::new_section(pg_bytes);
        (lf_global, lf_group, hf_global_section, pg_br)
    };

    // Re-extract Quantizer for the dequant path below (it was already
    // checked for presence above in both branches).
    let quantizer = lf_global
        .quantizer
        .ok_or_else(|| Error::InvalidData("JXL VarDCT round 13: Quantizer missing".into()))?;

    let lf_coeff = lf_group.lf_coeff.ok_or_else(|| {
        Error::InvalidData("JXL VarDCT round 13: LfCoefficients missing on VarDCT LfGroup".into())
    })?;
    let hf_meta = lf_group.hf_meta.ok_or_else(|| {
        Error::InvalidData("JXL VarDCT round 13: HfMetadata missing on VarDCT LfGroup".into())
    })?;

    // Derive DctSelect / HfMul from BlockInfo per FDIS C.5.4 prose.
    // The grid covers the LfGroup's pixel rectangle; for a single-
    // LfGroup frame that's the full frame.
    let lf_w = lf_group.mlf_group.lf_group_width;
    let lf_h = lf_group.mlf_group.lf_group_height;
    let dct_grid = crate::dct_select::derive_dct_select(&hf_meta, lf_w, lf_h)?;

    // F.1 LF dequantisation (Listing F.1) over the per-LfGroup
    // LfCoefficients. Unwrap the lf_quant Vec into a fixed-size [3]
    // array as expected by `dequant_lf`.
    if lf_coeff.lf_quant.len() != 3 {
        return Err(Error::InvalidData(format!(
            "JXL VarDCT round 13: LfCoefficients has {} channels, expected 3",
            lf_coeff.lf_quant.len()
        )));
    }
    let lf_quant: [Vec<i32>; 3] = [
        lf_coeff.lf_quant[0].clone(),
        lf_coeff.lf_quant[1].clone(),
        lf_coeff.lf_quant[2].clone(),
    ];
    let multipliers = crate::lf_dequant::LfMultipliers::compute(&lf_global.lf_dequant, &quantizer);
    let mut dequant = crate::lf_dequant::dequant_lf(
        &lf_quant,
        lf_coeff.lf_quant_widths,
        lf_coeff.lf_quant_heights,
        lf_coeff.extra_precision,
        &multipliers,
    );

    // F.2 adaptive LF smoothing (gated by kSkipAdaptiveLFSmoothing flag
    // + no channel subsampled).
    if crate::lf_dequant::should_apply_adaptive_lf_smoothing(fh) {
        crate::lf_dequant::apply_adaptive_lf_smoothing(&mut dequant, &multipliers);
    }

    // Integrated finish: §C.8.3 per-pass HF header + histogram-backed HF
    // decode + F.3 dequant + LLF merge + IDCT + CfL → XYB residual
    // planes → §6.2 crop → §L.2.2 XYB→RGB. The integrated `qdc_at`
    // supplies a zero quantised-LF DC triple, which the §C.13
    // `block_context` derivation consumes ONLY through the
    // `lf_thresholds` ladder. A bundle with empty `lf_thresholds` (the
    // common case, including the default §I.2.2 bundle and many custom
    // bundles that only override `qf_thresholds` / `block_ctx_map`)
    // therefore never reads `qdc`, so the zero triple is exact. A bundle
    // with non-empty `lf_thresholds` would mis-derive the context, so
    // reject it precisely — wiring the per-varblock LF-DC lookup is the
    // next step.
    let hbc = lf_global
        .hf_block_context
        .clone()
        .ok_or_else(|| Error::InvalidData("JXL VarDCT round 13: HfBlockContext missing".into()))?;
    let cfl = lf_global.lf_channel_correlation.ok_or_else(|| {
        Error::InvalidData("JXL VarDCT round 13: LfChannelCorrelation missing".into())
    })?;
    let lf_thresholds_present = hbc.lf_thresholds.iter().any(|t| !t.is_empty());
    if lf_thresholds_present {
        return Err(Error::Unsupported(format!(
            "jxl VarDCT integrated decode: HfBlockContext carries non-empty lf_thresholds \
             (nb_block_ctx={}) — the integrated qdc_at LF-DC lookup feeding the \
             BlockContextResolver is the next wiring step; bundles with empty lf_thresholds \
             decode end-to-end",
            hbc.nb_block_ctx
        )));
    }

    let finish_inputs = VarDctFinishInputs {
        fh,
        metadata,
        lf: dequant,
        grid: dct_grid,
        x_from_y: &hf_meta.x_from_y,
        b_from_y: &hf_meta.b_from_y,
        cfl,
        hf_block_context: hbc,
        frame_width: scaffold.width,
        frame_height: scaffold.height,
    };
    finish_vardct_decode(
        finish_inputs,
        &mut hf_global_section,
        &mut pass_group_br,
        pts,
    )
}

/// FDIS-side `Headers` returned by [`probe_fdis`]. Mirrors the
/// committee-draft [`Headers`] but uses the FDIS bundle types.
#[derive(Debug, Clone)]
pub struct HeadersFdis {
    pub signature: container::Signature,
    pub size: SizeHeaderFdis,
    pub metadata: ImageMetadataFdis,
}

/// FDIS-side probe: parse SizeHeader + full A.6 ImageMetadata. Falls
/// back to the committee-draft probe if the FDIS path errors (so that
/// container detection still works on edge cases the committee-draft
/// path tolerates).
pub fn probe_fdis(input: &[u8]) -> Result<HeadersFdis> {
    let signature = container::detect(input)
        .ok_or_else(|| Error::InvalidData("jxl probe: no JXL signature".into()))?;
    match signature {
        container::Signature::RawCodestream => probe_fdis_codestream(&input[2..], signature),
        container::Signature::Isobmff => {
            let codestream_owned = container::extract_codestream(input)?;
            probe_fdis_codestream(&codestream_owned, signature)
        }
    }
}

fn probe_fdis_codestream(
    codestream: &[u8],
    signature: container::Signature,
) -> Result<HeadersFdis> {
    let mut br = BitReader::new(codestream);
    let size = SizeHeaderFdis::read(&mut br)?;
    let metadata = ImageMetadataFdis::read(&mut br)?;
    Ok(HeadersFdis {
        signature,
        size,
        metadata,
    })
}

/// Inspect a JXL file (raw codestream or ISOBMFF-wrapped) and return the
/// signature type + parsed `SizeHeader` + `ImageMetadata` preamble.
///
/// This is the main API users can reach today: it covers identification,
/// dimensions and sample format without needing an actual decoder.
pub fn probe(input: &[u8]) -> Result<Headers> {
    parse_headers(input)
}

/// Encoder slot, always rejected. Exposed for completeness so callers
/// that wire an `Encoder` factory by codec id get a clean `Unsupported`
/// error instead of `CodecNotFound`.
pub fn make_encoder(_params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    Err(Error::Unsupported(
        "jxl encode is out of scope for this crate".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_factory_returns_live_decoder() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let dec = ctx
            .codecs
            .first_decoder(&params)
            .expect("expected live decoder");
        assert_eq!(dec.codec_id().as_str(), CODEC_ID_STR);
    }

    #[test]
    fn probe_rejects_non_jxl() {
        let err = probe(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn probe_accepts_minimal_raw_codestream() {
        // small=1, 8x8 square (ratio=1), all_default=1 → 10 bits total.
        // LSB-first packing: byte0 holds bits 0..=7, byte1 holds bits 8..=9.
        // bit0=1, bits1..=5=0, bits6..=8=001 (ratio=1), bit9=1 (all_default)
        // → byte0 = 0b01000001 = 0x41, byte1 = 0b00000010 = 0x02.
        let input = [0xFF, 0x0A, 0x41, 0x02];
        let h = probe(&input).unwrap();
        assert_eq!(h.size.width, 8);
        assert_eq!(h.size.height, 8);
        assert!(h.metadata.all_default);
    }

    #[test]
    fn encoder_factory_rejects_cleanly() {
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        assert!(matches!(make_encoder(&params), Err(Error::Unsupported(_))));
    }
}
