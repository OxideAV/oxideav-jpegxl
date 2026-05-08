# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Round 9 (2024-spec)** — synth_320 0-byte PassGroup blocker
  resolved via two underlying fixes plus per-group transforms support.

  **§F.3.1 unconditional HfGlobal slot fix** — the 2024 spec lists
  `HfGlobal` UNCONDITIONALLY in the TOC bullet list (not gated on
  `encoding == kVarDCT`); per NOTE 1, the slot is empty (0-byte) for
  `encoding == kModular`. Round 8's `num_toc_entries` /
  `Toc::read` skipped HfGlobal for kModular, off-by-oning every
  PassGroup index in multi-group kModular frames. The synth_320
  fixture (320×320 grey, num_groups=9) actually has 12 TOC entries
  (1 LfGlobal + 1 LfGroup + 1 HfGlobal + 9 PassGroup), not 11; the
  apparent "0-byte PassGroup[0][0]" was the HfGlobal slot reading.
  Also: `HfPass[num_passes]` is part of the `HfGlobal` section per
  Annex G.3 Table G.4 — it does NOT contribute additional TOC
  entries (round 8 had counted both, double-incorrect).

  **§F.3 first-paragraph zero-padding sub-reader** — "When decoding
  a section, no more bits are read from the codestream than 8 times
  the byte size indicated in the TOC; if fewer bits are read, then
  the remaining bits of the section all have the value zero." Round
  8's `BitReader` errored on EOF for section sub-readers, breaking
  PassGroup ANS decodes whose modular sub-bitstream legitimately
  consumes fewer real bits than the section's byte size (the missing
  bits are guaranteed by the spec to be zero). Added
  `BitReader::new_section` which pads EOF reads with zero values for
  per-TOC-section sub-readers (LfGlobal / LfGroup / HfGlobal /
  PassGroup); the legacy `BitReader::new` keeps the strict EOF for
  whole-codestream parsing so malformed top-level structures still
  error early.

  **Per-PassGroup transforms (Annex H.6 inside G.4.2)** — observed
  in cjxl 0.11.1's synth_320 edge groups: the encoder emits a
  per-group Palette transform (`begin_c=0, num_c=1, nb_colours=191`)
  for the 64-pixel-wide column-2 / row-2 groups, which is
  spec-legal per Table H.1 (every modular sub-bitstream has its own
  `transform[nb_transforms]` field). `decode_modular_group_into`
  now applies the transform layout adjustment to the per-group
  channel descs, decodes against the adjusted descs, and applies
  the inverse transforms LOCALLY before copying samples back into
  the parent image. `apply_transforms_to_channel_layout` is now
  `pub` so the per-group reuse path doesn't duplicate the table.
  A new `tests/round9_synth_320_toc.rs` integration test confirms
  the TOC layout is parsed correctly (12 entries, slot 2 is
  HfGlobal not PG[0][0]) and that the first 6 rows of the first
  two group columns decode pixel-for-pixel against the expected
  `(y + x) & 0xFF` gradient.

  **Status** — synth_320 reaches end-of-frame without erroring and
  about 21k of 102400 pixels match the expected gradient; the
  remaining ~80k pixels drift mid-decode in the smaller edge groups
  (PG[0][2,5,6,7,8] = 64-pixel-wide / 64-pixel-tall sections).
  Suspected residual issue: ANS state nuance specific to the F.3
  zero-padded tail or per-group WP / property bookkeeping that
  doesn't surface against the round-4 small fixtures (single-group,
  single-channel, no padding pressure on the ANS state). Full
  pixel-correctness is round-10 work.

- **Round 8 (2024-spec)** — two themes: round-7 SPECGAP partial
  resolution + VarDCT scaffolding.

  **Theme 1: ANS distribution C.2.5 SPECGAP (interpretation C, partial)**
  - `src/ans/distribution.rs` — `read_distribution` now returns
    `(D, log_eff)` instead of just `D`; `log_eff` is the effective
    log_alphabet_size for downstream alias-table sizing. For the
    common case (alphabet_size <= table_size) `log_eff` equals the
    signalled `log_alphabet_size`. For the SPECGAP case
    (alphabet_size > table_size), the logcounts loop iterates
    `min(alphabet_size, table_size)` entries; the encoder's
    advertised wider alphabet is treated as a soft cap because
    empirically cjxl 0.11.1 only serialises `table_size` per-symbol
    entries. Interpretations A (grow D to a power-of-2 >=
    alphabet_size) and B (drop writes at i >= table_size) were both
    tried and rejected — see the module-level docstring on
    `read_distribution` for the full rationale.
  - `src/ans/cluster.rs`, `src/modular_fdis.rs`, `src/toc.rs` —
    callers updated to consume the `(D, log_eff)` tuple and pass
    `log_eff` to `AliasTable::build`.
  - The synth_320 fixture's LfGlobal section now parses cleanly
    past the round-7 SPECGAP error, but PassGroup decode is blocked
    at a separate post-LfGlobal blocker (cjxl emits a 0-byte
    PassGroup[0][0] slot which contradicts the spec's per-group
    "all groups carry data per pass" rule). That secondary blocker
    is round-9+ work; the synth_320 fixture is left in
    `tests/fixtures/synth_320_grey/` unconsumed by tests pending
    that round.

  **Theme 2: VarDCT scaffolding**
  - New `src/vardct.rs` module: structural recognition of a
    VarDCT-encoded codestream + IDCT-II primitives for the smallest
    block size (8×8). `recognise_vardct_codestream(fh, metadata)`
    validates the round-8 envelope (single LF group, single pass,
    no extra channels, Grey/RGB colour) and returns a
    `VarDctScaffold` geometry record. `idct1d_8` and `idct2d_8x8`
    implement the spec's inverse DCT-II formula directly (O(N²),
    audit-friendly; faster Lee-style decompositions land alongside
    LF/HF subband decode in round 9+).
  - `src/lib.rs` — `decode_codestream`'s encoding gate now special-
    cases `Encoding::VarDct` to invoke
    `vardct::recognise_vardct_codestream` and emit a VarDCT-specific
    `Error::Unsupported` message rather than the generic round-7
    one.
  - End-to-end VarDCT pixel decode (LF subband decode, HF subband
    decode, dequant, inverse transform dispatch across block sizes
    8×8/8×16/16×8/16×16/32×32/64×64/DCT4/DCT8/IDENTITY/AFV,
    Chroma-from-Luma, Gaborish smoothing, EPF) is round-9+ work.

  **Tests**
  - `tests/round8_vardct_scaffold.rs` — verifies the 5 small
    lossless fixtures still pixel-correct (regression sentinel
    against the `(D, log_eff)` tuple refactor) plus VarDCT
    primitive sanity checks.
  - `src/ans/distribution.rs` — new
    `branch3_alphabet_size_above_table_size_is_truncated` sentinel
    test for the SPECGAP truncation behaviour.

- **Round 7 (2024-spec)** — four-piece refactor wiring the GlobalModular
  partial-decode path to per-PassGroup decode + post-PassGroup inverse
  transforms (Annex G.1.3 last paragraph + G.4.2). The orchestration
  is in place; pixel-correct decode of the committed multi-group
  fixture is blocked at a documented spec-vs-reference SPECGAP (cjxl
  0.11.1's multi-group ANS streams emit `alphabet_size > table_size`
  for log_alpha=5, which the spec text in C.2.5 implies should be
  rejected). Round-8 will resolve the SPECGAP once docs collaborator
  clarifies the alphabet cap.
  - **`src/global_modular.rs`** — `GlobalModular::read` now obeys
    G.1.3's "stops decoding at channels exceeding `group_dim`" rule.
    Channels too large for GlobalModular are zero-filled placeholders
    and `fully_decoded = false`; the bundle stashes
    `nb_meta_channels`, `transforms`, and `global_tree` for the
    per-PassGroup decode to consume. New
    `apply_inverse_transforms(image, transforms, bit_depth)` is the
    transform pass that the multi-group path invokes AFTER all
    PassGroups complete (G.4.2 last paragraph).
  - **`src/modular_fdis.rs`** — new public
    `decode_channels_at_stream(br, descs, tree, wp, stream_index)`
    threads the Table H.4 stream-index property through the channel-
    decode loop (the legacy `decode_channels` is a thin wrapper that
    passes `stream_index = 0`). `MaTreeFdis::cloned_with_fresh_state`
    lets per-section sub-bitstreams reuse the global tree's static
    shape + clustered distributions while reading a fresh ANS state
    init for each section (per H.2's "global MA tree and its clustered
    distributions are used as decoded from the GlobalModular section").
    `MaTreeFdis`, `EntropyStream`, `ClusterEntropy`, `HybridUintState`,
    `AnsDecoder` all gain `Clone`.
  - **`src/pass_group.rs`** —
    `decode_modular_group_into(br, fh, lf_global, pass_idx, group_idx)`
    decodes one PassGroup's modular sub-bitstream. The contributing-
    channel filter implements G.4.2's criterion (channel exceeds
    group_dim, hshift<3 OR vshift<3, minshift<=min(hshift,vshift)<
    maxshift, not already decoded). The decoded samples are copied
    back into `lf_global.global_modular.image` at the rectangle
    derived from the group's frame-coordinates origin shifted by
    hshift/vshift. `compute_pass_shift_range` now takes `num_passes`
    and models an implicit `n=num_ds` final-resolution entry that the
    spec text omits (documented SPECGAP — without it, single-pass
    frames would have minshift=maxshift=3 and decode no modular data).
  - **`src/toc.rs`** — TOC entries of value 0 are now accepted (an
    empty LfGroup or PassGroup section is legal when no channel
    matches that section's filter). Round 6 over-strictly rejected
    `entry == 0`.
  - **`src/ans/cluster.rs`** — `read_general_clustering` now handles
    the prefix-coded sub-stream branch (the simple-clustering path
    covered by the round-2..6 fixtures avoided this branch
    altogether).
  - **`src/lib.rs`** — `decode_codestream` reads each TOC slot as a
    fresh sub-bitstream-bounded `BitReader`, dispatches LfGlobal
    (slot 0), then iterates `pass_idx × group_idx` PassGroups (slots
    `1 + num_lf_groups + p*num_groups + g`), then applies inverse
    transforms over the assembled image. Single-group / single-pass
    frames continue to use the round-3..6 fast path so the five
    pixel-correct lossless fixtures remain regression-free.
  - **`tests/fixtures/synth_320_grey/`** — a 320×320 grey gradient
    encoded by cjxl 0.11.1 (`-d 0 -m 1 -e 1 -g 0 -R 0`) producing a
    9-group multi-group lossless modular fixture. Committed for round-8
    once the SPECGAP above is resolved.

- **Round 6 (2024-spec)** — Annex E.4 ICC profile decode + LfGroup /
  PassGroup type scaffolding.
  - **`src/icc.rs`** — full ICC profile decoder per Annex E.4. Reads
    `enc_size = U64()`, then 41 pre-clustered distributions (the
    existing `EntropyStream::read(br, 41)` infrastructure built for
    Modular), then `enc_size` bytes via `DecodeHybridVarLenUint`
    driven by the `IccContext(i, prev_byte, prev_prev_byte)`
    41-context function from E.4.1. The encoded byte stream is split
    into `output_size` (Varint) + `commands_size` (Varint) prefix +
    command stream + data stream, then walked through E.4.3 (header
    with predicted-byte ladder), E.4.4 (tag list with 21-tagcode
    switch + previous_tagstart / previous_tagsize accumulation), and
    E.4.5 (main content with command set 1 / 2 / 3 / 4 / 10 / 16-23
    + Nth-order predictor at orders 0/1/2). 14 unit tests
    (round-trip helpers + spec-listing edge cases incl. the example
    "shuffle of (1,2,3,4,5,6,7) at width 2 → (1,5,2,6,3,7,4)").
  - **`src/lf_group.rs`** — Annex G.2 type scaffolding. `LfGroup`
    bundle (Table G.3) + `LfCoefficients` (G.2.2 — VarDCT only) +
    `ModularLfGroup` (G.2.3 — always present) + `HfMetadata` (G.2.4).
    Per-LfGroup decode itself is round-7 work; the parser stub
    returns `Error::Unsupported` with a precise round-7 follow-up
    message. `ModularLfGroup::rect_for_index` does compute the
    per-LfGroup pixel rectangle in frame coordinates.
  - **`src/pass_group.rs`** — Annex G.4 type scaffolding. `PassGroup`
    bundle (Table G.5) + `ModularGroupData` (G.4.2). Per-PassGroup
    decode is round-7 work; `ModularGroupData::rect_for_index`
    computes per-group pixel rectangles. Plus
    `compute_pass_shift_range(pass_index, downsample, last_pass)`
    implementing the `(minshift, maxshift)` recurrence from the
    G.4.2 first paragraph: pass 0 starts at maxshift=3, subsequent
    passes inherit maxshift = previous pass's minshift; minshift
    comes from the smallest `log2(downsample[n])` over `n` with
    `last_pass[n] == p`, falling back to maxshift if no match.
  - **`lib::decode_codestream`** — when
    `metadata.colour_encoding.want_icc == true` the bit reader is
    now correctly advanced past the ICC stream via
    `icc::decode_encoded_icc_stream` + `icc::reconstruct_icc_profile`,
    instead of erroring with "Annex B ICC stream not yet wired". A
    minimal ICC.1 sanity check verifies the "acsp" magic at offset
    36; the decoded bytes are not propagated to `VideoFrame`
    (`oxideav_core::VideoFrame` has no ICC slot in 0.1.x).
    Multi-LfGroup / multi-group / multi-pass / VarDCT frames now
    fail with precise round-7-targeting error messages instead of
    the generic "TOC with N entries" rejection.

### Round-6 acceptance

- All 5 currently-pixel-correct fixtures still decode pixel-correct
  vs `expected.png`: pixel-1x1, gray-64x64, gradient-64x64-lossless,
  palette-32x32, grey_8x8_lossless. (No regression of the
  five-round single-group decode path.)
- 32 new unit tests (14 ICC + 8 LfGroup + 10 PassGroup); total test
  count goes from 211 to 243.
- `cargo clippy --all-targets -- -D warnings` clean.
- `cargo fmt --check` clean.

### Round-6 deferred (round-7 candidates)

- LfGroup / PassGroup actual decode wiring: blocked on four
  coordinated changes — GlobalModular `nb_meta_channels`-aware
  partial decode (G.1.3 last paragraph), `stream_index` threading
  through `decode_channels` (Table H.4 property 1), TOC permutation
  awareness, and inverse-transform application timing (post-PassGroup
  per G.4.2 last sentence). These four are too coupled to ship
  individually without regressing the five pixel-correct fixtures.
- Multi-group lossless modular fixture: docs corpus has no fixture
  in this category (the smallest multi-group fixture
  `large-1024x768-d2` is VarDCT). Round 7 should produce one via
  `cjxl input.png output.jxl -d 0 -e 7` against a 256×256+ lossless
  PNG and commit it to `tests/fixtures/`.
- ICC bytes propagation to `oxideav_core::VideoFrame`: the parsed
  ICC profile is currently discarded after sanity-check because
  there's no `VideoFrame::icc_profile` slot in `oxideav-core 0.1`.
  Round 8+ work should be coordinated with an `oxideav-core` minor
  release that adds the slot.
- XYB inverse transform (§C.5 / §K): deferred — no XYB fixture in
  current pixel-correct corpus. Synthetic XYB fixture would require
  encoder support which doesn't exist in this crate.

### SPECGAP entries (round 6)

None new. The Annex E.4 ICC pseudocode in the 2024 published edition
is complete and unambiguous; no round-7 SPECGAP pivot is required
for it.

- **Round 5 (2024-spec)** — RFC 7932 §3.5 prefix-code histogram Kraft
  early-stop fix; `grey_8x8_lossless.jxl` (cjxl 0.11.1, 180-byte
  emit) now decodes pixel-correct (all 64 bytes equal 128 as
  expected for a constant-grey PGM input).
  - **Root cause** — `read_complex_prefix` decoded all `count`
    code-lengths regardless of whether the running Kraft sum had
    already reached `1 << 15`. cjxl 0.11.1 emits histograms whose
    Kraft saturates mid-stream (specifically the cluster[1] histogram
    at bit 299..549 of the grey_8x8 fixture: 251 lengths reach
    Kraft = 32768 exactly; the remaining 6 lengths must be treated
    as implicit zeros per RFC 7932 §3.5).
  - **Fix** — track a running Kraft sum inside the lengths loop;
    once it reaches `1 << 15`, break early and rely on the initial
    `vec![0u32; alphabet_size]` to leave the trailing entries as
    implicit zeros. Repeat-16 (re-emit previous non-zero length) is
    also instrumented to short-circuit when its replication crosses
    the Kraft boundary.
  - **Bisect** — `tests/round5_grey_8x8_cluster_bisect.rs` walks the
    symbol-stream prelude bit-by-bit, decoding each cluster's prefix
    histogram and printing the clcl array, the Kraft sum, and the
    per-symbol code-length array. Cluster 1 was the failing one;
    the round-4 trace stopped at bit 563 with Kraft=32832 (64 over
    budget). `src/ans/prefix.rs` exposes a public `diagnose_complex_prefix`
    entry point that captures partial state even on failure.
  - **New API surface** — `read_prefix_code_traced` /
    `read_complex_prefix_traced` / `diagnose_complex_prefix` /
    `ClclTrace` are public so future bisect tests can reproduce the
    same per-cluster step-by-step trace without copy-paste.

- **Round 4 (2024-spec)** — three independent decoder bugs fixed; all
  three previously-blocked single-group docs fixtures
  (`gradient-64x64-lossless.jxl`, `palette-32x32.jxl`, plus the round-3
  baseline `gray-64x64.jxl`) now decode pixel-correct against their
  committed `expected.png` references via a new full-image PNG-decoder
  comparison harness (`tests/round4_pixel_correctness.rs`).
  - **2024-spec C.3.3 `ReadUint` formula fix** — round 3 computed the
    extra-bits count as `n = split_exponent + ((token - split) >>
    (msb + lsb))` but spec C.3.3 says
    `n = (split_exponent - msb_in_token - lsb_in_token) +
    ((token - split) >> (msb + lsb))`. The missing `- msb - lsb`
    inflated `n` by `(msb + lsb)` extra bits per above-split token,
    which is the root cause of "12× bits/token" over-consumption that
    blocked `gradient-64x64` and `palette-32x32` in round 3.
    `HybridUintConfig::read_uint` now uses the spec formula; the
    in-tree `encode_uint` round-trip helper was likewise updated to
    keep the existing round-trip unit tests passing.
  - **2024-spec H.5.2 Self-correcting predictor — three sign / formula
    fixes**:
    1. `subpred[3]` had `n8.wrapping_add(...)` in round 3; spec listing
       reads `subpred[3] = N3 - (...)`. Sign flipped to
       `wrapping_sub`.
    2. `error2weight` was missing the trailing `>> shift`. Spec:
       `4 + ((maxweight * ((1<<24) Idiv ((err_sum >> shift) + 1))) >> shift)`.
       The missing outer shift inflated weights non-uniformly across
       sub-predictors when their shifts differ, producing wrong
       sub-predictor mixing.
    3. `s = (sum_weights >> 1) - 1` per spec; round 3 omitted the
       `- 1`.
  - **2024-spec H.5.1 `err[i]` formula fix** — round 3 stored
    `abs(((subpred[i] + 3) >> 3) - true_value)`; spec is
    `(abs(subpred[i] - (true_value << 3)) + 3) >> 3`. These differ in
    rounding, producing wrong sub_err values that propagate to
    downstream WP weights.
  - **2024-spec H.5.2 sub_err edge cases** — when N or NW does not
    exist for the `err_sum[i]` neighbour gathering, spec says use 0
    (for N, W, WW) or N's value (for NW, NE). Round 3 used 0 for all
    out-of-range neighbours; corrected to use N's err for NW at
    column 0.
  - **2024-spec H.5.2 rightmost-column carry** — spec adds
    `err[i]_W` to `err_sum[i]` when `x == width - 1`. Round 3
    omitted this. Now applied via an explicit branch.
  - **2024-spec H.5 / H.4 max_error semantics** — round 3 used the
    PREVIOUS sample's max_error for property 15 of the CURRENT
    sample. Spec calls `wp_predict` first to get max_error for the
    current sample, then uses that as `property[15]` for the MA-tree
    decision. Restructured `decode_channels` to call WP up-front,
    use the result for both property 15 and (if the leaf picks
    predictor 6) the prediction value.
  - **`tests/round4_pixel_correctness.rs`** — full-image PNG-backed
    pixel-correctness harness (4 fixtures: `pixel-1x1`,
    `gray-64x64`, `gradient-64x64-lossless`, `palette-32x32`) plus
    a manual `palette_invasive_pixel_decode` diagnostic that walks
    decode_channels token-by-token printing bit positions, kept for
    round-5 work.
  - **`png` dev-dependency** (`png = "0.18"`) — pulled only by the
    test harness; no codec-semantics overlap with JPEG XL itself.
- **Round 3 (2024-spec)** — bit-alignment fix at the GlobalModular →
  inner-Modular boundary + ANS alias-mapping conditional-offset fix.
  After this round, `gray-64x64.jxl` decodes pixel-correct against
  its committed `expected.png` reference (gradient pattern
  `pixel(x, y) = ((x + y) * 2) & 0xff`, first scanline `0, 2, 4, …`).
  - **2024-spec C.3.2 (ANS state init position)** — round 1+2 read
    the ANS `u(32)` state initialiser EAGERLY at end of the entropy
    stream prelude inside `EntropyStream::read`. Empirical bisect
    against `cjxl 0.12.0` traces shows the state init is emitted
    AFTER the inner Modular sub-bitstream's ModularHeader (i.e.
    after `use_global_tree` / `WPHeader` / `nb_transforms` /
    `transforms`) and IMMEDIATELY before the first symbol decode.
    Round 3 splits the prelude reading from the state init reading
    via a new `EntropyStream::read_ans_state_init` method, which
    `decode_channels` invokes just before the first per-pixel
    `DecodeHybridVarLenUint` call. Position confirmed by tracing
    inner_use_global_tree against the expected `1` bit in cjxl's
    bytestream: bit 199 (gray-64x64), bit 338 (gradient-64x64),
    bit 359 (palette-32x32) all read `1` (true) once the state init
    is deferred — they were reading `0` (false) when the state init
    was eager.
  - **2024-spec C.2.6 (alias mapping conditional offset)** — round 1
    `AliasTable::lookup` always returned `offset = offsets[i] + pos`,
    but spec C.2.6 makes the formula CONDITIONAL on whether
    `pos >= cutoffs[i]`: in the "stays in own bucket" branch the
    offset is just `pos` (no `+ offsets[i]`). The unconditional
    formula caused incorrect ANS state evolution and triggered
    extra `u(16)` refills that ran the bitreader past EOF on
    small ANS-path fixtures. Round 3 adds the conditional.
  - **`gray-64x64.jxl` pixel-correct end-to-end** — first lossless
    Modular fixture > 1×1 to decode without EOF. Output checked
    against the gradient pattern in `docs/image/jpegxl/fixtures/
    gray-64x64/expected.png` first 16 pixels (0, 2, 4, …, 30) +
    histogram (min=0 max=252 mean=126.0).
  - **Diagnostic tooling**: `tests/round3_bit_alignment_bisect.rs`
    — eight tests (4 manual bisects + 4 production-path walks)
    that print bit positions at every spec milestone for the four
    target fixtures, with cross-reference comments against trace.
- **Round 2 (2024-spec)** — Inverse Modular transforms (Annex H.6) +
  full Self-correcting predictor (Annex H.5) + 2024-spec-correctness
  fixes for the entropy stream prelude (Annex C.2.1) and CLCL prefix
  decode (RFC 7932 §3.5), built additively on round 1's pixel-1x1
  pixel-correct decode.
  - **`modular_fdis::inverse_palette` (Annex H.6.4)** — full inverse
    palette transform incl. delta-palette via the verbatim
    `K_DELTA_PALETTE[72][3]` table (transcribed from FDIS Listing
    L.6), implicit colour extrapolation via the bitdepth-scaled
    formulas, and per-channel re-expansion from a single index
    channel + meta-channel palette to `num_c` colour channels.
  - **`modular_fdis::inverse_rct` (Annex H.6.3)** — all 6 RCT type
    modes (`type ∈ [0, 6]`) × 6 permutations = 42 `rct_type` codes,
    incl. the YCgCo branch (type==6) that uses the 4-step inverse.
    Channel triple `(A, B, C)` re-mapped to `(V[0], V[1], V[2])` via
    spec-formula permutations.
  - **`modular_fdis::horiz_isqueeze` / `vert_isqueeze` (Annex H.6.2)**
    — pair-merge inverse Squeeze step with the spec's `tendency()`
    function. Default-params (empty `squeeze_params`) defers to a
    later round.
  - **`global_modular::apply_transforms_to_channel_layout`** now
    handles Squeeze layout (channel dim halving + residu-channel
    insertion at `r + c - begin`).
  - **`global_modular`** applies inverse transforms in REVERSE order
    after `decode_channels` per H.6's "from last to first" rule,
    instead of erroring out as in round 1.
  - **`modular_fdis::WpState` + `wp_predict` (Annex H.5)** — full
    Self-correcting predictor with `true_err`, `sub_err[0..4]`
    per-channel arrays, 4 sub-predictor weights, and the H.5.2
    `error2weight` clamping. State updates after every sample
    decode regardless of whether predictor 6 was selected (so future
    predictor-6 calls see correct history).
  - **`modular_fdis::get_properties`** now wires `property[15]` to
    the WP `max_error` value (round 1 left it at 0).
  - **2024-spec C.2.1 fix in `ans::cluster::read_general_clustering`**:
    `use_prefix_code` ↔ `log_alphabet_size` mapping was reversed
    (round 1 fixed `EntropyStream::read` but missed the same swap
    in the cluster sub-stream).
  - **RFC 7932 §3.5 CLCL prefix-decode fix**: the 6-symbol
    code-length-code lookup interprets codewords as "bits parsed
    right to left" — the rightmost char of each codeword is the
    FIRST bit read. This is equivalent to LSB-first packing with
    no bit-reversal (round 1 incorrectly bit-reversed, breaking
    every fixture using complex-prefix codes).
  - **`bitreader::pu0` is now lenient** — does not enforce zero
    padding bits before byte boundaries. cjxl 0.12.0 emits non-zero
    padding on small fixtures (gradient-64x64, palette-32x32) at
    the metadata→frame_header alignment; the 2024 spec's text says
    the zero-padding is "for validity" only, not a decode-time
    requirement, and `djxl` accepts the same streams.
  - **`metadata_fdis::ImageMetadataFdis::read` tail dropped** — the
    FDIS-2021 `default_transform` Bool + `cw_mask` u(3) +
    per-mask F16 weight arrays were over-reading by 4-5 bits
    relative to libjxl's actual stream consumption. Round 2 leaves
    these at their defaults (`default_transform=true, cw_mask=0`)
    and SPECGAPs the exact gating condition.
  - **3 new soft fixture tests** (`r2_gradient_decode_attempt`,
    `r2_palette_decode_attempt`, `r2_gray_docs_decode_attempt`)
    against the docs/image/jpegxl/fixtures/ corpus. These currently
    fail at GlobalModular entropy stream prelude alignment in the
    complex-prefix path but the inverse-transform infrastructure
    they would feed is verified by unit tests.
  - **`pixel-1x1.jxl` regression-free** — the 1×1 RGB lossless
    acceptance fixture from round 1 still decodes to R=255 G=0 B=0.

- **Round 1 (2024-spec)** — Modular sub-bitstream pixel decode
  end-to-end against the final ISO/IEC 18181-1:2024 core spec (Annex
  H), built on top of the round-1..3 baseline:
  - `modular_fdis::evaluate_tree` walks decision-node MA trees per
    H.4.1, replacing the round-3 single-leaf-only restriction.
  - `modular_fdis::get_properties` computes the 16 base properties
    of Table H.4 plus per-previous-channel properties (4 each for
    every channel with matching dims/shifts).
  - `modular_fdis::Neighbours` materialises the 7 prediction
    neighbours per Table H.2 with the H.3 edge-case fallbacks.
  - `modular_fdis::predict` covers Table H.3 predictors 0-5 + 7-13;
    predictor 6 (Self-correcting) is implemented for the trivial
    (0, 0) origin case (returns 0 — full WP defers to round 2).
  - `modular_fdis::TransformInfo` + `TransformId` parses the H.7
    bundle for `nb_transforms > 0`; channel-list adjustment for
    Palette is applied; inverse Palette / Squeeze application defers
    to round 2 with a clean `Error::Unsupported` exit point.
  - `decode_codestream` accepts RGB images (3 channels) in addition
    to Grey, producing 3 / 1 plane VideoFrames respectively.
  - `pixel-1x1.jxl` (1×1 RGB lossless, 22 B fixture from
    `docs/image/jpegxl/fixtures/pixel-1x1/`) now decodes
    pixel-correct: R=255, G=0, B=0 (matches `expected.png`).
  - Black-box validator test for `djxl` confirms the binary decodes
    the same `gray-64x64` fixture; we never read djxl/cjxl source.
- **FDIS-2021 spec typo #5 documented and corrected**: D.3.1's
  `use_prefix_code` ↔ `log_alphabet_size` mapping was swapped in the
  FDIS 2021 text (`if use_prefix_code is 1 → log_alphabet_size = 5 +
  u(2)`); the 2024-published edition (C.2.1) reverses it (prefix →
  15, ANS → 5+u(2)) which matches the libjxl reference output
  observed via cjxl/djxl. The implementation in
  `modular_fdis::EntropyStream::read` now follows the 2024 reading.

### Removed

- **Decoder rounds 7-11 + encoder rounds 1-6 RETIRED 2026-05-08** under
  fruits-of-poisonous-tree. The `OxideAV/docs` repository retired
  `image/jpegxl/libjxl-trace-reverse-engineering.md` (the 792-line
  behavioural-trace writeup) on 2026-05-06 (commit `d732002`); the
  retire reasoning applies to any code authored by an agent that read
  that doc, even when no source was literally quoted. This crate's
  master was reset to `9d79695` (round-3 LfGlobal + GlobalModular +
  Modular sub-bitstream wiring, 2026-05-01) — the last commit before
  the retired trace doc landed in `OxideAV/docs` (`8931c26`,
  2026-05-02 22:55). The pre-retirement history is preserved on the
  `old` branch for forensics.
  - **Retired decoder commits**: `403f256` (round 7 — typo #6/#7 +
    MA-tree decodes), `06b4d00` (modular pre-check scope),
    `d49e583` (round 8 — prefix early-terminate),
    `ba225c2` / `1217a08` / `1540102` / `7827d96` / `52b1cfb` /
    `8258cdc` / `a2419a6` (round 9 — typo #8 + instrumentation),
    `ab5f94a` (round 10 — kRCT/kPalette/kSqueeze parsing + dispatch),
    `2e41c1d` (round 11 — Appendix B four-range index partition).
  - **Retired encoder commits**: `a53e041` / `198f9e4` / `5f35de8` /
    `f83a6d8` / `0c9b9d8` / `88f05ee` / `6215efc` / `39b2e73` /
    `dd8be6e` / `65195e5` / `1925527` / `fedb620` / `9804c79` (encoder
    rounds 1-6 — independent codec surface but authored within the
    same trace-doc-contaminated session window).
  - **Retired infrastructure commits**: `4f1b6bd` (CI workflow
    centralisation), `9a8b33d` (standalone-friendly registry feature),
    `2cb9943` (register_containers extension lookup), `dd68816`
    (register entry-point unification), `cde6f6a` (auto-register
    macro), `e4ea5b7` (`make_decoder` → `first_decoder` rename),
    `852ac81` (re-export `__oxideav_entry`), `9d3e999` (drop linkme
    dep). Re-applicable in non-narrative plumbing rounds later.
  - **Retired crates.io versions** (yank pending): v0.0.5 (published
    2026-05-04), v0.0.6 (2026-05-04), v0.0.7 (2026-05-05). Tags
    v0.0.5 / v0.0.6 / v0.0.7 deleted from `origin`. Version bumped
    0.0.4 → 0.0.8 in this commit to skip the yanked range.
  - **Forward path**: a strict-isolation `docs/image/jpegxl-cleanroom/`
    workspace with the four-role layout (Specifier / Extractor /
    Implementer / Auditor) — Specifier wall: ISO/IEC 18181-1 FDIS +
    18181-3 conformance corpus only, no libjxl source ever. Modelled
    after `docs/video/msmpeg4/`, `docs/video/magicyuv/`,
    `docs/audio/tta-cleanroom/`. Until that workspace exists, this
    crate ships only the round-1..3 ANS + headers + LfGlobal +
    GlobalModular wiring; no further decoder rounds will land.

### Changed

- API shim for the post-retire workspace: `register(ctx: &mut RuntimeContext)`
  + `register_codecs(reg: &mut CodecRegistry)` + `oxideav_core::register!`
  macro call (current registration pattern); the round-1..3 test that
  used `reg.make_decoder` now uses `ctx.codecs.first_decoder` to match
  the post-rename `oxideav-core` API.

### Added

- New `ans` module implementing the FDIS 18181-1:2021 Annex D entropy
  layer (round 1 of the committee-draft → FDIS migration). Submodules:
  - `ans::prefix` — Brotli (RFC 7932) §3.4 simple + §3.5 complex
    prefix codes, used by the `use_prefix_code == 1` histogram path
    of D.3.1.
  - `ans::alias` — alias-mapping table init + lookup (D.3.2,
    Listings D.1 + D.2). Implements Vose's alias method with the
    spec PDF's u/o/i variable typo corrected.
  - `ans::symbol` — 32-bit-state ANS reverse decoder (D.3.3,
    Listing D.3) including the `0x130000` end-of-stream check.
  - `ans::distribution` — ANS distribution decoder (D.3.4,
    Listing D.4) with the verbatim 128 × 2 `kLogCountLut` lookup
    table transcribed from p. 64 of the FDIS PDF.
  - `ans::cluster` — distribution clustering simple-path + the
    inverse move-to-front transform (D.3.5, Listing D.5).
  - `ans::hybrid` — hybrid-integer LZ77 decode driver (D.3.6,
    Listing D.6) with the verbatim 120 × 2 `kSpecialDistances`
    lookup table transcribed from p. 66 of the FDIS PDF, plus a
    1 MiB sliding window per stream.
  - `ans::hybrid_config` — `HybridUintConfig` decode + `ReadUint`
    (D.3.7, Listing D.7).
  Every allocation is bounded against the input length; the
  module ships 45 self-contained unit tests covering hand-built
  bitstreams from each spec listing plus four malicious-input
  cases (oversized log_alphabet_size, oversized alphabet, huge
  hybrid token, huge prefix-code alphabet).
  The committee-draft `abrac` / `begabrac` / `matree` / `modular`
  pipeline and the registered `make_decoder` are intentionally
  untouched — round 2 will wire the new ANS coder behind a
  FrameHeader + TOC entry point.
- `BitReader` gains `peek_bits(n)` / `advance_bits(n)` / `bits_remaining()`
  / `read_u8_value()` to support the ANS distribution decoder
  (D.3.4 reads `u(7)` for the kLogCountLut key without advancing,
  then advances by the table-derived step count).

- Modular sub-bitstream channel decoder per the 2019 committee draft
  (`arxiv-1908.03565v2`, Annexes C.9 + D.7), a stepping stone toward
  full FDIS 18181-1 support. New modules:
  - `abrac` — bit-level adaptive range coder (D.7).
  - `begabrac` — bounded-Exp-Golomb integer coder over a known signed
    range, layered on `abrac` (D.7.1).
  - `matree` — meta-adaptive decision tree that picks a per-context
    BEGABRAC for each pixel (D.7.2 / D.7.3).
  - `predictors` — five named pixel predictors (Zero, Average,
    Gradient, Left, Top) from C.9.3.1.
  - `modular` — channel-header parser plus the per-pixel property +
    predictor + entropy decode loop, exposed as
    `modular::decode_single_channel`.
  - `BitReader` gains `pu0()` (zero-padded byte align), `pu()`
    (byte-align value), `read_varint()` (A.3.1.5), and a `data()`
    accessor used by entropy coders that switch from bits to bytes.
- DoS-hardening of the Modular decode path against malformed
  channel headers and adversarial entropy streams:
  - `Channel::try_new` refuses dimensions larger than
    `MAX_CHANNEL_DIM` (32 768) per side or pixel counts above
    `MAX_CHANNEL_PIXELS` (256 M); the bitstream-driven entry point
    `decode_single_channel` now uses `try_new` so a forged
    width/height pair returns `InvalidData` instead of asking the
    allocator for terabytes.
  - `MaTree::decode` caps the bit-depth `n` at `MAX_VALUE_BIT_DEPTH`
    (32) so a pathological caller can't make each leaf BEGABRAC
    allocate gigabytes of mantissa context.
  - `decode_subtree` caps the total node count at
    `MAX_MA_TREE_NODES` (1 << 20) and recursion depth at
    `MAX_MA_TREE_DEPTH` (1024), preventing both heap exhaustion and
    stack overflow when the entropy stream keeps emitting "decision
    node" instead of "leaf".
- Regression tests for the hardening above, including a
  hand-crafted 1 M × 1 M channel-header fixture that asserts
  `decode_single_channel` rejects with `InvalidData` rather than
  allocating.

### Changed

- Crate description updated to mention the Modular sub-bitstream
  decode now landed (committee-draft path).
- Doc-comment in `lib.rs` updated to reflect the new module layout
  and the remaining gap toward FDIS 18181-1 (FrameHeader/TOC,
  Squeeze, VarDCT, ANS-based entropy).

### Removed

- `SPEC_BLOCKED.md`: the ISO/IEC 18181-1 normative spec (committee
  draft + FDIS) is now present in `docs/image/jpegxl/`, so the
  block is lifted. Migration to the FDIS layout (ANS entropy,
  FrameHeader, TOC, ImageMetadata FDIS shape) is tracked as the
  next round of work, not a block.

## [0.0.4](https://github.com/OxideAV/oxideav-jpegxl/compare/v0.0.3...v0.0.4) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- drop Cargo.lock — this crate is a library
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- bump to oxideav-core 0.1.1 + codec 0.1.1
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
