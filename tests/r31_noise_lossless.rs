//! Round 31 (parent-dispatch r16) — §F.3 single-TOC-entry section
//! zero-padding fix, exercised by the `noise-64x64-lossless` fixture
//! (`docs/image/jpegxl/fixtures/noise-64x64-lossless/`, encoded with
//! `cjxl SOURCE input.jxl -d 0 -e 7`, 13505 B).
//!
//! ## What round-30 deferred
//!
//! Round-30 left the noise fixture as the round-31 docs-gap target:
//! the decoder hard-rejected the mid-pixel-decode stage with
//! `unexpected end of JXL bitstream`. Bisecting showed the root cause
//! was that the **single-TOC-entry fast path** in
//! `lib::decode_codestream` was routing `LfGlobal::read` through the
//! non-padding main bit reader (`BitReader::new`), while the rest of
//! the decoder uses a padding section reader (`BitReader::new_section`)
//! that implements FDIS §F.3 first paragraph normatively:
//!
//! > "When decoding a section, no more bits are read from the codestream
//! > than 8 times the byte size indicated in the TOC; if fewer bits are
//! > read, then the remaining bits of the section all have the value
//! > zero."
//!
//! The high-entropy noise fixture (`cjxl -e 7`) packs the per-pixel
//! ANS / hybrid-uint reads so tightly that the last sample's
//! renormalisation refill reaches a few bits past the last byte; with
//! the non-padding reader those reads errored, with the section reader
//! they correctly read zero per §F.3 and the decode now runs to
//! completion. Six prior lossless fixtures happened to have enough
//! slack at the section tail that the bug never fired.
//!
//! ## What this test asserts
//!
//! After the §F.3 fix, `decode_one_frame` on the noise fixture returns
//! a 3-plane 64×64 8-bit RGB `VideoFrame` with byte-packed planes
//! (stride == width, no error). This nails down the §F.3 single-section
//! fast-path behaviour as a regression baseline.
//!
//! ## What this test does NOT yet assert (deferred)
//!
//! The decoded pixels are not yet byte-identical to `expected.png`.
//! Round-31 traced the first divergence to plane[0] (R) at (2, 3) —
//! i.e. after 194 of 4096 samples in plane 0 decode correctly. From
//! that sample on, ~98 % of samples diverge. The divergence point is
//! deterministic (re-runs hit the same (x, y)) and well within the
//! section's real (non-padding) byte budget, ruling out the §F.3 fix
//! as the source. The remaining gap is therefore a separate latent
//! state-evolution bug not exposed by the seven smaller fixtures
//! (whose MA trees have far fewer contexts — 6 or fewer — vs the
//! noise fixture's 84). Suspect areas:
//!
//!   * MA-tree leaf decode with `num_contexts > 16` (the leaf-stream
//!     EntropyStream's cluster_map is 84 → 3 clusters here);
//!   * the Self-correcting WP state when neighbour history is
//!     uncorrelated (high-entropy input stresses the WP state machine);
//!   * the hybrid-uint extra-bits path for large `n_extra` values
//!     (high-entropy → many above-`split` tokens).
//!
//! Bit-position bisects across the noise fixture's ~108 kbits would
//! need ~30 rounds of the per-token cluster trace already shipped in
//! `tests/round24_d1_disttrace.rs`. Deferring pixel-correctness to a
//! follow-up round; locking in the §F.3 fix here.
//!
//! Spec citations:
//!   * FDIS §F.3 first paragraph — section bit-budget + zero-pad rule.
//!   * FDIS Annex C.4 + C.9 — LfGlobal / GlobalModular sub-bitstream.
//!   * FDIS Annex G.1.3 — kModular channel order (R, G, B for this
//!     `colour_space = RGB` fixture, no extra channels).
//!
//! Black-box oracle (cross-check, NOT used at test time): `djxl
//! v0.11.1 input.jxl /tmp/out.png` produces the same PNG modulo
//! per-byte equality with `expected.png`.

use oxideav_jpegxl::decode_one_frame;

const NOISE_JXL: &[u8] = include_bytes!("fixtures/noise_64x64_lossless.jxl");

#[test]
fn noise_64x64_lossless_decodes_without_eof_error() {
    // Pre-round-31: this call errored with InvalidData("unexpected end
    // of JXL bitstream") mid-pixel-decode. Post-fix: it returns a
    // 3-plane 64×64 8-bit RGB VideoFrame.
    let vf = decode_one_frame(NOISE_JXL, None)
        .expect("noise-64x64-lossless must decode after §F.3 fast-path fix");
    assert_eq!(
        vf.planes.len(),
        3,
        "noise-64x64-lossless: expected 3 RGB planes, got {}",
        vf.planes.len(),
    );
    for (i, p) in vf.planes.iter().enumerate() {
        assert_eq!(p.stride, 64, "plane[{i}] stride must be width=64");
        assert_eq!(
            p.data.len(),
            64 * 64,
            "plane[{i}] must hold width*height=4096 samples",
        );
    }
}

/// Regression: confirm the seven pre-round-31 small lossless fixtures
/// (pixel-1x1, gray-64x64, gradient-64x64, palette-32x32, grey-8x8,
/// alpha-64x64, bit-depth-16) still decode after the §F.3 fast-path
/// fix.
#[test]
fn pre_round31_seven_lossless_fixtures_still_decode() {
    for (label, bytes) in [
        ("pixel-1x1", &include_bytes!("fixtures/pixel_1x1.jxl")[..]),
        (
            "gray-64x64",
            &include_bytes!("fixtures/gray_64x64_lossless.jxl")[..],
        ),
        (
            "gradient-64x64",
            &include_bytes!("fixtures/gradient_64x64_lossless.jxl")[..],
        ),
        (
            "palette-32x32",
            &include_bytes!("fixtures/palette_32x32.jxl")[..],
        ),
        (
            "grey-8x8",
            &include_bytes!("fixtures/grey_8x8_lossless.jxl")[..],
        ),
        (
            "alpha-64x64",
            &include_bytes!("fixtures/alpha_64x64.jxl")[..],
        ),
        (
            "bit-depth-16",
            &include_bytes!("fixtures/bit_depth_16.jxl")[..],
        ),
    ] {
        let _ = decode_one_frame(bytes, None)
            .unwrap_or_else(|e| panic!("post-r31 regression: {label} failed to decode: {e}"));
    }
}
