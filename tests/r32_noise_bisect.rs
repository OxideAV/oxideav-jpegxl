//! Round 32 — `noise-64x64-lossless` pixel-divergence bisect,
//! **promoted in round 278 to a full pixel-equality regression**.
//!
//! ## History (rounds 31..272)
//!
//! Round 31 fixed the §F.3 zero-pad on the single-TOC-entry LfGlobal
//! fast path so that the noise fixture decode-completes without an
//! EOF error. Round 32 bisected the remaining pixel divergence to a
//! single locus: the first divergent plane-0 byte at linear index
//! **194** (y=3, x=2), a `predictor == 6` (Self-correcting Weighted
//! Predictor, FDIS Annex E / §H.5.2) sample whose `wp_pred8 = 717`
//! was off-by-1-in-quotient-of-`>>3` from the spec value (709 per the
//! staged behavioural trace). Rounds 126..272 progressively localised
//! the defect to upstream WP state evolution (the stored `true_err`
//! at sample 129 was -21 from spec).
//!
//! ## Round-278 resolution
//!
//! Two FDIS Annex E reading fixes landed in `modular_fdis::wp_predict`:
//!
//! 1. `error2weight` (Listing E.2) computes the inner
//!    `(1 << 24) Idiv ((err_sum >> shift) + 1)` division FIRST and
//!    multiplies the truncated quotient by `maxweight` (the FDIS-2021
//!    parenthesisation), instead of multiplying `maxweight` into the
//!    numerator before dividing. The staged trace's 52 full-precision
//!    `(err_sum, weight)` cells (samples 188..200) discriminate the
//!    two readings: the Idiv-first reading matches all 52, the
//!    multiply-first reading mismatches 18.
//! 2. The `true_errNW` read falls back to `true_errN` when NW does
//!    not exist (x == 0) — the same H.5.2 "if NW or NE does not
//!    exist, the value of N is used instead" edge rule the err_sum
//!    accumulator reads already applied. The previous zero fallback
//!    corrupted every column-0 prediction and, through the stored
//!    true_err/sub_err history, produced the sample-129 Δ = -21.
//!
//! With both fixes a from-scratch Annex E state-evolution walk over
//! the fixture's decoded values reproduces every traced cell of
//! `docs/image/jpegxl/fixtures/noise-64x64-lossless/
//! wp-trace-sample-194.md` exactly, and the production decode is now
//! byte-exact against `expected.png` on all three planes — which is
//! what this test pins.

use oxideav_jpegxl::decode_one_frame;
use std::io::Cursor;

const NOISE_JXL: &[u8] = include_bytes!("fixtures/noise_64x64_lossless.jxl");
const EXPECTED_PNG: &[u8] = include_bytes!("fixtures/noise_64x64_lossless_expected.png");

#[test]
fn r32_noise_lossless_is_pixel_exact() {
    let vf = decode_one_frame(NOISE_JXL, None).expect("noise fixture must decode");
    let decoder = png::Decoder::new(Cursor::new(EXPECTED_PNG));
    let mut reader = decoder.read_info().expect("png info");
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).expect("png frame");
    let raw = &buf[..info.buffer_size()];

    assert_eq!(vf.planes.len(), 3, "noise fixture must decode 3 planes");
    for (c, plane) in vf.planes.iter().enumerate() {
        assert_eq!(
            plane.data.len(),
            4096,
            "plane[{c}] must hold 64*64=4096 byte samples",
        );
        for i in 0..4096 {
            assert_eq!(
                plane.data[i],
                raw[i * 3 + c],
                "plane[{c}][{i}] (y={}, x={}) dec={} exp={} — the \
                 noise-64x64-lossless fixture decoded byte-exact from \
                 round 278 onward (WP error2weight Idiv-first + \
                 true_errNW column-0 fallback); any mismatch is a WP \
                 state-evolution regression",
                i / 64,
                i % 64,
                plane.data[i],
                raw[i * 3 + c]
            );
        }
    }

    // The historical round-32 bisect anchor: sample 194 (y=3, x=2) of
    // plane 0 was the first divergent byte for 246 rounds. Call it out
    // individually so a regression names the locus directly.
    assert_eq!(
        vf.planes[0].data[194], 34,
        "plane[0][194] must decode the spec value 34 (round-32's \
         historical first-divergence pinned dec=35 until round 278)"
    );
}
