//! Round 488 — decoder regression fixtures for the transform and
//! restoration-filter corrections of this round.
//!
//! Provenance: each `r488_*.jxl` was written by this crate's VarDCT
//! encoder (a round-488 development build forcing particular coding
//! tools) from a synthesised picture — fractal plasma under a noise
//! texture with saturated flat shapes drawn on top; the matching
//! `_expected.png` is the output of the installed black-box decoder
//! (`djxl`) for that stream. Every stream decodes to within one 8-bit
//! code of the black-box output.
//!
//! * `r488_transforms` — a 384×256 picture whose 64×64 tiles each use
//!   one transform type, all 21 of DCT8×8 … DCT64×64, the rectangular
//!   DCTs (both orientations), DCT4×4, DCT4×8, DCT8×4, DCT2×2, IDENTITY
//!   and AFV0..3, at a small distance (high-frequency B coefficients
//!   present). Before round 488 the tall rectangular types above
//!   16×8 (LLF placement), DCT8×4 (orientation), AFV (DC terms) and the
//!   DCT8×8 B-channel weights were decoded differently.
//! * `r488_epf1_gab`, `r488_epf2_gab`, `r488_epf3` — the Gabor-like
//!   transform and one, two and three iterations of the
//!   edge-preserving filter over a varying quantisation field (the
//!   filter strength follows the quantisation width of each block).

mod common;

use common::png::decode_png;
use oxideav_jpegxl::{PixelFormat, decode};

fn check(name: &str) {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let jxl = std::fs::read(format!("{dir}/r488_{name}.jxl")).unwrap();
    let want = decode_png(&std::fs::read(format!("{dir}/r488_{name}_expected.png")).unwrap());
    let got = decode(&jxl).unwrap();
    assert_eq!(got.format, PixelFormat::Rgb24, "{name}");
    assert_eq!((got.width, got.height), (want.width, want.height), "{name}");
    let ours = got.as_bytes().unwrap();
    assert_eq!(ours.len(), want.data.len(), "{name}");
    let mut worst = 0u8;
    let mut sum = 0u64;
    for (&a, &b) in ours.iter().zip(&want.data) {
        let d = a.abs_diff(b);
        worst = worst.max(d);
        sum += u64::from(d);
    }
    let mean = sum as f64 / ours.len() as f64;
    assert!(
        worst <= 1 && mean < 0.3,
        "{name}: differs from the black-box decode (worst {worst}, mean {mean:.3})"
    );
}

#[test]
fn every_transform_type_matches_the_black_box_decode() {
    check("transforms");
}

#[test]
fn gabor_and_one_epf_iteration_match_the_black_box_decode() {
    check("epf1_gab");
}

#[test]
fn gabor_and_two_epf_iterations_match_the_black_box_decode() {
    check("epf2_gab");
}

#[test]
fn three_epf_iterations_match_the_black_box_decode() {
    check("epf3");
}
