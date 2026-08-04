//! Round 408 — Squeeze end-to-end pixel validation + §C.5.2
//! ModularLfGroup (multi-LfGroup Modular frames).
//!
//! Three encoder-generated responsive-mode (Squeeze) fixtures, each
//! against a black-box reference decode:
//!
//! * `sq_32` (32×32, single group): **bit-exact** — pins the Listing
//!   I.19 default-parameter sequence, the Listing I.21 tendency
//!   erratum, and the Listing D.8 `rleft = 0` column-0 rule
//!   end-to-end.
//! * `sq_512` (512×512, 4 groups, 1 LfGroup): **bit-exact** since
//!   round 420 — the round-408 "sporadic multi-group residual tail"
//!   (MAD 0.27) was never a group-boundary issue: it was the Listing
//!   I.21 tendency mis-rounding exact negative half-ties (`4A - 3C -
//!   B ≡ 6 mod 12`, ascending). Ties round HALF-AWAY-FROM-ZERO.
//! * `grayscale_public_university` (2880×1620, ISO/IEC 18181-3
//!   conformance, 2 LfGroups, LOSSY Squeeze + gab + 3-pass EPF): the
//!   Modular pyramid decode is entropy-verified in sync (all 87
//!   sub-bitstreams end on the D.3.3 ANS final-state invariant); the
//!   residual MAD is restoration-filter accuracy, not Squeeze:
//!   1.68 (r408, no filters) → 1.00 (r420: Gabor-like transform
//!   wired + the §J.2 weight-sign erratum keeping EPF sane). The
//!   remaining ≈1.0 is the §J.3-for-kModular sample-domain docs-gap
//!   (see `apply_modular_restoration_filters`).

use std::io::Cursor;

fn png_grey(bytes: &[u8]) -> (usize, usize, Vec<u8>) {
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    assert_eq!(info.color_type, png::ColorType::Grayscale);
    (info.width as usize, info.height as usize, buf)
}

fn decode_and_compare(jxl: &[u8], expected_png: &[u8]) -> (f64, u8) {
    let frame = oxideav_jpegxl::decode_one_frame(jxl, None).expect("decode");
    let (w, h, reference) = png_grey(expected_png);
    let plane = &frame.planes[0];
    let mut sum = 0u64;
    let mut max = 0u8;
    for y in 0..h {
        for x in 0..w {
            let d = plane.data[y * plane.stride + x].abs_diff(reference[y * w + x]);
            sum += d as u64;
            max = max.max(d);
        }
    }
    (sum as f64 / (w * h) as f64, max)
}

#[test]
fn squeeze_default_params_32x32_bit_exact() {
    let (mad, max) = decode_and_compare(
        include_bytes!("fixtures/sq_32.jxl"),
        include_bytes!("fixtures/sq_32_expected.png"),
    );
    assert_eq!(max, 0, "single-group Squeeze must be bit-exact (MAD {mad})");
}

#[test]
fn squeeze_multigroup_512_bit_exact() {
    let (mad, max) = decode_and_compare(
        include_bytes!("fixtures/sq_512.jxl"),
        include_bytes!("fixtures/sq_512_expected.png"),
    );
    // Round 420: the Listing I.21 half-tie erratum closed the
    // round-408 residual tail — multi-group Squeeze is bit-exact.
    assert_eq!(max, 0, "multi-group Squeeze must be bit-exact (MAD {mad})");
}

#[test]
fn conformance_grayscale_public_university_decodes() {
    let (mad, max) = decode_and_compare(
        include_bytes!("fixtures/conformance_grayscale_public_university.jxl"),
        include_bytes!("fixtures/conformance_grayscale_public_university_expected.png"),
    );
    // 2 LfGroups (2880 px wide) — pins the §C.5.2 ModularLfGroup walk
    // (channels with hshift >= 3 && vshift >= 3 decoded per LF group,
    // slices clamped to the channel extent). The Modular decode is
    // entropy-verified in sync; the ratchet bounds the remaining §J
    // restoration-filter accuracy. Was 1.8 before round 420 (no EPF),
    // 1.00 rounds 420-431 (literal sample-domain reading), 0.2909
    // since round 437 resolved the fdis-errata.md Part 9 kModular
    // sample-domain + grey-mapping SPECGAP by the prescribed grid
    // bisection (see `round437_modular_epf_posture`).
    assert!(
        mad < 0.32,
        "grayscale_public_university MAD ratchet regressed: {mad} (max {max})"
    );
}
