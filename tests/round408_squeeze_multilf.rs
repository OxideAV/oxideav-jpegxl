//! Round 408 — Squeeze end-to-end pixel validation + §C.5.2
//! ModularLfGroup (multi-LfGroup Modular frames).
//!
//! Three encoder-generated responsive-mode (Squeeze) fixtures, each
//! against a black-box reference decode:
//!
//! * `sq_32` (32×32, single group): **bit-exact** — pins the Listing
//!   I.19 default-parameter sequence, the Listing I.21 tendency
//!   FLOOR-division erratum, and the Listing D.8 `rleft = 0` column-0
//!   rule end-to-end.
//! * `sq_512` (512×512, 4 groups, 1 LfGroup): MAD ratchet — a
//!   PRE-EXISTING sporadic per-sample residual-decode divergence
//!   remains on multi-group Squeeze streams (~2 % of samples in the
//!   group-scoped residual channel decode to slightly wrong values;
//!   scattered, not group-boundary-aligned). Round 408 improved this
//!   stream's MAD from 1.02 to 0.27 (tendency floor); the ratchet
//!   pins the current bound so the tail can only shrink.
//! * `grayscale_public_university` (2880×1620, ISO/IEC 18181-3
//!   conformance, 2 LfGroups): decodes end-to-end through the new
//!   §C.5.2 per-LfGroup walk (was: hard `Unsupported`); same
//!   sporadic-residual tail, MAD ratchet 1.7 (was undecodable).

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
fn squeeze_multigroup_512_ratchet() {
    let (mad, max) = decode_and_compare(
        include_bytes!("fixtures/sq_512.jxl"),
        include_bytes!("fixtures/sq_512_expected.png"),
    );
    // Known tail: sporadic residual-channel decode divergence on
    // multi-group Squeeze (see module docs). Ratchet only — tighten
    // to bit-exact when the tail is closed.
    assert!(mad < 0.3, "sq_512 MAD ratchet regressed: {mad} (max {max})");
    assert!(max <= 8, "sq_512 max-error ratchet regressed: {max}");
}

#[test]
fn conformance_grayscale_public_university_decodes() {
    let (mad, max) = decode_and_compare(
        include_bytes!("fixtures/conformance_grayscale_public_university.jxl"),
        include_bytes!("fixtures/conformance_grayscale_public_university_expected.png"),
    );
    // 2 LfGroups (2880 px wide) — pins the §C.5.2 ModularLfGroup walk
    // (channels with hshift >= 3 && vshift >= 3 decoded per LF group,
    // slices clamped to the channel extent). Same sporadic-residual
    // tail as sq_512; ratchet only.
    assert!(
        mad < 1.8,
        "grayscale_public_university MAD ratchet regressed: {mad} (max {max})"
    );
}
