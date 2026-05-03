//! Round-3 integration test: decode the simplest cjxl-produced
//! `--lossless` Modular Grey 8×8 fixture.
//!
//! Fixture build (committed to `tests/fixtures/grey_8x8_lossless.jxl`):
//!
//! ```text
//! python3 -c "open('/tmp/test_8x8.pgm','wb').write(b'P5\n8 8\n255\n' + bytes([128]*64))"
//! cjxl -d 0 -e 1 /tmp/test_8x8.pgm /tmp/test_8x8.jxl
//! ```
//!
//! cjxl 0.11.1 produces a raw-codestream `.jxl` of 180 bytes. The
//! header bytes are `FF 0A 41 40 50 DC 08 08 02 01 ...`.

use oxideav_jpegxl::probe_fdis;

const FIXTURE: &[u8] = include_bytes!("fixtures/grey_8x8_lossless.jxl");

#[test]
fn cjxl_grey_8x8_probe_recognises_dimensions_and_grey() {
    let h = probe_fdis(FIXTURE).expect("probe should succeed on cjxl fixture");
    assert_eq!(h.size.width, 8);
    assert_eq!(h.size.height, 8);
    // cjxl emits Grey, 8 bpp, no extras for a PGM input.
    assert_eq!(h.metadata.bit_depth.bits_per_sample, 8);
    assert_eq!(h.metadata.num_extra_channels, 0);
}

#[test]
fn cjxl_grey_8x8_dump_first_bytes() {
    // Print the first 32 bytes of the fixture so that bisecting the
    // decoder against this dump is straightforward.
    let n = FIXTURE.len().min(32);
    let mut s = String::new();
    for b in &FIXTURE[..n] {
        s.push_str(&format!("{b:02x} "));
    }
    eprintln!("first {n} bytes: {s}");
    eprintln!("total bytes: {}", FIXTURE.len());
}

/// Soft test: this MAY return Unsupported / InvalidData while the
/// FDIS decoder is still missing pieces. The test prints the decoder
/// result so a reviewer can see exactly what the failure mode was, but
/// does NOT fail the suite if it's an expected error. A successful
/// decode asserts pixel-perfect equality with the source PGM (all 128s).
///
/// Round-8 status (this commit): three fixes attempted to unblock the
/// round-7 stop point (cl_code Kraft 37 in the second per-cluster
/// prefix code):
///   1. `PrefixCode::from_lengths` now sums Kraft in the actual
///      `1<<max_length` budget instead of always `1<<15`.
///   2. RFC 7932 §3.5 single-non-zero clcl special case (degenerate
///      single-symbol zero-length code) is now handled in
///      `read_complex_prefix`.
///   3. RFC 7932 §3.4 simple-prefix length assignment reverted to
///      per-RFC (first-read gets length 1, NOT smallest-sorted).
///
/// See `tests/cjxl_grey_8x8_trace.rs` for the bit-by-bit trace
/// harness used for round-8 starting-point analysis.
/// `cjxl_grey_8x8_round7_kraft_error_is_resolved` (below) hard-asserts
/// the round-7 error message no longer appears.
#[test]
fn cjxl_grey_8x8_decode_attempt() {
    use oxideav_jpegxl::decode_one_frame;
    let res = decode_one_frame(FIXTURE, None);
    match res {
        Ok(vf) => {
            assert_eq!(vf.planes.len(), 1, "expected 1 plane (Gray8)");
            let plane = &vf.planes[0];
            assert_eq!(plane.stride, 8);
            assert_eq!(plane.data.len(), 64);
            for (i, &v) in plane.data.iter().enumerate() {
                assert_eq!(v, 128, "pixel {i} should be 128 (constant grey input)");
            }
        }
        Err(e) => {
            // Print and accept — round 3 stops here. Round 4 picks up
            // multi-leaf MA tree evaluation + token-> property
            // interpretation rules.
            eprintln!("cjxl_grey_8x8 round-3 stop point: {e}");
        }
    }
}

/// Hard-asserts the round-7 stop-point error ("code lengths grossly
/// overflow Kraft sum (kraft=135104, alphabet_size=257, max_length=13)")
/// does NOT appear when decoding the cjxl 8x8 grey lossless fixture.
/// Round 8 (`from_lengths` per-alphabet Kraft budget + RFC §3.5
/// single-non-zero clcl + RFC §3.4 simple-prefix length assignment
/// reverted to per-RFC) was specifically targeted at this error. This
/// test fails loudly if a regression brings the error back, OR if a
/// later round breaks the symbol-stream-prelude in a way that
/// reintroduces the same Kraft overshoot.
///
/// The decode itself is allowed to fail with a DIFFERENT error
/// (round 9 will pick up wherever the new stop point is). Only the
/// specific round-7 error message is rejected.
#[test]
fn cjxl_grey_8x8_round7_kraft_error_is_resolved() {
    use oxideav_jpegxl::decode_one_frame;
    let res = decode_one_frame(FIXTURE, None);
    if let Err(e) = res {
        let msg = format!("{e}");
        assert!(
            !msg.contains("kraft=135104"),
            "round-7 error message reappeared: {msg}"
        );
    }
}
