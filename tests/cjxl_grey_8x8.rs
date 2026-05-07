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

/// Soft-was-once: the grey-8x8 fixture decode. As of round 5 the prefix-
/// code Kraft early-stop (RFC 7932 §3.5) is implemented; cjxl 0.11.1's
/// emitted complex prefix histograms now decode through the symbol-
/// stream prelude. The test stays soft (errors are logged not asserted)
/// because pixel-correctness on this fixture is not yet a hard target —
/// the round-5 hard test for grey_8x8 lives in
/// `tests/round5_grey_8x8_pixel_correctness.rs`.
#[test]
fn cjxl_grey_8x8_decode_attempt() {
    use oxideav_jpegxl::decode_one_frame;
    let res = decode_one_frame(FIXTURE, None);
    match res {
        Ok(vf) => {
            eprintln!("cjxl_grey_8x8 decoded: {} planes", vf.planes.len());
            if !vf.planes.is_empty() {
                let p = &vf.planes[0];
                eprintln!(
                    "  plane[0] stride={} len={} first 8 bytes: {:?}",
                    p.stride,
                    p.data.len(),
                    &p.data[..8.min(p.data.len())]
                );
            }
        }
        Err(e) => {
            eprintln!("cjxl_grey_8x8 round-5 stop point: {e}");
        }
    }
}

/// Round-5 hard test: pixel-correctness for grey_8x8_lossless.
///
/// The fixture is a constant-grey 8×8 PGM (all bytes = 128) encoded by
/// cjxl 0.11.1 with `--lossless`. Round 5's RFC 7932 §3.5 Kraft
/// early-stop fix unblocks the prefix-code histogram decode, after
/// which the symbol stream + per-pixel decode chain succeed.
#[test]
fn cjxl_grey_8x8_pixel_correct() {
    use oxideav_jpegxl::decode_one_frame;
    let vf = decode_one_frame(FIXTURE, None).expect("grey_8x8 should decode after round 5 fix");
    assert_eq!(vf.planes.len(), 1, "expected 1 plane (Gray8)");
    let plane = &vf.planes[0];
    assert_eq!(plane.stride, 8);
    assert_eq!(plane.data.len(), 64);
    for (i, &v) in plane.data.iter().enumerate() {
        assert_eq!(v, 128, "pixel {i} should be 128 (constant grey input)");
    }
}
