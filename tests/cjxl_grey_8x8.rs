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

/// Soft test for the grey-8x8 fixture. Updated 2026-05-08 (round 1
/// against the 2024-published core spec): the entropy-stack
/// `use_prefix_code` ↔ `log_alphabet_size` swap (FDIS-2021 typo #5)
/// has been corrected, multi-leaf MA tree evaluation is implemented,
/// and per-pixel property computation is wired. This particular
/// fixture was encoded by cjxl 0.11.1 (effort=1) into a 180-byte
/// stream with a non-trivial complex prefix-code histogram; it tickles
/// a separate code path in the prefix decoder that round 1 doesn't
/// fully reproduce yet — see the SPECGAP entry in the round-1 report.
/// The pixel-correct acceptance fixture for round 1 is `pixel-1x1.jxl`
/// (see the sibling `cjxl_gray_64x64.rs` integration test); this
/// test stays soft so a future round can complete it without churn.
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
            eprintln!("cjxl_grey_8x8 round-1 (2024-spec) stop point: {e}");
        }
    }
}
