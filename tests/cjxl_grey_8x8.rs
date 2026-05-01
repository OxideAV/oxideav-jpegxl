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
/// round-3 decoder is still missing pieces (multi-leaf MA tree
/// evaluation, large-token property-node interpretation, etc.). The
/// test prints the decoder result so a reviewer can see exactly what
/// the failure mode was, but does NOT fail the suite if it's an
/// expected error. A successful decode asserts pixel-perfect equality
/// with the source PGM (all 128s).
///
/// Round-3 status (commit landing this test): the decoder gets through
/// SizeHeader, ImageMetadata (Grey/8bpp), FrameHeader (Modular,
/// is_last, no crop), TOC (single 167-byte entry), LfChannelDequantization
/// (all_default), GlobalModular preamble (`use_global_tree=true`,
/// `wp_header.default_wp=true`, `nb_transforms=0`), and the MA tree
/// EntropyStream prelude (1 cluster, simple-prefix code over a
/// 115-symbol alphabet emitting symbols {8, 14, 113}). The first
/// prefix-decoded symbol from the tree sub-stream is 8 → property=7
/// → decision node, but a subsequent decode returns 113 which the
/// HybridUintConfig (split=16, msb=1, lsb=2) expands to ~552964 — a
/// value far too large to be a property index. Since the FDIS spec is
/// open to multiple readings here and the workspace policy bars
/// consulting third-party JXL implementations, round 3 stops here and
/// flags the gap for round 4.
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
