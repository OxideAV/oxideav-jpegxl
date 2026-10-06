//! Round 473 — Annex A JPEG reconstruction across the sampling matrix
//! `jbrd` can describe (4:4:4, 4:2:0, 4:2:2, 4:4:0) × baseline /
//! progressive × with / without a DRI restart interval, byte-exact
//! against the original JPEG files.
//!
//! The 4:4:0 lattice (`jpeg_upsampling` value 3, `{1, 2}` factors) is
//! new to the pin set this round; 4:2:0 / 4:2:2 were pinned in rounds
//! 448 / 451. Every `r473_*.jxl` is `cjxl --lossless_jpeg=1` of its
//! `.jpg` sibling (`cjpeg` from one 100×60 source, so the image is
//! MCU-padded on both axes for every subsampled class); the exact
//! command lines and SHA-256 digests are in
//! `tests/fixtures/r473_matrix_NOTES.md`. Two cells of the matrix
//! (4:2:0 and 4:2:2 progressive **with** DRI) are refused by `cjxl`
//! itself (`EncodeImageJXL() failed`) and therefore have no fixture.

use oxideav_jpegxl::jpeg_reconstruct::reconstruct_jpeg;

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/tests/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn assert_byte_exact(stem: &str) {
    let jxl = fixture(&format!("{stem}.jxl"));
    let jpg = fixture(&format!("{stem}.jpg"));
    let out = reconstruct_jpeg(&jxl).unwrap_or_else(|e| panic!("{stem}: {e:?}"));
    assert_eq!(out.len(), jpg.len(), "{stem}: reconstructed length differs");
    assert!(out == jpg, "{stem}: reconstruction is not byte-exact");
}

#[test]
fn matrix_444_byte_exact() {
    for v in ["base_nodri", "base_dri", "prog_nodri", "prog_dri"] {
        assert_byte_exact(&format!("r473_444_{v}"));
    }
}

#[test]
fn matrix_420_byte_exact() {
    for v in ["base_nodri", "base_dri", "prog_nodri"] {
        assert_byte_exact(&format!("r473_420_{v}"));
    }
}

#[test]
fn matrix_422_byte_exact() {
    for v in ["base_nodri", "base_dri", "prog_nodri"] {
        assert_byte_exact(&format!("r473_422_{v}"));
    }
}

#[test]
fn matrix_440_byte_exact() {
    for v in ["base_nodri", "base_dri", "prog_nodri", "prog_dri"] {
        assert_byte_exact(&format!("r473_440_{v}"));
    }
}

/// The 4:4:0 frame header really is the vertical-only lattice
/// (`jpeg_upsampling` = {0, 3, 0}: luma {1, 2}, chroma {1, 1}).
#[test]
fn matrix_440_is_the_vertical_lattice() {
    use oxideav_jpegxl::jpeg_reconstruct::decode_transcoded_coefficients;
    let jxl = fixture("r473_440_base_nodri.jxl");
    let cs = oxideav_jpegxl::extract_codestream(&jxl).unwrap();
    let tc = decode_transcoded_coefficients(&cs[2..]).unwrap();
    assert_eq!(tc.jpeg_upsampling, [0, 3, 0]);
    // 100×60 → MCU 8×16 → padded 104×64 → luma 13×8 blocks, chroma 13×4.
    assert_eq!((tc.bw, tc.bh), (13, 8));
    assert_eq!(tc.cdims, [(13, 4), (13, 8), (13, 4)]);
}
