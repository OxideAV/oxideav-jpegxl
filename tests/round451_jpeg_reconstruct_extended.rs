//! Round 451 — ISO/IEC 18181-2 Annex A JPEG bitstream reconstruction,
//! extended coverage: greyscale JPEGs, ICC-carrying APP2 chains,
//! MCU-padded dimensions and progressive (SOF2) scans.
//!
//! Every `r451_*.jxl` fixture is a `cjxl --lossless_jpeg=1` transcode of
//! the committed `r451_*.jpg` sibling (locally generated sources; the
//! reference decoder round-trips each pair byte-exactly, so the
//! original JPEG bytes arbitrate the whole chain).

use oxideav_jpegxl::jpeg_reconstruct::reconstruct_jpeg;

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/tests/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

fn assert_byte_exact(jxl_name: &str, jpg_name: &str) {
    let jxl = fixture(jxl_name);
    let jpg = fixture(jpg_name);
    let out = reconstruct_jpeg(&jxl).unwrap_or_else(|e| panic!("{jxl_name}: {e:?}"));
    assert_eq!(
        out.len(),
        jpg.len(),
        "{jxl_name}: reconstructed length differs"
    );
    assert!(out == jpg, "{jxl_name}: reconstruction is not byte-exact");
}

/// 96×80 greyscale gradient, sequential DCT: the jbrd `is_grey` path.
/// The JPEG frame lists a single luma component (JXL channel 1); the
/// codestream still decodes three channels (all-zero chroma).
#[test]
fn greyscale_byte_exact() {
    assert_byte_exact("r451_grey_g.jxl", "r451_grey_g.jpg");
}

/// 64×64 4:4:4 colour gradient, sequential DCT (control specimen for
/// this round's fixture generator).
#[test]
fn sequential_444_byte_exact() {
    assert_byte_exact("r451_seq_g.jxl", "r451_seq_g.jpg");
}

/// 100×60 4:2:0 gradient — image dims not a multiple of the 16×16
/// MCU. The F.2 rule ("size in blocks, divided rounding up by the
/// maximum sampling factor, then multiplied by the channel's factor")
/// pads the luma grid to 14×8 blocks; the padded blocks carry the
/// JPEG MCU dummy blocks' real coefficients.
#[test]
fn mcu_padded_420_byte_exact() {
    assert_byte_exact("r451_odd420_g.jxl", "r451_odd420_g.jpg");
}

/// The same geometry on noisy (plasma) content — denser HF streams
/// across the padded grid.
#[test]
fn mcu_padded_420_noise_byte_exact() {
    assert_byte_exact("r451_odd420.jxl", "r451_odd420.jpg");
}

/// MCU padding + DRI restart markers: per-interval byte alignment and
/// DC prediction resets across a padded 4:2:0 walk.
#[test]
fn mcu_padded_420_restart_byte_exact() {
    assert_byte_exact("r451_seq420_rst.jxl", "r451_seq420_rst.jpg");
}
