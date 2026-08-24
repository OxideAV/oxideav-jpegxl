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

/// Embedded ICC profile (A.9 kind 1): the codestream signals
/// `want_icc`, the Annex B / E.4 encoded ICC stream sits between
/// ImageMetadata and the frame, and reconstruction re-chunks the
/// decoded profile into `ICC_PROFILE` APP2 segments (chunk index /
/// count bytes, `am.length - 17` profile bytes per marker).
#[test]
fn icc_app2_byte_exact() {
    assert_byte_exact("r451_icc_g.jxl", "r451_icc_g.jpg");
}

/// Progressive (SOF2) 4:4:4 colour: multi-scan spectral selection +
/// successive approximation. Exercises DC first + refinement scans
/// and AC first + refinement scans with the shared EOB-run state
/// (10918-1 G.1.2) and buffered correction bits.
#[test]
fn progressive_444_byte_exact() {
    assert_byte_exact("r451_prog_g.jxl", "r451_prog_g.jpg");
}

/// Progressive + 4:2:0 + MCU-padded dims (89×53): interleaved DC
/// scans over the padded MCU grid, per-component AC scans over the
/// TRUE (unpadded) component block grids.
#[test]
fn progressive_420_padded_byte_exact() {
    assert_byte_exact("r451_prog420.jxl", "r451_prog420.jpg");
}

/// Progressive greyscale (72×40): single-component DC + AC scan
/// ladder through the is_grey component mapping.
#[test]
fn progressive_greyscale_byte_exact() {
    assert_byte_exact("r451_proggrey.jxl", "r451_proggrey.jpg");
}

/// Progressive with DRI restart markers (jpegtran -restart 2): the
/// pending EOB run must flush before every RSTn, with byte alignment
/// from the A.6 padding-bit source.
#[test]
fn progressive_restart_byte_exact() {
    assert_byte_exact("r451_prog_rst.jxl", "r451_prog_rst.jpg");
}

/// The round-451 CfL tie-rounding erratum, minimal specimen: a 16×16
/// noisy 4:4:4 transcode whose Cb prediction `tile × y × qY / (cf ×
/// qC)` lands EXACTLY on a half-way value (840/336 = 2.5). The wire
/// rounds ties TOWARD ZERO; rounds 448–450 used `f64::round`
/// (half-away) and reconstructed a silently wrong JPEG (±1 on one
/// chroma coefficient) that still passed every loudness guard.
#[test]
fn cfl_tie_rounding_16x16_byte_exact() {
    assert_byte_exact("r451_noise16.jxl", "r451_noise16.jpg");
}

/// Six tie hits across BOTH chroma channels (X and B) of a 64×64
/// noisy 4:4:4 transcode, positive and negative predictions.
#[test]
fn cfl_tie_rounding_64x64_byte_exact() {
    assert_byte_exact("r451_noise64.jxl", "r451_noise64.jpg");
}

/// The round-451 `CoeffNumNonzeroContext[21]` erratum, direct pin: a
/// 48×48 noisy 4:4:4 transcode whose first luma block walks its
/// remaining-NonZeros count through exactly 21 while the stream's
/// cluster map splits the 152-family coefficient contexts from the
/// 180-family. The in-crate table carried a ninth 152 at index 21
/// (Listing C.13 prints eight: 152 covers 13..=20, 180 covers
/// 21..=32); the misrouted read desynced the shared ANS state and the
/// walk under-delivered its declared NonZeros (the r444/r448 loud
/// class).
#[test]
fn cnnc21_desync_48x48_byte_exact() {
    assert_byte_exact("r451_noise48.jxl", "r451_noise48.jpg");
}

/// The round-451 alias-map exactly-full-bucket erratum, direct pin
/// (2024 IS C.2.6): a bucket holding exactly `bucket_size` mass goes
/// on NEITHER Vose worklist; the FDIS Listing D.1's plain `else`
/// queues it as underfull, and popping it permutes every later
/// (overfull, underfull) pairing — a different alias table whose
/// redirected slices decode wrong symbols. Reachable only through
/// histograms with an exactly-bucket-size entry — the near-uniform
/// non-dyadic shapes of noisy content. This 64×64 specimen previously
/// desynced into a Huffman-table refusal mid-reconstruction.
#[test]
fn alias_exact_bucket_64x64_byte_exact() {
    assert_byte_exact("r451_noise64hist.jxl", "r451_noise64hist.jpg");
}
