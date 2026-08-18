//! Round 448 — ISO/IEC 18181-2 Annex A JPEG bitstream reconstruction,
//! pinned BYTE-EXACT against the original JPEG files.
//!
//! Every `r448_*.jxl` fixture is a `cjxl --lossless_jpeg=1` transcode of
//! the committed `r448_*.jpg` sibling (locally generated sources; the
//! reference decoder round-trips each pair byte-exactly, so the original
//! JPEG bytes are the arbitration oracle for the whole chain: container
//! walk → jbrd bundle + Brotli tail → coefficient-level codestream
//! decode (RAW quant matrices, §C.8.3 entropy, integer chroma-from-luma
//! inversion) → 10918-1 entropy re-encode).
//!
//! The wire-arbitrated findings these pins hold down:
//!
//! * the 2021-FDIS vs 2024-IS `prev` divergence in the first HF
//!   coefficient read of a block (Listing C.14 vs I.4) — the 2024
//!   reading (`non_zeros > size/16 → prev = 0`) is on the wire;
//! * RAW dequantization matrices carry the JPEG quant tables with NO
//!   ZeroPadToByte between the F16 denominator and the modular
//!   sub-bitstream (the 2021 FDIS prints one; the 2024 listing and the
//!   wire agree there is none);
//! * the JXL cell layout (coefficients AND RAW quant tables) is the
//!   transpose of the JPEG raster;
//! * stored chroma is CfL-decorrelated and re-correlates in the integer
//!   domain as `c += round(k × y × qY[pos] / qC[pos])` with
//!   `k = base_correlation + tile / colour_factor` (no Listing F.2
//!   bias, no 0.8^(qm_scale−2) factor);
//! * entropy segments pad to byte boundaries with ONE bits when the
//!   jbrd signals `has_padding = false` (A.6 prints "zero").

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

/// 16×16 hard-edge content, natural coefficient orders
/// (`used_orders == 0`), single-table DHT/DQT segments, JFIF APP0.
#[test]
fn edge_16x16_byte_exact() {
    assert_byte_exact("r448_edge444.jxl", "r448_edge444.jpg");
}

/// 64×64 smooth gradient, custom coefficient orders
/// (`used_orders == 1`) — the fixture that arbitrated the Listing
/// C.14 / I.4 `prev` edition divergence and the integer CfL inversion.
#[test]
fn gradient_64x64_byte_exact() {
    assert_byte_exact("r448_grad444.jxl", "r448_grad444.jpg");
}

/// COM marker (A.10): the comment bytes travel verbatim in the jbrd
/// Brotli tail.
#[test]
fn com_marker_byte_exact() {
    assert_byte_exact("r448_com444.jxl", "r448_com444.jpg");
}

/// DRI + restart markers (A.4 / A.8): RSTm cycling, per-interval DC
/// prediction resets, and byte alignment before each marker.
#[test]
fn restart_interval_byte_exact() {
    assert_byte_exact("r448_rst444.jxl", "r448_rst444.jpg");
}

/// 16×16 hard edges at 4:2:0: the I.4 subsampled varblock walk
/// (chroma decoded only at even lattice positions, NonZeros
/// bookkeeping on the half-resolution grids) and the F.2 channel-dim
/// scaling — chroma planes halve, luma stays full.
#[test]
fn edge_16x16_420_byte_exact() {
    assert_byte_exact("r448_edge420.jxl", "r448_edge420.jpg");
}

/// 64×64 gradient at 4:2:0 — chroma-from-luma is skipped entirely for
/// subsampled frames (I.6 first sentence).
#[test]
fn gradient_64x64_420_byte_exact() {
    assert_byte_exact("r448_grad420.jxl", "r448_grad420.jpg");
}

/// 64×64 gradient at 4:2:2 — asymmetric per-axis lattices
/// (`jpeg_upsampling` value 2, {2,1} factors).
#[test]
fn gradient_64x64_422_byte_exact() {
    assert_byte_exact("r448_grad422.jxl", "r448_grad422.jpg");
}

/// 512×320 4:2:0 — a multi-group frame (2×2 PassGroup sections):
/// per-group section slicing, group-local NonZeros resets on the
/// subsampled lattices, and per-group entropy-stream lifecycle.
#[test]
fn big_512x320_420_byte_exact() {
    assert_byte_exact("r448_big420.jxl", "r448_big420.jpg");
}

/// The staged docs `jpeg-transcode` fixture (256×256 noisy PHOTO at
/// 4:2:0, quality 85, committed round 448 as
/// `jpeg_transcode.jxl` / `jpeg_transcode_original.jpg`): the
/// real-content flagship — container walk, jbrd, RAW tables,
/// subsampled §C.8.3 decode and the 10918-1 re-encode all in one pin.
#[test]
fn docs_photo_transcode_byte_exact() {
    assert_byte_exact("jpeg_transcode.jxl", "jpeg_transcode_original.jpg");
}

/// Noisy 4:4:4 content still hits the round-444 open §C.7.2 desync
/// class (near-uniform non-dyadic histograms): the walk
/// under-delivers declared NonZeros. Reconstruction must refuse
/// LOUDLY — a silently wrong JPEG is never acceptable output. This
/// pin flips to a byte-exact assertion when the desync class closes.
#[test]
fn noisy_photo_refuses_loudly() {
    let jxl = fixture("r448_noise444.jxl");
    let err = reconstruct_jpeg(&jxl).expect_err("desynced stream must not reconstruct");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("undelivered NonZeros") || msg.contains("desync"),
        "unexpected refusal shape: {msg}"
    );
}
