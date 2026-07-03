//! Round 355 — integrated single-LfGroup VarDCT decode reaches pixels.
//!
//! Earlier rounds drove a VarDCT codestream as far as the parsed §C.7
//! HfGlobal section + the Listing F.1 / F.2 LF dequant, then stopped
//! with a precise "remaining: §C.8.3 per-pass header reads + qdc_at LF
//! lookup + BlockContextResolver history" deferral. Round 355 wires that
//! remaining chain together: the §C.8.3 per-pass HF header, the
//! histogram-backed HF-coefficient entropy decode
//! (`reconstruct_lf_group_from_histogram`), the F.3 dequant + §I.2.4 LLF
//! merge + §I.2.3.2 IDCT + Annex G chroma-from-luma, the §6.2 crop, and
//! the §L.2.2 inverse-XYB → 8-bit RGB conversion — all on a real
//! codestream.
//!
//! The whole chain now *executes* end-to-end (the entry point
//! [`oxideav_jpegxl::decode_vardct_frame`] returns a 3-plane RGB
//! [`oxideav_core::VideoFrame`] at the logical frame extent), which it
//! never did before. Round 389 validated the output against the staged
//! reference decodes (see `round389_multi_group_vardct.rs` and the
//! round-362 ratchet) and lifted the public-path pixel withhold; these
//! tests pin the structural invariants of the integrated pipeline.
//!
//! Clean-room: behaviour is derived from the ISO/IEC 18181 spec PDFs +
//! the staged trace/errata material under `docs/image/jpegxl/`. No
//! external implementation source is consulted.

use oxideav_jpegxl::decode_one_frame;

const VARDCT_D1_JXL: &[u8] = include_bytes!("fixtures/vardct_256x256_d1.jxl");

/// The public decode path returns the integrated reconstruction's
/// pixels on the `vardct-256x256-d1` fixture (round 389 lifted the
/// rounds-355–385 withhold sentinel once the output was
/// reference-validated), and is byte-identical to the historical
/// tests/tooling entry `decode_vardct_frame_from_codestream`.
#[test]
fn vardct_d1_reaches_integrated_reconstruction() {
    let public =
        decode_one_frame(VARDCT_D1_JXL, None).expect("public VarDCT decode succeeds (round 389)");
    let alias = oxideav_jpegxl::decode_vardct_frame_from_codestream(VARDCT_D1_JXL, None)
        .expect("historical alias decodes");
    assert_eq!(public.planes.len(), 3);
    for (c, (p, a)) in public.planes.iter().zip(alias.planes.iter()).enumerate() {
        assert_eq!(p.data, a.data, "channel {c} public/alias byte-identical");
    }
}

/// Driving the integrated decoder via
/// [`oxideav_jpegxl::decode_vardct_frame_from_codestream`] produces a
/// correctly-*shaped* 3-plane RGB frame at
/// the 256×256 logical extent. This pins the pipeline's structural
/// invariants — three planes, each `256 × 256` bytes, stride 256 — with
/// the whole §C.8.3 → §L.2.2 chain having run to completion. (Pixel
/// values are deliberately NOT asserted: per-block HF scaling is not yet
/// reference-validated.)
#[test]
fn vardct_d1_integrated_frame_is_correctly_shaped() {
    let frame = oxideav_jpegxl::decode_vardct_frame_from_codestream(VARDCT_D1_JXL, None)
        .expect("integrated VarDCT reconstruction should run end-to-end on vardct-d1");
    assert_eq!(frame.planes.len(), 3, "RGB frame has three planes");
    for (ci, plane) in frame.planes.iter().enumerate() {
        assert_eq!(plane.stride, 256, "plane {ci} stride");
        assert_eq!(
            plane.data.len(),
            256 * 256,
            "plane {ci} has 256×256 byte samples"
        );
    }
}

/// Non-degeneracy: the reconstructed frame must carry real image content,
/// not a constant colour. This pins the LF channel-order reindex
/// (modular `(Y, X, B)` → Listing-F.1 `[X, Y, B]`): before that fix the
/// luma (Y) plane was dequantised with the X multiplier and fed to
/// `inverse_xyb_to_rgb` as chroma, collapsing every pixel of the frame
/// to a single constant colour (R=255, G=0, B=0). We don't assert exact
/// pixel values (HF scaling is not reference-validated), but a real photo
/// fixture must produce many distinct sample values spanning a wide range
/// in every plane.
#[test]
fn vardct_d1_reconstruction_is_not_a_constant_colour() {
    let frame = oxideav_jpegxl::decode_vardct_frame_from_codestream(VARDCT_D1_JXL, None)
        .expect("integrated VarDCT reconstruction should run end-to-end on vardct-d1");
    for (ci, plane) in frame.planes.iter().enumerate() {
        let mut seen = [false; 256];
        for &b in &plane.data {
            seen[b as usize] = true;
        }
        let distinct = seen.iter().filter(|&&s| s).count();
        let min = *plane.data.iter().min().unwrap();
        let max = *plane.data.iter().max().unwrap();
        assert!(
            distinct > 16,
            "plane {ci} is near-constant ({distinct} distinct values) — the LF channel-order \
             reindex regressed (frame collapsed toward a constant colour)"
        );
        assert!(
            (max - min) > 64,
            "plane {ci} sample range {min}..={max} is too narrow for the photo fixture — \
             likely an XYB channel-mapping regression"
        );
    }
}
