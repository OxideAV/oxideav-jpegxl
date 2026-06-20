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
//! never did before. The per-block HF coefficient scaling is not yet
//! validated bit-exact against a reference decode, so the *public*
//! [`oxideav_jpegxl::decode_one_frame`] path still withholds the
//! reconstructed pixels (returning a precise "runs end-to-end but
//! pixels not yet validated" `Error::Unsupported`) rather than risk a
//! silent misparse. These tests therefore pin the structural
//! invariants of the integrated pipeline, not pixel values.
//!
//! Clean-room: behaviour is derived from the ISO/IEC 18181 spec PDFs +
//! the staged trace/errata material under `docs/image/jpegxl/`. No
//! external implementation source is consulted.

use oxideav_core::Error;
use oxideav_jpegxl::decode_one_frame;

const VARDCT_D1_JXL: &[u8] = include_bytes!("fixtures/vardct_256x256_d1.jxl");

/// The public decode path now reaches the integrated reconstruction on
/// the `vardct-256x256-d1` fixture: the error is the round-355
/// "runs end-to-end" sentinel, proving the §C.8.3 → §L.2.2 chain
/// executed without aborting (rather than the older "remaining: …
/// per-pass header reads" deferral, or a sub-component parse error).
#[test]
fn vardct_d1_reaches_integrated_reconstruction() {
    let err = decode_one_frame(VARDCT_D1_JXL, None)
        .expect_err("public path withholds unvalidated VarDCT pixels");
    let msg = format!("{err}");
    assert!(
        matches!(err, Error::Unsupported(_)),
        "expected Unsupported sentinel; got {msg}"
    );
    assert!(
        msg.contains("runs end-to-end"),
        "expected the round-355 end-to-end sentinel (the integrated HF decode + IDCT + CfL + \
         crop + XYB→RGB chain ran to completion); got: {msg}"
    );
}

/// Driving the integrated decoder via
/// [`oxideav_jpegxl::decode_vardct_frame_from_codestream`] (the test/tool
/// entry that returns the reconstruction's pixels instead of the public
/// withhold sentinel) produces a correctly-*shaped* 3-plane RGB frame at
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
