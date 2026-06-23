//! Round 362 — VarDCT `vardct-256x256-d1` decode vs. a black-box
//! reference, with the divergence pinned numerically.
//!
//! Rounds 343..355 built the integrated single-LfGroup VarDCT decode so
//! it runs the whole §C.8.3 → §L.2.2 chain end-to-end and produces a
//! shaped 256×256×3 RGB frame. Round 355's caveat was that "the per-block
//! HF coefficient scaling is not yet validated bit-exact against a
//! reference decode", so the public [`oxideav_jpegxl::decode_one_frame`]
//! path withholds the reconstructed pixels.
//!
//! This round commits the missing **measurement**: a committed reference
//! PNG (decoded once, offline, by the `djxl` black-box validator from the
//! exact same `vardct_256x256_d1.jxl` fixture — its *source* is never
//! consulted, only its output bytes) and a test that quantifies how far
//! our reconstruction is from it, at the RGB output level.
//!
//! ## What the measurement shows (round 362 baseline)
//!
//! * The reference is an ordinary mid-tone photo: per-channel frame-means
//!   R ≈ 127, G ≈ 129, B ≈ 139, with a smooth spread of values.
//! * Our reconstruction **saturates ~99.8 % of samples** to 0 or 255: the
//!   internal XYB magnitudes are several times too large, so the
//!   non-linear XYB→RGB step (Listing L.1) clips almost everything to the
//!   `[0, 1]` rail. Mean absolute per-channel error is ~105–129 / 255.
//!
//! Diagnosis recorded for the next round (derived this round by tracing
//! the intermediate XYB planes against the reference inverted through the
//! default `OpsinInverseMatrix`):
//!
//! * The **Y (luma) plane** is ≈ 4.0× too large in the XYB domain (our
//!   frame-mean Y ≈ 1.83, reference ≈ 0.46). Y carries no
//!   chroma-from-luma term (Annex G: `Y = dY`) and its DC passes the spec
//!   IDCT with unit gain (Annex I.2.1: `in[k] = out[0] + …`), so this is
//!   a clean LF-coefficient *magnitude* error upstream of the IDCT — in
//!   the LfQuant modular decode or the Listing F.1 dequant — independent
//!   of every HF / CfL stage. The Listing F.1 dequant, the Table C.12
//!   Quantizer parse (`global_scale`, `quant_lf`), and the Table C.11
//!   `m_*_lf_unscaled` were each verified spec-conformant this round, so
//!   the ≈4× factor most likely originates in the LfQuant modular
//!   sub-bitstream decode (a missing/duplicated inverse transform — a
//!   power-of-two factor — is consistent with the clean 4×).
//! * The **X (red-green chroma) plane** swings far wider than it should
//!   (reference X ≈ 0; ours spans roughly ±2.6). For this fixture
//!   `x_from_y` is all-zero and `base_correlation_x == 0`, so the Annex G
//!   HF chroma-from-luma multiplier `kX` is exactly 0 — the X swing is
//!   therefore entirely the X HF AC coefficients, mis-scaled in the same
//!   family as the Y error, not a CfL artefact.
//!
//! These facts localise the remaining VarDCT divergence to the
//! coefficient-magnitude path (LfQuant decode + F.1/F.3 dequant), not to
//! the IDCT, placement, crop, or XYB→RGB transform. The next round's fix
//! has a concrete, reproducible target: drive the saturation fraction
//! asserted below toward zero.
//!
//! Clean-room: behaviour is derived from the ISO/IEC 18181 spec PDFs +
//! the staged trace/errata material under `docs/image/jpegxl/`. The
//! reference PNG is the opaque output of the `djxl` validator binary; no
//! external implementation *source* is consulted.

use std::io::Cursor;

const VARDCT_D1_JXL: &[u8] = include_bytes!("fixtures/vardct_256x256_d1.jxl");
const REF_PNG: &[u8] = include_bytes!("fixtures/vardct_256x256_d1_expected.png");

/// Decode the committed reference PNG into interleaved RGB pixels.
fn ref_rgb() -> (u32, u32, Vec<[u8; 3]>) {
    let dec = png::Decoder::new(Cursor::new(REF_PNG));
    let mut reader = dec.read_info().expect("png read_info");
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap_or(0)];
    let info = reader.next_frame(&mut buf).expect("png next_frame");
    assert_eq!(info.bit_depth, png::BitDepth::Eight, "8-bit reference");
    let data = &buf[..info.buffer_size()];
    let ch = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => panic!("unexpected reference colour type {other:?}"),
    };
    let mut px = Vec::with_capacity((info.width * info.height) as usize);
    for c in data.chunks_exact(ch) {
        px.push([c[0], c[1], c[2]]);
    }
    (info.width, info.height, px)
}

/// The `djxl` reference is an ordinary mid-tone photo: each channel's
/// frame-mean sits well inside the 8-bit range and the values are spread,
/// not railed. Pins the qualitative target the VarDCT decode must reach.
#[test]
fn reference_is_a_normal_mid_tone_photo() {
    let (w, h, px) = ref_rgb();
    assert_eq!((w, h), (256, 256), "reference fixture is 256×256");
    let n = px.len() as f64;
    let mut sum = [0u64; 3];
    let mut saturated = 0u64;
    for &p in &px {
        for k in 0..3 {
            sum[k] += p[k] as u64;
        }
        if p.iter().all(|&v| v == 0 || v == 255) {
            saturated += 1;
        }
    }
    let mean = [sum[0] as f64 / n, sum[1] as f64 / n, sum[2] as f64 / n];
    for (k, &m) in mean.iter().enumerate() {
        assert!(
            (80.0..180.0).contains(&m),
            "reference channel {k} frame-mean {m:.1} should be mid-tone (80..180)"
        );
    }
    let sat_frac = saturated as f64 / n;
    assert!(
        sat_frac < 0.02,
        "reference should barely saturate (got {:.1}% fully-railed pixels)",
        sat_frac * 100.0
    );
}

/// Pin the **current** VarDCT reconstruction divergence as a ratchet:
/// the integrated decode currently rails almost every sample to 0/255
/// because the internal XYB magnitude is far too large. When the
/// coefficient-magnitude fix lands, the saturation fraction will collapse
/// and this assertion will (correctly) demand an update — at which point
/// it becomes a true pixel-accuracy gate. A regression that makes the
/// output worse also trips it.
#[test]
fn current_vardct_output_is_oversaturated() {
    let frame = oxideav_jpegxl::decode_vardct_frame_from_codestream(VARDCT_D1_JXL, None)
        .expect("integrated VarDCT reconstruction runs end-to-end on vardct-d1");
    let (_w, _h, refpx) = ref_rgb();
    let r = &frame.planes[0].data;
    let g = &frame.planes[1].data;
    let b = &frame.planes[2].data;
    let n = r.len();
    assert_eq!(n, refpx.len(), "frame and reference are the same size");

    let mut railed = 0u64;
    let mut total_abs_err = [0u64; 3];
    for i in 0..n {
        let ours = [r[i], g[i], b[i]];
        if ours.iter().all(|&v| v == 0 || v == 255) {
            railed += 1;
        }
        for k in 0..3 {
            total_abs_err[k] += ours[k].abs_diff(refpx[i][k]) as u64;
        }
    }
    let railed_frac = railed as f64 / n as f64;

    // Round-362 baseline: ~99.8% of pixels are fully railed. Assert a
    // generous floor so the test is stable, yet still catches the fix
    // (which drops this dramatically). Documented divergence, not a
    // pass — the public `decode_one_frame` correctly withholds pixels.
    assert!(
        railed_frac > 0.80,
        "round-362 baseline expects heavy saturation (got {:.1}% railed). \
         If a coefficient-magnitude fix dropped this, update the ratchet \
         and tighten toward a real pixel-accuracy assertion.",
        railed_frac * 100.0
    );

    // The mean absolute error per channel is large at this baseline; pin
    // it is non-trivial so a future fix's improvement is visible.
    for (k, &e) in total_abs_err.iter().enumerate() {
        let mad = e as f64 / n as f64;
        assert!(
            mad > 20.0,
            "channel {k} MAD {mad:.1} unexpectedly small at the round-362 \
             baseline — did the magnitude fix land? Tighten this ratchet."
        );
    }
}
