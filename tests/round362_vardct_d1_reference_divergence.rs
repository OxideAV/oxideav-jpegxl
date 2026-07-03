//! Round 362 committed the VarDCT `vardct-256x256-d1` reference
//! measurement (a `djxl` black-box reference PNG + a divergence
//! ratchet). At that baseline the reconstruction railed ~99.8 % of
//! samples to 0/255 (per-channel MAD ~105–129/255) because the internal
//! XYB magnitudes were far too large.
//!
//! Round 385 root-caused and fixed the divergence in three pieces:
//!
//! 1. **Corrected Listing C.1 LF-multiplier reading** —
//!    `mXDC = 65536 / (m_x_lf_unscaled × global_scale × quant_lf)`
//!    (`global_scale` is 16.16 fixed-point; the `m_*_lf_unscaled` F16
//!    values are divisors). See `LfMultipliers::compute`.
//! 2. **Annex G CfL split into its Figure 2 branches** — the LF branch
//!    uses the frame-global `x_factor_lf / b_factor_lf` factors on the
//!    dequantised LF planes before the Listing I.16 LLF composition; the
//!    HF branch uses the per-64×64-tile `XFromY / BFromY` factors on the
//!    dequantised HF coefficients before the IDCT. (Previously one
//!    spatial CfL applied the HF factors to everything, crushing B by
//!    ~2× whenever `BFromY` differed from the LF factor.)
//! 3. **Annex G LF-factor bias corrected to `x_factor_lf - 128`** — the
//!    FDIS' `- 127` adds one excess `Y / colour_factor` term on both
//!    X and B at the default bundle (measured independently on both
//!    channels). See `chroma_from_luma::kx_kb_lf`.
//!
//! Two further round-385 fixes tightened this again: the §J restoration
//! filters (Gaborish + per-block-sigma EPF) now run on the integrated
//! path, and the Listing I.16 LLF normalisation was corrected (the LLF
//! block is the plain §I.2.1-normalised forward DCT of the LF block —
//! the literal `× ScaleF` reading left every LLF AC cell off by exactly
//! `ScaleF(8, 64, u)` per axis; see `llf_from_lf`).
//!
//! With all fixes the internal XYB frame-means match the reference's
//! forward-XYB means to ~4 decimal places (X 0.00014 vs 0.00016,
//! Y 0.45788 vs 0.45807, B 0.47213 vs 0.47229) and the sRGB-domain
//! per-channel MAD collapses to ≈ 4.2 / 2.6 / 3.3 with **zero** railed
//! pixels. The residual error is concentrated in the entropy-decoded HF
//! coefficients (the §C.8.3 per-block stream — under active
//! investigation), not the LF/LLF path.
//!
//! This suite is the tightened ratchet at the round-385 baseline.
//!
//! Comparison domain note: this crate's decode output is documented as
//! **linear** RGB (the §L.2.2 NOTE's "the transfer function is linear"
//! reading — see the crate README's plane-layout section), while the
//! `djxl` reference PNG is sRGB-encoded. The comparison below therefore
//! sRGB-encodes our linear output first.
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

/// sRGB-encode one linear 8-bit sample (the crate's documented linear
/// output) for comparison against the sRGB-encoded reference PNG.
fn linear_u8_to_srgb_u8(v: u8) -> u8 {
    let l = v as f64 / 255.0;
    let s = if l <= 0.003_130_8 {
        12.92 * l
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round().clamp(0.0, 255.0) as u8
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

/// Round-385 accuracy ratchet: after the Listing C.1 multiplier fix,
/// the Annex G CfL branch split, the LF-factor `-128` bias fix, the §J
/// filter wiring, and the Listing I.16 LLF-normalisation fix, the
/// integrated VarDCT reconstruction is a close match to the reference —
/// zero railed pixels, per-channel means within ±4/255, per-channel MAD
/// under 6/255 in the sRGB domain (measured ≈ 4.2 / 2.6 / 3.3). The
/// remaining gap is in the entropy-decoded HF coefficients; when that
/// path is fixed, tighten these bounds further.
#[test]
fn vardct_output_tracks_reference_within_hf_filter_gap() {
    let frame = oxideav_jpegxl::decode_vardct_frame_from_codestream(VARDCT_D1_JXL, None)
        .expect("integrated VarDCT reconstruction runs end-to-end on vardct-d1");
    let (_w, _h, refpx) = ref_rgb();
    let n = refpx.len();
    let srgb: Vec<[u8; 3]> = (0..n)
        .map(|i| {
            [
                linear_u8_to_srgb_u8(frame.planes[0].data[i]),
                linear_u8_to_srgb_u8(frame.planes[1].data[i]),
                linear_u8_to_srgb_u8(frame.planes[2].data[i]),
            ]
        })
        .collect();

    let mut railed = 0u64;
    let mut total_abs_err = [0u64; 3];
    let mut our_sum = [0u64; 3];
    let mut ref_sum = [0u64; 3];
    for i in 0..n {
        if srgb[i].iter().all(|&v| v == 0 || v == 255) {
            railed += 1;
        }
        for k in 0..3 {
            total_abs_err[k] += srgb[i][k].abs_diff(refpx[i][k]) as u64;
            our_sum[k] += srgb[i][k] as u64;
            ref_sum[k] += refpx[i][k] as u64;
        }
    }

    let railed_frac = railed as f64 / n as f64;
    assert!(
        railed_frac < 0.01,
        "railed fraction should be ~0 after the round-385 magnitude fixes \
         (got {:.2}%)",
        railed_frac * 100.0
    );

    for k in 0..3 {
        let our_mean = our_sum[k] as f64 / n as f64;
        let ref_mean = ref_sum[k] as f64 / n as f64;
        assert!(
            (our_mean - ref_mean).abs() < 4.0,
            "channel {k} frame-mean {our_mean:.1} should track reference {ref_mean:.1} \
             within ±4 (DC path is validated; a drift here is a MULTIPLIER regression)"
        );
        let mad = total_abs_err[k] as f64 / n as f64;
        assert!(
            mad < 6.0,
            "channel {k} MAD {mad:.2} exceeds the round-385 baseline bound of 6 \
             (measured ≈ 4.2 / 2.6 / 3.3; the residual is the entropy-decoded HF \
             coefficient path). A regression pushed it up — investigate before \
             loosening this ratchet."
        );
    }
}
