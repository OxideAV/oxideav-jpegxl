//! Round 437 — the §J.3-for-kModular SPECGAP (fdis-errata.md Part 9)
//! is RESOLVED by the in-crate bisection the errata catalogue
//! prescribes.
//!
//! Part 9 identifies two freedoms the FDIS leaves open for kModular
//! non-XYB frames — the Annex J sample domain (§9.3: the printed text
//! runs raw integer samples through XYB-calibrated constants, which
//! degenerates the EPF to a near-identity) and the 1-channel Grey
//! plane mapping (§9.2: Annex J is written for exactly three
//! channels) — and prescribes a bisection over the (domain × grey)
//! grid with the black-box reference decode as the residual oracle,
//! explicitly noting a behavioural trace is NOT needed.
//!
//! Round-437 measurement on `grayscale_public_university` (2880×1620,
//! lossy Squeeze, gab=1, epf_iters=3, `epf_sigma_for_modular = 20`):
//!
//! | domain        | C0          | YPlane      | Replicate3     |
//! |---------------|-------------|-------------|----------------|
//! | `Raw8Literal` | 1.0041 / 21 | 1.0009 / 21 | 1.0042 / 21    |
//! | `Normalised`  | 0.4157 / 23 | 1.9995 / 83 | **0.2909 / 8** |
//! | `SigmaScaled` | 0.4157 / 23 | 1.9995 / 83 | 0.2909 / 8     |
//!
//! The winner (`Normalised` + `Replicate3`) is shipped as
//! `ModularEpfPosture::default`; `Normalised/C0` reproduces the
//! previously reported ≈ 0.42 "sigma ≈ ×32" empirical best fit
//! exactly, confirming that fit was the missing domain normalisation.
//! This test CI-gates the arbitration: the shipped default must beat
//! every other grid point on the conformance stream (same pattern as
//! the round-393 §F.2 ramp arbitration).

use oxideav_jpegxl::{
    set_modular_epf_posture_override, ModularEpfDomain, ModularEpfPosture, ModularGreyMapping,
};
use std::io::Cursor;

fn png_grey(bytes: &[u8]) -> (usize, usize, Vec<u8>) {
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    assert_eq!(info.color_type, png::ColorType::Grayscale);
    (info.width as usize, info.height as usize, buf)
}

fn mad_under(posture: Option<ModularEpfPosture>) -> f64 {
    let jxl = include_bytes!("fixtures/conformance_grayscale_public_university.jxl");
    let expected = include_bytes!("fixtures/conformance_grayscale_public_university_expected.png");
    let (w, h, reference) = png_grey(expected);
    set_modular_epf_posture_override(posture);
    let frame = oxideav_jpegxl::decode_one_frame(jxl, None).expect("decode");
    set_modular_epf_posture_override(None);
    let plane = &frame.planes[0];
    let mut sum = 0u64;
    for y in 0..h {
        for x in 0..w {
            sum += plane.data[y * plane.stride + x].abs_diff(reference[y * w + x]) as u64;
        }
    }
    sum as f64 / (w * h) as f64
}

/// The shipped default must be the grid argmin on the conformance
/// stream — every other (domain × grey) point measures strictly worse.
#[test]
fn shipped_posture_is_the_grid_argmin() {
    let best = mad_under(None);
    assert!(
        best < 0.32,
        "shipped posture regressed on grayscale_public_university: MAD {best}"
    );
    for domain in [
        ModularEpfDomain::Raw8Literal,
        ModularEpfDomain::Normalised,
        ModularEpfDomain::SigmaScaled,
    ] {
        for grey in [
            ModularGreyMapping::C0,
            ModularGreyMapping::YPlane,
            ModularGreyMapping::Replicate3,
        ] {
            let p = ModularEpfPosture { domain, grey };
            if p == ModularEpfPosture::default() {
                continue;
            }
            let mad = mad_under(Some(p));
            // SigmaScaled/Replicate3 is weight-identical to the
            // shipped point (sigma-skip never fires at sigma = 20) up
            // to float rounding in `inv_sigma` (the ×255 factor moves
            // a handful of samples by one code); everything else must
            // be strictly worse.
            if domain == ModularEpfDomain::SigmaScaled && grey == ModularGreyMapping::Replicate3 {
                assert!(
                    (mad - best).abs() < 1e-3,
                    "N1/N2 weight-identity broke: {mad} vs {best}"
                );
            } else {
                assert!(
                    mad > best + 0.05,
                    "grid point {p:?} (MAD {mad}) no longer loses to the shipped \
                     default (MAD {best}) — re-run the Part 9 arbitration"
                );
            }
        }
    }
}

/// The default posture is the documented winner.
#[test]
fn default_posture_is_documented_winner() {
    let d = ModularEpfPosture::default();
    assert_eq!(d.domain, ModularEpfDomain::Normalised);
    assert_eq!(d.grey, ModularGreyMapping::Replicate3);
}
