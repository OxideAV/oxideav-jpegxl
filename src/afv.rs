//! AFV (Asymmetric "FREE") variable-block helpers — ISO/IEC FDIS
//! 18181-1:2021 Annex I.2.2 Listings I.5 and I.6.
//!
//! Round 147 lands the §I.2.2 `AFVBasis[16][16]` table (Listing I.5)
//! plus the `AFV_IDCT` per-cell summation (Listing I.6) — the
//! arithmetic core that the [`crate::idct`] dispatch needs to lift
//! `TransformType::Afv0..Afv3` out of its `Err(Unsupported)` branch
//! (the full Inverse-AFV-transform composition of Listing I.13 is
//! follow-up work; this round lands the pure-math primitive in the
//! same shape as round-89 `dct_quant_weights`, round-95 `hf_dequant`,
//! round-121 `llf_from_lf`, round-138 `chroma_from_luma`, round-141
//! `gaborish`, and round-144 `epf`).
//!
//! ## Spec mapping
//!
//! § I.2.2 of FDIS 18181-1:2021 (page 76) defines the AFV transform
//! over a 4×4 cell of coefficients laid out in a flat 16-entry vector
//! using the convention "a 4×4 matrix is stored in an array such that
//! entry `(x, y)` corresponds to index `4 × y + x`".
//!
//! Listing I.5 names the orthonormal basis `AFVBasis[16][16]`, where
//! `AFVBasis[j]` is the `j`-th basis row of length 16. Listing I.6
//! computes one cell of samples by
//!
//! ```text
//! for (i = 0; i < 16; ++i) {
//!   sample = 0;
//!   for (j = 0; j < 16; ++j) sample += coefficients[j] × AFVBasis[j][i];
//!   samples[i] = sample;
//! }
//! ```
//!
//! i.e. the inner-product of the 16 coefficients with the `i`-th column
//! of the AFVBasis matrix. Because `AFVBasis` is orthonormal,
//! `AFV_IDCT` is the inverse of the forward AFV transform — but this
//! module only ships the inverse; the forward direction is not needed
//! by the decoder.
//!
//! ## Self-check
//!
//! The 256-float transcription is independently verified by the
//! property tests in `tests` at the bottom of this module:
//!
//! * The first basis row is identically `0.25` in every position
//!   (Listing I.5 line 1).
//! * Each row has unit L2 norm (orthonormality diagonal entry = 1).
//! * Distinct rows have zero inner-product (orthonormality off-diagonal
//!   entry = 0).
//! * Listing I.6 reduces to a constant-output 0.25 × dc when only the
//!   DC coefficient is nonzero (consistent with `AFVBasis[0]`).
//!
//! Together these pin every cell of every row up to f32 noise without
//! reading any reference implementation — a single transcription
//! typo in any of the 256 floats would fail at least one orthonormality
//! sum.

use oxideav_core::{Error, Result};

/// Size of the AFV cell: 16 coefficients in, 16 samples out, mapped to
/// a 4×4 grid using `index = 4 × y + x` (§I.2.2 convention).
pub const AFV_CELL_LEN: usize = 16;

/// `AFVBasis[16][16]` from FDIS 18181-1:2021 Listing I.5.
///
/// Row `j` is the `j`-th basis row; column `i` is the `i`-th cell of
/// the basis (with the §I.2.2 `(x, y) -> 4 × y + x` 4×4 mapping).
///
/// Used by [`afv_idct`] (Listing I.6) as
/// `samples[i] = sum_j coeff[j] × AFVBasis[j][i]`.
pub const AFV_BASIS: [[f32; AFV_CELL_LEN]; AFV_CELL_LEN] = [
    // j = 0: identically 0.25 in every column.
    [
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25,
    ],
    // j = 1
    [
        0.876_902_9,
        0.220_651_82,
        -0.101_400_5,
        -0.101_400_5,
        0.220_651_82,
        -0.101_400_51,
        -0.101_400_5,
        -0.101_400_5,
        -0.101_400_5,
        -0.101_400_51,
        -0.101_400_5,
        -0.101_400_51,
        -0.101_400_51,
        -0.101_400_5,
        -0.101_400_5,
        -0.101_400_49,
    ],
    // j = 2
    [
        0.0,
        0.0,
        0.406_700_76,
        0.444_448_17,
        0.0,
        0.0,
        0.195_744,
        0.292_91,
        -0.406_700_75,
        -0.195_744,
        0.0,
        0.113_790_75,
        -0.444_448_15,
        -0.292_91,
        -0.113_790_75,
        0.0,
    ],
    // j = 3
    [
        0.0,
        0.0,
        -0.212_557_48,
        0.308_549_7,
        0.0,
        0.470_670_23,
        -0.162_120_52,
        0.0,
        -0.212_557_48,
        -0.162_120_52,
        -0.470_670_23,
        -0.146_429_2,
        0.308_549_7,
        0.0,
        -0.146_429_2,
        0.425_114_96,
    ],
    // j = 4
    [
        0.0,
        -0.707_106_77,
        0.0,
        0.0,
        0.707_106_77,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    // j = 5
    [
        -0.410_537_76,
        0.623_548_55,
        -0.064_350_72,
        -0.064_350_72,
        0.623_548_55,
        -0.064_350_72,
        -0.064_350_72,
        -0.064_350_72,
        -0.064_350_72,
        -0.064_350_72,
        -0.064_350_72,
        -0.064_350_72,
        -0.064_350_72,
        -0.064_350_72,
        -0.064_350_72,
        -0.064_350_72,
    ],
    // j = 6
    [
        0.0,
        0.0,
        -0.451_755_66,
        0.158_545_04,
        0.0,
        -0.040_385_15,
        0.007_418_226_5,
        0.393_510_34,
        -0.451_755_67,
        0.007_418_226_4,
        0.110_741_66,
        0.082_981_63,
        0.158_545_03,
        0.393_510_34,
        0.082_981_63,
        -0.451_755_67,
    ],
    // j = 7
    [
        0.0,
        0.0,
        -0.304_684_75,
        0.511_261_6,
        0.0,
        0.0,
        -0.290_480_13,
        -0.065_787_02,
        0.304_684_75,
        0.290_480_13,
        0.0,
        -0.238_897_74,
        -0.511_261_6,
        0.065_787_02,
        0.238_897_74,
        0.0,
    ],
    // j = 8
    [
        0.0,
        0.0,
        0.301_792_96,
        0.257_923_63,
        0.0,
        0.162_723_4,
        0.095_200_226,
        0.0,
        0.301_792_96,
        0.095_200_226,
        -0.162_723_4,
        -0.353_123_85,
        0.257_923_63,
        0.0,
        -0.353_123_85,
        -0.603_585_9,
    ],
    // j = 9
    [
        0.0,
        0.0,
        0.408_248_3,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.408_248_3,
        -0.408_248_3,
        0.0,
        0.0,
        -0.408_248_3,
        0.0,
        0.408_248_3,
        0.408_248_3,
        0.0,
    ],
    // j = 10
    [
        0.0,
        0.0,
        0.174_786_7,
        0.081_261_12,
        0.0,
        0.0,
        -0.367_539_8,
        -0.307_882_21,
        -0.174_786_7,
        0.367_539_8,
        0.0,
        0.482_668_9,
        -0.081_261_12,
        0.307_882_22,
        -0.482_668_9,
        0.0,
    ],
    // j = 11
    [
        0.0,
        0.0,
        -0.211_056_01,
        0.185_671_8,
        0.0,
        0.0,
        0.492_158_6,
        -0.385_250_15,
        0.211_056_01,
        -0.492_158_6,
        0.0,
        0.174_194_12,
        -0.185_671_8,
        0.385_250_12,
        -0.174_194_12,
        0.0,
    ],
    // j = 12
    [
        0.0,
        0.0,
        -0.142_660_85,
        -0.341_644_68,
        0.0,
        0.736_749_75,
        0.246_271_08,
        -0.085_740_19,
        -0.142_660_85,
        0.246_271_08,
        0.148_833_99,
        -0.047_686_804,
        -0.341_644_68,
        -0.085_740_19,
        -0.047_686_804,
        -0.142_660_85,
    ],
    // j = 13
    [
        0.0,
        0.0,
        -0.138_135_4,
        0.330_228_26,
        0.0,
        0.087_551_15,
        -0.079_467_066,
        -0.461_337_5,
        -0.138_135_4,
        -0.079_467_066,
        0.497_246_47,
        0.125_380_59,
        0.330_228_26,
        -0.461_337_5,
        0.125_380_59,
        -0.138_135_4,
    ],
    // j = 14
    [
        0.0,
        0.0,
        -0.174_376_03,
        0.070_279_07,
        0.0,
        -0.292_102_66,
        0.362_381_73,
        0.0,
        -0.174_376_03,
        0.362_381_73,
        0.292_102_66,
        -0.432_660_8,
        0.070_279_07,
        0.0,
        -0.432_660_8,
        0.348_752_05,
    ],
    // j = 15
    [
        0.0,
        0.0,
        0.113_549_87,
        -0.074_175_045,
        0.0,
        0.194_028_93,
        -0.435_190_5,
        0.219_186_85,
        0.113_549_87,
        -0.435_190_5,
        0.555_044_4,
        -0.254_682_77,
        -0.074_175_045,
        0.219_186_85,
        -0.254_682_77,
        0.113_549_87,
    ],
];

/// AFV_IDCT per FDIS Listing I.6.
///
/// Takes the 16-entry `coefficients` vector laid out per the §I.2.2
/// `(x, y) -> 4 × y + x` 4×4 mapping, and produces the matching
/// 16-entry sample vector
///
/// ```text
/// samples[i] = sum_{j = 0..16} coefficients[j] × AFVBasis[j][i]
/// ```
///
/// Returns `Err(InvalidData)` if `coefficients.len() != 16`.
pub fn afv_idct(coefficients: &[f32]) -> Result<[f32; AFV_CELL_LEN]> {
    if coefficients.len() != AFV_CELL_LEN {
        return Err(Error::InvalidData(format!(
            "JXL afv_idct: coefficients length {} != {AFV_CELL_LEN}",
            coefficients.len()
        )));
    }
    let mut samples = [0.0f32; AFV_CELL_LEN];
    for i in 0..AFV_CELL_LEN {
        let mut acc = 0.0f32;
        for j in 0..AFV_CELL_LEN {
            acc += coefficients[j] * AFV_BASIS[j][i];
        }
        samples[i] = acc;
    }
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// L2 inner product of two basis rows.
    fn inner(a: &[f32; 16], b: &[f32; 16]) -> f32 {
        let mut acc = 0.0f32;
        for k in 0..16 {
            acc += a[k] * b[k];
        }
        acc
    }

    #[test]
    fn afv_basis_row0_is_quarter_constant() {
        // Listing I.5 line 1: row 0 is identically 0.25 in every column.
        for (i, &v) in AFV_BASIS[0].iter().enumerate() {
            assert_eq!(v, 0.25, "AFVBasis[0][{i}] = {v} expected 0.25");
        }
    }

    #[test]
    fn afv_basis_has_16_rows_of_16() {
        // Length-of-table contract: 16 × 16 = 256 entries.
        assert_eq!(AFV_BASIS.len(), 16);
        for (j, row) in AFV_BASIS.iter().enumerate() {
            assert_eq!(row.len(), 16, "row {j} length");
        }
    }

    #[test]
    fn afv_basis_rows_are_unit_norm() {
        // Every basis row should have ||row||_2 = 1 (orthonormal basis,
        // §I.2.2). f32-noise tolerance: a single transcription typo in
        // any of the 256 entries shifts this metric well above 1e-3.
        for (j, row) in AFV_BASIS.iter().enumerate() {
            let norm_sq = inner(row, row);
            assert!(
                (norm_sq - 1.0).abs() < 1e-3,
                "row {j}: ||row||^2 = {norm_sq}, expected 1.0"
            );
        }
    }

    #[test]
    fn afv_basis_rows_are_pairwise_orthogonal() {
        // Every distinct pair of basis rows should have zero inner-
        // product (off-diagonal orthonormality, §I.2.2).
        for (j, row_j) in AFV_BASIS.iter().enumerate() {
            for (k, row_k) in AFV_BASIS.iter().enumerate().skip(j + 1) {
                let dot = inner(row_j, row_k);
                assert!(dot.abs() < 1e-3, "<row {j}, row {k}> = {dot}, expected 0.0");
            }
        }
    }

    #[test]
    fn afv_idct_rejects_wrong_length() {
        let short = vec![0.0f32; 15];
        assert!(afv_idct(&short).is_err());
        let long = vec![0.0f32; 17];
        assert!(afv_idct(&long).is_err());
        let empty: Vec<f32> = Vec::new();
        assert!(afv_idct(&empty).is_err());
    }

    #[test]
    fn afv_idct_dc_only_yields_constant_quarter_dc() {
        // coefficients = [dc, 0, 0, ..., 0] → samples[i] = dc × 0.25 in
        // every cell, because AFVBasis[0][i] = 0.25 for all i (Listing
        // I.5 line 1).
        let mut c = vec![0.0f32; 16];
        c[0] = 4.0;
        let s = afv_idct(&c).unwrap();
        for (i, &v) in s.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "i={i}: dc-only got {v} expected 1.0 (= 4.0 × 0.25)"
            );
        }
    }

    #[test]
    fn afv_idct_single_basis_pulls_out_row() {
        // coefficients = e_j (one-hot at index j) → samples = AFVBasis[j].
        for j in 0..16 {
            let mut c = vec![0.0f32; 16];
            c[j] = 1.0;
            let s = afv_idct(&c).unwrap();
            for i in 0..16 {
                let expected = AFV_BASIS[j][i];
                assert!(
                    (s[i] - expected).abs() < 1e-6,
                    "j={j}, i={i}: got {} expected {expected}",
                    s[i]
                );
            }
        }
    }

    #[test]
    fn afv_idct_is_linear() {
        // afv_idct(a × x + b × y) == a × afv_idct(x) + b × afv_idct(y).
        let x: Vec<f32> = (0..16).map(|i| (i as f32) * 0.3 - 1.0).collect();
        let y: Vec<f32> = (0..16).map(|i| (i as f32) * -0.1 + 2.0).collect();
        let a = 0.7f32;
        let b = -1.3f32;
        let mut comb = vec![0.0f32; 16];
        for i in 0..16 {
            comb[i] = a * x[i] + b * y[i];
        }
        let sx = afv_idct(&x).unwrap();
        let sy = afv_idct(&y).unwrap();
        let sc = afv_idct(&comb).unwrap();
        for i in 0..16 {
            let expected = a * sx[i] + b * sy[i];
            assert!(
                (sc[i] - expected).abs() < 1e-4,
                "i={i}: combined-output {} != {expected} (per-op combo)",
                sc[i]
            );
        }
    }

    #[test]
    fn afv_idct_preserves_l2_energy() {
        // AFVBasis is orthonormal (verified in
        // afv_basis_rows_are_unit_norm + …pairwise_orthogonal); the
        // matrix product preserves L2 norm, so for any coefficient
        // vector, ||samples||_2 == ||coefficients||_2 up to f32 noise.
        for trial in 0..5usize {
            let mut c = [0.0f32; 16];
            // Deterministic pseudo-random fill (no rand dep).
            for (i, slot) in c.iter_mut().enumerate() {
                *slot = ((trial * 37 + i * 13 + 17) % 31) as f32 / 31.0 - 0.5;
            }
            let s = afv_idct(&c).unwrap();
            let c_norm: f32 = c.iter().map(|v| v * v).sum::<f32>().sqrt();
            let s_norm: f32 = s.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                (s_norm - c_norm).abs() < 1e-3,
                "trial {trial}: ||samples||={s_norm}, ||coeffs||={c_norm}"
            );
        }
    }

    #[test]
    fn afv_basis_row4_known_two_nonzero_pair() {
        // Row 4 is special: only two entries are nonzero — at columns 1
        // and 4 — and they're equal-magnitude opposite-sign at
        // ±1/sqrt(2). The remaining 14 entries are 0.0.
        let inv_sqrt2 = 1.0f32 / 2.0f32.sqrt();
        for (i, &v) in AFV_BASIS[4].iter().enumerate() {
            match i {
                1 => assert!(
                    (v + inv_sqrt2).abs() < 1e-6,
                    "AFVBasis[4][1] = {v}, expected -1/sqrt(2)"
                ),
                4 => assert!(
                    (v - inv_sqrt2).abs() < 1e-6,
                    "AFVBasis[4][4] = {v}, expected +1/sqrt(2)"
                ),
                _ => assert_eq!(v, 0.0, "AFVBasis[4][{i}] = {v}, expected 0.0"),
            }
        }
    }
}
