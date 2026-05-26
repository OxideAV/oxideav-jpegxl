//! Integration tests for the round-147 `afv` pure-math primitive
//! (FDIS 18181-1:2021 Annex I.2.2 Listings I.5 + I.6).
//!
//! Mirrors the round-138 / round-141 / round-144 in-tree test layout:
//! the per-module unit tests under `src/afv.rs` cover the orthonormality
//! and linearity properties of the basis directly; this file exercises
//! the same primitive through its public API as a downstream consumer
//! would.

use oxideav_jpegxl::afv::{afv_idct, AFV_BASIS, AFV_CELL_LEN};

#[test]
fn public_api_cell_len_is_16() {
    assert_eq!(AFV_CELL_LEN, 16);
}

#[test]
fn public_api_basis_table_is_16_by_16() {
    assert_eq!(AFV_BASIS.len(), 16);
    for (j, row) in AFV_BASIS.iter().enumerate() {
        assert_eq!(row.len(), 16, "row {j} length");
    }
}

#[test]
fn public_api_dc_only_input_returns_quarter_constant() {
    // §I.2.2 Listing I.5 line 1: AFVBasis[0] = {0.25, ..., 0.25};
    // §I.2.2 Listing I.6: samples[i] = sum_j coeff[j] × AFVBasis[j][i].
    // For coeff = [c, 0, ..., 0], samples[i] = c × 0.25 in every cell.
    let mut c = vec![0.0f32; 16];
    c[0] = 8.0;
    let s = afv_idct(&c).expect("dc-only AFV_IDCT");
    for (i, &v) in s.iter().enumerate() {
        assert!(
            (v - 2.0).abs() < 1e-5,
            "i={i}: dc-only AFV got {v} expected 2.0 (= 8.0 × 0.25)"
        );
    }
}

#[test]
fn public_api_zero_input_zero_output() {
    let c = vec![0.0f32; 16];
    let s = afv_idct(&c).expect("zero AFV_IDCT");
    for (i, &v) in s.iter().enumerate() {
        assert_eq!(v, 0.0, "i={i}: zero input produced {v}");
    }
}

#[test]
fn public_api_each_basis_row_recoverable_via_one_hot() {
    // Pull each basis row out individually by one-hot coefficient.
    // This is the "orthonormality with itself" cross-check at the
    // public-API level.
    for j in 0..16 {
        let mut c = vec![0.0f32; 16];
        c[j] = 1.0;
        let s = afv_idct(&c).expect("one-hot AFV_IDCT");
        for i in 0..16 {
            let expected = AFV_BASIS[j][i];
            assert!(
                (s[i] - expected).abs() < 1e-6,
                "j={j}, i={i}: one-hot output {} != AFV_BASIS[{j}][{i}] = {expected}",
                s[i]
            );
        }
    }
}

#[test]
fn public_api_rejects_wrong_length() {
    assert!(afv_idct(&[0.0f32; 0]).is_err());
    assert!(afv_idct(&[0.0f32; 15]).is_err());
    assert!(afv_idct(&[0.0f32; 17]).is_err());
    assert!(afv_idct(&[0.0f32; 64]).is_err());
}

#[test]
fn public_api_inner_product_diagonal_is_one() {
    // <AFV_BASIS[j], AFV_BASIS[j]> = 1 for every j (orthonormality
    // diagonal). The §I.2.2 invariant is checked again at the public
    // API to pin the table's published contract.
    for (j, row) in AFV_BASIS.iter().enumerate() {
        let n: f32 = row.iter().map(|v| v * v).sum();
        assert!(
            (n - 1.0).abs() < 1e-3,
            "row {j}: ||AFV_BASIS[{j}]||^2 = {n}, expected 1.0"
        );
    }
}

#[test]
fn public_api_inner_product_off_diagonal_is_zero() {
    // <AFV_BASIS[j], AFV_BASIS[k]> = 0 for j != k (orthonormality
    // off-diagonal).
    for (j, row_j) in AFV_BASIS.iter().enumerate() {
        for (k, row_k) in AFV_BASIS.iter().enumerate().skip(j + 1) {
            let dot: f32 = row_j.iter().zip(row_k.iter()).map(|(a, b)| a * b).sum();
            assert!(
                dot.abs() < 1e-3,
                "<AFV_BASIS[{j}], AFV_BASIS[{k}]> = {dot}, expected 0.0"
            );
        }
    }
}

#[test]
fn public_api_energy_conserving_for_random_coeffs() {
    // L2-norm conservation: ||samples||_2 == ||coefficients||_2.
    // Five deterministic random trials.
    for trial in 0..5usize {
        let mut c = [0.0f32; 16];
        for (i, slot) in c.iter_mut().enumerate() {
            *slot = ((trial * 41 + i * 23 + 7) % 41) as f32 / 41.0 - 0.5;
        }
        let s = afv_idct(&c).unwrap();
        let cn: f32 = c.iter().map(|v| v * v).sum::<f32>().sqrt();
        let sn: f32 = s.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (sn - cn).abs() < 1e-3,
            "trial {trial}: ||samples||={sn}, ||coeffs||={cn}"
        );
    }
}
