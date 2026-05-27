//! Round 164 — `TransformType`-driven entry points for the §C.8.3
//! per-block HF coefficient decode loop (DCT16×16 + DCT16×8 dimensions
//! exercised end-to-end).
//!
//! Round 159 landed the raw `(num_blocks, size, natural_order)` entry
//! points; the per-block state machine is shape-agnostic (Listing C.14
//! parameterises on `num_blocks` and `size` only), but the round-159
//! tests only pinned the DCT8×8 shape. This round adds:
//!
//! * `pass_group_hf::transform_block_params(t)` →
//!   `(num_blocks, size)` per §I.2.4 opening paragraph + Listing C.14
//!   (`num_blocks = (bwidth / 8) × (bheight / 8)`,
//!   `size = bwidth × bheight`).
//! * `pass_group_hf::decode_block_coefficients_for_transform(t, ..)` —
//!   typed wrapper that derives `(num_blocks, size, natural_order)`
//!   from the [`TransformType`] and reduces to
//!   [`pass_group_hf::decode_block_coefficients`].
//! * `pass_group_hf::read_non_zeros_and_decode_block_for_transform(t, ..)` —
//!   analogous typed wrapper around
//!   [`pass_group_hf::read_non_zeros_and_decode_block`].
//!
//! All truth is from the FDIS PDF Listings C.13 + C.14 + §I.2.4 opening
//! paragraph. No external library consulted.

use oxideav_jpegxl::coeff_order::{natural_coeff_order, OrderId};
use oxideav_jpegxl::dct_select::TransformType;
use oxideav_jpegxl::pass_group_hf::{
    decode_block_coefficients, decode_block_coefficients_for_transform,
    read_non_zeros_and_decode_block_for_transform, transform_block_params,
};

/// `transform_block_params` for DCT16×16 returns `(4, 256)` per §I.2.4
/// opening paragraph: `bwidth = bheight = 16` → `num_blocks = (16/8)
/// × (16/8) = 4`, `size = 16 × 16 = 256`.
#[test]
fn transform_block_params_dct16x16() {
    assert_eq!(transform_block_params(TransformType::Dct16x16), (4, 256));
}

/// `transform_block_params` for the rectangular DCT16×8 returns `(2,
/// 128)`: `(bwidth, bheight) = (16, 8)` → `num_blocks = 2`,
/// `size = 128`.
#[test]
fn transform_block_params_dct16x8() {
    assert_eq!(transform_block_params(TransformType::Dct16x8), (2, 128));
}

/// Typed entry point at DCT16×16, all-zero `initial_non_zeros`:
/// no closure calls (the `non_zeros == 0` early-stop fires immediately),
/// every coefficient is zero, and the buffer is sized at `size = 256`.
#[test]
fn decode_block_coefficients_for_transform_dct16x16_all_zero_is_empty_block() {
    let mut calls = 0;
    let decoded = decode_block_coefficients_for_transform(TransformType::Dct16x16, 0, 0, 1, |_| {
        calls += 1;
        Ok(0)
    })
    .unwrap();
    assert_eq!(calls, 0);
    assert_eq!(decoded.coeffs.len(), 256);
    assert_eq!(decoded.coeffs_read, 0);
    assert_eq!(decoded.remaining_non_zeros, 0);
    assert!(decoded.coeffs.iter().all(|&v| v == 0));
}

/// Typed entry point at DCT16×16 with one non-zero. The loop reads
/// one symbol then halts; the coefficient lands at
/// `natural_coeff_order(Id2)[num_blocks]` (the first HF cell).
#[test]
fn decode_block_coefficients_for_transform_dct16x16_first_nonzero() {
    let order = natural_coeff_order(OrderId::Id2);
    let mut calls = 0;
    let decoded = decode_block_coefficients_for_transform(TransformType::Dct16x16, 1, 0, 1, |_| {
        calls += 1;
        // ucoeff = 2 → UnpackSigned(2) = 1
        Ok(2)
    })
    .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(decoded.coeffs_read, 1);
    assert_eq!(decoded.remaining_non_zeros, 0);
    let first_hf_pos = order[4] as usize;
    assert_eq!(decoded.coeffs[first_hf_pos], 1, "UnpackSigned(2) == +1");
    for (i, &v) in decoded.coeffs.iter().enumerate() {
        if i == first_hf_pos {
            continue;
        }
        assert_eq!(v, 0, "coefficient at slot {i} should be zero");
    }
}

/// Typed entry point at DCT16×16, three consecutive non-zeros. The
/// loop runs for k = 4, 5, 6 (three reads) then halts when non_zeros
/// reaches 0. Coefficients land at natural_order[4..7].
#[test]
fn decode_block_coefficients_for_transform_dct16x16_three_nonzeros() {
    let order = natural_coeff_order(OrderId::Id2);
    // ucoeff sequence [2, 4, 6] → signed [+1, +2, +3].
    let sequence = [2u32, 4, 6];
    let mut idx = 0;
    let decoded = decode_block_coefficients_for_transform(TransformType::Dct16x16, 3, 0, 1, |_| {
        let v = sequence[idx];
        idx += 1;
        Ok(v)
    })
    .unwrap();
    assert_eq!(decoded.coeffs_read, 3);
    assert_eq!(decoded.remaining_non_zeros, 0);
    assert_eq!(decoded.coeffs[order[4] as usize], 1);
    assert_eq!(decoded.coeffs[order[5] as usize], 2);
    assert_eq!(decoded.coeffs[order[6] as usize], 3);
    // Every other position is zero.
    for (i, &v) in decoded.coeffs.iter().enumerate() {
        if i == order[4] as usize || i == order[5] as usize || i == order[6] as usize {
            continue;
        }
        assert_eq!(v, 0, "coefficient at slot {i} should be zero");
    }
}

/// Full-density DCT16×16 varblock: `initial_non_zeros = size -
/// num_blocks = 252`. The loop reads exactly 252 times; every HF
/// position is non-zero; the four LLF positions (natural_order[0..4])
/// are untouched.
#[test]
fn decode_block_coefficients_for_transform_dct16x16_full_density() {
    let order = natural_coeff_order(OrderId::Id2);
    let mut calls = 0;
    let decoded =
        decode_block_coefficients_for_transform(TransformType::Dct16x16, 252, 0, 1, |_| {
            calls += 1;
            Ok(2) // every read returns ucoeff=2 → signed +1, non_zeros decrements
        })
        .unwrap();
    assert_eq!(calls, 252);
    assert_eq!(decoded.coeffs_read, 252);
    assert_eq!(decoded.remaining_non_zeros, 0);
    let nonzero = decoded.coeffs.iter().filter(|&&v| v != 0).count();
    assert_eq!(nonzero, 252);
    // The four LLF cells (natural_order[0..4]) are untouched.
    for (k, &pos) in order.iter().enumerate().take(4) {
        assert_eq!(
            decoded.coeffs[pos as usize], 0,
            "LLF cell at k={k} should be untouched",
        );
    }
}

/// Typed entry point for DCT16×16 against the raw entry point on the
/// matching natural-order vector yields the *same* DecodedHfBlock.
/// (The typed wrapper is plumbing only; it MUST NOT alter the
/// per-block state machine.)
#[test]
fn decode_block_coefficients_for_transform_dct16x16_matches_raw() {
    let order = natural_coeff_order(OrderId::Id2);
    let sequence = [2u32, 0, 4, 0, 0, 6];
    let mut idx_a = 0;
    let typed = decode_block_coefficients_for_transform(TransformType::Dct16x16, 3, 7, 15, |_| {
        let v = sequence[idx_a];
        idx_a += 1;
        Ok(v)
    })
    .unwrap();
    let mut idx_b = 0;
    let raw = decode_block_coefficients(&order, 4, 256, 3, 7, 15, |_| {
        let v = sequence[idx_b];
        idx_b += 1;
        Ok(v)
    })
    .unwrap();
    assert_eq!(typed, raw);
    // Both should have decoded 6 symbols (3 zeros + 3 non-zeros, the
    // third non-zero decrements non_zeros to 0).
    assert_eq!(typed.coeffs_read, 6);
}

/// Typed `read_non_zeros_and_decode_block_for_transform` at DCT16×16:
/// the first closure reads `non_zeros` against the
/// `NonZerosContext(predicted)` context, the second decodes that many
/// coefficients.
#[test]
fn read_non_zeros_and_decode_block_for_transform_dct16x16_threads_state() {
    let order = natural_coeff_order(OrderId::Id2);
    let predicted = 12u32;
    let mut coeff_calls = 0u32;
    let (decoded, non_zeros) = read_non_zeros_and_decode_block_for_transform(
        TransformType::Dct16x16,
        predicted,
        0,
        1,
        |_| Ok(2),
        |_| {
            coeff_calls += 1;
            Ok(2) // signed +1
        },
    )
    .unwrap();
    assert_eq!(non_zeros, 2);
    assert_eq!(coeff_calls, 2);
    assert_eq!(decoded.coeffs_read, 2);
    assert_eq!(decoded.coeffs[order[4] as usize], 1);
    assert_eq!(decoded.coeffs[order[5] as usize], 1);
}

/// DCT16×8 typed entry point, one non-zero. Coefficient lands at
/// natural_coeff_order(Id4)[num_blocks] = natural_order[2].
#[test]
fn decode_block_coefficients_for_transform_dct16x8_first_nonzero() {
    let order = natural_coeff_order(OrderId::Id4);
    assert_eq!(order.len(), 128);
    let mut calls = 0;
    let decoded = decode_block_coefficients_for_transform(TransformType::Dct16x8, 1, 0, 1, |_| {
        calls += 1;
        Ok(1) // UnpackSigned(1) = -1
    })
    .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(decoded.coeffs.len(), 128);
    let first_hf_pos = order[2] as usize;
    assert_eq!(decoded.coeffs[first_hf_pos], -1);
}

/// DCT8×16 typed entry point. Same OrderId::Id4 + same (num_blocks,
/// size) as DCT16×8 — the two share the same scan order — so the
/// behavioural outcome is identical at the per-block layer.
#[test]
fn decode_block_coefficients_for_transform_dct8x16_collapses_to_dct16x8() {
    let mut calls_a = 0;
    let a = decode_block_coefficients_for_transform(TransformType::Dct16x8, 2, 0, 1, |_| {
        calls_a += 1;
        Ok(2)
    })
    .unwrap();
    let mut calls_b = 0;
    let b = decode_block_coefficients_for_transform(TransformType::Dct8x16, 2, 0, 1, |_| {
        calls_b += 1;
        Ok(2)
    })
    .unwrap();
    assert_eq!(a, b);
    assert_eq!(calls_a, calls_b);
    assert_eq!(a.coeffs_read, 2);
}

/// Defensive validation at the typed layer: `initial_non_zeros > size
/// - num_blocks` is rejected. For DCT16×16 that's anything > 252.
#[test]
fn decode_block_coefficients_for_transform_dct16x16_rejects_oversize_initial_non_zeros() {
    let r = decode_block_coefficients_for_transform(TransformType::Dct16x16, 253, 0, 1, |_| Ok(0));
    assert!(r.is_err());
    // 252 is the maximum legal value.
    let ok = decode_block_coefficients_for_transform(TransformType::Dct16x16, 252, 0, 1, |_| Ok(0));
    assert!(ok.is_ok());
}

/// Larger transform sizes are also accepted at the typed layer.
/// DCT32×32 → (num_blocks, size) = (16, 1024). Smoke-test one
/// non-zero at the first HF slot (= natural_order[16]).
#[test]
fn decode_block_coefficients_for_transform_dct32x32_first_nonzero() {
    let order = natural_coeff_order(OrderId::Id3);
    assert_eq!(order.len(), 1024);
    let mut calls = 0;
    let decoded = decode_block_coefficients_for_transform(TransformType::Dct32x32, 1, 0, 1, |_| {
        calls += 1;
        Ok(2)
    })
    .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(decoded.coeffs.len(), 1024);
    let first_hf_pos = order[16] as usize;
    assert_eq!(decoded.coeffs[first_hf_pos], 1);
}
