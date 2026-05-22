//! Round-90 integration tests — HfPass + PassGroup HF structural
//! parse end-to-end at the *typed surface* level (no fixture decode).
//!
//! These tests exercise the round-90 contract:
//!
//! * §C.7.1 HfPass `used_orders == 0` fast path:
//!   - parses a single preset's `used_orders` value cleanly,
//!   - exposes the 13 natural coefficient orders verbatim per
//!     [`oxideav_jpegxl::coeff_order::natural_coeff_order`],
//!   - computes the §C.7.2 `num_histogram_distributions` correctly
//!     (`495 × num_hf_presets × nb_block_ctx`).
//!
//! * §C.8.3 PassGroup HF header `hfp = u(ceil(log2(num_hf_presets)))`:
//!   - parses cleanly for power-of-two and non-power-of-two preset
//!     counts,
//!   - rejects out-of-range `hfp` for non-power-of-two preset counts,
//!   - computes the `495 × nb_block_ctx × hfp` histogram offset
//!     correctly.
//!
//! * Listing C.13 BlockContext / NonZerosContext / CoefficientContext
//!   / PredictedNonZeros — exercised via the default 39-element
//!   `block_ctx_map` (the most common path) plus the spec's no-
//!   threshold shortcut.
//!
//! Pixel decode of a VarDCT fixture remains gated on the round-91+
//! ANS-stream wiring + `used_orders != 0` permutation reads + per-
//! block coefficient decode loop.

use oxideav_jpegxl::bitreader::BitReader;
use oxideav_jpegxl::coeff_order::{natural_coeff_order, OrderId, NUM_ORDERS};
use oxideav_jpegxl::hf_pass::{read_hf_pass_sequence, HfPass, ORDER_COEFFICIENT_COUNTS};
use oxideav_jpegxl::lf_global::HfBlockContext;
use oxideav_jpegxl::pass_group_hf::{
    block_context, coefficient_context, non_zeros_context, predicted_non_zeros, PassGroupHfHeader,
    COEFF_FREQ_CONTEXT, COEFF_NUM_NONZERO_CONTEXT,
};

/// Pack `(value, n_bits)` LSB-first. Mirrors the crate-private
/// `ans::test_helpers::pack_lsb` so integration tests don't depend on
/// it.
fn pack_lsb(parts: &[(u32, u32)]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let mut byte: u32 = 0;
    let mut bits: u32 = 0;
    for &(v, n) in parts {
        assert!(n <= 32);
        for i in 0..n {
            let bit = (v >> i) & 1;
            byte |= bit << bits;
            bits += 1;
            if bits == 8 {
                out.push(byte as u8);
                byte = 0;
                bits = 0;
            }
        }
    }
    if bits > 0 {
        out.push(byte as u8);
    }
    out
}

#[test]
fn hf_pass_used_orders_zero_yields_all_natural_orders() {
    let bytes = pack_lsb(&[(2, 2)]); // U32 selector 2 → Val(0)
    let mut br = BitReader::new(&bytes);
    let hp = HfPass::read(&mut br, 1, 15).unwrap();
    assert_eq!(hp.used_orders, 0);
    for i in 0..NUM_ORDERS as u32 {
        let o = OrderId::from_index(i).unwrap();
        let expected = natural_coeff_order(o);
        assert_eq!(hp.order_for(o), expected.as_slice(), "OrderId {i}");
        assert_eq!(
            hp.order_for(o).len(),
            ORDER_COEFFICIENT_COUNTS[i as usize] as usize
        );
    }
}

#[test]
fn hf_pass_histogram_distribution_count_matches_spec_formula() {
    // 495 × num_hf_presets × nb_block_ctx
    let bytes = pack_lsb(&[(2, 2)]);
    let mut br = BitReader::new(&bytes);
    let hp = HfPass::read(&mut br, 3, 7).unwrap();
    assert_eq!(hp.num_histogram_distributions, 495 * 3 * 7);
}

#[test]
fn hf_pass_used_orders_nonzero_unsupported_round_90() {
    // selector 3 (Bits(13)), payload = 1 → used_orders = 1.
    let bytes = pack_lsb(&[(3, 2), (1, 13)]);
    let mut br = BitReader::new(&bytes);
    let r = HfPass::read(&mut br, 1, 15);
    assert!(
        r.is_err(),
        "used_orders = 1 must error pending round 91 wiring"
    );
}

#[test]
fn read_hf_pass_sequence_threads_through_n_presets() {
    // Three consecutive presets, each with used_orders = 0.
    let bytes = pack_lsb(&[(2, 2), (2, 2), (2, 2)]);
    let mut br = BitReader::new(&bytes);
    let v = read_hf_pass_sequence(&mut br, 3, 15).unwrap();
    assert_eq!(v.len(), 3);
    for hp in &v {
        assert_eq!(hp.used_orders, 0);
    }
}

#[test]
fn pass_group_hf_header_single_preset_zero_bits() {
    // num_hf_presets = 1 → 0 bits for hfp → no reads.
    let bytes: Vec<u8> = vec![0];
    let mut br = BitReader::new(&bytes);
    let h = PassGroupHfHeader::read(&mut br, 1, 15).unwrap();
    assert_eq!(h.hfp, 0);
    assert_eq!(h.histogram_offset, 0);
}

#[test]
fn pass_group_hf_header_four_presets_hfp_2() {
    // num_hf_presets = 4 → 2 bits for hfp.
    let bytes = pack_lsb(&[(2, 2)]);
    let mut br = BitReader::new(&bytes);
    let h = PassGroupHfHeader::read(&mut br, 4, 15).unwrap();
    assert_eq!(h.hfp, 2);
    assert_eq!(h.histogram_offset, 495 * 15 * 2);
}

#[test]
fn pass_group_hf_header_rejects_oob_hfp() {
    // num_hf_presets = 3 → 2 bits. hfp = 3 is bit-legal but >= 3.
    let bytes = pack_lsb(&[(3, 2)]);
    let mut br = BitReader::new(&bytes);
    let r = PassGroupHfHeader::read(&mut br, 3, 1);
    assert!(r.is_err());
}

#[test]
fn block_context_default_map_table_b_channel() {
    // Default HfBlockContext (used_default = true) yields the spec
    // 39-element block_ctx_map and no thresholds. Compute BlockContext
    // for (c=2, s=0): idx = 2 × 13 + 0 = 26. block_ctx_map[26] = 7.
    let map = HfBlockContext::DEFAULT_BLOCK_CTX_MAP;
    let r = block_context(
        2,
        0,
        0,
        [0, 0, 0],
        &[],
        &[Vec::new(), Vec::new(), Vec::new()],
        &map,
    )
    .unwrap();
    assert_eq!(r, map[26] as u32);
}

#[test]
fn non_zeros_context_continuous_at_eight() {
    // predicted = 7 → block_ctx + 7 × nb.
    // predicted = 8 → block_ctx + (4 + 4) × nb = block_ctx + 8 × nb.
    // The two branches happen to coincide at the boundary.
    assert_eq!(non_zeros_context(7, 0, 1), 7);
    assert_eq!(non_zeros_context(8, 0, 1), 8);
}

#[test]
fn coefficient_context_spec_constants_reachable() {
    // Sanity that the listed tables can be referenced from outside
    // the crate at module-level visibility.
    assert_eq!(COEFF_FREQ_CONTEXT.len(), 64);
    assert_eq!(COEFF_NUM_NONZERO_CONTEXT.len(), 64);
    // Trivial CoefficientContext call (all zero inputs).
    let r = coefficient_context(0, 0, 1, 64, 0, 0, 1).unwrap();
    assert_eq!(r, 37);
}

#[test]
fn predicted_non_zeros_dispatch_table() {
    // (0, 0) → 32.
    assert_eq!(predicted_non_zeros(0, 0, |_, _| 0), 32);
    // (1, 0) → NonZeros(0, 0).
    assert_eq!(predicted_non_zeros(1, 0, |_, _| 17), 17);
    // (0, 1) → NonZeros(0, 0).
    assert_eq!(predicted_non_zeros(0, 1, |_, _| 17), 17);
    // (1, 1) → (NonZeros(1, 0) + NonZeros(0, 1) + 1) >> 1.
    assert_eq!(predicted_non_zeros(1, 1, |_, _| 10), (10 + 10 + 1) >> 1);
}

#[test]
fn pass_group_hf_select_pass_from_hf_pass_sequence() {
    // num_hf_presets = 2 → hfp = 1, select_pass returns passes[1].
    let bytes = pack_lsb(&[(1, 1)]);
    let mut br = BitReader::new(&bytes);
    let h = PassGroupHfHeader::read(&mut br, 2, 1).unwrap();
    let bytes2 = pack_lsb(&[(2, 2), (2, 2)]);
    let mut br2 = BitReader::new(&bytes2);
    let passes = read_hf_pass_sequence(&mut br2, 2, 1).unwrap();
    let chosen = h.select_pass(&passes).unwrap();
    assert_eq!(chosen.used_orders, 0);
}
