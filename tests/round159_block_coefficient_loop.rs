//! Round 159 — §C.8.3 per-block HF coefficient decode loop scaffolding.
//!
//! This integration test exercises the public scaffolding API from
//! `oxideav_jpegxl::pass_group_hf` against a hand-rolled symbol source,
//! independent of the (still un-landed) §C.7.2 entropy stream
//! histograms.
//!
//! Scope: DCT8×8 alone — `num_blocks = 1`, `size = 64`, the
//! `OrderId::Id0` natural-coefficient-order vector. Once §C.7.2 lands,
//! a follow-up round wires a real `EntropyStream` + `HybridUintState`
//! closure into the same primitive without touching this test (or the
//! primitive itself); these tests pin the per-block state machine on
//! its own.
//!
//! All truth is from the FDIS PDF Listing C.13 + Listing C.14 + the
//! surrounding §C.8.3 prose. No external library consulted.

use oxideav_jpegxl::coeff_order::{natural_coeff_order, OrderId};
use oxideav_jpegxl::pass_group_hf::{
    coefficient_context, decode_block_coefficients, non_zeros_context, prev_for_context,
    read_non_zeros_and_decode_block, COEFF_FREQ_CONTEXT, COEFF_NUM_NONZERO_CONTEXT,
};

/// Listing C.14: at the first iteration (k == num_blocks), prev depends
/// on `non_zeros > size / 16`.
#[test]
fn prev_for_context_dct8x8_threshold_at_5() {
    // size / 16 = 4 for DCT8×8 → strictly-greater comparison so the
    // crossover is at non_zeros == 5.
    for nz in 0..=4 {
        assert_eq!(
            prev_for_context(1, 1, 64, nz, |_| panic!("never called")),
            0
        );
    }
    for nz in 5..=63 {
        assert_eq!(
            prev_for_context(1, 1, 64, nz, |_| panic!("never called")),
            1
        );
    }
}

/// Listing C.14: subsequent iterations (k > num_blocks) flip prev based
/// on whether the coefficient at the previous k slot was non-zero.
#[test]
fn prev_for_context_follows_decoded_history() {
    // Construct a "decoded history": iteration k=2 has a non-zero
    // coefficient at the (k-1)=1 slot. Iteration k=3 sees prev=1;
    // every other k sees prev=0.
    let history = |kk: u32| kk == 1;
    assert_eq!(prev_for_context(2, 1, 64, 1, history), 1);
    assert_eq!(prev_for_context(3, 1, 64, 1, history), 0);
    assert_eq!(prev_for_context(4, 1, 64, 1, history), 0);
}

/// All-zero block (non_zeros = 0 going in) makes no symbol reads.
/// Listing C.14 prose: "If non_zeros reaches 0, the decoder stops
/// decoding further coefficients."
#[test]
fn decode_block_coefficients_zero_initial_non_zeros_is_empty_block() {
    let order = natural_coeff_order(OrderId::Id0);
    let mut calls = 0;
    let decoded = decode_block_coefficients(&order, 1, 64, 0, 0, 1, |_| {
        calls += 1;
        Ok(0)
    })
    .unwrap();
    assert_eq!(calls, 0);
    assert_eq!(decoded.coeffs_read, 0);
    assert_eq!(decoded.remaining_non_zeros, 0);
    assert!(decoded.coeffs.iter().all(|&v| v == 0));
}

/// A single non-zero coefficient at the first HF slot. Verifies:
/// * the closure is called exactly once;
/// * `UnpackSigned(1) == -1` lands at `natural_order[1]`;
/// * the remaining 63 positions are zero.
#[test]
fn decode_block_coefficients_first_nonzero_stops_immediately() {
    let order = natural_coeff_order(OrderId::Id0);
    let mut calls = 0;
    let decoded = decode_block_coefficients(&order, 1, 64, 1, 3, 15, |_| {
        calls += 1;
        Ok(1)
    })
    .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(decoded.coeffs_read, 1);
    assert_eq!(decoded.remaining_non_zeros, 0);
    let pos = order[1] as usize;
    assert_eq!(decoded.coeffs[pos], -1, "UnpackSigned(1) == -1");
    for (i, &v) in decoded.coeffs.iter().enumerate() {
        if i == pos {
            continue;
        }
        assert_eq!(v, 0);
    }
}

/// Three non-zero coefficients at the first three HF slots: the loop
/// runs to k = 4 then stops (non_zeros drops from 3 → 2 → 1 → 0 across
/// reads). Verifies the natural-order placement is right for the first
/// three HF cells of DCT8×8 (Table I.1 OrderId::Id0).
#[test]
fn decode_block_coefficients_three_nonzeros_at_consecutive_hf_slots() {
    let order = natural_coeff_order(OrderId::Id0);
    // ucoeff sequence: [2, 4, 6] → signed [+1, +2, +3].
    let sequence = [2u32, 4, 6];
    let mut idx = 0;
    let decoded = decode_block_coefficients(&order, 1, 64, 3, 0, 1, |_| {
        let v = sequence[idx];
        idx += 1;
        Ok(v)
    })
    .unwrap();
    assert_eq!(decoded.coeffs_read, 3);
    assert_eq!(decoded.remaining_non_zeros, 0);
    assert_eq!(decoded.coeffs[order[1] as usize], 1);
    assert_eq!(decoded.coeffs[order[2] as usize], 2);
    assert_eq!(decoded.coeffs[order[3] as usize], 3);
}

/// A fully-populated DCT8×8 block (non_zeros = 63 = size - num_blocks).
/// Every HF cell is non-zero; the loop reads `size - num_blocks` times;
/// the LLF cell (natural-order index 0) is untouched.
#[test]
fn decode_block_coefficients_full_density_reads_size_minus_one() {
    let order = natural_coeff_order(OrderId::Id0);
    let mut calls = 0;
    let decoded = decode_block_coefficients(&order, 1, 64, 63, 0, 1, |_| {
        calls += 1;
        Ok(2)
    })
    .unwrap();
    assert_eq!(calls, 63);
    let nonzero = decoded.coeffs.iter().filter(|&&v| v != 0).count();
    assert_eq!(nonzero, 63);
    assert_eq!(decoded.coeffs[order[0] as usize], 0);
}

/// CoefficientContext maps the (k, non_zeros, prev) triple through the
/// two ladder tables. We re-derive every read's context value from
/// the helper and compare against what the loop actually passes.
#[test]
fn decode_block_coefficients_context_threading_matches_listings_c13_c14() {
    let order = natural_coeff_order(OrderId::Id0);
    let block_ctx = 7;
    let nb_block_ctx = 15;
    let num_blocks = 1u32;
    let size = 64u32;

    // ucoeff sequence: [0, 0, 2] — first two zeros do NOT decrement
    // non_zeros, the third (ucoeff=2 → signed +1) does. Initial
    // non_zeros = 1 means the third read drops nz to 0 and the loop
    // stops → exactly 3 reads.
    let sequence = [0u32, 0, 2];
    let mut idx = 0;
    let mut seen_ctx: Vec<u32> = Vec::new();
    let _ = decode_block_coefficients(
        &order,
        num_blocks,
        size,
        1,
        block_ctx,
        nb_block_ctx,
        |ctx| {
            seen_ctx.push(ctx);
            let v = sequence[idx];
            idx += 1;
            Ok(v)
        },
    )
    .unwrap();
    assert_eq!(seen_ctx.len(), 3);

    // k=1: prev = 0 (non_zeros=1, 1>4 false).
    let expect0 = coefficient_context(1, 1, num_blocks, size, 0, block_ctx, nb_block_ctx).unwrap();
    assert_eq!(seen_ctx[0], expect0);
    // k=2: prev = 0 (ucoeff at k=1 was zero).
    let expect1 = coefficient_context(2, 1, num_blocks, size, 0, block_ctx, nb_block_ctx).unwrap();
    assert_eq!(seen_ctx[1], expect1);
    // k=3: prev = 0 (ucoeff at k=2 was zero).
    let expect2 = coefficient_context(3, 1, num_blocks, size, 0, block_ctx, nb_block_ctx).unwrap();
    assert_eq!(seen_ctx[2], expect2);
}

/// `read_non_zeros_and_decode_block` plumbs the
/// `NonZerosContext(predicted)` value to the first closure and the
/// per-iteration `CoefficientContext` to the second. The first
/// closure's return value seeds the second closure's loop.
#[test]
fn read_non_zeros_and_decode_block_threads_state_through_both_closures() {
    let order = natural_coeff_order(OrderId::Id0);
    let block_ctx = 2;
    let nb_block_ctx = 15;
    let predicted = 16;

    // NonZerosContext(16, 2, 15): predicted=16 → predicted >= 8 →
    // return 2 + 15 × (4 + 16/2) = 2 + 15 × 12 = 182.
    let expected_nz_ctx = non_zeros_context(predicted, block_ctx, nb_block_ctx);
    assert_eq!(expected_nz_ctx, 2 + 15 * (4 + 16 / 2));

    let mut got_nz_ctx = 0u32;
    let mut got_coeff_calls = 0u32;
    let (decoded, non_zeros) = read_non_zeros_and_decode_block(
        &order,
        1,
        64,
        predicted,
        block_ctx,
        nb_block_ctx,
        |ctx| {
            got_nz_ctx = ctx;
            Ok(2) // non_zeros = 2 → two HF cells, both non-zero below.
        },
        |_| {
            got_coeff_calls += 1;
            Ok(2) // signed +1
        },
    )
    .unwrap();
    assert_eq!(got_nz_ctx, expected_nz_ctx);
    assert_eq!(non_zeros, 2);
    assert_eq!(got_coeff_calls, 2);
    assert_eq!(decoded.coeffs_read, 2);
    assert_eq!(decoded.remaining_non_zeros, 0);
}

/// The two 64-element ladder tables are surfaced publicly. Spot-check
/// the boundaries: the spec sets `CoeffFreqContext[0..2] = 0`, then a
/// monotone non-decreasing 0..=30 ramp; `CoeffNumNonzeroContext[0..2] =
/// 0`, then increasing plateaus 31, 62, 93, 123, 152, 180, 206.
#[test]
fn coefficient_context_tables_are_publicly_visible_with_known_shape() {
    assert_eq!(COEFF_FREQ_CONTEXT.len(), 64);
    assert_eq!(COEFF_NUM_NONZERO_CONTEXT.len(), 64);

    // Monotone non-decreasing.
    for w in COEFF_FREQ_CONTEXT.windows(2) {
        assert!(w[1] >= w[0]);
    }
    for w in COEFF_NUM_NONZERO_CONTEXT.windows(2) {
        assert!(w[1] >= w[0]);
    }
    // Plateau ceilings.
    assert_eq!(*COEFF_FREQ_CONTEXT.last().unwrap(), 30);
    assert_eq!(*COEFF_NUM_NONZERO_CONTEXT.last().unwrap(), 206);
}

/// The scaffolding rejects malformed natural-order vectors. (Verifies
/// that the per-block primitive defensively validates its inputs
/// rather than panicking on bad caller data.)
#[test]
fn decode_block_coefficients_rejects_bad_inputs() {
    let good = natural_coeff_order(OrderId::Id0);
    // Bad: natural_order length mismatch.
    let short = good[..32].to_vec();
    assert!(decode_block_coefficients(&short, 1, 64, 0, 0, 1, |_| Ok(0)).is_err());
    // Bad: natural_order entry out of range.
    let mut oob = good.clone();
    oob[5] = 999;
    assert!(decode_block_coefficients(&oob, 1, 64, 0, 0, 1, |_| Ok(0)).is_err());
    // Bad: num_blocks = 0.
    assert!(decode_block_coefficients(&good, 0, 64, 0, 0, 1, |_| Ok(0)).is_err());
    // Bad: initial_non_zeros > size - num_blocks.
    assert!(decode_block_coefficients(&good, 1, 64, 64, 0, 1, |_| Ok(0)).is_err());
}

/// The scaffolding is plug-compatible with the existing
/// `non_zeros_context` + `coefficient_context` helpers. End-to-end
/// sanity: a block with one non-zero at slot 1 (ucoeff=2 → +1) sees
/// exactly one symbol decoded; its position is `natural_order[1]`.
#[test]
fn end_to_end_smoke_one_nonzero_at_first_hf_cell() {
    let order = natural_coeff_order(OrderId::Id0);
    let block_ctx = 0;
    let nb_block_ctx = 1;
    let predicted = 0;
    let (decoded, non_zeros) = read_non_zeros_and_decode_block(
        &order,
        1,
        64,
        predicted,
        block_ctx,
        nb_block_ctx,
        |_| Ok(1), // non_zeros = 1
        |_| Ok(2), // ucoeff = 2 → signed +1
    )
    .unwrap();
    assert_eq!(non_zeros, 1);
    assert_eq!(decoded.coeffs_read, 1);
    assert_eq!(decoded.coeffs[order[1] as usize], 1);
}
