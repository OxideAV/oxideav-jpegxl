//! Round 278 — pin the Listing E.2 `error2weight` **operand order**
//! (ISO/IEC FDIS 18181-1:2021 Annex E.2) against the staged clean-room
//! behavioural trace.
//!
//! ## The two readings
//!
//! The FDIS-2021 listing reads
//!
//! ```text
//! return 4 + (maxweight × ((1 << 24) Idiv ((err_sum >> shift) + 1)));
//! ```
//!
//! i.e. the `(1 << 24) Idiv denom` inner division truncates FIRST and
//! the quotient is then multiplied by `maxweight`. The pre-round-278
//! production code instead computed `maxweight * (1 << 24) / denom` —
//! multiplying into the numerator before dividing — which yields a
//! slightly larger product whenever the inner division truncates.
//! Round 191 documented the discrepancy but found it a no-op at the
//! sample-194 trace point (both readings give the same post-`>> sh`
//! weights there).
//!
//! ## Why the trace discriminates
//!
//! `docs/image/jpegxl/fixtures/noise-64x64-lossless/
//! wp-trace-sample-194.md` (surrounding-sample context, samples
//! 188..200) reports the FULL-PRECISION `weight_i` for 13 samples ×
//! 4 sub-predictors = 52 cells. All 52 match the Idiv-first reading
//! exactly; the multiply-first reading mismatches 18 of them. The
//! sharpest cell is sample 192's `err_sum_0 = 51` (shift = 0,
//! denom = 52): Idiv-first gives `4 + 13 × floor(2^24 / 52)` =
//! 4194298 (= the trace), multiply-first gives
//! `4 + floor(13 × 2^24 / 52)` = 4194308 (the division is exact once
//! 13 is in the numerator, so the +10 truncation loss reappears).
//!
//! The full-precision difference propagates through the Listing E.3
//! `>> (log_weight - 5)` rounding at some samples; together with the
//! `true_errNW` column-0 fallback this was the root cause of the
//! long-standing `noise-64x64-lossless` sample-129/-194 WP
//! state-evolution divergence (rounds 31..272) and the `synth_320`
//! (y=24, x=14) drift (round 10).
//!
//! ## WpHeader parameters in effect (trace doc table)
//!
//! `wp_w0 = 13`, `wp_w1 = wp_w2 = wp_w3 = 12`.

use oxideav_jpegxl::modular_fdis::error2weight_pub;

/// `(err_sum[4], weight[4])` per sample 188..=200, from the trace
/// doc's surrounding-sample context block (lines 130-168).
const TRACE_CELLS: &[(u32, [u32; 4], [u64; 4])] = &[
    (188, [472, 691, 780, 530], [454386, 285979, 256798, 370089]),
    (189, [342, 669, 781, 363], [634025, 299596, 256798, 547087]),
    (190, [228, 782, 667, 422], [940105, 256798, 299596, 474830]),
    (191, [337, 1214, 930, 806], [634025, 165568, 213273, 246727]),
    (192, [51, 265, 334, 235], [4194298, 740174, 599189, 853081]),
    (193, [222, 343, 338, 200], [973681, 585254, 585254, 986899]),
    (194, [438, 330, 416, 240], [495694, 599189, 474830, 825112]),
    (195, [456, 526, 461, 410], [470054, 381304, 433897, 483961]),
    (196, [502, 442, 399, 547], [432749, 449393, 503320, 359515]),
    (197, [421, 423, 347, 484], [514399, 474830, 571954, 412558]),
    (198, [556, 307, 518, 314], [389475, 645281, 381304, 629149]),
    (199, [676, 393, 422, 426], [317014, 503320, 474830, 466037]),
    (200, [795, 568, 575, 602], [272633, 349528, 349528, 331132]),
];

/// `wp_w0..wp_w3` for this fixture's first weighted-predicted channel.
const WCFG: [u32; 4] = [13, 12, 12, 12];

/// All 52 traced full-precision `(err_sum, weight)` cells must match
/// the production `error2weight`.
#[test]
fn all_52_trace_cells_match_production_error2weight() {
    for &(s, err_sums, weights) in TRACE_CELLS {
        for k in 0..4 {
            let got = error2weight_pub(err_sums[k], WCFG[k]);
            assert_eq!(
                got, weights[k],
                "error2weight(err_sum={}, maxweight={}) at trace sample \
                 {s} sub-predictor {k} must equal the trace's \
                 full-precision weight {}; got {}",
                err_sums[k], WCFG[k], weights[k], got
            );
        }
    }
}

/// The sharpest discriminating cell: at sample 192's `err_sum_0 = 51`
/// the multiply-first reading is +10 off the trace value, so this
/// single assert fails immediately if the operand order regresses.
#[test]
fn sample_192_err_sum_51_pins_the_idiv_first_operand_order() {
    // shift = floor(log2(52)) - 5 = 0; denom = 52.
    // Idiv-first:      4 + 13 * (2^24 Idiv 52) = 4 + 13 * 322638 = 4194298
    // multiply-first:  4 + (13 * 2^24) Idiv 52 = 4 + 4194304    = 4194308
    let got = error2weight_pub(51, 13);
    assert_eq!(
        got, 4194298,
        "error2weight(51, 13) must follow the FDIS-2021 Idiv-first \
         parenthesisation (trace value 4194298); 4194308 means the \
         maxweight multiplication was moved back into the numerator"
    );
}

/// Idiv-first ≤ multiply-first always (the inner truncation can only
/// lose magnitude); pin the bound across a sweep so a future
/// "simplification" that silently reorders the operations is caught
/// even off the traced cells.
#[test]
fn idiv_first_is_bounded_by_multiply_first_across_sweep() {
    for err_sum in 0..4096u32 {
        for &mw in &[12u32, 13] {
            let bits = 32 - (err_sum + 1).leading_zeros();
            let shift = bits.saturating_sub(1).saturating_sub(5);
            let denom = ((err_sum >> shift) as u64) + 1;
            let mul_first = 4 + ((mw as u64 * (1u64 << 24) / denom) >> shift);
            let got = error2weight_pub(err_sum, mw);
            assert!(
                got <= mul_first,
                "Idiv-first error2weight({err_sum}, {mw}) = {got} must \
                 never exceed the multiply-first form {mul_first}"
            );
        }
    }
}
