//! Splines image feature — ISO/IEC FDIS 18181-1:2021 §C.4.6 (codestream
//! decode) and §K.3 (rendering).
//!
//! A JXL frame whose `frame_header.flags` sets `kSplines` (§C.2.6,
//! `0x10`) carries a dictionary of *centripetal Catmull-Rom* splines that
//! are drawn on top of the frame — after patches (§K.2) and before noise
//! (§K.4) — by pixel-by-pixel addition of a Gaussian brush swept along
//! each spline's arc length (§K.3, Listing K.1).
//!
//! This module implements the **feature** in self-contained, spec-cited
//! layers so each stage is independently testable against the FDIS
//! listings (the crate's established synthetic-fixture pattern):
//!
//! 1. Coefficient post-processing (this file, §C.4.6): [`decode_double_delta`]
//!    (Listing C.4), [`quant_adjust_divisor`], [`K_CHANNEL_WEIGHT`],
//!    [`dequant_dct32`], [`recorrelate_xb`].
//! 2. [`continuous_idct`] — the per-arc-length coefficient evaluator
//!    defined at the head of §K.3.
//!
//! Later layers add the §K.1 control-point upsampling, arc-length
//! resampling, and the erf-based Gaussian splat, then the §C.4.6 entropy
//! parse that produces [`Spline`]s from the codestream.
//!
//! All maths follows the FDIS listings verbatim; no external decoder
//! source is consulted (see the crate README "History" note).

/// A 2-D point with element-wise arithmetic, as used by the §K.1
/// rendering listing (`Mirror`, control-point upsampling, arc-length
/// sampling). Coordinates are pixel positions and become fractional
/// during Catmull-Rom upsampling, so they are held as `f32`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    #[inline]
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// Per-channel weight applied to the dequantized DCT32 coefficients,
/// indexed by channel in the order X, Y, B, σ (FDIS §C.4.6):
/// `kChannelWeight[4] = {0.0042, 0.075, 0.07, 0.3333}`.
pub const K_CHANNEL_WEIGHT: [f32; 4] = [0.0042, 0.075, 0.07, 0.3333];

/// Number of DCT coefficients decoded per spline channel (DCT32, §C.4.6).
pub const SPLINE_DCT_LEN: usize = 32;

/// FDIS §C.4.6, Listing C.4 — `DecodeDoubleDelta`.
///
/// Reconstructs a coordinate sequence from a starting value and a list of
/// second-order (double) deltas:
///
/// ```text
/// current_value = starting_value; current_delta = 0;
/// for each delta d:
///   current_delta += d;
///   current_value += current_delta;
///   emit current_value;
/// ```
///
/// The returned vector has `deltas.len() + 1` entries, the first being
/// `starting_value` itself.
pub fn decode_double_delta(starting_value: i64, deltas: &[i64]) -> Vec<i64> {
    let mut out = Vec::with_capacity(deltas.len() + 1);
    out.push(starting_value);
    let mut current_value = starting_value;
    let mut current_delta: i64 = 0;
    for &d in deltas {
        current_delta += d;
        current_value += current_delta;
        out.push(current_value);
    }
    out
}

/// FDIS §C.4.6 — the `quant_adjust` divisor applied to the decoded DCT32
/// coefficients before the per-channel weighting:
///
/// > the DCT32 coefficients ... are divided by
/// > `quant_adjust >= 0 ? 1 + quant_adjust / 8 : 1 / (1 + quant_adjust / 8)`.
///
/// The divisions are real-valued (the coefficients are floating-point at
/// this stage).
pub fn quant_adjust_divisor(quant_adjust: i32) -> f32 {
    let qa = quant_adjust as f32;
    if quant_adjust >= 0 {
        1.0 + qa / 8.0
    } else {
        1.0 / (1.0 - qa / 8.0)
    }
}

/// FDIS §C.4.6 — dequantize one spline channel's 32 DCT coefficients.
///
/// Each raw (integer) coefficient is divided by [`quant_adjust_divisor`]
/// and multiplied by [`K_CHANNEL_WEIGHT`]`[channel]`, where `channel` is
/// in `[0, 4)` for X, Y, B, σ respectively.
///
/// Returns `None` when `channel >= 4`.
pub fn dequant_dct32(
    raw: &[i32; SPLINE_DCT_LEN],
    quant_adjust: i32,
    channel: usize,
) -> Option<[f32; SPLINE_DCT_LEN]> {
    if channel >= K_CHANNEL_WEIGHT.len() {
        return None;
    }
    let divisor = quant_adjust_divisor(quant_adjust);
    let weight = K_CHANNEL_WEIGHT[channel];
    let mut out = [0.0f32; SPLINE_DCT_LEN];
    for (o, &r) in out.iter_mut().zip(raw.iter()) {
        *o = (r as f32 / divisor) * weight;
    }
    Some(out)
}

/// FDIS §C.4.6 — recorrelate the X and B channels from the Y channel
/// before rendering:
///
/// > Before rendering splines, the decoder adds `Y × base_correlation_x`
/// > and `Y × base_correlation_b`, respectively, to the X and B channels.
///
/// The DCT is linear, so this per-coefficient add on the DCT32 vectors is
/// equivalent to the spatial recorrelation of the `ContinuousIDCT`
/// samples. `dct_y` is read (unmodified) into both `dct_x` and `dct_b`.
pub fn recorrelate_xb(
    dct_x: &mut [f32; SPLINE_DCT_LEN],
    dct_b: &mut [f32; SPLINE_DCT_LEN],
    dct_y: &[f32; SPLINE_DCT_LEN],
    base_correlation_x: f32,
    base_correlation_b: f32,
) {
    for ((x, b), &y) in dct_x.iter_mut().zip(dct_b.iter_mut()).zip(dct_y.iter()) {
        *x += base_correlation_x * y;
        *b += base_correlation_b * y;
    }
}

/// FDIS §K.3 — `ContinuousIDCT(dct, t)`.
///
/// Evaluates a DCT32 coefficient vector at continuous arc-length
/// parameter `t`:
///
/// ```text
/// ContinuousIDCT(dct, t) =
///   dct[0] + sum for k in [1, 32):
///     sqrt(2) × dct[k] × cos(k × (π / 32) × (t + 0.5))
/// ```
pub fn continuous_idct(dct: &[f32; SPLINE_DCT_LEN], t: f32) -> f32 {
    // Use f64 accumulation for the trig sum; the spec quantity is a real
    // value and f64 avoids order-dependent f32 rounding in the 31-term
    // sum. The result is returned as f32 (the frame pipeline is f32).
    const SQRT2: f64 = std::f64::consts::SQRT_2;
    let mut acc = dct[0] as f64;
    let phase = (t as f64 + 0.5) * (std::f64::consts::PI / 32.0);
    for (k, &c) in dct.iter().enumerate().skip(1) {
        acc += SQRT2 * c as f64 * (k as f64 * phase).cos();
    }
    acc as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_delta_first_value_is_starting_point() {
        // Listing C.4: the starting value is emitted unchanged first.
        let out = decode_double_delta(7, &[]);
        assert_eq!(out, vec![7]);
    }

    #[test]
    fn double_delta_constant_second_difference_is_quadratic() {
        // With every delta == 1 the second difference is constant, so the
        // sequence grows quadratically: starting 0, deltas [1,1,1,1]
        //   current_delta: 1,2,3,4  → values 1,3,6,10 (triangular numbers).
        let out = decode_double_delta(0, &[1, 1, 1, 1]);
        assert_eq!(out, vec![0, 1, 3, 6, 10]);
    }

    #[test]
    fn double_delta_zero_deltas_is_affine() {
        // Zero deltas → current_delta stays 0 → value stays at start.
        let out = decode_double_delta(5, &[0, 0, 0]);
        assert_eq!(out, vec![5, 5, 5, 5]);

        // A single non-zero first delta then zeros → straight line
        // (constant first difference).
        let out = decode_double_delta(2, &[3, 0, 0]);
        assert_eq!(out, vec![2, 5, 8, 11]);
    }

    #[test]
    fn quant_adjust_divisor_non_negative() {
        assert_eq!(quant_adjust_divisor(0), 1.0);
        assert_eq!(quant_adjust_divisor(8), 2.0);
        assert_eq!(quant_adjust_divisor(4), 1.5);
    }

    #[test]
    fn quant_adjust_divisor_negative_is_reciprocal() {
        // qa = -8 → 1 / (1 - (-8)/8) = 1 / 2 = 0.5.
        assert_eq!(quant_adjust_divisor(-8), 0.5);
        // qa = -4 → 1 / (1 + 4/8) = 1 / 1.5.
        assert!((quant_adjust_divisor(-4) - (1.0 / 1.5)).abs() < 1e-6);
    }

    #[test]
    fn dequant_applies_divisor_then_channel_weight() {
        let mut raw = [0i32; SPLINE_DCT_LEN];
        raw[0] = 16;
        raw[1] = 8;
        // channel 1 (Y), qa = 8 → divisor 2.0, weight 0.075.
        let out = dequant_dct32(&raw, 8, 1).unwrap();
        assert!((out[0] - (16.0 / 2.0) * 0.075).abs() < 1e-6);
        assert!((out[1] - (8.0 / 2.0) * 0.075).abs() < 1e-6);
        assert_eq!(out[2], 0.0);
    }

    #[test]
    fn dequant_rejects_out_of_range_channel() {
        let raw = [0i32; SPLINE_DCT_LEN];
        assert!(dequant_dct32(&raw, 0, 4).is_none());
    }

    #[test]
    fn recorrelate_adds_scaled_y_to_x_and_b() {
        let dct_y = {
            let mut y = [0.0f32; SPLINE_DCT_LEN];
            y[0] = 4.0;
            y[3] = -2.0;
            y
        };
        let mut dct_x = [1.0f32; SPLINE_DCT_LEN];
        let mut dct_b = [0.0f32; SPLINE_DCT_LEN];
        // base_correlation_x = 0.0 (default) leaves X untouched; b = 1.0.
        recorrelate_xb(&mut dct_x, &mut dct_b, &dct_y, 0.0, 1.0);
        assert_eq!(dct_x[0], 1.0);
        assert_eq!(dct_x[3], 1.0);
        assert_eq!(dct_b[0], 4.0);
        assert_eq!(dct_b[3], -2.0);

        // Non-zero base_correlation_x propagates too.
        let mut dct_x2 = [0.0f32; SPLINE_DCT_LEN];
        let mut dct_b2 = [0.0f32; SPLINE_DCT_LEN];
        recorrelate_xb(&mut dct_x2, &mut dct_b2, &dct_y, 0.5, 0.9921875);
        assert!((dct_x2[0] - 2.0).abs() < 1e-6);
        assert!((dct_b2[0] - 4.0 * 0.9921875).abs() < 1e-6);
    }

    #[test]
    fn continuous_idct_dc_only_is_flat() {
        // Only dct[0] non-zero → the value is dct[0] for every t.
        let mut dct = [0.0f32; SPLINE_DCT_LEN];
        dct[0] = 3.5;
        for &t in &[0.0f32, 5.0, 15.5, 31.0] {
            assert!((continuous_idct(&dct, t) - 3.5).abs() < 1e-5);
        }
    }

    #[test]
    fn continuous_idct_matches_manual_single_harmonic() {
        // dct[1] = 1.0, everything else 0 → value = sqrt(2)*cos(π/32*(t+0.5)).
        let mut dct = [0.0f32; SPLINE_DCT_LEN];
        dct[1] = 1.0;
        for &t in &[0.0f32, 7.0, 20.0] {
            let want = (2.0f64).sqrt() * ((t as f64 + 0.5) * std::f64::consts::PI / 32.0).cos();
            assert!((continuous_idct(&dct, t) as f64 - want).abs() < 1e-5);
        }
    }
}
