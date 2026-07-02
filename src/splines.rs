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

use crate::bitreader::unpack_signed;
use oxideav_core::{Error, Result};

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

    /// Element-wise `self + other`.
    #[inline]
    fn add(self, other: Point) -> Point {
        Point::new(self.x + other.x, self.y + other.y)
    }

    /// Element-wise `self - other`.
    #[inline]
    fn sub(self, other: Point) -> Point {
        Point::new(self.x - other.x, self.y - other.y)
    }

    /// Element-wise `self × scalar`.
    #[inline]
    fn scale(self, s: f32) -> Point {
        Point::new(self.x * s, self.y * s)
    }

    /// §K.3 `Mirror(center, point) = center + (center - point)`
    /// (= `2·center − point`, element-wise).
    #[inline]
    fn mirror(center: Point, point: Point) -> Point {
        center.add(center.sub(point))
    }

    /// Linear interpolation `self + s × (other − self)`.
    #[inline]
    fn lerp(self, other: Point, s: f32) -> Point {
        self.add(other.sub(self).scale(s))
    }
}

/// Number of interior sub-steps inserted between each pair of adjacent
/// control points by the §K.1 Catmull-Rom upsampling loop (`step` runs
/// over `[1, 16)`).
const UPSAMPLE_STEPS: u32 = 16;

/// FDIS §K.3, Listing K.1 (first block) — upsample the decoded control
/// points into a dense polyline via *centripetal* Catmull-Rom
/// interpolation.
///
/// A single control point yields itself. Otherwise the point list is
/// extended at both ends with mirror points, then for every window of
/// four consecutive extended points the segment between the two middle
/// points is subdivided into 16 sub-steps using the centripetal knot
/// spacing `t[k] = t[k-1] + ((Δx)² + (Δy)²)^0.25` (i.e. `√distance`). The
/// final control point is appended verbatim.
///
/// NOTE (spec faithfulness): the listing performs no guard against
/// coincident consecutive control points; when two points coincide a knot
/// interval is zero and the interpolation divides by zero, exactly as the
/// listing is written. Encoders do not emit coincident control points, so
/// this matches conformant streams.
pub fn upsample_control_points(control_points: &[Point]) -> Vec<Point> {
    if control_points.is_empty() {
        return Vec::new();
    }
    if control_points.len() == 1 {
        return vec![control_points[0]];
    }

    let n = control_points.len();
    // extended = [Mirror(S[0], S[1]), S..., Mirror(S[n-1], S[n-2])].
    let mut extended = Vec::with_capacity(n + 2);
    extended.push(Point::mirror(control_points[0], control_points[1]));
    extended.extend_from_slice(control_points);
    extended.push(Point::mirror(control_points[n - 1], control_points[n - 2]));

    let mut out = Vec::new();
    // for (i = 0; i < extended.size() - 3; ++i)
    for i in 0..extended.len().saturating_sub(3) {
        let p = [
            extended[i],
            extended[i + 1],
            extended[i + 2],
            extended[i + 3],
        ];
        out.push(p[1]);

        // Centripetal knot vector t[0..4], t[0] = 0.
        let mut t = [0.0f32; 4];
        for k in 1..4 {
            let d = p[k].sub(p[k - 1]);
            t[k] = t[k - 1] + (d.x * d.x + d.y * d.y).powf(0.25);
        }

        for step in 1..UPSAMPLE_STEPS {
            let knot = t[1] + (step as f32 / UPSAMPLE_STEPS as f32) * (t[2] - t[1]);
            // A[k] = p[k] + ((knot - t[k]) / (t[k+1] - t[k])) × (p[k+1] - p[k])
            let mut a = [Point::new(0.0, 0.0); 3];
            for (k, ak) in a.iter_mut().enumerate() {
                let s = (knot - t[k]) / (t[k + 1] - t[k]);
                *ak = p[k].lerp(p[k + 1], s);
            }
            // B[k] = A[k] + ((knot - t[k]) / (t[k+2] - t[k])) × (A[k+1] - A[k])
            let mut b = [Point::new(0.0, 0.0); 2];
            for (k, bk) in b.iter_mut().enumerate() {
                let s = (knot - t[k]) / (t[k + 2] - t[k]);
                *bk = a[k].lerp(a[k + 1], s);
            }
            let s = (knot - t[1]) / (t[2] - t[1]);
            out.push(b[0].lerp(b[1], s));
        }
    }
    out.push(control_points[n - 1]);
    out
}

/// A point sampled along a spline at (nominally) unit arc-length spacing,
/// together with the listing's `.arclength` weight `d` (1 for interior
/// samples, the fractional remainder for the final sample). Produced by
/// [`resample_by_arclength`] and consumed by the §K.1 Gaussian splat.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArcSample {
    pub point: Point,
    /// The listing's `.arclength` field (`d`) — a per-sample brightness
    /// weight, `1.0` except for the trailing partial sample.
    pub d: f32,
}

/// FDIS §K.3, Listing K.1 (middle block) — resample the upsampled
/// polyline into points spaced one unit apart along the arc length.
///
/// The first sample is the polyline's start (weight `1`). Thereafter the
/// walker accumulates segment lengths between consecutive upsampled
/// points; whenever the running length reaches `1` it interpolates a new
/// sample exactly one unit from the previous emitted sample (weight `1`).
/// When the polyline is exhausted mid-unit the leftover length is emitted
/// as the final sample's weight `d`.
pub fn resample_by_arclength(upsampled: &[Point]) -> Vec<ArcSample> {
    if upsampled.is_empty() {
        return Vec::new();
    }
    let mut all_samples = Vec::new();
    let mut current = upsampled[0];
    all_samples.push(ArcSample {
        point: current,
        d: 1.0,
    });
    let mut next = 0usize;
    while next < upsampled.len() {
        let mut previous = current;
        let mut arclength = 0.0f32;
        loop {
            if next == upsampled.len() {
                all_samples.push(ArcSample {
                    point: previous,
                    d: arclength,
                });
                break;
            }
            let d = upsampled[next].sub(previous);
            let arclength_to_next = (d.x * d.x + d.y * d.y).sqrt();
            if arclength + arclength_to_next >= 1.0 {
                let s = (1.0 - arclength) / arclength_to_next;
                current = previous.lerp(upsampled[next], s);
                all_samples.push(ArcSample {
                    point: current,
                    d: 1.0,
                });
                break;
            }
            arclength += arclength_to_next;
            previous = upsampled[next];
            next += 1;
        }
    }
    all_samples
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

/// Gauss error function, `erf(x)`.
///
/// The §K.1 Gaussian brush integrates the 2-D Gaussian over each pixel
/// cell as a product of `erf` differences. This is a standard rational
/// approximation of `erf` (maximum absolute error ≈ 1.5e-7), evaluated in
/// `f64` for headroom and returned as `f32`. It is a general numerical
/// identity for the error function, not derived from any codec source.
pub fn erf(x: f32) -> f32 {
    erf_f64(x as f64) as f32
}

fn erf_f64(x: f64) -> f64 {
    const P: f64 = 0.327_591_1;
    const A: [f64; 5] = [
        0.254_829_592,
        -0.284_496_736,
        1.421_413_741,
        -1.453_152_027,
        1.061_405_429,
    ];
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + P * ax);
    let poly = ((((A[4] * t + A[3]) * t + A[2]) * t + A[1]) * t + A[0]) * t;
    sign * (1.0 - poly * (-ax * ax).exp())
}

/// One decoded spline, ready to render: its control points plus the
/// dequantized and recorrelated DCT32 coefficients of the X, Y, B and σ
/// channels (§C.4.6). The coefficients are already through
/// [`dequant_dct32`] + [`recorrelate_xb`]; rendering only evaluates them.
#[derive(Debug, Clone)]
pub struct Spline {
    pub control_points: Vec<Point>,
    pub dct_x: [f32; SPLINE_DCT_LEN],
    pub dct_y: [f32; SPLINE_DCT_LEN],
    pub dct_b: [f32; SPLINE_DCT_LEN],
    pub dct_sigma: [f32; SPLINE_DCT_LEN],
}

/// `-2 × ln(0.1)` — the constant factor in the §K.1 `maximum_distance`
/// bound (`maximum_distance = -2 × log(0.1) × σ²`, natural log).
const NEG_TWO_LN_TENTH: f32 = 4.605_170_2;

impl Spline {
    /// FDIS §K.3, Listing K.1 (final block) — render this spline onto the
    /// three XYB planes by additive Gaussian splatting along its arc
    /// length.
    ///
    /// `x_plane` / `y_plane` / `b_plane` are row-major `width × height`
    /// buffers (the XYB frame planes); the spline's contribution is
    /// **added** to them. Coordinates outside a plane are clipped (the
    /// listing clamps `xbegin/xend/ybegin/yend` to the image bounds).
    ///
    /// ## Errata note (DCT32 arc parameter)
    ///
    /// Listing K.1 writes the `ContinuousIDCT` argument as
    /// `31 × arclength_from_start / arclength`. With
    /// `arclength_from_start = min(1, i / arclength) ∈ [0, 1]`, the extra
    /// `/ arclength` collapses the parameter to `≪ 1` for any spline
    /// longer than a few pixels, so every channel would evaluate to its DC
    /// coefficient alone — defeating the 32 encoded DCT coefficients per
    /// channel. The internally-consistent reading (used here) is
    /// `t = 31 × arclength_from_start`, mapping the normalized arc position
    /// onto the DCT32 domain `[0, 31]`. Recorded as a suspected FDIS typo.
    pub fn render(
        &self,
        x_plane: &mut [f32],
        y_plane: &mut [f32],
        b_plane: &mut [f32],
        width: usize,
        height: usize,
    ) -> Result<()> {
        let expected = width
            .checked_mul(height)
            .ok_or_else(|| Error::InvalidData("JXL splines: width × height overflow".into()))?;
        for (name, plane) in [("X", &*x_plane), ("Y", &*y_plane), ("B", &*b_plane)] {
            if plane.len() != expected {
                return Err(Error::InvalidData(format!(
                    "JXL splines: {name} plane length {} != width×height {expected}",
                    plane.len()
                )));
            }
        }
        if width == 0 || height == 0 {
            return Ok(());
        }

        let upsampled = upsample_control_points(&self.control_points);
        let samples = resample_by_arclength(&upsampled);
        if samples.len() < 2 {
            // A degenerate (single-point) spline has zero arc length and
            // draws nothing (Listing K.1: arclength = size − 2 + back.d).
            return Ok(());
        }
        let total = samples.len() as f32 - 2.0 + samples.last().unwrap().d;
        if !total.is_finite() || total <= 0.0 {
            return Ok(());
        }

        const SQRT2: f32 = std::f32::consts::SQRT_2;
        let w_i = width as i64;
        let h_i = height as i64;

        for (i, sample) in samples.iter().enumerate() {
            let arclength_from_start = (i as f32 / total).min(1.0);
            // Errata-corrected DCT32 arc parameter (see doc comment).
            let t = 31.0 * arclength_from_start;

            let sigma = continuous_idct(&self.dct_sigma, t);
            let s2s = SQRT2 * sigma;
            if !s2s.is_finite() || s2s.abs() < 1e-20 {
                // Zero/degenerate thickness → no visible brush.
                continue;
            }
            let maximum_distance = NEG_TWO_LN_TENTH * sigma * sigma;
            if !maximum_distance.is_finite() {
                continue;
            }

            let p = sample.point;
            let d = sample.d;

            let xbegin = ((p.x - maximum_distance + 0.5).floor() as i64).max(0);
            let xend = ((p.x + maximum_distance + 0.5).floor() as i64).min(w_i - 1);
            let ybegin = ((p.y - maximum_distance + 0.5).floor() as i64).max(0);
            let yend = ((p.y + maximum_distance + 0.5).floor() as i64).min(h_i - 1);
            if xend < xbegin || yend < ybegin {
                continue;
            }

            let vx = continuous_idct(&self.dct_x, t);
            let vy = continuous_idct(&self.dct_y, t);
            let vb = continuous_idct(&self.dct_b, t);
            // The geometric erf factor is identical across the three
            // channels (the listing's per-channel `× (erf..) × (erf..)`);
            // the per-sample `d × σ / 4` scale is likewise shared.
            let scale = d * sigma / 4.0;

            for x in xbegin..=xend {
                let xf = x as f32;
                let ex = erf((p.x - xf + 0.5) / s2s) - erf((p.x - xf - 0.5) / s2s);
                if ex == 0.0 {
                    continue;
                }
                for y in ybegin..=yend {
                    let yf = y as f32;
                    let ey = erf((p.y - yf + 0.5) / s2s) - erf((p.y - yf - 0.5) / s2s);
                    let g = scale * ex * ey;
                    let idx = (y as usize) * width + (x as usize);
                    x_plane[idx] += vx * g;
                    y_plane[idx] += vy * g;
                    b_plane[idx] += vb * g;
                }
            }
        }
        Ok(())
    }
}

/// Number of clustered distributions (contexts) in the §C.4.6 spline ANS
/// stream — one per `ReadHybridVarLenUint` context index 0..6.
pub const SPLINE_NUM_CONTEXTS: usize = 6;

/// FDIS §C.4.6 — decode the spline dictionary from an abstract
/// `ReadHybridVarLenUint(ctx)` source.
///
/// This carries the full Listing C.3 / Listing C.4 structure independent
/// of the entropy back-end: `read_uint(ctx)` returns the next hybrid
/// var-len uint for context `ctx`. [`decode_splines`] wraps it with the
/// real §D.3 [`crate::modular_fdis::EntropyStream`]; tests drive it with a
/// scripted token source.
///
/// Context assignment (per Listing C.3 and the per-spline steps):
/// `2` num_splines, `0` quant_adjust, `1` start coords, `3`
/// num_control_points, `4` control-point deltas, `5` DCT32 coefficients.
pub fn decode_splines_with<F>(
    mut read_uint: F,
    base_correlation_x: f32,
    base_correlation_b: f32,
) -> Result<Vec<Spline>>
where
    F: FnMut(u32) -> Result<u32>,
{
    // num_splines = ReadHybridVarLenUint(2) + 1;
    let num_splines = read_uint(2)?
        .checked_add(1)
        .ok_or_else(|| Error::InvalidData("JXL splines: num_splines overflow".into()))?
        as usize;
    // quant_adjust = UnpackSigned(ReadHybridVarLenUint(0));
    let quant_adjust = unpack_signed(read_uint(0)?);

    // Listing C.3 — starting coordinates (delta-coded after the first).
    let mut start = Vec::with_capacity(num_splines.min(1024));
    let (mut last_x, mut last_y): (i64, i64) = (0, 0);
    for i in 0..num_splines {
        let rx = read_uint(1)?;
        let ry = read_uint(1)?;
        let (x, y) = if i == 0 {
            (rx as i64, ry as i64)
        } else {
            (
                unpack_signed(rx) as i64 + last_x,
                unpack_signed(ry) as i64 + last_y,
            )
        };
        start.push((x, y));
        last_x = x;
        last_y = y;
    }

    // Per-spline: control points (double-delta) + 4×32 DCT coefficients.
    let mut out = Vec::with_capacity(num_splines.min(1024));
    for &(sp_x, sp_y) in &start {
        // num_control_points = 1 + ReadHybridVarLenUint(3);
        let num_cp = (read_uint(3)? as usize)
            .checked_add(1)
            .ok_or_else(|| Error::InvalidData("JXL splines: num_control_points overflow".into()))?;
        // Interleaved (x1, y1, x2, y2, …) second-order deltas.
        let mut dx = Vec::with_capacity((num_cp - 1).min(1 << 20));
        let mut dy = Vec::with_capacity((num_cp - 1).min(1 << 20));
        for _ in 0..num_cp - 1 {
            dx.push(unpack_signed(read_uint(4)?) as i64);
            dy.push(unpack_signed(read_uint(4)?) as i64);
        }
        let cx = decode_double_delta(sp_x, &dx);
        let cy = decode_double_delta(sp_y, &dy);
        let control_points: Vec<Point> = cx
            .iter()
            .zip(cy.iter())
            .map(|(&x, &y)| Point::new(x as f32, y as f32))
            .collect();

        // Four channels of 32 quantized DCT coefficients: X, Y, B, σ.
        let mut raw = [[0i32; SPLINE_DCT_LEN]; 4];
        for chan in raw.iter_mut() {
            for c in chan.iter_mut() {
                *c = unpack_signed(read_uint(5)?);
            }
        }
        let mut dct_x = dequant_dct32(&raw[0], quant_adjust, 0).unwrap();
        let dct_y = dequant_dct32(&raw[1], quant_adjust, 1).unwrap();
        let mut dct_b = dequant_dct32(&raw[2], quant_adjust, 2).unwrap();
        let dct_sigma = dequant_dct32(&raw[3], quant_adjust, 3).unwrap();
        recorrelate_xb(
            &mut dct_x,
            &mut dct_b,
            &dct_y,
            base_correlation_x,
            base_correlation_b,
        );

        out.push(Spline {
            control_points,
            dct_x,
            dct_y,
            dct_b,
            dct_sigma,
        });
    }
    Ok(out)
}

/// FDIS §C.4.6 — decode the spline dictionary from the codestream.
///
/// Reads the §D.3 six-distribution ANS prelude, its ANS-state init
/// (§C.3.2), then drives [`decode_splines_with`] against the resulting
/// entropy stream. `base_correlation_{x,b}` come from the frame's
/// LfChannelCorrelation bundle (§C.4.4).
pub fn decode_splines(
    br: &mut crate::bitreader::BitReader<'_>,
    base_correlation_x: f32,
    base_correlation_b: f32,
) -> Result<Vec<Spline>> {
    use crate::modular_fdis::{decode_uint_in_with_dist_pub, EntropyStream};
    let mut entropy = EntropyStream::read(br, SPLINE_NUM_CONTEXTS)?;
    entropy.read_ans_state_init(br)?;
    let mut hybrid = crate::ans::hybrid::HybridUintState::new(entropy.lz77, entropy.lz_len_conf);
    decode_splines_with(
        |ctx| decode_uint_in_with_dist_pub(&mut hybrid, &mut entropy, br, ctx, 0),
        base_correlation_x,
        base_correlation_b,
    )
}

/// FDIS §K.3 — render every spline in draw order onto the XYB planes
/// (additive; patches have already been drawn, noise follows).
pub fn render_splines(
    splines: &[Spline],
    x_plane: &mut [f32],
    y_plane: &mut [f32],
    b_plane: &mut [f32],
    width: usize,
    height: usize,
) -> Result<()> {
    for spline in splines {
        spline.render(x_plane, y_plane, b_plane, width, height)?;
    }
    Ok(())
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
    fn upsample_single_point_is_itself() {
        let out = upsample_control_points(&[Point::new(3.0, 4.0)]);
        assert_eq!(out, vec![Point::new(3.0, 4.0)]);
    }

    #[test]
    fn upsample_two_points_endpoints_and_length() {
        // Two control points: extended has 4 entries, one window (i=0)
        // emits p[1] then 15 interior sub-steps, then S.back() is
        // appended → 1 + 15 + 1 = 17 points, starting at S[0], ending at
        // S[1].
        let s0 = Point::new(0.0, 0.0);
        let s1 = Point::new(16.0, 0.0);
        let out = upsample_control_points(&[s0, s1]);
        assert_eq!(out.len(), 17);
        assert_eq!(out[0], s0);
        assert_eq!(*out.last().unwrap(), s1);
    }

    #[test]
    fn upsample_collinear_stays_on_the_line() {
        // Equally-spaced collinear control points on y = 5: the
        // centripetal interpolation must keep every upsampled sample on
        // that line, with monotone non-decreasing x.
        let pts: Vec<Point> = (0..4).map(|i| Point::new(i as f32 * 4.0, 5.0)).collect();
        let out = upsample_control_points(&pts);
        assert!(out.len() > pts.len());
        for w in out.windows(2) {
            assert!(
                (w[0].y - 5.0).abs() < 1e-3,
                "sample left the line: y = {}",
                w[0].y
            );
            assert!(
                w[1].x + 1e-3 >= w[0].x,
                "x must be monotone non-decreasing: {} then {}",
                w[0].x,
                w[1].x
            );
        }
        assert!((out.last().unwrap().y - 5.0).abs() < 1e-3);
    }

    #[test]
    fn upsample_empty_is_empty() {
        assert!(upsample_control_points(&[]).is_empty());
    }

    #[test]
    fn resample_empty_is_empty() {
        assert!(resample_by_arclength(&[]).is_empty());
    }

    #[test]
    fn resample_horizontal_line_is_unit_spaced() {
        // A dense horizontal polyline from x=0..=10 (0.25 spacing).
        let up: Vec<Point> = (0..=40).map(|i| Point::new(i as f32 * 0.25, 0.0)).collect();
        let s = resample_by_arclength(&up);
        // First sample is the start with weight 1.
        assert_eq!(s[0].point, Point::new(0.0, 0.0));
        assert_eq!(s[0].d, 1.0);
        // Consecutive emitted samples are ~1 unit apart in x, y unchanged.
        for w in s.windows(2) {
            assert!((w[0].point.y).abs() < 1e-4);
            let dx = w[1].point.x - w[0].point.x;
            // Interior gaps are ~1; the final (partial/zero-length) gap
            // may be anywhere in [0, 1].
            assert!((0.0..=1.0 + 1e-4).contains(&dx), "gap {dx}");
        }
        // Interior samples carry weight 1; only the trailing sample may be
        // fractional.
        for interior in &s[..s.len() - 1] {
            assert_eq!(interior.d, 1.0);
        }
        assert!(s.last().unwrap().d <= 1.0);
    }

    #[test]
    fn resample_endpoint_weight_is_fractional_remainder() {
        // Total length 2.5 → samples at arclength 0,1,2 (weight 1) and a
        // trailing 0.5 remainder.
        let up = vec![Point::new(0.0, 0.0), Point::new(2.5, 0.0)];
        let s = resample_by_arclength(&up);
        assert!(
            (s.last().unwrap().d - 0.5).abs() < 1e-4,
            "d = {}",
            s.last().unwrap().d
        );
    }

    #[test]
    fn erf_known_values() {
        assert!(erf(0.0).abs() < 1e-6);
        assert!((erf(0.5) - 0.520_499_9).abs() < 1e-4);
        assert!((erf(1.0) - 0.842_700_8).abs() < 1e-4);
        assert!((erf(3.0) - 1.0).abs() < 1e-4);
        // Odd symmetry.
        assert!((erf(-0.7) + erf(0.7)).abs() < 1e-6);
    }

    /// A dequantized Y-only DC spline: `continuous_idct` returns the DC
    /// coefficient for every arc position.
    fn dc_only(dc: f32) -> [f32; SPLINE_DCT_LEN] {
        let mut c = [0.0f32; SPLINE_DCT_LEN];
        c[0] = dc;
        c
    }

    #[test]
    fn render_horizontal_streak_is_symmetric_and_localised() {
        let w = 48usize;
        let h = 48usize;
        let mut x = vec![0.0f32; w * h];
        let mut y = vec![0.0f32; w * h];
        let mut b = vec![0.0f32; w * h];
        // A short horizontal spline at y = 24, Y-channel only, σ ≈ 1.5.
        let spline = Spline {
            control_points: vec![Point::new(12.0, 24.0), Point::new(36.0, 24.0)],
            dct_x: [0.0; SPLINE_DCT_LEN],
            dct_y: dc_only(10.0),
            dct_b: [0.0; SPLINE_DCT_LEN],
            dct_sigma: dc_only(1.5),
        };
        spline.render(&mut x, &mut y, &mut b, w, h).unwrap();

        // X and B planes are untouched (their DCT is zero).
        assert!(x.iter().all(|&v| v == 0.0));
        assert!(b.iter().all(|&v| v == 0.0));

        let at = |px: usize, py: usize| y[py * w + px];
        // The streak is present near the centre line.
        assert!(at(24, 24) > 0.0, "centre of streak must be positive");
        // Vertically symmetric about y = 24 for a mid-line column.
        assert!((at(24, 23) - at(24, 25)).abs() < 1e-3);
        assert!((at(24, 22) - at(24, 26)).abs() < 1e-3);
        // Energy decays away from the line (peak on the line).
        assert!(at(24, 24) > at(24, 27));
        // Far corner is essentially untouched.
        assert!(at(2, 2).abs() < 1e-3);
    }

    #[test]
    fn render_rejects_wrong_plane_length() {
        let spline = Spline {
            control_points: vec![Point::new(0.0, 0.0), Point::new(4.0, 0.0)],
            dct_x: [0.0; SPLINE_DCT_LEN],
            dct_y: dc_only(1.0),
            dct_b: [0.0; SPLINE_DCT_LEN],
            dct_sigma: dc_only(1.0),
        };
        let mut x = vec![0.0f32; 10];
        let mut good = vec![0.0f32; 16];
        assert!(spline
            .render(&mut x, &mut good.clone(), &mut good, 4, 4)
            .is_err());
    }

    #[test]
    fn render_single_control_point_draws_nothing() {
        let spline = Spline {
            control_points: vec![Point::new(8.0, 8.0)],
            dct_x: [0.0; SPLINE_DCT_LEN],
            dct_y: dc_only(10.0),
            dct_b: [0.0; SPLINE_DCT_LEN],
            dct_sigma: dc_only(2.0),
        };
        let mut x = vec![0.0f32; 256];
        let mut y = vec![0.0f32; 256];
        let mut b = vec![0.0f32; 256];
        spline.render(&mut x, &mut y, &mut b, 16, 16).unwrap();
        assert!(y.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn render_splines_batch_is_additive() {
        let w = 32usize;
        let h = 32usize;
        let spline = Spline {
            control_points: vec![Point::new(6.0, 16.0), Point::new(26.0, 16.0)],
            dct_x: [0.0; SPLINE_DCT_LEN],
            dct_y: dc_only(5.0),
            dct_b: [0.0; SPLINE_DCT_LEN],
            dct_sigma: dc_only(1.2),
        };
        let mut y1 = vec![0.0f32; w * h];
        let (mut zx, mut zb) = (vec![0.0f32; w * h], vec![0.0f32; w * h]);
        spline
            .render(&mut zx.clone(), &mut y1, &mut zb.clone(), w, h)
            .unwrap();

        // Two copies via render_splines double every sample.
        let mut y2 = vec![0.0f32; w * h];
        render_splines(
            &[spline.clone(), spline.clone()],
            &mut zx,
            &mut y2,
            &mut zb,
            w,
            h,
        )
        .unwrap();
        for (a, c) in y1.iter().zip(y2.iter()) {
            assert!((2.0 * a - c).abs() < 1e-4);
        }
    }

    #[test]
    fn decode_splines_with_parses_one_spline() {
        // Scripted ReadHybridVarLenUint returns; also records the ctx
        // sequence so the parse's context routing is verified.
        // ctx2 num_splines-1=0; ctx0 quant_adjust=0; ctx1 sp_x[0]=10,
        // sp_y[0]=24; ctx3 num_control_points-1=1; ctx4 x1 raw
        // (UnpackSigned=20), y1 raw (0).
        let mut script: Vec<u32> = vec![0, 0, 10, 24, 1, 40, 0];
        // X channel: 32 zeros.
        script.extend(std::iter::repeat_n(0, 32));
        // Y channel: index 0 = UnpackSigned(200) = 100, rest 0.
        script.push(200);
        script.extend(std::iter::repeat_n(0, 31));
        // B channel: 32 zeros.
        script.extend(std::iter::repeat_n(0, 32));
        // σ channel: index 0 = UnpackSigned(40) = 20, rest 0.
        script.push(40);
        script.extend(std::iter::repeat_n(0, 31));

        let expected_ctx_prefix = [2u32, 0, 1, 1, 3, 4, 4];

        let mut idx = 0usize;
        let mut ctxs: Vec<u32> = Vec::new();
        let splines = decode_splines_with(
            |ctx| {
                ctxs.push(ctx);
                let v = script[idx];
                idx += 1;
                Ok(v)
            },
            0.0,
            1.0,
        )
        .unwrap();

        // All scripted tokens consumed, correct context routing.
        assert_eq!(idx, script.len());
        assert_eq!(&ctxs[..expected_ctx_prefix.len()], &expected_ctx_prefix);
        assert!(ctxs[expected_ctx_prefix.len()..].iter().all(|&c| c == 5));

        assert_eq!(splines.len(), 1);
        let s = &splines[0];
        assert_eq!(
            s.control_points,
            vec![Point::new(10.0, 24.0), Point::new(30.0, 24.0)]
        );
        // qa = 0 → divisor 1. Y DC = 100 × 0.075 = 7.5.
        assert!((s.dct_y[0] - 7.5).abs() < 1e-4);
        // σ DC = 20 × 0.3333.
        assert!((s.dct_sigma[0] - 20.0 * 0.3333).abs() < 1e-4);
        // X DC = 0 + base_correlation_x(0) × Y = 0.
        assert_eq!(s.dct_x[0], 0.0);
        // B DC = 0 + base_correlation_b(1) × Y = 7.5.
        assert!((s.dct_b[0] - 7.5).abs() < 1e-4);
    }

    #[test]
    fn decode_splines_with_delta_coded_start_points() {
        // Two splines: second start point is delta-coded off the first.
        // sp[0] = (5, 5); sp[1] = (5 + UnpackSigned(4), 5 + UnpackSigned(2))
        //       = (5 + 2, 5 + 1) = (7, 6).
        let mut script: Vec<u32> = vec![
            1, // ctx2: num_splines - 1 = 1 → 2 splines
            0, // ctx0: quant_adjust = 0
            5, // ctx1: sp_x[0] = 5 (literal)
            5, // ctx1: sp_y[0] = 5
            4, // ctx1: sp_x[1] raw → UnpackSigned(4) = 2 → 7
            2, // ctx1: sp_y[1] raw → UnpackSigned(2) = 1 → 6
        ];
        // Spline 0: 1 control point (num_cp - 1 = 0), then 4×32 coeffs.
        script.push(0); // ctx3
        script.extend(std::iter::repeat_n(0, 128));
        // Spline 1: 1 control point, then 4×32 coeffs.
        script.push(0); // ctx3
        script.extend(std::iter::repeat_n(0, 128));

        let mut idx = 0usize;
        let splines = decode_splines_with(
            |_ctx| {
                let v = script[idx];
                idx += 1;
                Ok(v)
            },
            0.0,
            1.0,
        )
        .unwrap();
        assert_eq!(idx, script.len());
        assert_eq!(splines.len(), 2);
        assert_eq!(splines[0].control_points, vec![Point::new(5.0, 5.0)]);
        assert_eq!(splines[1].control_points, vec![Point::new(7.0, 6.0)]);
    }

    #[test]
    fn decode_splines_propagates_read_error() {
        // A read that fails partway is surfaced, not swallowed.
        let err = decode_splines_with(|_ctx| Err(Error::InvalidData("boom".into())), 0.0, 1.0);
        assert!(err.is_err());
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
