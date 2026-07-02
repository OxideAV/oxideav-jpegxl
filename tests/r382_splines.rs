//! End-to-end Splines image-feature pipeline — ISO/IEC FDIS
//! 18181-1:2021 §C.4.6 (decode) + §K.3 (render).
//!
//! Exercises the public `oxideav_jpegxl::splines` API as a whole: decode
//! a spline dictionary from a scripted `ReadHybridVarLenUint` token
//! source (the entropy back-end is covered by the module's own unit
//! tests) and render the resulting splines onto XYB planes, asserting the
//! drawn geometry. No external decoder source is consulted; the token
//! script is built by hand from the Listing C.3 structure.

use oxideav_jpegxl::splines::{decode_splines_with, render_splines, Point, Spline, SPLINE_DCT_LEN};

/// Build the token script for a single spline with two control points and
/// a chosen (Y, σ) DC value, everything else zero. Mirrors the reader's
/// context order: ctx2, ctx0, ctx1×2, ctx3, ctx4×2, ctx5×128.
fn script_one_spline(
    sp_x: u32,
    sp_y: u32,
    dx_raw: u32,
    dy_raw: u32,
    y_dc_raw: u32,
    sigma_dc_raw: u32,
) -> Vec<u32> {
    let mut s = vec![
        0,      // ctx2: num_splines - 1
        0,      // ctx0: quant_adjust (UnpackSigned → 0)
        sp_x,   // ctx1: sp_x[0]
        sp_y,   // ctx1: sp_y[0]
        1,      // ctx3: num_control_points - 1 = 1
        dx_raw, // ctx4: x1 delta raw
        dy_raw, // ctx4: y1 delta raw
    ];
    // X: 32 zeros.
    s.resize(s.len() + SPLINE_DCT_LEN, 0);
    // Y: DC then 31 zeros.
    s.push(y_dc_raw);
    s.resize(s.len() + SPLINE_DCT_LEN - 1, 0);
    // B: 32 zeros.
    s.resize(s.len() + SPLINE_DCT_LEN, 0);
    // σ: DC then 31 zeros.
    s.push(sigma_dc_raw);
    s.resize(s.len() + SPLINE_DCT_LEN - 1, 0);
    s
}

fn decode(script: &[u32], bcx: f32, bcb: f32) -> Vec<Spline> {
    let mut idx = 0usize;
    let out = decode_splines_with(
        |_ctx| {
            let v = script[idx];
            idx += 1;
            Ok(v)
        },
        bcx,
        bcb,
    )
    .expect("scripted spline decode must succeed");
    assert_eq!(idx, script.len(), "every scripted token must be consumed");
    out
}

#[test]
fn decode_then_render_draws_a_localised_streak() {
    // A horizontal spline from (12, 24) to (12 + UnpackSigned(40)=32, 24),
    // Y DC = UnpackSigned(200) = 100 → 100 × 0.075 = 7.5, σ DC =
    // UnpackSigned(8) = 4 → 4 × 0.3333 ≈ 1.33 (brush radius ≈ 8 px, so the
    // contribution stays local).
    let script = script_one_spline(12, 24, 40, 0, 200, 8);
    let splines = decode(&script, 0.0, 1.0);
    assert_eq!(splines.len(), 1);
    assert_eq!(
        splines[0].control_points,
        vec![Point::new(12.0, 24.0), Point::new(32.0, 24.0)]
    );

    let (w, h) = (48usize, 48usize);
    let mut x = vec![0.0f32; w * h];
    let mut y = vec![0.0f32; w * h];
    let mut b = vec![0.0f32; w * h];
    render_splines(&splines, &mut x, &mut y, &mut b, w, h).unwrap();

    let at = |plane: &[f32], px: usize, py: usize| plane[py * w + px];
    // The streak lifts the Y plane along the centre line.
    assert!(at(&y, 22, 24) > 0.0);
    // Base-correlation-b = 1.0 copies Y into B; X (bcx = 0) stays flat.
    assert!((at(&b, 22, 24) - at(&y, 22, 24)).abs() < 1e-4);
    assert!(x.iter().all(|&v| v == 0.0));
    // The contribution is localised: a far corner is untouched.
    assert!(at(&y, 2, 2).abs() < 1e-3);
}

#[test]
fn base_correlation_x_bleeds_luma_into_chroma() {
    // With base_correlation_x = 0.5 the X plane picks up half the Y DC.
    let script = script_one_spline(12, 24, 40, 0, 200, 8);
    let splines = decode(&script, 0.5, 1.0);
    // X DC = 0 + 0.5 × (100 × 0.075) = 3.75; Y DC = 7.5.
    assert!((splines[0].dct_x[0] - 3.75).abs() < 1e-4);
    assert!((splines[0].dct_y[0] - 7.5).abs() < 1e-4);

    let (w, h) = (48usize, 48usize);
    let mut x = vec![0.0f32; w * h];
    let mut y = vec![0.0f32; w * h];
    let mut b = vec![0.0f32; w * h];
    render_splines(&splines, &mut x, &mut y, &mut b, w, h).unwrap();
    let idx = 24 * w + 22;
    // X is exactly half of Y everywhere the brush touches (0.5 ratio).
    assert!(y[idx] > 0.0);
    assert!((x[idx] - 0.5 * y[idx]).abs() < 1e-4);
}

#[test]
fn empty_spline_set_leaves_planes_untouched() {
    let (w, h) = (16usize, 16usize);
    let mut x = vec![1.0f32; w * h];
    let mut y = vec![2.0f32; w * h];
    let mut b = vec![3.0f32; w * h];
    render_splines(&[], &mut x, &mut y, &mut b, w, h).unwrap();
    assert!(x.iter().all(|&v| v == 1.0));
    assert!(y.iter().all(|&v| v == 2.0));
    assert!(b.iter().all(|&v| v == 3.0));
}
