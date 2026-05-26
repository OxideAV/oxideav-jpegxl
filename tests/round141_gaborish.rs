//! Round-141 integration tests: Annex J.2 "Gabor-like transform"
//! pure-math primitive (`oxideav_jpegxl::gaborish`).
//!
//! Pins the cross-module composition of:
//!
//! * the round-1+ [`oxideav_jpegxl::frame_header::RestorationFilter`]
//!   bundle (Table C.9) — the per-frame Gaborish weights,
//! * the FDIS §J.2 normative 3×3 symmetric-convolution kernel
//!   `(centre = 1, edges = w1, corners = w2)` normalised to sum 1,
//! * the §6.5 `Mirror1D` boundary handling
//!
//! into a single per-channel bit-exact float pipeline that
//! downstream restoration-filter-wiring rounds can drop in without
//! re-deriving any of the §J.2 formulae.
//!
//! Round 141 does **not** drive this from a real frame — that wiring
//! is one or more follow-up rounds; the §I.2.5 LLF/HF reconstruction
//! and Annex G chroma-from-luma chain need to be composed first.
//! These tests pin the pure-math step at the formula level.

use oxideav_jpegxl::frame_header::RestorationFilter;
use oxideav_jpegxl::gaborish::{
    apply_channel, apply_channel_in_place, apply_xyb_planes_in_place, gab_kernel, mirror1d,
    sample_mirror,
};

/// FDIS §J.2 — at the default-bundle Gaborish weights (Table C.9
/// defaults `gab_C_weight1 = 0.115_169_525`,
/// `gab_C_weight2 = 0.061_248_592` for every channel), the kernel
/// sums to 1.0 within f32 round-off.
#[test]
fn j2_default_kernel_sums_to_one() {
    let rf = RestorationFilter::default();
    let k = gab_kernel(rf.gab_x_weight1, rf.gab_x_weight2).unwrap();
    let s: f32 = k.iter().sum();
    assert!(
        (s - 1.0).abs() < 1e-6,
        "default-X kernel sum {s} differs from 1.0"
    );
}

/// FDIS §J.2 — a constant-value plane is invariant under any
/// unit-sum kernel; pin this for a 9×9 plane filtered with the
/// default weights.
#[test]
fn j2_constant_plane_invariant_under_default_kernel() {
    let rf = RestorationFilter::default();
    let w = 9_usize;
    let h = 9_usize;
    let input = vec![2.0_f32; w * h];
    let mut output = vec![0.0_f32; w * h];
    apply_channel(
        &input,
        &mut output,
        w,
        h,
        rf.gab_x_weight1,
        rf.gab_x_weight2,
    )
    .unwrap();
    for &v in &output {
        assert!((v - 2.0).abs() < 1e-5);
    }
}

/// FDIS §J.2 — the convolution is a linear operator: the filtered
/// plane of `(a · input1 + b · input2)` equals
/// `a · filter(input1) + b · filter(input2)`. Pin this on a 4×4
/// plane with the default weights.
#[test]
fn j2_convolution_is_linear() {
    let rf = RestorationFilter::default();
    let w = 4_usize;
    let h = 4_usize;
    let p1: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let p2: Vec<f32> = (0..16).map(|v| (v as f32) * 0.3 - 1.0).collect();
    let a = 0.7_f32;
    let b = -1.3_f32;
    let mixed: Vec<f32> = p1.iter().zip(&p2).map(|(x, y)| a * x + b * y).collect();

    let mut f1 = vec![0.0_f32; 16];
    let mut f2 = vec![0.0_f32; 16];
    let mut fmix = vec![0.0_f32; 16];
    apply_channel(&p1, &mut f1, w, h, rf.gab_x_weight1, rf.gab_x_weight2).unwrap();
    apply_channel(&p2, &mut f2, w, h, rf.gab_x_weight1, rf.gab_x_weight2).unwrap();
    apply_channel(&mixed, &mut fmix, w, h, rf.gab_x_weight1, rf.gab_x_weight2).unwrap();

    for i in 0..16 {
        let expected = a * f1[i] + b * f2[i];
        assert!(
            (fmix[i] - expected).abs() < 1e-4,
            "linearity violated at sample {i}: filtered(mix)={} a·f1+b·f2={}",
            fmix[i],
            expected
        );
    }
}

/// FDIS §J.2 — the §6.5 Mirror1D semantics make a horizontally-
/// uniform plane invariant under the convolution along that axis.
/// Pin this for a `width=5`, `height=1` plane: each row sample is
/// `v[x]`, and the kernel reduces to `(2·w2 + w1)·v[x-1] + (1 +
/// 2·w1)·v[x] + (2·w2 + w1)·v[x+1]` (the vertical references all
/// mirror back to row 0). For a constant row this is just the
/// original value.
#[test]
fn j2_single_row_constant_is_invariant() {
    let rf = RestorationFilter::default();
    let input = vec![3.5_f32; 5];
    let mut output = vec![0.0_f32; 5];
    apply_channel(
        &input,
        &mut output,
        5,
        1,
        rf.gab_x_weight1,
        rf.gab_x_weight2,
    )
    .unwrap();
    for &v in &output {
        assert!((v - 3.5).abs() < 1e-5);
    }
}

/// FDIS §J.2 — `rf.gab` boolean controls whether the filter runs,
/// but this module unconditionally applies the math on the inputs
/// it is given (the caller honours the skip). The default
/// `RestorationFilter::default()` sets `gab = true`, so a real
/// pipeline call always runs the convolution.
#[test]
fn j2_default_rf_has_gab_enabled() {
    let rf = RestorationFilter::default();
    assert!(rf.gab);
}

/// FDIS §J.2 — `apply_xyb_planes_in_place` runs the per-channel
/// convolution on all three XYB planes, using the corresponding
/// `gab_x_*` / `gab_y_*` / `gab_b_*` weight pair. Pin this with
/// distinct per-channel weights (so an accidental channel-swap
/// would diverge).
#[test]
fn j2_apply_xyb_dispatches_per_channel_weights() {
    let rf = RestorationFilter {
        gab_x_weight1: 0.10,
        gab_x_weight2: 0.05,
        gab_y_weight1: 0.20,
        gab_y_weight2: 0.10,
        gab_b_weight1: 0.30,
        gab_b_weight2: 0.15,
        ..RestorationFilter::default()
    };

    let w = 5_usize;
    let h = 3_usize;
    let plane: Vec<f32> = (0..w * h).map(|v| v as f32).collect();
    let mut x = plane.clone();
    let mut y = plane.clone();
    let mut b = plane.clone();
    apply_xyb_planes_in_place(&mut x, &mut y, &mut b, w, h, &rf).unwrap();

    let mut x_ref = plane.clone();
    let mut y_ref = plane.clone();
    let mut b_ref = plane.clone();
    apply_channel_in_place(&mut x_ref, w, h, 0.10, 0.05).unwrap();
    apply_channel_in_place(&mut y_ref, w, h, 0.20, 0.10).unwrap();
    apply_channel_in_place(&mut b_ref, w, h, 0.30, 0.15).unwrap();
    assert_eq!(x, x_ref);
    assert_eq!(y, y_ref);
    assert_eq!(b, b_ref);
}

/// FDIS §6.5 — Mirror1D bottoms out in at most one reflection for
/// any `coord ∈ {-1, x, size}` with `x ∈ 0..size`, which is the
/// only range Gaborish exercises (a 3×3 kernel touches at most one
/// row/column outside the plane on each side).
#[test]
fn j2_mirror1d_only_one_reflection_needed_for_gaborish() {
    for size in 1..32_usize {
        // coord == -1 → 0
        assert_eq!(mirror1d(-1, size).unwrap(), 0);
        // coord == size → size - 1
        assert_eq!(mirror1d(size as i64, size).unwrap(), size - 1);
        // and every in-bounds coord is identity
        for coord in 0..size as i64 {
            assert_eq!(mirror1d(coord, size).unwrap(), coord as usize);
        }
    }
}

/// FDIS §J.2 — the convolution is genuinely a low-pass at the
/// default weights: the centre tap (≈ 0.586) dominates and the
/// total off-centre weight (≈ 0.414) spreads evenly to the 8
/// neighbours. Verify the centre tap > the sum of any one corner
/// + any one edge.
#[test]
fn j2_default_kernel_low_pass_shape() {
    let rf = RestorationFilter::default();
    let k = gab_kernel(rf.gab_x_weight1, rf.gab_x_weight2).unwrap();
    let centre = k[4];
    let one_edge = k[1];
    let one_corner = k[0];
    // Centre dominates.
    assert!(centre > 4.0 * one_edge);
    assert!(centre > 4.0 * one_corner);
    // Edges > corners.
    assert!(one_edge > one_corner);
}

/// FDIS §J.2 — `sample_mirror` directly exposes the §6.5 fetch
/// semantics for any future caller that needs them outside of the
/// per-channel convolution helper. Pin both in-bounds and
/// out-of-bounds against a 4×3 plane.
#[test]
fn j2_sample_mirror_in_and_out_of_bounds() {
    let plane: Vec<f32> = (0..4 * 3).map(|v| v as f32).collect();
    // Layout 4 wide × 3 tall:
    //   0  1  2  3
    //   4  5  6  7
    //   8  9 10 11
    assert_eq!(sample_mirror(&plane, 4, 3, 0, 0).unwrap(), 0.0);
    assert_eq!(sample_mirror(&plane, 4, 3, 3, 2).unwrap(), 11.0);
    // (-1, 0) mirrors to (0, 0) = 0.
    assert_eq!(sample_mirror(&plane, 4, 3, -1, 0).unwrap(), 0.0);
    // (4, 0) mirrors to (3, 0) = 3.
    assert_eq!(sample_mirror(&plane, 4, 3, 4, 0).unwrap(), 3.0);
    // (0, 3) mirrors to (0, 2) = 8.
    assert_eq!(sample_mirror(&plane, 4, 3, 0, 3).unwrap(), 8.0);
    // (-1, -1) mirrors to (0, 0) = 0.
    assert_eq!(sample_mirror(&plane, 4, 3, -1, -1).unwrap(), 0.0);
    // (4, 3) mirrors to (3, 2) = 11.
    assert_eq!(sample_mirror(&plane, 4, 3, 4, 3).unwrap(), 11.0);
}

/// FDIS §J.2 — when the gab weights are zero, the kernel collapses
/// to the identity and the plane is preserved exactly (no
/// round-off, since 1.0 × x is exact in f32).
#[test]
fn j2_zero_weights_preserves_plane_exactly() {
    let w = 6_usize;
    let h = 4_usize;
    let input: Vec<f32> = (0..w * h).map(|v| (v as f32) * 1.5 - 7.0).collect();
    let mut output = vec![0.0_f32; w * h];
    apply_channel(&input, &mut output, w, h, 0.0, 0.0).unwrap();
    assert_eq!(output, input);
}
