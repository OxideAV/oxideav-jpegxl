//! Round-144 integration tests: Annex J.3 "Edge-preserving filter"
//! pure-math primitive (`oxideav_jpegxl::epf`).
//!
//! Pins the cross-module composition of:
//!
//! * the round-1+ [`oxideav_jpegxl::frame_header::RestorationFilter`]
//!   bundle (Table C.9) — the per-frame EPF parameters
//!   (`epf_iters`, `epf_sharp_lut`, `epf_channel_scale`,
//!   `epf_pass*_zeroflush`, `epf_quant_mul`,
//!   `epf_pass*_sigma_scale`, `epf_border_sad_mul`,
//!   `epf_sigma_for_modular`),
//! * the FDIS §J.3 normative listings:
//!   - Listing J.1 — `DistanceStep0and1` and `DistanceStep2`
//!     (per-channel scaled L1 distances),
//!   - Listing J.2 — `Weight(distance, sigma)` decreasing-function
//!     kernel with zeroflush cutoff and border-position multiplier,
//!   - Listing J.3 — VarDCT-mode `sigma` derivation,
//!   - Listing J.4 — per-pass 5-tap / 13-tap weighted-average
//!     application,
//! * the §6.5 `Mirror1D` boundary handling reused from round-141
//!   [`oxideav_jpegxl::gaborish::mirror1d`].
//!
//! into a single per-pass three-channel f32 pipeline that
//! downstream restoration-filter-wiring rounds can drop in without
//! re-deriving any of the §J.3 formulae.
//!
//! Round 144 does **not** drive this from a real frame — the
//! per-frame wiring (calling each pass for each varblock under the
//! right `epf_iters` / per-block sigma / position-multiplier
//! conditions, with the output of pass `i` feeding pass `i+1`) is
//! a follow-up round's responsibility. These tests pin the
//! pure-math step at the formula level.

use oxideav_jpegxl::epf::{
    apply_step_13tap, apply_step_5tap, distance_step_0_and_1, distance_step_2, inv_sigma_for_pass,
    is_border_position, vardct_sigma_from_listing_j3, weight, Pass,
};
use oxideav_jpegxl::frame_header::RestorationFilter;

/// FDIS §J.3.2 Listing J.1 `DistanceStep0and1` — when fed the
/// default-bundle EPF channel scales (`epf_channel_scale = [40.0,
/// 5.0, 3.5]` from Table C.9 defaults), the self-distance of every
/// reference pixel is 0 regardless of input data, because the
/// formula degenerates to `|s(x,y,c) - s(x,y,c)| × ... = 0` for
/// the (cx, cy) = (0, 0) case.
#[test]
fn round144_distance_step_0_and_1_self_distance_default_scales() {
    let plane: Vec<f32> = (0..36).map(|v| (v as f32) * 0.13).collect();
    let rf = RestorationFilter::default();
    for x in 0..6_i64 {
        for y in 0..6_i64 {
            let d = distance_step_0_and_1(
                &plane,
                &plane,
                &plane,
                6,
                6,
                x,
                y,
                0,
                0,
                rf.epf_channel_scale,
            )
            .unwrap();
            assert_eq!(d, 0.0, "self-distance at ({x},{y}) = {d}");
        }
    }
}

/// FDIS §J.3.2 Listing J.1 `DistanceStep2` — the single-sample
/// reading (per the module-level "Spec ambiguity" note) gives a
/// distance of exactly `|s_ref - s_nbr| × Σ channel_scale[c]`
/// when all three planes carry the same offset.
#[test]
fn round144_distance_step_2_hand_derived_single_sample() {
    // Plane 1 is all-1, plane 2 is all-3, plane 3 is all-5. The
    // reference position has the same value as the neighbour
    // position (because the plane is constant): distance = 0.
    let p1 = vec![1.0_f32; 9];
    let p2 = vec![3.0_f32; 9];
    let p3 = vec![5.0_f32; 9];
    let rf = RestorationFilter::default();
    let d = distance_step_2(&p1, &p2, &p3, 3, 3, 1, 1, 1, 0, rf.epf_channel_scale).unwrap();
    assert_eq!(d, 0.0);
}

/// FDIS §J.3.2 Listing J.1 `DistanceStep2` — hand-derived:
/// constant-but-different planes still give 0 (within-channel
/// differences are 0). To get a non-zero distance the planes must
/// vary spatially. Drive the x-channel as `sample(x,y) = x`,
/// y-channel as `sample(x,y) = 2y`, b-channel as `sample = 0`;
/// neighbour offset (1, 1). Per channel:
///   x: |s(1,1)=1 - s(2,2)=2| = 1 × 40 = 40
///   y: |s(1,1)=2 - s(2,2)=4| = 2 × 5  = 10
///   b: 0                              = 0
///   total = 50
#[test]
fn round144_distance_step_2_hand_derived_spatially_varying() {
    let w = 3_usize;
    let h = 3_usize;
    let x_p: Vec<f32> = (0..w * h).map(|i| (i % w) as f32).collect();
    let y_p: Vec<f32> = (0..w * h).map(|i| (2 * (i / w)) as f32).collect();
    let b_p: Vec<f32> = vec![0.0; w * h];
    let rf = RestorationFilter::default();
    let d = distance_step_2(&x_p, &y_p, &b_p, w, h, 1, 1, 1, 1, rf.epf_channel_scale).unwrap();
    assert!((d - 50.0).abs() < 1e-5, "expected 50.0 got {d}");
}

/// FDIS §J.3.3 Listing J.2 `Weight()` — at distance 0 the
/// computed `v = 1.0` always passes the zeroflush cutoff for
/// either pass (defaults `epf_pass1_zeroflush = 0.45`,
/// `epf_pass2_zeroflush = 0.6`), returning weight 1.0.
#[test]
fn round144_weight_self_distance_yields_unit_weight() {
    let rf = RestorationFilter::default();
    let sigma = 1.0_f32;
    let inv_sigma = inv_sigma_for_pass(1.0, sigma).unwrap();
    let w_pass1 = weight(0.0, inv_sigma, 1.0, rf.epf_pass1_zeroflush);
    let w_pass2 = weight(0.0, inv_sigma, 1.0, rf.epf_pass2_zeroflush);
    assert!((w_pass1 - 1.0).abs() < 1e-7);
    assert!((w_pass2 - 1.0).abs() < 1e-7);
}

/// FDIS §J.3.3 Listing J.3 — VarDCT sigma derivation from default
/// `rf`: at `quantization_width = 1.0`, `sharpness = 0` (default
/// lut[0] = 0/7 = 0), sigma collapses to the 1e-4 clamp.
#[test]
fn round144_vardct_sigma_default_sharpness_zero_clamps() {
    let rf = RestorationFilter::default();
    let sigma = vardct_sigma_from_listing_j3(1.0, 0, &rf).unwrap();
    // lut[0] is 0 / 7 = 0, so sigma = 1.0 × 0.46 × 0 = 0 → clamped
    // to 1e-4.
    assert!((sigma - 1e-4).abs() < 1e-9, "sigma {sigma}");
}

/// FDIS §J.3.3 Listing J.3 — VarDCT sigma at sharpness = 7 (default
/// lut[7] = 7/7 = 1): sigma = quantization_width × 0.46 × 1.0.
#[test]
fn round144_vardct_sigma_default_sharpness_seven_full_quant() {
    let rf = RestorationFilter::default();
    let sigma = vardct_sigma_from_listing_j3(2.5, 7, &rf).unwrap();
    let expected = 2.5_f32 * rf.epf_quant_mul * rf.epf_sharp_lut[7];
    assert!(
        (sigma - expected).abs() < 1e-6,
        "sigma {sigma} != {expected}"
    );
}

/// FDIS §J.2 Listing J.2 — `inv_sigma` derivation at
/// `step_multiplier = epf_pass0_sigma_scale = 0.9` (default),
/// sigma derived above:
/// `inv_sigma = 0.9 × 4 × (sqrt(0.5) - 1) / sigma`.
#[test]
fn round144_inv_sigma_for_pass0_from_default_rf() {
    let rf = RestorationFilter::default();
    let sigma = vardct_sigma_from_listing_j3(2.5, 7, &rf).unwrap();
    let inv = inv_sigma_for_pass(rf.epf_pass0_sigma_scale, sigma).unwrap();
    let expected = rf.epf_pass0_sigma_scale * 4.0_f32 * (0.5_f32.sqrt() - 1.0) / sigma;
    assert!(
        (inv - expected).abs() < 1e-6,
        "inv_sigma {inv} != {expected}"
    );
    // The numeric is negative under positive step_multiplier and
    // positive sigma (sqrt(0.5) - 1 < 0).
    assert!(inv < 0.0, "inv_sigma {inv} expected negative");
}

/// FDIS §J.3.3 — `is_border_position` predicate hand-derived
/// against the "0 or 7 IMod 8" condition for a 16×16 grid.
#[test]
fn round144_is_border_position_grid_layout() {
    // Border rows (y % 8 ∈ {0, 7}) and border cols (x % 8 ∈ {0, 7})
    // form a periodic cross-hatch with non-border interior 6×6
    // blocks at (1..7, 1..7) within each 8×8 tile.
    for y in 0_usize..16 {
        for x in 0_usize..16 {
            let xm = x % 8;
            let ym = y % 8;
            let expected = xm == 0 || xm == 7 || ym == 0 || ym == 7;
            assert_eq!(
                is_border_position(x, y),
                expected,
                "is_border_position({x},{y}) mismatch"
            );
        }
    }
}

/// FDIS §J.3 — the EPF on a constant plane is invariant for both
/// 5-tap passes AND the 13-tap pass 0, because the (0,0) kernel tap
/// always weights the centre with weight ≥ 0 (typically 1.0) and
/// every other tap weights its mirror neighbour with weight ≥ 0
/// — and a weighted average of identical values is the same value.
#[test]
fn round144_epf_constant_plane_invariant_across_all_three_passes() {
    let w = 8_usize;
    let h = 8_usize;
    let rf = RestorationFilter::default();
    let val = 4.625_f32;
    let p = vec![val; w * h];

    // Pass 1: 5-tap with DistanceStep0and1
    let mut xo = vec![0.0_f32; w * h];
    let mut yo = vec![0.0_f32; w * h];
    let mut bo = vec![0.0_f32; w * h];
    apply_step_5tap(
        Pass::Pass1,
        &p,
        &p,
        &p,
        &mut xo,
        &mut yo,
        &mut bo,
        w,
        h,
        1.0,
        1.0,
        rf.epf_pass1_zeroflush,
        rf.epf_border_sad_mul,
        rf.epf_channel_scale,
    )
    .unwrap();
    for &v in xo.iter() {
        assert!((v - val).abs() < 1e-4, "pass1 drift {v} from {val}");
    }

    // Pass 2: 5-tap with DistanceStep2
    let mut xo = vec![0.0_f32; w * h];
    let mut yo = vec![0.0_f32; w * h];
    let mut bo = vec![0.0_f32; w * h];
    apply_step_5tap(
        Pass::Pass2,
        &p,
        &p,
        &p,
        &mut xo,
        &mut yo,
        &mut bo,
        w,
        h,
        1.0,
        rf.epf_pass2_sigma_scale,
        rf.epf_pass2_zeroflush,
        rf.epf_border_sad_mul,
        rf.epf_channel_scale,
    )
    .unwrap();
    for &v in xo.iter() {
        assert!((v - val).abs() < 1e-4);
    }

    // Pass 0: 13-tap with DistanceStep0and1
    let mut xo = vec![0.0_f32; w * h];
    let mut yo = vec![0.0_f32; w * h];
    let mut bo = vec![0.0_f32; w * h];
    apply_step_13tap(
        &p,
        &p,
        &p,
        &mut xo,
        &mut yo,
        &mut bo,
        w,
        h,
        1.0,
        rf.epf_pass0_sigma_scale,
        0.0, // pass-0 has no spec'd zeroflush; use minimum cutoff
        rf.epf_border_sad_mul,
        rf.epf_channel_scale,
    )
    .unwrap();
    for &v in xo.iter() {
        assert!((v - val).abs() < 1e-4);
    }
}

/// FDIS §J.3 — when the per-channel scales are all zero, every
/// distance is 0 (per channel: `0 × |dx|`), so every weight is 1.0,
/// and the output is the un-weighted arithmetic mean of the kernel
/// taps over Mirror1D-mirrored inputs. On a single-channel impulse
/// at the centre of a 5×5 plane with the 5-tap Pass-1 kernel, the
/// centre output averages 5 mirrored-equal samples (the centre = 1
/// and four cross neighbours = 0 in the impulse plane) so each
/// kernel-tap weight is 1.0 and the centre output equals 1 / 5.
#[test]
fn round144_epf_zero_channel_scale_collapses_to_uniform_mean() {
    let w = 5_usize;
    let h = 5_usize;
    let mut p = vec![0.0_f32; w * h];
    p[2 * w + 2] = 1.0; // centre impulse
    let z = vec![0.0_f32; w * h];
    let mut xo = vec![0.0_f32; w * h];
    let mut yo = vec![0.0_f32; w * h];
    let mut bo = vec![0.0_f32; w * h];
    apply_step_5tap(
        Pass::Pass1,
        &p,
        &z,
        &z,
        &mut xo,
        &mut yo,
        &mut bo,
        w,
        h,
        1.0,
        1.0,
        0.0,
        1.0,
        [0.0, 0.0, 0.0],
    )
    .unwrap();
    // Centre output averages 5 kernel taps (one centre = 1, four
    // cross neighbours = 0) ÷ 5 = 0.2.
    assert!((xo[2 * w + 2] - 0.2).abs() < 1e-5);
    // All four cross-shape neighbours of the centre (at (1,2),
    // (3,2), (2,1), (2,3)) have ONE kernel reference hitting the
    // impulse (the centre tap from THAT reference's neighbours
    // pointing back at the centre): 1/5 = 0.2.
    assert!((xo[2 * w + 1] - 0.2).abs() < 1e-5);
    assert!((xo[2 * w + 3] - 0.2).abs() < 1e-5);
    assert!((xo[w + 2] - 0.2).abs() < 1e-5);
    assert!((xo[3 * w + 2] - 0.2).abs() < 1e-5);
    // The y and b output planes are 0 / sum_w = 0 everywhere.
    for &v in yo.iter() {
        assert_eq!(v, 0.0);
    }
    for &v in bo.iter() {
        assert_eq!(v, 0.0);
    }
}

/// FDIS §J.3.1 pass dispatch ergonomics — [`Pass::Pass0`] is
/// rejected by [`apply_step_5tap`] (wrong kernel shape), and the
/// public API steers callers to [`apply_step_13tap`] for the pass-0
/// diamond.
#[test]
fn round144_pass0_routing_is_enforced() {
    let p = vec![1.0_f32; 9];
    let mut o = vec![0.0_f32; 9];
    let r = apply_step_5tap(
        Pass::Pass0,
        &p,
        &p,
        &p,
        &mut o.clone(),
        &mut o.clone(),
        &mut o,
        3,
        3,
        1.0,
        1.0,
        0.0,
        1.0,
        [40.0, 5.0, 3.5],
    );
    assert!(r.is_err(), "Pass0 should be rejected by apply_step_5tap");
}

/// FDIS §J.3 — the Modular-mode sigma is taken from
/// `rf.epf_sigma_for_modular` directly (per §J.3.3 prose:
/// "sigma is then computed as specified by Listing J.3 if the frame
/// encoding is kVarDCT, else it is set to rf.epf_sigma_for_modular").
/// Round 144 surfaces the value via [`RestorationFilter`] defaults
/// — the wiring round wires the encoding-mode branch.
#[test]
fn round144_modular_sigma_default_is_1_point_0() {
    let rf = RestorationFilter::default();
    assert!((rf.epf_sigma_for_modular - 1.0).abs() < 1e-7);
    // Per Listing J.3, this value is NOT subjected to the 1e-4
    // clamp (the clamp is only in the VarDCT branch); the default
    // 1.0 is well clear regardless.
    let inv = inv_sigma_for_pass(1.0, rf.epf_sigma_for_modular).unwrap();
    assert!(inv < 0.0);
}
