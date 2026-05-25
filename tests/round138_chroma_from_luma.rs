//! Round-138 integration tests: Annex G "Chroma from luma" pure-
//! math primitive (`oxideav_jpegxl::chroma_from_luma`).
//!
//! Pins the cross-module composition of:
//!
//! * the round-11 [`oxideav_jpegxl::lf_global::LfChannelCorrelation`]
//!   bundle (§C.4.4) — the per-frame colour-correlation parameters,
//! * the round-12+16 [`oxideav_jpegxl::lf_group::HfMetadata`]
//!   `x_from_y` / `b_from_y` per-64×64-tile factor channels (§C.5.4),
//! * the FDIS Annex G Listing G.1 reconstruction formula
//!
//! into a single per-sample / per-plane bit-exact float pipeline
//! that downstream VarDCT-wiring rounds can drop in without
//! re-deriving any of the G.1 formulae.
//!
//! Round 138 does **not** drive this from a real VarDCT bitstream —
//! that wiring is one or more follow-up rounds; the §F.3 HF
//! dequantisation and the §C.8.3 per-block ANS coefficient decode
//! both need to be glued in first. These tests pin the pure-math
//! step at the formula level.

use oxideav_jpegxl::chroma_from_luma::{
    apply_hf_plane_inplace, apply_lf_plane_inplace, apply_lf_sample, apply_sample, kx_kb_hf,
    kx_kb_lf, kx_kb_raw,
};
use oxideav_jpegxl::lf_global::LfChannelCorrelation;

/// FDIS Annex G — at the `all_default` `LfChannelCorrelation` bundle
/// (`colour_factor = 84`, `base_correlation_x = 0.0`,
/// `base_correlation_b = 1.0`, `x_factor_lf = b_factor_lf = 128`),
/// the LF derivation yields `(kX, kB) = (1/84, 1 + 1/84)`.
#[test]
fn annex_g_lf_default_bundle() {
    let cfl = LfChannelCorrelation::default();
    let (kx, kb) = kx_kb_lf(&cfl).unwrap();
    assert!((kx - 1.0_f32 / 84.0).abs() < 1e-7);
    assert!((kb - (1.0_f32 + 1.0_f32 / 84.0)).abs() < 1e-7);
}

/// FDIS Annex G — Listing G.1 line 3 (`Y = dY`) is the identity on
/// the Y channel for every input.
#[test]
fn listing_g1_y_identity() {
    let cfl = LfChannelCorrelation::default();
    for &dy in &[-1.5_f32, 0.0, 0.25, 100.0] {
        let (_, y, _) = apply_lf_sample(0.0, dy, 0.0, &cfl).unwrap();
        assert_eq!(y, dy);
    }
}

/// FDIS Annex G — Listing G.1 lines 4 & 5 collapse to dX / dB at
/// dY == 0 regardless of `(kX, kB)`.
#[test]
fn listing_g1_zero_y_passthrough() {
    let cfl = LfChannelCorrelation::default();
    let (x, _, b) = apply_lf_sample(-3.0, 0.0, 7.0, &cfl).unwrap();
    assert_eq!(x, -3.0);
    assert_eq!(b, 7.0);
}

/// FDIS Annex G — for HF coefficients, `(x_factor, b_factor)` come
/// from `XFromY` / `BFromY` "at the coordinates of the 64 × 64
/// rectangle containing the current sample" (last paragraph). With
/// the `all_default` bundle and per-tile factors `(0, 0)`, kX = 0.0
/// and kB = base_correlation_b = 1.0 exactly.
#[test]
fn annex_g_hf_zero_factors_collapse_to_base() {
    let cfl = LfChannelCorrelation::default();
    let (kx, kb) = kx_kb_hf(&cfl, 0, 0).unwrap();
    assert_eq!(kx, 0.0);
    assert_eq!(kb, 1.0);
}

/// FDIS Annex G — the per-tile compute `(kX, kB)` only depends on
/// `colour_factor`, `base_correlation_*` and the local `(x_factor,
/// b_factor)`. Two bundles with the same constants but different
/// `x_factor_lf` produce the same kX/kB when their per-tile HF
/// factors match.
#[test]
fn annex_g_hf_independent_of_lf_factors_field() {
    let cfl_a = LfChannelCorrelation {
        all_default: false,
        colour_factor: 84,
        base_correlation_x: 0.0,
        base_correlation_b: 1.0,
        x_factor_lf: 128,
        b_factor_lf: 128,
    };
    let cfl_b = LfChannelCorrelation {
        all_default: false,
        colour_factor: 84,
        base_correlation_x: 0.0,
        base_correlation_b: 1.0,
        x_factor_lf: 255,
        b_factor_lf: 0,
    };
    assert_eq!(
        kx_kb_hf(&cfl_a, 5, -3).unwrap(),
        kx_kb_hf(&cfl_b, 5, -3).unwrap()
    );
}

/// Cross-module check: applying the encoder-side decorrelation
/// `dX = X - kX × Y`, `dB = B - kB × Y` and then the round-138
/// reconstruction recovers (X, Y, B) exactly within f32 epsilon.
/// This pins the formula direction.
#[test]
fn annex_g_round_trip_against_forward_decorrelation() {
    let cfl = LfChannelCorrelation::default();
    let (kx, kb) = kx_kb_lf(&cfl).unwrap();
    let cases: &[(f32, f32, f32)] = &[
        (1.0, 2.0, 3.0),
        (-1.0, 0.5, 0.0),
        (10.0, -5.0, 2.5),
        (0.0, 0.0, 0.0),
    ];
    for &(x, y, b) in cases {
        let dx = x - kx * y;
        let db = b - kb * y;
        let (xr, yr, br) = apply_sample(dx, y, db, kx, kb);
        assert!((xr - x).abs() < 1e-5);
        assert_eq!(yr, y);
        assert!((br - b).abs() < 1e-5);
    }
}

/// A 64×64 single-tile HF plane reconstruction equals a constant-
/// (kX, kB) per-sample LF reconstruction with the appropriate
/// HF-derived (kX, kB). This validates the tile-lookup path on the
/// degenerate single-tile case.
#[test]
fn annex_g_hf_single_tile_matches_constant_apply() {
    let cfl = LfChannelCorrelation::default();
    let n = 64 * 64;
    let dy: Vec<f32> = (0..n).map(|i| (i as f32 * 0.125).sin()).collect();
    let mut dx_hf = vec![0.0_f32; n];
    let mut db_hf = vec![0.0_f32; n];
    let mut dx_lf_ref = vec![0.0_f32; n];
    let mut db_lf_ref = vec![0.0_f32; n];

    let x_from_y = vec![7_i32];
    let b_from_y = vec![-2_i32];

    apply_hf_plane_inplace(
        &mut dx_hf, &dy, &mut db_hf, 64, 64, &x_from_y, &b_from_y, &cfl,
    )
    .unwrap();

    let (kx, kb) = kx_kb_hf(&cfl, 7, -2).unwrap();
    for i in 0..n {
        let y = dy[i];
        dx_lf_ref[i] += kx * y;
        db_lf_ref[i] += kb * y;
    }

    for i in 0..n {
        assert!((dx_hf[i] - dx_lf_ref[i]).abs() < 1e-5);
        assert!((db_hf[i] - db_lf_ref[i]).abs() < 1e-5);
    }
}

/// FDIS Annex G — different per-tile `(x_factor, b_factor)` values
/// across a multi-tile HF plane yield different `(kX, kB)` per tile.
/// A `128 × 64` plane is exactly 2 tiles wide × 1 tile tall: left
/// half uses `(x_from_y[0], b_from_y[0])`, right half uses
/// `(x_from_y[1], b_from_y[1])`.
#[test]
fn annex_g_hf_per_tile_lookup_two_horizontal_tiles() {
    let cfl = LfChannelCorrelation::default();
    let w = 128_u32;
    let h = 64_u32;
    let n = (w * h) as usize;
    let dy = vec![1.0_f32; n];
    let mut dx = vec![0.0_f32; n];
    let mut db = vec![0.0_f32; n];
    let x_from_y = vec![10_i32, -10];
    let b_from_y = vec![5_i32, -5];

    apply_hf_plane_inplace(&mut dx, &dy, &mut db, w, h, &x_from_y, &b_from_y, &cfl).unwrap();

    let (kx_l, kb_l) = kx_kb_hf(&cfl, 10, 5).unwrap();
    let (kx_r, kb_r) = kx_kb_hf(&cfl, -10, -5).unwrap();
    // Left column (x=0) — tile (0,0).
    assert!((dx[0] - kx_l).abs() < 1e-6);
    assert!((db[0] - kb_l).abs() < 1e-6);
    // Right column (x=64) — tile (1,0).
    assert!((dx[64] - kx_r).abs() < 1e-6);
    assert!((db[64] - kb_r).abs() < 1e-6);
}

/// FDIS Annex G — on a partial-tile plane (e.g. 65 × 65) the tile
/// grid still rounds up: `ceil(65 / 64) = 2`, so the plane has
/// `2 × 2 = 4` tiles even though the last tile is mostly empty.
/// This pins the `div_ceil` lookup boundary.
#[test]
fn annex_g_hf_partial_tile_rounds_up() {
    let cfl = LfChannelCorrelation::default();
    let w = 65_u32;
    let h = 65_u32;
    let n = (w * h) as usize;
    let dy = vec![1.0_f32; n];
    let mut dx = vec![0.0_f32; n];
    let mut db = vec![0.0_f32; n];
    // Tile factors: (0,0)→1, (1,0)→2, (0,1)→3, (1,1)→4.
    let x_from_y = vec![1_i32, 2, 3, 4];
    let b_from_y = vec![0_i32; 4];
    apply_hf_plane_inplace(&mut dx, &dy, &mut db, w, h, &x_from_y, &b_from_y, &cfl).unwrap();

    // Sample (0,0) → tile (0,0).
    let (kx00, _) = kx_kb_hf(&cfl, 1, 0).unwrap();
    assert!((dx[0] - kx00).abs() < 1e-6);
    // Sample (64,0) → tile (1,0).
    let (kx10, _) = kx_kb_hf(&cfl, 2, 0).unwrap();
    assert!((dx[64] - kx10).abs() < 1e-6);
    // Sample (0,64) → tile (0,1).
    let (kx01, _) = kx_kb_hf(&cfl, 3, 0).unwrap();
    assert!((dx[64 * (w as usize)] - kx01).abs() < 1e-6);
    // Sample (64,64) → tile (1,1).
    let (kx11, _) = kx_kb_hf(&cfl, 4, 0).unwrap();
    assert!((dx[64 * (w as usize) + 64] - kx11).abs() < 1e-6);
}

/// FDIS Annex G — the LF plane API is in-place and idempotent
/// composition with itself does NOT round-trip (CfL is a one-way
/// reconstruction step, not an involution): applying twice with
/// the same dY adds the kX × Y / kB × Y deltas twice.
#[test]
fn annex_g_lf_plane_not_idempotent_double_application() {
    let cfl = LfChannelCorrelation::default();
    let dy = vec![4.0_f32, -2.0, 1.5];
    let mut dx = vec![0.0_f32; 3];
    let mut db = vec![0.0_f32; 3];
    let dx_save = dx.clone();
    let db_save = db.clone();
    apply_lf_plane_inplace(&mut dx, &dy, &mut db, &cfl).unwrap();
    let dx_after_one = dx.clone();
    let _db_after_one = db.clone();
    apply_lf_plane_inplace(&mut dx, &dy, &mut db, &cfl).unwrap();
    let (kx, kb) = kx_kb_lf(&cfl).unwrap();
    for i in 0..3 {
        // After one apply: dx_save + kx*dy.
        assert!((dx_after_one[i] - (dx_save[i] + kx * dy[i])).abs() < 1e-6);
        // After two applies: dx_save + 2*kx*dy.
        assert!((dx[i] - (dx_save[i] + 2.0 * kx * dy[i])).abs() < 1e-5);
        assert!((db[i] - (db_save[i] + 2.0 * kb * dy[i])).abs() < 1e-5);
    }
}

/// `kx_kb_raw` is the lowest-level helper: it does not touch
/// `LfChannelCorrelation` directly and is the building block both
/// `kx_kb_lf` and `kx_kb_hf` go through. Pinning its output at a
/// known dyadic-exact input shows the f32 arithmetic is the
/// straightforward `(a) + (b)/(c)` per Listing G.1.
#[test]
fn raw_dyadic_exact_input_pin() {
    // base_correlation_x = 0.0, base_correlation_b = 0.5,
    // colour_factor = 4, x_factor = 1, b_factor = -2 →
    // kX = 0 + 1/4 = 0.25, kB = 0.5 + (-2)/4 = 0.0.
    let (kx, kb) = kx_kb_raw(0.0, 0.5, 4, 1, -2);
    assert_eq!(kx, 0.25);
    assert_eq!(kb, 0.0);
}
