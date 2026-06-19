//! Round 340 — non-square VarDCT reconstruction + LF/HF cross-pass.
//!
//! Drives the round-340 one-call reconstruction
//! ([`oxideav_jpegxl::vardct_reconstruct::reconstruct_lf_group_cross_pass`])
//! across the rectangular (non-square) DCT families through the full
//! cross-pass accumulation → F.3 dequant → §I.2.4 LLF merge → §I.2.3.2
//! IDCT → §C.5.4 placement → Annex G CfL walk, asserting each
//! non-square footprint reconstructs to spatial samples with the
//! expected padded-plane geometry and a constant-LF flat output.
//!
//! Clean-room: all behaviour is derived from the FDIS spec PDF under
//! `docs/image/jpegxl/`; no external implementation source is consulted.

use oxideav_jpegxl::dct_quant_weights::materialise_default_dequant_set;
use oxideav_jpegxl::dct_select::{DctSelectCell, DctSelectGrid, TransformType};
use oxideav_jpegxl::frame_header::Passes;
use oxideav_jpegxl::hf_dequant::QmScaleFactors;
use oxideav_jpegxl::lf_dequant::LfDequantOutput;
use oxideav_jpegxl::lf_global::LfChannelCorrelation;
use oxideav_jpegxl::metadata_fdis::OpsinInverseMatrix;
use oxideav_jpegxl::multi_pass_decode::MultiPassThreeChannelOutput;
use oxideav_jpegxl::pass_group_hf::DecodedHfBlock;
use oxideav_jpegxl::varblock_walk::Varblock;
use oxideav_jpegxl::vardct_reconstruct::{reconstruct_lf_group_cross_pass, DequantContext};

fn passes(num_passes: u32, shift: Vec<u32>) -> Passes {
    Passes {
        num_passes,
        num_ds: 0,
        shift,
        downsample: Vec::new(),
        last_pass: Vec::new(),
    }
}

fn vb(x: u32, y: u32, t: TransformType) -> Varblock {
    Varblock {
        x,
        y,
        transform: t,
        hf_mul: 1,
    }
}

fn block(coeffs: Vec<i32>) -> DecodedHfBlock {
    DecodedHfBlock {
        coeffs,
        remaining_non_zeros: 0,
        coeffs_read: 0,
    }
}

fn qm() -> QmScaleFactors {
    QmScaleFactors {
        x_factor: 0.8,
        b_factor: 1.0,
    }
}

/// Build a single-varblock grid + matching LF image + zero-HF single
/// pass for transform `t`, run the reconstruction, and return the Y
/// plane dims plus a "all-finite" flag.
fn reconstruct_single(t: TransformType) -> (usize, usize, bool, f32) {
    // Footprint in 8×8 cells.
    let (bcols, brows) = t.block_dims();
    let set = materialise_default_dequant_set().unwrap();
    let p = passes(1, vec![]);

    // Build the grid: TopLeft at (0,0), Continuation elsewhere.
    let w = bcols;
    let h = brows;
    let mut cells = vec![DctSelectCell::Continuation; (w * h) as usize];
    cells[0] = DctSelectCell::TopLeft(t);
    let mut hf_mul = vec![0i32; (w * h) as usize];
    hf_mul[0] = 1;
    let grid = DctSelectGrid {
        cells,
        hf_mul,
        width_blocks: w,
        height_blocks: h,
    };

    // LF image is w×h samples (one per 8×8 cell), constant.
    let n_lf = (w * h) as usize;
    let lf = LfDequantOutput {
        samples: [vec![2.0; n_lf], vec![2.0; n_lf], vec![2.0; n_lf]],
        widths: [w, w, w],
        heights: [h, h, h],
    };

    // Coefficient grid size = (bcols*8) × (brows*8).
    let coeff_n = (bcols * 8 * brows * 8) as usize;
    let mp: MultiPassThreeChannelOutput = vec![vec![(
        vb(0, 0, t),
        [
            block(vec![0; coeff_n]),
            block(vec![0; coeff_n]),
            block(vec![0; coeff_n]),
        ],
        [0, 0, 0],
    )]];

    let oim = OpsinInverseMatrix::default();
    let qmv = qm();
    let dq = DequantContext {
        set: &set,
        oim: &oim,
        qm: &qmv,
    };

    // CfL factor tiles: ceil(plane/64) per axis.
    let pw = (bcols * 8) as usize;
    let ph = (brows * 8) as usize;
    let tiles = pw.div_ceil(64) * ph.div_ceil(64);

    let planes = reconstruct_lf_group_cross_pass(
        &p,
        &grid,
        &lf,
        &dq,
        &vec![0i32; tiles],
        &vec![0i32; tiles],
        &LfChannelCorrelation::default(),
        &mp,
    )
    .unwrap();

    let (dw, dh) = planes.dims();
    let mut all_finite = true;
    for y in 0..dh {
        for x in 0..dw {
            if !planes.y().get(x, y).unwrap().is_finite() {
                all_finite = false;
            }
        }
    }
    let v0 = planes.y().get(0, 0).unwrap();
    (dw, dh, all_finite, v0)
}

#[test]
fn non_square_rectangular_dcts_reconstruct_to_spatial_samples() {
    // Each rectangular DCT reconstructs to its (bcols*8 × brows*8) padded
    // plane with finite samples. Constant LF + zero HF → a flat block, so
    // every sample equals the DC value.
    let cases = [
        (TransformType::Dct16x8, 8usize, 16usize), // 1 col × 2 rows cells → 8 wide × 16 tall
        (TransformType::Dct8x16, 16, 8),           // 2 cols × 1 row → 16 wide × 8 tall
        (TransformType::Dct32x8, 8, 32),           // 1 × 4 → 8 wide × 32 tall
        (TransformType::Dct8x32, 32, 8),           // 4 × 1 → 32 wide × 8 tall
        (TransformType::Dct32x16, 16, 32),         // 2 × 4 → 16 wide × 32 tall
        (TransformType::Dct16x32, 32, 16),         // 4 × 2 → 32 wide × 16 tall
    ];
    for (t, want_w, want_h) in cases {
        let (dw, dh, finite, _v0) = reconstruct_single(t);
        assert_eq!((dw, dh), (want_w, want_h), "{t:?} padded plane dims");
        assert!(finite, "{t:?} produced a non-finite sample");
    }
}

#[test]
fn non_square_flat_lf_reconstructs_to_constant() {
    // Constant LF + zero HF → the inverse DCT of a DC-only coefficient
    // grid is a flat block. Check the Y plane is (approximately) constant
    // across the whole non-square footprint for DCT8×16.
    let set = materialise_default_dequant_set().unwrap();
    let p = passes(1, vec![]);
    let grid = DctSelectGrid {
        cells: vec![
            DctSelectCell::TopLeft(TransformType::Dct8x16),
            DctSelectCell::Continuation,
        ],
        hf_mul: vec![1, 0],
        width_blocks: 2,
        height_blocks: 1,
    };
    let lf = LfDequantOutput {
        samples: [vec![6.0, 6.0], vec![6.0, 6.0], vec![6.0, 6.0]],
        widths: [2, 2, 2],
        heights: [1, 1, 1],
    };
    let mp: MultiPassThreeChannelOutput = vec![vec![(
        vb(0, 0, TransformType::Dct8x16),
        [
            block(vec![0; 128]),
            block(vec![0; 128]),
            block(vec![0; 128]),
        ],
        [0, 0, 0],
    )]];
    let oim = OpsinInverseMatrix::default();
    let qmv = qm();
    let dq = DequantContext {
        set: &set,
        oim: &oim,
        qm: &qmv,
    };
    let planes = reconstruct_lf_group_cross_pass(
        &p,
        &grid,
        &lf,
        &dq,
        &[0i32; 1],
        &[0i32; 1],
        &LfChannelCorrelation::default(),
        &mp,
    )
    .unwrap();
    assert_eq!(planes.dims(), (16, 8));
    let v0 = planes.y().get(0, 0).unwrap();
    for y in 0..8 {
        for x in 0..16 {
            let v = planes.y().get(x, y).unwrap();
            assert!(
                (v - v0).abs() < 1e-3,
                "DCT8×16 flat-LF Y ({x},{y}) = {v} != {v0}"
            );
        }
    }
}

#[test]
fn two_pass_non_square_accumulation_reaches_spatial_output() {
    // A two-pass DCT16×8: pass 0 carries an HF coefficient left-shifted by
    // shift[0]=1, pass 1 a smaller delta. The accumulated coefficient is
    // non-zero, so the reconstruction is NOT flat — at least one sample
    // departs from the DC value. Proves the cross-pass accumulation
    // reaches the spatial output for a non-square transform.
    let set = materialise_default_dequant_set().unwrap();
    // DCT16×8 footprint = (bcols=1, brows=2) → 8 wide × 16 tall, 1×2 grid.
    let grid = DctSelectGrid {
        cells: vec![
            DctSelectCell::TopLeft(TransformType::Dct16x8),
            DctSelectCell::Continuation,
        ],
        hf_mul: vec![1, 0],
        width_blocks: 1,
        height_blocks: 2,
    };
    let lf = LfDequantOutput {
        samples: [vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]],
        widths: [1, 1, 1],
        heights: [2, 2, 2],
    };
    // Coefficient grid is 16×8 = 128 cells. Put an AC coefficient at a
    // raster cell that is NOT in the LLF prefix (cell 5).
    let mut c0 = vec![0i32; 128];
    c0[5] = 6;
    let mut c1 = vec![0i32; 128];
    c1[5] = 2;
    let mp: MultiPassThreeChannelOutput = vec![
        vec![(
            vb(0, 0, TransformType::Dct16x8),
            [block(vec![0; 128]), block(c0), block(vec![0; 128])],
            [0, 0, 0],
        )],
        vec![(
            vb(0, 0, TransformType::Dct16x8),
            [block(vec![0; 128]), block(c1), block(vec![0; 128])],
            [0, 0, 0],
        )],
    ];
    let p = passes(2, vec![1]);
    let oim = OpsinInverseMatrix::default();
    let qmv = qm();
    let dq = DequantContext {
        set: &set,
        oim: &oim,
        qm: &qmv,
    };
    let planes = reconstruct_lf_group_cross_pass(
        &p,
        &grid,
        &lf,
        &dq,
        &[0i32; 1],
        &[0i32; 1],
        &LfChannelCorrelation::default(),
        &mp,
    )
    .unwrap();
    assert_eq!(planes.dims(), (8, 16));
    // With a non-zero AC coefficient the block is not flat: some Y sample
    // differs from the (0,0) value beyond rounding.
    let v0 = planes.y().get(0, 0).unwrap();
    let mut non_flat = false;
    for y in 0..16 {
        for x in 0..8 {
            if (planes.y().get(x, y).unwrap() - v0).abs() > 1e-3 {
                non_flat = true;
            }
        }
    }
    assert!(
        non_flat,
        "non-square two-pass AC reconstruction must not be flat"
    );
}
