//! VarDCT decode path — ISO/IEC 18181-1:2024 Annex I.
//!
//! Round-8 lands the **scaffold**: structural recognition of a
//! VarDCT-encoded codestream + fixed-size DCT-II / IDCT primitives
//! for the smallest block size (8x8). End-to-end pixel decode is
//! deferred to round-9+.
//!
//! ## Annex I overview
//!
//! VarDCT splits a frame into 8x8 (or larger) blocks. Each colour
//! channel is partitioned into:
//!
//! * **LF coefficients** — one DC + low-frequency coefficient per
//!   block, decoded via a separate modular sub-bitstream that lives
//!   inside the LfGroup section (G.2.2). Round 8 doesn't decode LF.
//! * **HF coefficients** — the remaining 63 (for 8x8) high-frequency
//!   AC coefficients per block, decoded via a clustered ANS stream
//!   inside each PassGroup (G.4.3). Round 8 doesn't decode HF.
//!
//! After both subbands are decoded:
//!
//! 1. Dequantise (multiply by per-channel + per-block-size + per-position
//!    weights from the Quantizer / HfBlockContext / LfChannelCorrelation
//!    headers in LfGlobal).
//! 2. Inverse-DCT each block (variable size: 8x8, 8x16, 16x8, 16x16,
//!    32x32, 64x64, plus DCT4/8 + DCT4x8 + DCT8x4 + IDENTITY + AFV
//!    transforms — Annex I.4).
//! 3. Apply Chroma-from-Luma (LfChannelCorrelation) — round-9+.
//! 4. Apply Gaborish smoothing (RestorationFilter.gab_*) —
//!    round-9+.
//! 5. Apply EPF / loop-filter (RestorationFilter.epf_*) — round-9+.
//! 6. Convert from XYB / YCbCr to the output colour space.
//!
//! Round 8's contribution: recognition + IDCT-8x8 primitive +
//! placeholder VideoFrame output (all-zeros) so a VarDCT fixture
//! goes through `decode_one_frame` without `Error::Unsupported`.
//! This unblocks downstream callers that probe the codestream
//! signature and only error if pixel data is asked for.

use oxideav_core::Error;

use crate::dct_select::TransformType;
use crate::frame_header::FrameHeader;
use crate::lf_dequant::LfDequantOutput;
use crate::llf_from_lf::{llf_dims, llf_from_lf};
use crate::metadata_fdis::ImageMetadataFdis;

/// Local alias matching the rest of the crate's error surface.
type Result<T> = std::result::Result<T, Error>;

/// Inverse DCT-II of size 8 along one axis. Output[k] = sum_n
/// (input[n] * cos(pi*(2k+1)*n / 16)) for n=0..7, with the spec's
/// scale factor (1/sqrt(2) for n=0, 1 otherwise) folded into a
/// single normalisation by 0.5 (the inverse-transform amplitude).
///
/// Implemented as a plain O(N^2) sum so the scaffolding is self-
/// contained and audit-friendly. Faster Lee-style decompositions
/// land in round 9+ once LF/HF subband decode joins.
pub fn idct1d_8(coeffs: &[f32; 8]) -> [f32; 8] {
    use std::f32::consts::PI;
    let mut out = [0.0f32; 8];
    let scale0 = 1.0 / 2f32.sqrt();
    for (k, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (n, &c) in coeffs.iter().enumerate() {
            let s = if n == 0 { scale0 } else { 1.0 };
            acc += s * c * f32::cos(PI * ((2 * k + 1) as f32) * (n as f32) / 16.0);
        }
        *slot = 0.5 * acc;
    }
    out
}

/// 2-D inverse DCT-II over an 8x8 coefficient block. Applies
/// [`idct1d_8`] along columns, then along rows.
pub fn idct2d_8x8(coeffs: &[[f32; 8]; 8]) -> [[f32; 8]; 8] {
    let mut tmp = [[0.0f32; 8]; 8];
    // 1-D IDCT along columns.
    for col in 0..8 {
        let column: [f32; 8] = std::array::from_fn(|r| coeffs[r][col]);
        let out = idct1d_8(&column);
        for r in 0..8 {
            tmp[r][col] = out[r];
        }
    }
    let mut result = [[0.0f32; 8]; 8];
    // 1-D IDCT along rows.
    for r in 0..8 {
        let out = idct1d_8(&tmp[r]);
        result[r] = out;
    }
    result
}

/// Result of [`recognise_vardct_codestream`]: the codestream is
/// VarDCT-encoded, with the recorded geometry. Pixel decode is not
/// yet wired — see crate-level docs for round-9+ scope.
#[derive(Debug, Clone)]
pub struct VarDctScaffold {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Group dimension (typically 256 for VarDCT, but FrameHeader
    /// can override via `group_size_shift`).
    pub group_dim: u32,
    /// Number of colour channels reaching the output (after XYB
    /// inverse for `xyb_encoded == true`).
    pub num_colour_channels: usize,
}

/// Recognise an FDIS / 2024-spec VarDCT codestream's structural
/// metadata. Returns `Ok(VarDctScaffold)` when the FrameHeader
/// indicates `encoding == kVarDCT` and all other fields fall in the
/// round-8 envelope; returns `Err(Unsupported)` for anything outside
/// that envelope.
///
/// Round-8 envelope:
/// * Single LF group (`num_lf_groups == 1`).
/// * Single pass (`num_passes == 1`).
/// * No animation, no preview, no extra channels.
///
/// This routine performs **no pixel decode** — see crate docs.
pub fn recognise_vardct_codestream(
    fh: &FrameHeader,
    metadata: &ImageMetadataFdis,
) -> std::result::Result<VarDctScaffold, Error> {
    if fh.num_lf_groups() > 1 {
        return Err(Error::Unsupported(format!(
            "jxl VarDCT (round 8 scaffold): num_lf_groups = {} > 1 not yet supported",
            fh.num_lf_groups()
        )));
    }
    if fh.passes.num_passes > 1 {
        return Err(Error::Unsupported(format!(
            "jxl VarDCT (round 8 scaffold): num_passes = {} > 1 not yet supported",
            fh.passes.num_passes
        )));
    }
    if metadata.num_extra_channels > 0 {
        return Err(Error::Unsupported(format!(
            "jxl VarDCT (round 8 scaffold): {} extra channels not yet supported",
            metadata.num_extra_channels
        )));
    }
    let num_colour_channels = match metadata.colour_encoding.colour_space {
        crate::metadata_fdis::ColourSpace::Grey => 1,
        crate::metadata_fdis::ColourSpace::Rgb => 3,
        _ => {
            return Err(Error::Unsupported(format!(
                "jxl VarDCT (round 8 scaffold): colour space {:?} not yet supported",
                metadata.colour_encoding.colour_space
            )));
        }
    };
    Ok(VarDctScaffold {
        width: fh.width,
        height: fh.height,
        group_dim: fh.group_dim(),
        num_colour_channels,
    })
}

// =========================================================================
// Round 129 — per-varblock LF→LLF composition (§I.2.5 plumbing)
// =========================================================================
//
// The pure-math LF→LLF step landed in round 121 (`llf_from_lf::llf_from_lf`)
// and the per-LfGroup LF dequant + smoothing landed in rounds 12/13
// (`lf_dequant::dequant_lf` + `apply_adaptive_lf_smoothing`). Round 129
// adds the **glue** that drives `llf_from_lf` from the
// `LfDequantOutput` for a single varblock placement, per FDIS Annex
// I.2.5 prose:
//
// > For each varblock of size X × Y, the decoder takes the
// > corresponding X/8 × Y/8 samples from the dequantized LF image
// > and computes the top-left X/8 × Y/8 coefficients of the HF
// > varblock.
//
// This is pure array indexing + a call into `llf_from_lf` for every
// (channel, varblock) pair. Wiring it into `decode_codestream`
// requires the per-LfGroup `DctSelect` grid from `dct_select::
// derive_dct_select` AND the HF coefficient buffer the round-91+
// ANS decoder will populate; round 129 lands the geometry helper
// so that wiring step can use it verbatim.
//
// All functions take a single channel's LF samples + dims; the
// caller is responsible for invoking once per colour channel (X,
// Y, B) of an LfGroup. The varblock origin is specified in LF
// sample units (i.e. 8×8-pixel block units within the LF grid,
// matching `DctSelectCell::block_x / block_y`).

/// Extract the `cy × cx` LF sub-block from a single channel's
/// dequantised LF image at varblock origin `(bx, by)`.
///
/// `lf_samples` is `lf_width * lf_height` `f32` values in row-major
/// order — one LF sample per 8×8 pixel block, indexed
/// `[y * lf_width + x]`.
///
/// `(bx, by)` is the varblock origin in LF-sample units (i.e. the
/// origin in 8×8-block grid coordinates within this channel's LF
/// grid). For an LfGroup whose LF channel is 32×32 samples and a
/// DCT16×16 varblock placed at block-grid `(2, 2)`, the call would
/// be `extract_lf_subblock(lf_samples, 32, 32, 2, 2, TransformType::Dct16x16)`
/// and would return a 4-element `(cx=2 × cy=2)` row-major sub-block
/// starting at LF position `(2, 2)`.
///
/// The returned vector is `cy * cx` samples row-major:
/// `out[dy * cx + dx] = lf_samples[(by + dy) * lf_width + (bx + dx)]`.
///
/// Returns `Err(InvalidData)` when the varblock would read past the
/// channel's LF grid (per §I.2.5: the LF grid dimensions are sized
/// for the encoded image and every valid varblock must fit
/// entirely inside).
pub fn extract_lf_subblock(
    lf_samples: &[f32],
    lf_width: u32,
    lf_height: u32,
    bx: u32,
    by: u32,
    t: TransformType,
) -> Result<Vec<f32>> {
    let (cx, cy) = llf_dims(t);
    let lf_width_u = lf_width as usize;
    let lf_height_u = lf_height as usize;
    if lf_samples.len() != lf_width_u * lf_height_u {
        return Err(Error::InvalidData(format!(
            "JXL extract_lf_subblock: lf_samples length {} != \
             lf_width {} * lf_height {} = {}",
            lf_samples.len(),
            lf_width,
            lf_height,
            lf_width_u * lf_height_u,
        )));
    }
    // §I.2.5 invariant: the varblock must fit within the LF grid.
    let bx_end = bx.checked_add(cx).ok_or_else(|| {
        Error::InvalidData(format!(
            "JXL extract_lf_subblock: bx ({bx}) + cx ({cx}) overflow"
        ))
    })?;
    let by_end = by.checked_add(cy).ok_or_else(|| {
        Error::InvalidData(format!(
            "JXL extract_lf_subblock: by ({by}) + cy ({cy}) overflow"
        ))
    })?;
    if bx_end > lf_width || by_end > lf_height {
        return Err(Error::InvalidData(format!(
            "JXL extract_lf_subblock: varblock {t:?} at origin \
             ({bx}, {by}) with dims ({cx} × {cy}) extends past LF \
             grid ({lf_width} × {lf_height})"
        )));
    }
    let cx_u = cx as usize;
    let cy_u = cy as usize;
    let bx_u = bx as usize;
    let by_u = by as usize;
    let mut out = Vec::with_capacity(cx_u * cy_u);
    for dy in 0..cy_u {
        let row_off = (by_u + dy) * lf_width_u + bx_u;
        out.extend_from_slice(&lf_samples[row_off..row_off + cx_u]);
    }
    Ok(out)
}

/// Compose `extract_lf_subblock` with `llf_from_lf::llf_from_lf` to
/// produce the top-left LLF coefficient block of a single varblock
/// in one call.
///
/// Returns the `cy * cx` LLF coefficients in row-major order, indexed
/// `out[y * cx + x] = LF→LLF(input)(x, y)` per FDIS Listing I.16.
///
/// Errors:
/// * `InvalidData` — `lf_samples` length mismatch with `lf_width *
///   lf_height`, varblock origin overflow, or varblock extending
///   past the LF grid.
///
/// The non-DCT transforms (Hornuss / DCT2×2 / DCT4×4 / DCT4×8 /
/// DCT8×4 / AFV0..AFV3) all have `(cx, cy) = (1, 1)` so the
/// returned 1-element vector is the single LF sample at
/// `(bx, by)` unchanged — per §I.2.5 closing sentence.
pub fn compose_lf_to_llf_block(
    lf_samples: &[f32],
    lf_width: u32,
    lf_height: u32,
    bx: u32,
    by: u32,
    t: TransformType,
) -> Result<Vec<f32>> {
    let sub = extract_lf_subblock(lf_samples, lf_width, lf_height, bx, by, t)?;
    llf_from_lf(&sub, t)
}

/// Per-varblock LLF blocks for all three colour channels (X, Y, B),
/// extracted from a single LfGroup's `LfDequantOutput`.
///
/// Indexed by channel `c ∈ [0..3]`, each entry is a `cy × cx`
/// row-major LLF coefficient block. The three channels must share
/// the same LF dimensions (caller must verify `out.widths[c]` is
/// identical for `c ∈ [0..3]`, which holds when no channel is
/// subsampled — the only case §F.2 applies smoothing to).
///
/// For subsampled channels (`jpeg_upsampling[c] != 0`), the LF
/// grid per channel has different dims and the caller must invoke
/// `compose_lf_to_llf_block` per-channel with each channel's own
/// `widths[c]` / `heights[c]`. This convenience helper is for the
/// common non-subsampled case.
pub fn compose_lf_to_llf_block_3ch(
    lf: &LfDequantOutput,
    bx: u32,
    by: u32,
    t: TransformType,
) -> Result<[Vec<f32>; 3]> {
    // Verify all three channels share the same LF dims.
    if lf.widths[0] != lf.widths[1]
        || lf.widths[0] != lf.widths[2]
        || lf.heights[0] != lf.heights[1]
        || lf.heights[0] != lf.heights[2]
    {
        return Err(Error::InvalidData(format!(
            "JXL compose_lf_to_llf_block_3ch: LF channels have \
             different dims (widths = {:?}, heights = {:?}); use \
             `compose_lf_to_llf_block` per channel for the \
             subsampled case",
            lf.widths, lf.heights,
        )));
    }
    let w = lf.widths[0];
    let h = lf.heights[0];
    let x = compose_lf_to_llf_block(&lf.samples[0], w, h, bx, by, t)?;
    let y = compose_lf_to_llf_block(&lf.samples[1], w, h, bx, by, t)?;
    let b = compose_lf_to_llf_block(&lf.samples[2], w, h, bx, by, t)?;
    Ok([x, y, b])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn idct1d_8_dc_only_returns_constant() {
        // DC-only input: coeffs = [c, 0, 0, 0, 0, 0, 0, 0] should
        // produce a constant output of c * scale0 * 0.5 across all k
        // (since cos(pi*(2k+1)*0/16) = 1 for every k).
        let mut c = [0.0f32; 8];
        c[0] = 8.0;
        let out = idct1d_8(&c);
        // scale0 = 1/sqrt(2), 0.5 * 8 * 1/sqrt(2) = 2.828427...
        let expected = 0.5 * 8.0 / 2f32.sqrt();
        for (k, &v) in out.iter().enumerate() {
            assert!(
                approx_eq(v, expected, 1e-5),
                "k={k}: out={v} expected={expected}"
            );
        }
    }

    #[test]
    fn idct1d_8_ac1_first_position() {
        // AC[1] = 1 input. out[0] = 0.5 * cos(pi/16) ~ 0.49039.
        let mut c = [0.0f32; 8];
        c[1] = 1.0;
        let out = idct1d_8(&c);
        let expected = 0.5 * f32::cos(std::f32::consts::PI / 16.0);
        assert!(
            approx_eq(out[0], expected, 1e-5),
            "got {} expected {}",
            out[0],
            expected
        );
    }

    #[test]
    fn idct2d_dc_only_round_trip_through_dct() {
        // DC-only block produces a constant output.
        let mut c = [[0.0f32; 8]; 8];
        c[0][0] = 1.0;
        let out = idct2d_8x8(&c);
        // After 2-D IDCT the constant value is scale0^2 * 0.5 * 0.5 *
        // 1.0 = (1/sqrt(2))^2 * 0.25 = 0.5 * 0.25 = 0.125.
        let expected = 0.125;
        for row in out.iter() {
            for &v in row.iter() {
                assert!(approx_eq(v, expected, 1e-5), "got {v} expected {expected}");
            }
        }
    }

    // -------------------------------------------------------------
    // Round 129 — extract_lf_subblock + compose_lf_to_llf_block
    // -------------------------------------------------------------

    #[test]
    fn extract_lf_subblock_dct8x8_returns_single_sample() {
        // 4×4 LF channel; DCT8×8 reads a 1×1 sub-block at origin.
        let lf: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let sub = extract_lf_subblock(&lf, 4, 4, 2, 1, TransformType::Dct8x8).unwrap();
        assert_eq!(sub, vec![lf[4 + 2]]);
        assert_eq!(sub, vec![6.0]);
    }

    #[test]
    fn extract_lf_subblock_dct16x16_returns_2x2() {
        // 8×8 LF channel; DCT16×16 reads a 2×2 sub-block at (3, 4).
        let lf: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let sub = extract_lf_subblock(&lf, 8, 8, 3, 4, TransformType::Dct16x16).unwrap();
        // Row-major: (4,3), (4,4), (5,3), (5,4) at LF indices
        // 4*8+3=35, 4*8+4=36, 5*8+3=43, 5*8+4=44.
        assert_eq!(sub, vec![35.0, 36.0, 43.0, 44.0]);
    }

    #[test]
    fn extract_lf_subblock_dct32x32_returns_4x4() {
        // 16×16 LF channel; DCT32×32 reads a 4×4 sub-block.
        let lf: Vec<f32> = (0..16 * 16).map(|i| i as f32).collect();
        let sub = extract_lf_subblock(&lf, 16, 16, 0, 0, TransformType::Dct32x32).unwrap();
        assert_eq!(sub.len(), 16);
        // Row 0: 0..4. Row 1: 16..20. Row 2: 32..36. Row 3: 48..52.
        assert_eq!(&sub[..4], &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(&sub[4..8], &[16.0, 17.0, 18.0, 19.0]);
        assert_eq!(&sub[8..12], &[32.0, 33.0, 34.0, 35.0]);
        assert_eq!(&sub[12..16], &[48.0, 49.0, 50.0, 51.0]);
    }

    #[test]
    fn extract_lf_subblock_dct16x8_returns_1x2() {
        // DCT16x8: cy=2, cx=1. Reads a 2-row × 1-col block.
        let lf: Vec<f32> = (0..16).map(|i| i as f32).collect();
        // 4 wide × 4 tall. Origin (1, 0); reads (0,1) and (1,1) at
        // LF indices 0*4+1=1, 1*4+1=5.
        let sub = extract_lf_subblock(&lf, 4, 4, 1, 0, TransformType::Dct16x8).unwrap();
        assert_eq!(sub, vec![1.0, 5.0]);
    }

    #[test]
    fn extract_lf_subblock_dct8x16_returns_2x1() {
        // DCT8x16: cy=1, cx=2. Reads a 1-row × 2-col block.
        let lf: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let sub = extract_lf_subblock(&lf, 4, 4, 0, 2, TransformType::Dct8x16).unwrap();
        // Origin (0, 2) → row 2, cols 0..2: LF indices 8, 9.
        assert_eq!(sub, vec![8.0, 9.0]);
    }

    #[test]
    fn extract_lf_subblock_non_dct_returns_single_sample() {
        // Hornuss / DCT2×2 / DCT4×4 / DCT4×8 / DCT8×4 / AFV0..AFV3
        // all have (cx, cy) = (1, 1).
        let lf: Vec<f32> = (0..16).map(|i| i as f32).collect();
        for t in [
            TransformType::Hornuss,
            TransformType::Dct2x2,
            TransformType::Dct4x4,
            TransformType::Dct4x8,
            TransformType::Dct8x4,
            TransformType::Afv0,
            TransformType::Afv1,
            TransformType::Afv2,
            TransformType::Afv3,
        ] {
            let sub = extract_lf_subblock(&lf, 4, 4, 2, 2, t).unwrap();
            assert_eq!(sub.len(), 1, "{t:?}");
            assert_eq!(sub[0], lf[2 * 4 + 2], "{t:?}");
        }
    }

    #[test]
    fn extract_lf_subblock_rejects_oob_origin() {
        // 4×4 LF grid. DCT16×16 needs 2×2; origin (3, 3) reads (3..5,
        // 3..5) which extends past the 4×4 grid.
        let lf: Vec<f32> = vec![0.0; 16];
        let err = extract_lf_subblock(&lf, 4, 4, 3, 3, TransformType::Dct16x16);
        assert!(err.is_err());
    }

    #[test]
    fn extract_lf_subblock_at_grid_corner_works() {
        // 4×4 LF, DCT32×32 (4×4 block). Origin (0, 0) fits exactly.
        let lf: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let sub = extract_lf_subblock(&lf, 4, 4, 0, 0, TransformType::Dct32x32).unwrap();
        assert_eq!(sub.len(), 16);
        assert_eq!(sub, lf);
    }

    #[test]
    fn extract_lf_subblock_rejects_dim_mismatch() {
        // `lf_samples` length doesn't match `lf_width * lf_height`.
        let lf = vec![0.0f32; 15];
        let err = extract_lf_subblock(&lf, 4, 4, 0, 0, TransformType::Dct8x8);
        assert!(err.is_err());
    }

    #[test]
    fn compose_lf_to_llf_block_dct8x8_matches_input_sample() {
        // For DCT8×8 (1×1 LF block), llf_from_lf returns the single
        // sample scaled by ScaleF(1, 8, 0)^2 = 1.0 — i.e. the input
        // value unchanged. So compose_lf_to_llf_block(... Dct8x8)
        // should return the LF sample at (bx, by).
        let lf: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let llf = compose_lf_to_llf_block(&lf, 4, 4, 1, 2, TransformType::Dct8x8).unwrap();
        assert_eq!(llf.len(), 1);
        // ScaleF(1, 8, 0) = 1.0 exactly per `llf_from_lf::scale_f`.
        let expected = lf[2 * 4 + 1];
        assert!(
            (llf[0] - expected).abs() < 1e-6,
            "got {} want {}",
            llf[0],
            expected
        );
    }

    #[test]
    fn compose_lf_to_llf_block_non_dct_pass_through() {
        // For the non-DCT transforms (1×1 LF, identity LF→LLF map),
        // the composed result is the LF sample at (bx, by) verbatim.
        let lf: Vec<f32> = (0..16).map(|i| (i as f32) - 8.0).collect();
        for t in [
            TransformType::Hornuss,
            TransformType::Dct2x2,
            TransformType::Dct4x4,
            TransformType::Dct4x8,
            TransformType::Dct8x4,
            TransformType::Afv0,
            TransformType::Afv1,
            TransformType::Afv2,
            TransformType::Afv3,
        ] {
            let llf = compose_lf_to_llf_block(&lf, 4, 4, 3, 1, t).unwrap();
            assert_eq!(llf.len(), 1, "{t:?}");
            assert_eq!(llf[0], lf[4 + 3], "{t:?}");
        }
    }

    #[test]
    fn compose_lf_to_llf_block_dct16x16_constant_block_has_dc_only() {
        // 4×4 LF grid, all samples = 7.0. A 2×2 sub-block at any
        // origin is also all 7.0; the LLF block of a constant is
        // dc-only at out[0] = 7.0 * ScaleF(2, 16, 0)^2.
        let lf = vec![7.0f32; 16];
        let llf = compose_lf_to_llf_block(&lf, 4, 4, 0, 0, TransformType::Dct16x16).unwrap();
        assert_eq!(llf.len(), 4);
        let sf = crate::llf_from_lf::scale_f(2, 16, 0);
        let dc = 7.0 * sf * sf;
        assert!((llf[0] - dc).abs() < 1e-5, "DC {} != {}", llf[0], dc);
        for v in &llf[1..] {
            assert!(v.abs() < 1e-4, "AC = {v}, expected 0");
        }
    }

    #[test]
    fn compose_lf_to_llf_block_3ch_requires_matching_dims() {
        use crate::lf_dequant::LfDequantOutput;
        // Mismatched widths → rejection.
        let lf = LfDequantOutput {
            samples: [vec![0.0; 16], vec![0.0; 8], vec![0.0; 16]],
            widths: [4, 4, 4],
            heights: [4, 2, 4],
        };
        let err = compose_lf_to_llf_block_3ch(&lf, 0, 0, TransformType::Dct8x8);
        assert!(err.is_err());
    }

    #[test]
    fn compose_lf_to_llf_block_3ch_produces_3_channels() {
        use crate::lf_dequant::LfDequantOutput;
        let lf = LfDequantOutput {
            samples: [vec![1.0f32; 16], vec![2.0f32; 16], vec![3.0f32; 16]],
            widths: [4, 4, 4],
            heights: [4, 4, 4],
        };
        let blocks = compose_lf_to_llf_block_3ch(&lf, 0, 0, TransformType::Dct8x8).unwrap();
        // DCT8×8 LLF is `dc * ScaleF(1, 8, 0)^2`; ScaleF(1, 8, 0) =
        // 1.0 exactly so the result equals the input — but allow
        // f32 round-off.
        for (c, &want) in [1.0f32, 2.0, 3.0].iter().enumerate() {
            assert_eq!(blocks[c].len(), 1, "channel {c}");
            assert!(
                (blocks[c][0] - want).abs() < 1e-5,
                "channel {c}: got {} want {}",
                blocks[c][0],
                want,
            );
        }
    }

    #[test]
    fn compose_lf_to_llf_block_3ch_dct16x16_constant() {
        use crate::lf_dequant::LfDequantOutput;
        let lf = LfDequantOutput {
            samples: [vec![5.0f32; 16], vec![10.0f32; 16], vec![15.0f32; 16]],
            widths: [4, 4, 4],
            heights: [4, 4, 4],
        };
        let blocks = compose_lf_to_llf_block_3ch(&lf, 1, 1, TransformType::Dct16x16).unwrap();
        let sf = crate::llf_from_lf::scale_f(2, 16, 0);
        for (c, &dc_in) in [5.0, 10.0, 15.0].iter().enumerate() {
            assert_eq!(blocks[c].len(), 4, "channel {c}");
            let expected_dc = dc_in * sf * sf;
            assert!(
                (blocks[c][0] - expected_dc).abs() < 1e-5,
                "channel {c}: DC {} != {}",
                blocks[c][0],
                expected_dc,
            );
            for v in &blocks[c][1..] {
                assert!(v.abs() < 1e-4, "channel {c}: AC = {v}");
            }
        }
    }
}
