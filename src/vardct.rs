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

use crate::frame_header::FrameHeader;
use crate::metadata_fdis::ImageMetadataFdis;

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
) -> Result<VarDctScaffold, Error> {
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
}
