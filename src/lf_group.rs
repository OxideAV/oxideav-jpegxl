//! `LfGroup` bundle — ISO/IEC 18181-1:2024 Annex G.2.
//!
//! For an `kModular` frame the bundle reduces to a single
//! [`ModularLfGroup`] (G.2.3). For `kVarDCT` the bundle additionally
//! contains [`LfCoefficients`] (G.2.2) and [`HfMetadata`] (G.2.4).
//!
//! Round 6 ships the **type scaffolding only**: the bitstream parser
//! is stubbed with `Error::Unsupported`. Wiring the per-LfGroup decode
//! (and the per-PassGroup decode that follows it in the TOC layout)
//! is round-7 work, gated on first refactoring [`crate::global_modular::GlobalModular`]
//! to honour the spec's "stop decoding when remaining channel size
//! exceeds `group_dim`" rule (last paragraph of §G.1.3).
//!
//! ## Why this is round-6 scaffold-only
//!
//! Multi-group decode requires four coordinated pieces that all change
//! the existing single-group path:
//!
//! 1. `GlobalModular::read` must stop after `nb_meta_channels` plus any
//!    channel ≤ `group_dim` (G.1.3 last paragraph).
//! 2. The Modular sub-bitstream's `stream_index` property (Table H.4
//!    property 1) must reflect which sub-bitstream is being decoded —
//!    `0` for GlobalModular, `1 + lf_group_idx` for LfCoefficients,
//!    `1 + num_lf_groups + lf_group_idx` for ModularLfGroup, etc.
//!    Currently `decode_channels` hard-codes `stream_index = 0`.
//! 3. The TOC entry order (§F.3) must be permuted-aware; currently the
//!    decoder rejects any TOC with `entries.len() != 1`.
//! 4. Inverse transforms run AFTER all PassGroups complete, not after
//!    GlobalModular (last sentence of G.4.2).
//!
//! Doing (1)..(4) in a single round risks regressing the five
//! pixel-correct fixtures decoder rounds 1..5 stabilised. They are
//! listed here so round 7 can attack them as a coordinated unit.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::frame_header::{Encoding, FrameHeader};

/// LfGroup index within the frame, computed from
/// `(grid_y * num_lf_columns + grid_x)` per the spec's G.1.3 channel
/// stride model.
pub type LfGroupIndex = u32;

/// `LfGroup` bundle — Table G.3.
///
/// All coordinates inside this clause are relative to the top-left
/// corner of the current LF group, not the frame.
#[derive(Debug, Clone)]
pub struct LfGroup {
    /// LF coefficients (G.2.2). Only populated when `encoding == kVarDCT`.
    pub lf_coeff: Option<LfCoefficients>,
    /// Modular LF group residuals (G.2.3). Always populated.
    pub mlf_group: ModularLfGroup,
    /// HF metadata (G.2.4). Only populated when `encoding == kVarDCT`.
    pub hf_meta: Option<HfMetadata>,
}

/// `LfCoefficients` (G.2.2) — `kVarDCT` only. Round 7+ work.
#[derive(Debug, Clone)]
pub struct LfCoefficients {
    /// `extra_precision = u(2)` — additional fixed-point precision
    /// for the dequantised LF channels.
    pub extra_precision: u32,
    // The actual three modular sub-bitstream channels (X', Y', B' or
    // similar) follow as ceil(height/8) × ceil(width/8) integer arrays.
}

/// `ModularLfGroup` (G.2.3). Holds the per-LfGroup residuals for any
/// channel in the partially decoded GlobalModular image whose `hshift,
/// vshift` are both ≥ 3.
#[derive(Debug, Clone)]
pub struct ModularLfGroup {
    /// LfGroup index within the frame (0-based, raster order).
    pub lf_group_index: LfGroupIndex,
    /// Width of this LfGroup in pixels (capped at `8 * group_dim`).
    pub lf_group_width: u32,
    /// Height of this LfGroup in pixels.
    pub lf_group_height: u32,
}

/// `HfMetadata` (G.2.4) — `kVarDCT` only. Round 7+ work.
#[derive(Debug, Clone)]
pub struct HfMetadata {
    /// `nb_blocks - 1` is read as a `u(ceil(log2(ceil(width / 8) *
    /// ceil(height / 8))))` per spec.
    pub nb_blocks: u32,
}

impl LfGroup {
    /// Decode the LfGroup bundle at index `lf_group_index` per Table
    /// G.3. Returns `Error::Unsupported` in round 6 — the four
    /// coordination items above must land first.
    pub fn read(
        _br: &mut BitReader<'_>,
        fh: &FrameHeader,
        lf_group_index: LfGroupIndex,
    ) -> Result<Self> {
        let num_lf_groups = fh.num_lf_groups();
        if lf_group_index as u64 >= num_lf_groups {
            return Err(Error::InvalidData(format!(
                "JXL LfGroup: index {lf_group_index} >= num_lf_groups {num_lf_groups}"
            )));
        }
        Err(Error::Unsupported(
            "JXL LfGroup: per-LfGroup decode not yet wired (round 7 follow-up — see crate-level docs)".into(),
        ))
    }
}

impl ModularLfGroup {
    /// Compute the LfGroup pixel rectangle for a given
    /// `lf_group_index`. The frame is split into a grid of
    /// `8 × group_dim`-sized cells; the last column / row may be
    /// smaller if `frame_width` / `frame_height` is not a multiple of
    /// `8 × group_dim`.
    ///
    /// Returns `(x_origin, y_origin, lf_group_width, lf_group_height)`
    /// — origin in frame coordinates, dims in pixels.
    pub fn rect_for_index(
        fh: &FrameHeader,
        lf_group_index: LfGroupIndex,
    ) -> Result<(u32, u32, u32, u32)> {
        let lf_group_dim = fh
            .group_dim()
            .checked_mul(8)
            .ok_or_else(|| Error::InvalidData("JXL LfGroup: group_dim * 8 overflow".into()))?;
        let num_lf_columns = fh.width.div_ceil(lf_group_dim);
        let num_lf_rows = fh.height.div_ceil(lf_group_dim);
        let total = num_lf_columns as u64 * num_lf_rows as u64;
        if lf_group_index as u64 >= total {
            return Err(Error::InvalidData(format!(
                "JXL LfGroup: index {lf_group_index} out of grid {num_lf_columns}x{num_lf_rows}"
            )));
        }
        let grid_x = lf_group_index % num_lf_columns;
        let grid_y = lf_group_index / num_lf_columns;
        let x_origin = grid_x * lf_group_dim;
        let y_origin = grid_y * lf_group_dim;
        let w = (fh.width - x_origin).min(lf_group_dim);
        let h = (fh.height - y_origin).min(lf_group_dim);
        Ok((x_origin, y_origin, w, h))
    }
}

/// Reject a multi-LfGroup frame at decode time with a precise message
/// pinpointing the round-7 follow-up. Used by the top-level
/// `decode_codestream` when `num_lf_groups > 1`.
pub fn unsupported_multi_lf_group_error(num_lf_groups: u64, encoding: Encoding) -> Error {
    Error::Unsupported(format!(
        "jxl decoder (round 6): num_lf_groups = {num_lf_groups} (encoding = {encoding:?}) — \
         per-LfGroup decode (Annex G.2) is round-7 work; this round only handles single-LfGroup frames"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_header::FrameDecodeParams;

    fn build_fh(w: u32, h: u32) -> FrameHeader {
        let params = FrameDecodeParams {
            xyb_encoded: false,
            num_extra_channels: 0,
            have_animation: false,
            have_animation_timecodes: false,
            image_width: w,
            image_height: h,
        };
        let bytes = crate::ans::test_helpers::pack_lsb(&[(1, 1)]);
        let mut br = BitReader::new(&bytes);
        let mut fh = FrameHeader::read(&mut br, &params).unwrap();
        fh.width = w;
        fh.height = h;
        fh
    }

    #[test]
    fn rect_for_single_group_origin_zero() {
        let fh = build_fh(64, 64);
        let (x, y, w, h) = ModularLfGroup::rect_for_index(&fh, 0).unwrap();
        assert_eq!((x, y, w, h), (0, 0, 64, 64));
    }

    #[test]
    fn rect_for_index_out_of_range_errors() {
        let fh = build_fh(64, 64);
        assert!(ModularLfGroup::rect_for_index(&fh, 1).is_err());
    }

    #[test]
    fn rect_for_2x1_grid_at_default_group_dim() {
        // group_dim = 256 by default → lf_group_dim = 2048. So a
        // 4096x256 image has 2 LfGroups horizontally, 1 vertically.
        let mut fh = build_fh(4096, 256);
        fh.group_size_shift = 1; // group_dim 256 (default)
        let (x0, y0, w0, h0) = ModularLfGroup::rect_for_index(&fh, 0).unwrap();
        assert_eq!((x0, y0, w0, h0), (0, 0, 2048, 256));
        let (x1, y1, w1, h1) = ModularLfGroup::rect_for_index(&fh, 1).unwrap();
        assert_eq!((x1, y1, w1, h1), (2048, 0, 2048, 256));
    }

    #[test]
    fn lf_group_read_errors_in_round_6() {
        let fh = build_fh(64, 64);
        let bytes = vec![0u8; 16];
        let mut br = BitReader::new(&bytes);
        let r = LfGroup::read(&mut br, &fh, 0);
        assert!(matches!(r, Err(Error::Unsupported(_))));
    }

    #[test]
    fn lf_group_read_rejects_out_of_range_index() {
        let fh = build_fh(64, 64);
        let bytes = vec![0u8; 16];
        let mut br = BitReader::new(&bytes);
        let r = LfGroup::read(&mut br, &fh, 99);
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }
}
