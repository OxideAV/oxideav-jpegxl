//! `PassGroup` bundle — ISO/IEC 18181-1:2024 Annex G.4.
//!
//! A `PassGroup` carries the per-pass × per-group residuals for any
//! Modular channel that wasn't already decoded by GlobalModular or
//! ModularLfGroup, plus the VarDCT HF coefficients (when
//! `encoding == kVarDCT`).
//!
//! Round 6 ships **type scaffolding only**: parser stubs return
//! `Error::Unsupported`. See [`crate::lf_group`] for the round-7
//! coordination plan that has to land before this module's parser can
//! be wired.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::frame_header::FrameHeader;

/// `PassGroup` bundle — Table G.5.
#[derive(Debug, Clone)]
pub struct PassGroup {
    /// `(pass_index, group_index)` within the frame.
    pub pass_index: u32,
    pub group_index: u32,
    // Modular group data (G.4.2). VarDCT HF coefficients (I.4) are
    // separate and not represented here in round 6.
    pub modular_group: ModularGroupData,
}

/// `ModularGroupData` (G.4.2). Holds per-group modular residuals for
/// any channel in the partially decoded GlobalModular image whose
/// `min(hshift, vshift)` is in `[minshift, maxshift)`.
#[derive(Debug, Clone)]
pub struct ModularGroupData {
    /// Frame-coordinates rectangle this group covers.
    pub x_origin: u32,
    pub y_origin: u32,
    pub width: u32,
    pub height: u32,
}

impl PassGroup {
    /// Decode the PassGroup bundle at `(pass_index, group_index)` per
    /// Table G.5. Returns `Error::Unsupported` in round 6.
    pub fn read(
        _br: &mut BitReader<'_>,
        fh: &FrameHeader,
        pass_index: u32,
        group_index: u32,
    ) -> Result<Self> {
        let num_groups = fh.num_groups();
        let num_passes = fh.passes.num_passes;
        if group_index as u64 >= num_groups {
            return Err(Error::InvalidData(format!(
                "JXL PassGroup: group {group_index} >= num_groups {num_groups}"
            )));
        }
        if pass_index >= num_passes {
            return Err(Error::InvalidData(format!(
                "JXL PassGroup: pass {pass_index} >= num_passes {num_passes}"
            )));
        }
        Err(Error::Unsupported(
            "JXL PassGroup: per-pass per-group decode not yet wired (round 7 follow-up)".into(),
        ))
    }
}

impl ModularGroupData {
    /// Compute the group's pixel rectangle in frame coordinates from
    /// `group_index` and `frame_header`. The frame is split into a
    /// grid of `group_dim × group_dim`-sized cells.
    pub fn rect_for_index(fh: &FrameHeader, group_index: u32) -> Result<Self> {
        let g = fh.group_dim();
        let num_groups_x = fh.width.div_ceil(g);
        let num_groups_y = fh.height.div_ceil(g);
        let total = num_groups_x as u64 * num_groups_y as u64;
        if group_index as u64 >= total {
            return Err(Error::InvalidData(format!(
                "JXL PassGroup: group {group_index} out of grid {num_groups_x}x{num_groups_y}"
            )));
        }
        let grid_x = group_index % num_groups_x;
        let grid_y = group_index / num_groups_x;
        let x_origin = grid_x * g;
        let y_origin = grid_y * g;
        let w = (fh.width - x_origin).min(g);
        let h = (fh.height - y_origin).min(g);
        Ok(Self {
            x_origin,
            y_origin,
            width: w,
            height: h,
        })
    }
}

/// Compute `(minshift, maxshift)` per G.4.2 first paragraph.
///
/// * If `pass_index == 0` (or this is the only pass), `maxshift = 3`.
/// * Otherwise `maxshift = minshift_of_previous_pass`.
/// * `minshift = log2(downsample[n])` if there exists `n` such that
///   `current_pass_index == last_pass[n]`; otherwise `minshift =
///   maxshift` (this pass contains no modular data).
///
/// `downsample` and `last_pass` are arrays of length `num_passes - 1`
/// supplied by `frame_header.passes`. Returns `(minshift, maxshift)`.
pub fn compute_pass_shift_range(
    pass_index: u32,
    downsample: &[u32],
    last_pass: &[u32],
) -> (u32, u32) {
    let mut prev_minshift: u32 = 3; // seed for pass 0
    let mut minshift: u32 = 3;
    let mut maxshift: u32 = 3;
    for p in 0..=pass_index {
        // For pass 0, maxshift = 3. For subsequent passes, maxshift =
        // previous pass's minshift.
        maxshift = if p == 0 { 3 } else { prev_minshift };
        // Compute minshift for this pass: take the smallest
        // log2(downsample[n]) over all `n` with `last_pass[n] == p`.
        let mut found_for_this = false;
        let mut min_log_ds = u32::MAX;
        for (n, lp) in last_pass.iter().enumerate() {
            if *lp == p {
                let ds = downsample.get(n).copied().unwrap_or(1).max(1);
                let log_ds = u32::BITS - 1 - ds.leading_zeros();
                if log_ds < min_log_ds {
                    min_log_ds = log_ds;
                }
                found_for_this = true;
            }
        }
        minshift = if found_for_this {
            min_log_ds
        } else {
            // No match: this pass contains no modular data; minshift = maxshift.
            maxshift
        };
        prev_minshift = minshift;
    }
    (minshift, maxshift)
}

/// Reject a multi-group frame at decode time with a precise message.
pub fn unsupported_multi_group_error(num_groups: u64, num_passes: u32) -> Error {
    Error::Unsupported(format!(
        "jxl decoder (round 6): num_groups = {num_groups} × num_passes = {num_passes} — \
         per-PassGroup decode (Annex G.4) is round-7 work; this round only handles \
         single-group, single-pass frames"
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
    fn rect_for_single_group_default_group_dim() {
        let fh = build_fh(64, 64);
        let r = ModularGroupData::rect_for_index(&fh, 0).unwrap();
        assert_eq!((r.x_origin, r.y_origin, r.width, r.height), (0, 0, 64, 64));
    }

    #[test]
    fn rect_for_2x2_grid() {
        // 512×512 image at default group_dim=256 → 2x2 grid.
        let mut fh = build_fh(512, 512);
        fh.group_size_shift = 1; // group_dim = 256
        let r0 = ModularGroupData::rect_for_index(&fh, 0).unwrap();
        let r1 = ModularGroupData::rect_for_index(&fh, 1).unwrap();
        let r2 = ModularGroupData::rect_for_index(&fh, 2).unwrap();
        let r3 = ModularGroupData::rect_for_index(&fh, 3).unwrap();
        assert_eq!((r0.x_origin, r0.y_origin), (0, 0));
        assert_eq!((r1.x_origin, r1.y_origin), (256, 0));
        assert_eq!((r2.x_origin, r2.y_origin), (0, 256));
        assert_eq!((r3.x_origin, r3.y_origin), (256, 256));
        for r in [r0, r1, r2, r3] {
            assert_eq!((r.width, r.height), (256, 256));
        }
    }

    #[test]
    fn rect_out_of_range_errors() {
        let fh = build_fh(64, 64);
        assert!(ModularGroupData::rect_for_index(&fh, 1).is_err());
    }

    #[test]
    fn pass_group_read_errors_in_round_6() {
        let fh = build_fh(64, 64);
        let bytes = vec![0u8; 16];
        let mut br = BitReader::new(&bytes);
        let r = PassGroup::read(&mut br, &fh, 0, 0);
        assert!(matches!(r, Err(Error::Unsupported(_))));
    }

    #[test]
    fn pass_group_read_rejects_out_of_range_pass() {
        let fh = build_fh(64, 64);
        let bytes = vec![0u8; 16];
        let mut br = BitReader::new(&bytes);
        let r = PassGroup::read(&mut br, &fh, 99, 0);
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn shift_range_single_pass_default() {
        // Single pass: pass_index=0, no downsample/last_pass entries.
        // Per spec: maxshift=3 (only pass), minshift = if no `n` with
        // last_pass[n] == 0 then minshift = maxshift = 3.
        let (minshift, maxshift) = compute_pass_shift_range(0, &[], &[]);
        assert_eq!(minshift, 3);
        assert_eq!(maxshift, 3);
    }

    #[test]
    fn shift_range_two_passes_no_downsample() {
        // pass 0: no last_pass match → minshift = maxshift = 3.
        // pass 1: maxshift = 3, no match → minshift = 3.
        // (Unusual config; spec uses downsample to drive resolution
        //  pyramid passes.)
        let (m0, x0) = compute_pass_shift_range(0, &[1], &[0]);
        // last_pass[0] == 0 → log2(downsample[0])=log2(1)=0 → minshift=0.
        assert_eq!((m0, x0), (0, 3));
    }
}
