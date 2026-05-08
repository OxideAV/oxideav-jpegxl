//! `PassGroup` bundle — ISO/IEC 18181-1:2024 Annex G.4.
//!
//! A `PassGroup` carries the per-pass × per-group residuals for any
//! Modular channel that wasn't already decoded by GlobalModular or
//! ModularLfGroup, plus the VarDCT HF coefficients (when
//! `encoding == kVarDCT`).
//!
//! Round 7 wires the modular per-group decode end-to-end: each PassGroup
//! reads its own ModularHeader (Table H.1) for the channel rectangle
//! intersecting that group, and the decoded samples are copied back
//! into the partially decoded GlobalModular image at the correct frame
//! coordinates. Per G.4.2 last paragraph, inverse transforms run AFTER
//! all PassGroups complete (driven by `decode_codestream`).

use oxideav_core::{Error, Result};

use crate::bitreader::{BitReader, U32Dist};
use crate::frame_header::FrameHeader;
use crate::lf_global::LfGlobal;
use crate::modular_fdis::{
    decode_channels_at_stream, ChannelDesc, MaTreeFdis, TransformInfo, WpHeader,
};

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
    /// Compute the bundle's geometry without parsing the bitstream.
    /// Used by tests + the LfGroup geometry helpers; the real per-group
    /// decode is in [`decode_modular_group_into`].
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
        let modular_group = ModularGroupData::rect_for_index(fh, group_index)?;
        Ok(Self {
            pass_index,
            group_index,
            modular_group,
        })
    }
}

/// Decode a single PassGroup's modular sub-bitstream and copy the
/// decoded samples into `lf_global.global_modular.image` at the correct
/// frame-coordinates rectangle (G.4.2).
///
/// `br` is positioned at the start of this PassGroup's TOC slot (a
/// fresh, byte-aligned sub-reader covering exactly the slot's bytes).
/// `lf_global` is mutated in-place.
pub fn decode_modular_group_into(
    br: &mut BitReader<'_>,
    fh: &FrameHeader,
    lf_global: &mut LfGlobal,
    pass_index: u32,
    group_index: u32,
) -> Result<()> {
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
    let rect = ModularGroupData::rect_for_index(fh, group_index)?;
    let group_dim = fh.group_dim();
    let num_lf_groups = fh.num_lf_groups();

    // Compute (minshift, maxshift) for this pass per G.4.2.
    let (minshift, maxshift) = compute_pass_shift_range(
        pass_index,
        &fh.passes.downsample,
        &fh.passes.last_pass,
        num_passes,
    );

    // Determine which channels of the partially-decoded GlobalModular
    // image have data in this PassGroup. Per G.4.2: a channel
    // contributes when:
    //   * it is NOT a meta-channel,
    //   * its dimensions exceed group_dim (i.e. it wasn't decoded
    //     entirely inside GlobalModular),
    //   * hshift < 3 OR vshift < 3 (otherwise it's an LfGroup channel),
    //   * minshift <= min(hshift, vshift) < maxshift,
    //   * it has not already been decoded in a previous pass.
    let descs_full = lf_global.global_modular.image.descs.clone();
    let nb_meta = lf_global.global_modular.nb_meta_channels;

    // For each contributing channel, compute the per-group channel
    // descriptor: dims = (rect.width >> hshift, rect.height >> vshift)
    // — actually right-shifted offsets per the spec, dims taken at the
    // group rectangle. Per spec: "the group dimensions and the x,y
    // offsets are right-shifted by hshift (for x and width) and vshift
    // (for y and height)".
    struct PerGroupChannel {
        full_idx: usize,
        x0: u32,
        y0: u32,
        desc: ChannelDesc,
    }
    let mut contributing: Vec<PerGroupChannel> = Vec::new();
    for (full_idx, d) in descs_full.iter().enumerate() {
        if full_idx < nb_meta {
            continue;
        }
        // Channel was fully decoded in GlobalModular — skip.
        if d.width <= group_dim && d.height <= group_dim {
            continue;
        }
        // LfGroup channel — skip.
        if d.hshift >= 3 && d.vshift >= 3 {
            continue;
        }
        let m = d.hshift.min(d.vshift);
        // Pass-shift gate: minshift <= min(hshift, vshift) < maxshift.
        if !((minshift as i32) <= m && m < (maxshift as i32)) {
            continue;
        }
        // Compute group rect in this channel's coordinates.
        let hs = d.hshift.max(0) as u32;
        let vs = d.vshift.max(0) as u32;
        let x0 = rect.x_origin >> hs;
        let y0 = rect.y_origin >> vs;
        let w = rect.width >> hs;
        let h = rect.height >> vs;
        if w == 0 || h == 0 {
            continue;
        }
        contributing.push(PerGroupChannel {
            full_idx,
            x0,
            y0,
            desc: ChannelDesc {
                width: w,
                height: h,
                hshift: d.hshift,
                vshift: d.vshift,
            },
        });
    }

    if contributing.is_empty() {
        // Nothing to decode in this group; the TOC slot may still
        // contain padding, but per spec the slot byte length is fixed
        // and the bit reader will simply not read further.
        return Ok(());
    }

    // Read the inner ModularHeader (Table H.1).
    let inner_use_global_tree = br.read_bool()?;
    let wp_header = WpHeader::read(br)?;
    let nb_transforms = br.read_u32([
        U32Dist::Val(0),
        U32Dist::Val(1),
        U32Dist::BitsOffset(4, 2),
        U32Dist::BitsOffset(8, 18),
    ])?;
    const MAX_TRANSFORMS: u32 = 274;
    if nb_transforms > MAX_TRANSFORMS {
        return Err(Error::InvalidData(format!(
            "JXL PassGroup: nb_transforms {nb_transforms} exceeds {MAX_TRANSFORMS}"
        )));
    }
    let mut transforms: Vec<TransformInfo> = Vec::with_capacity(nb_transforms as usize);
    for _ in 0..nb_transforms {
        transforms.push(TransformInfo::read(br)?);
    }
    if !transforms.is_empty() {
        return Err(Error::Unsupported(
            "JXL PassGroup: transforms inside per-group ModularHeader not supported (round 7 \
             scope: lossless single-channel grey, no nested transforms)"
                .into(),
        ));
    }

    // Resolve the MA tree to use.
    let mut tree = if inner_use_global_tree {
        lf_global
            .global_modular
            .global_tree
            .as_ref()
            .ok_or_else(|| {
                Error::InvalidData(
                    "JXL PassGroup: inner sub-bitstream wants global tree but none was decoded \
                     in GlobalModular"
                        .into(),
                )
            })?
            .cloned_with_fresh_state()
    } else {
        MaTreeFdis::read(br)?
    };

    // Build the per-group descs list from the contributing channels.
    let group_descs: Vec<ChannelDesc> = contributing.iter().map(|c| c.desc).collect();

    // stream_index per Table H.4 last paragraph.
    let stream_index =
        (1 + 3 * num_lf_groups + 17 + num_groups * pass_index as u64 + group_index as u64) as i32;

    // Decode the modular group's channels.
    let group_image =
        decode_channels_at_stream(br, &group_descs, &mut tree, &wp_header, stream_index)?;

    // Copy decoded samples back into the parent image at the per-channel
    // rectangle (x0, y0, w, h). Per spec: "The decoded modular group
    // data is then copied into the partially decoded GlobalModular
    // image in the corresponding positions."
    for (k, c) in contributing.iter().enumerate() {
        let g_chan = &group_image.channels[k];
        let g_w = group_image.descs[k].width as usize;
        let g_h = group_image.descs[k].height as usize;
        let parent_desc = lf_global.global_modular.image.descs[c.full_idx];
        let parent_w = parent_desc.width as usize;
        let parent_chan = &mut lf_global.global_modular.image.channels[c.full_idx];
        for y in 0..g_h {
            let src_off = y * g_w;
            let dst_off = (c.y0 as usize + y) * parent_w + c.x0 as usize;
            parent_chan[dst_off..dst_off + g_w].copy_from_slice(&g_chan[src_off..src_off + g_w]);
        }
    }
    Ok(())
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
/// **Spec gap (round 7):** For the FINAL pass (`pass_index ==
/// num_passes - 1`) the spec is silent on how `minshift` resolves down
/// to `0` so that full-resolution channel data can be carried. The
/// only consistent reading is the implicit `n = num_ds` entry with
/// `downsample[num_ds] = 1` and `last_pass[num_ds] = num_passes - 1`.
/// Without this implicit entry, single-pass frames would have
/// `(minshift, maxshift) = (3, 3)` and the criterion
/// `minshift <= min(hshift, vshift) < maxshift` could never be true
/// for a typical hshift=0 channel, so no PassGroup would ever decode
/// any data. This implementation models that implicit entry to make
/// the gate work as intended.
///
/// `downsample` and `last_pass` are arrays of length `num_ds <
/// num_passes` supplied by `frame_header.passes`. Returns
/// `(minshift, maxshift)`.
pub fn compute_pass_shift_range(
    pass_index: u32,
    downsample: &[u32],
    last_pass: &[u32],
    num_passes: u32,
) -> (u32, u32) {
    let mut prev_minshift: u32 = 3; // seed for pass 0
    let mut minshift: u32 = 3;
    let mut maxshift: u32 = 3;
    let final_pass = num_passes.saturating_sub(1);
    for p in 0..=pass_index {
        // For pass 0, maxshift = 3. For subsequent passes, maxshift =
        // previous pass's minshift.
        maxshift = if p == 0 { 3 } else { prev_minshift };
        // Compute minshift for this pass: take the smallest
        // log2(downsample[n]) over all `n` with `last_pass[n] == p`,
        // PLUS the implicit `n=num_ds` entry that targets the final
        // pass at downsample=1 (full resolution).
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
        // Implicit final-pass entry (see SPECGAP above).
        if p == final_pass {
            // log2(1) = 0
            if 0 < min_log_ds {
                min_log_ds = 0;
            }
            found_for_this = true;
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
    fn pass_group_read_succeeds_round_7() {
        // Round 7: PassGroup::read no longer rejects; it returns the
        // bundle's geometry. The actual modular sub-bitstream decode
        // is in `decode_modular_group_into`.
        let fh = build_fh(64, 64);
        let bytes = vec![0u8; 16];
        let mut br = BitReader::new(&bytes);
        let pg = PassGroup::read(&mut br, &fh, 0, 0).unwrap();
        assert_eq!(pg.pass_index, 0);
        assert_eq!(pg.group_index, 0);
        assert_eq!(pg.modular_group.width, 64);
        assert_eq!(pg.modular_group.height, 64);
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
    fn shift_range_single_pass_implicit_final() {
        // Single pass: pass_index=0, num_passes=1, no downsample/
        // last_pass entries. Per round-7 SPECGAP reading: implicit
        // final-pass entry kicks in (downsample=1, last_pass=0) →
        // minshift=0. maxshift=3 (only pass).
        let (minshift, maxshift) = compute_pass_shift_range(0, &[], &[], 1);
        assert_eq!(minshift, 0);
        assert_eq!(maxshift, 3);
    }

    #[test]
    fn shift_range_two_passes_first_pass() {
        // pass 0 of two-pass with downsample=[1] last_pass=[0]:
        //   last_pass[0] == 0 → log2(1) = 0 → minshift=0.
        //   maxshift=3.
        let (m0, x0) = compute_pass_shift_range(0, &[1], &[0], 2);
        assert_eq!((m0, x0), (0, 3));
    }

    #[test]
    fn shift_range_two_passes_implicit_final() {
        // pass 1 (final) of two-pass with downsample=[2] last_pass=[0]:
        //   pass 0 → minshift=log2(2)=1, maxshift=3.
        //   pass 1 → maxshift=1; implicit final-pass entry
        //   (downsample=1, last_pass=1) → minshift=0.
        let (m1, x1) = compute_pass_shift_range(1, &[2], &[0], 2);
        assert_eq!((m1, x1), (0, 1));
    }
}
