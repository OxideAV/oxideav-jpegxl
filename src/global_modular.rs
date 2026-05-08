//! `GlobalModular` bundle — FDIS 18181-1 §C.4.8.
//!
//! Decodes a single Modular sub-bitstream (§C.9) holding all channels
//! whose dimensions fit in `kGroupDim × kGroupDim` (i.e. small images
//! like the round-3 cjxl `8×8` fixture). For larger images the bulk of
//! the pixel data lives in LfGroups and PassGroups (rounds 4+).
//!
//! Per spec, the channel layout depends on the frame's encoding:
//! * **Modular**: first comes Grey (1 channel) or RGB (3 channels) or
//!   Y'X'B' (3 channels) or YCbCr (3 channels), then any extra channels
//!   in ascending index order.
//! * **VarDCT**: 0 colour channels in GlobalModular — only extra
//!   channels live here (round 4 territory).
//!
//! Round-3 supports only the Modular case with a single Grey channel
//! (the simplest cjxl can emit). RGB and extras parse the channel
//! descriptions but reject early in `decode_channels`.
//!
//! Allocation bound: number of channels is computed from the metadata
//! (Grey vs RGB) and bounded by `MAX_CHANNELS` from
//! [`crate::modular_fdis`]. Per-channel pixel allocation is bounded by
//! `width × height` against the bit reader's remaining input length.

use oxideav_core::{Error, Result};

use crate::bitreader::{BitReader, U32Dist};
use crate::frame_header::{Encoding, FrameHeader};
use crate::metadata_fdis::{ColourSpace, ImageMetadataFdis};
use crate::modular_fdis::{
    decode_channels_at_stream, horiz_isqueeze, inverse_palette, inverse_rct, vert_isqueeze,
    ChannelDesc, MaTreeFdis, ModularImage, SqueezeParam, TransformId, TransformInfo, WpHeader,
    MAX_CHANNELS,
};

/// Decoded `GlobalModular` — the channel descriptions and the actual
/// pixel data for each channel.
#[derive(Debug, Clone)]
pub struct GlobalModular {
    /// True if a frame-wide MA tree was decoded in the LfGlobal section
    /// (rather than locally inside the modular sub-bitstream).
    pub global_tree_present: bool,
    /// True if the modular sub-bitstream inside this GlobalModular
    /// section reused the frame-wide tree (`global_tree_present` must
    /// also be true in that case).
    pub inner_used_global_tree: bool,
    pub wp_header: WpHeader,
    pub nb_transforms: u32,
    pub transforms: Vec<TransformInfo>,
    pub image: ModularImage,
    /// Number of leading meta-channels in `image.channels` (those
    /// inserted by Palette transforms). Used by per-group decode to
    /// know which channels are NOT split across groups.
    pub nb_meta_channels: usize,
    /// True if every non-meta colour/extra channel was fully decoded
    /// inside GlobalModular (small-image case per G.1.3 last paragraph).
    /// False when at least one channel exceeds `group_dim` in width or
    /// height — the per-group sections then carry the bulk of pixel
    /// data and the inverse transforms must wait until after all
    /// PassGroups complete.
    pub fully_decoded: bool,
    /// MA tree carried over to per-group sub-bitstreams that opt to
    /// reuse the global tree (`use_global_tree=true` inside their inner
    /// ModularHeader). Only present when GlobalModular's outer
    /// `use_global_tree` was true.
    pub global_tree: Option<MaTreeFdis>,
}

impl GlobalModular {
    /// Decode the GlobalModular section per ISO/IEC 18181-1:2024 G.1.3.
    ///
    /// Per the spec's last paragraph: only the first `nb_meta_channels`
    /// channels and any further channels that have a width and height
    /// both at most `group_dim` are decoded inside GlobalModular. Any
    /// channel exceeding `group_dim` in either dimension stops the
    /// channel-decode loop and is left for per-PassGroup decode (G.4.2).
    /// If `fully_decoded` is true, the inverse transforms have been
    /// applied. Otherwise, the caller must apply transforms via
    /// [`apply_inverse_transforms`] AFTER all PassGroups complete (last
    /// paragraph of G.4.2).
    pub fn read(
        br: &mut BitReader<'_>,
        fh: &FrameHeader,
        metadata: &ImageMetadataFdis,
    ) -> Result<Self> {
        // 1. use_global_tree per C.4.8 — `u(1)` flag indicating that an
        //    MA tree precedes the modular sub-bitstream. The tree
        //    decoded here is shared by ModularLfGroup / ModularGroup
        //    sub-bitstreams that follow. For the round-3 single-group
        //    fixture, every sub-bitstream is GlobalModular so the
        //    global tree IS the local tree.
        //
        //    Per FDIS C.4.8 first sentence: if `use_global_tree` is
        //    true, an MA tree is decoded first (D.4.2). Then the inner
        //    modular sub-bitstream's `use_global_tree` flag (in its
        //    `ModularHeader`, Table C.22) selects whether to reuse
        //    that tree or read a new one. Since both flags map to the
        //    same MA tree in the round-3 single-group case, we decode
        //    the tree in whichever bundle reaches it first.
        let global_use_tree = br.read_bool()?;
        let global_tree = if global_use_tree {
            Some(MaTreeFdis::read(br).map_err(|e| {
                Error::InvalidData(format!("JXL GlobalModular: global tree read failed: {e}"))
            })?)
        } else {
            None
        };

        // 2. Modular sub-bitstream (C.9). Per Table C.22:
        //    - use_global_tree (Bool),
        //    - WPHeader,
        //    - U32 nb_transforms,
        //    - TransformInfo[nb_transforms],
        //    - if !use_global_tree: MA tree + clustered distributions,
        //    - ANS state + per-channel decode.
        let inner_use_global_tree = br.read_bool()?;
        let wp_header = WpHeader::read(br)?;

        let nb_transforms = br.read_u32([
            U32Dist::Val(0),
            U32Dist::Val(1),
            U32Dist::BitsOffset(4, 2),
            U32Dist::BitsOffset(8, 18),
        ])?;
        // Bound: a malicious bitstream could supply an absurd value here.
        // The U32 distribution caps `nb_transforms` at 18 + 2^8 = 274,
        // which is well above any realistic image. We accept the cap
        // implied by the U32 distribution.
        const MAX_TRANSFORMS: u32 = 274;
        if nb_transforms > MAX_TRANSFORMS {
            return Err(Error::InvalidData(format!(
                "JXL GlobalModular: nb_transforms {nb_transforms} exceeds {MAX_TRANSFORMS}"
            )));
        }
        let mut transforms: Vec<TransformInfo> = Vec::with_capacity(nb_transforms as usize);
        for _ in 0..nb_transforms {
            transforms.push(TransformInfo::read(br)?);
        }

        // 3. Local MA tree + per-context distributions, OR reuse the
        //    global tree. We CLONE the global tree (with fresh ANS
        //    state) so the original can be retained on the bundle for
        //    later per-PassGroup sub-bitstreams that also opt in to
        //    `use_global_tree=true`.
        let mut tree = if inner_use_global_tree {
            global_tree
                .as_ref()
                .ok_or_else(|| {
                    Error::InvalidData(
                        "JXL GlobalModular: inner sub-bitstream wants global tree but none was decoded".into(),
                    )
                })?
                .cloned_with_fresh_state()
        } else {
            MaTreeFdis::read(br)?
        };

        // 4. Channel layout.
        let descs = derive_channel_descs(fh, metadata)?;
        if descs.is_empty() {
            return Err(Error::InvalidData(
                "JXL GlobalModular: zero channels — VarDCT path not yet supported".into(),
            ));
        }
        if descs.len() > MAX_CHANNELS {
            return Err(Error::InvalidData(format!(
                "JXL GlobalModular: {} channels exceeds cap {}",
                descs.len(),
                MAX_CHANNELS
            )));
        }

        // 4b. Adjust channel descriptions per H.6: transforms add
        //     meta-channels at the start of the channel list (Palette)
        //     or split channels into pairs (Squeeze).  RCT does not
        //     change the layout. The list returned is the layout AT
        //     THE TIME OF DECODE — i.e. with transform bookkeeping
        //     applied so the decoder reads the correct number of
        //     channels at the correct dimensions.
        let descs = apply_transforms_to_channel_layout(descs, &transforms)?;

        // Count meta-channels at the head of the descs list. Per H.1
        // these are the channels with hshift=-1, vshift=-1 (Palette
        // meta channel signature) inserted at the front.
        let nb_meta_channels = descs
            .iter()
            .take_while(|d| d.hshift == -1 && d.vshift == -1)
            .count();

        // 5. G.1.3 last paragraph — decode only the first
        //    `nb_meta_channels` channels and any further channel whose
        //    `width <= group_dim AND height <= group_dim`. Any further
        //    channel that exceeds group_dim stops the GlobalModular
        //    channel-decode loop and is deferred to per-PassGroup
        //    decode (G.4.2).
        let group_dim = fh.group_dim();
        let mut decoded_descs: Vec<ChannelDesc> = Vec::with_capacity(descs.len());
        let mut deferred_indices: Vec<usize> = Vec::new();
        let mut stop = false;
        for (idx, d) in descs.iter().enumerate() {
            if idx < nb_meta_channels {
                decoded_descs.push(*d);
                continue;
            }
            if !stop && d.width <= group_dim && d.height <= group_dim {
                decoded_descs.push(*d);
            } else {
                stop = true;
                deferred_indices.push(idx);
            }
        }
        let fully_decoded = deferred_indices.is_empty();

        // 6. Pixel decode (per Annex H.3) for the non-deferred subset.
        //    The full descs list is preserved in `image` (deferred
        //    channels become zero-filled placeholders that PassGroups
        //    fill in afterwards).
        let partial_image = decode_channels_at_stream(
            br,
            &decoded_descs,
            &mut tree,
            &wp_header,
            0, // GlobalModular: stream_index = 0.
        )?;

        // Reassemble into a full ModularImage with deferred channels as
        // zero buffers (will be filled by per-PassGroup decode in
        // G.4.2). decoded_descs ordering matches descs[0..decoded_count].
        let mut full_channels: Vec<Vec<i32>> = Vec::with_capacity(descs.len());
        let mut iter_decoded = partial_image.channels.into_iter();
        for (idx, d) in descs.iter().enumerate() {
            if deferred_indices.contains(&idx) {
                let n = (d.width as usize).saturating_mul(d.height as usize);
                full_channels.push(vec![0i32; n]);
            } else {
                full_channels.push(iter_decoded.next().ok_or_else(|| {
                    Error::InvalidData(
                        "JXL GlobalModular: decoded-channel iter exhausted prematurely".into(),
                    )
                })?);
            }
        }
        let mut image = ModularImage {
            channels: full_channels,
            descs,
        };

        // 7. Apply inverse transforms (Annex H.6) ONLY when the image
        //    is fully decoded inside GlobalModular. Otherwise defer to
        //    after all PassGroups complete (G.4.2 last paragraph).
        let bit_depth = metadata.bit_depth.bits_per_sample.max(1);
        if fully_decoded {
            apply_inverse_transforms(&mut image, &transforms, bit_depth)?;
        }

        // Stash the global tree on the bundle so per-PassGroup decode
        // can reuse it without re-reading. Per H.2: when a per-group
        // sub-bitstream's `use_global_tree=true`, "the global MA tree
        // and its clustered distributions are used as decoded from the
        // GlobalModular section". The stored tree carries the static
        // shape + clustered distributions; per-group reuse goes through
        // [`MaTreeFdis::cloned_with_fresh_state`] to reset the ANS
        // state for each new sub-bitstream.
        let global_tree_for_pass = global_tree;

        Ok(Self {
            global_tree_present: global_use_tree,
            inner_used_global_tree: inner_use_global_tree,
            wp_header,
            nb_transforms,
            transforms,
            image,
            nb_meta_channels,
            fully_decoded,
            global_tree: global_tree_for_pass,
        })
    }
}

/// Apply the modular inverse-transform sequence (RCT / Palette /
/// Squeeze) to `image` per Annex H.6, in REVERSE bitstream order. This
/// is invoked from [`GlobalModular::read`] for the small-image fast path
/// AND from `decode_codestream` AFTER all PassGroups complete (G.4.2
/// last paragraph) for the multi-group path.
pub fn apply_inverse_transforms(
    image: &mut ModularImage,
    transforms: &[TransformInfo],
    bit_depth: u32,
) -> Result<()> {
    for t in transforms.iter().rev() {
        match t.tr {
            TransformId::Rct => {
                let begin = t.begin_c.unwrap_or(0) as usize;
                let rct_type = t.rct_type.unwrap_or(0);
                inverse_rct(image, begin, rct_type)?;
            }
            TransformId::Palette => {
                let begin = t.begin_c.unwrap_or(0) as usize;
                let num_c = t.num_c.unwrap_or(1);
                let nb_colours = t.nb_colours.unwrap_or(0);
                let nb_deltas = t.nb_deltas.unwrap_or(0);
                let d_pred = t.d_pred.unwrap_or(0);
                inverse_palette(
                    image, begin, num_c, nb_colours, nb_deltas, d_pred, bit_depth,
                )?;
            }
            TransformId::Squeeze => {
                apply_inverse_squeeze(image, &t.squeeze_params)?;
            }
        }
    }
    Ok(())
}

/// Apply the Inverse Squeeze transform's per-step pair-merge for every
/// step in `squeeze_params`, in reverse order. This implements Listing
/// I.18 from FDIS / Annex H.6.2 from the 2024 edition.
///
/// Empty `squeeze_params` is the "default parameters" path; round 2
/// only handles the explicit-params case (the small fixtures don't
/// trigger default-param Squeeze; if they did the encoder would not
/// have emitted an explicit kSqueeze transform).
fn apply_inverse_squeeze(image: &mut ModularImage, squeeze_params: &[SqueezeParam]) -> Result<()> {
    if squeeze_params.is_empty() {
        return Err(Error::Unsupported(
            "JXL Modular Squeeze: default-params (empty) sequence not yet supported (round 2)"
                .into(),
        ));
    }
    // Inverse application: reverse-iterate the params.
    for sp in squeeze_params.iter().rev() {
        let begin = sp.begin_c as usize;
        let num_c = sp.num_c as usize;
        let end = begin + num_c - 1;
        let r = if sp.in_place {
            end + 1
        } else {
            // For "not in place" the residuals were appended at the very
            // end of the channel list; their count is num_c so they sit
            // at indices [channel_count - num_c .. channel_count).
            // Per spec: `r = channel.size() + begin - end - 1` (which is
            // `channel.size() - num_c`).
            image
                .channels
                .len()
                .saturating_sub(num_c)
                .saturating_add(begin)
                .saturating_sub(begin)
        };
        for c in begin..=end {
            // We pair channel[c] with channel[r + (c - begin)] (since
            // each iteration removes channel r, the residual stays at
            // the same index r as we step through c).
            let r_index = if sp.in_place { r + (c - begin) } else { r };
            if r_index >= image.channels.len() {
                return Err(Error::InvalidData(format!(
                    "JXL Modular Squeeze: residual channel index {r_index} out of range {}",
                    image.channels.len()
                )));
            }
            if c >= image.channels.len() || c == r_index {
                return Err(Error::InvalidData(format!(
                    "JXL Modular Squeeze: invalid channel pair c={c} r={r_index}"
                )));
            }
            // Compute output dims.
            let cd = image.descs[c];
            let rd = image.descs[r_index];
            let (merged, new_w, new_h) = if sp.horizontal {
                if cd.height != rd.height {
                    return Err(Error::InvalidData(
                        "JXL Modular Squeeze (horiz): channel pair height mismatch".into(),
                    ));
                }
                let (out, ow) = horiz_isqueeze(
                    &image.channels[c],
                    cd.width,
                    &image.channels[r_index],
                    rd.width,
                    cd.height,
                )?;
                (out, ow, cd.height)
            } else {
                if cd.width != rd.width {
                    return Err(Error::InvalidData(
                        "JXL Modular Squeeze (vert): channel pair width mismatch".into(),
                    ));
                }
                let (out, oh) = vert_isqueeze(
                    &image.channels[c],
                    cd.height,
                    &image.channels[r_index],
                    rd.height,
                    cd.width,
                )?;
                (out, cd.width, oh)
            };
            // Write back to channel[c] with new dims.
            image.channels[c] = merged;
            image.descs[c] = ChannelDesc {
                width: new_w,
                height: new_h,
                hshift: cd.hshift - if sp.horizontal { 1 } else { 0 },
                vshift: cd.vshift - if sp.horizontal { 0 } else { 1 },
            };
            // Remove the residual channel.
            image.channels.remove(r_index);
            image.descs.remove(r_index);
        }
    }
    Ok(())
}

/// Apply transform metadata to the channel layout so the decoded
/// channel data has the correct shape per H.6:
///
/// * `kPalette` — adds one meta-channel of dims `nb_colours × num_c`
///   at the front; the original `num_c` channels are reduced to a
///   single index channel.
/// * `kRCT` — no channel-list change.
/// * `kSqueeze` — for each step, halves one dim of `num_c` source
///   channels (round-up) and inserts a residu channel of the same
///   width × half-height (or half-width × height) for each.
///
/// Public since round 9 so per-PassGroup decode can reuse it for
/// nested per-group transforms (Palette / RCT / Squeeze inside a
/// PassGroup ModularHeader, observed in cjxl 0.11.1's synth_320
/// fixture's edge groups). See [`apply_inverse_transforms`] for the
/// reverse step.
pub fn apply_transforms_to_channel_layout(
    mut descs: Vec<ChannelDesc>,
    transforms: &[TransformInfo],
) -> Result<Vec<ChannelDesc>> {
    for t in transforms {
        match t.tr {
            TransformId::Palette => {
                // Per H.6.4: channels begin_c+1 .. begin_c+num_c-1 are
                // removed; one meta-channel is inserted at the start
                // of the list with width = nb_colours, height = num_c.
                let begin = t.begin_c.unwrap_or(0) as usize;
                let num_c = t.num_c.unwrap_or(1) as usize;
                let nb_colours = t.nb_colours.unwrap_or(0);
                if num_c == 0 {
                    return Err(Error::InvalidData(
                        "JXL Modular Palette: num_c must be >= 1".into(),
                    ));
                }
                if begin + num_c > descs.len() {
                    return Err(Error::InvalidData(format!(
                        "JXL Modular Palette: begin_c {begin} + num_c {num_c} exceeds channel count {}",
                        descs.len()
                    )));
                }
                // Remove num_c-1 channels starting from begin+1 (the
                // remaining channel at begin holds palette indices).
                for _ in 1..num_c {
                    if begin + 1 < descs.len() {
                        descs.remove(begin + 1);
                    }
                }
                // Insert the meta-channel at index 0.
                descs.insert(
                    0,
                    ChannelDesc {
                        width: nb_colours,
                        height: num_c as u32,
                        hshift: -1,
                        vshift: -1,
                    },
                );
            }
            TransformId::Rct => {
                // RCT does not change the channel list per H.6.3.
            }
            TransformId::Squeeze => {
                // Per Listing I.17 (FDIS) / Annex H.6.2 (2024). For each
                // step, halve one dim (round-up) of channels [begin..end]
                // and insert a residu channel for each at position
                // `r + c - begin` where `r = in_place ? end+1 : len`.
                let params = &t.squeeze_params;
                if params.is_empty() {
                    return Err(Error::Unsupported(
                        "JXL Modular Squeeze: default-params (empty) sequence not yet supported"
                            .into(),
                    ));
                }
                for sp in params {
                    let begin = sp.begin_c as usize;
                    let num_c = sp.num_c as usize;
                    let end = begin + num_c - 1;
                    if end >= descs.len() {
                        return Err(Error::InvalidData(format!(
                            "JXL Modular Squeeze: end {end} >= channel count {}",
                            descs.len()
                        )));
                    }
                    let r_base = if sp.in_place { end + 1 } else { descs.len() };
                    for (k, c) in (begin..=end).enumerate() {
                        let cd = descs[c];
                        let (new_w, new_h, residu_w, residu_h, dh, dv) = if sp.horizontal {
                            let nw = cd.width.div_ceil(2);
                            let rw = cd.width / 2;
                            (nw, cd.height, rw, cd.height, 1, 0)
                        } else {
                            let nh = cd.height.div_ceil(2);
                            let rh = cd.height / 2;
                            (cd.width, nh, cd.width, rh, 0, 1)
                        };
                        descs[c] = ChannelDesc {
                            width: new_w,
                            height: new_h,
                            hshift: cd.hshift + dh,
                            vshift: cd.vshift + dv,
                        };
                        let residu = ChannelDesc {
                            width: residu_w,
                            height: residu_h,
                            hshift: cd.hshift + dh,
                            vshift: cd.vshift + dv,
                        };
                        let insert_at = if sp.in_place { r_base + k } else { descs.len() };
                        descs.insert(insert_at, residu);
                    }
                }
            }
        }
    }
    Ok(descs)
}

/// Compute the channel descriptions for the GlobalModular section per
/// FDIS C.4.8.
///
/// For Modular encoding:
///   number of channels = (1 if grey/xyb-not-encoded/RGB-grey
///                         else 3) + num_extra_channels.
///
/// We collapse the spec's branchy expression into:
///   colour_count = if VarDCT { 0 } else if Grey-1ch { 1 } else { 3 }
///
/// Then channel dims are `(width, height)` for colour channels and
/// `ceil(width / 2^dim_shift) × ceil(height / 2^dim_shift)` for extras.
fn derive_channel_descs(
    fh: &FrameHeader,
    metadata: &ImageMetadataFdis,
) -> Result<Vec<ChannelDesc>> {
    let mut descs: Vec<ChannelDesc> = Vec::new();

    let colour_count: u32 = match fh.encoding {
        Encoding::VarDct => 0,
        Encoding::Modular => {
            // Grey-1ch happens when:
            //   !do_YCbCr && !xyb_encoded && colour_space == kGrey
            // (per FDIS C.4.8 channel-count formula).
            let is_grey_1ch = !fh.do_ycbcr
                && !metadata.xyb_encoded
                && metadata.colour_encoding.colour_space == ColourSpace::Grey;
            if is_grey_1ch {
                1
            } else {
                3
            }
        }
    };

    // Frame's pixel dimensions; if have_crop is set fh.width/height
    // already reflect that.
    let width = fh.width;
    let height = fh.height;

    for _ in 0..colour_count {
        descs.push(ChannelDesc {
            width,
            height,
            hshift: 0,
            vshift: 0,
        });
    }

    // Extra channels.
    for ec in metadata.extra_channel_info.iter() {
        let shift = ec.dim_shift as i32;
        let w = (width as u64).div_ceil(1u64 << shift) as u32;
        let h = (height as u64).div_ceil(1u64 << shift) as u32;
        descs.push(ChannelDesc {
            width: w,
            height: h,
            hshift: shift,
            vshift: shift,
        });
    }

    Ok(descs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_header::FrameDecodeParams;
    use crate::metadata_fdis::ImageMetadataFdis;

    fn build_metadata_grey_8bpp() -> ImageMetadataFdis {
        // Synthesise an ImageMetadataFdis matching cjxl's defaults
        // for a Grey 8bpp image: !xyb, colour_space=Grey,
        // num_extra_channels=0, bit_depth=8.
        // Easiest: read all_default then mutate the colour_space.
        let bytes = crate::ans::test_helpers::pack_lsb(&[(1, 1)]);
        let mut br = crate::bitreader::BitReader::new(&bytes);
        let mut m = ImageMetadataFdis::read(&mut br).unwrap();
        m.xyb_encoded = false;
        m.colour_encoding.colour_space = ColourSpace::Grey;
        m
    }

    fn build_modular_frame_header_8x8() -> FrameHeader {
        // FrameHeader for an 8x8 Modular frame, no extras, no animation.
        // We use the same all_default short-circuit + mutate the few
        // fields the FDIS path consults.
        let params = FrameDecodeParams {
            xyb_encoded: false,
            num_extra_channels: 0,
            have_animation: false,
            have_animation_timecodes: false,
            image_width: 8,
            image_height: 8,
        };
        let bytes = crate::ans::test_helpers::pack_lsb(&[(1, 1)]);
        let mut br = crate::bitreader::BitReader::new(&bytes);
        let mut fh = FrameHeader::read(&mut br, &params).unwrap();
        fh.encoding = Encoding::Modular;
        fh.do_ycbcr = false;
        fh.width = 8;
        fh.height = 8;
        fh
    }

    #[test]
    fn channel_descs_grey_image_one_channel() {
        let fh = build_modular_frame_header_8x8();
        let m = build_metadata_grey_8bpp();
        let d = derive_channel_descs(&fh, &m).unwrap();
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].width, 8);
        assert_eq!(d[0].height, 8);
    }

    #[test]
    fn channel_descs_rgb_image_three_channels() {
        let fh = build_modular_frame_header_8x8();
        let mut m = build_metadata_grey_8bpp();
        m.colour_encoding.colour_space = ColourSpace::Rgb;
        let d = derive_channel_descs(&fh, &m).unwrap();
        assert_eq!(d.len(), 3);
    }

    #[test]
    fn channel_descs_var_dct_zero_colour() {
        let mut fh = build_modular_frame_header_8x8();
        fh.encoding = Encoding::VarDct;
        let m = build_metadata_grey_8bpp();
        let d = derive_channel_descs(&fh, &m).unwrap();
        assert_eq!(d.len(), 0);
    }
}
