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
    decode_channels, ChannelDesc, MaTreeFdis, ModularImage, TransformId, TransformInfo, WpHeader,
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
}

impl GlobalModular {
    /// Decode the GlobalModular section per FDIS C.4.8.
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
            Some(MaTreeFdis::read(br)?)
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
        //    global tree.
        let mut tree = if inner_use_global_tree {
            global_tree.ok_or_else(|| {
                Error::InvalidData(
                    "JXL GlobalModular: inner sub-bitstream wants global tree but none was decoded"
                        .into(),
                )
            })?
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

        // 4b. Adjust channel descriptions per H.6 (transforms can add
        //     meta-channels at the start of the channel list, e.g.
        //     Palette adds 1 meta-channel for the colour table). In
        //     round 1 we only handle this for Palette; Squeeze + RCT
        //     are deferred since they require post-decode inverse work.
        let descs = apply_transforms_to_channel_layout(descs, &transforms)?;

        // 5. Pixel decode (per Annex H.3).
        let image = decode_channels(br, &descs, &mut tree)?;

        // 6. Inverse transforms — round 1 errors out cleanly when any
        //    non-trivial transform was signalled. Pixel-correct decode
        //    of fixtures using transforms is round-2 work.
        if !transforms.is_empty() {
            return Err(Error::Unsupported(format!(
                "JXL GlobalModular: inverse transforms (have {} of types {:?}) not yet applied (round 2)",
                transforms.len(),
                transforms.iter().map(|t| t.tr).collect::<Vec<_>>()
            )));
        }

        Ok(Self {
            global_tree_present: global_use_tree,
            inner_used_global_tree: inner_use_global_tree,
            wp_header,
            nb_transforms,
            transforms,
            image,
        })
    }
}

/// Apply transform metadata to the channel layout so the decoded
/// channel data has the correct shape per H.6:
/// * `kPalette` — adds one meta-channel of dims `nb_colours × num_c`
///   at the front; the original `num_c` channels are removed (kept as
///   palette indices in a single channel).
/// * `kRCT` — no channel-list change.
/// * `kSqueeze` — transforms channel dims; round 1 declines to model
///   it (the inverse pass is deferred).
fn apply_transforms_to_channel_layout(
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
                // Squeeze reshapes the channel list significantly; round 1
                // declines to model it. The inverse pass is round-2 work.
                return Err(Error::Unsupported(
                    "JXL Modular: kSqueeze transform not yet supported (round 2)".into(),
                ));
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
