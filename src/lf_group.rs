//! `LfGroup` bundle — ISO/IEC 18181-1:2024 Annex G.2 (= 2021 FDIS C.5).
//!
//! For an `kModular` frame the bundle reduces to a single
//! [`ModularLfGroup`] (G.2.3). For `kVarDCT` the bundle additionally
//! contains [`LfCoefficients`] (G.2.2) and [`HfMetadata`] (G.2.4).
//!
//! ## Round 11 — LF coefficients sub-bitstream wiring
//!
//! Round-11 lands [`LfCoefficients::read`] which:
//!
//! * Reads `extra_precision = u(2)` per FDIS C.5.3.
//! * Builds a per-LfGroup `ChannelDesc` list of 3 channels (X', Y', B')
//!   with dims `ceil(lf_group_width / 8) × ceil(lf_group_height / 8)`,
//!   optionally further right-shifted by `frame_header.jpeg_upsampling`
//!   (per the same C.5.3 paragraph).
//! * Drives a Modular sub-bitstream (Annex H) over those three channels
//!   with `stream_index = 1 + lf_group_index` per Table H.4.
//! * Stores the decoded `lf_quant` array on the [`LfCoefficients`]
//!   struct for round-12 dequantisation.
//!
//! [`LfGroup::read`] composes the LfCoefficients sub-bitstream with the
//! ModularLfGroup sub-bitstream (G.2.3) — for now ModularLfGroup is
//! decoded only as far as the empty-channel-list case (no GlobalModular
//! channel had `hshift, vshift` both ≥ 3). HfMetadata (G.2.4) is parsed
//! as a single placeholder field (`nb_blocks`); end-to-end HF metadata
//! decode lands round 12+.
//!
//! ## Round 6 / 7 history (still relevant)
//!
//! Multi-group decode required four coordinated pieces:
//!
//! 1. `GlobalModular::read` stops after `nb_meta_channels` plus any
//!    channel ≤ `group_dim` (G.1.3 last paragraph). [round 7]
//! 2. The Modular sub-bitstream's `stream_index` property (Table H.4
//!    property 1) reflects which sub-bitstream is being decoded —
//!    `0` for GlobalModular, `1 + lf_group_idx` for LfCoefficients,
//!    `1 + num_lf_groups + lf_group_idx` for ModularLfGroup. [round 7]
//! 3. The TOC entry order (§F.3) is permuted-aware. [round 2]
//! 4. Inverse transforms run AFTER all PassGroups complete (G.4.2).
//!    [round 7]

use oxideav_core::{Error, Result};

use crate::bitreader::{BitReader, U32Dist};
use crate::frame_header::{Encoding, FrameHeader};
use crate::lf_global::LfGlobal;
use crate::modular_fdis::{
    decode_channels_at_stream, ChannelDesc, MaTreeFdis, TransformInfo, WpHeader,
};

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

/// `LfCoefficients` (G.2.2) — `kVarDCT` only. Round 11 wires the
/// modular sub-bitstream decode for the X', Y', B' LF coefficient
/// channels per FDIS C.5.3.
#[derive(Debug, Clone)]
pub struct LfCoefficients {
    /// `extra_precision = u(2)` — additional fixed-point precision
    /// for the dequantised LF channels per Listing F.1.
    pub extra_precision: u32,
    /// The decoded per-channel LF coefficients. Length is always 3 (or 0
    /// when the `kUseLfFrame` flag is set, in which case the spec says
    /// to skip C.5.3 entirely). Each `Vec<i32>` has `width * height`
    /// elements row-major where `(width, height)` come from
    /// [`LfCoefficients::lf_quant_dims`].
    pub lf_quant: Vec<Vec<i32>>,
    /// Width of each channel in the [`LfCoefficients::lf_quant`] array,
    /// indexed by channel (0 = X, 1 = Y, 2 = B).
    pub lf_quant_widths: [u32; 3],
    /// Height of each channel.
    pub lf_quant_heights: [u32; 3],
}

impl LfCoefficients {
    /// Compute the per-channel `(width, height)` of the LF coefficient
    /// channels for an LfGroup of frame-coordinates rectangle
    /// `(lf_group_width, lf_group_height)`. Returns `[(w, h); 3]` for
    /// channels (X, Y, B) in that order. Per FDIS C.5.3 each channel has
    /// `ceil(group_height / 8) × ceil(group_width / 8)` samples; if
    /// `frame_header.jpeg_upsampling[c]` is set, the rows / columns are
    /// **optionally right-shifted by one** for that channel.
    pub fn lf_quant_dims(
        fh: &FrameHeader,
        lf_group_width: u32,
        lf_group_height: u32,
    ) -> [(u32, u32); 3] {
        // Base dims: ceil(group_dim / 8). The clause-text says "the
        // number of rows and columns is optionally right-shifted by one
        // according to frame_header.jpeg_upsampling". jpeg_upsampling is
        // a 3-element array; values 1 (=2x subsampling) and higher
        // shift by one (rows OR columns, both for the chroma plane).
        let base_w = lf_group_width.div_ceil(8);
        let base_h = lf_group_height.div_ceil(8);
        let mut out = [(base_w, base_h); 3];
        for (c, slot) in out.iter_mut().enumerate() {
            let shift = fh.jpeg_upsampling.get(c).copied().unwrap_or(0);
            // Spec: "optionally right-shifted by one" — the practical
            // interpretation (matching libjxl behaviour) is shift by 1
            // when jpeg_upsampling[c] != 0. Round-11 follows that
            // reading; conformance-fixture-driven validation defers to
            // round 12+ when end-to-end VarDCT pixel decode is wired.
            let s = if shift > 0 { 1u32 } else { 0u32 };
            *slot = (base_w >> s, base_h >> s);
        }
        out
    }

    /// Decode the LfCoefficients sub-bitstream per FDIS C.5.3.
    ///
    /// The caller has already positioned `br` at the LfGroup's section
    /// start AND verified that `frame_header.encoding == kVarDCT` AND
    /// that `kUseLfFrame` is NOT set in `frame_header.flags` (the spec's
    /// C.5.3 first sentence — when `kUseLfFrame` is set the entire
    /// subclause is skipped).
    ///
    /// `lf_global` carries the optional global tree the LF
    /// sub-bitstream's inner `use_global_tree=true` may reference.
    /// `stream_index` is `1 + lf_group_index` per Table H.4.
    pub fn read(
        br: &mut BitReader<'_>,
        fh: &FrameHeader,
        lf_global: &LfGlobal,
        lf_group_width: u32,
        lf_group_height: u32,
        lf_group_index: LfGroupIndex,
    ) -> Result<Self> {
        let extra_precision = br.read_bits(2)?;

        let dims = Self::lf_quant_dims(fh, lf_group_width, lf_group_height);
        // Per spec the LF coefficient channels have the same shifts that
        // jpeg_upsampling implies — but for the modular sub-bitstream's
        // ChannelDesc we only need the per-channel pixel dims; the
        // hshift / vshift fields are used by Annex H for predictor
        // bookkeeping and don't affect the LF decode. We pass through
        // the jpeg_upsampling shifts so the property[6..=10] (channel
        // shift) values are reported correctly to any MA tree that
        // branches on them.
        let descs: Vec<ChannelDesc> = (0..3)
            .map(|c| {
                let (w, h) = dims[c];
                let s = fh.jpeg_upsampling.get(c).copied().unwrap_or(0);
                let shift = if s > 0 { 1 } else { 0 };
                ChannelDesc {
                    width: w.max(1),
                    height: h.max(1),
                    hshift: shift,
                    vshift: shift,
                }
            })
            .collect();

        // Inner ModularHeader (Table H.1) per Annex H.2.
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
                "JXL LfCoefficients: nb_transforms {nb_transforms} exceeds {MAX_TRANSFORMS}"
            )));
        }
        let mut transforms: Vec<TransformInfo> = Vec::with_capacity(nb_transforms as usize);
        for _ in 0..nb_transforms {
            transforms.push(TransformInfo::read(br)?);
        }
        // Round-11 only handles the no-transform LF case. Per-LF
        // transforms (e.g. Squeeze inside an LfCoefficients section) are
        // round-12+ work.
        if !transforms.is_empty() {
            return Err(Error::Unsupported(format!(
                "JXL LfCoefficients: {} transforms inside LF sub-bitstream not yet supported \
                 (round 12+)",
                transforms.len()
            )));
        }

        let mut tree = if inner_use_global_tree {
            lf_global
                .global_modular
                .global_tree
                .as_ref()
                .ok_or_else(|| {
                    Error::InvalidData(
                        "JXL LfCoefficients: inner sub-bitstream wants global tree but none was \
                         decoded in GlobalModular"
                            .into(),
                    )
                })?
                .cloned_with_fresh_state()
        } else {
            MaTreeFdis::read(br)?
        };

        // stream_index per Table H.4: `1 + lf_group_idx` for LfCoefficients.
        let stream_index = 1i32 + lf_group_index as i32;

        let img = decode_channels_at_stream(br, &descs, &mut tree, &wp_header, stream_index)?;

        // Sanity check: 3 channels of expected dims.
        if img.channels.len() != 3 {
            return Err(Error::InvalidData(format!(
                "JXL LfCoefficients: expected 3 decoded channels, got {}",
                img.channels.len()
            )));
        }
        let lf_quant_widths = [img.descs[0].width, img.descs[1].width, img.descs[2].width];
        let lf_quant_heights = [
            img.descs[0].height,
            img.descs[1].height,
            img.descs[2].height,
        ];
        Ok(Self {
            extra_precision,
            lf_quant: img.channels,
            lf_quant_widths,
            lf_quant_heights,
        })
    }
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
    /// G.3. Round 11 wires LfCoefficients (G.2.2) end-to-end; the
    /// ModularLfGroup branch and HfMetadata are round-12+ work.
    ///
    /// `br` is positioned at the start of the LfGroup TOC slot.
    /// `lf_global` carries the global MA tree (if any) and the
    /// partially-decoded GlobalModular image (used by ModularLfGroup
    /// in round 12).
    ///
    /// Round-11 envelope:
    /// * `encoding == kVarDCT` (caller-checked).
    /// * `frame_header.flags & kUseLfFrame == 0` (LF coefficient bundle
    ///   is present, per C.5.3 first sentence).
    /// * The GlobalModular image has no channel with `hshift, vshift`
    ///   both ≥ 3 — i.e. the ModularLfGroup section is empty.
    pub fn read(
        br: &mut BitReader<'_>,
        fh: &FrameHeader,
        lf_global: &LfGlobal,
        lf_group_index: LfGroupIndex,
    ) -> Result<Self> {
        let num_lf_groups = fh.num_lf_groups();
        if lf_group_index as u64 >= num_lf_groups {
            return Err(Error::InvalidData(format!(
                "JXL LfGroup: index {lf_group_index} >= num_lf_groups {num_lf_groups}"
            )));
        }

        let (_x, _y, lf_w, lf_h) = ModularLfGroup::rect_for_index(fh, lf_group_index)?;
        let mlf_group = ModularLfGroup {
            lf_group_index,
            lf_group_width: lf_w,
            lf_group_height: lf_h,
        };

        // Sub-bitstream order per Table G.3: ModularLfGroup, then (if
        // VarDCT) LfCoefficients, then (if VarDCT) HfMetadata.
        // Round-11 supports only the all-channels-fit-GlobalModular
        // case for ModularLfGroup (= empty channel list). For VarDCT
        // we then read LfCoefficients. HfMetadata defers (the round-11
        // test exits early after LfCoefficients).
        if has_modular_lf_group_channels(lf_global, fh) {
            return Err(Error::Unsupported(
                "JXL LfGroup: ModularLfGroup with channels having hshift, vshift both >= 3 not \
                 yet supported (round 12+)"
                    .into(),
            ));
        }

        let (lf_coeff, hf_meta) = if fh.encoding == Encoding::VarDct {
            // C.5.3 first sentence: when kUseLfFrame is set, skip C.5.3.
            const K_USE_LF_FRAME: u64 = crate::frame_header::flags::USE_LF_FRAME;
            if (fh.flags & K_USE_LF_FRAME) != 0 {
                return Err(Error::Unsupported(
                    "JXL LfGroup: kUseLfFrame flag (LF reused from a separate LFFrame) not yet \
                     supported (round 12+)"
                        .into(),
                ));
            }
            let lf = LfCoefficients::read(br, fh, lf_global, lf_w, lf_h, lf_group_index)?;
            // HfMetadata (G.2.4) — round-11 stops here; the round-11
            // acceptance test exercises only LfCoefficients. Returning
            // a placeholder `HfMetadata { nb_blocks: 0 }` keeps the
            // type shape; round-12 will replace this with a real
            // parse + DctSelect / HfMul / XFromY / BFromY decode.
            (Some(lf), None::<HfMetadata>)
        } else {
            (None, None)
        };

        Ok(Self {
            lf_coeff,
            mlf_group,
            hf_meta,
        })
    }
}

/// Detect whether the partially-decoded GlobalModular image has any
/// channel that should land in the ModularLfGroup section (per FDIS
/// C.5.2: hshift, vshift both >= 3 AND not already decoded in
/// GlobalModular). Round-11 returns `false` for the small-image case
/// where every channel fit inside `group_dim` (so no per-LfGroup
/// channel filtering applies); a future round will extend this to
/// drive the ModularLfGroup decode.
fn has_modular_lf_group_channels(lf_global: &LfGlobal, fh: &FrameHeader) -> bool {
    let group_dim = fh.group_dim();
    let nb_meta = lf_global.global_modular.nb_meta_channels;
    for (idx, d) in lf_global.global_modular.image.descs.iter().enumerate() {
        if idx < nb_meta {
            continue;
        }
        if d.width <= group_dim && d.height <= group_dim {
            // Already fully decoded in GlobalModular — not eligible.
            continue;
        }
        if d.hshift >= 3 && d.vshift >= 3 {
            return true;
        }
    }
    false
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
    use crate::global_modular::GlobalModular;
    use crate::lf_global::LfChannelDequantization;
    use crate::modular_fdis::{ModularImage, WpHeader};

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

    /// Build a minimal empty-image LfGlobal stub for round-11 tests
    /// where the LfCoefficients sub-bitstream is the only thing being
    /// exercised. The GlobalModular's `image` is empty (no channels);
    /// `nb_meta_channels = 0`; tree is `None`.
    fn build_empty_lf_global() -> LfGlobal {
        LfGlobal {
            lf_dequant: LfChannelDequantization::default(),
            quantizer: None,
            hf_block_context: None,
            lf_channel_correlation: None,
            global_modular: GlobalModular {
                global_tree_present: false,
                inner_used_global_tree: false,
                wp_header: WpHeader::default(),
                nb_transforms: 0,
                transforms: Vec::new(),
                image: ModularImage {
                    channels: Vec::new(),
                    descs: Vec::new(),
                },
                nb_meta_channels: 0,
                fully_decoded: true,
                global_tree: None,
            },
        }
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
    fn lf_group_read_rejects_out_of_range_index() {
        let fh = build_fh(64, 64);
        let lf_global = build_empty_lf_global();
        let bytes = vec![0u8; 16];
        let mut br = BitReader::new(&bytes);
        let r = LfGroup::read(&mut br, &fh, &lf_global, 99);
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn lf_quant_dims_match_spec_no_subsampling() {
        // For an 8x8 LfGroup with no jpeg_upsampling, all three channels
        // have ceil(8/8) = 1 row, ceil(8/8) = 1 column.
        let fh = build_fh(8, 8);
        let dims = LfCoefficients::lf_quant_dims(&fh, 8, 8);
        for (c, &(w, h)) in dims.iter().enumerate() {
            assert_eq!((w, h), (1, 1), "channel {c}: got {w}x{h}");
        }
    }

    #[test]
    fn lf_quant_dims_64x64_no_subsampling() {
        // For a 64x64 LfGroup: ceil(64/8) = 8.
        let fh = build_fh(64, 64);
        let dims = LfCoefficients::lf_quant_dims(&fh, 64, 64);
        for &(w, h) in dims.iter() {
            assert_eq!((w, h), (8, 8));
        }
    }

    /// Round-11 acceptance test: hand-built minimal VarDCT bitstream
    /// covering LfGlobal (Quantizer + HfBlockContext + CfL +
    /// 0-channel GlobalModular) followed by LfGroup → LfCoefficients
    /// over 3 channels of dim 1×1 (one-block frame). The MA tree has
    /// one leaf with predictor=Zero, and the symbol stream uses
    /// prefix codes with alphabet_size=1 per cluster — every symbol
    /// decodes to 0, so the LF coefficients are all zero. IDCT
    /// inverse + dequant are out of scope for round 11 (round 12+).
    #[test]
    fn round11_lfgroup_minimal_vardct_one_block_parses() {
        use crate::ans::test_helpers::pack_lsb;
        use crate::lf_global::LfGlobal;
        use crate::metadata_fdis::ImageMetadataFdis;

        // Build the tightest possible VarDCT-encoding FrameHeader for
        // an 8×8 frame. We start from an `all_default=1` Modular header
        // (the existing helper already does that) and mutate the
        // encoding field in-place — the field is checked at decode
        // time by `LfGlobal::read` and `LfCoefficients::read`, not by
        // the FrameHeader byte layout.
        let mut fh = build_fh(8, 8);
        fh.encoding = Encoding::VarDct;
        // group_dim default (256). num_lf_groups = 1, num_groups = 1.
        // jpeg_upsampling = [0, 0, 0] from build_fh.

        // ImageMetadata for VarDCT path: Grey colour space, no extras.
        // Reuse the all_default=1 helper as the base; mutate
        // colour_space if needed. For VarDCT path we don't actually
        // consume the metadata in LfGlobal::read for the empty-channel
        // case, but we still need a valid bundle.
        let md_bytes = pack_lsb(&[(1, 1)]);
        let mut md_br = BitReader::new(&md_bytes);
        let metadata = ImageMetadataFdis::read(&mut md_br).unwrap();

        // Compose the LfGlobal bitstream piece-by-piece. Each tuple is
        // `(value, n_bits)` packed LSB-first by `pack_lsb`.
        let lf_global_bits: Vec<(u32, u32)> = vec![
            // 1. lf_dequant.all_default = 1
            (1, 1),
            // 2. Quantizer.global_scale: U32 sel=00 → BitsOffset(11, 1)
            //    → 2 bits selector + 11 bits payload (=0 → value 1).
            (0, 2),
            (0, 11),
            // 3. Quantizer.quant_lf: U32 sel=00 → Val(16). 2 bits.
            (0, 2),
            // 4. HfBlockContext.used_default = 1
            (1, 1),
            // 5. LfChannelCorrelation.all_default = 1
            (1, 1),
            // 6. GlobalModular (VarDCT, 0 channels):
            //    use_global_tree = 0
            (0, 1),
            //    inner_use_global_tree = 0
            (0, 1),
            //    WPHeader.default_wp = 1
            (1, 1),
            //    nb_transforms = U32 sel=00 → Val(0). 2 bits.
            (0, 2),
            //    (No tree, no distributions, no ANS state — descs empty.)
        ];

        let lf_global_bytes = pack_lsb(&lf_global_bits);
        let mut br_lfg = BitReader::new(&lf_global_bytes);
        let lf_global = LfGlobal::read(&mut br_lfg, &fh, &metadata)
            .expect("LfGlobal VarDCT minimum should parse");
        assert!(lf_global.quantizer.is_some());
        assert_eq!(lf_global.quantizer.unwrap().global_scale, 1);
        assert_eq!(lf_global.quantizer.unwrap().quant_lf, 16);
        assert!(lf_global.hf_block_context.is_some());
        assert!(lf_global.lf_channel_correlation.is_some());
        assert_eq!(lf_global.global_modular.image.channels.len(), 0);

        // Compose the LfGroup (LfCoefficients) bitstream.
        let lf_coeff_bits: Vec<(u32, u32)> = vec![
            // 1. extra_precision = u(2) = 0
            (0, 2),
            // 2. inner ModularHeader: inner_use_global_tree = 0
            (0, 1),
            //    WPHeader.default_wp = 1
            (1, 1),
            //    nb_transforms = 0 (U32 sel=00 → Val(0))
            (0, 2),
            // 3. MA tree-stream entropy prelude (D.3 over 6 distributions):
            //    lz77_enabled = 0
            (0, 1),
            //    clustering: is_simple = 1, nbits = 0 → 6 × u(0) = 0 bits
            (1, 1),
            (0, 2),
            //    use_prefix_code = 1, log_alphabet_size = 15 (implicit)
            (1, 1),
            //    1 × HybridUintConfig: split_exponent = 15 (u(4))
            //    msb_in_token / lsb_in_token: skipped because
            //    split_exponent == log_alphabet_size = 15
            (15, 4),
            //    per-cluster prefix count: u(1) = 0 → count = 1
            (0, 1),
            //    prefix code with alphabet_size = 1: 0 bits
            // 4. Tree decode (Listing D.9): 1 iteration, 0 bits since
            //    every decoded symbol is 0 (alphabet_size = 1 path).
            // 5. Symbol stream prelude (1 distribution):
            //    lz77_enabled = 0
            (0, 1),
            //    no clustering (num_dist == 1)
            //    use_prefix_code = 1
            (1, 1),
            //    1 × HybridUintConfig: split_exp = 15
            (15, 4),
            //    per-cluster prefix count: u(1) = 0 → count = 1
            (0, 1),
            //    prefix code with alphabet_size = 1: 0 bits
            // 6. Pixel decode: 3 samples × 0 bits = 0 bits.
        ];

        let lf_coeff_bytes = pack_lsb(&lf_coeff_bits);
        let mut br_lc = BitReader::new(&lf_coeff_bytes);
        let lf_group = LfGroup::read(&mut br_lc, &fh, &lf_global, 0)
            .expect("LfGroup minimal VarDCT should parse");

        // Assertions:
        let lf_coeff = lf_group.lf_coeff.expect("LfCoefficients should be present");
        assert_eq!(lf_coeff.extra_precision, 0);
        assert_eq!(lf_coeff.lf_quant.len(), 3);
        for (c, ch) in lf_coeff.lf_quant.iter().enumerate() {
            assert_eq!(lf_coeff.lf_quant_widths[c], 1);
            assert_eq!(lf_coeff.lf_quant_heights[c], 1);
            assert_eq!(ch.len(), 1);
            // All decoded LF coefficients must be 0 since the prefix
            // code has alphabet_size=1 and predictor=Zero.
            assert_eq!(
                ch[0], 0,
                "channel {c} LF[0,0] should be 0 for hand-built fixture"
            );
        }
        // ModularLfGroup geometry (G.2.3): the LF group rectangle is
        // the entire 8×8 frame for a single-LfGroup frame.
        assert_eq!(lf_group.mlf_group.lf_group_index, 0);
        assert_eq!(lf_group.mlf_group.lf_group_width, 8);
        assert_eq!(lf_group.mlf_group.lf_group_height, 8);
        // HfMetadata not yet wired in round 11.
        assert!(lf_group.hf_meta.is_none());
    }

    #[test]
    fn lf_quant_dims_with_jpeg_upsampling_chroma() {
        // jpeg_upsampling[1] = jpeg_upsampling[2] = 1 (4:2:0 subsampling
        // applied to chroma). Y' (0) stays 8x8; X' (1) and B' (2) right-
        // shifted by one to 4x4.
        let mut fh = build_fh(64, 64);
        fh.jpeg_upsampling = [0, 1, 1];
        let dims = LfCoefficients::lf_quant_dims(&fh, 64, 64);
        assert_eq!(dims[0], (8, 8));
        assert_eq!(dims[1], (4, 4));
        assert_eq!(dims[2], (4, 4));
    }
}
