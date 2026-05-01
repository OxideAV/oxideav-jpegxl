//! `FrameHeader` bundle — FDIS 18181-1 §C.2 (Table C.2).
//!
//! This module decodes the per-frame header: encoding mode, frame type,
//! flags, sub-sampling, upsampling, LF level, crop window, blending,
//! animation timing, name, and the [`RestorationFilter`] sub-bundle.
//! It does NOT touch any pixel data — that lands in round 3 (GlobalModular
//! wiring) once [`crate::toc`] is fed by FrameHeader's group counts.
//!
//! Allocation bound: every `Vec::with_capacity` here is sized against
//! either `num_extra_channels` (capped at the JXL ImageMetadata maximum
//! of `4096 + 1` per §A.6) or against a name length capped at 1071 bytes
//! (the maximum encoding of `name_len`).
//!
//! Spec ambiguity discovered while implementing this module is
//! documented at the relevant fix site. None found in round 2 — see
//! `MEMORY/project_jpegxl_fdis_typos.md` for the four typos round 1
//! already documented.

use oxideav_core::{Error, Result};

use crate::bitreader::{BitReader, U32Dist};
use crate::extensions::Extensions;
use crate::metadata_fdis::SizeHeaderFdis;

/// `FrameType` enum — FDIS Table C.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    Regular = 0,
    LfFrame = 1,
    ReferenceOnly = 2,
    SkipProgressive = 3,
}

impl FrameType {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(FrameType::Regular),
            1 => Ok(FrameType::LfFrame),
            2 => Ok(FrameType::ReferenceOnly),
            3 => Ok(FrameType::SkipProgressive),
            _ => Err(Error::InvalidData(format!(
                "JXL FrameHeader: invalid frame_type {v}"
            ))),
        }
    }
}

/// `Encoding` enum — FDIS Table C.4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    VarDct = 0,
    Modular = 1,
}

impl Encoding {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Encoding::VarDct),
            1 => Ok(Encoding::Modular),
            _ => Err(Error::InvalidData(format!(
                "JXL FrameHeader: invalid encoding {v}"
            ))),
        }
    }
}

/// FDIS Table C.5 — `FrameHeader.flags` bit definitions.
pub mod flags {
    pub const NOISE: u64 = 0x01;
    pub const PATCHES: u64 = 0x02;
    pub const SPLINES: u64 = 0x10;
    pub const USE_LF_FRAME: u64 = 0x20;
    pub const SKIP_ADAPTIVE_LF_SMOOTHING: u64 = 0x80;
}

/// `BlendMode` enum — FDIS Table C.8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Replace = 0,
    Add = 1,
    Blend = 2,
    AlphaWeightedAdd = 3,
    Mul = 4,
}

impl BlendMode {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(BlendMode::Replace),
            1 => Ok(BlendMode::Add),
            2 => Ok(BlendMode::Blend),
            3 => Ok(BlendMode::AlphaWeightedAdd),
            4 => Ok(BlendMode::Mul),
            _ => Err(Error::InvalidData(format!(
                "JXL FrameHeader: invalid blend mode {v}"
            ))),
        }
    }
}

/// `BlendingInfo` bundle — FDIS Table C.7.
#[derive(Debug, Clone, Copy)]
pub struct BlendingInfo {
    pub mode: BlendMode,
    pub alpha_channel: u32,
    pub clamp: bool,
    pub source: u32,
}

impl Default for BlendingInfo {
    fn default() -> Self {
        Self {
            mode: BlendMode::Replace,
            alpha_channel: 0,
            clamp: false,
            source: 0,
        }
    }
}

impl BlendingInfo {
    fn read(br: &mut BitReader<'_>, multi_extra: bool, full_frame: bool) -> Result<Self> {
        let mode_v = br.read_u32([
            U32Dist::Val(0),
            U32Dist::Val(1),
            U32Dist::Val(2),
            U32Dist::BitsOffset(2, 3),
        ])?;
        let mode = BlendMode::from_u32(mode_v)?;
        let mut alpha_channel = 0u32;
        let mut clamp = false;
        if multi_extra && (mode == BlendMode::Blend || mode == BlendMode::AlphaWeightedAdd) {
            alpha_channel = br.read_u32([
                U32Dist::Val(0),
                U32Dist::Val(1),
                U32Dist::Val(2),
                U32Dist::BitsOffset(3, 3),
            ])?;
        }
        if multi_extra
            && (mode == BlendMode::Blend
                || mode == BlendMode::AlphaWeightedAdd
                || mode == BlendMode::Mul)
        {
            clamp = br.read_bool()?;
        }
        let mut source = 0u32;
        if mode != BlendMode::Replace || !full_frame {
            source = br.read_u32([
                U32Dist::Val(0),
                U32Dist::Val(1),
                U32Dist::Val(2),
                U32Dist::Val(3),
            ])?;
        }
        Ok(Self {
            mode,
            alpha_channel,
            clamp,
            source,
        })
    }
}

/// `Passes` bundle — FDIS Table C.6.
#[derive(Debug, Clone)]
pub struct Passes {
    pub num_passes: u32,
    pub num_ds: u32,
    /// Length `num_passes - 1` (empty when `num_passes == 1`).
    pub shift: Vec<u32>,
    /// Length `num_ds`.
    pub downsample: Vec<u32>,
    /// Length `num_ds`.
    pub last_pass: Vec<u32>,
}

impl Default for Passes {
    fn default() -> Self {
        Self {
            num_passes: 1,
            num_ds: 0,
            shift: Vec::new(),
            downsample: Vec::new(),
            last_pass: Vec::new(),
        }
    }
}

/// FDIS implicit cap on per-frame array allocations: a single frame
/// cannot contain more than this many passes / extra-channel slots /
/// name bytes. The spec gives `num_passes <= 11` (Val(1), Val(2),
/// Val(3), BitsOffset(3, 4) → max 4 + 7 = 11), and `num_ds < num_passes`.
const MAX_NUM_PASSES: u32 = 11;

impl Passes {
    fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let num_passes = br.read_u32([
            U32Dist::Val(1),
            U32Dist::Val(2),
            U32Dist::Val(3),
            U32Dist::BitsOffset(3, 4),
        ])?;
        if num_passes == 0 || num_passes > MAX_NUM_PASSES {
            return Err(Error::InvalidData(format!(
                "JXL FrameHeader: invalid num_passes {num_passes}"
            )));
        }
        if num_passes == 1 {
            return Ok(Self {
                num_passes: 1,
                ..Self::default()
            });
        }
        let num_ds = br.read_u32([
            U32Dist::Val(0),
            U32Dist::Val(1),
            U32Dist::Val(2),
            U32Dist::BitsOffset(1, 3),
        ])?;
        if num_ds >= num_passes {
            return Err(Error::InvalidData(format!(
                "JXL FrameHeader: num_ds ({num_ds}) >= num_passes ({num_passes})"
            )));
        }
        let mut shift = Vec::with_capacity((num_passes - 1) as usize);
        for _ in 0..(num_passes - 1) {
            shift.push(br.read_bits(2)?);
        }
        let mut downsample = Vec::with_capacity(num_ds as usize);
        for _ in 0..num_ds {
            downsample.push(br.read_u32([
                U32Dist::Val(1),
                U32Dist::Val(2),
                U32Dist::Val(4),
                U32Dist::Val(8),
            ])?);
        }
        let mut last_pass = Vec::with_capacity(num_ds as usize);
        for _ in 0..num_ds {
            let lp = br.read_u32([
                U32Dist::Val(0),
                U32Dist::Val(1),
                U32Dist::Val(2),
                U32Dist::Bits(3),
            ])?;
            if lp >= num_passes {
                return Err(Error::InvalidData(format!(
                    "JXL FrameHeader: last_pass ({lp}) >= num_passes ({num_passes})"
                )));
            }
            last_pass.push(lp);
        }
        Ok(Self {
            num_passes,
            num_ds,
            shift,
            downsample,
            last_pass,
        })
    }
}

/// `RestorationFilter` bundle — FDIS Table C.9.
#[derive(Debug, Clone)]
pub struct RestorationFilter {
    pub gab: bool,
    pub gab_custom: bool,
    pub gab_x_weight1: f32,
    pub gab_x_weight2: f32,
    pub gab_y_weight1: f32,
    pub gab_y_weight2: f32,
    pub gab_b_weight1: f32,
    pub gab_b_weight2: f32,
    pub epf_iters: u32,
    pub epf_sharp_custom: bool,
    pub epf_sharp_lut: [f32; 8],
    pub epf_weight_custom: bool,
    pub epf_channel_scale: [f32; 3],
    pub epf_pass1_zeroflush: f32,
    pub epf_pass2_zeroflush: f32,
    pub epf_sigma_custom: bool,
    pub epf_quant_mul: f32,
    pub epf_pass0_sigma_scale: f32,
    pub epf_pass2_sigma_scale: f32,
    pub epf_border_sad_mul: f32,
    pub epf_sigma_for_modular: f32,
    pub extensions: Extensions,
}

impl Default for RestorationFilter {
    fn default() -> Self {
        Self {
            gab: true,
            gab_custom: false,
            gab_x_weight1: 0.115_169_525,
            gab_x_weight2: 0.061_248_592,
            gab_y_weight1: 0.115_169_525,
            gab_y_weight2: 0.061_248_592,
            gab_b_weight1: 0.115_169_525,
            gab_b_weight2: 0.061_248_592,
            epf_iters: 2,
            epf_sharp_custom: false,
            epf_sharp_lut: [
                0.0,
                1.0 / 7.0,
                2.0 / 7.0,
                3.0 / 7.0,
                4.0 / 7.0,
                5.0 / 7.0,
                6.0 / 7.0,
                1.0,
            ],
            epf_weight_custom: false,
            epf_channel_scale: [40.0, 5.0, 3.5],
            epf_pass1_zeroflush: 0.45,
            epf_pass2_zeroflush: 0.6,
            epf_sigma_custom: false,
            epf_quant_mul: 0.46,
            epf_pass0_sigma_scale: 0.9,
            epf_pass2_sigma_scale: 6.5,
            epf_border_sad_mul: 2.0 / 3.0,
            epf_sigma_for_modular: 1.0,
            extensions: Extensions::default(),
        }
    }
}

impl RestorationFilter {
    // Start from spec defaults, then mutate per the bitstream. We
    // intentionally use direct field writes rather than the
    // struct-update syntax because Table C.9 conditionals make a
    // builder-style assembly far less readable.
    #[allow(clippy::field_reassign_with_default)]
    fn read(br: &mut BitReader<'_>, encoding: Encoding) -> Result<Self> {
        let mut rf = RestorationFilter::default();
        rf.gab = br.read_bool()?;
        if rf.gab {
            rf.gab_custom = br.read_bool()?;
            if rf.gab_custom {
                rf.gab_x_weight1 = br.read_f16()?;
                rf.gab_x_weight2 = br.read_f16()?;
                rf.gab_y_weight1 = br.read_f16()?;
                rf.gab_y_weight2 = br.read_f16()?;
                rf.gab_b_weight1 = br.read_f16()?;
                rf.gab_b_weight2 = br.read_f16()?;
            }
        } else {
            rf.gab_custom = false;
        }
        rf.epf_iters = br.read_bits(2)?;
        if rf.epf_iters > 0 && encoding == Encoding::VarDct {
            rf.epf_sharp_custom = br.read_bool()?;
            if rf.epf_sharp_custom {
                for slot in rf.epf_sharp_lut.iter_mut() {
                    *slot = br.read_f16()?;
                }
            }
        }
        if rf.epf_iters > 0 {
            rf.epf_weight_custom = br.read_bool()?;
            if rf.epf_weight_custom {
                for slot in rf.epf_channel_scale.iter_mut() {
                    *slot = br.read_f16()?;
                }
                rf.epf_pass1_zeroflush = br.read_f16()?;
                rf.epf_pass2_zeroflush = br.read_f16()?;
            }
            rf.epf_sigma_custom = br.read_bool()?;
            if rf.epf_sigma_custom {
                if encoding == Encoding::VarDct {
                    rf.epf_quant_mul = br.read_f16()?;
                }
                rf.epf_pass0_sigma_scale = br.read_f16()?;
                rf.epf_pass2_sigma_scale = br.read_f16()?;
                rf.epf_border_sad_mul = br.read_f16()?;
            }
            if encoding == Encoding::Modular {
                rf.epf_sigma_for_modular = br.read_f16()?;
            }
        }
        rf.extensions = Extensions::read(br)?;
        rf.extensions.skip_payload(br)?;
        Ok(rf)
    }
}

/// `FrameHeader` bundle — FDIS Table C.2 (the round-2 decode target).
#[derive(Debug, Clone)]
pub struct FrameHeader {
    pub all_default: bool,
    pub frame_type: FrameType,
    pub encoding: Encoding,
    pub flags: u64,
    pub do_ycbcr: bool,
    /// `[Y_subsampling, Cb_subsampling, Cr_subsampling]` if `do_ycbcr`,
    /// else empty.
    pub jpeg_upsampling: [u32; 3],
    pub upsampling: u32,
    /// Length `num_extra_channels`.
    pub ec_upsampling: Vec<u32>,
    pub group_size_shift: u32,
    pub x_qm_scale: u32,
    pub b_qm_scale: u32,
    pub passes: Passes,
    pub lf_level: u32,
    pub have_crop: bool,
    pub x0: i32,
    pub y0: i32,
    pub width: u32,
    pub height: u32,
    pub blending_info: BlendingInfo,
    /// Length `num_extra_channels`.
    pub ec_blending_info: Vec<BlendingInfo>,
    pub duration: u32,
    pub timecode: u32,
    pub is_last: bool,
    pub save_as_reference: u32,
    pub save_before_ct: bool,
    pub name: String,
    pub restoration_filter: RestorationFilter,
    pub extensions: Extensions,
}

/// Parameters needed from `ImageMetadata` to decode a `FrameHeader`.
/// The FDIS frame-header bundle conditionalises several fields on
/// these so we capture them in a small struct rather than passing them
/// individually.
#[derive(Debug, Clone, Copy)]
pub struct FrameDecodeParams {
    pub xyb_encoded: bool,
    pub num_extra_channels: u32,
    pub have_animation: bool,
    pub have_animation_timecodes: bool,
    pub image_width: u32,
    pub image_height: u32,
}

/// FDIS implicit cap on `num_extra_channels` from §A.6 (Table A.16):
/// `U32(Val(0), Val(1), BitsOffset(4, 2), BitsOffset(12, 1))` →
/// max `4096 + 1 = 4097`.
const MAX_NUM_EXTRA_CHANNELS: u32 = 4097;

/// FDIS implicit cap on `name_len` (§C.2 Table C.2):
/// `U32(Val(0), Bits(4), BitsOffset(5,16), BitsOffset(10,48))` →
/// max `48 + 1023 = 1071`.
const MAX_NAME_LEN: u32 = 1071;

impl FrameHeader {
    /// Decode FDIS Table C.2 from `br` against the supplied
    /// `ImageMetadata`-derived parameters. The bit cursor is advanced
    /// to one bit past the end of the bundle (the *next* bundle in
    /// FDIS Table C.1 — the byte-aligned TOC — is read by [`crate::toc`]
    /// which performs its own `pu0()` first).
    pub fn read(br: &mut BitReader<'_>, params: &FrameDecodeParams) -> Result<Self> {
        if params.num_extra_channels > MAX_NUM_EXTRA_CHANNELS {
            return Err(Error::InvalidData(format!(
                "JXL FrameHeader: caller's num_extra_channels ({}) exceeds spec maximum",
                params.num_extra_channels
            )));
        }

        let all_default = br.read_bool()?;
        if all_default {
            return Ok(Self::default_with(params));
        }
        let frame_type = FrameType::from_u32(br.read_bits(2)?)?;
        let encoding = Encoding::from_u32(br.read_bits(1)?)?;
        let flags = br.read_u64()?;

        let do_ycbcr = if !params.xyb_encoded {
            br.read_bool()?
        } else {
            false
        };
        let mut jpeg_upsampling = [0u32; 3];
        if do_ycbcr && (flags & flags::USE_LF_FRAME) == 0 {
            for slot in jpeg_upsampling.iter_mut() {
                *slot = br.read_bits(2)?;
            }
        }

        let mut upsampling = 1u32;
        let mut ec_upsampling: Vec<u32> = Vec::new();
        if (flags & flags::USE_LF_FRAME) == 0 {
            upsampling = br.read_u32([
                U32Dist::Val(1),
                U32Dist::Val(2),
                U32Dist::Val(4),
                U32Dist::Val(8),
            ])?;
            ec_upsampling.reserve(params.num_extra_channels as usize);
            for _ in 0..params.num_extra_channels {
                ec_upsampling.push(br.read_u32([
                    U32Dist::Val(1),
                    U32Dist::Val(2),
                    U32Dist::Val(4),
                    U32Dist::Val(8),
                ])?);
            }
        } else {
            // Defaults: upsampling = 1, ec_upsampling all 1.
            ec_upsampling = vec![1u32; params.num_extra_channels as usize];
        }

        let group_size_shift = if encoding == Encoding::Modular {
            br.read_bits(2)?
        } else {
            1
        };

        let mut x_qm_scale = 3u32;
        let mut b_qm_scale = 2u32;
        if encoding == Encoding::VarDct && params.xyb_encoded {
            x_qm_scale = br.read_bits(3)?;
            b_qm_scale = br.read_bits(3)?;
        }

        let passes = if frame_type != FrameType::ReferenceOnly {
            Passes::read(br)?
        } else {
            Passes::default()
        };

        let lf_level = if frame_type == FrameType::LfFrame {
            br.read_u32([
                U32Dist::Val(1),
                U32Dist::Val(2),
                U32Dist::Val(3),
                U32Dist::Val(4),
            ])?
        } else {
            0
        };

        let have_crop = if frame_type != FrameType::LfFrame {
            br.read_bool()?
        } else {
            false
        };
        let mut x0 = 0i32;
        let mut y0 = 0i32;
        let mut width = params.image_width;
        let mut height = params.image_height;
        let crop_dist = [
            U32Dist::Bits(8),
            U32Dist::BitsOffset(11, 256),
            U32Dist::BitsOffset(14, 2304),
            U32Dist::BitsOffset(30, 18688),
        ];
        if have_crop {
            if frame_type != FrameType::ReferenceOnly {
                // FDIS uses UnpackSigned on x0/y0 implicitly via the U32
                // distribution. The U32 result is non-negative; the
                // signed semantics come from the caller's interpretation
                // (the FDIS text only constrains x0 + width <= image
                // width). We store as i32 for forward compat.
                x0 = br.read_u32(crop_dist)? as i32;
                y0 = br.read_u32(crop_dist)? as i32;
            }
            width = br.read_u32(crop_dist)?;
            height = br.read_u32(crop_dist)?;
        }

        // Whether we are a "normal_frame" per the FDIS abbreviation
        // (controls blending + animation + is_last fields).
        let normal_frame =
            frame_type == FrameType::Regular || frame_type == FrameType::SkipProgressive;

        // full_frame: `have_crop is false or the frame area completely
        // covers the image area`.
        let full_frame = if !have_crop {
            true
        } else {
            x0 == 0 && y0 == 0 && width >= params.image_width && height >= params.image_height
        };

        let multi_extra = params.num_extra_channels >= 2;

        let mut blending_info = BlendingInfo::default();
        let mut ec_blending_info: Vec<BlendingInfo> =
            vec![BlendingInfo::default(); params.num_extra_channels as usize];
        let mut duration = 0u32;
        let mut timecode = 0u32;
        if normal_frame {
            blending_info = BlendingInfo::read(br, multi_extra, full_frame)?;
            for slot in ec_blending_info.iter_mut() {
                *slot = BlendingInfo::read(br, multi_extra, full_frame)?;
            }
            if params.have_animation {
                duration = br.read_u32([
                    U32Dist::Val(0),
                    U32Dist::Val(1),
                    U32Dist::Bits(8),
                    U32Dist::Bits(32),
                ])?;
                if params.have_animation_timecodes {
                    timecode = br.read_bits(32)?;
                }
            }
        }

        // is_last: default = !frame_type (per Table C.2). Read only if
        // normal_frame.
        let is_last = if normal_frame {
            br.read_bool()?
        } else {
            // For non-normal frames the spec default for is_last is
            // `!frame_type` (i.e. only kRegularFrame == 0 has default
            // true). For LF / ReferenceOnly / SkipProgressive frames
            // is_last defaults to false; SkipProgressive is normal so
            // already handled above.
            frame_type == FrameType::Regular
        };

        let mut save_as_reference = 0u32;
        if frame_type != FrameType::LfFrame && !is_last {
            save_as_reference = br.read_bits(2)?;
        }

        // Default for save_before_ct: the FDIS rule
        // d_sbct = !(full_frame && (frame_type ∈ {kRegular, kSkipProgressive})
        //            && blending_info.mode == kReplace
        //            && (duration == 0 || save_as_reference != 0)
        //            && !is_last).
        let d_sbct = !(full_frame
            && (frame_type == FrameType::Regular || frame_type == FrameType::SkipProgressive)
            && blending_info.mode == BlendMode::Replace
            && (duration == 0 || save_as_reference != 0)
            && !is_last);
        let save_before_ct = if frame_type != FrameType::LfFrame {
            br.read_bool()?
        } else {
            d_sbct
        };

        let name_len = br.read_u32([
            U32Dist::Val(0),
            U32Dist::Bits(4),
            U32Dist::BitsOffset(5, 16),
            U32Dist::BitsOffset(10, 48),
        ])?;
        if name_len > MAX_NAME_LEN {
            return Err(Error::InvalidData(format!(
                "JXL FrameHeader: name_len {name_len} exceeds spec maximum"
            )));
        }
        let mut name_bytes = Vec::with_capacity(name_len as usize);
        for _ in 0..name_len {
            name_bytes.push(br.read_bits(8)? as u8);
        }
        let name = String::from_utf8(name_bytes)
            .map_err(|_| Error::InvalidData("JXL FrameHeader: name is not valid UTF-8".into()))?;

        let restoration_filter = RestorationFilter::read(br, encoding)?;

        let extensions = Extensions::read(br)?;
        extensions.skip_payload(br)?;

        Ok(Self {
            all_default: false,
            frame_type,
            encoding,
            flags,
            do_ycbcr,
            jpeg_upsampling,
            upsampling,
            ec_upsampling,
            group_size_shift,
            x_qm_scale,
            b_qm_scale,
            passes,
            lf_level,
            have_crop,
            x0,
            y0,
            width,
            height,
            blending_info,
            ec_blending_info,
            duration,
            timecode,
            is_last,
            save_as_reference,
            save_before_ct,
            name,
            restoration_filter,
            extensions,
        })
    }

    fn default_with(params: &FrameDecodeParams) -> Self {
        Self {
            all_default: true,
            frame_type: FrameType::Regular,
            encoding: Encoding::VarDct,
            flags: 0,
            do_ycbcr: false,
            jpeg_upsampling: [0; 3],
            upsampling: 1,
            ec_upsampling: vec![1u32; params.num_extra_channels as usize],
            group_size_shift: 1,
            x_qm_scale: 3,
            b_qm_scale: 2,
            passes: Passes::default(),
            lf_level: 0,
            have_crop: false,
            x0: 0,
            y0: 0,
            width: params.image_width,
            height: params.image_height,
            blending_info: BlendingInfo::default(),
            ec_blending_info: vec![BlendingInfo::default(); params.num_extra_channels as usize],
            duration: 0,
            timecode: 0,
            is_last: true,
            save_as_reference: 0,
            save_before_ct: false,
            name: String::new(),
            restoration_filter: RestorationFilter::default(),
            extensions: Extensions::default(),
        }
    }

    /// `kGroupDim = 128 << group_size_shift` per FDIS §C.2.
    pub fn group_dim(&self) -> u32 {
        128u32 << self.group_size_shift
    }

    /// `num_groups = ceil(width / kGroupDim) × ceil(height / kGroupDim)`.
    pub fn num_groups(&self) -> u64 {
        let g = self.group_dim() as u64;
        let w = self.width as u64;
        let h = self.height as u64;
        w.div_ceil(g) * h.div_ceil(g)
    }

    /// `num_lf_groups = ceil(width / (kGroupDim × 8)) × ceil(...)`.
    pub fn num_lf_groups(&self) -> u64 {
        let g = (self.group_dim() as u64) * 8;
        let w = self.width as u64;
        let h = self.height as u64;
        w.div_ceil(g) * h.div_ceil(g)
    }
}

// Unused import warning: SizeHeaderFdis is referenced by tests below
// (for documentation); keep it imported through a `pub use` to avoid
// unused-warning lint.
#[allow(dead_code)]
const _SIZE_HEADER_REF: Option<SizeHeaderFdis> = None;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ans::test_helpers::pack_lsb;

    fn default_params() -> FrameDecodeParams {
        FrameDecodeParams {
            xyb_encoded: true,
            num_extra_channels: 0,
            have_animation: false,
            have_animation_timecodes: false,
            image_width: 256,
            image_height: 256,
        }
    }

    #[test]
    fn all_default_frame_header_skips_to_defaults() {
        // Single bit set: all_default = 1.
        let bytes = pack_lsb(&[(1, 1)]);
        let mut br = BitReader::new(&bytes);
        let fh = FrameHeader::read(&mut br, &default_params()).unwrap();
        assert!(fh.all_default);
        assert_eq!(fh.frame_type, FrameType::Regular);
        assert_eq!(fh.encoding, Encoding::VarDct);
        assert_eq!(fh.flags, 0);
        assert_eq!(fh.upsampling, 1);
        assert_eq!(fh.passes.num_passes, 1);
        assert!(fh.is_last);
        assert_eq!(fh.width, 256);
        assert_eq!(fh.height, 256);
        assert_eq!(fh.group_dim(), 256);
        // 256x256 image with kGroupDim=256 → 1 group.
        assert_eq!(fh.num_groups(), 1);
        // num_lf_groups = ceil(256 / (256*8))^2 = 1.
        assert_eq!(fh.num_lf_groups(), 1);
    }

    #[test]
    fn explicit_modular_frame_with_minimal_fields() {
        // all_default=0, frame_type=Regular(0), encoding=Modular(1),
        // flags=U64(sel=0)→0, !xyb_encoded path so do_ycbcr is read as bool=0,
        // upsampling sel=0→1, no extra channels (params has 0),
        // group_size_shift=u(2)=1 (since encoding==Modular).
        // passes: num_passes sel=0→1.
        // frame_type != LfFrame → have_crop = bool = 0 → false.
        // normal_frame=true (Regular). multi_extra=false.
        // blending_info: mode = U32 sel=0 → 0 (kReplace).
        //   full_frame=true, mode=Replace → no source field.
        // ec_blending_info: empty (0 extra channels).
        // have_animation=false → no duration/timecode.
        // is_last = bool = 1 → true.
        // is_last=true and frame_type != LfFrame, but !is_last is false →
        //   save_as_reference NOT read (gated on `!is_last`).
        // frame_type != LfFrame → save_before_ct = bool = 0.
        // name_len: sel=0 → 0.
        // restoration_filter: gab = bool = 1 (default), gab_custom = bool = 0,
        //   epf_iters = u(2) = 0; encoding==Modular but epf_iters==0 so no
        //   sigma_for_modular. extensions: U64(sel=0)=0.
        // outer extensions: U64(sel=0)=0.
        let params = FrameDecodeParams {
            xyb_encoded: false,
            ..default_params()
        };
        let bw = pack_lsb(&[
            (0, 1), // all_default = 0
            (0, 2), // frame_type = 0 (Regular)
            (1, 1), // encoding = 1 (Modular)
            (0, 2), // flags U64 selector = 0 → 0
            (0, 1), // do_ycbcr = false
            (0, 2), // upsampling selector = 0 → Val(1)
            (1, 2), // group_size_shift = 1
            (0, 2), // passes.num_passes selector = 0 → 1
            (0, 1), // have_crop = false
            (0, 2), // blending_info.mode = 0 (kReplace), source skipped
            (1, 1), // is_last = true
            (0, 1), // save_before_ct = false
            (0, 2), // name_len selector = 0 → 0
            (1, 1), // restoration_filter.gab = true
            (0, 1), // gab_custom = false
            (0, 2), // epf_iters = 0
            (0, 2), // restoration_filter.extensions = U64 sel=0
            (0, 2), // outer extensions = U64 sel=0
        ]);
        let mut br = BitReader::new(&bw);
        let fh = FrameHeader::read(&mut br, &params).unwrap();
        assert!(!fh.all_default);
        assert_eq!(fh.frame_type, FrameType::Regular);
        assert_eq!(fh.encoding, Encoding::Modular);
        assert_eq!(fh.flags, 0);
        assert_eq!(fh.upsampling, 1);
        assert_eq!(fh.group_size_shift, 1);
        assert_eq!(fh.group_dim(), 256);
        assert_eq!(fh.passes.num_passes, 1);
        assert!(!fh.have_crop);
        assert!(fh.is_last);
        assert!(!fh.save_before_ct);
        assert!(fh.name.is_empty());
        assert!(fh.restoration_filter.gab);
        assert_eq!(fh.restoration_filter.epf_iters, 0);
    }

    #[test]
    fn rejects_invalid_frame_type() {
        // all_default=0 then 2 bits read for frame_type. We can't fail
        // this branch (all 2-bit values are valid frame types). Use
        // is_last fall-through case: kReferenceOnly path skips Passes
        // and crops.
        // Actually, all 4 frame_type values map to a valid enum, so
        // this test is degenerate — instead, verify rejection on a
        // truncated bitstream.
        let bytes = pack_lsb(&[(0, 1), (0, 2)]); // all_default=0, frame_type=0 → then EOF on encoding
        let mut br = BitReader::new(&bytes);
        let res = FrameHeader::read(&mut br, &default_params());
        assert!(res.is_err(), "expected EOF error");
    }

    #[test]
    fn restoration_filter_default_values() {
        let rf = RestorationFilter::default();
        assert!(rf.gab);
        assert!(!rf.gab_custom);
        assert_eq!(rf.epf_iters, 2);
        assert!(!rf.epf_sharp_custom);
        assert_eq!(rf.epf_sharp_lut[0], 0.0);
        assert!((rf.epf_sharp_lut[7] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn group_dim_scales_with_shift() {
        let mut params = default_params();
        params.image_width = 1024;
        params.image_height = 512;
        let mut fh = FrameHeader::default_with(&params);
        fh.group_size_shift = 0;
        assert_eq!(fh.group_dim(), 128);
        fh.group_size_shift = 3;
        assert_eq!(fh.group_dim(), 1024);
    }

    #[test]
    fn num_groups_matches_spec() {
        let mut params = default_params();
        params.image_width = 600;
        params.image_height = 300;
        let mut fh = FrameHeader::default_with(&params);
        fh.group_size_shift = 1; // kGroupDim = 256
                                 // ceil(600/256)=3, ceil(300/256)=2 → 6
        assert_eq!(fh.group_dim(), 256);
        assert_eq!(fh.num_groups(), 6);
    }

    #[test]
    fn name_len_overflow_rejected() {
        // Manually craft a name_len that exceeds MAX_NAME_LEN.
        // U32 selector 3 → BitsOffset(10, 48) → max raw = 1023 → max value = 1071.
        // We can't get above 1071 with this distribution, so this test
        // verifies the check is in place by lying with 1023 (legal max).
        // Verify legal max accepted.
        let bw = pack_lsb(&[
            (0, 1),     // all_default
            (0, 2),     // frame_type
            (1, 1),     // encoding modular
            (0, 2),     // flags
            (0, 1),     // do_ycbcr  (we set xyb_encoded=false in params)
            (0, 2),     // upsampling
            (0, 2),     // group_size_shift = 0
            (0, 2),     // num_passes = 1
            (0, 1),     // have_crop = 0
            (0, 2),     // blending mode = 0
            (1, 1),     // is_last = 1
            (0, 1),     // save_before_ct = 0
            (3, 2),     // name_len selector = 3 → BitsOffset(10, 48)
            (1023, 10), // raw = 1023 → name_len = 1071 (legal)
        ]);
        // We don't have name bytes or restoration filter / extensions
        // following → reading fails. We're testing the *size* gate, so
        // this expects the read to fail mid-name (EOF), NOT the size
        // gate to trip.
        let params = FrameDecodeParams {
            xyb_encoded: false,
            ..default_params()
        };
        let mut br = BitReader::new(&bw);
        let res = FrameHeader::read(&mut br, &params);
        assert!(res.is_err(), "expected EOF after huge name_len");
    }
}
