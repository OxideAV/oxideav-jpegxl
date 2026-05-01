//! `ImageMetadata` (full FDIS A.6 form), `ColourEncoding` (A.4),
//! `ToneMapping` (A.6 Table A.18), `ExtraChannelInfo` (A.9),
//! `AnimationHeader` (A.7), `OpsinInverseMatrix` (A.8), and the
//! `SizeHeader` re-decoded against the FDIS published in 2021.
//!
//! This module is **additive** alongside the original [`crate::metadata`]
//! parsers (which decode just enough of the preamble for `probe()` and
//! were authored against the 2019 committee draft). The FDIS fully
//! parses every field, including ColourEncoding + ToneMapping +
//! ExtraChannelInfo + the `default_transform` / cw_mask sections that
//! were stubbed in the original. Round-3 wiring will switch the
//! aggregator's `make_decoder` over to this module.
//!
//! Allocation bounds: every per-channel / per-name / per-weights array
//! is sized by an upstream `num_*` field whose maximum is fixed by the
//! `U32` distribution it was decoded from (see `MAX_NUM_EXTRA_CHANNELS`
//! and `MAX_NAME_LEN`).

use oxideav_core::{Error, Result};

use crate::bitreader::{unpack_signed, BitReader, U32Dist};
use crate::extensions::Extensions;

/// Re-decoded `SizeHeader` per FDIS Table A.3. Identical wire format to
/// the existing [`crate::metadata::SizeHeader`]; lifted here so the
/// FDIS-side parsers do not have to depend on the older module.
#[derive(Debug, Clone, Copy)]
pub struct SizeHeaderFdis {
    pub width: u32,
    pub height: u32,
    pub small: bool,
    pub ratio: u8,
}

const FIXED_ASPECT_RATIOS: [(u32, u32); 7] =
    [(1, 1), (12, 10), (4, 3), (3, 2), (16, 9), (5, 4), (2, 1)];

impl SizeHeaderFdis {
    /// FDIS Table A.3.
    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let small = br.read_bool()?;
        let height = if small {
            (br.read_bits(5)? + 1) * 8
        } else {
            br.read_u32([
                U32Dist::BitsOffset(9, 1),
                U32Dist::BitsOffset(13, 1),
                U32Dist::BitsOffset(18, 1),
                U32Dist::BitsOffset(30, 1),
            ])?
        };
        let ratio = br.read_bits(3)? as u8;
        let width = if ratio == 0 {
            if small {
                (br.read_bits(5)? + 1) * 8
            } else {
                br.read_u32([
                    U32Dist::BitsOffset(9, 1),
                    U32Dist::BitsOffset(13, 1),
                    U32Dist::BitsOffset(18, 1),
                    U32Dist::BitsOffset(30, 1),
                ])?
            }
        } else if ratio <= 7 {
            let (num, den) = FIXED_ASPECT_RATIOS[(ratio - 1) as usize];
            ((height as u64 * num as u64) / den as u64) as u32
        } else {
            return Err(Error::InvalidData(format!(
                "JXL SizeHeader (FDIS): invalid ratio {ratio}"
            )));
        };
        if width == 0 || height == 0 {
            return Err(Error::InvalidData(
                "JXL SizeHeader (FDIS): zero-dimensional image".into(),
            ));
        }
        Ok(Self {
            width,
            height,
            small,
            ratio,
        })
    }
}

/// `BitDepth` per FDIS Table A.15.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BitDepthFdis {
    pub float_sample: bool,
    pub bits_per_sample: u32,
    /// Only meaningful when `float_sample == true`. Equals
    /// `exp_bits_minus_one + 1`.
    pub exp_bits: u32,
}

impl BitDepthFdis {
    pub const DEFAULT: Self = Self {
        float_sample: false,
        bits_per_sample: 8,
        exp_bits: 0,
    };

    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let float_sample = br.read_bool()?;
        if !float_sample {
            let bits_per_sample = br.read_u32([
                U32Dist::Val(8),
                U32Dist::Val(10),
                U32Dist::Val(12),
                U32Dist::BitsOffset(6, 1),
            ])?;
            if !(1..=31).contains(&bits_per_sample) {
                return Err(Error::InvalidData(format!(
                    "JXL BitDepth: integer bits_per_sample {bits_per_sample} out of [1, 31]"
                )));
            }
            Ok(Self {
                float_sample: false,
                bits_per_sample,
                exp_bits: 0,
            })
        } else {
            let bits_per_sample = br.read_u32([
                U32Dist::Val(32),
                U32Dist::Val(16),
                U32Dist::Val(24),
                U32Dist::BitsOffset(6, 1),
            ])?;
            let exp_bits_minus_one = br.read_bits(4)?;
            let exp_bits = exp_bits_minus_one + 1;
            if !(2..=8).contains(&exp_bits) {
                return Err(Error::InvalidData(format!(
                    "JXL BitDepth: exp_bits {exp_bits} out of [2, 8]"
                )));
            }
            let mantissa_bits = bits_per_sample as i64 - exp_bits as i64 - 1;
            if !(2..=23).contains(&mantissa_bits) {
                return Err(Error::InvalidData(format!(
                    "JXL BitDepth: float bits_per_sample {bits_per_sample} produces invalid mantissa {mantissa_bits}"
                )));
            }
            Ok(Self {
                float_sample: true,
                bits_per_sample,
                exp_bits,
            })
        }
    }
}

/// `Customxy` per FDIS Table A.6 — a (x, y) chromaticity coordinate
/// pair scaled by 1e6.
#[derive(Debug, Clone, Copy)]
pub struct Customxy {
    pub x: i32,
    pub y: i32,
}

impl Customxy {
    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let dist = [
            U32Dist::Bits(19),
            U32Dist::BitsOffset(19, 524288),
            U32Dist::BitsOffset(20, 1048576),
            U32Dist::BitsOffset(21, 2097152),
        ];
        let ux = br.read_u32(dist)?;
        let uy = br.read_u32(dist)?;
        Ok(Self {
            x: unpack_signed(ux),
            y: unpack_signed(uy),
        })
    }
}

/// `ColourSpace` enum — FDIS Table A.7.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColourSpace {
    Rgb = 0,
    Grey = 1,
    Xyb = 2,
    Unknown = 3,
}

impl ColourSpace {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(ColourSpace::Rgb),
            1 => Ok(ColourSpace::Grey),
            2 => Ok(ColourSpace::Xyb),
            3 => Ok(ColourSpace::Unknown),
            _ => Err(Error::InvalidData(format!(
                "JXL ColourSpace: invalid value {v}"
            ))),
        }
    }
}

/// `WhitePoint` enum — FDIS Table A.8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhitePoint {
    D65 = 1,
    Custom = 2,
    E = 10,
    Dci = 11,
}

impl WhitePoint {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            1 => Ok(WhitePoint::D65),
            2 => Ok(WhitePoint::Custom),
            10 => Ok(WhitePoint::E),
            11 => Ok(WhitePoint::Dci),
            _ => Err(Error::InvalidData(format!(
                "JXL WhitePoint: invalid value {v}"
            ))),
        }
    }
}

/// `Primaries` enum — FDIS Table A.9.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Primaries {
    SRgb = 1,
    Custom = 2,
    P2100 = 9,
    P3 = 11,
}

impl Primaries {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            1 => Ok(Primaries::SRgb),
            2 => Ok(Primaries::Custom),
            9 => Ok(Primaries::P2100),
            11 => Ok(Primaries::P3),
            _ => Err(Error::InvalidData(format!(
                "JXL Primaries: invalid value {v}"
            ))),
        }
    }
}

/// `TransferFunction` enum — FDIS Table A.10.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferFunction {
    Bt709 = 1,
    Unknown = 2,
    Linear = 8,
    SRgb = 13,
    Pq = 16,
    Dci = 17,
    Hlg = 18,
}

impl TransferFunction {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            1 => Ok(TransferFunction::Bt709),
            2 => Ok(TransferFunction::Unknown),
            8 => Ok(TransferFunction::Linear),
            13 => Ok(TransferFunction::SRgb),
            16 => Ok(TransferFunction::Pq),
            17 => Ok(TransferFunction::Dci),
            18 => Ok(TransferFunction::Hlg),
            _ => Err(Error::InvalidData(format!(
                "JXL TransferFunction: invalid value {v}"
            ))),
        }
    }
}

/// `RenderingIntent` enum — FDIS Table A.12.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderingIntent {
    Perceptual = 0,
    Relative = 1,
    Saturation = 2,
    Absolute = 3,
}

impl RenderingIntent {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(RenderingIntent::Perceptual),
            1 => Ok(RenderingIntent::Relative),
            2 => Ok(RenderingIntent::Saturation),
            3 => Ok(RenderingIntent::Absolute),
            _ => Err(Error::InvalidData(format!(
                "JXL RenderingIntent: invalid value {v}"
            ))),
        }
    }
}

/// FDIS §9.2.8 — `Enum(...)` reads `U32(Val(0), Val(1), BitsOffset(4, 2),
/// BitsOffset(6, 18))`. Caps at 63.
fn read_enum_u32(br: &mut BitReader<'_>) -> Result<u32> {
    let v = br.read_u32([
        U32Dist::Val(0),
        U32Dist::Val(1),
        U32Dist::BitsOffset(4, 2),
        U32Dist::BitsOffset(6, 18),
    ])?;
    if v > 63 {
        return Err(Error::InvalidData(
            "JXL Enum(): value > 63 (per §9.2.8)".into(),
        ));
    }
    Ok(v)
}

/// `CustomTransferFunction` per FDIS Table A.11.
#[derive(Debug, Clone, Copy)]
pub struct CustomTransferFunction {
    pub have_gamma: bool,
    /// Only valid when `have_gamma == true`. Decoded from `u(24)`.
    /// Real opto-electrical exponent is `gamma / 1e7`.
    pub gamma: u32,
    pub transfer_function: TransferFunction,
}

impl Default for CustomTransferFunction {
    fn default() -> Self {
        Self {
            have_gamma: false,
            gamma: 10_000_000,
            transfer_function: TransferFunction::SRgb,
        }
    }
}

impl CustomTransferFunction {
    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let have_gamma = br.read_bool()?;
        if have_gamma {
            let gamma = br.read_bits(24)?;
            // gamma represents an exponent in (0, 1] when divided by 1e7.
            if gamma == 0 || gamma > 10_000_000 {
                return Err(Error::InvalidData(format!(
                    "JXL CustomTransferFunction: gamma {gamma} out of (0, 10_000_000]"
                )));
            }
            Ok(Self {
                have_gamma: true,
                gamma,
                transfer_function: TransferFunction::SRgb,
            })
        } else {
            let v = read_enum_u32(br)?;
            Ok(Self {
                have_gamma: false,
                gamma: 10_000_000,
                transfer_function: TransferFunction::from_u32(v)?,
            })
        }
    }
}

/// `ColourEncoding` bundle — FDIS Table A.13.
#[derive(Debug, Clone)]
pub struct ColourEncoding {
    pub all_default: bool,
    pub want_icc: bool,
    pub colour_space: ColourSpace,
    pub white_point: WhitePoint,
    pub white: Option<Customxy>,
    pub primaries: Primaries,
    pub red: Option<Customxy>,
    pub green: Option<Customxy>,
    pub blue: Option<Customxy>,
    pub tf: CustomTransferFunction,
    pub rendering_intent: RenderingIntent,
}

impl Default for ColourEncoding {
    fn default() -> Self {
        Self {
            all_default: true,
            want_icc: false,
            colour_space: ColourSpace::Rgb,
            white_point: WhitePoint::D65,
            white: None,
            primaries: Primaries::SRgb,
            red: None,
            green: None,
            blue: None,
            tf: CustomTransferFunction::default(),
            rendering_intent: RenderingIntent::Relative,
        }
    }
}

impl ColourEncoding {
    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let all_default = br.read_bool()?;
        if all_default {
            return Ok(Self::default());
        }
        let want_icc = br.read_bool()?;
        let colour_space = ColourSpace::from_u32(read_enum_u32(br)?)?;
        let use_desc = !want_icc;
        let not_xyb = colour_space != ColourSpace::Xyb;

        let mut white_point = WhitePoint::D65;
        let mut white = None;
        if use_desc && not_xyb {
            white_point = WhitePoint::from_u32(read_enum_u32(br)?)?;
            if white_point == WhitePoint::Custom {
                white = Some(Customxy::read(br)?);
            }
        }

        let mut primaries = Primaries::SRgb;
        let (mut red, mut green, mut blue) = (None, None, None);
        if use_desc && not_xyb && colour_space != ColourSpace::Grey {
            primaries = Primaries::from_u32(read_enum_u32(br)?)?;
            if primaries == Primaries::Custom {
                red = Some(Customxy::read(br)?);
                green = Some(Customxy::read(br)?);
                blue = Some(Customxy::read(br)?);
            }
        }

        let tf = if use_desc {
            CustomTransferFunction::read(br)?
        } else {
            CustomTransferFunction::default()
        };
        let rendering_intent = if use_desc {
            RenderingIntent::from_u32(read_enum_u32(br)?)?
        } else {
            RenderingIntent::Relative
        };

        Ok(Self {
            all_default: false,
            want_icc,
            colour_space,
            white_point,
            white,
            primaries,
            red,
            green,
            blue,
            tf,
            rendering_intent,
        })
    }
}

/// `ToneMapping` bundle — FDIS Table A.18.
#[derive(Debug, Clone, Copy)]
pub struct ToneMapping {
    pub all_default: bool,
    pub intensity_target: f32,
    pub min_nits: f32,
    pub relative_to_max_display: bool,
    pub linear_below: f32,
}

impl Default for ToneMapping {
    fn default() -> Self {
        Self {
            all_default: true,
            intensity_target: 255.0,
            min_nits: 0.0,
            relative_to_max_display: false,
            linear_below: 0.0,
        }
    }
}

impl ToneMapping {
    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let all_default = br.read_bool()?;
        if all_default {
            return Ok(Self::default());
        }
        let intensity_target = br.read_f16()?;
        let min_nits = br.read_f16()?;
        let relative_to_max_display = br.read_bool()?;
        let linear_below = br.read_f16()?;
        // Spec rule: intensity_target > 0 AND 0 <= min_nits <= intensity_target.
        // F16-decoded values are guaranteed finite (read_f16 rejects
        // NaN/Inf) so a direct comparison is well-defined.
        if intensity_target <= 0.0 {
            return Err(Error::InvalidData(
                "JXL ToneMapping: intensity_target must be > 0".into(),
            ));
        }
        if min_nits < 0.0 || min_nits > intensity_target {
            return Err(Error::InvalidData(
                "JXL ToneMapping: min_nits out of [0, intensity_target]".into(),
            ));
        }
        Ok(Self {
            all_default: false,
            intensity_target,
            min_nits,
            relative_to_max_display,
            linear_below,
        })
    }
}

/// `ExtraChannelType` enum — FDIS Table A.21.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtraChannelType {
    Alpha = 0,
    Depth = 1,
    SpotColour = 2,
    SelectionMask = 3,
    Black = 4,
    Cfa = 5,
    Thermal = 6,
    NonOptional = 15,
    Optional = 16,
}

impl ExtraChannelType {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(ExtraChannelType::Alpha),
            1 => Ok(ExtraChannelType::Depth),
            2 => Ok(ExtraChannelType::SpotColour),
            3 => Ok(ExtraChannelType::SelectionMask),
            4 => Ok(ExtraChannelType::Black),
            5 => Ok(ExtraChannelType::Cfa),
            6 => Ok(ExtraChannelType::Thermal),
            15 => Ok(ExtraChannelType::NonOptional),
            16 => Ok(ExtraChannelType::Optional),
            _ => Err(Error::InvalidData(format!(
                "JXL ExtraChannelType: invalid value {v}"
            ))),
        }
    }
}

/// `ExtraChannelInfo` bundle — FDIS Table A.22.
#[derive(Debug, Clone)]
pub struct ExtraChannelInfo {
    pub all_default: bool,
    pub kind: ExtraChannelType,
    pub bit_depth: BitDepthFdis,
    pub dim_shift: u32,
    pub name: String,
    pub alpha_associated: bool,
    pub spot_red: f32,
    pub spot_green: f32,
    pub spot_blue: f32,
    pub spot_solidity: f32,
    pub cfa_channel: u32,
}

impl Default for ExtraChannelInfo {
    fn default() -> Self {
        Self {
            all_default: true,
            kind: ExtraChannelType::Alpha,
            bit_depth: BitDepthFdis::DEFAULT,
            dim_shift: 0,
            name: String::new(),
            alpha_associated: false,
            spot_red: 0.0,
            spot_green: 0.0,
            spot_blue: 0.0,
            spot_solidity: 0.0,
            cfa_channel: 1,
        }
    }
}

const MAX_EC_NAME_LEN: u32 = 1071;

impl ExtraChannelInfo {
    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let all_default = br.read_bool()?;
        if all_default {
            return Ok(Self::default());
        }
        let kind = ExtraChannelType::from_u32(read_enum_u32(br)?)?;
        let bit_depth = BitDepthFdis::read(br)?;
        let dim_shift = br.read_u32([
            U32Dist::Val(0),
            U32Dist::Val(3),
            U32Dist::Val(4),
            U32Dist::BitsOffset(3, 1),
        ])?;
        let name_len = br.read_u32([
            U32Dist::Val(0),
            U32Dist::Bits(4),
            U32Dist::BitsOffset(5, 16),
            U32Dist::BitsOffset(10, 48),
        ])?;
        if name_len > MAX_EC_NAME_LEN {
            return Err(Error::InvalidData(format!(
                "JXL ExtraChannelInfo: name_len {name_len} exceeds spec maximum"
            )));
        }
        let mut name_bytes = Vec::with_capacity(name_len as usize);
        for _ in 0..name_len {
            name_bytes.push(br.read_bits(8)? as u8);
        }
        let name = String::from_utf8(name_bytes)
            .map_err(|_| Error::InvalidData("JXL ExtraChannelInfo: name not valid UTF-8".into()))?;
        let alpha_associated = if kind == ExtraChannelType::Alpha {
            br.read_bool()?
        } else {
            false
        };
        let (spot_red, spot_green, spot_blue, spot_solidity) =
            if kind == ExtraChannelType::SpotColour {
                (
                    br.read_f16()?,
                    br.read_f16()?,
                    br.read_f16()?,
                    br.read_f16()?,
                )
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };
        let cfa_channel = if kind == ExtraChannelType::Cfa {
            br.read_u32([
                U32Dist::Val(1),
                U32Dist::Bits(2),
                U32Dist::BitsOffset(4, 3),
                U32Dist::BitsOffset(8, 19),
            ])?
        } else {
            1
        };
        Ok(Self {
            all_default: false,
            kind,
            bit_depth,
            dim_shift,
            name,
            alpha_associated,
            spot_red,
            spot_green,
            spot_blue,
            spot_solidity,
            cfa_channel,
        })
    }
}

/// `AnimationHeader` bundle — FDIS Table A.19.
#[derive(Debug, Clone, Copy)]
pub struct AnimationHeader {
    pub tps_numerator: u32,
    pub tps_denominator: u32,
    pub num_loops: u32,
    pub have_timecodes: bool,
}

impl AnimationHeader {
    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let tps_numerator = br.read_u32([
            U32Dist::Val(100),
            U32Dist::Val(1000),
            U32Dist::BitsOffset(10, 1),
            U32Dist::BitsOffset(30, 1),
        ])?;
        let tps_denominator = br.read_u32([
            U32Dist::Val(1),
            U32Dist::Val(1001),
            U32Dist::BitsOffset(8, 1),
            U32Dist::BitsOffset(10, 1),
        ])?;
        let num_loops = br.read_u32([
            U32Dist::Val(0),
            U32Dist::Bits(3),
            U32Dist::Bits(16),
            U32Dist::Bits(32),
        ])?;
        let have_timecodes = br.read_bool()?;
        Ok(Self {
            tps_numerator,
            tps_denominator,
            num_loops,
            have_timecodes,
        })
    }
}

/// `PreviewHeader` bundle — FDIS Table A.5. Width/height are bounded to
/// 4096 per the spec.
#[derive(Debug, Clone, Copy)]
pub struct PreviewHeader {
    pub width: u32,
    pub height: u32,
}

impl PreviewHeader {
    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let div8 = br.read_bool()?;
        let height = if div8 {
            let h_div8 = br.read_u32([
                U32Dist::Val(16),
                U32Dist::Val(32),
                U32Dist::BitsOffset(5, 1),
                U32Dist::BitsOffset(9, 33),
            ])?;
            h_div8 * 8
        } else {
            br.read_u32([
                U32Dist::BitsOffset(6, 1),
                U32Dist::BitsOffset(8, 65),
                U32Dist::BitsOffset(10, 321),
                U32Dist::BitsOffset(12, 1345),
            ])?
        };
        let ratio = br.read_bits(3)? as u8;
        let width = if ratio == 0 {
            if div8 {
                let w_div8 = br.read_u32([
                    U32Dist::Val(16),
                    U32Dist::Val(32),
                    U32Dist::BitsOffset(5, 1),
                    U32Dist::BitsOffset(9, 33),
                ])?;
                w_div8 * 8
            } else {
                br.read_u32([
                    U32Dist::BitsOffset(6, 1),
                    U32Dist::BitsOffset(8, 65),
                    U32Dist::BitsOffset(10, 321),
                    U32Dist::BitsOffset(12, 1345),
                ])?
            }
        } else if ratio <= 7 {
            let (num, den) = FIXED_ASPECT_RATIOS[(ratio - 1) as usize];
            ((height as u64 * num as u64) / den as u64) as u32
        } else {
            return Err(Error::InvalidData(format!(
                "JXL PreviewHeader: invalid ratio {ratio}"
            )));
        };
        if width == 0 || height == 0 || width > 4096 || height > 4096 {
            return Err(Error::InvalidData(format!(
                "JXL PreviewHeader: dimensions {width}x{height} out of bounds"
            )));
        }
        Ok(Self { width, height })
    }
}

/// `OpsinInverseMatrix` per FDIS Table A.20.
///
/// We currently store the all_default flag plus the parsed values; if
/// all_default is true, the values are the FDIS-listed defaults. The
/// upsampling weight tables (cw_mask) live alongside `ImageMetadataFdis`
/// because they're not part of this bundle.
#[derive(Debug, Clone, Copy)]
pub struct OpsinInverseMatrix {
    pub all_default: bool,
    pub inv_mat: [[f32; 3]; 3],
    pub opsin_bias: [f32; 3],
    pub quant_bias: [f32; 3],
    pub quant_bias_numerator: f32,
}

impl Default for OpsinInverseMatrix {
    fn default() -> Self {
        // The FDIS spec gives these constants in full f64 precision.
        // We store as f32 (matching the F16-decoded payload type) and
        // tolerate rounding by writing the source spec value as `f64
        // as f32` so the audit trail remains intact.
        Self {
            all_default: true,
            inv_mat: [
                [
                    11.031_566_901_960_783_f64 as f32,
                    -9.866_943_921_568_629_f64 as f32,
                    -0.164_622_996_470_588_26_f64 as f32,
                ],
                [
                    -3.254_147_380_392_157_f64 as f32,
                    4.418_770_392_156_863_f64 as f32,
                    -0.164_622_996_470_588_26_f64 as f32,
                ],
                [
                    -3.658_851_286_274_509_7_f64 as f32,
                    2.712_923_047_058_823_5_f64 as f32,
                    1.945_928_239_215_686_3_f64 as f32,
                ],
            ],
            opsin_bias: [
                -0.003_793_073_255_275_449_3_f64 as f32,
                -0.003_793_073_255_275_449_3_f64 as f32,
                -0.003_793_073_255_275_449_3_f64 as f32,
            ],
            quant_bias: [
                (1.0 - 0.054_650_073_307_154_01_f64) as f32,
                (1.0 - 0.070_054_498_917_485_93_f64) as f32,
                (1.0 - 0.049_935_103_337_343_655_f64) as f32,
            ],
            quant_bias_numerator: 0.145,
        }
    }
}

impl OpsinInverseMatrix {
    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let all_default = br.read_bool()?;
        if all_default {
            return Ok(Self::default());
        }
        let mut inv_mat = [[0f32; 3]; 3];
        for row in inv_mat.iter_mut() {
            for v in row.iter_mut() {
                *v = br.read_f16()?;
            }
        }
        let mut opsin_bias = [0f32; 3];
        for v in opsin_bias.iter_mut() {
            *v = br.read_f16()?;
        }
        let mut quant_bias = [0f32; 3];
        for v in quant_bias.iter_mut() {
            *v = br.read_f16()?;
        }
        let quant_bias_numerator = br.read_f16()?;
        Ok(Self {
            all_default: false,
            inv_mat,
            opsin_bias,
            quant_bias,
            quant_bias_numerator,
        })
    }
}

/// Full `ImageMetadata` bundle — FDIS Table A.16. Decoded against the
/// 2021 published spec, including ColourEncoding, ToneMapping, and the
/// `default_transform` / cw_mask custom-upsampling-weights tail.
#[derive(Debug, Clone)]
pub struct ImageMetadataFdis {
    pub all_default: bool,
    pub extra_fields: bool,
    pub orientation: u8,
    pub have_intr_size: bool,
    pub intrinsic_size: Option<SizeHeaderFdis>,
    pub have_preview: bool,
    pub preview: Option<PreviewHeader>,
    pub have_animation: bool,
    pub animation: Option<AnimationHeader>,
    pub bit_depth: BitDepthFdis,
    pub modular_16bit_buffers: bool,
    pub num_extra_channels: u32,
    pub extra_channel_info: Vec<ExtraChannelInfo>,
    pub xyb_encoded: bool,
    pub colour_encoding: ColourEncoding,
    pub tone_mapping: ToneMapping,
    pub extensions: Extensions,
    pub default_transform: bool,
    pub opsin_inverse_matrix: OpsinInverseMatrix,
    pub cw_mask: u32,
}

const MAX_NUM_EXTRA_CHANNELS: u32 = 4097;

impl ImageMetadataFdis {
    /// FDIS Table A.16. Note that the cw_mask custom-weight arrays are
    /// not stored — they're decoded and immediately discarded since
    /// applying them is the upsampling routine's job (Annex L), not the
    /// metadata parser's.
    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let all_default = br.read_bool()?;
        let mut out = Self::defaults();
        out.all_default = all_default;
        if all_default {
            // Even when all_default, the bundle still emits the trailing
            // `default_transform` Bool() — the FDIS table shows it
            // unconditional. We just leave it at its (true) default and
            // do NOT consume bits for it; the spec says all_default
            // shortcircuits the entire bundle.
            return Ok(out);
        }
        out.extra_fields = br.read_bool()?;
        if out.extra_fields {
            out.orientation = (br.read_bits(3)? + 1) as u8;
            out.have_intr_size = br.read_bool()?;
            if out.have_intr_size {
                out.intrinsic_size = Some(SizeHeaderFdis::read(br)?);
            }
            out.have_preview = br.read_bool()?;
            if out.have_preview {
                out.preview = Some(PreviewHeader::read(br)?);
            }
            out.have_animation = br.read_bool()?;
            if out.have_animation {
                out.animation = Some(AnimationHeader::read(br)?);
            }
        }

        out.bit_depth = BitDepthFdis::read(br)?;
        out.modular_16bit_buffers = br.read_bool()?;
        out.num_extra_channels = br.read_u32([
            U32Dist::Val(0),
            U32Dist::Val(1),
            U32Dist::BitsOffset(4, 2),
            U32Dist::BitsOffset(12, 1),
        ])?;
        if out.num_extra_channels > MAX_NUM_EXTRA_CHANNELS {
            return Err(Error::InvalidData(format!(
                "JXL ImageMetadata: num_extra_channels {} exceeds spec maximum",
                out.num_extra_channels
            )));
        }
        // Per-extra-channel ExtraChannelInfo costs at least 1 bit
        // (all_default); refuse a count exceeding our remaining input.
        if (out.num_extra_channels as usize) > br.bits_remaining() {
            return Err(Error::InvalidData(
                "JXL ImageMetadata: num_extra_channels exceeds remaining input".into(),
            ));
        }
        out.extra_channel_info = Vec::with_capacity(out.num_extra_channels as usize);
        for _ in 0..out.num_extra_channels {
            out.extra_channel_info.push(ExtraChannelInfo::read(br)?);
        }
        out.xyb_encoded = br.read_bool()?;
        out.colour_encoding = ColourEncoding::read(br)?;
        if out.extra_fields {
            out.tone_mapping = ToneMapping::read(br)?;
        }
        out.extensions = Extensions::read(br)?;
        out.extensions.skip_payload(br)?;

        out.default_transform = br.read_bool()?;
        if out.default_transform && out.xyb_encoded {
            // FDIS Table A.16 row says
            //   default_transform && xyb_encoded → OpsinInverseMatrix
            // Note: this reads as opsin_inverse_matrix when *both* hold,
            // unlike the more common "non-default reads it" pattern.
            out.opsin_inverse_matrix = OpsinInverseMatrix::read(br)?;
        }
        if out.default_transform {
            out.cw_mask = br.read_bits(3)?;
            // Skip the per-mask custom upsampling weights (15+55+210
            // F16 fields). We bound against bits_remaining before
            // reading.
            if (out.cw_mask & 1) != 0 {
                Self::skip_f16_array(br, 15)?;
            }
            if (out.cw_mask & 2) != 0 {
                Self::skip_f16_array(br, 55)?;
            }
            if (out.cw_mask & 4) != 0 {
                Self::skip_f16_array(br, 210)?;
            }
        }
        Ok(out)
    }

    fn skip_f16_array(br: &mut BitReader<'_>, n: usize) -> Result<()> {
        // Each F16 read consumes 16 bits.
        if n.saturating_mul(16) > br.bits_remaining() {
            return Err(Error::InvalidData(
                "JXL ImageMetadata: cw_mask array exceeds remaining input".into(),
            ));
        }
        for _ in 0..n {
            let _ = br.read_f16()?;
        }
        Ok(())
    }

    fn defaults() -> Self {
        Self {
            all_default: true,
            extra_fields: false,
            orientation: 1,
            have_intr_size: false,
            intrinsic_size: None,
            have_preview: false,
            preview: None,
            have_animation: false,
            animation: None,
            bit_depth: BitDepthFdis::DEFAULT,
            modular_16bit_buffers: true,
            num_extra_channels: 0,
            extra_channel_info: Vec::new(),
            xyb_encoded: true,
            colour_encoding: ColourEncoding::default(),
            tone_mapping: ToneMapping::default(),
            extensions: Extensions::default(),
            default_transform: true,
            opsin_inverse_matrix: OpsinInverseMatrix::default(),
            cw_mask: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ans::test_helpers::pack_lsb;

    #[test]
    fn image_metadata_all_default_short_circuit() {
        let bytes = pack_lsb(&[(1, 1)]);
        let mut br = BitReader::new(&bytes);
        let im = ImageMetadataFdis::read(&mut br).unwrap();
        assert!(im.all_default);
        assert_eq!(im.orientation, 1);
        assert_eq!(im.bit_depth, BitDepthFdis::DEFAULT);
        assert!(im.xyb_encoded);
    }

    #[test]
    fn bit_depth_integer_8() {
        // float_sample = 0, bits_per_sample sel=0 → Val(8).
        let bytes = pack_lsb(&[(0, 1), (0, 2)]);
        let mut br = BitReader::new(&bytes);
        let bd = BitDepthFdis::read(&mut br).unwrap();
        assert!(!bd.float_sample);
        assert_eq!(bd.bits_per_sample, 8);
    }

    #[test]
    fn bit_depth_float_32_8() {
        // float_sample = 1, bits_per_sample sel=0 → Val(32),
        // exp_bits_minus_one = 7 → exp_bits = 8.
        let bytes = pack_lsb(&[(1, 1), (0, 2), (7, 4)]);
        let mut br = BitReader::new(&bytes);
        let bd = BitDepthFdis::read(&mut br).unwrap();
        assert!(bd.float_sample);
        assert_eq!(bd.bits_per_sample, 32);
        assert_eq!(bd.exp_bits, 8);
    }

    #[test]
    fn colour_encoding_all_default() {
        let bytes = pack_lsb(&[(1, 1)]);
        let mut br = BitReader::new(&bytes);
        let ce = ColourEncoding::read(&mut br).unwrap();
        assert!(ce.all_default);
        assert_eq!(ce.colour_space, ColourSpace::Rgb);
        assert_eq!(ce.white_point, WhitePoint::D65);
    }

    #[test]
    fn tone_mapping_all_default() {
        let bytes = pack_lsb(&[(1, 1)]);
        let mut br = BitReader::new(&bytes);
        let tm = ToneMapping::read(&mut br).unwrap();
        assert!(tm.all_default);
        assert_eq!(tm.intensity_target, 255.0);
    }

    #[test]
    fn extra_channel_info_all_default_is_alpha() {
        let bytes = pack_lsb(&[(1, 1)]);
        let mut br = BitReader::new(&bytes);
        let eci = ExtraChannelInfo::read(&mut br).unwrap();
        assert!(eci.all_default);
        assert_eq!(eci.kind, ExtraChannelType::Alpha);
    }

    #[test]
    fn animation_header_default_path() {
        // tps_num sel=0 → Val(100), tps_den sel=0 → Val(1),
        // num_loops sel=0 → Val(0), have_timecodes = 0.
        let bytes = pack_lsb(&[(0, 2), (0, 2), (0, 2), (0, 1)]);
        let mut br = BitReader::new(&bytes);
        let ah = AnimationHeader::read(&mut br).unwrap();
        assert_eq!(ah.tps_numerator, 100);
        assert_eq!(ah.tps_denominator, 1);
        assert_eq!(ah.num_loops, 0);
        assert!(!ah.have_timecodes);
    }

    #[test]
    fn customxy_round_trip() {
        // ux = 0 (sel=0), uy = 1 (sel=0, bits=1)
        // Distribution[0] is Bits(19); pick ux raw = 4 → unpacks to 2.
        // Pick uy raw = 5 → unpacks to -3.
        let bytes = pack_lsb(&[(0, 2), (4, 19), (0, 2), (5, 19)]);
        let mut br = BitReader::new(&bytes);
        let xy = Customxy::read(&mut br).unwrap();
        assert_eq!(xy.x, 2);
        assert_eq!(xy.y, -3);
    }

    #[test]
    fn opsin_inverse_matrix_all_default() {
        let bytes = pack_lsb(&[(1, 1)]);
        let mut br = BitReader::new(&bytes);
        let om = OpsinInverseMatrix::read(&mut br).unwrap();
        assert!(om.all_default);
        assert!((om.inv_mat[0][0] - 11.031_567).abs() < 0.01);
    }

    #[test]
    fn malicious_huge_extra_channels_rejected() {
        // Build:
        //   all_default = 0
        //   extra_fields = 0
        //   bit_depth: float_sample=0, bps sel=0 → 8
        //   modular_16bit = 1
        //   num_extra_channels U32 sel=3 → BitsOffset(12, 1) raw=4095 → 4096
        // After this we'd need 4096 ExtraChannelInfo decodes; with no
        // remaining input the bound check should fail.
        let bytes = pack_lsb(&[
            (0, 1),     // all_default
            (0, 1),     // extra_fields
            (0, 1),     // bit_depth float_sample
            (0, 2),     // bits sel=0 → 8
            (1, 1),     // modular_16bit
            (3, 2),     // num_extra_channels selector = 3
            (4095, 12), // raw=4095 → 4096
        ]);
        let mut br = BitReader::new(&bytes);
        let res = ImageMetadataFdis::read(&mut br);
        assert!(res.is_err(), "expected huge num_extra_channels rejection");
    }
}
