//! `LfGlobal` bundle — FDIS 18181-1 §C.4 (Table C.10).
//!
//! For a Modular-only frame with no Patches/Splines/Noise flags set
//! (the common case for `cjxl --lossless` output of small images), the
//! bundle reduces to:
//!
//! * [`LfChannelDequantization`] — three F16 LF dequant weights
//!   (§C.4.2 Table C.11), and
//! * [`crate::global_modular::GlobalModular`] — wraps a Modular
//!   sub-bitstream (§C.4.8 + §C.9). Round 3 covers this minimum.
//!
//! Patches (§C.4.5), Splines (§C.4.6), NoiseParameters (§C.4.7),
//! Quantizer (§C.4.3), HF Block Context (§C.8.4), and
//! LfChannelCorrelation (§C.4.4) are deferred to round 4 — they are
//! only needed when `frame_header.flags` enables the corresponding
//! feature or when `encoding == kVarDCT`.
//!
//! Allocation bound: this bundle reads at most a handful of fixed-size
//! fields. The only variable allocation is in the embedded
//! `GlobalModular` (a Modular sub-bitstream — see that module for its
//! own bounds).

use oxideav_core::{Error, Result};

use crate::bitreader::{BitReader, U32Dist};
use crate::frame_header::{flags, Encoding, FrameHeader};
use crate::global_modular::GlobalModular;
use crate::metadata_fdis::ImageMetadataFdis;

/// `LfChannelDequantization` per FDIS Table C.11. Three F16 multipliers
/// (X, Y, B) used to dequantize LF coefficients in VarDCT mode. For
/// Modular-mode frames the values are still decoded but unused by the
/// pixel path; we keep them for forward compatibility.
#[derive(Debug, Clone, Copy)]
pub struct LfChannelDequantization {
    pub all_default: bool,
    pub m_x_lf_unscaled: f32,
    pub m_y_lf_unscaled: f32,
    pub m_b_lf_unscaled: f32,
}

impl Default for LfChannelDequantization {
    fn default() -> Self {
        Self {
            all_default: true,
            m_x_lf_unscaled: 4096.0,
            m_y_lf_unscaled: 512.0,
            m_b_lf_unscaled: 256.0,
        }
    }
}

impl LfChannelDequantization {
    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let all_default = br.read_bool()?;
        if all_default {
            return Ok(Self::default());
        }
        let m_x_lf_unscaled = br.read_f16()?;
        let m_y_lf_unscaled = br.read_f16()?;
        let m_b_lf_unscaled = br.read_f16()?;
        Ok(Self {
            all_default: false,
            m_x_lf_unscaled,
            m_y_lf_unscaled,
            m_b_lf_unscaled,
        })
    }
}

/// `Quantizer` bundle — FDIS Table C.12 (§C.4.3). Two integer fields
/// `global_scale` and `quant_lf` parameterise the per-channel LF
/// dequantisation: `mXDC = m_x_lf_unscaled / (global_scale × quant_lf)`,
/// and similarly for Y / B (Listing C.1).
///
/// Round-11 wiring: parsed when `encoding == kVarDCT` so the LfGlobal
/// bit-position advances past the bundle. The decoded values are stored
/// for later round-12 consumption (LF-dequant + IDCT). For Modular-mode
/// frames the bundle is not present (skipped per Table C.10's
/// `frame_header.encoding == kVarDCT` condition row).
#[derive(Debug, Clone, Copy)]
pub struct Quantizer {
    /// `global_scale = U32(BitsOffset(11,1), BitsOffset(11, 2049),
    /// BitsOffset(12, 4097), BitsOffset(16, 8193))`.
    pub global_scale: u32,
    /// `quant_lf = U32(Val(16), BitsOffset(5,1), BitsOffset(8,1),
    /// BitsOffset(16,1))`.
    pub quant_lf: u32,
}

impl Quantizer {
    /// Decode the Quantizer bundle (Table C.12). Both fields default to
    /// their non-zero defaults (no `all_default` short-circuit on this
    /// bundle — both `U32` distributions always read).
    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let global_scale = br.read_u32([
            U32Dist::BitsOffset(11, 1),
            U32Dist::BitsOffset(11, 2049),
            U32Dist::BitsOffset(12, 4097),
            U32Dist::BitsOffset(16, 8193),
        ])?;
        let quant_lf = br.read_u32([
            U32Dist::Val(16),
            U32Dist::BitsOffset(5, 1),
            U32Dist::BitsOffset(8, 1),
            U32Dist::BitsOffset(16, 1),
        ])?;
        Ok(Self {
            global_scale,
            quant_lf,
        })
    }
}

/// `LfChannelCorrelation` bundle — FDIS Table C.13 (§C.4.4). Drives the
/// chroma-from-luma reconstruction (Annex G) for VarDCT mode. Round-11
/// only consumes the bits — actual CfL application defers to round-12.
#[derive(Debug, Clone, Copy)]
pub struct LfChannelCorrelation {
    pub all_default: bool,
    /// `colour_factor` — denominator for kX / kB on HF coefficients
    /// (default 84).
    pub colour_factor: u32,
    /// `base_correlation_x` (default 0.0).
    pub base_correlation_x: f32,
    /// `base_correlation_b` (default 1.0).
    pub base_correlation_b: f32,
    /// `x_factor_lf` u(8) (default 128 — per spec the value used in CfL
    /// is `x_factor_lf - 127` so default delta is 1).
    pub x_factor_lf: u32,
    /// `b_factor_lf` u(8) (default 128).
    pub b_factor_lf: u32,
}

impl Default for LfChannelCorrelation {
    fn default() -> Self {
        Self {
            all_default: true,
            colour_factor: 84,
            base_correlation_x: 0.0,
            base_correlation_b: 1.0,
            x_factor_lf: 128,
            b_factor_lf: 128,
        }
    }
}

impl LfChannelCorrelation {
    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let all_default = br.read_bool()?;
        if all_default {
            return Ok(Self::default());
        }
        let colour_factor = br.read_u32([
            U32Dist::Val(84),
            U32Dist::Val(256),
            U32Dist::BitsOffset(8, 2),
            U32Dist::BitsOffset(16, 258),
        ])?;
        let base_correlation_x = br.read_f16()?;
        let base_correlation_b = br.read_f16()?;
        let x_factor_lf = br.read_bits(8)?;
        let b_factor_lf = br.read_bits(8)?;
        Ok(Self {
            all_default: false,
            colour_factor,
            base_correlation_x,
            base_correlation_b,
            x_factor_lf,
            b_factor_lf,
        })
    }
}

/// `HfBlockContext` bundle — FDIS Listing C.15 (§C.8.4). Describes the
/// HF block-context model. Round-11 only handles the `u(1) == 1`
/// default-table fast path — that consumes a single bit and selects the
/// 39-element default `block_ctx_map`. The non-default branch (per-LF
/// thresholds + qf thresholds + clustering map) returns
/// `Error::Unsupported` until full HF decode lands.
#[derive(Debug, Clone)]
pub struct HfBlockContext {
    /// True when the default 39-element `block_ctx_map` was selected.
    pub used_default: bool,
    /// The 39-element block context map per Listing C.15.
    pub block_ctx_map: Vec<u8>,
    /// `nb_block_ctx = max(block_ctx_map) + 1` per §C.8.3.
    pub nb_block_ctx: u32,
}

impl HfBlockContext {
    /// Default 39-element table per Listing C.15 first branch.
    pub const DEFAULT_BLOCK_CTX_MAP: [u8; 39] = [
        0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14, 7,
        8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,
    ];

    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let used_default = br.read_bool()?;
        if used_default {
            let map = Self::DEFAULT_BLOCK_CTX_MAP.to_vec();
            let nb = (*map.iter().max().unwrap_or(&0) as u32) + 1;
            return Ok(Self {
                used_default: true,
                block_ctx_map: map,
                nb_block_ctx: nb,
            });
        }
        Err(Error::Unsupported(
            "JXL LfGlobal: HfBlockContext non-default-table branch (per-LF thresholds + qf \
             thresholds + clustering map) not yet supported (round 12+)"
                .into(),
        ))
    }
}

/// `LfGlobal` bundle — FDIS Table C.10. Round-11 widens the round-3
/// Modular-only subset with the VarDCT VarDct-specific bundles
/// (`Quantizer`, `HfBlockContext`, `LfChannelCorrelation`) so that
/// the bit-position advances correctly past the LfGlobal slot and the
/// downstream LfGroup parser (round 11 too) can reach the LF
/// coefficients sub-bitstream.
///
/// Patches / Splines / NoiseParameters are still rejected with a precise
/// `Error::Unsupported` — round-12+ work.
#[derive(Debug, Clone)]
pub struct LfGlobal {
    /// Always present (defaulted).
    pub lf_dequant: LfChannelDequantization,
    /// Present when `encoding == kVarDCT`, else `None`.
    pub quantizer: Option<Quantizer>,
    /// Present when `encoding == kVarDCT`, else `None`.
    pub hf_block_context: Option<HfBlockContext>,
    /// Present when `encoding == kVarDCT`, else `None`.
    pub lf_channel_correlation: Option<LfChannelCorrelation>,
    /// Always present (the frame's GlobalModular sub-bitstream).
    pub global_modular: GlobalModular,
}

impl LfGlobal {
    /// Decode the LfGlobal bundle. Currently rejects:
    ///
    /// * any of `flags::PATCHES | SPLINES | NOISE` set,
    /// * `encoding == kVarDCT` (Quantizer / HfBlockContext / CfL all
    ///   read in that path).
    ///
    /// These limits will be relaxed in round 4 once Patches / Splines /
    /// VarDCT land. The router in `crate::lib::make_decoder` does not
    /// have to know about them — the FDIS bundle naturally short-circuits
    /// on any flag bit it can't yet parse.
    pub fn read(
        br: &mut BitReader<'_>,
        fh: &FrameHeader,
        metadata: &ImageMetadataFdis,
    ) -> Result<Self> {
        if (fh.flags & flags::PATCHES) != 0 {
            return Err(Error::Unsupported(
                "JXL LfGlobal: Patches not yet supported (round 4)".into(),
            ));
        }
        if (fh.flags & flags::SPLINES) != 0 {
            return Err(Error::Unsupported(
                "JXL LfGlobal: Splines not yet supported (round 4)".into(),
            ));
        }
        if (fh.flags & flags::NOISE) != 0 {
            return Err(Error::Unsupported(
                "JXL LfGlobal: NoiseParameters not yet supported (round 4)".into(),
            ));
        }

        let lf_dequant = LfChannelDequantization::read(br)?;

        // VarDCT bundles per Table C.10 condition rows: when
        // `encoding == kVarDCT` read Quantizer (C.4.3), HfBlockContext
        // (C.8.4), and LfChannelCorrelation (C.4.4) before
        // GlobalModular. Round-11 wires these so the bit-position
        // advances correctly into LfGroup territory.
        let (quantizer, hf_block_context, lf_channel_correlation) =
            if fh.encoding == Encoding::VarDct {
                let q = Quantizer::read(br)?;
                let hbc = HfBlockContext::read(br)?;
                let cfl = LfChannelCorrelation::read(br)?;
                (Some(q), Some(hbc), Some(cfl))
            } else {
                (None, None, None)
            };

        // C.4.8 GlobalModular.
        let global_modular = GlobalModular::read(br, fh, metadata)?;

        Ok(Self {
            lf_dequant,
            quantizer,
            hf_block_context,
            lf_channel_correlation,
            global_modular,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ans::test_helpers::pack_lsb;

    #[test]
    fn lf_dequant_defaults_match_spec() {
        let bytes = pack_lsb(&[(1, 1)]);
        let mut br = BitReader::new(&bytes);
        let d = LfChannelDequantization::read(&mut br).unwrap();
        assert!(d.all_default);
        assert_eq!(d.m_x_lf_unscaled, 4096.0);
        assert_eq!(d.m_y_lf_unscaled, 512.0);
        assert_eq!(d.m_b_lf_unscaled, 256.0);
    }

    #[test]
    fn quantizer_minimum_encoding() {
        // global_scale: selector 0 (BitsOffset(11, 1)), payload = 0
        //   → value = 1.
        // quant_lf: selector 0 (Val(16)) → no payload → value = 16.
        // Total: 2 bits + 11 bits + 2 bits = 15 bits.
        let bytes = pack_lsb(&[(0, 2), (0, 11), (0, 2)]);
        let mut br = BitReader::new(&bytes);
        let q = Quantizer::read(&mut br).unwrap();
        assert_eq!(q.global_scale, 1);
        assert_eq!(q.quant_lf, 16);
    }

    #[test]
    fn quantizer_typical_values() {
        // global_scale: selector 0 (BitsOffset(11, 1)), payload = 1535
        //   → value = 1536. quant_lf: selector 1 (BitsOffset(5, 1)),
        //   payload = 31 → value = 32.
        let bytes = pack_lsb(&[(0, 2), (1535, 11), (1, 2), (31, 5)]);
        let mut br = BitReader::new(&bytes);
        let q = Quantizer::read(&mut br).unwrap();
        assert_eq!(q.global_scale, 1536);
        assert_eq!(q.quant_lf, 32);
    }

    #[test]
    fn lf_channel_correlation_defaults() {
        // all_default = 1.
        let bytes = pack_lsb(&[(1, 1)]);
        let mut br = BitReader::new(&bytes);
        let cfl = LfChannelCorrelation::read(&mut br).unwrap();
        assert!(cfl.all_default);
        assert_eq!(cfl.colour_factor, 84);
        assert_eq!(cfl.base_correlation_x, 0.0);
        assert_eq!(cfl.base_correlation_b, 1.0);
        assert_eq!(cfl.x_factor_lf, 128);
        assert_eq!(cfl.b_factor_lf, 128);
    }

    #[test]
    fn hf_block_context_default_table() {
        // u(1) = 1 → default 39-element table; nb_block_ctx = 14 + 1 = 15.
        let bytes = pack_lsb(&[(1, 1)]);
        let mut br = BitReader::new(&bytes);
        let hbc = HfBlockContext::read(&mut br).unwrap();
        assert!(hbc.used_default);
        assert_eq!(hbc.block_ctx_map.len(), 39);
        assert_eq!(hbc.nb_block_ctx, 15);
        assert_eq!(hbc.block_ctx_map[0], 0);
        assert_eq!(hbc.block_ctx_map[7], 5);
        assert_eq!(hbc.block_ctx_map[38], 14);
    }

    #[test]
    fn hf_block_context_non_default_rejected_round_11() {
        // u(1) = 0 → non-default branch returns Unsupported.
        let bytes = pack_lsb(&[(0, 1)]);
        let mut br = BitReader::new(&bytes);
        let r = HfBlockContext::read(&mut br);
        assert!(matches!(r, Err(Error::Unsupported(_))));
    }

    #[test]
    fn lf_dequant_explicit_values() {
        // all_default = 0, then three F16 values: 1.0, -2.0, 0.5.
        // F16 1.0 = 0x3C00, -2.0 = 0xC000, 0.5 = 0x3800.
        let mut bw = Vec::new();
        // bit 0: all_default = 0, then 16 bits = 0x3C00, etc.
        // pack_lsb expects (value, n_bits) — but F16 is read via
        // read_bits(16), and we pack the byte LSB-first ordering; F16
        // bits 0..15 correspond to the 16 packed bits.
        bw.extend_from_slice(&pack_lsb(&[
            (0, 1),
            (0x3C00, 16),
            (0xC000, 16),
            (0x3800, 16),
        ]));
        let mut br = BitReader::new(&bw);
        let d = LfChannelDequantization::read(&mut br).unwrap();
        assert!(!d.all_default);
        assert_eq!(d.m_x_lf_unscaled, 1.0);
        assert_eq!(d.m_y_lf_unscaled, -2.0);
        assert_eq!(d.m_b_lf_unscaled, 0.5);
    }
}
