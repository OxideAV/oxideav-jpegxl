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

use crate::bitreader::BitReader;
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

/// `LfGlobal` bundle — FDIS Table C.10 — Modular-encoding subset.
///
/// Reading the full bundle requires Patches / Splines / Quantizer /
/// HfBlockContext / LfChannelCorrelation, none of which are wired in
/// round 3. We *fail* if the frame header signals any of those features
/// instead of silently producing wrong output.
#[derive(Debug, Clone)]
pub struct LfGlobal {
    /// Always present (defaulted).
    pub lf_dequant: LfChannelDequantization,
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

        if fh.encoding == Encoding::VarDct {
            return Err(Error::Unsupported(
                "JXL LfGlobal: VarDCT path (Quantizer + HfBlockContext + CfL) not yet supported"
                    .into(),
            ));
        }

        // C.4.8 GlobalModular.
        let global_modular = GlobalModular::read(br, fh, metadata)?;

        Ok(Self {
            lf_dequant,
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
