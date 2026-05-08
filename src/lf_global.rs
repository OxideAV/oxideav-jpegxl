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

use crate::ans::cluster::{num_clusters, read_clustering};
use crate::bitreader::{unpack_signed, BitReader, U32Dist};
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

/// `HfBlockContext` bundle — ISO/IEC 18181-1:2024 §I.2.2 (was FDIS
/// Listing C.15). Describes the HF block-context model.
///
/// Two encodings:
///
/// 1. **Default**: `u(1) == 1` selects the 39-element fixed table from
///    the spec.
/// 2. **Custom**: `u(1) == 0` reads:
///    * `nb_lf_thr[i] = u(4)` for `i in 0..3`, then `nb_lf_thr[i]` thresholds
///      decoded as `UnpackSigned(ReadThreshold())` per channel.
///    * `nb_qf_thr = u(4)`, then `nb_qf_thr` qf thresholds, each
///      `1 + U32(u(2), 4+u(3), 12+u(5), 44+u(8))`.
///    * `bsize = 39 * (nb_qf_thr+1) * (nb_lf_thr[0]+1) * (nb_lf_thr[1]+1) * (nb_lf_thr[2]+1)`.
///    * `block_ctx_map = ReadBlockCtxMap()` which is the standard C.2.2
///      clustering with `num_dist = bsize` (skipping when `bsize == 1`
///      → `block_ctx_map = [0]`).
///    * Spec invariants: `bsize ≤ 39 * 64` and resulting
///      `num_clusters ≤ 16`.
#[derive(Debug, Clone)]
pub struct HfBlockContext {
    /// True when the default 39-element `block_ctx_map` was selected.
    pub used_default: bool,
    /// The block context map per §I.2.2. For the default branch this is
    /// the 39-element fixed table; for the custom branch it has `bsize`
    /// elements.
    pub block_ctx_map: Vec<u8>,
    /// `nb_block_ctx = max(block_ctx_map) + 1` per §C.8.3 / I.2.2.
    pub nb_block_ctx: u32,
    /// Per-channel LF thresholds `lf_thresholds[c]`. Empty for the
    /// default branch.
    pub lf_thresholds: [Vec<i32>; 3],
    /// QF thresholds. Empty for the default branch.
    pub qf_thresholds: Vec<u32>,
}

/// Read a `Threshold` per §I.2.2:
/// `U32(u(4), 16 + u(8), 272 + u(16), 65808 + u(32))`.
///
/// The 32-bit branch is read via `read_u32` only handles up to `u(32)`
/// returning a `u32`, but the spec's 4th selector reaches `65808 + u(32)`
/// which can overflow `u32`. We model the field as `u32` and saturate on
/// overflow with `Error::InvalidData` since a threshold value > u32::MAX
/// cannot match any real decoded HF value.
fn read_threshold(br: &mut BitReader<'_>) -> Result<u32> {
    let sel = br.read_bits(2)?;
    let v: u64 = match sel {
        0 => br.read_bits(4)? as u64,
        1 => 16u64 + br.read_bits(8)? as u64,
        2 => 272u64 + br.read_bits(16)? as u64,
        _ => {
            // Reading u(32) into a 64-bit accumulator. read_bits returns
            // u32; reading 32 bits in two halves keeps us inside the
            // BitReader API.
            let lo = br.read_bits(16)? as u64;
            let hi = br.read_bits(16)? as u64;
            65808u64 + ((hi << 16) | lo)
        }
    };
    if v > u32::MAX as u64 {
        return Err(Error::InvalidData(format!(
            "JXL HfBlockContext: ReadThreshold = {v} exceeds u32::MAX"
        )));
    }
    Ok(v as u32)
}

impl HfBlockContext {
    /// Default 39-element table per §I.2.2 first branch.
    pub const DEFAULT_BLOCK_CTX_MAP: [u8; 39] = [
        0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14, 7,
        8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,
    ];

    /// Spec invariant: `nb_lf_thr[i]` is read as `u(4)`, max 15. The
    /// custom-branch `bsize` cap from the spec ("num_dist ≤ 39 * 64")
    /// constrains the product `(nb_qf_thr+1) * Π (nb_lf_thr[i]+1)` to
    /// at most 64 — the largest legal product.
    const MAX_BSIZE: u32 = 39 * 64;
    /// Spec invariant: "the resulting num_clusters ≤ 16".
    const MAX_NUM_CLUSTERS: u32 = 16;

    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let used_default = br.read_bool()?;
        if used_default {
            let map = Self::DEFAULT_BLOCK_CTX_MAP.to_vec();
            let nb = (*map.iter().max().unwrap_or(&0) as u32) + 1;
            return Ok(Self {
                used_default: true,
                block_ctx_map: map,
                nb_block_ctx: nb,
                lf_thresholds: [Vec::new(), Vec::new(), Vec::new()],
                qf_thresholds: Vec::new(),
            });
        }

        // Custom branch — §I.2.2 second arm.
        let mut lf_thresholds: [Vec<i32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut nb_lf_thr = [0u32; 3];
        for i in 0..3 {
            let n = br.read_bits(4)?;
            nb_lf_thr[i] = n;
            let mut v = Vec::with_capacity(n as usize);
            for _ in 0..n {
                let raw = read_threshold(br)?;
                v.push(unpack_signed(raw));
            }
            lf_thresholds[i] = v;
        }
        let nb_qf_thr = br.read_bits(4)?;
        let mut qf_thresholds: Vec<u32> = Vec::with_capacity(nb_qf_thr as usize);
        for _ in 0..nb_qf_thr {
            let raw = br.read_u32([
                U32Dist::Bits(2),
                U32Dist::BitsOffset(3, 4),
                U32Dist::BitsOffset(5, 12),
                U32Dist::BitsOffset(8, 44),
            ])?;
            qf_thresholds.push(raw.checked_add(1).ok_or_else(|| {
                Error::InvalidData("JXL HfBlockContext: qf_threshold overflow".into())
            })?);
        }

        // bsize = 39 * (nb_qf_thr+1) * Π (nb_lf_thr[i]+1).
        let bsize_u64 = 39u64
            * (nb_qf_thr as u64 + 1)
            * (nb_lf_thr[0] as u64 + 1)
            * (nb_lf_thr[1] as u64 + 1)
            * (nb_lf_thr[2] as u64 + 1);
        if bsize_u64 > Self::MAX_BSIZE as u64 {
            return Err(Error::InvalidData(format!(
                "JXL HfBlockContext: bsize {bsize_u64} > 39*64={} (spec invariant)",
                Self::MAX_BSIZE
            )));
        }
        let bsize = bsize_u64 as u32;

        // ReadBlockCtxMap: standard C.2.2 clustering with num_dist = bsize.
        // When bsize == 1, the spec skips the procedure and returns
        // clusters = [0].
        let block_ctx_map_u32: Vec<u32> = if bsize <= 1 {
            vec![0u32; bsize as usize]
        } else {
            read_clustering(br, bsize as usize)?
        };
        let nb = num_clusters(&block_ctx_map_u32);
        if nb > Self::MAX_NUM_CLUSTERS {
            return Err(Error::InvalidData(format!(
                "JXL HfBlockContext: num_clusters {nb} > {} (spec invariant)",
                Self::MAX_NUM_CLUSTERS
            )));
        }
        // Cluster indices fit in u8 (max 16 < 256).
        let block_ctx_map: Vec<u8> = block_ctx_map_u32.iter().map(|&v| v as u8).collect();

        Ok(Self {
            used_default: false,
            block_ctx_map,
            nb_block_ctx: nb,
            lf_thresholds,
            qf_thresholds,
        })
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
    fn hf_block_context_custom_minimum_zero_thresholds() {
        // u(1) = 0 → custom branch.
        // nb_lf_thr[0] = 0 (4 bits), nb_lf_thr[1] = 0, nb_lf_thr[2] = 0,
        // nb_qf_thr = 0 (4 bits). bsize = 39 * 1 * 1 * 1 * 1 = 39.
        // ReadBlockCtxMap with num_dist = 39:
        //   is_simple = 1 (1 bit), nbits = 0 (2 bits) → 39 × u(0) = all 0
        //   → block_ctx_map = [0; 39], num_clusters = 1.
        let bytes = pack_lsb(&[
            (0, 1), // u(1) = 0 → custom
            (0, 4), // nb_lf_thr[0] = 0
            (0, 4), // nb_lf_thr[1] = 0
            (0, 4), // nb_lf_thr[2] = 0
            (0, 4), // nb_qf_thr = 0
            (1, 1), // is_simple = 1
            (0, 2), // nbits = 0 → all cluster indices read as u(0) = 0
        ]);
        let mut br = BitReader::new(&bytes);
        let hbc = HfBlockContext::read(&mut br).unwrap();
        assert!(!hbc.used_default);
        assert_eq!(hbc.block_ctx_map.len(), 39);
        assert_eq!(hbc.nb_block_ctx, 1);
        for &v in &hbc.block_ctx_map {
            assert_eq!(v, 0);
        }
        for ch in &hbc.lf_thresholds {
            assert!(ch.is_empty());
        }
        assert!(hbc.qf_thresholds.is_empty());
    }

    #[test]
    fn hf_block_context_custom_with_qf_threshold() {
        // u(1) = 0 → custom.
        // nb_lf_thr[0..3] = 0, nb_qf_thr = 1.
        // qf_threshold #0: U32 selector 0 → u(2) value 3 → +1 = 4.
        // bsize = 39 * 2 * 1 * 1 * 1 = 78. Clustering with num_dist=78:
        //   is_simple = 1, nbits = 0 → all 0. num_clusters = 1.
        let bytes = pack_lsb(&[
            (0, 1), // u(1) = 0
            (0, 4),
            (0, 4),
            (0, 4),
            (1, 4), // nb_qf_thr = 1
            (0, 2), // U32 sel = 0 (u(2))
            (3, 2), // u(2) = 3 → qf_threshold = 4
            (1, 1), // is_simple
            (0, 2), // nbits = 0
        ]);
        let mut br = BitReader::new(&bytes);
        let hbc = HfBlockContext::read(&mut br).unwrap();
        assert_eq!(hbc.qf_thresholds, vec![4]);
        assert_eq!(hbc.block_ctx_map.len(), 78);
        assert_eq!(hbc.nb_block_ctx, 1);
    }

    #[test]
    fn hf_block_context_custom_simple_clustering_bit_exact() {
        // Round-trip: build a 1+12+0+...+39*nbits bit-exact custom
        // HfBlockContext bitstream, decode it, then verify exactly the
        // expected number of bits were consumed and `block_ctx_map`
        // matches the encoder input.
        //
        // Format:
        // - u(1) = 0 (custom)
        // - nb_lf_thr[0..3] = 0 (4 bits each, 12 bits total)
        // - nb_qf_thr = 0 (4 bits)
        // - is_simple = 1 (1 bit)
        // - nbits = 2 (2 bits)
        // - 39 × u(2): the cluster indices.
        // Total: 1 + 12 + 4 + 1 + 2 + 39*2 = 98 bits.
        let map: [u32; 39] = [
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        ];
        let mut bits: Vec<(u32, u32)> = vec![
            (0, 1), // u(1) = 0 → custom
            (0, 4),
            (0, 4),
            (0, 4),
            (0, 4), // nb_qf_thr = 0
            (1, 1), // is_simple
            (2, 2), // nbits = 2
        ];
        for &v in &map {
            bits.push((v, 2));
        }
        let bytes = pack_lsb(&bits);
        let mut br = BitReader::new(&bytes);
        let bits_before = br.bits_read();
        let hbc = HfBlockContext::read(&mut br).unwrap();
        let bits_after = br.bits_read();
        let consumed = bits_after - bits_before;
        assert_eq!(consumed, 1 + 12 + 4 + 1 + 2 + 39 * 2);
        for (i, &v) in map.iter().enumerate() {
            assert_eq!(hbc.block_ctx_map[i] as u32, v, "cell {i}");
        }
        assert_eq!(hbc.nb_block_ctx, 3);
    }

    #[test]
    fn hf_block_context_custom_oversized_bsize_rejected() {
        // nb_lf_thr[0]=15, nb_lf_thr[1]=15, nb_lf_thr[2]=15, nb_qf_thr=15
        // → bsize = 39 * 16 * 16 * 16 * 16 = 39 * 65536 ≫ 39*64.
        // Must error before clustering is attempted.
        let bytes = pack_lsb(&[
            (0, 1),
            (15, 4),
            (15, 4),
            // Need 15 thresholds for channel 0 — give a bunch of zeros.
            (0, 2),
            (0, 4),
            (0, 2),
            (0, 4),
            (0, 2),
            (0, 4),
            (0, 2),
            (0, 4),
            (0, 2),
            (0, 4),
            (0, 2),
            (0, 4),
            (0, 2),
            (0, 4),
            (0, 2),
            (0, 4),
            (0, 2),
            (0, 4),
            (0, 2),
            (0, 4),
            (0, 2),
            (0, 4),
            (0, 2),
            (0, 4),
            (0, 2),
            (0, 4),
            (0, 2),
            (0, 4),
            (15, 4), // nb_lf_thr[1]=15
        ]);
        // Provide enough trailing zeros to satisfy further reads
        // (channel 1 needs 15 thresholds + channel 2 nb_lf_thr + thresholds
        // + nb_qf_thr + qf thresholds before bsize check, but with all
        // zeros early the bsize-product test rejects regardless of how
        // far the parser actually got — what matters is no panic).
        let mut padded = bytes.clone();
        padded.extend_from_slice(&[0u8; 256]);
        let mut br = BitReader::new(&padded);
        let r = HfBlockContext::read(&mut br);
        assert!(matches!(r, Err(Error::InvalidData(_))));
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
