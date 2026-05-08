//! `HfGlobal` bundle — ISO/IEC 18181-1:2024 Annex C.6 (= 2021 FDIS C.6).
//!
//! ## Round 13 scope
//!
//! This module wires the **default-fast-path** of HfGlobal:
//!
//! 1. **C.6.2 Dequantization matrices** — first reads `u(1)`. When the
//!    bit is `1`, all 17 matrix slots take their default encoding from
//!    Table C.20 (C.6.3) — round 13 only handles this fast path. The
//!    non-default branch (11 sets of `encoding_mode = u(3)` + per-mode
//!    `ReadDctParams()` machinery from Listing C.7) returns
//!    `Error::Unsupported` until a future round wires the full table.
//!
//! 2. **C.6.4 Number of HF decoding presets** — reads
//!    `num_hf_presets_minus_1 = u(ceil(log2(num_groups)))`. The result is
//!    stored on [`HfGlobal::num_hf_presets`] for later C.7 / C.8.3
//!    consumption. The bit count is 0 for single-group frames (only
//!    legal value: 0 → `num_hf_presets = 1`).
//!
//! ## Why fast-path first
//!
//! The five small lossless fixtures exercising round 1..11 are all
//! `kModular` — they don't have an HfGlobal section to read (the slot is
//! 0-byte per F.3.1 round-9 fix). The minimal VarDCT fixture from round
//! 11 doesn't currently reach HfGlobal either since the test only
//! exercises LfGlobal + LfGroup. Round 13's HfGlobal lands as
//! infrastructure: the parser is unit-tested standalone so a future
//! round that wires multi-group VarDCT pixel decode can drive
//! [`HfGlobal::read`] directly.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// `HfGlobal` bundle — Table C.17. For a `kVarDCT` frame the bundle
/// contains:
///
/// * **Dequantization matrices** (C.6.2). Round 13 supports only the
///   default-encoding fast path (`u(1) == 1`) — the per-matrix mode +
///   parameters branch is round-14+ work.
/// * **Number of HF decoding presets** (C.6.4). Always read.
#[derive(Debug, Clone, Copy)]
pub struct HfGlobal {
    /// `true` when the codestream signaled the C.6.2 default-encoding
    /// fast path (`u(1) == 1`). Round-13 always sets this to `true`
    /// since the non-default branch returns `Error::Unsupported`.
    pub dequant_default: bool,
    /// `num_hf_presets` per C.6.4. The codestream encodes
    /// `num_hf_presets - 1` so this value is at least 1.
    pub num_hf_presets: u32,
}

impl HfGlobal {
    /// Decode the HfGlobal bundle (Table C.17). The caller has positioned
    /// `br` at the start of the HfGlobal TOC slot AND verified that
    /// `frame_header.encoding == kVarDCT` (the bundle is empty for
    /// `kModular`).
    ///
    /// `num_groups` parameterises the bit-count of `num_hf_presets - 1`
    /// per C.6.4: `u(ceil(log2(num_groups)))`. For single-group frames
    /// `num_groups == 1` and the field uses 0 bits (legal value: 0 →
    /// `num_hf_presets = 1`).
    pub fn read(br: &mut BitReader<'_>, num_groups: u64) -> Result<Self> {
        // C.6.2 first sentence: read u(1). When 1, all dequant matrices
        // take their default encoding from C.6.3.
        let dequant_default = br.read_bool()?;
        if !dequant_default {
            return Err(Error::Unsupported(
                "JXL HfGlobal: per-matrix dequantization parameters (C.6.2 non-default-encoding \
                 branch) not yet supported (round 14+) — only the u(1)=1 default-fast-path is \
                 wired in round 13"
                    .into(),
            ));
        }

        // C.6.4: num_hf_presets_minus_1 = u(ceil(log2(num_groups))).
        // For num_groups == 0 the spec implicitly forbids the case (a
        // VarDCT frame must have at least one group); be defensive.
        if num_groups == 0 {
            return Err(Error::InvalidData(
                "JXL HfGlobal: num_groups = 0 (a VarDCT frame must have at least one group)".into(),
            ));
        }
        let nbits = ceil_log2_u64(num_groups);
        let num_hf_presets_minus_1 = if nbits == 0 { 0 } else { br.read_bits(nbits)? };
        // num_hf_presets = num_hf_presets_minus_1 + 1.
        let num_hf_presets = num_hf_presets_minus_1
            .checked_add(1)
            .ok_or_else(|| Error::InvalidData("JXL HfGlobal: num_hf_presets overflow".into()))?;
        if (num_hf_presets as u64) > num_groups {
            return Err(Error::InvalidData(format!(
                "JXL HfGlobal: num_hf_presets {num_hf_presets} exceeds num_groups {num_groups}"
            )));
        }
        Ok(Self {
            dequant_default,
            num_hf_presets,
        })
    }
}

/// `ceil(log2(n))` for `n >= 1`. `0` when `n == 1`.
fn ceil_log2_u64(n: u64) -> u32 {
    if n <= 1 {
        return 0;
    }
    64 - (n - 1).leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ans::test_helpers::pack_lsb;

    #[test]
    fn hf_global_default_fast_path_one_group() {
        // u(1) = 1 (default), num_groups = 1 → no bits for
        // num_hf_presets_minus_1, value = 0 → num_hf_presets = 1.
        let bytes = pack_lsb(&[(1, 1)]);
        let mut br = BitReader::new(&bytes);
        let hf = HfGlobal::read(&mut br, 1).unwrap();
        assert!(hf.dequant_default);
        assert_eq!(hf.num_hf_presets, 1);
    }

    #[test]
    fn hf_global_default_fast_path_four_groups() {
        // u(1) = 1, num_groups = 4 → ceil(log2(4)) = 2 bits for
        // num_hf_presets_minus_1. Encode value 2 → num_hf_presets = 3.
        let bytes = pack_lsb(&[(1, 1), (2, 2)]);
        let mut br = BitReader::new(&bytes);
        let hf = HfGlobal::read(&mut br, 4).unwrap();
        assert!(hf.dequant_default);
        assert_eq!(hf.num_hf_presets, 3);
    }

    #[test]
    fn hf_global_default_fast_path_three_groups() {
        // num_groups = 3 → ceil(log2(3)) = 2 bits. Encode value 0 →
        // num_hf_presets = 1.
        let bytes = pack_lsb(&[(1, 1), (0, 2)]);
        let mut br = BitReader::new(&bytes);
        let hf = HfGlobal::read(&mut br, 3).unwrap();
        assert_eq!(hf.num_hf_presets, 1);
    }

    #[test]
    fn hf_global_non_default_returns_unsupported() {
        // u(1) = 0 → non-default-encoding path → Unsupported.
        let bytes = pack_lsb(&[(0, 1)]);
        let mut br = BitReader::new(&bytes);
        let r = HfGlobal::read(&mut br, 1);
        assert!(matches!(r, Err(Error::Unsupported(_))));
    }

    #[test]
    fn hf_global_num_groups_zero_rejected() {
        let bytes = pack_lsb(&[(1, 1)]);
        let mut br = BitReader::new(&bytes);
        let r = HfGlobal::read(&mut br, 0);
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn hf_global_num_hf_presets_exceeds_num_groups_rejected() {
        // num_groups = 2 → ceil(log2(2)) = 1 bit. Encode value 1 →
        // num_hf_presets = 2 (= num_groups, OK). With value 0 → 1, OK.
        // Force exceed by num_groups = 2 + value 3 (impossible in 1 bit
        // — instead use num_groups=3, val=3 → num_hf_presets=4 > 3).
        let bytes = pack_lsb(&[(1, 1), (3, 2)]);
        let mut br = BitReader::new(&bytes);
        let r = HfGlobal::read(&mut br, 3);
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn ceil_log2_edges() {
        assert_eq!(ceil_log2_u64(0), 0);
        assert_eq!(ceil_log2_u64(1), 0);
        assert_eq!(ceil_log2_u64(2), 1);
        assert_eq!(ceil_log2_u64(3), 2);
        assert_eq!(ceil_log2_u64(4), 2);
        assert_eq!(ceil_log2_u64(8), 3);
        assert_eq!(ceil_log2_u64(9), 4);
    }
}
