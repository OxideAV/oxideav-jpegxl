//! `HfCoefficientHistogramSize` — typed sizing primitive for the
//! §C.7.2 HF coefficient histogram block.
//!
//! ISO/IEC FDIS 18181-1:2021 §C.7.2 reads, in full:
//!
//! > Let `nb_block_ctx` be equal to `max(block_ctx_map)+1`. The
//! > decoder reads a histogram with `495 × num_hf_presets ×
//! > nb_block_ctx` clustered distributions D from the codestream as
//! > specified in D.3.
//!
//! §C.8.3 then re-uses the same constants for the per-pass routing
//! offset:
//!
//! > The decoder read `hfp = u(ceil(log2(num_hf_presets)))`, which
//! > indicates the coefficient order to be used for this group as
//! > well as the offset in the histogram, which is given by
//! > `offset = 495 × nb_block_ctx × hfp`.
//!
//! Round 238 lifts the `495u64 * num_hf_presets * nb_block_ctx`
//! arithmetic (which currently lives inline in [`crate::hf_pass`],
//! [`crate::pass_group_hf`], and [`crate::multi_pass_hf_header`])
//! into a typed primitive so the spec constant has one home, the
//! defensive zero-input guard runs once, and the per-pass offset
//! derivation shares the same `nb_block_ctx` factor.
//!
//! The primitive is sizing-only — it does NOT read any bits from a
//! [`BitReader`], does NOT materialise the histogram block itself
//! (the [`crate::modular_fdis::EntropyStream::read`] call against
//! `num_distributions()` clustered distributions remains a follow-up
//! step), and does NOT compute per-context offsets beyond the
//! per-pass `offset_for_hfp` (which §C.8.3 spells out explicitly).

use oxideav_core::{Error, Result};

/// The §C.7.2 spec constant — distributions per HF preset per block
/// context. The factor of 495 = 11 × 45 partitions the 64-coefficient
/// run/level alphabet across the §C.8.3 `NonZerosContext` +
/// `CoefficientContext` lookup table.
pub const PER_PRESET_PER_BLOCK_CTX: u64 = 495;

/// §C.7.2 sizing descriptor for the HF coefficient histogram block.
///
/// Constructed from the HfGlobal `num_hf_presets` (§I.2.6) plus
/// either:
///
/// * the LfGlobal HfBlockContext `nb_block_ctx` directly
///   (`max(block_ctx_map) + 1` already computed during
///   [`crate::lf_global::HfBlockContext`] decode), via [`Self::new`];
/// * the raw `block_ctx_map` slice, via [`Self::from_block_ctx_map`],
///   which re-derives `nb_block_ctx = max(map) + 1` per §C.7.2 line 1.
///
/// All arithmetic is u64 to avoid overflow at the upper bound. The
/// §C.7.2 read size grows as `495 × num_hf_presets × nb_block_ctx`:
/// with the spec-permitted maxima the product stays well under
/// `2^32`, but downstream multiplication by per-distribution sizes
/// stays in u64 for headroom.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HfCoefficientHistogramSize {
    /// `num_hf_presets` — §I.2.6 HfGlobal field, guaranteed ≥ 1.
    pub num_hf_presets: u32,
    /// `nb_block_ctx = max(block_ctx_map) + 1` — §I.2.2 HfBlockContext
    /// bundle, guaranteed ≥ 1.
    pub nb_block_ctx: u32,
}

impl HfCoefficientHistogramSize {
    /// Direct constructor from the two HfGlobal / HfBlockContext
    /// fields. Rejects zero inputs (HfGlobal §I.2.6 + HfBlockContext
    /// §I.2.2 both guarantee ≥ 1; this is a defensive guard against
    /// upstream constructor bugs).
    pub fn new(num_hf_presets: u32, nb_block_ctx: u32) -> Result<Self> {
        if num_hf_presets == 0 {
            return Err(Error::InvalidData(
                "JXL HfCoefficientHistogramSize: num_hf_presets must be ≥ 1 (HfGlobal §I.2.6 \
                 invariant)"
                    .into(),
            ));
        }
        if nb_block_ctx == 0 {
            return Err(Error::InvalidData(
                "JXL HfCoefficientHistogramSize: nb_block_ctx must be ≥ 1 (HfBlockContext §I.2.2 \
                 invariant — max(block_ctx_map) + 1)"
                    .into(),
            ));
        }
        Ok(Self {
            num_hf_presets,
            nb_block_ctx,
        })
    }

    /// Derive from a decoded `block_ctx_map` slice + the HfGlobal
    /// `num_hf_presets`, computing `nb_block_ctx = max(map) + 1` per
    /// §C.7.2 line 1. Rejects an empty `block_ctx_map`.
    pub fn from_block_ctx_map(block_ctx_map: &[u8], num_hf_presets: u32) -> Result<Self> {
        if block_ctx_map.is_empty() {
            return Err(Error::InvalidData(
                "JXL HfCoefficientHistogramSize: block_ctx_map is empty (HfBlockContext §I.2.2 \
                 guarantees ≥ 1 entry)"
                    .into(),
            ));
        }
        let max_ctx = *block_ctx_map.iter().max().expect("non-empty checked above");
        // `max + 1` cannot overflow u8 → u32 since max_ctx ≤ 255.
        let nb_block_ctx = max_ctx as u32 + 1;
        Self::new(num_hf_presets, nb_block_ctx)
    }

    /// `495 × nb_block_ctx` — distributions per single HF preset.
    /// This is the `histogram_offset` step the §C.8.3 routing uses
    /// when stepping from `hfp = h` to `hfp = h + 1`.
    pub fn per_preset(&self) -> u64 {
        PER_PRESET_PER_BLOCK_CTX * self.nb_block_ctx as u64
    }

    /// `495 × num_hf_presets × nb_block_ctx` — the total number of
    /// clustered distributions the §C.7.2 read consumes from the
    /// codestream.
    pub fn num_distributions(&self) -> u64 {
        self.per_preset() * self.num_hf_presets as u64
    }

    /// `495 × nb_block_ctx × hfp` — the §C.8.3 per-pass offset for a
    /// given `hfp`. Rejects `hfp >= num_hf_presets` with `InvalidData`
    /// (matches the existing [`crate::pass_group_hf::PassGroupHfHeader`]
    /// range check).
    pub fn offset_for_hfp(&self, hfp: u32) -> Result<u64> {
        if hfp >= self.num_hf_presets {
            return Err(Error::InvalidData(format!(
                "JXL HfCoefficientHistogramSize: hfp {hfp} ≥ num_hf_presets {}",
                self.num_hf_presets
            )));
        }
        Ok(self.per_preset() * hfp as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The §C.7.2 default-shape worked example: the default
    /// HfBlockContext (§I.2.2) ships a 39-entry `block_ctx_map`
    /// whose maximum value is 14, so `nb_block_ctx = 15`. With a
    /// single-preset HfGlobal (`num_hf_presets = 1`), §C.7.2 reads
    /// `495 × 1 × 15 = 7425` clustered distributions.
    #[test]
    fn r238_default_block_ctx_map_15_contexts_one_preset() {
        // Build the default 39-entry shape used by
        // `HfBlockContext::DEFAULT_BLOCK_CTX_MAP`: 14 is the highest
        // cluster index, appearing at the last few entries.
        let mut default_map = vec![0u8; 39];
        default_map[38] = 14;
        let size = HfCoefficientHistogramSize::from_block_ctx_map(&default_map, 1).unwrap();
        assert_eq!(size.num_hf_presets, 1);
        assert_eq!(size.nb_block_ctx, 15);
        assert_eq!(size.per_preset(), 7425);
        assert_eq!(size.num_distributions(), 7425);
        assert_eq!(size.offset_for_hfp(0).unwrap(), 0);
        // Single-preset → hfp = 1 is out of range.
        assert!(size.offset_for_hfp(1).is_err());
    }

    /// Multi-preset multi-context arithmetic worked example:
    /// `num_hf_presets = 4`, `nb_block_ctx = 15` → per-preset 7425,
    /// total 29 700, per-pass offsets stepping by 7425.
    #[test]
    fn r238_multi_preset_multi_context_arithmetic() {
        let size = HfCoefficientHistogramSize::new(4, 15).unwrap();
        assert_eq!(size.per_preset(), 7425);
        assert_eq!(size.num_distributions(), 29_700);
        assert_eq!(size.offset_for_hfp(0).unwrap(), 0);
        assert_eq!(size.offset_for_hfp(1).unwrap(), 7425);
        assert_eq!(size.offset_for_hfp(2).unwrap(), 14_850);
        assert_eq!(size.offset_for_hfp(3).unwrap(), 22_275);
        assert!(size.offset_for_hfp(4).is_err());
    }

    /// Defensive guards against zero / empty inputs.
    #[test]
    fn r238_rejects_zero_inputs() {
        assert!(HfCoefficientHistogramSize::new(0, 15).is_err());
        assert!(HfCoefficientHistogramSize::new(4, 0).is_err());
        assert!(HfCoefficientHistogramSize::from_block_ctx_map(&[], 1).is_err());
    }

    /// The `PER_PRESET_PER_BLOCK_CTX` spec constant matches the
    /// §C.7.2 literal.
    #[test]
    fn r238_per_preset_per_block_ctx_is_495() {
        assert_eq!(PER_PRESET_PER_BLOCK_CTX, 495);
    }

    /// `from_block_ctx_map` matches `new` when the caller has already
    /// computed `nb_block_ctx`. Round-trip on an arbitrary map.
    #[test]
    fn r238_from_block_ctx_map_matches_new() {
        let map = [0u8, 1, 2, 3, 4, 5, 6, 7, 5, 3, 1, 0]; // max = 7
        let derived = HfCoefficientHistogramSize::from_block_ctx_map(&map, 2).unwrap();
        let direct = HfCoefficientHistogramSize::new(2, 8).unwrap();
        assert_eq!(derived, direct);
        assert_eq!(derived.num_distributions(), 495 * 2 * 8);
    }
}
