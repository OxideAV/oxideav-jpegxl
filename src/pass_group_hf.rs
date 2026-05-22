//! `PassGroup` HF coefficients — ISO/IEC FDIS 18181-1:2021 §C.8.3 +
//! §C.8.4 supporting machinery.
//!
//! ## Scope (round 90)
//!
//! Round 90 lands the **structural** entry-points to the per-group
//! HF coefficient decode:
//!
//! * [`PassGroupHfHeader::read`] — the §C.8.3 first line
//!   `hfp = u(ceil(log2(num_hf_presets)))`. Selects (a) the
//!   coefficient order from the per-pass `HfPass` array and (b) the
//!   histogram offset `495 × nb_block_ctx × hfp`.
//! * [`block_context`], [`non_zeros_context`],
//!   [`coefficient_context`], [`predicted_non_zeros`] — direct
//!   transcriptions of Listing C.13's helper functions, plus the
//!   PredictedNonZeros per-position recurrence specified in the
//!   prose right after Listing C.13.
//!
//! The actual per-block ANS coefficient decode loop is deferred —
//! it requires the shared per-pass ANS stream that §C.7.2's
//! `495 × num_hf_presets × nb_block_ctx` histograms feed. That land
//! happens in a round after round 91 (which wires the §C.7.2
//! histograms + the `used_orders != 0` permutation reads).
//!
//! ## Listing C.13 — HF context (verbatim from FDIS p. 55)
//!
//! ```text
//! BlockContext() {
//!   idx = (c < 2 ? c ^ 1 : 2) × 13 + s;
//!   idx ×= (qf_thresholds.size() + 1);
//!   for (t : qf_thresholds) if (qf > t) idx++;
//!   for (i = 0; i < 3; i++) idx ×= (lf_thresholds[i].size() + 1);
//!   lf_idx = 0;
//!   for (t : lf_thresholds[0]) if (qdc[0] > t) lf_idx++;
//!   lf_idx ×= (lf_thresholds[2].size() + 1);
//!   for (t : lf_thresholds[2]) if (qdc[2] > t) lf_idx++;
//!   lf_idx ×= (lf_thresholds[0].size() + 1);
//!   for (t : lf_thresholds[1]) if (qdc[1] > t) lf_idx++;
//!   return block_ctx_map[idx + lf_idx];
//! }
//! NonZerosContext(predicted) {
//!   if (predicted > 64) predicted = 64;
//!   if (predicted < 8) return BlockContext() + nb_block_ctx × predicted;
//!   return BlockContext() + nb_block_ctx × (4 + predicted Idiv 2);
//! }
//! CoefficientContext(k, non_zeros, num_blocks, size, prev) {
//!   non_zeros = (non_zeros + num_blocks - 1) Idiv num_blocks;
//!   k = k Idiv num_blocks;
//!   return (CoeffNumNonzeroContext[non_zeros] + CoeffFreqContext[k]) × 2 +
//!     prev + BlockContext() × 458 + 37 × nb_block_ctx;
//! }
//! ```
//!
//! With the per-position predictor (PredictedNonZeros) defined right
//! after Listing C.13:
//!
//! * `(x, y) == (0, 0)` → 32
//! * `x == 0 && y != 0` → `NonZeros(x, y - 1)`
//! * `x != 0 && y == 0` → `NonZeros(x - 1, y)`
//! * else → `(NonZeros(x, y - 1) + NonZeros(x - 1, y) + 1) >> 1`
//!
//! And the spec's listings for the two 64-element ladder tables
//! (`CoeffFreqContext`, `CoeffNumNonzeroContext`) reproduced
//! verbatim below.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::hf_pass::HfPass;

/// `CoeffFreqContext[k]` — Listing C.13 prelude. Indexed by the
/// position-within-natural-order, divided by `num_blocks` per the
/// `k Idiv num_blocks` line in `CoefficientContext`.
pub const COEFF_FREQ_CONTEXT: [u32; 64] = [
    0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19,
    20, 20, 21, 21, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27,
    27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
];

/// `CoeffNumNonzeroContext[k]` — Listing C.13 prelude. Indexed by
/// the *remaining* `non_zeros` count.
pub const COEFF_NUM_NONZERO_CONTEXT: [u32; 64] = [
    0, 0, 31, 62, 62, 93, 93, 93, 93, 123, 123, 123, 123, 152, 152, 152, 152, 152, 152, 152, 152,
    152, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206,
];

/// Per-group HF coefficients header — the round-90 typed surface for
/// §C.8.3's `hfp` selector + the histogram-offset derivation.
#[derive(Debug, Clone, Copy)]
pub struct PassGroupHfHeader {
    /// `hfp = u(ceil(log2(num_hf_presets)))`. Selects which HfPass
    /// preset's coefficient orders apply to this group. Must satisfy
    /// `hfp < num_hf_presets` (the bit-width caps it naturally when
    /// `num_hf_presets` is a power of two; the spec reads it as a
    /// bare `u(nbits)` so a non-power-of-two `num_hf_presets` could
    /// allow an out-of-range `hfp` that we reject defensively).
    pub hfp: u32,
    /// `offset = 495 × nb_block_ctx × hfp`. The starting index into
    /// the per-pass histogram array for this group's ANS reads.
    pub histogram_offset: u64,
}

impl PassGroupHfHeader {
    /// Parse `hfp` per §C.8.3 first line.
    ///
    /// `nb_block_ctx` is inherited from the LfGlobal HfBlockContext
    /// bundle (§I.2.2) and used to compute `histogram_offset`.
    pub fn read(br: &mut BitReader<'_>, num_hf_presets: u32, nb_block_ctx: u32) -> Result<Self> {
        if num_hf_presets == 0 {
            return Err(Error::InvalidData(
                "JXL PassGroup HF: num_hf_presets must be ≥ 1 (HfGlobal §I.2.6 invariant)".into(),
            ));
        }
        let nbits = ceil_log2_u32(num_hf_presets);
        let hfp = if nbits == 0 { 0 } else { br.read_bits(nbits)? };
        if hfp >= num_hf_presets {
            return Err(Error::InvalidData(format!(
                "JXL PassGroup HF: hfp {hfp} ≥ num_hf_presets {num_hf_presets}"
            )));
        }
        let histogram_offset = 495u64 * nb_block_ctx as u64 * hfp as u64;
        Ok(Self {
            hfp,
            histogram_offset,
        })
    }

    /// Look up the coefficient orders for this group's chosen pass
    /// preset. The caller passes the per-pass [`HfPass`] vector that
    /// [`crate::hf_pass::read_hf_pass_sequence`] returned for the
    /// current pass.
    pub fn select_pass<'a>(&self, passes: &'a [HfPass]) -> Result<&'a HfPass> {
        passes.get(self.hfp as usize).ok_or_else(|| {
            Error::InvalidData(format!(
                "JXL PassGroup HF: hfp {} out of HfPass array length {}",
                self.hfp,
                passes.len()
            ))
        })
    }
}

/// `block_context()` per Listing C.13 — verbatim translation.
///
/// `c` is the channel (0=X, 1=Y, 2=B); `s` is the Order ID (Table
/// I.1) for the current varblock's DctSelect; `qf` is the HfMul of
/// the current varblock; `qdc[3]` are the quantised LF values of the
/// top-left 8×8 block within the current varblock (with chroma
/// subsampling factored in if needed).
///
/// `qf_thresholds`, `lf_thresholds`, and `block_ctx_map` are from
/// the LfGlobal HfBlockContext bundle (§I.2.2).
#[allow(clippy::too_many_arguments)]
pub fn block_context(
    c: u32,
    s: u32,
    qf: i32,
    qdc: [i32; 3],
    qf_thresholds: &[u32],
    lf_thresholds: &[Vec<i32>; 3],
    block_ctx_map: &[u8],
) -> Result<u32> {
    // idx = (c < 2 ? c ^ 1 : 2) × 13 + s
    let c_term = if c < 2 { c ^ 1 } else { 2 };
    let mut idx: u64 = (c_term as u64) * 13 + (s as u64);
    // idx ×= qf_thresholds.size() + 1
    idx *= (qf_thresholds.len() as u64) + 1;
    // for (t : qf_thresholds) if (qf > t) idx++
    for &t in qf_thresholds {
        if qf > t as i32 {
            idx += 1;
        }
    }
    // for (i = 0; i < 3; i++) idx ×= lf_thresholds[i].size() + 1
    for thr in lf_thresholds {
        idx *= (thr.len() as u64) + 1;
    }
    // lf_idx = 0
    let mut lf_idx: u64 = 0;
    // for (t : lf_thresholds[0]) if (qdc[0] > t) lf_idx++
    for &t in &lf_thresholds[0] {
        if qdc[0] > t {
            lf_idx += 1;
        }
    }
    // lf_idx ×= lf_thresholds[2].size() + 1
    lf_idx *= (lf_thresholds[2].len() as u64) + 1;
    for &t in &lf_thresholds[2] {
        if qdc[2] > t {
            lf_idx += 1;
        }
    }
    // lf_idx ×= lf_thresholds[0].size() + 1
    lf_idx *= (lf_thresholds[0].len() as u64) + 1;
    for &t in &lf_thresholds[1] {
        if qdc[1] > t {
            lf_idx += 1;
        }
    }
    let total = idx + lf_idx;
    if total >= block_ctx_map.len() as u64 {
        return Err(Error::InvalidData(format!(
            "JXL PassGroup HF: block_context idx {total} ≥ block_ctx_map len {}",
            block_ctx_map.len()
        )));
    }
    Ok(block_ctx_map[total as usize] as u32)
}

/// `NonZerosContext(predicted)` per Listing C.13.
///
/// `block_ctx` is the already-computed [`block_context`] value;
/// `nb_block_ctx` is the LfGlobal HfBlockContext invariant.
pub fn non_zeros_context(predicted: u32, block_ctx: u32, nb_block_ctx: u32) -> u32 {
    let mut pred = predicted;
    if pred > 64 {
        pred = 64;
    }
    if pred < 8 {
        return block_ctx + nb_block_ctx * pred;
    }
    block_ctx + nb_block_ctx * (4 + pred / 2)
}

/// `CoefficientContext(k, non_zeros, num_blocks, size, prev)` per
/// Listing C.13.
///
/// The `size` parameter of the spec listing is unused in the
/// listing body, but kept in the signature so callers thread the
/// spec-named arguments unchanged.
pub fn coefficient_context(
    k: u32,
    non_zeros: u32,
    num_blocks: u32,
    _size: u32,
    prev: u32,
    block_ctx: u32,
    nb_block_ctx: u32,
) -> Result<u32> {
    if num_blocks == 0 {
        return Err(Error::InvalidData(
            "JXL PassGroup HF: coefficient_context num_blocks must be ≥ 1".into(),
        ));
    }
    // non_zeros = (non_zeros + num_blocks - 1) Idiv num_blocks
    let nz = non_zeros.div_ceil(num_blocks);
    let nz = nz.min(63); // index COEFF_NUM_NONZERO_CONTEXT safely
                         // k = k Idiv num_blocks
    let k_div = (k / num_blocks).min(63); // index COEFF_FREQ_CONTEXT safely
                                          // (CoeffNumNonzeroContext[non_zeros] + CoeffFreqContext[k]) × 2 + prev
                                          //   + BlockContext() × 458 + 37 × nb_block_ctx
    let base = (COEFF_NUM_NONZERO_CONTEXT[nz as usize] + COEFF_FREQ_CONTEXT[k_div as usize]) * 2;
    let total: u64 =
        (base as u64) + (prev as u64) + (block_ctx as u64) * 458 + 37 * nb_block_ctx as u64;
    Ok(total as u32)
}

/// `PredictedNonZeros(x, y)` per the spec prose after Listing C.13.
///
/// `non_zeros_grid(x, y)` returns the `NonZeros(x, y)` field of the
/// already-decoded block at position `(x, y)`; for the recurrence
/// to be well-defined the caller must seed it with `NonZeros` values
/// as each block of the raster-order walk completes.
pub fn predicted_non_zeros<F>(x: u32, y: u32, non_zeros_at: F) -> u32
where
    F: Fn(u32, u32) -> u32,
{
    if x == 0 && y == 0 {
        return 32;
    }
    if x == 0 {
        return non_zeros_at(x, y - 1);
    }
    if y == 0 {
        return non_zeros_at(x - 1, y);
    }
    (non_zeros_at(x, y - 1) + non_zeros_at(x - 1, y) + 1) >> 1
}

fn ceil_log2_u32(n: u32) -> u32 {
    if n <= 1 {
        return 0;
    }
    32 - (n - 1).leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ans::test_helpers::pack_lsb;

    #[test]
    fn ceil_log2_u32_edges() {
        assert_eq!(ceil_log2_u32(0), 0);
        assert_eq!(ceil_log2_u32(1), 0);
        assert_eq!(ceil_log2_u32(2), 1);
        assert_eq!(ceil_log2_u32(3), 2);
        assert_eq!(ceil_log2_u32(4), 2);
        assert_eq!(ceil_log2_u32(8), 3);
        assert_eq!(ceil_log2_u32(9), 4);
    }

    #[test]
    fn pass_group_hf_header_single_preset() {
        // num_hf_presets = 1 → 0 bits for hfp → hfp = 0.
        let bytes = pack_lsb(&[(0, 0)]);
        let mut br = BitReader::new(&bytes);
        let h = PassGroupHfHeader::read(&mut br, 1, 15).unwrap();
        assert_eq!(h.hfp, 0);
        assert_eq!(h.histogram_offset, 0);
    }

    #[test]
    fn pass_group_hf_header_two_presets_hfp_1() {
        // num_hf_presets = 2 → 1 bit for hfp.
        let bytes = pack_lsb(&[(1, 1)]);
        let mut br = BitReader::new(&bytes);
        let h = PassGroupHfHeader::read(&mut br, 2, 15).unwrap();
        assert_eq!(h.hfp, 1);
        // 495 × 15 × 1 = 7425
        assert_eq!(h.histogram_offset, 7425);
    }

    #[test]
    fn pass_group_hf_header_rejects_oob_hfp() {
        // num_hf_presets = 3 → ceil_log2 = 2 bits. A value of 3 in
        // 2 bits is legal at the bit-reader level but >= num_hf_presets
        // → reject.
        let bytes = pack_lsb(&[(3, 2)]);
        let mut br = BitReader::new(&bytes);
        let r = PassGroupHfHeader::read(&mut br, 3, 1);
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn pass_group_hf_header_zero_presets_rejected() {
        let bytes = pack_lsb(&[(0, 0)]);
        let mut br = BitReader::new(&bytes);
        let r = PassGroupHfHeader::read(&mut br, 0, 1);
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn block_context_default_table_no_thresholds() {
        // With the default 39-element block_ctx_map and no thresholds
        // (qf/lf_thresholds all empty) the BlockContext formula collapses
        // to:
        //   idx = (c < 2 ? c ^ 1 : 2) × 13 + s
        // For (c=0, s=0): idx = 1 × 13 + 0 = 13.
        // block_ctx_map[13] = 7 (per the DEFAULT_BLOCK_CTX_MAP at index 13).
        let map = crate::lf_global::HfBlockContext::DEFAULT_BLOCK_CTX_MAP;
        let r = block_context(
            0,
            0,
            0,
            [0, 0, 0],
            &[],
            &[Vec::new(), Vec::new(), Vec::new()],
            &map,
        )
        .unwrap();
        assert_eq!(r, map[13] as u32);
        // (c=1, s=0): c ^ 1 = 0 → idx = 0. block_ctx_map[0] = 0.
        let r = block_context(
            1,
            0,
            0,
            [0, 0, 0],
            &[],
            &[Vec::new(), Vec::new(), Vec::new()],
            &map,
        )
        .unwrap();
        assert_eq!(r, map[0] as u32);
        // (c=2, s=0): c >= 2 → 2 × 13 + 0 = 26. map[26] = 7.
        let r = block_context(
            2,
            0,
            0,
            [0, 0, 0],
            &[],
            &[Vec::new(), Vec::new(), Vec::new()],
            &map,
        )
        .unwrap();
        assert_eq!(r, map[26] as u32);
    }

    #[test]
    fn block_context_invalid_idx_rejected() {
        // Single-element map → idx >= 1 must fail.
        let map = [0u8; 1];
        let r = block_context(
            0,
            0,
            0,
            [0, 0, 0],
            &[],
            &[Vec::new(), Vec::new(), Vec::new()],
            &map,
        );
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn non_zeros_context_branches() {
        // predicted < 8 → return block_ctx + nb × predicted.
        assert_eq!(non_zeros_context(0, 5, 15), 5);
        assert_eq!(non_zeros_context(7, 5, 15), 5 + 15 * 7);
        // predicted >= 8 → return block_ctx + nb × (4 + predicted / 2).
        // predicted = 8 → 4 + 4 = 8 → 5 + 15 * 8 = 125.
        assert_eq!(non_zeros_context(8, 5, 15), 5 + 15 * 8);
        // predicted = 64 → 4 + 32 = 36 → 5 + 15 * 36 = 545.
        assert_eq!(non_zeros_context(64, 5, 15), 5 + 15 * 36);
        // predicted = 100 → clamped to 64 → same as above.
        assert_eq!(non_zeros_context(100, 5, 15), 5 + 15 * 36);
    }

    #[test]
    fn coefficient_context_basic() {
        // num_blocks = 1, non_zeros = 0, k = 0, prev = 0,
        // block_ctx = 0, nb_block_ctx = 1.
        //   nz = (0 + 0) / 1 = 0 → COEFF_NUM_NONZERO_CONTEXT[0] = 0
        //   k_div = 0 → COEFF_FREQ_CONTEXT[0] = 0
        //   base = (0 + 0) × 2 = 0
        //   total = 0 + 0 + 0 × 458 + 37 × 1 = 37
        let r = coefficient_context(0, 0, 1, 64, 0, 0, 1).unwrap();
        assert_eq!(r, 37);
        // non_zeros = 32, k = 5, num_blocks = 1, prev = 1, block_ctx = 3,
        // nb_block_ctx = 15.
        //   nz = 32 → COEFF_NUM_NONZERO_CONTEXT[32] = 180
        //   k_div = 5 → COEFF_FREQ_CONTEXT[5] = 4
        //   base = (180 + 4) × 2 = 368
        //   total = 368 + 1 + 3 × 458 + 37 × 15 = 368 + 1 + 1374 + 555 = 2298
        let r = coefficient_context(5, 32, 1, 64, 1, 3, 15).unwrap();
        assert_eq!(r, 2298);
    }

    #[test]
    fn coefficient_context_rejects_zero_num_blocks() {
        let r = coefficient_context(0, 0, 0, 64, 0, 0, 1);
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn predicted_non_zeros_origin_is_32() {
        assert_eq!(predicted_non_zeros(0, 0, |_, _| 99), 32);
    }

    #[test]
    fn predicted_non_zeros_first_row() {
        // x != 0 && y == 0 → NonZeros(x - 1, 0). With a constant
        // non_zeros_at = 7, predicted = 7.
        assert_eq!(predicted_non_zeros(1, 0, |_, _| 7), 7);
        assert_eq!(predicted_non_zeros(5, 0, |_, _| 7), 7);
    }

    #[test]
    fn predicted_non_zeros_first_column() {
        // x == 0 && y != 0 → NonZeros(0, y - 1).
        assert_eq!(predicted_non_zeros(0, 1, |_, _| 12), 12);
    }

    #[test]
    fn predicted_non_zeros_interior_averages_two_neighbours() {
        // (1, 1): (NonZeros(1, 0) + NonZeros(0, 1) + 1) >> 1.
        // Return 4 for (1, 0) and 6 for (0, 1) → (4 + 6 + 1) >> 1 = 5.
        let nz = |x: u32, y: u32| {
            if x == 1 && y == 0 {
                4
            } else if x == 0 && y == 1 {
                6
            } else {
                0
            }
        };
        assert_eq!(predicted_non_zeros(1, 1, nz), 5);
    }

    #[test]
    fn pass_group_hf_header_select_pass_works() {
        // Two passes; hfp = 1 must pick passes[1].
        let bytes = pack_lsb(&[(1, 1)]);
        let mut br = BitReader::new(&bytes);
        let h = PassGroupHfHeader::read(&mut br, 2, 1).unwrap();
        // Build dummy passes by reading two used_orders=0 HfPass bundles.
        let bytes2 = pack_lsb(&[(2, 2), (2, 2)]);
        let mut br2 = BitReader::new(&bytes2);
        let passes = crate::hf_pass::read_hf_pass_sequence(&mut br2, 2, 1).unwrap();
        let chosen = h.select_pass(&passes).unwrap();
        // Either preset's `used_orders` is 0 → same num_histogram_distributions.
        assert_eq!(chosen.used_orders, 0);
    }

    #[test]
    fn pass_group_hf_header_select_pass_out_of_range() {
        // Forge an hfp out of range by hand (the read path can't
        // produce it, but the helper still needs to defend).
        let h = PassGroupHfHeader {
            hfp: 5,
            histogram_offset: 0,
        };
        let bytes = pack_lsb(&[(2, 2)]);
        let mut br = BitReader::new(&bytes);
        let passes = crate::hf_pass::read_hf_pass_sequence(&mut br, 1, 1).unwrap();
        assert!(h.select_pass(&passes).is_err());
    }

    #[test]
    fn coeff_freq_context_well_formed() {
        // Listing C.13 prelude — spot check key positions.
        assert_eq!(COEFF_FREQ_CONTEXT[0], 0);
        assert_eq!(COEFF_FREQ_CONTEXT[1], 0);
        assert_eq!(COEFF_FREQ_CONTEXT[2], 1);
        assert_eq!(COEFF_FREQ_CONTEXT[15], 14);
        assert_eq!(COEFF_FREQ_CONTEXT[16], 15);
        assert_eq!(COEFF_FREQ_CONTEXT[63], 30);
    }

    #[test]
    fn coeff_num_nonzero_context_well_formed() {
        assert_eq!(COEFF_NUM_NONZERO_CONTEXT[0], 0);
        assert_eq!(COEFF_NUM_NONZERO_CONTEXT[1], 0);
        assert_eq!(COEFF_NUM_NONZERO_CONTEXT[2], 31);
        assert_eq!(COEFF_NUM_NONZERO_CONTEXT[3], 62);
        assert_eq!(COEFF_NUM_NONZERO_CONTEXT[5], 93);
        assert_eq!(COEFF_NUM_NONZERO_CONTEXT[9], 123);
        assert_eq!(COEFF_NUM_NONZERO_CONTEXT[13], 152);
        assert_eq!(COEFF_NUM_NONZERO_CONTEXT[22], 180);
        assert_eq!(COEFF_NUM_NONZERO_CONTEXT[33], 206);
        assert_eq!(COEFF_NUM_NONZERO_CONTEXT[63], 206);
    }
}
