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
use crate::coeff_order::{
    coefficient_count, natural_coeff_order, order_id_for_transform, varblock_size_for_order,
};
use crate::dct_select::TransformType;
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

// ----------------------------------------------------------------------------
// Round 159 — §C.8.3 per-block HF coefficient decode loop scaffolding.
//
// Bounded scope: this lands the per-block raster-order coefficient walk
// from FDIS Listing C.13 prose + Listing C.14 (`prev` for context),
// parameterised over the symbol-source (so the §C.7.2 entropy stream
// can plug in once it lands) and over `(num_blocks, size)` (so the
// scaffolding stays usable beyond DCT8×8). The two helpers below are
// deliberately small — Listings C.13 / C.14 + the §C.8.3 prose
// transcribed almost verbatim into a `for k in [num_blocks, size)`
// loop — because every wiring choice (per-channel non_zeros read,
// histogram-offset bookkeeping, NonZeros-grid maintenance) belongs to
// the per-group driver above the per-block step. The driver itself is
// follow-up; this round only commits the per-block primitive + its
// stand-alone tests.
//
// The DCT8×8-alone shape (num_blocks = 1, size = 64) is the simplest
// case that exercises the full state machine:
//   * `prev` per Listing C.14 — `k == num_blocks` is the first
//     iteration, so the `if (k == num_blocks)` branch fires for k = 1.
//   * `non_zeros` decrements every nonzero coefficient and the loop
//     stops as soon as it hits 0 (the spec prose right after Listing
//     C.14: "If `ucoeff != 0`, the decoder decreases `non_zeros` by 1.
//     If `non_zeros` reaches 0, the decoder stops decoding further
//     coefficients for the current block.").
//   * `CoefficientContext` indexes the two ladder tables
//     `COEFF_FREQ_CONTEXT` / `COEFF_NUM_NONZERO_CONTEXT` and adds the
//     histogram offset.
// ----------------------------------------------------------------------------

/// `prev` per FDIS Listing C.14.
///
/// `prev = 1` iff:
///   * `k == num_blocks` and `non_zeros > size / 16`, **or**
///   * `k != num_blocks` and the previously-decoded coefficient at
///     position `k - 1` is non-zero.
///
/// Otherwise `prev = 0`. (Listing C.14 verbatim — the FDIS text uses
/// `0` as the "previous coefficient was zero / this is the first read
/// and non_zeros is low" sentinel, `1` as the "previous coefficient was
/// non-zero / first read with high non_zeros" sentinel.)
///
/// Caller responsibility: `prev_coeff_nonzero(k - 1)` is undefined when
/// `k == num_blocks`, so we never call it in that branch.
pub fn prev_for_context<F>(
    k: u32,
    num_blocks: u32,
    size: u32,
    non_zeros: u32,
    prev_nonzero: F,
) -> u32
where
    F: Fn(u32) -> bool,
{
    if k == num_blocks {
        if non_zeros > size / 16 {
            1
        } else {
            0
        }
    } else if prev_nonzero(k - 1) {
        1
    } else {
        0
    }
}

/// Decoded block bundle returned by [`decode_block_coefficients`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedHfBlock {
    /// Quantised HF coefficients placed in **natural-order index
    /// space** (NOT in raster order): `coeffs[ natural_order[k] ]`
    /// for `k in [num_blocks, size)` and 0 elsewhere. The caller maps
    /// the natural-order index back to a `(y, x)` in the varblock per
    /// `natural_order[k] = y * bwidth + x`.
    pub coeffs: Vec<i32>,
    /// `non_zeros` count after the per-block loop completed (always 0
    /// when the loop runs to completion via the `if non_zeros reaches
    /// 0` early-stop; non-zero only if every coefficient in the block
    /// was actually non-zero — which the spec disallows for a real
    /// block since `non_zeros` must reach 0 before the final `k`).
    pub remaining_non_zeros: u32,
    /// Number of coefficients the loop actually decoded (the symbol
    /// reads performed against the entropy stream). Useful for tests
    /// that want to assert "the loop terminated after N reads".
    pub coeffs_read: u32,
}

/// FDIS §C.8.3 per-block coefficient decode loop — Listing C.14 +
/// the surrounding prose.
///
/// Walks `k` from `num_blocks` to `size`, computing the
/// `CoefficientContext` per Listing C.13, reading a `ucoeff` symbol
/// from the caller-supplied `decode_symbol` closure (which is expected
/// to issue a `D[ctx + offset]` read against the §C.7.2 entropy
/// stream), placing `UnpackSigned(ucoeff)` at position
/// `natural_order[k]` in the output block, and stopping as soon as
/// `non_zeros` reaches 0.
///
/// The closure interface keeps this primitive independent of the
/// (still un-landed) §C.7.2 histogram array. A real consumer will
/// wrap a `EntropyStream` + `HybridUintState` + the per-group
/// histogram offset into the closure; tests can hand-roll a symbol
/// table.
///
/// The natural-order vector `natural_order` MUST have length `size`
/// and contain a permutation of `[0, size)` (Listing C.12). For
/// DCT8×8 alone, callers pass `crate::coeff_order::natural_coeff_order(
/// OrderId::Id0)` (the LLF prefix is the single cell `(0, 0)`, so the
/// HF tail starts at `k = 1`).
///
/// `block_ctx` is the already-computed `BlockContext()` value for the
/// current varblock (§C.13). `nb_block_ctx` is the LfGlobal
/// HfBlockContext invariant.
///
/// Returns a [`DecodedHfBlock`] with the coefficient buffer in
/// natural-order **index space**, the final `non_zeros` value, and
/// the number of symbol reads performed.
pub fn decode_block_coefficients<F>(
    natural_order: &[u32],
    num_blocks: u32,
    size: u32,
    initial_non_zeros: u32,
    block_ctx: u32,
    nb_block_ctx: u32,
    mut decode_symbol: F,
) -> Result<DecodedHfBlock>
where
    F: FnMut(u32) -> Result<u32>,
{
    if num_blocks == 0 {
        return Err(Error::InvalidData(
            "JXL block coeff loop: num_blocks must be ≥ 1".into(),
        ));
    }
    if size == 0 || num_blocks > size {
        return Err(Error::InvalidData(format!(
            "JXL block coeff loop: invalid (num_blocks={num_blocks}, size={size})"
        )));
    }
    if natural_order.len() != size as usize {
        return Err(Error::InvalidData(format!(
            "JXL block coeff loop: natural_order len {} != size {size}",
            natural_order.len()
        )));
    }
    // Sanity-check: every natural-order entry must be in [0, size).
    for &p in natural_order {
        if p >= size {
            return Err(Error::InvalidData(format!(
                "JXL block coeff loop: natural_order entry {p} >= size {size}"
            )));
        }
    }
    if initial_non_zeros > size - num_blocks {
        return Err(Error::InvalidData(format!(
            "JXL block coeff loop: initial_non_zeros {initial_non_zeros} > size - num_blocks ({})",
            size - num_blocks
        )));
    }

    let mut coeffs = vec![0i32; size as usize];
    let mut non_zeros = initial_non_zeros;
    let mut coeffs_read: u32 = 0;

    // Tracks "was the natural-order coefficient at position (k-1)
    // non-zero" for Listing C.14. Indexed by k (so prev_nonzero[k-1]).
    // For k == num_blocks we read prev via the size/16 path so the
    // [k=num_blocks-1] slot is never accessed; we still size the vec to
    // `size` for uniform indexing.
    let mut prev_nonzero = vec![false; size as usize];

    let mut k = num_blocks;
    while k < size {
        if non_zeros == 0 {
            // Spec: "If non_zeros reaches 0, the decoder stops
            // decoding further coefficients for the current block."
            break;
        }
        let prev = prev_for_context(k, num_blocks, size, non_zeros, |kk| {
            prev_nonzero[kk as usize]
        });
        let ctx = coefficient_context(
            k,
            non_zeros,
            num_blocks,
            size,
            prev,
            block_ctx,
            nb_block_ctx,
        )?;
        let ucoeff = decode_symbol(ctx)?;
        coeffs_read += 1;
        let signed = crate::bitreader::unpack_signed(ucoeff);
        let pos = natural_order[k as usize] as usize;
        coeffs[pos] = signed;
        if ucoeff != 0 {
            prev_nonzero[k as usize] = true;
            non_zeros = non_zeros.saturating_sub(1);
        }
        k += 1;
    }

    Ok(DecodedHfBlock {
        coeffs,
        remaining_non_zeros: non_zeros,
        coeffs_read,
    })
}

/// Convenience wrapper: read the per-channel `non_zeros` count from the
/// caller-supplied closure, then drive [`decode_block_coefficients`].
///
/// `predicted_non_zeros` is the [`predicted_non_zeros`] value for the
/// current varblock's top-left position; `block_ctx` is the
/// [`block_context`] result. The closure issues a `D[
/// NonZerosContext(predicted) + offset ]` read.
///
/// Returns `(decoded_block, non_zeros)` so the caller can update its
/// NonZeros-grid bookkeeping per the spec's:
///
/// > NonZeros(x, y) is then (non_zeros + num_blocks - 1) Idiv num_blocks.
#[allow(clippy::too_many_arguments)]
pub fn read_non_zeros_and_decode_block<F, G>(
    natural_order: &[u32],
    num_blocks: u32,
    size: u32,
    predicted: u32,
    block_ctx: u32,
    nb_block_ctx: u32,
    mut read_non_zeros: F,
    decode_symbol: G,
) -> Result<(DecodedHfBlock, u32)>
where
    F: FnMut(u32) -> Result<u32>,
    G: FnMut(u32) -> Result<u32>,
{
    let nz_ctx = non_zeros_context(predicted, block_ctx, nb_block_ctx);
    let non_zeros = read_non_zeros(nz_ctx)?;
    if non_zeros > size - num_blocks {
        return Err(Error::InvalidData(format!(
            "JXL block coeff loop: non_zeros {non_zeros} > size - num_blocks ({}) at predicted={predicted}",
            size - num_blocks
        )));
    }
    let decoded = decode_block_coefficients(
        natural_order,
        num_blocks,
        size,
        non_zeros,
        block_ctx,
        nb_block_ctx,
        decode_symbol,
    )?;
    Ok((decoded, non_zeros))
}

// ----------------------------------------------------------------------------
// Round 164 — `TransformType`-driven entry points for the per-block loop.
//
// Bounded scope: this lifts the round-159 raw-`(num_blocks, size)` API into
// a typed wrapper that derives `num_blocks` and `size` from a
// [`TransformType`] (Table C.16 + §I.2.4 opening paragraph), threading the
// matching natural-order vector through the existing pure scaffolding.
//
// Listing C.14 is itself shape-agnostic — `num_blocks` and `size` enter
// only as arithmetic — so this round adds the typed entry points + their
// integration tests at DCT16×16 / DCT16×8 dimensions to pin "the
// scaffolding stays usable beyond DCT8×8" (round-159 module docs) from a
// caller's perspective. No new bit-level reads happen here; both
// wrappers reduce to the existing `decode_block_coefficients` /
// `read_non_zeros_and_decode_block` after parameter derivation.
//
// Per §I.2.4 opening paragraph + Listing C.14's `num_blocks` symbol:
//
// * `num_blocks = (bwidth / 8) × (bheight / 8)` — the LLF cell count of
//   the varblock (its top-left rectangle the spec stamps as "the LLF
//   prefix"); equals 1 for the 8×8-output transforms (DCT8×8, Hornuss,
//   DCT2×2, DCT4×4, DCT4×8, DCT8×4, AFV0..AFV3), 4 for DCT16×16, 2 for
//   the rectangular DCT16×8 / DCT8×16, and scales up to 1024 for
//   DCT256×256.
// * `size = bwidth × bheight` — the total coefficient count of the
//   varblock = [`coefficient_count`] applied to the transform's
//   [`OrderId`]. 64 for the 8×8-output transforms, 256 for DCT16×16,
//   128 for DCT16×8, …, 65536 for DCT256×256.
// ----------------------------------------------------------------------------

/// `(num_blocks, size)` for a [`TransformType`] per §I.2.4 opening
/// paragraph + Listing C.14.
///
/// * `num_blocks = (bwidth / 8) × (bheight / 8)` — the LLF prefix cell
///   count of the varblock (its top-left rectangle the spec stamps as
///   "the LLF prefix").
/// * `size = bwidth × bheight` — the total coefficient count of the
///   varblock.
///
/// Both are pure derivations from the [`varblock_size_for_order`]
/// table; this helper keeps the per-call lookup terse for the typed
/// per-block entry points.
pub fn transform_block_params(t: TransformType) -> (u32, u32) {
    let oid = order_id_for_transform(t);
    let (bwidth, bheight) = varblock_size_for_order(oid);
    let num_blocks = (bwidth / 8) * (bheight / 8);
    let size = coefficient_count(oid);
    (num_blocks, size)
}

/// Typed [`decode_block_coefficients`] driven from a [`TransformType`].
///
/// Picks `(num_blocks, size)` via [`transform_block_params`] and the
/// natural-order vector via [`natural_coeff_order`] keyed by
/// [`order_id_for_transform`]. The rest of the per-block state machine
/// is identical to the raw entry point.
///
/// Defensively rejects `initial_non_zeros > size - num_blocks` (the
/// spec invariant Listing C.14 maintains across reads); the typed
/// wrapper also guarantees the natural-order length matches `size`,
/// so callers cannot accidentally pass a DCT8×8 order against a
/// DCT16×16 transform.
pub fn decode_block_coefficients_for_transform<F>(
    t: TransformType,
    initial_non_zeros: u32,
    block_ctx: u32,
    nb_block_ctx: u32,
    decode_symbol: F,
) -> Result<DecodedHfBlock>
where
    F: FnMut(u32) -> Result<u32>,
{
    let (num_blocks, size) = transform_block_params(t);
    let oid = order_id_for_transform(t);
    let order = natural_coeff_order(oid);
    decode_block_coefficients(
        &order,
        num_blocks,
        size,
        initial_non_zeros,
        block_ctx,
        nb_block_ctx,
        decode_symbol,
    )
}

/// Typed [`read_non_zeros_and_decode_block`] driven from a
/// [`TransformType`].
///
/// Identical shape to the raw entry point but with `(num_blocks, size)`
/// and natural-order derived from `t` per [`transform_block_params`]
/// and [`natural_coeff_order`] and [`order_id_for_transform`]. The
/// caller still owns the two closures and the NonZeros-grid
/// bookkeeping above this primitive (per the FDIS prose right before
/// Listing C.14: `NonZeros(x, y) = (non_zeros + num_blocks - 1) Idiv
/// num_blocks`).
#[allow(clippy::too_many_arguments)]
pub fn read_non_zeros_and_decode_block_for_transform<F, G>(
    t: TransformType,
    predicted: u32,
    block_ctx: u32,
    nb_block_ctx: u32,
    read_non_zeros: F,
    decode_symbol: G,
) -> Result<(DecodedHfBlock, u32)>
where
    F: FnMut(u32) -> Result<u32>,
    G: FnMut(u32) -> Result<u32>,
{
    let (num_blocks, size) = transform_block_params(t);
    let oid = order_id_for_transform(t);
    let order = natural_coeff_order(oid);
    read_non_zeros_and_decode_block(
        &order,
        num_blocks,
        size,
        predicted,
        block_ctx,
        nb_block_ctx,
        read_non_zeros,
        decode_symbol,
    )
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

    // ------------------------------------------------------------------------
    // Round 159 — per-block coefficient decode loop tests.
    // ------------------------------------------------------------------------

    use crate::coeff_order::{natural_coeff_order, OrderId};

    #[test]
    fn prev_for_context_first_iteration_high_non_zeros() {
        // k == num_blocks, non_zeros > size / 16 → prev = 1.
        // For DCT8×8: num_blocks = 1, size = 64, size/16 = 4. With
        // non_zeros = 5 the first read uses prev = 1.
        assert_eq!(prev_for_context(1, 1, 64, 5, |_| panic!("never called")), 1);
    }

    #[test]
    fn prev_for_context_first_iteration_low_non_zeros() {
        // k == num_blocks, non_zeros <= size / 16 → prev = 0.
        // For DCT8×8: non_zeros = 4 → 4 > 64 / 16 == 4 is false → prev = 0.
        assert_eq!(prev_for_context(1, 1, 64, 4, |_| panic!("never called")), 0);
        assert_eq!(prev_for_context(1, 1, 64, 0, |_| panic!("never called")), 0);
    }

    #[test]
    fn prev_for_context_subsequent_iteration_follows_prev_nonzero_flag() {
        // k > num_blocks → prev = prev_nonzero(k - 1) ? 1 : 0.
        // Stub a closure that says "the (k-1)=2 slot was non-zero" only
        // when k == 3.
        let f = |kk: u32| kk == 2;
        assert_eq!(prev_for_context(3, 1, 64, 7, f), 1);
        assert_eq!(prev_for_context(4, 1, 64, 7, f), 0);
        assert_eq!(prev_for_context(5, 1, 64, 7, f), 0);
    }

    #[test]
    fn decode_block_coefficients_rejects_zero_num_blocks() {
        let order = natural_coeff_order(OrderId::Id0);
        let r = decode_block_coefficients(&order, 0, 64, 0, 0, 1, |_| Ok(0));
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn decode_block_coefficients_rejects_bad_natural_order_len() {
        let bad = vec![0u32; 32]; // half of size = 64
        let r = decode_block_coefficients(&bad, 1, 64, 0, 0, 1, |_| Ok(0));
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn decode_block_coefficients_rejects_oob_natural_order_entry() {
        let mut bad = natural_coeff_order(OrderId::Id0);
        bad[5] = 99; // 99 >= size = 64
        let r = decode_block_coefficients(&bad, 1, 64, 0, 0, 1, |_| Ok(0));
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn decode_block_coefficients_rejects_too_many_non_zeros() {
        let order = natural_coeff_order(OrderId::Id0);
        // size - num_blocks = 64 - 1 = 63, so 64 is too many.
        let r = decode_block_coefficients(&order, 1, 64, 64, 0, 1, |_| Ok(0));
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn decode_block_coefficients_all_zero_block_reads_nothing() {
        // initial_non_zeros = 0 → the spec's "If non_zeros reaches 0,
        // the decoder stops" fires before any read happens. The loop
        // body must not even compute prev / call the closure.
        let order = natural_coeff_order(OrderId::Id0);
        let mut call_count = 0;
        let decoded = decode_block_coefficients(&order, 1, 64, 0, 0, 1, |_| {
            call_count += 1;
            Ok(0)
        })
        .unwrap();
        assert_eq!(call_count, 0);
        assert_eq!(decoded.coeffs_read, 0);
        assert_eq!(decoded.remaining_non_zeros, 0);
        assert!(decoded.coeffs.iter().all(|&v| v == 0));
        assert_eq!(decoded.coeffs.len(), 64);
    }

    #[test]
    fn decode_block_coefficients_single_nonzero_stops_after_one_read() {
        // initial_non_zeros = 1, closure returns ucoeff = 1 on the
        // first call. UnpackSigned(1) = -1 (1 is odd → -((1>>1)+1) =
        // -1). non_zeros decrements to 0 → the loop should stop after
        // exactly one read; subsequent k iterations must NOT call the
        // closure.
        let order = natural_coeff_order(OrderId::Id0);
        let mut call_count = 0;
        let decoded = decode_block_coefficients(&order, 1, 64, 1, 3, 15, |_ctx| {
            call_count += 1;
            // Return non-zero ucoeff on first read; never reached again.
            Ok(1)
        })
        .unwrap();
        assert_eq!(call_count, 1);
        assert_eq!(decoded.coeffs_read, 1);
        assert_eq!(decoded.remaining_non_zeros, 0);
        // The first read lands at natural_order[1] (k = num_blocks = 1).
        // For DCT8×8 the LLF prefix is the single (0, 0) cell → natural
        // order index 0 is the LLF cell; natural_order[1] is the first
        // HF cell. For OrderId::Id0 the HF tail starts with the smallest
        // `(key1, key2)` from Listing I.14. We don't pin the exact
        // raster index here (covered by `coeff_order` tests); we just
        // assert the rest of the buffer is zero.
        let nonzero_positions: Vec<(usize, i32)> = decoded
            .coeffs
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v != 0)
            .map(|(i, &v)| (i, v))
            .collect();
        assert_eq!(nonzero_positions.len(), 1);
        assert_eq!(nonzero_positions[0].1, -1); // unpack_signed(1) == -1
    }

    #[test]
    fn decode_block_coefficients_zero_ucoeff_does_not_decrement_non_zeros() {
        // initial_non_zeros = 2, closure returns ucoeff = 0 then 2
        // then 4 then... The first read returns 0 → non_zeros stays at
        // 2; the second read returns 2 (signed = +1) → non_zeros drops
        // to 1; the third read returns 4 (signed = +2) → non_zeros
        // drops to 0 → loop stops. Expect exactly 3 reads.
        let order = natural_coeff_order(OrderId::Id0);
        let sequence = [0u32, 2, 4];
        let mut idx = 0;
        let decoded = decode_block_coefficients(&order, 1, 64, 2, 0, 1, |_ctx| {
            let v = sequence[idx];
            idx += 1;
            Ok(v)
        })
        .unwrap();
        assert_eq!(decoded.coeffs_read, 3);
        assert_eq!(decoded.remaining_non_zeros, 0);
        let nonzero_count = decoded.coeffs.iter().filter(|&&v| v != 0).count();
        assert_eq!(nonzero_count, 2);
    }

    #[test]
    fn decode_block_coefficients_prev_flag_tracks_decoded_nonzero() {
        // Verify Listing C.14's "else { prev = ([[decoded coeff at
        // position (k - 1) is 0]]) ? 0 : 1; }" branch fires by
        // observing the context value passed to the closure.
        //
        // Plan: with initial_non_zeros = 2 and a sequence [2, 0, 2]
        // (signed [+1, 0, +1]), iteration k = num_blocks + 1 must see
        // prev = 1 (k-1's coeff was non-zero), iteration k = num_blocks
        // + 2 must see prev = 0 (k-1's coeff was zero). The loop runs
        // 3 times: nz=2 → 1 (first +1) → 1 (zero, no decrement) → 0
        // (second +1) → stop. We don't pin the absolute ctx value
        // (CoefficientContext is a tested helper above) — we re-derive
        // the expected ctx from the spec formula and compare.
        let order = natural_coeff_order(OrderId::Id0);
        let sequence = [2u32, 0, 2]; // signed [+1, 0, +1] — 2 non-zero
        let mut idx = 0;
        let mut seen_ctx: Vec<u32> = Vec::new();
        let block_ctx = 4;
        let nb_block_ctx = 15;
        let num_blocks = 1;
        let size = 64;
        let _ = decode_block_coefficients(
            &order,
            num_blocks,
            size,
            2,
            block_ctx,
            nb_block_ctx,
            |ctx| {
                seen_ctx.push(ctx);
                let v = sequence[idx];
                idx += 1;
                Ok(v)
            },
        )
        .unwrap();

        // The loop runs exactly 3 times (nz=2 → 1 → 1 → 0 → stop).
        assert_eq!(seen_ctx.len(), 3);

        // k = 1, num_blocks = 1, non_zeros = 2. k == num_blocks → prev
        // path is "non_zeros > size / 16 ? 1 : 0". 2 > 64/16=4 is false
        // → prev = 0.
        let expected_ctx_0 =
            coefficient_context(1, 2, num_blocks, size, 0, block_ctx, nb_block_ctx).unwrap();
        assert_eq!(seen_ctx[0], expected_ctx_0);

        // k = 2, non_zeros after first nonzero is 1. prev = 1 (the
        // previous coefficient was non-zero, ucoeff=2 ≠ 0).
        let expected_ctx_1 =
            coefficient_context(2, 1, num_blocks, size, 1, block_ctx, nb_block_ctx).unwrap();
        assert_eq!(seen_ctx[1], expected_ctx_1);

        // k = 3, non_zeros still 1 (ucoeff=0 didn't decrement). prev =
        // 0 (the previous coefficient WAS zero).
        let expected_ctx_2 =
            coefficient_context(3, 1, num_blocks, size, 0, block_ctx, nb_block_ctx).unwrap();
        assert_eq!(seen_ctx[2], expected_ctx_2);
    }

    #[test]
    fn decode_block_coefficients_places_at_natural_order_position() {
        // initial_non_zeros = 1, the first read returns ucoeff = 2 (signed
        // +1). The non-zero must land at natural_order[1] (k = num_blocks
        // for DCT8×8 with num_blocks=1). Verify the placement against
        // crate::coeff_order::natural_coeff_order(OrderId::Id0)[1].
        let order = natural_coeff_order(OrderId::Id0);
        let decoded = decode_block_coefficients(&order, 1, 64, 1, 0, 1, |_| Ok(2)).unwrap();
        let expected_pos = order[1] as usize;
        assert_eq!(decoded.coeffs[expected_pos], 1);
        // every other slot must be zero.
        for (i, &v) in decoded.coeffs.iter().enumerate() {
            if i == expected_pos {
                continue;
            }
            assert_eq!(v, 0, "slot {i} should be zero, got {v}");
        }
    }

    #[test]
    fn read_non_zeros_and_decode_block_threads_predicted_through_context() {
        // The first closure (read_non_zeros) sees the
        // NonZerosContext(predicted, block_ctx, nb_block_ctx) value;
        // returning a specific non_zeros count should feed straight
        // into the second closure's coefficient-context derivation.
        let order = natural_coeff_order(OrderId::Id0);
        let block_ctx = 5;
        let nb_block_ctx = 15;
        let predicted = 32;

        let mut saw_nz_ctx: Option<u32> = None;
        let read_non_zeros = |ctx: u32| -> Result<u32> {
            saw_nz_ctx = Some(ctx);
            // Return 1 → one coefficient to decode.
            Ok(1)
        };
        let mut saw_coeff_ctx: Option<u32> = None;
        let decode_symbol = |ctx: u32| -> Result<u32> {
            saw_coeff_ctx = Some(ctx);
            // Return 2 → signed +1.
            Ok(2)
        };
        let (decoded, non_zeros) = read_non_zeros_and_decode_block(
            &order,
            1,
            64,
            predicted,
            block_ctx,
            nb_block_ctx,
            read_non_zeros,
            decode_symbol,
        )
        .unwrap();

        // The NonZerosContext closure saw exactly the spec formula
        // (non_zeros_context returns block_ctx + nb_block_ctx ×
        // (4 + 32/2) for predicted = 32).
        let expected_nz_ctx = non_zeros_context(predicted, block_ctx, nb_block_ctx);
        assert_eq!(saw_nz_ctx, Some(expected_nz_ctx));
        assert_eq!(non_zeros, 1);

        // The CoefficientContext closure saw the k=1, non_zeros=1
        // formula with prev = 0 (1 > 64/16=4 is false).
        let expected_coeff_ctx =
            coefficient_context(1, 1, 1, 64, 0, block_ctx, nb_block_ctx).unwrap();
        assert_eq!(saw_coeff_ctx, Some(expected_coeff_ctx));
        assert_eq!(decoded.coeffs_read, 1);
        assert_eq!(decoded.remaining_non_zeros, 0);
    }

    #[test]
    fn read_non_zeros_and_decode_block_rejects_oversized_non_zeros() {
        // non_zeros must satisfy `non_zeros <= size - num_blocks`. If
        // the closure returns a too-large count we must reject before
        // the inner loop runs.
        let order = natural_coeff_order(OrderId::Id0);
        let read_non_zeros = |_| Ok(64); // size - num_blocks = 63
        let r = read_non_zeros_and_decode_block(
            &order,
            1,
            64,
            0,
            0,
            1,
            read_non_zeros,
            |_| -> Result<u32> { panic!("decode_symbol should not be called") },
        );
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn decode_block_coefficients_full_block_decodes_all_size_reads() {
        // initial_non_zeros = size - num_blocks (the maximum allowed,
        // every HF coefficient non-zero). Every closure call returns
        // 2 → signed +1 → non_zeros decrements each step → the loop
        // runs exactly `size - num_blocks = 63` times for DCT8×8.
        let order = natural_coeff_order(OrderId::Id0);
        let mut call_count = 0;
        let decoded = decode_block_coefficients(&order, 1, 64, 63, 0, 1, |_| {
            call_count += 1;
            Ok(2)
        })
        .unwrap();
        assert_eq!(call_count, 63);
        assert_eq!(decoded.coeffs_read, 63);
        assert_eq!(decoded.remaining_non_zeros, 0);
        // 63 non-zero slots in the natural-order tail (positions
        // natural_order[1..64]).
        let nonzero_count = decoded.coeffs.iter().filter(|&&v| v != 0).count();
        assert_eq!(nonzero_count, 63);
        // Position natural_order[0] (the LLF cell) is untouched (== 0).
        assert_eq!(decoded.coeffs[order[0] as usize], 0);
    }

    // --- Round 164: typed `TransformType` entry points -----------------

    /// `transform_block_params` derives `(num_blocks, size)` exactly per
    /// §I.2.4 opening paragraph. Spot-check the four canonical shapes
    /// this round exercises.
    #[test]
    fn transform_block_params_known_shapes() {
        // DCT8×8 / Hornuss / DCT2×2 / DCT4×4 / DCT4×8 / DCT8×4 / AFV*:
        // num_blocks = 1, size = 64.
        assert_eq!(transform_block_params(TransformType::Dct8x8), (1, 64));
        assert_eq!(transform_block_params(TransformType::Hornuss), (1, 64));
        assert_eq!(transform_block_params(TransformType::Dct2x2), (1, 64));
        assert_eq!(transform_block_params(TransformType::Dct4x4), (1, 64));
        assert_eq!(transform_block_params(TransformType::Dct4x8), (1, 64));
        assert_eq!(transform_block_params(TransformType::Dct8x4), (1, 64));
        assert_eq!(transform_block_params(TransformType::Afv0), (1, 64));
        assert_eq!(transform_block_params(TransformType::Afv3), (1, 64));
        // DCT16×16 → num_blocks = 4, size = 256.
        assert_eq!(transform_block_params(TransformType::Dct16x16), (4, 256));
        // DCT16×8 → (bwidth, bheight) = (16, 8) → num_blocks = 2,
        // size = 128.
        assert_eq!(transform_block_params(TransformType::Dct16x8), (2, 128));
        // DCT8×16 → same OrderId::Id4, so (num_blocks, size) collapses
        // to (2, 128) too.
        assert_eq!(transform_block_params(TransformType::Dct8x16), (2, 128));
        // DCT32×32 → num_blocks = 16, size = 1024.
        assert_eq!(transform_block_params(TransformType::Dct32x32), (16, 1024));
        // DCT256×256 → num_blocks = 1024, size = 65536. (Sanity:
        // num_blocks * 64 == size for every "square" DCTNxN since
        // bwidth = bheight = N → num_blocks = (N/8)^2 → size = N^2 =
        // num_blocks * 64.)
        assert_eq!(
            transform_block_params(TransformType::Dct256x256),
            (1024, 65536)
        );
    }

    /// `transform_block_params` round-trips every Table C.16 entry
    /// through the `num_blocks * 64 == size`-for-DCT-NxN invariant
    /// (and the rectangular DCT shapes). Catches typos in
    /// [`varblock_size_for_order`] without re-pinning every cell.
    #[test]
    fn transform_block_params_size_equals_bwidth_times_bheight() {
        let all = [
            TransformType::Dct8x8,
            TransformType::Hornuss,
            TransformType::Dct2x2,
            TransformType::Dct4x4,
            TransformType::Dct16x16,
            TransformType::Dct32x32,
            TransformType::Dct16x8,
            TransformType::Dct8x16,
            TransformType::Dct32x8,
            TransformType::Dct8x32,
            TransformType::Dct32x16,
            TransformType::Dct16x32,
            TransformType::Dct4x8,
            TransformType::Dct8x4,
            TransformType::Afv0,
            TransformType::Afv1,
            TransformType::Afv2,
            TransformType::Afv3,
            TransformType::Dct64x64,
            TransformType::Dct64x32,
            TransformType::Dct32x64,
            TransformType::Dct128x128,
            TransformType::Dct128x64,
            TransformType::Dct64x128,
            TransformType::Dct256x256,
            TransformType::Dct256x128,
            TransformType::Dct128x256,
        ];
        for t in all {
            let (num_blocks, size) = transform_block_params(t);
            let oid = order_id_for_transform(t);
            let (bwidth, bheight) = varblock_size_for_order(oid);
            assert!(num_blocks >= 1, "{t:?}: num_blocks {num_blocks} < 1",);
            assert_eq!(
                num_blocks,
                (bwidth / 8) * (bheight / 8),
                "{t:?}: num_blocks derivation",
            );
            assert_eq!(size, bwidth * bheight, "{t:?}: size derivation",);
            // num_blocks * 64 == size (every 8×8-tiled rectangular DCT
            // varblock is a whole number of 8×8 cells).
            assert_eq!(num_blocks * 64, size, "{t:?}: num_blocks * 64 != size",);
        }
    }

    /// Typed entry point for DCT8×8 reduces to the raw entry point on
    /// the matching natural-order vector and dimensions.
    #[test]
    fn decode_block_coefficients_for_transform_dct8x8_matches_raw() {
        let order = natural_coeff_order(OrderId::Id0);
        let typed =
            decode_block_coefficients_for_transform(TransformType::Dct8x8, 1, 0, 1, |_| Ok(2))
                .unwrap();
        let raw = decode_block_coefficients(&order, 1, 64, 1, 0, 1, |_| Ok(2)).unwrap();
        assert_eq!(typed, raw);
    }

    /// Typed entry point for DCT16×16 walks the (num_blocks=4, size=256)
    /// scaffolding: a single non-zero at the first HF slot terminates
    /// after one symbol read; the coefficient lands at
    /// `natural_coeff_order(Id2)[num_blocks]`.
    #[test]
    fn decode_block_coefficients_for_transform_dct16x16_first_nonzero() {
        let order = natural_coeff_order(OrderId::Id2);
        assert_eq!(order.len(), 256);
        let mut calls = 0;
        let decoded =
            decode_block_coefficients_for_transform(TransformType::Dct16x16, 1, 0, 1, |_| {
                calls += 1;
                Ok(1)
            })
            .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(decoded.coeffs.len(), 256);
        // num_blocks for DCT16×16 is 4 → first HF slot is k = 4.
        let first_hf_pos = order[4] as usize;
        assert_eq!(decoded.coeffs[first_hf_pos], -1, "UnpackSigned(1) == -1");
        // Every other slot is zero.
        for (i, &v) in decoded.coeffs.iter().enumerate() {
            if i == first_hf_pos {
                continue;
            }
            assert_eq!(v, 0);
        }
    }

    /// `prev` Listing C.14 threshold for DCT16×16 is at
    /// `non_zeros == size/16 + 1 = 17`. Exercises the
    /// `non_zeros > size / 16` branch with the typed entry point's
    /// `(num_blocks, size) = (4, 256)`.
    #[test]
    fn prev_for_context_dct16x16_threshold_at_17() {
        // size / 16 = 16 for DCT16×16 → crossover at non_zeros == 17.
        for nz in 0..=16 {
            assert_eq!(
                prev_for_context(4, 4, 256, nz, |_| panic!("never called")),
                0,
                "DCT16×16 prev at nz={nz}",
            );
        }
        for nz in 17..=63 {
            assert_eq!(
                prev_for_context(4, 4, 256, nz, |_| panic!("never called")),
                1,
                "DCT16×16 prev at nz={nz}",
            );
        }
    }

    /// `read_non_zeros_and_decode_block_for_transform` threads
    /// `(num_blocks, size, natural_order)` through both closures.
    #[test]
    fn read_non_zeros_and_decode_block_for_transform_dct16x16() {
        let order = natural_coeff_order(OrderId::Id2);
        let predicted = 4u32;
        // NonZerosContext(4, 0, 1) = 0 + 1 × 4 = 4 (predicted < 8 branch).
        let expected_nz_ctx = non_zeros_context(predicted, 0, 1);
        assert_eq!(expected_nz_ctx, 4);
        let mut nz_seen = u32::MAX;
        let mut coeff_calls = 0u32;
        let (decoded, non_zeros) = read_non_zeros_and_decode_block_for_transform(
            TransformType::Dct16x16,
            predicted,
            0,
            1,
            |ctx| {
                nz_seen = ctx;
                Ok(3) // three non-zeros for this block
            },
            |_| {
                coeff_calls += 1;
                Ok(2) // signed +1 each
            },
        )
        .unwrap();
        assert_eq!(nz_seen, expected_nz_ctx);
        assert_eq!(non_zeros, 3);
        assert_eq!(coeff_calls, 3);
        assert_eq!(decoded.coeffs_read, 3);
        assert_eq!(decoded.remaining_non_zeros, 0);
        // Three +1 values at positions natural_order[4..7] (DCT16×16
        // num_blocks = 4 so HF tail starts at k = 4).
        assert_eq!(decoded.coeffs[order[4] as usize], 1);
        assert_eq!(decoded.coeffs[order[5] as usize], 1);
        assert_eq!(decoded.coeffs[order[6] as usize], 1);
    }

    /// Rectangular DCT16×8 varblock — `(num_blocks, size) = (2, 128)`.
    /// All-zero `initial_non_zeros` short-circuits to no symbol reads.
    #[test]
    fn decode_block_coefficients_for_transform_dct16x8_all_zero() {
        let mut calls = 0;
        let decoded =
            decode_block_coefficients_for_transform(TransformType::Dct16x8, 0, 0, 1, |_| {
                calls += 1;
                Ok(0)
            })
            .unwrap();
        assert_eq!(calls, 0);
        assert_eq!(decoded.coeffs.len(), 128);
        assert_eq!(decoded.coeffs_read, 0);
        assert!(decoded.coeffs.iter().all(|&v| v == 0));
    }

    /// Typed entry point defensively rejects an initial_non_zeros that
    /// exceeds `size - num_blocks` for the chosen transform.
    /// For DCT16×16: max legal = 256 - 4 = 252.
    #[test]
    fn decode_block_coefficients_for_transform_rejects_oob_non_zeros() {
        let r =
            decode_block_coefficients_for_transform(TransformType::Dct16x16, 253, 0, 1, |_| Ok(0));
        assert!(matches!(r, Err(Error::InvalidData(_))));
        // 252 is OK at the validation layer (the loop itself will run
        // 252 reads). Smoke-test that with a closure that returns 0
        // (which never decrements non_zeros, so the loop exits on
        // `k == size`).
        let mut calls = 0;
        let decoded =
            decode_block_coefficients_for_transform(TransformType::Dct16x16, 252, 0, 1, |_| {
                calls += 1;
                Ok(0)
            })
            .unwrap();
        // The loop walks k from num_blocks=4 to size=256 (= 252
        // iterations) since every read returns 0 → non_zeros never
        // decrements.
        assert_eq!(calls, 252);
        assert_eq!(decoded.coeffs_read, 252);
        // All coefficients are zero (every UnpackSigned(0) = 0).
        assert!(decoded.coeffs.iter().all(|&v| v == 0));
    }
}
