//! Cross-pass HF coefficient accumulation —
//! ISO/IEC FDIS 18181-1:2021 §C.8.3 closing prose + Table C.6 `Passes`.
//!
//! ## Scope
//!
//! The per-LfGroup multi-pass decode driver
//! ([`crate::multi_pass_decode::decode_multi_pass_three_channels_with_resolver`])
//! produces, for a `num_passes`-pass frame, one
//! [`crate::pass_group_hf::DecodedHfBlock`] **per pass** per channel per
//! varblock — `out[p][i]` is the `i`-th varblock decoded in pass `p`.
//! Those per-pass coefficient blocks are *deltas*, not the final
//! coefficients: the spec partitions the HF coefficients of each block
//! across the passes and the final quantised coefficient grid is the
//! **sum across passes** of the (per-pass-left-shifted) deltas.
//!
//! This module is the pure-arithmetic stage that turns the per-pass
//! `DecodedHfBlock` stack into a single accumulated quantised
//! coefficient grid per varblock — the buffer the F.3 dequant →
//! §I.2.4 LLF merge → §I.2.3.2 inverse-DCT walk
//! ([`crate::block_dequant::decode_block_to_residual_with_llf`])
//! consumes. It owns no bit reads, no entropy state, no spec
//! re-derivation: it is the cross-pass analogue of the per-block
//! placement geometry in [`crate::residual_plane`].
//!
//! ## FDIS prose anchors
//!
//! Two spec sentences fully specify the accumulation:
//!
//! **Table C.6 `shift[i]`** (§C.4 frame-header `Passes` bundle):
//!
//! > `shift[i]` indicates the amount by which the HF coefficients of the
//! > pass with index `i` in the range `[0, num_passes)` are left-shifted
//! > immediately after entropy decoding. The last pass behaves as if it
//! > had a shift value of 0.
//!
//! So the per-pass left-shift is
//!
//! ```text
//! pass_shift(p) = shift[p]   for p in [0, num_passes - 1)
//! pass_shift(num_passes - 1) = 0
//! ```
//!
//! and the parsed [`crate::frame_header::Passes::shift`] vector has
//! length `num_passes - 1` (the last pass's implicit 0 is *not* stored).
//!
//! **§C.8.3 closing line** (right after the per-block coefficient loop):
//!
//! > If this is not the first pass, the decoder adds decoded HF
//! > coefficients to previously-decoded ones.
//!
//! Combined: the accumulated quantised coefficient at raster cell `c` is
//!
//! ```text
//! acc[c] = sum over p in [0, num_passes) of (delta[p][c] << pass_shift(p))
//! ```
//!
//! The shift is applied to the *quantised* integer coefficient (before
//! any F.3 dequantisation); the dequant + inverse-DCT then run once on
//! the accumulated grid, exactly as they do for the single-pass case.
//!
//! ## Non-square + LLF
//!
//! The accumulation is a flat cell-wise integer sum, so it is uniform
//! across **every** [`crate::dct_select::TransformType`] — the square,
//! rectangular (non-square: DCT8×16 / DCT16×8 / DCT32×8 / … and their
//! larger relatives), and non-DCT families alike. Each
//! [`crate::pass_group_hf::DecodedHfBlock::coeffs`] is already in the
//! shared `bwidth × bheight` raster layout (the per-pass decode placed
//! every coefficient at `natural_order[k]`), so the cross-pass sum needs
//! no per-transform reshaping. The LLF (DC subband) prefix is **not**
//! touched here: the §C.8.3 per-block loop only reads HF symbols for
//! `k ∈ [num_blocks, size)`, so every LLF cell of every per-pass
//! `DecodedHfBlock` is zero, and the accumulated grid likewise carries
//! zero in its `cy × cx` top-left cells. The LF-image → coefficient
//! seeding happens downstream in
//! [`crate::block_dequant::merge_llf_into_block`] after dequant — this
//! module's output is exactly the HF-only accumulated quantised grid
//! that single-pass decode would have produced for a one-pass frame.

use oxideav_core::{Error, Result};

use crate::frame_header::Passes;
use crate::pass_group_hf::DecodedHfBlock;

/// The per-pass left-shift applied to a pass's HF coefficients
/// immediately after entropy decoding, per FDIS Table C.6 `shift[i]`.
///
/// * For `p ∈ [0, num_passes - 1)` this is `passes.shift[p]`.
/// * For the last pass (`p == num_passes - 1`) this is `0` — the spec's
///   "the last pass behaves as if it had a shift value of 0".
///
/// `passes.shift` has length `num_passes - 1` (the parsed bundle never
/// stores the last pass's implicit 0), so this helper bridges the
/// "stored length is one short" convention.
///
/// Errors with [`Error::InvalidData`] if `p >= passes.num_passes`, or if
/// the stored `shift` vector length disagrees with `num_passes - 1` (a
/// malformed `Passes` bundle).
pub fn pass_shift(passes: &Passes, p: u32) -> Result<u32> {
    if p >= passes.num_passes {
        return Err(Error::InvalidData(format!(
            "JXL cross_pass: pass index {p} >= num_passes {}",
            passes.num_passes
        )));
    }
    // num_passes >= 1 is a Passes invariant; single-pass frames have an
    // empty shift vector and the only valid p (0) is the last pass.
    let expect_len = (passes.num_passes - 1) as usize;
    if passes.shift.len() != expect_len {
        return Err(Error::InvalidData(format!(
            "JXL cross_pass: Passes.shift length {} != num_passes - 1 ({expect_len})",
            passes.shift.len()
        )));
    }
    // Last pass → implicit 0.
    if p == passes.num_passes - 1 {
        return Ok(0);
    }
    Ok(passes.shift[p as usize])
}

/// Accumulate one varblock's per-pass quantised HF coefficient blocks
/// (a single channel) into one quantised coefficient grid, applying the
/// per-pass left-shift of §C.8.3.
///
/// `per_pass[p]` is the pass-`p` [`DecodedHfBlock`] for this
/// (channel, varblock) pair; its `coeffs` is the `bwidth × bheight`
/// raster-layout quantised block (HF cells decoded, LLF cells zero).
/// All passes share the same coefficient-grid length (the transform is
/// chosen once per varblock and applies to every pass), so the blocks
/// must all have the same `coeffs.len()`.
///
/// The returned grid is `sum_p (per_pass[p].coeffs[c] << pass_shift(p))`
/// for each raster cell `c`, in `i32`. The left-shift and the running
/// sum are checked: a shift that would overflow `i32`, or a partial sum
/// that would overflow `i32`, surfaces as [`Error::InvalidData`] rather
/// than wrapping (a conformant codestream's quantised coefficients stay
/// well inside `i32`, but a malformed stream must not silently wrap).
///
/// `passes.num_passes` MUST equal `per_pass.len()`; a mismatch is
/// [`Error::InvalidData`]. An empty grid (`coeffs.len() == 0`) is
/// rejected — every covered transform has a non-empty coefficient grid.
pub fn accumulate_block_across_passes(
    passes: &Passes,
    per_pass: &[&DecodedHfBlock],
) -> Result<Vec<i32>> {
    if per_pass.len() as u32 != passes.num_passes {
        return Err(Error::InvalidData(format!(
            "JXL cross_pass: per_pass len {} != num_passes {}",
            per_pass.len(),
            passes.num_passes
        )));
    }
    if per_pass.is_empty() {
        return Err(Error::InvalidData(
            "JXL cross_pass: num_passes must be >= 1".into(),
        ));
    }
    let n = per_pass[0].coeffs.len();
    if n == 0 {
        return Err(Error::InvalidData(
            "JXL cross_pass: empty coefficient grid".into(),
        ));
    }
    for (p, blk) in per_pass.iter().enumerate() {
        if blk.coeffs.len() != n {
            return Err(Error::InvalidData(format!(
                "JXL cross_pass: pass {p} coeffs length {} != pass 0 length {n}",
                blk.coeffs.len()
            )));
        }
    }

    // Pre-compute each pass's shift once (validates p < num_passes and
    // the stored shift-vector length).
    let mut shifts = Vec::with_capacity(per_pass.len());
    for p in 0..passes.num_passes {
        shifts.push(pass_shift(passes, p)?);
    }

    let mut acc = vec![0i32; n];
    for (p, blk) in per_pass.iter().enumerate() {
        let s = shifts[p];
        // A left-shift of an i32 by >= 32 is undefined behaviour in the
        // arithmetic sense and always overflows for a non-zero operand;
        // reject up front. (Conformant `shift[i]` is a u(2) field → at
        // most 3, but a malformed Passes bundle could carry more.)
        if s >= 32 {
            return Err(Error::InvalidData(format!(
                "JXL cross_pass: pass {p} shift {s} >= 32 (would overflow i32)"
            )));
        }
        for (c, &q) in blk.coeffs.iter().enumerate() {
            let shifted = (q as i64) << s;
            if shifted < i32::MIN as i64 || shifted > i32::MAX as i64 {
                return Err(Error::InvalidData(format!(
                    "JXL cross_pass: pass {p} cell {c} coefficient {q} << {s} \
                     overflows i32"
                )));
            }
            let sum = acc[c] as i64 + shifted;
            if sum < i32::MIN as i64 || sum > i32::MAX as i64 {
                return Err(Error::InvalidData(format!(
                    "JXL cross_pass: cell {c} accumulated sum overflows i32 \
                     at pass {p}"
                )));
            }
            acc[c] = sum as i32;
        }
    }
    Ok(acc)
}

/// One varblock's accumulated quantised coefficient grids for all three
/// XYB channels — index `0 = X`, `1 = Y`, `2 = B`.
///
/// Each entry is the [`accumulate_block_across_passes`] result for that
/// channel: the cross-pass sum of the per-pass left-shifted quantised
/// blocks, in the shared `bwidth × bheight` raster layout.
pub type AccumulatedVarblock = [Vec<i32>; 3];

/// Accumulate the full multi-pass output of one LfGroup into per-varblock
/// three-channel quantised coefficient grids.
///
/// `multi_pass[p][i]` is the pass-`p` decode of the `i`-th varblock (the
/// [`crate::multi_pass_decode::MultiPassThreeChannelOutput`] shape:
/// `(Varblock, [DecodedHfBlock; 3], [u32; 3])`). This driver transposes
/// the `[pass][varblock]` grid into per-varblock per-channel stacks and
/// runs [`accumulate_block_across_passes`] once per (varblock, channel),
/// returning `out[i]` = the `i`-th varblock's three accumulated grids in
/// the canonical raster walk order.
///
/// All passes MUST agree on the varblock count and the per-varblock
/// [`crate::varblock_walk::Varblock`] placement (the §C.8.3 invariant:
/// the DctSelect grid is decoded once and every pass walks it
/// identically); a per-pass length mismatch is [`Error::InvalidData`].
/// The returned [`crate::varblock_walk::Varblock`] for each entry is
/// taken from pass 0 (they are identical across passes by that
/// invariant) and returned alongside the accumulated grids so the caller
/// can drive placement without re-walking the grid.
///
/// For `num_passes == 1` this is a pure copy (`pass_shift(0) == 0`): the
/// accumulated grid equals pass 0's coefficients, so a single-pass frame
/// flows through unchanged — the cross-pass stage is a no-op for the
/// common case and only does real work for genuinely progressive
/// (multi-pass) frames.
pub fn accumulate_three_channel_multi_pass(
    passes: &Passes,
    multi_pass: &crate::multi_pass_decode::MultiPassThreeChannelOutput,
) -> Result<Vec<(crate::varblock_walk::Varblock, AccumulatedVarblock)>> {
    if multi_pass.len() as u32 != passes.num_passes {
        return Err(Error::InvalidData(format!(
            "JXL cross_pass: multi_pass pass count {} != num_passes {}",
            multi_pass.len(),
            passes.num_passes
        )));
    }
    if multi_pass.is_empty() {
        return Err(Error::InvalidData(
            "JXL cross_pass: num_passes must be >= 1".into(),
        ));
    }
    let nb = multi_pass[0].len();
    for (p, pass) in multi_pass.iter().enumerate() {
        if pass.len() != nb {
            return Err(Error::InvalidData(format!(
                "JXL cross_pass: pass {p} varblock count {} != pass 0 count {nb}",
                pass.len()
            )));
        }
    }

    let mut out = Vec::with_capacity(nb);
    for i in 0..nb {
        // The Varblock placement is identical across passes (§C.8.3); use
        // pass 0's, and verify the others agree to catch a mis-wired
        // multi-pass driver.
        let vb0 = multi_pass[0][i].0;
        for (p, pass) in multi_pass.iter().enumerate().skip(1) {
            let vbp = pass[i].0;
            if vbp.x != vb0.x || vbp.y != vb0.y || vbp.transform != vb0.transform {
                return Err(Error::InvalidData(format!(
                    "JXL cross_pass: pass {p} varblock {i} placement ({},{},{:?}) \
                     differs from pass 0 ({},{},{:?})",
                    vbp.x, vbp.y, vbp.transform, vb0.x, vb0.y, vb0.transform
                )));
            }
        }

        let mut channels: [Vec<i32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for (c, slot) in channels.iter_mut().enumerate() {
            let per_pass: Vec<&DecodedHfBlock> =
                multi_pass.iter().map(|pass| &pass[i].1[c]).collect();
            *slot = accumulate_block_across_passes(passes, &per_pass)?;
        }
        out.push((vb0, channels));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dct_select::TransformType;
    use crate::varblock_walk::Varblock;

    fn passes(num_passes: u32, shift: Vec<u32>) -> Passes {
        Passes {
            num_passes,
            num_ds: 0,
            shift,
            downsample: Vec::new(),
            last_pass: Vec::new(),
        }
    }

    fn block(coeffs: Vec<i32>) -> DecodedHfBlock {
        DecodedHfBlock {
            coeffs,
            remaining_non_zeros: 0,
            coeffs_read: 0,
        }
    }

    fn vb(x: u32, y: u32, t: TransformType) -> Varblock {
        Varblock {
            x,
            y,
            transform: t,
            hf_mul: 1,
        }
    }

    // ---- pass_shift ----

    #[test]
    fn pass_shift_single_pass_is_zero() {
        let p = passes(1, vec![]);
        assert_eq!(pass_shift(&p, 0).unwrap(), 0);
    }

    #[test]
    fn pass_shift_last_pass_is_implicit_zero() {
        // 3 passes, shift vector length 2 (passes 0 and 1). Pass 2 (last)
        // is the implicit 0.
        let p = passes(3, vec![2, 1]);
        assert_eq!(pass_shift(&p, 0).unwrap(), 2);
        assert_eq!(pass_shift(&p, 1).unwrap(), 1);
        assert_eq!(pass_shift(&p, 2).unwrap(), 0);
    }

    #[test]
    fn pass_shift_rejects_out_of_range_pass() {
        let p = passes(2, vec![3]);
        assert!(pass_shift(&p, 2).is_err());
    }

    #[test]
    fn pass_shift_rejects_wrong_shift_vector_length() {
        // num_passes 3 needs a shift vector of length 2.
        let p = passes(3, vec![1]);
        assert!(pass_shift(&p, 0).is_err());
    }

    // ---- accumulate_block_across_passes ----

    #[test]
    fn accumulate_single_pass_is_identity() {
        // One pass, shift 0 → accumulated == pass 0 coeffs.
        let p = passes(1, vec![]);
        let b = block(vec![1, -2, 3, 0]);
        let acc = accumulate_block_across_passes(&p, &[&b]).unwrap();
        assert_eq!(acc, vec![1, -2, 3, 0]);
    }

    #[test]
    fn accumulate_two_passes_sums_with_shift() {
        // Pass 0 shift = 2 (<<2 = ×4); pass 1 (last) shift 0.
        // cell0: 1<<2 + 10 = 14; cell1: (-3)<<2 + 5 = -7; cell2: 0 + 0 = 0.
        let p = passes(2, vec![2]);
        let b0 = block(vec![1, -3, 0]);
        let b1 = block(vec![10, 5, 0]);
        let acc = accumulate_block_across_passes(&p, &[&b0, &b1]).unwrap();
        // pass0 << 2 (×4) + pass1: (1·4 + 10), (-3·4 + 5), 0.
        assert_eq!(acc, vec![14, -7, 0]);
    }

    #[test]
    fn accumulate_three_passes() {
        // shifts: pass0=3 (×8), pass1=1 (×2), pass2(last)=0.
        let p = passes(3, vec![3, 1]);
        let b0 = block(vec![2, 0]);
        let b1 = block(vec![1, 4]);
        let b2 = block(vec![5, -1]);
        let acc = accumulate_block_across_passes(&p, &[&b0, &b1, &b2]).unwrap();
        // cell0: 2*8 + 1*2 + 5 = 16+2+5 = 23.
        // cell1: 0*8 + 4*2 + (-1) = 0+8-1 = 7.
        assert_eq!(acc, vec![23, 7]);
    }

    #[test]
    fn accumulate_rejects_pass_count_mismatch() {
        let p = passes(2, vec![1]);
        let b = block(vec![0; 4]);
        // Only one block for a 2-pass frame.
        assert!(accumulate_block_across_passes(&p, &[&b]).is_err());
    }

    #[test]
    fn accumulate_rejects_length_mismatch() {
        let p = passes(2, vec![1]);
        let b0 = block(vec![0; 4]);
        let b1 = block(vec![0; 3]);
        assert!(accumulate_block_across_passes(&p, &[&b0, &b1]).is_err());
    }

    #[test]
    fn accumulate_rejects_empty_grid() {
        let p = passes(1, vec![]);
        let b = block(vec![]);
        assert!(accumulate_block_across_passes(&p, &[&b]).is_err());
    }

    #[test]
    fn accumulate_rejects_overflowing_shift() {
        // A malformed Passes carrying shift 40 must reject, not wrap.
        let p = passes(2, vec![40]);
        let b0 = block(vec![1]);
        let b1 = block(vec![0]);
        assert!(accumulate_block_across_passes(&p, &[&b0, &b1]).is_err());
    }

    #[test]
    fn accumulate_rejects_overflowing_shifted_value() {
        // 1 << 30 fits i32, but a large coefficient << a large shift
        // overflows; the i64 guard catches it.
        let p = passes(2, vec![30]);
        let b0 = block(vec![8]); // 8 << 30 = 2^33, overflows i32.
        let b1 = block(vec![0]);
        assert!(accumulate_block_across_passes(&p, &[&b0, &b1]).is_err());
    }

    #[test]
    fn accumulate_rejects_overflowing_sum() {
        // Two passes each contributing near i32::MAX → sum overflows.
        let p = passes(2, vec![0]);
        let b0 = block(vec![i32::MAX]);
        let b1 = block(vec![1]);
        assert!(accumulate_block_across_passes(&p, &[&b0, &b1]).is_err());
    }

    #[test]
    fn accumulate_nonsquare_transform_grid_uniform() {
        // A non-square DCT8×16 grid is 16 × 8 = 128 cells; the cross-pass
        // sum is the same flat cell-wise rule as the square case. Two
        // passes with a distinguishing pattern.
        let p = passes(2, vec![1]);
        let n: i32 = 128;
        let c0: Vec<i32> = (0..n).collect();
        let c1: Vec<i32> = (0..n).map(|i| n - i).collect();
        let b0 = block(c0.clone());
        let b1 = block(c1.clone());
        let acc = accumulate_block_across_passes(&p, &[&b0, &b1]).unwrap();
        for c in 0..n as usize {
            assert_eq!(acc[c], c0[c] * 2 + c1[c], "cell {c}");
        }
    }

    // ---- accumulate_three_channel_multi_pass ----

    fn tcv(
        v: Varblock,
        x: Vec<i32>,
        y: Vec<i32>,
        b: Vec<i32>,
    ) -> crate::block_context_resolver::ThreeChannelVarblock {
        (v, [block(x), block(y), block(b)], [0, 0, 0])
    }

    #[test]
    fn multi_pass_single_pass_copies_through() {
        // One pass, two varblocks, three channels. Output equals input
        // (shift 0).
        let p = passes(1, vec![]);
        let mp = vec![vec![
            tcv(
                vb(0, 0, TransformType::Dct8x8),
                vec![1; 64],
                vec![2; 64],
                vec![3; 64],
            ),
            tcv(
                vb(1, 0, TransformType::Dct8x8),
                vec![4; 64],
                vec![5; 64],
                vec![6; 64],
            ),
        ]];
        let out = accumulate_three_channel_multi_pass(&p, &mp).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0.x, 0);
        assert_eq!(out[0].1[0], vec![1; 64]);
        assert_eq!(out[0].1[1], vec![2; 64]);
        assert_eq!(out[0].1[2], vec![3; 64]);
        assert_eq!(out[1].0.x, 1);
        assert_eq!(out[1].1[2], vec![6; 64]);
    }

    #[test]
    fn multi_pass_two_passes_accumulates_per_channel() {
        // Two passes, shift[0] = 2, one varblock.
        let p = passes(2, vec![2]);
        let mp = vec![
            // pass 0
            vec![tcv(
                vb(0, 0, TransformType::Dct8x8),
                vec![1; 64],
                vec![0; 64],
                vec![3; 64],
            )],
            // pass 1 (last, shift 0)
            vec![tcv(
                vb(0, 0, TransformType::Dct8x8),
                vec![10; 64],
                vec![7; 64],
                vec![1; 64],
            )],
        ];
        let out = accumulate_three_channel_multi_pass(&p, &mp).unwrap();
        assert_eq!(out.len(), 1);
        // X: 1<<2 + 10 = 14; Y: 0<<2 + 7 = 7; B: 3<<2 + 1 = 13.
        assert_eq!(out[0].1[0], vec![14; 64]);
        assert_eq!(out[0].1[1], vec![7; 64]);
        assert_eq!(out[0].1[2], vec![13; 64]);
    }

    #[test]
    fn multi_pass_rejects_pass_count_mismatch() {
        let p = passes(2, vec![1]);
        let mp = vec![vec![tcv(
            vb(0, 0, TransformType::Dct8x8),
            vec![0; 64],
            vec![0; 64],
            vec![0; 64],
        )]];
        assert!(accumulate_three_channel_multi_pass(&p, &mp).is_err());
    }

    #[test]
    fn multi_pass_rejects_varblock_count_mismatch() {
        let p = passes(2, vec![1]);
        let mp = vec![
            vec![tcv(
                vb(0, 0, TransformType::Dct8x8),
                vec![0; 64],
                vec![0; 64],
                vec![0; 64],
            )],
            // pass 1 has zero varblocks → mismatch.
            vec![],
        ];
        assert!(accumulate_three_channel_multi_pass(&p, &mp).is_err());
    }

    #[test]
    fn multi_pass_rejects_inconsistent_varblock_placement() {
        // pass 1 reports a different transform for varblock 0.
        let p = passes(2, vec![1]);
        let mp = vec![
            vec![tcv(
                vb(0, 0, TransformType::Dct8x8),
                vec![0; 64],
                vec![0; 64],
                vec![0; 64],
            )],
            vec![tcv(
                vb(0, 0, TransformType::Dct16x16),
                vec![0; 256],
                vec![0; 256],
                vec![0; 256],
            )],
        ];
        assert!(accumulate_three_channel_multi_pass(&p, &mp).is_err());
    }
}
