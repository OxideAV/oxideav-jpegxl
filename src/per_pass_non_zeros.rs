//! Per-pass `NonZeros(x, y)` grid container —
//! ISO/IEC FDIS 18181-1:2021 §C.8.3 + Listing C.13.
//!
//! ## Scope (round 190)
//!
//! Round 190 lands the typed scaffolding that owns the per-pass
//! array of [`crate::per_channel_non_zeros::PerChannelNonZerosGrids`]
//! cells referenced throughout FDIS §C.8.3.
//!
//! A frame's VarDCT path is decoded in `num_passes` ordered passes
//! (declared in the [`crate::frame_header::FrameHeader`]'s
//! [`crate::frame_header::Passes`] field). Each pass scans every
//! `PassGroup` once: §C.8.3 specifies that within a pass, every
//! varblock is decoded in raster order, and each channel of that
//! varblock maintains its own `NonZeros(x, y)` state. Between
//! passes the per-channel `NonZeros(x, y)` bookkeeping is reset
//! because the per-pass histogram is selected by `hfp` from the
//! per-pass `HfPass` array — a different pass uses a different
//! histogram and the prediction recurrence is keyed against the
//! current pass's own coefficient counts, not a previous pass's.
//!
//! Round 183 lifts the per-channel container; round 190 layers the
//! per-pass container above it. Both are pure-control-flow
//! primitives — no bit reads, no spec re-derivation, no histogram
//! materialisation. The next layer above this is the per-LfGroup
//! driver that walks the varblock-shape grid (still follow-up,
//! tracked under the §C.7.2 entropy-histograms gap noted in round
//! 177's #799 spec issue).
//!
//! ## Scope boundary
//!
//! The §C.7.2 entropy histogram array, the per-pass `EntropyStream`
//! / `HybridUintState` wiring, the per-LfGroup varblock-shape grid
//! itself, the per-channel `BlockContext()` history threading, and
//! the per-pass `hfp` selection from the [`crate::hf_pass::HfPass`]
//! array remain follow-up work. The `decode_symbol` and
//! `read_non_zeros` closures abstract over them at the per-block
//! level; rounds 177 / 183 / 190 abstract storage / channel routing
//! / pass routing at the next layers up.
//!
//! ## FDIS prose — per-pass `NonZeros(x, y)` keying
//!
//! From the §C.8.3 paragraph that introduces per-pass coefficient
//! decoding:
//!
//! > For each pass `p ∈ [0, num_passes)` the per-channel
//! > `NonZeros(x, y)` bookkeeping is owned per-pass: a varblock's
//! > prediction recurrence reads its own pass's history, not the
//! > previous pass's. The `hfp` selector at the top of each
//! > `PassGroup` references the per-pass `HfPass` array
//! > (Listing C.13's `BlockContext()` is evaluated against the
//! > per-pass histograms).
//!
//! That per-pass keying is what this round captures.
//!
//! ## Round-177 → round-183 → round-190 layering
//!
//! * Round 177 [`crate::non_zeros_grid::NonZerosGrid`] — single-channel,
//!   single-pass position grid.
//! * Round 183 [`crate::per_channel_non_zeros::PerChannelNonZerosGrids`]
//!   — per-channel container of per-position grids.
//! * Round 190 [`PerPassNonZerosGrids`] (this module) — per-pass
//!   container of per-channel grids.
//!
//! Each layer keeps the same five-rule API surface — `new` /
//! `new_uniform` / per-coordinate `get` / `set` / `predicted` /
//! `update_after_block(_for_transform)` / a typed driver that
//! threads the round-177 [`crate::non_zeros_grid::decode_block_at`]
//! through the routing chain.
//!
//! No bit reads, no spec re-derivation — same pure-control-flow
//! primitive shape as round-89 [`crate::dct_quant_weights`],
//! round-95 [`crate::hf_dequant`], round-121 [`crate::llf_from_lf`],
//! round-138 [`crate::chroma_from_luma`], round-141
//! [`crate::gaborish`], round-144 [`crate::epf`], round-147
//! [`crate::afv::afv_idct`], round-159 / 164
//! [`crate::pass_group_hf`], round-177
//! [`crate::non_zeros_grid`], and round-183
//! [`crate::per_channel_non_zeros`].

use oxideav_core::{Error, Result};

use crate::dct_select::TransformType;
use crate::pass_group_hf::DecodedHfBlock;
use crate::per_channel_non_zeros::PerChannelNonZerosGrids;

/// Per-pass container of [`PerChannelNonZerosGrids`] — the typed
/// wrapper that owns one per-channel container per pass index,
/// indexed by `p ∈ [0, num_passes)`.
///
/// Each per-pass per-channel container is independently shaped — a
/// frame may use different per-channel grid shapes between passes
/// (passes can subset the channel set when only a subset of
/// channels participate in a partial pass), and the per-pass `hfp`
/// selector ensures each pass reads from its own histogram space.
/// Round 190 owns the routing, not the histogram-derived shape:
/// the caller passes per-pass `dims` slices in.
#[derive(Debug, Clone)]
pub struct PerPassNonZerosGrids {
    passes: Vec<PerChannelNonZerosGrids>,
}

impl PerPassNonZerosGrids {
    /// Build a per-pass container from a slice of per-pass per-channel
    /// dimension lists. Each entry is a `&[(u32, u32)]` slice that
    /// gets validated by [`PerChannelNonZerosGrids::new`]
    /// (zero-channel or zero-dim / oversize-dim per-channel entries
    /// rejected).
    ///
    /// Returns [`Error::InvalidData`] if `pass_dims` is empty (a
    /// frame with zero passes has no useful interpretation per FDIS
    /// §C.2 `num_passes ≥ 1`) or any per-pass `PerChannelNonZerosGrids::new`
    /// call fails.
    pub fn new(pass_dims: &[&[(u32, u32)]]) -> Result<Self> {
        if pass_dims.is_empty() {
            return Err(Error::InvalidData(
                "JXL PerPassNonZerosGrids: must have at least one pass".into(),
            ));
        }
        let mut passes = Vec::with_capacity(pass_dims.len());
        for dims in pass_dims {
            passes.push(PerChannelNonZerosGrids::new(dims)?);
        }
        Ok(Self { passes })
    }

    /// Convenience builder: every pass shares the same per-channel
    /// `(width, height)` for every channel. Useful when the caller
    /// already knows the frame has uniform per-pass shapes (the
    /// common case for an unsubsampled VarDCT frame).
    pub fn new_uniform(
        num_passes: u32,
        num_channels: u32,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        if num_passes == 0 {
            return Err(Error::InvalidData(
                "JXL PerPassNonZerosGrids: num_passes must be ≥ 1".into(),
            ));
        }
        let mut passes = Vec::with_capacity(num_passes as usize);
        for _ in 0..num_passes {
            passes.push(PerChannelNonZerosGrids::new_uniform(
                num_channels,
                width,
                height,
            )?);
        }
        Ok(Self { passes })
    }

    /// Number of passes owned by this container.
    pub fn num_passes(&self) -> u32 {
        self.passes.len() as u32
    }

    /// Borrow the `p`-th per-pass container. Returns
    /// [`Error::InvalidData`] for `p >= num_passes`.
    pub fn pass(&self, p: u32) -> Result<&PerChannelNonZerosGrids> {
        self.passes.get(p as usize).ok_or_else(|| {
            Error::InvalidData(format!(
                "JXL PerPassNonZerosGrids::pass: p={p} out of range (num_passes={})",
                self.num_passes()
            ))
        })
    }

    /// Mutably borrow the `p`-th per-pass container. Returns
    /// [`Error::InvalidData`] for `p >= num_passes`.
    pub fn pass_mut(&mut self, p: u32) -> Result<&mut PerChannelNonZerosGrids> {
        let num = self.num_passes();
        self.passes.get_mut(p as usize).ok_or_else(|| {
            Error::InvalidData(format!(
                "JXL PerPassNonZerosGrids::pass_mut: p={p} out of range (num_passes={num})"
            ))
        })
    }

    /// Per-pass per-channel `PredictedNonZeros(x, y)` lookup.
    /// Delegates to [`PerChannelNonZerosGrids::predicted`] on the
    /// `p`-th pass.
    pub fn predicted(&self, p: u32, c: u32, x: u32, y: u32) -> Result<u32> {
        self.pass(p)?.predicted(c, x, y)
    }

    /// Per-pass per-channel `NonZeros(x, y)` read. Delegates to
    /// [`PerChannelNonZerosGrids::get`] on the `p`-th pass.
    pub fn get(&self, p: u32, c: u32, x: u32, y: u32) -> Result<u32> {
        self.pass(p)?.get(c, x, y)
    }

    /// Per-pass per-channel `NonZeros(x, y)` write. Delegates to
    /// [`PerChannelNonZerosGrids::set`] on the `p`-th pass.
    pub fn set(&mut self, p: u32, c: u32, x: u32, y: u32, value: u32) -> Result<()> {
        self.pass_mut(p)?.set(c, x, y, value)
    }

    /// Per-pass per-channel `(non_zeros + num_blocks - 1) Idiv
    /// num_blocks` update. Delegates to
    /// [`PerChannelNonZerosGrids::update_after_block`] on the `p`-th
    /// pass.
    #[allow(clippy::too_many_arguments)]
    pub fn update_after_block(
        &mut self,
        p: u32,
        c: u32,
        x: u32,
        y: u32,
        non_zeros: u32,
        num_blocks: u32,
    ) -> Result<u32> {
        self.pass_mut(p)?
            .update_after_block(c, x, y, non_zeros, num_blocks)
    }

    /// Per-pass per-channel `TransformType`-driven update. Delegates
    /// to [`PerChannelNonZerosGrids::update_after_block_for_transform`]
    /// on the `p`-th pass.
    #[allow(clippy::too_many_arguments)]
    pub fn update_after_block_for_transform(
        &mut self,
        p: u32,
        c: u32,
        x: u32,
        y: u32,
        non_zeros: u32,
        t: TransformType,
    ) -> Result<u32> {
        self.pass_mut(p)?
            .update_after_block_for_transform(c, x, y, non_zeros, t)
    }

    /// Per-pass per-channel typed driver: invokes the round-183
    /// [`PerChannelNonZerosGrids::decode_block_at_for_channel`]
    /// against the `p`-th pass's per-channel container. Mirrors the
    /// `(DecodedHfBlock, raw_non_zeros)` return shape of the lower
    /// layers.
    ///
    /// The Listing C.13 `BlockContext()` channel-keying and the
    /// per-pass `hfp` histogram selection are the caller's
    /// responsibility: `p` selects which per-pass container the call
    /// routes through, `c` selects the channel inside it, but the
    /// `block_ctx` value passed in must already have been computed
    /// with the matching `c` (via [`crate::pass_group_hf::block_context`])
    /// and the `read_non_zeros` / `decode_symbol` closures must
    /// already be bound to the per-pass `hfp`-selected histograms.
    /// This keeps the container a pure storage + routing primitive
    /// and lets the per-LfGroup driver own the histogram + context
    /// derivation.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_block_at_for_pass_channel<F, G>(
        &mut self,
        p: u32,
        c: u32,
        x: u32,
        y: u32,
        t: TransformType,
        block_ctx: u32,
        nb_block_ctx: u32,
        read_non_zeros: F,
        decode_symbol: G,
    ) -> Result<(DecodedHfBlock, u32)>
    where
        F: FnMut(u32) -> Result<u32>,
        G: FnMut(u32) -> Result<u32>,
    {
        let pass = self.pass_mut(p)?;
        pass.decode_block_at_for_channel(
            c,
            x,
            y,
            t,
            block_ctx,
            nb_block_ctx,
            read_non_zeros,
            decode_symbol,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_empty_pass_list() {
        let r = PerPassNonZerosGrids::new(&[]);
        assert!(r.is_err());
    }

    #[test]
    fn new_propagates_per_channel_validation() {
        // Empty per-channel dims slice on the first pass → error.
        let empty: &[(u32, u32)] = &[];
        let r = PerPassNonZerosGrids::new(&[empty]);
        assert!(r.is_err());
        // Zero-dim on any pass → error.
        let bad: &[(u32, u32)] = &[(0, 8)];
        let r2 = PerPassNonZerosGrids::new(&[bad]);
        assert!(r2.is_err());
    }

    #[test]
    fn new_accepts_two_passes_chroma_subsampled() {
        // 4:2:0-ish: pass 0 at (Y 16×16, Cb 8×8, Cr 8×8);
        // pass 1 with the same shape.
        let pass0: [(u32, u32); 3] = [(16, 16), (8, 8), (8, 8)];
        let pass1: [(u32, u32); 3] = [(16, 16), (8, 8), (8, 8)];
        let p = PerPassNonZerosGrids::new(&[&pass0[..], &pass1[..]]).unwrap();
        assert_eq!(p.num_passes(), 2);
        assert_eq!(p.pass(0).unwrap().num_channels(), 3);
        assert_eq!(p.pass(0).unwrap().grid(0).unwrap().width(), 16);
        assert_eq!(p.pass(1).unwrap().grid(2).unwrap().width(), 8);
    }

    #[test]
    fn new_accepts_per_pass_different_channel_counts() {
        // Pass 0 is a "DC-only" preview with one channel; pass 1 is
        // the full three-channel pass. The per-pass storage tolerates
        // ragged per-pass shapes — driver is responsible for the
        // semantic choice, not the container.
        let pass0: [(u32, u32); 1] = [(8, 8)];
        let pass1: [(u32, u32); 3] = [(8, 8), (8, 8), (8, 8)];
        let p = PerPassNonZerosGrids::new(&[&pass0[..], &pass1[..]]).unwrap();
        assert_eq!(p.pass(0).unwrap().num_channels(), 1);
        assert_eq!(p.pass(1).unwrap().num_channels(), 3);
    }

    #[test]
    fn new_uniform_two_passes_three_channels() {
        let p = PerPassNonZerosGrids::new_uniform(2, 3, 8, 8).unwrap();
        assert_eq!(p.num_passes(), 2);
        for pp in 0..p.num_passes() {
            assert_eq!(p.pass(pp).unwrap().num_channels(), 3);
            for c in 0..3 {
                assert_eq!(p.pass(pp).unwrap().grid(c).unwrap().width(), 8);
            }
        }
    }

    #[test]
    fn new_uniform_rejects_zero_passes() {
        let r = PerPassNonZerosGrids::new_uniform(0, 3, 8, 8);
        assert!(r.is_err());
    }

    #[test]
    fn new_uniform_propagates_per_channel_validation() {
        // zero channels per pass → per-channel layer rejects.
        let r = PerPassNonZerosGrids::new_uniform(2, 0, 8, 8);
        assert!(r.is_err());
        // zero width on a pass → per-grid layer rejects.
        let r2 = PerPassNonZerosGrids::new_uniform(2, 3, 0, 8);
        assert!(r2.is_err());
    }

    #[test]
    fn pass_oob_errors() {
        let p = PerPassNonZerosGrids::new_uniform(2, 3, 4, 4).unwrap();
        assert!(p.pass(2).is_err());
        assert!(p.pass(100).is_err());
    }

    #[test]
    fn pass_mut_oob_errors() {
        let mut p = PerPassNonZerosGrids::new_uniform(2, 3, 4, 4).unwrap();
        assert!(p.pass_mut(2).is_err());
    }

    #[test]
    fn predicted_origin_is_32_for_every_pass_and_channel() {
        let p = PerPassNonZerosGrids::new_uniform(2, 3, 4, 4).unwrap();
        for pp in 0..p.num_passes() {
            for c in 0..p.pass(pp).unwrap().num_channels() {
                assert_eq!(p.predicted(pp, c, 0, 0).unwrap(), 32);
            }
        }
    }

    #[test]
    fn predicted_propagates_per_pass_per_channel_get() {
        // Seed two passes × three channels with distinct values;
        // verify per-pass per-channel predicted lookup reads back the
        // matching cell.
        let mut p = PerPassNonZerosGrids::new_uniform(2, 3, 4, 4).unwrap();
        // Pass 0 seeds.
        p.set(0, 0, 0, 0, 10).unwrap();
        p.set(0, 1, 0, 0, 20).unwrap();
        p.set(0, 2, 0, 0, 30).unwrap();
        // Pass 1 seeds (distinct).
        p.set(1, 0, 0, 0, 11).unwrap();
        p.set(1, 1, 0, 0, 22).unwrap();
        p.set(1, 2, 0, 0, 33).unwrap();
        // At (1, 0) on pass 0: predicted = NonZeros(0, 0) per channel.
        assert_eq!(p.predicted(0, 0, 1, 0).unwrap(), 10);
        assert_eq!(p.predicted(0, 1, 1, 0).unwrap(), 20);
        assert_eq!(p.predicted(0, 2, 1, 0).unwrap(), 30);
        // Pass 1 reads its own history — NOT pass 0's.
        assert_eq!(p.predicted(1, 0, 1, 0).unwrap(), 11);
        assert_eq!(p.predicted(1, 1, 1, 0).unwrap(), 22);
        assert_eq!(p.predicted(1, 2, 1, 0).unwrap(), 33);
    }

    #[test]
    fn predicted_oob_pass_errors() {
        let p = PerPassNonZerosGrids::new_uniform(2, 3, 4, 4).unwrap();
        assert!(p.predicted(2, 0, 0, 0).is_err());
    }

    #[test]
    fn predicted_oob_channel_errors() {
        let p = PerPassNonZerosGrids::new_uniform(2, 3, 4, 4).unwrap();
        assert!(p.predicted(0, 3, 0, 0).is_err());
    }

    #[test]
    fn predicted_oob_position_errors() {
        let p = PerPassNonZerosGrids::new_uniform(2, 3, 4, 4).unwrap();
        assert!(p.predicted(0, 0, 4, 0).is_err());
        assert!(p.predicted(0, 0, 0, 4).is_err());
    }

    #[test]
    fn get_set_per_pass_independent() {
        // A write to pass 0 must not leak into pass 1.
        let mut p = PerPassNonZerosGrids::new_uniform(2, 3, 2, 2).unwrap();
        p.set(0, 0, 0, 0, 7).unwrap();
        assert_eq!(p.get(0, 0, 0, 0).unwrap(), 7);
        // Same coordinates on pass 1 — still zero.
        assert_eq!(p.get(1, 0, 0, 0).unwrap(), 0);
        // Other channels on pass 0 untouched.
        assert_eq!(p.get(0, 1, 0, 0).unwrap(), 0);
        assert_eq!(p.get(0, 2, 0, 0).unwrap(), 0);
    }

    #[test]
    fn set_oob_pass_errors() {
        let mut p = PerPassNonZerosGrids::new_uniform(2, 3, 2, 2).unwrap();
        assert!(p.set(2, 0, 0, 0, 5).is_err());
    }

    #[test]
    fn get_oob_pass_errors() {
        let p = PerPassNonZerosGrids::new_uniform(2, 3, 2, 2).unwrap();
        assert!(p.get(2, 0, 0, 0).is_err());
    }

    #[test]
    fn update_after_block_per_pass_independent() {
        // num_blocks = 4, non_zeros = 17 → ceil(17/4) = 5 on the
        // updated pass / channel; everything else untouched.
        let mut p = PerPassNonZerosGrids::new_uniform(2, 3, 2, 2).unwrap();
        let v = p.update_after_block(1, 1, 0, 0, 17, 4).unwrap();
        assert_eq!(v, 5);
        assert_eq!(p.get(1, 1, 0, 0).unwrap(), 5);
        // Other pass / channel: zero.
        assert_eq!(p.get(0, 1, 0, 0).unwrap(), 0);
        assert_eq!(p.get(1, 0, 0, 0).unwrap(), 0);
        assert_eq!(p.get(1, 2, 0, 0).unwrap(), 0);
    }

    #[test]
    fn update_after_block_for_transform_per_pass() {
        // DCT8×8 / DCT16×16 / DCT32×32 → num_blocks {1, 4, 16}.
        // raw_non_zeros = 17 reduces to {17, 5, 2}. Verify per-pass
        // routing applies the requested TransformType, writing the
        // value to every covered cell per the §C.8.3 "for each block
        // in the current varblock" prose (grids sized for the
        // largest 4×4-cell footprint).
        let mut p = PerPassNonZerosGrids::new_uniform(3, 3, 4, 4).unwrap();
        let v0 = p
            .update_after_block_for_transform(0, 0, 0, 0, 17, TransformType::Dct8x8)
            .unwrap();
        let v1 = p
            .update_after_block_for_transform(1, 1, 0, 0, 17, TransformType::Dct16x16)
            .unwrap();
        let v2 = p
            .update_after_block_for_transform(2, 2, 0, 0, 17, TransformType::Dct32x32)
            .unwrap();
        assert_eq!(v0, 17, "ceil(17/1) = 17");
        assert_eq!(v1, 5, "ceil(17/4) = 5");
        assert_eq!(v2, 2, "ceil(17/16) = 2");
        // Footprint writeback per (pass, channel): DCT16x16 wrote the
        // 2×2 footprint on (pass 1, channel 1); DCT32x32 the full 4×4
        // footprint on (pass 2, channel 2); cross-pass isolation
        // intact.
        assert_eq!(p.get(1, 1, 1, 1).unwrap(), 5, "DCT16x16 2×2 footprint");
        assert_eq!(p.get(2, 2, 3, 3).unwrap(), 2, "DCT32x32 4×4 footprint");
        assert_eq!(p.get(0, 0, 1, 1).unwrap(), 0, "pass 0 only wrote (0,0)");
    }

    #[test]
    fn decode_block_at_for_pass_channel_routes_correctly() {
        // Drive the typed per-pass per-channel driver: a raw_non_zeros
        // = 3 call on (pass 1, channel 0) at (0, 0) must update only
        // pass 1's channel 0 grid; everything else stays zero.
        let mut p = PerPassNonZerosGrids::new_uniform(2, 3, 2, 2).unwrap();
        let read_non_zeros = |_ctx: u32| -> Result<u32> { Ok(3u32) };
        let decode_symbol = |_ctx: u32| -> Result<u32> { Ok(0u32) };
        let (_decoded, raw_non_zeros) = p
            .decode_block_at_for_pass_channel(
                1,
                0,
                0,
                0,
                TransformType::Dct8x8,
                0,
                1,
                read_non_zeros,
                decode_symbol,
            )
            .unwrap();
        assert_eq!(raw_non_zeros, 3);
        assert_eq!(p.get(1, 0, 0, 0).unwrap(), 3);
        // Other pass / channel: still zero.
        assert_eq!(p.get(0, 0, 0, 0).unwrap(), 0);
        assert_eq!(p.get(1, 1, 0, 0).unwrap(), 0);
        assert_eq!(p.get(1, 2, 0, 0).unwrap(), 0);
    }

    #[test]
    fn decode_block_at_for_pass_channel_oob_pass_errors() {
        let mut p = PerPassNonZerosGrids::new_uniform(2, 3, 2, 2).unwrap();
        let read_non_zeros = |_ctx: u32| -> Result<u32> { Ok(0u32) };
        let decode_symbol = |_ctx: u32| -> Result<u32> { Ok(0u32) };
        let r = p.decode_block_at_for_pass_channel(
            2, // out of range
            0,
            0,
            0,
            TransformType::Dct8x8,
            0,
            1,
            read_non_zeros,
            decode_symbol,
        );
        assert!(r.is_err());
    }

    #[test]
    fn decode_block_at_for_pass_channel_oob_channel_errors() {
        let mut p = PerPassNonZerosGrids::new_uniform(2, 3, 2, 2).unwrap();
        let read_non_zeros = |_ctx: u32| -> Result<u32> { Ok(0u32) };
        let decode_symbol = |_ctx: u32| -> Result<u32> { Ok(0u32) };
        let r = p.decode_block_at_for_pass_channel(
            0,
            3, // out of range
            0,
            0,
            TransformType::Dct8x8,
            0,
            1,
            read_non_zeros,
            decode_symbol,
        );
        assert!(r.is_err());
    }

    #[test]
    fn decode_block_at_for_pass_channel_dct16x16_ceil_divides() {
        // Per-pass routing must propagate the TransformType through
        // to the post-update step: raw_non_zeros = 17 with DCT16×16
        // (num_blocks = 4) stores ceil(17/4) = 5 on every cell of
        // the 2×2 footprint (§C.8.3 prose).
        let mut p = PerPassNonZerosGrids::new_uniform(2, 3, 2, 2).unwrap();
        let read_non_zeros = |_ctx: u32| -> Result<u32> { Ok(17u32) };
        let decode_symbol = |_ctx: u32| -> Result<u32> { Ok(0u32) };
        let (_decoded, raw_non_zeros) = p
            .decode_block_at_for_pass_channel(
                0,
                1,
                0,
                0,
                TransformType::Dct16x16,
                0,
                1,
                read_non_zeros,
                decode_symbol,
            )
            .unwrap();
        assert_eq!(raw_non_zeros, 17);
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            assert_eq!(p.get(0, 1, x, y).unwrap(), 5, "footprint cell ({x},{y})");
        }
        assert_eq!(p.get(1, 1, 0, 0).unwrap(), 0, "pass 1 untouched");
    }

    #[test]
    fn two_pass_three_channel_raster_walk_independent_evolution() {
        // Decode raw_non_zeros = {7, 9, 11} at (0, 0) on pass 0 across
        // its three channels, then a different sequence
        // {2, 4, 6} on pass 1 at the same coordinates. Verify the
        // per-pass per-channel grids evolve independently — pass 1's
        // predict at (1, 0) reads back pass 1's history, not pass 0's.
        let mut p = PerPassNonZerosGrids::new_uniform(2, 3, 2, 1).unwrap();
        // Pass 0 step.
        for (c, nz) in [(0u32, 7u32), (1, 9), (2, 11)] {
            let read_non_zeros = |_ctx: u32| -> Result<u32> { Ok(nz) };
            let decode_symbol = |_ctx: u32| -> Result<u32> { Ok(0u32) };
            let _ = p
                .decode_block_at_for_pass_channel(
                    0,
                    c,
                    0,
                    0,
                    TransformType::Dct8x8,
                    0,
                    1,
                    read_non_zeros,
                    decode_symbol,
                )
                .unwrap();
        }
        // Pass 1 step.
        for (c, nz) in [(0u32, 2u32), (1, 4), (2, 6)] {
            let read_non_zeros = |_ctx: u32| -> Result<u32> { Ok(nz) };
            let decode_symbol = |_ctx: u32| -> Result<u32> { Ok(0u32) };
            let _ = p
                .decode_block_at_for_pass_channel(
                    1,
                    c,
                    0,
                    0,
                    TransformType::Dct8x8,
                    0,
                    1,
                    read_non_zeros,
                    decode_symbol,
                )
                .unwrap();
        }
        // Pass 0's predicted at (1, 0) reads pass 0's own (0, 0).
        assert_eq!(p.predicted(0, 0, 1, 0).unwrap(), 7);
        assert_eq!(p.predicted(0, 1, 1, 0).unwrap(), 9);
        assert_eq!(p.predicted(0, 2, 1, 0).unwrap(), 11);
        // Pass 1's predicted reads pass 1's own (0, 0).
        assert_eq!(p.predicted(1, 0, 1, 0).unwrap(), 2);
        assert_eq!(p.predicted(1, 1, 1, 0).unwrap(), 4);
        assert_eq!(p.predicted(1, 2, 1, 0).unwrap(), 6);
    }

    #[test]
    fn per_pass_chroma_subsampled_shapes_persist() {
        // Each pass owns chroma-subsampled per-channel grids: Y at
        // 16×16, Cb/Cr at 8×8. Verify per-pass channel-shape
        // independence.
        let pass0: [(u32, u32); 3] = [(16, 16), (8, 8), (8, 8)];
        let pass1: [(u32, u32); 3] = [(16, 16), (8, 8), (8, 8)];
        let mut p = PerPassNonZerosGrids::new(&[&pass0[..], &pass1[..]]).unwrap();
        // Write to a Y-grid cell only valid for pass 0's channel 0.
        p.set(0, 0, 15, 15, 99).unwrap();
        assert_eq!(p.get(0, 0, 15, 15).unwrap(), 99);
        // Pass 0 channel 1's (15, 15) is OOB — chroma only 8×8.
        assert!(p.set(0, 1, 15, 15, 50).is_err());
        // Same as above, but on pass 1.
        assert!(p.set(1, 2, 15, 15, 50).is_err());
        // Inside chroma range works on pass 1.
        p.set(1, 1, 7, 7, 77).unwrap();
        assert_eq!(p.get(1, 1, 7, 7).unwrap(), 77);
    }

    #[test]
    fn update_after_block_oob_pass_errors() {
        let mut p = PerPassNonZerosGrids::new_uniform(2, 3, 2, 2).unwrap();
        // Out-of-range pass.
        assert!(p.update_after_block(2, 0, 0, 0, 5, 1).is_err());
        // Out-of-range channel propagates from the lower layer.
        assert!(p.update_after_block(0, 3, 0, 0, 5, 1).is_err());
    }

    #[test]
    fn pass_independent_default_three_pass_dct8x8_chain() {
        // Worked example: three passes × three channels at DCT8×8
        // (num_blocks = 1, identity update). Decode raw_non_zeros = 5
        // at (0, 0) on each (pass, channel = 0) and verify pass-1 and
        // pass-2 predict at (1, 0) read back the value 5 written on
        // their own pass and not the others'.
        let mut p = PerPassNonZerosGrids::new_uniform(3, 3, 2, 1).unwrap();
        for pp in 0..3 {
            let read_non_zeros = |_ctx: u32| -> Result<u32> { Ok(5u32) };
            let decode_symbol = |_ctx: u32| -> Result<u32> { Ok(0u32) };
            let _ = p
                .decode_block_at_for_pass_channel(
                    pp,
                    0,
                    0,
                    0,
                    TransformType::Dct8x8,
                    0,
                    1,
                    read_non_zeros,
                    decode_symbol,
                )
                .unwrap();
        }
        // Each pass independently chains.
        for pp in 0..3 {
            assert_eq!(p.predicted(pp, 0, 1, 0).unwrap(), 5);
        }
        // And the other two channels on every pass are still default.
        for pp in 0..3 {
            assert_eq!(p.predicted(pp, 1, 0, 0).unwrap(), 32);
            assert_eq!(p.predicted(pp, 2, 0, 0).unwrap(), 32);
        }
    }

    #[test]
    fn update_does_not_overflow_at_u32_max_via_pass_route() {
        // The lower layer's saturating_add path must remain available
        // through the per-pass route — pathological raw_non_zeros =
        // u32::MAX with num_blocks = 1 stores u32::MAX without panic.
        let mut p = PerPassNonZerosGrids::new_uniform(2, 3, 1, 1).unwrap();
        let v = p.update_after_block(0, 0, 0, 0, u32::MAX, 1).unwrap();
        assert_eq!(v, u32::MAX);
    }
}
