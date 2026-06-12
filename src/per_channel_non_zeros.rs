//! Per-channel `NonZeros(x, y)` grid container —
//! ISO/IEC FDIS 18181-1:2021 §C.8.3 + Listing C.13 prelude.
//!
//! ## Scope (round 183)
//!
//! Round 183 lands the typed scaffolding that owns the per-channel
//! array of [`crate::non_zeros_grid::NonZerosGrid`] cells referenced
//! throughout FDIS Listing C.13 / Listing C.14.
//!
//! Listing C.13's `BlockContext()` takes a channel index `c`
//! explicitly (`idx = (c < 2 ? c ^ 1 : 2) × 13 + s`) and the
//! `NonZeros` bookkeeping is described per-channel: each channel
//! tracks its own `NonZeros(x, y)` grid because chroma subsampling +
//! `TransformType` heterogeneity means the per-channel grid shapes
//! differ. The round-177 [`crate::non_zeros_grid::NonZerosGrid`] is
//! the single-channel primitive; round 183 adds the typed container
//! that holds one such grid per channel.
//!
//! The container's per-channel shapes are caller-supplied (so a
//! callsite that already knows e.g. `(width_y, height_y)` from the
//! per-LfGroup varblock-shape grid and the chroma-subsampled
//! `(width_chroma, height_chroma)` from the chroma plane geometry
//! can construct the container in one call). The container does not
//! re-derive the chroma-subsampling math; it only owns storage +
//! per-channel routing.
//!
//! ## Scope boundary
//!
//! The §C.7.2 entropy histogram array, the per-pass `EntropyStream`
//! / `HybridUintState` wiring, the per-LfGroup varblock-shape grid
//! itself, and the per-channel `BlockContext()` history threading
//! remain follow-up work (the `decode_symbol` and `read_non_zeros`
//! closures abstract over them at the per-block level, the per-grid
//! primitive at the per-position level, and now round 183 at the
//! per-channel level — but the next layer is the per-LfGroup grid).
//!
//! ## §C.8.3 prose — per-channel `NonZeros(x, y)` keying
//!
//! From the FDIS prose right before Listing C.13:
//!
//! > For each channel `c` of the pass group, `BlockContext()` is
//! > computed and the resulting `NonZeros(x, y)` grid is owned
//! > per-channel.
//!
//! And from the §C.8.3 paragraph that introduces per-channel
//! coefficient decoding:
//!
//! > The HF coefficients of a varblock are decoded in YCbCr / XYB
//! > order: the Y / Y' channel first, then Cb / X, then Cr / B.
//! > Each channel has its own `BlockContext()` invocation (with `c`
//! > set appropriately) and updates its own `NonZeros(x, y)` cell.
//!
//! That per-channel keying is what this round captures.
//!
//! ## Round-177 round-183 layering
//!
//! Round 177 lifts the single-channel grid; round 183 lifts the
//! per-channel container. Both are pure-control-flow primitives —
//! no bit reads, no spec re-derivation. The same five-layer shape
//! the rest of the round-89 / 95 / 121 / 138 / 141 / 144 / 147 / 159
//! / 164 / 177 cascade follows.

use oxideav_core::{Error, Result};

use crate::dct_select::TransformType;
use crate::non_zeros_grid::{decode_block_at, NonZerosGrid};
use crate::pass_group_hf::DecodedHfBlock;

/// Canonical per-frame channel count for the VarDCT path (YCbCr /
/// XYB: Y / Y', Cb / X, Cr / B).
pub const DEFAULT_NUM_CHANNELS: u32 = 3;

/// Per-channel container of [`NonZerosGrid`]s — the typed wrapper
/// that owns one grid per channel, indexed by `c ∈ [0, num_channels)`.
///
/// Each per-channel grid is independently shaped: the Y / Y' grid is
/// typically larger than the Cb / X and Cr / B grids under chroma
/// subsampling (Listing C.13's `BlockContext()` factors `c` into
/// `(c < 2 ? c ^ 1 : 2) × 13`, so the channel index drives both
/// histogram offset and grid shape).
#[derive(Debug, Clone)]
pub struct PerChannelNonZerosGrids {
    grids: Vec<NonZerosGrid>,
}

impl PerChannelNonZerosGrids {
    /// Build a per-channel container from `(width, height)` pairs,
    /// one per channel. Each pair is validated by
    /// [`NonZerosGrid::new`] (zero or `> 65535` dims are rejected).
    ///
    /// Returns [`Error::InvalidData`] if `dims` is empty (a frame
    /// with zero channels has no useful interpretation) or any pair
    /// fails [`NonZerosGrid::new`].
    pub fn new(dims: &[(u32, u32)]) -> Result<Self> {
        if dims.is_empty() {
            return Err(Error::InvalidData(
                "JXL PerChannelNonZerosGrids: must have at least one channel".into(),
            ));
        }
        let mut grids = Vec::with_capacity(dims.len());
        for &(w, h) in dims {
            grids.push(NonZerosGrid::new(w, h)?);
        }
        Ok(Self { grids })
    }

    /// Convenience builder: every channel shares the same
    /// `(width, height)`. Useful when the caller already knows the
    /// grid is unsubsampled (e.g. 4:4:4 YCbCr or XYB without
    /// chroma subsampling).
    pub fn new_uniform(num_channels: u32, width: u32, height: u32) -> Result<Self> {
        if num_channels == 0 {
            return Err(Error::InvalidData(
                "JXL PerChannelNonZerosGrids: num_channels must be ≥ 1".into(),
            ));
        }
        let dims: Vec<(u32, u32)> = (0..num_channels).map(|_| (width, height)).collect();
        Self::new(&dims)
    }

    /// Number of channels owned by this container.
    pub fn num_channels(&self) -> u32 {
        self.grids.len() as u32
    }

    /// Borrow the `c`-th per-channel grid. Returns
    /// [`Error::InvalidData`] for `c >= num_channels`.
    pub fn grid(&self, c: u32) -> Result<&NonZerosGrid> {
        self.grids.get(c as usize).ok_or_else(|| {
            Error::InvalidData(format!(
                "JXL PerChannelNonZerosGrids::grid: c={c} out of range (num_channels={})",
                self.num_channels()
            ))
        })
    }

    /// Mutably borrow the `c`-th per-channel grid. Returns
    /// [`Error::InvalidData`] for `c >= num_channels`.
    pub fn grid_mut(&mut self, c: u32) -> Result<&mut NonZerosGrid> {
        let num = self.num_channels();
        self.grids.get_mut(c as usize).ok_or_else(|| {
            Error::InvalidData(format!(
                "JXL PerChannelNonZerosGrids::grid_mut: c={c} out of range (num_channels={num})"
            ))
        })
    }

    /// Per-channel `PredictedNonZeros(x, y)` lookup. Delegates to
    /// [`NonZerosGrid::predicted`] on the `c`-th grid.
    pub fn predicted(&self, c: u32, x: u32, y: u32) -> Result<u32> {
        self.grid(c)?.predicted(x, y)
    }

    /// Per-channel `NonZeros(x, y)` read. Delegates to
    /// [`NonZerosGrid::get`] on the `c`-th grid.
    pub fn get(&self, c: u32, x: u32, y: u32) -> Result<u32> {
        self.grid(c)?.get(x, y)
    }

    /// Per-channel `NonZeros(x, y)` write. Delegates to
    /// [`NonZerosGrid::set`] on the `c`-th grid.
    pub fn set(&mut self, c: u32, x: u32, y: u32, value: u32) -> Result<()> {
        self.grid_mut(c)?.set(x, y, value)
    }

    /// Per-channel `(non_zeros + num_blocks - 1) Idiv num_blocks`
    /// update. Delegates to [`NonZerosGrid::update_after_block`] on
    /// the `c`-th grid.
    pub fn update_after_block(
        &mut self,
        c: u32,
        x: u32,
        y: u32,
        non_zeros: u32,
        num_blocks: u32,
    ) -> Result<u32> {
        self.grid_mut(c)?
            .update_after_block(x, y, non_zeros, num_blocks)
    }

    /// Per-channel `TransformType`-driven update. Delegates to
    /// [`NonZerosGrid::update_after_block_for_transform`] on the
    /// `c`-th grid.
    pub fn update_after_block_for_transform(
        &mut self,
        c: u32,
        x: u32,
        y: u32,
        non_zeros: u32,
        t: TransformType,
    ) -> Result<u32> {
        self.grid_mut(c)?
            .update_after_block_for_transform(x, y, non_zeros, t)
    }

    /// Per-channel typed driver: invokes the round-177
    /// [`decode_block_at`] against the `c`-th grid. Mirrors the
    /// single-channel driver's `(DecodedHfBlock, raw_non_zeros)`
    /// return shape.
    ///
    /// The Listing C.13 `BlockContext()` channel-keying is the
    /// caller's responsibility: `c` selects which per-channel grid
    /// the call routes through, but the `block_ctx` value passed in
    /// must already have been computed with the matching `c` (via
    /// [`crate::pass_group_hf::block_context`]). This keeps the
    /// container a pure storage + routing primitive and lets the
    /// per-LfGroup driver own the BlockContext call.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_block_at_for_channel<F, G>(
        &mut self,
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
        let grid = self.grid_mut(c)?;
        decode_block_at(
            grid,
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
    fn new_rejects_empty_dims() {
        let r = PerChannelNonZerosGrids::new(&[]);
        assert!(r.is_err());
    }

    #[test]
    fn new_rejects_zero_dim_in_any_channel() {
        // First channel valid, second has zero width → error.
        let r = PerChannelNonZerosGrids::new(&[(8, 8), (0, 4)]);
        assert!(r.is_err());
        // Likewise zero height.
        let r2 = PerChannelNonZerosGrids::new(&[(8, 0)]);
        assert!(r2.is_err());
    }

    #[test]
    fn new_rejects_oversize_dim_in_any_channel() {
        let big = u16::MAX as u32 + 1;
        let r = PerChannelNonZerosGrids::new(&[(8, 8), (big, 4)]);
        assert!(r.is_err());
    }

    #[test]
    fn new_accepts_three_channels_chroma_subsampled() {
        // Y at 16×16 varblocks, Cb/Cr at 8×8 (4:2:0).
        let p = PerChannelNonZerosGrids::new(&[(16, 16), (8, 8), (8, 8)]).unwrap();
        assert_eq!(p.num_channels(), 3);
        assert_eq!(p.grid(0).unwrap().width(), 16);
        assert_eq!(p.grid(0).unwrap().height(), 16);
        assert_eq!(p.grid(1).unwrap().width(), 8);
        assert_eq!(p.grid(2).unwrap().width(), 8);
    }

    #[test]
    fn new_uniform_three_channels() {
        let p = PerChannelNonZerosGrids::new_uniform(3, 4, 4).unwrap();
        assert_eq!(p.num_channels(), 3);
        for c in 0..3 {
            assert_eq!(p.grid(c).unwrap().width(), 4);
            assert_eq!(p.grid(c).unwrap().height(), 4);
        }
    }

    #[test]
    fn new_uniform_rejects_zero_channels() {
        let r = PerChannelNonZerosGrids::new_uniform(0, 4, 4);
        assert!(r.is_err());
    }

    #[test]
    fn new_uniform_propagates_grid_validation() {
        let r = PerChannelNonZerosGrids::new_uniform(3, 0, 4);
        assert!(r.is_err());
    }

    #[test]
    fn grid_oob_channel_errors() {
        let p = PerChannelNonZerosGrids::new_uniform(3, 4, 4).unwrap();
        assert!(p.grid(3).is_err());
        assert!(p.grid(100).is_err());
    }

    #[test]
    fn grid_mut_oob_channel_errors() {
        let mut p = PerChannelNonZerosGrids::new_uniform(3, 4, 4).unwrap();
        assert!(p.grid_mut(3).is_err());
    }

    #[test]
    fn predicted_origin_is_32_for_every_channel() {
        let p = PerChannelNonZerosGrids::new_uniform(3, 4, 4).unwrap();
        for c in 0..3 {
            assert_eq!(p.predicted(c, 0, 0).unwrap(), 32);
        }
    }

    #[test]
    fn predicted_propagates_per_channel_get() {
        // Seed three channels with different non_zeros values; verify
        // the per-channel predicted lookup picks the matching channel's
        // grid.
        let mut p = PerChannelNonZerosGrids::new_uniform(3, 4, 4).unwrap();
        p.set(0, 0, 0, 10).unwrap();
        p.set(1, 0, 0, 20).unwrap();
        p.set(2, 0, 0, 30).unwrap();
        // At (1, 0): predicted = NonZeros(0, 0) on each channel.
        assert_eq!(p.predicted(0, 1, 0).unwrap(), 10);
        assert_eq!(p.predicted(1, 1, 0).unwrap(), 20);
        assert_eq!(p.predicted(2, 1, 0).unwrap(), 30);
    }

    #[test]
    fn predicted_oob_position_errors() {
        let p = PerChannelNonZerosGrids::new_uniform(3, 4, 4).unwrap();
        assert!(p.predicted(0, 4, 0).is_err());
        assert!(p.predicted(0, 0, 4).is_err());
    }

    #[test]
    fn predicted_oob_channel_errors() {
        let p = PerChannelNonZerosGrids::new_uniform(3, 4, 4).unwrap();
        assert!(p.predicted(3, 0, 0).is_err());
    }

    #[test]
    fn get_set_per_channel_independent() {
        // Each channel's grid is independent — a write to (c=0, 0, 0)
        // does not bleed into (c=1, 0, 0).
        let mut p = PerChannelNonZerosGrids::new_uniform(3, 2, 2).unwrap();
        p.set(0, 0, 0, 7).unwrap();
        assert_eq!(p.get(0, 0, 0).unwrap(), 7);
        assert_eq!(p.get(1, 0, 0).unwrap(), 0);
        assert_eq!(p.get(2, 0, 0).unwrap(), 0);
    }

    #[test]
    fn update_after_block_per_channel_ceiling_divide() {
        // Round-177 grid post-Listing-C.14 formula:
        // (non_zeros + num_blocks - 1) Idiv num_blocks.
        // At num_blocks = 4, non_zeros = 17 → ceil(17/4) = 5.
        let mut p = PerChannelNonZerosGrids::new_uniform(3, 2, 2).unwrap();
        let v = p.update_after_block(1, 0, 0, 17, 4).unwrap();
        assert_eq!(v, 5);
        assert_eq!(p.get(1, 0, 0).unwrap(), 5);
        // Other channels untouched.
        assert_eq!(p.get(0, 0, 0).unwrap(), 0);
        assert_eq!(p.get(2, 0, 0).unwrap(), 0);
    }

    #[test]
    fn update_after_block_for_transform_per_channel() {
        // DCT8×8 has num_blocks = 1; DCT16×16 has num_blocks = 4;
        // DCT32×32 has num_blocks = 16. raw_non_zeros = 17 reduces
        // to {17, 5, 2} respectively. Verify per-channel routing
        // applies the channel-specific TransformType, writing the
        // value to every covered cell per the §C.8.3 "for each block
        // in the current varblock" prose (grids sized for the
        // largest 4×4-cell footprint).
        let mut p = PerChannelNonZerosGrids::new_uniform(3, 4, 4).unwrap();
        let v0 = p
            .update_after_block_for_transform(0, 0, 0, 17, TransformType::Dct8x8)
            .unwrap();
        let v1 = p
            .update_after_block_for_transform(1, 0, 0, 17, TransformType::Dct16x16)
            .unwrap();
        let v2 = p
            .update_after_block_for_transform(2, 0, 0, 17, TransformType::Dct32x32)
            .unwrap();
        assert_eq!(v0, 17, "ceil(17/1) = 17");
        assert_eq!(v1, 5, "ceil(17/4) = 5");
        assert_eq!(v2, 2, "ceil(17/16) = 2");
        // Footprint writeback per channel: channel 0 (DCT8x8) wrote
        // only (0, 0); channel 1 (DCT16x16) the 2×2 footprint;
        // channel 2 (DCT32x32) the full 4×4 footprint.
        assert_eq!(p.get(0, 1, 0).unwrap(), 0, "DCT8x8 footprint is 1×1");
        assert_eq!(p.get(1, 1, 1).unwrap(), 5, "DCT16x16 2×2 footprint");
        assert_eq!(p.get(1, 2, 0).unwrap(), 0, "outside DCT16x16 footprint");
        assert_eq!(p.get(2, 3, 3).unwrap(), 2, "DCT32x32 4×4 footprint");
    }

    #[test]
    fn decode_block_at_for_channel_routes_per_channel() {
        // Drive the typed per-channel driver: a raw_non_zeros = 3
        // call on channel 0 at (0, 0) updates only channel 0's grid;
        // channels 1 and 2 remain zero.
        let mut p = PerChannelNonZerosGrids::new_uniform(3, 2, 2).unwrap();
        let read_non_zeros = |_ctx: u32| -> Result<u32> { Ok(3u32) };
        let decode_symbol = |_ctx: u32| -> Result<u32> { Ok(0u32) };
        let (_decoded, raw_non_zeros) = p
            .decode_block_at_for_channel(
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
        assert_eq!(p.get(0, 0, 0).unwrap(), 3);
        assert_eq!(p.get(1, 0, 0).unwrap(), 0);
        assert_eq!(p.get(2, 0, 0).unwrap(), 0);
    }

    #[test]
    fn decode_block_at_for_channel_oob_channel_errors() {
        let mut p = PerChannelNonZerosGrids::new_uniform(3, 2, 2).unwrap();
        let read_non_zeros = |_ctx: u32| -> Result<u32> { Ok(0u32) };
        let decode_symbol = |_ctx: u32| -> Result<u32> { Ok(0u32) };
        let r = p.decode_block_at_for_channel(
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
    fn decode_block_at_for_channel_dct16x16_ceil_divides() {
        // Per-channel routing must also propagate the TransformType
        // through to the post-update step: raw_non_zeros = 17 with
        // DCT16×16 (num_blocks = 4) stores ceil(17/4) = 5 on every
        // cell of the 2×2 footprint (§C.8.3 prose).
        let mut p = PerChannelNonZerosGrids::new_uniform(3, 2, 2).unwrap();
        let read_non_zeros = |_ctx: u32| -> Result<u32> { Ok(17u32) };
        let decode_symbol = |_ctx: u32| -> Result<u32> { Ok(0u32) };
        let (_decoded, raw_non_zeros) = p
            .decode_block_at_for_channel(
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
            assert_eq!(p.get(1, x, y).unwrap(), 5, "footprint cell ({x},{y})");
        }
        assert_eq!(p.get(0, 0, 0).unwrap(), 0, "channel 0 untouched");
    }

    #[test]
    fn three_channel_raster_walk_independent_evolution() {
        // Drive a 2-step raster walk on three channels with different
        // raw_non_zeros sequences. Verify the per-channel grids evolve
        // independently and that channel-1's prediction at (1, 0)
        // reads back channel-1's (0, 0) cell — not channel-0's.
        let mut p = PerChannelNonZerosGrids::new_uniform(3, 2, 1).unwrap();
        // Step 1 — at (0, 0): channel 0 gets 4, channel 1 gets 12,
        // channel 2 gets 20.
        for (c, nz) in [(0u32, 4u32), (1, 12), (2, 20)] {
            let read_non_zeros = |_ctx: u32| -> Result<u32> { Ok(nz) };
            let decode_symbol = |_ctx: u32| -> Result<u32> { Ok(0u32) };
            let _ = p
                .decode_block_at_for_channel(
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
        // After step 1, predicted at (1, 0) on each channel must read
        // back the channel-specific (0, 0) cell.
        assert_eq!(p.predicted(0, 1, 0).unwrap(), 4);
        assert_eq!(p.predicted(1, 1, 0).unwrap(), 12);
        assert_eq!(p.predicted(2, 1, 0).unwrap(), 20);
    }

    #[test]
    fn default_num_channels_is_three() {
        assert_eq!(DEFAULT_NUM_CHANNELS, 3);
    }

    #[test]
    fn chroma_subsampled_shapes_persist() {
        // Y at 16×16 varblocks, Cb/Cr at 8×8 (4:2:0 conceptually).
        // Verify each channel's grid honours its own dimensions on
        // get / set / OOB.
        let mut p = PerChannelNonZerosGrids::new(&[(16, 16), (8, 8), (8, 8)]).unwrap();
        p.set(0, 15, 15, 50).unwrap();
        assert_eq!(p.get(0, 15, 15).unwrap(), 50);
        // (15, 15) is out-of-range for channels 1 and 2 (only 8×8).
        assert!(p.set(1, 15, 15, 99).is_err());
        assert!(p.get(2, 15, 15).is_err());
        // Inside the chroma grid range works.
        p.set(1, 7, 7, 77).unwrap();
        assert_eq!(p.get(1, 7, 7).unwrap(), 77);
    }

    #[test]
    fn set_oob_channel_errors() {
        let mut p = PerChannelNonZerosGrids::new_uniform(3, 2, 2).unwrap();
        assert!(p.set(3, 0, 0, 5).is_err());
    }

    #[test]
    fn get_oob_channel_errors() {
        let p = PerChannelNonZerosGrids::new_uniform(3, 2, 2).unwrap();
        assert!(p.get(3, 0, 0).is_err());
    }
}
