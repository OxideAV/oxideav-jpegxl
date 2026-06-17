//! Per-LfGroup VarDCT residual-plane assembly — places each varblock's
//! `R × C` spatial residual block (the output of the Annex F.3 dequant +
//! Annex I.2.3 inverse-DCT stage) into a single-channel spatial plane at
//! the varblock's pixel origin.
//!
//! ## Scope (round 306)
//!
//! [`crate::block_dequant::decode_block_to_residual`] (rounds 286 / 293 /
//! 300) is the per-block decode walk: given a decoded quantised block, a
//! [`TransformType`], a channel, and the per-varblock `HfMul`, it
//! dequantises every coefficient (F.3) and runs the inverse transform
//! (I.2.3.2) to produce the block's spatial residual samples — an
//! `R × C` row-major buffer where `(R, C)` is the transform's pixel
//! shape. That primitive is now complete for **every** [`TransformType`].
//!
//! What was still missing — and what this module adds — is the **spatial
//! placement** stage directly above it: a single channel's varblocks are
//! laid out across the LfGroup by the [`DctSelectGrid`] (one top-left
//! cell per varblock, in 8×8-block grid units), and each varblock's
//! decoded residual block must be written into the channel's spatial
//! plane at the pixel origin `(bx * 8, by * 8)` of its top-left cell.
//!
//! This module is the pure-geometry composition layer between
//! [`crate::block_dequant`] (which produces one residual block) and the
//! restoration filters — chroma-from-luma (Annex G), Gaborish (Annex
//! J.2), and EPF (Annex J.3) — which all consume an **assembled
//! per-channel plane**.
//!
//! ## Plane geometry (padded block grid)
//!
//! The [`DctSelectGrid`] has `width_blocks × height_blocks` cells where
//! `width_blocks = ceil(lf_w / 8)` and `height_blocks = ceil(lf_h / 8)`
//! (per §C.5.4). A varblock whose footprint reaches the right or bottom
//! edge of the grid therefore covers pixels past the LfGroup's actual
//! `lf_w × lf_h` extent. The decoder reconstructs into a buffer sized to
//! the **padded block grid** and crops to `lf_w × lf_h` as a final step.
//!
//! To keep this primitive free of any crop ambiguity, the assembled
//! plane is exactly the padded block grid:
//!
//! * `plane_width  = width_blocks  * 8`
//! * `plane_height = height_blocks * 8`
//!
//! Every covered varblock's residual block fits inside this plane by
//! construction (`derive_dct_select` already rejects a varblock whose
//! footprint spills past the grid), so placement is total and
//! unconditional — no per-edge clamping. The caller crops the assembled
//! plane to `lf_w × lf_h` afterwards.
//!
//! ## Block / pixel geometry invariant
//!
//! For a varblock with transform `t`:
//!
//! * [`crate::dct_select::TransformType::block_dims`] gives the footprint
//!   `(bcols, brows)` in 8×8-cell units;
//! * [`crate::idct::dct_pixel_dims`] (DCT family) or
//!   [`crate::idct::non_dct_pixel_dims`] (non-DCT family) gives the pixel
//!   shape `(R, C) = (rows, cols)`.
//!
//! These are consistent: `C == bcols * 8` and `R == brows * 8` for every
//! [`TransformType`]. The residual block produced by
//! [`crate::block_dequant::decode_block_to_residual`] is `R × C`
//! row-major (`block[r * C + c]`), and it is written to plane pixel
//! `(px = bx * 8 + c, py = by * 8 + r)` — i.e.
//! `plane[py * plane_width + px] = block[r * C + c]`.
//!
//! ## Not in scope
//!
//! The residual produced by the IDCT already carries the LLF (DC
//! subband) contribution at its prefix cells — the LLF is loaded into
//! the top-left coefficients of the block *before* the inverse transform
//! (§I.2.5 + the per-block decode walk), so no separate DC add happens
//! at placement time. Chroma-from-luma, Gaborish, and EPF run on the
//! assembled plane and remain caller-side concerns above this primitive.
//! This module performs **no** bit reads, **no** spec re-derivation, and
//! **no** histogram materialisation.

use crate::block_dequant::covered_grid_dims;
use crate::chroma_from_luma::apply_hf_plane_inplace;
use crate::dct_select::{DctSelectGrid, TransformType};
use crate::idct::{dct_pixel_dims, non_dct_pixel_dims};
use crate::lf_dequant::LfDequantOutput;
use crate::lf_global::LfChannelCorrelation;
use crate::varblock_walk::{Varblock, VarblockWalk};
use crate::vardct::compose_lf_to_llf_block;
use oxideav_core::{Error, Result};

/// A single-channel spatial residual plane sized to the padded block
/// grid of a [`DctSelectGrid`].
///
/// `samples` is row-major `width × height` `f32`; `samples[y * width +
/// x]` is the residual at pixel `(x, y)`. Both dimensions are multiples
/// of 8 (`width = width_blocks * 8`, `height = height_blocks * 8`).
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualPlane {
    /// Plane width in pixels (`= grid.width_blocks * 8`).
    pub width: usize,
    /// Plane height in pixels (`= grid.height_blocks * 8`).
    pub height: usize,
    /// Row-major residual samples, length `width * height`.
    pub samples: Vec<f32>,
}

impl ResidualPlane {
    /// Allocate a zero-filled plane sized to the padded block grid of
    /// `grid`.
    ///
    /// Errors with [`Error::InvalidData`] if `width_blocks * height_blocks
    /// * 64` overflows `usize`.
    pub fn for_grid(grid: &DctSelectGrid) -> Result<Self> {
        let width = (grid.width_blocks as usize)
            .checked_mul(8)
            .ok_or_else(|| Error::InvalidData("JXL residual plane: width overflow".into()))?;
        let height = (grid.height_blocks as usize)
            .checked_mul(8)
            .ok_or_else(|| Error::InvalidData("JXL residual plane: height overflow".into()))?;
        let n = width
            .checked_mul(height)
            .ok_or_else(|| Error::InvalidData("JXL residual plane: area overflow".into()))?;
        Ok(Self {
            width,
            height,
            samples: vec![0.0f32; n],
        })
    }

    /// Read the residual at pixel `(x, y)`, or `None` if out of range.
    pub fn get(&self, x: usize, y: usize) -> Option<f32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(self.samples[y * self.width + x])
    }

    /// Crop this plane to a `target_width × target_height` extent by
    /// keeping the top-left rectangle and discarding the right and bottom
    /// padding — the §6.2 "cropping at the right and bottom as necessary"
    /// step that turns the padded block-grid reconstruction into the
    /// logical channel extent.
    ///
    /// The padded plane has dimensions `width_blocks·8 × height_blocks·8`
    /// where `width_blocks = ceil(target_width / 8)` and
    /// `height_blocks = ceil(target_height / 8)` (§C.5.4), so a valid
    /// target is at most this plane's dimensions and at most 7 pixels
    /// smaller along each axis. Both bounds are enforced.
    ///
    /// Per §6.2 the decoder "ensures the decoded image has the dimensions
    /// specified in SizeHeader by cropping at the right and bottom" — the
    /// retained samples are exactly `self.get(x, y)` for `x < target_width`
    /// and `y < target_height`; no resampling, averaging, or edge handling
    /// is performed.
    ///
    /// Errors with [`Error::InvalidData`] if either target dimension is
    /// zero or exceeds this plane's corresponding dimension (a crop can
    /// only shrink — never grow — a padded reconstruction).
    pub fn crop_to(&self, target_width: usize, target_height: usize) -> Result<ResidualPlane> {
        if target_width == 0 || target_height == 0 {
            return Err(Error::InvalidData(format!(
                "JXL residual plane crop: zero target dimension \
                 ({target_width}×{target_height})"
            )));
        }
        if target_width > self.width || target_height > self.height {
            return Err(Error::InvalidData(format!(
                "JXL residual plane crop: target {target_width}×{target_height} \
                 exceeds padded plane {}×{} (crop only shrinks)",
                self.width, self.height
            )));
        }
        // Common case: target already matches the plane (LfGroup whose
        // logical extent is an exact multiple of 8). Avoid the copy.
        if target_width == self.width && target_height == self.height {
            return Ok(self.clone());
        }
        let mut out = Vec::with_capacity(target_width * target_height);
        for y in 0..target_height {
            let row_start = y * self.width;
            out.extend_from_slice(&self.samples[row_start..row_start + target_width]);
        }
        Ok(ResidualPlane {
            width: target_width,
            height: target_height,
            samples: out,
        })
    }
}

/// Pixel shape `(R, C) = (rows, cols)` of a transform's residual block,
/// from either the plain-DCT or the non-DCT pixel-dims table.
///
/// Returns `None` only if a future [`TransformType`] lacks a pixel-dims
/// mapping in both tables (no current variant does).
pub fn block_pixel_dims(t: TransformType) -> Option<(usize, usize)> {
    dct_pixel_dims(t).or_else(|| non_dct_pixel_dims(t))
}

/// Write one `R × C` row-major residual block into `plane` at the pixel
/// origin of varblock `vb` (`px = vb.x * 8`, `py = vb.y * 8`).
///
/// `block` must have length `R * C` where `(R, C) = block_pixel_dims(t)`
/// for `vb.transform`. The block is copied verbatim:
/// `plane[(py + r) * width + (px + c)] = block[r * C + c]`.
///
/// Errors:
/// * [`Error::Unsupported`] if `vb.transform` has no pixel-dims mapping.
/// * [`Error::InvalidData`] if `block.len() != R * C`, or if the block's
///   footprint at `(px, py)` would extend past the plane bounds (a
///   malformed grid / plane mismatch — a grid from `derive_dct_select`
///   plus a plane from [`ResidualPlane::for_grid`] never trips this).
pub fn place_block(plane: &mut ResidualPlane, vb: &Varblock, block: &[f32]) -> Result<()> {
    let (rows, cols) = block_pixel_dims(vb.transform).ok_or_else(|| {
        Error::Unsupported(format!(
            "JXL residual plane: {:?} has no pixel-dims mapping",
            vb.transform
        ))
    })?;
    if block.len() != rows * cols {
        return Err(Error::InvalidData(format!(
            "JXL residual plane: block length {} != R * C ({rows} * {cols} = {}) for {:?}",
            block.len(),
            rows * cols,
            vb.transform
        )));
    }
    let px = (vb.x as usize)
        .checked_mul(8)
        .ok_or_else(|| Error::InvalidData("JXL residual plane: px overflow".into()))?;
    let py = (vb.y as usize)
        .checked_mul(8)
        .ok_or_else(|| Error::InvalidData("JXL residual plane: py overflow".into()))?;
    if px + cols > plane.width || py + rows > plane.height {
        return Err(Error::InvalidData(format!(
            "JXL residual plane: {:?} block at pixel ({px},{py}) covering {cols}×{rows} \
             spills past plane {}×{}",
            vb.transform, plane.width, plane.height
        )));
    }
    let pw = plane.width;
    for r in 0..rows {
        let dst = (py + r) * pw + px;
        let src = r * cols;
        plane.samples[dst..dst + cols].copy_from_slice(&block[src..src + cols]);
    }
    Ok(())
}

/// Assemble a full single-channel residual plane by walking `grid` in
/// raster order and invoking `residual_at(&vb)` once per varblock to
/// obtain each block's `R × C` spatial residual samples (the output of
/// [`crate::block_dequant::decode_block_to_residual`]), placing each into
/// the returned [`ResidualPlane`].
///
/// The closure owns the per-varblock decode (coefficient lookup, F.3
/// dequant, IDCT); this driver owns only the grid walk and the spatial
/// placement geometry. Closure errors propagate verbatim without writing
/// a partial block. The walk order is the canonical §C.8.3 raster order
/// (top-left cells row-major; continuation cells skipped).
///
/// Defensive: rejects any varblock whose transform lacks a pixel-dims
/// mapping or whose residual length / footprint is inconsistent with the
/// plane (via [`place_block`]).
pub fn assemble_channel_plane<F>(grid: &DctSelectGrid, mut residual_at: F) -> Result<ResidualPlane>
where
    F: FnMut(&Varblock) -> Result<Vec<f32>>,
{
    // Validate every covered varblock has a grid-dims mapping up front so
    // a malformed transform surfaces before any allocation work.
    let mut plane = ResidualPlane::for_grid(grid)?;
    let mut walk = VarblockWalk::new(grid);
    while let Some(vb) = walk.next()? {
        // Defence-in-depth: confirm the transform is one this stack can
        // place (covered_grid_dims == has a pixel-dims mapping).
        if covered_grid_dims(vb.transform).is_none() {
            return Err(Error::Unsupported(format!(
                "JXL residual plane: {:?} is not a covered transform",
                vb.transform
            )));
        }
        let block = residual_at(&vb)?;
        place_block(&mut plane, &vb, &block)?;
    }
    Ok(plane)
}

/// The three XYB residual planes of a single LfGroup, all sized to the
/// same padded block grid: index `0 = X`, `1 = Y`, `2 = B` (the VarDCT
/// channel ordering of Listing C.13).
///
/// Each plane is a [`ResidualPlane`] over the LfGroup's padded block
/// grid (`width_blocks·8 × height_blocks·8`); the caller crops all three
/// to `lf_w × lf_h` after restoration filtering.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelResidualPlanes {
    /// The X / Y / B residual planes, in that channel order.
    pub planes: [ResidualPlane; 3],
}

impl ChannelResidualPlanes {
    /// Borrow the X plane (channel 0).
    pub fn x(&self) -> &ResidualPlane {
        &self.planes[0]
    }
    /// Borrow the Y plane (channel 1).
    pub fn y(&self) -> &ResidualPlane {
        &self.planes[1]
    }
    /// Borrow the B plane (channel 2).
    pub fn b(&self) -> &ResidualPlane {
        &self.planes[2]
    }
    /// The shared plane dimensions `(width, height)` in pixels.
    pub fn dims(&self) -> (usize, usize) {
        (self.planes[0].width, self.planes[0].height)
    }

    /// Crop all three XYB planes to a `target_width × target_height`
    /// extent, dropping the right/bottom padding of the block grid
    /// (§6.2). This is the post-restoration step that turns the padded
    /// per-LfGroup reconstruction into the logical channel extent
    /// (`lf_w × lf_h`, or the §6.2.1 cropped frame extent for the whole
    /// image).
    ///
    /// All three planes share the padded grid geometry (the non-subsampled
    /// VarDCT case these planes are produced for), so a single target pair
    /// crops every channel identically via [`ResidualPlane::crop_to`];
    /// see that method for the shrink-only contract and the §6.2 cropping
    /// semantics. Any per-channel error propagates verbatim.
    pub fn crop_to(
        &self,
        target_width: usize,
        target_height: usize,
    ) -> Result<ChannelResidualPlanes> {
        let x = self.planes[0].crop_to(target_width, target_height)?;
        let y = self.planes[1].crop_to(target_width, target_height)?;
        let b = self.planes[2].crop_to(target_width, target_height)?;
        Ok(ChannelResidualPlanes { planes: [x, y, b] })
    }
}

/// Assemble all three XYB residual planes of an LfGroup by walking the
/// shared [`DctSelectGrid`] once per channel.
///
/// `residual_at(channel, &vb)` is invoked once per `(channel, varblock)`
/// pair — three full raster walks of the grid, channel order `0 = X`,
/// `1 = Y`, `2 = B` — and returns the channel's `R × C` row-major
/// spatial residual block (the [`crate::block_dequant::decode_block_to_residual`]
/// output) for that varblock.
///
/// In VarDCT mode all three channels share one `DctSelectGrid`: the
/// transform choice is decoded once per varblock (§C.5.4) and applies to
/// every channel, and Annex G CfL "is skipped if any channel is
/// subsampled," so when CfL applies the three planes have identical
/// geometry. This driver therefore reuses the single grid for all three
/// channel walks and produces three planes of identical dimensions.
///
/// The per-varblock decode (coefficient lookup, F.3 dequant, IDCT) lives
/// entirely in the closure; this driver owns only the per-channel grid
/// walk and the spatial placement geometry (delegated to
/// [`assemble_channel_plane`]). Closure errors propagate verbatim.
pub fn assemble_three_channel_planes<F>(
    grid: &DctSelectGrid,
    mut residual_at: F,
) -> Result<ChannelResidualPlanes>
where
    F: FnMut(usize, &Varblock) -> Result<Vec<f32>>,
{
    let x = assemble_channel_plane(grid, |vb| residual_at(0, vb))?;
    let y = assemble_channel_plane(grid, |vb| residual_at(1, vb))?;
    let b = assemble_channel_plane(grid, |vb| residual_at(2, vb))?;
    Ok(ChannelResidualPlanes { planes: [x, y, b] })
}

/// Apply Annex G chroma-from-luma to three assembled XYB residual planes
/// in place.
///
/// The X (channel 0) and B (channel 2) residual planes carry the
/// `dX` / `dB` chroma-residual samples; the Y plane (channel 1) carries
/// the luma `dY`. Per Listing G.1 each X / B sample is restored as
/// `X = dX + kX·Y` / `B = dB + kB·Y`, with `(kX, kB)` looked up per the
/// 64×64 tile containing the sample (the HF path of Annex G). After this
/// call `planes[0]` holds the final `X` plane, `planes[2]` the final `B`
/// plane, and `planes[1]` (`Y`) is unchanged.
///
/// `x_from_y` / `b_from_y` are the per-64×64-tile factor channels from
/// [`crate::lf_group::HfMetadata`], each of length
/// `ceil(width / 64) × ceil(height / 64)` over the shared plane
/// dimensions (row-major).
///
/// Errors (via [`apply_hf_plane_inplace`]): factor-plane length mismatch,
/// or `cfl.colour_factor == 0`.
pub fn apply_chroma_from_luma(
    planes: &mut ChannelResidualPlanes,
    x_from_y: &[i32],
    b_from_y: &[i32],
    cfl: &LfChannelCorrelation,
) -> Result<()> {
    let (w, h) = planes.dims();
    let width = u32::try_from(w)
        .map_err(|_| Error::InvalidData("JXL CfL: plane width exceeds u32".into()))?;
    let height = u32::try_from(h)
        .map_err(|_| Error::InvalidData("JXL CfL: plane height exceeds u32".into()))?;
    // Split the array so X and B are mutable while Y is a shared borrow.
    let [x_plane, y_plane, b_plane] = &mut planes.planes;
    apply_hf_plane_inplace(
        &mut x_plane.samples,
        &y_plane.samples,
        &mut b_plane.samples,
        width,
        height,
        x_from_y,
        b_from_y,
        cfl,
    )
}

/// One-call per-LfGroup three-channel reconstruction: assemble the X / Y
/// / B residual planes from the shared grid, then apply Annex G CfL,
/// returning the final XYB planes (still on the padded block grid; the
/// caller crops to `lf_w × lf_h`).
///
/// Equivalent to [`assemble_three_channel_planes`] followed by
/// [`apply_chroma_from_luma`]. Gaborish (Annex J.2) and EPF (Annex J.3)
/// run on the returned planes and remain caller-side concerns above this
/// primitive.
pub fn reconstruct_three_channel_planes<F>(
    grid: &DctSelectGrid,
    x_from_y: &[i32],
    b_from_y: &[i32],
    cfl: &LfChannelCorrelation,
    residual_at: F,
) -> Result<ChannelResidualPlanes>
where
    F: FnMut(usize, &Varblock) -> Result<Vec<f32>>,
{
    let mut planes = assemble_three_channel_planes(grid, residual_at)?;
    apply_chroma_from_luma(&mut planes, x_from_y, b_from_y, cfl)?;
    Ok(planes)
}

/// LF-aware counterpart of [`assemble_three_channel_planes`]: assemble
/// the X / Y / B residual planes from the shared grid, threading each
/// varblock's separately-decoded **LLF** (DC subband) coefficients into
/// the per-block decode.
///
/// For every varblock walked (per channel `c ∈ [0..3]`) this driver
/// extracts that channel's `cy × cx` row-major LLF block from `lf` at the
/// varblock's `(vb.x, vb.y)` 8×8-block grid origin — via
/// [`crate::vardct::compose_lf_to_llf_block`] (`extract_lf_subblock` then
/// Listing I.16 `llf_from_lf`) — and hands it to the caller's
/// `decode_with_llf(channel, vb, &llf)` closure, which owns the F.3
/// dequant → §I.2.4 LLF merge → §I.2.3.2 IDCT walk (i.e.
/// [`crate::block_dequant::decode_block_to_residual_with_llf`]). The
/// returned residual block is the **complete** per-channel spatial
/// block, not the HF-only residual of
/// [`assemble_three_channel_planes`].
///
/// `lf` carries the dequantised LF samples of this LfGroup for all three
/// channels ([`crate::lf_dequant::dequant_lf`] output). The three LF
/// channels must share identical dims — the non-subsampled case §F.2
/// applies to; a per-channel-dims subsampled LfGroup must drive the
/// per-channel `compose_lf_to_llf_block` path directly. Mismatched LF
/// dims surface as [`Error::InvalidData`] before any decode work.
///
/// The grid walk, plane geometry, and placement are exactly those of
/// [`assemble_channel_plane`]; only the per-block residual source folds
/// in the LLF. Closure errors and `compose_lf_to_llf_block` errors
/// (varblock origin overflow, varblock spilling past the LF grid)
/// propagate verbatim without writing a partial plane.
pub fn assemble_three_channel_planes_with_lf<F>(
    grid: &DctSelectGrid,
    lf: &LfDequantOutput,
    mut decode_with_llf: F,
) -> Result<ChannelResidualPlanes>
where
    F: FnMut(usize, &Varblock, &[f32]) -> Result<Vec<f32>>,
{
    // The three LF channels must agree on dims (non-subsampled case);
    // reject up front so a mismatch can't silently mis-address one
    // channel's LLF extraction mid-walk.
    if lf.widths[0] != lf.widths[1]
        || lf.widths[0] != lf.widths[2]
        || lf.heights[0] != lf.heights[1]
        || lf.heights[0] != lf.heights[2]
    {
        return Err(Error::InvalidData(format!(
            "JXL reconstruct_with_lf: LF channels have different dims \
             (widths = {:?}, heights = {:?}); the subsampled case must \
             drive compose_lf_to_llf_block per channel",
            lf.widths, lf.heights,
        )));
    }
    let lf_w = lf.widths[0];
    let lf_h = lf.heights[0];

    let assemble_one = |c: usize, decode: &mut F| -> Result<ResidualPlane> {
        assemble_channel_plane(grid, |vb| {
            let llf =
                compose_lf_to_llf_block(&lf.samples[c], lf_w, lf_h, vb.x, vb.y, vb.transform)?;
            decode(c, vb, &llf)
        })
    };

    let x = assemble_one(0, &mut decode_with_llf)?;
    let y = assemble_one(1, &mut decode_with_llf)?;
    let b = assemble_one(2, &mut decode_with_llf)?;
    Ok(ChannelResidualPlanes { planes: [x, y, b] })
}

/// One-call LF-aware per-LfGroup three-channel reconstruction: the
/// LF-aware counterpart of [`reconstruct_three_channel_planes`].
///
/// Assembles the X / Y / B residual planes from the shared grid with
/// each varblock's LLF (DC subband) coefficients folded in
/// ([`assemble_three_channel_planes_with_lf`]), then applies Annex G
/// chroma-from-luma ([`apply_chroma_from_luma`]). The returned XYB
/// planes are the final per-LfGroup residual planes on the padded block
/// grid; the caller crops to `lf_w × lf_h` and runs Gaborish (Annex J.2)
/// + EPF (Annex J.3) above this primitive.
///
/// `lf` supplies the LfGroup's dequantised LF samples and
/// `decode_with_llf(channel, vb, &llf)` owns the per-block F.3 dequant →
/// §I.2.4 LLF merge → §I.2.3.2 IDCT walk
/// ([`crate::block_dequant::decode_block_to_residual_with_llf`]); see
/// [`assemble_three_channel_planes_with_lf`] for the LLF-extraction and
/// LF-dims contract.
pub fn reconstruct_three_channel_planes_with_lf<F>(
    grid: &DctSelectGrid,
    lf: &LfDequantOutput,
    x_from_y: &[i32],
    b_from_y: &[i32],
    cfl: &LfChannelCorrelation,
    decode_with_llf: F,
) -> Result<ChannelResidualPlanes>
where
    F: FnMut(usize, &Varblock, &[f32]) -> Result<Vec<f32>>,
{
    let mut planes = assemble_three_channel_planes_with_lf(grid, lf, decode_with_llf)?;
    apply_chroma_from_luma(&mut planes, x_from_y, b_from_y, cfl)?;
    Ok(planes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dct_select::{DctSelectCell, DctSelectGrid};

    /// Build a grid directly from a cell list + hf_mul + dims (bypassing
    /// `derive_dct_select` so tests pin placement geometry independently
    /// of BlockInfo parsing).
    fn grid(cells: Vec<DctSelectCell>, hf_mul: Vec<i32>, w: u32, h: u32) -> DctSelectGrid {
        DctSelectGrid {
            cells,
            hf_mul,
            width_blocks: w,
            height_blocks: h,
        }
    }

    /// A 1×1 grid holding a single DCT8×8 varblock.
    fn single_dct8x8() -> DctSelectGrid {
        grid(
            vec![DctSelectCell::TopLeft(TransformType::Dct8x8)],
            vec![1],
            1,
            1,
        )
    }

    fn vb(x: u32, y: u32, t: TransformType) -> Varblock {
        Varblock {
            x,
            y,
            transform: t,
            hf_mul: 1,
        }
    }

    #[test]
    fn block_pixel_dims_covers_every_transform() {
        use TransformType as T;
        // DCT family from dct_pixel_dims.
        assert_eq!(block_pixel_dims(T::Dct8x8), Some((8, 8)));
        assert_eq!(block_pixel_dims(T::Dct16x8), Some((16, 8)));
        assert_eq!(block_pixel_dims(T::Dct8x16), Some((8, 16)));
        assert_eq!(block_pixel_dims(T::Dct256x256), Some((256, 256)));
        // Non-DCT family — all 8×8.
        for t in [
            T::Hornuss,
            T::Dct2x2,
            T::Dct4x4,
            T::Dct4x8,
            T::Dct8x4,
            T::Afv0,
            T::Afv1,
            T::Afv2,
            T::Afv3,
        ] {
            assert_eq!(block_pixel_dims(t), Some((8, 8)), "{t:?}");
        }
    }

    #[test]
    fn pixel_dims_match_block_dims_times_8_for_every_transform() {
        // The placement geometry invariant: C == bcols*8, R == brows*8.
        use TransformType as T;
        for t in [
            T::Dct8x8,
            T::Dct16x16,
            T::Dct32x32,
            T::Dct16x8,
            T::Dct8x16,
            T::Dct32x8,
            T::Dct8x32,
            T::Dct32x16,
            T::Dct16x32,
            T::Dct64x64,
            T::Dct64x32,
            T::Dct32x64,
            T::Dct128x128,
            T::Dct128x64,
            T::Dct64x128,
            T::Dct256x256,
            T::Dct256x128,
            T::Dct128x256,
            T::Hornuss,
            T::Dct2x2,
            T::Dct4x4,
            T::Dct4x8,
            T::Dct8x4,
            T::Afv0,
            T::Afv1,
            T::Afv2,
            T::Afv3,
        ] {
            let (rows, cols) = block_pixel_dims(t).unwrap();
            let (bcols, brows) = t.block_dims();
            assert_eq!(cols, bcols as usize * 8, "{t:?} cols vs block_dims");
            assert_eq!(rows, brows as usize * 8, "{t:?} rows vs block_dims");
        }
    }

    #[test]
    fn for_grid_sizes_to_padded_block_grid() {
        // 3 cols × 2 rows of cells → 24 × 16 pixel plane.
        let g = grid(
            vec![DctSelectCell::TopLeft(TransformType::Dct8x8); 6],
            vec![1; 6],
            3,
            2,
        );
        let p = ResidualPlane::for_grid(&g).unwrap();
        assert_eq!(p.width, 24);
        assert_eq!(p.height, 16);
        assert_eq!(p.samples.len(), 24 * 16);
        assert!(p.samples.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn place_single_dct8x8_at_origin() {
        let g = single_dct8x8();
        let mut p = ResidualPlane::for_grid(&g).unwrap();
        // A block whose value equals its raster index.
        let block: Vec<f32> = (0..64).map(|i| i as f32).collect();
        place_block(&mut p, &vb(0, 0, TransformType::Dct8x8), &block).unwrap();
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(p.get(c, r), Some((r * 8 + c) as f32), "({c},{r})");
            }
        }
    }

    #[test]
    fn place_block_at_offset_origin() {
        // 3×3 cell grid; place a DCT8×8 block at grid cell (2, 1) → pixel
        // origin (16, 8).
        let g = grid(
            vec![DctSelectCell::TopLeft(TransformType::Dct8x8); 9],
            vec![1; 9],
            3,
            3,
        );
        let mut p = ResidualPlane::for_grid(&g).unwrap();
        let block = vec![7.0f32; 64];
        place_block(&mut p, &vb(2, 1, TransformType::Dct8x8), &block).unwrap();
        // The 8×8 region [16..24) × [8..16) is 7.0; everything else 0.
        for y in 0..p.height {
            for x in 0..p.width {
                let inside = (16..24).contains(&x) && (8..16).contains(&y);
                let want = if inside { 7.0 } else { 0.0 };
                assert_eq!(p.get(x, y), Some(want), "({x},{y})");
            }
        }
    }

    #[test]
    fn place_rectangular_dct16x8_writes_tall_block() {
        // DCT16×8 = 16 rows × 8 cols = 8px wide × 16px tall; block_dims
        // (1, 2) = 1 col × 2 rows of cells. Grid must be at least 1×2.
        let g = grid(
            vec![
                DctSelectCell::TopLeft(TransformType::Dct16x8),
                DctSelectCell::Continuation,
            ],
            vec![1, 0],
            1,
            2,
        );
        let mut p = ResidualPlane::for_grid(&g).unwrap();
        assert_eq!((p.width, p.height), (8, 16));
        // R×C = 16×8 row-major block: value = r (the row).
        let (rows, cols) = block_pixel_dims(TransformType::Dct16x8).unwrap();
        assert_eq!((rows, cols), (16, 8));
        let block: Vec<f32> = (0..rows * cols).map(|i| (i / cols) as f32).collect();
        place_block(&mut p, &vb(0, 0, TransformType::Dct16x8), &block).unwrap();
        for y in 0..16 {
            for x in 0..8 {
                assert_eq!(p.get(x, y), Some(y as f32), "({x},{y})");
            }
        }
    }

    #[test]
    fn place_rejects_wrong_block_length() {
        let g = single_dct8x8();
        let mut p = ResidualPlane::for_grid(&g).unwrap();
        let err =
            place_block(&mut p, &vb(0, 0, TransformType::Dct8x8), &vec![0.0; 63]).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)), "got {err:?}");
    }

    #[test]
    fn place_rejects_footprint_spill() {
        // A 1×1 plane (8×8 px) cannot hold a DCT16×16 (16×16 px) block.
        let g = single_dct8x8();
        let mut p = ResidualPlane::for_grid(&g).unwrap();
        let block = vec![0.0f32; 256];
        let err = place_block(&mut p, &vb(0, 0, TransformType::Dct16x16), &block).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)), "got {err:?}");
    }

    #[test]
    fn assemble_uniform_dct8x8_grid_in_raster_order() {
        // 2×2 grid of DCT8×8; the closure returns a constant block whose
        // value encodes the varblock raster index, proving placement
        // order + position.
        let cells = vec![DctSelectCell::TopLeft(TransformType::Dct8x8); 4];
        let g = grid(cells, vec![1; 4], 2, 2);
        let mut idx = 0u32;
        let p = assemble_channel_plane(&g, |v| {
            // Raster order must be (0,0), (1,0), (0,1), (1,1).
            let expect = v.y * 2 + v.x;
            assert_eq!(expect, idx, "raster order at vb ({},{})", v.x, v.y);
            idx += 1;
            Ok(vec![expect as f32; 64])
        })
        .unwrap();
        assert_eq!(idx, 4, "exactly 4 varblocks visited");
        assert_eq!((p.width, p.height), (16, 16));
        // Quadrant values: top-left=0, top-right=1, bottom-left=2,
        // bottom-right=3.
        assert_eq!(p.get(0, 0), Some(0.0));
        assert_eq!(p.get(8, 0), Some(1.0));
        assert_eq!(p.get(0, 8), Some(2.0));
        assert_eq!(p.get(8, 8), Some(3.0));
    }

    #[test]
    fn assemble_mixed_transform_grid() {
        // A DCT16×16 (2×2 cells) at (0,0) plus a DCT8×8 at (2,0) and a
        // DCT8×8 at (3,0) on a 4×2 grid; remaining cells DCT8×8.
        // Layout (cells, row-major, w=4 h=2):
        //   row0: TL(16) Cont    TL(8)  TL(8)
        //   row1: Cont   Cont    TL(8)  TL(8)
        let cells = vec![
            DctSelectCell::TopLeft(TransformType::Dct16x16),
            DctSelectCell::Continuation,
            DctSelectCell::TopLeft(TransformType::Dct8x8),
            DctSelectCell::TopLeft(TransformType::Dct8x8),
            DctSelectCell::Continuation,
            DctSelectCell::Continuation,
            DctSelectCell::TopLeft(TransformType::Dct8x8),
            DctSelectCell::TopLeft(TransformType::Dct8x8),
        ];
        let g = grid(cells, vec![1, 0, 1, 1, 0, 0, 1, 1], 4, 2);
        let p = assemble_channel_plane(&g, |v| {
            let (rows, cols) = block_pixel_dims(v.transform).unwrap();
            // Distinct constant per transform: 16×16 → 16, 8×8 → 8.
            Ok(vec![rows as f32; rows * cols])
        })
        .unwrap();
        assert_eq!((p.width, p.height), (32, 16));
        // The DCT16×16 fills pixels [0,16)×[0,16) with 16.0.
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(p.get(x, y), Some(16.0), "DCT16 ({x},{y})");
            }
        }
        // The DCT8×8 at grid (2,0) fills [16,24)×[0,8) with 8.0.
        for y in 0..8 {
            for x in 16..24 {
                assert_eq!(p.get(x, y), Some(8.0), "DCT8 ({x},{y})");
            }
        }
        // The DCT8×8 at grid (2,1) fills [16,24)×[8,16) with 8.0.
        assert_eq!(p.get(16, 8), Some(8.0));
    }

    #[test]
    fn assemble_propagates_closure_error() {
        let g = single_dct8x8();
        let err = assemble_channel_plane(&g, |_| {
            Err(Error::InvalidData("boom".into())) as Result<Vec<f32>>
        })
        .unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)), "got {err:?}");
    }

    #[test]
    fn assemble_empty_grid_is_empty_plane() {
        let g = grid(vec![], vec![], 0, 0);
        let p = assemble_channel_plane(&g, |_| unreachable!("no varblocks")).unwrap();
        assert_eq!((p.width, p.height), (0, 0));
        assert!(p.samples.is_empty());
    }

    #[test]
    fn assemble_rejects_residual_empty_cell() {
        // A grid with an Empty cell (malformed) errors during the walk.
        let g = grid(vec![DctSelectCell::Empty], vec![0], 1, 1);
        let err = assemble_channel_plane(&g, |_| Ok(vec![0.0; 64])).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)), "got {err:?}");
    }

    #[test]
    fn get_out_of_range_is_none() {
        let g = single_dct8x8();
        let p = ResidualPlane::for_grid(&g).unwrap();
        assert_eq!(p.get(8, 0), None);
        assert_eq!(p.get(0, 8), None);
        assert!(p.get(7, 7).is_some());
    }

    #[test]
    fn assemble_three_channels_visits_each_channel_in_order() {
        // 1×1 DCT8×8 grid: the closure should be called once per channel,
        // in order 0=X, 1=Y, 2=B, with the same varblock each time.
        let g = single_dct8x8();
        let mut calls: Vec<usize> = Vec::new();
        let planes = assemble_three_channel_planes(&g, |c, v| {
            assert_eq!((v.x, v.y), (0, 0));
            calls.push(c);
            // Tag each channel's plane with a distinct constant.
            Ok(vec![(c as f32 + 1.0) * 10.0; 64])
        })
        .unwrap();
        assert_eq!(calls, vec![0, 1, 2]);
        assert_eq!(planes.dims(), (8, 8));
        assert_eq!(planes.x().get(0, 0), Some(10.0));
        assert_eq!(planes.y().get(0, 0), Some(20.0));
        assert_eq!(planes.b().get(0, 0), Some(30.0));
    }

    #[test]
    fn three_channel_planes_share_dims() {
        // 2×2 DCT8×8 grid → all three planes are 16×16.
        let cells = vec![DctSelectCell::TopLeft(TransformType::Dct8x8); 4];
        let g = grid(cells, vec![1; 4], 2, 2);
        let planes = assemble_three_channel_planes(&g, |_c, _v| Ok(vec![0.0f32; 64])).unwrap();
        assert_eq!(planes.dims(), (16, 16));
        for p in &planes.planes {
            assert_eq!((p.width, p.height), (16, 16));
        }
    }

    #[test]
    fn assemble_three_channels_propagates_channel_error() {
        // Fail only on the B channel (index 2); the X+Y walks must have
        // run first, so the error surfaces from the third walk.
        let g = single_dct8x8();
        let err = assemble_three_channel_planes(&g, |c, _v| {
            if c == 2 {
                Err(Error::InvalidData("b boom".into()))
            } else {
                Ok(vec![0.0f32; 64])
            }
        })
        .unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)), "got {err:?}");
    }

    #[test]
    fn cfl_default_correlation_restores_x_and_b() {
        // Single 8×8 DCT block; default LfChannelCorrelation:
        //   kX = base_correlation_x + x_factor/colour_factor
        //      = 0.0 + 0/84 = 0.0
        //   kB = base_correlation_b + b_factor/colour_factor
        //      = 1.0 + 0/84 = 1.0
        // (XFromY / BFromY tile factors are all 0 here.)
        // So final X = dX + 0·Y = dX (unchanged); B = dB + 1·Y.
        let g = single_dct8x8();
        let mut planes = assemble_three_channel_planes(&g, |c, _v| {
            // dX = 3, dY = 5, dB = 7 everywhere.
            let v = match c {
                0 => 3.0,
                1 => 5.0,
                _ => 7.0,
            };
            Ok(vec![v; 64])
        })
        .unwrap();
        // One 8×8 plane → ceil(8/64) = 1 tile in each axis → 1 factor cell.
        let x_from_y = vec![0i32; 1];
        let b_from_y = vec![0i32; 1];
        let cfl = LfChannelCorrelation::default();
        apply_chroma_from_luma(&mut planes, &x_from_y, &b_from_y, &cfl).unwrap();
        // X unchanged (kX = 0), Y unchanged, B = 7 + 1·5 = 12.
        assert_eq!(planes.x().get(0, 0), Some(3.0));
        assert_eq!(planes.y().get(0, 0), Some(5.0));
        assert_eq!(planes.b().get(0, 0), Some(12.0));
    }

    #[test]
    fn cfl_nonzero_tile_factor_changes_kx() {
        // x_factor = 84 → kX = 0 + 84/84 = 1.0; X = dX + 1·Y.
        // b_factor = -84 → kB = 1 + (-84)/84 = 0.0; B = dB + 0·Y = dB.
        let g = single_dct8x8();
        let mut planes = assemble_three_channel_planes(&g, |c, _v| {
            let v = match c {
                0 => 2.0,
                1 => 10.0,
                _ => 4.0,
            };
            Ok(vec![v; 64])
        })
        .unwrap();
        let x_from_y = vec![84i32; 1];
        let b_from_y = vec![-84i32; 1];
        let cfl = LfChannelCorrelation::default();
        apply_chroma_from_luma(&mut planes, &x_from_y, &b_from_y, &cfl).unwrap();
        // X = 2 + 1·10 = 12; Y = 10; B = 4 + 0·10 = 4.
        assert_eq!(planes.x().get(0, 0), Some(12.0));
        assert_eq!(planes.y().get(0, 0), Some(10.0));
        assert_eq!(planes.b().get(0, 0), Some(4.0));
    }

    #[test]
    fn cfl_rejects_wrong_factor_plane_length() {
        let g = single_dct8x8();
        let mut planes = assemble_three_channel_planes(&g, |_c, _v| Ok(vec![0.0f32; 64])).unwrap();
        // 8×8 plane needs exactly 1 factor cell; supply 2.
        let err = apply_chroma_from_luma(
            &mut planes,
            &[0i32; 2],
            &[0i32; 1],
            &LfChannelCorrelation::default(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)), "got {err:?}");
    }

    #[test]
    fn cfl_rejects_zero_colour_factor() {
        let g = single_dct8x8();
        let mut planes = assemble_three_channel_planes(&g, |_c, _v| Ok(vec![0.0f32; 64])).unwrap();
        let cfl = LfChannelCorrelation {
            colour_factor: 0,
            ..LfChannelCorrelation::default()
        };
        let err = apply_chroma_from_luma(&mut planes, &[0i32; 1], &[0i32; 1], &cfl).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)), "got {err:?}");
    }

    #[test]
    fn reconstruct_equals_assemble_then_cfl() {
        // The one-call driver must equal the two-step composition.
        let cells = vec![DctSelectCell::TopLeft(TransformType::Dct8x8); 4];
        let g = grid(cells, vec![1; 4], 2, 2);
        let resid = |c: usize, v: &Varblock| -> Result<Vec<f32>> {
            // A value that varies by channel and varblock position.
            let base = (c as f32) * 100.0 + (v.y * 2 + v.x) as f32;
            Ok(vec![base; 64])
        };
        // 16×16 plane → ceil(16/64) = 1 tile.
        let x_from_y = vec![42i32; 1];
        let b_from_y = vec![-21i32; 1];
        let cfl = LfChannelCorrelation::default();

        let mut step = assemble_three_channel_planes(&g, resid).unwrap();
        apply_chroma_from_luma(&mut step, &x_from_y, &b_from_y, &cfl).unwrap();

        let one = reconstruct_three_channel_planes(&g, &x_from_y, &b_from_y, &cfl, resid).unwrap();
        assert_eq!(one, step);
    }

    #[test]
    fn reconstruct_multi_tile_uses_per_tile_factor() {
        // A 9×9 cell grid → 72×72 px plane → ceil(72/64) = 2 tiles per
        // axis = 4 factor cells. Tile (0,0) covers pixels [0,64); tile
        // (1,1) covers [64,72). Use distinct x_factor per tile and check
        // a sample in each tile picks up the matching kX.
        let cells = vec![DctSelectCell::TopLeft(TransformType::Dct8x8); 81];
        let g = grid(cells, vec![1; 81], 9, 9);
        // dX = 0, dY = 84 everywhere → X = 0 + kX·84 = x_factor (since
        // kX = x_factor/84 with default base 0 and colour_factor 84).
        let planes = reconstruct_three_channel_planes(
            &g,
            // tiles row-major: (0,0)=1, (1,0)=2, (0,1)=3, (1,1)=4.
            &[1i32, 2, 3, 4],
            &[0i32; 4],
            &LfChannelCorrelation::default(),
            |c, _v| Ok(vec![if c == 1 { 84.0 } else { 0.0 }; 64]),
        )
        .unwrap();
        // X = x_factor/84·84 = x_factor (within f32 rounding). Each
        // sampled pixel falls in a distinct tile, so it must pick up the
        // matching tile's x_factor.
        let close = |got: Option<f32>, want: f32| {
            let v = got.unwrap();
            assert!((v - want).abs() < 1e-3, "got {v}, want {want}");
        };
        close(planes.x().get(0, 0), 1.0); // tile (0,0)
        close(planes.x().get(64, 0), 2.0); // tile (1,0)
        close(planes.x().get(0, 64), 3.0); // tile (0,1)
        close(planes.x().get(64, 64), 4.0); // tile (1,1)
    }

    #[test]
    fn with_lf_threads_per_channel_llf_block() {
        // Single DCT8×8 varblock (cx=cy=1 → LLF is the LF sample). Feed a
        // distinct LF sample per channel and a decode closure that simply
        // returns a flat block of the LLF value; the assembled plane must
        // carry each channel's own LLF across its 8×8 footprint.
        let g = single_dct8x8();
        let lf = LfDequantOutput {
            samples: [vec![10.0], vec![20.0], vec![30.0]],
            widths: [1, 1, 1],
            heights: [1, 1, 1],
        };
        let planes = assemble_three_channel_planes_with_lf(&g, &lf, |_c, _vb, llf| {
            assert_eq!(llf.len(), 1);
            Ok(vec![llf[0]; 64])
        })
        .unwrap();
        // DCT8×8 LLF = LF sample within f32 rounding (ScaleF(1,8,0) ≈ 1).
        for (c, &want) in [10.0f32, 20.0, 30.0].iter().enumerate() {
            for r in 0..8 {
                for x in 0..8 {
                    let v = planes.planes[c].get(x, r).unwrap();
                    assert!((v - want).abs() < 1e-3, "channel {c} cell ({x},{r}) = {v}");
                }
            }
        }
    }

    #[test]
    fn with_lf_passes_correct_block_coords_to_llf_extraction() {
        // A 2×1 grid: two side-by-side DCT8×8 varblocks. The 2×1 LF image
        // gives the left varblock LF cell 0 and the right varblock LF cell
        // 1, so the per-varblock block-grid origin must thread through
        // `compose_lf_to_llf_block` into the closure's LLF.
        let g = grid(
            vec![
                DctSelectCell::TopLeft(TransformType::Dct8x8),
                DctSelectCell::TopLeft(TransformType::Dct8x8),
            ],
            vec![1, 1],
            2,
            1,
        );
        let lf = LfDequantOutput {
            samples: [vec![3.0, 7.0], vec![3.0, 7.0], vec![3.0, 7.0]],
            widths: [2, 2, 2],
            heights: [1, 1, 1],
        };
        let planes =
            assemble_three_channel_planes_with_lf(&g, &lf, |_c, _vb, llf| Ok(vec![llf[0]; 64]))
                .unwrap();
        // Left block (pixels x<8) carries LF 3; right block (x>=8) LF 7.
        assert!((planes.planes[1].get(0, 0).unwrap() - 3.0).abs() < 1e-3);
        assert!((planes.planes[1].get(8, 0).unwrap() - 7.0).abs() < 1e-3);
    }

    #[test]
    fn with_lf_rejects_mismatched_channel_dims() {
        let g = single_dct8x8();
        let lf = LfDequantOutput {
            samples: [vec![1.0], vec![1.0, 2.0], vec![1.0]],
            widths: [1, 2, 1],
            heights: [1, 1, 1],
        };
        let err = assemble_three_channel_planes_with_lf(&g, &lf, |_, _, _| Ok(vec![0.0; 64]))
            .unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)), "got {err:?}");
    }

    #[test]
    fn reconstruct_with_lf_equals_assemble_with_lf_then_cfl() {
        // The one-call LF-aware driver equals the two-step composition
        // (assemble_with_lf then apply_chroma_from_luma), mirroring the
        // non-LF `reconstruct_equals_assemble_then_cfl` invariant.
        let cells = vec![DctSelectCell::TopLeft(TransformType::Dct8x8); 4];
        let g = grid(cells, vec![1; 4], 2, 2);
        let lf = LfDequantOutput {
            samples: [
                vec![1.0, 2.0, 3.0, 4.0],
                vec![5.0, 6.0, 7.0, 8.0],
                vec![9.0, 10.0, 11.0, 12.0],
            ],
            widths: [2, 2, 2],
            heights: [2, 2, 2],
        };
        let decode = |_c: usize, _vb: &Varblock, llf: &[f32]| Ok(vec![llf[0]; 64]);
        let x_from_y = vec![42i32; 1];
        let b_from_y = vec![-21i32; 1];
        let cfl = LfChannelCorrelation::default();

        let mut step = assemble_three_channel_planes_with_lf(&g, &lf, decode).unwrap();
        apply_chroma_from_luma(&mut step, &x_from_y, &b_from_y, &cfl).unwrap();

        let one =
            reconstruct_three_channel_planes_with_lf(&g, &lf, &x_from_y, &b_from_y, &cfl, decode)
                .unwrap();
        assert_eq!(one, step);
    }

    // ---- §6.2 crop (right/bottom padding removal) ----

    /// Build a padded plane whose sample at `(x, y)` equals `y*width+x`,
    /// so a crop's retained samples are trivially checkable.
    fn ramp_plane(width: usize, height: usize) -> ResidualPlane {
        ResidualPlane {
            width,
            height,
            samples: (0..width * height).map(|i| i as f32).collect(),
        }
    }

    #[test]
    fn crop_keeps_top_left_rectangle() {
        // 24×16 padded plane (3×2 blocks) cropped to a 17×9 logical extent
        // (ceil/8 = 3×2 blocks, the §C.5.4 padding relationship).
        let p = ramp_plane(24, 16);
        let c = p.crop_to(17, 9).unwrap();
        assert_eq!(c.width, 17);
        assert_eq!(c.height, 9);
        assert_eq!(c.samples.len(), 17 * 9);
        // Every retained sample equals the original at the same (x, y):
        // §6.2 crops at the right and bottom, top-left rectangle kept.
        for y in 0..9 {
            for x in 0..17 {
                assert_eq!(c.get(x, y), p.get(x, y), "({x},{y})");
            }
        }
    }

    #[test]
    fn crop_to_full_dims_is_identity() {
        let p = ramp_plane(16, 8);
        let c = p.crop_to(16, 8).unwrap();
        assert_eq!(c, p);
    }

    #[test]
    fn crop_one_pixel_drops_last_row_and_column() {
        // 8×8 → 7×7 keeps columns 0..7 of rows 0..7, discarding the last
        // row and the last column of every kept row.
        let p = ramp_plane(8, 8);
        let c = p.crop_to(7, 7).unwrap();
        assert_eq!((c.width, c.height), (7, 7));
        assert_eq!(c.get(6, 0), Some(6.0)); // last kept column of row 0
        assert_eq!(c.get(0, 6), Some(48.0)); // (0,6) = 6*8+0
        assert_eq!(c.get(7, 0), None); // dropped column
        assert_eq!(c.get(0, 7), None); // dropped row
    }

    #[test]
    fn crop_rejects_growth_and_zero() {
        let p = ramp_plane(16, 8);
        assert!(p.crop_to(17, 8).is_err()); // wider than padded plane
        assert!(p.crop_to(16, 9).is_err()); // taller than padded plane
        assert!(p.crop_to(0, 8).is_err()); // zero width
        assert!(p.crop_to(16, 0).is_err()); // zero height
    }

    #[test]
    fn channel_crop_applies_to_all_three_planes() {
        // Distinct ramps per channel so a mis-wired channel would show.
        let planes = ChannelResidualPlanes {
            planes: [ramp_plane(16, 16), ramp_plane(16, 16), ramp_plane(16, 16)],
        };
        let c = planes.crop_to(13, 11).unwrap();
        assert_eq!(c.dims(), (13, 11));
        for ch in 0..3 {
            assert_eq!((c.planes[ch].width, c.planes[ch].height), (13, 11));
            // Spot-check a retained interior sample matches the source.
            assert_eq!(c.planes[ch].get(12, 10), planes.planes[ch].get(12, 10));
        }
    }
}
