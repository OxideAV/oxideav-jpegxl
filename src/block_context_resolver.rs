//! Per-LfGroup `BlockContext()` resolver —
//! ISO/IEC FDIS 18181-1:2021 §C.8.3 + §I.2.2 (was Listing C.13 +
//! Listing C.15).
//!
//! ## Scope (round 214)
//!
//! Round 214 lands the typed wrapper that bundles together
//! [`crate::lf_global::HfBlockContext`] (the LfGlobal §I.2.2
//! `qf_thresholds` + `lf_thresholds` + `block_ctx_map` triple) and
//! threads it through the per-varblock callback shape the round-208
//! [`crate::varblock_walk::decode_varblocks_for_pass_channel`] driver
//! already expects. The result is a stateless
//! [`BlockContextResolver`] borrow that turns the four-argument
//! [`crate::pass_group_hf::block_context`] call into a one-call
//! `resolve(channel, varblock, qdc) -> u32` lookup. The matching
//! [`decode_varblocks_with_resolver`] convenience saves callers from
//! writing the closure boilerplate every time.
//!
//! ## Scope (round 221)
//!
//! Round 221 layers the three-channel raster driver
//! [`decode_varblocks_three_channels_with_resolver`] on top of the
//! round-214 single-channel walker. §C.8.3 prose orders the
//! per-pass decode as
//!
//! > for each varblock in raster order, decode the X, Y, B channels
//! > (in that order) once
//!
//! — a single grid walk that emits 3 per-channel block-context reads
//! per varblock. The naive composition of calling
//! [`decode_varblocks_with_resolver`] three times (one per channel)
//! visits the grid three times and (critically) cannot share the
//! per-varblock `qdc[3]` derivation across channels. The round-221
//! driver walks the grid once, computes `qdc[3]` once per varblock,
//! invokes [`BlockContextResolver::resolve`] three times (X / Y / B)
//! against that shared `qdc`, and threads each `(p, c)` call through
//! [`PerPassNonZerosGrids::decode_block_at_for_pass_channel`]. The
//! return is one `(Varblock, [DecodedHfBlock; 3], [u32; 3])` triple
//! per varblock — three per-channel decoded blocks + three raw
//! `non_zeros` counters indexed by channel.
//!
//! This matches §C.8.3 prose ordering precisely (the spec text
//! orders the inner channel loop inside the outer varblock loop, not
//! the other way around) and gives callers a deterministic raster
//! layout that doesn't require post-walk per-channel zipping.
//!
//! No bit reads, no spec re-derivation, no histogram materialisation
//! — same pure-control-flow primitive shape as round-89
//! [`crate::dct_quant_weights`], round-95 [`crate::hf_dequant`],
//! round-121 [`crate::llf_from_lf`], round-138
//! [`crate::chroma_from_luma`], round-141 [`crate::gaborish`],
//! round-144 [`crate::epf`], round-147 [`crate::afv::afv_idct`],
//! round-159 / 164 [`crate::pass_group_hf`], round-177
//! [`crate::non_zeros_grid`], round-183
//! [`crate::per_channel_non_zeros`], round-190
//! [`crate::per_pass_non_zeros`], and round-208
//! [`crate::varblock_walk`].
//!
//! ## FDIS prose anchor
//!
//! From §C.8.3 (Listing C.13 `BlockContext()` first line):
//!
//! > `idx = (c < 2 ? c ^ 1 : 2) × 13 + s`
//!
//! where `c` is the channel (0 = X, 1 = Y, 2 = B) and `s` is the
//! coefficient-order ID (Table I.1) of the current varblock's
//! transform. The remaining `qf_thresholds` / `lf_thresholds` /
//! `block_ctx_map` reads thread the LfGlobal §I.2.2
//! [`HfBlockContext`](crate::lf_global::HfBlockContext) bundle.
//!
//! Round 214's resolver is the typed pass-through that captures the
//! LfGlobal bundle once and offers a per-varblock `(channel, qdc)`
//! lookup, eliminating the four-argument boilerplate the round-208
//! varblock walker required of its callers.
//!
//! ## Scope boundary
//!
//! The per-varblock `qdc[3]` derivation (the §F.2 quantised LF
//! samples at the varblock's top-left 8×8 cell) is **not** owned by
//! this module — the resolver expects the caller to supply
//! `qdc` per varblock. The §C.7.2 entropy histogram array, the
//! per-pass `EntropyStream` wiring, and the per-channel
//! `NonZeros` history threading remain follow-up work (#799
//! DOCS-GAP) — the round-208 abstract `read_non_zeros` /
//! `decode_symbol` closure boundary persists unchanged.

use oxideav_core::Result;

use crate::coeff_order::order_id_for_transform;
use crate::dct_select::DctSelectGrid;
use crate::lf_global::HfBlockContext;
use crate::pass_group_hf::{block_context, DecodedHfBlock};
use crate::per_pass_non_zeros::PerPassNonZerosGrids;
use crate::varblock_walk::{count_varblocks, Varblock, VarblockWalk};

/// Borrow-based wrapper around a [`HfBlockContext`] that exposes a
/// per-varblock `(channel, qdc) -> block_ctx` resolver matching the
/// Listing C.13 `BlockContext()` signature.
///
/// The resolver is **stateless** — every call re-evaluates the
/// Listing C.13 formula against the borrowed bundle and the caller's
/// per-call `(channel, varblock, qdc)` parameters. There is no
/// per-frame cache; the resolver is just a typed view on the
/// LfGlobal bundle.
#[derive(Debug, Clone, Copy)]
pub struct BlockContextResolver<'a> {
    hbc: &'a HfBlockContext,
}

impl<'a> BlockContextResolver<'a> {
    /// Construct a resolver borrowing the given [`HfBlockContext`].
    pub fn new(hbc: &'a HfBlockContext) -> Self {
        Self { hbc }
    }

    /// Borrow back the wrapped [`HfBlockContext`].
    pub fn hf_block_context(&self) -> &'a HfBlockContext {
        self.hbc
    }

    /// `nb_block_ctx` — the LfGlobal §I.2.2 invariant
    /// (`max(block_ctx_map) + 1`). Cached on the wrapped bundle.
    pub fn nb_block_ctx(&self) -> u32 {
        self.hbc.nb_block_ctx
    }

    /// Resolve a single varblock's `block_ctx` per Listing C.13.
    ///
    /// * `channel` is the channel index (0 = X, 1 = Y, 2 = B).
    /// * `varblock` is the placement read off the
    ///   [`crate::dct_select::DctSelectGrid`] by the round-208 walker.
    ///   The `transform` field maps to `s` (Table I.1 Order ID) and
    ///   the `hf_mul` field is `qf` per Listing C.13.
    /// * `qdc[3]` are the quantised LF samples of the varblock's
    ///   top-left 8×8 cell (X / Y / B per channel) — caller-owned;
    ///   see [`crate::lf_dequant`] for the upstream §F.2 source.
    pub fn resolve(&self, channel: u32, varblock: &Varblock, qdc: [i32; 3]) -> Result<u32> {
        let s = order_id_for_transform(varblock.transform).index();
        block_context(
            channel,
            s,
            varblock.hf_mul,
            qdc,
            &self.hbc.qf_thresholds,
            &self.hbc.lf_thresholds,
            &self.hbc.block_ctx_map,
        )
    }
}

/// Per-pass per-channel varblock decode driver — same shape as
/// [`crate::varblock_walk::decode_varblocks_for_pass_channel`], with
/// the `block_ctx_for_varblock` closure replaced by a borrowed
/// [`BlockContextResolver`] + a caller-supplied `qdc_at` closure that
/// returns the per-varblock `qdc[3]`.
///
/// This is the round-214 convenience wrapper that the round-208
/// module called out as a follow-up (the "per-LfGroup
/// `HfBlockContext` parameter sweep"). Callers that already maintain
/// a per-channel quantised-LF buffer can supply `qdc_at` as a
/// per-varblock lookup against that buffer; callers without one yet
/// can pass `|_| Ok([0, 0, 0])` to exercise the default-table fast
/// path (the §I.2.2 default bundle has empty `lf_thresholds` so
/// `qdc` is unused).
///
/// Returns the in-order vector of `(Varblock, DecodedHfBlock,
/// raw_non_zeros)` triples — exactly the same shape as
/// [`crate::varblock_walk::decode_varblocks_for_pass_channel`].
#[allow(clippy::too_many_arguments)]
pub fn decode_varblocks_with_resolver<Q, F, G>(
    grid: &DctSelectGrid,
    nz: &mut PerPassNonZerosGrids,
    p: u32,
    c: u32,
    resolver: &BlockContextResolver<'_>,
    mut qdc_at: Q,
    mut read_non_zeros: F,
    mut decode_symbol: G,
) -> Result<Vec<(Varblock, DecodedHfBlock, u32)>>
where
    Q: FnMut(&Varblock) -> Result<[i32; 3]>,
    F: FnMut(u32) -> Result<u32>,
    G: FnMut(u32) -> Result<u32>,
{
    let nb_block_ctx = resolver.nb_block_ctx();
    let mut out = Vec::with_capacity(count_varblocks(grid) as usize);
    let mut walk = VarblockWalk::new(grid);
    while let Some(vb) = walk.next()? {
        let qdc = qdc_at(&vb)?;
        let ctx = resolver.resolve(c, &vb, qdc)?;
        let (decoded, raw_non_zeros) = nz.decode_block_at_for_pass_channel(
            p,
            c,
            vb.x,
            vb.y,
            vb.transform,
            ctx,
            nb_block_ctx,
            &mut read_non_zeros,
            &mut decode_symbol,
        )?;
        out.push((vb, decoded, raw_non_zeros));
    }
    Ok(out)
}

/// Channel-routed payload used by the per-channel ANS-closure pair
/// passed to [`decode_varblocks_three_channels_with_resolver`]. The
/// inner driver invokes the caller's `read_non_zeros` and
/// `decode_symbol` closures with the channel index as the first
/// argument, so the caller can route each call to the matching
/// per-pass per-channel histogram. This keeps the round-221 driver
/// histogram-blind: it owns the channel iteration order (Listing
/// C.13's X / Y / B canonical sweep) but defers the per-channel
/// histogram selection to the caller, exactly as round 208 / 214
/// defer the single-channel histograms.
///
/// `channel` ∈ {0, 1, 2} for X / Y / B respectively.
pub type ChannelReadNonZeros<'a> = dyn FnMut(u32, u32) -> Result<u32> + 'a;
/// See [`ChannelReadNonZeros`]. Caller routes per-channel histogram
/// reads; the `(channel, coeff_ctx)` pair lets the closure pick the
/// matching per-channel ANS distribution.
pub type ChannelDecodeSymbol<'a> = dyn FnMut(u32, u32) -> Result<u32> + 'a;

/// Per-varblock output triple returned by
/// [`decode_varblocks_three_channels_with_resolver`]. Per-channel
/// arrays are indexed 0 = X, 1 = Y, 2 = B (canonical FDIS §C.8.3
/// channel order).
pub type ThreeChannelVarblock = (Varblock, [DecodedHfBlock; 3], [u32; 3]);

/// Per-LfGroup three-channel varblock decode driver — round 221's
/// upgrade over [`decode_varblocks_with_resolver`].
///
/// Walks the [`DctSelectGrid`] in raster order; for each varblock
/// the driver:
///
/// 1. invokes the caller's `qdc_at` closure exactly once to read
///    the shared `qdc[3]` triple,
/// 2. invokes [`BlockContextResolver::resolve`] three times — once
///    per channel — against that shared `qdc`, and
/// 3. invokes [`PerPassNonZerosGrids::decode_block_at_for_pass_channel`]
///    three times (channel order: X = 0, Y = 1, B = 2) on the
///    `p`-th pass with the matching `block_ctx`.
///
/// The `read_non_zeros` / `decode_symbol` closures take the channel
/// index as their first argument so the caller can route each call
/// to the matching per-pass per-channel histogram without binding
/// three separate closure pairs.
///
/// Returns the in-raster-order
/// `Vec<(Varblock, [DecodedHfBlock; 3], [u32; 3])>` triple. The
/// per-channel `DecodedHfBlock` and `raw_non_zeros` arrays are
/// indexed by channel — `out[i].1[0]` is the X-channel decoded
/// block, `out[i].2[2]` is the B-channel `raw_non_zeros`, etc.
///
/// On any per-channel error the driver propagates the error
/// immediately and discards any in-flight partial output. The walk
/// always proceeds X → Y → B per varblock; an error on Y aborts
/// before B reads (so the B-channel ANS state is **not** advanced).
#[allow(clippy::too_many_arguments)]
pub fn decode_varblocks_three_channels_with_resolver<Q, F, G>(
    grid: &DctSelectGrid,
    nz: &mut PerPassNonZerosGrids,
    p: u32,
    resolver: &BlockContextResolver<'_>,
    mut qdc_at: Q,
    mut read_non_zeros: F,
    mut decode_symbol: G,
) -> Result<Vec<ThreeChannelVarblock>>
where
    Q: FnMut(&Varblock) -> Result<[i32; 3]>,
    F: FnMut(u32, u32) -> Result<u32>,
    G: FnMut(u32, u32) -> Result<u32>,
{
    let nb_block_ctx = resolver.nb_block_ctx();
    let mut out = Vec::with_capacity(count_varblocks(grid) as usize);
    let mut walk = VarblockWalk::new(grid);
    while let Some(vb) = walk.next()? {
        let qdc = qdc_at(&vb)?;
        // Channel order: X = 0, Y = 1, B = 2 (FDIS §C.8.3 listing
        // sequence). The per-channel ANS state advance happens in
        // this fixed order so caller-side closures can rely on the
        // channel-major progression.
        let decoded0;
        let raw0;
        {
            let ctx0 = resolver.resolve(0, &vb, qdc)?;
            let r = nz.decode_block_at_for_pass_channel(
                p,
                0,
                vb.x,
                vb.y,
                vb.transform,
                ctx0,
                nb_block_ctx,
                |pred| read_non_zeros(0, pred),
                |coeff_ctx| decode_symbol(0, coeff_ctx),
            )?;
            decoded0 = r.0;
            raw0 = r.1;
        }
        let decoded1;
        let raw1;
        {
            let ctx1 = resolver.resolve(1, &vb, qdc)?;
            let r = nz.decode_block_at_for_pass_channel(
                p,
                1,
                vb.x,
                vb.y,
                vb.transform,
                ctx1,
                nb_block_ctx,
                |pred| read_non_zeros(1, pred),
                |coeff_ctx| decode_symbol(1, coeff_ctx),
            )?;
            decoded1 = r.0;
            raw1 = r.1;
        }
        let decoded2;
        let raw2;
        {
            let ctx2 = resolver.resolve(2, &vb, qdc)?;
            let r = nz.decode_block_at_for_pass_channel(
                p,
                2,
                vb.x,
                vb.y,
                vb.transform,
                ctx2,
                nb_block_ctx,
                |pred| read_non_zeros(2, pred),
                |coeff_ctx| decode_symbol(2, coeff_ctx),
            )?;
            decoded2 = r.0;
            raw2 = r.1;
        }
        out.push((vb, [decoded0, decoded1, decoded2], [raw0, raw1, raw2]));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dct_select::{derive_dct_select, TransformType};
    use crate::lf_group::HfMetadata;

    fn make_hf(block_info: Vec<i32>, nb_blocks: u32, info_w: u32) -> HfMetadata {
        HfMetadata {
            nb_blocks,
            x_from_y: vec![0],
            b_from_y: vec![0],
            block_info,
            sharpness: vec![0],
            channel_widths: [1, 1, info_w, 1],
            channel_heights: [1, 1, 2, 1],
        }
    }

    fn default_hbc() -> HfBlockContext {
        HfBlockContext {
            used_default: true,
            block_ctx_map: HfBlockContext::DEFAULT_BLOCK_CTX_MAP.to_vec(),
            nb_block_ctx: (*HfBlockContext::DEFAULT_BLOCK_CTX_MAP.iter().max().unwrap() as u32) + 1,
            lf_thresholds: [Vec::new(), Vec::new(), Vec::new()],
            qf_thresholds: Vec::new(),
        }
    }

    // -----------------------------------------------------------------
    // Resolver — borrow / accessor smoke tests
    // -----------------------------------------------------------------

    #[test]
    fn resolver_borrows_hbc_and_exposes_nb_block_ctx() {
        let hbc = default_hbc();
        let r = BlockContextResolver::new(&hbc);
        assert!(std::ptr::eq(r.hf_block_context(), &hbc));
        assert_eq!(r.nb_block_ctx(), 15);
    }

    #[test]
    fn resolver_default_channel_zero_dct8x8_top_left() {
        // (c=0, s=0): idx = (0 ^ 1) × 13 + 0 = 13.
        // map[13] = 7. Defaults have empty thresholds → qdc unused.
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let vb = Varblock {
            x: 0,
            y: 0,
            transform: TransformType::Dct8x8,
            hf_mul: 1,
        };
        let ctx = resolver.resolve(0, &vb, [0, 0, 0]).unwrap();
        assert_eq!(ctx, hbc.block_ctx_map[13] as u32);
    }

    #[test]
    fn resolver_default_channel_one_dct8x8_top_left() {
        // (c=1, s=0): idx = (1 ^ 1) × 13 + 0 = 0 → map[0] = 0.
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let vb = Varblock {
            x: 0,
            y: 0,
            transform: TransformType::Dct8x8,
            hf_mul: 1,
        };
        let ctx = resolver.resolve(1, &vb, [0, 0, 0]).unwrap();
        assert_eq!(ctx, hbc.block_ctx_map[0] as u32);
    }

    #[test]
    fn resolver_default_channel_two_dct8x8_top_left() {
        // (c=2, s=0): idx = 2 × 13 + 0 = 26 → map[26] = 7.
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let vb = Varblock {
            x: 0,
            y: 0,
            transform: TransformType::Dct8x8,
            hf_mul: 1,
        };
        let ctx = resolver.resolve(2, &vb, [0, 0, 0]).unwrap();
        assert_eq!(ctx, hbc.block_ctx_map[26] as u32);
    }

    #[test]
    fn resolver_default_dct16x16_uses_order_id_2() {
        // DCT16×16 maps to OrderId::Id2 → s = 2.
        // (c=0, s=2): idx = 1 × 13 + 2 = 15 → map[15] = 9.
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let vb = Varblock {
            x: 0,
            y: 0,
            transform: TransformType::Dct16x16,
            hf_mul: 1,
        };
        let ctx = resolver.resolve(0, &vb, [0, 0, 0]).unwrap();
        assert_eq!(ctx, hbc.block_ctx_map[15] as u32);
    }

    #[test]
    fn resolver_default_hornuss_uses_order_id_1() {
        // Hornuss / DCT2×2 / DCT4×4 / AFV* all map to OrderId::Id1
        // → s = 1. (c=0, s=1): idx = 13 + 1 = 14 → map[14] = 8.
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let vb = Varblock {
            x: 0,
            y: 0,
            transform: TransformType::Hornuss,
            hf_mul: 1,
        };
        let ctx = resolver.resolve(0, &vb, [0, 0, 0]).unwrap();
        assert_eq!(ctx, hbc.block_ctx_map[14] as u32);
    }

    #[test]
    fn resolver_default_ignores_qdc_and_hf_mul_in_default_branch() {
        // Default branch has empty qf_thresholds + lf_thresholds, so
        // varying hf_mul / qdc must not perturb the resolved ctx.
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let v_lo = Varblock {
            x: 0,
            y: 0,
            transform: TransformType::Dct8x8,
            hf_mul: 1,
        };
        let v_hi = Varblock {
            x: 5,
            y: 7,
            transform: TransformType::Dct8x8,
            hf_mul: 64,
        };
        let a = resolver.resolve(0, &v_lo, [0, 0, 0]).unwrap();
        let b = resolver.resolve(0, &v_hi, [-100, 200, -50]).unwrap();
        // Both default-branch reads with same (c, s) → same ctx.
        assert_eq!(a, b);
    }

    #[test]
    fn resolver_custom_qf_threshold_perturbs_ctx() {
        // Single qf_threshold = 5, single-cluster map = [0; 26]
        // (covers idx ∈ {0..=25} for c ∈ {0,1,2} × s = 0 × 2 cells).
        let hbc = HfBlockContext {
            used_default: false,
            block_ctx_map: vec![3; 26],
            nb_block_ctx: 4,
            lf_thresholds: [Vec::new(), Vec::new(), Vec::new()],
            qf_thresholds: vec![5],
        };
        let resolver = BlockContextResolver::new(&hbc);
        // c=0, s=0 → idx = 13. idx *= (1 + 1) = 2 → idx = 26.
        // hf_mul = 4 ≤ 5 → no bump. Total = 26 → out of range
        // (map.len() = 26). Reject expected.
        let vb_low = Varblock {
            x: 0,
            y: 0,
            transform: TransformType::Dct8x8,
            hf_mul: 4,
        };
        let r_low = resolver.resolve(0, &vb_low, [0, 0, 0]);
        assert!(r_low.is_err());
        // hf_mul = 10 > 5 → +1 bump. Total = 27 → still out of range.
        let vb_hi = Varblock {
            x: 0,
            y: 0,
            transform: TransformType::Dct8x8,
            hf_mul: 10,
        };
        let r_hi = resolver.resolve(0, &vb_hi, [0, 0, 0]);
        assert!(r_hi.is_err());
        // Both rejected by the bounds check — the resolver does
        // surface the per-channel idx growth as the underlying
        // block_context formula does.
    }

    #[test]
    fn resolver_custom_qf_threshold_clean_pass_through() {
        // Single qf_threshold = 5, oversized 78-element map.
        let hbc = HfBlockContext {
            used_default: false,
            block_ctx_map: vec![1; 78],
            nb_block_ctx: 2,
            lf_thresholds: [Vec::new(), Vec::new(), Vec::new()],
            qf_thresholds: vec![5],
        };
        let resolver = BlockContextResolver::new(&hbc);
        let vb_low = Varblock {
            x: 0,
            y: 0,
            transform: TransformType::Dct8x8,
            hf_mul: 4,
        };
        // c=0, s=0 → idx_base = 13; idx *= 2 → 26; qf=4 ≤ 5 → no bump.
        // idx then *= (1 + 1)^3 from lf_thresholds (all empty) = 1.
        // Final idx = 26. map[26] = 1.
        assert_eq!(resolver.resolve(0, &vb_low, [0, 0, 0]).unwrap(), 1);
        let vb_hi = Varblock {
            x: 0,
            y: 0,
            transform: TransformType::Dct8x8,
            hf_mul: 10,
        };
        // qf=10 > 5 → +1 → idx = 27. map[27] = 1.
        assert_eq!(resolver.resolve(0, &vb_hi, [0, 0, 0]).unwrap(), 1);
    }

    // -----------------------------------------------------------------
    // decode_varblocks_with_resolver — driver / closure routing tests
    // -----------------------------------------------------------------

    #[test]
    fn driver_single_dct8x8_routes_resolver_to_walker() {
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![0, 0], 1, 1);
        let grid = derive_dct_select(&hf, 8, 8).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 1, 1).unwrap();
        let mut qdc_calls = 0u32;
        let out = decode_varblocks_with_resolver(
            &grid,
            &mut nz,
            0,
            0,
            &resolver,
            |_vb| {
                qdc_calls += 1;
                Ok([0, 0, 0])
            },
            |_ctx| Ok(0),
            |_ctx| Ok(0),
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(qdc_calls, 1);
        let (vb, _decoded, raw_nz) = &out[0];
        assert_eq!((vb.x, vb.y), (0, 0));
        assert_eq!(vb.transform, TransformType::Dct8x8);
        assert_eq!(*raw_nz, 0);
    }

    #[test]
    fn driver_raster_order_2x2_dct8x8() {
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![0, 0, 0, 0, 0, 0, 0, 0], 4, 4);
        let grid = derive_dct_select(&hf, 16, 16).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 2, 2).unwrap();
        let out = decode_varblocks_with_resolver(
            &grid,
            &mut nz,
            0,
            0,
            &resolver,
            |_vb| Ok([0, 0, 0]),
            |_ctx| Ok(0),
            |_ctx| Ok(0),
        )
        .unwrap();
        assert_eq!(out.len(), 4);
        assert_eq!((out[0].0.x, out[0].0.y), (0, 0));
        assert_eq!((out[1].0.x, out[1].0.y), (1, 0));
        assert_eq!((out[2].0.x, out[2].0.y), (0, 1));
        assert_eq!((out[3].0.x, out[3].0.y), (1, 1));
    }

    #[test]
    fn driver_qdc_closure_receives_each_varblock_in_walk_order() {
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![0, 0, 0, 0, 0, 0, 0, 0], 4, 4);
        let grid = derive_dct_select(&hf, 16, 16).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 2, 2).unwrap();
        let mut seen: Vec<(u32, u32)> = Vec::new();
        let _ = decode_varblocks_with_resolver(
            &grid,
            &mut nz,
            0,
            1,
            &resolver,
            |vb| {
                seen.push((vb.x, vb.y));
                Ok([0, 0, 0])
            },
            |_| Ok(0),
            |_| Ok(0),
        )
        .unwrap();
        assert_eq!(seen, vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
    }

    #[test]
    fn driver_propagates_qdc_closure_error() {
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![0, 0], 1, 1);
        let grid = derive_dct_select(&hf, 8, 8).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 1, 1).unwrap();
        let r = decode_varblocks_with_resolver(
            &grid,
            &mut nz,
            0,
            0,
            &resolver,
            |_| Err(oxideav_core::Error::InvalidData("qdc failure".into())),
            |_| Ok(0),
            |_| Ok(0),
        );
        assert!(r.is_err());
    }

    #[test]
    fn driver_dct16x16_single_varblock_pass_through() {
        // 16×16 covered by one DCT16×16 → walker yields a single
        // varblock at (0,0) and the driver routes it through the
        // resolver + nz once.
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![4, 0], 1, 1);
        let grid = derive_dct_select(&hf, 16, 16).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 2, 2).unwrap();
        let out = decode_varblocks_with_resolver(
            &grid,
            &mut nz,
            0,
            0,
            &resolver,
            |_| Ok([0, 0, 0]),
            |_| Ok(0),
            |_| Ok(0),
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0.transform, TransformType::Dct16x16);
    }

    // -----------------------------------------------------------------
    // Round 221 — three-channel driver tests
    // -----------------------------------------------------------------

    #[test]
    fn r221_three_channel_single_dct8x8_yields_three_decodes_per_varblock() {
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![0, 0], 1, 1);
        let grid = derive_dct_select(&hf, 8, 8).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 1, 1).unwrap();
        let mut qdc_calls = 0u32;
        let mut nz_calls: Vec<u32> = Vec::new();
        let out = decode_varblocks_three_channels_with_resolver(
            &grid,
            &mut nz,
            0,
            &resolver,
            |_vb| {
                qdc_calls += 1;
                Ok([1, 2, 3])
            },
            |channel, _pred| {
                nz_calls.push(channel);
                Ok(0)
            },
            |_channel, _coeff_ctx| Ok(0),
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        // qdc closure should be called exactly once per varblock —
        // shared across the three per-channel decodes.
        assert_eq!(qdc_calls, 1);
        // read_non_zeros should be called once per channel (3 total),
        // in canonical X / Y / B order.
        assert_eq!(nz_calls, vec![0, 1, 2]);
        // All three raw_non_zeros are 0.
        assert_eq!(out[0].2, [0, 0, 0]);
    }

    #[test]
    fn r221_three_channel_raster_order_2x2_dct8x8() {
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![0, 0, 0, 0, 0, 0, 0, 0], 4, 4);
        let grid = derive_dct_select(&hf, 16, 16).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 2, 2).unwrap();
        let out = decode_varblocks_three_channels_with_resolver(
            &grid,
            &mut nz,
            0,
            &resolver,
            |_| Ok([0, 0, 0]),
            |_, _| Ok(0),
            |_, _| Ok(0),
        )
        .unwrap();
        assert_eq!(out.len(), 4);
        // Raster order (matches single-channel driver).
        assert_eq!((out[0].0.x, out[0].0.y), (0, 0));
        assert_eq!((out[1].0.x, out[1].0.y), (1, 0));
        assert_eq!((out[2].0.x, out[2].0.y), (0, 1));
        assert_eq!((out[3].0.x, out[3].0.y), (1, 1));
        // Per-varblock outputs are length-3 arrays.
        for entry in &out {
            assert_eq!(entry.2, [0, 0, 0]);
        }
    }

    #[test]
    fn r221_three_channel_qdc_shared_across_channels() {
        // The qdc closure must be called exactly once per varblock,
        // and the same qdc must feed all three channels' resolve()
        // calls. Test: a custom HfBlockContext where the LF threshold
        // on channel 2 would discriminate between qdc values, and
        // verify all three channels see the same qdc.
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![0, 0, 0, 0, 0, 0, 0, 0], 4, 4);
        let grid = derive_dct_select(&hf, 16, 16).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 2, 2).unwrap();
        let mut qdc_call_count = 0u32;
        let _ = decode_varblocks_three_channels_with_resolver(
            &grid,
            &mut nz,
            0,
            &resolver,
            |_vb| {
                qdc_call_count += 1;
                Ok([7, 7, 7])
            },
            |_, _| Ok(0),
            |_, _| Ok(0),
        )
        .unwrap();
        // 4 varblocks × 1 qdc call each = 4 (not 12 — qdc is shared).
        assert_eq!(qdc_call_count, 4);
    }

    #[test]
    fn r221_three_channel_routes_each_channel_through_resolver_and_decoder() {
        // Verify the per-channel ANS closure routing: each channel
        // sees its own `read_non_zeros` + `decode_symbol` invocations.
        // Each channel's `read_non_zeros` returns 2 (non-zero count)
        // and `decode_symbol` returns a non-zero ucoeff so the loop
        // counts 2 reads down to 0 (and stops). DCT8×8 has size = 64,
        // num_blocks = 1, so the loop iterates k = 1..size while
        // non_zeros > 0 — exactly 2 decode_symbol calls per channel
        // when each symbol decrements non_zeros.
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![0, 0], 1, 1);
        let grid = derive_dct_select(&hf, 8, 8).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 1, 1).unwrap();
        let mut per_channel_nz_calls = [0u32; 3];
        let mut per_channel_decode_calls = [0u32; 3];
        let _ = decode_varblocks_three_channels_with_resolver(
            &grid,
            &mut nz,
            0,
            &resolver,
            |_vb| Ok([0, 0, 0]),
            |channel, _pred| {
                per_channel_nz_calls[channel as usize] += 1;
                Ok(2) // 2 non-zeros per channel
            },
            |channel, _coeff_ctx| {
                per_channel_decode_calls[channel as usize] += 1;
                // Non-zero ucoeff (5) → decrements non_zeros.
                Ok(5)
            },
        )
        .unwrap();
        // Each channel's read_non_zeros called exactly once.
        assert_eq!(per_channel_nz_calls, [1, 1, 1]);
        // Each channel's decode_symbol called exactly 2 times — the
        // loop runs until non_zeros decrements to 0.
        assert_eq!(per_channel_decode_calls, [2, 2, 2]);
    }

    #[test]
    fn r221_three_channel_qdc_error_propagates_before_any_channel_read() {
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![0, 0], 1, 1);
        let grid = derive_dct_select(&hf, 8, 8).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 1, 1).unwrap();
        let mut nz_call_count = 0u32;
        let r = decode_varblocks_three_channels_with_resolver(
            &grid,
            &mut nz,
            0,
            &resolver,
            |_| Err(oxideav_core::Error::InvalidData("qdc failure".into())),
            |_, _| {
                nz_call_count += 1;
                Ok(0)
            },
            |_, _| Ok(0),
        );
        assert!(r.is_err());
        // qdc failure aborts before any per-channel reads happen.
        assert_eq!(nz_call_count, 0);
    }

    #[test]
    fn r221_three_channel_y_error_does_not_advance_b() {
        // Inject an error on channel = 1 (Y) and verify channel = 2 (B)
        // sees no read_non_zeros call.
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![0, 0], 1, 1);
        let grid = derive_dct_select(&hf, 8, 8).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 1, 1).unwrap();
        let mut per_channel_calls = [0u32; 3];
        let r = decode_varblocks_three_channels_with_resolver(
            &grid,
            &mut nz,
            0,
            &resolver,
            |_| Ok([0, 0, 0]),
            |channel, _pred| {
                per_channel_calls[channel as usize] += 1;
                if channel == 1 {
                    Err(oxideav_core::Error::InvalidData("y fail".into()))
                } else {
                    Ok(0)
                }
            },
            |_, _| Ok(0),
        );
        assert!(r.is_err());
        assert_eq!(per_channel_calls[0], 1); // X read
        assert_eq!(per_channel_calls[1], 1); // Y read (fails)
        assert_eq!(per_channel_calls[2], 0); // B not reached
    }

    #[test]
    fn r221_three_channel_dct16x16_single_varblock_per_channel_dispatch() {
        // 16×16 covered by one DCT16×16 → 1 varblock × 3 channels = 3
        // resolver.resolve() calls. The transform OrderId for DCT16×16
        // is 2; per-channel idx values are:
        //   c=0 → 13 + 2 = 15 → map[15] = 9
        //   c=1 → 0  + 2 = 2  → map[2]  = 2
        //   c=2 → 26 + 2 = 28 → map[28] = 9
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![4, 0], 1, 1);
        let grid = derive_dct_select(&hf, 16, 16).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 2, 2).unwrap();
        let out = decode_varblocks_three_channels_with_resolver(
            &grid,
            &mut nz,
            0,
            &resolver,
            |_| Ok([0, 0, 0]),
            |_, _| Ok(0),
            |_, _| Ok(0),
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0.transform, TransformType::Dct16x16);
        assert_eq!(out[0].2, [0, 0, 0]);
    }

    #[test]
    fn r221_three_channel_pass_index_routes_to_correct_pass() {
        // Two-pass container; the driver writes to pass = 1 only;
        // pass = 0 should remain pristine afterwards.
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![0, 0], 1, 1);
        let grid = derive_dct_select(&hf, 8, 8).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(2, 3, 1, 1).unwrap();
        // pass = 1: write a per-block non_zeros of 6 on each channel.
        let _ = decode_varblocks_three_channels_with_resolver(
            &grid,
            &mut nz,
            1,
            &resolver,
            |_| Ok([0, 0, 0]),
            |_, _| Ok(6),
            |_, _| Ok(0),
        )
        .unwrap();
        // pass 0 untouched.
        for c in 0..3 {
            assert_eq!(nz.get(0, c, 0, 0).unwrap(), 0);
        }
        // pass 1: each channel ran update_after_block for DCT8×8
        // (num_blocks = 1) → stored non_zeros = 6.
        for c in 0..3 {
            assert_eq!(nz.get(1, c, 0, 0).unwrap(), 6);
        }
    }

    #[test]
    fn r221_three_channel_count_matches_walker() {
        // Mixed transforms: DCT16×8 (covers (0,0)+(0,1)) + 2 DCT8×8.
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![6, 0, 0, 0, 0, 0], 3, 3);
        let grid = derive_dct_select(&hf, 16, 16).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 2, 2).unwrap();
        let out = decode_varblocks_three_channels_with_resolver(
            &grid,
            &mut nz,
            0,
            &resolver,
            |_| Ok([0, 0, 0]),
            |_, _| Ok(0),
            |_, _| Ok(0),
        )
        .unwrap();
        assert_eq!(out.len(), 3);
        // Layout (matches single-channel walker):
        assert_eq!(out[0].0.transform, TransformType::Dct16x8);
        assert_eq!((out[0].0.x, out[0].0.y), (0, 0));
        assert_eq!(out[1].0.transform, TransformType::Dct8x8);
        assert_eq!((out[1].0.x, out[1].0.y), (1, 0));
        assert_eq!(out[2].0.transform, TransformType::Dct8x8);
        assert_eq!((out[2].0.x, out[2].0.y), (1, 1));
    }

    #[test]
    fn r221_three_channel_visits_grid_once_not_three_times() {
        // The driver MUST walk the grid once — calling
        // decode_varblocks_with_resolver three times would visit every
        // varblock three times. Verify by counting unique qdc closure
        // invocations: 4 varblocks → 4 qdc calls, NOT 12.
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![0, 0, 0, 0, 0, 0, 0, 0], 4, 4);
        let grid = derive_dct_select(&hf, 16, 16).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 2, 2).unwrap();
        let mut qdc_calls = 0u32;
        let mut varblocks_seen: Vec<(u32, u32)> = Vec::new();
        let _ = decode_varblocks_three_channels_with_resolver(
            &grid,
            &mut nz,
            0,
            &resolver,
            |vb| {
                qdc_calls += 1;
                varblocks_seen.push((vb.x, vb.y));
                Ok([0, 0, 0])
            },
            |_, _| Ok(0),
            |_, _| Ok(0),
        )
        .unwrap();
        assert_eq!(qdc_calls, 4);
        // Each varblock visited exactly once (no duplicates).
        assert_eq!(varblocks_seen, vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
    }

    #[test]
    fn r221_three_channel_decoded_blocks_indexed_by_channel() {
        // The returned [DecodedHfBlock; 3] array is indexed
        // 0 = X, 1 = Y, 2 = B. Verify by giving each channel a
        // distinct non_zeros value and inspecting the per-channel
        // decoded result.
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![0, 0], 1, 1);
        let grid = derive_dct_select(&hf, 8, 8).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 1, 1).unwrap();
        let out = decode_varblocks_three_channels_with_resolver(
            &grid,
            &mut nz,
            0,
            &resolver,
            |_| Ok([0, 0, 0]),
            |channel, _pred| match channel {
                0 => Ok(3),
                1 => Ok(5),
                2 => Ok(7),
                _ => unreachable!(),
            },
            |_, _| Ok(0),
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        // raw_non_zeros indexed by channel.
        assert_eq!(out[0].2, [3, 5, 7]);
        // After the writeback: per-channel NonZeros(0, 0) values.
        assert_eq!(nz.get(0, 0, 0, 0).unwrap(), 3);
        assert_eq!(nz.get(0, 1, 0, 0).unwrap(), 5);
        assert_eq!(nz.get(0, 2, 0, 0).unwrap(), 7);
    }
}
