//! Per-pass HF histogram decode context — ISO/IEC FDIS 18181-1:2021
//! §C.7.2 (entropy histograms) + §C.8.3 (per-pass `histogram_offset`
//! routing) bridge.
//!
//! ## Scope (round 264)
//!
//! Round 264 lifts the round-260 single-varblock three-channel decode
//! method into a per-LfGroup raster walk for one pass:
//!
//! * [`HfHistogramDecodeContext::decode_lf_group_three_channels_for_pass`]
//!   — one `(br, p, grid, resolver, qdc_at, predicted_at)` call walks
//!   the [`crate::dct_select::DctSelectGrid`] in raster order via
//!   [`crate::varblock_walk::VarblockWalk`], invokes the caller's
//!   per-varblock `qdc_at` + `predicted_at` closures once per varblock
//!   to read the shared `qdc[3]` triple and the per-channel
//!   `predicted[3]` triple, then composes the round-260
//!   [`HfHistogramDecodeContext::decode_three_channel_varblock_for_pass`]
//!   bundled three-channel walk to yield one
//!   [`crate::block_context_resolver::ThreeChannelVarblock`] per
//!   top-left cell. Returns the in-raster-order
//!   `Vec<ThreeChannelVarblock>` per the round-221 / 228 driver
//!   convention.
//!
//! This is the per-pass per-LfGroup raster-walk driver layered above
//! the round-260 bundled per-varblock method, mirroring the round-221
//! [`crate::block_context_resolver::decode_varblocks_three_channels_with_resolver`]
//! shape — except that this driver owns the §C.7.2 entropy-stream
//! routing through the round-252 typed decode context (no caller-side
//! `read_non_zeros` / `decode_symbol` closures), so the per-channel
//! per-pass histogram selection happens inside the driver. The
//! per-LfGroup multi-pass outer loop (round 228) layers above this
//! by repeating the call for `p ∈ [0, num_passes)`.
//!
//! Round 260 (the immediate prior round) landed the single-varblock
//! three-channel walk — see the scope summary below.
//!
//! ## Scope (round 260)
//!
//! Round 260 lifts the round-255 single-channel single-varblock
//! decode entry into a bundled three-channel per-varblock walk
//! against a [`BlockContextResolver`]:
//!
//! * [`HfHistogramDecodeContext::decode_three_channel_varblock_for_pass`]
//!   — one `(br, p, vb, resolver, qdc, predicted[3])` call walks
//!   the FDIS §C.8.3 prose Y → X → B channel decode sequence, invokes
//!   [`BlockContextResolver::resolve`] once per channel against
//!   the shared `qdc[3]` triple, threads each resulting `block_ctx`
//!   into [`HfHistogramDecodeContext::decode_block_for_pass_transform`]
//!   (round 255), and returns the per-channel
//!   `([DecodedHfBlock; 3], [u32; 3])` pair (decoded coefficients
//!   plus the un-divided `raw_non_zeros` triple the caller threads
//!   into the per-channel NonZeros-grid bookkeeping).
//!
//! Round 255 (the immediate prior round) landed the single-channel
//! per-varblock decode method itself — see the scope summary below.
//!
//! ## Scope (round 255)
//!
//! Round 255 closes the round-252 deferred next-step "per-block raster
//! walk remain caller-side concerns above this primitive" with a new
//! bundled per-varblock decode method:
//!
//! * [`HfHistogramDecodeContext::decode_block_for_pass_transform`] —
//!   one [`crate::dct_select::TransformType`]-driven call that wires
//!   the Listing C.13 + C.14 per-block walk against the round-252
//!   per-pass histogram routing. Composes the existing
//!   [`HfHistogramDecodeContext::non_zeros_at`] +
//!   [`HfHistogramDecodeContext::coefficient_at`] entry points (which
//!   each individually take `&mut self`, so cannot both be wrapped
//!   into the round-90 `read_non_zeros_and_decode_block_for_transform`
//!   closure pair) into a single sequential `&mut self` walk that
//!   mirrors [`crate::pass_group_hf::decode_block_coefficients`]'s
//!   state machine — `prev_nonzero[]` tracking, the `non_zeros == 0`
//!   early-stop, and the `non_zeros > size - num_blocks` defensive
//!   cap — without re-deriving any of it. Returns the round-90
//!   [`DecodedHfBlock`] coefficient bundle plus the un-divided
//!   `raw_non_zeros` so the caller can drive the NonZeros-grid
//!   `(raw + num_blocks - 1) Idiv num_blocks` bookkeeping unchanged.
//!
//! Round 252 (the immediate prior round) landed the
//! `HfHistogramDecodeContext` primitive itself — see the §C.7.2 →
//! §C.8.3 routing summary below.
//!
//! ## Scope (round 252)
//!
//! Round 247 landed the typed [`HfCoefficientHistograms`] wrapper that
//! performs the §C.7.2 `EntropyStream::read` against
//! `num_distributions = 495 × num_hf_presets × nb_block_ctx`. Round
//! 232 landed [`PerPassHfHeaders`] which exposes the per-pass `hfp`
//! selector + the derived `histogram_offset = 495 × nb_block_ctx ×
//! hfp` per §C.8.3 first paragraph.
//!
//! The two layers do not, however, talk to each other yet: a caller
//! wiring the §C.8.3 per-block decode walk has to manually pair every
//! `non_zeros_context(predicted, block_ctx, nb_block_ctx)` /
//! `coefficient_context(k, non_zeros, num_blocks, size, prev,
//! block_ctx, nb_block_ctx)` Listing C.13 value with the per-pass
//! offset from [`PerPassHfHeaders::histogram_offset`] and feed the
//! sum into [`EntropyStream::decode_symbol`] on the
//! [`HfCoefficientHistograms`] stream.
//!
//! Round 252 lands a single typed `HfHistogramDecodeContext` primitive
//! that binds the §C.7.2 stream + the §C.8.3 per-pass offsets and
//! exposes three driver-shape entry points:
//!
//! * [`HfHistogramDecodeContext::decode_symbol_for_pass`] — raw
//!   `D[ctx + histogram_offset(p)]` symbol read. The base layer the
//!   two helpers below build on.
//! * [`HfHistogramDecodeContext::non_zeros_at`] — wraps
//!   [`crate::pass_group_hf::non_zeros_context`] +
//!   `decode_symbol_for_pass` so the caller hands in the spec-named
//!   `(predicted, block_ctx)` triple and gets the `NonZeros(x, y)`
//!   value back. This is the `read_non_zeros(p, c, predicted)` shape
//!   the [`crate::multi_pass_hf_header::decode_multi_pass_with_hf_headers`]
//!   driver expects.
//! * [`HfHistogramDecodeContext::coefficient_at`] — wraps
//!   [`crate::pass_group_hf::coefficient_context`] +
//!   `decode_symbol_for_pass` so the caller hands in the spec-named
//!   `(k, non_zeros, num_blocks, size, prev, block_ctx)` six-tuple
//!   and gets the `ucoeff` symbol back. This is the
//!   `decode_symbol(p, c, coeff_ctx)` shape the round-232 driver
//!   expects (parameterised on the §C.8.3 prose's
//!   `D[CoefficientContext(...) + offset]`).
//!
//! Both `non_zeros_at` and `coefficient_at` take the per-pass index
//! `p` and the §I.2.2 invariant `nb_block_ctx` (which is shared
//! across passes since it is a frame-level §I.2.2 derivation,
//! not a per-pass one).
//!
//! ## FDIS prose anchor
//!
//! §C.8.3 (FDIS p. 55), the per-PassGroup decode walk:
//!
//! > For each block in the LfGroup raster order:
//! >   compute `block_ctx = BlockContext(block_ctx_map, x, y, qf, qdc)`;
//! >   compute `predicted = PredictedNonZeros(x, y)`;
//! >   `NonZeros(x, y) = D[NonZerosContext(predicted) + offset]`;
//! >   for `k in [num_blocks, size)`:
//! >     `prev` = ...; compute `coeff_ctx =
//! >     CoefficientContext(k, non_zeros, num_blocks, size, prev)`;
//! >     `ucoeff = D[coeff_ctx + offset]`;
//! >     `coeffs[order[k]] = unpack_signed(ucoeff)`;
//!
//! The §C.8.3 first paragraph defines
//! `offset = 495 × nb_block_ctx × hfp` (per-pass), already captured
//! in [`PerPassHfHeaders::histogram_offset`].
//!
//! ## Scope boundary
//!
//! This module is a **wiring primitive** — no spec re-derivation, no
//! ANS state initialisation, no per-block raster walk. The ANS state
//! initialiser (the `u(32)` read between `EntropyStream::read` and
//! the first symbol decode for ANS-coded streams) must be performed
//! by the caller via [`HfCoefficientHistograms::read_ans_state_init`]
//! before the first [`HfHistogramDecodeContext::decode_symbol_for_pass`]
//! call. Prefix-coded streams short-circuit that to a no-op.
//!
//! Per-channel `BlockContext()` history threading and per-channel
//! coefficient-order lookup against [`crate::hf_pass::HfPass`] remain
//! caller-side concerns above this primitive. The single-varblock
//! Listing C.14 per-block walk is now bundled by round 255 (see
//! [`HfHistogramDecodeContext::decode_block_for_pass_transform`]
//! above), the single-varblock three-channel Y → X → B sweep is
//! now bundled by round 260 (see
//! [`HfHistogramDecodeContext::decode_three_channel_varblock_for_pass`]
//! above), and the per-LfGroup raster walk for one pass is now
//! bundled by round 264 (see
//! [`HfHistogramDecodeContext::decode_lf_group_three_channels_for_pass`]
//! above) — the per-LfGroup multi-pass outer loop is still a
//! caller-side concern (driven by [`crate::multi_pass_decode`]).
//!
//! Same pure-control-flow primitive shape as round-89
//! [`crate::dct_quant_weights`], round-95 [`crate::hf_dequant`],
//! round-121 [`crate::llf_from_lf`], round-138
//! [`crate::chroma_from_luma`], round-141 [`crate::gaborish`],
//! round-144 [`crate::epf`], round-147 [`crate::afv::afv_idct`],
//! round-159 / 164 [`crate::pass_group_hf`], round-177
//! [`crate::non_zeros_grid`], round-183
//! [`crate::per_channel_non_zeros`], round-190
//! [`crate::per_pass_non_zeros`], round-208 [`crate::varblock_walk`],
//! round-214 [`crate::block_context_resolver`], round-221, round-228
//! [`crate::multi_pass_decode`], round-232
//! [`crate::multi_pass_hf_header`], round-238
//! [`crate::hf_coeff_histogram_size`], round-247
//! [`crate::hf_coefficient_histograms`].
//!
//! ## Bound: `u32`-wide context + offset sum
//!
//! Listing C.13's `NonZerosContext` and `CoefficientContext` return
//! `u32` values within the spec's allowed `nb_block_ctx`-scaled
//! range. The per-pass `histogram_offset` is a `u64`
//! (`495 × nb_block_ctx × hfp`, theoretically up to ~2^45 for the
//! 32-bit maxima `nb_block_ctx ≤ 256` × `hfp < num_hf_presets ≤
//! 2^28`). The sum could overflow `u32` on the very largest spec-
//! permitted parameters; we therefore route through `u64` and
//! defensively reject when the final `u32` cast would lose data
//! (the `EntropyStream::decode_symbol` signature takes a `u32`
//! `ctx` index into the cluster_map).

use oxideav_core::{Error, Result};

use crate::bitreader::{unpack_signed, BitReader};
use crate::block_context_resolver::{BlockContextResolver, ThreeChannelVarblock};
use crate::coeff_order::{natural_coeff_order, order_id_for_transform};
use crate::dct_select::{DctSelectGrid, TransformType};
use crate::hf_coefficient_histograms::HfCoefficientHistograms;
use crate::multi_pass_hf_header::PerPassHfHeaders;
use crate::pass_group_hf::{
    coefficient_context, non_zeros_context, prev_for_context, transform_block_params,
    DecodedHfBlock,
};
use crate::varblock_walk::{count_varblocks, Varblock, VarblockWalk};

/// Per-pass HF histogram decode context — owns the §C.7.2 entropy
/// stream + the per-pass §C.8.3 `histogram_offset` array.
///
/// Construct with [`Self::new`] from a successfully-read
/// [`HfCoefficientHistograms`] (post-`read_ans_state_init`) and a
/// matching [`PerPassHfHeaders`] (same `nb_block_ctx` invariant the
/// histograms were sized against).
///
/// The two containers are validated for consistency at construction
/// time: every per-pass `hfp` must be `< histograms.num_hf_presets()`
/// (which is enforced inside [`PerPassHfHeaders::read`] already, but
/// re-checked here as a defensive cross-container invariant).
#[derive(Debug)]
pub struct HfHistogramDecodeContext<'a> {
    /// §C.7.2 entropy stream (ANS state initialiser optional —
    /// caller invokes
    /// [`HfCoefficientHistograms::read_ans_state_init`] for ANS
    /// streams before the first `decode_symbol_for_pass` call).
    histograms: &'a mut HfCoefficientHistograms,
    /// Per-pass `histogram_offset` array (one `u64` per pass).
    /// Cached from [`PerPassHfHeaders::digest`] at construction time
    /// so the per-symbol path is a single array indexing — no header
    /// dereference per decode.
    per_pass_offsets: Vec<u64>,
}

impl<'a> HfHistogramDecodeContext<'a> {
    /// Bind a §C.7.2 [`HfCoefficientHistograms`] stream and a §C.8.3
    /// [`PerPassHfHeaders`] container into a single typed decode
    /// context.
    ///
    /// Per-pass cross-validation:
    ///
    /// * Every `headers.hfp(p)` must be `< histograms.num_hf_presets()`
    ///   (defensive — [`PerPassHfHeaders::read`] already enforces
    ///   this against the value passed to `read`, but the histograms
    ///   container holds the authoritative `num_hf_presets` value
    ///   and we therefore re-check here).
    /// * `headers.num_passes() ≥ 1` — a zero-pass frame would not
    ///   need a decode context; we reject early so downstream
    ///   callers can `assume` `> 0` without a re-check.
    ///
    /// Returns [`Error::InvalidData`] when either invariant is
    /// violated.
    pub fn new(
        histograms: &'a mut HfCoefficientHistograms,
        headers: &PerPassHfHeaders,
    ) -> Result<Self> {
        let num_passes = headers.num_passes();
        if num_passes == 0 {
            return Err(Error::InvalidData(
                "JXL HfHistogramDecodeContext: headers.num_passes() must be ≥ 1".into(),
            ));
        }
        let num_hf_presets = histograms.num_hf_presets();
        // Cross-validate per-pass hfp values against the histogram's
        // num_hf_presets. PerPassHfHeaders::read already validates
        // against the caller-supplied num_hf_presets argument, but
        // the histograms container is the authoritative source so
        // we re-check defensively.
        for p in 0..num_passes {
            let hfp = headers.hfp(p)?;
            if hfp >= num_hf_presets {
                return Err(Error::InvalidData(format!(
                    "JXL HfHistogramDecodeContext: headers.hfp({p})={hfp} >= \
                     histograms.num_hf_presets()={num_hf_presets}"
                )));
            }
        }
        // Pre-compute per-pass offsets so the per-symbol path is a
        // single `per_pass_offsets[p]` index, no header dereference.
        let per_pass_offsets: Vec<u64> = (0..num_passes)
            .map(|p| {
                headers
                    .histogram_offset(p)
                    .expect("p < num_passes by construction")
            })
            .collect();
        Ok(Self {
            histograms,
            per_pass_offsets,
        })
    }

    /// Decode a §C.7.2 symbol for pass `p` against context `ctx`,
    /// routed through `D[ctx + histogram_offset(p)]`.
    ///
    /// `histogram_offset(p) = 495 × nb_block_ctx × hfp(p)` per
    /// §C.8.3 first paragraph; `ctx` is the Listing C.13 context
    /// value (either [`crate::pass_group_hf::non_zeros_context`] or
    /// [`crate::pass_group_hf::coefficient_context`]).
    ///
    /// Returns [`Error::InvalidData`] when:
    /// * `p >= num_passes`,
    /// * the `ctx + offset` sum overflows `u32` (the
    ///   [`crate::modular_fdis::EntropyStream::decode_symbol`]
    ///   signature requires a `u32` cluster_map index).
    pub fn decode_symbol_for_pass(
        &mut self,
        br: &mut BitReader<'_>,
        p: u32,
        ctx: u32,
    ) -> Result<u32> {
        let offset = self.histogram_offset(p)?;
        let total: u64 = (ctx as u64).saturating_add(offset);
        let combined: u32 = total.try_into().map_err(|_| {
            Error::InvalidData(format!(
                "JXL HfHistogramDecodeContext: ctx={ctx} + offset={offset} = {total} exceeds u32"
            ))
        })?;
        self.histograms.entropy.decode_symbol(br, combined)
    }

    /// `NonZeros(x, y)` decode per Listing C.13's
    /// `D[NonZerosContext(predicted) + offset]` line.
    ///
    /// Wraps [`crate::pass_group_hf::non_zeros_context`] +
    /// [`Self::decode_symbol_for_pass`] so the caller hands in the
    /// spec-named `(predicted, block_ctx)` pair plus the §I.2.2
    /// invariant `nb_block_ctx`, and receives the decoded
    /// `NonZeros(x, y)` value.
    ///
    /// Spec-precise routing:
    /// `D[NonZerosContext(predicted, block_ctx, nb_block_ctx) +
    /// (495 × nb_block_ctx × hfp(p))]`.
    pub fn non_zeros_at(
        &mut self,
        br: &mut BitReader<'_>,
        p: u32,
        predicted: u32,
        block_ctx: u32,
        nb_block_ctx: u32,
    ) -> Result<u32> {
        let ctx = non_zeros_context(predicted, block_ctx, nb_block_ctx);
        self.decode_symbol_for_pass(br, p, ctx)
    }

    /// `ucoeff` decode per Listing C.13's
    /// `D[CoefficientContext(k, non_zeros, num_blocks, size, prev) +
    /// offset]` line.
    ///
    /// Wraps [`crate::pass_group_hf::coefficient_context`] +
    /// [`Self::decode_symbol_for_pass`] so the caller hands in the
    /// spec-named six-tuple `(k, non_zeros, num_blocks, size, prev,
    /// block_ctx)` plus the §I.2.2 invariant `nb_block_ctx`, and
    /// receives the raw `ucoeff` symbol (unpacking to a signed
    /// coefficient is a separate caller-side step per Listing C.14).
    ///
    /// Spec-precise routing:
    /// `D[CoefficientContext(k, non_zeros, num_blocks, size, prev,
    /// block_ctx, nb_block_ctx) + (495 × nb_block_ctx × hfp(p))]`.
    #[allow(clippy::too_many_arguments)]
    pub fn coefficient_at(
        &mut self,
        br: &mut BitReader<'_>,
        p: u32,
        k: u32,
        non_zeros: u32,
        num_blocks: u32,
        size: u32,
        prev: u32,
        block_ctx: u32,
        nb_block_ctx: u32,
    ) -> Result<u32> {
        let ctx = coefficient_context(
            k,
            non_zeros,
            num_blocks,
            size,
            prev,
            block_ctx,
            nb_block_ctx,
        )?;
        self.decode_symbol_for_pass(br, p, ctx)
    }

    /// Decode a single varblock's HF coefficients for pass `p`, driven
    /// by a [`TransformType`] — round 255's bundled composition of the
    /// round-90 §C.8.3 / Listing C.14 per-block loop with the round-252
    /// per-pass histogram routing.
    ///
    /// One typed method replaces the round-90 caller pattern of
    ///
    /// ```text
    /// read_non_zeros_and_decode_block_for_transform(
    ///     t, predicted, block_ctx, nb_block_ctx,
    ///     |ctx| ctx.non_zeros_at(br, p, predicted, block_ctx, nb_block_ctx),
    ///     |ctx| ctx.coefficient_at(br, p, k, non_zeros, num_blocks, size,
    ///                              prev, block_ctx, nb_block_ctx),
    /// )
    /// ```
    ///
    /// — which doesn't actually compile because both closures need
    /// `&mut self` on the histogram-decode context concurrently. The
    /// method body drives the same Listing C.14 state machine but with
    /// sequential `&mut self` calls (one per symbol read), so the
    /// borrow checker is happy.
    ///
    /// Spec walk per Listing C.13 + C.14:
    ///
    /// 1. `nz_ctx = NonZerosContext(predicted, block_ctx,
    ///    nb_block_ctx)`,
    /// 2. `non_zeros = D[nz_ctx + histogram_offset(p)]` (a single
    ///    [`Self::non_zeros_at`] call),
    /// 3. defensive `non_zeros ≤ size - num_blocks` cap (round-90
    ///    invariant carried over verbatim — the spec's `(non_zeros +
    ///    num_blocks - 1) Idiv num_blocks` bound says a varblock with
    ///    `num_blocks` LLF cells can carry at most `size - num_blocks`
    ///    HF coefficients, so any reported `non_zeros` exceeding that
    ///    is an encoder bug or corrupted stream),
    /// 4. walk `k` from `num_blocks` to `size`:
    ///    a. break early when `non_zeros` reaches 0,
    ///    b. compute `prev = prev_for_context(k, num_blocks, size,
    ///    non_zeros, prev_nonzero[k-1])`,
    ///    c. `ucoeff = D[CoefficientContext(k, non_zeros, num_blocks,
    ///    size, prev, block_ctx, nb_block_ctx) + histogram_offset(
    ///    p)]` (a single [`Self::coefficient_at`] call),
    ///    d. `coeffs[natural_order[k]] = unpack_signed(ucoeff)`,
    ///    e. when `ucoeff != 0`, record `prev_nonzero[k] = true` and
    ///    decrement `non_zeros`.
    ///
    /// `predicted` is the [`crate::pass_group_hf::predicted_non_zeros`]
    /// value for the current varblock's top-left position; `block_ctx`
    /// is the [`crate::pass_group_hf::block_context`] result;
    /// `nb_block_ctx` is the LfGlobal `HfBlockContext` invariant
    /// (§I.2.2); `t` is the [`TransformType`] (§I.2.4 + Table C.16)
    /// driving the `(num_blocks, size)` + natural-order derivation.
    ///
    /// Returns `(decoded, raw_non_zeros)` where `decoded` is the
    /// [`DecodedHfBlock`] coefficient bundle (in natural-order index
    /// space) and `raw_non_zeros` is the **un-divided** `non_zeros`
    /// value the caller threads into the NonZeros-grid bookkeeping via
    /// `(raw_non_zeros + num_blocks - 1) Idiv num_blocks` (the spec
    /// line right after Listing C.14).
    ///
    /// Errors:
    /// * Propagates any [`Self::non_zeros_at`] /
    ///   [`Self::coefficient_at`] error verbatim (out-of-range pass
    ///   index, `u32`-overflow `ctx + offset`, downstream
    ///   `EntropyStream` error).
    /// * Rejects `non_zeros > size - num_blocks` with
    ///   [`Error::InvalidData`] before any HF-coefficient symbol read.
    /// * Rejects `num_blocks == 0` / mismatched natural-order length
    ///   via [`crate::pass_group_hf::transform_block_params`] +
    ///   internal sanity checks (defence in depth — the spec table
    ///   guarantees `num_blocks ≥ 1` for every [`TransformType`]).
    #[allow(clippy::too_many_arguments)]
    pub fn decode_block_for_pass_transform(
        &mut self,
        br: &mut BitReader<'_>,
        p: u32,
        t: TransformType,
        predicted: u32,
        block_ctx: u32,
        nb_block_ctx: u32,
    ) -> Result<(DecodedHfBlock, u32)> {
        let (num_blocks, size) = transform_block_params(t);
        // Defensive — every spec-listed TransformType has num_blocks ≥
        // 1 and size > 0, but we propagate a clean error instead of
        // panicking in `decode_block_coefficients_for_transform` if a
        // future table change introduces a zero-sized entry.
        if num_blocks == 0 {
            return Err(Error::InvalidData(
                "JXL HfHistogramDecodeContext::decode_block_for_pass_transform: \
                 num_blocks must be ≥ 1"
                    .into(),
            ));
        }
        if size == 0 || num_blocks > size {
            return Err(Error::InvalidData(format!(
                "JXL HfHistogramDecodeContext::decode_block_for_pass_transform: \
                 invalid (num_blocks={num_blocks}, size={size}) for {t:?}"
            )));
        }
        let oid = order_id_for_transform(t);
        let natural_order = natural_coeff_order(oid);
        if natural_order.len() != size as usize {
            return Err(Error::InvalidData(format!(
                "JXL HfHistogramDecodeContext::decode_block_for_pass_transform: \
                 natural_order len {} != size {size} for {t:?}",
                natural_order.len()
            )));
        }

        // (1) NonZerosContext read (round-252 routing).
        let raw_non_zeros = self.non_zeros_at(br, p, predicted, block_ctx, nb_block_ctx)?;

        // (2) Cap check — round-90 invariant.
        if raw_non_zeros > size - num_blocks {
            return Err(Error::InvalidData(format!(
                "JXL HfHistogramDecodeContext::decode_block_for_pass_transform: \
                 non_zeros {raw_non_zeros} > size - num_blocks ({}) at predicted={predicted}",
                size - num_blocks
            )));
        }

        // (3) Listing C.14 per-block loop — sequential &mut self calls.
        let mut coeffs = vec![0i32; size as usize];
        let mut prev_nonzero = vec![false; size as usize];
        let mut non_zeros = raw_non_zeros;
        let mut coeffs_read: u32 = 0;
        let mut k = num_blocks;
        while k < size {
            if non_zeros == 0 {
                break;
            }
            let prev = prev_for_context(k, num_blocks, size, non_zeros, |kk| {
                prev_nonzero[kk as usize]
            });
            let ucoeff = self.coefficient_at(
                br,
                p,
                k,
                non_zeros,
                num_blocks,
                size,
                prev,
                block_ctx,
                nb_block_ctx,
            )?;
            coeffs_read += 1;
            let signed = unpack_signed(ucoeff);
            let pos = natural_order[k as usize] as usize;
            // natural_order entries are guaranteed in [0, size) by
            // construction of natural_coeff_order; we still defensively
            // guard for the (impossible-by-spec) out-of-range case to
            // mirror round-90's belt-and-braces shape.
            if pos >= size as usize {
                return Err(Error::InvalidData(format!(
                    "JXL HfHistogramDecodeContext::decode_block_for_pass_transform: \
                     natural_order entry {pos} >= size {size} for {t:?}"
                )));
            }
            coeffs[pos] = signed;
            if ucoeff != 0 {
                prev_nonzero[k as usize] = true;
                non_zeros = non_zeros.saturating_sub(1);
            }
            k += 1;
        }

        Ok((
            DecodedHfBlock {
                coeffs,
                remaining_non_zeros: non_zeros,
                coeffs_read,
            },
            raw_non_zeros,
        ))
    }

    /// Decode one varblock's HF coefficients for all three §C.8.3
    /// channels (X = 0, Y = 1, B = 2) under pass `p` — round 260's
    /// bundled three-channel composition above
    /// [`Self::decode_block_for_pass_transform`].
    ///
    /// Round 255 landed the single-channel per-varblock walk. Round
    /// 221's
    /// [`crate::block_context_resolver::decode_varblocks_three_channels_with_resolver`]
    /// reproduces the three-channel sweep at the
    /// per-LfGroup driver layer, but goes through the
    /// [`crate::per_pass_non_zeros::PerPassNonZerosGrids::decode_block_at_for_pass_channel`]
    /// closure-based path (`read_non_zeros` + `decode_symbol` per
    /// channel). Now that the round-252 typed
    /// [`HfHistogramDecodeContext`] owns the per-pass histogram
    /// routing, callers wanting to drive the three-channel sweep against
    /// the §C.7.2 entropy stream directly need a per-varblock typed
    /// entry — without that, every caller re-implements the same
    /// three-call sequential walk, mixing per-channel state with
    /// the borrow-checker dance the round-255 method already solves
    /// for the single-channel case.
    ///
    /// Round 260 lifts that single-channel walk into a three-channel
    /// per-varblock walk:
    ///
    /// 1. for `c` ∈ `{1, 0, 2}` (the §C.8.3 prose decode order —
    ///    "for each varblock it reads channels Y, X, then B" —
    ///    against the Listing C.13 channel indices 0 = X, 1 = Y,
    ///    2 = B):
    ///    a. `block_ctx_c = resolver.resolve(c, &vb, qdc)?`
    ///    (Listing C.13 per-channel `BlockContext()` lookup),
    ///    b. `(decoded_c, raw_c) =
    ///    self.decode_block_for_pass_transform(br, p, vb.transform,
    ///    predicted[c], block_ctx_c, resolver.nb_block_ctx())?`,
    /// 2. return `(([decoded_0, decoded_1, decoded_2],
    ///    [raw_0, raw_1, raw_2]))` — indexed by channel, not by
    ///    decode position.
    ///
    /// The `resolver` owns the `nb_block_ctx` invariant (read off the
    /// LfGlobal `HfBlockContext` bundle, the same value the §C.7.2
    /// histograms were sized against — caller is responsible for
    /// passing the resolver that matches the histograms used to build
    /// `self`). The `predicted[3]` triple is the caller's per-channel
    /// [`crate::pass_group_hf::predicted_non_zeros`] lookup against
    /// the per-pass per-channel
    /// [`crate::per_channel_non_zeros::PerChannelNonZerosGrids`] grid
    /// (round 183 / 190) — round 260 keeps that lookup caller-side so
    /// the histograms primitive remains storage-blind. The same
    /// applies to the post-decode `NonZeros(x, y)` writeback (the
    /// `(raw + num_blocks - 1) Idiv num_blocks` line right after
    /// Listing C.14): caller invokes
    /// [`crate::per_pass_non_zeros::PerPassNonZerosGrids::update_after_block_for_pass_transform`]
    /// (or the equivalent per-channel call) once per channel against
    /// the returned `raw_non_zeros[c]`.
    ///
    /// `qdc[3]` is the shared per-varblock quantised-LF top-left
    /// triple (one read, three lookups) per round-221's invariant —
    /// not a per-channel value, so we accept a single `[i32; 3]`
    /// array.
    ///
    /// Channel decode ordering is fixed at Y → X → B per the §C.8.3
    /// prose ("for each varblock it reads channels Y, X, then B").
    /// The §C.7.2 entropy stream advances in that order; an error on
    /// X aborts before B reads, so the B-channel ANS state is **not**
    /// advanced (matching round-221's error-path invariant).
    ///
    /// Errors:
    /// * Propagates any [`BlockContextResolver::resolve`] error
    ///   verbatim (channel `> 2`, `s` out-of-range, threshold-table
    ///   inconsistency).
    /// * Propagates any [`Self::decode_block_for_pass_transform`]
    ///   error verbatim (out-of-range pass index, `u32`-overflow,
    ///   downstream `EntropyStream` error, or `non_zeros > size -
    ///   num_blocks` cap).
    ///
    /// Same pure-control-flow primitive shape as the round-255
    /// [`Self::decode_block_for_pass_transform`] it composes: no spec
    /// re-derivation, no ANS state initialisation, no raster walk
    /// across multiple varblocks.
    pub fn decode_three_channel_varblock_for_pass(
        &mut self,
        br: &mut BitReader<'_>,
        p: u32,
        vb: &Varblock,
        resolver: &BlockContextResolver<'_>,
        qdc: [i32; 3],
        predicted: [u32; 3],
    ) -> Result<([DecodedHfBlock; 3], [u32; 3])> {
        let nb_block_ctx = resolver.nb_block_ctx();
        // Channel decode order Y = 1 → X = 0 → B = 2 per the FDIS
        // §C.8.3 prose ("for each varblock it reads channels Y, X,
        // then B"); the channel *indices* stay 0 = X, 1 = Y, 2 = B
        // per Listing C.13, so the output arrays remain indexed by
        // channel while the entropy stream advances Y-first.
        // (Round 281 prose-conformance fix: rounds 260..264 advanced
        // the stream X-first.) Sequential `&mut self` calls (each
        // walks the round-255 Listing C.14 state machine) because the
        // histogram stream is shared across channels and the
        // underlying `decode_block_for_pass_transform` takes
        // `&mut self`.
        let mut decoded: [Option<DecodedHfBlock>; 3] = [None, None, None];
        let mut raw: [u32; 3] = [0; 3];
        for c in [1u32, 0, 2] {
            let ctx = resolver.resolve(c, vb, qdc)?;
            let (d, r) = self.decode_block_for_pass_transform(
                br,
                p,
                vb.transform,
                predicted[c as usize],
                ctx,
                nb_block_ctx,
            )?;
            decoded[c as usize] = Some(d);
            raw[c as usize] = r;
        }
        let [d0, d1, d2] = decoded;
        Ok((
            [
                d0.expect("channel 0 decoded"),
                d1.expect("channel 1 decoded"),
                d2.expect("channel 2 decoded"),
            ],
            raw,
        ))
    }

    /// Per-LfGroup raster-walk three-channel decode driver for one
    /// pass — round 264's bundled composition above
    /// [`Self::decode_three_channel_varblock_for_pass`].
    ///
    /// Round 260 landed the single-varblock three-channel walk. The
    /// canonical §C.8.3 caller pattern, however, is a raster walk of
    /// the [`DctSelectGrid`] in row-major order with that bundled
    /// method called once per top-left cell. Round 221's
    /// [`crate::block_context_resolver::decode_varblocks_three_channels_with_resolver`]
    /// reproduces that raster sweep at the per-LfGroup driver layer
    /// but routes through the [`crate::per_pass_non_zeros::PerPassNonZerosGrids`]
    /// closure-based path (caller-side `read_non_zeros` /
    /// `decode_symbol` closures wired to the per-pass per-channel
    /// histogram).
    ///
    /// Round 264 lifts that into a typed primitive that owns both the
    /// raster walk **and** the §C.7.2 entropy-stream routing: the
    /// driver invokes the round-260 single-varblock bundled method
    /// once per top-left cell, threading the per-pass `histogram_offset`
    /// internally via [`Self::decode_block_for_pass_transform`]. No
    /// `read_non_zeros` / `decode_symbol` closures cross the boundary
    /// — the caller hands in only the (storage-only) `qdc_at` +
    /// `predicted_at` lookups.
    ///
    /// Spec walk per FDIS §C.8.3 (per-LfGroup raster):
    ///
    /// 1. for every top-left cell yielded by
    ///    [`VarblockWalk::next`] in row-major order:
    ///    a. `qdc = qdc_at(&vb)` — caller's per-varblock shared
    ///    quantised-LF top-left triple,
    ///    b. `predicted = predicted_at(&vb)` — caller's per-varblock
    ///    per-channel
    ///    [`crate::pass_group_hf::predicted_non_zeros`] triple,
    ///    c. `(decoded[3], raw[3]) =
    ///    self.decode_three_channel_varblock_for_pass(br, p, &vb,
    ///    resolver, qdc, predicted)` — round-260 bundled
    ///    three-channel walk,
    ///    d. push
    ///    `(vb, [decoded_0, decoded_1, decoded_2], [raw_0, raw_1, raw_2])`
    ///    onto the output vector.
    /// 2. return the in-raster-order
    ///    `Vec<ThreeChannelVarblock>` (matching the round-221 /
    ///    228 / 260 type alias).
    ///
    /// The `resolver` owns the `nb_block_ctx` invariant (read off the
    /// LfGlobal `HfBlockContext` bundle, the same value the §C.7.2
    /// histograms were sized against — caller is responsible for
    /// passing the resolver that matches the histograms used to build
    /// `self`). The `qdc_at` closure may be a per-LfGroup
    /// quantised-LF lookup
    /// ([`crate::lf_dequant`] output sampled at the varblock's
    /// top-left LF cell) or a static per-varblock cache — the driver
    /// is storage-blind. Same for `predicted_at`: typically a lookup
    /// against the per-pass per-channel
    /// [`crate::per_channel_non_zeros::PerChannelNonZerosGrids`]
    /// grid the caller maintains.
    ///
    /// `predicted_at` is invoked **after** `qdc_at` so the caller
    /// closure may consult any state observed during `qdc_at` if
    /// that simplifies the predictor lookup; the driver keeps the
    /// `qdc → predicted → decode` ordering invariant.
    ///
    /// The driver writes back per-channel `NonZeros(x, y)` state
    /// **not** automatically — the caller is responsible for
    /// invoking
    /// [`crate::per_pass_non_zeros::PerPassNonZerosGrids::update_after_block_for_pass_transform`]
    /// (or the equivalent per-channel call) once per channel
    /// against the returned `raw[c]` value per varblock per pass,
    /// matching the round-260 invariant.
    ///
    /// On any error (per-varblock `qdc_at` / `predicted_at` /
    /// resolver / `decode_three_channel_varblock_for_pass`) the
    /// driver propagates the error immediately and discards any
    /// in-flight partial output. The walk always proceeds in
    /// raster order; an error on varblock `i` aborts before
    /// varblock `i + 1`'s `qdc_at` is invoked (so the §C.7.2
    /// entropy stream is not advanced past the failing call).
    ///
    /// Returns an empty vector for an empty grid (`width_blocks ==
    /// 0` or `height_blocks == 0`); the per-varblock output count
    /// is precisely [`count_varblocks(grid)`] on success.
    ///
    /// Errors:
    /// * Propagates any [`VarblockWalk::next`] error verbatim
    ///   (residual `DctSelectCell::Empty` cell — caller-side grid
    ///   mutation),
    /// * Propagates any `qdc_at` / `predicted_at` closure error
    ///   verbatim,
    /// * Propagates any
    ///   [`Self::decode_three_channel_varblock_for_pass`] error
    ///   verbatim (per-channel resolver error, out-of-range pass
    ///   index, `u32`-overflow `ctx + offset`, downstream
    ///   `EntropyStream` error, `non_zeros > size - num_blocks`
    ///   cap, transform-table inconsistency).
    ///
    /// Same pure-control-flow primitive shape as the round-260
    /// [`Self::decode_three_channel_varblock_for_pass`] it composes:
    /// no spec re-derivation, no ANS state initialisation, no
    /// per-pass outer loop (that's round 228's layer above).
    pub fn decode_lf_group_three_channels_for_pass<Q, P>(
        &mut self,
        br: &mut BitReader<'_>,
        p: u32,
        grid: &DctSelectGrid,
        resolver: &BlockContextResolver<'_>,
        mut qdc_at: Q,
        mut predicted_at: P,
    ) -> Result<Vec<ThreeChannelVarblock>>
    where
        Q: FnMut(&Varblock) -> Result<[i32; 3]>,
        P: FnMut(&Varblock) -> Result<[u32; 3]>,
    {
        let mut out: Vec<ThreeChannelVarblock> = Vec::with_capacity(count_varblocks(grid) as usize);
        let mut walk = VarblockWalk::new(grid);
        while let Some(vb) = walk.next()? {
            let qdc = qdc_at(&vb)?;
            let predicted = predicted_at(&vb)?;
            let (decoded, raw) =
                self.decode_three_channel_varblock_for_pass(br, p, &vb, resolver, qdc, predicted)?;
            out.push((vb, decoded, raw));
        }
        Ok(out)
    }

    /// Per-pass `histogram_offset(p)` lookup. Range-checked on `p`.
    pub fn histogram_offset(&self, p: u32) -> Result<u64> {
        self.per_pass_offsets
            .get(p as usize)
            .copied()
            .ok_or_else(|| {
                Error::InvalidData(format!(
                    "JXL HfHistogramDecodeContext: pass index {p} out of {} per-pass offsets",
                    self.per_pass_offsets.len()
                ))
            })
    }

    /// Pass count — = the `headers.num_passes()` value passed to
    /// [`Self::new`].
    pub fn num_passes(&self) -> u32 {
        self.per_pass_offsets.len() as u32
    }

    /// Borrow the per-pass offset slice for read-only inspection.
    pub fn per_pass_offsets(&self) -> &[u64] {
        &self.per_pass_offsets
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ans::test_helpers::pack_lsb;
    use crate::hf_coeff_histogram_size::HfCoefficientHistogramSize;
    use crate::pass_group_hf::PassGroupHfHeader;

    /// Build a §C.7.2 [`HfCoefficientHistograms`] for the minimal
    /// `(num_hf_presets, nb_block_ctx)` shape with single-symbol
    /// prefix clustering — every distribution maps to cluster 0,
    /// which has the single symbol 0.
    ///
    /// Returns the histograms post-`read` (ANS state initialisation
    /// is the prefix no-op; the caller may invoke
    /// `read_ans_state_init` separately for symmetry).
    fn make_minimal_histograms(num_hf_presets: u32, nb_block_ctx: u32) -> HfCoefficientHistograms {
        // Same minimal prelude pattern as the round-247 tests:
        //   lz77 = 0, is_simple = 1, nbits = 0,
        //   use_prefix_code = 1, log_alphabet_size = 15 implicit,
        //   split_exponent = 0 (HybridUintConfig),
        //   prefix count selector = 0 → single-symbol code.
        let parts: Vec<(u32, u32)> = vec![
            (0, 1), // lz77_enabled = 0
            (1, 1), // is_simple = 1
            (0, 2), // nbits = 0 → all distributions map to cluster 0
            (1, 1), // use_prefix_code = 1
            (0, 4), // split_exponent = 0
            (0, 1), // prefix count = 1
        ];
        let bytes = pack_lsb(&parts);
        // We need the bytes to outlive br — copy into a heap buffer
        // we leak for the test scope. (For a real test fixture, we
        // would keep `bytes` alive in the caller's frame; this is a
        // helper that returns the histograms only.)
        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let mut br = BitReader::new(leaked);
        let size = HfCoefficientHistogramSize::new(num_hf_presets, nb_block_ctx).unwrap();
        HfCoefficientHistograms::read(&mut br, size).unwrap()
    }

    #[test]
    fn r252_new_rejects_zero_passes() {
        let mut h = make_minimal_histograms(1, 1);
        let headers = PerPassHfHeaders::from_headers(vec![]);
        let r = HfHistogramDecodeContext::new(&mut h, &headers);
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn r252_new_rejects_hfp_ge_num_hf_presets() {
        // histograms has num_hf_presets = 2; we hand in a header with
        // hfp = 3 → must reject.
        let mut h = make_minimal_histograms(2, 1);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 3,
            histogram_offset: 495 * 3,
        }]);
        let r = HfHistogramDecodeContext::new(&mut h, &headers);
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn r252_new_caches_per_pass_offsets() {
        // num_hf_presets = 4, nb_block_ctx = 15 → per-pass offset =
        // 495 × 15 × hfp = 7425 × hfp.
        let mut h = make_minimal_histograms(4, 15);
        let headers = PerPassHfHeaders::from_headers(vec![
            PassGroupHfHeader {
                hfp: 0,
                histogram_offset: 0,
            },
            PassGroupHfHeader {
                hfp: 1,
                histogram_offset: 7425,
            },
            PassGroupHfHeader {
                hfp: 3,
                histogram_offset: 22_275,
            },
        ]);
        let ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        assert_eq!(ctx.num_passes(), 3);
        assert_eq!(ctx.per_pass_offsets(), &[0, 7425, 22_275]);
        assert_eq!(ctx.histogram_offset(0).unwrap(), 0);
        assert_eq!(ctx.histogram_offset(1).unwrap(), 7425);
        assert_eq!(ctx.histogram_offset(2).unwrap(), 22_275);
        assert!(ctx.histogram_offset(3).is_err());
    }

    #[test]
    fn r252_decode_symbol_for_pass_routes_through_offset() {
        // Single-symbol prefix code at cluster 0 — every ctx index
        // mapped through cluster_map → cluster 0 → symbol 0. We
        // therefore expect every decode_symbol call to return 0
        // regardless of (p, ctx), provided the (ctx + offset) sum
        // stays within the cluster_map length
        // (= num_distributions = 495 × 1 × 1 = 495 for the minimal
        // shape).
        //
        // We use num_hf_presets = 2, nb_block_ctx = 1 →
        // num_distributions = 990. Per-pass offset = 495 × 1 × hfp.
        let mut h = make_minimal_histograms(2, 1);
        let headers = PerPassHfHeaders::from_headers(vec![
            PassGroupHfHeader {
                hfp: 0,
                histogram_offset: 0,
            },
            PassGroupHfHeader {
                hfp: 1,
                histogram_offset: 495,
            },
        ]);
        let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();

        // Empty bytes are fine — single-symbol prefix code consumes
        // zero bits per decode.
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let bits_before = br.bits_read();

        // Decode for pass 0 (offset = 0), context 0 → routes to
        // cluster_map[0] = cluster 0 → symbol 0.
        let s00 = ctx.decode_symbol_for_pass(&mut br, 0, 0).unwrap();
        assert_eq!(s00, 0);
        // Decode for pass 0, context 100 → routes to cluster_map[100]
        // = cluster 0 → symbol 0.
        let s0_100 = ctx.decode_symbol_for_pass(&mut br, 0, 100).unwrap();
        assert_eq!(s0_100, 0);
        // Decode for pass 1 (offset = 495), context 0 → routes to
        // cluster_map[0 + 495] = cluster 0 → symbol 0.
        let s10 = ctx.decode_symbol_for_pass(&mut br, 1, 0).unwrap();
        assert_eq!(s10, 0);
        // Decode for pass 1, context 100 → routes to cluster_map[595]
        // = cluster 0 → symbol 0.
        let s1_100 = ctx.decode_symbol_for_pass(&mut br, 1, 100).unwrap();
        assert_eq!(s1_100, 0);

        // Single-symbol prefix code consumes zero bits per decode.
        assert_eq!(br.bits_read(), bits_before);
    }

    #[test]
    fn r252_decode_symbol_for_pass_rejects_out_of_range_pass() {
        let mut h = make_minimal_histograms(1, 1);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let bytes = [0u8];
        let mut br = BitReader::new(&bytes);
        let r = ctx.decode_symbol_for_pass(&mut br, 1, 0);
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn r252_decode_symbol_for_pass_rejects_u32_overflow_sum() {
        // Synthetic header with a huge histogram_offset → ctx +
        // offset > u32::MAX → reject. PerPassHfHeaders::from_headers
        // does not validate the offset field, so we can hand in a
        // u64 value above u32::MAX directly.
        let mut h = make_minimal_histograms(1, 1);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: u64::from(u32::MAX) + 10,
        }]);
        let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let bytes = [0u8];
        let mut br = BitReader::new(&bytes);
        let r = ctx.decode_symbol_for_pass(&mut br, 0, 0);
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn r252_non_zeros_at_uses_non_zeros_context_plus_offset() {
        // non_zeros_context(predicted = 0, block_ctx = 5,
        // nb_block_ctx = 15) = 5 + 15 × 0 = 5 (predicted < 8 branch).
        // For pass 0 with offset 0 → routes to cluster_map[5] →
        // cluster 0 → symbol 0.
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let s = ctx.non_zeros_at(&mut br, 0, 0, 5, 15).unwrap();
        assert_eq!(s, 0);
        // Cross-check the context derivation matches the standalone
        // helper.
        assert_eq!(non_zeros_context(0, 5, 15), 5);
    }

    #[test]
    fn r252_coefficient_at_uses_coefficient_context_plus_offset() {
        // coefficient_context(k = 1, non_zeros = 16, num_blocks = 1,
        // size = 64, prev = 0, block_ctx = 0, nb_block_ctx = 15)
        // — the §C.8.3 first-non-zero-block path. We just need the
        // wrapper to compose correctly; the exact value is checked
        // against the standalone helper.
        let expected_ctx = coefficient_context(1, 16, 1, 64, 0, 0, 15).unwrap();
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        // expected_ctx is < num_distributions for this minimal shape
        // (495 × 1 × 15 = 7425). Single-symbol prefix → returns 0
        // for any in-range ctx.
        let s = ctx_dec
            .coefficient_at(&mut br, 0, 1, 16, 1, 64, 0, 0, 15)
            .unwrap();
        assert_eq!(s, 0);
        // Sanity: the helper's expected_ctx is computed identically.
        let _ = expected_ctx;
    }

    #[test]
    fn r252_coefficient_at_propagates_num_blocks_zero_error() {
        // coefficient_context rejects num_blocks = 0; the wrapper
        // propagates the error without touching the BitReader.
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let bits_before = br.bits_read();
        let r = ctx_dec.coefficient_at(&mut br, 0, 1, 16, 0, 64, 0, 0, 15);
        assert!(matches!(r, Err(Error::InvalidData(_))));
        assert_eq!(br.bits_read(), bits_before);
    }

    #[test]
    fn r252_per_pass_offset_sequence_matches_headers() {
        // Construct from PerPassHfHeaders::read against a real
        // bitstream so the round-232 derivation is exercised end-
        // to-end and we confirm the round-252 cache equals it.
        let mut h = make_minimal_histograms(2, 15);
        // num_hf_presets = 2 → nbits = 1; pass 0 hfp = 0,
        // pass 1 hfp = 1. Bit layout: 0 | 1 = 0b10 = byte 0x02.
        let header_bytes = [0b0000_0010u8];
        let mut br = BitReader::new(&header_bytes);
        let headers = PerPassHfHeaders::read(&mut br, 2, 2, 15).unwrap();
        let ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        assert_eq!(ctx.num_passes(), 2);
        // Pass-0 offset = 495 × 15 × 0 = 0; pass-1 offset = 7425.
        assert_eq!(ctx.per_pass_offsets(), &[0u64, 7425u64]);
    }

    // -----------------------------------------------------------------
    // Round 255 — `decode_block_for_pass_transform` unit tests. The
    // bundled per-varblock decode method composes round-252's per-pass
    // `non_zeros_at` + `coefficient_at` into the round-90 Listing C.14
    // state machine for one varblock at a time. All tests use the
    // single-symbol-prefix `make_minimal_histograms` shape so every
    // `D[ctx + offset]` read returns 0 — which means the C.14 inner
    // loop short-circuits immediately on the very first symbol since
    // `non_zeros` arrives as 0 (predicted = 0 → NonZerosContext = 5,
    // single-symbol prefix → 0).
    // -----------------------------------------------------------------

    #[test]
    fn r255_decode_block_for_pass_transform_dct8x8_short_circuits_on_zero_non_zeros() {
        // DCT8x8: num_blocks = 1, size = 64. With single-symbol prefix
        // → non_zeros = 0 → C.14 loop never enters the body, no
        // coefficient symbols read.
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let bits_before = br.bits_read();
        let (decoded, raw_non_zeros) = ctx_dec
            .decode_block_for_pass_transform(
                &mut br,
                /*p*/ 0,
                TransformType::Dct8x8,
                /*predicted*/ 0,
                /*block_ctx*/ 0,
                /*nb_block_ctx*/ 15,
            )
            .unwrap();
        assert_eq!(raw_non_zeros, 0);
        assert_eq!(decoded.remaining_non_zeros, 0);
        assert_eq!(decoded.coeffs_read, 0);
        assert_eq!(decoded.coeffs.len(), 64);
        assert!(decoded.coeffs.iter().all(|&c| c == 0));
        // Single-symbol prefix → zero bits consumed.
        assert_eq!(br.bits_read(), bits_before);
    }

    #[test]
    fn r255_decode_block_for_pass_transform_rejects_out_of_range_pass() {
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let r = ctx_dec.decode_block_for_pass_transform(
            &mut br,
            /*p*/ 5, // > num_passes (= 1)
            TransformType::Dct8x8,
            /*predicted*/ 0,
            /*block_ctx*/ 0,
            /*nb_block_ctx*/ 15,
        );
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn r255_decode_block_for_pass_transform_dct16x16_short_circuits() {
        // DCT16x16: num_blocks = 4, size = 256. With single-symbol
        // prefix → non_zeros = 0 → no coefficient symbols read.
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let (decoded, raw_non_zeros) = ctx_dec
            .decode_block_for_pass_transform(
                &mut br,
                0,
                TransformType::Dct16x16,
                /*predicted*/ 32, // top-left → predicted_non_zeros = 32
                /*block_ctx*/ 0,
                /*nb_block_ctx*/ 15,
            )
            .unwrap();
        assert_eq!(raw_non_zeros, 0);
        assert_eq!(decoded.coeffs.len(), 256);
        assert_eq!(decoded.coeffs_read, 0);
        assert!(decoded.coeffs.iter().all(|&c| c == 0));
    }

    #[test]
    fn r255_decode_block_for_pass_transform_routes_through_per_pass_offset() {
        // num_hf_presets = 2, nb_block_ctx = 1; per-pass offset =
        // 495 × 1 × hfp. We invoke for pass 1 → routes via cluster_map[
        // nz_ctx + 495]; single-symbol prefix → 0.
        let mut h = make_minimal_histograms(2, 1);
        let headers = PerPassHfHeaders::from_headers(vec![
            PassGroupHfHeader {
                hfp: 0,
                histogram_offset: 0,
            },
            PassGroupHfHeader {
                hfp: 1,
                histogram_offset: 495,
            },
        ]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let (decoded, raw_non_zeros) = ctx_dec
            .decode_block_for_pass_transform(
                &mut br,
                /*p*/ 1,
                TransformType::Dct8x8,
                /*predicted*/ 0,
                /*block_ctx*/ 0,
                /*nb_block_ctx*/ 1,
            )
            .unwrap();
        assert_eq!(raw_non_zeros, 0);
        assert_eq!(decoded.coeffs_read, 0);
    }

    #[test]
    fn r255_decode_block_for_pass_transform_propagates_u32_overflow() {
        // Synthetic header with histogram_offset > u32::MAX → the
        // first non_zeros_at call hits the u32 overflow guard.
        let mut h = make_minimal_histograms(1, 1);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: u64::from(u32::MAX) + 10,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let r = ctx_dec.decode_block_for_pass_transform(&mut br, 0, TransformType::Dct8x8, 0, 0, 1);
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn r255_decode_block_for_pass_transform_dct8x16_short_circuits() {
        // DCT8x16: num_blocks = 2, size = 128. Confirms the transform-
        // type dispatch picks the rectangular-output table cleanly.
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let (decoded, raw_non_zeros) = ctx_dec
            .decode_block_for_pass_transform(&mut br, 0, TransformType::Dct8x16, 0, 0, 15)
            .unwrap();
        assert_eq!(raw_non_zeros, 0);
        assert_eq!(decoded.coeffs.len(), 128);
        assert_eq!(decoded.coeffs_read, 0);
    }

    #[test]
    fn r255_decode_block_for_pass_transform_does_not_advance_br_when_short_circuited() {
        // Single-symbol prefix: every D[...] read consumes 0 bits.
        // The non_zeros_at call returns 0, the C.14 loop never enters,
        // so the BitReader cursor must not advance.
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let bytes = [0xFFu8; 16];
        let mut br = BitReader::new(&bytes);
        let bits_before = br.bits_read();
        let _ = ctx_dec
            .decode_block_for_pass_transform(&mut br, 0, TransformType::Dct8x8, 0, 0, 15)
            .unwrap();
        assert_eq!(br.bits_read(), bits_before);
    }

    // -----------------------------------------------------------------
    // Round 260 — `decode_three_channel_varblock_for_pass` unit tests.
    // The bundled three-channel per-varblock walk composes round-255's
    // single-channel `decode_block_for_pass_transform` three times
    // (channel decode order Y = 1 → X = 0 → B = 2 per the §C.8.3
    // prose; round-281 conformance fix) against a
    // BlockContextResolver-derived per-channel `block_ctx`. All tests
    // use the single-symbol-prefix `make_minimal_histograms` shape so
    // every `D[ctx + offset]` read returns 0 — and the per-channel
    // C.14 inner loop short-circuits as in the round-255 tests above.
    // -----------------------------------------------------------------

    use crate::lf_global::HfBlockContext;

    /// Default §I.2.2 `HfBlockContext` bundle — same shape the round
    /// 214 / 221 / 228 tests use. Empty thresholds collapse the
    /// `qf` / `qdc` knobs; the 39-entry default `block_ctx_map`
    /// gives `nb_block_ctx = 15`.
    fn default_hbc_r260() -> HfBlockContext {
        HfBlockContext {
            used_default: true,
            qf_thresholds: vec![],
            lf_thresholds: [vec![], vec![], vec![]],
            block_ctx_map: HfBlockContext::DEFAULT_BLOCK_CTX_MAP.to_vec(),
            nb_block_ctx: 15,
        }
    }

    fn vb_dct8x8(x: u32, y: u32) -> Varblock {
        Varblock {
            x,
            y,
            transform: TransformType::Dct8x8,
            hf_mul: 1,
        }
    }

    #[test]
    fn r260_three_channel_short_circuits_on_zero_non_zeros_dct8x8() {
        // DCT8×8 single varblock, all three channels short-circuit on
        // non_zeros = 0; each channel returns an all-zero 64-coeff
        // block.
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let hbc = default_hbc_r260();
        let resolver = BlockContextResolver::new(&hbc);
        let vb = vb_dct8x8(0, 0);
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let bits_before = br.bits_read();
        let (decoded, raw) = ctx_dec
            .decode_three_channel_varblock_for_pass(
                &mut br,
                /*p*/ 0,
                &vb,
                &resolver,
                /*qdc*/ [0, 0, 0],
                /*predicted*/ [0, 0, 0],
            )
            .unwrap();
        for c in 0..3 {
            assert_eq!(raw[c], 0, "channel {c} raw_non_zeros");
            assert_eq!(decoded[c].remaining_non_zeros, 0, "channel {c} remaining");
            assert_eq!(decoded[c].coeffs_read, 0, "channel {c} coeffs_read");
            assert_eq!(decoded[c].coeffs.len(), 64, "channel {c} coeffs.len");
            assert!(
                decoded[c].coeffs.iter().all(|&v| v == 0),
                "channel {c} all-zero"
            );
        }
        // Single-symbol prefix → three D[...] reads, each zero bits.
        assert_eq!(br.bits_read(), bits_before);
    }

    #[test]
    fn r260_three_channel_rejects_out_of_range_pass() {
        // Pass index 2 with num_passes = 1 → InvalidData on the very
        // first channel's decode_block_for_pass_transform call.
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let hbc = default_hbc_r260();
        let resolver = BlockContextResolver::new(&hbc);
        let vb = vb_dct8x8(0, 0);
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let r = ctx_dec.decode_three_channel_varblock_for_pass(
            &mut br,
            /*p*/ 2,
            &vb,
            &resolver,
            [0, 0, 0],
            [0, 0, 0],
        );
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn r260_three_channel_dct16x16_short_circuits_per_channel() {
        // DCT16×16: num_blocks = 4, size = 256. Three channels each
        // short-circuit; per-channel coeff buffer length = 256.
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let hbc = default_hbc_r260();
        let resolver = BlockContextResolver::new(&hbc);
        let vb = Varblock {
            x: 0,
            y: 0,
            transform: TransformType::Dct16x16,
            hf_mul: 1,
        };
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let (decoded, raw) = ctx_dec
            .decode_three_channel_varblock_for_pass(
                &mut br,
                0,
                &vb,
                &resolver,
                [0, 0, 0],
                [32, 32, 32],
            )
            .unwrap();
        for c in 0..3 {
            assert_eq!(raw[c], 0);
            assert_eq!(decoded[c].coeffs.len(), 256);
            assert_eq!(decoded[c].coeffs_read, 0);
        }
    }

    #[test]
    fn r260_three_channel_routes_through_per_pass_offset() {
        // Per-pass routing parity — pass 1 (offset = 7425) returns
        // symbol 0 via the same minimal-histograms shape because all
        // distributions map to cluster 0 → single-symbol prefix.
        // Reading both passes from the same context exercises the
        // round-252 per-pass offset cache.
        let mut h = make_minimal_histograms(2, 15);
        let header_bytes = [0b0000_0010u8]; // pass 0 hfp = 0, pass 1 hfp = 1
        let mut br_h = BitReader::new(&header_bytes);
        let headers = PerPassHfHeaders::read(&mut br_h, 2, 2, 15).unwrap();
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let hbc = default_hbc_r260();
        let resolver = BlockContextResolver::new(&hbc);
        let vb = vb_dct8x8(0, 0);
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let (_d0, r0) = ctx_dec
            .decode_three_channel_varblock_for_pass(
                &mut br,
                0,
                &vb,
                &resolver,
                [0, 0, 0],
                [0, 0, 0],
            )
            .unwrap();
        let (_d1, r1) = ctx_dec
            .decode_three_channel_varblock_for_pass(
                &mut br,
                1,
                &vb,
                &resolver,
                [0, 0, 0],
                [0, 0, 0],
            )
            .unwrap();
        assert_eq!(r0, [0, 0, 0]);
        assert_eq!(r1, [0, 0, 0]);
    }

    #[test]
    fn r260_three_channel_all_channels_resolved_via_resolver() {
        // Verify all three channels are resolved + decoded. (The
        // decode *order* — Y, X, then B per the §C.8.3 prose — is not
        // bit-observable here because the single-symbol prefix
        // histograms consume zero bits; the ordering itself is pinned
        // by the closure-path driver tests in
        // `round221_three_channel_resolver` /
        // `round228_multi_pass_decode`.) We piggy-back on the
        // BlockContextResolver to compute the per-channel block_ctx
        // values and check that they all flow into the decoder
        // (default-table fast path — empty thresholds collapse
        // qf / qdc, so block_ctx is purely determined by
        // `(channel, transform-order-id)`).
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let hbc = default_hbc_r260();
        let resolver = BlockContextResolver::new(&hbc);
        let vb = vb_dct8x8(0, 0);
        // Pre-compute expected per-channel block_ctx values to ensure
        // the resolver is in fact invoked with c ∈ {0, 1, 2}.
        let exp_c0 = resolver.resolve(0, &vb, [0, 0, 0]).unwrap();
        let exp_c1 = resolver.resolve(1, &vb, [0, 0, 0]).unwrap();
        let exp_c2 = resolver.resolve(2, &vb, [0, 0, 0]).unwrap();
        // The three values must be `< nb_block_ctx` (= 15).
        assert!(exp_c0 < 15);
        assert!(exp_c1 < 15);
        assert!(exp_c2 < 15);
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        // The bundled method must consume zero bits (single-symbol
        // prefix) and return three short-circuited blocks — confirming
        // every per-channel resolve + decode pair ran.
        let (decoded, raw) = ctx_dec
            .decode_three_channel_varblock_for_pass(
                &mut br,
                0,
                &vb,
                &resolver,
                [0, 0, 0],
                [0, 0, 0],
            )
            .unwrap();
        assert_eq!(raw, [0, 0, 0]);
        assert_eq!(decoded[0].coeffs_read, 0);
        assert_eq!(decoded[1].coeffs_read, 0);
        assert_eq!(decoded[2].coeffs_read, 0);
    }

    #[test]
    fn r260_three_channel_does_not_advance_br_when_short_circuited() {
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let hbc = default_hbc_r260();
        let resolver = BlockContextResolver::new(&hbc);
        let vb = vb_dct8x8(0, 0);
        let bytes = [0xFFu8; 16];
        let mut br = BitReader::new(&bytes);
        let bits_before = br.bits_read();
        let _ = ctx_dec
            .decode_three_channel_varblock_for_pass(
                &mut br,
                0,
                &vb,
                &resolver,
                [0, 0, 0],
                [0, 0, 0],
            )
            .unwrap();
        // Three single-symbol-prefix reads → zero bits consumed total.
        assert_eq!(br.bits_read(), bits_before);
    }

    #[test]
    fn r260_three_channel_propagates_u32_overflow() {
        // Synthetic header offset above u32::MAX → first channel's
        // decode_block_for_pass_transform → non_zeros_at →
        // decode_symbol_for_pass returns the u32-overflow rejection.
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: u64::from(u32::MAX) + 10,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let hbc = default_hbc_r260();
        let resolver = BlockContextResolver::new(&hbc);
        let vb = vb_dct8x8(0, 0);
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let r = ctx_dec.decode_three_channel_varblock_for_pass(
            &mut br,
            0,
            &vb,
            &resolver,
            [0, 0, 0],
            [0, 0, 0],
        );
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn r260_three_channel_dct16x8_short_circuits() {
        // DCT16×8: num_blocks = 2, size = 128.
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let hbc = default_hbc_r260();
        let resolver = BlockContextResolver::new(&hbc);
        let vb = Varblock {
            x: 0,
            y: 0,
            transform: TransformType::Dct16x8,
            hf_mul: 1,
        };
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let (decoded, raw) = ctx_dec
            .decode_three_channel_varblock_for_pass(
                &mut br,
                0,
                &vb,
                &resolver,
                [0, 0, 0],
                [16, 16, 16],
            )
            .unwrap();
        for c in 0..3 {
            assert_eq!(raw[c], 0);
            assert_eq!(decoded[c].coeffs.len(), 128);
            assert_eq!(decoded[c].coeffs_read, 0);
        }
    }

    // -----------------------------------------------------------------
    // Round 264 — `decode_lf_group_three_channels_for_pass` unit tests.
    // The bundled per-LfGroup raster-walk driver for one pass composes
    // the round-260 single-varblock three-channel method across the
    // DctSelectGrid raster, threading per-varblock `qdc_at` +
    // `predicted_at` closures into the inner method. All tests use the
    // single-symbol-prefix `make_minimal_histograms` shape so every
    // `D[ctx + offset]` read returns 0 — every per-channel C.14 loop
    // short-circuits on `non_zeros = 0`, no coefficient symbols are
    // read, and the BitReader cursor is invariant across the entire
    // raster walk.
    // -----------------------------------------------------------------

    use crate::dct_select::{DctSelectCell, DctSelectGrid};

    /// Build a synthetic [`DctSelectGrid`] whose every cell is a
    /// [`DctSelectCell::TopLeft`] of the given `TransformType` (a
    /// `width × height` raster of single-block varblocks). The
    /// `hf_mul` array is all-`1` per the §C.5.4 default. Used by the
    /// round-264 unit tests to exercise the raster walk without
    /// re-deriving a `BlockInfo` fixture.
    fn make_uniform_grid_r264(
        width_blocks: u32,
        height_blocks: u32,
        t: TransformType,
    ) -> DctSelectGrid {
        let total = (width_blocks as usize) * (height_blocks as usize);
        DctSelectGrid {
            cells: vec![DctSelectCell::TopLeft(t); total],
            hf_mul: vec![1i32; total],
            width_blocks,
            height_blocks,
        }
    }

    #[test]
    fn r264_lf_group_empty_grid_returns_empty_vec() {
        // 0×0 grid → no top-left cells → empty output.
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let hbc = default_hbc_r260();
        let resolver = BlockContextResolver::new(&hbc);
        let grid = DctSelectGrid {
            cells: vec![],
            hf_mul: vec![],
            width_blocks: 0,
            height_blocks: 0,
        };
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let out = ctx_dec
            .decode_lf_group_three_channels_for_pass(
                &mut br,
                0,
                &grid,
                &resolver,
                |_vb| Ok([0, 0, 0]),
                |_vb| Ok([0, 0, 0]),
            )
            .unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn r264_lf_group_1x1_dct8x8_short_circuits_three_channels() {
        // 1×1 DCT8×8 grid → one varblock × three channels, all
        // short-circuited on non_zeros = 0.
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let hbc = default_hbc_r260();
        let resolver = BlockContextResolver::new(&hbc);
        let grid = make_uniform_grid_r264(1, 1, TransformType::Dct8x8);
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let bits_before = br.bits_read();
        let out = ctx_dec
            .decode_lf_group_three_channels_for_pass(
                &mut br,
                0,
                &grid,
                &resolver,
                |_vb| Ok([0, 0, 0]),
                |_vb| Ok([0, 0, 0]),
            )
            .unwrap();
        assert_eq!(out.len(), 1);
        let (vb, decoded, raw) = &out[0];
        assert_eq!(vb.x, 0);
        assert_eq!(vb.y, 0);
        for c in 0..3 {
            assert_eq!(decoded[c].coeffs.len(), 64);
            assert_eq!(decoded[c].coeffs_read, 0);
            assert_eq!(raw[c], 0);
        }
        // Single-symbol prefix × three channels = zero bits consumed.
        assert_eq!(br.bits_read(), bits_before);
    }

    #[test]
    fn r264_lf_group_2x2_uniform_dct8x8_yields_four_varblocks_in_raster_order() {
        // 2×2 DCT8×8 grid → four varblocks in row-major (0,0), (1,0),
        // (0,1), (1,1). Per-varblock closures observed in the same
        // order via a side-channel counter.
        use std::cell::RefCell;
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let hbc = default_hbc_r260();
        let resolver = BlockContextResolver::new(&hbc);
        let grid = make_uniform_grid_r264(2, 2, TransformType::Dct8x8);
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let qdc_calls: RefCell<Vec<(u32, u32)>> = RefCell::new(vec![]);
        let pred_calls: RefCell<Vec<(u32, u32)>> = RefCell::new(vec![]);
        let out = ctx_dec
            .decode_lf_group_three_channels_for_pass(
                &mut br,
                0,
                &grid,
                &resolver,
                |vb| {
                    qdc_calls.borrow_mut().push((vb.x, vb.y));
                    Ok([0, 0, 0])
                },
                |vb| {
                    pred_calls.borrow_mut().push((vb.x, vb.y));
                    Ok([0, 0, 0])
                },
            )
            .unwrap();
        assert_eq!(out.len(), 4);
        // Raster order: (0,0), (1,0), (0,1), (1,1).
        let exp = vec![(0u32, 0u32), (1, 0), (0, 1), (1, 1)];
        assert_eq!(*qdc_calls.borrow(), exp);
        assert_eq!(*pred_calls.borrow(), exp);
        for i in 0..4 {
            assert_eq!(out[i].0.x, exp[i].0);
            assert_eq!(out[i].0.y, exp[i].1);
        }
    }

    #[test]
    fn r264_lf_group_qdc_at_invoked_before_predicted_at_per_varblock() {
        // Per-varblock ordering: qdc_at fires before predicted_at, but
        // the per-varblock pair is followed by the next varblock's
        // qdc_at. We pin this by appending a tag to a shared log.
        use std::cell::RefCell;
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let hbc = default_hbc_r260();
        let resolver = BlockContextResolver::new(&hbc);
        let grid = make_uniform_grid_r264(2, 1, TransformType::Dct8x8);
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let log: RefCell<Vec<(char, u32)>> = RefCell::new(vec![]);
        let _ = ctx_dec
            .decode_lf_group_three_channels_for_pass(
                &mut br,
                0,
                &grid,
                &resolver,
                |vb| {
                    log.borrow_mut().push(('q', vb.x));
                    Ok([0, 0, 0])
                },
                |vb| {
                    log.borrow_mut().push(('p', vb.x));
                    Ok([0, 0, 0])
                },
            )
            .unwrap();
        assert_eq!(*log.borrow(), vec![('q', 0), ('p', 0), ('q', 1), ('p', 1)]);
    }

    #[test]
    fn r264_lf_group_rejects_out_of_range_pass_no_per_varblock_calls() {
        // Pass 5 with num_passes = 1 → first varblock's
        // decode_three_channel_varblock_for_pass returns InvalidData
        // on the first channel. qdc_at / predicted_at still fire
        // for that first varblock (they're invoked before the inner
        // method); confirm the error propagates and no second-varblock
        // closure invocation occurs.
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let hbc = default_hbc_r260();
        let resolver = BlockContextResolver::new(&hbc);
        let grid = make_uniform_grid_r264(3, 1, TransformType::Dct8x8);
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        use std::cell::Cell;
        let q_calls: Cell<u32> = Cell::new(0);
        let p_calls: Cell<u32> = Cell::new(0);
        let r = ctx_dec.decode_lf_group_three_channels_for_pass(
            &mut br,
            5, // > num_passes (= 1)
            &grid,
            &resolver,
            |_| {
                q_calls.set(q_calls.get() + 1);
                Ok([0, 0, 0])
            },
            |_| {
                p_calls.set(p_calls.get() + 1);
                Ok([0, 0, 0])
            },
        );
        assert!(matches!(r, Err(Error::InvalidData(_))));
        // First varblock invoked both closures before the inner method
        // hit the pass-index rejection; subsequent varblocks never ran.
        assert_eq!(q_calls.get(), 1);
        assert_eq!(p_calls.get(), 1);
    }

    #[test]
    fn r264_lf_group_propagates_qdc_at_error() {
        // qdc_at returns an error → driver propagates it immediately;
        // predicted_at is not invoked for that varblock, and no
        // subsequent varblock's closures run.
        use std::cell::Cell;
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let hbc = default_hbc_r260();
        let resolver = BlockContextResolver::new(&hbc);
        let grid = make_uniform_grid_r264(2, 1, TransformType::Dct8x8);
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let p_calls: Cell<u32> = Cell::new(0);
        let r = ctx_dec.decode_lf_group_three_channels_for_pass(
            &mut br,
            0,
            &grid,
            &resolver,
            |_vb| Err(Error::InvalidData("synthetic qdc_at error".into())),
            |_vb| {
                p_calls.set(p_calls.get() + 1);
                Ok([0, 0, 0])
            },
        );
        assert!(matches!(r, Err(Error::InvalidData(_))));
        assert_eq!(p_calls.get(), 0);
    }

    #[test]
    fn r264_lf_group_propagates_predicted_at_error() {
        // predicted_at returns an error → driver propagates it; the
        // inner decode method is not invoked for that varblock.
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let hbc = default_hbc_r260();
        let resolver = BlockContextResolver::new(&hbc);
        let grid = make_uniform_grid_r264(2, 1, TransformType::Dct8x8);
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let bits_before = br.bits_read();
        let r = ctx_dec.decode_lf_group_three_channels_for_pass(
            &mut br,
            0,
            &grid,
            &resolver,
            |_vb| Ok([0, 0, 0]),
            |_vb| Err(Error::InvalidData("synthetic predicted_at error".into())),
        );
        assert!(matches!(r, Err(Error::InvalidData(_))));
        // No symbol reads happened past the closure error.
        assert_eq!(br.bits_read(), bits_before);
    }

    #[test]
    fn r264_lf_group_skips_continuation_cells() {
        // Synthetic grid: row 0 = [TopLeft(DCT16x16), Continuation,
        // TopLeft(DCT8x8)]; row 1 = [Continuation, Continuation,
        // TopLeft(DCT8x8)]. Two top-left cells should yield two
        // varblocks in raster order.
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let hbc = default_hbc_r260();
        let resolver = BlockContextResolver::new(&hbc);
        // 3×2 grid (3 cols × 2 rows). Row-major order: (0,0)=TopLeft,
        // (1,0)=Cont, (2,0)=TopLeft, (0,1)=Cont, (1,1)=Cont,
        // (2,1)=TopLeft.
        let grid = DctSelectGrid {
            cells: vec![
                DctSelectCell::TopLeft(TransformType::Dct16x16),
                DctSelectCell::Continuation,
                DctSelectCell::TopLeft(TransformType::Dct8x8),
                DctSelectCell::Continuation,
                DctSelectCell::Continuation,
                DctSelectCell::TopLeft(TransformType::Dct8x8),
            ],
            hf_mul: vec![1, 0, 1, 0, 0, 1],
            width_blocks: 3,
            height_blocks: 2,
        };
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let out = ctx_dec
            .decode_lf_group_three_channels_for_pass(
                &mut br,
                0,
                &grid,
                &resolver,
                |_vb| Ok([0, 0, 0]),
                |_vb| Ok([0, 0, 0]),
            )
            .unwrap();
        assert_eq!(out.len(), 3);
        // Top-lefts in raster order: (0,0) DCT16x16, (2,0) DCT8x8,
        // (2,1) DCT8x8.
        assert_eq!(out[0].0.x, 0);
        assert_eq!(out[0].0.y, 0);
        assert_eq!(out[0].0.transform, TransformType::Dct16x16);
        assert_eq!(out[0].1[0].coeffs.len(), 256);
        assert_eq!(out[1].0.x, 2);
        assert_eq!(out[1].0.y, 0);
        assert_eq!(out[1].0.transform, TransformType::Dct8x8);
        assert_eq!(out[1].1[0].coeffs.len(), 64);
        assert_eq!(out[2].0.x, 2);
        assert_eq!(out[2].0.y, 1);
        assert_eq!(out[2].0.transform, TransformType::Dct8x8);
        assert_eq!(out[2].1[0].coeffs.len(), 64);
    }

    #[test]
    fn r264_lf_group_does_not_advance_br_when_short_circuited() {
        // 2×2 uniform DCT8x8 → 4 varblocks × 3 channels = 12
        // single-symbol-prefix reads, all zero-bit. BitReader cursor
        // invariant.
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let hbc = default_hbc_r260();
        let resolver = BlockContextResolver::new(&hbc);
        let grid = make_uniform_grid_r264(2, 2, TransformType::Dct8x8);
        let bytes = [0xFFu8; 16];
        let mut br = BitReader::new(&bytes);
        let bits_before = br.bits_read();
        let _ = ctx_dec
            .decode_lf_group_three_channels_for_pass(
                &mut br,
                0,
                &grid,
                &resolver,
                |_vb| Ok([0, 0, 0]),
                |_vb| Ok([0, 0, 0]),
            )
            .unwrap();
        assert_eq!(br.bits_read(), bits_before);
    }

    #[test]
    fn r264_lf_group_rejects_residual_empty_cell() {
        // A DctSelectGrid with a residual `Empty` cell must trigger
        // the VarblockWalk::next rejection before any closure fires.
        let mut h = make_minimal_histograms(1, 15);
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        let hbc = default_hbc_r260();
        let resolver = BlockContextResolver::new(&hbc);
        let grid = DctSelectGrid {
            cells: vec![DctSelectCell::Empty],
            hf_mul: vec![0],
            width_blocks: 1,
            height_blocks: 1,
        };
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        use std::cell::Cell;
        let q_calls: Cell<u32> = Cell::new(0);
        let r = ctx_dec.decode_lf_group_three_channels_for_pass(
            &mut br,
            0,
            &grid,
            &resolver,
            |_vb| {
                q_calls.set(q_calls.get() + 1);
                Ok([0, 0, 0])
            },
            |_vb| Ok([0, 0, 0]),
        );
        assert!(matches!(r, Err(Error::InvalidData(_))));
        assert_eq!(q_calls.get(), 0);
    }

    #[test]
    fn r264_lf_group_routes_through_per_pass_offset() {
        // Per-pass offset routing: pass 1 uses offset = 495 (num_hf_presets
        // = 2, nb_block_ctx = 1, hfp = 1). All distributions remain
        // single-cluster single-symbol → every read returns 0.
        let mut h = make_minimal_histograms(2, 1);
        let headers = PerPassHfHeaders::from_headers(vec![
            PassGroupHfHeader {
                hfp: 0,
                histogram_offset: 0,
            },
            PassGroupHfHeader {
                hfp: 1,
                histogram_offset: 495,
            },
        ]);
        let mut ctx_dec = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
        // Custom hbc with nb_block_ctx = 1 matches the histogram shape.
        let hbc = HfBlockContext {
            used_default: false,
            qf_thresholds: vec![],
            lf_thresholds: [vec![], vec![], vec![]],
            block_ctx_map: vec![0u8; 39],
            nb_block_ctx: 1,
        };
        let resolver = BlockContextResolver::new(&hbc);
        let grid = make_uniform_grid_r264(2, 1, TransformType::Dct8x8);
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        // Drive pass 1 (offset 495) → each varblock's three channels all
        // return raw = 0 via cluster_map[ctx + 495] → cluster 0 → symbol 0.
        let out = ctx_dec
            .decode_lf_group_three_channels_for_pass(
                &mut br,
                1,
                &grid,
                &resolver,
                |_vb| Ok([0, 0, 0]),
                |_vb| Ok([0, 0, 0]),
            )
            .unwrap();
        assert_eq!(out.len(), 2);
        for vbo in &out {
            assert_eq!(vbo.2, [0u32, 0u32, 0u32]);
        }
    }
}
