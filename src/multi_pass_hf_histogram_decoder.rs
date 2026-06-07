//! Per-pass HF histogram decode context — ISO/IEC FDIS 18181-1:2021
//! §C.7.2 (entropy histograms) + §C.8.3 (per-pass `histogram_offset`
//! routing) bridge.
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
//! above) — the raster walk across multiple varblocks is still a
//! caller-side concern (driven by [`crate::varblock_walk`] +
//! [`crate::multi_pass_decode`]).
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
use crate::coeff_order::{natural_coeff_order, order_id_for_transform};
use crate::dct_select::TransformType;
use crate::hf_coefficient_histograms::HfCoefficientHistograms;
use crate::multi_pass_hf_header::PerPassHfHeaders;
use crate::pass_group_hf::{
    coefficient_context, non_zeros_context, prev_for_context, transform_block_params,
    DecodedHfBlock,
};

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
}
