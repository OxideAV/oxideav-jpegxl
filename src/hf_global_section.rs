//! `HfGlobalSection` — the full §C.7 HfGlobal-section read.
//!
//! ## Scope (round 349)
//!
//! The HfGlobal TOC slot of a VarDCT frame (Table C.17 / §C.6) is read
//! in three consecutive pieces, all on the **same** bit cursor with no
//! byte alignment between them:
//!
//! 1. **§I.2.4 + §I.2.6 dequant-matrix bundle + `num_hf_presets`** —
//!    parsed by [`HfGlobal::read`]. The bit cursor stops immediately
//!    after `num_hf_presets_minus_1`.
//! 2. **§C.7.1 HfPass sequence** — `num_hf_presets` consecutive
//!    [`HfPass`] bundles (Listing C.12: `used_orders` selector + the
//!    permuted / natural coefficient orders), parsed by
//!    [`read_hf_pass_sequence`].
//! 3. **§C.7.2 HF-coefficient histograms** — the
//!    `495 × num_hf_presets × nb_block_ctx` clustered-distribution
//!    entropy block ([`HfCoefficientHistograms::read`]).
//!
//! Prior rounds built each of those three primitives but never tied
//! them together: [`HfGlobal::read`] returned after step 1, and the
//! integrated VarDCT decode path (`decode_vardct_round13` in `lib.rs`)
//! bailed with `Error::Unsupported` before steps 2 + 3 ran. This module
//! is the bundle that performs all three reads in spec order, so the
//! frame-level VarDCT decode can hand the parsed
//! [`HfCoefficientHistograms`] plus the per-preset coefficient orders
//! to [`crate::multi_pass_hf_histogram_decoder::HfHistogramDecodeContext`].
//!
//! ## Read order is fixed (no byte alignment between pieces)
//!
//! Per §C.6 the three pieces are a single contiguous bit sequence
//! inside the HfGlobal section: HfGlobal (dequant + presets), then the
//! HfPass sequence, then the histogram block. There is **no** byte
//! alignment between them — the caller passes one [`BitReader`] through
//! all three reads.
//!
//! ## The ANS-state init is per PassGroup stream, NOT part of this read
//!
//! Per D.3.3 the `u(32)` ANS state initialiser is read "immediately
//! before reading the first symbol from a new ANS stream". The symbols
//! routed through the §C.7.2 histograms live in the **PassGroup**
//! sections (§C.8.3) — one entropy stream per section — so the state
//! init belongs to each PassGroup's own reader, right after that
//! section's `hfp` header. Rounds 349–385 read one state init at the
//! end of this section instead; that was invisible on single-TOC
//! single-group frames (0-bit `hfp`, shared cursor) but wrong for
//! multi-entry TOCs. Round 389 moved it to the PassGroup decode.
//!
//! ## `nb_block_ctx` provenance
//!
//! The `nb_block_ctx` invariant that sizes both the HfPass histogram
//! count and the §C.7.2 distribution count comes from the LfGlobal
//! `HfBlockContext` (§I.2.2, `nb_block_ctx = max(block_ctx_map) + 1`),
//! NOT from anything inside the HfGlobal section. The caller threads it
//! in from `lf_global.hf_block_context`.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::hf_coefficient_histograms::HfCoefficientHistograms;
use crate::hf_global::HfGlobal;
use crate::hf_pass::{read_hf_pass_sequence, HfPass};
use crate::multi_pass_hf_header::PerPassHfHeaders;
use crate::multi_pass_hf_histogram_decoder::HfHistogramDecodeContext;

/// One pass's slice of the HfGlobal section (Table C.1 lists `HfPass
/// hf_pass[num_passes]` after HfGlobal): the §C.7.1 per-preset
/// coefficient-order bundles (Listing C.12, read `num_hf_presets`
/// times) followed by that pass's §C.7.2 HF-coefficient histogram
/// block (`495 × num_hf_presets × nb_block_ctx` clustered
/// distributions).
#[derive(Debug)]
pub struct HfPassData {
    /// §C.7.1 per-preset coefficient-order bundles. Length =
    /// `num_hf_presets`.
    pub presets: Vec<HfPass>,
    /// §C.7.2 HF-coefficient histogram entropy block for this pass.
    /// The per-stream ANS state initialiser is **not** yet read — each
    /// PassGroup section is its own entropy stream, so the caller
    /// invokes [`HfCoefficientHistograms::read_ans_state_init`] on
    /// that section's reader (after its `hfp` header) before the
    /// first symbol decode.
    pub histograms: HfCoefficientHistograms,
}

impl HfPassData {
    /// Build the single-pass histogram decode context a `(pass,
    /// group)` PassGroup section decodes against: this pass's §C.7.2
    /// histograms bound to the section's `hfp` selection, with the
    /// `hfp`-selected §C.7.1 coefficient orders attached. The caller
    /// has already read the section's ANS state init.
    pub fn single_pass_context<'a>(
        &'a mut self,
        headers: &PerPassHfHeaders,
    ) -> Result<HfHistogramDecodeContext<'a>> {
        if headers.num_passes() != 1 {
            return Err(Error::InvalidData(format!(
                "JXL HfPassData::single_pass_context: expected a single-pass header \
                 (one hfp per PassGroup section), got {} passes",
                headers.num_passes()
            )));
        }
        let Self {
            presets,
            histograms,
        } = self;
        let mut ctx = HfHistogramDecodeContext::new(histograms, headers)?;
        let hfp = headers.hfp(0)? as usize;
        let preset = presets.get(hfp).ok_or_else(|| {
            Error::InvalidData(format!(
                "JXL HfPassData: hfp {hfp} out of {} preset bundles",
                presets.len()
            ))
        })?;
        ctx.set_pass_orders(vec![preset])?;
        Ok(ctx)
    }
}

/// The fully-read HfGlobal TOC section of a VarDCT frame: the §I.2.4 /
/// §I.2.6 [`HfGlobal`] bundle followed by `num_passes` [`HfPassData`]
/// slices (§C.7.1 orders + §C.7.2 histograms per pass — Table C.1's
/// `hf_pass[num_passes]`, carried in the HfGlobal TOC slot per §C.3.1
/// "one for HfGlobal followed by HfPass data for all the passes").
///
/// Construct with [`Self::read`], which performs every read on a
/// single contiguous bit cursor in spec order.
#[derive(Debug)]
pub struct HfGlobalSection {
    /// §I.2.4 dequant-matrix bundle + §I.2.6 `num_hf_presets`.
    pub hf_global: HfGlobal,
    /// Per-pass §C.7 data, length = `num_passes`.
    pub passes: Vec<HfPassData>,
}

impl HfGlobalSection {
    /// Read the complete HfGlobal section from `br`.
    ///
    /// * `br` must be positioned at the start of the HfGlobal TOC slot
    ///   (the §I.2.4 `u(1)` default-encoding flag), exactly where
    ///   [`HfGlobal::read`] expects to begin.
    /// * `num_groups` parameterises the §I.2.6 `num_hf_presets`
    ///   bit-width (`u(ceil(log2(num_groups)))`).
    /// * `nb_block_ctx` is the LfGlobal §I.2.2 `HfBlockContext`
    ///   invariant (`max(block_ctx_map) + 1`); it sizes both the
    ///   §C.7.1 per-pass histogram-distribution count and the §C.7.2
    ///   total (`495 × num_hf_presets × nb_block_ctx`).
    ///
    /// On return `br` is positioned immediately after the §C.7.2
    /// histogram block — i.e. at the end of the HfGlobal section's
    /// defined bits. (The per-stream ANS state init is read later, on
    /// each PassGroup section's own reader — see the module notes.)
    ///
    /// Returns [`Error::InvalidData`] when any of the three sub-reads
    /// rejects (e.g. a §C.7.1 `used_orders` cap violation, or a
    /// §C.7.2 distribution-count overflow on a 32-bit target).
    pub fn read(
        br: &mut BitReader<'_>,
        num_groups: u64,
        nb_block_ctx: u32,
        num_passes: u32,
    ) -> Result<Self> {
        if nb_block_ctx == 0 {
            return Err(Error::InvalidData(
                "JXL HfGlobalSection: nb_block_ctx must be ≥ 1".into(),
            ));
        }
        if num_passes == 0 {
            return Err(Error::InvalidData(
                "JXL HfGlobalSection: num_passes must be ≥ 1".into(),
            ));
        }

        // Step 1 — §I.2.4 dequant matrices + §I.2.6 num_hf_presets.
        let hf_global = HfGlobal::read(br, num_groups)?;
        let num_hf_presets = hf_global.num_hf_presets;

        // Step 2 — Table C.1 `hf_pass[num_passes]`: for each pass, the
        // §C.7.1 order-bundle sequence (`num_hf_presets` bundles) then
        // that pass's §C.7.2 histogram block, all on the same
        // contiguous bit cursor (no byte alignment).
        //
        // The ANS-state initialiser is NOT read here. Per D.3.3 the
        // `u(32)` state init is read "immediately before reading the
        // first symbol from a new ANS stream" — and the symbols routed
        // through these histograms are read from the **PassGroup**
        // sections (§C.8.3), each of which is its own entropy stream
        // with its own state init on its own section reader (after
        // that section's `hfp` header). Reading the init here (a)
        // consumed 32 bits past the real end of a multi-entry TOC's
        // HfGlobal slot, and (b) left the multi-group PassGroup decode
        // without its per-section re-init. The single-TOC single-group
        // case was unaffected only because `hfp` is a 0-bit read there
        // (num_hf_presets == 1), so "after HfGlobal" and "after hfp"
        // were the same cursor position.
        let mut passes = Vec::with_capacity(num_passes as usize);
        for _ in 0..num_passes {
            let presets = read_hf_pass_sequence(br, num_hf_presets, nb_block_ctx)?;
            let histograms = HfCoefficientHistograms::read_after_hf_pass_sequence(
                br,
                num_hf_presets,
                nb_block_ctx,
            )?;
            passes.push(HfPassData {
                presets,
                histograms,
            });
        }

        Ok(Self { hf_global, passes })
    }

    /// `num_hf_presets` (§I.2.6) — also the length of every pass's
    /// preset list.
    pub fn num_hf_presets(&self) -> u32 {
        self.hf_global.num_hf_presets
    }

    /// `nb_block_ctx` (§I.2.2) recovered from the pass-0 histogram
    /// sizing descriptor — equals the value passed to [`Self::read`].
    pub fn nb_block_ctx(&self) -> u32 {
        self.passes[0].histograms.nb_block_ctx()
    }

    /// Per-pass [`HfPassData`] lookup (mutable — the decode context
    /// borrows the pass's histograms mutably for the ANS state).
    pub fn pass_data_mut(&mut self, p: u32) -> Result<&mut HfPassData> {
        let n = self.passes.len();
        self.passes.get_mut(p as usize).ok_or_else(|| {
            Error::InvalidData(format!(
                "JXL HfGlobalSection: pass index {p} out of {n} HfPass slices"
            ))
        })
    }

    /// Pass-0 per-preset [`HfPass`] lookup. Returns
    /// [`Error::InvalidData`] when `preset >= num_hf_presets`.
    pub fn hf_pass(&self, preset: u32) -> Result<&HfPass> {
        self.passes[0].presets.get(preset as usize).ok_or_else(|| {
            Error::InvalidData(format!(
                "JXL HfGlobalSection: preset index {preset} out of {} HfPass bundles",
                self.passes[0].presets.len()
            ))
        })
    }

    /// Borrow the pass-0 §C.7.2 histogram block. Mutable so the caller
    /// can run the per-PassGroup-stream `read_ans_state_init` and
    /// construct a [`HfHistogramDecodeContext`] (which borrows the
    /// histograms mutably for the ANS decode state).
    pub fn histograms_mut(&mut self) -> &mut HfCoefficientHistograms {
        &mut self.passes[0].histograms
    }

    /// Bind this section's §C.7.2 histograms to a per-frame §C.8.3
    /// [`PerPassHfHeaders`] (the per-pass `hfp` / `histogram_offset`
    /// sequence read at the start of each pass's PassGroup payload) to
    /// produce the [`HfHistogramDecodeContext`] the per-LfGroup VarDCT
    /// decode walks against.
    ///
    /// This is the bridge from the parsed HfGlobal section to the
    /// histogram-backed decode: the §C.7.2 distributions live in
    /// `self.histograms` (read by [`Self::read`]; the caller runs the
    /// per-PassGroup-stream ANS state init separately); the per-pass
    /// `histogram_offset` routing lives in `headers`.
    /// [`HfHistogramDecodeContext::new`] cross-validates every
    /// `headers.hfp(p) < num_hf_presets` against this section's
    /// authoritative `num_hf_presets`.
    ///
    /// The returned context borrows `self.histograms` mutably (it owns
    /// the per-symbol ANS decode state), so the section is borrowed for
    /// the lifetime of the decode.
    ///
    /// The context also carries the per-pass §C.7.1 coefficient-order
    /// sources: for each pass `p`, the `headers.hfp(p)`-selected
    /// [`crate::hf_pass::HfPass`] bundle from this section, so the
    /// §C.8.3 Listing C.14 `coeffs[order[k]]` placement uses the
    /// signalled (possibly permuted) order rather than the bare natural
    /// order.
    /// NOTE: this binding routes every header pass through the
    /// **pass-0** histogram block, which is only correct for
    /// single-pass frames (each pass owns its own §C.7.2 block —
    /// multi-pass callers use [`HfPassData::single_pass_context`] per
    /// PassGroup section instead). Rejected when this section carries
    /// more than one pass and the headers claim more than one.
    pub fn decode_context<'a>(
        &'a mut self,
        headers: &PerPassHfHeaders,
    ) -> Result<HfHistogramDecodeContext<'a>> {
        if self.passes.len() != 1 && headers.num_passes() != 1 {
            return Err(Error::InvalidData(format!(
                "JXL HfGlobalSection::decode_context: {}-pass section with {}-pass \
                 headers — multi-pass decodes bind per-pass contexts via \
                 HfPassData::single_pass_context",
                self.passes.len(),
                headers.num_passes()
            )));
        }
        let pass0 = &mut self.passes[0];
        let HfPassData {
            presets,
            histograms,
        } = pass0;
        let mut ctx = HfHistogramDecodeContext::new(histograms, headers)?;
        let mut orders = Vec::with_capacity(headers.num_passes() as usize);
        for p in 0..headers.num_passes() {
            let hfp = headers.hfp(p)? as usize;
            let hf_pass = presets.get(hfp).ok_or_else(|| {
                Error::InvalidData(format!(
                    "JXL HfGlobalSection: pass {p} hfp {hfp} out of {} HfPass bundles",
                    presets.len()
                ))
            })?;
            orders.push(hf_pass);
        }
        ctx.set_pass_orders(orders)?;
        Ok(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ans::test_helpers::pack_lsb;

    /// The minimal prefix-coded §C.7.2 histogram prelude (single cluster,
    /// `nbits = 0`, single-symbol prefix code), shared by the chaining
    /// tests. This is exactly the prelude the `hf_coefficient_histograms`
    /// suite validates byte-for-byte (`r247_read_with_minimal_prelude_*`).
    fn histogram_prelude_parts() -> Vec<(u32, u32)> {
        vec![
            (0, 1), // lz77_enabled = 0
            (1, 1), // is_simple = 1
            (0, 2), // nbits = 0 → all distributions → cluster 0
            (1, 1), // use_prefix_code = 1 → log_alphabet_size = 15
            (0, 4), // split_exponent = 0
            (0, 1), // prefix count selector = 0 → count = 1 (single-symbol)
        ]
    }

    /// Single-group, default-encoding VarDCT frame: `num_groups == 1`
    /// (zero preset bits → `num_hf_presets == 1`), one HfPass with
    /// `used_orders == 0` (all natural orders, no permutation stream),
    /// then the minimal prefix-coded §C.7.2 histogram block — all on
    /// one contiguous LSB-first bit cursor.
    ///
    /// Wire layout:
    ///   - HfGlobal: `u(1) = 1` (dequant default), 0 preset bits.
    ///   - HfPass[0]: `used_orders` selector. The §C.7.1 `U32` selector
    ///     is `U32(Val(0x5F), Val(0x13), Val(0), Bits(13))`; the 2-bit
    ///     selector code `0b10` (= 2, LSB-first) picks index 2
    ///     (`Val(0)`) → `used_orders == 0` (natural orders, no entropy
    ///     read inside the pass).
    ///   - §C.7.2 histograms: the minimal prefix prelude above.
    ///
    /// Asserts the three pieces chain in spec order and the bundle
    /// surfaces the expected preset count, orders, and histogram shape.
    #[test]
    fn single_group_default_encoding_natural_orders_chains() {
        let mut parts: Vec<(u32, u32)> = vec![
            (1, 1), // HfGlobal: dequant_default = 1; num_groups == 1 → 0 preset bits
            (2, 2), // HfPass[0] used_orders selector index 2 (Val(0)) → used_orders = 0
        ];
        parts.extend(histogram_prelude_parts());
        let bytes = pack_lsb(&parts);
        let mut br = BitReader::new(&bytes);

        let section = HfGlobalSection::read(&mut br, 1, 1, 1).unwrap();

        // HfGlobal: default encoding, one preset.
        assert!(section.hf_global.dequant_default);
        assert_eq!(section.num_hf_presets(), 1);
        assert!(section.hf_global.dequant_matrices.is_empty());

        // HfPass[0]: used_orders == 0 → every order is the natural order.
        assert_eq!(section.passes.len(), 1);
        assert_eq!(section.passes[0].presets.len(), 1);
        assert_eq!(section.hf_pass(0).unwrap().used_orders, 0);
        assert!(section.hf_pass(1).is_err());

        // §C.7.2 histograms: 495 × 1 × 1 distributions, single cluster.
        assert_eq!(section.passes[0].histograms.num_distributions(), 495);
        assert_eq!(section.nb_block_ctx(), 1);
        assert!(section.passes[0].histograms.entropy.use_prefix_code);
        assert_eq!(section.passes[0].histograms.entropy.cluster_map.len(), 495);
        assert_eq!(section.passes[0].histograms.entropy.entropies.len(), 1);
    }

    /// The cursor position after [`HfGlobalSection::read`] is exactly
    /// where an independent HfGlobal → HfPass → histograms re-read on
    /// the same bytes lands — i.e. no bits are skipped or double-read
    /// across the three sub-reads.
    #[test]
    fn cursor_matches_independent_piecewise_read() {
        use crate::hf_coefficient_histograms::HfCoefficientHistograms;
        use crate::hf_global::HfGlobal;
        use crate::hf_pass::read_hf_pass_sequence;

        let mut parts: Vec<(u32, u32)> = vec![(1, 1), (2, 2)];
        parts.extend(histogram_prelude_parts());
        let bytes = pack_lsb(&parts);

        // Bundled read.
        let mut br_bundle = BitReader::new(&bytes);
        let _section = HfGlobalSection::read(&mut br_bundle, 1, 1, 1).unwrap();
        let bundle_bits = br_bundle.bits_read();

        // Piecewise read of the same three pieces in the same order.
        // (No ANS-state init: that is a per-PassGroup-stream read, not
        // part of the HfGlobal section — see the module notes.)
        let mut br_pieces = BitReader::new(&bytes);
        let hg = HfGlobal::read(&mut br_pieces, 1).unwrap();
        let _passes = read_hf_pass_sequence(&mut br_pieces, hg.num_hf_presets, 1).unwrap();
        let _histos =
            HfCoefficientHistograms::read_after_hf_pass_sequence(&mut br_pieces, 1, 1).unwrap();
        let pieces_bits = br_pieces.bits_read();

        assert_eq!(bundle_bits, pieces_bits);
    }

    /// `decode_context` binds the section's §C.7.2 histograms to a
    /// per-frame §C.8.3 [`PerPassHfHeaders`], producing the
    /// [`HfHistogramDecodeContext`] the per-LfGroup decode walks
    /// against. A single-pass `hfp = 0` header yields offset 0.
    #[test]
    fn decode_context_binds_histograms_to_per_pass_headers() {
        use crate::multi_pass_hf_header::PerPassHfHeaders;
        use crate::pass_group_hf::PassGroupHfHeader;

        let mut parts: Vec<(u32, u32)> = vec![(1, 1), (2, 2)];
        parts.extend(histogram_prelude_parts());
        let bytes = pack_lsb(&parts);
        let mut br = BitReader::new(&bytes);
        let mut section = HfGlobalSection::read(&mut br, 1, 1, 1).unwrap();

        // Single pass, hfp = 0 → histogram_offset = 0.
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let ctx = section.decode_context(&headers).unwrap();
        assert_eq!(ctx.num_passes(), 1);
        assert_eq!(ctx.histogram_offset(0).unwrap(), 0);
    }

    /// `decode_context` rejects a per-pass header whose `hfp` exceeds the
    /// section's authoritative `num_hf_presets` (the cross-container
    /// invariant `HfHistogramDecodeContext::new` enforces).
    #[test]
    fn decode_context_rejects_out_of_range_hfp() {
        use crate::multi_pass_hf_header::PerPassHfHeaders;
        use crate::pass_group_hf::PassGroupHfHeader;

        let mut parts: Vec<(u32, u32)> = vec![(1, 1), (2, 2)];
        parts.extend(histogram_prelude_parts());
        let bytes = pack_lsb(&parts);
        let mut br = BitReader::new(&bytes);
        let mut section = HfGlobalSection::read(&mut br, 1, 1, 1).unwrap();
        assert_eq!(section.num_hf_presets(), 1);

        // hfp = 1 ≥ num_hf_presets = 1 → rejected.
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 1,
            histogram_offset: 495,
        }]);
        let r = section.decode_context(&headers);
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn nb_block_ctx_zero_rejected() {
        let bytes = pack_lsb(&[(1, 1)]);
        let mut br = BitReader::new(&bytes);
        let bits_before = br.bits_read();
        let r = HfGlobalSection::read(&mut br, 1, 0, 1);
        assert!(matches!(r, Err(Error::InvalidData(_))));
        // The guard runs before any HfGlobal bits are consumed.
        assert_eq!(br.bits_read(), bits_before);
    }

    /// Table C.1 `hf_pass[num_passes]`: a two-pass section carries TWO
    /// (§C.7.1 orders + §C.7.2 histograms) slices after the shared
    /// HfGlobal bundle, on one contiguous cursor — and the cursor lands
    /// exactly where a piecewise re-read of the same five pieces does.
    #[test]
    fn two_pass_section_reads_two_hf_pass_slices() {
        use crate::hf_coefficient_histograms::HfCoefficientHistograms;
        use crate::hf_global::HfGlobal;
        use crate::hf_pass::read_hf_pass_sequence;

        let mut parts: Vec<(u32, u32)> = vec![
            (1, 1), // HfGlobal: dequant_default = 1; num_groups == 1 → 0 preset bits
        ];
        // Pass 0: used_orders = Val(0) + minimal histogram block.
        parts.push((2, 2));
        parts.extend(histogram_prelude_parts());
        // Pass 1: same shape.
        parts.push((2, 2));
        parts.extend(histogram_prelude_parts());
        let bytes = pack_lsb(&parts);

        let mut br = BitReader::new(&bytes);
        let section = HfGlobalSection::read(&mut br, 1, 1, 2).unwrap();
        assert_eq!(section.passes.len(), 2);
        for p in &section.passes {
            assert_eq!(p.presets.len(), 1);
            assert_eq!(p.histograms.num_distributions(), 495);
        }
        let bundle_bits = br.bits_read();

        // Piecewise.
        let mut br2 = BitReader::new(&bytes);
        let hg = HfGlobal::read(&mut br2, 1).unwrap();
        for _ in 0..2 {
            let _ = read_hf_pass_sequence(&mut br2, hg.num_hf_presets, 1).unwrap();
            let _ = HfCoefficientHistograms::read_after_hf_pass_sequence(&mut br2, 1, 1).unwrap();
        }
        assert_eq!(bundle_bits, br2.bits_read());
    }

    /// `HfPassData::single_pass_context` binds one pass's histograms +
    /// the hfp-selected orders; a multi-pass header is rejected.
    #[test]
    fn single_pass_context_binds_one_pass() {
        use crate::multi_pass_hf_header::PerPassHfHeaders;
        use crate::pass_group_hf::PassGroupHfHeader;

        let mut parts: Vec<(u32, u32)> = vec![(1, 1), (2, 2)];
        parts.extend(histogram_prelude_parts());
        parts.push((2, 2));
        parts.extend(histogram_prelude_parts());
        let bytes = pack_lsb(&parts);
        let mut br = BitReader::new(&bytes);
        let mut section = HfGlobalSection::read(&mut br, 1, 1, 2).unwrap();

        let one = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        for p in 0..2 {
            let ctx = section
                .pass_data_mut(p)
                .unwrap()
                .single_pass_context(&one)
                .unwrap();
            assert_eq!(ctx.num_passes(), 1);
            assert_eq!(ctx.histogram_offset(0).unwrap(), 0);
        }
        assert!(section.pass_data_mut(2).is_err());

        let two = PerPassHfHeaders::from_headers(vec![
            PassGroupHfHeader {
                hfp: 0,
                histogram_offset: 0,
            };
            2
        ]);
        let r = section.pass_data_mut(0).unwrap().single_pass_context(&two);
        assert!(matches!(r, Err(Error::InvalidData(_))));

        // decode_context on a multi-pass section with multi-pass
        // headers is likewise rejected (that binding routes through
        // pass 0's histograms only).
        let r = section.decode_context(&two);
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }
}
