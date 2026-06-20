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
//!    entropy block ([`HfCoefficientHistograms::read`]), followed by
//!    the ANS-state initialiser (`u(32)`, a no-op for prefix streams)
//!    read via [`HfCoefficientHistograms::read_ans_state_init`].
//!
//! Prior rounds built each of those three primitives but never tied
//! them together: [`HfGlobal::read`] returned after step 1, and the
//! integrated VarDCT decode path (`decode_vardct_round13` in `lib.rs`)
//! bailed with `Error::Unsupported` before steps 2 + 3 ran. This module
//! is the bundle that performs all three reads in spec order, so the
//! frame-level VarDCT decode can hand a ready-to-decode
//! [`HfCoefficientHistograms`] (post-`read_ans_state_init`) plus the
//! per-preset coefficient orders to
//! [`crate::multi_pass_hf_histogram_decoder::HfHistogramDecodeContext`].
//!
//! ## Read order is fixed (no byte alignment between pieces)
//!
//! Per §C.6 the three pieces are a single contiguous bit sequence
//! inside the HfGlobal section: HfGlobal (dequant + presets), then the
//! HfPass sequence, then the histogram block. There is **no** byte
//! alignment between them — the caller passes one [`BitReader`] through
//! all three reads. The ANS-state init is part of the §C.7.2 read (it
//! immediately follows the clustered distributions, per §C.3.2), so it
//! is performed here rather than deferred to the first symbol decode.
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

/// The fully-read HfGlobal TOC section of a VarDCT frame: the §I.2.4 /
/// §I.2.6 [`HfGlobal`] bundle, the §C.7.1 per-preset [`HfPass`]
/// sequence, and the §C.7.2 [`HfCoefficientHistograms`] entropy block
/// (with its ANS state already initialised).
///
/// Construct with [`Self::read`], which performs all three reads on a
/// single contiguous bit cursor in spec order.
#[derive(Debug)]
pub struct HfGlobalSection {
    /// §I.2.4 dequant-matrix bundle + §I.2.6 `num_hf_presets`.
    pub hf_global: HfGlobal,
    /// §C.7.1 per-preset coefficient-order bundles. Length =
    /// `hf_global.num_hf_presets`.
    pub hf_passes: Vec<HfPass>,
    /// §C.7.2 HF-coefficient histogram entropy block, with
    /// `read_ans_state_init` already applied. Ready to back a
    /// [`crate::multi_pass_hf_histogram_decoder::HfHistogramDecodeContext`].
    pub histograms: HfCoefficientHistograms,
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
    /// ANS-state initialiser — i.e. at the end of the HfGlobal section.
    ///
    /// Returns [`Error::InvalidData`] when any of the three sub-reads
    /// rejects (e.g. a §C.7.1 `used_orders` cap violation, or a
    /// §C.7.2 distribution-count overflow on a 32-bit target).
    pub fn read(br: &mut BitReader<'_>, num_groups: u64, nb_block_ctx: u32) -> Result<Self> {
        if nb_block_ctx == 0 {
            return Err(Error::InvalidData(
                "JXL HfGlobalSection: nb_block_ctx must be ≥ 1".into(),
            ));
        }

        // Step 1 — §I.2.4 dequant matrices + §I.2.6 num_hf_presets.
        let hf_global = HfGlobal::read(br, num_groups)?;
        let num_hf_presets = hf_global.num_hf_presets;

        // Step 2 — §C.7.1 HfPass sequence (num_hf_presets bundles).
        let hf_passes = read_hf_pass_sequence(br, num_hf_presets, nb_block_ctx)?;

        // Step 3 — §C.7.2 histogram block + ANS-state init, on the same
        // contiguous bit cursor (no byte alignment).
        let mut histograms =
            HfCoefficientHistograms::read_after_hf_pass_sequence(br, num_hf_presets, nb_block_ctx)?;
        histograms.read_ans_state_init(br)?;

        Ok(Self {
            hf_global,
            hf_passes,
            histograms,
        })
    }

    /// `num_hf_presets` (§I.2.6) — also the length of [`Self::hf_passes`].
    pub fn num_hf_presets(&self) -> u32 {
        self.hf_global.num_hf_presets
    }

    /// `nb_block_ctx` (§I.2.2) recovered from the histogram sizing
    /// descriptor — equals the value passed to [`Self::read`].
    pub fn nb_block_ctx(&self) -> u32 {
        self.histograms.nb_block_ctx()
    }

    /// Per-preset [`HfPass`] lookup. Returns [`Error::InvalidData`]
    /// when `preset >= num_hf_presets`.
    pub fn hf_pass(&self, preset: u32) -> Result<&HfPass> {
        self.hf_passes.get(preset as usize).ok_or_else(|| {
            Error::InvalidData(format!(
                "JXL HfGlobalSection: preset index {preset} out of {} HfPass bundles",
                self.hf_passes.len()
            ))
        })
    }

    /// Borrow the §C.7.2 histogram block (post-`read_ans_state_init`).
    /// Mutable so the caller can construct a
    /// [`HfHistogramDecodeContext`] (which borrows the histograms
    /// mutably for the ANS decode state).
    pub fn histograms_mut(&mut self) -> &mut HfCoefficientHistograms {
        &mut self.histograms
    }

    /// Bind this section's §C.7.2 histograms to a per-frame §C.8.3
    /// [`PerPassHfHeaders`] (the per-pass `hfp` / `histogram_offset`
    /// sequence read at the start of each pass's PassGroup payload) to
    /// produce the [`HfHistogramDecodeContext`] the per-LfGroup VarDCT
    /// decode walks against.
    ///
    /// This is the bridge from the parsed HfGlobal section to the
    /// histogram-backed decode: the §C.7.2 stream + its ANS-state init
    /// live in `self.histograms` (already read by [`Self::read`]); the
    /// per-pass `histogram_offset` routing lives in `headers`.
    /// [`HfHistogramDecodeContext::new`] cross-validates every
    /// `headers.hfp(p) < num_hf_presets` against this section's
    /// authoritative `num_hf_presets`.
    ///
    /// The returned context borrows `self.histograms` mutably (it owns
    /// the per-symbol ANS decode state), so the section is borrowed for
    /// the lifetime of the decode.
    pub fn decode_context<'a>(
        &'a mut self,
        headers: &PerPassHfHeaders,
    ) -> Result<HfHistogramDecodeContext<'a>> {
        HfHistogramDecodeContext::new(&mut self.histograms, headers)
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

        let section = HfGlobalSection::read(&mut br, 1, 1).unwrap();

        // HfGlobal: default encoding, one preset.
        assert!(section.hf_global.dequant_default);
        assert_eq!(section.num_hf_presets(), 1);
        assert!(section.hf_global.dequant_matrices.is_empty());

        // HfPass[0]: used_orders == 0 → every order is the natural order.
        assert_eq!(section.hf_passes.len(), 1);
        assert_eq!(section.hf_pass(0).unwrap().used_orders, 0);
        assert!(section.hf_pass(1).is_err());

        // §C.7.2 histograms: 495 × 1 × 1 distributions, single cluster.
        assert_eq!(section.histograms.num_distributions(), 495);
        assert_eq!(section.nb_block_ctx(), 1);
        assert!(section.histograms.entropy.use_prefix_code);
        assert_eq!(section.histograms.entropy.cluster_map.len(), 495);
        assert_eq!(section.histograms.entropy.entropies.len(), 1);
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
        let _section = HfGlobalSection::read(&mut br_bundle, 1, 1).unwrap();
        let bundle_bits = br_bundle.bits_read();

        // Piecewise read of the same three pieces in the same order.
        let mut br_pieces = BitReader::new(&bytes);
        let hg = HfGlobal::read(&mut br_pieces, 1).unwrap();
        let _passes = read_hf_pass_sequence(&mut br_pieces, hg.num_hf_presets, 1).unwrap();
        let mut histos =
            HfCoefficientHistograms::read_after_hf_pass_sequence(&mut br_pieces, 1, 1).unwrap();
        histos.read_ans_state_init(&mut br_pieces).unwrap();
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
        let mut section = HfGlobalSection::read(&mut br, 1, 1).unwrap();

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
        let mut section = HfGlobalSection::read(&mut br, 1, 1).unwrap();
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
        let r = HfGlobalSection::read(&mut br, 1, 0);
        assert!(matches!(r, Err(Error::InvalidData(_))));
        // The guard runs before any HfGlobal bits are consumed.
        assert_eq!(br.bits_read(), bits_before);
    }
}
