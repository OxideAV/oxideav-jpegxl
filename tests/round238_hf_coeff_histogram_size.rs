//! Round 238 integration coverage —
//! [`oxideav_jpegxl::hf_coeff_histogram_size::HfCoefficientHistogramSize`].
//!
//! ISO/IEC FDIS 18181-1:2021 §C.7.2 read-size derivation
//! (`num_distributions = 495 × num_hf_presets × nb_block_ctx`) and
//! §C.8.3 per-pass routing offset (`offset = 495 × nb_block_ctx ×
//! hfp`), both routed through one typed primitive so the spec
//! constant has one home and the two existing call sites
//! ([`oxideav_jpegxl::hf_pass::HfPass::read`] and
//! [`oxideav_jpegxl::pass_group_hf::PassGroupHfHeader::read`]) compute
//! the same numbers as the primitive itself.

use oxideav_jpegxl::bitreader::BitReader;
use oxideav_jpegxl::hf_coeff_histogram_size::{
    HfCoefficientHistogramSize, PER_PRESET_PER_BLOCK_CTX,
};
use oxideav_jpegxl::hf_pass::HfPass;
use oxideav_jpegxl::pass_group_hf::PassGroupHfHeader;

#[test]
fn r238_integration_spec_constant_is_495() {
    // §C.7.2 reads literally `495 × num_hf_presets × nb_block_ctx`;
    // pin the constant at the public boundary.
    assert_eq!(PER_PRESET_PER_BLOCK_CTX, 495);
}

#[test]
fn r238_integration_default_hbc_shape_one_preset_7425() {
    // The default 39-entry `block_ctx_map` shipped by HfBlockContext
    // (§I.2.2) has max value 14 → nb_block_ctx = 15. With
    // num_hf_presets = 1 the §C.7.2 read consumes 7425 clustered
    // distributions.
    let default_map = vec![
        7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 9, 9, 10, 11, 12, 13, 14, 0, 0, 0, 0, 7, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let size = HfCoefficientHistogramSize::from_block_ctx_map(&default_map, 1).unwrap();
    assert_eq!(size.nb_block_ctx, 15);
    assert_eq!(size.num_distributions(), 7425);
    assert_eq!(size.per_preset(), 7425);
    assert_eq!(size.offset_for_hfp(0).unwrap(), 0);
}

#[test]
fn r238_integration_hf_pass_read_matches_primitive() {
    // HfPass::read computes its `num_histogram_distributions` field
    // by routing through HfCoefficientHistogramSize. Driving it with
    // a `used_orders = 0` payload (selector 2 → Val(0), 2 bits LSB
    // value 2 → byte 0b0000_0010) gives a trivial natural-order
    // HfPass that we can compare against the primitive's expected
    // size for the same (num_hf_presets, nb_block_ctx) pair.
    for &(num_hf_presets, nb_block_ctx) in
        &[(1u32, 15u32), (2u32, 15u32), (4u32, 7u32), (8u32, 1u32)]
    {
        let bytes = [0b0000_0010u8];
        let mut br = BitReader::new(&bytes);
        let hp = HfPass::read(&mut br, num_hf_presets, nb_block_ctx).unwrap();
        let size = HfCoefficientHistogramSize::new(num_hf_presets, nb_block_ctx).unwrap();
        assert_eq!(
            hp.num_histogram_distributions,
            size.num_distributions(),
            "HfPass::read mismatch for ({num_hf_presets}, {nb_block_ctx})"
        );
    }
}

#[test]
fn r238_integration_pass_group_hf_offset_matches_primitive() {
    // PassGroupHfHeader::read derives `histogram_offset = 495 ×
    // nb_block_ctx × hfp` through the same primitive. Drive a
    // single-preset (num_hf_presets = 1 → 0 bits read) header with an
    // empty BitReader payload, then a 4-preset header where we read
    // 2 bits to encode hfp.
    // Single-preset: nbits(ceil_log2(1)) = 0 → hfp = 0, offset = 0.
    let bytes = [0u8];
    let mut br = BitReader::new(&bytes);
    let hdr = PassGroupHfHeader::read(&mut br, 1, 15).unwrap();
    assert_eq!(hdr.hfp, 0);
    assert_eq!(hdr.histogram_offset, 0);
    let size_one = HfCoefficientHistogramSize::new(1, 15).unwrap();
    assert_eq!(hdr.histogram_offset, size_one.offset_for_hfp(0).unwrap());

    // 4-preset with hfp = 3: nbits = 2, LSB-first value 3 → bits 11.
    let bytes = [0b0000_0011u8];
    let mut br = BitReader::new(&bytes);
    let hdr = PassGroupHfHeader::read(&mut br, 4, 15).unwrap();
    assert_eq!(hdr.hfp, 3);
    let size_four = HfCoefficientHistogramSize::new(4, 15).unwrap();
    assert_eq!(hdr.histogram_offset, size_four.offset_for_hfp(3).unwrap());
    // Sanity: matches the bare 495 × 15 × 3 arithmetic literal too.
    assert_eq!(hdr.histogram_offset, 495 * 15 * 3);
}

#[test]
fn r238_integration_per_preset_steps_uniformly() {
    // The per-pass offset arithmetic steps uniformly by `per_preset()`
    // as hfp grows from 0 to num_hf_presets - 1. Pin this for the
    // upstream §C.8.3 multi-pass driver which routes the offset as
    // a per-pass constant.
    let size = HfCoefficientHistogramSize::new(8, 15).unwrap();
    let step = size.per_preset();
    assert_eq!(step, 495 * 15);
    for hfp in 0..8u32 {
        assert_eq!(size.offset_for_hfp(hfp).unwrap(), step * hfp as u64);
    }
    assert!(size.offset_for_hfp(8).is_err());
}

#[test]
fn r238_integration_rejects_zero_inputs() {
    // Defensive: zero inputs are rejected even though the public
    // entry-points (HfGlobal §I.2.6, HfBlockContext §I.2.2) both
    // guarantee ≥ 1; the typed primitive defends against upstream
    // constructor bugs.
    assert!(HfCoefficientHistogramSize::new(0, 15).is_err());
    assert!(HfCoefficientHistogramSize::new(4, 0).is_err());
    assert!(HfCoefficientHistogramSize::from_block_ctx_map(&[], 4).is_err());
}
