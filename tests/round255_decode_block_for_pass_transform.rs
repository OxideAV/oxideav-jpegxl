//! Round 255 integration coverage —
//! [`oxideav_jpegxl::multi_pass_hf_histogram_decoder::HfHistogramDecodeContext::decode_block_for_pass_transform`].
//!
//! ISO/IEC FDIS 18181-1:2021 §C.8.3 + Listing C.13 + Listing C.14 —
//! the bundled per-varblock decode method that composes round 252's
//! per-pass histogram routing
//! ([`HfHistogramDecodeContext::non_zeros_at`] /
//! [`HfHistogramDecodeContext::coefficient_at`]) with the round-90
//! Listing C.14 state machine into a single
//! [`TransformType`]-typed call.
//!
//! These integration tests pin the public-surface invariants from a
//! consumer's vantage point (the per-LfGroup multi-pass driver in
//! [`oxideav_jpegxl::multi_pass_decode`] is the eventual caller):
//!
//! * Default-prefix short-circuit (single-symbol prefix → `non_zeros
//!   == 0` → no coefficient symbols read) holds across DCT8×8 /
//!   DCT16×16 / DCT16×8 / DCT8×16 / DCT4×4 transforms and the
//!   returned `coeffs` vector is the correct length for each
//!   transform's `(num_blocks, size)` derivation.
//! * The per-pass `histogram_offset` is honoured — passing `p = 1`
//!   against a 2-preset histogram bundle routes through
//!   `cluster_map[ctx + 495]` rather than `cluster_map[ctx]`.
//! * The defensive rejections (`p >= num_passes`, `u32` overflow on
//!   `ctx + offset`) bubble out as [`oxideav_core::Error`] without
//!   panicking.
//! * No bit reads on a short-circuited block — the [`BitReader`]
//!   cursor must not advance.

use oxideav_jpegxl::bitreader::BitReader;
use oxideav_jpegxl::dct_select::TransformType;
use oxideav_jpegxl::hf_coeff_histogram_size::HfCoefficientHistogramSize;
use oxideav_jpegxl::hf_coefficient_histograms::HfCoefficientHistograms;
use oxideav_jpegxl::multi_pass_hf_header::PerPassHfHeaders;
use oxideav_jpegxl::multi_pass_hf_histogram_decoder::HfHistogramDecodeContext;
use oxideav_jpegxl::pass_group_hf::PassGroupHfHeader;

/// §D.3 prelude bytes for the minimal single-cluster, single-symbol
/// prefix-coded histogram block. Mirrors the round-252 integration
/// test helper exactly; documented in detail there.
fn minimal_prefix_prelude_bytes() -> [u8; 2] {
    [0b0001_0010, 0b0000_0000]
}

fn make_minimal_histograms(num_hf_presets: u32, nb_block_ctx: u32) -> HfCoefficientHistograms {
    let bytes = minimal_prefix_prelude_bytes();
    let mut br = BitReader::new(&bytes);
    let size = HfCoefficientHistogramSize::new(num_hf_presets, nb_block_ctx).unwrap();
    HfCoefficientHistograms::read(&mut br, size).unwrap()
}

fn single_pass_headers() -> PerPassHfHeaders {
    PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
        hfp: 0,
        histogram_offset: 0,
    }])
}

#[test]
fn r255_integration_dct8x8_short_circuits_with_zero_non_zeros() {
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let bits_before = br.bits_read();
    let (decoded, raw) = ctx
        .decode_block_for_pass_transform(&mut br, 0, TransformType::Dct8x8, 0, 0, 15)
        .unwrap();
    // DCT8x8: size = 64, num_blocks = 1.
    assert_eq!(decoded.coeffs.len(), 64);
    assert_eq!(raw, 0);
    assert_eq!(decoded.remaining_non_zeros, 0);
    assert_eq!(decoded.coeffs_read, 0);
    assert!(decoded.coeffs.iter().all(|&c| c == 0));
    // Zero-bit-consuming prefix → BitReader cursor unchanged.
    assert_eq!(br.bits_read(), bits_before);
}

#[test]
fn r255_integration_dct16x16_yields_256_coeff_vec_short_circuit() {
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let (decoded, raw) = ctx
        .decode_block_for_pass_transform(
            &mut br,
            0,
            TransformType::Dct16x16,
            32, // predicted_non_zeros at (0, 0)
            0,
            15,
        )
        .unwrap();
    // DCT16x16: size = 256, num_blocks = 4.
    assert_eq!(decoded.coeffs.len(), 256);
    assert_eq!(raw, 0);
    assert_eq!(decoded.coeffs_read, 0);
}

#[test]
fn r255_integration_dct16x8_yields_128_coeff_vec_short_circuit() {
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let (decoded, raw) = ctx
        .decode_block_for_pass_transform(&mut br, 0, TransformType::Dct16x8, 0, 0, 15)
        .unwrap();
    // DCT16x8: size = 128, num_blocks = 2.
    assert_eq!(decoded.coeffs.len(), 128);
    assert_eq!(raw, 0);
}

#[test]
fn r255_integration_dct8x16_yields_128_coeff_vec_short_circuit() {
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let (decoded, raw) = ctx
        .decode_block_for_pass_transform(&mut br, 0, TransformType::Dct8x16, 0, 0, 15)
        .unwrap();
    // DCT8x16: size = 128, num_blocks = 2.
    assert_eq!(decoded.coeffs.len(), 128);
    assert_eq!(raw, 0);
}

#[test]
fn r255_integration_dct4x4_yields_64_coeff_vec_short_circuit() {
    // DCT4x4 is one of the 8x8-output transforms (block_dims = (1,1)),
    // so size = 64, num_blocks = 1 — same shape as DCT8x8. Run the
    // short-circuit check to confirm the dispatch wires through.
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let (decoded, raw) = ctx
        .decode_block_for_pass_transform(&mut br, 0, TransformType::Dct4x4, 0, 0, 15)
        .unwrap();
    assert_eq!(decoded.coeffs.len(), 64);
    assert_eq!(raw, 0);
}

#[test]
fn r255_integration_per_pass_offset_routes_through_cluster_map() {
    // num_hf_presets = 2, nb_block_ctx = 1; per-pass offset = 495 × hfp.
    // The single-cluster cluster_map sends every index through →
    // cluster 0, which has the single symbol 0.
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
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    // Pass 0 — offset 0.
    let (d0, raw0) = ctx
        .decode_block_for_pass_transform(&mut br, 0, TransformType::Dct8x8, 0, 0, 1)
        .unwrap();
    assert_eq!(raw0, 0);
    assert_eq!(d0.coeffs.len(), 64);
    // Pass 1 — offset 495.
    let (d1, raw1) = ctx
        .decode_block_for_pass_transform(&mut br, 1, TransformType::Dct8x8, 0, 0, 1)
        .unwrap();
    assert_eq!(raw1, 0);
    assert_eq!(d1.coeffs.len(), 64);
}

#[test]
fn r255_integration_rejects_out_of_range_pass() {
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let r = ctx.decode_block_for_pass_transform(&mut br, 7, TransformType::Dct8x8, 0, 0, 15);
    assert!(r.is_err());
}

#[test]
fn r255_integration_rejects_u32_overflow_offset() {
    let mut h = make_minimal_histograms(1, 1);
    let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
        hfp: 0,
        histogram_offset: u64::from(u32::MAX) + 100,
    }]);
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let r = ctx.decode_block_for_pass_transform(&mut br, 0, TransformType::Dct8x8, 0, 0, 1);
    assert!(r.is_err());
}

#[test]
fn r255_integration_does_not_advance_br_on_short_circuit() {
    // Single-symbol prefix → every D[...] read consumes 0 bits, and
    // the C.14 loop short-circuits on non_zeros == 0. The BitReader
    // cursor must therefore stay where it started.
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let bytes = [0xFFu8; 16];
    let mut br = BitReader::new(&bytes);
    let bits_before = br.bits_read();
    let _ = ctx
        .decode_block_for_pass_transform(&mut br, 0, TransformType::Dct8x8, 0, 0, 15)
        .unwrap();
    assert_eq!(br.bits_read(), bits_before);
}

#[test]
fn r255_integration_round_trip_with_per_pass_hf_headers_read() {
    // Drive PerPassHfHeaders::read against a real bitstream, then
    // hand the resulting headers to HfHistogramDecodeContext + invoke
    // the round-255 bundled method to confirm the per-pass offset
    // derivation end-to-end equals 495 × nb_block_ctx × hfp.
    let mut h = make_minimal_histograms(2, 15);
    // num_hf_presets = 2 → nbits = 1; pass 0 hfp = 0, pass 1 hfp = 1.
    // Bit layout: 0 | 1 = 0b10 = byte 0x02.
    let header_bytes = [0b0000_0010u8];
    let mut hbr = BitReader::new(&header_bytes);
    let headers = PerPassHfHeaders::read(&mut hbr, 2, 2, 15).unwrap();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    assert_eq!(ctx.num_passes(), 2);
    assert_eq!(ctx.histogram_offset(0).unwrap(), 0);
    assert_eq!(ctx.histogram_offset(1).unwrap(), 7425); // 495 × 15 × 1
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let (_, raw0) = ctx
        .decode_block_for_pass_transform(&mut br, 0, TransformType::Dct8x8, 0, 0, 15)
        .unwrap();
    let (_, raw1) = ctx
        .decode_block_for_pass_transform(&mut br, 1, TransformType::Dct8x8, 0, 0, 15)
        .unwrap();
    assert_eq!(raw0, 0);
    assert_eq!(raw1, 0);
}
