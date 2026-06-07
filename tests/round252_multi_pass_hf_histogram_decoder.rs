//! Round 252 integration coverage —
//! [`oxideav_jpegxl::multi_pass_hf_histogram_decoder::HfHistogramDecodeContext`].
//!
//! ISO/IEC FDIS 18181-1:2021 §C.7.2 (entropy histograms) + §C.8.3
//! (per-pass `histogram_offset` routing) — the typed bridge that
//! wires the round-247 [`HfCoefficientHistograms`] entropy stream
//! to the round-232 [`PerPassHfHeaders`] per-pass `(hfp,
//! histogram_offset)` array, exposing the §C.8.3 driver-shape
//! `(p, c, ctx) -> symbol` decode surface.
//!
//! These integration tests pin the public-surface invariants:
//!
//! * `HfHistogramDecodeContext::new` validates per-pass `hfp <
//!   histograms.num_hf_presets()` against the histograms container
//!   (independent of the value [`PerPassHfHeaders::read`] was
//!   constructed against).
//! * `HfHistogramDecodeContext::decode_symbol_for_pass(p, ctx)`
//!   routes `D[ctx + histogram_offset(p)]` through the underlying
//!   §C.7.2 [`EntropyStream`].
//! * `HfHistogramDecodeContext::non_zeros_at` composes
//!   [`non_zeros_context`] + the per-pass offset routing.
//! * `HfHistogramDecodeContext::coefficient_at` composes
//!   [`coefficient_context`] + the per-pass offset routing, and
//!   propagates the `num_blocks == 0` rejection from
//!   `coefficient_context` without touching the [`BitReader`].
//! * Round-trip with [`PerPassHfHeaders::read`] driven off a real
//!   bitstream — the per-pass offsets cached at construction time
//!   match the values [`PerPassHfHeaders::histogram_offset`]
//!   independently exposes.

use oxideav_jpegxl::bitreader::BitReader;
use oxideav_jpegxl::hf_coeff_histogram_size::HfCoefficientHistogramSize;
use oxideav_jpegxl::hf_coefficient_histograms::HfCoefficientHistograms;
use oxideav_jpegxl::multi_pass_hf_header::PerPassHfHeaders;
use oxideav_jpegxl::multi_pass_hf_histogram_decoder::HfHistogramDecodeContext;
use oxideav_jpegxl::pass_group_hf::{coefficient_context, non_zeros_context, PassGroupHfHeader};

/// §D.3 prelude that reads as a single-cluster, prefix-coded
/// histogram block where every distribution maps to cluster 0 and
/// the per-cluster prefix code is a single-symbol code (symbol 0).
///
/// Layout (LSB-first):
///   bit 0       : lz77_enabled = 0
///   bit 1       : is_simple = 1            (D.3.5 simple clustering)
///   bits 2-3    : nbits = 0                (all distributions → cluster 0)
///   bit 4       : use_prefix_code = 1      (log_alphabet_size = 15 implicit)
///   bits 5-8    : split_exponent = 0       (HybridUintConfig::read u(4))
///   bit 9       : prefix count selector = 0 → count = 1 (single-symbol cluster)
///
/// Total = 10 bits. Byte 0 has bits 1 and 4 set → 0b0001_0010 = 0x12;
/// byte 1 is all-zero.
fn minimal_prefix_prelude_bytes() -> [u8; 2] {
    [0b0001_0010, 0b0000_0000]
}

fn make_minimal_histograms(num_hf_presets: u32, nb_block_ctx: u32) -> HfCoefficientHistograms {
    let bytes = minimal_prefix_prelude_bytes();
    let mut br = BitReader::new(&bytes);
    let size = HfCoefficientHistogramSize::new(num_hf_presets, nb_block_ctx).unwrap();
    HfCoefficientHistograms::read(&mut br, size).unwrap()
}

#[test]
fn r252_integration_new_rejects_zero_passes() {
    let mut h = make_minimal_histograms(1, 1);
    let headers = PerPassHfHeaders::from_headers(vec![]);
    let r = HfHistogramDecodeContext::new(&mut h, &headers);
    assert!(r.is_err());
}

#[test]
fn r252_integration_new_rejects_hfp_ge_num_hf_presets() {
    // histograms num_hf_presets = 2; header carries hfp = 5 → reject.
    let mut h = make_minimal_histograms(2, 1);
    let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
        hfp: 5,
        histogram_offset: 495 * 5,
    }]);
    let r = HfHistogramDecodeContext::new(&mut h, &headers);
    assert!(r.is_err());
}

#[test]
fn r252_integration_per_pass_offsets_cached_at_construction() {
    // num_hf_presets = 4, nb_block_ctx = 15 → offset = 7425 × hfp.
    let mut h = make_minimal_histograms(4, 15);
    let headers = PerPassHfHeaders::from_headers(vec![
        PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        },
        PassGroupHfHeader {
            hfp: 2,
            histogram_offset: 14_850,
        },
        PassGroupHfHeader {
            hfp: 3,
            histogram_offset: 22_275,
        },
    ]);
    let ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    assert_eq!(ctx.num_passes(), 3);
    assert_eq!(ctx.per_pass_offsets(), &[0u64, 14_850u64, 22_275u64]);
    assert_eq!(ctx.histogram_offset(0).unwrap(), 0);
    assert_eq!(ctx.histogram_offset(1).unwrap(), 14_850);
    assert_eq!(ctx.histogram_offset(2).unwrap(), 22_275);
    assert!(ctx.histogram_offset(3).is_err());
}

#[test]
fn r252_integration_decode_symbol_for_pass_zero_bits_single_symbol() {
    // Single-symbol prefix → every decode returns 0 and consumes 0 bits.
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
    let bits_before = br.bits_read();
    for p in 0..2u32 {
        for ctx_val in [0u32, 17u32, 100u32] {
            let s = ctx.decode_symbol_for_pass(&mut br, p, ctx_val).unwrap();
            assert_eq!(s, 0, "(p={p}, ctx={ctx_val}) should decode to 0");
        }
    }
    assert_eq!(br.bits_read(), bits_before);
}

#[test]
fn r252_integration_decode_symbol_for_pass_out_of_range_pass() {
    let mut h = make_minimal_histograms(1, 1);
    let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
        hfp: 0,
        histogram_offset: 0,
    }]);
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let bytes = [0u8];
    let mut br = BitReader::new(&bytes);
    let r = ctx.decode_symbol_for_pass(&mut br, 1, 0);
    assert!(r.is_err());
}

#[test]
fn r252_integration_non_zeros_at_routes_through_offset() {
    let mut h = make_minimal_histograms(2, 15);
    let headers = PerPassHfHeaders::from_headers(vec![
        PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        },
        PassGroupHfHeader {
            hfp: 1,
            histogram_offset: 7425,
        },
    ]);
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    // Spec-precise: D[NonZerosContext(predicted=3, block_ctx=2, nb_block_ctx=15) + offset(p)]
    //             = D[(2 + 15 × 3) + 0]         (p=0)
    //             = D[47 + 0]
    //             = cluster_map[47] → cluster 0 → symbol 0.
    let s0 = ctx.non_zeros_at(&mut br, 0, 3, 2, 15).unwrap();
    assert_eq!(s0, 0);
    // For p=1, offset = 7425 — still within the cluster_map of length 14850.
    let s1 = ctx.non_zeros_at(&mut br, 1, 3, 2, 15).unwrap();
    assert_eq!(s1, 0);
    // Cross-check the standalone helper composes to the same context.
    assert_eq!(non_zeros_context(3, 2, 15), 47);
}

#[test]
fn r252_integration_coefficient_at_routes_through_offset() {
    let mut h = make_minimal_histograms(2, 15);
    let headers = PerPassHfHeaders::from_headers(vec![
        PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        },
        PassGroupHfHeader {
            hfp: 1,
            histogram_offset: 7425,
        },
    ]);
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    // §C.8.3 first non-zero block path for DCT8×8 (num_blocks = 1,
    // size = 64). The spec line is
    // ucoeff = D[CoefficientContext(k, non_zeros, num_blocks, size,
    //                              prev, block_ctx, nb_block_ctx)
    //            + offset].
    // We pick k = 1, non_zeros = 16, num_blocks = 1, size = 64,
    // prev = 0, block_ctx = 0, nb_block_ctx = 15 — same arguments
    // checked by the standalone-helper round-159 tests.
    let expected_ctx = coefficient_context(1, 16, 1, 64, 0, 0, 15).unwrap();
    let s = ctx
        .coefficient_at(&mut br, 0, 1, 16, 1, 64, 0, 0, 15)
        .unwrap();
    assert_eq!(s, 0);
    // For pass 1 the offset is 7425 — ctx + offset still < cluster_map
    // length (14850) for this small expected_ctx.
    let s1 = ctx
        .coefficient_at(&mut br, 1, 1, 16, 1, 64, 0, 0, 15)
        .unwrap();
    assert_eq!(s1, 0);
    let _ = expected_ctx;
}

#[test]
fn r252_integration_coefficient_at_propagates_num_blocks_zero() {
    let mut h = make_minimal_histograms(1, 15);
    let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
        hfp: 0,
        histogram_offset: 0,
    }]);
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let bits_before = br.bits_read();
    let r = ctx.coefficient_at(&mut br, 0, 1, 16, 0, 64, 0, 0, 15);
    assert!(r.is_err());
    assert_eq!(br.bits_read(), bits_before);
}

#[test]
fn r252_integration_round_trip_with_read_headers() {
    // Construct headers from a real bitstream (round-232
    // PerPassHfHeaders::read) and verify the round-252 cache matches
    // the round-232 derivation.
    let mut h = make_minimal_histograms(4, 15);
    // num_hf_presets = 4 → nbits = 2. Pass 0 hfp = 1, pass 1 hfp = 3.
    // Layout (LSB-first per pass): 01 | 11 = 0b1101 = 0x0D.
    let header_bytes = [0b0000_1101u8];
    let mut hbr = BitReader::new(&header_bytes);
    let headers = PerPassHfHeaders::read(&mut hbr, 2, 4, 15).unwrap();
    assert_eq!(headers.hfp(0).unwrap(), 1);
    assert_eq!(headers.hfp(1).unwrap(), 3);
    assert_eq!(headers.histogram_offset(0).unwrap(), 7425);
    assert_eq!(headers.histogram_offset(1).unwrap(), 22_275);

    let ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    assert_eq!(ctx.num_passes(), 2);
    assert_eq!(ctx.per_pass_offsets(), &[7425u64, 22_275u64]);
    assert_eq!(
        ctx.histogram_offset(0).unwrap(),
        headers.histogram_offset(0).unwrap()
    );
    assert_eq!(
        ctx.histogram_offset(1).unwrap(),
        headers.histogram_offset(1).unwrap()
    );
}
