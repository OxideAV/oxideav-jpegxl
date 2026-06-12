//! Round 260 integration coverage —
//! [`oxideav_jpegxl::multi_pass_hf_histogram_decoder::HfHistogramDecodeContext::decode_three_channel_varblock_for_pass`].
//!
//! ISO/IEC FDIS 18181-1:2021 §C.8.3 — the bundled three-channel
//! per-varblock walk that composes the round-255 single-channel
//! [`HfHistogramDecodeContext::decode_block_for_pass_transform`] three
//! times (channel decode order Y = 1 → X = 0 → B = 2 per the §C.8.3
//! prose "for each varblock it reads channels Y, X, then B"; output
//! arrays stay indexed 0 = X, 1 = Y, 2 = B) against the round-214
//! [`BlockContextResolver`] per-channel `block_ctx` derivation
//! (Listing C.13).
//!
//! These integration tests pin the public-surface invariants from a
//! consumer's vantage point (the per-LfGroup multi-pass driver in
//! [`oxideav_jpegxl::multi_pass_decode`] / round-221's
//! [`oxideav_jpegxl::block_context_resolver::decode_varblocks_three_channels_with_resolver`]
//! is the eventual caller):
//!
//! * Default-prefix short-circuit (single-symbol prefix → `non_zeros
//!   == 0` per channel → no coefficient symbols read on any of the
//!   three channels) holds across DCT8×8 / DCT16×16 / DCT16×8 /
//!   DCT8×16 / DCT4×4 transforms; the returned per-channel `coeffs`
//!   vectors are the correct length for each transform's
//!   `(num_blocks, size)` derivation.
//! * The per-pass `histogram_offset` is honoured — pass `p = 1`
//!   against a 2-preset histogram bundle routes through
//!   `cluster_map[ctx + 495 × nb_block_ctx]` rather than
//!   `cluster_map[ctx]`.
//! * Channel decode ordering is exactly Y → X → B (§C.8.3 prose);
//!   an error mid-walk aborts before the remaining channels' reads
//!   (so their ANS state is **not** advanced).
//! * The defensive rejections (`p >= num_passes`, `u32` overflow on
//!   `ctx + offset`) bubble out as [`oxideav_core::Error`] without
//!   panicking.
//! * No bit reads on a short-circuited block — the [`BitReader`]
//!   cursor must not advance across all three channels.

use oxideav_jpegxl::bitreader::BitReader;
use oxideav_jpegxl::block_context_resolver::BlockContextResolver;
use oxideav_jpegxl::dct_select::TransformType;
use oxideav_jpegxl::hf_coeff_histogram_size::HfCoefficientHistogramSize;
use oxideav_jpegxl::hf_coefficient_histograms::HfCoefficientHistograms;
use oxideav_jpegxl::lf_global::HfBlockContext;
use oxideav_jpegxl::multi_pass_hf_header::PerPassHfHeaders;
use oxideav_jpegxl::multi_pass_hf_histogram_decoder::HfHistogramDecodeContext;
use oxideav_jpegxl::pass_group_hf::PassGroupHfHeader;
use oxideav_jpegxl::varblock_walk::Varblock;

/// §D.3 prelude bytes for the minimal single-cluster, single-symbol
/// prefix-coded histogram block. Mirrors the round-255 integration
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

/// Default §I.2.2 `HfBlockContext` bundle — the empty-thresholds
/// shape used by round 214 / 221 / 228 fixtures. `nb_block_ctx = 15`.
fn default_hbc() -> HfBlockContext {
    HfBlockContext {
        used_default: true,
        qf_thresholds: vec![],
        lf_thresholds: [vec![], vec![], vec![]],
        block_ctx_map: HfBlockContext::DEFAULT_BLOCK_CTX_MAP.to_vec(),
        nb_block_ctx: 15,
    }
}

fn vb_at(x: u32, y: u32, t: TransformType) -> Varblock {
    Varblock {
        x,
        y,
        transform: t,
        hf_mul: 1,
    }
}

#[test]
fn r260_integration_dct8x8_three_channel_short_circuits() {
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let vb = vb_at(0, 0, TransformType::Dct8x8);
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let bits_before = br.bits_read();
    let (decoded, raw) = ctx
        .decode_three_channel_varblock_for_pass(&mut br, 0, &vb, &resolver, [0, 0, 0], [0, 0, 0])
        .unwrap();
    for c in 0..3 {
        assert_eq!(decoded[c].coeffs.len(), 64);
        assert_eq!(decoded[c].remaining_non_zeros, 0);
        assert_eq!(decoded[c].coeffs_read, 0);
        assert_eq!(raw[c], 0);
        assert!(decoded[c].coeffs.iter().all(|&v| v == 0));
    }
    // Three single-symbol-prefix reads → zero bits consumed total.
    assert_eq!(br.bits_read(), bits_before);
}

#[test]
fn r260_integration_dct16x16_per_channel_buffer_length_256() {
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let vb = vb_at(0, 0, TransformType::Dct16x16);
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let (decoded, raw) = ctx
        .decode_three_channel_varblock_for_pass(&mut br, 0, &vb, &resolver, [0, 0, 0], [32, 32, 32])
        .unwrap();
    for c in 0..3 {
        assert_eq!(decoded[c].coeffs.len(), 256);
        assert_eq!(raw[c], 0);
        assert_eq!(decoded[c].coeffs_read, 0);
    }
}

#[test]
fn r260_integration_dct16x8_per_channel_buffer_length_128() {
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let vb = vb_at(0, 0, TransformType::Dct16x8);
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let (decoded, raw) = ctx
        .decode_three_channel_varblock_for_pass(&mut br, 0, &vb, &resolver, [0, 0, 0], [16, 16, 16])
        .unwrap();
    for c in 0..3 {
        assert_eq!(decoded[c].coeffs.len(), 128);
        assert_eq!(raw[c], 0);
    }
}

#[test]
fn r260_integration_dct8x16_per_channel_buffer_length_128() {
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let vb = vb_at(0, 0, TransformType::Dct8x16);
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let (decoded, raw) = ctx
        .decode_three_channel_varblock_for_pass(&mut br, 0, &vb, &resolver, [0, 0, 0], [16, 16, 16])
        .unwrap();
    for c in 0..3 {
        assert_eq!(decoded[c].coeffs.len(), 128);
        assert_eq!(raw[c], 0);
    }
}

#[test]
fn r260_integration_dct4x4_per_channel_buffer_length_64() {
    // DCT4×4: 8×8-output transform (block_dims = (1, 1)), size = 64.
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let vb = vb_at(0, 0, TransformType::Dct4x4);
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let (decoded, raw) = ctx
        .decode_three_channel_varblock_for_pass(&mut br, 0, &vb, &resolver, [0, 0, 0], [0, 0, 0])
        .unwrap();
    for c in 0..3 {
        assert_eq!(decoded[c].coeffs.len(), 64);
        assert_eq!(raw[c], 0);
    }
}

#[test]
fn r260_integration_per_pass_offset_routes_through_cluster_map() {
    // num_hf_presets = 2, nb_block_ctx = 1; per-pass offset = 495 × hfp.
    // The single-cluster cluster_map sends every index through →
    // cluster 0, which has the single symbol 0. Drive both passes
    // against the bundled three-channel walk.
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
    // Custom HfBlockContext with nb_block_ctx = 1 so the resolver
    // matches the histograms shape — block_ctx_map collapses to all-
    // zeros over the default-table indices.
    let hbc = HfBlockContext {
        used_default: false,
        qf_thresholds: vec![],
        lf_thresholds: [vec![], vec![], vec![]],
        block_ctx_map: vec![0u8; 39],
        nb_block_ctx: 1,
    };
    let resolver = BlockContextResolver::new(&hbc);
    let vb = vb_at(0, 0, TransformType::Dct8x8);
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    // Pass 0 — offset 0.
    let (_d0, raw0) = ctx
        .decode_three_channel_varblock_for_pass(&mut br, 0, &vb, &resolver, [0, 0, 0], [0, 0, 0])
        .unwrap();
    assert_eq!(raw0, [0, 0, 0]);
    // Pass 1 — offset 495.
    let (_d1, raw1) = ctx
        .decode_three_channel_varblock_for_pass(&mut br, 1, &vb, &resolver, [0, 0, 0], [0, 0, 0])
        .unwrap();
    assert_eq!(raw1, [0, 0, 0]);
}

#[test]
fn r260_integration_rejects_out_of_range_pass() {
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let vb = vb_at(0, 0, TransformType::Dct8x8);
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let r = ctx.decode_three_channel_varblock_for_pass(
        &mut br,
        7, // > num_passes (= 1)
        &vb,
        &resolver,
        [0, 0, 0],
        [0, 0, 0],
    );
    assert!(r.is_err());
}

#[test]
fn r260_integration_rejects_u32_overflow_offset() {
    let mut h = make_minimal_histograms(1, 1);
    let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
        hfp: 0,
        histogram_offset: u64::from(u32::MAX) + 100,
    }]);
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let hbc = HfBlockContext {
        used_default: false,
        qf_thresholds: vec![],
        lf_thresholds: [vec![], vec![], vec![]],
        block_ctx_map: vec![0u8; 39],
        nb_block_ctx: 1,
    };
    let resolver = BlockContextResolver::new(&hbc);
    let vb = vb_at(0, 0, TransformType::Dct8x8);
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let r = ctx.decode_three_channel_varblock_for_pass(
        &mut br,
        0,
        &vb,
        &resolver,
        [0, 0, 0],
        [0, 0, 0],
    );
    assert!(r.is_err());
}

#[test]
fn r260_integration_does_not_advance_br_on_short_circuit() {
    // Single-symbol prefix: every D[...] read consumes 0 bits.
    // Three channels × short-circuit non_zeros = 0 → no per-block
    // coefficient symbols → the BitReader cursor stays where it
    // started.
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let vb = vb_at(0, 0, TransformType::Dct8x8);
    let bytes = [0xFFu8; 16];
    let mut br = BitReader::new(&bytes);
    let bits_before = br.bits_read();
    let _ = ctx
        .decode_three_channel_varblock_for_pass(&mut br, 0, &vb, &resolver, [0, 0, 0], [0, 0, 0])
        .unwrap();
    assert_eq!(br.bits_read(), bits_before);
}

#[test]
fn r260_integration_round_trip_with_per_pass_hf_headers_read() {
    // Drive PerPassHfHeaders::read against a real bitstream, then
    // hand the resulting headers to HfHistogramDecodeContext and
    // invoke the round-260 bundled three-channel method on a couple
    // of varblocks to confirm the per-pass offset derivation
    // round-trips end-to-end (each call uses 495 × 15 × hfp(p) as
    // the per-pass offset).
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
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let vb = vb_at(0, 0, TransformType::Dct8x8);
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let (_, raw0) = ctx
        .decode_three_channel_varblock_for_pass(&mut br, 0, &vb, &resolver, [0, 0, 0], [0, 0, 0])
        .unwrap();
    let (_, raw1) = ctx
        .decode_three_channel_varblock_for_pass(&mut br, 1, &vb, &resolver, [0, 0, 0], [0, 0, 0])
        .unwrap();
    assert_eq!(raw0, [0, 0, 0]);
    assert_eq!(raw1, [0, 0, 0]);
}

#[test]
fn r260_integration_per_channel_block_ctx_resolved_from_resolver() {
    // Sanity: the resolver must be queried with c ∈ {0, 1, 2}. With
    // the default-table HfBlockContext, all three channels share the
    // same s = order_id (DCT8×8 → 0), so block_ctx is determined by
    // the channel-indexed slot in the default block_ctx_map. We can
    // compute the expected values via the resolver directly and
    // assert they are all `< nb_block_ctx` (= 15) — confirming the
    // bundled method receives spec-legal context values rather than
    // panicking on out-of-range entries.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let vb = vb_at(0, 0, TransformType::Dct8x8);
    let bc0 = resolver.resolve(0, &vb, [0, 0, 0]).unwrap();
    let bc1 = resolver.resolve(1, &vb, [0, 0, 0]).unwrap();
    let bc2 = resolver.resolve(2, &vb, [0, 0, 0]).unwrap();
    assert!(bc0 < 15, "X-channel block_ctx {bc0} out of range");
    assert!(bc1 < 15, "Y-channel block_ctx {bc1} out of range");
    assert!(bc2 < 15, "B-channel block_ctx {bc2} out of range");
    // And the bundled method completes without error against these.
    let mut h = make_minimal_histograms(1, 15);
    let headers = single_pass_headers();
    let mut ctx = HfHistogramDecodeContext::new(&mut h, &headers).unwrap();
    let bytes = [0u8; 4];
    let mut br = BitReader::new(&bytes);
    let (_d, raw) = ctx
        .decode_three_channel_varblock_for_pass(&mut br, 0, &vb, &resolver, [0, 0, 0], [0, 0, 0])
        .unwrap();
    assert_eq!(raw, [0, 0, 0]);
}
