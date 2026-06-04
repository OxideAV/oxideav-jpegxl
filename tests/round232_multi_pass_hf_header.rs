//! Round 232 integration coverage —
//! [`oxideav_jpegxl::multi_pass_hf_header::PerPassHfHeaders`] and
//! [`oxideav_jpegxl::multi_pass_hf_header::decode_multi_pass_with_hf_headers`].
//!
//! ISO/IEC FDIS 18181-1:2021 §C.8.3 first-paragraph
//! `hfp = u(ceil(log2(num_hf_presets)))` per-pass header read +
//! per-pass `histogram_offset = 495 × nb_block_ctx × hfp` routing.

use oxideav_jpegxl::bitreader::BitReader;
use oxideav_jpegxl::block_context_resolver::BlockContextResolver;
use oxideav_jpegxl::dct_select::{derive_dct_select, TransformType};
use oxideav_jpegxl::lf_global::HfBlockContext;
use oxideav_jpegxl::lf_group::HfMetadata;
use oxideav_jpegxl::multi_pass_hf_header::{
    decode_multi_pass_with_hf_headers, read_and_decode_multi_pass_with_hf_headers, PassHfDigest,
    PerPassHfHeaders,
};
use oxideav_jpegxl::pass_group_hf::PassGroupHfHeader;
use oxideav_jpegxl::per_pass_non_zeros::PerPassNonZerosGrids;

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
        qf_thresholds: vec![],
        lf_thresholds: [vec![], vec![], vec![]],
        block_ctx_map: vec![
            7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 9, 9, 10, 11, 12, 13, 14, 0, 0, 0, 0, 7,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
        nb_block_ctx: 15,
    }
}

#[test]
fn r232_integration_two_pass_header_sequence_offsets() {
    // num_passes = 2, num_hf_presets = 2 → nbits = 1.
    // Pass-0 hfp = 1, pass-1 hfp = 0 → bits 1, 0 → byte 0b01 = 0x01.
    let data = [0b0000_0001u8];
    let mut br = BitReader::new(&data);
    let headers = PerPassHfHeaders::read(&mut br, 2, 2, 15).unwrap();
    assert_eq!(headers.num_passes(), 2);
    assert_eq!(headers.hfp(0).unwrap(), 1);
    assert_eq!(headers.hfp(1).unwrap(), 0);
    assert_eq!(headers.histogram_offset(0).unwrap(), 7425);
    assert_eq!(headers.histogram_offset(1).unwrap(), 0);
    assert_eq!(br.bits_read(), 2);
}

#[test]
fn r232_integration_digest_round_trip_via_bit_read() {
    // num_passes = 4, num_hf_presets = 4 → nbits = 2.
    // hfps = (3, 1, 0, 2). JXL packs bits LSB-first: byte =
    // (3 << 0) | (1 << 2) | (0 << 4) | (2 << 6) = 0x87.
    let data = [0x87u8];
    let mut br = BitReader::new(&data);
    let headers = PerPassHfHeaders::read(&mut br, 4, 4, 15).unwrap();
    let digest = headers.digest();
    assert_eq!(digest.len(), 4);
    assert_eq!(
        digest[0],
        PassHfDigest {
            hfp: 3,
            histogram_offset: 22275
        }
    );
    assert_eq!(
        digest[1],
        PassHfDigest {
            hfp: 1,
            histogram_offset: 7425
        }
    );
    assert_eq!(
        digest[2],
        PassHfDigest {
            hfp: 0,
            histogram_offset: 0
        }
    );
    assert_eq!(
        digest[3],
        PassHfDigest {
            hfp: 2,
            histogram_offset: 14850
        }
    );
}

#[test]
fn r232_integration_inline_read_and_decode_single_dct8x8() {
    // num_passes = 1, num_hf_presets = 1 → 0 bits consumed for hfp.
    // Use a 1-byte buffer (unused) so the BitReader has room.
    let data = [0u8];
    let mut br = BitReader::new(&data);
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![0, 0], 1, 1);
    let grid = derive_dct_select(&hf, 8, 8).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 1, 1).unwrap();
    let (headers, out) = read_and_decode_multi_pass_with_hf_headers(
        &mut br,
        &grid,
        &mut nz,
        &resolver,
        1,
        15,
        |_p, _vb| Ok([0, 0, 0]),
        |_p, _c, _pred, _offset| Ok(0),
        |_p, _c, _coef, _offset| Ok(0),
    )
    .unwrap();
    assert_eq!(headers.num_passes(), 1);
    assert_eq!(headers.hfp(0).unwrap(), 0);
    assert_eq!(headers.histogram_offset(0).unwrap(), 0);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].len(), 1);
    assert_eq!(out[0][0].0.transform, TransformType::Dct8x8);
    // num_hf_presets = 1 → nbits = 0 → no bits consumed.
    assert_eq!(br.bits_read(), 0);
}

#[test]
fn r232_integration_inline_read_and_decode_4x4_grid_two_passes() {
    // 4×4 DCT8×8 grid, num_passes = 2, num_hf_presets = 2.
    // Pass-0 hfp = 0, pass-1 hfp = 1 → bits 0, 1 → 0b10 = 0x02.
    let data = [0b0000_0010u8];
    let mut br = BitReader::new(&data);
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let block_info = vec![0; 32];
    let hf = make_hf(block_info, 16, 16);
    let grid = derive_dct_select(&hf, 32, 32).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(2, 3, 4, 4).unwrap();
    let mut per_pass_offset_seen: Vec<(u32, u64)> = Vec::new();
    let (headers, out) = read_and_decode_multi_pass_with_hf_headers(
        &mut br,
        &grid,
        &mut nz,
        &resolver,
        2,
        15,
        |_p, _vb| Ok([0, 0, 0]),
        |p, _c, _pred, offset| {
            per_pass_offset_seen.push((p, offset));
            Ok(0)
        },
        |_p, _c, _coef, _offset| Ok(0),
    )
    .unwrap();
    assert_eq!(headers.hfp(0).unwrap(), 0);
    assert_eq!(headers.hfp(1).unwrap(), 1);
    // 2 passes × 16 varblocks each.
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].len(), 16);
    assert_eq!(out[1].len(), 16);
    // read_non_zeros: 2 passes × 16 varblocks × 3 channels = 96 calls.
    assert_eq!(per_pass_offset_seen.len(), 96);
    // Pass-0 invocations precede pass-1 invocations.
    let pass0_count = per_pass_offset_seen.iter().filter(|(p, _)| *p == 0).count();
    let pass1_count = per_pass_offset_seen.iter().filter(|(p, _)| *p == 1).count();
    assert_eq!(pass0_count, 48);
    assert_eq!(pass1_count, 48);
    // Per-pass offsets: pass-0 sees 0, pass-1 sees 7425.
    for &(p, off) in &per_pass_offset_seen {
        if p == 0 {
            assert_eq!(off, 0);
        } else {
            assert_eq!(off, 7425);
        }
    }
}

#[test]
fn r232_integration_three_pass_distinct_offsets_via_pre_built_headers() {
    // 3 passes, hfps = (0, 1, 0), nb_block_ctx = 15.
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
            hfp: 0,
            histogram_offset: 0,
        },
    ]);
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![0, 0], 1, 1);
    let grid = derive_dct_select(&hf, 8, 8).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(3, 3, 1, 1).unwrap();
    let mut per_pass_offsets: Vec<u64> = vec![u64::MAX; 3];
    let out = decode_multi_pass_with_hf_headers(
        &grid,
        &headers,
        &mut nz,
        &resolver,
        |_p, _vb| Ok([0, 0, 0]),
        |p, _c, _pred, offset| {
            per_pass_offsets[p as usize] = offset;
            Ok(0)
        },
        |_p, _c, _coef, _offset| Ok(0),
    )
    .unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(per_pass_offsets, vec![0, 7425, 0]);
}

#[test]
fn r232_integration_mixed_transform_offset_routing() {
    // Mixed transforms: DCT16×8 (covers (0,0)+(0,1)) + 2 DCT8×8.
    // 2 passes, distinct hfps. Verify per-pass offsets reach the
    // read_non_zeros closure once per (pass, channel) per varblock.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![6, 0, 0, 0, 0, 0], 3, 3);
    let grid = derive_dct_select(&hf, 16, 16).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(2, 3, 2, 2).unwrap();
    let headers = PerPassHfHeaders::from_headers(vec![
        PassGroupHfHeader {
            hfp: 1,
            histogram_offset: 7425,
        },
        PassGroupHfHeader {
            hfp: 3,
            histogram_offset: 22275,
        },
    ]);
    let mut offsets_per_pass: Vec<u64> = vec![u64::MAX; 2];
    let out = decode_multi_pass_with_hf_headers(
        &grid,
        &headers,
        &mut nz,
        &resolver,
        |_p, _vb| Ok([0, 0, 0]),
        |p, _c, _pred, offset| {
            offsets_per_pass[p as usize] = offset;
            Ok(0)
        },
        |_p, _c, _coef, _offset| Ok(0),
    )
    .unwrap();
    assert_eq!(out.len(), 2);
    for pass_out in &out {
        assert_eq!(pass_out.len(), 3);
        assert_eq!(pass_out[0].0.transform, TransformType::Dct16x8);
        assert_eq!(pass_out[1].0.transform, TransformType::Dct8x8);
        assert_eq!(pass_out[2].0.transform, TransformType::Dct8x8);
    }
    assert_eq!(offsets_per_pass, vec![7425, 22275]);
}

#[test]
fn r232_integration_per_pass_headers_get_out_of_range() {
    let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
        hfp: 0,
        histogram_offset: 0,
    }]);
    assert!(headers.get(0).is_ok());
    assert!(headers.get(1).is_err());
    assert!(headers.histogram_offset(1).is_err());
    assert!(headers.hfp(1).is_err());
}

#[test]
fn r232_integration_num_passes_mismatch_rejected() {
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![0, 0], 1, 1);
    let grid = derive_dct_select(&hf, 8, 8).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(2, 3, 1, 1).unwrap();
    let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
        hfp: 0,
        histogram_offset: 0,
    }]);
    let r = decode_multi_pass_with_hf_headers(
        &grid,
        &headers,
        &mut nz,
        &resolver,
        |_p, _vb| Ok([0, 0, 0]),
        |_p, _c, _pred, _offset| Ok(0),
        |_p, _c, _coef, _offset| Ok(0),
    );
    assert!(r.is_err());
}

#[test]
fn r232_integration_inline_read_consumes_correct_bit_count() {
    // num_hf_presets = 8 → nbits = 3. num_passes = 5 →
    // 5 × 3 = 15 bits consumed.
    let data = [0u8; 2];
    let mut br = BitReader::new(&data);
    let headers = PerPassHfHeaders::read(&mut br, 5, 8, 15).unwrap();
    assert_eq!(headers.num_passes(), 5);
    for p in 0..5 {
        assert_eq!(headers.hfp(p).unwrap(), 0);
    }
    assert_eq!(br.bits_read(), 15);
}

#[test]
fn r232_integration_offset_zero_at_hfp_zero_regardless_of_nb_block_ctx() {
    // hfp = 0 always yields offset = 0 since the formula is
    // multiplicative.
    let data = [0u8; 4];
    let mut br = BitReader::new(&data);
    let headers = PerPassHfHeaders::read(&mut br, 3, 2, 100).unwrap();
    for p in 0..3 {
        assert_eq!(headers.hfp(p).unwrap(), 0);
        assert_eq!(headers.histogram_offset(p).unwrap(), 0);
    }
}

#[test]
fn r232_integration_offset_scales_with_nb_block_ctx() {
    // Same hfp = 1 but nb_block_ctx = 100 → offset = 495 × 100 = 49500.
    let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
        hfp: 1,
        histogram_offset: 495 * 100,
    }]);
    assert_eq!(headers.histogram_offset(0).unwrap(), 49500);
}

#[test]
fn r232_integration_inline_read_bit_position_advances_through_walk() {
    // Verify the BitReader is left at the post-header position after
    // a successful inline read+decode (the per-LfGroup VarDCT
    // continuation is the caller's responsibility, but the per-pass
    // hfp bits must be exactly consumed).
    let data = [0b0000_0010u8, 0xFF];
    let mut br = BitReader::new(&data);
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![0, 0], 1, 1);
    let grid = derive_dct_select(&hf, 8, 8).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(2, 3, 1, 1).unwrap();
    let _ = read_and_decode_multi_pass_with_hf_headers(
        &mut br,
        &grid,
        &mut nz,
        &resolver,
        2,
        15,
        |_p, _vb| Ok([0, 0, 0]),
        |_p, _c, _pred, _offset| Ok(0),
        |_p, _c, _coef, _offset| Ok(0),
    )
    .unwrap();
    // 2 bits consumed (one per pass).
    assert_eq!(br.bits_read(), 2);
}
