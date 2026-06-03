//! Round 214 integration tests — per-LfGroup `BlockContext()`
//! resolver (ISO/IEC FDIS 18181-1:2021 §C.8.3 Listing C.13 +
//! §I.2.2 LfGlobal `HfBlockContext` bundle).
//!
//! These exercise the
//! [`block_context_resolver::BlockContextResolver`] +
//! [`block_context_resolver::decode_varblocks_with_resolver`] surface
//! end-to-end against the round-13 [`dct_select::DctSelectGrid`], the
//! round-190 [`per_pass_non_zeros::PerPassNonZerosGrids`], and the
//! round-208 [`varblock_walk::Varblock`] descriptor.
//!
//! Pure-control-flow primitive: no bit reads, no histogram
//! materialisation. The closures abstract over the §C.7.2 entropy
//! decode (#799 DOCS-GAP).

use oxideav_jpegxl::block_context_resolver::{
    decode_varblocks_with_resolver, BlockContextResolver,
};
use oxideav_jpegxl::dct_select::{derive_dct_select, TransformType};
use oxideav_jpegxl::lf_global::HfBlockContext;
use oxideav_jpegxl::lf_group::HfMetadata;
use oxideav_jpegxl::per_pass_non_zeros::PerPassNonZerosGrids;
use oxideav_jpegxl::varblock_walk::Varblock;

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

#[test]
fn r214_resolver_default_dct8x8_c0_matches_listing_c13() {
    // c=0, s=0 → idx = 1 × 13 + 0 = 13 → default map[13] = 7.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let vb = Varblock {
        x: 0,
        y: 0,
        transform: TransformType::Dct8x8,
        hf_mul: 1,
    };
    assert_eq!(resolver.resolve(0, &vb, [0, 0, 0]).unwrap(), 7);
}

#[test]
fn r214_resolver_default_dct8x8_c1_matches_listing_c13() {
    // c=1, s=0 → idx = (1^1) × 13 + 0 = 0 → default map[0] = 0.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let vb = Varblock {
        x: 0,
        y: 0,
        transform: TransformType::Dct8x8,
        hf_mul: 1,
    };
    assert_eq!(resolver.resolve(1, &vb, [0, 0, 0]).unwrap(), 0);
}

#[test]
fn r214_resolver_default_dct8x8_c2_matches_listing_c13() {
    // c=2 → c_term = 2; idx = 26 → default map[26] = 7.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let vb = Varblock {
        x: 0,
        y: 0,
        transform: TransformType::Dct8x8,
        hf_mul: 1,
    };
    assert_eq!(resolver.resolve(2, &vb, [0, 0, 0]).unwrap(), 7);
}

#[test]
fn r214_resolver_nb_block_ctx_default_is_15() {
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    assert_eq!(resolver.nb_block_ctx(), 15);
}

#[test]
fn r214_resolver_default_dct16x16_s_is_order_id_2() {
    // OrderId for Dct16x16 is Id2 → s = 2.
    // c=0 → idx = 13 + 2 = 15 → map[15] = 9.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let vb = Varblock {
        x: 0,
        y: 0,
        transform: TransformType::Dct16x16,
        hf_mul: 1,
    };
    assert_eq!(resolver.resolve(0, &vb, [0, 0, 0]).unwrap(), 9);
}

#[test]
fn r214_resolver_default_dct32x32_s_is_order_id_3() {
    // OrderId for Dct32x32 is Id3 → s = 3.
    // c=0 → idx = 13 + 3 = 16 → map[16] = 9.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let vb = Varblock {
        x: 0,
        y: 0,
        transform: TransformType::Dct32x32,
        hf_mul: 1,
    };
    assert_eq!(resolver.resolve(0, &vb, [0, 0, 0]).unwrap(), 9);
}

#[test]
fn r214_resolver_default_dct16x8_and_dct8x16_share_order_id_4() {
    // Both shapes → OrderId::Id4 → s = 4.
    // c=0 → idx = 13 + 4 = 17 → map[17] = 10.
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let vb_a = Varblock {
        x: 0,
        y: 0,
        transform: TransformType::Dct16x8,
        hf_mul: 1,
    };
    let vb_b = Varblock {
        x: 0,
        y: 0,
        transform: TransformType::Dct8x16,
        hf_mul: 1,
    };
    let a = resolver.resolve(0, &vb_a, [0, 0, 0]).unwrap();
    let b = resolver.resolve(0, &vb_b, [0, 0, 0]).unwrap();
    assert_eq!(a, b);
    assert_eq!(a, 10);
}

#[test]
fn r214_driver_routes_through_walker_single_dct8x8() {
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![0, 0], 1, 1);
    let grid = derive_dct_select(&hf, 8, 8).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 1, 1).unwrap();
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
    assert_eq!(out[0].0.transform, TransformType::Dct8x8);
}

#[test]
fn r214_driver_2x2_dct8x8_yields_four_varblocks_in_raster_order() {
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let hf = make_hf(vec![0, 0, 0, 0, 0, 0, 0, 0], 4, 4);
    let grid = derive_dct_select(&hf, 16, 16).unwrap();
    let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 2, 2).unwrap();
    let out = decode_varblocks_with_resolver(
        &grid,
        &mut nz,
        0,
        1,
        &resolver,
        |_| Ok([0, 0, 0]),
        |_| Ok(0),
        |_| Ok(0),
    )
    .unwrap();
    assert_eq!(out.len(), 4);
    let xs: Vec<(u32, u32)> = out.iter().map(|(v, _, _)| (v.x, v.y)).collect();
    assert_eq!(xs, vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
}

#[test]
fn r214_driver_qdc_closure_called_once_per_varblock_in_order() {
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
        0,
        &resolver,
        |vb| {
            seen.push((vb.x, vb.y));
            Ok([0, 0, 0])
        },
        |_| Ok(0),
        |_| Ok(0),
    )
    .unwrap();
    assert_eq!(seen.len(), 4);
    assert_eq!(seen, vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
}

#[test]
fn r214_driver_propagates_qdc_closure_error() {
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
        |_| Err(oxideav_core::Error::InvalidData("qdc-fail".into())),
        |_| Ok(0),
        |_| Ok(0),
    );
    assert!(r.is_err());
}

#[test]
fn r214_resolver_default_branch_qdc_is_irrelevant() {
    // Default branch has empty lf_thresholds, so any qdc value
    // produces the same ctx as qdc = [0; 3].
    let hbc = default_hbc();
    let resolver = BlockContextResolver::new(&hbc);
    let vb = Varblock {
        x: 0,
        y: 0,
        transform: TransformType::Dct8x8,
        hf_mul: 1,
    };
    let a = resolver.resolve(0, &vb, [0, 0, 0]).unwrap();
    let b = resolver.resolve(0, &vb, [i32::MIN, i32::MAX, 0]).unwrap();
    assert_eq!(a, b);
}
