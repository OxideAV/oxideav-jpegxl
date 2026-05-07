//! Round-3 bit-alignment bisect against the gray-64x64 trace.
//!
//! Walks the GlobalModular section step by step on the same fixture
//! covered by `cjxl_gray_64x64_pipeline_walkthrough` and prints the
//! exact bit position before and after each spec step. The trace
//! committed at `docs/image/jpegxl/fixtures/gray-64x64/trace.txt`
//! gives us authoritative bit budgets:
//!
//! * `HEADER total_bits=35` — ImageMetadata
//! * `FRAME_HEADER bits=27`
//! * `TOC bits=21`
//! * `ENTROPY` (tree, 6 ctx, prefix, log_alpha=15) bits=25
//! * `MODULAR_TREE nodes=3 leaves=2`
//! * `ENTROPY` (symbol, 2 ctx, ANS, log_alpha=5) bits=65
//! * `DC_GLOBAL_END bits_consumed=188` (so DC_GLOBAL section spans
//!   bit 88..276)
//!
//! Verification target: stop at each spec milestone and confirm
//! `br.bits_read()` matches the bit budget the trace would imply.
//! The first divergence is the bug.

use oxideav_jpegxl::ans::cluster::read_clustering;
use oxideav_jpegxl::bitreader::{BitReader, U32Dist};
use oxideav_jpegxl::container;
use oxideav_jpegxl::frame_header::{FrameDecodeParams, FrameHeader};
use oxideav_jpegxl::lf_global::LfChannelDequantization;
use oxideav_jpegxl::metadata_fdis::{ImageMetadataFdis, SizeHeaderFdis};
use oxideav_jpegxl::modular_fdis::{EntropyStream, MaNode};
use oxideav_jpegxl::toc::Toc;

const FIXTURE: &[u8] = include_bytes!("fixtures/gray_64x64_lossless.jxl");

fn run_pipeline(label: &str, fixture: &[u8]) {
    eprintln!("\n========= {label} =========");
    eprintln!("=== fixture {} bytes ===", fixture.len());
    let sig = container::detect(fixture).unwrap();
    let codestream: Vec<u8> = match sig {
        container::Signature::RawCodestream => fixture[2..].to_vec(),
        container::Signature::Isobmff => container::extract_codestream(fixture).unwrap().to_vec(),
    };
    let mut br = BitReader::new(&codestream);

    let size = SizeHeaderFdis::read(&mut br).unwrap();
    eprintln!("after SizeHeader: bits={}", br.bits_read());

    let metadata = ImageMetadataFdis::read(&mut br).unwrap();
    eprintln!("after ImageMetadata: bits={}", br.bits_read());

    br.pu0().unwrap();
    eprintln!("after byte-align (pu0): bits={}", br.bits_read());

    let fh_params = FrameDecodeParams {
        xyb_encoded: metadata.xyb_encoded,
        num_extra_channels: metadata.num_extra_channels,
        have_animation: metadata.have_animation,
        have_animation_timecodes: metadata
            .animation
            .map(|a| a.have_timecodes)
            .unwrap_or(false),
        image_width: size.width,
        image_height: size.height,
    };
    let fh = FrameHeader::read(&mut br, &fh_params).unwrap();
    eprintln!("after FrameHeader: bits={}", br.bits_read());

    let _toc = Toc::read(&mut br, &fh).unwrap();
    let dc_global_begin = br.bits_read();
    eprintln!("after TOC = DC_GLOBAL_BEGIN: bits={}", dc_global_begin);

    // ---- DC_GLOBAL section ----

    let lf_dequant = LfChannelDequantization::read(&mut br).unwrap();
    eprintln!(
        "after LfChannelDequantization (all_default={}): bits={} (rel {})",
        lf_dequant.all_default,
        br.bits_read(),
        br.bits_read() - dc_global_begin
    );

    let global_use_tree = br.read_bit().unwrap();
    eprintln!(
        "  global_use_tree = {} (bits={} rel {})",
        global_use_tree,
        br.bits_read(),
        br.bits_read() - dc_global_begin
    );

    if global_use_tree != 1 {
        eprintln!("  no global tree → skip");
        return;
    }

    // Tree-stream prelude
    let tree_prelude_begin = br.bits_read();
    let tree_stream = match EntropyStream::read(&mut br, 6) {
        Ok(s) => {
            eprintln!(
                "  tree-stream prelude OK: +{} bits → {} (rel {}) [use_prefix={} log_alpha={} clusters={}]",
                br.bits_read() - tree_prelude_begin,
                br.bits_read(),
                br.bits_read() - dc_global_begin,
                s.use_prefix_code,
                s.log_alphabet_size,
                s.entropies.len(),
            );
            s
        }
        Err(e) => {
            eprintln!("  tree-stream prelude FAIL at bits={}: {e}", br.bits_read());
            return;
        }
    };

    // MA tree decode — manually walk node-by-node printing bit positions.
    let tree_decode_begin = br.bits_read();
    eprintln!("--- MA tree decode begin: bits={tree_decode_begin} ---");
    let mut nodes: Vec<MaNode> = Vec::new();
    let mut nodes_left: u32 = 1;
    let mut ctx_id: u32 = 0;

    use oxideav_jpegxl::ans::hybrid::HybridUintState;
    let mut tree_hybrid = HybridUintState::new(tree_stream.lz77, tree_stream.lz_len_conf);
    let mut tree_stream = tree_stream;

    while nodes_left > 0 {
        let node_begin = br.bits_read();
        let property_plus_1 =
            match decode_uint_in_walk(&mut tree_hybrid, &mut tree_stream, &mut br, 1) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!(
                        "  node[{}] property decode FAIL at bits={} (rel {}): {e}",
                        nodes.len(),
                        br.bits_read(),
                        br.bits_read() - node_begin
                    );
                    return;
                }
            };
        let property = property_plus_1 as i64 - 1;

        if property < 0 {
            // Leaf
            let predictor =
                decode_uint_in_walk(&mut tree_hybrid, &mut tree_stream, &mut br, 2).unwrap();
            let uoffset =
                decode_uint_in_walk(&mut tree_hybrid, &mut tree_stream, &mut br, 3).unwrap();
            let mul_log =
                decode_uint_in_walk(&mut tree_hybrid, &mut tree_stream, &mut br, 4).unwrap();
            let mul_bits =
                decode_uint_in_walk(&mut tree_hybrid, &mut tree_stream, &mut br, 5).unwrap();
            eprintln!(
                "  node[{}] LEAF ctx={ctx_id} predictor={predictor} uoffset={uoffset} mul_log={mul_log} mul_bits={mul_bits} (+{} bits → {})",
                nodes.len(),
                br.bits_read() - node_begin,
                br.bits_read()
            );
            nodes.push(MaNode::Leaf(oxideav_jpegxl::modular_fdis::MaLeaf {
                ctx: ctx_id,
                predictor,
                offset: oxideav_jpegxl::bitreader::unpack_signed(uoffset),
                multiplier: (mul_bits + 1) << mul_log,
            }));
            ctx_id += 1;
            nodes_left -= 1;
        } else {
            let uvalue =
                decode_uint_in_walk(&mut tree_hybrid, &mut tree_stream, &mut br, 0).unwrap();
            let value = oxideav_jpegxl::bitreader::unpack_signed(uvalue);
            let nodes_now = nodes.len() as u32;
            let left_child = nodes_now + nodes_left;
            let right_child = nodes_now + nodes_left + 1;
            eprintln!(
                "  node[{}] DECISION property={property} value={value} left={left_child} right={right_child} (+{} bits → {})",
                nodes.len(),
                br.bits_read() - node_begin,
                br.bits_read()
            );
            nodes.push(MaNode::Decision {
                property: property as u32,
                value,
                left_child,
                right_child,
            });
            nodes_left += 2;
            nodes_left -= 1;
        }
    }
    let tree_decode_end = br.bits_read();
    eprintln!(
        "--- MA tree decode DONE: nodes={} ctx_id={ctx_id} (consumed {} bits, total at {} rel {}) ---",
        nodes.len(),
        tree_decode_end - tree_decode_begin,
        tree_decode_end,
        tree_decode_end - dc_global_begin,
    );

    // Symbol stream prelude — TRACE: bits=65 (after MA tree).
    let sym_begin = br.bits_read();
    let num_ctx = nodes.len().div_ceil(2);
    eprintln!(
        "--- symbol stream prelude begin at bit {sym_begin} (rel {}) num_ctx={num_ctx} ---",
        sym_begin - dc_global_begin
    );
    match EntropyStream::read(&mut br, num_ctx) {
        Ok(s) => eprintln!(
            "  symbol-stream prelude OK: +{} bits → {} (rel {}) [use_prefix={} log_alpha={} clusters={}]",
            br.bits_read() - sym_begin,
            br.bits_read(),
            br.bits_read() - dc_global_begin,
            s.use_prefix_code,
            s.log_alphabet_size,
            s.entropies.len(),
        ),
        Err(e) => eprintln!(
            "  symbol-stream prelude FAIL at bits={} (rel {}): {e}",
            br.bits_read(),
            br.bits_read() - sym_begin
        ),
    }
}

fn decode_uint_in_walk(
    hybrid: &mut oxideav_jpegxl::ans::hybrid::HybridUintState,
    entropy: &mut EntropyStream,
    br: &mut BitReader<'_>,
    ctx: u32,
) -> oxideav_core::Result<u32> {
    use oxideav_jpegxl::ans::hybrid_config::HybridUintConfig;
    let cluster_map_clone = entropy.cluster_map.clone();
    let configs_clone = entropy.configs.clone();
    let cfg_for = |c: u32| -> HybridUintConfig {
        let cl = cluster_map_clone.get(c as usize).copied().unwrap_or(0) as usize;
        configs_clone[cl.min(configs_clone.len().saturating_sub(1))]
    };
    hybrid.decode(
        br,
        ctx,
        ctx,
        0,
        |br_inner, c| entropy.decode_symbol(br_inner, c),
        cfg_for,
    )
}

#[test]
fn r3_bisect_gray_64x64_dc_global() {
    run_pipeline("gray_64x64_lossless", FIXTURE);
}

const GRADIENT: &[u8] = include_bytes!("fixtures/gradient_64x64_lossless.jxl");
const PALETTE: &[u8] = include_bytes!("fixtures/palette_32x32.jxl");
const GREY_8X8: &[u8] = include_bytes!("fixtures/grey_8x8_lossless.jxl");

#[test]
fn r3_bisect_gradient_64x64() {
    run_pipeline("gradient_64x64_lossless", GRADIENT);
}

#[test]
fn r3_bisect_palette_32x32() {
    run_pipeline("palette_32x32", PALETTE);
}

#[test]
fn r3_bisect_grey_8x8() {
    run_pipeline("grey_8x8_lossless", GREY_8X8);
}

/// Production path: call MaTreeFdis::read directly, then continue
/// with ModularHeader reads, printing each step's bit position.
fn run_production_path(label: &str, fixture: &[u8]) {
    use oxideav_jpegxl::modular_fdis::{MaTreeFdis, TransformInfo, WpHeader};
    eprintln!("\n========= production path: {label} =========");
    let sig = container::detect(fixture).unwrap();
    let codestream: Vec<u8> = match sig {
        container::Signature::RawCodestream => fixture[2..].to_vec(),
        container::Signature::Isobmff => container::extract_codestream(fixture).unwrap().to_vec(),
    };
    let mut br = BitReader::new(&codestream);

    let size = SizeHeaderFdis::read(&mut br).unwrap();
    let metadata = ImageMetadataFdis::read(&mut br).unwrap();
    br.pu0().unwrap();
    let fh_params = FrameDecodeParams {
        xyb_encoded: metadata.xyb_encoded,
        num_extra_channels: metadata.num_extra_channels,
        have_animation: metadata.have_animation,
        have_animation_timecodes: metadata
            .animation
            .map(|a| a.have_timecodes)
            .unwrap_or(false),
        image_width: size.width,
        image_height: size.height,
    };
    let fh = FrameHeader::read(&mut br, &fh_params).unwrap();
    let _toc = Toc::read(&mut br, &fh).unwrap();
    let dc_global_begin = br.bits_read();
    eprintln!("DC_GLOBAL_BEGIN at bit {}", dc_global_begin);

    let lf_dequant = LfChannelDequantization::read(&mut br).unwrap();
    eprintln!(
        "after LfChannelDequantization (all_default={}): bits={} (rel {})",
        lf_dequant.all_default,
        br.bits_read(),
        br.bits_read() - dc_global_begin
    );

    let global_use_tree = br.read_bool().unwrap();
    eprintln!(
        "global_use_tree={}: bits={} (rel {})",
        global_use_tree,
        br.bits_read(),
        br.bits_read() - dc_global_begin
    );

    let _tree = if global_use_tree {
        match MaTreeFdis::read(&mut br) {
            Ok(t) => {
                eprintln!(
                    "MaTreeFdis::read OK: nodes={} num_ctx={} (bits={} rel {})",
                    t.nodes.len(),
                    t.num_ctx,
                    br.bits_read(),
                    br.bits_read() - dc_global_begin
                );
                Some(t)
            }
            Err(e) => {
                eprintln!(
                    "MaTreeFdis::read FAIL at bits={} (rel {}): {e}",
                    br.bits_read(),
                    br.bits_read() - dc_global_begin
                );
                return;
            }
        }
    } else {
        None
    };

    // Now read the inner Modular sub-bitstream's ModularHeader.
    let inner_use_global_tree = match br.read_bool() {
        Ok(v) => {
            eprintln!(
                "inner_use_global_tree={v}: bits={} (rel {})",
                br.bits_read(),
                br.bits_read() - dc_global_begin
            );
            v
        }
        Err(e) => {
            eprintln!("inner_use_global_tree FAIL at bits={}: {e}", br.bits_read());
            return;
        }
    };

    let wp_header = match WpHeader::read(&mut br) {
        Ok(h) => {
            eprintln!(
                "WpHeader (default_wp={}): bits={} (rel {})",
                h.default_wp,
                br.bits_read(),
                br.bits_read() - dc_global_begin
            );
            h
        }
        Err(e) => {
            eprintln!("WpHeader FAIL at bits={}: {e}", br.bits_read());
            return;
        }
    };
    let _ = wp_header;
    let _ = inner_use_global_tree;

    let nb_transforms = match br.read_u32([
        U32Dist::Val(0),
        U32Dist::Val(1),
        U32Dist::BitsOffset(4, 2),
        U32Dist::BitsOffset(8, 18),
    ]) {
        Ok(v) => {
            eprintln!(
                "nb_transforms={}: bits={} (rel {})",
                v,
                br.bits_read(),
                br.bits_read() - dc_global_begin
            );
            v
        }
        Err(e) => {
            eprintln!("nb_transforms FAIL at bits={}: {e}", br.bits_read());
            return;
        }
    };

    for k in 0..nb_transforms {
        match TransformInfo::read(&mut br) {
            Ok(t) => eprintln!(
                "transform[{k}] OK: tr={:?} begin_c={:?} num_c={:?} nb_colours={:?} bits={} (rel {})",
                t.tr,
                t.begin_c,
                t.num_c,
                t.nb_colours,
                br.bits_read(),
                br.bits_read() - dc_global_begin
            ),
            Err(e) => {
                eprintln!(
                    "transform[{k}] FAIL at bits={} (rel {}): {e}",
                    br.bits_read(),
                    br.bits_read() - dc_global_begin
                );
                return;
            }
        }
    }
}

#[test]
fn r3_production_gray_64x64() {
    run_production_path("gray-64x64", FIXTURE);
}
#[test]
fn r3_production_gradient() {
    run_production_path("gradient-64x64", GRADIENT);
}
#[test]
fn r3_production_palette() {
    run_production_path("palette-32x32", PALETTE);
}
#[test]
fn r3_production_grey_8x8() {
    run_production_path("grey_8x8", GREY_8X8);
}

// helper
#[allow(dead_code)]
fn _read_clustering_helper(br: &mut BitReader<'_>, n: usize) {
    let _ = read_clustering(br, n);
}

// pull in U32Dist to shut up unused warning
#[allow(dead_code)]
const _U32_KEEP: U32Dist = U32Dist::Bits(2);
