//! Round-5 instrumented bisect: find the failing cluster prefix
//! decode in the grey_8x8 fixture.
//!
//! Round-4 left grey_8x8_lossless failing at bit 563 of the symbol-
//! stream prelude (after the MA tree finishes at bit 181). The
//! failure mode is Kraft-overflow inside `PrefixCode::from_lengths`,
//! triggered by one of the four per-cluster prefix histograms.
//! This test reproduces the prelude up to the point of failure and
//! prints, for each cluster, the bit position, the count, the
//! decoded clcl array, and the resulting per-symbol code-length array.

use oxideav_jpegxl::ans::cluster::{num_clusters, read_clustering};
use oxideav_jpegxl::ans::hybrid::{HybridUintState, Lz77Params};
use oxideav_jpegxl::ans::hybrid_config::HybridUintConfig;
use oxideav_jpegxl::ans::prefix::{diagnose_complex_prefix, read_prefix_code_traced, ClclTrace};
use oxideav_jpegxl::bitreader::{unpack_signed, BitReader, U32Dist};
use oxideav_jpegxl::container;
use oxideav_jpegxl::frame_header::{FrameDecodeParams, FrameHeader};
use oxideav_jpegxl::lf_global::LfChannelDequantization;
use oxideav_jpegxl::metadata_fdis::{ImageMetadataFdis, SizeHeaderFdis};
use oxideav_jpegxl::modular_fdis::{EntropyStream, MaLeaf, MaNode};
use oxideav_jpegxl::toc::Toc;

const FIXTURE: &[u8] = include_bytes!("fixtures/grey_8x8_lossless.jxl");

#[test]
fn r5_grey_8x8_per_cluster_prefix_trace() {
    let sig = container::detect(FIXTURE).unwrap();
    let codestream: Vec<u8> = match sig {
        container::Signature::RawCodestream => FIXTURE[2..].to_vec(),
        container::Signature::Isobmff => container::extract_codestream(FIXTURE).unwrap().to_vec(),
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
    let _fh = FrameHeader::read(&mut br, &fh_params).unwrap();
    let _toc = Toc::read(&mut br, &_fh).unwrap();
    let _lf_dequant = LfChannelDequantization::read(&mut br).unwrap();

    let global_use_tree = br.read_bit().unwrap();
    assert_eq!(global_use_tree, 1);

    // Walk the tree-stream + tree decode manually so we can stop right
    // before the symbol-stream prelude.
    let mut tree_stream = EntropyStream::read(&mut br, 6).unwrap();
    tree_stream.read_ans_state_init(&mut br).unwrap();
    let mut tree_hybrid = HybridUintState::new(tree_stream.lz77, tree_stream.lz_len_conf);

    let mut nodes: Vec<MaNode> = Vec::new();
    let mut nodes_left: u32 = 1;
    let mut ctx_id: u32 = 0;
    while nodes_left > 0 {
        let property_plus_1 =
            decode_uint_in_walk(&mut tree_hybrid, &mut tree_stream, &mut br, 1).unwrap();
        let property = property_plus_1 as i64 - 1;
        if property < 0 {
            let predictor =
                decode_uint_in_walk(&mut tree_hybrid, &mut tree_stream, &mut br, 2).unwrap();
            let uoffset =
                decode_uint_in_walk(&mut tree_hybrid, &mut tree_stream, &mut br, 3).unwrap();
            let mul_log =
                decode_uint_in_walk(&mut tree_hybrid, &mut tree_stream, &mut br, 4).unwrap();
            let mul_bits =
                decode_uint_in_walk(&mut tree_hybrid, &mut tree_stream, &mut br, 5).unwrap();
            nodes.push(MaNode::Leaf(MaLeaf {
                ctx: ctx_id,
                predictor,
                offset: unpack_signed(uoffset),
                multiplier: (mul_bits + 1) << mul_log,
            }));
            ctx_id += 1;
            nodes_left -= 1;
        } else {
            let uvalue =
                decode_uint_in_walk(&mut tree_hybrid, &mut tree_stream, &mut br, 0).unwrap();
            let value = unpack_signed(uvalue);
            let nodes_now = nodes.len() as u32;
            let left_child = nodes_now + nodes_left;
            let right_child = nodes_now + nodes_left + 1;
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
    let num_ctx = nodes.len().div_ceil(2);
    eprintln!(
        "MA tree decoded: nodes={} num_ctx={} bits={}",
        nodes.len(),
        num_ctx,
        br.bits_read()
    );
    eprintln!(
        "symbol stream num_ctx={num_ctx} bits_at_prelude_begin={}",
        br.bits_read()
    );

    // ---- Symbol stream prelude, manually walked ----

    let prelude_start = br.bits_read();

    let lz77_enabled = br.read_bit().unwrap() == 1;
    eprintln!(
        "lz77_enabled={lz77_enabled} bits={} (rel {})",
        br.bits_read(),
        br.bits_read() - prelude_start
    );

    let _lz77 = if lz77_enabled {
        let min_symbol = br
            .read_u32([
                U32Dist::Val(224),
                U32Dist::Val(512),
                U32Dist::Val(4096),
                U32Dist::BitsOffset(15, 8),
            ])
            .unwrap();
        let min_length = br
            .read_u32([
                U32Dist::Val(3),
                U32Dist::Val(4),
                U32Dist::BitsOffset(2, 5),
                U32Dist::BitsOffset(8, 9),
            ])
            .unwrap();
        eprintln!(
            "  lz77 min_symbol={min_symbol} min_length={min_length} bits={} (rel {})",
            br.bits_read(),
            br.bits_read() - prelude_start
        );
        Lz77Params {
            enabled: true,
            min_symbol,
            min_length,
        }
    } else {
        Lz77Params::default()
    };

    let _lz_len_conf = if lz77_enabled {
        let cfg = HybridUintConfig::read(&mut br, 8).unwrap();
        eprintln!(
            "  lz_len_conf={:?} bits={} (rel {})",
            cfg,
            br.bits_read(),
            br.bits_read() - prelude_start
        );
        cfg
    } else {
        HybridUintConfig {
            split_exponent: 8,
            msb_in_token: 0,
            lsb_in_token: 0,
            split: 256,
        }
    };

    let effective_num_dist = if lz77_enabled { num_ctx + 1 } else { num_ctx };

    // Clustering for 4 distributions.
    let cluster_map = if effective_num_dist > 1 {
        read_clustering(&mut br, effective_num_dist).unwrap()
    } else {
        vec![0u32; effective_num_dist]
    };
    eprintln!(
        "after clustering: cluster_map={:?} bits={} (rel {})",
        cluster_map,
        br.bits_read(),
        br.bits_read() - prelude_start
    );

    let n_clusters = num_clusters(&cluster_map) as usize;
    eprintln!("n_clusters={n_clusters}");

    // use_prefix_code + log_alphabet_size.
    let use_prefix_code = br.read_bit().unwrap() == 1;
    let log_alphabet_size = if use_prefix_code {
        15
    } else {
        5 + br.read_bits(2).unwrap()
    };
    eprintln!(
        "use_prefix_code={use_prefix_code} log_alphabet_size={log_alphabet_size} bits={} (rel {})",
        br.bits_read(),
        br.bits_read() - prelude_start
    );
    assert!(use_prefix_code, "grey_8x8 uses prefix code path");

    // Per-cluster HybridUintConfig.
    let mut configs: Vec<HybridUintConfig> = Vec::with_capacity(n_clusters);
    for c in 0..n_clusters {
        let cfg = HybridUintConfig::read(&mut br, log_alphabet_size).unwrap();
        eprintln!(
            "  cluster[{c}] HybridUintConfig: {:?} bits={} (rel {})",
            cfg,
            br.bits_read(),
            br.bits_read() - prelude_start
        );
        configs.push(cfg);
    }

    // Per-cluster symbol counts (Annex C.2.1 / D.3.1 — prefix path).
    let mut counts = Vec::with_capacity(n_clusters);
    for c in 0..n_clusters {
        let count = if br.read_bit().unwrap() == 0 {
            1u32
        } else {
            let n = br.read_bits(4).unwrap();
            assert!(n <= 14, "n {n} > 14");
            1 + (1 << n) + br.read_bits(n).unwrap()
        };
        eprintln!(
            "  cluster[{c}] count={count} bits={} (rel {})",
            br.bits_read(),
            br.bits_read() - prelude_start
        );
        counts.push(count);
    }

    // Per-cluster prefix-code histogram with trace.
    for (c, &count) in counts.iter().enumerate() {
        let cluster_begin = br.bits_read();
        eprintln!(
            "--- cluster[{c}] prefix-code at bit {cluster_begin} (rel {}), count={count} ---",
            cluster_begin - prelude_start
        );
        match read_prefix_code_traced(&mut br, count) {
            Ok((code, trace)) => {
                eprintln!(
                    "  cluster[{c}] OK: kind={:?} clcl={:?} kraft_clcl={} non_zero={} alphabet_size={} max_length={} bits_consumed={} (after at bit {} rel {})",
                    trace.kind,
                    trace.clcl,
                    trace.kraft_clcl,
                    trace.non_zero,
                    code.alphabet_size,
                    trace.max_length,
                    br.bits_read() - cluster_begin,
                    br.bits_read(),
                    br.bits_read() - prelude_start,
                );
                if let Some(lengths) = &trace.code_lengths {
                    let nonzero_lens: Vec<(usize, u32)> = lengths
                        .iter()
                        .enumerate()
                        .filter(|(_, &l)| l != 0)
                        .map(|(i, &l)| (i, l))
                        .collect();
                    eprintln!("  cluster[{c}] non-zero code-lengths: {:?}", nonzero_lens);
                    let kraft: u64 = lengths
                        .iter()
                        .filter(|&&l| l != 0)
                        .map(|&l| 1u64 << (15 - l))
                        .sum();
                    eprintln!(
                        "  cluster[{c}] code-length kraft sum = {} (target = {})",
                        kraft,
                        1u64 << 15
                    );
                }
            }
            Err(e) => {
                eprintln!(
                    "  cluster[{c}] FAIL at bit {} (rel {}): {e}",
                    br.bits_read(),
                    br.bits_read() - prelude_start
                );
                // Re-run the cluster from the start of the histogram
                // using the diagnose path that captures partial state
                // even on Kraft overflow.
                let mut br2 = BitReader::new(&codestream);
                for _ in 0..cluster_begin {
                    let _ = br2.read_bit().unwrap();
                }
                let kind = br2.read_bits(2).unwrap();
                eprintln!("  cluster[{c}] re-trace: kind={kind}");
                let trace = diagnose_complex_prefix(&mut br2, count, kind);
                eprintln!(
                    "  cluster[{c}] re-trace: clcl={:?} kraft={} non_zero={}",
                    trace.clcl, trace.kraft_clcl, trace.non_zero
                );
                if let Some(lengths) = trace.code_lengths {
                    let nonzero_lens: Vec<(usize, u32)> = lengths
                        .iter()
                        .enumerate()
                        .filter(|(_, &l)| l != 0)
                        .map(|(i, &l)| (i, l))
                        .collect();
                    eprintln!(
                        "  cluster[{c}] re-trace lengths (non-zero): {:?}",
                        nonzero_lens
                    );
                    let total_lens: usize = lengths.len();
                    let nz: usize = nonzero_lens.len();
                    let kraft: u64 = lengths
                        .iter()
                        .filter(|&&l| l != 0)
                        .map(|&l| 1u64 << (15 - l))
                        .sum();
                    eprintln!(
                        "  cluster[{c}] re-trace: total={total_lens} non_zero={nz} kraft_sum={kraft} (target = {})",
                        1u64 << 15
                    );
                    eprintln!(
                        "  cluster[{c}] re-trace bits read = {}",
                        br2.bits_read() - cluster_begin
                    );
                }
                let _ = ClclTrace::default();
                return;
            }
        }
    }
    eprintln!(
        "ALL CLUSTERS OK at bit {} (rel {})",
        br.bits_read(),
        br.bits_read() - prelude_start
    );
}

#[test]
fn r5_compare_with_existing_entropy_stream_read() {
    // Drive the un-instrumented EntropyStream::read against the same
    // fixture and confirm where it falls.
    let sig = container::detect(FIXTURE).unwrap();
    let codestream: Vec<u8> = match sig {
        container::Signature::RawCodestream => FIXTURE[2..].to_vec(),
        container::Signature::Isobmff => container::extract_codestream(FIXTURE).unwrap().to_vec(),
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
    let _fh = FrameHeader::read(&mut br, &fh_params).unwrap();
    let _toc = Toc::read(&mut br, &_fh).unwrap();
    let _lf_dequant = LfChannelDequantization::read(&mut br).unwrap();
    let _ = br.read_bit().unwrap();
    let tree = oxideav_jpegxl::modular_fdis::MaTreeFdis::read(&mut br);
    eprintln!(
        "MaTreeFdis::read result: err={:?} bits={}",
        tree.as_ref().err(),
        br.bits_read()
    );
}

fn decode_uint_in_walk(
    hybrid: &mut HybridUintState,
    entropy: &mut EntropyStream,
    br: &mut BitReader<'_>,
    ctx: u32,
) -> oxideav_core::Result<u32> {
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

// pull in U32Dist to shut up unused warning
#[allow(dead_code)]
const _U32_KEEP: U32Dist = U32Dist::Bits(2);
