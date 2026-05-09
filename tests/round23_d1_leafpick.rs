//! Round-23 d1 LfCoefficients leaf-pick property dump + WP y=0 audit
//! (Auditor mode). Per the round-23 dispatch, this test:
//!
//! 1. Re-decodes the d1 LfCoefficients sub-bitstream under WP bias 3
//!    (spec) and 4 (auditor) with the [`LEAF_PICK_TRACE_TARGET`] set
//!    to Y' sample 22 (channel 0, x=22, y=0) — the round-22 first-
//!    divergent sample location.
//! 2. Dumps the full property vector at that sample (16 base + any
//!    previous-channel properties), the WP intermediates (te_*, n8/w8/
//!    nw8/ne8, wp_pred8, max_error), every interior node visited
//!    during the MA-tree walk, and the final leaf chosen.
//! 3. Side-by-side compares bias=3 vs bias=4 to flag the first
//!    decision node whose chosen branch flips between the two runs.
//! 4. Sanity-audits the WP edge-case at y=0 / NE-boundary by checking
//!    that te_n, te_nw, te_ne all evaluate to 0 at the trace target
//!    (they should: y=0 means the full top row of WP state is empty
//!    and te_ne falls back to te_n per spec H.5.2).
//!
//! The test does NOT assert on the leaf-pick result (Auditor mode) —
//! all output is via `eprintln!` under `--nocapture`. Sentinels for
//! per-fixture decode + ANS final-state are inherited from rounds
//! 11..22.

use std::sync::atomic::Ordering;

use oxideav_jpegxl::ans::symbol::{
    ANS_FINAL_STATE, LATEST_ANS_CALL_COUNT, LATEST_ANS_STATE, STATE_TRACE_ENABLED,
};
use oxideav_jpegxl::bitreader::BitReader;
use oxideav_jpegxl::container;
use oxideav_jpegxl::frame_header::{Encoding, FrameDecodeParams, FrameHeader};
use oxideav_jpegxl::global_modular::GlobalModular;
use oxideav_jpegxl::lf_global::{
    HfBlockContext, LfChannelCorrelation, LfChannelDequantization, LfGlobal, Quantizer,
};
use oxideav_jpegxl::lf_group::LfCoefficients;
use oxideav_jpegxl::metadata_fdis::{ImageMetadataFdis, SizeHeaderFdis};
use oxideav_jpegxl::modular_fdis::{
    encode_leaf_pick_target, MaNode, MaTreeFdis, LEAF_PICK_LOG, LEAF_PICK_LOG_ENABLED,
    LEAF_PICK_TRACE_BUF, LEAF_PICK_TRACE_LEAF, LEAF_PICK_TRACE_PROPS, LEAF_PICK_TRACE_TARGET,
    LEAF_PICK_TRACE_WP, WP_ROUND_BIAS,
};
use oxideav_jpegxl::toc::Toc;

const VARDCT_D1_JXL: &[u8] = include_bytes!("fixtures/vardct_256x256_d1.jxl");

const PROP_NAMES: [&str; 16] = [
    "channel(c)",
    "stream_index",
    "y",
    "x",
    "abs(N)",
    "abs(W)",
    "N",
    "W",
    "prop8(W-grad@x-1)",
    "grad(W+N-NW)",
    "W-NW",
    "NW-N",
    "N-NE",
    "N-NN",
    "W-WW",
    "max_error",
];

#[derive(Debug, Clone, Default)]
struct LeafPickRun {
    final_state: u32,
    n_calls: usize,
    props: Vec<i32>,
    wp: Vec<i32>,
    steps: Vec<(u32, u32, i32, i32, u32)>,
    leaf: Option<(u32, u32, i32, u32)>,
    /// First-256 dump of channel 0 (Y') for divergence-spot context.
    y_first_256: Vec<i32>,
}

fn decode_with_target_and_bias(
    target_channel: u32,
    target_x: u32,
    target_y: u32,
    bias: i32,
) -> LeafPickRun {
    let sig = container::detect(VARDCT_D1_JXL).expect("d1 has JXL signature");
    let codestream: Vec<u8> = match sig {
        container::Signature::RawCodestream => VARDCT_D1_JXL[2..].to_vec(),
        container::Signature::Isobmff => container::extract_codestream(VARDCT_D1_JXL)
            .unwrap()
            .to_vec(),
    };
    let mut br = BitReader::new(&codestream);
    let size = SizeHeaderFdis::read(&mut br).expect("SizeHeader");
    let metadata = ImageMetadataFdis::read(&mut br).expect("ImageMetadata");
    if metadata.colour_encoding.want_icc {
        return LeafPickRun::default();
    }
    br.pu0().expect("byte-align");
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
    let fh = FrameHeader::read(&mut br, &fh_params).expect("FrameHeader");
    assert_eq!(fh.encoding, Encoding::VarDct);
    let toc = Toc::read(&mut br, &fh).expect("TOC");
    let frame_data_start = br.bytes_consumed();
    let frame_bytes = &br.data()[frame_data_start..];
    let lf_global_bytes = &frame_bytes[0..toc.entries[0] as usize];

    let mut shared_br = BitReader::new_section(lf_global_bytes);
    let lf_dequant = LfChannelDequantization::read(&mut shared_br).unwrap();
    let quantizer = Quantizer::read(&mut shared_br).unwrap();
    let hbc = HfBlockContext::read(&mut shared_br).unwrap();
    let cfl = LfChannelCorrelation::read(&mut shared_br).unwrap();
    let global_modular = GlobalModular::read(&mut shared_br, &fh, &metadata).unwrap();
    let lf_global = LfGlobal {
        lf_dequant,
        quantizer: Some(quantizer),
        hf_block_context: Some(hbc),
        lf_channel_correlation: Some(cfl),
        global_modular,
    };
    let lf_w = fh.width.min(fh.group_dim() * 8);
    let lf_h = fh.height.min(fh.group_dim() * 8);

    let mut shared_br = BitReader::new_section(lf_global_bytes);
    shared_br.advance_bits(1026).unwrap();

    LATEST_ANS_STATE.store(0, Ordering::Relaxed);
    LATEST_ANS_CALL_COUNT.store(0, Ordering::Relaxed);
    STATE_TRACE_ENABLED.store(true, Ordering::Relaxed);
    WP_ROUND_BIAS.store(bias, Ordering::Relaxed);

    LEAF_PICK_TRACE_TARGET.store(
        encode_leaf_pick_target(target_channel, target_x, target_y),
        Ordering::Relaxed,
    );
    LEAF_PICK_TRACE_BUF.with(|b| b.borrow_mut().clear());
    LEAF_PICK_TRACE_PROPS.with(|s| s.borrow_mut().clear());
    LEAF_PICK_TRACE_WP.with(|s| s.borrow_mut().clear());
    LEAF_PICK_TRACE_LEAF.with(|s| *s.borrow_mut() = None);

    let lfc = LfCoefficients::read(&mut shared_br, &fh, &lf_global, lf_w, lf_h, 0).ok();

    STATE_TRACE_ENABLED.store(false, Ordering::Relaxed);
    LEAF_PICK_TRACE_TARGET.store(u64::MAX, Ordering::Relaxed);
    WP_ROUND_BIAS.store(3, Ordering::Relaxed);

    let final_state = LATEST_ANS_STATE.load(Ordering::Relaxed);
    let n_calls = LATEST_ANS_CALL_COUNT.load(Ordering::Relaxed);
    let props = LEAF_PICK_TRACE_PROPS.with(|s| s.borrow().clone());
    let wp = LEAF_PICK_TRACE_WP.with(|s| s.borrow().clone());
    let steps = LEAF_PICK_TRACE_BUF.with(|b| b.borrow().clone());
    let leaf = LEAF_PICK_TRACE_LEAF.with(|s| *s.borrow());
    let y_first_256 = lfc
        .as_ref()
        .map(|c| {
            let n = c.lf_quant[0].len().min(256);
            c.lf_quant[0][0..n].to_vec()
        })
        .unwrap_or_default();
    LeafPickRun {
        final_state,
        n_calls,
        props,
        wp,
        steps,
        leaf,
        y_first_256,
    }
}

fn dump_run(label: &str, run: &LeafPickRun) {
    eprintln!(
        "[r23] === {label}: final_state=0x{:08x} n_calls={} ===",
        run.final_state, run.n_calls
    );
    eprintln!(
        "[r23]   |final - sentinel(0x{:08x})| = {}",
        ANS_FINAL_STATE,
        (run.final_state as i64 - ANS_FINAL_STATE as i64).unsigned_abs()
    );
    eprintln!(
        "[r23]   property vector at trace target ({} entries):",
        run.props.len()
    );
    for (k, v) in run.props.iter().enumerate() {
        let name = if k < 16 {
            PROP_NAMES[k].to_string()
        } else {
            // Beyond 16: previous-channel props in 4-tuples per spec D.8.
            let off = k - 16;
            let chan_back = (off / 4) + 1; // 1, 2, 3, ...
            let kind = match off % 4 {
                0 => "abs(rC)",
                1 => "rC",
                2 => "abs(rC-rG)",
                _ => "rC-rG",
            };
            format!("prev_ch[-{chan_back}].{kind}")
        };
        eprintln!("[r23]     prop[{k:2}] {name:>22}  = {v}");
    }
    if !run.wp.is_empty() {
        eprintln!(
            "[r23]   WP intermediates: te_w={} te_n={} te_nw={} te_ne={} w8={} n8={} nw8={} ne8={} wp_pred8={} max_error={}",
            run.wp[0], run.wp[1], run.wp[2], run.wp[3],
            run.wp[4], run.wp[5], run.wp[6], run.wp[7], run.wp[8], run.wp[9],
        );
    }
    eprintln!(
        "[r23]   MA-tree decision steps ({} interior nodes traversed):",
        run.steps.len()
    );
    for (k, &(node_idx, prop_idx, value, pv, branch)) in run.steps.iter().enumerate() {
        let pname = if (prop_idx as usize) < 16 {
            PROP_NAMES[prop_idx as usize].to_string()
        } else {
            format!("prop[{prop_idx}]")
        };
        let dir = if branch == 1 {
            "LEFT (pv > value)"
        } else {
            "RIGHT (pv <= value)"
        };
        eprintln!(
            "[r23]     step[{k:2}] node={node_idx:3} prop={prop_idx:2} ({pname:>22}) value={value:8} pv={pv:8}  → {dir}",
        );
    }
    if let Some((ctx, predictor, offset, multiplier)) = run.leaf {
        eprintln!(
            "[r23]   final leaf: ctx={ctx} predictor={predictor} offset={offset} multiplier={multiplier}",
        );
    } else {
        eprintln!("[r23]   final leaf: <none — decode failed before reaching trace target?>");
    }
}

fn dump_diff(a: &LeafPickRun, b: &LeafPickRun) {
    eprintln!("[r23] === Side-by-side diff (run A = bias 3 spec, run B = bias 4 auditor) ===");
    if a.props.len() != b.props.len() {
        eprintln!(
            "[r23]   property-vector length mismatch: A={} vs B={}",
            a.props.len(),
            b.props.len()
        );
    }
    let n_props = a.props.len().min(b.props.len());
    for (k, (av, bv)) in a.props.iter().zip(b.props.iter()).enumerate().take(n_props) {
        if av != bv {
            let name = if k < 16 {
                PROP_NAMES[k].to_string()
            } else {
                format!("prop[{k}]")
            };
            eprintln!("[r23]   property differ at [{k:2}] {name:>22}: A={av} B={bv}",);
        }
    }
    let n_steps = a.steps.len().min(b.steps.len());
    for k in 0..n_steps {
        if a.steps[k] != b.steps[k] {
            eprintln!(
                "[r23]   step differ at [{k:2}]: A={:?} B={:?}",
                a.steps[k], b.steps[k]
            );
            // Highlight the branch flip — the most actionable signal.
            let (an, ap, av, apv, ab) = a.steps[k];
            let (bn, bp, bv, bpv, bb) = b.steps[k];
            if an == bn && ap == bp && av == bv && ab != bb {
                eprintln!(
                    "[r23]     >>> WRONG-BRANCH FLIP at node {an} on prop {ap}: A pv={apv} branch={ab}, B pv={bpv} branch={bb}",
                );
            }
        }
    }
    if a.steps.len() != b.steps.len() {
        eprintln!(
            "[r23]   step-count diff: A={} B={} (paths diverged in length)",
            a.steps.len(),
            b.steps.len()
        );
    }
    if a.leaf != b.leaf {
        eprintln!("[r23]   FINAL LEAF DIFFERS: A={:?} B={:?}", a.leaf, b.leaf);
    } else {
        eprintln!("[r23]   final leaf identical: {:?}", a.leaf);
    }
}

fn dump_y_around(label: &str, run: &LeafPickRun, x_target: usize) {
    if run.y_first_256.is_empty() {
        return;
    }
    let lo = x_target.saturating_sub(8);
    let hi = (x_target + 8).min(run.y_first_256.len());
    let line: Vec<String> = (lo..hi)
        .map(|i| {
            if i == x_target {
                format!("[*{}*]", run.y_first_256[i])
            } else {
                format!("{}", run.y_first_256[i])
            }
        })
        .collect();
    eprintln!(
        "[r23]   {label} Y' samples [{lo}..{hi}] around x_target={x_target}: {}",
        line.join(" ")
    );
}

/// Round-23 supplement: dump the full MA tree used by d1
/// LfCoefficients. The tree is shared across the 3072 samples; if it
/// has only a few interior nodes (as observed at sample 22 where only
/// 2 decisions are visited), then the leaf-context space is small and
/// the only knobs are property[1] (stream_index) and property[15]
/// (max_error). This shows the FULL tree topology.
#[test]
fn d1_ma_tree_topology_round_23() {
    eprintln!("[r23] === Round-23 supplement: dump d1 LfCoefficients MA tree topology ===");
    let sig = container::detect(VARDCT_D1_JXL).expect("d1 has JXL signature");
    let codestream: Vec<u8> = match sig {
        container::Signature::RawCodestream => VARDCT_D1_JXL[2..].to_vec(),
        container::Signature::Isobmff => container::extract_codestream(VARDCT_D1_JXL)
            .unwrap()
            .to_vec(),
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
    let toc = Toc::read(&mut br, &fh).unwrap();
    let frame_data_start = br.bytes_consumed();
    let frame_bytes = &br.data()[frame_data_start..];
    let lf_global_bytes = &frame_bytes[0..toc.entries[0] as usize];

    let mut shared_br = BitReader::new_section(lf_global_bytes);
    let _lf_dequant = LfChannelDequantization::read(&mut shared_br).unwrap();
    let _quantizer = Quantizer::read(&mut shared_br).unwrap();
    let _hbc = HfBlockContext::read(&mut shared_br).unwrap();
    let _cfl = LfChannelCorrelation::read(&mut shared_br).unwrap();
    let global_modular = GlobalModular::read(&mut shared_br, &fh, &metadata).unwrap();

    if let Some(global_tree) = global_modular.global_tree.as_ref() {
        let nodes: &Vec<MaNode> = &global_tree.nodes;
        eprintln!(
            "[r23] Global MA tree: {} nodes, num_ctx={}",
            nodes.len(),
            global_tree.num_ctx
        );
        for (i, node) in nodes.iter().enumerate() {
            match node {
                MaNode::Decision {
                    property,
                    value,
                    left_child,
                    right_child,
                } => {
                    let pname = if (*property as usize) < 16 {
                        PROP_NAMES[*property as usize]
                    } else {
                        "prev-channel"
                    };
                    eprintln!(
                        "[r23]   node[{i:3}] DECISION prop={property} ({pname}) value={value} → L={left_child} R={right_child}",
                    );
                }
                MaNode::Leaf(l) => {
                    eprintln!(
                        "[r23]   node[{i:3}] LEAF      ctx={} predictor={} offset={} multiplier={}",
                        l.ctx, l.predictor, l.offset, l.multiplier,
                    );
                }
            }
        }
        eprintln!(
            "[r23] EntropyStream: use_prefix_code={} log_alphabet_size={} configs={} entropies={} cluster_map(len={})={:?}",
            global_tree.entropy.use_prefix_code,
            global_tree.entropy.log_alphabet_size,
            global_tree.entropy.configs.len(),
            global_tree.entropy.entropies.len(),
            global_tree.entropy.cluster_map.len(),
            global_tree.entropy.cluster_map,
        );
        eprintln!("[r23] Per-cluster HybridUintConfig:");
        for (i, c) in global_tree.entropy.configs.iter().enumerate() {
            eprintln!(
                "[r23]   cluster[{i}]: split_exponent={} split={} msb_in_token={} lsb_in_token={}",
                c.split_exponent, c.split, c.msb_in_token, c.lsb_in_token,
            );
        }
    } else {
        eprintln!("[r23] no global tree present in d1's GlobalModular");
    }
    let _ = MaTreeFdis::clone; // use the import path
}

/// Round-23 supplement: dump leaf-pick at d1 Y' sample 0 (x=0, y=0) —
/// the first sample of LfCoefficients. Useful as a sanity-check
/// baseline since at the origin all neighbours collapse to 0 and the
/// MA tree's first decision (property[1] > 2) goes RIGHT (stream_index
/// = 1 for LfCoefficients), then the second (property[15] > 0) goes
/// RIGHT (max_error = 0), so the leaf SHOULD be ctx=1.
#[test]
fn d1_leafpick_at_y_sample_0_round_23() {
    eprintln!("[r23] === Round-23 supplement: leaf-pick at d1 Y' sample 0 (origin) ===");
    let run3 = decode_with_target_and_bias(0, 0, 0, 3);
    dump_run("Bias=3 (spec) at sample 0 (origin)", &run3);
}

#[test]
fn d1_leafpick_at_y_sample_22_round_23() {
    eprintln!(
        "[r23] === Round-23 leaf-pick property dump at d1 Y' sample 22 (channel=0, x=22, y=0) ==="
    );
    let target_channel = 0u32;
    let target_x = 22u32;
    let target_y = 0u32;

    let run3 = decode_with_target_and_bias(target_channel, target_x, target_y, 3);
    dump_run("Bias=3 (spec)", &run3);
    dump_y_around("bias=3", &run3, target_x as usize);

    eprintln!();
    let run4 = decode_with_target_and_bias(target_channel, target_x, target_y, 4);
    dump_run("Bias=4 (auditor toggle)", &run4);
    dump_y_around("bias=4", &run4, target_x as usize);

    eprintln!();
    dump_diff(&run3, &run4);

    // WP y=0 / NE-boundary audit. At (x=22, y=0):
    //  * te_n must be 0 (no row above).
    //  * te_nw must be 0 (no NW since y=0).
    //  * te_ne must be 0 (NE row above doesn't exist; te_ne_raw falls
    //    back to te_n which is 0).
    //  * te_w may be nonzero (carried from sample 21's true_err set).
    if !run3.wp.is_empty() {
        let (te_w, te_n, te_nw, te_ne) = (run3.wp[0], run3.wp[1], run3.wp[2], run3.wp[3]);
        eprintln!();
        eprintln!(
            "[r23] === WP y=0 / NE-boundary audit (bias=3) === te_w={te_w} te_n={te_n} te_nw={te_nw} te_ne={te_ne}",
        );
        if te_n != 0 || te_nw != 0 || te_ne != 0 {
            eprintln!("[r23]   *** WP-y0 BUG: te_n / te_nw / te_ne nonzero at y=0 sample (these should be 0 per spec H.5.2 fallbacks) ***");
        } else {
            eprintln!(
                "[r23]   WP-y0 boundary clean: top-row te_n / te_nw / te_ne all 0 as expected"
            );
        }
        // Cross-check: max_error at y=0, x>0 must equal te_w (per
        // Listing E.4 — the others are 0 so abs() comparisons keep te_w).
        if run3.wp.len() >= 10 {
            let max_error = run3.wp[9];
            if max_error == te_w {
                eprintln!("[r23]   max_error == te_w ({te_w}) as expected for y=0/x>0 sample");
            } else {
                eprintln!(
                    "[r23]   *** max_error mismatch at y=0/x>0: max_error={} but te_w={} ***",
                    max_error, te_w
                );
            }
        }
    }

    // Auditor mode — never assert.
}

/// Round-23 supplement: the round-22 sample-dump showed Y' samples 0..21
/// are bit-identical between bias=3 and bias=4. This re-verifies that
/// invariant (so we know the audit really starts at sample 22) and dumps
/// the leaf-pick at sample 21 too — the LAST sample where they agree
/// — so we can compare the leaf-pick path at the boundary.
#[test]
fn d1_leafpick_at_y_sample_21_round_23() {
    eprintln!("[r23] === Round-23 supplement: leaf-pick at d1 Y' sample 21 (last identical) ===");
    let run3 = decode_with_target_and_bias(0, 21, 0, 3);
    dump_run("Bias=3 (spec) at sample 21", &run3);
    let run4 = decode_with_target_and_bias(0, 21, 0, 4);
    dump_run("Bias=4 (auditor) at sample 21", &run4);
    dump_diff(&run3, &run4);
}

type LeafLog = Vec<(u32, u32, u32, u32, i32)>;

fn decode_with_log_and_bias(bias: i32) -> (u32, usize, LeafLog) {
    let sig = container::detect(VARDCT_D1_JXL).expect("d1 has JXL signature");
    let codestream: Vec<u8> = match sig {
        container::Signature::RawCodestream => VARDCT_D1_JXL[2..].to_vec(),
        container::Signature::Isobmff => container::extract_codestream(VARDCT_D1_JXL)
            .unwrap()
            .to_vec(),
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
    let toc = Toc::read(&mut br, &fh).unwrap();
    let frame_data_start = br.bytes_consumed();
    let frame_bytes = &br.data()[frame_data_start..];
    let lf_global_bytes = &frame_bytes[0..toc.entries[0] as usize];

    let mut shared_br = BitReader::new_section(lf_global_bytes);
    let lf_dequant = LfChannelDequantization::read(&mut shared_br).unwrap();
    let quantizer = Quantizer::read(&mut shared_br).unwrap();
    let hbc = HfBlockContext::read(&mut shared_br).unwrap();
    let cfl = LfChannelCorrelation::read(&mut shared_br).unwrap();
    let global_modular = GlobalModular::read(&mut shared_br, &fh, &metadata).unwrap();
    let lf_global = LfGlobal {
        lf_dequant,
        quantizer: Some(quantizer),
        hf_block_context: Some(hbc),
        lf_channel_correlation: Some(cfl),
        global_modular,
    };
    let lf_w = fh.width.min(fh.group_dim() * 8);
    let lf_h = fh.height.min(fh.group_dim() * 8);

    let mut shared_br = BitReader::new_section(lf_global_bytes);
    shared_br.advance_bits(1026).unwrap();

    LATEST_ANS_STATE.store(0, Ordering::Relaxed);
    LATEST_ANS_CALL_COUNT.store(0, Ordering::Relaxed);
    STATE_TRACE_ENABLED.store(true, Ordering::Relaxed);
    WP_ROUND_BIAS.store(bias, Ordering::Relaxed);
    LEAF_PICK_LOG.with(|b| b.borrow_mut().clear());
    LEAF_PICK_LOG_ENABLED.store(true, Ordering::Relaxed);

    let _ = LfCoefficients::read(&mut shared_br, &fh, &lf_global, lf_w, lf_h, 0).ok();

    LEAF_PICK_LOG_ENABLED.store(false, Ordering::Relaxed);
    STATE_TRACE_ENABLED.store(false, Ordering::Relaxed);
    WP_ROUND_BIAS.store(3, Ordering::Relaxed);

    let log = LEAF_PICK_LOG.with(|b| b.borrow().clone());
    (
        LATEST_ANS_STATE.load(Ordering::Relaxed),
        LATEST_ANS_CALL_COUNT.load(Ordering::Relaxed),
        log,
    )
}

/// Round-23 path-2 follow-up: bisect the FIRST leaf-pick divergence
/// between bias=3 and bias=4. Sample 22 was confirmed identical at the
/// leaf-pick level (same ctx, predictor, multiplier, offset) and same
/// property[15]; the SAMPLE values diverge purely from the bias delta in
/// the predictor-6 rounding step. This finds the first downstream
/// sample where the LEAF actually flips — that's the first place where
/// the ANS chain consumes different tokens between the two runs.
#[test]
fn d1_first_leaf_flip_bisect_round_23() {
    eprintln!("[r23] === Round-23 follow-up: bisect first leaf-flip between bias=3 and bias=4 ===");
    let (state3, calls3, log3) = decode_with_log_and_bias(3);
    let (state4, calls4, log4) = decode_with_log_and_bias(4);
    eprintln!(
        "[r23] bias=3: final_state=0x{state3:08x} calls={calls3} log_len={}",
        log3.len()
    );
    eprintln!(
        "[r23] bias=4: final_state=0x{state4:08x} calls={calls4} log_len={}",
        log4.len()
    );

    let n = log3.len().min(log4.len());
    let mut first_ctx_flip: Option<usize> = None;
    let mut first_p15_diff: Option<usize> = None;
    let mut ctx_flip_count = 0usize;
    let mut p15_diff_count = 0usize;
    for k in 0..n {
        let (c3, x3, y3, ctx3, p15_3) = log3[k];
        let (c4, x4, y4, ctx4, p15_4) = log4[k];
        // Sanity: positions must align (same per-sample iteration order).
        assert_eq!((c3, x3, y3), (c4, x4, y4));
        if ctx3 != ctx4 {
            ctx_flip_count += 1;
            if first_ctx_flip.is_none() {
                first_ctx_flip = Some(k);
            }
        }
        if p15_3 != p15_4 {
            p15_diff_count += 1;
            if first_p15_diff.is_none() {
                first_p15_diff = Some(k);
            }
        }
    }
    if let Some(k) = first_p15_diff {
        let (c, x, y, ctx3, p15_3) = log3[k];
        let (_, _, _, ctx4, p15_4) = log4[k];
        eprintln!(
            "[r23] FIRST property[15] diff at log_idx={k} (channel={c} x={x} y={y}): bias3 ctx={ctx3} max_error={p15_3} | bias4 ctx={ctx4} max_error={p15_4}",
        );
    } else {
        eprintln!("[r23] property[15] never diverges across the LfCoefficients decode");
    }

    // Count ctx histograms across the whole decode (bias=3 only — the
    // spec-conformant run). For LfCoefficients, only ctx 0 and ctx 1
    // appear (per the MA tree shown by `d1_ma_tree_topology_round_23`).
    let mut ctx_hist = [0usize; 16];
    for &(_, _, _, ctx, _) in &log3 {
        if (ctx as usize) < 16 {
            ctx_hist[ctx as usize] += 1;
        }
    }
    eprintln!("[r23] bias=3 leaf-ctx histogram: {ctx_hist:?}");
    if let Some(k) = first_ctx_flip {
        let (c, x, y, ctx3, p15_3) = log3[k];
        let (_, _, _, ctx4, p15_4) = log4[k];
        eprintln!(
            "[r23] FIRST leaf-ctx flip at log_idx={k} (channel={c} x={x} y={y}): bias3 ctx={ctx3} max_error={p15_3} | bias4 ctx={ctx4} max_error={p15_4}",
        );
        eprintln!(
            "[r23]   ctx_flip count over {n} samples: {ctx_flip_count}; p15_diff count: {p15_diff_count}",
        );

        // Dump the surrounding context (5 samples before, 5 after) for
        // both runs to see how the leaf-pick chain evolves around the
        // first divergence.
        eprintln!("[r23] surrounding samples (5 before, 5 after):");
        let lo = k.saturating_sub(5);
        let hi = (k + 6).min(n);
        for j in lo..hi {
            let (c, x, y, ctx3, p15_3) = log3[j];
            let (_, _, _, ctx4, p15_4) = log4[j];
            let mark = if j == k { ">>>" } else { "   " };
            eprintln!(
                "[r23]   {mark} log[{j:4}] ch={c} x={x:2} y={y:2}  bias3 ctx={ctx3} p15={p15_3:6}  |  bias4 ctx={ctx4} p15={p15_4:6}",
            );
        }
    } else {
        eprintln!("[r23] LEAF NEVER FLIPS — yet final state still differs ({:#x} vs {:#x}). This means the divergence is purely from token VALUE differences (different multiplier/offset interpretation) rather than leaf-context selection. The token COUNT and CONTEXT are identical → ANS state diverges due to different ANS distributions per context if and only if multiplier/offset propagate differently, OR (more likely) the SAME-leaf decode reads MORE bits because the prediction differs and the diff DIFFERS, requiring different hybrid-uint extra bits.", state3, state4);
    }
    // Auditor-mode — never assert.
}
