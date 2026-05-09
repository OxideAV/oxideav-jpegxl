//! Round-19 diagnostic: extend the per-token trace with cluster index,
//! ANS-state refill bit count, and a `read_clustering` prelude bit
//! count. Drives the d1 LfCoefficients sub-bitstream and prints
//! statistics that disambiguate the round-18 candidates:
//!
//! 1. Is the cluster_map degenerate (all 0)? — would point at
//!    `read_clustering`.
//! 2. Does the cluster_map land all 16 contexts in 5 clusters as cjxl
//!    expects? — would point downstream at distribution / alias.
//! 3. How many bits does the leaf-level entropy-stream prelude consume?
//!    cjxl claims 602; ours can be measured here in isolation.
//!
//! The test never asserts; it emits diagnostic output under
//! `--nocapture` and is shipped as Auditor-mode evidence backing
//! `crates/oxideav-jpegxl/round19-d1-cluster.md`.

use std::sync::atomic::Ordering;

use oxideav_jpegxl::ans::hybrid_config::{with_trace_records, TRACE_ENABLED};
use oxideav_jpegxl::ans::symbol::{STATE_TRACE_BUF, STATE_TRACE_ENABLED};
use oxideav_jpegxl::bitreader::BitReader;
use oxideav_jpegxl::container;
use oxideav_jpegxl::frame_header::{Encoding, FrameDecodeParams, FrameHeader};
use oxideav_jpegxl::global_modular::GlobalModular;
use oxideav_jpegxl::lf_global::{
    HfBlockContext, LfChannelCorrelation, LfChannelDequantization, LfGlobal, Quantizer,
};
use oxideav_jpegxl::lf_group::LfCoefficients;
use oxideav_jpegxl::metadata_fdis::{ImageMetadataFdis, SizeHeaderFdis};
use oxideav_jpegxl::toc::Toc;

const VARDCT_D1_JXL: &[u8] = include_bytes!("fixtures/vardct_256x256_d1.jxl");

#[test]
fn d1_cluster_and_refill_trace_round_19() {
    // Pre-frame setup mirrors round-17/round-18 traces.
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
        return;
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
    assert_eq!(fh.encoding, Encoding::VarDct, "d1 must be VarDCT");
    let toc = Toc::read(&mut br, &fh).expect("TOC");
    assert_eq!(toc.entries.len(), 1);

    let frame_data_start = br.bytes_consumed();
    let frame_bytes = &br.data()[frame_data_start..];
    let lf_global_bytes = &frame_bytes[0..toc.entries[0] as usize];

    let mut shared_br = BitReader::new_section(lf_global_bytes);
    let lf_dequant = LfChannelDequantization::read(&mut shared_br).expect("LfChannelDequant");
    let quantizer = Quantizer::read(&mut shared_br).expect("Quantizer");
    let hbc = HfBlockContext::read(&mut shared_br).expect("HfBlockContext");
    let cfl = LfChannelCorrelation::read(&mut shared_br).expect("LfChannelCorrelation");
    // Enable trace during GlobalModular::read so the inner
    // [r19-prelude] line fires for the leaf-stream EntropyStream::read.
    TRACE_ENABLED.store(true, Ordering::Relaxed);
    let global_modular =
        GlobalModular::read(&mut shared_br, &fh, &metadata).expect("GlobalModular");
    TRACE_ENABLED.store(false, Ordering::Relaxed);
    // Drain any spurious trace records (no read_uint should have been
    // called, but be defensive).
    with_trace_records(|_| {});
    assert_eq!(shared_br.bits_read(), 1026, "LfGlobal must end at bit 1026");

    let lf_global = LfGlobal {
        lf_dequant,
        quantizer: Some(quantizer),
        hf_block_context: Some(hbc),
        lf_channel_correlation: Some(cfl),
        global_modular,
    };

    // Inspect the global tree exposed by GlobalModular: report the
    // leaf-level entropy-stream's cluster_map / configs / log_alphabet_size.
    if let Some(tree) = lf_global.global_modular.global_tree.as_ref() {
        let entropy = &tree.entropy;
        eprintln!(
            "[r19] global tree: {} nodes, num_ctx={}, log_alpha={}, use_prefix={}, n_clusters={}",
            tree.nodes.len(),
            tree.num_ctx,
            entropy.log_alphabet_size,
            entropy.use_prefix_code,
            entropy.configs.len(),
        );
        eprintln!("[r19] cluster_map ({} entries):", entropy.cluster_map.len());
        for (i, c) in entropy.cluster_map.iter().enumerate() {
            eprintln!("[r19]   ctx[{i}] -> cluster {c}");
        }
        for (i, cfg) in entropy.configs.iter().enumerate() {
            eprintln!(
                "[r19]   cfg[{i}] split_exp={} msb={} lsb={} split={}",
                cfg.split_exponent, cfg.msb_in_token, cfg.lsb_in_token, cfg.split,
            );
        }
        // Distinct clusters used = max(cluster_map) + 1 (FDIS num_clusters).
        let n_used = entropy.cluster_map.iter().copied().max().unwrap_or(0) + 1;
        eprintln!(
            "[r19] cluster_map: {} contexts -> {} distinct clusters",
            entropy.cluster_map.len(),
            n_used,
        );
        // Dump the per-cluster distribution arrays so we can verify
        // they're decoded sensibly.
        for (i, ent) in entropy.entropies.iter().enumerate() {
            match ent {
                oxideav_jpegxl::modular_fdis::ClusterEntropy::Ans { dist, .. } => {
                    let total: u32 = dist.iter().map(|&x| x as u32).sum();
                    let nonzero = dist.iter().filter(|&&x| x != 0).count();
                    eprintln!(
                        "[r19] D[cluster {i}] sum={total} nonzero={nonzero}/{} entries:",
                        dist.len()
                    );
                    let dump: Vec<String> = dist
                        .iter()
                        .enumerate()
                        .filter(|(_, &x)| x != 0)
                        .map(|(j, &x)| format!("[{j}]={x}"))
                        .collect();
                    eprintln!("[r19]   {}", dump.join(" "));
                }
                oxideav_jpegxl::modular_fdis::ClusterEntropy::Prefix { .. } => {
                    eprintln!("[r19] D[cluster {i}] PrefixCode (not dumped)");
                }
            }
        }
        // Dump the actual MA-tree node structure so we can see how
        // decisions branch and which leaves are reachable.
        eprintln!("[r19] MA tree nodes (DFS order):");
        for (i, n) in tree.nodes.iter().enumerate() {
            match n {
                oxideav_jpegxl::modular_fdis::MaNode::Decision {
                    property,
                    value,
                    left_child,
                    right_child,
                } => {
                    eprintln!(
                        "[r19]   #{i} DEC: prop[{property}] > {value} ? L={left_child} : R={right_child}",
                    );
                }
                oxideav_jpegxl::modular_fdis::MaNode::Leaf(l) => {
                    eprintln!(
                        "[r19]   #{i} LEAF: ctx={} pred={} off={} mul={}",
                        l.ctx, l.predictor, l.offset, l.multiplier
                    );
                }
            }
        }
    } else {
        eprintln!("[r19] WARN: global tree absent — d1 should always have one");
    }

    let lf_w = fh.width.min(fh.group_dim() * 8);
    let lf_h = fh.height.min(fh.group_dim() * 8);

    // Decode LfCoefficients with tracing on.
    let mut shared_br = BitReader::new_section(lf_global_bytes);
    shared_br.advance_bits(1026).unwrap();
    let bp = shared_br.bits_read();
    with_trace_records(|_| {});
    STATE_TRACE_BUF.with(|b| b.borrow_mut().clear());
    STATE_TRACE_ENABLED.store(true, Ordering::Relaxed);
    TRACE_ENABLED.store(true, Ordering::Relaxed);
    let _lfc = LfCoefficients::read(&mut shared_br, &fh, &lf_global, lf_w, lf_h, 0)
        .expect("LfCoefficients should not error");
    TRACE_ENABLED.store(false, Ordering::Relaxed);
    STATE_TRACE_ENABLED.store(false, Ordering::Relaxed);
    let consumed = shared_br.bits_read() - bp;
    eprintln!("[r19] LfCoefficients consumed {consumed} bits (cjxl LfGroup TOTAL = 11728)");

    // Dump first 30 ANS state transitions for sanity-check.
    STATE_TRACE_BUF.with(|b| {
        let v = b.borrow();
        eprintln!("[r19] first {} ANS state transitions (pre, idx, sym, off, prob, new, refill):", v.len());
        for (i, (pre, idx, sym, off, prob, new, refill)) in v.iter().enumerate() {
            eprintln!(
                "[r19]   #{i}: pre=0x{pre:08x} idx=0x{idx:03x} sym={sym} off={off} prob={prob} new=0x{new:08x} refill={refill}",
            );
        }
    });

    with_trace_records(|recs| {
        eprintln!("[r19] {} read_uint records captured", recs.len());

        // Cluster-usage histogram (count + bit-sums).
        use std::collections::BTreeMap;
        let mut by_cluster: BTreeMap<u32, (u64, u64, u64, u64)> = BTreeMap::new();
        // (calls, extra_bits, refill_bits, distinct ctx count via bitset)
        let mut ctx_per_cluster: BTreeMap<u32, std::collections::BTreeSet<u32>> = BTreeMap::new();
        let mut by_ctx: BTreeMap<u32, u64> = BTreeMap::new();
        let mut total_refill: u64 = 0;
        let mut total_extra: u64 = 0;
        for r in recs {
            let entry = by_cluster.entry(r.cluster).or_default();
            entry.0 += 1;
            entry.1 += r.n_extra_bits as u64;
            entry.2 += r.ans_refill_bits as u64;
            entry.3 = 0; // filled below
            ctx_per_cluster.entry(r.cluster).or_default().insert(r.ctx);
            *by_ctx.entry(r.ctx).or_insert(0) += 1;
            total_extra += r.n_extra_bits as u64;
            total_refill += r.ans_refill_bits as u64;
        }
        eprintln!("[r19] per-cluster usage:");
        for (c, (calls, extra, refill, _)) in &by_cluster {
            let ctxs = ctx_per_cluster.get(c).map(|s| s.len()).unwrap_or(0);
            eprintln!(
                "[r19]   cluster {c}: {calls} calls, {extra} extra-bits, {refill} refill-bits (={} refills), ctxs={ctxs}",
                refill / 16,
            );
        }
        eprintln!("[r19] per-ctx usage:");
        for (c, n) in &by_ctx {
            eprintln!("[r19]   ctx {c}: {n} calls");
        }
        eprintln!(
            "[r19] TOTAL: {} calls, {total_extra} extra-bits, {total_refill} refill-bits ({} refills)",
            recs.len(),
            total_refill / 16,
        );
        // Reconstruct: total_consumed = 32 (state init) + total_extra + total_refill
        let reconstructed = 32 + total_extra + total_refill;
        eprintln!(
            "[r19] consumed_breakdown: 32 (state init) + {total_extra} (extra) + {total_refill} (refill) = {reconstructed}, observed = {consumed}",
        );
    });
}
