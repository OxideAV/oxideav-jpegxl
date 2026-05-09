//! Round-21 diagnostic: per-cluster distribution dump + cluster-1
//! alias-table 256-entry dump for the d1 LfCoefficients sub-bitstream.
//!
//! Per the round-21 dispatch (path 1 + path 2 of the round-20
//! candidates):
//!
//! 1. Walk the per-cluster `read_distribution` calls in the prelude;
//!    for each cluster (0..=4) dump the alphabet_size,
//!    `HybridUintConfig`, the full ANS distribution table (length =
//!    `1 << log_alphabet_size`), and the first 30 alias-table entries.
//! 2. Audit the alias-table self-map branch (Vose pump's
//!    `cutoffs[u] == bucket_size` path) — the round-3 fix to
//!    `pos < cutoffs[i] → offset = pos` is one piece of this; the
//!    spec's `else if (cutoffs[i] < bucket_size)` filter on the
//!    initial underfull queue (skipping equal buckets) is another and
//!    is checked by inspecting whether any cluster ever has a
//!    `D[i] == bucket_size` entry that would trigger the divergence.
//!
//! The test never asserts (Auditor mode); it emits diagnostic output
//! under `--nocapture` and is shipped as evidence backing
//! `crates/oxideav-jpegxl/round21-d1-distbisect.md`.

use oxideav_jpegxl::bitreader::BitReader;
use oxideav_jpegxl::container;
use oxideav_jpegxl::frame_header::{FrameDecodeParams, FrameHeader};
use oxideav_jpegxl::global_modular::GlobalModular;
use oxideav_jpegxl::lf_global::{
    HfBlockContext, LfChannelCorrelation, LfChannelDequantization, Quantizer,
};
use oxideav_jpegxl::metadata_fdis::{ImageMetadataFdis, SizeHeaderFdis};
use oxideav_jpegxl::toc::Toc;

const VARDCT_D1_JXL: &[u8] = include_bytes!("fixtures/vardct_256x256_d1.jxl");

#[test]
fn d1_per_cluster_distribution_and_alias_dump_round_21() {
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
    let toc = Toc::read(&mut br, &fh).expect("TOC");
    assert_eq!(toc.entries.len(), 1);

    let frame_data_start = br.bytes_consumed();
    let frame_bytes = &br.data()[frame_data_start..];
    let lf_global_bytes = &frame_bytes[0..toc.entries[0] as usize];

    let mut shared_br = BitReader::new_section(lf_global_bytes);
    let _lf_dequant = LfChannelDequantization::read(&mut shared_br).expect("LfChannelDequant");
    let _quantizer = Quantizer::read(&mut shared_br).expect("Quantizer");
    let _hbc = HfBlockContext::read(&mut shared_br).expect("HfBlockContext");
    let _cfl = LfChannelCorrelation::read(&mut shared_br).expect("LfChannelCorrelation");
    let global_modular =
        GlobalModular::read(&mut shared_br, &fh, &metadata).expect("GlobalModular");
    assert_eq!(shared_br.bits_read(), 1026, "LfGlobal must end at bit 1026");

    let tree = global_modular
        .global_tree
        .as_ref()
        .expect("d1 should have a global tree");
    let entropy = &tree.entropy;
    eprintln!(
        "[r21] global tree: {} nodes, num_ctx={}, log_alpha={}, n_clusters={}",
        tree.nodes.len(),
        tree.num_ctx,
        entropy.log_alphabet_size,
        entropy.entropies.len(),
    );

    let table_size = 1usize << entropy.log_alphabet_size;
    let bucket_size = 1u32 << (12 - entropy.log_alphabet_size);
    eprintln!(
        "[r21] table_size={table_size}, bucket_size={bucket_size}, log_bucket={}",
        12 - entropy.log_alphabet_size
    );

    // Path 1: per-cluster distribution dump.
    for (cluster_idx, ent) in entropy.entropies.iter().enumerate() {
        let cfg = entropy.configs[cluster_idx];
        match ent {
            oxideav_jpegxl::modular_fdis::ClusterEntropy::Ans { dist, alias } => {
                let total: u32 = dist.iter().map(|&x| x as u32).sum();
                let nonzero = dist.iter().filter(|&&x| x != 0).count();
                let max_val = dist.iter().copied().max().unwrap_or(0);
                let n_at_bucket = dist.iter().filter(|&&x| x as u32 == bucket_size).count();
                let n_above_bucket = dist.iter().filter(|&&x| x as u32 > bucket_size).count();
                let n_below_bucket_nonzero = dist
                    .iter()
                    .filter(|&&x| x != 0 && (x as u32) < bucket_size)
                    .count();
                eprintln!(
                    "[r21] cluster {cluster_idx}: cfg(split_exp={} msb={} lsb={} split={})",
                    cfg.split_exponent, cfg.msb_in_token, cfg.lsb_in_token, cfg.split,
                );
                eprintln!(
                    "[r21]   D sum={total} nonzero={nonzero}/{} max={max_val} | bucket-stats: above={n_above_bucket} at={n_at_bucket} below_nz={n_below_bucket_nonzero}",
                    dist.len()
                );
                let dump: Vec<String> = dist
                    .iter()
                    .enumerate()
                    .filter(|(_, &x)| x != 0)
                    .map(|(j, &x)| format!("[{j}]={x}"))
                    .collect();
                eprintln!("[r21]   D nonzero entries: {}", dump.join(" "));

                // First 30 alias-table entries.
                eprintln!("[r21]   first 30 alias entries (sym/off/cut):");
                let n = alias.symbols.len().min(30);
                for i in 0..n {
                    eprintln!(
                        "[r21]     i={i:3}: sym={:3} off={:5} cut={:5}",
                        alias.symbols[i], alias.offsets[i], alias.cutoffs[i]
                    );
                }

                // Path 2: cluster-1 full alias-table dump (predictor-6
                // leaf used by d1 LfCoeff slot 1).
                if cluster_idx == 1 {
                    eprintln!(
                        "[r21] CLUSTER 1 FULL ALIAS TABLE ({} entries):",
                        alias.symbols.len()
                    );
                    for i in 0..alias.symbols.len() {
                        eprintln!(
                            "[r21]   slot[{i:3}]: sym={:3} off={:5} cut={:5}",
                            alias.symbols[i], alias.offsets[i], alias.cutoffs[i]
                        );
                    }
                    // Self-map branch sentinel: count slots where
                    // cutoffs[i] == 0 AND symbols[i] == i (the
                    // post-Vose self-map case from the trailing loop).
                    let n_self_map = (0..alias.symbols.len())
                        .filter(|&i| alias.cutoffs[i] == 0 && alias.symbols[i] as usize == i)
                        .count();
                    eprintln!(
                        "[r21] cluster 1 self-map slots: {n_self_map}/{}",
                        alias.symbols.len()
                    );
                }
            }
            oxideav_jpegxl::modular_fdis::ClusterEntropy::Prefix { .. } => {
                eprintln!("[r21] cluster {cluster_idx}: PrefixCode (not dumped)");
            }
        }
    }

    // Cluster-map echo to make this self-contained.
    eprintln!(
        "[r21] cluster_map ({} entries): {:?}",
        entropy.cluster_map.len(),
        entropy.cluster_map
    );
}
