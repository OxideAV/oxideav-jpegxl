//! Round-24 d1 LfCoefficients per-cluster ANS distribution byte-trace
//! + per-call alias-mapping invariant audit (Auditor mode).
//!
//! Round 24 implements the two top priorities from the round-23 close-out:
//!
//! 1. **Per-cluster ANS distribution byte-trace for clusters 0 + 1.**
//!    Round 21 dumped the per-cluster `D[]` non-zero entries and the
//!    first 30 alias entries; round 24 dumps the FULL post-decode
//!    distribution alongside per-bucket
//!    `(symbol, cutoff, offset, in_redirect_destination)` reconciliation
//!    so we can compare the alias build against an independent FDIS
//!    C.2.6 re-derivation.
//!
//! 2. **Per-call alias-mapping invariant check.** For each of the first
//!    30 ANS reads against the d1 LfCoefficients sub-stream, walk the
//!    state trace and re-derive `(symbol, offset)` directly from the
//!    cluster's alias table per the spec C.3.2 procedure
//!    (`AliasMapping(state & 0xFFF)`); compare against the value the
//!    decoder actually used. Any divergence at read M would be a smoking
//!    gun for the build-vs-lookup bug.
//!
//! Cluster identification is by `prob` matching: r23 confirmed that
//! LfCoefficients touches only cluster 0 (ctx 0, `max_error > 0` leaf)
//! and cluster 1 (ctx 1, `max_error <= 0` leaf). Each STATE_TRACE_BUF
//! row carries `prob = D[symbol]`; given the dump shows clusters 0 and
//! 1 have largely-disjoint non-zero supports, `prob` uniquely
//! identifies the cluster for most reads.
//!
//! Auditor mode: never asserts. Output via `eprintln!` under
//! `--nocapture`. Evidence backs `crates/oxideav-jpegxl/round24-d1-disttrace.md`.

use std::sync::atomic::Ordering;

use oxideav_jpegxl::ans::alias::AliasTable;
use oxideav_jpegxl::ans::symbol::{
    LATEST_ANS_CALL_COUNT, LATEST_ANS_STATE, STATE_TRACE_BUF, STATE_TRACE_ENABLED,
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
use oxideav_jpegxl::modular_fdis::{ClusterEntropy, WP_ROUND_BIAS};
use oxideav_jpegxl::toc::Toc;

const VARDCT_D1_JXL: &[u8] = include_bytes!("fixtures/vardct_256x256_d1.jxl");

/// Dump the FULL distribution and alias table for a single cluster.
fn dump_cluster(cluster_idx: usize, dist: &[u16], alias: &AliasTable, log_alphabet_size: u32) {
    let table_size = 1usize << log_alphabet_size;
    let log_bucket_size = 12 - log_alphabet_size;
    let bucket_size = 1u32 << log_bucket_size;
    let total: u32 = dist.iter().map(|&x| x as u32).sum();

    eprintln!("[r24] === CLUSTER {cluster_idx} FULL DISTRIBUTION + ALIAS DUMP ===");
    eprintln!(
        "[r24]   table_size={table_size} log_alpha={log_alphabet_size} bucket_size={bucket_size} log_bucket={log_bucket_size}"
    );
    eprintln!("[r24]   sum(D)={total} (must equal 4096)");

    // Full D[] dump including zeros.
    eprintln!("[r24]   FULL D[] (all {} entries):", dist.len());
    for chunk_start in (0..dist.len()).step_by(8) {
        let end = (chunk_start + 8).min(dist.len());
        let line: Vec<String> = (chunk_start..end)
            .map(|j| format!("[{j:2}]={:5}", dist[j]))
            .collect();
        eprintln!("[r24]     {}", line.join(" "));
    }

    // Per-bucket alias-table dump with reconciled meaning.
    eprintln!(
        "[r24]   FULL alias table ({} buckets):",
        alias.symbols.len()
    );
    eprintln!("[r24]     bucket  D[i]  cut  sym  off  | meaning");
    for i in 0..alias.symbols.len() {
        let d_val = dist.get(i).copied().unwrap_or(0);
        let cut = alias.cutoffs[i];
        let sym = alias.symbols[i];
        let off = alias.offsets[i];
        let meaning = if cut == 0 && sym as usize == i {
            format!("self-map (every pos returns ({i}, pos))")
        } else if cut == 0 {
            format!("ALL pos → redirect to sym={sym} off=off+pos={off}+pos")
        } else if cut as u32 == bucket_size {
            "(unreachable cut == bucket_size)".to_string()
        } else {
            format!("pos<{cut}: ({i}, pos); pos>={cut}: ({sym}, off+pos={off}+pos)")
        };
        eprintln!("[r24]     {i:6}  {d_val:4}  {cut:3}  {sym:3}  {off:3}  | {meaning}");
    }

    // Spec-D.3 invariant: count probability mass routed to each symbol.
    // For each bucket i:
    //   "in cutoff" range [0, cutoffs[i]) maps to symbol=i (so contributes
    //                                                  cutoffs[i] mass to D[i]).
    //   "redirect" range [cutoffs[i], bucket_size) maps to symbol=symbols[i]
    //                                                  (so contributes
    //                                                  bucket_size - cutoffs[i]
    //                                                  mass to D[symbols[i]]).
    let mut routed = vec![0u32; table_size];
    for i in 0..alias.symbols.len() {
        let cut = alias.cutoffs[i] as u32;
        let sym = alias.symbols[i] as usize;
        if cut == 0 && sym == i {
            // self-map: bucket_size mass to symbol=i.
            routed[i] = routed[i].saturating_add(bucket_size);
        } else {
            routed[i] = routed[i].saturating_add(cut);
            routed[sym] = routed[sym].saturating_add(bucket_size - cut);
        }
    }
    let routed_total: u32 = routed.iter().sum();
    eprintln!(
        "[r24]   alias-routed total = {routed_total} (must equal {})",
        bucket_size * (table_size as u32)
    );
    eprintln!("[r24]   per-symbol routed-mass vs declared-D divergence:");
    let mut any_divergence = false;
    for (j, &routed_val) in routed.iter().enumerate().take(table_size) {
        let d_val = dist.get(j).copied().unwrap_or(0) as u32;
        if routed_val != d_val {
            any_divergence = true;
            eprintln!(
                "[r24]     symbol {j:2}: D={d_val} routed={routed_val} (delta={:+})",
                routed_val as i64 - d_val as i64
            );
        }
    }
    if !any_divergence {
        eprintln!("[r24]     (none — alias table routes mass identically to D[])");
    }
}

#[test]
fn d1_per_cluster_distribution_byte_trace_round_24() {
    eprintln!("[r24] === Round-24 path (1): per-cluster D[] full byte-trace ===");

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

    let frame_data_start = br.bytes_consumed();
    let frame_bytes = &br.data()[frame_data_start..];
    let lf_global_bytes = &frame_bytes[0..toc.entries[0] as usize];

    let mut shared_br = BitReader::new_section(lf_global_bytes);
    let _ = LfChannelDequantization::read(&mut shared_br).unwrap();
    let _ = Quantizer::read(&mut shared_br).unwrap();
    let _ = HfBlockContext::read(&mut shared_br).unwrap();
    let _ = LfChannelCorrelation::read(&mut shared_br).unwrap();
    let global_modular = GlobalModular::read(&mut shared_br, &fh, &metadata).unwrap();

    let tree = global_modular
        .global_tree
        .as_ref()
        .expect("d1 has a global tree");
    let entropy = &tree.entropy;
    let log_alphabet_size = entropy.log_alphabet_size;

    // r23 identified: cluster_map[ctx] for ctx 0+1 → cluster 0, cluster 1.
    // Dump those two specifically; cluster 2..=4 already covered in r21.
    for cluster_idx in [0usize, 1] {
        if let Some(ent) = entropy.entropies.get(cluster_idx) {
            match ent {
                ClusterEntropy::Ans { dist, alias } => {
                    dump_cluster(cluster_idx, dist, alias, log_alphabet_size);
                }
                ClusterEntropy::Prefix { .. } => {
                    eprintln!(
                        "[r24] cluster {cluster_idx} is Prefix (not Ans) — skipping byte-trace"
                    );
                }
            }
        }
    }
}

/// Re-decode the d1 LfCoefficients sub-stream with state-tracing on,
/// then walk the captured trace and verify the alias-mapping invariant
/// per-call against the cluster's alias table.
#[test]
fn d1_per_call_alias_mapping_invariant_round_24() {
    eprintln!("[r24] === Round-24 path (2): per-call alias-mapping invariant audit ===");

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
    assert_eq!(fh.encoding, Encoding::VarDct);
    let toc = Toc::read(&mut br, &fh).expect("TOC");

    let frame_data_start = br.bytes_consumed();
    let frame_bytes = &br.data()[frame_data_start..];
    let lf_global_bytes = &frame_bytes[0..toc.entries[0] as usize];

    // First pass: collect the entropy stream and alias tables.
    let mut shared_br = BitReader::new_section(lf_global_bytes);
    let lf_dequant = LfChannelDequantization::read(&mut shared_br).unwrap();
    let quantizer = Quantizer::read(&mut shared_br).unwrap();
    let hbc = HfBlockContext::read(&mut shared_br).unwrap();
    let cfl = LfChannelCorrelation::read(&mut shared_br).unwrap();
    let global_modular = GlobalModular::read(&mut shared_br, &fh, &metadata).unwrap();

    let tree = global_modular.global_tree.as_ref().unwrap();
    let log_alphabet_size = tree.entropy.log_alphabet_size;
    let log_bucket_size = 12 - log_alphabet_size;
    let bucket_size = 1u32 << log_bucket_size;

    // Snapshot cluster 0 + cluster 1's distributions and alias tables.
    let snap = |c: usize| -> (Vec<u16>, AliasTable) {
        match &tree.entropy.entropies[c] {
            ClusterEntropy::Ans { dist, alias } => (dist.clone(), alias.clone()),
            _ => panic!("cluster {c} is not Ans"),
        }
    };
    let (d0, a0) = snap(0);
    let (d1, a1) = snap(1);

    let lf_global = LfGlobal {
        lf_dequant,
        quantizer: Some(quantizer),
        hf_block_context: Some(hbc),
        lf_channel_correlation: Some(cfl),
        global_modular,
    };
    let lf_w = fh.width.min(fh.group_dim() * 8);
    let lf_h = fh.height.min(fh.group_dim() * 8);

    // Second pass: enable trace + decode LfCoefficients to populate
    // STATE_TRACE_BUF.
    let mut shared_br = BitReader::new_section(lf_global_bytes);
    shared_br.advance_bits(1026).unwrap();

    LATEST_ANS_STATE.store(0, Ordering::Relaxed);
    LATEST_ANS_CALL_COUNT.store(0, Ordering::Relaxed);
    STATE_TRACE_BUF.with(|b| b.borrow_mut().clear());
    STATE_TRACE_ENABLED.store(true, Ordering::Relaxed);
    WP_ROUND_BIAS.store(3, Ordering::Relaxed);

    let _ = LfCoefficients::read(&mut shared_br, &fh, &lf_global, lf_w, lf_h, 0).ok();

    STATE_TRACE_ENABLED.store(false, Ordering::Relaxed);

    // Walk the first 30 trace rows and verify the alias-mapping invariant.
    let trace: Vec<(u32, u32, u16, u32, u32, u32, u32)> =
        STATE_TRACE_BUF.with(|b| b.borrow().clone());
    let n = trace.len().min(30);
    eprintln!(
        "[r24] state trace captured {} rows, auditing first {n} ANS reads",
        trace.len()
    );
    eprintln!(
        "[r24]   header: pre_state | slot | cluster_guess | sym | off_obs | sym_re | off_re | OK?"
    );

    let mut violations = 0usize;
    let mut first_violation: Option<usize> = None;
    let mut cluster_distribution = [0usize; 5];
    for (k, row) in trace.iter().enumerate().take(n) {
        let (pre_state, slot, sym, off, prob, _new_state, _refill) = *row;
        // Identify cluster: prob = D[sym]. Try cluster 0 then cluster 1.
        let prob_at_0 = d0.get(sym as usize).copied().unwrap_or(0) as u32;
        let prob_at_1 = d1.get(sym as usize).copied().unwrap_or(0) as u32;
        let (cluster_id, dist, alias) = if prob == prob_at_0 && prob == prob_at_1 {
            // Ambiguous — try both, report which matches the alias.
            // Default to cluster 0; flag as "ambiguous" below.
            (0, &d0, &a0)
        } else if prob == prob_at_0 {
            (0, &d0, &a0)
        } else if prob == prob_at_1 {
            (1, &d1, &a1)
        } else {
            // Cluster not 0 or 1 — log and skip invariant check.
            eprintln!(
                "[r24]   row {k:2}: pre=0x{pre_state:08x} slot={slot} sym={sym} prob={prob} — NOT cluster 0 or 1 (D0[sym]={prob_at_0} D1[sym]={prob_at_1})"
            );
            continue;
        };
        cluster_distribution[cluster_id] += 1;

        // Re-derive alias mapping per spec C.3.2 + C.2.6.
        let i = (slot >> log_bucket_size) as usize;
        let pos = slot & (bucket_size - 1);
        let cut = alias.cutoffs.get(i).copied().unwrap_or(0) as u32;
        let in_redirect = pos >= cut;
        let sym_re = if in_redirect {
            alias.symbols.get(i).copied().unwrap_or(0)
        } else {
            i as u16
        };
        let off_re = if in_redirect {
            alias.offsets.get(i).copied().unwrap_or(0) as u32 + pos
        } else {
            pos
        };
        let ok = sym_re == sym && off_re == off;
        if !ok {
            violations += 1;
            if first_violation.is_none() {
                first_violation = Some(k);
            }
        }
        eprintln!(
            "[r24]   row {k:2}: pre=0x{pre_state:08x} slot={slot:4} c={cluster_id} sym={sym:3} off_obs={off:5} sym_re={sym_re:3} off_re={off_re:5}  OK?={ok}  (D[{sym}]={prob}, cut={cut} in_redirect={in_redirect})"
        );

        // Verify the post-state too. Note: STATE_TRACE_BUF stores the
        // PRE-refill `new_state` value (the result of D[sym]*(pre>>12)+off
        // BEFORE the `(state<<16)|u(16)` refill). So compare against
        // `new_state_pre_refill` directly.
        let prob_re = dist.get(sym as usize).copied().unwrap_or(0) as u32;
        let new_state_pre_refill = prob_re * (pre_state >> 12) + off_re;
        if new_state_pre_refill != _new_state {
            eprintln!(
                "[r24]      ALSO: state-update mismatch! re={new_state_pre_refill} obs={_new_state} (refill={_refill})"
            );
        }
    }
    eprintln!(
        "[r24] === path-2 summary: {violations}/{n} alias-invariant violations, first at row {:?} ===",
        first_violation
    );
    eprintln!(
        "[r24]   cluster usage in first {n} reads: c0={} c1={} c2={} c3={} c4={}",
        cluster_distribution[0],
        cluster_distribution[1],
        cluster_distribution[2],
        cluster_distribution[3],
        cluster_distribution[4],
    );

    // Also verify across the FULL trace. For each call, try BOTH
    // cluster 0 and cluster 1 alias tables. The "true" cluster is
    // determined by the leaf-pick at that sample (r23 dump found
    // ctx 0 = max_error > 0 → cluster 0; ctx 1 = max_error <= 0 →
    // cluster 1). We don't have the leaf log here, so we declare a
    // violation only if BOTH candidate aliases fail to reproduce
    // (sym, off) — that proves it's not just a cluster-attribution
    // false alarm.
    let mut hard_violations: Vec<(usize, u32, u32, u16, u32, u32)> = Vec::new();
    let mut full_distribution = [0usize; 5];
    let mut unknown_cluster_calls = 0usize;
    let mut ambiguous_calls = 0usize;
    let try_cluster = |alias: &AliasTable, slot: u32| -> (u16, u32, u32, bool) {
        let i = (slot >> log_bucket_size) as usize;
        let pos = slot & (bucket_size - 1);
        let cut = alias.cutoffs.get(i).copied().unwrap_or(0) as u32;
        let in_redirect = pos >= cut;
        let sym_re = if in_redirect {
            alias.symbols.get(i).copied().unwrap_or(0)
        } else {
            i as u16
        };
        let off_re = if in_redirect {
            alias.offsets.get(i).copied().unwrap_or(0) as u32 + pos
        } else {
            pos
        };
        (sym_re, off_re, cut, in_redirect)
    };
    for (k, row) in trace.iter().enumerate() {
        let (_pre_state, slot, sym, off, prob, _new_state, _refill) = *row;
        let (s0, o0, _cut0, _ir0) = try_cluster(&a0, slot);
        let (s1, o1, _cut1, _ir1) = try_cluster(&a1, slot);
        let prob0 = d0.get(s0 as usize).copied().unwrap_or(0) as u32;
        let prob1 = d1.get(s1 as usize).copied().unwrap_or(0) as u32;
        let c0_match = s0 == sym && o0 == off && prob0 == prob;
        let c1_match = s1 == sym && o1 == off && prob1 == prob;
        if c0_match && c1_match {
            ambiguous_calls += 1;
            // Default attribute to whichever has prob match more uniquely;
            // both work, no violation.
            full_distribution[0] += 1;
        } else if c0_match {
            full_distribution[0] += 1;
        } else if c1_match {
            full_distribution[1] += 1;
        } else {
            unknown_cluster_calls += 1;
            hard_violations.push((k, _pre_state, slot, sym, off, prob));
        }
    }
    eprintln!(
        "[r24] full-trace audit: {} hard violations (neither c0 nor c1 alias reproduces (sym,off,prob)); ambiguous={ambiguous_calls}",
        hard_violations.len()
    );
    eprintln!(
        "[r24]   full-trace cluster usage: c0={} c1={} unknown={unknown_cluster_calls}",
        full_distribution[0], full_distribution[1]
    );
    if !hard_violations.is_empty() {
        eprintln!("[r24]   first 10 hard-violation rows (k, pre, slot, sym, off, prob):");
        for (k, pre, slot, sym, off, prob) in hard_violations.iter().take(10) {
            let (s0, o0, _, _) = try_cluster(&a0, *slot);
            let (s1, o1, _, _) = try_cluster(&a1, *slot);
            eprintln!(
                "[r24]     k={k:4}: pre=0x{pre:08x} slot={slot} obs=(sym={sym},off={off},prob={prob}) c0=(sym={s0},off={o0}) c1=(sym={s1},off={o1})"
            );
        }
    }
}
