//! Round-25 d1 LfCoefficients per-sample state-range dump (Auditor mode).
//!
//! Per the round-25 dispatch: extend round-23's leaf-pick audit beyond
//! sample 22 to capture the actual first-divergent ctx-flip at sample
//! 79. Round 23 saw the first ctx-flip there but didn't record the
//! property/leaf-pick state. With round 24's confirmation that ANS reads
//! are bit-correct (cluster-0 / cluster-1 D[] sum to 4096; alias-mapping
//! invariant holds across all 3072 calls; per-call state arithmetic
//! exact), the divergence at sample 79 must come from cluster/ctx
//! selection or from upstream sample reads feeding wrong inputs to WP.
//!
//! This test:
//! 1. Re-decodes the d1 LfCoefficients sub-bitstream under WP bias 3
//!    (spec) and 4 (auditor) with `set_rich_range(22, 79)`.
//! 2. Dumps the per-sample rich state for samples 22..=79 inclusive
//!    (the first 80 samples of channel 0, covering the first ctx-flip
//!    at sample 79 confirmed by round 23).
//! 3. Compares bias=3 vs bias=4 for each sample and classifies the
//!    divergence at the first ctx-flip:
//!    * Did the props differ? Which props?
//!    * Did the WP intermediates differ? Which?
//!    * Did the leaf differ?
//!    * Did the decoded sample value differ?
//!
//! The test does NOT assert on the result (Auditor mode); all output is
//! via `eprintln!` under `--nocapture`. Sentinels for per-fixture decode
//! + ANS final-state are inherited from rounds 11..24.

use std::sync::atomic::Ordering;
use std::sync::Mutex;

// Round-25: serialise the two tests in this file. The r25 instrumentation
// uses thread-local rich-range bounds (so cross-test contamination there
// is impossible) but `WP_ROUND_BIAS` and `STATE_TRACE_ENABLED` are
// process-global atomics that the harness's parallel test scheduler can
// otherwise clobber across threads. A simple mutex is the cheapest fix
// here without re-architecting the global atomics into thread-locals
// (which would require touching production paths).
static R25_TEST_LOCK: Mutex<()> = Mutex::new(());

use oxideav_jpegxl::ans::symbol::{LATEST_ANS_CALL_COUNT, LATEST_ANS_STATE, STATE_TRACE_ENABLED};
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
    set_rich_range, RichLeafPickLog, RICH_LEAF_PICK_LOG, WP_ROUND_BIAS,
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

const WP_NAMES: [&str; 10] = [
    "te_w",
    "te_n",
    "te_nw",
    "te_ne",
    "w8",
    "n8",
    "nw8",
    "ne8",
    "wp_pred8",
    "max_error",
];

fn decode_with_rich_range(
    bias: i32,
    range_lo: u32,
    range_hi: u32,
) -> (u32, usize, Vec<RichLeafPickLog>) {
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
    assert_eq!(fh.encoding, Encoding::VarDct);
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

    RICH_LEAF_PICK_LOG.with(|b| b.borrow_mut().clear());
    set_rich_range(range_lo, range_hi);

    let _ = LfCoefficients::read(&mut shared_br, &fh, &lf_global, lf_w, lf_h, 0).ok();

    set_rich_range(u32::MAX, u32::MAX);
    STATE_TRACE_ENABLED.store(false, Ordering::Relaxed);
    WP_ROUND_BIAS.store(3, Ordering::Relaxed);

    let log = RICH_LEAF_PICK_LOG.with(|b| b.borrow().clone());
    (
        LATEST_ANS_STATE.load(Ordering::Relaxed),
        LATEST_ANS_CALL_COUNT.load(Ordering::Relaxed),
        log,
    )
}

fn dump_rich_entry(idx: u32, e: &RichLeafPickLog) {
    let (c, x, y, props, wp, leaf, token, diff, pred, v) = e;
    eprintln!(
        "[r25]  log[{idx:3}] ch={c} x={x:2} y={y:2}  leaf(ctx={} pred={} off={} mult={})  token={token} diff={diff} pred={pred} v={v}",
        leaf.0, leaf.1, leaf.2, leaf.3,
    );
    let prop_show: Vec<String> = props
        .iter()
        .enumerate()
        .filter(|(k, _)| *k >= 4) // 0..=3 are stable (channel, stream_index, y, x)
        .map(|(k, v)| {
            let nm = if k < 16 { PROP_NAMES[k] } else { "prev_ch" };
            format!("[{k}]{nm}={v}")
        })
        .collect();
    eprintln!("[r25]    props: {}", prop_show.join("  "));
    let wp_show: Vec<String> = wp
        .iter()
        .enumerate()
        .map(|(k, v)| format!("{}={v}", WP_NAMES[k]))
        .collect();
    eprintln!("[r25]    wp:    {}", wp_show.join("  "));
}

fn diff_props(a: &[i32], b: &[i32]) -> Vec<(usize, i32, i32)> {
    let n = a.len().min(b.len());
    let mut out = Vec::new();
    for k in 0..n {
        if a[k] != b[k] {
            out.push((k, a[k], b[k]));
        }
    }
    out
}

fn diff_wp(a: &[i32; 10], b: &[i32; 10]) -> Vec<(usize, i32, i32)> {
    let mut out = Vec::new();
    for k in 0..10 {
        if a[k] != b[k] {
            out.push((k, a[k], b[k]));
        }
    }
    out
}

fn fmt_prop_diff(d: &[(usize, i32, i32)]) -> String {
    if d.is_empty() {
        return "-".to_string();
    }
    d.iter()
        .map(|(k, a, b)| {
            let nm = if *k < 16 { PROP_NAMES[*k] } else { "prev_ch" };
            format!("[{k}]{nm}({a}→{b})")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn fmt_wp_diff(d: &[(usize, i32, i32)]) -> String {
    if d.is_empty() {
        return "-".to_string();
    }
    d.iter()
        .map(|(k, a, b)| format!("{}({a}→{b})", WP_NAMES[*k]))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn d1_per_sample_state_range_22_to_79_round_25() {
    let _g = R25_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    eprintln!(
        "[r25] === Round-25 d1 LfCoefficients per-sample rich-state dump for log_idx 22..=79 ==="
    );
    let (state3, calls3, log3) = decode_with_rich_range(3, 22, 79);
    let (state4, calls4, log4) = decode_with_rich_range(4, 22, 79);
    eprintln!(
        "[r25] bias=3: final_state=0x{state3:08x} ans_calls={calls3} rich_entries={}",
        log3.len()
    );
    eprintln!(
        "[r25] bias=4: final_state=0x{state4:08x} ans_calls={calls4} rich_entries={}",
        log4.len()
    );
    assert_eq!(
        log3.len(),
        log4.len(),
        "rich-log lengths must match between biases (same iteration order)"
    );
    if log3.is_empty() {
        eprintln!("[r25] no rich-log entries captured; aborting");
        return;
    }

    eprintln!();
    eprintln!("[r25] === Side-by-side per-sample diff (bias=3 vs bias=4) for log_idx 22..=79 ===");
    eprintln!(
        "[r25]   format: log_idx ch x y  | leaf_diff | prop_diff | wp_diff | token_diff | val_diff"
    );

    let mut first_ctx_flip: Option<usize> = None;
    let mut first_prop_diff_kind: Option<(usize, usize)> = None; // (log_idx, prop_idx)
    let mut first_wp_diff_kind: Option<(usize, usize)> = None;
    let mut first_token_diff: Option<usize> = None;
    let mut first_val_diff: Option<usize> = None;

    for k in 0..log3.len() {
        let a = &log3[k];
        let b = &log4[k];
        let log_idx = 22 + k as u32;
        // sanity: same identity
        assert_eq!((a.0, a.1, a.2), (b.0, b.1, b.2));
        let prop_diffs = diff_props(&a.3, &b.3);
        let wp_diffs = diff_wp(&a.4, &b.4);
        let leaf_diff = a.5 != b.5;
        let token_diff = a.6 != b.6;
        let val_diff = a.9 != b.9;
        if leaf_diff && first_ctx_flip.is_none() {
            first_ctx_flip = Some(k);
        }
        if !prop_diffs.is_empty() && first_prop_diff_kind.is_none() {
            first_prop_diff_kind = Some((k, prop_diffs[0].0));
        }
        if !wp_diffs.is_empty() && first_wp_diff_kind.is_none() {
            first_wp_diff_kind = Some((k, wp_diffs[0].0));
        }
        if token_diff && first_token_diff.is_none() {
            first_token_diff = Some(k);
        }
        if val_diff && first_val_diff.is_none() {
            first_val_diff = Some(k);
        }

        let leaf_str = if leaf_diff {
            format!("LEAF! {:?}→{:?}", a.5, b.5)
        } else {
            "-".to_string()
        };
        let token_str = if token_diff {
            format!("token({}→{})", a.6, b.6)
        } else {
            "-".to_string()
        };
        let val_str = if val_diff {
            format!("v({}→{})", a.9, b.9)
        } else {
            "-".to_string()
        };
        let prop_str = fmt_prop_diff(&prop_diffs);
        let wp_str = fmt_wp_diff(&wp_diffs);
        eprintln!(
            "[r25]  log[{log_idx:3}] ch={} x={:2} y={:2} | {} | props:{} | wp:{} | {} | {}",
            a.0, a.1, a.2, leaf_str, prop_str, wp_str, token_str, val_str
        );
    }

    eprintln!();
    eprintln!("[r25] === First-divergence summary ===");
    eprintln!(
        "[r25]   first val_diff       at offset {:?} (log_idx {:?})",
        first_val_diff,
        first_val_diff.map(|k| 22 + k)
    );
    eprintln!(
        "[r25]   first wp_diff_kind   at offset {:?} (log_idx {:?}) → wp[{}]={}",
        first_wp_diff_kind.map(|x| x.0),
        first_wp_diff_kind.map(|x| 22 + x.0),
        first_wp_diff_kind
            .map(|x| x.1.to_string())
            .unwrap_or_else(|| "?".to_string()),
        first_wp_diff_kind.map(|x| WP_NAMES[x.1]).unwrap_or("?")
    );
    eprintln!(
        "[r25]   first prop_diff_kind at offset {:?} (log_idx {:?}) → prop[{}]={}",
        first_prop_diff_kind.map(|x| x.0),
        first_prop_diff_kind.map(|x| 22 + x.0),
        first_prop_diff_kind
            .map(|x| x.1.to_string())
            .unwrap_or_else(|| "?".to_string()),
        first_prop_diff_kind
            .map(|x| if x.1 < 16 { PROP_NAMES[x.1] } else { "prev_ch" })
            .unwrap_or("?")
    );
    eprintln!(
        "[r25]   first token_diff     at offset {:?} (log_idx {:?})",
        first_token_diff,
        first_token_diff.map(|k| 22 + k)
    );
    eprintln!(
        "[r25]   first ctx_flip       at offset {:?} (log_idx {:?})",
        first_ctx_flip,
        first_ctx_flip.map(|k| 22 + k)
    );

    if let Some(k) = first_ctx_flip {
        eprintln!();
        eprintln!(
            "[r25] === FIRST CTX-FLIP — full rich entries for both biases at log_idx {} ===",
            22 + k as u32
        );
        eprintln!("[r25] --- bias=3 (spec) ---");
        dump_rich_entry(22 + k as u32, &log3[k]);
        eprintln!("[r25] --- bias=4 (auditor) ---");
        dump_rich_entry(22 + k as u32, &log4[k]);
    }

    // Auditor mode — never assert on the diagnostic output.
}

/// Round-25 supplement: also dump the rich entries for log_idx 21..=24
/// to confirm the boundary handoff between the round-22-confirmed
/// "identical up to sample 21" and the round-25 audit window starting at
/// sample 22.
#[test]
fn d1_per_sample_state_boundary_21_to_24_round_25() {
    let _g = R25_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    eprintln!("[r25] === Round-25 supplement: boundary rich-state dump for log_idx 21..=24 ===");
    let (_, _, log3) = decode_with_rich_range(3, 21, 24);
    let (_, _, log4) = decode_with_rich_range(4, 21, 24);
    assert_eq!(log3.len(), log4.len());
    for k in 0..log3.len() {
        let log_idx = 21 + k as u32;
        eprintln!("[r25] --- log[{log_idx}] bias=3 ---");
        dump_rich_entry(log_idx, &log3[k]);
        eprintln!("[r25] --- log[{log_idx}] bias=4 ---");
        dump_rich_entry(log_idx, &log4[k]);
    }
}
