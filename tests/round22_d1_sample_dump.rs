//! Round-22 d1 LfCoefficients sample-dump + WP rounding toggle
//! diagnostic (Auditor mode).
//!
//! Per the round-22 dispatch:
//!
//! 1. Decode the d1 LfCoefficients sub-bitstream and dump the first
//!    256 `lf_quant` samples per channel (Y, X, B) with the
//!    spec-default WP rounding bias (`(pred + 3) >> 3`).
//! 2. Re-decode the same sub-bitstream with the auditor-only WP bias
//!    flipped to `(pred + 4) >> 3` and report the post-decode ANS
//!    final state delta. The sentinel from §D.3.3 is `0x00130000` —
//!    if the +4 bias brings the final state closer to the sentinel,
//!    that's a strong indication that Table H.3's `+3` is a typo and
//!    the spec intends `+4` (which several open-source decoders
//!    observed empirically before the 2024 publication).
//!
//! The test does NOT assert (Auditor mode); all output is via
//! `eprintln!` under `--nocapture`. Evidence is captured in
//! `crates/oxideav-jpegxl/round22-d1-sampledump.md`.
//!
//! Note: `lf_quant` channel ordering follows the spec's `[X, Y, B]`
//! storage order with a `Y' / X' / B'` chroma-from-luma scheme; the
//! display labels ("Y / X / B") below match the bitstream order.

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
use oxideav_jpegxl::modular_fdis::WP_ROUND_BIAS;
use oxideav_jpegxl::toc::Toc;

const VARDCT_D1_JXL: &[u8] = include_bytes!("fixtures/vardct_256x256_d1.jxl");

fn decode_lfcoeff_with_bias(bias: i32) -> (Option<LfCoefficients>, u32, usize) {
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
        return (None, 0, 0);
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

    let lfc = LfCoefficients::read(&mut shared_br, &fh, &lf_global, lf_w, lf_h, 0).ok();

    STATE_TRACE_ENABLED.store(false, Ordering::Relaxed);
    // Restore default bias for any subsequent test invocations.
    WP_ROUND_BIAS.store(3, Ordering::Relaxed);

    let final_state = LATEST_ANS_STATE.load(Ordering::Relaxed);
    let n_calls = LATEST_ANS_CALL_COUNT.load(Ordering::Relaxed);
    (lfc, final_state, n_calls)
}

fn dump_first_n(label: &str, ch: usize, samples: &[i32], n: usize) {
    let n = n.min(samples.len());
    eprintln!(
        "[r22] {label} ch={ch} (first {n}/{} samples):",
        samples.len()
    );
    // Print 16 per row.
    for row in 0..n.div_ceil(16) {
        let start = row * 16;
        let end = (start + 16).min(n);
        let line: Vec<String> = samples[start..end]
            .iter()
            .map(|v| format!("{v:5}"))
            .collect();
        eprintln!("[r22]   [{start:3}..{end:3}]  {}", line.join(" "));
    }
    // Stats.
    let nonzero = samples.iter().take(n).filter(|&&v| v != 0).count();
    let min_v = samples.iter().take(n).copied().min().unwrap_or(0);
    let max_v = samples.iter().take(n).copied().max().unwrap_or(0);
    let sum: i64 = samples.iter().take(n).map(|&v| v as i64).sum();
    let mean = sum as f64 / n as f64;
    eprintln!("[r22]   stats: nonzero={nonzero}/{n}, min={min_v}, max={max_v}, mean={mean:.2}");
}

#[test]
fn d1_lfcoefficients_sample_dump_and_wp_rounding_toggle_round_22() {
    eprintln!(
        "[r22] === Path (a): lf_quant first-256 dump per channel under spec-default WP +3 ==="
    );
    let (lfc_3, final_state_3, n_calls_3) = decode_lfcoeff_with_bias(3);
    eprintln!("[r22] WP +3 (spec-default): final_state=0x{final_state_3:08x} n_calls={n_calls_3}");
    eprintln!(
        "[r22]   sentinel ANS_FINAL_STATE = 0x{ANS_FINAL_STATE:08x}; matches? {}",
        final_state_3 == ANS_FINAL_STATE
    );
    let delta_3 = (final_state_3 as i64 - ANS_FINAL_STATE as i64).unsigned_abs();
    eprintln!("[r22]   |final - sentinel| = {delta_3} (= 0x{delta_3:08x})");

    if let Some(lfc) = lfc_3.as_ref() {
        eprintln!(
            "[r22]   lf_quant_widths={:?} heights={:?}",
            lfc.lf_quant_widths, lfc.lf_quant_heights
        );
        // d1 fixture is 256x256 → LfGroup is 256x256, lf_quant is
        // ceil(256/8) = 32 wide × 32 tall = 1024 samples per channel.
        // 3 channels × 1024 = 3072 samples → 3072 ANS calls is a
        // possible but not exact match; round-19 confirmed.
        for c in 0..lfc.lf_quant.len() {
            dump_first_n("lf_quant", c, &lfc.lf_quant[c], 256);
        }
    } else {
        eprintln!("[r22]   LfCoefficients::read FAILED under WP +3 — no samples to dump");
    }

    eprintln!();
    eprintln!("[r22] === Path (c): re-decode under non-spec WP +4 bias (auditor toggle) ===");
    let (lfc_4, final_state_4, n_calls_4) = decode_lfcoeff_with_bias(4);
    eprintln!(
        "[r22] WP +4 (auditor toggle): final_state=0x{final_state_4:08x} n_calls={n_calls_4}"
    );
    eprintln!(
        "[r22]   sentinel ANS_FINAL_STATE = 0x{ANS_FINAL_STATE:08x}; matches? {}",
        final_state_4 == ANS_FINAL_STATE
    );
    let delta_4 = (final_state_4 as i64 - ANS_FINAL_STATE as i64).unsigned_abs();
    eprintln!("[r22]   |final - sentinel| = {delta_4} (= 0x{delta_4:08x})");

    eprintln!();
    eprintln!("[r22] === Path (c) supplement: re-decode under WP +0 / +7 (sanity sweep) ===");
    for sweep_bias in [0i32, 7] {
        let (_lfc, fs, calls) = decode_lfcoeff_with_bias(sweep_bias);
        let d = (fs as i64 - ANS_FINAL_STATE as i64).unsigned_abs();
        eprintln!("[r22]   bias={sweep_bias}: final=0x{fs:08x} calls={calls} |delta|={d}");
    }

    eprintln!();
    eprintln!("[r22] === WP +3 vs +4 final-state comparison ===");
    eprintln!("[r22]   bias=3: 0x{final_state_3:08x} (calls={n_calls_3}, |delta|={delta_3})");
    eprintln!("[r22]   bias=4: 0x{final_state_4:08x} (calls={n_calls_4}, |delta|={delta_4})");
    if delta_4 < delta_3 {
        eprintln!(
            "[r22]   *** WP +4 is CLOSER to sentinel by {} — Table H.3 may want +4 ***",
            delta_3 - delta_4
        );
    } else if delta_4 > delta_3 {
        eprintln!(
            "[r22]   WP +4 is FARTHER from sentinel by {} — spec-literal +3 stays",
            delta_4 - delta_3
        );
    } else {
        eprintln!("[r22]   WP +3 and +4 yield identical final-state deltas (no signal)");
    }

    if let (Some(a), Some(b)) = (lfc_3.as_ref(), lfc_4.as_ref()) {
        // Spot-check sample-stream divergence: count how many samples
        // differ between bias=3 and bias=4 across the first 256 of
        // each channel.
        for c in 0..a.lf_quant.len().min(b.lf_quant.len()) {
            let n = a.lf_quant[c].len().min(b.lf_quant[c].len()).min(256);
            let mut diffs = 0usize;
            let mut first_diff = None;
            for i in 0..n {
                if a.lf_quant[c][i] != b.lf_quant[c][i] {
                    diffs += 1;
                    if first_diff.is_none() {
                        first_diff = Some((i, a.lf_quant[c][i], b.lf_quant[c][i]));
                    }
                }
            }
            eprintln!(
                "[r22]   ch={c}: WP+3 vs +4 differ at {diffs}/{n} of first 256 samples; first={:?}",
                first_diff
            );
        }
    } else {
        eprintln!("[r22]   one or both decode paths failed; per-sample diff skipped");
    }

    // Auditor mode — never assert.
}
