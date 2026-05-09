//! Round-18 diagnostic: per-token bit accounting in the d1 LfCoefficients
//! sub-bitstream. Drives the LfCoefficients decode end-to-end with the
//! per-call trace in `oxideav_jpegxl::ans::hybrid_config::TRACE_ENABLED`
//! switched on, then prints summary statistics on the captured records.
//!
//! See `crates/oxideav-jpegxl/round17-d1-bisect.md` for ground truth and
//! the bit-position evidence that motivates this trace.

use std::sync::atomic::Ordering;

use oxideav_jpegxl::ans::hybrid_config::{with_trace_records, TRACE_ENABLED};
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
fn d1_per_token_trace_round_18() {
    // === Pre-frame setup (mirrors round-17 trace test). ===
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

    // Decode LfGlobal end-to-end, but throw the result away — we only
    // need the global tree it leaves on the LfGlobal struct.
    let mut shared_br = BitReader::new_section(lf_global_bytes);
    let lf_dequant = LfChannelDequantization::read(&mut shared_br).expect("LfChannelDequant");
    let quantizer = Quantizer::read(&mut shared_br).expect("Quantizer");
    let hbc = HfBlockContext::read(&mut shared_br).expect("HfBlockContext");
    let cfl = LfChannelCorrelation::read(&mut shared_br).expect("LfChannelCorrelation");
    let global_modular =
        GlobalModular::read(&mut shared_br, &fh, &metadata).expect("GlobalModular");
    assert_eq!(shared_br.bits_read(), 1026, "LfGlobal must end at bit 1026");

    let lf_global = LfGlobal {
        lf_dequant,
        quantizer: Some(quantizer),
        hf_block_context: Some(hbc),
        lf_channel_correlation: Some(cfl),
        global_modular,
    };

    let lf_w = fh.width.min(fh.group_dim() * 8);
    let lf_h = fh.height.min(fh.group_dim() * 8);

    // === Decode LfCoefficients with tracing on. ===
    let mut shared_br = BitReader::new_section(lf_global_bytes);
    shared_br.advance_bits(1026).unwrap();
    let bp = shared_br.bits_read();

    // Drain any stale records from a previous test invocation on the
    // same thread.
    with_trace_records(|_| {});
    TRACE_ENABLED.store(true, Ordering::Relaxed);
    let lfc = LfCoefficients::read(&mut shared_br, &fh, &lf_global, lf_w, lf_h, 0)
        .expect("LfCoefficients should not error");
    TRACE_ENABLED.store(false, Ordering::Relaxed);
    let consumed = shared_br.bits_read() - bp;

    eprintln!("[r18] LfCoefficients consumed {consumed} bits (cjxl LfGroup TOTAL = 11728)");
    eprintln!(
        "[r18] dims: ch0={}x{} ch1={}x{} ch2={}x{}",
        lfc.lf_quant_widths[0],
        lfc.lf_quant_heights[0],
        lfc.lf_quant_widths[1],
        lfc.lf_quant_heights[1],
        lfc.lf_quant_widths[2],
        lfc.lf_quant_heights[2],
    );

    with_trace_records(|recs| {
        eprintln!("[r18] {} read_uint records captured", recs.len());

        // Histogram of (split_exp, msb, lsb) configs encountered.
        use std::collections::HashMap;
        let mut by_cfg: HashMap<(u32, u32, u32), (u64, u64)> = HashMap::new();
        for r in recs {
            let key = (r.split_exponent, r.msb_in_token, r.lsb_in_token);
            let entry = by_cfg.entry(key).or_default();
            entry.0 += 1; // call count
            entry.1 += r.n_extra_bits as u64; // total extra bits
        }
        let mut keys: Vec<_> = by_cfg.keys().copied().collect();
        keys.sort();
        let mut total_extra: u64 = 0;
        let mut total_calls: u64 = 0;
        for key in &keys {
            let (n_calls, n_extra) = by_cfg[key];
            total_calls += n_calls;
            total_extra += n_extra;
            eprintln!(
                "[r18]   cfg(split_exp={}, msb={}, lsb={}): {} calls, {} extra-bit-reads, avg {:.3} extra/call",
                key.0, key.1, key.2, n_calls, n_extra, n_extra as f64 / n_calls as f64
            );
        }
        eprintln!("[r18] TOTAL: {total_calls} calls, {total_extra} extra-bits-read");

        // Distribution of n_extra_bits values
        let mut hist_n: HashMap<u32, u32> = HashMap::new();
        for r in recs {
            *hist_n.entry(r.n_extra_bits).or_insert(0) += 1;
        }
        let mut ks: Vec<_> = hist_n.keys().copied().collect();
        ks.sort();
        eprintln!("[r18] histogram of n_extra_bits:");
        for k in &ks {
            eprintln!("[r18]   n={k} -> {} times", hist_n[k]);
        }

        // Token magnitude histogram
        let mut hist_t: HashMap<u32, u32> = HashMap::new();
        for r in recs {
            let bucket = if r.token == 0 {
                0
            } else {
                32 - r.token.leading_zeros()
            };
            *hist_t.entry(bucket).or_insert(0) += 1;
        }
        let mut ks: Vec<_> = hist_t.keys().copied().collect();
        ks.sort();
        eprintln!("[r18] histogram of token magnitude (1+log2(token)):");
        for k in &ks {
            eprintln!("[r18]   bucket={k} -> {} times", hist_t[k]);
        }

        // Print first 12 records for spot-check
        eprintln!("[r18] first 12 records:");
        for (i, r) in recs.iter().take(12).enumerate() {
            eprintln!(
                "[r18]   #{i}: token={} cfg(split_exp={}, msb={}, lsb={}) n_extra={} value={}",
                r.token, r.split_exponent, r.msb_in_token, r.lsb_in_token, r.n_extra_bits, r.value
            );
        }
    });
}
