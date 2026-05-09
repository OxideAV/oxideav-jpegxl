//! Round-20 ANS state walk: log every ANS state value during the
//! `LfCoefficients` decode and identify the call index at which the
//! state first reaches `ANS_FINAL_STATE` (`0x00130000`). That call
//! index minus 1 is the true number of LF samples in the cjxl bitstream
//! — and the difference vs our 3072 tells us how many extra samples
//! we're decoding.

use std::sync::atomic::Ordering;

use oxideav_jpegxl::ans::symbol::{ANS_FINAL_STATE, STATE_TRACE_BUF, STATE_TRACE_ENABLED};
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

/// Temporarily widen the cap inside `decode_symbol_with_refill`'s
/// trace recorder by capturing all 3072 transitions. We do this by
/// directly poking at `STATE_TRACE_BUF` with a borrowed `Vec` reserve.
/// (The cap inside the recorder is `30`; this test instead reads the
/// trace's `_` placeholder via STATE_TRACE_BUF after the fact.)
///
/// Easier alternative: just enable tracing and read STATE_TRACE_BUF —
/// we'll see the first 30. To see ALL transitions we either need to
/// raise the cap in src or use the per-call hook. For round 20 we
/// settle for first-30 + final summary; the 3072-by-3072 walk would
/// be next-round work.
#[test]
fn d1_lfcoefficients_full_state_walk_round_20() {
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

    STATE_TRACE_BUF.with(|b| b.borrow_mut().clear());
    STATE_TRACE_ENABLED.store(true, Ordering::Relaxed);
    let _lfc = LfCoefficients::read(&mut shared_br, &fh, &lf_global, lf_w, lf_h, 0).unwrap();
    STATE_TRACE_ENABLED.store(false, Ordering::Relaxed);

    STATE_TRACE_BUF.with(|b| {
        let v = b.borrow();
        eprintln!(
            "[r20-walk-state] captured {} transitions (cap = 30 inside recorder)",
            v.len()
        );
        let mut found_final_at: Option<usize> = None;
        for (i, &(pre, _idx, _sym, _off, _prob, new, _refill)) in v.iter().enumerate() {
            if pre == ANS_FINAL_STATE {
                eprintln!(
                    "[r20-walk-state] PRE-state @ call {i} = 0x{ANS_FINAL_STATE:08x} (FINAL!)"
                );
                found_final_at.get_or_insert(i);
            }
            if new == ANS_FINAL_STATE {
                eprintln!(
                    "[r20-walk-state] POST-state @ call {i} = 0x{ANS_FINAL_STATE:08x} (FINAL!)"
                );
                found_final_at.get_or_insert(i);
            }
        }
        if found_final_at.is_none() {
            eprintln!(
                "[r20-walk-state] No call in the first {} transitions reached ANS_FINAL_STATE.",
                v.len()
            );
        }
    });
}
