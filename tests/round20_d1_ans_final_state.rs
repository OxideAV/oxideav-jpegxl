//! Round-20 ANS end-of-stream check on d1's LfCoefficients.
//!
//! Per FDIS D.3.3 last sentence: "After the decoder reads the last
//! symbol in a given stream, state is 0x130000". If our `LfCoefficients`
//! decoder over- or under-runs, the post-loop ANS state will NOT be
//! `0x130000` and we'll have a precise oracle for "stop at iteration K"
//! bisects.
//!
//! This test enables [`STATE_TRACE_ENABLED`] (which side-publishes the
//! latest state to [`LATEST_ANS_STATE`]), runs `LfCoefficients::read`,
//! then reads the trailing state and call count.
//!
//! `LATEST_ANS_STATE` only updates when the trace flag is on for the
//! duration of every `decode_symbol_with_refill`; we run the trace
//! over the whole LfCoefficients sub-bitstream (including its 6-bit
//! prelude + 32-bit state init).

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
use oxideav_jpegxl::toc::Toc;

const VARDCT_D1_JXL: &[u8] = include_bytes!("fixtures/vardct_256x256_d1.jxl");

#[test]
fn d1_lfcoefficients_ans_final_state_round_20() {
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

    // Reset the side channels and turn tracing on for the whole
    // LfCoefficients sub-bitstream.
    LATEST_ANS_STATE.store(0, Ordering::Relaxed);
    LATEST_ANS_CALL_COUNT.store(0, Ordering::Relaxed);
    STATE_TRACE_ENABLED.store(true, Ordering::Relaxed);

    let _lfc = LfCoefficients::read(&mut shared_br, &fh, &lf_global, lf_w, lf_h, 0).unwrap();

    STATE_TRACE_ENABLED.store(false, Ordering::Relaxed);
    let final_state = LATEST_ANS_STATE.load(Ordering::Relaxed);
    let n_calls = LATEST_ANS_CALL_COUNT.load(Ordering::Relaxed);

    eprintln!(
        "[r20-final] LfCoefficients final ANS state = 0x{:08x} after {} decode_symbol calls",
        final_state, n_calls
    );
    eprintln!(
        "[r20-final]   ANS_FINAL_STATE (D.3.3 sentinel) = 0x{:08x}",
        ANS_FINAL_STATE
    );
    let matches = final_state == ANS_FINAL_STATE;
    eprintln!(
        "[r20-final]   final_state == ANS_FINAL_STATE? {}  ← if FALSE, our LfCoeff per-sample loop is over- or under-running",
        matches
    );
    if !matches {
        // For debugging, also report low-bit truncation patterns —
        // shifting state right by 16 reveals if a byte boundary issue
        // would have settled things.
        eprintln!(
            "[r20-final]   state >> 16 = 0x{:04x} (vs 0x0013 in final state)",
            final_state >> 16
        );
    }
    // We don't assert here — Auditor mode just records the diagnostic.
    // A future round can flip this to an assertion once the bug is
    // identified.
}
