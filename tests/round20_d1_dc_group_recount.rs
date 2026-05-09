//! Round-20 (Auditor pivot) — re-verify `bits_consumed` semantics in the
//! cjxl `JXL_TRACE` output for `vardct_256x256_d1.jxl`.
//!
//! Round 17 / 18 / 19 all assumed cjxl's `DC_GROUP_END id=0
//! bits_consumed=12754` was an *absolute file-position* counter, in
//! which case `DC_GROUP` itself spans `12754 − 1026 = 11728` bits and
//! our `LfCoefficients` consuming `11995` bits would be 267-over the
//! ENTIRE LfGroup budget (LfCoeff + ModularLfGroup + HfMetadata).
//!
//! Round 20 falsifies that interpretation. Inside the same trace,
//! `AC_GLOBAL_END num_histograms=1 bits_consumed=307` cannot be a
//! cumulative file position — `307 < 1026`, so the section starts in
//! the middle of `DC_GLOBAL`. Therefore `bits_consumed` is **section-
//! local**: the `307` is the size of `AC_GLOBAL` itself. By the same
//! interpretation, `DC_GLOBAL_END bits_consumed=1026` says LfGlobal is
//! 1026 bits (matches our trace exactly), and **`DC_GROUP_END
//! bits_consumed=12754` says the LfGroup (LfCoefficients +
//! ModularLfGroup + HfMetadata) is 12754 bits, *not* 11728**.
//!
//! Under the corrected reading the implied HfMetadata budget is
//! `12754 − 11995 = 759` bits. The 267-bit "overshoot" is illusory.
//!
//! This test prints the corrected per-section budget and hex-dumps the
//! 759 bits at file offset 13021..13780 so a future round can walk the
//! HfMetadata sub-bitstream with confidence in the bit boundary.
//! It also re-runs HfMetadata::read at bit 13021 and reports where the
//! `Squeeze begin_c=39` symptom surfaces relative to the 759-bit
//! budget.

use oxideav_jpegxl::bitreader::BitReader;
use oxideav_jpegxl::container;
use oxideav_jpegxl::frame_header::{Encoding, FrameDecodeParams, FrameHeader};
use oxideav_jpegxl::global_modular::GlobalModular;
use oxideav_jpegxl::lf_global::{
    HfBlockContext, LfChannelCorrelation, LfChannelDequantization, LfGlobal, Quantizer,
};
use oxideav_jpegxl::lf_group::{HfMetadata, LfCoefficients};
use oxideav_jpegxl::metadata_fdis::{ImageMetadataFdis, SizeHeaderFdis};
use oxideav_jpegxl::toc::Toc;

const VARDCT_D1_JXL: &[u8] = include_bytes!("fixtures/vardct_256x256_d1.jxl");

/// cjxl `JXL_TRACE` reports for d1 (from
/// `docs/image/jpegxl/fixtures/vardct-256x256-d1/trace.txt`):
///
/// ```text
/// DC_GLOBAL_END  bits_consumed=1026
/// DC_GROUP_END   bits_consumed=12754
/// AC_GLOBAL_END  bits_consumed=307
/// ```
///
/// `307 < 1026` proves these counters are section-local sizes, NOT
/// cumulative file positions. So the LfGroup section is **12754** bits
/// long, and our LfCoefficients (11995 bits) leaves **759** bits for
/// HfMetadata.
const CJXL_LFGROUP_TOTAL_BITS: usize = 12754;

#[test]
fn d1_dc_group_recount_round_20() {
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
    assert_eq!(toc.entries.len(), 1);

    let frame_data_start = br.bytes_consumed();
    let frame_bytes = &br.data()[frame_data_start..];
    let lf_global_bytes = &frame_bytes[0..toc.entries[0] as usize];

    eprintln!(
        "[r20] codestream {} B, frame data start byte {}, TOC entry {} B",
        codestream.len(),
        frame_data_start,
        toc.entries[0]
    );

    // === LfGlobal — must end at section-local bit 1026.
    let mut shared_br = BitReader::new_section(lf_global_bytes);
    let lf_dequant = LfChannelDequantization::read(&mut shared_br).expect("LfChannelDequant");
    let quantizer = Quantizer::read(&mut shared_br).expect("Quantizer");
    let hbc = HfBlockContext::read(&mut shared_br).expect("HfBlockContext");
    let cfl = LfChannelCorrelation::read(&mut shared_br).expect("LfChannelCorrelation");
    let global_modular =
        GlobalModular::read(&mut shared_br, &fh, &metadata).expect("GlobalModular");
    let lf_global_end = shared_br.bits_read();
    assert_eq!(
        lf_global_end, 1026,
        "LfGlobal must end at section-local bit 1026 (cjxl DC_GLOBAL_END)"
    );
    let lf_global = LfGlobal {
        lf_dequant,
        quantizer: Some(quantizer),
        hf_block_context: Some(hbc),
        lf_channel_correlation: Some(cfl),
        global_modular,
    };

    let lf_w = fh.width.min(fh.group_dim() * 8);
    let lf_h = fh.height.min(fh.group_dim() * 8);

    // === LfCoefficients — currently consumes 11995 bits (per round 19).
    let mut shared_br = BitReader::new_section(lf_global_bytes);
    shared_br.advance_bits(1026).unwrap();
    let lfc_start = shared_br.bits_read();
    let _lfc = LfCoefficients::read(&mut shared_br, &fh, &lf_global, lf_w, lf_h, 0)
        .expect("LfCoefficients should not error");
    let lfc_end = shared_br.bits_read();
    let lfc_bits = lfc_end - lfc_start;

    // === Compute corrected DC_GROUP boundaries under section-local
    // semantics for `bits_consumed`.
    //
    // DC_GLOBAL_END=1026  → DC_GLOBAL section is 1026 bits.
    // DC_GROUP_END=12754  → DC_GROUP section is 12754 bits.
    //
    // DC_GROUP starts at section-local bit 1026 (immediately after
    // DC_GLOBAL). So DC_GROUP spans section-local bits
    // [1026, 1026 + 12754) = [1026, 13780).
    let dc_group_start = lf_global_end; // 1026
    let dc_group_end = dc_group_start + CJXL_LFGROUP_TOTAL_BITS; // 13780
    let hfm_start = lfc_end;
    let hfm_budget = dc_group_end.saturating_sub(hfm_start);

    eprintln!("[r20] === BIT-POSITION SUMMARY (section-local) ===");
    eprintln!(
        "[r20]   DC_GLOBAL: [0, {}) = {} bits   (cjxl: 1026)",
        lf_global_end, lf_global_end
    );
    eprintln!(
        "[r20]   DC_GROUP : [{}, {}) = {} bits  (cjxl: 12754)",
        dc_group_start,
        dc_group_end,
        dc_group_end - dc_group_start
    );
    eprintln!(
        "[r20]     LfCoefficients: [{}, {}) = {} bits   (our measurement)",
        lfc_start, lfc_end, lfc_bits
    );
    eprintln!("[r20]     ModularLfGroup: 0 bits (no qualifying channels for d1)");
    eprintln!(
        "[r20]     HfMetadata implied budget: [{}, {}) = {} bits",
        hfm_start, dc_group_end, hfm_budget
    );

    // The 267-bit "overshoot" claimed by r17/r18/r19 — re-evaluate.
    let r19_overshoot_claim_old = lfc_bits as i64 - 11728i64;
    let r20_overshoot_actual = (lfc_end as i64) - dc_group_end as i64;
    eprintln!(
        "[r20]   r17/r18/r19 claimed LfCoeff 'overshoot' vs 11728 = {} bits (BAD READ)",
        r19_overshoot_claim_old
    );
    eprintln!(
        "[r20]   r20 actual LfCoeff vs DC_GROUP_END (=13780)         = {} bits (negative = within budget)",
        r20_overshoot_actual
    );
    assert!(
        r20_overshoot_actual <= 0,
        "Under the section-local reading, LfCoefficients (ending at bit {}) must NOT overrun \
         DC_GROUP_END (=bit {})",
        lfc_end,
        dc_group_end
    );
    assert!(
        hfm_budget > 0,
        "HfMetadata implied budget must be > 0 under the corrected reading"
    );

    // === Hex-dump the 759-bit HfMetadata window for downstream walk.
    let hfm_byte_start = hfm_start / 8;
    let hfm_bit_off = hfm_start % 8;
    let hfm_byte_end = dc_group_end.div_ceil(8);
    let win = &lf_global_bytes[hfm_byte_start..hfm_byte_end.min(lf_global_bytes.len())];
    eprintln!(
        "[r20] HfMetadata window: byte [{}..{}) ({} B), bit offset within first byte = {}",
        hfm_byte_start,
        hfm_byte_end,
        win.len(),
        hfm_bit_off
    );
    let hex: String = win
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!("[r20]   hex: {hex}");

    // === Re-run HfMetadata::read at lfc_end and report what happens
    // relative to the 759-bit budget.
    let bp_hfm = shared_br.bits_read();
    eprintln!(
        "[r20] HfMetadata::read starts at bit {} (= section-local bit cursor)",
        bp_hfm
    );
    let r = HfMetadata::read(
        &mut shared_br,
        &lf_global,
        &metadata,
        lf_w,
        lf_h,
        0,
        fh.num_lf_groups(),
    );
    match r {
        Ok(hfm) => {
            let consumed = shared_br.bits_read() - bp_hfm;
            eprintln!(
                "[r20] HfMetadata OK consumed {} bits (budget {}); slack {} bits",
                consumed,
                hfm_budget,
                hfm_budget as i64 - consumed as i64
            );
            eprintln!(
                "[r20]   nb_blocks={} channel_widths={:?} channel_heights={:?}",
                hfm.nb_blocks, hfm.channel_widths, hfm.channel_heights
            );
        }
        Err(e) => {
            let err_at = shared_br.bits_read();
            let consumed = err_at - bp_hfm;
            eprintln!(
                "[r20] HfMetadata ERR @ bit {} (consumed {} of budget {}); err = {:?}",
                err_at, consumed, hfm_budget, e
            );
            // The r16/r17 symptom: `Squeeze begin_c=39 + num_c >
            // channel_count`. We expect the SAME error at the SAME
            // 233-bit offset under the new reading — proving that
            // begin_c=39 is what's actually on the wire at this byte
            // boundary, not a bit-position drift artefact.
            assert_eq!(
                consumed, 233,
                "round-17 trace pinpointed the Squeeze error at HfMetadata-relative bit 233; \
                 if this changed, the upstream LfCoefficients drifted"
            );
        }
    }
}
