//! Round-17 Auditor diagnostic: bit-position trace of the d1 fixture.
//!
//! Replicates the relevant slice of `decode_one_frame` step-by-step
//! against `vardct_256x256_d1.jxl`, recording the bit cursor before/after
//! each major sub-component (LfGlobal sub-bundles + LfGroup sub-bundles).
//!
//! This is the round-17 diagnostic deliverable: it captures bit-precise
//! evidence of the divergence between our decoder and the empirical
//! ground truth in
//! `docs/image/jpegxl/fixtures/vardct-256x256-d1/trace.txt` (cjxl
//! verbose-trace output, written by the docs collaborator).
//!
//! Test does NOT assert correctness; it just runs the trace through and
//! confirms that `LfGlobal::read` succeeds at the expected boundary
//! (codestream-relative bit 1026) and that `LfCoefficients::read`
//! over-consumes, leaving `HfMetadata::read` to surface the round-16
//! "Squeeze begin_c=39" garbage symptom downstream. See
//! `crates/oxideav-jpegxl/round17-d1-bisect.md` for the analysis +
//! r18 candidate.

use oxideav_jpegxl::bitreader::{BitReader, U32Dist};
use oxideav_jpegxl::container;
use oxideav_jpegxl::frame_header::{Encoding, FrameDecodeParams, FrameHeader};
use oxideav_jpegxl::global_modular::GlobalModular;
use oxideav_jpegxl::lf_global::{
    HfBlockContext, LfChannelCorrelation, LfChannelDequantization, LfGlobal, Quantizer,
};
use oxideav_jpegxl::lf_group::{HfMetadata, LfCoefficients};
use oxideav_jpegxl::metadata_fdis::{ImageMetadataFdis, SizeHeaderFdis};
use oxideav_jpegxl::modular_fdis::WpHeader;
use oxideav_jpegxl::toc::Toc;

const VARDCT_D1_JXL: &[u8] = include_bytes!("fixtures/vardct_256x256_d1.jxl");

/// Walk through the d1 codestream, reporting bit-cursor positions at
/// each LfGlobal / LfGroup sub-component. Output is captured by `cargo
/// test -- --nocapture` so the trace can be diffed against
/// `docs/image/jpegxl/fixtures/vardct-256x256-d1/trace.txt`.
#[test]
fn d1_bit_position_walk_round_17() {
    // === Pre-frame: detect signature, strip container if needed. ===
    let sig = container::detect(VARDCT_D1_JXL).expect("d1 has JXL signature");
    let codestream: Vec<u8> = match sig {
        container::Signature::RawCodestream => VARDCT_D1_JXL[2..].to_vec(),
        container::Signature::Isobmff => container::extract_codestream(VARDCT_D1_JXL)
            .unwrap()
            .to_vec(),
    };

    let mut br = BitReader::new(&codestream);
    eprintln!("[r17-trace] === PRE-FRAME ===");
    eprintln!("[r17-trace] codestream = {} bytes", codestream.len());

    // SizeHeader → ImageMetadata → byte-align → FrameHeader → TOC.
    let size = SizeHeaderFdis::read(&mut br).expect("SizeHeader");
    eprintln!(
        "[r17-trace] after SizeHeader      = {} bits ({}x{})",
        br.bits_read(),
        size.width,
        size.height
    );
    let metadata = ImageMetadataFdis::read(&mut br).expect("ImageMetadata");
    eprintln!(
        "[r17-trace] after ImageMetadata   = {} bits (xyb_encoded={}, want_icc={})",
        br.bits_read(),
        metadata.xyb_encoded,
        metadata.colour_encoding.want_icc
    );
    if metadata.colour_encoding.want_icc {
        // d1 doesn't have ICC — bail rather than complicate the trace.
        return;
    }
    br.pu0().expect("byte-align");
    eprintln!(
        "[r17-trace] after byte-align      = {} bits",
        br.bits_read()
    );

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
    eprintln!(
        "[r17-trace] after FrameHeader     = {} bits (encoding={:?}, group_dim={})",
        br.bits_read(),
        fh.encoding,
        fh.group_dim()
    );
    assert_eq!(fh.encoding, Encoding::VarDct, "d1 must be VarDCT");

    let toc = Toc::read(&mut br, &fh).expect("TOC");
    eprintln!(
        "[r17-trace] after TOC             = {} bits (entries={:?})",
        br.bits_read(),
        toc.entries
    );

    // d1 is single-TOC: 1 entry containing all sections concatenated.
    assert_eq!(toc.entries.len(), 1);
    assert_eq!(fh.num_groups(), 1);
    assert_eq!(fh.num_lf_groups(), 1);
    assert_eq!(fh.passes.num_passes, 1);

    let frame_data_start = br.bytes_consumed();
    let frame_bytes = &br.data()[frame_data_start..];
    let lf_global_bytes = &frame_bytes[0..toc.entries[0] as usize];

    // === LfGlobal sub-bundles ===
    let mut shared_br = BitReader::new_section(lf_global_bytes);
    eprintln!("[r17-trace] === LF GLOBAL ===");

    let lf_dequant = LfChannelDequantization::read(&mut shared_br).expect("LfChannelDequant");
    eprintln!(
        "[r17-trace]   after LfChannelDequant   = {} (default={})",
        shared_br.bits_read(),
        lf_dequant.all_default
    );
    let quantizer = Quantizer::read(&mut shared_br).expect("Quantizer");
    eprintln!(
        "[r17-trace]   after Quantizer          = {} (global_scale={}, quant_lf={})",
        shared_br.bits_read(),
        quantizer.global_scale,
        quantizer.quant_lf
    );
    let bp = shared_br.bits_read();
    let hbc = HfBlockContext::read(&mut shared_br).expect("HfBlockContext");
    eprintln!(
        "[r17-trace]   after HfBlockContext     = {} (consumed {} bits, used_default={}, nb_block_ctx={})",
        shared_br.bits_read(),
        shared_br.bits_read() - bp,
        hbc.used_default,
        hbc.nb_block_ctx
    );
    let cfl = LfChannelCorrelation::read(&mut shared_br).expect("LfChannelCorrelation");
    eprintln!(
        "[r17-trace]   after LfChannelCorr      = {} (default={})",
        shared_br.bits_read(),
        cfl.all_default
    );
    let bp = shared_br.bits_read();
    let global_modular =
        GlobalModular::read(&mut shared_br, &fh, &metadata).expect("GlobalModular");
    eprintln!(
        "[r17-trace]   after GlobalModular      = {} (consumed {} bits, fully_decoded={}, tree_present={})",
        shared_br.bits_read(),
        shared_br.bits_read() - bp,
        global_modular.fully_decoded,
        global_modular.global_tree_present
    );
    eprintln!(
        "[r17-trace] LF GLOBAL END         = {} bits (cjxl trace says 1026)",
        shared_br.bits_read()
    );

    let lf_global = LfGlobal {
        lf_dequant,
        quantizer: Some(quantizer),
        hf_block_context: Some(hbc),
        lf_channel_correlation: Some(cfl),
        global_modular,
    };

    // === LfGroup sub-bundles (manual breakdown, not via LfGroup::read) ===
    let lf_w = fh.width.min(fh.group_dim() * 8);
    let lf_h = fh.height.min(fh.group_dim() * 8);
    eprintln!("[r17-trace] === LF GROUP === (lf_dim={}x{})", lf_w, lf_h);

    // LfCoefficients ModularHeader breakdown (without consuming the
    // section — re-create a separate reader at bit 1026). The earlier
    // `shared_br` is no longer used; we drop it at end-of-scope below.
    let _ = shared_br;
    let mut probe_br = BitReader::new_section(lf_global_bytes);
    probe_br.advance_bits(1026).unwrap();
    let _ = probe_br.read_bits(2).unwrap(); // extra_precision
    let iugt = probe_br.read_bool().unwrap();
    let wp = WpHeader::read(&mut probe_br).unwrap();
    let nbt = probe_br
        .read_u32([
            U32Dist::Val(0),
            U32Dist::Val(1),
            U32Dist::BitsOffset(4, 2),
            U32Dist::BitsOffset(8, 18),
        ])
        .unwrap();
    eprintln!(
        "[r17-trace]   LfCoeff ModularHeader: iugt={} default_wp={} nb_transforms={} (consumed {} bits)",
        iugt,
        wp.default_wp,
        nbt,
        probe_br.bits_read() - 1026
    );

    // Run LfCoefficients::read end-to-end on a fresh reader.
    let mut shared_br = BitReader::new_section(lf_global_bytes);
    shared_br.advance_bits(1026).unwrap();
    let bp = shared_br.bits_read();
    let lfc = LfCoefficients::read(&mut shared_br, &fh, &lf_global, lf_w, lf_h, 0)
        .expect("LfCoefficients should not error (just over-consume)");
    eprintln!(
        "[r17-trace]   after LfCoefficients     = {} (consumed {} bits, extra_prec={}, dims=[{}x{},{}x{},{}x{}])",
        shared_br.bits_read(),
        shared_br.bits_read() - bp,
        lfc.extra_precision,
        lfc.lf_quant_widths[0],
        lfc.lf_quant_heights[0],
        lfc.lf_quant_widths[1],
        lfc.lf_quant_heights[1],
        lfc.lf_quant_widths[2],
        lfc.lf_quant_heights[2],
    );
    // Sanity-check the decoded LF samples (these LOOK plausible — see
    // the bisect doc — but the bit count consumed is wrong).
    for (c, ch) in lfc.lf_quant.iter().enumerate() {
        let mn = ch.iter().min().copied().unwrap_or(0);
        let mx = ch.iter().max().copied().unwrap_or(0);
        let first8: Vec<i32> = ch.iter().take(8).copied().collect();
        eprintln!(
            "[r17-trace]     LfCoeff[ch{}] (n={}): min={} max={} first8={:?}",
            c,
            ch.len(),
            mn,
            mx,
            first8
        );
    }

    // HfMetadata::read — surfaces the round-16 SqueezeParam.begin_c=39
    // symptom because LfCoefficients over-consumed.
    eprintln!(
        "[r17-trace]   HfMetadata starts at      = {} bits",
        shared_br.bits_read()
    );
    let bp_hfm = shared_br.bits_read();
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
        Ok(_) => {
            eprintln!(
                "[r17-trace]   HfMetadata OK            = {} (consumed {} bits)",
                shared_br.bits_read(),
                shared_br.bits_read() - bp_hfm
            );
        }
        Err(e) => {
            eprintln!(
                "[r17-trace]   HfMetadata ERR @ bit {} (consumed {} bits) — {:?}",
                shared_br.bits_read(),
                shared_br.bits_read() - bp_hfm,
                e
            );
        }
    }

    eprintln!(
        "[r17-trace] === SUMMARY ===\n\
         [r17-trace]   cjxl trace:                 DC_GLOBAL_END=1026  DC_GROUP_END=12754  (total LfGroup = 11728 bits)\n\
         [r17-trace]   our decoder LfCoefficients alone = 11995 bits → 267+ bits over-consumed before HfMetadata\n\
         [r17-trace]   r18 candidate: see crates/oxideav-jpegxl/round17-d1-bisect.md"
    );
}
