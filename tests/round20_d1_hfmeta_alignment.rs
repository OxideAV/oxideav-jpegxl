//! Round-20 alignment scan: try parsing HfMetadata's first few fields
//! at every offset in `[lfc_end - 270, lfc_end + 10)` to find the bit
//! position where the fields look sensible (iugt=true default_wp=true
//! nb_transforms=0).
//!
//! Hypothesis: our LfCoefficients consumed N bits but the *correct*
//! cjxl-encoded LfCoefficients consumed `N - delta` bits for some
//! `delta > 0`. By scanning offsets we can identify the cleanest
//! starting point for HfMetadata, and `delta` will tell us how many
//! extra bits LfCoefficients over-read.

use oxideav_jpegxl::bitreader::{BitReader, U32Dist};
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
fn d1_hfmeta_alignment_scan_round_20() {
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
    let _lfc = LfCoefficients::read(&mut shared_br, &fh, &lf_global, lf_w, lf_h, 0).unwrap();
    let lfc_end = shared_br.bits_read();

    eprintln!("[r20-scan] LfCoefficients ends at bit {lfc_end} (cjxl LfGroup ends at 13780)");
    eprintln!(
        "[r20-scan] Scanning HfMetadata candidate start positions in [lfc_end-270, lfc_end+5):"
    );
    eprintln!("[r20-scan]   ideal: nb_blocks_minus_1 < 1024, iugt=true, default_wp=true, nb_transforms in {{0,1}}");

    let scan_lo = lfc_end.saturating_sub(270);
    let scan_hi = lfc_end + 5;

    let mut best_offsets: Vec<(i64, usize, u32, bool, bool, u32)> = Vec::new();

    for trial_start in scan_lo..scan_hi {
        let mut br = BitReader::new_section(lf_global_bytes);
        if br.advance_bits(trial_start as u32).is_err() {
            continue;
        }
        // nb_blocks_minus_1 = u(10) for 256x256 LfGroup.
        let Ok(nbm1) = br.read_bits(10) else { continue };
        if nbm1 + 1 > 1024 {
            continue;
        }
        let Ok(iugt) = br.read_bool() else { continue };
        // WPHeader: 1 bit if default_wp=1, else 51 more bits.
        let Ok(default_wp) = br.read_bool() else {
            continue;
        };
        if !default_wp {
            // Skip WPHeader's 51 bits.
            if br.read_bits(51).is_err() {
                continue;
            }
        }
        let Ok(nb_transforms) = br.read_u32([
            U32Dist::Val(0),
            U32Dist::Val(1),
            U32Dist::BitsOffset(4, 2),
            U32Dist::BitsOffset(8, 18),
        ]) else {
            continue;
        };
        if nb_transforms > 4 {
            continue;
        }
        // Score: prefer default_wp=true and iugt=true and small nb_transforms.
        let mut score = 0i32;
        if default_wp {
            score += 4;
        }
        if iugt {
            score += 2;
        }
        if nb_transforms == 0 {
            score += 4;
        } else if nb_transforms == 1 {
            score += 2;
        }
        if nbm1 < 100 {
            score += 1;
        } else if nbm1 < 600 {
            score += 2;
        } else if nbm1 < 1024 {
            score += 1;
        }
        let delta = trial_start as i64 - lfc_end as i64;
        eprintln!(
            "[r20-scan]   delta={:+5} bit={:5} nbm1={:4} iugt={:5} dwp={:5} nbt={} score={}",
            delta, trial_start, nbm1, iugt, default_wp, nb_transforms, score
        );
        if score >= 8 {
            best_offsets.push((delta, trial_start, nbm1, iugt, default_wp, nb_transforms));
        }
    }

    eprintln!("[r20-scan] === high-score candidates (score >= 8) ===");
    for (delta, bp, nbm1, iugt, dwp, nbt) in &best_offsets {
        eprintln!(
            "[r20-scan]   delta={:+5}  bit={}  nb_blocks_minus_1={}  iugt={}  default_wp={}  nb_transforms={}",
            delta, bp, nbm1, iugt, dwp, nbt
        );
    }
    if best_offsets.is_empty() {
        eprintln!("[r20-scan] No high-score candidates found in scan range — HfMetadata's prefix may be more exotic, or LfCoeff drift is outside [-270, +5).");
    }
}
