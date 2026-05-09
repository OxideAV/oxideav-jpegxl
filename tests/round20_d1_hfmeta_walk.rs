//! Round-20 deep walk: replicate `HfMetadata::read` byte-for-byte at
//! file-section bit 13021 with per-field bit accounting, to identify
//! where exactly the 233-bit prefix consumed by our impl diverges from
//! a sensible cjxl-encoded HfMetadata.
//!
//! Per round-20 (`round20-d1-dc_group-recount` test), `bits_consumed`
//! in the cjxl trace is **section-local**, so DC_GROUP is 12754 bits
//! (not 11728) and HfMetadata's slot is `[13021, 13780)` = 759 bits.
//! The "JXL Modular Squeeze: end 40 >= channel count 4" error fires
//! 233 bits into HfMetadata's slot. This walk parses the same field
//! sequence with no abort to surface every TransformInfo decision
//! point.

use oxideav_jpegxl::bitreader::{BitReader, U32Dist};
use oxideav_jpegxl::container;
use oxideav_jpegxl::frame_header::{Encoding, FrameDecodeParams, FrameHeader};
use oxideav_jpegxl::global_modular::GlobalModular;
use oxideav_jpegxl::lf_global::{
    HfBlockContext, LfChannelCorrelation, LfChannelDequantization, LfGlobal, Quantizer,
};
use oxideav_jpegxl::lf_group::LfCoefficients;
use oxideav_jpegxl::metadata_fdis::{ImageMetadataFdis, SizeHeaderFdis};
use oxideav_jpegxl::modular_fdis::WpHeader;
use oxideav_jpegxl::toc::Toc;

const VARDCT_D1_JXL: &[u8] = include_bytes!("fixtures/vardct_256x256_d1.jxl");

#[test]
fn d1_hfmeta_field_walk_round_20() {
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

    // === LfGlobal + LfCoefficients to position the cursor at HfMeta start.
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
    let hfm_start = shared_br.bits_read();
    eprintln!("[r20-walk] HfMetadata starts at bit {hfm_start} (= cjxl-relative HfMeta-bit 0)");

    // Read nb_blocks_minus_1 = u(10) for d1 (256x256 LfGroup → 1024 max blocks).
    let nb_blocks_minus_1 = shared_br.read_bits(10).unwrap();
    eprintln!(
        "[r20-walk] @bit {} (HfMeta+{}): nb_blocks_minus_1 = u(10) = {} → nb_blocks={}",
        shared_br.bits_read(),
        shared_br.bits_read() - hfm_start,
        nb_blocks_minus_1,
        nb_blocks_minus_1 + 1
    );

    // inner_use_global_tree = u(1)
    let iugt = shared_br.read_bool().unwrap();
    eprintln!(
        "[r20-walk] @bit {} (HfMeta+{}): inner_use_global_tree = u(1) = {}",
        shared_br.bits_read(),
        shared_br.bits_read() - hfm_start,
        iugt
    );

    // WPHeader
    let pre_wp = shared_br.bits_read();
    let wp = WpHeader::read(&mut shared_br).unwrap();
    let wp_bits = shared_br.bits_read() - pre_wp;
    eprintln!(
        "[r20-walk] @bit {} (HfMeta+{}): WPHeader (default_wp={}) consumed {} bits",
        shared_br.bits_read(),
        shared_br.bits_read() - hfm_start,
        wp.default_wp,
        wp_bits
    );

    // nb_transforms = U32([Val(0), Val(1), BitsOffset(4,2), BitsOffset(8,18)])
    let pre_nbt = shared_br.bits_read();
    let nb_transforms = shared_br
        .read_u32([
            U32Dist::Val(0),
            U32Dist::Val(1),
            U32Dist::BitsOffset(4, 2),
            U32Dist::BitsOffset(8, 18),
        ])
        .unwrap();
    let nbt_bits = shared_br.bits_read() - pre_nbt;
    eprintln!(
        "[r20-walk] @bit {} (HfMeta+{}): nb_transforms = U32 = {} ({} bits)",
        shared_br.bits_read(),
        shared_br.bits_read() - hfm_start,
        nb_transforms,
        nbt_bits
    );

    if nb_transforms == 0 {
        eprintln!("[r20-walk] No transforms — HfMetadata channel list is the 4-channel baseline.");
        return;
    }

    // Per-transform walk.
    for ti in 0..nb_transforms.min(8) {
        let pre_t = shared_br.bits_read();
        let tr_raw = shared_br.read_bits(2).unwrap();
        let tr_name = match tr_raw {
            0 => "RCT",
            1 => "Palette",
            2 => "Squeeze",
            _ => "Reserved",
        };
        eprintln!(
            "[r20-walk] @bit {} (HfMeta+{}): transform[{}].kind = u(2) = {} ({})",
            shared_br.bits_read(),
            shared_br.bits_read() - hfm_start,
            ti,
            tr_raw,
            tr_name
        );

        match tr_raw {
            0 => {
                // RCT: begin_c, rct_type
                let bc = shared_br
                    .read_u32([
                        U32Dist::Bits(3),
                        U32Dist::BitsOffset(6, 8),
                        U32Dist::BitsOffset(10, 72),
                        U32Dist::BitsOffset(13, 1096),
                    ])
                    .unwrap();
                let rt = shared_br
                    .read_u32([
                        U32Dist::Val(6),
                        U32Dist::Bits(2),
                        U32Dist::BitsOffset(4, 2),
                        U32Dist::BitsOffset(6, 10),
                    ])
                    .unwrap();
                eprintln!(
                    "[r20-walk]   RCT begin_c={} rct_type={} (consumed {} bits)",
                    bc,
                    rt,
                    shared_br.bits_read() - pre_t
                );
            }
            1 => {
                // Palette: begin_c, num_c, nb_colours, nb_deltas, d_pred
                let bc = shared_br
                    .read_u32([
                        U32Dist::Bits(3),
                        U32Dist::BitsOffset(6, 8),
                        U32Dist::BitsOffset(10, 72),
                        U32Dist::BitsOffset(13, 1096),
                    ])
                    .unwrap();
                let nc = shared_br
                    .read_u32([
                        U32Dist::Val(1),
                        U32Dist::Val(3),
                        U32Dist::Val(4),
                        U32Dist::BitsOffset(13, 1),
                    ])
                    .unwrap();
                let ncol = shared_br
                    .read_u32([
                        U32Dist::BitsOffset(8, 0),
                        U32Dist::BitsOffset(10, 256),
                        U32Dist::BitsOffset(12, 1280),
                        U32Dist::BitsOffset(16, 5376),
                    ])
                    .unwrap();
                let nd = shared_br
                    .read_u32([
                        U32Dist::Val(0),
                        U32Dist::BitsOffset(8, 1),
                        U32Dist::BitsOffset(10, 257),
                        U32Dist::BitsOffset(16, 1281),
                    ])
                    .unwrap();
                let dp = shared_br.read_bits(4).unwrap();
                eprintln!(
                    "[r20-walk]   Palette begin_c={} num_c={} nb_col={} nb_d={} d_pred={} ({} bits)",
                    bc,
                    nc,
                    ncol,
                    nd,
                    dp,
                    shared_br.bits_read() - pre_t
                );
            }
            2 => {
                // Squeeze: num_sq, then num_sq × SqueezeParam
                let num_sq = shared_br
                    .read_u32([
                        U32Dist::Val(0),
                        U32Dist::BitsOffset(4, 1),
                        U32Dist::BitsOffset(6, 9),
                        U32Dist::BitsOffset(8, 41),
                    ])
                    .unwrap();
                eprintln!(
                    "[r20-walk]   Squeeze num_sq = {} (transform now {} bits in)",
                    num_sq,
                    shared_br.bits_read() - pre_t
                );
                for sq in 0..num_sq.min(8) {
                    let pre_sq = shared_br.bits_read();
                    let horizontal = shared_br.read_bool().unwrap();
                    let in_place = shared_br.read_bool().unwrap();
                    let bc = shared_br
                        .read_u32([
                            U32Dist::Bits(3),
                            U32Dist::BitsOffset(6, 8),
                            U32Dist::BitsOffset(10, 72),
                            U32Dist::BitsOffset(13, 1096),
                        ])
                        .unwrap();
                    let nc = shared_br
                        .read_u32([
                            U32Dist::Val(1),
                            U32Dist::Val(2),
                            U32Dist::Val(3),
                            U32Dist::BitsOffset(4, 4),
                        ])
                        .unwrap();
                    eprintln!(
                        "[r20-walk]     SP[{}] horizontal={} in_place={} begin_c={} num_c={} ({} bits)",
                        sq,
                        horizontal,
                        in_place,
                        bc,
                        nc,
                        shared_br.bits_read() - pre_sq
                    );
                }
            }
            _ => {
                eprintln!("[r20-walk]   reserved transform kind, stop");
                return;
            }
        }
        eprintln!(
            "[r20-walk] @bit {} (HfMeta+{}): transform[{}] total {} bits",
            shared_br.bits_read(),
            shared_br.bits_read() - hfm_start,
            ti,
            shared_br.bits_read() - pre_t
        );
        // Stop early if we've blown past the budget.
        if shared_br.bits_read() - hfm_start > 759 {
            eprintln!(
                "[r20-walk] WARNING: walked past the 759-bit HfMeta budget; bit {} (= +{})",
                shared_br.bits_read(),
                shared_br.bits_read() - hfm_start
            );
            return;
        }
    }
}
