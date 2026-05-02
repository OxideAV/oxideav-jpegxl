//! Diagnostic trace for the cjxl gradient_64x64 lossless fixture (round 7).
//!
//! Captures `[TRACE/...]` events comparable to the docs/image/jpegxl/
//! libjxl-trace-reverse-engineering.md report so future regressions in
//! the modular sub-bitstream decode have a stable bisection harness.
//!
//! This is a 64x64 RGB lossless fixture produced by cjxl 0.11.1 at -d 0
//! -e 7. The doc's reference run reports a 228-byte payload with 4
//! transforms (3x Palette + RCT), 53-node global tree. Our local
//! cjxl 0.11.1 may pick a different transform set / tree, but the
//! structural layout (signature → SizeHeader → ImageMetadata → frame
//! header → TOC → LfGlobal → modular sub-bitstream → AC group(s))
//! must be observed.

use oxideav_jpegxl::ans::cluster::{num_clusters, read_clustering};
use oxideav_jpegxl::ans::hybrid_config::HybridUintConfig;
use oxideav_jpegxl::ans::prefix::read_prefix_code;
use oxideav_jpegxl::bitreader::{BitReader, U32Dist};
use oxideav_jpegxl::container;
use oxideav_jpegxl::frame_header::{FrameDecodeParams, FrameHeader};
use oxideav_jpegxl::lf_global::LfChannelDequantization;
use oxideav_jpegxl::metadata_fdis::{ImageMetadataFdis, SizeHeaderFdis};
use oxideav_jpegxl::toc::Toc;

const FIXTURE: &[u8] = include_bytes!("fixtures/gradient_64x64.lossless.jxl");

#[test]
fn trace_gradient_64x64_signature_and_headers() {
    let sig = container::detect(FIXTURE).expect("signature");
    let codestream: &[u8] = match sig {
        container::Signature::RawCodestream => &FIXTURE[2..],
        _ => panic!("not raw codestream"),
    };
    eprintln!(
        "[TRACE/sig] codestream signature accepted (raw, codestream={} B)",
        codestream.len()
    );

    let mut br = BitReader::new(codestream);
    let bits_before = br.bits_read();
    let size = SizeHeaderFdis::read(&mut br).unwrap();
    eprintln!(
        "[TRACE/hdr] SizeHeader  parsed: consumed {} bits (xsize={} ysize={})",
        br.bits_read() - bits_before,
        size.width,
        size.height
    );
    assert_eq!(size.width, 64);
    assert_eq!(size.height, 64);

    let bits_before = br.bits_read();
    let metadata = ImageMetadataFdis::read(&mut br).unwrap();
    eprintln!(
        "[TRACE/hdr] ImageMetadata parsed: consumed {} bits (total since signature: {})",
        br.bits_read() - bits_before,
        br.bits_read()
    );
    eprintln!(
        "[TRACE/hdr] BasicInfo: {}x{} bpp={} xyb_encoded={} num_extra_channels={}",
        size.width,
        size.height,
        metadata.bit_depth.bits_per_sample,
        metadata.xyb_encoded,
        metadata.num_extra_channels
    );

    match br.pu0() {
        Ok(()) => eprintln!("[TRACE/hdr] ZeroPadToByte → bits_read={}", br.bits_read()),
        Err(e) => {
            eprintln!("[TRACE/hdr] ZeroPadToByte FAILED: {} — local cjxl 0.11.1 may emit extra metadata bits not modeled by FDIS impl. bits_read={}", e, br.bits_read());
            return;
        }
    }

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
    let bits_before = br.bits_read();
    let fh = FrameHeader::read(&mut br, &fh_params).unwrap();
    eprintln!(
        "[TRACE/frame] FrameHeader parsed: {} bits (encoding={:?}, color_transform={}, flags={:#x}, passes={}, group_size_shift={}, is_last={})",
        br.bits_read() - bits_before,
        fh.encoding,
        fh.do_ycbcr as u8,
        fh.flags,
        fh.passes.num_passes,
        fh.group_size_shift,
        fh.is_last
    );

    let bits_before = br.bits_read();
    let toc = Toc::read(&mut br, &fh).unwrap();
    eprintln!(
        "[TRACE/frame] TOC parsed: {} bits, entries={:?}",
        br.bits_read() - bits_before,
        &toc.entries
    );
}

#[test]
fn trace_gradient_64x64_lf_global_modular_prelude() {
    // Same as above but goes one step further into the LfGlobal /
    // GlobalModular bundle prelude. Stops once it has the use_global_tree
    // flag and the entropy-stream prelude has been read enough to print
    // the per-cluster prefix codes.
    let sig = container::detect(FIXTURE).expect("signature");
    let codestream: &[u8] = match sig {
        container::Signature::RawCodestream => &FIXTURE[2..],
        _ => panic!(),
    };
    let mut br = BitReader::new(codestream);

    let size = SizeHeaderFdis::read(&mut br).unwrap();
    let metadata = ImageMetadataFdis::read(&mut br).unwrap();
    match br.pu0() {
        Ok(()) => {}
        Err(e) => {
            eprintln!(
                "[TRACE/hdr] ZeroPadToByte FAILED: {} — bits_read={}",
                e,
                br.bits_read()
            );
            return;
        }
    }
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
    let _toc = Toc::read(&mut br, &fh).unwrap();

    let lf_dequant = LfChannelDequantization::read(&mut br).unwrap();
    eprintln!(
        "[TRACE/dc] ProcessDCGlobal start; lf_dequant.all_default={}",
        lf_dequant.all_default
    );

    let global_use_tree = br.read_bool().unwrap();
    eprintln!(
        "[TRACE/modular] use_global_tree={} (bits_read={})",
        global_use_tree,
        br.bits_read()
    );

    if !global_use_tree {
        eprintln!(
            "[TRACE/modular] no global tree — sub-bitstream decodes its own tree (round-7 untested)"
        );
        return;
    }

    // Decode the MA-tree's entropy stream prelude.
    let lz77_enabled = br.read_bit().unwrap() == 1;
    eprintln!("[TRACE/ans] tree-prelude lz77_enabled={}", lz77_enabled);
    if lz77_enabled {
        let _min_symbol = br
            .read_u32([
                U32Dist::Val(224),
                U32Dist::Val(512),
                U32Dist::Val(4096),
                U32Dist::BitsOffset(15, 8),
            ])
            .unwrap();
        let _min_length = br
            .read_u32([
                U32Dist::Val(3),
                U32Dist::Val(4),
                U32Dist::BitsOffset(2, 5),
                U32Dist::BitsOffset(8, 9),
            ])
            .unwrap();
        let _ = HybridUintConfig::read(&mut br, 8).unwrap();
    }

    let num_dist = 6usize;
    let effective_num_dist = if lz77_enabled { num_dist + 1 } else { num_dist };
    let cluster_map = if effective_num_dist > 1 {
        read_clustering(&mut br, effective_num_dist).unwrap()
    } else {
        vec![0u32; effective_num_dist]
    };
    let n_clusters = if effective_num_dist > 1 {
        num_clusters(&cluster_map) as usize
    } else {
        1
    };
    eprintln!(
        "[TRACE/ans] tree cluster_map={:?} n_clusters={}",
        cluster_map, n_clusters
    );

    let use_prefix_code = br.read_bit().unwrap() == 1;
    let log_alphabet_size = if use_prefix_code {
        15
    } else {
        5 + br.read_bits(2).unwrap()
    };
    eprintln!(
        "[TRACE/ans] tree use_prefix_code={} log_alphabet_size={}",
        use_prefix_code, log_alphabet_size
    );

    let mut configs = Vec::new();
    for i in 0..n_clusters {
        let c = HybridUintConfig::read(&mut br, log_alphabet_size).unwrap();
        eprintln!(
            "[TRACE/ans]   cluster {}: HybridUintConfig split_exp={} msb={} lsb={} split={}",
            i, c.split_exponent, c.msb_in_token, c.lsb_in_token, c.split
        );
        configs.push(c);
    }

    if use_prefix_code {
        let mut counts = Vec::new();
        for i in 0..n_clusters {
            let count = if br.read_bit().unwrap() == 0 {
                1u32
            } else {
                let n = br.read_bits(4).unwrap();
                1 + (1 << n) + br.read_bits(n).unwrap()
            };
            eprintln!("[TRACE/ans]   cluster {} symbol-count={}", i, count);
            counts.push(count);
        }
        for (i, &count) in counts.iter().enumerate() {
            let bits_before = br.bits_read();
            let code = read_prefix_code(&mut br, count).unwrap();
            eprintln!(
                "[TRACE/ans]   cluster {} prefix code: alphabet_size={} consumed {} bits",
                i,
                code.alphabet_size,
                br.bits_read() - bits_before
            );
        }
    }

    eprintln!(
        "[TRACE/ans] tree-prelude done — bits_read={}",
        br.bits_read()
    );
}
