//! Diagnostic trace for the cjxl 8x8 grey lossless fixture.
//!
//! Round 4 (this commit): the fixture's MA-tree T sub-stream prelude
//! parses correctly (1 cluster, simple-prefix code over a 115-symbol
//! alphabet emitting symbols {8, 14, 113}, HybridUintConfig with
//! split=16/msb=1/lsb=2). With this code, the prefix decode of token
//! 113 (a length-2 code, second-most-common in the stream) hybrid-
//! expands to ~552965 — far outside any property index for a single
//! Grey channel (max 15 per FDIS Table D.2).
//!
//! The decode failure at "JXL MA tree: property 552964 too large"
//! happens after 3 successful (prop=7 left, value=4) decision-node
//! reads: every fourth tree iteration's T[1] read pulls a token-113
//! from the prefix code, which the spec's ReadUint formula
//! (`n = split_exp + ((token-split) >> (msb+lsb))`) blows up to
//! ~552k. This rules out the obvious bugs (HybridUintConfig field
//! order, prefix code length assignment for NSYM=3 — both
//! interpretations gave the same problem; see typo memo #5 + RFC
//! 7932 §3.4 cross-check) but does not isolate the actual divergence.
//!
//! Trace output (run with `cargo test --offline -j 4 --test
//! cjxl_grey_8x8_trace -- --nocapture`) is the round-5 starting
//! point. The test asserts NOTHING; it just prints the prelude bit
//! positions, the prefix code mapping, and the first 30 MA-tree
//! decode iterations so a follow-up agent can bisect against a
//! known-good decoder reference once one is found.

use oxideav_jpegxl::ans::cluster::{num_clusters, read_clustering};
use oxideav_jpegxl::ans::hybrid_config::HybridUintConfig;
use oxideav_jpegxl::ans::prefix::read_prefix_code;
use oxideav_jpegxl::bitreader::{BitReader, U32Dist};
use oxideav_jpegxl::container;
use oxideav_jpegxl::frame_header::{FrameDecodeParams, FrameHeader};
use oxideav_jpegxl::lf_global::LfChannelDequantization;
use oxideav_jpegxl::metadata_fdis::{ImageMetadataFdis, SizeHeaderFdis};
use oxideav_jpegxl::toc::Toc;

const FIXTURE: &[u8] = include_bytes!("fixtures/grey_8x8_lossless.jxl");

fn raw_to_bytes(raw: u32, nbits: u32) -> Vec<u8> {
    // LSB-first packing: bit 0 of raw goes to bit 0 of byte 0.
    let nbytes = nbits.div_ceil(8) as usize;
    let mut bytes = vec![0u8; nbytes.max(1)];
    for i in 0..nbits {
        let bit = (raw >> i) & 1;
        bytes[(i / 8) as usize] |= (bit as u8) << (i % 8);
    }
    bytes
}

#[test]
fn trace_prelude_step_by_step() {
    let sig = container::detect(FIXTURE).expect("signature");
    let codestream: &[u8] = match sig {
        container::Signature::RawCodestream => &FIXTURE[2..],
        _ => panic!("not raw codestream"),
    };
    let mut br = BitReader::new(codestream);
    let size = SizeHeaderFdis::read(&mut br).unwrap();
    eprintln!("after size: bits_read={}", br.bits_read());
    let metadata = ImageMetadataFdis::read(&mut br).unwrap();
    eprintln!(
        "metadata: all_default={} extra_fields={} num_extra_channels={} xyb_encoded={} default_transform={} cw_mask={}",
        metadata.all_default, metadata.extra_fields, metadata.num_extra_channels, metadata.xyb_encoded, metadata.default_transform, metadata.cw_mask
    );
    eprintln!(
        "after metadata: bits_read={} {}x{} bpp={}",
        br.bits_read(),
        size.width,
        size.height,
        metadata.bit_depth.bits_per_sample
    );
    br.pu0().unwrap();
    eprintln!("after pu0: bits_read={}", br.bits_read());

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
    eprintln!(
        "after fh: bits_read={} encoding={:?} {}x{} flags={:#x}",
        br.bits_read(),
        fh.encoding,
        fh.width,
        fh.height,
        fh.flags
    );
    let toc = Toc::read(&mut br, &fh).unwrap();
    eprintln!(
        "after toc: bits_read={} entries={:?}",
        br.bits_read(),
        &toc.entries
    );

    let lf_dequant = LfChannelDequantization::read(&mut br).unwrap();
    eprintln!(
        "after lf_dequant: bits_read={} all_default={}",
        br.bits_read(),
        lf_dequant.all_default
    );

    let global_use_tree = br.read_bool().unwrap();
    eprintln!(
        "use_global_tree={} bits_read={}",
        global_use_tree,
        br.bits_read()
    );

    // EntropyStream prelude for the MA tree's 6 T-distributions.
    let lz77_enabled = br.read_bit().unwrap() == 1;
    eprintln!("MA tree lz77_enabled={}", lz77_enabled);
    let lz_len_conf = if lz77_enabled {
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
        HybridUintConfig::read(&mut br, 8).unwrap()
    } else {
        HybridUintConfig {
            split_exponent: 8,
            msb_in_token: 0,
            lsb_in_token: 0,
            split: 256,
        }
    };
    let _ = lz_len_conf;

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
        "MA tree cluster_map={:?} n_clusters={} bits_read={}",
        cluster_map,
        n_clusters,
        br.bits_read()
    );

    let use_prefix_code = br.read_bit().unwrap() == 1;
    let log_alphabet_size = if use_prefix_code {
        5 + br.read_bits(2).unwrap()
    } else {
        15
    };
    eprintln!(
        "MA tree use_prefix_code={} log_alphabet_size={} bits_read={}",
        use_prefix_code,
        log_alphabet_size,
        br.bits_read()
    );

    let mut configs = Vec::with_capacity(n_clusters);
    for i in 0..n_clusters {
        let c = HybridUintConfig::read(&mut br, log_alphabet_size).unwrap();
        eprintln!(
            "  cluster {}: HybridUintConfig split_exp={} msb={} lsb={} split={}",
            i, c.split_exponent, c.msb_in_token, c.lsb_in_token, c.split
        );
        configs.push(c);
    }
    eprintln!("after configs: bits_read={}", br.bits_read());

    if use_prefix_code {
        let mut counts = Vec::with_capacity(n_clusters);
        for i in 0..n_clusters {
            let count = if br.read_bit().unwrap() == 0 {
                1u32
            } else {
                let n = br.read_bits(4).unwrap();
                1 + (1 << n) + br.read_bits(n).unwrap()
            };
            eprintln!("  cluster {} symbol count={}", i, count);
            counts.push(count);
        }
        let mut codes = Vec::new();
        for (i, &count) in counts.iter().enumerate() {
            let bits_before = br.bits_read();
            let code = read_prefix_code(&mut br, count).unwrap();
            let bits_after = br.bits_read();
            eprintln!(
                "  cluster {} prefix code: alphabet_size={} (consumed {} bits, total bits_read={})",
                i,
                code.alphabet_size,
                bits_after - bits_before,
                bits_after
            );
            codes.push(code);
        }
        // Enumerate the 1..=4-bit codes the prefix table actually maps.
        for (i, code) in codes.iter().enumerate() {
            eprintln!("  cluster {} prefix code mapping (short codes):", i);
            for nbits in 1..=4 {
                for raw in 0..(1u32 << nbits) {
                    let bytes = raw_to_bytes(raw, nbits);
                    let mut br_inner = BitReader::new(&bytes);
                    if let Ok(sym) = code.decode(&mut br_inner) {
                        let consumed = br_inner.bits_read() as u32;
                        if consumed == nbits {
                            eprintln!(
                                "    bits {:#0w$b} ({} bits) → symbol {}",
                                raw,
                                nbits,
                                sym,
                                w = (nbits + 2) as usize
                            );
                        }
                    }
                }
            }
        }
        eprintln!("after prefix codes: bits_read={}", br.bits_read());

        // Simulate MA tree decode iterations (Listing D.9). Stops at 30
        // iterations or first error — whichever comes first.
        let cfg = configs[0];
        let code = &codes[0];
        let read_uint = |br: &mut BitReader<'_>| -> Result<u32, String> {
            let token = code.decode(br).map_err(|e| format!("{}", e))?;
            if token < cfg.split {
                Ok(token)
            } else {
                cfg.read_uint(br, token).map_err(|e| format!("{}", e))
            }
        };
        let mut nodes_left = 1u32;
        let mut node_no = 0u32;
        while nodes_left > 0 && node_no < 30 {
            let bits_before = br.bits_read();
            let prop_plus_1 = match read_uint(&mut br) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("node {}: prop read error: {}", node_no, e);
                    break;
                }
            };
            let property = prop_plus_1 as i64 - 1;
            if property < 0 {
                let pred = read_uint(&mut br).unwrap_or(u32::MAX);
                let uoffset = read_uint(&mut br).unwrap_or(u32::MAX);
                let mul_log = read_uint(&mut br).unwrap_or(u32::MAX);
                let mul_bits = read_uint(&mut br).unwrap_or(u32::MAX);
                let bits_after = br.bits_read();
                eprintln!(
                    "  node {}: LEAF pred={} uoff={} mul_log={} mul_bits={} ({} bits, total={})",
                    node_no,
                    pred,
                    uoffset,
                    mul_log,
                    mul_bits,
                    bits_after - bits_before,
                    bits_after
                );
                nodes_left -= 1;
            } else {
                let uvalue = read_uint(&mut br).unwrap_or(u32::MAX);
                let bits_after = br.bits_read();
                eprintln!(
                    "  node {}: DECISION prop={} uvalue={} ({} bits, total={})",
                    node_no,
                    property,
                    uvalue,
                    bits_after - bits_before,
                    bits_after
                );
                nodes_left += 1;
            }
            node_no += 1;
        }
    }
}
