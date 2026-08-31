//! Round 393 — §C.5 multi-LfGroup VarDCT framing on the
//! `large-3072x2048-multigroup` fixture (docs commit 7024774): a
//! single-pass Regular VarDCT frame whose DC spans **2 LF groups**
//! (2×1) and whose AC spans **96 groups** (12×8), with a **permuted
//! TOC** (large-image progressive group ordering) and the long-form
//! SizeHeader.
//!
//! Structural pieces landed and pinned here:
//!
//! * §C.3.2 TOC permutation over a full D.3 sub-stream — cjxl's
//!   large-image TOC permutations ship with **LZ77 enabled**, which
//!   the previous ANS-only hand-rolled prelude rejected. The permuted
//!   100-entry TOC (1 LfGlobal + 2 LfGroup + 1 HfGlobal + 96
//!   PassGroup) now decodes and the §C.3.3 byte offsets follow
//!   `group_offsets` (NOT a running entry sum — that misframes every
//!   section of a permuted stream).
//! * §C.5 LfGroup tiling — both LfGroups parse (LfCoefficients +
//!   HfMetadata) and assemble into frame-level canvases at their
//!   `group_dim × 8` raster offsets.
//!
//! The remaining boundary is pinned precisely: the stream signals
//! `used_orders = 0x5F` (§C.7.1 custom coefficient orders), whose
//! permutation sub-stream reading (C.3.2 `end` endpoint-vs-count +
//! `D[prev_elem]` context) is underdetermined by the FDIS text and not
//! covered by any staged trace — a docs-gap filed this round. Until it
//! is pinned, the decode surfaces a loud error from the §C.7.1 range
//! guards (never a silent misparse). When the trace lands, replace
//! `full_decode_stops_at_c71_custom_orders` with the pixel-MAD ratchet
//! against `expected.png`.
//!
//! Clean-room: `input.jxl` / `expected.png` are black-box validator
//! artefacts staged under `docs/image/jpegxl/fixtures/`; behaviour is
//! derived from the ISO/IEC 18181-1 FDIS. No external implementation
//! source is consulted.

use oxideav_jpegxl::bitreader::BitReader;
use oxideav_jpegxl::container;
use oxideav_jpegxl::frame_header::{Encoding, FrameDecodeParams, FrameHeader, RfEdition};
use oxideav_jpegxl::metadata_fdis::{ImageMetadataFdis, SizeHeaderFdis};
use oxideav_jpegxl::toc::Toc;

const JXL: &[u8] = include_bytes!("fixtures/large_3072x2048_multigroup.jxl");

struct Parsed {
    fh: FrameHeader,
    metadata: ImageMetadataFdis,
    toc: Toc,
    frame_bytes_start: usize,
    codestream: Vec<u8>,
}

fn parse_to_toc() -> Parsed {
    let sig = container::detect(JXL).expect("signature");
    let codestream: Vec<u8> = match sig {
        container::Signature::RawCodestream => JXL[2..].to_vec(),
        container::Signature::Isobmff => container::extract_codestream(JXL).unwrap().to_vec(),
    };
    let mut br = BitReader::new(&codestream);
    let size = SizeHeaderFdis::read(&mut br).expect("SizeHeader (long form)");
    assert_eq!((size.width, size.height), (3072, 2048));
    let metadata = ImageMetadataFdis::read(&mut br).expect("ImageMetadata");
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
    let fh =
        FrameHeader::read_with_edition(&mut br, &fh_params, RfEdition::V2024).expect("FrameHeader");
    let toc = Toc::read(&mut br, &fh).expect("permuted TOC decodes");
    let frame_bytes_start = br.bytes_consumed();
    Parsed {
        fh,
        metadata,
        toc,
        frame_bytes_start,
        codestream,
    }
}

/// The §C.3 framing pins from the fixture notes: 96 AC groups (12×8),
/// 2 LF groups (2×1), single pass, permuted 100-entry TOC.
#[test]
fn framing_matches_fixture_notes() {
    let p = parse_to_toc();
    assert_eq!(p.fh.encoding, Encoding::VarDct);
    assert_eq!(p.fh.num_groups(), 96, "12×8 AC groups");
    assert_eq!(p.fh.num_lf_groups(), 2, "2×1 LF groups");
    assert_eq!(p.fh.passes.num_passes, 1);
    assert!(p.toc.permuted, "fixture notes: TOC is permuted");
    assert_eq!(
        p.toc.entries.len(),
        100,
        "1 LfGlobal + 2 LfGroup + 1 HfGlobal + 96 PassGroup"
    );
    // §C.3.3: with a permutation the wire offsets are NOT a running
    // sum of the canonical entry sizes.
    let mut running = 0u64;
    let mut monotonic = true;
    for (i, &e) in p.toc.entries.iter().enumerate() {
        if p.toc.group_offsets[i] != running {
            monotonic = false;
        }
        running += e as u64;
    }
    assert!(
        !monotonic,
        "a permuted TOC must reorder section offsets (progressive group order)"
    );
}

/// Both LfGroups parse through LfGlobal → LfCoefficients + HfMetadata
/// at their §C.3.3 permuted byte offsets, with the §C.5 edge-tile
/// dimensions (2048 px + 1024 px columns).
#[test]
fn both_lf_groups_parse_at_permuted_offsets() {
    let p = parse_to_toc();
    let frame_bytes = &p.codestream[p.frame_bytes_start..];
    let range = |idx: usize| -> &[u8] {
        let start = p.toc.group_offsets[idx] as usize;
        let len = p.toc.entries[idx] as usize;
        &frame_bytes[start..start + len]
    };
    let mut lf_br = BitReader::new_section(range(0));
    let lf_global = oxideav_jpegxl::lf_global::LfGlobal::read(&mut lf_br, &p.fh, &p.metadata)
        .expect("LfGlobal parses");
    for (lg, want_w) in [(0u32, 2048u32), (1, 1024)] {
        let mut br = BitReader::new_section(range(1 + lg as usize));
        let group =
            oxideav_jpegxl::lf_group::LfGroup::read(&mut br, &p.fh, &lf_global, &p.metadata, lg)
                .unwrap_or_else(|e| panic!("LfGroup {lg} parses: {e:?}"));
        assert_eq!(group.mlf_group.lf_group_width, want_w, "LfGroup {lg} width");
        assert_eq!(group.mlf_group.lf_group_height, 2048);
        let lf_coeff = group.lf_coeff.as_ref().expect("VarDCT LfCoefficients");
        assert_eq!(
            lf_coeff.lf_quant_widths,
            [want_w / 8; 3],
            "LF samples = blocks"
        );
        assert!(group.hf_meta.is_some(), "HfMetadata parses");
    }
}

/// Round 454: the §C.7.1 multi-preset boundary is CLOSED (the FDIS
/// "read `num_hf_presets` times" lead-in is superseded by 2024
/// §I.3.1's one-order-bundle-per-pass layout — see
/// `round437_custom_orders_boundary`), so this 2-preset stream now
/// decodes end to end. Pixel ratchet against the staged black-box
/// reference decode: measured at MAD 0.60 / 0.41 / 0.47 on landing —
/// the same sub-1/255 band as the other photo VarDCT fixtures.
#[test]
fn full_decode_within_pixel_ratchet() {
    use std::io::Cursor;
    let expected = include_bytes!("fixtures/large_3072x2048_multigroup_expected.png");
    let decoder = png::Decoder::new(Cursor::new(&expected[..]));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    assert_eq!(info.color_type, png::ColorType::Rgb);
    let (w, h) = (info.width as usize, info.height as usize);

    let frame = oxideav_jpegxl::decode_vardct_frame_from_codestream(JXL, None)
        .expect("round 454: the permuted-TOC 2-preset stream decodes end to end");
    assert_eq!(frame.planes.len(), 3);
    for c in 0..3usize {
        let plane = &frame.planes[c];
        let mut sum = 0u64;
        for y in 0..h {
            for x in 0..w {
                let d = plane.data[y * plane.stride + x].abs_diff(buf[(y * w + x) * 3 + c]);
                sum += d as u64;
            }
        }
        let mad = sum as f64 / (w * h) as f64;
        assert!(
            mad < 1.0,
            "channel {c}: MAD {mad:.3} exceeds the sub-1/255 photo band ratchet"
        );
    }
}
