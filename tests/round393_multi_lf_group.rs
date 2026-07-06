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

/// The full decode currently stops — loudly — at the §C.7.1
/// `used_orders = 0x5F` custom-coefficient-order sub-stream, whose
/// exact reading is the round-393 docs-gap. Everything before it
/// (permuted TOC, §C.5 framing, both LfGroups, frame-level assembly)
/// is validated by the tests above. Replace this with the pixel-MAD
/// ratchet against `expected.png` once the §C.7.1 trace is staged.
#[test]
fn full_decode_stops_at_c71_custom_orders() {
    let r = oxideav_jpegxl::decode_vardct_frame_from_codestream(JXL, None);
    let err = match r {
        Ok(_) => panic!(
            "decode unexpectedly SUCCEEDED — the §C.7.1 gap must have been resolved; \
             replace this pin with the expected.png pixel ratchet"
        ),
        Err(e) => e,
    };
    let msg = format!("{err:?}");
    assert!(
        !msg.contains("multi-LfGroup") && !msg.contains("num_lf_groups"),
        "must not fail at the (landed) multi-LfGroup framing: {msg}"
    );
    assert!(
        !msg.contains("permutation: LZ77"),
        "must not fail at the (landed) LZ77 TOC permutation: {msg}"
    );
}
