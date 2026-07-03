//! Round 389 — multi-group VarDCT framing on the `large-1024x768-d2`
//! fixture (12 × 256×256 groups, one LfGroup, 15-entry TOC).
//!
//! Clean-room: behaviour is derived from the ISO/IEC 18181 spec PDFs +
//! the staged trace/errata material under `docs/image/jpegxl/`. The
//! fixture's `trace.txt` (an instrumented `djxl` black-box decode) pins
//! the on-wire section sizes; `expected.png` is the reference decode.
//! No external implementation source is consulted.

use oxideav_jpegxl::bitreader::BitReader;
use oxideav_jpegxl::frame_header::{Encoding, FrameDecodeParams, FrameHeader};
use oxideav_jpegxl::metadata_fdis::{ImageMetadataFdis, SizeHeaderFdis};
use oxideav_jpegxl::toc::Toc;

const JXL: &[u8] = include_bytes!("fixtures/large_1024x768_d2.jxl");
const REF_PNG: &[u8] = include_bytes!("fixtures/large_1024x768_d2_expected.png");

/// Parse the codestream prelude + FrameHeader + TOC of the raw-codestream
/// fixture, returning `(frame_header, toc, frame_bytes)` where
/// `frame_bytes` starts at the first (byte-aligned) TOC section.
fn parse_to_sections(codestream: &[u8]) -> (FrameHeader, Toc, &[u8]) {
    let mut br = BitReader::new(codestream);
    let size = SizeHeaderFdis::read(&mut br).expect("SizeHeader");
    let metadata = ImageMetadataFdis::read(&mut br).expect("ImageMetadata");
    assert!(!metadata.colour_encoding.want_icc, "fixture carries no ICC");
    br.pu0().expect("byte align");
    let params = FrameDecodeParams {
        xyb_encoded: metadata.xyb_encoded,
        num_extra_channels: metadata.num_extra_channels,
        have_animation: metadata.have_animation,
        have_animation_timecodes: false,
        image_width: size.width,
        image_height: size.height,
    };
    let fh = FrameHeader::read(&mut br, &params).expect("FrameHeader");
    let toc = Toc::read(&mut br, &fh).expect("TOC");
    let start = br.bytes_consumed();
    (fh, toc, &codestream[start..])
}

/// The frame geometry matches the fixture's documented shape: VarDCT,
/// single pass, 12 groups (4×3 grid of 256×256), one LfGroup.
#[test]
fn frame_header_geometry_matches_fixture_notes() {
    let (fh, _, _) = parse_to_sections(&JXL[2..]);
    assert_eq!(fh.encoding, Encoding::VarDct);
    assert_eq!(fh.passes.num_passes, 1);
    assert_eq!((fh.width, fh.height), (1024, 768));
    assert_eq!(fh.group_dim(), 256);
    assert_eq!(fh.num_groups(), 12);
    assert_eq!(fh.num_lf_groups(), 1);
}

/// The 15-entry TOC decodes to exactly the section sizes the fixture's
/// black-box decode trace records (`TOC ... sizes=17,8840,33,20,18,16,
/// 14,18,16,16,14,16,16,18,14`), unpermuted, summing to the remaining
/// frame bytes.
#[test]
fn toc_matches_black_box_trace() {
    let (_, toc, frame_bytes) = parse_to_sections(&JXL[2..]);
    assert!(!toc.permuted);
    assert_eq!(
        toc.entries,
        vec![17, 8840, 33, 20, 18, 16, 14, 18, 16, 16, 14, 16, 16, 18, 14],
        "TOC section sizes must match the fixture trace"
    );
    let total: u64 = toc.entries.iter().map(|&e| e as u64).sum();
    assert_eq!(total, frame_bytes.len() as u64);
}

/// Regression pin for the round-389 D.3.5 general-clustering fix: the
/// §C.7 HfGlobal section of this fixture cluster-maps
/// `495 × 1 × nb_block_ctx(5) = 2475` HF-coefficient contexts inside a
/// 33-byte section (the fixture trace records the same stream as
/// `num_contexts=2475 num_histograms=4 ... bits=250`). An ANS-coded
/// cluster index costs far less than one bit amortised, so a
/// `num_distributions ≤ bits_remaining` heuristic must NOT reject it.
#[test]
fn hf_global_section_parses_within_33_bytes() {
    let (fh, toc, frame_bytes) = parse_to_sections(&JXL[2..]);

    // Slice sections: LfGlobal(0), LfGroup(1), HfGlobal(2).
    let mut starts = Vec::new();
    let mut acc = 0usize;
    for &e in &toc.entries {
        starts.push(acc);
        acc += e as usize;
    }
    let sect = |i: usize| -> &[u8] { &frame_bytes[starts[i]..starts[i] + toc.entries[i] as usize] };

    let mut lf_br = BitReader::new_section(sect(0));
    let metadata = {
        // Re-parse metadata (parse_to_sections drops it); cheap.
        let mut br = BitReader::new(&JXL[2..]);
        SizeHeaderFdis::read(&mut br).unwrap();
        ImageMetadataFdis::read(&mut br).unwrap()
    };
    let lf_global = oxideav_jpegxl::lf_global::LfGlobal::read(&mut lf_br, &fh, &metadata)
        .expect("LfGlobal parses");
    let hbc = lf_global
        .hf_block_context
        .as_ref()
        .expect("VarDCT LfGlobal carries HfBlockContext");
    assert_eq!(
        hbc.nb_block_ctx, 5,
        "fixture uses the default block-context map"
    );

    let mut hg_br = BitReader::new_section(sect(2));
    let hg = oxideav_jpegxl::hf_global::HfGlobal::read(&mut hg_br, fh.num_groups())
        .expect("HfGlobal (dequant + num_hf_presets) parses");
    assert_eq!(hg.num_hf_presets, 1);
    let hf_passes = oxideav_jpegxl::hf_pass::read_hf_pass_sequence(
        &mut hg_br,
        hg.num_hf_presets,
        hbc.nb_block_ctx,
    )
    .expect("§C.7.1 coefficient orders parse");
    assert_eq!(hf_passes.len(), 1);
    let histos =
        oxideav_jpegxl::hf_coefficient_histograms::HfCoefficientHistograms::read_after_hf_pass_sequence(
            &mut hg_br,
            hg.num_hf_presets,
            hbc.nb_block_ctx,
        )
        .expect("§C.7.2 histograms (2475 contexts) parse inside the 33-byte section");
    assert_eq!(histos.num_distributions(), 2475);
    // The whole section is 33 bytes = 264 bits; everything read so far
    // must have come from real section bits (no zero-padding reads).
    assert!(
        hg_br.bits_read() <= 264,
        "HfGlobal section parse overran its 33-byte TOC slot: {} bits",
        hg_br.bits_read()
    );
}

/// End-to-end multi-group VarDCT decode accuracy, measured in the XYB
/// domain (the pre-§L.2.2 planes via the `VARDCT_XYB_CAPTURE`
/// diagnostic hook) against the reference decode's PNG inverted
/// through the spec **forward** XYB transform. All 12 PassGroup
/// sections decode with group-local coordinates (§C.8.1): per-group
/// `hfp` headers, per-section ANS state init, group-local NonZeros
/// grids, and the pasted-together frame matches the reference to
/// per-pixel XYB MAD ≈ 7e-5 / 1.4e-3 / 9e-4 (X / Y / B) — measured
/// values at landing; thresholds double them.
///
/// Clean-room: the reference values are the `djxl` validator's opaque
/// output PNG inverted through the ISO/IEC 18181-1 forward XYB math
/// (Annex L.2 + the default OpsinInverseMatrix).
#[test]
fn multi_group_decode_matches_reference_in_xyb() {
    use oxideav_jpegxl::metadata_fdis::{OpsinInverseMatrix, ToneMapping};
    use std::io::Cursor;

    // Decode with the XYB capture hook armed.
    oxideav_jpegxl::VARDCT_XYB_CAPTURE.with(|s| *s.borrow_mut() = None);
    oxideav_jpegxl::set_vardct_xyb_capture_armed(true);
    let r = oxideav_jpegxl::decode_vardct_frame_from_codestream(JXL, None);
    oxideav_jpegxl::set_vardct_xyb_capture_armed(false);
    let frame = r.expect("multi-group VarDCT decode runs end-to-end");
    assert_eq!(frame.planes.len(), 3);
    assert_eq!(frame.planes[0].stride, 1024);
    assert_eq!(frame.planes[0].data.len(), 1024 * 768);
    let ours = oxideav_jpegxl::VARDCT_XYB_CAPTURE
        .with(|s| s.borrow_mut().take())
        .expect("XYB capture populated");

    // Invert the reference PNG through the spec forward XYB transform.
    let oim = OpsinInverseMatrix::default();
    let tm = ToneMapping::default();
    let a = oim.inv_mat;
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    let fwd = [
        [
            (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det,
            (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / det,
            (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det,
        ],
        [
            (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det,
            (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / det,
            (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det,
        ],
        [
            (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det,
            (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / det,
            (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det,
        ],
    ];
    let itscale = 255.0 / tm.intensity_target;
    let s2l = |c: f32| -> f32 {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    let dec = png::Decoder::new(Cursor::new(REF_PNG));
    let mut reader = dec.read_info().expect("png read_info");
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap_or(0)];
    let info = reader.next_frame(&mut buf).expect("png next_frame");
    let ch = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => panic!("unexpected reference colour type {other:?}"),
    };
    assert_eq!((info.width, info.height), (1024, 768));

    let n = 1024usize * 768;
    let mut mads = [0f64; 3];
    for i in 0..n {
        let rl = s2l(buf[i * ch] as f32 / 255.0) / itscale;
        let gl = s2l(buf[i * ch + 1] as f32 / 255.0) / itscale;
        let bl = s2l(buf[i * ch + 2] as f32 / 255.0) / itscale;
        let lm = fwd[0][0] * rl + fwd[0][1] * gl + fwd[0][2] * bl;
        let mm = fwd[1][0] * rl + fwd[1][1] * gl + fwd[1][2] * bl;
        let sm = fwd[2][0] * rl + fwd[2][1] * gl + fwd[2][2] * bl;
        let gl_ = (lm - oim.opsin_bias[0]).cbrt() + oim.opsin_bias[0].cbrt();
        let gm_ = (mm - oim.opsin_bias[1]).cbrt() + oim.opsin_bias[1].cbrt();
        let gs_ = (sm - oim.opsin_bias[2]).cbrt() + oim.opsin_bias[2].cbrt();
        mads[0] += (ours[0][i] as f64 - ((gl_ - gm_) * 0.5) as f64).abs();
        mads[1] += (ours[1][i] as f64 - ((gl_ + gm_) * 0.5) as f64).abs();
        mads[2] += (ours[2][i] as f64 - gs_ as f64).abs();
    }
    for (c, name, tol) in [(0usize, "X", 1.5e-4), (1, "Y", 3e-3), (2, "B", 2e-3)] {
        let mad = mads[c] / n as f64;
        assert!(
            mad < tol,
            "XYB {name}: per-pixel MAD {mad:.6} exceeds {tol} — multi-group framing regressed"
        );
    }
}
