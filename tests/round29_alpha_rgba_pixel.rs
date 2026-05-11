//! Round 29 (parent-dispatch r14) pixel-correctness tests for the
//! `alpha-64x64` 4-channel RGBA Modular lossless fixture and an
//! ISOBMFF-wrapped re-validation of the existing gray-64x64 fixture
//! (now that the FF 0A jxlc-payload signature is stripped before
//! `decode_codestream` per the lib-level fix).
//!
//! `alpha-64x64` is 64×64 RGBA lossless modular (`cjxl SOURCE input.jxl
//! -d 0 -e 7`) from the docs cleanroom fixture corpus
//! (`docs/image/jpegxl/fixtures/alpha-64x64/`). It exercises:
//!
//!   * `ImageMetadata.num_extra_channels = 1`, ExtraChannelInfo[0] of
//!     type `Alpha` (per FDIS A.6 + A.9 + Table A.22).
//!   * Four per-channel Palette transforms (one per R, G, B, A) per
//!     FDIS H.6 + Table H.4.
//!   * A 4-plane VideoFrame output where plane[3] is the decoded alpha
//!     channel — exercised here by byte-for-byte comparison against
//!     the committed `alpha-64x64/expected.png` (PNG ColorType=Rgba,
//!     8-bit).
//!
//! The two fixes that unblock this fixture:
//!
//!   1. `decode_one_frame` now strips the 2-byte `FF 0A` codestream
//!      signature from the jxlc/jxlp payload before calling
//!      `decode_codestream` (FDIS Annex B.1). The previous code only
//!      stripped FF 0A on the RawCodestream branch, so any
//!      ISOBMFF-wrapped JXL misaligned by 16 bits at the SizeHeader
//!      parse. No round-1..28 fixture covered that branch
//!      end-to-end so the bug went unnoticed.
//!   2. The post-Modular channel-count check that previously rejected
//!      `n_chans != expected_chans` now also accepts
//!      `n_chans == expected_chans + metadata.num_extra_channels`,
//!      mapping the extra channels into trailing VideoFrame planes
//!      (FDIS Annex G.1.3 colour-then-extras channel-order rule).

use oxideav_jpegxl::decode_one_frame;
use png::ColorType;
use std::io::Cursor;

const ALPHA_JXL: &[u8] = include_bytes!("fixtures/alpha_64x64.jxl");
const ALPHA_PNG: &[u8] = include_bytes!("fixtures/alpha_64x64_expected.png");

/// Decode an 8-bit PNG into `(width, height, planes_in_R-G-B-A_order)`.
fn png_to_planes_rgba(bytes: &[u8]) -> (u32, u32, Vec<Vec<u8>>) {
    let dec = png::Decoder::new(Cursor::new(bytes));
    let mut reader = dec.read_info().expect("png read_info");
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap_or(0)];
    let info = reader.next_frame(&mut buf).expect("png next_frame");
    let (w, h) = (info.width, info.height);
    let bytes = &buf[..info.buffer_size()];
    assert_eq!(info.bit_depth, png::BitDepth::Eight, "expect 8-bit PNG");
    assert!(
        matches!(info.color_type, ColorType::Rgba),
        "alpha-64x64/expected.png must be RGBA"
    );
    let n = (w * h) as usize;
    let (mut r, mut g, mut b, mut a) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for px in bytes.chunks_exact(4) {
        r.push(px[0]);
        g.push(px[1]);
        b.push(px[2]);
        a.push(px[3]);
    }
    (w, h, vec![r, g, b, a])
}

fn assert_rgba_planes_equal(ours: &[Vec<u8>], theirs: &[Vec<u8>], w: u32, h: u32) {
    assert_eq!(ours.len(), 4, "expected 4 planes (R, G, B, A)");
    assert_eq!(theirs.len(), 4, "reference has 4 planes (R, G, B, A)");
    let labels = ["R", "G", "B", "A"];
    for (idx, (a, b)) in ours.iter().zip(theirs.iter()).enumerate() {
        assert_eq!(
            a.len(),
            (w * h) as usize,
            "plane[{idx}] ({}) len {} expected {}",
            labels[idx],
            a.len(),
            w * h
        );
        assert_eq!(
            b.len(),
            (w * h) as usize,
            "ref plane[{idx}] ({}) len mismatch",
            labels[idx]
        );
        if a != b {
            for (px_idx, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
                if av != bv {
                    let x = (px_idx as u32) % w;
                    let y = (px_idx as u32) / w;
                    panic!(
                        "alpha-64x64: plane[{idx}] ({}) mismatch at ({x}, {y}): ours={av} theirs={bv}",
                        labels[idx]
                    );
                }
            }
        }
    }
}

#[test]
fn alpha_64x64_rgba_pixel_correct_vs_expected_png() {
    let vf = decode_one_frame(ALPHA_JXL, None).expect("alpha-64x64 must decode");
    assert_eq!(
        vf.planes.len(),
        4,
        "alpha-64x64: expected 4 RGBA planes (R, G, B, A), got {}",
        vf.planes.len()
    );
    let (w, h, ref_planes) = png_to_planes_rgba(ALPHA_PNG);
    assert_eq!((w, h), (64, 64));
    let ours: Vec<Vec<u8>> = vf.planes.iter().map(|p| p.data.clone()).collect();
    assert_rgba_planes_equal(&ours, &ref_planes, w, h);
}

/// Regression: confirm the five pre-round-29 lossless fixtures still
/// pass under the new channel-count contract + ISOBMFF FF 0A strip.
#[test]
fn five_pre_round29_fixtures_still_pass() {
    // Smallest fixture: pixel-1x1 (raw codestream, RGB single pixel).
    let bytes = include_bytes!("fixtures/pixel_1x1.jxl");
    let vf = decode_one_frame(bytes, None).expect("pixel-1x1");
    assert_eq!(vf.planes.len(), 3);
    assert_eq!(vf.planes[0].data, vec![255u8]);

    // gray-64x64 (raw codestream, single-channel grey).
    let bytes = include_bytes!("fixtures/gray_64x64_lossless.jxl");
    let vf = decode_one_frame(bytes, None).expect("gray-64x64");
    assert_eq!(vf.planes.len(), 1);
    assert_eq!(vf.planes[0].data.len(), 64 * 64);

    // gradient-64x64 (raw codestream, 3-channel RGB).
    let bytes = include_bytes!("fixtures/gradient_64x64_lossless.jxl");
    let vf = decode_one_frame(bytes, None).expect("gradient-64x64");
    assert_eq!(vf.planes.len(), 3);
    assert_eq!(vf.planes[0].data.len(), 64 * 64);

    // palette-32x32 (raw codestream, 3-channel RGB via Palette).
    let bytes = include_bytes!("fixtures/palette_32x32.jxl");
    let vf = decode_one_frame(bytes, None).expect("palette-32x32");
    assert_eq!(vf.planes.len(), 3);
    assert_eq!(vf.planes[0].data.len(), 32 * 32);

    // grey_8x8_lossless (raw codestream, 8x8 grey).
    let bytes = include_bytes!("fixtures/grey_8x8_lossless.jxl");
    let vf = decode_one_frame(bytes, None).expect("grey-8x8");
    assert_eq!(vf.planes.len(), 1);
    assert_eq!(vf.planes[0].data.len(), 8 * 8);
}

/// Regression for the `decode_one_frame` ISOBMFF path: wrap an
/// existing pixel-correct raw codestream in a minimal ISOBMFF (JXL
/// signature box + jxlc box carrying `FF 0A || codestream_tail`) and
/// verify the decoded pixels are byte-identical to the raw-path
/// decode.
///
/// Before this round, `decode_one_frame` on the ISOBMFF branch did
/// NOT strip the `FF 0A` codestream signature from the jxlc/jxlp
/// payload, so `SizeHeader::read` started 16 bits into the
/// codestream and produced silent miscompares (or downstream parse
/// errors that surfaced far from the root cause — e.g. the
/// bit-depth-16 fixture happened to misparse the TOC `permuted` flag
/// as 1 and tripped the "LZ77-enabled TOC sub-stream not supported"
/// error in `toc::decode_permutation`).
#[test]
fn isobmff_wraps_raw_codestream_decodes_identically() {
    // ISOBMFF JXL signature box (12 bytes).
    let mut isobmff = vec![
        0x00, 0x00, 0x00, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A,
    ];
    // Minimal `ftyp` box (size=20, major=jxl, minor=0, compat=jxl).
    isobmff.extend_from_slice(&[
        0x00, 0x00, 0x00, 0x14, b'f', b't', b'y', b'p', b'j', b'x', b'l', b' ', 0x00, 0x00, 0x00,
        0x00, b'j', b'x', b'l', b' ',
    ]);
    // jxlc box wrapping the raw gradient-64x64 codestream (FF 0A prefix
    // intact — the jxlc payload IS a codestream).
    let raw = include_bytes!("fixtures/gradient_64x64_lossless.jxl");
    let jxlc_size = (8 + raw.len()) as u32;
    isobmff.extend_from_slice(&jxlc_size.to_be_bytes());
    isobmff.extend_from_slice(b"jxlc");
    isobmff.extend_from_slice(raw);

    let vf_raw = decode_one_frame(raw, None).expect("raw decode");
    let vf_iso = decode_one_frame(&isobmff, None).expect("isobmff decode");
    assert_eq!(vf_raw.planes.len(), vf_iso.planes.len());
    for (i, (a, b)) in vf_raw.planes.iter().zip(vf_iso.planes.iter()).enumerate() {
        assert_eq!(
            a.data, b.data,
            "isobmff decode plane[{i}] must match raw decode plane[{i}]"
        );
    }
}
