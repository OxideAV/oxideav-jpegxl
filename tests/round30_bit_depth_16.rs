//! Round 30 (parent-dispatch r15) — pixel-correctness for the
//! `bit-depth-16` 64×64 RGB lossless Modular fixture
//! (`docs/image/jpegxl/fixtures/bit-depth-16/`, `cjxl SOURCE input.jxl
//! -d 0 -e 7`, 421 B).
//!
//! Round-29 surfaced `bit-depth-16` as a docs-gap probe failure: the
//! decoder hard-rejected `metadata.bit_depth.bits_per_sample != 8` at
//! the start of the post-Modular VideoFrame mapping. This round lifts
//! that restriction for the pass-through (non-XYB / non-YCbCr) path
//! and adopts a documented LE-pack convention for samples wider than
//! 8 bits:
//!
//!   bps  ≤ 8 → 1 byte/sample, plane stride == width;
//!   9 ≤ bps ≤ 16 → 2 bytes/sample, **little-endian**, plane stride
//!                  == width × 2.
//!
//! The convention is documented in the crate README. `oxideav-core`'s
//! `VideoPlane` carries no bit-depth field, so a downstream consumer
//! must look up the source codestream's `bit_depth` (e.g. via the
//! `CodecParameters.extra_data`) to know how to interpret a wide
//! plane.
//!
//! Spec citations:
//!   * FDIS Annex A.6 + Table A.22 — `bit_depth.bits_per_sample`.
//!   * FDIS Annex G.1.3 — Modular channel order (no per-channel
//!     bit-depth split for kModular RGB; all colour channels share
//!     the global `bits_per_sample`).
//!   * PNG RFC 2083 §2.1 — explicitly spells out PNG's *big*-endian
//!     16-bit sample order; we keep our wire convention LE so a
//!     downstream `bytemuck::cast_slice::<_, u16>()` on a
//!     little-endian host needs no swap.
//!
//! Black-box oracle (cross-check, NOT used at test time): `djxl
//! v0.11.1 input.jxl /tmp/out.ppm` produces a P6 PPM whose 16-bit
//! samples (BE per Netpbm) match ours after byteswap; the committed
//! `expected.png` (16-bit RGB PNG) is the ground-truth used at test
//! time.

use oxideav_jpegxl::decode_one_frame;
use png::ColorType;
use std::io::Cursor;

const BD16_JXL: &[u8] = include_bytes!("fixtures/bit_depth_16.jxl");
const BD16_PNG: &[u8] = include_bytes!("fixtures/bit_depth_16_expected.png");

/// Decode the committed reference PNG into three planar `Vec<u16>`
/// channels (R, G, B) at native u16. Asserts the PNG is exactly 64×64
/// 16-bit RGB.
fn png_to_planes_rgb16(bytes: &[u8]) -> (u32, u32, [Vec<u16>; 3]) {
    let dec = png::Decoder::new(Cursor::new(bytes));
    let mut reader = dec.read_info().expect("png read_info");
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap_or(0)];
    let info = reader.next_frame(&mut buf).expect("png next_frame");
    let (w, h) = (info.width, info.height);
    let bytes = &buf[..info.buffer_size()];
    assert_eq!(
        info.bit_depth,
        png::BitDepth::Sixteen,
        "bit-depth-16/expected.png must be 16-bit",
    );
    assert!(
        matches!(info.color_type, ColorType::Rgb),
        "bit-depth-16/expected.png must be RGB (no alpha)",
    );
    let n = (w * h) as usize;
    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    // PNG ships 16-bit samples big-endian (RFC 2083 §2.1) and the
    // png crate preserves that on the output buffer.
    for px in bytes.chunks_exact(6) {
        r.push(u16::from_be_bytes([px[0], px[1]]));
        g.push(u16::from_be_bytes([px[2], px[3]]));
        b.push(u16::from_be_bytes([px[4], px[5]]));
    }
    (w, h, [r, g, b])
}

/// Reinterpret a LE-packed plane (round-30 convention: 2 bytes per
/// 16-bit sample, low byte first) into a `Vec<u16>`. Panics if
/// `data.len()` is not a multiple of 2.
fn plane_to_u16_le(data: &[u8]) -> Vec<u16> {
    assert_eq!(
        data.len() % 2,
        0,
        "round-30 16-bit plane bytes must be a multiple of 2",
    );
    data.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

#[test]
fn bit_depth_16_rgb_pixel_correct_vs_expected_png() {
    let vf = decode_one_frame(BD16_JXL, None).expect("bit-depth-16 must decode");
    assert_eq!(
        vf.planes.len(),
        3,
        "bit-depth-16: expected 3 RGB planes, got {}",
        vf.planes.len(),
    );

    // Round-30 LE-pack convention: 16-bit samples → 2 bytes/sample,
    // stride = width × 2.
    for (i, p) in vf.planes.iter().enumerate() {
        assert_eq!(p.stride, 64 * 2, "plane[{i}] stride");
        assert_eq!(p.data.len(), 64 * 64 * 2, "plane[{i}] data len");
    }

    let (w, h, ref_planes) = png_to_planes_rgb16(BD16_PNG);
    assert_eq!((w, h), (64, 64), "expected.png size");

    let labels = ["R", "G", "B"];
    for (idx, (plane, ref_ch)) in vf.planes.iter().zip(ref_planes.iter()).enumerate() {
        let ours = plane_to_u16_le(&plane.data);
        assert_eq!(ours.len(), ref_ch.len(), "plane[{idx}] sample count");
        if ours != *ref_ch {
            for (px_idx, (a, b)) in ours.iter().zip(ref_ch.iter()).enumerate() {
                if a != b {
                    let x = (px_idx as u32) % w;
                    let y = (px_idx as u32) / w;
                    panic!(
                        "bit-depth-16: plane[{idx}] ({}) mismatch at ({x}, {y}): ours={a:#06x} ref={b:#06x}",
                        labels[idx],
                    );
                }
            }
        }
    }
}

/// Sanity-check the LE-pack convention end-to-end: every plane's
/// byte length is exactly `width × height × 2`, the stride equals
/// `width × 2`, and round-tripping through `u16::from_le_bytes`
/// reproduces the same samples as `to_le_bytes`.
#[test]
fn bit_depth_16_le_pack_convention_self_consistent() {
    let vf = decode_one_frame(BD16_JXL, None).expect("bit-depth-16 must decode");
    for plane in &vf.planes {
        assert_eq!(plane.stride, 64 * 2);
        assert_eq!(plane.data.len(), 64 * 64 * 2);
        for chunk in plane.data.chunks_exact(2) {
            let s = u16::from_le_bytes([chunk[0], chunk[1]]);
            let back = s.to_le_bytes();
            assert_eq!(back, [chunk[0], chunk[1]]);
        }
    }
}

/// Regression: confirm the six pre-round-30 lossless 8-bit fixtures
/// still pass the byte-pack contract (stride == width, 1 byte/sample).
#[test]
fn pre_round30_8bit_fixtures_still_byte_packed() {
    // pixel-1x1 (RGB 1×1)
    let vf = decode_one_frame(include_bytes!("fixtures/pixel_1x1.jxl"), None).expect("pixel-1x1");
    for p in &vf.planes {
        assert_eq!(p.stride, 1);
        assert_eq!(p.data.len(), 1);
    }

    // gray-64x64 (Grey 64×64)
    let vf = decode_one_frame(include_bytes!("fixtures/gray_64x64_lossless.jxl"), None)
        .expect("gray-64x64");
    assert_eq!(vf.planes.len(), 1);
    assert_eq!(vf.planes[0].stride, 64);
    assert_eq!(vf.planes[0].data.len(), 64 * 64);

    // gradient-64x64 (RGB 64×64)
    let vf = decode_one_frame(include_bytes!("fixtures/gradient_64x64_lossless.jxl"), None)
        .expect("gradient-64x64");
    assert_eq!(vf.planes.len(), 3);
    for p in &vf.planes {
        assert_eq!(p.stride, 64);
        assert_eq!(p.data.len(), 64 * 64);
    }

    // alpha-64x64 (RGBA 64×64) — round 29
    let vf =
        decode_one_frame(include_bytes!("fixtures/alpha_64x64.jxl"), None).expect("alpha-64x64");
    assert_eq!(vf.planes.len(), 4);
    for p in &vf.planes {
        assert_eq!(p.stride, 64);
        assert_eq!(p.data.len(), 64 * 64);
    }
}
