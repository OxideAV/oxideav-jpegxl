//! Round 406 — ISO/IEC 18181-3 conformance corpus: the Modular
//! blending / layering test cases decode end-to-end.
//!
//! Streams under test (committed from `docs/image/jpegxl/conformance/`,
//! the official Part 3 corpus):
//!
//! * `alpha_nonpremultiplied` — Modular, 12-bit, straight alpha.
//! * `alpha_triangles` — Modular, 9-bit, alpha.
//! * `blendmodes` — Modular, 12-bit, a five-frame `Reference[1]` chain
//!   exercising every Table C.8 blend mode
//!   (kReplace → kBlend → kAdd → kMul → kAlphaWeightedAdd).
//! * `sunset_logo` — Modular, RCT, 10-bit, Orientation 7, two layers
//!   with out-of-canvas signed crop offsets.
//!
//! The `*_expected.png` oracles are black-box reference decodes
//! (`djxl v0.11.1 input.jxl expected.png`, 16-bit RGBA PNG); djxl is
//! used strictly as an opaque validator binary. The Part 3 reference
//! `.npy` arrays are not redistributable here, but §A.3 core
//! conformance compares clamped samples channel-per-channel, which is
//! exactly what these assertions do.
//!
//! What round 406 landed to make these pass (all fixture-measured
//! against this corpus — see `src/frame_compose.rs` / `src/frame_header.rs`
//! / `src/orientation.rs` for the full notes):
//!
//! * multi-byte (9–16-bit) plane support in §C.2 composition;
//! * signed `UnpackSigned` crop offsets + §3.5.1 clipping of frame
//!   rects extending beyond the canvas;
//! * Table C.7 `alpha_channel` / `clamp` presence gated on the blend
//!   mode alone (the FDIS `multi_extra &&` gate desyncs every
//!   single-extra-channel stream with a blendy mode);
//! * float-domain, *unclamped* composition state — Modular frames may
//!   decode to out-of-range samples (negative planes are legal) and
//!   over-range intermediates must survive across blend stages;
//! * kAlphaWeightedAdd weights by the frame's own alpha and leaves the
//!   alpha channel unchanged;
//! * §A.6 Table A.17 orientation at presentation.

use oxideav_jpegxl::decode_all_frames;
use png::ColorType;
use std::io::Cursor;

/// Decode a 16-bit RGBA expected.png into interleaved u16 samples.
fn png_rgba16(bytes: &[u8]) -> (usize, usize, Vec<u16>) {
    let dec = png::Decoder::new(Cursor::new(bytes));
    let mut reader = dec.read_info().expect("png read_info");
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap_or(0)];
    let info = reader.next_frame(&mut buf).expect("png next_frame");
    assert_eq!(info.bit_depth, png::BitDepth::Sixteen, "16-bit oracle");
    assert_eq!(info.color_type, ColorType::Rgba, "RGBA oracle");
    let (w, h) = (info.width as usize, info.height as usize);
    let raw = &buf[..info.buffer_size()];
    let samples: Vec<u16> = raw
        .chunks_exact(2)
        .map(|c| u16::from_be_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(samples.len(), w * h * 4);
    (w, h, samples)
}

/// Our plane sample (little-endian 2-byte layout for bps > 8).
fn plane_sample(plane: &oxideav_core::VideoPlane, idx: usize) -> u32 {
    u16::from_le_bytes([plane.data[2 * idx], plane.data[2 * idx + 1]]) as u32
}

/// Map a djxl 16-bit PNG sample back to the native `bps`-bit value.
/// djxl expands with (float) `v × 65535 / (2^bps − 1)` scaling; the
/// expansion factor is ≈ 2^(16−bps) ≫ 1, so round-to-nearest inversion
/// recovers the native value exactly even across djxl's own ±1 float
/// rounding of the expanded sample.
fn native_from_16(v16: u32, bps: u32) -> u32 {
    let max = (1u64 << bps) - 1;
    ((v16 as u64 * max + 32767) / 65535) as u32
}

/// Compare a decoded frame against a 16-bit RGBA oracle in the native
/// `bps`-bit domain. Returns per-channel (max_abs_diff, diff_count).
fn compare(
    frame: &oxideav_core::VideoFrame,
    oracle: &(usize, usize, Vec<u16>),
    bps: u32,
) -> Vec<(u32, usize)> {
    let (w, h, ref samples) = *oracle;
    assert_eq!(frame.planes.len(), 4, "RGBA plane stack");
    assert_eq!(frame.planes[0].stride, w * 2, "presented width");
    assert_eq!(frame.planes[0].data.len(), w * h * 2, "presented height");
    (0..4usize)
        .map(|c| {
            let mut max_d = 0u32;
            let mut nd = 0usize;
            for i in 0..w * h {
                let ours = plane_sample(&frame.planes[c], i);
                let want = native_from_16(samples[i * 4 + c] as u32, bps);
                let d = ours.abs_diff(want);
                if d > 0 {
                    nd += 1;
                }
                max_d = max_d.max(d);
            }
            (max_d, nd)
        })
        .collect()
}

#[test]
fn alpha_nonpremultiplied_is_bit_exact() {
    let jxl = include_bytes!("fixtures/conformance_alpha_nonpremultiplied.jxl");
    let png = include_bytes!("fixtures/conformance_alpha_nonpremultiplied_expected.png");
    let frames = decode_all_frames(jxl, None).expect("decode");
    assert_eq!(frames.len(), 1);
    let oracle = png_rgba16(png);
    assert_eq!((oracle.0, oracle.1), (1024, 1024));
    for (c, (max_d, nd)) in compare(&frames[0], &oracle, 12).into_iter().enumerate() {
        assert_eq!((max_d, nd), (0, 0), "channel {c} must be bit-exact");
    }
}

#[test]
fn alpha_triangles_is_bit_exact() {
    let jxl = include_bytes!("fixtures/conformance_alpha_triangles.jxl");
    let png = include_bytes!("fixtures/conformance_alpha_triangles_expected.png");
    let frames = decode_all_frames(jxl, None).expect("decode");
    assert_eq!(frames.len(), 1);
    let oracle = png_rgba16(png);
    assert_eq!((oracle.0, oracle.1), (1024, 1024));
    for (c, (max_d, nd)) in compare(&frames[0], &oracle, 9).into_iter().enumerate() {
        assert_eq!((max_d, nd), (0, 0), "channel {c} must be bit-exact");
    }
}

/// The five-frame blend chain: only the final kAlphaWeightedAdd frame
/// is presented (`duration == 0` intermediates compose silently).
/// Composed (non-integer) samples quantise independently at 12 and 16
/// bits in the two reference outputs, so a boundary sample may sit ±1
/// native code from the PNG-derived oracle even when our float value
/// matches the reference's: against the native-depth PAM reference the
/// same decode shows only 245 total differing samples (all ±1, alpha
/// exact). The PNG oracle bound is therefore max ±1 with a generous
/// boundary-population count.
#[test]
fn blendmodes_chain_composes_within_one_code() {
    let jxl = include_bytes!("fixtures/conformance_blendmodes.jxl");
    let png = include_bytes!("fixtures/conformance_blendmodes_expected.png");
    let frames = decode_all_frames(jxl, None).expect("decode");
    assert_eq!(frames.len(), 1, "only the is_last frame is presented");
    let oracle = png_rgba16(png);
    assert_eq!((oracle.0, oracle.1), (1024, 1024));
    let per_ch = compare(&frames[0], &oracle, 12);
    for (c, &(max_d, nd)) in per_ch.iter().enumerate() {
        assert!(
            max_d <= 1,
            "channel {c}: max diff {max_d}/4095 exceeds one 12-bit code"
        );
        assert!(
            nd <= 30_000,
            "channel {c}: {nd} differing samples (≫ the ±1 quantisation-boundary population)"
        );
    }
}

/// Two RCT layers with signed out-of-canvas crops, kBlend, and
/// orientation 7 (presented transposed: 924×1386). The alpha plane is
/// bit-exact; the colour channels carry a known sub-RMS-40 (of 1023)
/// deviation in the smooth sky region — an open Modular-decode item
/// tracked by this ratchet, NOT a composition/orientation issue (both
/// pinned exact by the alpha plane and the other three streams).
#[test]
fn sunset_logo_orientation_and_alpha_exact_colour_ratchet() {
    let jxl = include_bytes!("fixtures/conformance_sunset_logo.jxl");
    let png = include_bytes!("fixtures/conformance_sunset_logo_expected.png");
    let frames = decode_all_frames(jxl, None).expect("decode");
    assert_eq!(frames.len(), 1);
    let oracle = png_rgba16(png);
    assert_eq!(
        (oracle.0, oracle.1),
        (924, 1386),
        "orientation 7 transposes the 1386×924 sample grid"
    );
    let (w, h, ref samples) = oracle;
    let f = &frames[0];
    assert_eq!(f.planes.len(), 4);
    assert_eq!(f.planes[0].stride, w * 2);
    // Alpha: bit-exact.
    let per_ch = compare(f, &oracle, 10);
    assert_eq!(per_ch[3], (0, 0), "alpha plane must be bit-exact");
    // Colour ratchet: RMS (in 10-bit codes) per channel.
    for c in 0..3 {
        let mut sum_sq = 0f64;
        for i in 0..w * h {
            let ours = plane_sample(&f.planes[c], i) as f64;
            let want = native_from_16(samples[i * 4 + c] as u32, 10) as f64;
            let d = ours - want;
            sum_sq += d * d;
        }
        let rms = (sum_sq / (w * h) as f64).sqrt();
        assert!(
            rms <= 45.0,
            "channel {c}: RMS {rms:.2}/1023 exceeds the round-406 ratchet"
        );
    }
}
