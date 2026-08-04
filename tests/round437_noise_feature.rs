//! Round 437 — the §K.4 kNoise image feature decodes end to end.
//!
//! The staged `noise-feature-256x256` fixture (photon-noise ISO 3200,
//! `FrameHeader.flags = 0x1` kNoise, lossy VarDCT d1.5) pins the whole
//! chain: §C.4.7 `NoiseParameters` LUT parse in LfGlobal → §K.4
//! per-group XorShift128Plus/SplitMix64 pseudorandom channels →
//! frame-level 5×5 Laplacian-like convolution (§6.5 mirroring) →
//! Listing K.5 strength-modulated injection into the XYB planes with
//! the §C.4.4 base correlations — validated against the black-box
//! reference decode.
//!
//! Round-437 baseline: per-channel sRGB MAD 0.9169 / 0.7884 / 0.8821,
//! max error 7 / 5 / 5 — the same sub-1/255 band as the noise-free
//! VarDCT fixtures (d1: 0.66 / 0.47 / 0.61), i.e. the deterministic
//! noise render itself introduces no additional error class beyond
//! the known VarDCT float-accuracy tail. The noise is NOT a no-op:
//! the ISO-3200 injection has multi-code amplitude, so a missing or
//! misseeded render would blow these bounds immediately.

use std::io::Cursor;

fn png_rgb(bytes: &[u8]) -> (usize, usize, Vec<u8>) {
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    assert_eq!(info.color_type, png::ColorType::Rgb);
    (info.width as usize, info.height as usize, buf)
}

#[test]
fn noise_feature_decodes_within_ratchet() {
    let jxl = include_bytes!("fixtures/noise_feature_256x256.jxl");
    let expected = include_bytes!("fixtures/noise_feature_256x256_expected.png");
    let (w, h, reference) = png_rgb(expected);
    let frame = oxideav_jpegxl::decode_one_frame(jxl, None).expect("kNoise stream must decode");
    assert_eq!(frame.planes.len(), 3);
    let bounds = [(0.95, 9u8), (0.85, 7u8), (0.95, 7u8)];
    for (c, (mad_max, abs_max)) in bounds.iter().enumerate() {
        let plane = &frame.planes[c];
        let mut sum = 0u64;
        let mut max = 0u8;
        for y in 0..h {
            for x in 0..w {
                let d = plane.data[y * plane.stride + x].abs_diff(reference[(y * w + x) * 3 + c]);
                sum += d as u64;
                max = max.max(d);
            }
        }
        let mad = sum as f64 / (w * h) as f64;
        assert!(
            mad < *mad_max && max <= *abs_max,
            "noise fixture ch{c} regressed: MAD {mad} (ratchet {mad_max}), max {max} \
             (ratchet {abs_max})"
        );
    }
}
