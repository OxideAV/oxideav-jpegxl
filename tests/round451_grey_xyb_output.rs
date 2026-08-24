//! Round 451 — the xyb_encoded → Grey output hand-off.
//!
//! Per C.4.8 an `xyb_encoded` kModular frame ALWAYS carries the three
//! (Y', X', B') channels, even when `colour_space == kGrey`; the
//! §L.2.2 inverse XYB produces (R, G, B) and the single grey output
//! plane is the GREEN channel (the luminance carrier of the opsin
//! matrix — R = G = B for genuinely grey content up to quantisation).
//! Wire-arbitrated pixel-wise against the black-box reference decode
//! of a locally generated 64×64 lossy-Modular grey stream
//! (`cjxl -d 1 --modular=1` from a colour-type-0 PNG). Before round
//! 451 this path refused loudly at the channel-count contract
//! ("3 channels but colour_space wants 1").

use std::io::Cursor;

#[test]
fn grey_xyb_output_reference_band() {
    let jxl = include_bytes!("fixtures/r451_grey_xyb.jxl");
    let png = include_bytes!("fixtures/r451_grey_xyb_expected.png");
    let decoder = png::Decoder::new(Cursor::new(&png[..]));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    assert_eq!(info.color_type, png::ColorType::Grayscale);
    let (w, h) = (info.width as usize, info.height as usize);
    let frame = oxideav_jpegxl::decode_one_frame(jxl, None).expect("grey xyb stream decodes");
    assert_eq!(frame.planes.len(), 1, "grey output must be a single plane");
    let p = &frame.planes[0];
    let mut sum = 0u64;
    let mut max = 0u8;
    for y in 0..h {
        for x in 0..w {
            let d = p.data[y * p.stride + x].abs_diff(buf[y * w + x]);
            sum += d as u64;
            max = max.max(d);
        }
    }
    let mad = sum as f64 / (w * h) as f64;
    assert!(
        mad < 0.5 && max <= 1,
        "grey xyb output out of band: MAD {mad} (bound 0.5), max {max} (bound 1)"
    );
}
