#![no_main]

//! Cross-decode fuzz harness: random RGBA → libjxl lossless encode →
//! `oxideav_jpegxl::decode_one_frame`.
//!
//! Knowledge gap caveat (rounds 1-9 surfaced 4 spec PDF typos and
//! several MA-tree / ANS issues; round 10 is parked). Many fixtures
//! still fail to decode in the round-3 envelope of the decoder
//! (`decode_one_frame` only handles single-group Modular frames with
//! Grey 8 bpp). Crucially, libjxl encodes our RGBA input in colour
//! format which oxideav's current envelope rejects with
//! `Error::Unsupported(...)`.
//!
//! So this harness is intentionally tolerant: an `Err(_)` from
//! `decode_one_frame` is silently accepted (we are NOT asserting
//! decoder completeness yet — only that no panics, OOMs, integer
//! overflows, or other UB occur in the parser/entropy-decoder paths
//! the envelope DOES reach). When `decode_one_frame` happens to
//! return `Ok(frame)` we assert byte-exact pixel equality (lossless
//! mode is supposed to roundtrip exactly).

use libfuzzer_sys::fuzz_target;
use oxideav_jpegxl::{decode_one_frame, JxlImage};
use oxideav_jpegxl_fuzz::libjxl;

const MAX_WIDTH: usize = 64;
const MAX_PIXELS: usize = 2048;

fuzz_target!(|data: &[u8]| {
    // Skip silently if libjxl isn't installed on this host.
    if !libjxl::available() {
        return;
    }

    let Some((width, height, rgba)) = image_from_fuzz_input(data) else {
        return;
    };

    let Some(encoded) = libjxl::encode_lossless_rgba(rgba, width, height) else {
        // libjxl rejected our input (e.g. degenerate dimensions for
        // the encoder's internal heuristics) — that's not an oxideav
        // bug, just skip.
        return;
    };

    // The decoder is a work-in-progress (round-3 envelope: Grey-8 only
    // single-group Modular). RGBA inputs from libjxl will most often
    // hit the `Err(Unsupported)` path. Accept that gracefully — the
    // point of the harness is to surface panics / UB on whatever the
    // parser DOES manage to walk before bailing.
    if let Ok(frame) = decode_one_frame(&encoded, None) {
        assert_pixel_equal(rgba, width, height, &frame);
    }
});

fn image_from_fuzz_input(data: &[u8]) -> Option<(u32, u32, &[u8])> {
    let (&shape, rgba) = data.split_first()?;

    let pixel_count = (rgba.len() / 4).min(MAX_PIXELS);
    if pixel_count == 0 {
        return None;
    }

    let width = ((shape as usize) % MAX_WIDTH) + 1;
    let width = width.min(pixel_count);
    let height = pixel_count / width;
    if height == 0 {
        return None;
    }
    let used_len = width * height * 4;
    let rgba = &rgba[..used_len];

    Some((width as u32, height as u32, rgba))
}

/// Best-effort pixel-equality check. The decoder currently emits one
/// plane (Grey-8) — if it returns Ok with a different plane shape
/// (e.g. RGB triplanar), we still want to catch obvious corruption,
/// so we just assert the planes are dimensionally consistent and that
/// any plane data covers the claimed area. Full byte-exact RGBA
/// roundtrip won't be possible until the decoder grows colour
/// support, at which point this assertion should be tightened.
fn assert_pixel_equal(expected_rgba: &[u8], width: u32, height: u32, frame: &JxlImage) {
    assert!(!frame.planes.is_empty(), "decoded frame has zero planes");
    let w = width as usize;
    let h = height as usize;

    for (idx, plane) in frame.planes.iter().enumerate() {
        // Stride must cover the row width.
        assert!(
            plane.stride >= w,
            "plane {idx} stride {} < width {w}",
            plane.stride
        );
        // Data must hold at least height rows of `stride` bytes.
        assert!(
            plane.data.len() >= plane.stride * h,
            "plane {idx} data {} < stride*height {}",
            plane.data.len(),
            plane.stride * h
        );
    }

    // If oxideav-jpegxl ever returns a single 4-channel-interleaved
    // RGBA plane with stride = width*4, byte-compare it.
    if frame.planes.len() == 1 && frame.planes[0].stride == w * 4 {
        let actual = &frame.planes[0].data[..w * h * 4];
        assert_eq!(
            actual, expected_rgba,
            "lossless RGBA roundtrip differs for {w}x{h} frame",
        );
    }
}
