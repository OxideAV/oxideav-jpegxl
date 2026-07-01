//! Multi-frame codestream iteration — `decode_all_frames`.
//!
//! A JXL codestream can carry more than one frame (§C.1): a shared
//! prelude (SizeHeader / ImageMetadata / ICC) followed by a byte-aligned
//! array of FrameHeader + TOC + section groups, terminated by the frame
//! whose FrameHeader sets `is_last` (§C.2). `decode_all_frames` reads the
//! prelude once and walks the array frame-by-frame.
//!
//! Fixture: `docs/image/jpegxl/fixtures/animation-3frame` (78 B) — three
//! 32×32 Regular Modular frames with `is_last = 0, 0, 1` and
//! `have_animation = 1`. Its three frames are solid red, green, and blue
//! respectively; `expected.png` is the first (red) frame.

use oxideav_jpegxl::{decode_all_frames, decode_one_frame};

const ANIM_FIXTURE: &[u8] = include_bytes!("fixtures/animation_3frame.jxl");

/// Uniform-colour frames encode as a single (r, g, b) triple repeated
/// over every pixel; assert the whole plane matches.
fn assert_solid(vf: &oxideav_core::VideoFrame, r: u8, g: u8, b: u8) {
    assert_eq!(vf.planes.len(), 3, "RGB frame must have three planes");
    for (plane, (chan, want)) in vf.planes.iter().zip([("R", r), ("G", g), ("B", b)]) {
        assert_eq!(plane.stride, 32, "{chan} plane stride");
        assert_eq!(plane.data.len(), 32 * 32, "{chan} plane sample count");
        assert!(
            plane.data.iter().all(|&v| v == want),
            "{chan} plane must be solid {want}"
        );
    }
}

#[test]
fn animation_3frame_decodes_all_three_frames() {
    let frames = decode_all_frames(ANIM_FIXTURE, None)
        .expect("multi-frame codestream must decode all three frames");
    assert_eq!(
        frames.len(),
        3,
        "the fixture's frame array terminates at the third frame (is_last)"
    );
    // Red, then green, then blue — one solid colour per frame.
    assert_solid(&frames[0], 255, 0, 0);
    assert_solid(&frames[1], 0, 255, 0);
    assert_solid(&frames[2], 0, 0, 255);
}

#[test]
fn decode_one_frame_returns_the_first_of_the_array() {
    // The single-frame entry point yields exactly the first frame the
    // multi-frame walk produces.
    let first = decode_one_frame(ANIM_FIXTURE, None).expect("first frame decodes");
    let all = decode_all_frames(ANIM_FIXTURE, None).expect("all frames decode");
    assert_eq!(first.planes.len(), all[0].planes.len());
    for (a, b) in first.planes.iter().zip(all[0].planes.iter()) {
        assert_eq!(a.stride, b.stride);
        assert_eq!(a.data, b.data);
    }
}

#[test]
fn first_frame_pts_flows_through_multi_frame_walk() {
    // `pts` is applied to the first frame; later frames carry None
    // (per-frame animation timing is not yet mapped onto pts).
    let frames = decode_all_frames(ANIM_FIXTURE, Some(4242)).expect("decode with pts");
    assert_eq!(frames[0].pts, Some(4242));
    assert_eq!(frames[1].pts, None);
    assert_eq!(frames[2].pts, None);
}

/// A single-frame codestream (one of the 2021-layout lossless fixtures)
/// walks to exactly one frame via `decode_all_frames`.
#[test]
fn single_frame_codestream_yields_one_frame() {
    let bytes = &include_bytes!("fixtures/gray_64x64_lossless.jxl")[..];
    let frames = decode_all_frames(bytes, None).expect("single-frame decode");
    assert_eq!(
        frames.len(),
        1,
        "a single-frame codestream is_last on frame 0"
    );
    // Same pixels as the single-frame entry point.
    let one = decode_one_frame(bytes, None).expect("single-frame decode via one-frame API");
    assert_eq!(frames[0].planes.len(), one.planes.len());
    for (a, b) in frames[0].planes.iter().zip(one.planes.iter()) {
        assert_eq!(a.data, b.data);
    }
}
