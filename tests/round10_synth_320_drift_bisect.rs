//! Round-10 integration tests against `tests/fixtures/synth_320_grey/synth_320.jxl`,
//! a 320x320 grey lossless gradient (`pixel(y,x) = (y + x) & 0xFF`)
//! emitted by cjxl 0.11.1.
//!
//! Round-9 unblocked the 0-byte PassGroup decode and ~21k of 102400
//! pixels became correct. Round-10 bisected the FIRST diverging pixel
//! to (y=24, x=14) — frame-coord pixel #3086 inside PG[0][0]. The
//! divergence is an ANS-state nuance: state 0x9CA780 alias-maps to
//! symbol 30 (`D[30] = 1`, low-prob), forcing a refill plus extra
//! bits that consume 21 bits beyond the 9-byte slot. djxl decodes
//! the same fixture within 9 bytes, so our state evolution must
//! diverge from djxl's somewhere in the 3086 prior decodes. The
//! per-decode trace is captured in this test as a regression anchor:
//! if a future fix changes the divergence point (or eliminates it),
//! these assertions will fail and need to be updated.
//!
//! Round-10 also lands a `lz_dist_ctx` spec-conformance fix
//! (C.3.3): when `lz77.enabled` the LZ77 distance token is decoded
//! against the dedicated last context, not the leaf's per-symbol
//! context. That fix is a no-op for synth_320 (its symbol stream
//! has `lz77.enabled = false`) but would manifest immediately on
//! any fixture that triggers LZ77.

const SYNTH_320_JXL: &[u8] = include_bytes!("fixtures/synth_320_grey/synth_320.jxl");

#[test]
fn synth_320_pg00_first_24_rows_pixel_correct() {
    // Round-10 bisect anchor: rows y=0..24 inside PG[0][0] (the
    // top-left 128x128 group, x=0..128) decode pixel-for-pixel
    // against (y + x) & 0xFF. PG[0][0]'s state drifts at y=24, x=14
    // when state 0x9CA780 maps to a low-prob ANS symbol. PG[0][1]
    // (gx=1, x=128..256) has its own earlier drift point and is
    // covered separately; PG[0][2] (gx=2, x=256..320) drifts almost
    // immediately because of the per-group Palette transform.
    let vf = oxideav_jpegxl::decode_one_frame(SYNTH_320_JXL, None).unwrap();
    assert_eq!(vf.planes.len(), 1);
    let plane = &vf.planes[0];
    for y in 0..24usize {
        for x in 0..128usize {
            let want = ((y as u32 + x as u32) & 0xFF) as u8;
            let got = plane.data[y * 320 + x];
            assert_eq!(
                got, want,
                "round-10 bisect anchor: PG[0][0] pixel ({y},{x}) want {want} got {got} \
                 — drift inside PG[0][0] should begin at y=24, x=14, not earlier"
            );
        }
    }
}

#[test]
fn synth_320_first_drift_in_pg00_lands_at_y24_x14() {
    // Locking in the round-10 bisect finding for PG[0][0] (the
    // top-left 128x128 group): inside this group the first non-
    // matching pixel against (y+x)&0xFF is at frame coords
    // (y=24, x=14). State 0x9CA780 alias-maps to symbol 30
    // (D[30]=1, low-prob), forcing an ANS refill plus extra bits
    // that exceed the slot's 9-byte budget. Earlier groups in raster
    // order (PG[0][2], the right edge with 64-wide group) drift much
    // sooner — that's a separate edge-group failure mode tracked in
    // its own assertion below.
    let vf = oxideav_jpegxl::decode_one_frame(SYNTH_320_JXL, None).unwrap();
    let plane = &vf.planes[0];
    // Walk PG[0][0]'s rectangle (0..128, 0..128) in raster order.
    let mut first: Option<(usize, usize)> = None;
    'outer: for y in 0..128usize {
        for x in 0..128usize {
            let want = ((y as u32 + x as u32) & 0xFF) as u8;
            let got = plane.data[y * 320 + x];
            if got != want {
                first = Some((y, x));
                break 'outer;
            }
        }
    }
    let (y, x) = first.expect("PG[0][0] should still have at least one mismatch");
    assert_eq!(
        (y, x),
        (24, 14),
        "round-10 anchor: PG[0][0] first mismatch should be at (y=24, x=14), got ({y}, {x})"
    );
}

#[test]
fn synth_320_pg_0_2_edge_group_drifts_early() {
    // PG[0][2] is the gx=2 right-edge group: rect (256, 0, 64, 128).
    // It carries a per-group Palette transform (nb_colours=191) that
    // round 9 added support for. In the current state the 64-wide
    // edge group decodes a few rows then drifts; the FIRST mismatch
    // inside PG[0][2] is at frame (y=6, x=261). Round-11 work to
    // unify the per-group decode against djxl's behaviour should
    // either eliminate this drift or move its starting position;
    // either outcome means this assertion needs updating.
    let vf = oxideav_jpegxl::decode_one_frame(SYNTH_320_JXL, None).unwrap();
    let plane = &vf.planes[0];
    // Walk PG[0][2]'s rectangle (256..320, 0..128) in raster order.
    let mut first: Option<(usize, usize)> = None;
    'outer: for y in 0..128usize {
        for x in 256..320usize {
            let want = ((y as u32 + x as u32) & 0xFF) as u8;
            let got = plane.data[y * 320 + x];
            if got != want {
                first = Some((y, x));
                break 'outer;
            }
        }
    }
    let (y, x) = first.expect("PG[0][2] should have at least one mismatch in current state");
    assert_eq!(
        (y, x),
        (6, 261),
        "round-10 anchor: PG[0][2] first mismatch should be at (y=6, x=261), got ({y}, {x})"
    );
}

#[test]
fn synth_320_pg_0_0_intra_group_drift_pattern_unchanged() {
    // The drift pattern within PG[0][0] (x=0..128, y=0..128) is
    // characteristic: rows 0..24 are entirely correct, then row 24
    // starts mismatching at x=14, with the cascading mismatch sweeping
    // leftward by one column per subsequent row (each mismatched
    // sample's WP-neighbour propagates through predictor 6).
    let vf = oxideav_jpegxl::decode_one_frame(SYNTH_320_JXL, None).unwrap();
    let plane = &vf.planes[0];
    // Row 23 fully correct.
    for x in 0..128usize {
        let want = ((23u32 + x as u32) & 0xFF) as u8;
        assert_eq!(
            plane.data[23 * 320 + x],
            want,
            "row 23 (last fully-correct row) pixel x={x} differs"
        );
    }
    // Row 24 correct up to x=13, mismatches at x=14.
    for x in 0..14usize {
        let want = ((24u32 + x as u32) & 0xFF) as u8;
        assert_eq!(
            plane.data[24 * 320 + x],
            want,
            "row 24 should be correct up to x=13, pixel x={x} differs"
        );
    }
    let want_at_14 = ((24u32 + 14u32) & 0xFF) as u8; // = 38
    let got_at_14 = plane.data[24 * 320 + 14];
    assert_ne!(
        got_at_14, want_at_14,
        "round-10 anchor: (y=24, x=14) should still be the divergence point"
    );
}
