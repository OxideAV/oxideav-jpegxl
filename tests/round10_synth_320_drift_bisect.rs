//! Round-10 integration tests against `tests/fixtures/synth_320_grey/synth_320.jxl`,
//! a 320x320 grey lossless gradient (`pixel(y,x) = (y + x) & 0xFF`)
//! emitted by cjxl 0.11.1 — **promoted in round 278 to a full
//! pixel-correctness regression**.
//!
//! ## History
//!
//! Round-9 unblocked the 0-byte PassGroup decode and ~21k of 102400
//! pixels became correct. Round-10 bisected the FIRST diverging pixel
//! to (y=24, x=14) — frame-coord pixel #3086 inside PG[0][0] — and
//! pinned the drift anchors as regression baselines: PG[0][0] first
//! mismatch at (24, 14), PG[0][2] (the right-edge group with the
//! per-group Palette transform) at (6, 261), and the characteristic
//! leftward-sweeping cascade pattern. Rounds 19/126/272 used these
//! anchors to validate WP reading choices (`s_init - 1`, the
//! `sub_err` 8x-domain reading) by checking whether a candidate
//! change moved the drift EARLIER (worse) or later (better).
//!
//! ## Round-278 resolution
//!
//! The same two FDIS Annex E fixes that made `noise-64x64-lossless`
//! byte-exact (see `r32_noise_bisect.rs`: Listing E.2 `error2weight`
//! inner-Idiv-first operand order, pinned by the staged behavioural
//! trace's 52 full-precision weight cells; `true_errNW → true_errN`
//! fallback at x == 0) eliminated the synth_320 drift entirely: all
//! 102400 pixels now decode to `(y + x) & 0xFF`. The drift-anchor
//! assertions are accordingly promoted to whole-image equality.
//!
//! Round-10 also landed a `lz_dist_ctx` spec-conformance fix
//! (C.3.3): when `lz77.enabled` the LZ77 distance token is decoded
//! against the dedicated last context, not the leaf's per-symbol
//! context. That fix is a no-op for synth_320 (its symbol stream
//! has `lz77.enabled = false`) but would manifest immediately on
//! any fixture that triggers LZ77.

const SYNTH_320_JXL: &[u8] = include_bytes!("fixtures/synth_320_grey/synth_320.jxl");

#[test]
fn synth_320_pg00_first_24_rows_pixel_correct() {
    // Historical round-10 bisect anchor (rows y=0..24 inside PG[0][0]
    // were the maximal correct prefix until round 278). Kept as a
    // fast-failing subset with a precise failure message.
    let vf = oxideav_jpegxl::decode_one_frame(SYNTH_320_JXL, None).unwrap();
    assert_eq!(vf.planes.len(), 1);
    let plane = &vf.planes[0];
    for y in 0..24usize {
        for x in 0..128usize {
            let want = ((y as u32 + x as u32) & 0xFF) as u8;
            let got = plane.data[y * 320 + x];
            assert_eq!(
                got, want,
                "PG[0][0] pixel ({y},{x}) want {want} got {got} — this \
                 prefix has been pixel-correct since round 10"
            );
        }
    }
}

#[test]
fn synth_320_whole_image_pixel_correct() {
    // Round-278 promotion: the WP error2weight Idiv-first operand
    // order + the true_errNW column-0 fallback (see module doc)
    // removed the round-10 (y=24, x=14) drift anchor and every
    // downstream mismatch. The whole 320x320 frame must decode to
    // the synthetic gradient exactly.
    let vf = oxideav_jpegxl::decode_one_frame(SYNTH_320_JXL, None).unwrap();
    let plane = &vf.planes[0];
    let mut mismatches = 0usize;
    let mut first: Option<(usize, usize)> = None;
    for y in 0..320usize {
        for x in 0..320usize {
            let want = ((y as u32 + x as u32) & 0xFF) as u8;
            if plane.data[y * 320 + x] != want {
                mismatches += 1;
                if first.is_none() {
                    first = Some((y, x));
                }
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "synth_320 must decode pixel-exact (102400/102400) from round \
         278 onward; got {mismatches} mismatches, first at {first:?}. \
         A regression here means a WP reading was changed — check \
         `modular_fdis::wp_predict` (error2weight operand order, \
         true_err border fallbacks, sub_err reading)."
    );
}
