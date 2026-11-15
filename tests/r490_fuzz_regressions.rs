//! Round 490 — fuzz findings pinned as regression tests.

use oxideav_jpegxl::{DecodeOptions, decode, decode_all, decode_all_with};

/// `decode_full` finding: a 1590-byte Modular stream whose samples
/// reach the top of the 32-bit range. The self-correcting predictor's
/// sub-predictor errors (H.5.1, `(|subpred − 8·value| + 3) >> 3`) then
/// exceed 2^31; kept as 32-bit patterns, one of them was widened with
/// its sign into the modulo-2^32 error sum of the last column and the
/// addition overflowed (a panic with overflow checks on). The sum is
/// now 32-bit wrapping arithmetic throughout, which is what "Umod
/// (1 << 32)" asks for.
#[test]
fn weighted_predictor_error_sum_wraps_on_full_range_samples() {
    let bytes = include_bytes!("fixtures/fuzz_r490_wp_error_sum.jxl");
    // Whatever the verdict, it is a verdict: no panic, and the same
    // one from every entry point.
    let first = decode(bytes).map(|i| (i.width, i.height, i.format));
    let all = decode_all(bytes).map(|f| f.len());
    assert_eq!(first.is_ok(), all.is_ok(), "{first:?} / {all:?}");
    let layers = decode_all_with(bytes, &DecodeOptions::default().with_coalesce(false));
    assert_eq!(layers.is_ok(), all.is_ok());
    // And the same on a worker budget.
    let threaded = decode_all_with(bytes, &DecodeOptions::default().with_threads(4));
    assert_eq!(threaded.is_ok(), all.is_ok());
}

/// `decode_vardct` finding: a 10.7 KiB VarDCT stream with an HF
/// coefficient token that decodes to `u32::MAX`. `UnpackSigned` of it
/// is −2^31, but the routine computed `u / 2 + 1` first, one past the
/// signed range (a panic with overflow checks on; the library has had
/// this since the routine was written). It is the complement of
/// `u / 2` now.
#[test]
fn unpack_signed_of_the_largest_token_does_not_overflow() {
    let bytes = include_bytes!("fixtures/fuzz_r490_unpack_signed.jxl");
    let first = decode(bytes).map(|i| (i.width, i.height, i.format));
    let strict = oxideav_jpegxl::decode_with(bytes, &DecodeOptions::default().with_strict(true))
        .map(|i| (i.width, i.height, i.format));
    let all = decode_all(bytes).map(|f| f.len());
    // No panic; a decode that succeeds does so from every entry point.
    if let (Ok(a), Ok(b)) = (&first, &strict) {
        assert_eq!(a, b);
    }
    assert_eq!(first.is_ok(), all.is_ok(), "{first:?} / {all:?}");
    let threaded = decode_all_with(bytes, &DecodeOptions::default().with_threads(4));
    assert_eq!(threaded.is_ok(), all.is_ok());
}
