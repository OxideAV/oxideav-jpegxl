//! Round 469 — fuzz findings pinned as regression tests.

use oxideav_jpegxl::{decode, decode_all, Error};

/// Round-469 fuzz finding (`decode_full`, libFuzzer OOM): a 68-byte raw
/// codestream whose LZ77-enabled tree sub-stream repeats a node pattern
/// up to the D.4.2 `1 << 26` bound; growing the node vector that far
/// cost > 2 GiB before the bound fired. The tree is now also budgeted
/// at 64 nodes per remaining codestream byte (+ 2^16), so the input is
/// rejected after ~66 k nodes.
#[test]
fn hostile_ma_tree_is_rejected_within_budget() {
    const OOM: [u8; 68] = [
        0xff, 0x0a, 0x47, 0x04, 0x03, 0x88, 0x09, 0x08, 0x00, 0x92, 0x09, 0x00, 0x00, 0x40, 0x00,
        0x4b, 0x46, 0x15, 0x23, 0x38, 0x09, 0x00, 0x00, 0x00, 0x18, 0x8b, 0x15, 0x80, 0x5d, 0xfe,
        0x7f, 0x00, 0x09, 0x05, 0x00, 0xcc, 0x7b, 0xfc, 0x01, 0x08, 0x00, 0x92, 0x00, 0x54, 0xa8,
        0xfc, 0x01, 0x08, 0x00, 0x92, 0x05, 0x00, 0x40, 0x2e, 0x00, 0x4b, 0x80, 0x15, 0x8b, 0xfe,
        0xc2, 0x7f, 0x05, 0x00, 0x4c, 0xca, 0xfc, 0x01,
    ];
    let started = std::time::Instant::now();
    let err = decode_all(&OOM).unwrap_err();
    assert!(matches!(err, Error::InvalidData(_)), "{err}");
    assert!(err.to_string().contains("tree budget"), "{err}");
    assert!(decode(&OOM).is_err());
    assert!(
        started.elapsed() < std::time::Duration::from_secs(10),
        "hostile tree took {:?}",
        started.elapsed()
    );
}

/// Round-469 fuzz finding (`decode_partial`, libFuzzer slow-unit): a
/// 75-byte 64×64 XYB Modular stream whose kSplines dictionary carries
/// 217 splines × 217 control points marching millions of samples off
/// canvas — the §K.4 renderer took 65 s. Annex M (Table M.1) bounds the
/// total control points at `min(2^20, fwidth × fheight / 2)` (2048
/// here) and the estimated area reached; both are now enforced before
/// rendering (level 5 unless the container's `jxll` box says 10).
#[test]
fn hostile_spline_dictionary_is_rejected_by_annex_m() {
    const SLOW: [u8; 75] = [
        0xff, 0x0a, 0x4f, 0x06, 0xd8, 0x13, 0x08, 0x00, 0x84, 0x00, 0x06, 0x6c, 0x6c, 0x6c, 0x6c,
        0x6c, 0x6c, 0x6c, 0x6c, 0x6c, 0x6c, 0x6c, 0x6c, 0x6c, 0x6c, 0x6c, 0x6c, 0x6c, 0x6c, 0x6c,
        0x6c, 0x6c, 0x6c, 0x6c, 0x6c, 0x6c, 0x6c, 0x6c, 0x6c, 0x6c, 0x6c, 0x9d, 0x93, 0x93, 0x39,
        0x8a, 0xdb, 0x94, 0x93, 0xc6, 0xa1, 0x4a, 0x76, 0x05, 0x68, 0x8c, 0xb3, 0x0e, 0x21, 0x4c,
        0xd2, 0x61, 0x00, 0x67, 0x67, 0x67, 0x67, 0x67, 0x67, 0x67, 0x67, 0x67, 0x67, 0x9f, 0xed,
    ];
    let started = std::time::Instant::now();
    let err = decode(&SLOW).unwrap_err();
    assert!(matches!(err, Error::InvalidData(_)), "{err}");
    assert!(err.to_string().contains("Annex M"), "{err}");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "hostile splines took {:?}",
        started.elapsed()
    );
}
