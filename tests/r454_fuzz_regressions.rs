//! Round 454 — regression pins for the fuzz-battery findings.
//!
//! Each case is a minimized hostile input the r454 cargo-fuzz targets
//! produced against the library surface. The pins assert the decode
//! entry points return a clean `Err` (or `Ok`) instead of panicking:
//!
//! 1. `matree` — a Modular channel whose MA tree names a decision
//!    property beyond the caller-supplied property table (the
//!    BEGABRAC1 read spans `[0, n_props + 12)`); previously an
//!    index-out-of-bounds panic in `decode_subtree`.
//! 2. `extensions` — a Bundle Extensions field whose `extension_bits`
//!    entries sum past `u64::MAX`; previously an add-with-overflow
//!    panic (debug) in `Extensions::payload_bits`.
//! 3. `icc` — an E.4.5 `Shuffle` shape with more than one missing
//!    matrix element (len 114, width 4): the uniform row indexing
//!    read one byte past the input; previously an
//!    index-out-of-bounds panic in `shuffle`.

/// r454 fuzz artifact for finding 1 (decode_modular target). The
/// harness's first four bytes choose the channel geometry; the
/// remainder is the channel payload.
const MATREE_PROP_OOB: &[u8] = &[
    255, 10, 79, 64, 0, 128, 80, 220, 8, 0, 96, 0, 24, 75, 1, 1, 63, 213,
];

/// r454 fuzz artifact for finding 2 (parse_headers target).
const EXTENSION_BITS_OVERFLOW: &[u8] = &[
    255, 10, 79, 64, 255, 255, 255, 255, 255, 255, 11, 31, 0, 172, 0, 129, 11, 208, 31, 198, 198,
    198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198,
    198, 198, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 239,
    255, 255, 255, 255, 255, 255, 255, 172, 0, 129, 11, 208, 31, 0, 0, 199, 68, 96, 119, 17,
];

/// r454 fuzz artifact for finding 3 (parse_icc target) — fed to
/// `reconstruct_icc_profile` as an already-entropy-decoded stream.
const ICC_SHUFFLE_OOB: &[u8] = &[
    255, 10, 18, 215, 11, 64, 3, 114, 47, 0, 0, 0, 0, 0, 0, 0, 65, 65, 65, 65, 111, 0, 122, 32, 0,
    0, 0, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
    75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 203, 75, 0, 65, 65, 65, 65, 75, 75, 75, 41, 0, 0,
    0, 0, 0, 0, 0, 75, 27, 0, 0, 0, 255, 255, 255, 194, 169, 151, 201, 255, 255, 252, 194, 109,
    110, 116, 0, 255, 252, 160, 33, 0, 0, 0, 0, 96, 71, 18, 18, 18, 18, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    200, 27, 0, 0, 0, 255, 255, 255, 194, 169, 151, 201, 255, 255, 252, 194, 109, 110, 116, 0, 255,
    252, 200, 200, 200, 200, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    71, 64, 24, 215, 11, 64, 3, 114, 47, 0, 0, 0, 0, 0, 0, 0, 65, 65, 65, 65, 111, 0, 122, 32, 0,
    0, 0, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
    75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 160, 75, 75, 75, 0, 33, 0, 0, 0, 65, 65, 65, 65, 75,
    75, 75, 75, 16, 0, 0, 0, 0, 0, 208, 255, 255, 180, 180, 176, 75, 75, 75, 75, 75, 75, 75, 0, 96,
    71, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
];

fn modular_geometry(data: &[u8]) -> (u32, u32, &[u8]) {
    // Mirrors the decode_modular fuzz harness's geometry derivation.
    let w = 1 + (u32::from(data[0]) | (u32::from(data[1] & 1) << 8)) % 512;
    let h = 1 + (u32::from(data[2]) | (u32::from(data[3] & 1) << 8)) % 512;
    (w, h, &data[4..])
}

#[test]
fn matree_out_of_table_property_is_invalid_data_not_panic() {
    let (w, h, payload) = modular_geometry(MATREE_PROP_OOB);
    let err = oxideav_jpegxl::modular::decode_single_channel(payload, w, h, 4)
        .expect_err("hostile MA tree must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("property"),
        "expected the property-range rejection, got: {msg}"
    );
}

#[test]
fn extension_bits_sum_overflow_no_panic() {
    // The probe walks SizeHeader + ImageMetadata; the hostile
    // Extensions field lives in the metadata tail. The saturating
    // total either skips cleanly (when the remaining input happens to
    // cover the claimed bits) or surfaces `Err` from `skip_payload` -
    // the pin is that the summation never overflows.
    let _ = oxideav_jpegxl::probe_fdis(EXTENSION_BITS_OVERFLOW);
    let _ = oxideav_jpegxl::probe(EXTENSION_BITS_OVERFLOW);
}

#[test]
fn icc_shuffle_multi_missing_elements_no_oob() {
    // Must return (Ok or Err) without panicking; the artifact drives
    // the E.4 command interpreter into a Shuffle with len % width
    // leaving more than one matrix cell missing.
    let _ = oxideav_jpegxl::icc::reconstruct_icc_profile(ICC_SHUFFLE_OOB);
}

/// r454 fuzz artifact for finding 4 (decode_modular target, second
/// wave): a channel header declaring near-full-range i32 min/max
/// overflowed the derived `2*min - max` property bounds.
const PROPERTY_RANGE_OVERFLOW: &[u8] = &[
    96, 65, 7, 192, 4, 255, 65, 255, 255, 255, 255, 4, 0, 255, 255, 70, 0, 172, 0, 129, 0, 217,
    255, 11, 192, 199, 175, 96, 119, 13,
];

/// r454 fuzz artifact for finding 5 (parse_icc target, second wave):
/// a general-clustering stream decoding a cluster index of u32::MAX,
/// whose `+ 1` in `num_clusters` overflowed.
const CLUSTER_INDEX_OVERFLOW: &[u8] = &[
    17, 47, 215, 1, 165, 116, 255, 255, 255, 1, 165, 116, 255, 255, 255, 255, 255, 255, 255, 86, 9,
    0, 1, 165, 116, 109, 116, 25, 1, 0, 0, 243, 32, 0, 0, 0, 5, 0, 15, 114, 255, 255, 41, 255, 86,
    9, 0, 1, 165, 116, 109, 116, 25, 116, 25, 0, 243, 243, 1, 0, 1, 0, 0, 243, 32, 0, 0, 0, 5, 0,
    15, 114, 116, 25, 0, 243, 243, 1, 0, 0, 0, 0, 0, 0, 109, 241, 243, 50, 243, 243, 243, 243, 40,
    0, 15, 114, 0, 241, 0, 0, 0, 0, 0, 109, 151, 241, 243, 50, 243, 243, 243, 243, 40, 0, 15, 114,
    0, 241, 151,
];

#[test]
fn property_range_extremes_no_overflow() {
    let (w, h, payload) = modular_geometry(PROPERTY_RANGE_OVERFLOW);
    let _ = oxideav_jpegxl::modular::decode_single_channel(payload, w, h, 4);
}

#[test]
fn cluster_index_saturates_no_overflow() {
    let mut br = oxideav_jpegxl::bitreader::BitReader::new(CLUSTER_INDEX_OVERFLOW);
    let _ = oxideav_jpegxl::icc::decode_encoded_icc_stream(&mut br);
}

/// r454 fuzz artifact for finding 7 (decode_modular target, third
/// wave; also reproduced by the first CI Fuzz leg): decoded
/// neighbour samples at i32 extremes overflowed the additive
/// gradient properties in `compute_properties` (D.7.2).
const PROPERTY_GRADIENT_OVERFLOW: &[u8] = &[
    96, 65, 7, 4, 0, 255, 249, 255, 255, 7, 255, 255, 65, 255, 255, 255, 129, 4, 28, 255, 255, 70,
    31, 0, 173, 129, 93, 0, 0, 75, 192, 199, 47, 96, 119, 13,
];

#[test]
fn property_gradient_extremes_no_overflow() {
    let (w, h, payload) = modular_geometry(PROPERTY_GRADIENT_OVERFLOW);
    let _ = oxideav_jpegxl::modular::decode_single_channel(payload, w, h, 4);
}

/// r454 fuzz artifact for finding 8 (decode_modular target, fourth
/// wave): a channel header with `lower == i32::MIN` overflowed the
/// i32 negation in the BEGABRAC signed-range split.
const BEGABRAC_NEG_OVERFLOW: &[u8] = &[
    96, 65, 7, 4, 0, 255, 255, 65, 255, 255, 255, 255, 7, 4, 0, 255, 214, 7, 255, 13,
];

#[test]
fn begabrac_neg_extreme_no_overflow() {
    let (w, h, payload) = modular_geometry(BEGABRAC_NEG_OVERFLOW);
    let _ = oxideav_jpegxl::modular::decode_single_channel(payload, w, h, 4);
}

/// r454 fuzz artifact for finding 9 (decode_modular target, soak
/// wave): neighbours at i32 extremes overflowed the Gradient
/// predictor's i32 sum.
const GRADIENT_PREDICTOR_OVERFLOW: &[u8] = &[
    96, 65, 142, 1, 8, 252, 128, 192, 230, 4, 154, 10, 190, 127, 127, 192, 197, 58, 29, 255,
];

#[test]
fn gradient_predictor_extremes_no_overflow() {
    let (w, h, payload) = modular_geometry(GRADIENT_PREDICTOR_OVERFLOW);
    let _ = oxideav_jpegxl::modular::decode_single_channel(payload, w, h, 4);
}
