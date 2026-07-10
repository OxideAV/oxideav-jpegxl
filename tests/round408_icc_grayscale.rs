//! Round 408 — the ImageMetadata-tail → ICC-start boundary and the
//! Annex B ICC entropy decode, pinned on the ISO/IEC 18181-3
//! `grayscale` conformance stream (VarDCT, embedded 912-byte Grey ICC
//! profile, `want_icc == true`) plus a locally synthesised profile
//! fixture.
//!
//! What this file pins:
//!
//! * The Table A.16 tail layout (unconditional `default_transform`
//!   with INVERTED gating) via the metadata-tail gating trace — the
//!   `grayscale` ImageMetadata ends at bit 25.
//! * The ICC stream framing: `enc_size = U64()` is coded at the very
//!   next bit after ImageMetadata (bit 25 — NO §B.2 `ZeroPadToByte()`),
//!   and the entropy stream ends at bit 1939, padding to the frame
//!   header at byte 243.
//! * The Listing B.1 `IccContext` digit class and the simple-prefix
//!   first-symbol code assignment: the embedded 912-byte GRAY/ADBE
//!   profile reproduces the reference tooling's header fields and a
//!   structurally consistent tag set, and the synthesised
//!   `r408_custom.icc` (digit-heavy text tags) round-trips
//!   byte-exactly through a real encoder embedding.
//! * The remaining §C.7.1 boundary: the full grayscale frame decode
//!   still fails LOUDLY in the HfGlobal area (custom coefficient
//!   orders — `used_orders = 0x14`); when that gap closes this test's
//!   final assertion flips to a pixel comparison.

use oxideav_jpegxl::icc::{set_icc_start_trace_armed, ICC_START_TRACE};
use oxideav_jpegxl::metadata_fdis::{
    set_metadata_tail_trace_armed, ImageMetadataFdis, SizeHeaderFdis, METADATA_TAIL_TRACE,
};

const GRAYSCALE: &[u8] = include_bytes!("fixtures/conformance_grayscale.jxl");
const ICC_DIGITS: &[u8] = include_bytes!("fixtures/r408_icc_digits.jxl");
const CUSTOM_PROFILE: &[u8] = include_bytes!("fixtures/r408_custom.icc");

/// Decode a want_icc stream's prelude and return (tail trace, icc
/// start trace, decoded profile).
fn decode_icc_prelude(
    file: &[u8],
) -> (
    oxideav_jpegxl::metadata_fdis::MetadataTailTrace,
    oxideav_jpegxl::icc::IccStartTrace,
    Vec<u8>,
) {
    let cs = oxideav_jpegxl::container::extract_codestream(file).unwrap();
    let cs = &cs.as_ref()[2..];
    let mut br = oxideav_jpegxl::bitreader::BitReader::new(cs);
    let _size = SizeHeaderFdis::read(&mut br).expect("SizeHeader");
    set_metadata_tail_trace_armed(true);
    let md = ImageMetadataFdis::read(&mut br).expect("ImageMetadata");
    set_metadata_tail_trace_armed(false);
    let tail = METADATA_TAIL_TRACE
        .with(|s| s.borrow_mut().take())
        .expect("tail trace armed");
    assert!(md.colour_encoding.want_icc, "fixture must be want_icc");
    set_icc_start_trace_armed(true);
    let encoded = oxideav_jpegxl::icc::decode_encoded_icc_stream(&mut br).expect("Annex B decode");
    set_icc_start_trace_armed(false);
    let icc = ICC_START_TRACE
        .with(|s| s.borrow_mut().take())
        .expect("icc trace armed");
    let profile = oxideav_jpegxl::icc::reconstruct_icc_profile(&encoded).expect("reconstruct");
    (tail, icc, profile)
}

/// The metadata-tail gating trace + ICC-start boundary on the 18181-3
/// grayscale conformance stream — the round-408 SPECGAP resolution.
#[test]
fn grayscale_metadata_tail_and_icc_start() {
    let (tail, icc, profile) = decode_icc_prelude(GRAYSCALE);

    // Head flags.
    assert!(!tail.all_default);
    assert!(!tail.extra_fields);
    assert!(tail.xyb_encoded);
    assert_eq!(tail.num_extra_channels, 0);
    assert!(tail.want_icc);
    assert_eq!(tail.extensions_mask, 0);

    // Tail fields: default_transform present (1 bit, value 1 → no
    // opsin / cw_mask / weights follow under the inverted gating).
    let dt = tail
        .fields
        .iter()
        .find(|f| f.name == "default_transform")
        .expect("default_transform trace entry");
    assert!(dt.present);
    assert_eq!(dt.bits, 1);
    for name in ["opsin_inverse_matrix", "cw_mask", "up2_weight"] {
        let f = tail.fields.iter().find(|f| f.name == name).unwrap();
        assert!(!f.present, "{name} must be absent");
        assert_eq!(f.bits, 0);
    }
    assert_eq!(tail.bit_offset_end_of_image_metadata, 25);

    // ICC start: enc_size at the very next bit (NO byte-align), and
    // the entropy stream's end pads to the frame at byte 243.
    assert_eq!(icc.bit_offset_of_enc_size, 25);
    assert_eq!(icc.enc_size, 841);
    assert_eq!(icc.bit_offset_after_icc, 1939);

    // The embedded profile: 912-byte GRAY / ADBE / relative-intent —
    // the exact fields the reference tooling reports for this stream.
    assert_eq!(profile.len(), 912);
    assert_eq!(&profile[0..4], &912u32.to_be_bytes());
    assert_eq!(&profile[4..8], b"ADBE");
    assert_eq!(&profile[16..20], b"GRAY");
    assert_eq!(&profile[36..40], b"acsp");
    assert_eq!(&profile[64..68], &1u32.to_be_bytes());
    // Structural consistency: every tag-table entry in bounds.
    let ntags = u32::from_be_bytes(profile[128..132].try_into().unwrap()) as usize;
    assert_eq!(ntags, 5);
    for i in 0..ntags {
        let off = u32::from_be_bytes(profile[136 + 12 * i..140 + 12 * i].try_into().unwrap());
        let sz = u32::from_be_bytes(profile[140 + 12 * i..144 + 12 * i].try_into().unwrap());
        assert!(
            (off + sz) as usize <= profile.len(),
            "tag {i} out of bounds"
        );
    }
    assert_eq!(&profile[132..136], b"cprt");
    // cprt text: "Copyright 1999 Adobe Systems Incorporated" — the
    // digit run that derailed the pre-408 IccContext (missing digit
    // class).
    let cprt_off = u32::from_be_bytes(profile[136..140].try_into().unwrap()) as usize;
    assert_eq!(&profile[cprt_off..cprt_off + 4], b"text");
    assert_eq!(
        &profile[cprt_off + 8..cprt_off + 8 + 41],
        b"Copyright 1999 Adobe Systems Incorporated"
    );
}

/// Byte-exact ICC round-trip through a real encoder embedding of a
/// locally synthesised digit-heavy grey profile.
#[test]
fn synthesised_profile_round_trips_byte_exact() {
    let (_tail, _icc, profile) = decode_icc_prelude(ICC_DIGITS);
    assert_eq!(profile.len(), CUSTOM_PROFILE.len());
    assert_eq!(
        profile, CUSTOM_PROFILE,
        "embedded ICC must round-trip byte-exactly"
    );
}

/// The full grayscale frame decode reaches the §C.7.1 custom
/// coefficient orders and fails LOUDLY there (never a silent
/// misparse). Flip to a pixel assertion when the C.3.2 permutation
/// semantics gap closes.
#[test]
fn grayscale_frame_decode_fails_loudly_at_c71_boundary() {
    let err = oxideav_jpegxl::decode_one_frame(GRAYSCALE, None)
        .expect_err("grayscale frame decode is expected to stop at the §C.7.1 boundary");
    let msg = format!("{err}");
    assert!(
        msg.contains("HybridUintConfig") || msg.contains("coeff permutation"),
        "expected a loud §C.7-area error, got: {msg}"
    );
}
