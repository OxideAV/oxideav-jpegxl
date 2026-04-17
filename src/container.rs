//! JPEG XL container detection.
//!
//! A JXL file is either a raw codestream (begins with `FF 0A`) or an
//! ISOBMFF-style box-wrapped file (begins with the 12-byte JXL signature
//! box `00 00 00 0C 4A 58 4C 20 0D 0A 87 0A`). The wrapper carries the
//! codestream in one or more `jxlc` / `jxlp` boxes alongside optional
//! metadata boxes (`jbrd`, `Exif`, `xml `, `jumb`, ...).
//!
//! This module exposes signature detection and a minimal demuxer that
//! extracts the concatenated codestream payload from the wrapper.

use oxideav_core::{Error, Result};

/// Raw codestream magic: `FF 0A` (2 bytes, little-endian reading of `0x0AFF`).
pub const RAW_CODESTREAM_SIGNATURE: [u8; 2] = [0xFF, 0x0A];

/// ISOBMFF-wrapped signature box: 12 bytes — a box of size 12, type
/// `JXL ` (0x4A584C20), followed by the 4-byte payload `0D 0A 87 0A`.
pub const ISOBMFF_SIGNATURE: [u8; 12] = [
    0x00, 0x00, 0x00, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A,
];

/// Which wrapping the input uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signature {
    /// Raw codestream, `FF 0A …`.
    RawCodestream,
    /// ISOBMFF box-wrapped container, signature box `00 00 00 0C JXL␣ 0D 0A 87 0A`.
    Isobmff,
}

/// Detect which JXL signature (if any) is at the start of `data`.
///
/// Returns `None` if neither signature matches. Does not consume or copy
/// any bytes beyond peeking at the prefix.
pub fn detect(data: &[u8]) -> Option<Signature> {
    if data.len() >= 12 && data[..12] == ISOBMFF_SIGNATURE {
        return Some(Signature::Isobmff);
    }
    if data.len() >= 2 && data[..2] == RAW_CODESTREAM_SIGNATURE {
        return Some(Signature::RawCodestream);
    }
    None
}

/// Extract the codestream bytes from a JXL input regardless of wrapping.
///
/// For raw inputs this is a zero-copy slice of `data`. For ISOBMFF inputs
/// this walks the top-level boxes and concatenates the payloads of all
/// `jxlc` + `jxlp` boxes in order.
///
/// Note: `jxlp` ordering is technically governed by a 4-byte index prefix
/// whose high bit marks the last partial; this implementation accepts
/// `jxlp` payloads in file order after stripping the index, which matches
/// compliant encoders but is not a full spec-conforming reorderer.
pub fn extract_codestream<'a>(data: &'a [u8]) -> Result<std::borrow::Cow<'a, [u8]>> {
    match detect(data) {
        Some(Signature::RawCodestream) => Ok(std::borrow::Cow::Borrowed(data)),
        Some(Signature::Isobmff) => {
            let mut out: Vec<u8> = Vec::new();
            let mut pos = 0usize;
            while pos + 8 <= data.len() {
                let size32 = u32::from_be_bytes([
                    data[pos],
                    data[pos + 1],
                    data[pos + 2],
                    data[pos + 3],
                ]);
                let box_type = &data[pos + 4..pos + 8];
                let (header_len, box_len) = match size32 {
                    1 => {
                        if pos + 16 > data.len() {
                            return Err(Error::InvalidData(
                                "JXL ISOBMFF: truncated large-size box header".into(),
                            ));
                        }
                        let large = u64::from_be_bytes([
                            data[pos + 8],
                            data[pos + 9],
                            data[pos + 10],
                            data[pos + 11],
                            data[pos + 12],
                            data[pos + 13],
                            data[pos + 14],
                            data[pos + 15],
                        ]) as usize;
                        (16usize, large)
                    }
                    0 => (8usize, data.len() - pos),
                    n => (8usize, n as usize),
                };
                if box_len < header_len || pos + box_len > data.len() {
                    return Err(Error::InvalidData(
                        "JXL ISOBMFF: box size overruns file".into(),
                    ));
                }
                let payload = &data[pos + header_len..pos + box_len];
                match box_type {
                    b"jxlc" => out.extend_from_slice(payload),
                    b"jxlp" => {
                        if payload.len() < 4 {
                            return Err(Error::InvalidData(
                                "JXL ISOBMFF: jxlp box too short for index".into(),
                            ));
                        }
                        out.extend_from_slice(&payload[4..]);
                    }
                    _ => {}
                }
                pos += box_len;
            }
            if out.is_empty() {
                return Err(Error::InvalidData(
                    "JXL ISOBMFF: no jxlc/jxlp codestream box found".into(),
                ));
            }
            Ok(std::borrow::Cow::Owned(out))
        }
        None => Err(Error::InvalidData(
            "not a JPEG XL file: signature mismatch".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_raw_codestream() {
        assert_eq!(detect(&[0xFF, 0x0A, 0x00]), Some(Signature::RawCodestream));
    }

    #[test]
    fn detects_isobmff() {
        let mut buf = ISOBMFF_SIGNATURE.to_vec();
        buf.push(0);
        assert_eq!(detect(&buf), Some(Signature::Isobmff));
    }

    #[test]
    fn rejects_other() {
        assert!(detect(&[0x89, 0x50, 0x4E, 0x47]).is_none());
        assert!(detect(&[]).is_none());
    }

    #[test]
    fn extracts_raw_as_borrowed() {
        let data = [0xFF, 0x0A, 0x01, 0x02, 0x03];
        let cow = extract_codestream(&data).unwrap();
        assert!(matches!(cow, std::borrow::Cow::Borrowed(_)));
        assert_eq!(&*cow, &data[..]);
    }

    #[test]
    fn extracts_isobmff_jxlc_payload() {
        // Signature box (12) + jxlc box (size=12, payload 4 bytes).
        let mut buf = ISOBMFF_SIGNATURE.to_vec();
        buf.extend_from_slice(&[0, 0, 0, 12, b'j', b'x', b'l', b'c']);
        buf.extend_from_slice(&[0xFF, 0x0A, 0x55, 0x77]);
        let cs = extract_codestream(&buf).unwrap();
        assert_eq!(&*cs, &[0xFF, 0x0A, 0x55, 0x77]);
    }

    #[test]
    fn rejects_isobmff_without_codestream() {
        let mut buf = ISOBMFF_SIGNATURE.to_vec();
        // ftyp-ish box with no jxlc/jxlp.
        buf.extend_from_slice(&[0, 0, 0, 16, b'f', b't', b'y', b'p', 0, 0, 0, 0, 0, 0, 0, 0]);
        let err = extract_codestream(&buf).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }
}
