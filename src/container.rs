//! JPEG XL file format (ISO/IEC 18181-2:2024) — the box-structured
//! container.
//!
//! A JXL file is either a raw codestream (begins with `FF 0A`) or a
//! box-structured file (18181-2 Clause 5) that begins with the 12-byte
//! JPEG XL Signature box `00 00 00 0C 4A 58 4C 20 0D 0A 87 0A`. The box
//! structure carries the codestream in exactly one `jxlc` box or one or
//! more `jxlp` partial-codestream boxes, alongside optional metadata
//! boxes: `jxll` (level), `jxli` (frame index), `Exif`, `xml `, `brob`
//! (Brotli-compressed metadata), `jumb` (JUMBF) and `jbrd` (JPEG
//! bitstream reconstruction data, see [`crate::jpeg_bitstream`]).
//!
//! This module implements the Clause 8 binary box format and the
//! Clause 9 box-type semantics: [`detect`] (signature detection),
//! [`BoxIter`] (the raw box walk), and [`JxlFile::parse`] (the typed,
//! validated view of a whole file). [`extract_codestream`] remains the
//! decoder-facing helper that yields the concatenated codestream bytes
//! regardless of wrapping.

use oxideav_core::{Error, Result};
use std::borrow::Cow;

/// Raw codestream magic: `FF 0A` (18181-2 Annex B.2 "magic numbers").
pub const RAW_CODESTREAM_SIGNATURE: [u8; 2] = [0xFF, 0x0A];

/// JPEG XL Signature box (18181-2 §9.1): exactly these 12 bytes — a box
/// of size 12, type `JXL ` (0x4A584C20), payload `0D 0A 87 0A`.
pub const ISOBMFF_SIGNATURE: [u8; 12] = [
    0x00, 0x00, 0x00, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A,
];

/// File Type box (18181-2 §9.2): exactly these 20 bytes — size 20,
/// type `ftyp`, brand `jxl `, minor version 0, compatible brand `jxl `.
pub const FILE_TYPE_BOX: [u8; 20] = [
    0x00, 0x00, 0x00, 0x14, b'f', b't', b'y', b'p', b'j', b'x', b'l', b' ', 0x00, 0x00, 0x00, 0x00,
    b'j', b'x', b'l', b' ',
];

/// Which wrapping the input uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signature {
    /// Raw codestream, `FF 0A …`.
    RawCodestream,
    /// Box-structured container, signature box `00 00 00 0C JXL␣ 0D 0A 87 0A`.
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

/// One box as laid out by 18181-2 Table 4: `LBox` (u32) + `TBox`
/// (4 bytes) + optional `XLBox` (u64, present iff `LBox == 1`) +
/// `DBox` (the remaining bytes).
#[derive(Debug, Clone, Copy)]
pub struct RawBox<'a> {
    /// The 4-byte `TBox` box type.
    pub box_type: [u8; 4],
    /// The `DBox` payload (excludes the header fields).
    pub payload: &'a [u8],
    /// Byte offset of the box header from the start of the file.
    pub offset: usize,
}

/// Iterator over the top-level boxes of a box-structured JXL file.
///
/// Yields `Err` once (then `None`) if a box header is truncated or a
/// declared box size overruns the file / is smaller than its header
/// (Table 4: `LBox >= 8` unless 0 or 1; `XLBox >= 16`).
pub struct BoxIter<'a> {
    data: &'a [u8],
    pos: usize,
    failed: bool,
}

impl<'a> BoxIter<'a> {
    /// Walk the boxes of `data` from its first byte. The caller is
    /// expected to have checked [`detect`] first; the walk itself does
    /// not require the first box to be the signature box.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            failed: false,
        }
    }
}

impl<'a> Iterator for BoxIter<'a> {
    type Item = Result<RawBox<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed || self.pos >= self.data.len() {
            return None;
        }
        let offset = self.pos;
        let rest = &self.data[self.pos..];
        if rest.len() < 8 {
            self.failed = true;
            return Some(Err(Error::InvalidData(
                "JXL file format: truncated box header".into(),
            )));
        }
        let lbox = u32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]);
        let box_type = [rest[4], rest[5], rest[6], rest[7]];
        let (header_len, box_len) = match lbox {
            0 => (8usize, rest.len()),
            1 => {
                if rest.len() < 16 {
                    self.failed = true;
                    return Some(Err(Error::InvalidData(
                        "JXL file format: truncated XLBox header".into(),
                    )));
                }
                let xl = u64::from_be_bytes([
                    rest[8], rest[9], rest[10], rest[11], rest[12], rest[13], rest[14], rest[15],
                ]);
                if xl < 16 {
                    self.failed = true;
                    return Some(Err(Error::InvalidData(
                        "JXL file format: XLBox size below 16".into(),
                    )));
                }
                if xl > rest.len() as u64 {
                    self.failed = true;
                    return Some(Err(Error::InvalidData(
                        "JXL file format: box size overruns file".into(),
                    )));
                }
                (16usize, xl as usize)
            }
            n if n < 8 => {
                self.failed = true;
                return Some(Err(Error::InvalidData(
                    "JXL file format: LBox size below 8".into(),
                )));
            }
            n => {
                if n as usize > rest.len() {
                    self.failed = true;
                    return Some(Err(Error::InvalidData(
                        "JXL file format: box size overruns file".into(),
                    )));
                }
                (8usize, n as usize)
            }
        };
        self.pos += box_len;
        Some(Ok(RawBox {
            box_type,
            payload: &rest[header_len..box_len],
            offset,
        }))
    }
}

/// Read a `Varint()` (18181-1:2024 E.4.2 / FDIS Listing 9.2) from a byte
/// cursor: little-endian base-128, 7 value bits per byte, high bit set on
/// every byte except the last, at most 63 value bits.
fn read_varint(data: &[u8], pos: &mut usize) -> Result<u64> {
    let mut value: u64 = 0;
    let mut shift = 0u32;
    loop {
        let b = *data
            .get(*pos)
            .ok_or_else(|| Error::InvalidData("JXL Varint: truncated".into()))?;
        *pos += 1;
        value += u64::from(b & 127) << shift;
        if b <= 127 {
            break;
        }
        shift += 7;
        if shift >= 63 {
            return Err(Error::InvalidData("JXL Varint: exceeds 63 bits".into()));
        }
    }
    Ok(value)
}

fn read_u32_be(data: &[u8], pos: &mut usize) -> Result<u32> {
    let s = data
        .get(*pos..*pos + 4)
        .ok_or_else(|| Error::InvalidData("JXL file format: truncated u32".into()))?;
    *pos += 4;
    Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
}

/// One entry of the Frame Index box (18181-2 Table 9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameIndexEntry {
    /// `OFFi` — offset of the start byte of this frame relative to the
    /// start byte of the previous indexed frame in the codestream (for
    /// the first entry: from the first byte of the codestream). Offsets
    /// count bytes in the concatenated codestream, not in the file.
    pub offset: u64,
    /// `Ti` — duration in ticks between the start of this frame and the
    /// start of the next indexed frame (for the last entry: to the end
    /// of the stream). A tick lasts `tick_numerator / tick_denominator`
    /// seconds.
    pub duration_ticks: u64,
    /// `Fi` — the number of presented frames after which the next
    /// indexed frame occurs (for the last entry: the number of presented
    /// frames after this one in the remainder of the stream).
    pub frames_until_next: u64,
}

/// Parsed Frame Index box (`jxli`, 18181-2 §9.8).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameIndex {
    /// `TNUM` — numerator of the tick unit.
    pub tick_numerator: u32,
    /// `TDEN` — denominator of the tick unit (non-zero; zero is
    /// rejected as ill-formed per §9.8).
    pub tick_denominator: u32,
    /// The `NF` indexed keyframes.
    pub entries: Vec<FrameIndexEntry>,
}

impl FrameIndex {
    /// Parse a Frame Index box payload per Table 9.
    pub fn parse(payload: &[u8]) -> Result<Self> {
        let mut pos = 0usize;
        let nf = read_varint(payload, &mut pos)?;
        let tick_numerator = read_u32_be(payload, &mut pos)?;
        let tick_denominator = read_u32_be(payload, &mut pos)?;
        if tick_denominator == 0 {
            return Err(Error::InvalidData(
                "JXL jxli: tick denominator is 0 (ill-formed)".into(),
            ));
        }
        if nf == 0 {
            // §9.8: "The first frame shall always be listed."
            return Err(Error::InvalidData(
                "JXL jxli: NF == 0 but the first frame shall always be listed".into(),
            ));
        }
        if nf > (payload.len() as u64) {
            // Each entry costs at least 3 bytes; cheap bomb guard before
            // allocating.
            return Err(Error::InvalidData("JXL jxli: NF overruns box".into()));
        }
        let mut entries = Vec::with_capacity(nf as usize);
        for _ in 0..nf {
            let offset = read_varint(payload, &mut pos)?;
            let duration_ticks = read_varint(payload, &mut pos)?;
            let frames_until_next = read_varint(payload, &mut pos)?;
            entries.push(FrameIndexEntry {
                offset,
                duration_ticks,
                frames_until_next,
            });
        }
        if pos != payload.len() {
            return Err(Error::InvalidData(
                "JXL jxli: trailing bytes after the last index entry".into(),
            ));
        }
        Ok(Self {
            tick_numerator,
            tick_denominator,
            entries,
        })
    }
}

/// Metadata box kind for the ordered [`JxlFile::metadata`] list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataKind {
    /// `Exif` box (§9.5): payload is a 4-byte tiff-header offset
    /// followed by the Exif payload.
    Exif,
    /// `xml ` box (§9.6): payload is a well-formed XML document
    /// (e.g. XMP metadata).
    Xml,
    /// `jumb` box (§9.4): a JUMBF superbox (ISO/IEC 19566-5). Carried
    /// opaque; its inner boxes are outside 18181-2's syntactic scope.
    Jumbf,
}

/// One metadata box, in file order.
#[derive(Debug, Clone, Copy)]
pub struct MetadataBox<'a> {
    pub kind: MetadataKind,
    /// The box payload. When `brotli_compressed` is set this is the
    /// still-compressed `brob` payload *after* the 4-byte payload box
    /// type; Brotli-decompressing it yields the equivalent plain box
    /// content (§9.7).
    pub payload: &'a [u8],
    /// Set when the box was wrapped in a Brotli-compressed `brob` box.
    pub brotli_compressed: bool,
}

impl MetadataBox<'_> {
    /// The box content with any `brob` wrapping removed (§9.7): a
    /// Brotli-compressed box "shall be treated as if it is a box of the
    /// type given by the first 4 bytes of its contents, with a contents
    /// equal to the Brotli-decompressed data obtained from the remaining
    /// bytes". `max_output` caps the decompressed size
    /// (decompression-bomb defence).
    pub fn content(&self, max_output: usize) -> Result<Cow<'_, [u8]>> {
        if !self.brotli_compressed {
            return Ok(Cow::Borrowed(self.payload));
        }
        crate::jpeg_bitstream::brotli_decompress(self.payload, max_output).map(Cow::Owned)
    }
}

/// Typed, validated view of a box-structured JPEG XL file
/// (18181-2 Clauses 5, 8 and 9).
#[derive(Debug)]
pub struct JxlFile<'a> {
    /// Level box value (`jxll`, §9.3); 5 when absent.
    pub level: u8,
    /// The complete codestream: the single `jxlc` payload (borrowed) or
    /// the concatenation of all `jxlp` partial payloads in index order
    /// (owned).
    pub codestream: Cow<'a, [u8]>,
    /// The Frame Index box (`jxli`, §9.8), if present (zero or one).
    pub frame_index: Option<FrameIndex>,
    /// The JPEG Bitstream Reconstruction Data box payload (`jbrd`,
    /// §9.11), if present. Never `brob`-wrapped (§9.7 forbids it).
    pub jbrd: Option<&'a [u8]>,
    /// `Exif` / `xml ` / `jumb` metadata boxes in file order, including
    /// `brob`-wrapped equivalents.
    pub metadata: Vec<MetadataBox<'a>>,
}

impl<'a> JxlFile<'a> {
    /// Parse and validate a box-structured JXL file.
    ///
    /// Enforces the "shall" requirements of Clause 9: signature box
    /// first, File Type box second (both byte-exact), at most one Level
    /// box and only as the third box, exactly one `jxlc` XOR one or more
    /// `jxlp` (with the §9.10 index sequence: consecutive from 0, high
    /// bit set on exactly the last), zero or one `jxli`. Unrecognized
    /// box types are skipped per Clause 5.
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        if detect(data) != Some(Signature::Isobmff) {
            return Err(Error::InvalidData(
                "JXL file format: signature box mismatch".into(),
            ));
        }
        let mut level = 5u8;
        let mut frame_index: Option<FrameIndex> = None;
        let mut jbrd: Option<&'a [u8]> = None;
        let mut metadata: Vec<MetadataBox<'a>> = Vec::new();
        let mut jxlc: Option<&'a [u8]> = None;
        let mut jxlp_parts: Vec<&'a [u8]> = Vec::new();
        let mut jxlp_done = false;

        for (idx, item) in BoxIter::new(data).enumerate() {
            let b = item?;
            match idx {
                0 => {
                    // §9.1: the signature box shall be the first box and
                    // contain exactly the 12 signature bytes (checked by
                    // detect(), which also pins the declared size 12).
                    debug_assert_eq!(&b.box_type, b"JXL ");
                    continue;
                }
                1 => {
                    // §9.2: the File Type box shall be the second box and
                    // shall contain exactly the 20 bytes of FILE_TYPE_BOX.
                    if b.box_type != *b"ftyp"
                        || b.offset != 12
                        || data.get(12..32) != Some(&FILE_TYPE_BOX[..])
                    {
                        return Err(Error::InvalidData(
                            "JXL file format: second box is not the exact File Type box".into(),
                        ));
                    }
                    continue;
                }
                _ => {}
            }
            match &b.box_type {
                b"jxll" => {
                    // §9.3: at most one; if present it shall be the third
                    // box, immediately after the File Type box.
                    if idx != 2 {
                        return Err(Error::InvalidData(
                            "JXL jxll: Level box present but not the third box".into(),
                        ));
                    }
                    if b.payload.len() != 1 {
                        return Err(Error::InvalidData(
                            "JXL jxll: Level box payload is not exactly one byte".into(),
                        ));
                    }
                    level = b.payload[0];
                }
                b"jxlc" => {
                    if jxlc.is_some() || !jxlp_parts.is_empty() {
                        return Err(Error::InvalidData(
                            "JXL file format: more than one codestream box population \
                             (exactly one jxlc XOR one or more jxlp)"
                                .into(),
                        ));
                    }
                    jxlc = Some(b.payload);
                }
                b"jxlp" => {
                    if jxlc.is_some() {
                        return Err(Error::InvalidData(
                            "JXL file format: jxlp present alongside jxlc".into(),
                        ));
                    }
                    if jxlp_done {
                        return Err(Error::InvalidData(
                            "JXL jxlp: partial codestream box after the final-index box".into(),
                        ));
                    }
                    if b.payload.len() < 4 {
                        return Err(Error::InvalidData(
                            "JXL jxlp: box too short for its index field".into(),
                        ));
                    }
                    let index = u32::from_be_bytes([
                        b.payload[0],
                        b.payload[1],
                        b.payload[2],
                        b.payload[3],
                    ]);
                    // §9.10: index modulo 2^31 is 0 for the first box and
                    // increments by 1; only the last box has index >= 2^31.
                    let seq = index & 0x7FFF_FFFF;
                    if seq as usize != jxlp_parts.len() {
                        return Err(Error::InvalidData(format!(
                            "JXL jxlp: index {} out of sequence (expected {})",
                            seq,
                            jxlp_parts.len()
                        )));
                    }
                    if index & 0x8000_0000 != 0 {
                        jxlp_done = true;
                    }
                    jxlp_parts.push(&b.payload[4..]);
                }
                b"jxli" => {
                    // §9.8: zero or one Frame Index boxes.
                    if frame_index.is_some() {
                        return Err(Error::InvalidData(
                            "JXL jxli: more than one Frame Index box".into(),
                        ));
                    }
                    frame_index = Some(FrameIndex::parse(b.payload)?);
                }
                b"jbrd" => {
                    if jbrd.is_some() {
                        return Err(Error::InvalidData(
                            "JXL jbrd: more than one JPEG Bitstream Reconstruction box".into(),
                        ));
                    }
                    jbrd = Some(b.payload);
                }
                b"Exif" => metadata.push(MetadataBox {
                    kind: MetadataKind::Exif,
                    payload: b.payload,
                    brotli_compressed: false,
                }),
                b"xml " => metadata.push(MetadataBox {
                    kind: MetadataKind::Xml,
                    payload: b.payload,
                    brotli_compressed: false,
                }),
                b"jumb" => metadata.push(MetadataBox {
                    kind: MetadataKind::Jumbf,
                    payload: b.payload,
                    brotli_compressed: false,
                }),
                b"brob" => {
                    // §9.7: first 4 payload bytes name the wrapped box
                    // type, which shall not be `brob`, shall not start
                    // with `jxl` and shall not be `jbrd`.
                    if b.payload.len() < 4 {
                        return Err(Error::InvalidData(
                            "JXL brob: box too short for its payload box type".into(),
                        ));
                    }
                    let inner: [u8; 4] = [b.payload[0], b.payload[1], b.payload[2], b.payload[3]];
                    if &inner == b"brob" || &inner == b"jbrd" || inner.starts_with(b"jxl") {
                        return Err(Error::InvalidData(format!(
                            "JXL brob: forbidden payload box type {:?}",
                            String::from_utf8_lossy(&inner)
                        )));
                    }
                    let kind = match &inner {
                        b"Exif" => Some(MetadataKind::Exif),
                        b"xml " => Some(MetadataKind::Xml),
                        b"jumb" => Some(MetadataKind::Jumbf),
                        // Unrecognized wrapped types are skipped like any
                        // other unrecognized box.
                        _ => None,
                    };
                    if let Some(kind) = kind {
                        metadata.push(MetadataBox {
                            kind,
                            payload: &b.payload[4..],
                            brotli_compressed: true,
                        });
                    }
                }
                // Clause 5: boxes with an unrecognized type shall be
                // ignored and skipped.
                _ => {}
            }
        }

        let codestream: Cow<'a, [u8]> = match (jxlc, jxlp_parts.is_empty()) {
            (Some(cs), true) => Cow::Borrowed(cs),
            (None, false) => {
                if !jxlp_done {
                    return Err(Error::InvalidData(
                        "JXL jxlp: no partial codestream box carries the final index \
                         (>= 2^31)"
                            .into(),
                    ));
                }
                Cow::Owned(jxlp_parts.concat())
            }
            (None, true) => {
                return Err(Error::InvalidData(
                    "JXL file format: no jxlc / jxlp codestream box found".into(),
                ));
            }
            (Some(_), false) => unreachable!("rejected while walking"),
        };

        Ok(Self {
            level,
            codestream,
            frame_index,
            jbrd,
            metadata,
        })
    }
}

/// Extract the codestream bytes from a JXL input regardless of wrapping.
///
/// For raw inputs this is a zero-copy slice of `data`. For box-structured
/// inputs this is [`JxlFile::parse`]'s validated codestream: the single
/// `jxlc` payload (borrowed) or the concatenation of all `jxlp` partial
/// payloads in index order (owned).
pub fn extract_codestream(data: &[u8]) -> Result<Cow<'_, [u8]>> {
    match detect(data) {
        Some(Signature::RawCodestream) => Ok(Cow::Borrowed(data)),
        Some(Signature::Isobmff) => Ok(JxlFile::parse(data)?.codestream),
        None => Err(Error::InvalidData(
            "not a JPEG XL file: signature mismatch".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_container(extra_boxes: &[u8]) -> Vec<u8> {
        let mut buf = ISOBMFF_SIGNATURE.to_vec();
        buf.extend_from_slice(&FILE_TYPE_BOX);
        buf.extend_from_slice(extra_boxes);
        buf
    }

    fn boxed(box_type: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut b = Vec::with_capacity(8 + payload.len());
        b.extend_from_slice(&u32::to_be_bytes(8 + payload.len() as u32));
        b.extend_from_slice(box_type);
        b.extend_from_slice(payload);
        b
    }

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
        assert!(matches!(cow, Cow::Borrowed(_)));
        assert_eq!(&*cow, &data[..]);
    }

    #[test]
    fn extracts_isobmff_jxlc_payload() {
        let buf = minimal_container(&boxed(b"jxlc", &[0xFF, 0x0A, 0x55, 0x77]));
        let cs = extract_codestream(&buf).unwrap();
        assert!(matches!(cs, Cow::Borrowed(_)));
        assert_eq!(&*cs, &[0xFF, 0x0A, 0x55, 0x77]);
    }

    #[test]
    fn extracts_isobmff_jxlp_payload_strips_index() {
        let buf = minimal_container(&boxed(b"jxlp", &[0x80, 0, 0, 0, 0xFF, 0x0A, 0x42, 0x42]));
        let cs = extract_codestream(&buf).unwrap();
        assert_eq!(&*cs, &[0xFF, 0x0A, 0x42, 0x42]);
    }

    #[test]
    fn extracts_isobmff_jxlp_concatenates_in_order() {
        let mut boxes = boxed(b"jxlp", &[0, 0, 0, 0, 0xFF, 0x0A]);
        boxes.extend_from_slice(&boxed(b"jxlp", &[0x80, 0, 0, 1, 0xAB, 0xCD]));
        let buf = minimal_container(&boxes);
        let cs = extract_codestream(&buf).unwrap();
        assert_eq!(&*cs, &[0xFF, 0x0A, 0xAB, 0xCD]);
    }

    #[test]
    fn rejects_jxlp_out_of_sequence() {
        let mut boxes = boxed(b"jxlp", &[0, 0, 0, 0, 0xFF, 0x0A]);
        boxes.extend_from_slice(&boxed(b"jxlp", &[0x80, 0, 0, 2, 0xAB, 0xCD]));
        let buf = minimal_container(&boxes);
        assert!(extract_codestream(&buf).is_err());
    }

    #[test]
    fn rejects_jxlp_without_final_index() {
        let buf = minimal_container(&boxed(b"jxlp", &[0, 0, 0, 0, 0xFF, 0x0A]));
        assert!(extract_codestream(&buf).is_err());
    }

    #[test]
    fn rejects_jxlp_after_final_index() {
        let mut boxes = boxed(b"jxlp", &[0x80, 0, 0, 0, 0xFF, 0x0A]);
        boxes.extend_from_slice(&boxed(b"jxlp", &[0x80, 0, 0, 1, 0xAB]));
        let buf = minimal_container(&boxes);
        assert!(extract_codestream(&buf).is_err());
    }

    #[test]
    fn rejects_mixed_jxlc_and_jxlp() {
        let mut boxes = boxed(b"jxlc", &[0xFF, 0x0A]);
        boxes.extend_from_slice(&boxed(b"jxlp", &[0x80, 0, 0, 0, 0xAB]));
        let buf = minimal_container(&boxes);
        assert!(extract_codestream(&buf).is_err());
        // And the reverse order.
        let mut boxes = boxed(b"jxlp", &[0, 0, 0, 0, 0xAB]);
        boxes.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
        let buf = minimal_container(&boxes);
        assert!(extract_codestream(&buf).is_err());
    }

    #[test]
    fn rejects_double_jxlc() {
        let mut boxes = boxed(b"jxlc", &[0xFF, 0x0A]);
        boxes.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
        let buf = minimal_container(&boxes);
        assert!(extract_codestream(&buf).is_err());
    }

    #[test]
    fn rejects_jxlp_too_short_for_index() {
        let buf = minimal_container(&boxed(b"jxlp", &[0, 0]));
        assert!(extract_codestream(&buf).is_err());
    }

    #[test]
    fn extracts_isobmff_large_size_box() {
        // LBox=1 → 64-bit XLBox follows. Carry a single 4-byte jxlc.
        let mut boxes = vec![0, 0, 0, 1, b'j', b'x', b'l', b'c'];
        // XLBox = 8 (header) + 8 (XLBox) + 4 (payload) = 20.
        boxes.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 20]);
        boxes.extend_from_slice(&[0xFF, 0x0A, 0x33, 0x44]);
        let buf = minimal_container(&boxes);
        let cs = extract_codestream(&buf).unwrap();
        assert_eq!(&*cs, &[0xFF, 0x0A, 0x33, 0x44]);
    }

    #[test]
    fn extracts_last_box_size_zero() {
        // LBox=0 → the box extends to the end of the file.
        let buf = minimal_container(&[0, 0, 0, 0, b'j', b'x', b'l', b'c', 0xFF, 0x0A, 0x99]);
        let cs = extract_codestream(&buf).unwrap();
        assert_eq!(&*cs, &[0xFF, 0x0A, 0x99]);
    }

    #[test]
    fn rejects_truncated_large_size_header() {
        let buf = minimal_container(&[0, 0, 0, 1, b'j', b'x', b'l', b'c', 0, 0, 0]);
        assert!(extract_codestream(&buf).is_err());
    }

    #[test]
    fn rejects_box_overrunning_file() {
        let buf = minimal_container(&[0, 0, 4, 0, b'j', b'x', b'l', b'c']);
        assert!(extract_codestream(&buf).is_err());
    }

    #[test]
    fn rejects_lbox_below_8() {
        let buf = minimal_container(&[0, 0, 0, 7, b'j', b'x', b'l', b'c']);
        assert!(extract_codestream(&buf).is_err());
    }

    #[test]
    fn rejects_isobmff_without_codestream() {
        let buf = minimal_container(&boxed(b"free", &[0; 4]));
        assert!(extract_codestream(&buf).is_err());
    }

    #[test]
    fn rejects_wrong_second_box() {
        // Signature box followed by something other than the exact ftyp.
        let mut buf = ISOBMFF_SIGNATURE.to_vec();
        buf.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
        assert!(extract_codestream(&buf).is_err());
        // Right type, wrong brand bytes.
        let mut wrong_ftyp = FILE_TYPE_BOX;
        wrong_ftyp[8] = b'a';
        let mut buf = ISOBMFF_SIGNATURE.to_vec();
        buf.extend_from_slice(&wrong_ftyp);
        buf.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
        assert!(extract_codestream(&buf).is_err());
    }

    #[test]
    fn level_box_parsed_third() {
        let mut boxes = boxed(b"jxll", &[10]);
        boxes.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
        let buf = minimal_container(&boxes);
        let f = JxlFile::parse(&buf).unwrap();
        assert_eq!(f.level, 10);
    }

    #[test]
    fn level_defaults_to_5() {
        let buf = minimal_container(&boxed(b"jxlc", &[0xFF, 0x0A]));
        let f = JxlFile::parse(&buf).unwrap();
        assert_eq!(f.level, 5);
    }

    #[test]
    fn rejects_level_box_not_third() {
        let mut boxes = boxed(b"jxlc", &[0xFF, 0x0A]);
        boxes.extend_from_slice(&boxed(b"jxll", &[5]));
        let buf = minimal_container(&boxes);
        assert!(JxlFile::parse(&buf).is_err());
    }

    #[test]
    fn frame_index_round_trip() {
        // NF=2, TNUM=1, TDEN=30, entries (0, 3, 0) and (200, 1, 5).
        let mut payload = vec![2u8];
        payload.extend_from_slice(&1u32.to_be_bytes());
        payload.extend_from_slice(&30u32.to_be_bytes());
        payload.extend_from_slice(&[0, 3, 0]);
        payload.extend_from_slice(&[0xC8, 0x01, 1, 5]); // 200 as Varint = C8 01
        let mut boxes = boxed(b"jxli", &payload);
        boxes.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
        let buf = minimal_container(&boxes);
        let f = JxlFile::parse(&buf).unwrap();
        let fi = f.frame_index.unwrap();
        assert_eq!(fi.tick_numerator, 1);
        assert_eq!(fi.tick_denominator, 30);
        assert_eq!(
            fi.entries,
            vec![
                FrameIndexEntry {
                    offset: 0,
                    duration_ticks: 3,
                    frames_until_next: 0
                },
                FrameIndexEntry {
                    offset: 200,
                    duration_ticks: 1,
                    frames_until_next: 5
                },
            ]
        );
    }

    #[test]
    fn rejects_frame_index_zero_tick_denominator() {
        let mut payload = vec![1u8];
        payload.extend_from_slice(&1u32.to_be_bytes());
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&[0, 1, 0]);
        assert!(FrameIndex::parse(&payload).is_err());
    }

    #[test]
    fn collects_metadata_boxes_in_order() {
        let mut boxes = boxed(b"Exif", &[0, 0, 0, 0, b'M', b'M']);
        boxes.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
        boxes.extend_from_slice(&boxed(b"xml ", b"<x/>"));
        let buf = minimal_container(&boxes);
        let f = JxlFile::parse(&buf).unwrap();
        assert_eq!(f.metadata.len(), 2);
        assert_eq!(f.metadata[0].kind, MetadataKind::Exif);
        assert!(!f.metadata[0].brotli_compressed);
        assert_eq!(f.metadata[1].kind, MetadataKind::Xml);
        assert_eq!(f.metadata[1].payload, b"<x/>");
    }

    #[test]
    fn brob_wrapped_metadata_recorded() {
        let mut payload = b"xml ".to_vec();
        payload.extend_from_slice(&[1, 2, 3]); // compressed bytes (opaque here)
        let mut boxes = boxed(b"brob", &payload);
        boxes.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
        let buf = minimal_container(&boxes);
        let f = JxlFile::parse(&buf).unwrap();
        assert_eq!(f.metadata.len(), 1);
        assert_eq!(f.metadata[0].kind, MetadataKind::Xml);
        assert!(f.metadata[0].brotli_compressed);
        assert_eq!(f.metadata[0].payload, &[1, 2, 3]);
    }

    #[test]
    fn rejects_brob_forbidden_payload_types() {
        for inner in [&b"brob"[..], b"jbrd", b"jxlc", b"jxlp", b"jxll", b"jxli"] {
            let mut payload = inner.to_vec();
            payload.push(0);
            let mut boxes = boxed(b"brob", &payload);
            boxes.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
            let buf = minimal_container(&boxes);
            assert!(
                JxlFile::parse(&buf).is_err(),
                "brob wrapping {:?} must be rejected",
                String::from_utf8_lossy(inner)
            );
        }
    }

    #[test]
    fn jbrd_captured_and_duplicate_rejected() {
        let mut boxes = boxed(b"jbrd", &[0xAA, 0xBB]);
        boxes.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
        let buf = minimal_container(&boxes);
        let f = JxlFile::parse(&buf).unwrap();
        assert_eq!(f.jbrd, Some(&[0xAA, 0xBB][..]));

        let mut boxes = boxed(b"jbrd", &[0xAA]);
        boxes.extend_from_slice(&boxed(b"jbrd", &[0xBB]));
        boxes.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A]));
        let buf = minimal_container(&boxes);
        assert!(JxlFile::parse(&buf).is_err());
    }

    #[test]
    fn unknown_boxes_skipped() {
        let mut boxes = boxed(b"abcd", &[9; 7]);
        boxes.extend_from_slice(&boxed(b"jxlc", &[0xFF, 0x0A, 0x01]));
        boxes.extend_from_slice(&boxed(b"wxyz", &[]));
        let buf = minimal_container(&boxes);
        let f = JxlFile::parse(&buf).unwrap();
        assert_eq!(&*f.codestream, &[0xFF, 0x0A, 0x01]);
    }
}
