//! JPEG Bitstream Reconstruction Data (ISO/IEC 18181-2:2024 §9.11).
//!
//! The `jbrd` box carries everything a decoder needs — beyond the JPEG XL
//! codestream itself — to regenerate the *original* JPEG file byte-exactly
//! from a losslessly recompressed JPEG XL file: the JPEG's marker
//! sequence, entropy-coding side data (Huffman code definitions, scan
//! scripts, restart interval, encoder quirks like extra ZRL symbols and
//! non-zero padding bits) and the verbatim bytes of segments that have no
//! codestream equivalent (unknown APPn payloads, COM segments,
//! unrecognized data, trailing garbage), the latter group Brotli-
//! compressed (IETF RFC 7932) at the end of the box.
//!
//! This module implements the §9.11 bundle parse (Tables 11 to 18) over
//! the 18181-1 B.2 bit primitives ([`crate::bitreader::BitReader`]) and
//! the splitting of the decompressed trailing stream. The Annex A
//! reconstruction procedure that consumes this data lives in
//! [`crate::jpeg_reconstruct`].

use crate::bitreader::{BitReader, U32Dist};
use oxideav_core::{Error, Result};

/// Decompress a Brotli stream (IETF RFC 7932, the format 18181-2 §9.7 and
/// §9.11 mandate) capping the output at `max_output` bytes as a
/// decompression-bomb defence.
pub fn brotli_decompress(data: &[u8], max_output: usize) -> Result<Vec<u8>> {
    compcol::vec::decompress_to_vec_capped::<compcol::brotli::Brotli>(data, max_output as u64)
        .map_err(|e| match e {
            compcol::Error::OutputLimitExceeded => Error::InvalidData(format!(
                "JXL Brotli stream: decompressed output exceeds the {max_output}-byte cap"
            )),
            other => Error::InvalidData(format!("JXL Brotli stream: {other:?}")),
        })
}

/// Hard cap for a single decompressed metadata / trailing-data stream.
/// Every length the trailing stream can legitimately carry is bounded by
/// the Table 11 fields (16-bit segment lengths, 22-bit tail length), so
/// 64 MiB is far beyond any conforming stream.
pub const MAX_BROTLI_OUTPUT: usize = 64 << 20;

/// `AppMarker` bundle (Table 12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppMarker {
    /// Marker payload disposition: 0 = unknown (verbatim bytes in the
    /// trailing stream), 1 = ICC profile fragment, 2 = Exif metadata,
    /// 3 = XMP metadata.
    pub kind: u32,
    /// Number of bytes of the segment after its `0xFF` byte.
    pub length: u32,
}

/// `QuantTable` bundle (Table 13).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantTable {
    /// `Pq` — 0 for 8-bit factors, 1 for 16-bit factors.
    pub precision: u32,
    /// `Tq` — quantization table destination identifier.
    pub index: u32,
    /// Whether this table ends its DQT segment.
    pub is_last: bool,
}

/// `HuffmanCode` bundle (Table 14).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HuffmanCode {
    /// `Tc` — false for a DC (or lossless) table, true for an AC table.
    pub is_ac: bool,
    /// `Th` — Huffman table destination identifier.
    pub id: u32,
    /// Whether this code ends its DHT segment.
    pub is_last: bool,
    /// Number of codes of each length 0..=16 — 17 slots, wire-arbitrated
    /// on real reconstruction data: with 16 slots the first table's
    /// Kraft sum is 1/2 (an impossible output of any Huffman optimiser)
    /// and every following table misparses; with a leading length-0 slot
    /// the sum is exactly 1 and each following table's header lands on
    /// conforming values to the bit. The A.3 storage rule maps
    /// `counts[1..=16]` to the DHT `L_i` bytes with the last non-zero
    /// count decremented by one before storing (dropping the sentinel
    /// code, see `values`).
    pub counts: [u32; 17],
    /// The symbol values, `sum(counts)` of them. The value 256 (the
    /// maximum the Table 14 selector reaches, one past the 8-bit symbol
    /// range) marks the sentinel symbol that exists in the code but is
    /// not emitted into the DHT segment.
    pub values: Vec<u32>,
}

/// `ScanComponentInfo` bundle (Table 16).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanComponentInfo {
    /// Index into the frame's component list.
    pub comp_idx: u32,
    /// `Ta` — AC entropy table selector.
    pub ac_tbl_idx: u32,
    /// `Td` — DC entropy table selector.
    pub dc_tbl_idx: u32,
}

/// `ScanInfo` bundle (Table 15).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanInfo {
    /// `Ss` — start of spectral selection.
    pub ss: u32,
    /// `Se` — end of spectral selection.
    pub se: u32,
    /// `Al` — successive approximation bit position, low.
    pub al: u32,
    /// `Ah` — successive approximation bit position, high.
    pub ah: u32,
    /// The scan's components, in scan order.
    pub components: Vec<ScanComponentInfo>,
    /// Hint for progressive decoding; not needed to rebuild the bytes.
    pub last_needed_pass: u32,
}

/// `ExtraZeroRun` bundle (Table 18).
///
/// Table 18 prints the second field's name as `run_length`, but the
/// Annex A.6 procedure addresses this bundle's members as `ezr.num_runs`
/// and matches its *block index* against the currently serialized block
/// index — the field is used as the block index of the block before
/// which `num_runs` extra "ZRL" symbols are emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtraZeroRun {
    /// Number of extra "ZRL" (run of 16 zeros) symbols to emit.
    pub num_runs: u32,
    /// Block index (in the current scan) before which they are emitted.
    pub block_idx: u32,
}

/// `ScanMoreInfo` bundle (Table 17).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanMoreInfo {
    /// Block indices (in the scan) at which "Encode_EOBRUN" is invoked
    /// before encoding the block (A.6).
    pub reset_points: Vec<u32>,
    /// Extra "ZRL" emissions (A.6).
    pub extra_zero_runs: Vec<ExtraZeroRun>,
}

/// The verbatim byte streams recovered from the Brotli-compressed tail of
/// the `jbrd` box (§9.11 final paragraphs).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TrailingData {
    /// For each `app_marker[i]` with kind 0, its `length` verbatim bytes
    /// (the APPn segment after its `0xFF` byte); `None` for kinds 1..=3,
    /// whose payloads come from other boxes / the codestream.
    pub app_data: Vec<Option<Vec<u8>>>,
    /// For each COM marker, `com_length[i]` verbatim bytes (the segment
    /// after its `0xFF 0xFE` bytes).
    pub com_data: Vec<Vec<u8>>,
    /// For each `0xFF` entry of the marker array, `intermarker_length[i]`
    /// verbatim bytes of unrecognized data.
    pub intermarker_data: Vec<Vec<u8>>,
    /// `tail_data_length` bytes appended after the EOI marker.
    pub tail_data: Vec<u8>,
}

/// Parsed JPEG Bitstream Reconstruction Data (`JPEGBitstream` bundle,
/// Table 11, plus the decompressed trailing streams).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JpegBitstreamData {
    /// Whether the JPEG is a 1-component greyscale image.
    pub is_grey: bool,
    /// The marker array: each element is `0xC0 + Bits(6)`, ending with
    /// the `0xD9` (EOI) entry. Segment types are dispatched from this
    /// array by the Annex A procedure.
    pub markers: Vec<u8>,
    /// One entry per APPn marker (`0xE0..=0xEF`) in `markers` order.
    pub app_markers: Vec<AppMarker>,
    /// One entry per COM marker (`0xFE`): the segment length after the
    /// `0xFF 0xFE` bytes.
    pub com_lengths: Vec<u32>,
    /// The QuantTable entities serialized across the DQT segments.
    pub quant_tables: Vec<QuantTable>,
    /// Component-set type: 0 = single component id 1; 1 = ids {1,2,3};
    /// 2 = ids {'R','G','B'}; 3 = explicit ids.
    pub comp_type: u32,
    /// Frame component identifiers (`Ci` in the SOF segment).
    pub component_ids: Vec<u32>,
    /// Per-component quantization table selector (`Tqi` in the SOF).
    pub component_q_idx: Vec<u32>,
    /// The HuffmanCode entities serialized across the DHT segments.
    pub huffman_codes: Vec<HuffmanCode>,
    /// One entry per SOS marker (`0xDA`) in `markers` order.
    pub scan_infos: Vec<ScanInfo>,
    /// DRI restart interval (0 when no `0xDD` marker occurs).
    pub restart_interval: u32,
    /// One entry per SOS marker, parallel to `scan_infos`.
    pub scan_more_infos: Vec<ScanMoreInfo>,
    /// One length per `0xFF` marker entry (unrecognized data).
    pub intermarker_lengths: Vec<u32>,
    /// Number of trailing bytes after EOI.
    pub tail_data_length: u32,
    /// Whether entropy-coded segments end with recorded (rather than
    /// all-zero) padding bits.
    pub has_padding: bool,
    /// The recorded padding bits, consumed in order by A.6.
    pub padding_bits: Vec<bool>,
    /// The decompressed trailing byte streams.
    pub trailing: TrailingData,
}

const U32_D: fn(u32) -> U32Dist = U32Dist::Val;

/// Selector table shared by `num_reset_points` and `num_extra_zero_runs`
/// (Table 17).
const COUNT_DIST: [U32Dist; 4] = [
    U32Dist::Val(0),
    U32Dist::BitsOffset(2, 1),
    U32Dist::BitsOffset(4, 4),
    U32Dist::BitsOffset(16, 20),
];

/// Selector table shared by `reset_point` (Table 17) and the extra zero
/// run block index (Table 18).
const BLOCK_IDX_DIST: [U32Dist; 4] = [
    U32Dist::Val(0),
    U32Dist::BitsOffset(3, 1),
    U32Dist::BitsOffset(5, 9),
    U32Dist::BitsOffset(28, 41),
];

impl JpegBitstreamData {
    /// Parse a `jbrd` box payload: the Table 11 bundle followed by the
    /// single Brotli stream carrying the verbatim data bytes.
    pub fn parse(payload: &[u8]) -> Result<Self> {
        let mut r = BitReader::new(payload);

        let is_grey = r.read_bool()?;

        // Marker array: 0xC0 + Bits(6) each, until 0xD9 (EOI).
        let mut markers: Vec<u8> = Vec::new();
        loop {
            let m = 0xC0u8 + r.read_bits(6)? as u8;
            markers.push(m);
            if m == 0xD9 {
                break;
            }
            if markers.len() > payload.len().saturating_mul(8) {
                return Err(Error::InvalidData(
                    "JXL jbrd: marker array never reaches EOI".into(),
                ));
            }
        }
        let num_app_markers = markers
            .iter()
            .filter(|&&m| (0xE0..=0xEF).contains(&m))
            .count();
        let num_com_markers = markers.iter().filter(|&&m| m == 0xFE).count();
        let num_scans = markers.iter().filter(|&&m| m == 0xDA).count();
        let num_intermarker = markers.iter().filter(|&&m| m == 0xFF).count();
        let has_dri = markers.contains(&0xDD);

        let mut app_markers = Vec::with_capacity(num_app_markers);
        for _ in 0..num_app_markers {
            let kind = r.read_u32([
                U32_D(0),
                U32_D(1),
                U32Dist::BitsOffset(1, 2),
                U32Dist::BitsOffset(2, 4),
            ])?;
            let length = 1 + r.read_bits(16)?;
            app_markers.push(AppMarker { kind, length });
        }

        let mut com_lengths = Vec::with_capacity(num_com_markers);
        for _ in 0..num_com_markers {
            com_lengths.push(1 + r.read_bits(16)?);
        }

        let num_quant_tables = 1 + r.read_bits(2)?;
        let mut quant_tables = Vec::with_capacity(num_quant_tables as usize);
        for _ in 0..num_quant_tables {
            quant_tables.push(QuantTable {
                precision: r.read_bits(1)?,
                index: r.read_bits(2)?,
                is_last: r.read_bool()?,
            });
        }

        let comp_type = r.read_bits(2)?;
        let num_comp = if comp_type == 3 {
            1 + r.read_bits(2)?
        } else if comp_type == 0 {
            1
        } else {
            3
        };
        let component_ids: Vec<u32> = if comp_type == 3 {
            (0..num_comp)
                .map(|_| r.read_bits(8))
                .collect::<Result<_>>()?
        } else {
            match comp_type {
                0 => vec![1],
                1 => vec![1, 2, 3],
                2 => vec![u32::from(b'R'), u32::from(b'G'), u32::from(b'B')],
                _ => unreachable!(),
            }
        };
        let component_q_idx: Vec<u32> = (0..num_comp)
            .map(|_| r.read_bits(2))
            .collect::<Result<_>>()?;

        let num_huff = r.read_u32([
            U32_D(4),
            U32Dist::BitsOffset(3, 2),
            U32Dist::BitsOffset(4, 10),
            U32Dist::BitsOffset(6, 26),
        ])?;
        let mut huffman_codes = Vec::with_capacity(num_huff as usize);
        for _ in 0..num_huff {
            let is_ac = r.read_bool()?;
            let id = r.read_bits(2)?;
            let is_last = r.read_bool()?;
            let mut counts = [0u32; 17];
            for c in counts.iter_mut() {
                *c = r.read_u32([
                    U32_D(0),
                    U32_D(1),
                    U32Dist::BitsOffset(3, 2),
                    U32Dist::Bits(8),
                ])?;
            }
            let total: u32 = counts.iter().sum();
            let mut values = Vec::with_capacity(total as usize);
            for _ in 0..total {
                values.push(r.read_u32([
                    U32Dist::Bits(2),
                    U32Dist::BitsOffset(2, 4),
                    U32Dist::BitsOffset(4, 8),
                    U32Dist::BitsOffset(8, 1),
                ])?);
            }
            huffman_codes.push(HuffmanCode {
                is_ac,
                id,
                is_last,
                counts,
                values,
            });
        }

        let mut scan_infos = Vec::with_capacity(num_scans);
        for _ in 0..num_scans {
            let num_comps = 1 + r.read_bits(2)?;
            let ss = r.read_bits(6)?;
            let se = r.read_bits(6)?;
            let al = r.read_bits(4)?;
            let ah = r.read_bits(4)?;
            let components: Vec<ScanComponentInfo> = (0..num_comps)
                .map(|_| -> Result<ScanComponentInfo> {
                    Ok(ScanComponentInfo {
                        comp_idx: r.read_bits(2)?,
                        ac_tbl_idx: r.read_bits(2)?,
                        dc_tbl_idx: r.read_bits(2)?,
                    })
                })
                .collect::<Result<_>>()?;
            let last_needed_pass =
                r.read_u32([U32_D(0), U32_D(1), U32_D(2), U32Dist::BitsOffset(3, 3)])?;
            scan_infos.push(ScanInfo {
                ss,
                se,
                al,
                ah,
                components,
                last_needed_pass,
            });
        }

        let restart_interval = if has_dri { r.read_bits(16)? } else { 0 };

        let mut scan_more_infos = Vec::with_capacity(num_scans);
        for _ in 0..num_scans {
            let num_reset_points = r.read_u32(COUNT_DIST)?;
            let reset_points: Vec<u32> = (0..num_reset_points)
                .map(|_| r.read_u32(BLOCK_IDX_DIST))
                .collect::<Result<_>>()?;
            let num_extra_zero_runs = r.read_u32(COUNT_DIST)?;
            let extra_zero_runs: Vec<ExtraZeroRun> = (0..num_extra_zero_runs)
                .map(|_| -> Result<ExtraZeroRun> {
                    Ok(ExtraZeroRun {
                        num_runs: r.read_u32([
                            U32_D(1),
                            U32Dist::BitsOffset(2, 2),
                            U32Dist::BitsOffset(4, 5),
                            U32Dist::BitsOffset(8, 20),
                        ])?,
                        block_idx: r.read_u32(BLOCK_IDX_DIST)?,
                    })
                })
                .collect::<Result<_>>()?;
            scan_more_infos.push(ScanMoreInfo {
                reset_points,
                extra_zero_runs,
            });
        }

        let intermarker_lengths: Vec<u32> = (0..num_intermarker)
            .map(|_| r.read_bits(16))
            .collect::<Result<_>>()?;

        let tail_data_length = r.read_u32([
            U32_D(0),
            U32Dist::BitsOffset(8, 1),
            U32Dist::BitsOffset(16, 257),
            U32Dist::BitsOffset(22, 65793),
        ])?;

        let has_padding = r.read_bool()?;
        let mut padding_bits = Vec::new();
        if has_padding {
            let nbit = r.read_bits(24)?;
            if nbit as usize > payload.len().saturating_mul(8) {
                return Err(Error::InvalidData(
                    "JXL jbrd: padding bit count overruns box".into(),
                ));
            }
            padding_bits.reserve(nbit as usize);
            for _ in 0..nbit {
                padding_bits.push(r.read_bool()?);
            }
        }

        // The single Brotli stream follows at the next byte boundary.
        //
        // Hostile-input fence (r454 fuzz): the exact decompressed size
        // is already determined by the Table 11 fields — the sum of
        // every verbatim segment length plus the post-EOI tail — and
        // the split loop below rejects any surplus anyway. Cap the
        // decompressor at that exact total (bounded by
        // MAX_BROTLI_OUTPUT) instead of always allowing the full
        // 64 MiB, so a tiny hostile payload cannot force a
        // multi-megabyte transient allocation per parse.
        let expected_total: u64 = app_markers
            .iter()
            .filter(|am| am.kind == 0)
            .map(|am| am.length as u64)
            .sum::<u64>()
            + com_lengths.iter().map(|&l| l as u64).sum::<u64>()
            + intermarker_lengths.iter().map(|&l| l as u64).sum::<u64>()
            + tail_data_length as u64;
        if expected_total > MAX_BROTLI_OUTPUT as u64 {
            return Err(Error::InvalidData(format!(
                "JXL jbrd: declared trailing streams total {expected_total} bytes \
                 (cap {MAX_BROTLI_OUTPUT})"
            )));
        }
        let brotli_offset = r.bits_read().div_ceil(8);
        let decompressed = brotli_decompress(&payload[brotli_offset..], expected_total as usize)?;

        // Split it: unknown-APPn payloads, COM payloads, unrecognized
        // segment data, then the post-EOI tail (§9.11).
        let mut pos = 0usize;
        let mut take = |n: usize, what: &str| -> Result<Vec<u8>> {
            let s = decompressed.get(pos..pos + n).ok_or_else(|| {
                Error::InvalidData(format!(
                    "JXL jbrd: Brotli stream too short for {what} ({n} bytes at {pos})"
                ))
            })?;
            pos += n;
            Ok(s.to_vec())
        };
        let mut app_data = Vec::with_capacity(app_markers.len());
        for am in &app_markers {
            if am.kind == 0 {
                app_data.push(Some(take(am.length as usize, "app_data")?));
            } else {
                app_data.push(None);
            }
        }
        let mut com_data = Vec::with_capacity(com_lengths.len());
        for &len in &com_lengths {
            com_data.push(take(len as usize, "com_data")?);
        }
        let mut intermarker_data = Vec::with_capacity(intermarker_lengths.len());
        for &len in &intermarker_lengths {
            intermarker_data.push(take(len as usize, "intermarker_data")?);
        }
        let tail_data = take(tail_data_length as usize, "tail_data")?;
        if pos != decompressed.len() {
            return Err(Error::InvalidData(format!(
                "JXL jbrd: {} unconsumed bytes after the trailing streams",
                decompressed.len() - pos
            )));
        }

        Ok(Self {
            is_grey,
            markers,
            app_markers,
            com_lengths,
            quant_tables,
            comp_type,
            component_ids,
            component_q_idx,
            huffman_codes,
            scan_infos,
            restart_interval,
            scan_more_infos,
            intermarker_lengths,
            tail_data_length,
            has_padding,
            padding_bits,
            trailing: TrailingData {
                app_data,
                com_data,
                intermarker_data,
                tail_data,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brotli_round_trip() {
        let raw = b"the quick brown fox jumps over the lazy dog";
        let compressed =
            compcol::vec::compress_to_vec::<compcol::brotli::Brotli>(raw).expect("compress");
        let out = brotli_decompress(&compressed, 1024).unwrap();
        assert_eq!(out, raw);
    }

    #[test]
    fn brotli_cap_enforced() {
        let raw = vec![0u8; 4096];
        let compressed =
            compcol::vec::compress_to_vec::<compcol::brotli::Brotli>(&raw).expect("compress");
        assert!(brotli_decompress(&compressed, 16).is_err());
    }
}
