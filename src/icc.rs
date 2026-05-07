//! ICC profile decoder — ISO/IEC 18181-1:2024 Annex E.4.
//!
//! When `metadata.colour_encoding.want_icc == true`, the codestream
//! carries an embedded ICC profile encoded with a JPEG-XL-specific
//! entropy + bytecode scheme. This module decodes that stream into the
//! final ICC byte sequence.
//!
//! The decode is in three layers:
//!
//! 1. [`decode_encoded_icc_stream`] — entry point. Reads `enc_size`
//!    (`U64()` per Listing 9.1) from the JXL codestream, then reads
//!    `41` pre-clustered distributions per [`crate::modular_fdis::EntropyStream::read`] (Annex C.1),
//!    then `enc_size` integers in `[0, 255]` using
//!    [`crate::ans::hybrid::HybridUintState`] driven by the
//!    `IccContext(i, prev_byte, prev_prev_byte)` 41-context function
//!    in [`icc_context`]. The result is the *encoded* ICC byte stream.
//!
//! 2. After step 1, the encoded ICC byte stream is split into
//!    `output_size` (Varint) + `commands_size` (Varint) prefix, then a
//!    command stream of `commands_size` bytes, then a data stream of
//!    the remaining bytes. (E.4.2 + Table E.10.)
//!
//! 3. [`reconstruct_icc_profile`] walks the command stream against the
//!    data stream, materialising the three concatenated parts of the
//!    ICC profile output: ICC header (E.4.3), ICC tag list (E.4.4),
//!    and main content (E.4.5). The result is exactly `output_size`
//!    bytes long.
//!
//! The `Varint` layer (E.4.2 second listing) is independent from JXL's
//! `U64()` and is implemented as [`varint_read`] over a byte cursor.
//!
//! ## Spec divergence audit (round 6)
//!
//! Spec listings transcribed verbatim into the code below, with the
//! following explicit reading choices:
//!
//! * E.4.1 `IccContext`: spec listing computes `p1` then `p2`, returning
//!   `1 + p1 + p2 * 8`. The `i <= 128` early-return path means contexts
//!   for the first 129 output bytes are always 0, regardless of `b1` /
//!   `b2`. For `i == 0` the context is also 0 because `b1 == b2 == 0`
//!   (initialised) which falls through to the `b1 < 16 ? p1=2: ...`
//!   path; but the `i <= 128` early return covers that explicitly.
//!
//! * E.4.4 `Varint() == 0` indicates the tag list is finalised. The
//!   spec's listing reads `v = Varint(); num_tags = v - 1; if (num_tags
//!   == -1) { ... }` — `num_tags` is treated as a *signed* value here
//!   (interpreting the 64-bit unsigned underflow when `v == 0`) so that
//!   `0 - 1 == 0xFFFF_FFFF_FFFF_FFFF` is the sentinel. We compare on
//!   `v == 0` directly to make the intent unambiguous.
//!
//! * E.4.5 main content `command == 4` Nth-order predictor: the inner
//!   loop reads `prev[j]` from earlier output positions. The spec's
//!   "stride * 4 < number of bytes already output" guard requires the
//!   predictor not to address before the start of the ICC profile
//!   output. We enforce that strictly.
//!
//! ## Wall
//!
//! ICC.1:2010 / ISO 15076-1 was used only to establish that bytes 36-39
//! of the decoded output are the magic string `"acsp"` for a valid
//! display ICC profile, and that bytes 0-3 are the profile size as a
//! big-endian u32. No ICC vendor's source / lcms source consulted.

use oxideav_core::{Error, Result};

use crate::ans::hybrid::HybridUintState;
use crate::bitreader::BitReader;
use crate::modular_fdis::{decode_uint_in_with_dist_pub, EntropyStream};

/// Fixed number of pre-clustered distributions for the ICC entropy
/// stream — Annex E.4.1 first paragraph.
pub const ICC_NUM_DIST: usize = 41;

/// Sanity bound on the encoded byte-stream size. The spec doesn't cap
/// `enc_size` directly; ICC profiles in practice are < 1 MB. We accept
/// up to 64 MB to give very generous headroom while still rejecting
/// malicious gigabyte-class allocations.
pub const MAX_ENC_SIZE: u64 = 64 * 1024 * 1024;

/// Sanity bound on the *output* ICC profile size (the `output_size`
/// Varint inside the encoded stream). 64 MB is well above any realistic
/// ICC profile; ICC.1 itself caps profile size to 4 GB via the
/// big-endian `u32` size field, but allocating that much would be
/// catastrophic in a bitstream-validation path.
pub const MAX_OUTPUT_SIZE: u64 = 64 * 1024 * 1024;

/// Compute the ICC entropy-decode context per E.4.1 listing.
///
/// `i` is the *byte* index within the encoded ICC stream (0-based).
/// `b1` is the previous byte (or 0 if this is the first byte); `b2` is
/// the byte before that (or 0).
///
/// Returns a context index in `[0, 41)`.
pub fn icc_context(i: usize, b1: u8, b2: u8) -> u32 {
    if i <= 128 {
        return 0;
    }
    let p1: u32 = if b1.is_ascii_lowercase() || b1.is_ascii_uppercase() {
        0
    } else if b1 == b'.' || b1 == b',' {
        1
    } else if b1 <= 1 {
        2 + (b1 as u32)
    } else if b1 > 1 && b1 < 16 {
        4
    } else if b1 > 240 && b1 < 255 {
        5
    } else if b1 == 255 {
        6
    } else {
        7
    };
    let p2: u32 = if b2.is_ascii_lowercase() || b2.is_ascii_uppercase() {
        0
    } else if b2 == b'.' || b2 == b',' {
        1
    } else if b2 < 16 {
        2
    } else if b2 > 240 {
        3
    } else {
        4
    };
    1 + p1 + p2 * 8
}

/// Decode the encoded ICC byte stream from a JXL codestream's bit
/// reader, per Annex E.4.1.
///
/// Reads `enc_size = U64()`, then 41 pre-clustered distributions
/// (`EntropyStream::read(br, 41)`), then `enc_size` bytes via
/// `DecodeHybridVarLenUint(IccContext(...))`. Returns the encoded ICC
/// byte stream (length `enc_size`).
///
/// The caller is responsible for then handing the returned bytes to
/// [`reconstruct_icc_profile`] to obtain the final ICC profile.
pub fn decode_encoded_icc_stream(br: &mut BitReader<'_>) -> Result<Vec<u8>> {
    let enc_size = br.read_u64()?;
    if enc_size == 0 {
        return Err(Error::InvalidData(
            "JXL ICC: enc_size == 0 (no encoded byte stream)".into(),
        ));
    }
    if enc_size > MAX_ENC_SIZE {
        return Err(Error::InvalidData(format!(
            "JXL ICC: enc_size {enc_size} exceeds cap {MAX_ENC_SIZE}"
        )));
    }
    if enc_size > br.bits_remaining() as u64 {
        return Err(Error::InvalidData(
            "JXL ICC: enc_size exceeds remaining input bits".into(),
        ));
    }
    let enc_size = enc_size as usize;

    let mut entropy = EntropyStream::read(br, ICC_NUM_DIST)?;
    entropy.read_ans_state_init(br)?;

    let mut hybrid = HybridUintState::new(entropy.lz77, entropy.lz_len_conf);

    let mut bytes = Vec::with_capacity(enc_size);
    let mut prev_byte: u8 = 0;
    let mut prev_prev_byte: u8 = 0;
    for i in 0..enc_size {
        let ctx = icc_context(i, prev_byte, prev_prev_byte);
        let v = decode_uint_in_with_dist_pub(&mut hybrid, &mut entropy, br, ctx, 0)?;
        if v > 255 {
            return Err(Error::InvalidData(format!(
                "JXL ICC: decoded byte value {v} > 255 at index {i}"
            )));
        }
        let byte = v as u8;
        bytes.push(byte);
        prev_prev_byte = prev_byte;
        prev_byte = byte;
    }
    Ok(bytes)
}

/// Read a Varint per E.4.2 listing (separate from JXL's U64). Reads
/// 7-bit groups from the byte cursor with continuation bit set on every
/// byte except the last; up to 9 bytes total (63 bits of data + the
/// final low-bit-cleared byte).
///
/// Returns `(value, bytes_consumed)`. Errors if the stream ends before
/// a terminator byte is found, or if `shift` would exceed 56 (which
/// would overflow the 64-bit accumulator beyond the spec's 63-bit
/// guarantee).
pub fn varint_read(stream: &[u8], pos: &mut usize) -> Result<u64> {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        if shift > 56 {
            return Err(Error::InvalidData(
                "JXL ICC: Varint shift > 56 (would overflow 64 bits)".into(),
            ));
        }
        let b = *stream.get(*pos).ok_or_else(|| {
            Error::InvalidData("JXL ICC: Varint truncated (no terminator)".into())
        })?;
        *pos += 1;
        value |= ((b & 0x7F) as u64) << shift;
        if b <= 0x7F {
            return Ok(value);
        }
        shift += 7;
    }
}

/// Reconstruct the ICC profile from the encoded byte stream produced by
/// [`decode_encoded_icc_stream`]. Returns exactly `output_size` bytes.
///
/// Walks the command stream against the data stream per E.4.3 (header),
/// E.4.4 (tag list), and E.4.5 (main content), in that order.
pub fn reconstruct_icc_profile(encoded: &[u8]) -> Result<Vec<u8>> {
    // Read output_size and commands_size at the start of the encoded
    // byte stream (E.4.2).
    let mut pos = 0usize;
    let output_size = varint_read(encoded, &mut pos)?;
    let commands_size = varint_read(encoded, &mut pos)?;
    if output_size > MAX_OUTPUT_SIZE {
        return Err(Error::InvalidData(format!(
            "JXL ICC: output_size {output_size} exceeds cap {MAX_OUTPUT_SIZE}"
        )));
    }
    if commands_size > encoded.len() as u64 {
        return Err(Error::InvalidData(format!(
            "JXL ICC: commands_size {commands_size} exceeds remaining encoded byte stream {}",
            encoded.len() - pos
        )));
    }
    // Command stream is `[pos .. pos + commands_size)`; data stream is
    // `[pos + commands_size .. encoded.len())`.
    let cmd_start = pos;
    let cmd_end = cmd_start
        .checked_add(commands_size as usize)
        .ok_or_else(|| Error::InvalidData("JXL ICC: commands_size + offset overflow".into()))?;
    if cmd_end > encoded.len() {
        return Err(Error::InvalidData(
            "JXL ICC: commands stream extends beyond encoded byte stream".into(),
        ));
    }
    let mut cmd_pos = cmd_start;
    let mut data_pos = cmd_end;

    let mut output: Vec<u8> = Vec::with_capacity(output_size as usize);

    // === E.4.3 ICC header ===
    let header_size = std::cmp::min(128u64, output_size);
    if data_pos + (header_size as usize) > encoded.len() {
        return Err(Error::InvalidData(
            "JXL ICC: header data extends beyond encoded byte stream".into(),
        ));
    }
    for i in 0..(header_size as usize) {
        let p = predict_header_byte(i, &output, output_size);
        let e = encoded[data_pos];
        data_pos += 1;
        let out_byte = p.wrapping_add(e);
        output.push(out_byte);
    }
    if output_size <= 128 {
        // E.4.3: "If output_size is smaller than or equal to 128, then
        // the above procedure has produced the full output, the ICC
        // decoder is finished and the remaining subclauses are skipped."
        return Ok(output);
    }

    // === E.4.4 ICC tag list ===
    if cmd_pos < cmd_end {
        let tag_list_finished = decode_tag_list(
            encoded,
            &mut cmd_pos,
            &mut data_pos,
            cmd_end,
            &mut output,
            output_size,
        )?;
        if tag_list_finished {
            return Ok(output);
        }
    }

    // === E.4.5 main content ===
    decode_main_content(
        encoded,
        &mut cmd_pos,
        &mut data_pos,
        cmd_end,
        &mut output,
        output_size,
    )?;

    if output.len() != output_size as usize {
        return Err(Error::InvalidData(format!(
            "JXL ICC: decoded output {} bytes != output_size {}",
            output.len(),
            output_size
        )));
    }
    Ok(output)
}

/// Compute the predicted byte `p` for output position `i` in the ICC
/// header, per the E.4.3 listing. `header` is the partially-built ICC
/// output so far (positions `0..i` already populated), `output_size` is
/// the final ICC profile size.
fn predict_header_byte(i: usize, header: &[u8], output_size: u64) -> u8 {
    // Positions 0..3: predicted as bytes of `output_size` interpreted
    // as big-endian u32. The spec says "output_size[i]" meaning byte i
    // of output_size encoded as an unsigned 32-bit integer in big
    // endian order.
    let h40 = header.get(40).copied().unwrap_or(0);
    let h41 = header.get(41).copied().unwrap_or(0);
    if i < 4 {
        let v = (output_size as u32).to_be_bytes();
        v[i]
    } else if i == 8 {
        4
    } else if (12..=23).contains(&i) {
        // "mntrRGB XYZ " — one space after RGB, one after XYZ.
        let s = b"mntrRGB XYZ ";
        s[i - 12]
    } else if (36..=39).contains(&i) {
        let s = b"acsp";
        s[i - 36]
    } else if (i == 41 || i == 42) && h40 == b'A' {
        b'P'
    } else if i == 43 && h40 == b'A' {
        b'L'
    } else if i == 41 && h40 == b'M' {
        b'S'
    } else if i == 42 && h40 == b'M' {
        b'F'
    } else if i == 43 && h40 == b'M' {
        b'T'
    } else if i == 42 && h40 == b'S' && h41 == b'G' {
        b'I'
    } else if i == 43 && h40 == b'S' && h41 == b'G' {
        32 // ' '
    } else if i == 42 && h40 == b'S' && h41 == b'U' {
        b'N'
    } else if i == 43 && h40 == b'S' && h41 == b'U' {
        b'W'
    } else if i == 70 {
        246
    } else if i == 71 {
        214
    } else if i == 73 {
        1
    } else if i == 78 {
        211
    } else if i == 79 {
        45
    } else if (80..84).contains(&i) {
        let idx = 4 + i - 80;
        header.get(idx).copied().unwrap_or(0)
    } else {
        0
    }
}

/// E.4.4 ICC tag list decode. Returns `true` if the tag list reached
/// the "finalise full ICC profile" branch (`v - 1 == -1`, i.e. `v ==
/// 0`); `false` if it ran to the end and decode should continue with
/// E.4.5 main content.
fn decode_tag_list(
    encoded: &[u8],
    cmd_pos: &mut usize,
    data_pos: &mut usize,
    cmd_end: usize,
    output: &mut Vec<u8>,
    output_size: u64,
) -> Result<bool> {
    if *cmd_pos >= cmd_end {
        return Ok(true);
    }
    let v = varint_read(encoded, cmd_pos)?;
    if v == 0 {
        // Spec: num_tags = v - 1 = -1 → output nothing, finalise.
        return Ok(true);
    }
    let num_tags = v - 1;
    if num_tags > MAX_OUTPUT_SIZE {
        return Err(Error::InvalidData(format!(
            "JXL ICC: tag-list num_tags {num_tags} exceeds cap {MAX_OUTPUT_SIZE}"
        )));
    }
    // Append `num_tags` to output as big-endian unsigned 32-bit integer
    // (4 bytes).
    if num_tags > u32::MAX as u64 {
        return Err(Error::InvalidData(format!(
            "JXL ICC: tag-list num_tags {num_tags} exceeds u32"
        )));
    }
    push_u32_be(output, num_tags as u32, output_size)?;
    let mut previous_tagstart: u64 = (num_tags as u64) * 12 + 128;
    let mut previous_tagsize: u64 = 0;
    loop {
        if *cmd_pos >= cmd_end {
            // End of command stream — decoder finished.
            return Ok(true);
        }
        let command = encoded[*cmd_pos];
        *cmd_pos += 1;
        let tagcode = command & 63;
        if tagcode == 0 {
            // Tag list done — proceed to E.4.5.
            return Ok(false);
        }
        // Resolve tag bytes per the spec's switch.
        let tag: [u8; 4] = if tagcode == 1 {
            // 4 custom bytes from the data stream.
            if *data_pos + 4 > encoded.len() {
                return Err(Error::InvalidData(
                    "JXL ICC: tag-list tagcode=1 data underflow".into(),
                ));
            }
            let mut t = [0u8; 4];
            t.copy_from_slice(&encoded[*data_pos..*data_pos + 4]);
            *data_pos += 4;
            t
        } else if tagcode == 2 {
            *b"rTRC"
        } else if tagcode == 3 {
            *b"rXYZ"
        } else if (4..21).contains(&tagcode) {
            // Predefined table index `tagcode - 4`.
            let strings: [&[u8; 4]; 17] = [
                b"cprt", b"wtpt", b"bkpt", b"rXYZ", b"gXYZ", b"bXYZ", b"kXYZ", b"rTRC", b"gTRC",
                b"bTRC", b"kTRC", b"chad", b"desc", b"chrm", b"dmnd", b"dmdd", b"lumi",
            ];
            *strings[(tagcode - 4) as usize]
        } else {
            // tagcode in [21, 64) is reserved by spec; the spec listing
            // says "this branch is not reached", but be defensive and
            // reject rather than panic.
            return Err(Error::InvalidData(format!(
                "JXL ICC: tag-list tagcode {tagcode} reserved/invalid"
            )));
        };
        // tagstart default = previous_tagstart + previous_tagsize; if
        // command & 64 != 0, tagstart = Varint().
        let mut tagstart = previous_tagstart.saturating_add(previous_tagsize);
        if (command & 64) != 0 {
            tagstart = varint_read(encoded, cmd_pos)?;
        }
        // tagsize default = previous_tagsize. Unless the tag is one of
        // the 7 fixed-20-byte tags, in which case tagsize = 20. If
        // command & 128 != 0, tagsize = Varint().
        let mut tagsize = previous_tagsize;
        let known20 = [
            b"rXYZ", b"gXYZ", b"bXYZ", b"kXYZ", b"wtpt", b"bkpt", b"lumi",
        ];
        if known20.iter().any(|s| **s == tag) {
            tagsize = 20;
        }
        if (command & 128) != 0 {
            tagsize = varint_read(encoded, cmd_pos)?;
        }
        previous_tagstart = tagstart;
        previous_tagsize = tagsize;

        // Append tag, tagstart, tagsize as 4 + 4 + 4 = 12 bytes.
        output.extend_from_slice(&tag);
        push_u32_be(output, tag_to_u32(tagstart)?, output_size)?;
        push_u32_be(output, tag_to_u32(tagsize)?, output_size)?;

        // tagcode 2: also append gTRC / bTRC tags.
        if tagcode == 2 {
            output.extend_from_slice(b"gTRC");
            push_u32_be(output, tag_to_u32(tagstart)?, output_size)?;
            push_u32_be(output, tag_to_u32(tagsize)?, output_size)?;
            output.extend_from_slice(b"bTRC");
            push_u32_be(output, tag_to_u32(tagstart)?, output_size)?;
            push_u32_be(output, tag_to_u32(tagsize)?, output_size)?;
        } else if tagcode == 3 {
            // tagcode 3: also append gXYZ and bXYZ tags.
            output.extend_from_slice(b"gXYZ");
            push_u32_be(output, tag_to_u32(tagstart + tagsize)?, output_size)?;
            push_u32_be(output, tag_to_u32(tagsize)?, output_size)?;
            output.extend_from_slice(b"bXYZ");
            push_u32_be(output, tag_to_u32(tagstart + 2 * tagsize)?, output_size)?;
            push_u32_be(output, tag_to_u32(tagsize)?, output_size)?;
        }
    }
}

fn tag_to_u32(v: u64) -> Result<u32> {
    if v > u32::MAX as u64 {
        return Err(Error::InvalidData(format!(
            "JXL ICC: tagstart/tagsize {v} exceeds u32"
        )));
    }
    Ok(v as u32)
}

fn push_u32_be(output: &mut Vec<u8>, v: u32, output_size: u64) -> Result<()> {
    if (output.len() as u64).saturating_add(4) > output_size {
        return Err(Error::InvalidData(
            "JXL ICC: output overflow appending u32 BE".into(),
        ));
    }
    output.extend_from_slice(&v.to_be_bytes());
    Ok(())
}

/// E.4.5 main content decode. Walks the rest of the command stream
/// emitting bytes per the spec's command set:
///
/// * `command == 1` — append `num` raw bytes from the data stream.
/// * `command == 2 | 3` — append shuffled bytes (width 2 or 4).
/// * `command == 4` — Nth-order predictor (order 0..3, width 1/2/4)
///   with optional shuffle and stride.
/// * `command == 10` — append "XYZ " + 4 zero bytes + 12 bytes from
///   the data stream.
/// * `command in [16, 24)` — append a 4-byte ASCII string + 4 zero
///   bytes (one of "XYZ ", "desc", "text", "mluc", "para", "curv",
///   "sf32", "gbd ").
fn decode_main_content(
    encoded: &[u8],
    cmd_pos: &mut usize,
    data_pos: &mut usize,
    cmd_end: usize,
    output: &mut Vec<u8>,
    output_size: u64,
) -> Result<()> {
    while *cmd_pos < cmd_end {
        let command = encoded[*cmd_pos];
        *cmd_pos += 1;
        match command {
            1 => {
                let num = varint_read(encoded, cmd_pos)?;
                if num == 0 {
                    return Err(Error::InvalidData(
                        "JXL ICC: main content command=1 num==0".into(),
                    ));
                }
                let num_usize = u64_to_usize_bounded(num, output_size)?;
                if *data_pos + num_usize > encoded.len() {
                    return Err(Error::InvalidData(
                        "JXL ICC: main content command=1 data underflow".into(),
                    ));
                }
                output.extend_from_slice(&encoded[*data_pos..*data_pos + num_usize]);
                *data_pos += num_usize;
            }
            2 | 3 => {
                let width = if command == 2 { 2usize } else { 4usize };
                let num = varint_read(encoded, cmd_pos)?;
                if num == 0 {
                    return Err(Error::InvalidData(format!(
                        "JXL ICC: main content command={command} num==0"
                    )));
                }
                let num_usize = u64_to_usize_bounded(num, output_size)?;
                if *data_pos + num_usize > encoded.len() {
                    return Err(Error::InvalidData(format!(
                        "JXL ICC: main content command={command} data underflow"
                    )));
                }
                let mut bytes = vec![0u8; num_usize];
                bytes.copy_from_slice(&encoded[*data_pos..*data_pos + num_usize]);
                *data_pos += num_usize;
                shuffle(&mut bytes, width);
                output.extend_from_slice(&bytes);
            }
            4 => {
                if *cmd_pos >= cmd_end {
                    return Err(Error::InvalidData(
                        "JXL ICC: main content command=4 truncated flags".into(),
                    ));
                }
                let flags = encoded[*cmd_pos];
                *cmd_pos += 1;
                let width = ((flags & 3) as usize) + 1;
                if width == 3 {
                    return Err(Error::InvalidData(
                        "JXL ICC: main content command=4 width==3 invalid".into(),
                    ));
                }
                let order = ((flags & 12) >> 2) as usize;
                if order == 3 {
                    return Err(Error::InvalidData(
                        "JXL ICC: main content command=4 order==3 invalid".into(),
                    ));
                }
                let mut stride = width;
                if (flags & 16) != 0 {
                    let s = varint_read(encoded, cmd_pos)?;
                    let s_usize = u64_to_usize_bounded(s, output_size)?;
                    if (s_usize.saturating_mul(4)) >= output.len() {
                        return Err(Error::InvalidData(
                            "JXL ICC: main content command=4 stride * 4 >= output_so_far".into(),
                        ));
                    }
                    if s_usize < width {
                        return Err(Error::InvalidData(
                            "JXL ICC: main content command=4 stride < width".into(),
                        ));
                    }
                    stride = s_usize;
                }
                let num = varint_read(encoded, cmd_pos)?;
                if num == 0 {
                    return Err(Error::InvalidData(
                        "JXL ICC: main content command=4 num==0".into(),
                    ));
                }
                let num_usize = u64_to_usize_bounded(num, output_size)?;
                if *data_pos + num_usize > encoded.len() {
                    return Err(Error::InvalidData(
                        "JXL ICC: main content command=4 data underflow".into(),
                    ));
                }
                let mut bytes = vec![0u8; num_usize];
                bytes.copy_from_slice(&encoded[*data_pos..*data_pos + num_usize]);
                *data_pos += num_usize;
                if width == 2 || width == 4 {
                    shuffle(&mut bytes, width);
                }
                // Now run the Nth-order predictor:
                //   for i in [0, num) step width:
                //     N = order + 1
                //     prev[j] = output[stride*(j+1) bytes before output_so_far],
                //               interpreted as big-endian unsigned width-byte integer
                //     compute predicted p; emit val = (bytes[i+j] + (p >> ...)) & 255
                let mut i = 0usize;
                while i < num_usize {
                    let n_order = order + 1;
                    let mut prev = [0u128; 3];
                    for (j, prev_j) in prev.iter_mut().enumerate().take(n_order) {
                        let back = stride.checked_mul(j + 1).ok_or_else(|| {
                            Error::InvalidData("JXL ICC: main content stride overflow".into())
                        })?;
                        if back > output.len() {
                            return Err(Error::InvalidData(
                                "JXL ICC: main content predictor reads before output start".into(),
                            ));
                        }
                        let off = output.len() - back;
                        if off + width > output.len() {
                            return Err(Error::InvalidData(
                                "JXL ICC: main content predictor reads past output end".into(),
                            ));
                        }
                        let mut acc: u128 = 0;
                        for &b in &output[off..off + width] {
                            acc = (acc << 8) | b as u128;
                        }
                        *prev_j = acc;
                    }
                    let p: u128 = match order {
                        0 => prev[0],
                        1 => prev[0].wrapping_mul(2).wrapping_sub(prev[1]),
                        2 => prev[0]
                            .wrapping_mul(3)
                            .wrapping_sub(prev[1].wrapping_mul(3))
                            .wrapping_add(prev[2]),
                        _ => unreachable!(),
                    };
                    for j in 0..width {
                        if i + j >= num_usize {
                            break;
                        }
                        let shift = 8 * (width - 1 - j) as u32;
                        let val = (bytes[i + j] as u128).wrapping_add(p >> shift) & 0xFF;
                        if (output.len() as u64) >= output_size {
                            return Err(Error::InvalidData(
                                "JXL ICC: main content command=4 output overflow".into(),
                            ));
                        }
                        output.push(val as u8);
                    }
                    i += width;
                }
            }
            10 => {
                // Append "XYZ " (4 bytes) + 4 zero bytes + 12 data bytes.
                ensure_room(output, output_size, 4 + 4 + 12)?;
                output.extend_from_slice(b"XYZ ");
                output.extend_from_slice(&[0u8; 4]);
                if *data_pos + 12 > encoded.len() {
                    return Err(Error::InvalidData(
                        "JXL ICC: main content command=10 data underflow".into(),
                    ));
                }
                output.extend_from_slice(&encoded[*data_pos..*data_pos + 12]);
                *data_pos += 12;
            }
            cmd if (16..24).contains(&cmd) => {
                let strings: [&[u8; 4]; 8] = [
                    b"XYZ ", b"desc", b"text", b"mluc", b"para", b"curv", b"sf32", b"gbd ",
                ];
                let s = strings[(cmd - 16) as usize];
                ensure_room(output, output_size, 8)?;
                output.extend_from_slice(s);
                output.extend_from_slice(&[0u8; 4]);
            }
            _ => {
                return Err(Error::InvalidData(format!(
                    "JXL ICC: main content unknown command {command}"
                )));
            }
        }
    }
    Ok(())
}

fn ensure_room(output: &[u8], output_size: u64, n: usize) -> Result<()> {
    if (output.len() as u64).saturating_add(n as u64) > output_size {
        return Err(Error::InvalidData(
            "JXL ICC: main content output overflow".into(),
        ));
    }
    Ok(())
}

fn u64_to_usize_bounded(v: u64, output_size: u64) -> Result<usize> {
    if v > output_size {
        return Err(Error::InvalidData(format!(
            "JXL ICC: byte count {v} exceeds output_size {output_size}"
        )));
    }
    Ok(v as usize)
}

/// E.4.5 `Shuffle(bytes, width)`. Bytes are inserted in raster order
/// into a matrix with `width` rows. The last column may have missing
/// elements at the bottom if `len(bytes)` is not a multiple of `width`;
/// those are skipped. Then the matrix is transposed and bytes are
/// overwritten with elements of the transposed matrix read in raster
/// order, not including the missing elements.
///
/// Example: `(1, 2, 3, 4, 5, 6, 7)` with `width=2` becomes
/// `(1, 5, 2, 6, 3, 7, 4)`.
fn shuffle(bytes: &mut [u8], width: usize) {
    if width <= 1 || bytes.len() <= 1 {
        return;
    }
    let n = bytes.len();
    let cols = n.div_ceil(width);
    // Matrix has `width` rows × `cols` columns. Bytes are placed in
    // row-major order: matrix[r][c] = bytes[r * cols + c]. The last
    // column may be short — its filled rows count is
    // `last_col_filled_rows`; rows >= that in the last column are
    // missing. Total missing = width * cols - n.
    let last_col_filled_rows = n - (cols - 1) * width;
    let src: Vec<u8> = bytes.to_vec();
    let mut out_pos = 0usize;
    for c in 0..cols {
        for r in 0..width {
            // Skip missing elements at the bottom of the last column.
            if c == cols - 1 && r >= last_col_filled_rows {
                continue;
            }
            // Source position: row-major matrix index.
            let in_index = r * cols + c;
            bytes[out_pos] = src[in_index];
            out_pos += 1;
        }
    }
    debug_assert_eq!(out_pos, n);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icc_context_first_129_bytes_are_zero() {
        for i in 0..=128 {
            assert_eq!(icc_context(i, 0, 0), 0);
            assert_eq!(icc_context(i, b'A', b'B'), 0);
            assert_eq!(icc_context(i, 255, 255), 0);
        }
    }

    #[test]
    fn icc_context_letters_collapse_to_p1_zero() {
        // For i > 128, b1 letter, b2 letter → p1=0, p2=0 → context = 1.
        assert_eq!(icc_context(200, b'A', b'B'), 1);
        assert_eq!(icc_context(200, b'z', b'a'), 1);
    }

    #[test]
    fn icc_context_max_p1_p2() {
        // To hit p1 = 7 (the catch-all "else") b1 must NOT be a letter,
        // NOT '.'/',', NOT <= 1, NOT in (1,16), NOT in (240,255), NOT
        // == 255 — i.e. b1 in [16, 240]. To hit p2 = 4 (the catch-all)
        // b2 must satisfy b2 in [16, 240] too. Pick 200 for both.
        // ctx = 1 + 7 + 4*8 = 40.
        assert_eq!(icc_context(200, 200, 200), 1 + 7 + 4 * 8);
        assert!(icc_context(200, 200, 200) < ICC_NUM_DIST as u32);
    }

    #[test]
    fn icc_context_byte_value_1_path() {
        // b1 == 1: p1 = 2 + b1 = 3.
        // b2 == 200 (catch-all): p2 = 4.
        // ctx = 1 + 3 + 32 = 36.
        assert_eq!(icc_context(200, 1, 200), 1 + 3 + 4 * 8);
    }

    #[test]
    fn icc_context_byte_value_255_path() {
        // b1 == 255: p1 = 6. b2 == 0: p2 = 2 (b2 < 16).
        assert_eq!(icc_context(200, 255, 0), 1 + 6 + 2 * 8);
    }

    #[test]
    fn icc_context_in_range() {
        // Exhaustive check that the returned context fits in [0, 41).
        for i in [0usize, 100, 129, 200, 1024] {
            for b1 in 0u8..=255 {
                for b2 in 0u8..=255 {
                    let c = icc_context(i, b1, b2);
                    assert!(
                        c < ICC_NUM_DIST as u32,
                        "ctx {} out of range for i={} b1={} b2={}",
                        c,
                        i,
                        b1,
                        b2
                    );
                }
            }
        }
    }

    #[test]
    fn varint_simple_one_byte() {
        let bytes = [42u8];
        let mut pos = 0;
        assert_eq!(varint_read(&bytes, &mut pos).unwrap(), 42);
        assert_eq!(pos, 1);
    }

    #[test]
    fn varint_zero_value() {
        let bytes = [0u8];
        let mut pos = 0;
        assert_eq!(varint_read(&bytes, &mut pos).unwrap(), 0);
        assert_eq!(pos, 1);
    }

    #[test]
    fn varint_two_byte() {
        // 7-bit groups: 0x80 | 1, 0x02 → value = 1 | (2 << 7) = 257.
        let bytes = [0x81u8, 0x02];
        let mut pos = 0;
        assert_eq!(varint_read(&bytes, &mut pos).unwrap(), 257);
        assert_eq!(pos, 2);
    }

    #[test]
    fn varint_truncated_errors() {
        let bytes = [0x80u8]; // continuation bit set, no follow-up
        let mut pos = 0;
        assert!(varint_read(&bytes, &mut pos).is_err());
    }

    #[test]
    fn varint_overflow_errors() {
        // 9 continuation bytes + a 10th non-continuation would overflow
        // 64-bit; the tighter check is shift > 56.
        let bytes = [0xFFu8; 16];
        let mut pos = 0;
        assert!(varint_read(&bytes, &mut pos).is_err());
    }

    #[test]
    fn shuffle_width_2_example_from_spec() {
        // Spec example: (1,2,3,4,5,6,7) with width 2 → (1,5,2,6,3,7,4).
        let mut data = [1u8, 2, 3, 4, 5, 6, 7];
        shuffle(&mut data, 2);
        assert_eq!(data, [1, 5, 2, 6, 3, 7, 4]);
    }

    #[test]
    fn shuffle_width_1_is_noop() {
        let mut data = [1u8, 2, 3, 4];
        shuffle(&mut data, 1);
        assert_eq!(data, [1, 2, 3, 4]);
    }

    #[test]
    fn shuffle_width_4_full_matrix() {
        // 8 bytes, width 4 → matrix 4 rows × 2 cols, no missing.
        // Matrix:
        //   row0: 1 2
        //   row1: 3 4
        //   row2: 5 6
        //   row3: 7 8
        // Transposed: 2 cols become 2 rows, 4 cols
        //   row0: 1 3 5 7
        //   row1: 2 4 6 8
        // Output in raster order: 1 3 5 7 2 4 6 8
        let mut data = [1u8, 2, 3, 4, 5, 6, 7, 8];
        shuffle(&mut data, 4);
        assert_eq!(data, [1, 3, 5, 7, 2, 4, 6, 8]);
    }

    #[test]
    fn predict_header_pos0_is_output_size_be() {
        let header = [];
        let p0 = predict_header_byte(0, &header, 0x12345678);
        assert_eq!(p0, 0x12);
        let p1 = predict_header_byte(1, &header, 0x12345678);
        assert_eq!(p1, 0x34);
    }

    #[test]
    fn predict_header_acsp_at_36() {
        // Positions 36-39 always predict to 'a','c','s','p'.
        let h = vec![0u8; 36];
        assert_eq!(predict_header_byte(36, &h, 1024), b'a');
        assert_eq!(predict_header_byte(37, &h, 1024), b'c');
        assert_eq!(predict_header_byte(38, &h, 1024), b's');
        assert_eq!(predict_header_byte(39, &h, 1024), b'p');
    }

    #[test]
    fn reconstruct_minimal_64_byte_output() {
        // A round-trip-style test: build an encoded byte stream that
        // claims output_size=64 and commands_size=0, with 64 data
        // bytes that round-trip through predict_header_byte.
        // For positions 0..3 the predicted value is the BE bytes of
        // output_size=64. So for the output to be all-zero the encoded
        // delta at positions 0..3 must be the negation of those bytes.
        // We instead make the output exactly the predicted-header
        // sequence: encoded bytes 0..63 are all 0.
        let mut encoded: Vec<u8> = Vec::new();
        // output_size = 64 (Varint): single byte 0x40.
        encoded.push(0x40);
        // commands_size = 0 (Varint): single byte 0x00.
        encoded.push(0x00);
        // 64 data bytes, all zero → output equals predicted-header
        // sequence.
        encoded.extend_from_slice(&[0u8; 64]);
        let result = reconstruct_icc_profile(&encoded).unwrap();
        assert_eq!(result.len(), 64);
        // Check the predictable bits of the predicted header:
        // bytes 0..4 = (64 as u32 BE) = [0, 0, 0, 64]
        assert_eq!(&result[0..4], &[0, 0, 0, 64]);
        // byte 8 = 4 ('rendering intent prediction')
        assert_eq!(result[8], 4);
        // bytes 12..24 = "mntrRGB XYZ "
        assert_eq!(&result[12..24], b"mntrRGB XYZ ");
        // bytes 36..40 = "acsp"
        assert_eq!(&result[36..40], b"acsp");
    }

    #[test]
    fn reconstruct_rejects_truncated_header() {
        let encoded = [0x80, 0x40, 0x00]; // output_size=64 (2-byte Varint), commands_size=0, NO data bytes
        let r = reconstruct_icc_profile(&encoded);
        assert!(r.is_err(), "expected truncation error, got {r:?}");
    }

    #[test]
    fn reconstruct_rejects_oversize_output() {
        // output_size = MAX_OUTPUT_SIZE + 1 — should be rejected.
        let mut encoded: Vec<u8> = Vec::new();
        // Varint encoding of MAX_OUTPUT_SIZE + 1 = 64 MiB + 1 = 67_108_865
        let mut v: u64 = MAX_OUTPUT_SIZE + 1;
        loop {
            let mut byte = (v & 0x7F) as u8;
            v >>= 7;
            if v != 0 {
                byte |= 0x80;
            }
            encoded.push(byte);
            if v == 0 {
                break;
            }
        }
        encoded.push(0); // commands_size = 0
        let r = reconstruct_icc_profile(&encoded);
        assert!(r.is_err(), "expected oversize-output rejection");
    }

    #[test]
    fn reconstruct_short_output_of_size_4() {
        // output_size = 4 (≤ 128 → header-only path), encoded
        // delta bytes = 0 so output = predicted header bytes 0..3 =
        // BE bytes of output_size = [0, 0, 0, 4].
        let encoded = [0x04u8, 0x00, 0x00, 0x00, 0x00, 0x00];
        let result = reconstruct_icc_profile(&encoded).unwrap();
        assert_eq!(result, vec![0, 0, 0, 4]);
    }
}
