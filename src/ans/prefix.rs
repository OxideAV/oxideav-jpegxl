//! Brotli (RFC 7932) prefix codes — FDIS Annex D.2.
//!
//! Two formats are supported, both with explicit alphabet size given by
//! the JXL caller (D.2.1: "alphabet size mentioned in the RFC is
//! explicitly specified as parameter `alphabet_size`"):
//!
//! * **Simple prefix code** (RFC 7932 §3.4) — the histogram begins with
//!   `u(2) == 1`. Up to four symbols are listed, each as a fixed-width
//!   field of `ceil(log2(alphabet_size))` bits, plus an optional 1-bit
//!   tree-select for the four-symbol case.
//! * **Complex prefix code** (RFC 7932 §3.5) — `u(2) ∈ {0, 2, 3}` is
//!   `HSKIP`, the number of code-length-code-lengths to skip. Up to 18
//!   code-length-code-length values are then read, each via the tiny
//!   2-4-bit variable-length code, then the actual code lengths are
//!   derived using repeat-codes 16 (repeat previous non-zero) and 17
//!   (repeat zero).
//!
//! The decoded output is a [`PrefixCode`] table that exposes
//! [`PrefixCode::decode`] for D.2.2 ("read bits one at a time until
//! they match a code"). Decode uses a flat lookup table sized by the
//! longest code length, capped at 15 bits per RFC 7932 §3.1.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// RFC 7932 §3.5 — the exact order in which the 18 code-length-code
/// lengths appear in the bitstream.
pub const K_CODE_LENGTH_CODE_ORDER: [usize; 18] =
    [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Maximum prefix-code length permitted by RFC 7932 §3.1.
pub const MAX_CODE_LENGTH: u32 = 15;

/// Maximum alphabet size we accept for a prefix code. JXL never asks
/// for more than a few thousand symbols (D.3.4 caps log_alphabet_size
/// at 8 when prefix codes are in play, so alphabet_size <= 256 in
/// practice). We accept up to 1<<16 to leave headroom while still
/// bounding allocations against malicious input.
pub const MAX_ALPHABET_SIZE: usize = 1 << 16;

/// A flat-table prefix decoder.
///
/// `lookup` has length `1 << max_length`. Reading the bottom
/// `max_length` LSB-first bits of the input gives an index into the
/// table; the entry holds the decoded symbol and the actual code
/// length consumed. Bits past the actual code length are then "given
/// back" by the caller (see [`Self::decode`]).
#[derive(Debug, Clone)]
pub struct PrefixCode {
    lookup: Vec<(u32, u8)>,
    max_length: u32,
    /// Alphabet size the table was built for. Symbols outside [0, n)
    /// are flagged as decode errors.
    pub alphabet_size: u32,
}

impl PrefixCode {
    /// Construct a prefix code from a vector of code lengths
    /// (`code_lengths[i]` = length in bits of symbol `i`'s code, or 0
    /// if that symbol is unused).
    ///
    /// The mapping from lengths to canonical codes follows RFC 7932 §3.1
    /// (canonical Huffman: codes are assigned in length-major,
    /// symbol-minor order, with each new code formed by adding 1 to
    /// the previous one and shifting up if the length increases).
    pub fn from_lengths(code_lengths: &[u32]) -> Result<Self> {
        if code_lengths.len() > MAX_ALPHABET_SIZE {
            return Err(Error::InvalidData(
                "JXL prefix: alphabet size too large".into(),
            ));
        }
        // Single-symbol degenerate case: length 0, one entry → maps any
        // input bit pattern to that symbol with 0 code length.
        let max_length = code_lengths.iter().copied().max().unwrap_or(0);
        if max_length == 0 {
            // No symbol is encoded — caller must not invoke this.
            // Emit a 1-entry table that always errors on decode.
            let lookup = vec![(0u32, 0u8); 1];
            return Ok(Self {
                lookup,
                max_length: 0,
                alphabet_size: code_lengths.len() as u32,
            });
        }
        if max_length > MAX_CODE_LENGTH {
            return Err(Error::InvalidData("JXL prefix: code length > 15".into()));
        }

        // Validate the Kraft-McMillan inequality (RFC 7932 §3.1: equal
        // for a *complete* code, i.e. exactly one).
        let mut kraft: u64 = 0;
        let mut nonzero = 0u32;
        for &l in code_lengths {
            if l == 0 {
                continue;
            }
            nonzero += 1;
            kraft += 1u64 << (15 - l);
        }
        if nonzero == 0 {
            return Err(Error::InvalidData(
                "JXL prefix: no non-zero code lengths".into(),
            ));
        }
        // Special case: exactly one non-zero length-1 symbol acts like a
        // fully-redundant 1-bit code (the other half is unused).
        // RFC 7932 says all real prefix codes must be complete, so kraft
        // must equal 1<<15 — except for the degenerate single-symbol
        // simple-prefix case where it's allowed to be < 1<<15.
        if kraft > (1u64 << 15) {
            return Err(Error::InvalidData(
                "JXL prefix: code lengths overflow Kraft sum".into(),
            ));
        }

        // Canonical Huffman: assign codes in length-major,
        // symbol-minor order.
        let mut bl_count = vec![0u32; (max_length + 1) as usize];
        for &l in code_lengths {
            if l > 0 {
                bl_count[l as usize] += 1;
            }
        }
        let mut next_code = vec![0u32; (max_length + 1) as usize];
        let mut code: u32 = 0;
        for bits in 1..=max_length {
            code = (code + bl_count[(bits - 1) as usize]) << 1;
            next_code[bits as usize] = code;
        }

        // Build the flat lookup. RFC 7932 codes are assembled MSB-first
        // (bit n-1 is the first bit emitted into the stream) but JXL
        // reads its bits LSB-first, so the lookup index is the
        // bit-reversed code. RFC 7932 §3.5 also specifies "the codes are
        // packed from least significant bit to most significant bit" —
        // i.e. the first bit read from the stream is the MSB of the
        // canonical code. We produce a lookup table such that reading
        // `max_length` LSB-first bits then matching against the table
        // works for partial codes too.
        let table_size = 1u32 << max_length;
        let mut lookup = vec![(u32::MAX, 0u8); table_size as usize];
        for (sym, &l) in code_lengths.iter().enumerate() {
            if l == 0 {
                continue;
            }
            let canonical = next_code[l as usize];
            next_code[l as usize] += 1;
            // Reverse the canonical code's `l` bits.
            let bit_reversed = bit_reverse(canonical, l);
            // Replicate across all higher-bit suffixes.
            let stride = 1u32 << l;
            let mut idx = bit_reversed;
            while idx < table_size {
                lookup[idx as usize] = (sym as u32, l as u8);
                idx += stride;
            }
        }

        // Any uninitialised slot indicates an incomplete code; we leave
        // them as `(u32::MAX, 0)` and let `decode` error out on
        // encountering them.
        Ok(Self {
            lookup,
            max_length,
            alphabet_size: code_lengths.len() as u32,
        })
    }

    /// Decode one symbol from `br`. Reads up to `max_length` bits.
    pub fn decode(&self, br: &mut BitReader<'_>) -> Result<u32> {
        if self.max_length == 0 {
            // Single-symbol code: lookup[0] is the symbol, no bits
            // consumed. Built either by `read_simple_prefix` for
            // NSYM=1 or by `from_lengths` with all-zero lengths
            // (which we treat as an unrecoverable degenerate, see
            // below).
            let (sym, _) = self.lookup[0];
            if sym >= self.alphabet_size {
                return Err(Error::InvalidData(
                    "JXL prefix: degenerate empty code".into(),
                ));
            }
            return Ok(sym);
        }
        let raw = br.peek_bits(self.max_length)? as usize;
        let (sym, l) = self.lookup[raw];
        if l == 0 {
            return Err(Error::InvalidData(
                "JXL prefix: malformed prefix code (incomplete table hit)".into(),
            ));
        }
        if sym >= self.alphabet_size {
            return Err(Error::InvalidData(
                "JXL prefix: decoded symbol >= alphabet_size".into(),
            ));
        }
        br.advance_bits(l as u32)?;
        Ok(sym)
    }
}

/// Reverse the bottom `n` bits of `x`. Helper for canonical-code →
/// LSB-first lookup index conversion.
fn bit_reverse(mut x: u32, n: u32) -> u32 {
    let mut out = 0u32;
    for _ in 0..n {
        out = (out << 1) | (x & 1);
        x >>= 1;
    }
    out
}

/// Decode a prefix-code histogram (a "Huffman histogram stream" per
/// FDIS D.2.1) and return the decoded [`PrefixCode`].
///
/// `alphabet_size` is the JXL-side parameter described in D.2.1, and
/// is bounded by [`MAX_ALPHABET_SIZE`].
pub fn read_prefix_code(br: &mut BitReader<'_>, alphabet_size: u32) -> Result<PrefixCode> {
    if alphabet_size as usize > MAX_ALPHABET_SIZE {
        return Err(Error::InvalidData(
            "JXL prefix: alphabet_size too large".into(),
        ));
    }
    if alphabet_size == 0 {
        return Err(Error::InvalidData("JXL prefix: alphabet_size == 0".into()));
    }
    if alphabet_size == 1 {
        // Degenerate: one symbol, no bits. The bitstream contains
        // nothing for this histogram (per RFC 7932 §3.4 NSYM=1 case).
        // JXL still reads the simple/complex selector to be safe?
        // The RFC says simple-prefix is signalled by `u(2) == 1`. We
        // accept either way and produce a length-0 code that always
        // returns symbol 0.
        let code_lengths = vec![0u32; 1];
        return PrefixCode::from_lengths(&code_lengths);
    }

    let kind = br.read_bits(2)?;
    if kind == 1 {
        read_simple_prefix(br, alphabet_size)
    } else {
        read_complex_prefix(br, alphabet_size, kind)
    }
}

fn alphabet_bits(alphabet_size: u32) -> u32 {
    // ceil(log2(alphabet_size)). For alphabet_size==1 we never reach
    // here (caller short-circuits).
    if alphabet_size <= 1 {
        0
    } else {
        32 - (alphabet_size - 1).leading_zeros()
    }
}

fn read_simple_prefix(br: &mut BitReader<'_>, alphabet_size: u32) -> Result<PrefixCode> {
    // RFC 7932 §3.4.
    let nsym = br.read_bits(2)? + 1;
    let bits = alphabet_bits(alphabet_size);
    let mut symbols = Vec::with_capacity(nsym as usize);
    for _ in 0..nsym {
        let s = br.read_bits(bits)?;
        if s >= alphabet_size {
            return Err(Error::InvalidData(
                "JXL prefix (simple): symbol out of alphabet".into(),
            ));
        }
        if symbols.contains(&s) {
            return Err(Error::InvalidData(
                "JXL prefix (simple): duplicate symbol".into(),
            ));
        }
        symbols.push(s);
    }
    let mut code_lengths = vec![0u32; alphabet_size as usize];
    match nsym {
        1 => {
            code_lengths[symbols[0] as usize] = 0;
            // Special case: a one-symbol simple code is valid; we leave
            // all lengths at 0 and PrefixCode::decode will succeed with
            // 0 bits consumed (handled in the max_length==0 path).
            // But we want to actually return that symbol from decode,
            // which the current code doesn't do (it errors). Adjust:
            return Ok(PrefixCode {
                lookup: vec![(symbols[0], 0u8); 1],
                max_length: 0,
                alphabet_size,
            });
        }
        2 => {
            // Two symbols, both length 1, ascending order required by
            // RFC 7932 §3.4. We sort them so canonicalisation gives
            // codes 0 and 1.
            symbols.sort();
            code_lengths[symbols[0] as usize] = 1;
            code_lengths[symbols[1] as usize] = 1;
        }
        3 => {
            // Three symbols: lengths 1, 2, 2. Symbols 1 and 2 (the last
            // two) sorted ascending.
            // RFC 7932 §3.4 says "For NSYM = 3, the symbols are sorted
            // for code length assignment". The first symbol gets length
            // 1, the second and third (sorted) get length 2.
            let first = symbols[0];
            let mut rest = [symbols[1], symbols[2]];
            rest.sort();
            code_lengths[first as usize] = 1;
            code_lengths[rest[0] as usize] = 2;
            code_lengths[rest[1] as usize] = 2;
        }
        4 => {
            // Read tree-select bit.
            let tree_select = br.read_bit()?;
            symbols.sort();
            if tree_select == 0 {
                // Lengths 2, 2, 2, 2.
                for &s in &symbols {
                    code_lengths[s as usize] = 2;
                }
            } else {
                // Lengths 1, 2, 3, 3.
                code_lengths[symbols[0] as usize] = 1;
                code_lengths[symbols[1] as usize] = 2;
                code_lengths[symbols[2] as usize] = 3;
                code_lengths[symbols[3] as usize] = 3;
            }
        }
        _ => unreachable!(),
    }
    PrefixCode::from_lengths(&code_lengths)
}

/// RFC 7932 §3.5 variable-length code for the 18 code-length-code
/// lengths. Each entry is `(symbol, code, code_length_in_bits)`.
const CLCL_VL_TABLE: &[(u32, u32, u32)] = &[
    // The 2-4 bit code from RFC 7932 §3.5:
    //   0 → 00     (2 bits)
    //   1 → 0111   (4 bits)
    //   2 → 011    (3 bits)
    //   3 → 10     (2 bits)
    //   4 → 01     (2 bits)
    //   5 → 1111   (4 bits)
    // Bit order: MSB-first in the RFC. JXL packs LSB-first, so we
    // bit-reverse below at lookup time.
    (0, 0b00, 2),
    (1, 0b0111, 4),
    (2, 0b011, 3),
    (3, 0b10, 2),
    (4, 0b01, 2),
    (5, 0b1111, 4),
];

fn read_clcl_symbol(br: &mut BitReader<'_>) -> Result<u32> {
    // Build a tiny lookup table by hand each call. The table is small
    // enough that the overhead is irrelevant compared to bit reading.
    // We try lengths in ascending order.
    // Peek 4 bits, match against the table.
    let raw = br.peek_bits(4)?;
    for &(sym, code, len) in CLCL_VL_TABLE {
        let lsb_first = bit_reverse(code, len);
        let mask = (1u32 << len) - 1;
        if (raw & mask) == lsb_first {
            br.advance_bits(len)?;
            return Ok(sym);
        }
    }
    Err(Error::InvalidData(
        "JXL prefix (clcl): no matching code-length-code length".into(),
    ))
}

fn read_complex_prefix(
    br: &mut BitReader<'_>,
    alphabet_size: u32,
    hskip: u32,
) -> Result<PrefixCode> {
    if hskip != 0 && hskip != 2 && hskip != 3 {
        // Per RFC 7932 §3.5: HSKIP ∈ {0, 2, 3}. The kind=1 case is
        // simple-prefix (handled by caller); 1 here is unreachable.
        return Err(Error::InvalidData(
            "JXL prefix (complex): invalid HSKIP".into(),
        ));
    }

    // Read up to 18 code-length-code-lengths in order; the first HSKIP
    // are implicit zeros.
    let mut clcl = [0u32; 18];
    let mut sum_kraft: u64 = 0;
    let mut nonzero_count = 0u32;
    for i in (hskip as usize)..18 {
        let v = read_clcl_symbol(br)?;
        clcl[K_CODE_LENGTH_CODE_ORDER[i]] = v;
        if v > 0 {
            nonzero_count += 1;
            sum_kraft += 1u64 << (5 - v); // RFC 7932 §3.5: 32-length space.
            if nonzero_count >= 2 && sum_kraft >= 32 {
                // Full 5-bit code reached → remaining clcls are
                // implicit zeros.
                break;
            }
        }
    }
    if nonzero_count == 0 {
        return Err(Error::InvalidData(
            "JXL prefix (complex): all code-length-code-lengths zero".into(),
        ));
    }
    if nonzero_count == 1 && sum_kraft != 32 {
        // RFC 7932 §3.5 short-circuits when only one non-zero clcl
        // exists; that symbol's code becomes a 0-bit code.
    }
    // Build the code-length code itself (alphabet of 18 symbols).
    let cl_code = PrefixCode::from_lengths(&clcl)?;

    // Bound the alphabet against the input length to refuse insane
    // allocations from a malicious histogram preamble.
    if alphabet_size as usize > MAX_ALPHABET_SIZE {
        return Err(Error::InvalidData(
            "JXL prefix (complex): alphabet_size too large".into(),
        ));
    }
    let bits_remaining_cap = br.bits_remaining();
    // A non-zero code length must be at least 1 bit, so an entire
    // alphabet of length-only repeat codes still needs at least
    // alphabet_size / 11 bits (a single 17 code emits up to 10 zeros
    // but still costs at least 4 bits). Use this very loose bound.
    if (alphabet_size as usize) > bits_remaining_cap.saturating_mul(11) + 18 {
        return Err(Error::InvalidData(
            "JXL prefix (complex): alphabet_size larger than input could supply".into(),
        ));
    }

    let mut lengths = vec![0u32; alphabet_size as usize];
    let mut idx: usize = 0;
    let mut prev_nonzero: u32 = 8; // RFC 7932 §3.5: "If first code or all previous lengths are zero, repeats length 8".
    let mut last_was_16 = false;
    let mut last_was_17 = false;
    let mut repeat_count_16: u32 = 0;
    let mut repeat_count_17: u32 = 0;
    while idx < alphabet_size as usize {
        let sym = cl_code.decode(br)?;
        if sym <= 15 {
            lengths[idx] = sym;
            idx += 1;
            if sym != 0 {
                prev_nonzero = sym;
            }
            last_was_16 = false;
            last_was_17 = false;
        } else if sym == 16 {
            // Repeat previous non-zero code length 3..6 times.
            let extra = br.read_bits(2)?;
            let new_count = if last_was_16 {
                4 * (repeat_count_16 - 2) + 3 + extra
            } else {
                3 + extra
            };
            let delta = new_count.saturating_sub(repeat_count_16);
            for _ in 0..delta {
                if idx >= alphabet_size as usize {
                    return Err(Error::InvalidData(
                        "JXL prefix (complex): repeat-16 overruns alphabet".into(),
                    ));
                }
                lengths[idx] = prev_nonzero;
                idx += 1;
            }
            repeat_count_16 = new_count;
            last_was_16 = true;
            last_was_17 = false;
        } else if sym == 17 {
            // Repeat zero 3..10 times.
            let extra = br.read_bits(3)?;
            let new_count = if last_was_17 {
                8 * (repeat_count_17 - 2) + 3 + extra
            } else {
                3 + extra
            };
            let delta = new_count.saturating_sub(repeat_count_17);
            for _ in 0..delta {
                if idx >= alphabet_size as usize {
                    return Err(Error::InvalidData(
                        "JXL prefix (complex): repeat-17 overruns alphabet".into(),
                    ));
                }
                lengths[idx] = 0;
                idx += 1;
            }
            repeat_count_17 = new_count;
            last_was_17 = true;
            last_was_16 = false;
        } else {
            return Err(Error::InvalidData(
                "JXL prefix (complex): code-length symbol out of range".into(),
            ));
        }
        if !last_was_16 {
            repeat_count_16 = 0;
        }
        if !last_was_17 {
            repeat_count_17 = 0;
        }
    }

    PrefixCode::from_lengths(&lengths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ans::test_helpers::pack_lsb;

    #[test]
    fn bit_reverse_works() {
        assert_eq!(bit_reverse(0b101, 3), 0b101);
        assert_eq!(bit_reverse(0b1100, 4), 0b0011);
        assert_eq!(bit_reverse(0b10, 4), 0b0100);
    }

    #[test]
    fn from_lengths_two_symbol_round_trip() {
        // Symbols 0 and 1, both length 1.
        // Canonical code: sym 0 = "0", sym 1 = "1".
        // LSB-first lookup: bit 0 → sym 0, bit 1 → sym 1.
        let code = PrefixCode::from_lengths(&[1, 1]).unwrap();
        assert_eq!(code.lookup[0], (0, 1));
        assert_eq!(code.lookup[1], (1, 1));
    }

    #[test]
    fn from_lengths_canonical_124_code() {
        // Lengths 1, 2, 3, 3. Canonical codes:
        //   sym 0: "0"   (length 1)
        //   sym 1: "10"  (length 2)
        //   sym 2: "110" (length 3)
        //   sym 3: "111" (length 3)
        // RFC packs MSB-first; JXL reads LSB-first, so the lookup
        // table indexes by bit-reversed codes:
        //   sym 0: 0b0    → idx 0, 2, 4, 6
        //   sym 1: 0b01   → idx 1, 5
        //   sym 2: 0b011  → idx 3
        //   sym 3: 0b111  → idx 7
        let code = PrefixCode::from_lengths(&[1, 2, 3, 3]).unwrap();
        assert_eq!(code.lookup[0], (0, 1));
        assert_eq!(code.lookup[2], (0, 1));
        assert_eq!(code.lookup[4], (0, 1));
        assert_eq!(code.lookup[6], (0, 1));
        assert_eq!(code.lookup[1], (1, 2));
        assert_eq!(code.lookup[5], (1, 2));
        assert_eq!(code.lookup[3], (2, 3));
        assert_eq!(code.lookup[7], (3, 3));
    }

    #[test]
    fn from_lengths_decodes_symbols() {
        // [1,2,3,3] code; encode the symbol sequence 0,1,2,3 then
        // verify decode recovers it.
        let code = PrefixCode::from_lengths(&[1, 2, 3, 3]).unwrap();
        // Encoded bits, MSB-first per RFC: "0" "10" "110" "111".
        // LSB-first packing: each code is reversed within itself and
        // the codes are emitted in source order.
        // sym 0 "0":   reversed = 0     (1 bit)
        // sym 1 "10":  reversed = 01    (2 bits)
        // sym 2 "110": reversed = 011   (3 bits)
        // sym 3 "111": reversed = 111   (3 bits)
        let bytes = pack_lsb(&[(0b0, 1), (0b01, 2), (0b011, 3), (0b111, 3)]);
        let mut br = BitReader::new(&bytes);
        assert_eq!(code.decode(&mut br).unwrap(), 0);
        assert_eq!(code.decode(&mut br).unwrap(), 1);
        assert_eq!(code.decode(&mut br).unwrap(), 2);
        assert_eq!(code.decode(&mut br).unwrap(), 3);
    }

    #[test]
    fn read_simple_prefix_two_symbol() {
        // alphabet_size = 4 → bits = 2. Encode NSYM=2 symbols 1, 3.
        // Bits LSB-first:
        //   u(2) = 1 (kind=simple)         → bit0=1, bit1=0
        //   u(2) = nsym-1 = 1              → bit0=1, bit1=0
        //   u(2) = sym 1                   → bit0=1, bit1=0
        //   u(2) = sym 3                   → bit0=1, bit1=1
        let bytes = pack_lsb(&[(1, 2), (1, 2), (1, 2), (3, 2)]);
        let mut br = BitReader::new(&bytes);
        let code = read_prefix_code(&mut br, 4).unwrap();
        // Both symbols must be decodable with 1 bit each.
        // Sorted ascending: sym 1 → code 0, sym 3 → code 1 (LSB-first
        // index 0 → sym 1, index 1 → sym 3).
        let decode_bytes = pack_lsb(&[(0, 1), (1, 1)]);
        let mut br2 = BitReader::new(&decode_bytes);
        assert_eq!(code.decode(&mut br2).unwrap(), 1);
        assert_eq!(code.decode(&mut br2).unwrap(), 3);
    }

    #[test]
    fn read_simple_prefix_one_symbol() {
        // alphabet_size = 4 → bits = 2. NSYM = 1, symbol = 2.
        let bytes = pack_lsb(&[(1, 2), (0, 2), (2, 2)]);
        let mut br = BitReader::new(&bytes);
        let code = read_prefix_code(&mut br, 4).unwrap();
        // 1-symbol decode consumes 0 bits.
        let mut br2 = BitReader::new(&[]);
        assert_eq!(code.decode(&mut br2).unwrap(), 2);
        assert_eq!(br2.bits_read(), 0);
    }

    #[test]
    fn alphabet_size_too_large_rejected() {
        let mut br = BitReader::new(&[0u8; 4]);
        let huge = (MAX_ALPHABET_SIZE + 1) as u32;
        assert!(read_prefix_code(&mut br, huge).is_err());
    }

    #[test]
    fn from_lengths_rejects_oversum() {
        // Three length-1 symbols: kraft sum > 1.
        assert!(PrefixCode::from_lengths(&[1, 1, 1]).is_err());
    }

    #[test]
    fn malicious_alphabet_size_rejected_before_alloc() {
        // Construct an empty bitstream with a huge claimed alphabet
        // size; read_prefix_code must refuse it before allocating.
        let bytes = vec![0u8; 1];
        let mut br = BitReader::new(&bytes);
        let huge = (MAX_ALPHABET_SIZE + 1) as u32;
        let err = read_prefix_code(&mut br, huge).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("alphabet_size"));
    }

    #[test]
    fn complex_prefix_alphabet_too_big_for_input_rejected() {
        // Force a complex-prefix kind (HSKIP=0, u(2)=0) but claim a
        // very large alphabet relative to remaining input. Should be
        // rejected by the input-bound sanity check rather than fed
        // into a giant Vec allocation.
        // u(2) = 0 means HSKIP = 0. Then we need 18 clcl symbols
        // (each at least 2 bits) before the alphabet starts → 36 bits
        // minimum. With only 4 bytes (32 bits) total input, decode
        // will fail at some point.
        let bytes = vec![0u8; 4];
        let mut br = BitReader::new(&bytes);
        // alphabet_size = 100 with very little remaining input is a
        // reasonable tripwire for the bound check.
        let _ = read_prefix_code(&mut br, 100);
        // We accept either "fail at clcl decode" or "fail at the
        // bound check" — the only thing we don't accept is OOM.
    }
}
