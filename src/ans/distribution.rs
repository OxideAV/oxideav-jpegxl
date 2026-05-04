//! ANS distribution decoding — FDIS Annex D.3.4 (Listing D.4, p. 63-64)
//! and the verbatim 128 × 2 `kLogCountLut` lookup table.
//!
//! The output of [`read_distribution`] is a `Vec<u16>` of length
//! `1 << log_alphabet_size` whose entries sum to `1 << 12`, suitable
//! for feeding into `AliasTable::build` (D.3.2).
//!
//! ## Spec typo notes
//!
//! Two FDIS PDF artefacts are documented inline:
//!
//! * The two-symbol short path writes `D[v2] = (1 << 1) – D[v1]`. That
//!   is unambiguously `(1 << 12) – D[v1]`; distributions sum to 4096.
//! * The "uniform" branch writes the same `floor((1 << 12) /
//!   alphabet_size)` value to both halves of the loop, which would not
//!   sum to 4096. The standard reading is `floor + 1` for the first
//!   `(1 << 12) Umod alphabet_size` entries and `floor` for the rest.
//!
//! Both fixes are minimal and required for the decoded distribution to
//! satisfy the `sum to 4096` invariant the rest of D.3 relies on.

use crate::error::{JxlError as Error, Result};

use crate::bitreader::BitReader;

/// FDIS D.3.4 — `kLogCountLut[128][2]` (p. 64, verbatim from
/// Listing D.4 in the published PDF).
///
/// Each entry is `(advance_bits, log_count_value)`: `peek u(7)`, then
/// advance the bitstream by `[0]` bits and use `[1]` as `logcounts[i]`.
/// `logcounts[i] == 13` triggers a run-length-encoded zero/equal block.
///
/// **Audit trail:** transcribed from FDIS 18181-1:2021 Listing D.4
/// page 64 of the FDIS PDF (also reachable in `/tmp/fdis.txt` lines
/// 4145-4161 if the local extracted text is on disk).
pub const K_LOG_COUNT_LUT: [[u8; 2]; 128] = [
    [3, 10],
    [7, 12],
    [3, 7],
    [4, 3],
    [3, 6],
    [3, 8],
    [3, 9],
    [4, 5],
    [3, 10],
    [4, 4],
    [3, 7],
    [4, 1],
    [3, 6],
    [3, 8],
    [3, 9],
    [4, 2],
    [3, 10],
    [5, 0],
    [3, 7],
    [4, 3],
    [3, 6],
    [3, 8],
    [3, 9],
    [4, 5],
    [3, 10],
    [4, 4],
    [3, 7],
    [4, 1],
    [3, 6],
    [3, 8],
    [3, 9],
    [4, 2],
    [3, 10],
    [6, 11],
    [3, 7],
    [4, 3],
    [3, 6],
    [3, 8],
    [3, 9],
    [4, 5],
    [3, 10],
    [4, 4],
    [3, 7],
    [4, 1],
    [3, 6],
    [3, 8],
    [3, 9],
    [4, 2],
    [3, 10],
    [5, 0],
    [3, 7],
    [4, 3],
    [3, 6],
    [3, 8],
    [3, 9],
    [4, 5],
    [3, 10],
    [4, 4],
    [3, 7],
    [4, 1],
    [3, 6],
    [3, 8],
    [3, 9],
    [4, 2],
    [3, 10],
    [7, 13],
    [3, 7],
    [4, 3],
    [3, 6],
    [3, 8],
    [3, 9],
    [4, 5],
    [3, 10],
    [4, 4],
    [3, 7],
    [4, 1],
    [3, 6],
    [3, 8],
    [3, 9],
    [4, 2],
    [3, 10],
    [5, 0],
    [3, 7],
    [4, 3],
    [3, 6],
    [3, 8],
    [3, 9],
    [4, 5],
    [3, 10],
    [4, 4],
    [3, 7],
    [4, 1],
    [3, 6],
    [3, 8],
    [3, 9],
    [4, 2],
    [3, 10],
    [6, 11],
    [3, 7],
    [4, 3],
    [3, 6],
    [3, 8],
    [3, 9],
    [4, 5],
    [3, 10],
    [4, 4],
    [3, 7],
    [4, 1],
    [3, 6],
    [3, 8],
    [3, 9],
    [4, 2],
    [3, 10],
    [5, 0],
    [3, 7],
    [4, 3],
    [3, 6],
    [3, 8],
    [3, 9],
    [4, 5],
    [3, 10],
    [4, 4],
    [3, 7],
    [4, 1],
    [3, 6],
    [3, 8],
    [3, 9],
    [4, 2],
];

/// Hard cap on alphabet size for D.3.4 reads, matching the
/// `log_alphabet_size <= 15` ceiling from D.3.1. Used to bound
/// `Vec::with_capacity` against malformed bitstreams.
pub const ALPHABET_SIZE_MAX: usize = 1 << 15;

/// Decode an ANS distribution per FDIS D.3.4.
///
/// Returns a `Vec<u16>` of length `1 << log_alphabet_size` whose entries
/// sum to `1 << 12 = 4096` (the ANS_TAB_SIZE invariant).
pub fn read_distribution(br: &mut BitReader<'_>, log_alphabet_size: u32) -> Result<Vec<u16>> {
    if log_alphabet_size > 15 {
        return Err(Error::InvalidData(
            "JXL ANS distribution: log_alphabet_size > 15".into(),
        ));
    }
    let table_size: usize = 1usize << log_alphabet_size;
    if table_size > ALPHABET_SIZE_MAX {
        return Err(Error::InvalidData(
            "JXL ANS distribution: alphabet size too large".into(),
        ));
    }

    let mut d = vec![0u16; table_size];

    // Branch 1: explicit single-symbol or two-symbol distribution.
    if br.read_bit()? == 1 {
        let ns = br.read_bit()? + 1;
        if ns == 1 {
            let x = br.read_u8_value()? as usize;
            if x >= table_size {
                return Err(Error::InvalidData(
                    "JXL ANS distribution (1-sym): symbol out of alphabet".into(),
                ));
            }
            d[x] = 4096;
        } else {
            let v1 = br.read_u8_value()? as usize;
            let v2 = br.read_u8_value()? as usize;
            if v1 == v2 {
                return Err(Error::InvalidData(
                    "JXL ANS distribution (2-sym): v1 == v2".into(),
                ));
            }
            if v1 >= table_size || v2 >= table_size {
                return Err(Error::InvalidData(
                    "JXL ANS distribution (2-sym): symbol out of alphabet".into(),
                ));
            }
            let p = br.read_bits(12)?;
            if p > 4096 {
                return Err(Error::InvalidData(
                    "JXL ANS distribution (2-sym): probability > 4096".into(),
                ));
            }
            // Spec PDF reads `(1 << 1) - D[v1]` — clear typo, see module
            // docs. The correct sum is `1 << 12 = 4096`.
            d[v1] = p as u16;
            d[v2] = (4096 - p) as u16;
        }
        return Ok(d);
    }

    // Branch 2: flat / uniform distribution.
    if br.read_bit()? == 1 {
        let alphabet_size = br.read_u8_value()? as usize + 1;
        if alphabet_size > table_size {
            return Err(Error::InvalidData(
                "JXL ANS distribution (flat): alphabet_size > table_size".into(),
            ));
        }
        if alphabet_size == 0 {
            return Err(Error::InvalidData(
                "JXL ANS distribution (flat): alphabet_size = 0".into(),
            ));
        }
        // Spec PDF prints both halves of the partition with the same
        // floor — clear typo. The first `4096 mod alphabet_size`
        // entries get `floor + 1`, the rest get `floor`, so the entries
        // sum to exactly 4096.
        let floor_v = (4096u32 / alphabet_size as u32) as u16;
        let remainder = 4096usize % alphabet_size;
        for slot in d.iter_mut().take(remainder) {
            *slot = floor_v + 1;
        }
        for slot in d.iter_mut().take(alphabet_size).skip(remainder) {
            *slot = floor_v;
        }
        return Ok(d);
    }

    // Branch 3: general path with kLogCountLut.
    let mut len: u32 = 0;
    while len < 3 {
        if br.read_bit()? == 1 {
            len += 1;
        } else {
            break;
        }
    }
    let shift = br.read_bits(len)? + (1u32 << len) - 1;
    if shift > (1u32 << 12) + 1 {
        return Err(Error::InvalidData(
            "JXL ANS distribution: shift out of range".into(),
        ));
    }
    let alphabet_size = br.read_u8_value()? as usize + 3;
    if alphabet_size > table_size {
        return Err(Error::InvalidData(
            "JXL ANS distribution: alphabet_size > table_size".into(),
        ));
    }

    // logcounts is bounded by alphabet_size which is bounded by
    // table_size which is at most 2^15. ALPHABET_SIZE_MAX gates this.
    let mut logcounts = vec![0u8; alphabet_size];
    let mut same = vec![0u32; alphabet_size];
    let mut omit_log: i32 = -1;
    let mut omit_pos: i32 = -1;

    let mut i: usize = 0;
    while i < alphabet_size {
        let h = br.peek_bits(7)? as usize;
        let advance = K_LOG_COUNT_LUT[h][0] as u32;
        br.advance_bits(advance)?;
        let lc = K_LOG_COUNT_LUT[h][1];
        logcounts[i] = lc;
        if lc == 13 {
            let rle = br.read_u8_value()? as usize;
            same[i] = (rle + 5) as u32;
            // Skip the next (rle + 3) entries, but cap at alphabet_size.
            let skip = rle
                .checked_add(3)
                .ok_or_else(|| Error::InvalidData("JXL ANS distribution: rle overflow".into()))?;
            i = i.checked_add(1 + skip).ok_or_else(|| {
                Error::InvalidData("JXL ANS distribution: rle index overflow".into())
            })?;
            if i > alphabet_size {
                return Err(Error::InvalidData(
                    "JXL ANS distribution: rle overruns alphabet".into(),
                ));
            }
            continue;
        }
        if lc as i32 > omit_log {
            omit_log = lc as i32;
            omit_pos = i as i32;
        }
        i += 1;
    }
    if omit_pos < 0 {
        return Err(Error::InvalidData(
            "JXL ANS distribution: omit_pos undefined".into(),
        ));
    }
    let omit_pos_u = omit_pos as usize;
    if omit_pos_u + 1 < alphabet_size && logcounts[omit_pos_u + 1] == 13 {
        return Err(Error::InvalidData(
            "JXL ANS distribution: omit_pos followed by rle".into(),
        ));
    }

    let mut total_count: u32 = 0;
    let mut prev: u16 = 0;
    let mut numsame: u32 = 0;
    for i in 0..alphabet_size {
        if same[i] != 0 {
            numsame = same[i] - 1;
            prev = if i > 0 { d[i - 1] } else { 0 };
        }
        if numsame > 0 {
            d[i] = prev;
            numsame -= 1;
            total_count = total_count
                .checked_add(d[i] as u32)
                .ok_or_else(|| Error::InvalidData("JXL ANS distribution: total overflow".into()))?;
        } else {
            let code = logcounts[i];
            if i == omit_pos_u || code == 0 {
                continue;
            }
            if code == 1 {
                d[i] = 1;
                total_count = total_count.checked_add(1).ok_or_else(|| {
                    Error::InvalidData("JXL ANS distribution: total overflow".into())
                })?;
            } else {
                // bitcount = min(max(0, shift – ((12 – code + 1) >> 1)),
                //                code - 1)
                let inner = ((12i32 - code as i32 + 1) >> 1).max(0) as u32;
                let raw = (shift as i32 - inner as i32).max(0) as u32;
                let bitcount = raw.min(code as u32 - 1);
                let extra = br.read_bits(bitcount)?;
                let val = (1u32 << (code - 1)) + (extra << (code as u32 - 1 - bitcount));
                if val > 4096 {
                    return Err(Error::InvalidData(
                        "JXL ANS distribution: per-symbol value > 4096".into(),
                    ));
                }
                d[i] = val as u16;
                total_count = total_count.checked_add(val).ok_or_else(|| {
                    Error::InvalidData("JXL ANS distribution: total overflow".into())
                })?;
            }
        }
    }
    if total_count > 4096 {
        return Err(Error::InvalidData(
            "JXL ANS distribution: total_count > 4096".into(),
        ));
    }
    d[omit_pos_u] = (4096 - total_count) as u16;
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ans::test_helpers::pack_lsb;

    #[test]
    fn k_log_count_lut_size() {
        assert_eq!(K_LOG_COUNT_LUT.len(), 128);
    }

    #[test]
    fn single_symbol_distribution_round_trip() {
        // u(1)=1 (explicit) ; u(1)=0 (ns=1) ; U8() encodes symbol 5.
        // U8 for 5: bit0=1 (nonzero), then n=u(3)=2 (since 5 = 1<<2 + 1),
        // then u(2) = 1 → value = 1 + (1<<2) = 5.
        // Bit stream LSB→MSB:
        //   1, 0,  // explicit, ns=1
        //   1,     // U8 nonzero
        //   0,1,0, // n = 2 (LSB first: bit0=0,bit1=1,bit2=0)
        //   1,0,   // u(2) = 1 (LSB first)
        let bytes = pack_lsb(&[(1, 1), (0, 1), (1, 1), (2, 3), (1, 2)]);
        let mut br = BitReader::new(&bytes);
        let d = read_distribution(&mut br, 3).unwrap();
        assert_eq!(d.len(), 8);
        assert_eq!(d[5], 4096);
        let sum: u32 = d.iter().map(|&x| x as u32).sum();
        assert_eq!(sum, 4096);
    }

    #[test]
    fn two_symbol_distribution_round_trip() {
        // u(1)=1 (explicit) ; u(1)=1 (ns=2) ; U8(v1=2) ; U8(v2=4) ; u(12) prob.
        // U8(2): nonzero=1, n = 1 (since 2 = 1<<1 + 0), u(1)=0.
        //   bits: 1, 1,0,0, 0
        //   → flag=1, n=1, payload=0
        // U8(4): nonzero=1, n = 2 (since 4 = 1<<2 + 0), u(2)=0.
        //   bits: 1, 0,1,0, 0,0
        // u(12) for prob = 1234.
        let bytes = pack_lsb(&[
            (1, 1),
            (1, 1), // explicit, ns=2
            (1, 1),
            (1, 3),
            (0, 1), // U8 v1=2
            (1, 1),
            (2, 3),
            (0, 2),     // U8 v2=4
            (1234, 12), // probability
        ]);
        let mut br = BitReader::new(&bytes);
        let d = read_distribution(&mut br, 3).unwrap();
        assert_eq!(d[2], 1234);
        assert_eq!(d[4], 4096 - 1234);
        let sum: u32 = d.iter().map(|&x| x as u32).sum();
        assert_eq!(sum, 4096);
    }

    #[test]
    fn flat_distribution_round_trip() {
        // u(1)=0 (not explicit) ; u(1)=1 (flat) ; U8(alphabet_size-1).
        // alphabet_size = 5 → U8 reads value 4. U8(4) bits: 1, 0,1,0, 0,0
        let bytes = pack_lsb(&[(0, 1), (1, 1), (1, 1), (2, 3), (0, 2)]);
        let mut br = BitReader::new(&bytes);
        let d = read_distribution(&mut br, 3).unwrap();
        // 4096 / 5 = 819 remainder 1. First 1 entry = 820, next 4 = 819.
        // 820 + 4*819 = 820 + 3276 = 4096.
        assert_eq!(d[0], 820);
        assert_eq!(d[1], 819);
        assert_eq!(d[2], 819);
        assert_eq!(d[3], 819);
        assert_eq!(d[4], 819);
        assert_eq!(d[5], 0);
        let sum: u32 = d.iter().map(|&x| x as u32).sum();
        assert_eq!(sum, 4096);
    }

    #[test]
    fn explicit_single_symbol_out_of_range_rejected() {
        // log_alphabet_size = 3 → table_size = 8. Encode symbol 200.
        // U8(200): nonzero=1, find n s.t. 200 = (1<<n) + extra; 200 = 128 + 72,
        // n = 7, extra = 72 (u(7) = 0b1001000).
        let bytes = pack_lsb(&[(1, 1), (0, 1), (1, 1), (7, 3), (72, 7)]);
        let mut br = BitReader::new(&bytes);
        assert!(read_distribution(&mut br, 3).is_err());
    }

    #[test]
    fn rejects_log_alphabet_size_too_large() {
        let bytes = vec![0u8; 4];
        let mut br = BitReader::new(&bytes);
        assert!(read_distribution(&mut br, 16).is_err());
    }

    #[test]
    fn malicious_alphabet_size_capped_at_table_size() {
        // log_alphabet_size = 3 → table_size = 8. Force the bitstream
        // to claim alphabet_size = 256+3 = 259 (U8 max + 3); this
        // exceeds table_size and must be rejected, NOT trigger an
        // allocation of the malicious size.
        // Branch 3: u(1)=0, u(1)=0, then `len` and `shift` reads.
        // U8(255): nonzero=1, n=u(3)=7 → bits 1,1,1, then u(7)=127.
        let bytes = pack_lsb(&[
            (0, 1),
            (0, 1), // not-explicit, not-flat
            (0, 1), // len = 0 (loop break)
            (0, 0), // shift = u(0) + (1<<0) - 1 = 0
            (1, 1),
            (7, 3),
            (127, 7), // U8 = 255 → alphabet_size = 258
        ]);
        let mut br = BitReader::new(&bytes);
        assert!(read_distribution(&mut br, 3).is_err());
    }

    #[test]
    fn malicious_log_alphabet_size_rejected_before_alloc() {
        // Even with log_alphabet_size = 16 (one above the cap), the
        // alphabet would be 64KiB entries — we error before allocating.
        let bytes = vec![0u8; 4];
        let mut br = BitReader::new(&bytes);
        let err = read_distribution(&mut br, 16).unwrap_err();
        // Sanity: error mentions log_alphabet_size or > 15.
        let msg = format!("{err:?}");
        assert!(msg.contains("log_alphabet_size") || msg.contains("> 15"));
    }

    #[test]
    fn flat_zero_alphabet_is_rejected() {
        // u(1)=0, u(1)=1, U8 = 0 → alphabet_size = 1, but we need to
        // force alphabet_size = 0 which U8+1 cannot do. Instead this
        // path should produce alphabet_size=1; verify the smallest valid
        // case still works (single-symbol uniform with probability 4096).
        let bytes = pack_lsb(&[
            (0, 1),
            (1, 1),
            (0, 1), // U8 = 0
        ]);
        let mut br = BitReader::new(&bytes);
        let d = read_distribution(&mut br, 3).unwrap();
        assert_eq!(d[0], 4096);
    }
}
