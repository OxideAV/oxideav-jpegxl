//! Alias mapping — FDIS Annex D.3.2 (Listings D.1 and D.2, p. 62).
//!
//! Builds three lookup tables (`symbols`, `offsets`, `cutoffs`) from a
//! probability distribution `D` of size `1 << log_alphabet_size`, then
//! exposes `lookup(x)` that maps an index `x` (the low 12 bits of the
//! current ANS state) to a `(symbol, offset)` pair as required by D.3.3.
//!
//! The construction uses Vose's alias method ("overfull/underfull"
//! buckets). The spec text in Listing D.1 uses `i` as the loop variable
//! inside the overfull/underfull pop loop, which is a known PDF
//! formatting glitch — we follow the *intent* of Vose, which is to
//! write `symbols[u]` and `offsets[u]` (i.e. into the underfull slot
//! that we just popped). This matches the only sensible reading: the
//! variable `i` was already used by the preceding loops to populate
//! `cutoffs`; here we want to record that the underfull bucket `u`
//! redirects its post-cutoff range to symbol `o` at offset
//! `cutoffs[o]`.

use crate::error::{JxlError as Error, Result};

/// Maximum supported `log_alphabet_size`. FDIS D.3.1 caps this at 15
/// for the ANS path; we cap a bit lower than the bitreader's allocation
/// guard to stay well clear of any 1-bit overflow.
pub const LOG_ALPHABET_SIZE_MAX: u32 = 15;

/// 1 << 12 — the ANS distribution sum.
pub const ANS_TAB_SIZE: u32 = 1 << 12;

/// The three precomputed lookup tables for the alias method.
#[derive(Debug, Clone)]
pub struct AliasTable {
    /// Each index maps to either its own symbol (when below `cutoffs[i]`)
    /// or to `symbols[i]` (when at/above the cutoff).
    pub symbols: Vec<u16>,
    /// The offset to add to the in-bucket position when redirecting.
    pub offsets: Vec<u16>,
    /// The cutoff inside each bucket between "stays here" and "redirected".
    pub cutoffs: Vec<u16>,
    /// log2(bucket_size) = 12 - log_alphabet_size.
    pub log_bucket_size: u32,
}

impl AliasTable {
    /// Build the alias table for distribution `d` (length must be
    /// `1 << log_alphabet_size`, entries non-negative summing to
    /// `1 << 12`).
    ///
    /// `log_alphabet_size` is bounded by [`LOG_ALPHABET_SIZE_MAX`].
    pub fn build(d: &[u16], log_alphabet_size: u32) -> Result<Self> {
        if log_alphabet_size > LOG_ALPHABET_SIZE_MAX {
            return Err(Error::InvalidData(
                "JXL AliasTable: log_alphabet_size out of range".into(),
            ));
        }
        let table_size = 1usize << log_alphabet_size;
        if d.len() != table_size {
            return Err(Error::InvalidData(
                "JXL AliasTable: distribution length mismatch".into(),
            ));
        }

        let log_bucket_size = 12 - log_alphabet_size;
        let bucket_size: u32 = 1u32 << log_bucket_size;

        // Identify single-symbol distributions (Listing D.1 short path).
        let mut nonzero_count = 0usize;
        let mut single_idx = 0usize;
        let mut max_symbol = 0usize;
        for (i, &v) in d.iter().enumerate() {
            if v != 0 {
                nonzero_count += 1;
                single_idx = i;
                max_symbol = i;
            }
        }
        if nonzero_count == 0 {
            return Err(Error::InvalidData(
                "JXL AliasTable: distribution has zero non-zero symbols".into(),
            ));
        }
        let total: u32 = d.iter().map(|&v| v as u32).sum();
        if total != ANS_TAB_SIZE {
            return Err(Error::InvalidData(
                "JXL AliasTable: distribution does not sum to 4096".into(),
            ));
        }

        // Single-symbol shortcut from Listing D.1.
        if nonzero_count == 1 {
            let mut symbols = vec![0u16; ANS_TAB_SIZE as usize];
            let mut offsets = vec![0u16; ANS_TAB_SIZE as usize];
            let cutoffs = vec![0u16; ANS_TAB_SIZE as usize];
            for i in 0..ANS_TAB_SIZE as usize {
                symbols[i] = single_idx as u16;
                offsets[i] = (bucket_size as usize * i) as u16;
            }
            return Ok(Self {
                symbols,
                offsets,
                cutoffs,
                log_bucket_size,
            });
        }

        // General case (Listing D.1 main path).
        let mut symbols = vec![0u16; table_size];
        let mut offsets = vec![0u16; table_size];
        let mut cutoffs = vec![0u16; table_size];
        let mut overfull: Vec<usize> = Vec::with_capacity(table_size);
        let mut underfull: Vec<usize> = Vec::with_capacity(table_size);

        // The FDIS Listing D.1 writes `for (i = 0; i < max_symbol; i++)`
        // strictly, then `for (i = max_symbol; i < table_size; i++)
        // cutoffs[i] = 0`. With `max_symbol` defined as the highest
        // index with non-zero probability, the strict reading omits
        // the bucket at index `max_symbol` itself, which is precisely
        // where Vose needs the highest-probability symbol's mass to
        // start. The only consistent interpretation (and the one that
        // makes Vose's algorithm terminate) is `i <= max_symbol` /
        // `i > max_symbol`. We implement that.
        for i in 0..=max_symbol {
            cutoffs[i] = d[i];
            if d[i] as u32 > bucket_size {
                overfull.push(i);
            } else {
                underfull.push(i);
            }
        }
        for (i, slot) in cutoffs
            .iter_mut()
            .enumerate()
            .take(table_size)
            .skip(max_symbol + 1)
        {
            *slot = 0;
            underfull.push(i);
        }

        // Vose pump. The FDIS Listing D.1 spec text mixes the loop
        // variables `o`, `u`, and `i` in this block — `cutoffs[u] -= by`
        // is unambiguously meant to be `cutoffs[o] -= by` (the
        // overfull bucket loses `by`), and `symbols[i] / offsets[i]`
        // are unambiguously `symbols[u] / offsets[u]` (we are
        // redirecting the underfull bucket to symbol `o`). Anything
        // else makes Vose's algorithm impossible to terminate. We
        // implement the corrected version, which is the standard
        // textbook Vose alias method.
        while !overfull.is_empty() {
            let o = overfull.pop().ok_or_else(|| {
                Error::InvalidData("JXL AliasTable: overfull pop on empty".into())
            })?;
            let u = underfull.pop().ok_or_else(|| {
                Error::InvalidData(
                    "JXL AliasTable: underfull exhausted before overfull (malformed distribution)"
                        .into(),
                )
            })?;
            let by = bucket_size - cutoffs[u] as u32;
            // Overfull bucket gives up `by` probability mass.
            let new_o = cutoffs[o] as u32 - by;
            symbols[u] = o as u16;
            offsets[u] = new_o as u16;
            cutoffs[o] = new_o as u16;
            if new_o < bucket_size {
                underfull.push(o);
            } else if new_o > bucket_size {
                overfull.push(o);
            }
            // If new_o == bucket_size we leave it alone — the final
            // reconciliation loop will mark it self-mapping.
        }

        // Final reconciliation (Listing D.1 trailing loop).
        for i in 0..table_size {
            if (cutoffs[i] as u32) == bucket_size {
                symbols[i] = i as u16;
                offsets[i] = 0;
                cutoffs[i] = 0;
            } else {
                offsets[i] = (offsets[i] as u32 - cutoffs[i] as u32) as u16;
            }
        }

        Ok(Self {
            symbols,
            offsets,
            cutoffs,
            log_bucket_size,
        })
    }

    /// `AliasMapping(x)` per FDIS Listing D.2.
    ///
    /// `x` is masked into `[0, 4096)` before lookup; the caller is
    /// expected to pass `state & 0xFFF` from the ANS decode loop. The
    /// returned `(symbol, offset)` is then plugged into the standard
    /// ANS update `state = D[symbol] * (state >> 12) + offset`.
    pub fn lookup(&self, x: u32) -> (u16, u32) {
        let bucket_size: u32 = 1u32 << self.log_bucket_size;
        let i = (x >> self.log_bucket_size) as usize;
        let pos = x & (bucket_size - 1);
        // The build path validates lengths so `i < symbols.len()` always
        // holds for a well-built table; we still guard to never panic
        // on an out-of-bounds state.
        if i >= self.symbols.len() {
            return (0, 0);
        }
        let symbol = if pos >= self.cutoffs[i] as u32 {
            self.symbols[i]
        } else {
            i as u16
        };
        let offset = self.offsets[i] as u32 + pos;
        (symbol, offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_log_alphabet_size_too_large() {
        let d = vec![0u16; 1 << 16];
        assert!(AliasTable::build(&d, 16).is_err());
    }

    #[test]
    fn rejects_wrong_length_distribution() {
        let d = vec![0u16; 7];
        assert!(AliasTable::build(&d, 3).is_err());
    }

    #[test]
    fn rejects_distribution_not_summing_to_4096() {
        let mut d = vec![0u16; 8];
        d[0] = 1000; // sum = 1000, not 4096.
        assert!(AliasTable::build(&d, 3).is_err());
    }

    #[test]
    fn single_symbol_distribution_short_path() {
        let mut d = vec![0u16; 8];
        d[3] = 4096;
        let t = AliasTable::build(&d, 3).unwrap();
        // Every bucket in the 4096-element table maps to symbol 3.
        for i in 0..ANS_TAB_SIZE {
            let (sym, _) = t.lookup(i);
            assert_eq!(sym, 3);
        }
    }

    #[test]
    fn uniform_distribution_each_bucket_self_maps() {
        // table_size = 8, each entry = 512 = bucket_size.
        let d = vec![512u16; 8];
        let t = AliasTable::build(&d, 3).unwrap();
        // bucket_size = 512, so every bucket's cutoff is 512 = bucket_size,
        // which the trailing loop sets to self-mapping (offset=0, cutoff=0).
        for i in 0..8 {
            assert_eq!(t.symbols[i], i as u16);
            assert_eq!(t.offsets[i], 0);
            assert_eq!(t.cutoffs[i], 0);
        }
        // Lookup: symbol = i, offset = pos.
        for x in 0..ANS_TAB_SIZE {
            let (sym, off) = t.lookup(x);
            let expected_sym = (x >> 9) as u16;
            let expected_off = x & 0x1FF;
            assert_eq!(sym, expected_sym);
            assert_eq!(off, expected_off);
        }
    }

    #[test]
    fn two_symbol_skewed_distribution_aliases_correctly() {
        // Two symbols, table_size = 8, bucket_size = 512.
        // Put weight 4000 on symbol 0, 96 on symbol 1.
        let mut d = vec![0u16; 8];
        d[0] = 4000;
        d[1] = 96;
        let t = AliasTable::build(&d, 3).unwrap();
        // The composite mapping should cover the range [0, 4096) and
        // produce symbol 0 exactly 4000 times and symbol 1 exactly 96
        // times overall.
        let mut count0 = 0u32;
        let mut count1 = 0u32;
        let mut other = 0u32;
        for x in 0..ANS_TAB_SIZE {
            let (sym, _) = t.lookup(x);
            match sym {
                0 => count0 += 1,
                1 => count1 += 1,
                _ => other += 1,
            }
        }
        assert_eq!(count0 + count1 + other, ANS_TAB_SIZE);
        assert_eq!(other, 0, "alias table emitted unsupported symbol");
        assert_eq!(count0, 4000);
        assert_eq!(count1, 96);
    }
}
