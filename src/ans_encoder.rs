//! ANS entropy ENCODER — round-3 inverse of `crate::ans::symbol::AnsDecoder`
//! and `crate::ans::distribution::read_distribution`.
//!
//! The decoder side (FDIS Annex D.3.3 + D.3.4) is already in
//! `src/ans/`. This module is its forward (encoder-side) twin: given a
//! sequence of symbols and a per-symbol probability distribution, it
//! emits the on-wire ANS bitstream that the existing decoder will read
//! back to recover the same symbol sequence.
//!
//! ## High-level algorithm
//!
//! 1. **Quantise** raw symbol counts into a non-negative `D[]` array of
//!    length `1 << log_alphabet_size` with `sum(D) == 4096`. Round-3 uses
//!    `log_alphabet_size = 5` (alphabet size 32) for the symbol stream
//!    so the preamble fits in tens of bytes — the kSpecial-distance and
//!    kLogCountLut tables stay tiny.
//!
//! 2. **Inverse alias table** — for each symbol `s` with `D[s]` non-zero,
//!    pre-compute `inv_alias[s][k]` (`k in [0, D[s])`) by iterating
//!    through all 4096 alias-table indices and asking
//!    [`AliasTable::lookup`] which `(symbol, offset)` each maps to.
//!
//! 3. **Reverse-walk encode** — process symbols in REVERSE order. For
//!    each symbol `s`:
//!    - While `state >= D[s] << 20` (i.e. the inverse update would
//!      overflow the 32-bit state), push the low 16 bits of `state`
//!      onto a stack and shift `state` right by 16.
//!    - Apply the inverse update:
//!      `state = ((state / D[s]) << 12) | inv_alias[s][state mod D[s]]`.
//!
//! 4. **Bitstream emission**: write `state` as a leading `u(32)` (the
//!    decoder's `AnsDecoder::new` reads this first), then pop the
//!    refill stack in LIFO order writing each `u(16)` (so the decoder
//!    reads them in encoder-emission order — i.e. as it decoded each
//!    symbol forward).
//!
//! 5. **Distribution preamble** is emitted via [`write_distribution`].
//!    Round-3 uses the *uniform / two-symbol / single-symbol* short
//!    paths from FDIS D.3.4 where applicable, falling back to the
//!    general `kLogCountLut` path. The encoder targets the simplest
//!    encoding the decoder accepts, NOT the smallest possible — round-4
//!    can switch to a frequency-tuned `logcounts` table.
//!
//! ## Bit budget vs round-2 prefix
//!
//! Round-2 emits exactly 4 bits per symbol (16-symbol uniform Huffman
//! code) regardless of frequency. Round-3 ANS approaches the entropy
//! bound `H(D) = -sum(p_i * log2(p_i))`. For a Gradient-residual
//! channel where ~80% of residuals are 0, H ≈ 0.7 bits/pixel — a
//! ~5.7× compression improvement over round 2. The cost is a one-time
//! ~30-50 byte distribution preamble (negligible above ~100 pixels).

use oxideav_core::{Error, Result};

use crate::ans::alias::AliasTable;
use crate::ans::distribution::K_LOG_COUNT_LUT;
use crate::ans::symbol::ANS_FINAL_STATE;
use crate::bitwriter::BitWriter;

/// ANS table size invariant — distributions sum to this.
pub const ANS_TAB_SIZE: u32 = 1 << 12;

/// Precision bits — `state & ((1 << 12) - 1)` gives the alias index.
pub const ANS_LOG_TAB_SIZE: u32 = 12;

/// Number of refill bits per `u(16)` emission. Hard-coded by FDIS
/// Annex D.3.3 (the `(state << 16) | u(16)` refill).
pub const ANS_REFILL_BITS: u32 = 16;

/// Quantise raw symbol histogram counts into a `D[]` array of length
/// `1 << log_alphabet_size` with `sum(D) == 4096`.
///
/// Algorithm: per-symbol target = `round(count_i / total * 4096)`,
/// clamped to `[1, 4096]` for non-zero counts; zero counts stay 0.
/// Then a final adjustment loop nudges entries by ±1 until the sum
/// hits exactly 4096 — taking from the largest entry to lower values
/// (or adding to the largest) to minimise distortion.
///
/// `counts.len()` must be `<= 1 << log_alphabet_size`. Symbols beyond
/// `counts.len()` get probability 0.
pub fn quantise_distribution(counts: &[u32], log_alphabet_size: u32) -> Result<Vec<u16>> {
    if log_alphabet_size > 15 {
        return Err(Error::other(
            "ANS encoder: log_alphabet_size > 15 not supported",
        ));
    }
    let table_size: usize = 1usize << log_alphabet_size;
    if counts.len() > table_size {
        return Err(Error::other(
            "ANS encoder: counts.len() exceeds 1 << log_alphabet_size",
        ));
    }
    let total: u64 = counts.iter().map(|&c| c as u64).sum();
    if total == 0 {
        return Err(Error::other(
            "ANS encoder: cannot quantise an all-zero histogram",
        ));
    }

    let mut d: Vec<u16> = vec![0u16; table_size];

    // Initial pass: floor(c * 4096 / total), clamped to [1, 4096].
    // Track a "leftover fractional" to allocate spare units fairly.
    let mut nonzero_indices: Vec<usize> = Vec::new();
    let mut sum_assigned: u32 = 0;
    for (i, &c) in counts.iter().enumerate() {
        if c == 0 {
            continue;
        }
        nonzero_indices.push(i);
        let target = ((c as u64 * ANS_TAB_SIZE as u64) / total).max(1) as u32;
        d[i] = target.min(ANS_TAB_SIZE) as u16;
        sum_assigned += d[i] as u32;
    }
    if nonzero_indices.is_empty() {
        return Err(Error::other(
            "ANS encoder: histogram has no non-zero counts",
        ));
    }

    // If only one symbol is non-zero, give it the full mass — the alias
    // table builder uses a short-circuit path for single-symbol
    // distributions.
    if nonzero_indices.len() == 1 {
        let idx = nonzero_indices[0];
        d[idx] = ANS_TAB_SIZE as u16;
        return Ok(d);
    }

    // Adjust to hit exactly 4096 by nudging entries.
    use core::cmp::Ordering;
    match sum_assigned.cmp(&ANS_TAB_SIZE) {
        Ordering::Equal => {}
        Ordering::Less => {
            let mut deficit = ANS_TAB_SIZE - sum_assigned;
            // Add to the largest entries first.
            let mut sorted: Vec<usize> = nonzero_indices.clone();
            sorted.sort_by_key(|&i| std::cmp::Reverse(d[i]));
            let mut k = 0usize;
            while deficit > 0 {
                let i = sorted[k % sorted.len()];
                if (d[i] as u32) < ANS_TAB_SIZE {
                    d[i] += 1;
                    deficit -= 1;
                }
                k += 1;
                // Safety: nonzero_indices is non-empty and at least one
                // entry is < ANS_TAB_SIZE (since sum < ANS_TAB_SIZE and
                // each entry <= ANS_TAB_SIZE).
                if k > sorted.len() * (ANS_TAB_SIZE as usize) {
                    return Err(Error::other(
                        "ANS encoder: deficit allocation did not converge (BUG)",
                    ));
                }
            }
        }
        Ordering::Greater => {
            let mut excess = sum_assigned - ANS_TAB_SIZE;
            // Remove from the largest entries first, but never drop a
            // non-zero entry below 1.
            let mut sorted: Vec<usize> = nonzero_indices.clone();
            sorted.sort_by_key(|&i| std::cmp::Reverse(d[i]));
            let mut k = 0usize;
            while excess > 0 {
                let i = sorted[k % sorted.len()];
                if d[i] > 1 {
                    d[i] -= 1;
                    excess -= 1;
                }
                k += 1;
                if k > sorted.len() * (ANS_TAB_SIZE as usize) {
                    return Err(Error::other(
                        "ANS encoder: excess removal did not converge (BUG)",
                    ));
                }
            }
        }
    }

    // Sanity check.
    let final_sum: u32 = d.iter().map(|&v| v as u32).sum();
    if final_sum != ANS_TAB_SIZE {
        return Err(Error::other(format!(
            "ANS encoder: post-quantise sum {final_sum} != 4096 (BUG)"
        )));
    }

    Ok(d)
}

/// Build the inverse-alias table for a quantised distribution `d`.
///
/// `inv_alias[s]` is a `Vec<u16>` of length `D[s]` such that for each
/// `k in [0, D[s])`, `inv_alias[s][k]` is the alias-table index
/// `x in [0, 4096)` whose `AliasMapping(x)` returns `(s, k)`.
///
/// Construction: walk all 4096 alias-table indices once, calling
/// `alias.lookup(x)` and recording the inverse mapping.
pub fn build_inverse_alias(d: &[u16], alias: &AliasTable) -> Result<Vec<Vec<u16>>> {
    let mut inv: Vec<Vec<u16>> = d.iter().map(|&p| vec![0u16; p as usize]).collect();
    let mut filled: Vec<u16> = vec![0u16; d.len()];
    for x in 0..ANS_TAB_SIZE {
        let (sym, off) = alias.lookup(x);
        let s = sym as usize;
        if s >= d.len() {
            return Err(Error::other(
                "ANS encoder: alias.lookup returned symbol >= d.len()",
            ));
        }
        let p = d[s] as u32;
        if off >= p {
            return Err(Error::other(format!(
                "ANS encoder: alias.lookup returned offset {off} >= D[{s}]={p}"
            )));
        }
        if filled[s] >= p as u16 {
            return Err(Error::other(format!(
                "ANS encoder: inv_alias[{s}] over-filled (alias table inconsistent)"
            )));
        }
        inv[s][off as usize] = x as u16;
        filled[s] += 1;
    }
    for (s, &p) in d.iter().enumerate() {
        if filled[s] != p {
            return Err(Error::other(format!(
                "ANS encoder: inv_alias[{s}] filled {} != D[{s}]={p}",
                filled[s]
            )));
        }
    }
    Ok(inv)
}

/// Encode a symbol sequence through ANS into the supplied
/// [`BitWriter`]. Returns the number of bits emitted (state + refills,
/// excluding the distribution preamble which the caller emits
/// separately via [`write_distribution`]).
///
/// **Bit ordering**: the leading `u(32)` is the final encoder state; the
/// decoder reads this via `BitReader::read_bits(32)` and stores it in
/// its `state` field. Each subsequent `u(16)` is an LSB-first refill,
/// emitted in the order the decoder will read them.
pub fn encode_symbols(
    bw: &mut BitWriter,
    symbols: &[u16],
    d: &[u16],
    inv_alias: &[Vec<u16>],
) -> Result<()> {
    // Validate the distribution sums.
    let sum: u32 = d.iter().map(|&v| v as u32).sum();
    if sum != ANS_TAB_SIZE {
        return Err(Error::other(format!(
            "ANS encoder: distribution does not sum to 4096 (got {sum})"
        )));
    }
    // Pre-validate inv_alias dimensions.
    if inv_alias.len() != d.len() {
        return Err(Error::other(format!(
            "ANS encoder: inv_alias len {} != d.len() {}",
            inv_alias.len(),
            d.len()
        )));
    }

    // Reverse-walk encode. State starts at the magic decoder-end
    // sentinel ANS_FINAL_STATE so the decoder, which initialises with
    // the leading u(32), starts at the correct state.
    let mut state: u32 = ANS_FINAL_STATE;
    let mut refill_stack: Vec<u16> = Vec::with_capacity(symbols.len() / 4);

    // Iterate symbols in REVERSE order.
    for &sym in symbols.iter().rev() {
        let s = sym as usize;
        if s >= d.len() {
            return Err(Error::other(format!(
                "ANS encoder: symbol {sym} out of distribution range {}",
                d.len()
            )));
        }
        let p = d[s] as u32;
        if p == 0 {
            return Err(Error::other(format!(
                "ANS encoder: symbol {sym} has zero probability"
            )));
        }
        // Renormalise: while state >= D[s] << 20, push the low 16 bits.
        // The threshold p * (1 << 20) ensures the inverse update result
        // stays below 1 << 32. The encoder pushes bits in opposite order
        // to how the decoder will read them, then we emit the stack in
        // reverse to restore decoder-order.
        //
        // Computed in u64 to avoid overflow when p == 4096
        // (4096 << 20 = 2^32, exactly out of u32 range).
        let max_state: u64 = (p as u64) << 20;
        while (state as u64) >= max_state {
            refill_stack.push((state & 0xFFFF) as u16);
            state >>= ANS_REFILL_BITS;
        }
        // Apply inverse update.
        let q = state / p;
        let r = state - q * p;
        if (r as usize) >= inv_alias[s].len() {
            return Err(Error::other(format!(
                "ANS encoder: inv_alias[{s}] index {r} out of bounds (len {})",
                inv_alias[s].len()
            )));
        }
        let alias_index = inv_alias[s][r as usize] as u32;
        // state' = (q << 12) | alias_index. Both fit in 32 bits given
        // the pre-renormalisation guard.
        state = (q << ANS_LOG_TAB_SIZE) | alias_index;
    }

    // Emit the leading u(32) state.
    // BitWriter::write_bits supports up to 32 bits.
    bw.write_bits(state, 32)?;
    // Emit refills in REVERSE of stack-push order so the decoder reads
    // them in the order it expects (i.e. encoder-emission order).
    while let Some(refill) = refill_stack.pop() {
        bw.write_bits(refill as u32, ANS_REFILL_BITS)?;
    }
    Ok(())
}

/// Convenience: build the alias table + inverse alias table + encode in
/// one call. Intended for callers that don't need to share the alias
/// table across multiple streams.
pub fn encode_symbols_with_dist(
    bw: &mut BitWriter,
    symbols: &[u16],
    d: &[u16],
    log_alphabet_size: u32,
) -> Result<()> {
    let alias = AliasTable::build(d, log_alphabet_size)?;
    let inv = build_inverse_alias(d, &alias)?;
    encode_symbols(bw, symbols, d, &inv)
}

/// Emit an ANS distribution preamble per FDIS D.3.4.
///
/// Round-3 uses the simplest encoding the decoder accepts:
///
/// * **1 non-zero symbol** → `u(1)=1, u(1)=0, U8(symbol)`.
/// * **2 non-zero symbols** → `u(1)=1, u(1)=1, U8(v1), U8(v2), u(12)=D[v1]`.
/// * **Otherwise** → fall back to the **flat** branch
///   (`u(1)=0, u(1)=1, U8(alphabet_size - 1)`) when the input is roughly
///   uniform over its alphabet, OR the general kLogCountLut path with a
///   rounded-up "power of 2" approximation of the actual distribution.
///
/// The kLogCountLut general path emits each symbol as either:
/// * `logcount = 0`  → probability 0  (no extra bits)
/// * `logcount = 1`  → probability 1  (no extra bits)
/// * `logcount = k`  (2..=12) → probability in `[2^(k-1), 2^k)` with
///   `bitcount = min(max(0, shift - ((12 - k + 1) >> 1)), k - 1)` extra
///   bits. We use `shift = 12 + 1` (the maximum) so the omit-position
///   carries the residual entries.
///
/// The omitted symbol gets `4096 - sum(others)` filled in by the
/// decoder.
pub fn write_distribution(bw: &mut BitWriter, d: &[u16], log_alphabet_size: u32) -> Result<()> {
    let table_size: usize = 1usize << log_alphabet_size;
    if d.len() != table_size {
        return Err(Error::other(format!(
            "ANS encoder: distribution length {} != 1 << log_alphabet_size {}",
            d.len(),
            table_size
        )));
    }
    let sum: u32 = d.iter().map(|&v| v as u32).sum();
    if sum != ANS_TAB_SIZE {
        return Err(Error::other(format!(
            "ANS encoder: distribution sum {sum} != 4096"
        )));
    }

    let mut nonzero_indices: Vec<usize> = Vec::new();
    for (i, &v) in d.iter().enumerate() {
        if v != 0 {
            nonzero_indices.push(i);
        }
    }
    if nonzero_indices.is_empty() {
        return Err(Error::other(
            "ANS encoder: distribution has no non-zero entries",
        ));
    }

    // Branch 1a: exactly one non-zero symbol.
    if nonzero_indices.len() == 1 {
        let idx = nonzero_indices[0];
        if d[idx] != ANS_TAB_SIZE as u16 {
            return Err(Error::other(
                "ANS encoder: single non-zero symbol must carry full 4096 mass",
            ));
        }
        bw.write_bit(1); // explicit
        bw.write_bit(0); // ns = 0 → 1 symbol
        bw.write_u8_value(idx as u32)?;
        return Ok(());
    }

    // Branch 1b: exactly two non-zero symbols.
    if nonzero_indices.len() == 2 {
        let i1 = nonzero_indices[0];
        let i2 = nonzero_indices[1];
        bw.write_bit(1); // explicit
        bw.write_bit(1); // ns = 1 → 2 symbols
        bw.write_u8_value(i1 as u32)?;
        bw.write_u8_value(i2 as u32)?;
        bw.write_bits(d[i1] as u32, 12)?;
        return Ok(());
    }

    // Branch 2: flat (uniform) — use this only if `d` actually IS the
    // uniform distribution over its first `n` entries.
    if is_flat(d) {
        let alphabet = nonzero_indices.last().copied().unwrap() + 1;
        bw.write_bit(0); // not explicit
        bw.write_bit(1); // flat
        bw.write_u8_value((alphabet - 1) as u32)?;
        return Ok(());
    }

    // Branch 3: general kLogCountLut path.
    write_distribution_general(bw, d, log_alphabet_size)
}

fn is_flat(d: &[u16]) -> bool {
    // Flat means: there's an `n` such that d[0..n] is entirely
    // `floor(4096/n)` or `floor(4096/n)+1` (with the +1 entries first),
    // and d[n..] is all zero.
    let mut n = d.len();
    while n > 0 && d[n - 1] == 0 {
        n -= 1;
    }
    if n == 0 {
        return false;
    }
    let floor_v = (ANS_TAB_SIZE / n as u32) as u16;
    let remainder = (ANS_TAB_SIZE as usize) % n;
    for (i, &v) in d.iter().enumerate().take(n) {
        let expected = if i < remainder { floor_v + 1 } else { floor_v };
        if v != expected {
            return false;
        }
    }
    true
}

/// Compute the inverse of the K_LOG_COUNT_LUT lookup. For each
/// `logcount in 0..=13` returns the LSB-first 7-bit pattern that decodes
/// to `(advance_bits, logcount)`. We pick the *smallest* such 7-bit
/// pattern (and matching `advance_bits`) so multiple round-trips reuse
/// the same encoded form.
///
/// Returns `(pattern, advance_bits)`.
fn invert_log_count_lut(logcount: u8) -> (u32, u32) {
    for h in 0..128u32 {
        if K_LOG_COUNT_LUT[h as usize][1] == logcount {
            return (h, K_LOG_COUNT_LUT[h as usize][0] as u32);
        }
    }
    // Should never happen for logcount in [0, 13].
    (0, 3)
}

/// General-path distribution encoder. Picks a sensible `shift` and
/// emits each non-zero entry's `logcount` plus the extra bits.
fn write_distribution_general(bw: &mut BitWriter, d: &[u16], log_alphabet_size: u32) -> Result<()> {
    bw.write_bit(0); // not explicit
    bw.write_bit(0); // not flat → general path

    // Find the highest non-zero index → alphabet_size = highest + 1.
    let mut alphabet_size: usize = 0;
    for (i, &v) in d.iter().enumerate() {
        if v != 0 {
            alphabet_size = i + 1;
        }
    }
    if alphabet_size < 3 {
        return Err(Error::other(
            "ANS encoder: general path requires alphabet_size >= 3 (use 1-/2-symbol path otherwise)",
        ));
    }
    let table_size: usize = 1usize << log_alphabet_size;
    if alphabet_size > table_size {
        return Err(Error::other(
            "ANS encoder: alphabet_size > table_size in general path",
        ));
    }

    // FDIS D.3.4: read `len` zeros (1-bit each) capped at 3, then read
    // `len` bits for shift = u(len) + (1 << len) - 1. We always emit
    // len = 3 → shift = u(3) + 7. To get `shift = 14` (≥ 12, so all
    // values get exactly `code - 1` extra bits per the bitcount
    // formula), we'd need u(3) = 7 → shift = 14. shift > 4096 + 1 is
    // rejected by the decoder, so we cap below the limit.
    //
    // Pick shift = 13: u(3) = 6 → shift = 6 + 7 = 13. With shift = 13
    // and code k, bitcount = min(max(0, 13 - ((12-k+1)>>1)), k-1) =
    // min(13 - ((13-k)>>1), k-1). For k=12: bitcount = min(13-0, 11)
    // = 11. For k=2: bitcount = min(13 - 5, 1) = 1 (= k-1). So with
    // shift=13 every code gets exactly k-1 extra bits, which means
    // each non-zero entry encodes its full 12-bit value as bit (k-1)
    // is the implicit leading 1 + (k-1) bits of payload.
    //
    // Wait — bitcount must equal `code - 1` for the round-trip to
    // recover the exact value. The decoder formula is:
    //   val = (1 << (code - 1)) + (extra << (code - 1 - bitcount))
    // For round-trip, we want `val` to be exactly the encoded
    // probability. This requires `extra` to carry the bits below the
    // leading 1. With bitcount = code - 1, val = (1 << (code-1)) +
    // extra; extra = val - (1 << (code-1)) ∈ [0, 1 << (code-1)) which
    // is `code - 1` bits — exactly what we need.
    //
    // Per the formula, bitcount = min(max(0, shift - ((12-code+1)>>1)), code-1).
    // For bitcount = code - 1, we need shift - ((13-code)>>1) >= code - 1,
    // i.e. shift >= code - 1 + ((13-code)>>1).
    // For code = 12: shift >= 11 + 0 = 11.
    // For code = 2:  shift >= 1 + 5 = 6.
    // So shift = 11 is the minimum that gives bitcount = code - 1 for
    // ALL values of code. Pick shift = 11: u(3) = 4 → shift = 4 + 7 = 11.
    let len: u32 = 3;
    let shift_minus_offset: u32 = 4; // u(3) = 4 → shift = 4 + 7 = 11
    let shift: u32 = shift_minus_offset + (1u32 << len) - 1;
    bw.write_bit(1); // first bit of `len` chain
    bw.write_bit(1); // second
    bw.write_bit(1); // third (len = 3 reached → break)
    bw.write_bits(shift_minus_offset, len)?;
    debug_assert_eq!(shift, 11);

    // alphabet_size: U8(alphabet_size - 3).
    bw.write_u8_value((alphabet_size - 3) as u32)?;

    // Pick the omit_pos = the FIRST index whose actual code is the
    // maximum across all entries. The decoder picks omit_pos as the
    // FIRST index with the highest logcount seen. By placing the omit
    // marker at the first-max-code position, we don't conflict with
    // later code-equal entries.
    let codes: Vec<u8> = (0..alphabet_size).map(|i| code_for(d[i] as u32)).collect();
    let max_code = *codes.iter().max().unwrap();
    let omit_pos = codes
        .iter()
        .position(|&c| c == max_code)
        .ok_or_else(|| Error::other("ANS encoder: no max-code position (BUG)"))?;

    // For each entry in [0, alphabet_size): emit logcount via the
    // K_LOG_COUNT_LUT inverse. For non-omit entries with code >= 2,
    // ALSO emit the `bitcount` extra bits.
    //
    // logcount derivation: see [`code_for`]. We never emit code 13
    // (RLE escape) — the round-3 encoder doesn't use RLE.
    //
    // Critical: at omit_pos the decoder reads ONLY the logcount (via
    // K_LOG_COUNT_LUT advance), NOT the extra bits — see modular_fdis
    // /distribution.rs. Emitting extra bits at omit_pos would desync
    // the bitstream.
    for i in 0..alphabet_size {
        let code = codes[i];
        let (pattern, advance) = invert_log_count_lut(code);
        bw.write_bits(pattern, advance)?;
        if i == omit_pos {
            // Decoder skips extra-bits decode for omit_pos and overwrites
            // d[omit_pos] = 4096 - sum(others). So we emit ZERO extra
            // bits here. The actual probability d[omit_pos] is recovered
            // by the decoder's final pass.
            continue;
        }
        if code == 13 {
            return Err(Error::other(
                "ANS encoder: code 13 (RLE escape) not used in round 3",
            ));
        }
        if code >= 2 {
            let v = d[i] as u32;
            // Extra bits: bitcount = min(max(0, shift - ((12 - code + 1) >> 1)), code - 1).
            let inner = ((12i32 - code as i32 + 1) >> 1).max(0) as u32;
            let raw = (shift as i32 - inner as i32).max(0) as u32;
            let bitcount = raw.min(code as u32 - 1);
            // val = (1 << (code-1)) + (extra << (code-1-bitcount))
            // For bitcount = code-1 (shift=11 in our config),
            //   extra = v - (1 << (code-1))
            let base = 1u32 << (code as u32 - 1);
            if v < base {
                return Err(Error::other(format!(
                    "ANS encoder: v={v} < base={base} for code={code}"
                )));
            }
            let above = v - base;
            let shift_amt = (code as u32).saturating_sub(1).saturating_sub(bitcount);
            let extra = above >> shift_amt;
            if bitcount < 32 && extra >= (1u32 << bitcount) {
                return Err(Error::other(format!(
                    "ANS encoder: extra {extra} doesn't fit in {bitcount} bits"
                )));
            }
            bw.write_bits(extra, bitcount)?;
        }
        // codes 0 and 1 emit no extra bits.
    }
    Ok(())
}

/// The K_LOG_COUNT_LUT logcount value for a probability `v` in
/// `[0, 4096]`. Returns `0` for `v==0`, `1` for `v==1`, otherwise
/// `floor(log2(v)) + 1` clamped to `[2, 12]`.
///
/// The decoder uses the formula `D[i] = (1 << (code - 1)) +
/// (extra << (code - 1 - bitcount))` for `code >= 2` (D.3.4). With
/// `shift = 11` we have `bitcount = code - 1` so `extra = v -
/// (1 << (code - 1))`, which uniquely recovers `v`.
fn code_for(v: u32) -> u8 {
    if v == 0 {
        0
    } else if v == 1 {
        1
    } else {
        let lz = v.leading_zeros();
        let bits = 32 - lz;
        bits.clamp(2, 12) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ans::distribution::read_distribution;
    use crate::ans::symbol::AnsDecoder;
    use crate::bitreader::BitReader;

    /// Round-trip a small symbol stream through encoder + decoder using
    /// the single-symbol short path.
    #[test]
    fn single_symbol_round_trip() {
        let log_alpha = 5u32;
        let table_size = 1usize << log_alpha;
        let mut d = vec![0u16; table_size];
        d[7] = ANS_TAB_SIZE as u16;
        let alias = AliasTable::build(&d, log_alpha).unwrap();
        let inv = build_inverse_alias(&d, &alias).unwrap();

        let symbols = vec![7u16; 10];
        let mut bw = BitWriter::new();
        encode_symbols(&mut bw, &symbols, &d, &inv).unwrap();
        let bytes = bw.finish();

        let mut br = BitReader::new(&bytes);
        let mut dec = AnsDecoder::new(&mut br).unwrap();
        for &expected in &symbols {
            let s = dec.decode_symbol(&mut br, &d, &alias).unwrap();
            assert_eq!(s, expected);
        }
        assert!(dec.final_state(), "final state should be ANS_FINAL_STATE");
    }

    /// Round-trip a 2-symbol skewed distribution.
    #[test]
    fn two_symbol_skewed_round_trip() {
        let log_alpha = 5u32;
        let table_size = 1usize << log_alpha;
        let mut d = vec![0u16; table_size];
        d[0] = 4000;
        d[1] = 96;
        let alias = AliasTable::build(&d, log_alpha).unwrap();
        let inv = build_inverse_alias(&d, &alias).unwrap();

        // A mostly-zero stream with occasional ones.
        let mut symbols = Vec::new();
        for i in 0..100u16 {
            symbols.push(if i % 13 == 7 { 1 } else { 0 });
        }
        let mut bw = BitWriter::new();
        encode_symbols(&mut bw, &symbols, &d, &inv).unwrap();
        let bytes = bw.finish();

        let mut br = BitReader::new(&bytes);
        let mut dec = AnsDecoder::new(&mut br).unwrap();
        for &expected in &symbols {
            let s = dec.decode_symbol(&mut br, &d, &alias).unwrap();
            assert_eq!(s, expected);
        }
        assert!(dec.final_state());
    }

    /// Round-trip a uniform distribution over 8 symbols.
    #[test]
    fn uniform_distribution_round_trip() {
        let log_alpha = 5u32;
        let table_size = 1usize << log_alpha;
        let mut d = vec![0u16; table_size];
        // Uniform over 8 symbols: each gets 4096/8 = 512.
        for slot in d.iter_mut().take(8) {
            *slot = 512;
        }
        let alias = AliasTable::build(&d, log_alpha).unwrap();
        let inv = build_inverse_alias(&d, &alias).unwrap();

        let symbols: Vec<u16> = (0..50u16).map(|i| (i % 8) as u16).collect();
        let mut bw = BitWriter::new();
        encode_symbols(&mut bw, &symbols, &d, &inv).unwrap();
        let bytes = bw.finish();

        let mut br = BitReader::new(&bytes);
        let mut dec = AnsDecoder::new(&mut br).unwrap();
        for &expected in &symbols {
            let s = dec.decode_symbol(&mut br, &d, &alias).unwrap();
            assert_eq!(s, expected);
        }
        assert!(dec.final_state());
    }

    /// Quantise a frequency histogram and verify the result sums to 4096.
    #[test]
    fn quantise_distribution_sums_to_4096() {
        let counts = vec![100u32, 50, 25, 10, 5, 1];
        let d = quantise_distribution(&counts, 5).unwrap();
        let sum: u32 = d.iter().map(|&v| v as u32).sum();
        assert_eq!(sum, 4096);
        // Each non-zero count should give a non-zero probability.
        for (i, &c) in counts.iter().enumerate() {
            if c > 0 {
                assert!(d[i] > 0, "symbol {i} (count {c}) got zero probability");
            }
        }
    }

    /// Quantise a single-symbol histogram → 4096 on that symbol.
    #[test]
    fn quantise_single_symbol_full_mass() {
        let mut counts = vec![0u32; 10];
        counts[3] = 50;
        let d = quantise_distribution(&counts, 5).unwrap();
        assert_eq!(d[3], 4096);
        for (i, &v) in d.iter().enumerate() {
            if i != 3 {
                assert_eq!(v, 0, "non-source symbol {i} should be zero");
            }
        }
    }

    /// End-to-end: write distribution preamble + encoded symbols, then
    /// read distribution + decode symbols.
    #[test]
    fn distribution_preamble_round_trip_single_symbol() {
        let log_alpha = 5u32;
        let table_size = 1usize << log_alpha;
        let mut d = vec![0u16; table_size];
        d[5] = ANS_TAB_SIZE as u16;
        let mut bw = BitWriter::new();
        write_distribution(&mut bw, &d, log_alpha).unwrap();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let d_decoded = read_distribution(&mut br, log_alpha).unwrap();
        assert_eq!(d, d_decoded);
    }

    #[test]
    fn distribution_preamble_round_trip_two_symbol() {
        let log_alpha = 5u32;
        let table_size = 1usize << log_alpha;
        let mut d = vec![0u16; table_size];
        d[2] = 1234;
        d[7] = 4096 - 1234;
        let mut bw = BitWriter::new();
        write_distribution(&mut bw, &d, log_alpha).unwrap();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let d_decoded = read_distribution(&mut br, log_alpha).unwrap();
        assert_eq!(d, d_decoded);
    }

    #[test]
    fn distribution_preamble_round_trip_flat() {
        let log_alpha = 5u32;
        let table_size = 1usize << log_alpha;
        let mut d = vec![0u16; table_size];
        // 4096 / 8 = 512 exactly → flat over 8 symbols.
        for slot in d.iter_mut().take(8) {
            *slot = 512;
        }
        let mut bw = BitWriter::new();
        write_distribution(&mut bw, &d, log_alpha).unwrap();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let d_decoded = read_distribution(&mut br, log_alpha).unwrap();
        assert_eq!(d, d_decoded);
    }

    /// End-to-end: distribution preamble + encoded symbols, both round-tripped.
    #[test]
    fn distribution_plus_symbols_round_trip() {
        let log_alpha = 5u32;
        let table_size = 1usize << log_alpha;
        // A 3-symbol distribution forcing the general path.
        let mut counts = vec![0u32; table_size];
        counts[0] = 80;
        counts[1] = 15;
        counts[2] = 5;
        let d = quantise_distribution(&counts, log_alpha).unwrap();
        let alias = AliasTable::build(&d, log_alpha).unwrap();
        let inv = build_inverse_alias(&d, &alias).unwrap();

        // Symbols sampled from the histogram.
        let mut symbols = Vec::with_capacity(100);
        for i in 0..100u16 {
            symbols.push(match i % 20 {
                0..=15 => 0u16,
                16..=18 => 1u16,
                _ => 2u16,
            });
        }

        let mut bw = BitWriter::new();
        write_distribution(&mut bw, &d, log_alpha).unwrap();
        encode_symbols(&mut bw, &symbols, &d, &inv).unwrap();
        let bytes = bw.finish();

        let mut br = BitReader::new(&bytes);
        let d_decoded = read_distribution(&mut br, log_alpha).unwrap();
        assert_eq!(d, d_decoded);
        let alias2 = AliasTable::build(&d_decoded, log_alpha).unwrap();
        let mut dec = AnsDecoder::new(&mut br).unwrap();
        for &expected in &symbols {
            let s = dec.decode_symbol(&mut br, &d_decoded, &alias2).unwrap();
            assert_eq!(s, expected);
        }
        assert!(dec.final_state());
    }
}
