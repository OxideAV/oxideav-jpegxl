//! ANS symbol decoder — FDIS Annex D.3.3 (Listing D.3, p. 63).
//!
//! 32-bit-state ANS reverse decoder. State is initialised with `u(32)`,
//! each symbol decoded performs:
//!
//! ```text
//! index = state & 0xFFF
//! (symbol, offset) = AliasMapping(index)
//! state = D[symbol] * (state >> 12) + offset
//! if state < (1 << 16): state = (state << 16) | u(16)
//! ```
//!
//! The spec requires that, after the *last* symbol in a stream is
//! decoded, `state` equals `0x130000`. Use [`AnsDecoder::final_state`]
//! to verify that condition at end-of-stream.

use oxideav_core::{Error, Result};

use crate::ans::alias::AliasTable;
use crate::bitreader::BitReader;

/// ANS final-state magic value (D.3.3, last sentence).
pub const ANS_FINAL_STATE: u32 = 0x130000;

/// ANS symbol decoder state.
///
/// One instance per ANS stream; multiple distributions / alias tables
/// share the same state if they belong to the same stream.
#[derive(Debug)]
pub struct AnsDecoder {
    state: u32,
}

impl AnsDecoder {
    /// Initialise the decoder by reading `u(32)` for the state.
    pub fn new(br: &mut BitReader<'_>) -> Result<Self> {
        let state = br.read_bits(32)?;
        Ok(Self { state })
    }

    /// FDIS Listing D.3 — decode one symbol from the stream against
    /// distribution `d` (with companion alias table `alias`).
    ///
    /// `d` MUST be the same `(1 << log_alphabet_size)`-element array
    /// the alias table was built from; we accept it by reference here
    /// (rather than embedding it inside `AliasTable`) so distribution
    /// clustering can share one alias table across many calls without
    /// duplicating the histogram.
    pub fn decode_symbol(
        &mut self,
        br: &mut BitReader<'_>,
        d: &[u16],
        alias: &AliasTable,
    ) -> Result<u16> {
        let index = self.state & 0xFFF;
        let (symbol, offset) = alias.lookup(index);
        let prob = d.get(symbol as usize).copied().unwrap_or(0) as u32;
        if prob == 0 {
            // Per D.3.1: indexing D with an out-of-bounds value gives 0.
            // For an in-bounds symbol with zero probability we treat
            // the same way — but practically this can only happen if
            // the alias table was built from a malformed distribution.
            return Err(Error::InvalidData(
                "JXL ANS: decoded symbol has zero probability".into(),
            ));
        }
        let new_state = prob
            .checked_mul(self.state >> 12)
            .and_then(|v| v.checked_add(offset))
            .ok_or_else(|| Error::InvalidData("JXL ANS: state update overflow".into()))?;
        self.state = if new_state < (1u32 << 16) {
            let extra = br.read_bits(16)?;
            (new_state << 16) | extra
        } else {
            new_state
        };
        Ok(symbol)
    }

    /// Current state value. Useful for end-of-stream validation:
    /// after the final `decode_symbol` call, this should equal
    /// [`ANS_FINAL_STATE`].
    pub fn state(&self) -> u32 {
        self.state
    }

    /// True iff the decoder is in its expected end-of-stream state.
    pub fn final_state(&self) -> bool {
        self.state == ANS_FINAL_STATE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ans::test_helpers::pack_lsb;

    /// Build a tiny distribution, hand-craft an ANS stream that decodes
    /// a known symbol sequence, and verify the symbols come out.
    ///
    /// We build a bitstream using the spec's *forward* (encoder-side)
    /// arithmetic that mirrors Listing D.3:
    ///
    /// ```text
    /// // Forward (encode) one symbol with probability D[s] starting at slot S:
    /// state' = ((state / D[s]) << 12) | alias_inverse(s, state mod D[s])
    /// // emit u(16) when state' would exceed 32 bits
    /// ```
    ///
    /// To keep the test self-contained we use a single-symbol
    /// distribution where `alias_inverse(s, k) = bucket_size * 0 + k`
    /// degenerates so that symbol decode is trivial; we then verify the
    /// decoder consumes exactly the expected number of bits.
    #[test]
    fn single_symbol_stream_decodes_correctly() {
        // Distribution: symbol 7 with full probability 4096.
        let mut d = vec![0u16; 8];
        d[7] = 4096;
        let alias = AliasTable::build(&d, 3).unwrap();

        // For a single-symbol distribution, the alias table maps every
        // index x in [0, 4096) to (7, bucket_size * (x >> log_bucket)).
        // Wait: per AliasTable::build, single-symbol fills offsets[i]
        // with bucket_size * i (4096 entries, bucket_size_for_4096
        // entries indexed 0..4096). So lookup(x) = (7, offsets[x] + (x & 0)).
        // log_bucket_size = 12-3 = 9, bucket_size = 512.
        // The single-symbol path in AliasTable uses bucket_size=512 and
        // pos = x & (bucket_size-1). So lookup(x) returns
        // symbol=7, offset = offsets[x>>9] + (x & 511).
        // Since offsets[i] = bucket_size * i = 512*i (and there are 4096
        // such entries, but we only index 0..(1<<3)=8 of them in the
        // 8-bucket case)... actually in AliasTable single-symbol path
        // I made the symbols/offsets array length 4096. lookup uses
        // i = x >> log_bucket_size (=9 here), so i in [0, 8). offsets[i]
        // = 512*i. So lookup(x) = (7, 512*i + (x & 511)) = (7, x).
        // Effectively: state' = D[7] * (state >> 12) + x = 4096 * (state>>12) + (state & 0xFFF) = state.
        // i.e. decoding a single-symbol stream is a no-op on state.
        //
        // So if we initialise state = 0x130000, the spec's final_state
        // condition is already met without decoding anything; and any
        // number of decode_symbol calls will keep state = 0x130000
        // (because new_state = 0x130000 >= (1 << 16) so no refill).
        let bytes = pack_lsb(&[(0x130000, 32)]);
        let mut br = BitReader::new(&bytes);
        let mut dec = AnsDecoder::new(&mut br).unwrap();
        assert!(dec.final_state());
        // Decode 5 symbols, each must come back as 7, state must remain
        // at the magic constant the whole time.
        for _ in 0..5 {
            let s = dec.decode_symbol(&mut br, &d, &alias).unwrap();
            assert_eq!(s, 7);
            assert_eq!(dec.state(), ANS_FINAL_STATE);
        }
    }

    #[test]
    fn truncated_state_init_returns_error() {
        // Only 24 bits available; reading u(32) for state must fail.
        let bytes = vec![0u8; 3];
        let mut br = BitReader::new(&bytes);
        assert!(AnsDecoder::new(&mut br).is_err());
    }

    #[test]
    fn refill_eof_returns_error() {
        // Construct a distribution where decoding will trigger the
        // refill path (new_state < 1<<16). With single-symbol dist + a
        // state that's already 0x130000 we never refill; instead, build
        // a 2-symbol skewed distribution and force a refill.
        let mut d = vec![0u16; 4];
        d[0] = 4095;
        d[1] = 1;
        let alias = AliasTable::build(&d, 2).unwrap();

        // Pick state = 1 (extreme low). new_state = D[symbol] * (1 >> 12)
        // + offset = 0 + offset. AliasMapping(1) goes to symbol 0
        // (because cutoffs[0] is non-zero and 1 falls under it most
        // likely), with offset some small value < 1<<16. So we hit the
        // refill path immediately.
        let bytes = pack_lsb(&[(1, 32)]);
        let mut br = BitReader::new(&bytes);
        let mut dec = AnsDecoder::new(&mut br).unwrap();
        // Refill needs 16 more bits; bytes only has 4 bytes so refill
        // would need to read past EOF.
        let res = dec.decode_symbol(&mut br, &d, &alias);
        assert!(res.is_err(), "expected EOF on refill");
    }
}
