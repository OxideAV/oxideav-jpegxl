//! Hybrid integer coding — FDIS Annex D.3.6 (Listing D.6, p. 66).
//!
//! Implements `DecodeHybridVarLenUint(ctx)` plus the verbatim 120 × 2
//! `kSpecialDistances` lookup table. The decoder maintains a 1 MiB
//! sliding window for LZ77-style copy operations.

use oxideav_core::{Error, Result};

use super::hybrid_config::HybridUintConfig;
use crate::bitreader::BitReader;

/// LZ77 sliding-window size from FDIS D.3.6 — `window[(idx) & 0xFFFFF]`.
/// 1 MiB of `u32` slots.
pub const WINDOW_SIZE: usize = 1 << 20;

/// LZ77 max distance — also `1 << 20`, per the `min(distance, ..., 1 << 20)`
/// clamp in Listing D.6.
pub const MAX_DISTANCE: u32 = 1 << 20;

/// FDIS D.3.6 — `kSpecialDistances[120][2]` (p. 66, verbatim from
/// Listing D.6 in the published PDF).
///
/// Each entry is `(dx, dy)` — a 2-D offset selected when
/// `dist_multiplier != 0` and the decoded `distance` field is below 120.
///
/// **Audit trail:** transcribed from FDIS 18181-1:2021 Listing D.6
/// page 66 of the FDIS PDF (also reachable in `/tmp/fdis.txt` lines
/// 4275-4290 if the local extracted text is on disk).
pub const K_SPECIAL_DISTANCES: [[i32; 2]; 120] = [
    [0, 1],
    [1, 0],
    [1, 1],
    [-1, 1],
    [0, 2],
    [2, 0],
    [1, 2],
    [-1, 2],
    [2, 1],
    [-2, 1],
    [2, 2],
    [-2, 2],
    [0, 3],
    [3, 0],
    [1, 3],
    [-1, 3],
    [3, 1],
    [-3, 1],
    [2, 3],
    [-2, 3],
    [3, 2],
    [-3, 2],
    [0, 4],
    [4, 0],
    [1, 4],
    [-1, 4],
    [4, 1],
    [-4, 1],
    [3, 3],
    [-3, 3],
    [2, 4],
    [-2, 4],
    [4, 2],
    [-4, 2],
    [0, 5],
    [3, 4],
    [-3, 4],
    [4, 3],
    [-4, 3],
    [5, 0],
    [1, 5],
    [-1, 5],
    [5, 1],
    [-5, 1],
    [2, 5],
    [-2, 5],
    [5, 2],
    [-5, 2],
    [4, 4],
    [-4, 4],
    [3, 5],
    [-3, 5],
    [5, 3],
    [-5, 3],
    [0, 6],
    [6, 0],
    [1, 6],
    [-1, 6],
    [6, 1],
    [-6, 1],
    [2, 6],
    [-2, 6],
    [6, 2],
    [-6, 2],
    [4, 5],
    [-4, 5],
    [5, 4],
    [-5, 4],
    [3, 6],
    [-3, 6],
    [6, 3],
    [-6, 3],
    [0, 7],
    [7, 0],
    [1, 7],
    [-1, 7],
    [5, 5],
    [-5, 5],
    [7, 1],
    [-7, 1],
    [4, 6],
    [-4, 6],
    [6, 4],
    [-6, 4],
    [2, 7],
    [-2, 7],
    [7, 2],
    [-7, 2],
    [3, 7],
    [-3, 7],
    [7, 3],
    [-7, 3],
    [5, 6],
    [-5, 6],
    [6, 5],
    [-6, 5],
    [8, 0],
    [4, 7],
    [-4, 7],
    [7, 4],
    [-7, 4],
    [8, 1],
    [8, 2],
    [6, 6],
    [-6, 6],
    [8, 3],
    [5, 7],
    [-5, 7],
    [7, 5],
    [-7, 5],
    [8, 4],
    [6, 7],
    [-6, 7],
    [7, 6],
    [-7, 6],
    [8, 5],
    [7, 7],
    [-7, 7],
    [8, 6],
    [8, 7],
];

/// LZ77 settings (`LZ77Params`) per FDIS D.3.1 Table D.1.
#[derive(Debug, Clone, Copy)]
pub struct Lz77Params {
    pub enabled: bool,
    pub min_symbol: u32,
    pub min_length: u32,
}

impl Default for Lz77Params {
    fn default() -> Self {
        Self {
            enabled: false,
            min_symbol: 224,
            min_length: 3,
        }
    }
}

/// Streaming state for `DecodeHybridVarLenUint` (D.3.6).
///
/// The actual symbol-decode callback is supplied by the caller so this
/// module does not have to know whether the underlying entropy stream
/// is ANS or prefix-coded — both are valid per D.3.1.
///
/// `dist_multiplier` is 0 unless the call site (frame decode path)
/// specifies otherwise; the LZ77 distance branch using
/// [`K_SPECIAL_DISTANCES`] is gated on a non-zero `dist_multiplier`.
#[derive(Debug)]
pub struct HybridUintState {
    /// 1 MiB sliding window for LZ77 copies. Indexed via `idx & 0xFFFFF`.
    window: Vec<u32>,
    /// Number of integers decoded so far.
    num_decoded: u64,
    /// LZ77 copy state.
    num_to_copy: u32,
    copy_pos: u64,
    /// LZ77 parameters (read from the bitstream up-front).
    pub lz77: Lz77Params,
    /// HybridUintConfig for the LZ77 length symbols (only used when
    /// LZ77 is enabled).
    pub lz_len_conf: HybridUintConfig,
    /// Context id used to decode the LZ77 distance — typically the last
    /// non-LZ77 context's id; tracked across calls.
    last_ctx: u32,
}

impl HybridUintState {
    /// Allocate a fresh state. **Window allocation is fixed at
    /// `WINDOW_SIZE` regardless of input length** — this is the spec's
    /// invariant, but it caps memory at 4 MiB (`u32` × 1 MiB) per
    /// stream. The caller must arrange to share or reuse states across
    /// sub-streams that belong to the same frame.
    pub fn new(lz77: Lz77Params, lz_len_conf: HybridUintConfig) -> Self {
        Self {
            window: vec![0u32; WINDOW_SIZE],
            num_decoded: 0,
            num_to_copy: 0,
            copy_pos: 0,
            lz77,
            lz_len_conf,
            last_ctx: 0,
        }
    }

    /// Push a literal value through the window without involving the
    /// LZ77 path. Used by callers that drive non-hybrid decode (e.g.
    /// the *clustering map* sub-stream of D.3.5) and want to exercise
    /// the LZ77 buffer's invariants.
    pub fn push_literal(&mut self, value: u32) {
        self.window[(self.num_decoded as usize) & (WINDOW_SIZE - 1)] = value;
        self.num_decoded += 1;
    }

    /// Number of integers decoded through this state so far.
    pub fn num_decoded(&self) -> u64 {
        self.num_decoded
    }

    /// FDIS Listing D.6 — `DecodeHybridVarLenUint(ctx)`.
    ///
    /// * `br` is the underlying bit reader. Extra-bit reads `u(n)` for
    ///   tokens above `split` come from this bit reader.
    /// * `read_token(br, ctx)` reads one entropy-coded token for the
    ///   given context. The closure gets the bit reader so it can drive
    ///   either the ANS state machine or the prefix-code state machine
    ///   (per D.3.1's `use_prefix_code` switch).
    /// * `ctx_lz` is the "LZ77 distance" context id (`num_dists` in spec).
    /// * `configs(ctx)` returns the `HybridUintConfig` for the named
    ///   context.
    /// * `dist_multiplier` is 0 unless the caller specifies otherwise.
    pub fn decode<R, C>(
        &mut self,
        br: &mut BitReader<'_>,
        ctx: u32,
        ctx_lz: u32,
        dist_multiplier: u32,
        mut read_token: R,
        configs: C,
    ) -> Result<u32>
    where
        R: FnMut(&mut BitReader<'_>, u32) -> Result<u32>,
        C: Fn(u32) -> HybridUintConfig,
    {
        // Iterative rewrite of the spec's tail recursion to avoid any
        // possibility of unbounded stack growth on a malicious LZ77
        // stream that immediately re-enters itself.
        loop {
            if self.num_to_copy > 0 {
                let r = self.window[(self.copy_pos as usize) & (WINDOW_SIZE - 1)];
                self.copy_pos = self.copy_pos.wrapping_add(1);
                self.num_to_copy -= 1;
                self.window[(self.num_decoded as usize) & (WINDOW_SIZE - 1)] = r;
                self.num_decoded = self.num_decoded.wrapping_add(1);
                return Ok(r);
            }

            let token = read_token(br, ctx)?;
            self.last_ctx = ctx;

            if self.lz77.enabled && token >= self.lz77.min_symbol {
                let len_token = token - self.lz77.min_symbol;
                let length_payload = self.lz_len_conf.read_uint(br, len_token)?;
                self.num_to_copy = length_payload
                    .checked_add(self.lz77.min_length)
                    .ok_or_else(|| Error::InvalidData("JXL LZ77: copy length overflow".into()))?;
                let dist_token = read_token(br, ctx_lz)?;
                let cfg = configs(ctx_lz);
                let mut distance = cfg.read_uint(br, dist_token)?;
                if dist_multiplier == 0 {
                    distance = distance.checked_add(1).ok_or_else(|| {
                        Error::InvalidData("JXL LZ77: distance+1 overflow".into())
                    })?;
                } else if distance < 120 {
                    let entry = K_SPECIAL_DISTANCES[distance as usize];
                    // Spec PDF re-indexes kSpecialDistances after the
                    // first lookup mutates `distance` — clear PDF
                    // artefact. We capture both fields of the same row
                    // eagerly, which is the only consistent reading.
                    let dx = entry[0];
                    let dy = entry[1];
                    let signed: i64 = dx as i64 + (dist_multiplier as i64) * (dy as i64);
                    if signed < 1 {
                        return Err(Error::InvalidData(
                            "JXL LZ77: special-distance result < 1".into(),
                        ));
                    }
                    distance = signed.min(MAX_DISTANCE as i64) as u32;
                } else {
                    distance = distance
                        .checked_sub(119)
                        .ok_or_else(|| Error::InvalidData("JXL LZ77: distance underflow".into()))?;
                }
                let max_d = self.num_decoded.min(MAX_DISTANCE as u64) as u32;
                if max_d == 0 {
                    return Err(Error::InvalidData(
                        "JXL LZ77: copy before any literal decoded".into(),
                    ));
                }
                let distance = distance.min(max_d);
                self.copy_pos = self.num_decoded.wrapping_sub(distance as u64);
                continue;
            }

            // Regular literal.
            let cfg = configs(ctx);
            let r = cfg.read_uint(br, token)?;
            self.window[(self.num_decoded as usize) & (WINDOW_SIZE - 1)] = r;
            self.num_decoded = self.num_decoded.wrapping_add(1);
            return Ok(r);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_special_distances_size() {
        assert_eq!(K_SPECIAL_DISTANCES.len(), 120);
    }

    #[test]
    fn lz77_disabled_passes_through_token_below_split() {
        let lz77 = Lz77Params::default();
        let lz_len_conf = HybridUintConfig {
            split_exponent: 4,
            msb_in_token: 0,
            lsb_in_token: 0,
            split: 16,
        };
        let mut state = HybridUintState::new(lz77, lz_len_conf);
        let cfg = HybridUintConfig {
            split_exponent: 8,
            msb_in_token: 0,
            lsb_in_token: 0,
            split: 256,
        };
        // Token = 5, below split = 256 → returns 5 without LZ77.
        let bytes: [u8; 0] = [];
        let mut br = BitReader::new(&bytes);
        let mut tokens = vec![5u32];
        let value = state
            .decode(
                &mut br,
                0,
                1,
                0,
                |_br, _ctx| Ok(tokens.remove(0)),
                |_ctx| cfg,
            )
            .unwrap();
        assert_eq!(value, 5);
        assert_eq!(state.num_decoded(), 1);
    }

    #[test]
    fn lz77_copy_repeats_previous_literal() {
        // First decode literal 42 (token = 42 < split, no LZ77 trigger).
        // Then feed an LZ77 token = min_symbol = 224, length-payload
        // token = 0 → copy length = 0 + min_length = 3, distance token
        // = 0 → distance = 0 + 1 = 1 (dist_multiplier == 0).
        let lz77 = Lz77Params {
            enabled: true,
            min_symbol: 224,
            min_length: 3,
        };
        let lz_len_conf = HybridUintConfig {
            split_exponent: 4,
            msb_in_token: 0,
            lsb_in_token: 0,
            split: 16,
        };
        let mut state = HybridUintState::new(lz77, lz_len_conf);
        let cfg = HybridUintConfig {
            split_exponent: 8,
            msb_in_token: 0,
            lsb_in_token: 0,
            split: 256,
        };
        let bytes: [u8; 0] = [];
        let mut br = BitReader::new(&bytes);

        // Literal first.
        let mut tokens = vec![42u32];
        let v0 = state
            .decode(
                &mut br,
                0,
                1,
                0,
                |_br, _ctx| Ok(tokens.remove(0)),
                |_ctx| cfg,
            )
            .unwrap();
        assert_eq!(v0, 42);

        // LZ77 trigger + length-payload token + distance token.
        let mut tokens = vec![224u32, 0u32, 0u32];
        let v1 = state
            .decode(
                &mut br,
                0,
                1,
                0,
                |_br, _ctx| Ok(tokens.remove(0)),
                |_ctx| cfg,
            )
            .unwrap();
        assert_eq!(v1, 42, "first LZ77 copy must repeat the previous literal");
        // Two more copies left.
        let v2 = state
            .decode(
                &mut br,
                0,
                1,
                0,
                |_br, _ctx| panic!("should not call read_token during copy"),
                |_ctx| cfg,
            )
            .unwrap();
        assert_eq!(v2, 42);
        let v3 = state
            .decode(
                &mut br,
                0,
                1,
                0,
                |_br, _ctx| panic!("should not call read_token during copy"),
                |_ctx| cfg,
            )
            .unwrap();
        assert_eq!(v3, 42);
        assert_eq!(state.num_decoded(), 4);
    }

    #[test]
    fn lz77_copy_before_any_literal_rejected() {
        let lz77 = Lz77Params {
            enabled: true,
            min_symbol: 0,
            min_length: 3,
        };
        let lz_len_conf = HybridUintConfig {
            split_exponent: 4,
            msb_in_token: 0,
            lsb_in_token: 0,
            split: 16,
        };
        let mut state = HybridUintState::new(lz77, lz_len_conf);
        let cfg = HybridUintConfig {
            split_exponent: 8,
            msb_in_token: 0,
            lsb_in_token: 0,
            split: 256,
        };
        let bytes: [u8; 0] = [];
        let mut br = BitReader::new(&bytes);
        // First call tries to copy with num_decoded = 0 → must fail.
        let mut tokens = vec![0u32, 0u32, 0u32];
        let err = state.decode(
            &mut br,
            0,
            1,
            0,
            |_br, _ctx| Ok(tokens.remove(0)),
            |_ctx| cfg,
        );
        assert!(err.is_err());
    }
}
