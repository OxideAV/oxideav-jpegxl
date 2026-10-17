//! Entropy *encoder* — the inverse of Annex C with LZ77 off: the C.2.1
//! prelude (LZ77 flag, C.2.2 simple clustering, hybrid-integer
//! configurations, then either per-cluster symbol counts + RFC 7932
//! §3.4 / §3.5 prefix codes, or `5 + u(2)` alphabet bits + C.2.5 ANS
//! distributions) and the C.3.3 hybrid-integer symbol stream (C.3.2
//! rANS: one 32-bit state per stream, 12-bit probabilities, 16-bit
//! renormalisation, final state `0x130000`).
//!
//! Two passes: [`SymbolStats`] collects every `(context, value)` the
//! caller will emit, [`EntropyCodes::build`] /
//! [`EntropyCodes::build_with`] turn the statistics into codes, and the
//! caller then writes the prelude followed — possibly after intervening
//! bundle fields, as the GlobalModular layout requires — by the symbols
//! in the same order through [`EntropyCodes::write_symbols`]. An ANS
//! stream is built backwards (the state the decoder ends on is the
//! encoder's starting point), so the symbols must be presented twice.

use crate::ans::alias::AliasTable;
use crate::ans::distribution::K_LOG_COUNT_LUT;
use crate::ans::prefix::{K_CODE_LENGTH_CODE_ORDER, MAX_CODE_LENGTH};
use crate::bitwriter::BitWriter;
use crate::error::{Error, Result};

/// The C.3.2 state every ANS stream ends on.
pub const ANS_FINAL_STATE: u32 = 0x130000;

/// Which symbol coding a stream uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coding {
    /// RFC 7932 prefix codes — whole bits per symbol, no state.
    Prefix,
    /// C.2.5 / C.3.2 ANS — fractional bits per symbol, one 32-bit
    /// state (and its initialiser) per stream.
    Ans,
}

/// `log_alphabet_size` of every prefix-coded stream (C.2.1).
pub const PREFIX_LOG_ALPHABET_SIZE: u32 = 15;

/// Largest cluster count the simple clustering (`u(2)` bits per entry)
/// can name.
pub const MAX_SIMPLE_CLUSTERS: usize = 8;

/// `ceil(log2(x))` for `x >= 1` (`0` for `x == 1`).
fn ceil_log2(x: u32) -> u32 {
    if x <= 1 {
        0
    } else {
        32 - (x - 1).leading_zeros()
    }
}

/// Hybrid-integer configuration (C.2.3) chosen by the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridConfig {
    pub split_exponent: u32,
    pub msb_in_token: u32,
    pub lsb_in_token: u32,
}

impl HybridConfig {
    /// `split = 16`, two MSBs in the token: the shape the Modular
    /// residual streams use.
    pub const RESIDUALS: Self = Self {
        split_exponent: 4,
        msb_in_token: 2,
        lsb_in_token: 0,
    };
    /// `split = 256`: every byte-valued symbol is its own token (ICC).
    pub const BYTES: Self = Self {
        split_exponent: 8,
        msb_in_token: 0,
        lsb_in_token: 0,
    };

    fn split(&self) -> u32 {
        1 << self.split_exponent
    }

    /// Write the configuration (C.2.3) for a stream whose alphabet is
    /// `1 << log_alphabet_size`.
    pub fn write(&self, w: &mut BitWriter, log_alphabet_size: u32) {
        w.write_bits(ceil_log2(log_alphabet_size + 1), self.split_exponent);
        if self.split_exponent != log_alphabet_size {
            w.write_bits(ceil_log2(self.split_exponent + 1), self.msb_in_token);
            w.write_bits(
                ceil_log2(self.split_exponent - self.msb_in_token + 1),
                self.lsb_in_token,
            );
        }
    }

    /// Token, extra-bit count and extra bits for `value` — the inverse
    /// of `ReadUint` (C.3.3).
    pub fn tokenize(&self, value: u32) -> (u32, u32, u32) {
        let split = self.split();
        if value < split {
            return (value, 0, 0);
        }
        let msb = self.msb_in_token;
        let lsb = self.lsb_in_token;
        let l = 31 - value.leading_zeros();
        let n = l - msb - lsb;
        let msb_bits = (value >> (n + lsb)) & ((1 << msb) - 1);
        let lsb_bits = value & ((1 << lsb) - 1);
        let token =
            split + (((l - self.split_exponent) << (msb + lsb)) | (msb_bits << lsb) | lsb_bits);
        let extra = (value >> lsb) & (((1u64 << n) - 1) as u32);
        (token, n, extra)
    }
}

/// Canonical codes for `lengths` (RFC 7932 §3.2 — the assignment
/// `PrefixCode::from_lengths` mirrors): `codes[sym]` is meaningful only
/// where `lengths[sym] != 0`.
fn canonical_codes(lengths: &[u8]) -> Vec<u32> {
    let max_len = lengths.iter().copied().max().unwrap_or(0) as usize;
    let mut bl_count = vec![0u32; max_len + 1];
    for &l in lengths {
        if l > 0 {
            bl_count[l as usize] += 1;
        }
    }
    let mut next_code = vec![0u32; max_len + 1];
    let mut code = 0u32;
    for bits in 1..=max_len {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }
    let mut codes = vec![0u32; lengths.len()];
    for (sym, &l) in lengths.iter().enumerate() {
        if l > 0 {
            codes[sym] = next_code[l as usize];
            next_code[l as usize] += 1;
        }
    }
    codes
}

fn bit_reverse(mut x: u32, n: u32) -> u32 {
    let mut out = 0;
    for _ in 0..n {
        out = (out << 1) | (x & 1);
        x >>= 1;
    }
    out
}

/// Huffman code lengths for `counts`, no code longer than `limit`
/// bits (counts are halved and retried until the tree fits — the
/// flattening converges because equal counts give `ceil(log2(n))`-bit
/// codes). Symbols with a zero count get length 0; a single used
/// symbol gets length 1 (callers that want a bit-free single-symbol
/// code use the simple prefix code instead).
pub fn limited_huffman_lengths(counts: &[u64], limit: u8) -> Vec<u8> {
    let n = counts.len();
    let mut lengths = vec![0u8; n];
    let used: Vec<usize> = (0..n).filter(|&i| counts[i] > 0).collect();
    if used.is_empty() {
        return lengths;
    }
    if used.len() == 1 {
        lengths[used[0]] = 1;
        return lengths;
    }
    let mut weights: Vec<u64> = counts.to_vec();
    loop {
        // (weight, node) heap-less Huffman: sort each round (alphabets
        // are tiny — at most a few hundred symbols).
        let mut parent = vec![usize::MAX; 2 * n];
        let mut nodes: Vec<(u64, usize)> = used.iter().map(|&i| (weights[i], i)).collect();
        let mut next = n;
        while nodes.len() > 1 {
            nodes.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
            let (wa, a) = nodes.pop().unwrap();
            let (wb, b) = nodes.pop().unwrap();
            parent[a] = next;
            parent[b] = next;
            nodes.push((wa + wb, next));
            next += 1;
        }
        let mut max = 0u8;
        for &i in &used {
            let mut depth = 0u8;
            let mut p = parent[i];
            while p != usize::MAX {
                depth += 1;
                p = parent[p];
            }
            lengths[i] = depth;
            max = max.max(depth);
        }
        if max <= limit {
            return lengths;
        }
        for &i in &used {
            weights[i] = weights[i].div_ceil(2);
        }
    }
}

/// A prefix code ready to be written and used.
#[derive(Debug, Clone)]
pub struct PrefixCodeWriter {
    /// Alphabet size signalled to the reader (`count`).
    alphabet_size: u32,
    lengths: Vec<u8>,
    codes: Vec<u32>,
    /// Symbols of a simple code in transmission order (`None` for a
    /// complex code).
    simple: Option<Vec<u32>>,
    /// Tree-select bit of a four-symbol simple code.
    tree_select: bool,
}

impl PrefixCodeWriter {
    /// Build the code for an alphabet of `counts.len()` symbols.
    pub fn build(counts: &[u64]) -> Self {
        let alphabet_size = counts.len() as u32;
        let mut used: Vec<u32> = (0..alphabet_size)
            .filter(|&s| counts[s as usize] > 0)
            .collect();
        // Most frequent first: the simple code gives the short codes
        // to the first-transmitted symbols.
        used.sort_by(|&a, &b| counts[b as usize].cmp(&counts[a as usize]).then(a.cmp(&b)));
        let mut lengths = vec![0u8; counts.len()];
        if alphabet_size == 1 || used.is_empty() {
            // No bits are read at all for a one-symbol alphabet.
            return Self {
                alphabet_size,
                lengths,
                codes: vec![0; counts.len()],
                simple: None,
                tree_select: false,
            };
        }
        if used.len() <= 4 {
            let mut tree_select = false;
            match used.len() {
                1 => {}
                2 => {
                    lengths[used[0] as usize] = 1;
                    lengths[used[1] as usize] = 1;
                }
                3 => {
                    lengths[used[0] as usize] = 1;
                    lengths[used[1] as usize] = 2;
                    lengths[used[2] as usize] = 2;
                }
                _ => {
                    let c: Vec<u64> = used.iter().map(|&s| counts[s as usize]).collect();
                    let flat = 2 * (c[0] + c[1] + c[2] + c[3]);
                    let skew = c[0] + 2 * c[1] + 3 * (c[2] + c[3]);
                    if skew < flat {
                        tree_select = true;
                        lengths[used[0] as usize] = 1;
                        lengths[used[1] as usize] = 2;
                        lengths[used[2] as usize] = 3;
                        lengths[used[3] as usize] = 3;
                    } else {
                        for &s in &used {
                            lengths[s as usize] = 2;
                        }
                    }
                }
            }
            let codes = canonical_codes(&lengths);
            return Self {
                alphabet_size,
                lengths,
                codes,
                simple: Some(used),
                tree_select,
            };
        }
        let lengths = limited_huffman_lengths(counts, MAX_CODE_LENGTH as u8);
        let codes = canonical_codes(&lengths);
        Self {
            alphabet_size,
            lengths,
            codes,
            simple: None,
            tree_select: false,
        }
    }

    /// Write the histogram (C.2.4: RFC 7932 §3.4 / §3.5).
    pub fn write(&self, w: &mut BitWriter) {
        if self.alphabet_size <= 1 {
            return;
        }
        if let Some(symbols) = &self.simple {
            let alphabet_bits = ceil_log2(self.alphabet_size);
            w.write_bits(2, 1);
            w.write_bits(2, symbols.len() as u32 - 1);
            for &s in symbols {
                w.write_bits(alphabet_bits, s);
            }
            if symbols.len() == 4 {
                w.write_bool(self.tree_select);
            }
            return;
        }
        // Complex code: run-length the code lengths into the 18-symbol
        // code-length alphabet.
        let last_nonzero = self
            .lengths
            .iter()
            .rposition(|&l| l != 0)
            .expect("complex code has symbols");
        let mut rle: Vec<(u8, u32, u32)> = Vec::new(); // (symbol, extra bits count, extra)
        let mut i = 0usize;
        while i <= last_nonzero {
            let l = self.lengths[i];
            let mut run = 1usize;
            while i + run <= last_nonzero && self.lengths[i + run] == l {
                run += 1;
            }
            if l == 0 {
                let mut left = run;
                while left >= 3 {
                    let k = left.min(10);
                    rle.push((17, 3, (k - 3) as u32));
                    left -= k;
                    if left >= 3 {
                        // A literal breaks the "repeated 17 modifies the
                        // count" chain so the next 17 starts afresh.
                        rle.push((0, 0, 0));
                        left -= 1;
                    }
                }
                for _ in 0..left {
                    rle.push((0, 0, 0));
                }
            } else {
                // The literal establishes the length to repeat.
                rle.push((l, 0, 0));
                let mut left = run - 1;
                while left >= 3 {
                    let k = left.min(6);
                    rle.push((16, 2, (k - 3) as u32));
                    left -= k;
                    if left >= 3 {
                        rle.push((l, 0, 0));
                        left -= 1;
                    }
                }
                for _ in 0..left {
                    rle.push((l, 0, 0));
                }
            }
            i += run;
        }
        let mut cl_counts = [0u64; 18];
        for &(s, _, _) in &rle {
            cl_counts[s as usize] += 1;
        }
        let mut cl_lengths = limited_huffman_lengths(&cl_counts, 5);
        if cl_counts.iter().filter(|&&c| c > 0).count() == 1 {
            // Two one-bit codes keep the code-length code complete.
            let only = cl_counts.iter().position(|&c| c > 0).unwrap();
            let other = if only == 0 { 1 } else { 0 };
            cl_lengths[only] = 1;
            cl_lengths[other] = 1;
        }
        let cl_codes = canonical_codes(&cl_lengths);
        // HSKIP = 0.
        w.write_bits(2, 0);
        let mut kraft = 0u32;
        let mut nonzero = 0u32;
        for &sym in K_CODE_LENGTH_CODE_ORDER.iter() {
            let l = cl_lengths[sym] as u32;
            let (code, bits) = match l {
                0 => (0b00, 2),
                1 => (0b0111, 4),
                2 => (0b011, 3),
                3 => (0b10, 2),
                4 => (0b01, 2),
                5 => (0b1111, 4),
                _ => unreachable!("code-length code lengths are at most 5"),
            };
            w.write_bits(bits, code);
            if l > 0 {
                nonzero += 1;
                kraft += 32 >> l;
                if nonzero >= 2 && kraft >= 32 {
                    break;
                }
            }
        }
        for &(s, nbits, extra) in &rle {
            let l = cl_lengths[s as usize] as u32;
            w.write_bits(l, bit_reverse(cl_codes[s as usize], l));
            w.write_bits(nbits, extra);
        }
    }

    /// Emit `symbol` with this code.
    pub fn write_symbol(&self, w: &mut BitWriter, symbol: u32) {
        let l = self.lengths[symbol as usize] as u32;
        debug_assert!(
            self.alphabet_size <= 1 || l > 0 || self.simple.as_ref().is_some_and(|s| s.len() == 1),
            "symbol {symbol} has no code"
        );
        w.write_bits(l, bit_reverse(self.codes[symbol as usize], l));
    }

    /// The count field (`alphabet_size`) as C.2.1 reads it for the
    /// prefix path.
    fn write_count(&self, w: &mut BitWriter) {
        let count = self.alphabet_size.max(1);
        if count == 1 {
            w.write_bool(false);
        } else {
            w.write_bool(true);
            let n = 31 - (count - 1).leading_zeros();
            w.write_bits(4, n);
            w.write_bits(n, count - 1 - (1 << n));
        }
    }
}

/// One ANS cluster: the 12-bit distribution and, per symbol, the slot
/// each offset lives in under the decoder's alias mapping (C.2.6).
#[derive(Debug, Clone)]
struct AnsCluster {
    /// `1 << log_alphabet_size` entries summing to 4096.
    dist: Vec<u16>,
    /// `slot_of[sym][offset]` for `offset < dist[sym]`.
    slot_of: Vec<Vec<u16>>,
}

/// Scale `counts` to a 4096-sum distribution with every used symbol
/// at least 1 (the largest symbol absorbs the rounding).
fn normalize_counts(counts: &[u64]) -> Vec<u16> {
    let total: u64 = counts.iter().sum();
    let mut d = vec![0u16; counts.len()];
    if total == 0 {
        return d;
    }
    let mut sum = 0u32;
    let mut largest = 0usize;
    for (i, &c) in counts.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let v = ((c as u128 * 4096 + (total as u128) / 2) / total as u128) as u32;
        let v = v.clamp(1, 4095);
        d[i] = v as u16;
        sum += v;
        if counts[i] > counts[largest] || d[largest] == 0 {
            largest = i;
        }
    }
    // Settle on exactly 4096: move the difference onto the largest
    // symbol, then if that would drive it below 1 take from the others.
    let target = 4096i64;
    let mut diff = target - i64::from(sum);
    let l = i64::from(d[largest]);
    if l + diff >= 1 {
        d[largest] = (l + diff) as u16;
        diff = 0;
    }
    let mut i = 0usize;
    while diff != 0 {
        if d[i] > 1 && diff < 0 {
            d[i] -= 1;
            diff += 1;
        } else if diff > 0 && d[i] > 0 {
            d[i] += 1;
            diff -= 1;
        }
        i = (i + 1) % d.len();
    }
    d
}

/// `u8()` of C.2.5: `0` as one bit, else `1`, `u(3)` exponent, `u(n)`.
fn write_u8_value(w: &mut BitWriter, v: u32) {
    if v == 0 {
        w.write_bool(false);
        return;
    }
    w.write_bool(true);
    let n = 31 - v.leading_zeros();
    w.write_bits(3, n);
    w.write_bits(n, v - (1 << n));
}

/// The log-count prefix code of C.2.5 (`K_LOG_COUNT_LUT` inverted):
/// `(bits, length)` for each value 0..=13.
fn logcount_codes() -> [(u32, u32); 14] {
    let mut out = [(0u32, 0u32); 14];
    let mut seen = [false; 14];
    for (h, &[len, value]) in K_LOG_COUNT_LUT.iter().enumerate() {
        let v = value as usize;
        if v < 14 && !seen[v] {
            seen[v] = true;
            out[v] = ((h as u32) & ((1 << len) - 1), u32::from(len));
        }
    }
    out
}

/// Write one C.2.5 distribution for `dist` (4096-sum, length
/// `1 << log_alphabet_size`): the one- and two-symbol short forms, else
/// the log-count form at full precision (`shift = 13`).
fn write_distribution(w: &mut BitWriter, dist: &[u16]) {
    let used: Vec<usize> = (0..dist.len()).filter(|&i| dist[i] > 0).collect();
    match used.len() {
        0 | 1 => {
            w.write_bool(true);
            w.write_bool(false);
            write_u8_value(w, used.first().copied().unwrap_or(0) as u32);
            return;
        }
        2 => {
            w.write_bool(true);
            w.write_bool(true);
            write_u8_value(w, used[0] as u32);
            write_u8_value(w, used[1] as u32);
            w.write_bits(12, u32::from(dist[used[0]]));
            return;
        }
        _ => {}
    }
    w.write_bool(false);
    w.write_bool(false); // not flat
                         // shift = 13: three 1-bits then u(3) = 13 - 7.
    w.write_bits(3, 0b111);
    w.write_bits(3, 6);
    let alphabet_size = (used[used.len() - 1] + 1).max(3);
    write_u8_value(w, (alphabet_size - 3) as u32);
    let codes = logcount_codes();
    let logcount = |v: u16| -> u32 {
        if v == 0 {
            0
        } else {
            32 - u32::from(v).leading_zeros()
        }
    };
    // The omitted symbol: first occurrence of the largest log count.
    let mut omit = 0usize;
    for i in 0..alphabet_size {
        if logcount(dist[i]) > logcount(dist[omit]) {
            omit = i;
        }
    }
    for &d in &dist[..alphabet_size] {
        let (bits, len) = codes[logcount(d) as usize];
        w.write_bits(len, bits);
    }
    for (i, &d) in dist[..alphabet_size].iter().enumerate() {
        let v = u32::from(d);
        let c = logcount(d);
        if i == omit || c <= 1 {
            continue;
        }
        let inner = (13 - c) >> 1;
        let bitcount = (13u32.saturating_sub(inner)).min(c - 1);
        debug_assert_eq!(bitcount, c - 1, "shift 13 keeps full precision");
        w.write_bits(bitcount, (v - (1 << (c - 1))) >> (c - 1 - bitcount));
    }
}

impl AnsCluster {
    fn build(counts: &[u64], log_alphabet_size: u32) -> Result<Self> {
        let table = 1usize << log_alphabet_size;
        let mut dist = normalize_counts(counts);
        dist.resize(table, 0);
        if dist.iter().all(|&d| d == 0) {
            dist[0] = 4096;
        }
        let alias = AliasTable::build(&dist, log_alphabet_size)?;
        let mut slot_of: Vec<Vec<u16>> = dist
            .iter()
            .map(|&d| vec![u16::MAX; usize::from(d)])
            .collect();
        for x in 0..4096u32 {
            let (sym, off) = alias.lookup(x);
            let slots = slot_of.get_mut(usize::from(sym)).ok_or_else(|| {
                Error::invalid("JXL writer: alias symbol beyond the distribution")
            })?;
            let cell = slots.get_mut(off as usize).ok_or_else(|| {
                Error::invalid("JXL writer: alias offset beyond the symbol's probability")
            })?;
            *cell = x as u16;
        }
        if slot_of.iter().flatten().any(|&s| s == u16::MAX) {
            return Err(Error::invalid(
                "JXL writer: alias mapping does not cover every offset",
            ));
        }
        Ok(Self { dist, slot_of })
    }
}

/// Symbol statistics of one entropy-coded stream: `num_dist` contexts
/// mapped onto clusters, one hybrid configuration per cluster.
#[derive(Debug, Clone)]
pub struct SymbolStats {
    cluster_map: Vec<u32>,
    configs: Vec<HybridConfig>,
    counts: Vec<Vec<u64>>,
}

impl SymbolStats {
    /// `cluster_map[ctx]` names the cluster of context `ctx`; the
    /// clusters are `0..configs.len()` and every one must be named.
    pub fn new(cluster_map: Vec<u32>, configs: Vec<HybridConfig>) -> Result<Self> {
        let n_clusters = configs.len();
        if n_clusters == 0 || n_clusters > MAX_SIMPLE_CLUSTERS {
            return Err(Error::invalid(format!(
                "JXL writer: {n_clusters} clusters (1..={MAX_SIMPLE_CLUSTERS} supported)"
            )));
        }
        for &c in &cluster_map {
            if c as usize >= n_clusters {
                return Err(Error::invalid("JXL writer: cluster index out of range"));
            }
        }
        for c in 0..n_clusters as u32 {
            if !cluster_map.contains(&c) {
                return Err(Error::invalid(format!("JXL writer: cluster {c} unused")));
            }
        }
        Ok(Self {
            cluster_map,
            configs,
            counts: vec![Vec::new(); n_clusters],
        })
    }

    /// One context per cluster, all with `config`.
    pub fn identity(num_dist: usize, config: HybridConfig) -> Result<Self> {
        Self::new((0..num_dist as u32).collect(), vec![config; num_dist])
    }

    /// Every context in cluster 0.
    pub fn single(num_dist: usize, config: HybridConfig) -> Result<Self> {
        Self::new(vec![0; num_dist], vec![config])
    }

    /// Record one symbol.
    pub fn add(&mut self, ctx: u32, value: u32) {
        let cluster = self.cluster_map[ctx as usize] as usize;
        let (token, _, _) = self.configs[cluster].tokenize(value);
        let counts = &mut self.counts[cluster];
        if counts.len() <= token as usize {
            counts.resize(token as usize + 1, 0);
        }
        counts[token as usize] += 1;
    }
}

/// The codes of one stream, ready to write the C.2.1 prelude and the
/// symbols.
#[derive(Debug, Clone)]
pub struct EntropyCodes {
    cluster_map: Vec<u32>,
    configs: Vec<HybridConfig>,
    codes: Vec<PrefixCodeWriter>,
    /// `Some((log_alphabet_size, clusters))` for an ANS stream.
    ans: Option<(u32, Vec<AnsCluster>)>,
}

impl EntropyCodes {
    /// Build per-cluster prefix codes from the statistics.
    pub fn build(stats: &SymbolStats) -> Self {
        let codes = stats
            .counts
            .iter()
            .map(|c| {
                if c.is_empty() {
                    PrefixCodeWriter::build(&[1])
                } else {
                    PrefixCodeWriter::build(c)
                }
            })
            .collect();
        Self {
            cluster_map: stats.cluster_map.clone(),
            configs: stats.configs.clone(),
            codes,
            ans: None,
        }
    }

    /// Build with the given coding. ANS needs every token below 256
    /// (`log_alphabet_size <= 8`); larger tokens fall back to prefix
    /// codes.
    pub fn build_with(stats: &SymbolStats, coding: Coding) -> Result<Self> {
        let mut codes = Self::build(stats);
        if coding == Coding::Prefix {
            return Ok(codes);
        }
        let max_tokens = stats
            .counts
            .iter()
            .map(|c| c.len())
            .max()
            .unwrap_or(0)
            .max(1);
        if max_tokens > 256 {
            return Ok(codes);
        }
        let log_alphabet_size = (5..=8u32)
            .find(|&l| (1usize << l) >= max_tokens)
            .expect("max_tokens <= 256");
        let clusters = stats
            .counts
            .iter()
            .map(|c| AnsCluster::build(c, log_alphabet_size))
            .collect::<Result<Vec<_>>>()?;
        codes.ans = Some((log_alphabet_size, clusters));
        Ok(codes)
    }

    /// The coding this stream uses.
    pub fn coding(&self) -> Coding {
        if self.ans.is_some() {
            Coding::Ans
        } else {
            Coding::Prefix
        }
    }

    /// Write the C.2.1 prelude: LZ77 off, the clustering, the coding
    /// flag, the configurations, then the prefix counts + codes or the
    /// ANS alphabet size + distributions.
    pub fn write_prelude(&self, w: &mut BitWriter) {
        w.write_bool(false); // lz77.enabled
        let num_dist = self.cluster_map.len();
        if num_dist > 1 {
            let n_clusters = self.configs.len() as u32;
            w.write_bool(true); // is_simple
            let nbits = ceil_log2(n_clusters);
            w.write_bits(2, nbits);
            for &c in &self.cluster_map {
                w.write_bits(nbits, c);
            }
        }
        match &self.ans {
            None => {
                w.write_bool(true); // use_prefix_code
                for cfg in &self.configs {
                    cfg.write(w, PREFIX_LOG_ALPHABET_SIZE);
                }
                for code in &self.codes {
                    code.write_count(w);
                }
                for code in &self.codes {
                    code.write(w);
                }
            }
            Some((log_alphabet_size, clusters)) => {
                w.write_bool(false); // use_prefix_code
                w.write_bits(2, log_alphabet_size - 5);
                for cfg in &self.configs {
                    cfg.write(w, *log_alphabet_size);
                }
                for c in clusters {
                    write_distribution(w, &c.dist);
                }
            }
        }
    }

    /// Write one hybrid integer in context `ctx` (prefix streams only;
    /// an ANS stream goes through [`write_symbols`](Self::write_symbols)).
    pub fn write(&self, w: &mut BitWriter, ctx: u32, value: u32) {
        debug_assert!(self.ans.is_none(), "ANS streams are written as a whole");
        let cluster = self.cluster_map[ctx as usize] as usize;
        let (token, nbits, extra) = self.configs[cluster].tokenize(value);
        self.codes[cluster].write_symbol(w, token);
        w.write_bits(nbits, extra);
    }

    /// Write a whole symbol stream `(context, value)` in order. For an
    /// ANS stream this emits the `u(32)` state initialiser first (the
    /// reader takes it right before the first symbol), then per symbol
    /// the optional 16-bit renormalisation word followed by the hybrid
    /// integer's extra bits; the stream ends on [`ANS_FINAL_STATE`].
    pub fn write_symbols<I>(&self, w: &mut BitWriter, symbols: I)
    where
        I: Iterator<Item = (u32, u32)> + DoubleEndedIterator + Clone,
    {
        let Some((_, clusters)) = &self.ans else {
            for (ctx, v) in symbols {
                self.write(w, ctx, v);
            }
            return;
        };
        // Backward pass: from the final state to the initial one,
        // recording which symbols carry a renormalisation word.
        let mut state: u32 = ANS_FINAL_STATE;
        let mut words: Vec<Option<u16>> = Vec::new();
        for (ctx, v) in symbols.clone().rev() {
            let cluster = self.cluster_map[ctx as usize] as usize;
            let (token, _, _) = self.configs[cluster].tokenize(v);
            let c = &clusters[cluster];
            let f = u32::from(c.dist[token as usize]);
            debug_assert!(f > 0, "token {token} has probability 0");
            let mut word = None;
            // `f << 20` reaches 2^32 for a one-symbol cluster: compare
            // in 64 bits (such a symbol never renormalises).
            if u64::from(state) >= (u64::from(f) << 20) {
                word = Some((state & 0xFFFF) as u16);
                state >>= 16;
            }
            let slot = u32::from(c.slot_of[token as usize][(state % f) as usize]);
            state = ((state / f) << 12) | slot;
            words.push(word);
        }
        words.reverse();
        w.write_bits(32, state);
        for ((ctx, v), word) in symbols.zip(words) {
            let cluster = self.cluster_map[ctx as usize] as usize;
            let (_, nbits, extra) = self.configs[cluster].tokenize(v);
            if let Some(word) = word {
                w.write_bits(16, u32::from(word));
            }
            w.write_bits(nbits, extra);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ans::hybrid::HybridUintState;
    use crate::bitreader::BitReader;
    use crate::modular_fdis::{decode_uint_in_with_dist_pub, EntropyStream};

    fn round_trip(stats_fn: impl Fn() -> SymbolStats, symbols: &[(u32, u32)]) {
        for coding in [Coding::Prefix, Coding::Ans] {
            let mut stats = stats_fn();
            for &(c, v) in symbols {
                stats.add(c, v);
            }
            let codes = EntropyCodes::build_with(&stats, coding).unwrap();
            let mut w = BitWriter::new();
            codes.write_prelude(&mut w);
            // Something between the prelude and the symbols, as the
            // Modular layout has.
            w.write_bits(5, 0b10110);
            codes.write_symbols(&mut w, symbols.iter().copied());
            w.write_bits(3, 0b101);
            let bytes = w.into_bytes();
            let num_dist = stats.cluster_map.len();
            let mut br = BitReader::new(&bytes);
            let mut es = EntropyStream::read(&mut br, num_dist).expect("prelude");
            assert_eq!(
                es.use_prefix_code,
                coding == Coding::Prefix || codes.coding() == Coding::Prefix
            );
            assert_eq!(br.read_bits(5).unwrap(), 0b10110);
            es.read_ans_state_init(&mut br).unwrap();
            let mut hy = HybridUintState::new(es.lz77, es.lz_len_conf);
            for &(c, v) in symbols {
                let got = decode_uint_in_with_dist_pub(&mut hy, &mut es, &mut br, c, 0).unwrap();
                assert_eq!(got, v, "{coding:?} ctx {c}");
            }
            if let Some(ans) = &es.ans_state {
                assert!(ans.final_state(), "{coding:?}: stream ends on 0x130000");
            }
            assert_eq!(
                br.read_bits(3).unwrap(),
                0b101,
                "{coding:?}: bit position after the stream"
            );
        }
    }

    #[test]
    fn tokenize_inverts_read_uint() {
        for cfg in [HybridConfig::RESIDUALS, HybridConfig::BYTES] {
            for v in [
                0u32,
                1,
                15,
                16,
                17,
                31,
                32,
                255,
                256,
                1000,
                65535,
                1 << 20,
                u32::MAX,
            ] {
                let (token, n, extra) = cfg.tokenize(v);
                let mut w = BitWriter::new();
                w.write_bits(n, extra);
                let bytes = w.into_bytes();
                let mut br = BitReader::new(&bytes);
                let rc = crate::ans::hybrid_config::HybridUintConfig {
                    split_exponent: cfg.split_exponent,
                    msb_in_token: cfg.msb_in_token,
                    lsb_in_token: cfg.lsb_in_token,
                    split: 1 << cfg.split_exponent,
                };
                assert_eq!(rc.read_uint(&mut br, token).unwrap(), v, "{cfg:?} {v}");
            }
        }
    }

    #[test]
    fn single_context_streams_round_trip() {
        // One symbol only (no histogram bits at all).
        round_trip(
            || SymbolStats::single(1, HybridConfig::RESIDUALS).unwrap(),
            &[(0, 3); 10],
        );
        // Two, three, four symbols (simple codes, both tree shapes).
        round_trip(
            || SymbolStats::single(1, HybridConfig::RESIDUALS).unwrap(),
            &[(0, 0), (0, 1), (0, 0), (0, 0)],
        );
        round_trip(
            || SymbolStats::single(1, HybridConfig::RESIDUALS).unwrap(),
            &[(0, 5), (0, 1), (0, 5), (0, 9), (0, 5)],
        );
        round_trip(
            || SymbolStats::single(1, HybridConfig::RESIDUALS).unwrap(),
            &[
                (0, 5),
                (0, 1),
                (0, 5),
                (0, 9),
                (0, 5),
                (0, 2),
                (0, 5),
                (0, 5),
            ],
        );
        round_trip(
            || SymbolStats::single(1, HybridConfig::RESIDUALS).unwrap(),
            &[(0, 5), (0, 1), (0, 9), (0, 2)],
        );
        // Complex code with big values (extra bits) and zero runs.
        let mut syms = Vec::new();
        for i in 0..2000u32 {
            syms.push((0, (i * 7919) % 300));
            syms.push((0, 1 << (i % 20)));
        }
        round_trip(
            || SymbolStats::single(1, HybridConfig::RESIDUALS).unwrap(),
            &syms,
        );
    }

    #[test]
    fn clustered_streams_round_trip() {
        let mut syms = Vec::new();
        for i in 0..500u32 {
            syms.push((i % 3, i % 17));
            syms.push((2, 1000 + i));
        }
        round_trip(
            || SymbolStats::identity(3, HybridConfig::RESIDUALS).unwrap(),
            &syms,
        );
        round_trip(
            || SymbolStats::single(6, HybridConfig::RESIDUALS).unwrap(),
            &[(0, 1), (1, 7), (5, 300), (2, 0), (4, 12), (3, 2)],
        );
        // A context that is never used still gets a (bit-free) code.
        round_trip(
            || SymbolStats::identity(2, HybridConfig::BYTES).unwrap(),
            &[(0, 200), (0, 201), (0, 200)],
        );
    }

    #[test]
    fn flat_bytes_use_a_complex_code_of_equal_lengths() {
        let syms: Vec<(u32, u32)> = (0..256u32).map(|v| (0, v)).collect();
        round_trip(
            || SymbolStats::single(1, HybridConfig::BYTES).unwrap(),
            &syms,
        );
    }

    #[test]
    fn ans_distributions_are_normalised_and_skewed_streams_cost_under_a_bit() {
        let d = normalize_counts(&[1, 0, 1_000_000, 3, 0, 7]);
        assert_eq!(d.iter().map(|&x| u32::from(x)).sum::<u32>(), 4096);
        assert!(d[0] >= 1 && d[3] >= 1 && d[5] >= 1 && d[1] == 0 && d[4] == 0);
        assert_eq!(normalize_counts(&[0, 5]), vec![0, 4096]);
        // 4000 symbols, one rare: ANS spends well under one bit each.
        let mut syms = vec![(0u32, 0u32); 4000];
        syms[1234] = (0, 3);
        let mut stats = SymbolStats::single(1, HybridConfig::RESIDUALS).unwrap();
        for &(c, v) in &syms {
            stats.add(c, v);
        }
        let codes = EntropyCodes::build_with(&stats, Coding::Ans).unwrap();
        let mut w = BitWriter::new();
        codes.write_prelude(&mut w);
        codes.write_symbols(&mut w, syms.iter().copied());
        assert!(
            w.byte_len() < 60,
            "{} bytes for 4000 near-constant symbols",
            w.byte_len()
        );
        round_trip(
            || SymbolStats::single(1, HybridConfig::RESIDUALS).unwrap(),
            &syms,
        );
        // Three symbols, then a full 256-token alphabet (log 8).
        round_trip(
            || SymbolStats::single(1, HybridConfig::BYTES).unwrap(),
            &[(0, 9), (0, 9), (0, 200), (0, 9), (0, 42)],
        );
    }

    #[test]
    fn a_constant_cluster_shares_the_state_with_a_varied_one() {
        // Context 0 always 0 (a 4096-probability symbol), context 1 varied:
        // the one-symbol cluster must never renormalise.
        let mut syms = Vec::new();
        for i in 0..3000u32 {
            syms.push((0, 0));
            syms.push((1, (i * 7919) % 97));
        }
        round_trip(
            || SymbolStats::identity(2, HybridConfig::RESIDUALS).unwrap(),
            &syms,
        );
        round_trip(
            || SymbolStats::identity(3, HybridConfig::RESIDUALS).unwrap(),
            &[(2, 5); 300],
        );
    }

    #[test]
    fn huffman_lengths_respect_the_limit() {
        let counts: Vec<u64> = (0..40u64).map(|i| 1u64 << i).collect();
        let l = limited_huffman_lengths(&counts, 15);
        assert!(l.iter().all(|&x| x <= 15 && x > 0));
        let kraft: u64 = l.iter().map(|&x| 1u64 << (15 - x)).sum();
        assert_eq!(kraft, 1 << 15, "limited code stays complete");
        let l5 = limited_huffman_lengths(&counts[..18], 5);
        assert!(l5.iter().all(|&x| x <= 5));
    }
}
