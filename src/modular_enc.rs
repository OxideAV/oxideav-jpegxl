//! Modular sub-bitstream **encoder** core (Annex H, written side).
//!
//! * [`scan_channel`] walks one channel in raster order and hands every
//!   sample's H.3 neighbourhood, H.4.1 property vector and H.5
//!   self-correcting prediction to a callback — the exact quantities the
//!   decoder derives ([`crate::modular_fdis`]), computed on rolling row
//!   state.
//! * [`learn_tree`] learns an MA tree (H.4.2) from a sample of the
//!   picture: every node takes the property / threshold split and the
//!   per-side predictors that minimise the estimated coded size, as long
//!   as the gain pays for the extra context.
//! * [`encode_section`] turns a section into `(context, value)` symbols
//!   under a tree, closed-loop when a leaf multiplier quantises.
//!
//! Every decision here (sampling, split search, stopping rule) is this
//! crate's own design; the format only fixes how a decoder evaluates
//! the result.

use crate::bitwriter::pack_signed;

/// Predictors of Table H.3.
pub(crate) const NUM_PREDICTORS: usize = 14;
/// Properties 0..=15 of Table H.4.
const BASE_PROPS: usize = 16;
/// Previous channels whose four properties each are considered.
const MAX_PREV: usize = 3;
/// Length of the property vector handed out by [`scan_channel`].
pub(crate) const MAX_PROPS: usize = BASE_PROPS + 4 * MAX_PREV;

/// One channel of a Modular image (or of a group rectangle of it).
#[derive(Debug, Clone)]
pub(crate) struct Chan {
    pub w: usize,
    pub h: usize,
    pub hshift: i32,
    pub vshift: i32,
    pub data: Vec<i32>,
}

impl Chan {
    pub(crate) fn new(w: usize, h: usize, data: Vec<i32>) -> Self {
        debug_assert_eq!(data.len(), w * h);
        Self {
            w,
            h,
            hshift: 0,
            vshift: 0,
            data,
        }
    }

    /// The `rw × rh` rectangle at `(x0, y0)` as a channel of its own.
    pub(crate) fn rect(&self, x0: usize, y0: usize, rw: usize, rh: usize) -> Self {
        let mut data = Vec::with_capacity(rw * rh);
        for y in 0..rh {
            let o = (y0 + y) * self.w + x0;
            data.extend_from_slice(&self.data[o..o + rw]);
        }
        Self {
            w: rw,
            h: rh,
            hshift: self.hshift,
            vshift: self.vshift,
            data,
        }
    }
}

/// One independently decoded sub-bitstream: its channels and the
/// stream index of property 1.
#[derive(Debug, Clone)]
pub(crate) struct Section {
    pub stream_index: i32,
    pub chans: Vec<Chan>,
}

impl Section {
    pub(crate) fn num_samples(&self) -> usize {
        self.chans.iter().map(|c| c.w * c.h).sum()
    }
}

/// What the callback of [`scan_channel`] sees for one sample.
pub(crate) struct Sample<'a> {
    pub props: &'a [i32; MAX_PROPS],
    /// Properties that exist for this channel (`16 + 4 ×` matching
    /// previous channels, capped at [`MAX_PROPS`]).
    pub nprops: usize,
    /// `W, N, NW, NE, NN, NEE, WW` with the H.3 edge rules.
    pub nb: [i32; 7],
    /// The self-correcting prediction (0 when the scan runs without it).
    pub wp: i32,
    /// The sample to code.
    pub value: i32,
}

/// Prediction of Table H.3 for `predictor` from a sample's
/// neighbourhood.
#[inline]
pub(crate) fn predict(predictor: u32, nb: &[i32; 7], wp: i32) -> i32 {
    let [w, n, nw, ne, nn, nee, ww] = *nb;
    match predictor {
        0 => 0,
        1 => w,
        2 => n,
        3 => ((i64::from(w) + i64::from(n)) / 2) as i32,
        4 => {
            if (i64::from(n) - i64::from(nw)).abs() < (i64::from(w) - i64::from(nw)).abs() {
                w
            } else {
                n
            }
        }
        5 => w.wrapping_add(n).wrapping_sub(nw).clamp(w.min(n), w.max(n)),
        6 => wp,
        7 => ne,
        8 => nw,
        9 => ww,
        10 => ((i64::from(w) + i64::from(nw)) / 2) as i32,
        11 => ((i64::from(nw) + i64::from(n)) / 2) as i32,
        12 => ((i64::from(n) + i64::from(ne)) / 2) as i32,
        _ => {
            let s = 6 * i64::from(n) - 2 * i64::from(nn)
                + 7 * i64::from(w)
                + i64::from(ww)
                + i64::from(nee)
                + 3 * i64::from(ne)
                + 8;
            (s / 16) as i32
        }
    }
}

/// `(1 << 24) Idiv d` for `d` in `1..=64` (index 0 unused).
const DIV24: [u32; 65] = {
    let mut t = [0u32; 65];
    let mut d = 1;
    while d <= 64 {
        t[d] = (1u32 << 24) / d as u32;
        d += 1;
    }
    t
};

/// Default `WPHeader` parameters (H.5.1).
const WP_P1: i64 = 16;
const WP_P2: i64 = 10;
const WP_P3: [i64; 5] = [7, 7, 7, 0, 0];
const WP_W: [u64; 4] = [13, 12, 12, 12];

#[inline]
fn error2weight(err_sum: u32, maxweight: u64) -> u64 {
    let bits = 32 - (err_sum.wrapping_add(1)).leading_zeros();
    let shift = bits.saturating_sub(1).saturating_sub(5);
    let denom = ((err_sum >> shift) as usize + 1).min(64);
    4 + ((maxweight * u64::from(DIV24[denom])) >> shift)
}

/// Rolling state of the self-correcting predictor for one channel.
struct WpRows {
    te_prev: Vec<i32>,
    te_cur: Vec<i32>,
    se_prev: [Vec<i32>; 4],
    se_cur: [Vec<i32>; 4],
}

impl WpRows {
    fn new(w: usize) -> Self {
        Self {
            te_prev: vec![0; w],
            te_cur: vec![0; w],
            se_prev: std::array::from_fn(|_| vec![0; w]),
            se_cur: std::array::from_fn(|_| vec![0; w]),
        }
    }

    fn next_row(&mut self) {
        std::mem::swap(&mut self.te_prev, &mut self.te_cur);
        for k in 0..4 {
            std::mem::swap(&mut self.se_prev[k], &mut self.se_cur[k]);
        }
    }

    /// H.5.2: `(prediction, sub-predictions, max_error)` in the
    /// eight-times domain for the sample at `(x, y)`.
    #[inline]
    fn predict(&self, nb: &[i32; 7], x: usize, y: usize, w: usize) -> (i32, [i32; 4], i32) {
        let [wv, n, nw, ne, nn, _nee, _ww] = *nb;
        let n8 = n.wrapping_shl(3);
        let w8 = wv.wrapping_shl(3);
        let nw8 = nw.wrapping_shl(3);
        let ne8 = ne.wrapping_shl(3);
        let nn8 = nn.wrapping_shl(3);
        let te_w = if x > 0 { self.te_cur[x - 1] } else { 0 };
        let te_n = if y > 0 { self.te_prev[x] } else { 0 };
        let te_nw = if x > 0 && y > 0 {
            self.te_prev[x - 1]
        } else {
            te_n
        };
        let te_ne = if x + 1 < w && y > 0 {
            self.te_prev[x + 1]
        } else {
            te_n
        };
        let p0 = w8.wrapping_add(ne8).wrapping_sub(n8);
        let p1 = n8.wrapping_sub(
            (((i64::from(te_w) + i64::from(te_n) + i64::from(te_ne)) * WP_P1) >> 5) as i32,
        );
        let p2 = w8.wrapping_sub(
            (((i64::from(te_w) + i64::from(te_n) + i64::from(te_nw)) * WP_P2) >> 5) as i32,
        );
        let p3 = n8.wrapping_sub(
            ((i64::from(te_nw) * WP_P3[0]
                + i64::from(te_n) * WP_P3[1]
                + i64::from(te_ne) * WP_P3[2]
                + (i64::from(nn8) - i64::from(n8)) * WP_P3[3]
                + (i64::from(nw8) - i64::from(w8)) * WP_P3[4])
                >> 5) as i32,
        );
        let preds = [p0, p1, p2, p3];
        let mut weights = [0u64; 4];
        for k in 0..4 {
            let n_se = if y > 0 { self.se_prev[k][x] } else { 0 };
            let w_se = if x > 0 { self.se_cur[k][x - 1] } else { 0 };
            let nw_se = if x > 0 && y > 0 {
                self.se_prev[k][x - 1]
            } else {
                n_se
            };
            let ww_se = if x > 1 { self.se_cur[k][x - 2] } else { 0 };
            let ne_se = if x + 1 < w && y > 0 {
                self.se_prev[k][x + 1]
            } else {
                n_se
            };
            let mut err_sum = (i64::from(n_se)
                + i64::from(w_se)
                + i64::from(nw_se)
                + i64::from(ww_se)
                + i64::from(ne_se)) as u64
                & 0xFFFF_FFFF;
            if x + 1 == w {
                err_sum = (err_sum + w_se as u64) & 0xFFFF_FFFF;
            }
            weights[k] = error2weight(err_sum as u32, WP_W[k]);
        }
        let sum_pre = weights[0] + weights[1] + weights[2] + weights[3];
        let log_weight = 64 - sum_pre.leading_zeros();
        let sh = log_weight.saturating_sub(5);
        let shifted = [
            weights[0] >> sh,
            weights[1] >> sh,
            weights[2] >> sh,
            weights[3] >> sh,
        ];
        let sum = shifted[0] + shifted[1] + shifted[2] + shifted[3];
        let mut s: i64 = (sum >> 1) as i64 - 1;
        for (p, sh) in preds.iter().zip(&shifted) {
            s += i64::from(*p) * *sh as i64;
        }
        let recip = if (1..=64).contains(&sum) {
            i64::from(DIV24[sum as usize])
        } else {
            (1i64 << 24) / (sum as i64).max(1)
        };
        let mut prediction = ((s * recip) >> 24) as i32;
        if (te_n ^ te_w) | (te_n ^ te_nw) <= 0 {
            let lo = w8.min(n8).min(ne8);
            let hi = w8.max(n8).max(ne8);
            prediction = prediction.clamp(lo, hi);
        }
        let mut max_error = te_w;
        if te_n.unsigned_abs() > max_error.unsigned_abs() {
            max_error = te_n;
        }
        if te_nw.unsigned_abs() > max_error.unsigned_abs() {
            max_error = te_nw;
        }
        if te_ne.unsigned_abs() > max_error.unsigned_abs() {
            max_error = te_ne;
        }
        (prediction, preds, max_error)
    }

    #[inline]
    fn store(&mut self, x: usize, pred8: i32, preds: &[i32; 4], v: i32) {
        let v8 = v.wrapping_shl(3);
        self.te_cur[x] = pred8.wrapping_sub(v8);
        for (row, p) in self.se_cur.iter_mut().zip(preds) {
            row[x] = ((p.wrapping_sub(v8).unsigned_abs().wrapping_add(3)) >> 3) as i32;
        }
    }
}

/// Walk channel `i` of `chans` in raster order. For every sample the
/// callback receives the decoder-side view (neighbours, properties,
/// self-correcting prediction when `use_wp`) and returns the value the
/// decoder will hold at that position — the sample itself when coding
/// losslessly — which is written back before the next sample.
pub(crate) fn scan_channel<F>(
    chans: &mut [Chan],
    i: usize,
    stream_index: i32,
    use_wp: bool,
    mut f: F,
) where
    F: FnMut(&Sample<'_>) -> i32,
{
    let (prev, rest) = chans.split_at_mut(i);
    let cur = &mut rest[0];
    let (w, h) = (cur.w, cur.h);
    if w == 0 || h == 0 {
        return;
    }
    let refs: Vec<&Chan> = prev
        .iter()
        .rev()
        .filter(|p| p.w == w && p.h == h && p.hshift == cur.hshift && p.vshift == cur.vshift)
        .take(MAX_PREV)
        .collect();
    let nprops = BASE_PROPS + 4 * refs.len();
    let mut wp_rows = use_wp.then(|| WpRows::new(w));
    let mut props = [0i32; MAX_PROPS];
    props[0] = i as i32;
    props[1] = stream_index;
    let d = &mut cur.data;
    for y in 0..h {
        let row = y * w;
        props[2] = y as i32;
        for x in 0..w {
            let wv = if x > 0 {
                d[row + x - 1]
            } else if y > 0 {
                d[row - w + x]
            } else {
                0
            };
            let n = if y > 0 { d[row - w + x] } else { wv };
            let nw = if x > 0 && y > 0 {
                d[row - w + x - 1]
            } else {
                wv
            };
            let ne = if x + 1 < w && y > 0 {
                d[row - w + x + 1]
            } else {
                n
            };
            let nn = if y > 1 { d[row - 2 * w + x] } else { n };
            let nee = if x + 2 < w && y > 0 {
                d[row - w + x + 2]
            } else {
                ne
            };
            let ww = if x > 1 { d[row + x - 2] } else { wv };
            let nb = [wv, n, nw, ne, nn, nee, ww];
            let (pred8, preds, max_error) = match &wp_rows {
                Some(rows) => rows.predict(&nb, x, y, w),
                None => (0, [0; 4], 0),
            };
            props[3] = x as i32;
            props[4] = n.unsigned_abs() as i32;
            props[5] = wv.unsigned_abs() as i32;
            props[6] = n;
            props[7] = wv;
            props[8] = if x > 0 {
                let w_prev = if x > 1 {
                    d[row + x - 2]
                } else if y > 0 {
                    d[row - w + x - 1]
                } else {
                    0
                };
                let n_prev = if y > 0 { d[row - w + x - 1] } else { w_prev };
                let nw_prev = if x > 1 && y > 0 {
                    d[row - w + x - 2]
                } else {
                    w_prev
                };
                wv.wrapping_sub(w_prev.wrapping_add(n_prev).wrapping_sub(nw_prev))
            } else {
                n
            };
            props[9] = wv.wrapping_add(n).wrapping_sub(nw);
            props[10] = wv.wrapping_sub(nw);
            props[11] = nw.wrapping_sub(n);
            props[12] = n.wrapping_sub(ne);
            props[13] = n.wrapping_sub(nn);
            props[14] = wv.wrapping_sub(ww);
            props[15] = max_error;
            for (k, r) in refs.iter().enumerate() {
                let rd = &r.data;
                let r_c = rd[row + x];
                let r_w = if x > 0 { rd[row + x - 1] } else { 0 };
                let r_n = if y > 0 { rd[row - w + x] } else { r_w };
                let r_nw = if x > 0 && y > 0 {
                    rd[row - w + x - 1]
                } else {
                    r_w
                };
                let g = r_w.wrapping_add(r_n).wrapping_sub(r_nw);
                let r_g = g.clamp(r_w.min(r_n), r_w.max(r_n));
                let o = BASE_PROPS + 4 * k;
                props[o] = r_c.unsigned_abs() as i32;
                props[o + 1] = r_c;
                props[o + 2] = r_c.wrapping_sub(r_g).unsigned_abs() as i32;
                props[o + 3] = r_c.wrapping_sub(r_g);
            }
            let sample = Sample {
                props: &props,
                nprops,
                nb,
                wp: pred8.wrapping_add(3) >> 3,
                value: d[row + x],
            };
            let v = f(&sample);
            d[row + x] = v;
            if let Some(rows) = &mut wp_rows {
                rows.store(x, pred8, &preds, v);
            }
        }
        if let Some(rows) = &mut wp_rows {
            rows.next_row();
        }
    }
}

// ---- the tree ------------------------------------------------------------

/// One MA-tree node, in bitstream (breadth-first) order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Node {
    /// `property[prop] > value` goes to `gt`, else to `le`.
    Split {
        prop: u32,
        value: i32,
        gt: u32,
        le: u32,
    },
    Leaf {
        ctx: u32,
        predictor: u32,
        multiplier: u32,
    },
}

/// An MA tree in H.4.2 order: node 0 is the root, a decision's
/// children follow every node already queued, leaves take contexts in
/// reading order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Tree {
    pub nodes: Vec<Node>,
    pub num_ctx: usize,
    /// Some leaf predicts with the self-correcting predictor or some
    /// decision reads property 15.
    pub uses_wp: bool,
}

/// A tree under construction (children by arena index).
#[derive(Debug, Clone)]
enum Build {
    Split {
        prop: u32,
        value: i32,
        gt: usize,
        le: usize,
    },
    Leaf {
        predictor: u32,
    },
}

fn finish_tree(arena: &[Build], root: usize, multiplier: u32) -> Tree {
    let mut order = std::collections::VecDeque::new();
    order.push_back(root);
    let mut nodes = Vec::with_capacity(arena.len());
    let mut num_ctx = 0u32;
    let mut uses_wp = false;
    // Children of the decision at output index k land at the back of
    // the queue: `k + queue length` and the slot after it.
    while let Some(a) = order.pop_front() {
        match arena[a] {
            Build::Split {
                prop,
                value,
                gt,
                le,
            } => {
                let k = nodes.len() as u32;
                let pending = order.len() as u32;
                nodes.push(Node::Split {
                    prop,
                    value,
                    gt: k + pending + 1,
                    le: k + pending + 2,
                });
                order.push_back(gt);
                order.push_back(le);
                uses_wp |= prop == 15;
            }
            Build::Leaf { predictor } => {
                nodes.push(Node::Leaf {
                    ctx: num_ctx,
                    predictor,
                    multiplier,
                });
                num_ctx += 1;
                uses_wp |= predictor == 6;
            }
        }
    }
    Tree {
        nodes,
        num_ctx: num_ctx as usize,
        uses_wp,
    }
}

impl Tree {
    /// A one-leaf tree.
    pub(crate) fn single(predictor: u32, multiplier: u32) -> Self {
        finish_tree(&[Build::Leaf { predictor }], 0, multiplier)
    }

    /// `(context, predictor, multiplier)` of the leaf `props` reach.
    #[inline]
    pub(crate) fn leaf(&self, props: &[i32; MAX_PROPS]) -> (u32, u32, u32) {
        let mut k = 0usize;
        loop {
            match self.nodes[k] {
                Node::Split {
                    prop,
                    value,
                    gt,
                    le,
                } => {
                    k = if props[prop as usize] > value { gt } else { le } as usize;
                }
                Node::Leaf {
                    ctx,
                    predictor,
                    multiplier,
                } => return (ctx, predictor, multiplier),
            }
        }
    }

    /// The H.4.2 tree stream as `(context, value)` symbols.
    pub(crate) fn symbols(&self) -> Vec<(u32, u32)> {
        let mut out = Vec::with_capacity(self.nodes.len() * 4);
        for n in &self.nodes {
            match *n {
                Node::Split { prop, value, .. } => {
                    out.push((1, prop + 1));
                    out.push((0, pack_signed(value)));
                }
                Node::Leaf {
                    predictor,
                    multiplier,
                    ..
                } => {
                    out.push((1, 0));
                    out.push((2, predictor));
                    out.push((3, 0)); // offset
                    let mul_log = multiplier.trailing_zeros().min(30);
                    out.push((4, mul_log));
                    out.push((5, (multiplier >> mul_log) - 1));
                }
            }
        }
        out
    }
}

/// Divide a residual by the leaf multiplier, rounding half away from
/// zero.
#[inline]
pub(crate) fn quantise(r: i64, m: i64) -> i64 {
    if m == 1 {
        r
    } else if r >= 0 {
        (r + m / 2) / m
    } else {
        -((-r + m / 2) / m)
    }
}

/// Code every channel of `section` under `tree`: the `(context, packed
/// residual)` symbols in decode order. With a multiplier above one the
/// section's samples are replaced by what the decoder reconstructs.
pub(crate) fn encode_section(section: &mut Section, tree: &Tree) -> Vec<(u32, u32)> {
    let mut out = Vec::with_capacity(section.num_samples());
    for i in 0..section.chans.len() {
        scan_channel(
            &mut section.chans,
            i,
            section.stream_index,
            tree.uses_wp,
            |s| {
                let (ctx, predictor, multiplier) = tree.leaf(s.props);
                let pred = predict(predictor, &s.nb, s.wp);
                let r = i64::from(s.value) - i64::from(pred);
                if multiplier == 1 {
                    out.push((ctx, pack_signed(r as i32)));
                    s.value
                } else {
                    let m = i64::from(multiplier);
                    let q = quantise(r, m).clamp(i64::from(i32::MIN) + 1, i64::from(i32::MAX));
                    out.push((ctx, pack_signed(q as i32)));
                    (i64::from(pred) + q * m).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
                }
            },
        );
    }
    out
}

// ---- learning ------------------------------------------------------------

/// Knobs of [`learn_tree`].
#[derive(Debug, Clone)]
pub(crate) struct LearnParams {
    /// Samples the learner works on (the picture is subsampled to
    /// about this many).
    pub max_samples: usize,
    /// Bits a split has to save, at full-picture scale, to be kept:
    /// the price of one more context (its histogram and tree nodes).
    pub split_bits: f64,
    /// Upper bound on leaves.
    pub max_leaves: usize,
    /// Residual quantiser (leaf multiplier); 1 = lossless.
    pub multiplier: u32,
    /// Offer the self-correcting predictor and property 15.
    pub use_wp: bool,
}

impl Default for LearnParams {
    fn default() -> Self {
        Self {
            max_samples: 320_000,
            split_bits: 96.0,
            max_leaves: 768,
            multiplier: 1,
            use_wp: true,
        }
    }
}

/// Token and extra-bit count of a packed residual under the
/// `split 16, two MSBs in token` hybrid configuration — the learner's
/// cost model.
#[inline]
fn cost_token(v: u32) -> u8 {
    if v < 16 {
        v as u8
    } else {
        let l = 31 - v.leading_zeros();
        (16 + ((l - 4) << 2) + ((v >> (l - 2)) & 3)) as u8
    }
}

#[inline]
fn token_extra_bits(t: u8) -> u32 {
    if t < 16 {
        0
    } else {
        u32::from((t - 16) >> 2) + 4 - 2
    }
}

const TOKENS: usize = 16 + 28 * 4;
/// Thresholds tried per property.
const MAX_THRESHOLDS: usize = 63;
/// Predictors kept in lock-step during a node's split search.
const SEARCH_PREDICTORS: usize = 3;

struct Learner {
    n: usize,
    /// `n × MAX_PROPS` threshold buckets.
    bucket: Vec<u8>,
    /// `n × NUM_PREDICTORS` residual tokens.
    tok: Vec<u8>,
    /// Properties available per sample.
    nprops: Vec<u8>,
    thresholds: Vec<Vec<i32>>,
    /// `c × log2(c)`.
    clog: Vec<f64>,
    predictors: Vec<u32>,
    /// Sampled bits a split has to save.
    min_gain: f64,
    max_leaves: usize,
    leaves: usize,
    arena: Vec<Build>,
}

impl Learner {
    #[inline]
    fn f(&self, c: u32) -> f64 {
        match self.clog.get(c as usize) {
            Some(&v) => v,
            None => f64::from(c) * f64::from(c).log2(),
        }
    }

    /// Cost in bits of the node's samples under each predictor.
    fn predictor_costs(&self, idx: &[u32]) -> Vec<(f64, u32)> {
        let mut out = Vec::with_capacity(self.predictors.len());
        for &p in &self.predictors {
            let mut hist = [0u32; TOKENS];
            let mut extra = 0u64;
            for &s in idx {
                let t = self.tok[s as usize * NUM_PREDICTORS + p as usize];
                hist[t as usize] += 1;
                extra += u64::from(token_extra_bits(t));
            }
            let mut cost = self.f(idx.len() as u32) + extra as f64;
            for &c in &hist {
                if c > 1 {
                    cost -= self.f(c);
                }
            }
            out.push((cost, p));
        }
        // Ties go to the predictors that behave best when the cost
        // model is blind (flat or heavily quantised data).
        let rank = |p: u32| match p {
            5 => 0,
            6 => 1,
            1 => 2,
            2 => 3,
            0 => 20,
            other => 4 + other,
        };
        out.sort_by(|a, b| a.0.total_cmp(&b.0).then(rank(a.1).cmp(&rank(b.1))));
        out
    }

    fn build(&mut self, idx: Vec<u32>, depth: usize) -> usize {
        let costs = self.predictor_costs(&idx);
        let (node_cost, node_pred) = costs[0];
        let n = idx.len();
        if n < 32 || depth >= 48 || self.leaves >= self.max_leaves || node_cost <= self.min_gain {
            self.arena.push(Build::Leaf {
                predictor: node_pred,
            });
            return self.arena.len() - 1;
        }
        let search: Vec<u32> = costs
            .iter()
            .take(SEARCH_PREDICTORS)
            .map(|&(_, p)| p)
            .collect();
        let k_n = search.len();
        let min_props = idx
            .iter()
            .map(|&s| self.nprops[s as usize])
            .min()
            .unwrap_or(0) as usize;
        // Total histograms per search predictor.
        let mut total = vec![[0u32; TOKENS]; k_n];
        let mut total_extra = vec![0u64; k_n];
        for &s in &idx {
            for (k, &p) in search.iter().enumerate() {
                let t = self.tok[s as usize * NUM_PREDICTORS + p as usize];
                total[k][t as usize] += 1;
                total_extra[k] += u64::from(token_extra_bits(t));
            }
        }
        let total_s: Vec<f64> = total
            .iter()
            .map(|h| h.iter().map(|&c| self.f(c)).sum())
            .collect();
        let mut best = (node_cost - self.min_gain, usize::MAX, 0usize);
        let mut order: Vec<u32> = vec![0; n];
        for prop in 0..min_props {
            if prop == 1 || self.thresholds[prop].is_empty() {
                continue;
            }
            let nb = self.thresholds[prop].len() + 1;
            // Counting sort by bucket.
            let mut starts = vec![0usize; nb + 1];
            for &s in &idx {
                starts[self.bucket[s as usize * MAX_PROPS + prop] as usize + 1] += 1;
            }
            if starts.iter().filter(|&&c| c > 0).count() < 2 {
                continue;
            }
            for b in 0..nb {
                starts[b + 1] += starts[b];
            }
            let mut fill = starts.clone();
            for &s in &idx {
                let b = self.bucket[s as usize * MAX_PROPS + prop] as usize;
                order[fill[b]] = s;
                fill[b] += 1;
            }
            let mut left = vec![[0u32; TOKENS]; k_n];
            let mut right = total.clone();
            let mut s_left = vec![0.0f64; k_n];
            let mut s_right = total_s.clone();
            let mut extra_left = vec![0u64; k_n];
            for b in 0..nb - 1 {
                for &s in &order[starts[b]..starts[b + 1]] {
                    for (k, &p) in search.iter().enumerate() {
                        let t = self.tok[s as usize * NUM_PREDICTORS + p as usize] as usize;
                        let cl = left[k][t];
                        s_left[k] += self.f(cl + 1) - self.f(cl);
                        left[k][t] = cl + 1;
                        let cr = right[k][t];
                        s_right[k] += self.f(cr - 1) - self.f(cr);
                        right[k][t] = cr - 1;
                        extra_left[k] += u64::from(token_extra_bits(t as u8));
                    }
                }
                let n_left = starts[b + 1];
                let n_right = n - n_left;
                if starts[b + 1] == starts[b] || n_left < 12 || n_right < 12 {
                    continue;
                }
                let mut cl = f64::INFINITY;
                let mut cr = f64::INFINITY;
                for k in 0..k_n {
                    cl = cl.min(self.f(n_left as u32) - s_left[k] + extra_left[k] as f64);
                    cr = cr.min(
                        self.f(n_right as u32) - s_right[k]
                            + (total_extra[k] - extra_left[k]) as f64,
                    );
                }
                if cl + cr < best.0 {
                    best = (cl + cr, prop, b);
                }
            }
        }
        if best.1 == usize::MAX {
            self.arena.push(Build::Leaf {
                predictor: node_pred,
            });
            return self.arena.len() - 1;
        }
        let (_, prop, b) = best;
        // Samples of buckets 0..=b have `property <= threshold[b]`.
        let (le, gt): (Vec<u32>, Vec<u32>) = idx
            .into_iter()
            .partition(|&s| self.bucket[s as usize * MAX_PROPS + prop] as usize <= b);
        let slot = self.arena.len();
        self.arena.push(Build::Leaf {
            predictor: node_pred,
        });
        self.leaves += 1;
        let gt_node = self.build(gt, depth + 1);
        let le_node = self.build(le, depth + 1);
        self.arena[slot] = Build::Split {
            prop: prop as u32,
            value: self.thresholds[prop][b],
            gt: gt_node,
            le: le_node,
        };
        slot
    }
}

/// Learn an MA tree for `sections` (left untouched).
pub(crate) fn learn_tree(sections: &[Section], params: &LearnParams) -> Tree {
    let total: usize = sections.iter().map(Section::num_samples).sum();
    let multiplier = params.multiplier.max(1);
    if total == 0 {
        return Tree::single(0, multiplier);
    }
    // Which sections to scan and with what per-sample probability.
    let target = params.max_samples.max(1024);
    let stride = if total > target * 8 {
        (total / (target * 8)).max(1)
    } else {
        1
    };
    let picked: Vec<usize> = (0..sections.len()).step_by(stride).collect();
    let scanned: usize = picked.iter().map(|&s| sections[s].num_samples()).sum();
    // Probability as a 32-bit threshold.
    let keep = if scanned <= target {
        u64::from(u32::MAX) + 1
    } else {
        ((target as u128 * (1u128 << 32)) / scanned as u128) as u64
    };
    let fraction = (keep as f64 / (1u64 << 32) as f64) * (scanned as f64 / total as f64);
    let m = i64::from(multiplier);
    let mut raw_props: Vec<i32> = Vec::new();
    let mut tok: Vec<u8> = Vec::new();
    let mut nprops: Vec<u8> = Vec::new();
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    for &si in &picked {
        let mut section = sections[si].clone();
        for i in 0..section.chans.len() {
            scan_channel(
                &mut section.chans,
                i,
                section.stream_index,
                params.use_wp,
                |s| {
                    rng = rng
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    if (rng >> 32) < keep {
                        raw_props.extend_from_slice(s.props);
                        nprops.push(s.nprops as u8);
                        for p in 0..NUM_PREDICTORS as u32 {
                            let r = i64::from(s.value) - i64::from(predict(p, &s.nb, s.wp));
                            let q =
                                quantise(r, m).clamp(i64::from(i32::MIN) + 1, i64::from(i32::MAX));
                            tok.push(cost_token(pack_signed(q as i32)));
                        }
                    }
                    if m == 1 {
                        s.value
                    } else {
                        // Quantised coding is closed-loop: learn on the
                        // kind of neighbourhood the decoder will see.
                        let pred = predict(5, &s.nb, s.wp);
                        let q = quantise(i64::from(s.value) - i64::from(pred), m);
                        (i64::from(pred) + q * m).clamp(i64::from(i32::MIN), i64::from(i32::MAX))
                            as i32
                    }
                },
            );
        }
    }
    let n = nprops.len();
    if n == 0 {
        return Tree::single(5, multiplier);
    }
    // Thresholds: quantiles of each property's sampled values.
    let mut thresholds: Vec<Vec<i32>> = Vec::with_capacity(MAX_PROPS);
    let mut bucket = vec![0u8; n * MAX_PROPS];
    let mut column: Vec<i32> = Vec::with_capacity(n);
    for prop in 0..MAX_PROPS {
        column.clear();
        column.extend((0..n).map(|s| raw_props[s * MAX_PROPS + prop]));
        column.sort_unstable();
        let mut t: Vec<i32> = Vec::new();
        for q in 1..=MAX_THRESHOLDS {
            let v = column[(q * n / (MAX_THRESHOLDS + 1)).min(n - 1)];
            // `property > v` must leave something on both sides.
            if v < column[n - 1] && t.last() != Some(&v) {
                t.push(v);
            }
        }
        // Few distinct values: make sure the smallest ones can be
        // isolated too.
        if column[0] < column[n - 1] && t.first() != Some(&column[0]) && t.len() < MAX_THRESHOLDS {
            t.insert(0, column[0]);
        }
        for s in 0..n {
            let v = raw_props[s * MAX_PROPS + prop];
            bucket[s * MAX_PROPS + prop] = t.partition_point(|&x| x < v) as u8;
        }
        thresholds.push(t);
    }
    drop(raw_props);
    let table = (n + 2).min(1 << 20);
    let mut clog = vec![0.0f64; table];
    for (c, slot) in clog.iter_mut().enumerate().skip(2) {
        *slot = c as f64 * (c as f64).log2();
    }
    let predictors: Vec<u32> = (0..NUM_PREDICTORS as u32)
        .filter(|&p| params.use_wp || p != 6)
        .collect();
    if !params.use_wp {
        thresholds[15].clear();
    }
    let mut learner = Learner {
        n,
        bucket,
        tok,
        nprops,
        thresholds,
        clog,
        predictors,
        // A split chosen among many candidates overfits the sample by a
        // few bits; the floor keeps those out.
        min_gain: (params.split_bits * fraction).max(14.0),
        max_leaves: params.max_leaves.max(1),
        leaves: 1,
        arena: Vec::new(),
    };
    let all: Vec<u32> = (0..learner.n as u32).collect();
    let root = learner.build(all, 0);
    finish_tree(&learner.arena, root, multiplier)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular_fdis::{ChannelDesc, ModularImage, Neighbours, WpHeader, WpState};

    fn noisy(w: usize, h: usize, seed: u32) -> Vec<i32> {
        let mut s = seed | 1;
        (0..w * h)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let x = (i % w) as i32;
                let y = (i / w) as i32;
                (x * 3 + y * 2 + (s % 23) as i32 - 11).clamp(-40, 300)
            })
            .collect()
    }

    #[test]
    fn scan_matches_the_decoder_properties_and_weighted_prediction() {
        let (w, h) = (13usize, 9usize);
        let chans = vec![
            Chan::new(w, h, noisy(w, h, 7)),
            Chan::new(w, h, noisy(w, h, 99)),
        ];
        let img = ModularImage {
            channels: chans.iter().map(|c| c.data.clone()).collect(),
            descs: vec![
                ChannelDesc {
                    width: w as u32,
                    height: h as u32,
                    hshift: 0,
                    vshift: 0,
                };
                2
            ],
        };
        let wp = WpHeader::default();
        for i in 0..2 {
            let mut state = WpState::new(w as u32, h as u32);
            let mut work = chans.clone();
            let mut pos = 0usize;
            scan_channel(&mut work, i, 5, true, |s| {
                let (x, y) = ((pos % w) as i32, (pos / w) as i32);
                let nb = Neighbours::at(&img, i, x, y);
                let (p8, _, max_error) =
                    crate::modular_fdis::wp_predict_pub(&state, &nb, x, y, &wp);
                let want = crate::modular_fdis::get_properties(&img, i, x, y, 5, max_error);
                assert_eq!(&s.props[..want.len()], &want[..], "props at {x},{y} ch {i}");
                assert_eq!(s.nprops, want.len());
                assert_eq!(s.wp, (p8 + 3) >> 3, "wp at {x},{y}");
                crate::modular_fdis::wp_store_pub(&mut state, &nb, x, y, &wp, s.value);
                pos += 1;
                s.value
            });
            assert_eq!(pos, w * h);
        }
    }

    #[test]
    fn tree_order_matches_the_reader() {
        // root(split) -> gt: leaf A, le: split -> (leaf B, leaf C)
        let arena = vec![
            Build::Split {
                prop: 0,
                value: 0,
                gt: 1,
                le: 2,
            },
            Build::Leaf { predictor: 1 },
            Build::Split {
                prop: 3,
                value: 7,
                gt: 3,
                le: 4,
            },
            Build::Leaf { predictor: 2 },
            Build::Leaf { predictor: 5 },
        ];
        let t = finish_tree(&arena, 0, 1);
        assert_eq!(t.num_ctx, 3);
        assert_eq!(
            t.nodes[0],
            Node::Split {
                prop: 0,
                value: 0,
                gt: 1,
                le: 2
            }
        );
        assert!(matches!(
            t.nodes[1],
            Node::Leaf {
                ctx: 0,
                predictor: 1,
                ..
            }
        ));
        assert_eq!(
            t.nodes[2],
            Node::Split {
                prop: 3,
                value: 7,
                gt: 3,
                le: 4
            }
        );
        let mut props = [0i32; MAX_PROPS];
        props[0] = 1;
        assert_eq!(t.leaf(&props).1, 1);
        props[0] = 0;
        props[3] = 8;
        assert_eq!(t.leaf(&props).1, 2);
        props[3] = 7;
        assert_eq!(t.leaf(&props).1, 5);
    }

    #[test]
    fn learning_separates_two_differently_behaved_channels() {
        let (w, h) = (64usize, 64usize);
        // Channel 0: horizontal ramp (West predicts exactly after the
        // first column); channel 1: vertical ramp.
        let a: Vec<i32> = (0..w * h).map(|i| ((i % w) * 5) as i32).collect();
        let b: Vec<i32> = (0..w * h).map(|i| ((i / w) * 7) as i32).collect();
        let sections = vec![Section {
            stream_index: 0,
            chans: vec![Chan::new(w, h, a), Chan::new(w, h, b)],
        }];
        let tree = learn_tree(&sections, &LearnParams::default());
        let mut s = sections[0].clone();
        let syms = encode_section(&mut s, &tree);
        let nonzero = syms.iter().filter(|&&(_, v)| v != 0).count();
        assert!(nonzero < 256, "{nonzero} non-zero residuals, tree {tree:?}");
    }
}
