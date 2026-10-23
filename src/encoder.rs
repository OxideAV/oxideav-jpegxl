//! JPEG XL **encoder** — a Modular (Annex H) codestream writer.
//!
//! What it writes, in codestream order:
//!
//! * the signature, `SizeHeader` (D.2) and `ImageMetadata` (D.3) —
//!   integer samples at 1..=16 bits, grey or RGB, an optional alpha
//!   extra channel, the enumerated `ColourEncoding` (E.2) or an
//!   embedded ICC profile (E.4, written as one verbatim command so
//!   the profile bytes round-trip exactly), the `AnimationHeader`
//!   (D.3.4) for multi-frame input;
//! * one `kModular` frame per picture (F.2): no upsampling, no crop,
//!   `kReplace` blending, no restoration filter, `group_size_shift`
//!   from the options; a global MA tree (H.4.2) with one leaf per
//!   channel — the leaf's predictor is picked per channel among West /
//!   North / Select / Gradient by residual energy, its `multiplier`
//!   is the quality quantiser (`1` = lossless) — and the YCoCg RCT
//!   (H.6.3, `rct_type = 6`) in front of three colour channels;
//! * the channel residuals (H.3), closed-loop quantised when
//!   `multiplier > 1`, entropy coded with prefix codes
//!   ([`crate::entropy_write`]); the whole picture in `GlobalModular`
//!   when it fits one group, else one `PassGroup` section per group
//!   (G.4.2) sharing the global tree.
//!
//! Everything produced here is decodable by [`crate::decode`] and
//! round-trips bit-exactly at quality 100 (the integration tests pin
//! that on every layout, and against a black-box decoder where one is
//! installed).

use crate::bitreader::U32Dist;
use crate::bitwriter::{BitWriter, pack_signed};
use crate::entropy_write::{Coding, EntropyCodes, HybridConfig, SymbolStats};
use crate::error::{Error, Result};
use crate::icc::icc_context;
use crate::image::ColorInfo;

/// `U32(1 + u(9), 1 + u(13), 1 + u(18), 1 + u(30))` — SizeHeader dims.
const SIZE_DIST: [U32Dist; 4] = [
    U32Dist::BitsOffset(9, 1),
    U32Dist::BitsOffset(13, 1),
    U32Dist::BitsOffset(18, 1),
    U32Dist::BitsOffset(30, 1),
];

/// `U32(u(10), 1024 + u(14), 17408 + u(22), 4211712 + u(30))` — TOC.
const TOC_DIST: [U32Dist; 4] = [
    U32Dist::Bits(10),
    U32Dist::BitsOffset(14, 1024),
    U32Dist::BitsOffset(22, 17408),
    U32Dist::BitsOffset(30, 4211712),
];

/// `U32(0, 1, 2 + u(4), 18 + u(8))` — `nb_transforms`.
const NB_TRANSFORMS_DIST: [U32Dist; 4] = [
    U32Dist::Val(0),
    U32Dist::Val(1),
    U32Dist::BitsOffset(4, 2),
    U32Dist::BitsOffset(8, 18),
];

/// Largest section the TOC can describe.
const MAX_SECTION_BYTES: usize = 4211712 + (1 << 30) - 1;

/// Streams with at least this many symbols are ANS coded (fractional
/// bits per symbol); shorter ones keep prefix codes, whose histograms
/// are cheaper than the 32-bit ANS state initialiser.
const ANS_MIN_SYMBOLS: usize = 64;

fn coding_for(num_symbols: usize) -> Coding {
    if num_symbols >= ANS_MIN_SYMBOLS {
        Coding::Ans
    } else {
        Coding::Prefix
    }
}

/// Colour signalling of the codestream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColourSpec {
    /// The enumerated E.2 description: H.273 primaries / transfer code
    /// points (anything unmapped falls back to sRGB).
    Enumerated { primaries: u8, transfer: u8 },
    /// An embedded ICC profile (E.4).
    Icc(Vec<u8>),
}

impl ColourSpec {
    /// From the contract's [`ColorInfo`].
    pub fn from_color_info(c: &ColorInfo) -> Self {
        Self::Enumerated {
            primaries: c.primaries,
            transfer: c.transfer,
        }
    }
}

/// `AnimationHeader` fields (D.3.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnimationSpec {
    pub tps_numerator: u32,
    pub tps_denominator: u32,
    pub num_loops: u32,
}

/// The image-level parameters of one codestream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodeSpec {
    pub width: u32,
    pub height: u32,
    /// Integer sample depth of the colour channels and the alpha
    /// channel, 1..=16.
    pub bits: u32,
    /// One colour channel instead of three.
    pub grey: bool,
    /// An alpha extra channel follows the colour channels.
    pub alpha: bool,
    pub colour: ColourSpec,
    /// `Some` writes `have_animation` and per-frame durations.
    pub animation: Option<AnimationSpec>,
    /// 0..=100; 100 is lossless (see [`quantizer_for_quality`]).
    pub quality: u8,
    /// `group_dim = 128 << group_size_shift`, 0..=3.
    pub group_size_shift: u32,
}

impl EncodeSpec {
    fn num_channels(&self) -> usize {
        (if self.grey { 1 } else { 3 }) + usize::from(self.alpha)
    }
    fn group_dim(&self) -> u32 {
        128 << self.group_size_shift
    }
    fn num_groups(&self) -> u64 {
        let g = u64::from(self.group_dim());
        u64::from(self.width).div_ceil(g) * u64::from(self.height).div_ceil(g)
    }
    fn num_lf_groups(&self) -> u64 {
        let g = u64::from(self.group_dim()) * 8;
        u64::from(self.width).div_ceil(g) * u64::from(self.height).div_ceil(g)
    }
}

/// One picture to encode: `channels` in codestream order (grey or R,
/// G, B, then alpha), each `width × height` samples in raster order,
/// already in the integer range of `bits`.
#[derive(Debug, Clone)]
pub struct FrameInput {
    pub channels: Vec<Vec<i32>>,
    /// Frame duration in animation ticks (ignored without an
    /// `AnimationSpec`).
    pub duration: u32,
}

/// The Modular residual quantiser for `quality` at `bits` per sample:
/// `1` (lossless) at 100, otherwise `round(2^((100 − q) / 12))` scaled
/// by `2^(bits − 8)` so the step is the same fraction of the sample
/// range at every depth. Monotone: a lower quality never yields a
/// smaller step.
pub fn quantizer_for_quality(quality: u8, bits: u32) -> u32 {
    let q = quality.min(100);
    if q == 100 {
        return 1;
    }
    let base = (2f64.powf(f64::from(100 - q) / 12.0)).round() as u32;
    let base = base.max(2);
    let scaled = if bits > 8 {
        u64::from(base) << (bits - 8)
    } else if bits < 8 {
        (u64::from(base) >> (8 - bits)).max(1)
    } else {
        u64::from(base)
    };
    scaled.min(1 << 24) as u32
}

// ---- headers ---------------------------------------------------------------

/// Signature + `SizeHeader` + `ImageMetadata` (+ ICC stream), then
/// `ZeroPadToByte` — what precedes the first frame.
pub fn write_prelude(w: &mut BitWriter, spec: &EncodeSpec) -> Result<()> {
    if spec.width == 0 || spec.height == 0 {
        return Err(Error::invalid("JXL encoder: zero image dimension"));
    }
    if spec.width > (1 << 30) || spec.height > (1 << 30) {
        return Err(Error::unsupported(
            "JXL encoder: dimensions beyond the SizeHeader range (2^30)",
        ));
    }
    if spec.bits == 0 || spec.bits > 16 {
        return Err(Error::unsupported(format!(
            "JXL encoder: {} bits per sample (1..=16 supported)",
            spec.bits
        )));
    }
    if spec.group_size_shift > 3 {
        return Err(Error::invalid("JXL encoder: group_size_shift > 3"));
    }
    w.write_bits(8, 0xFF);
    w.write_bits(8, 0x0A);
    write_size_header(w, spec.width, spec.height)?;

    // ImageMetadata (Table D.3).
    w.write_bool(false); // all_default
    let extra_fields = spec.animation.is_some();
    w.write_bool(extra_fields);
    if extra_fields {
        w.write_bits(3, 0); // orientation 1
        w.write_bool(false); // have_intr_size
        w.write_bool(false); // have_preview
        w.write_bool(true); // have_animation
        let a = spec.animation.expect("extra_fields implies animation");
        if a.tps_numerator == 0 || a.tps_denominator == 0 {
            return Err(Error::invalid(
                "JXL encoder: animation ticks per second is not a rate",
            ));
        }
        w.write_u32(
            [
                U32Dist::Val(100),
                U32Dist::Val(1000),
                U32Dist::BitsOffset(10, 1),
                U32Dist::BitsOffset(30, 1),
            ],
            a.tps_numerator,
        )?;
        w.write_u32(
            [
                U32Dist::Val(1),
                U32Dist::Val(1001),
                U32Dist::BitsOffset(8, 1),
                U32Dist::BitsOffset(10, 1),
            ],
            a.tps_denominator,
        )?;
        w.write_u32(
            [
                U32Dist::Val(0),
                U32Dist::Bits(3),
                U32Dist::Bits(16),
                U32Dist::Bits(32),
            ],
            a.num_loops,
        )?;
        w.write_bool(false); // have_timecodes
    }
    write_bit_depth(w, spec.bits)?;
    // modular_16bit_buffers: signed 16-bit holds every sample and
    // inverse-RCT intermediate only up to 8-bit input.
    w.write_bool(spec.bits <= 8);
    w.write_u32(
        [
            U32Dist::Val(0),
            U32Dist::Val(1),
            U32Dist::BitsOffset(4, 2),
            U32Dist::BitsOffset(12, 1),
        ],
        u32::from(spec.alpha),
    )?;
    if spec.alpha {
        // ExtraChannelInfo (Table D.8): the all-default alpha channel is
        // 8-bit; any other depth spells the bundle out.
        if spec.bits == 8 {
            w.write_bool(true);
        } else {
            w.write_bool(false);
            w.write_enum(0)?; // kAlpha
            write_bit_depth(w, spec.bits)?;
            w.write_u32(
                [
                    U32Dist::Val(0),
                    U32Dist::Val(3),
                    U32Dist::Val(4),
                    U32Dist::BitsOffset(3, 1),
                ],
                0,
            )?; // dim_shift
            w.write_u32(
                [
                    U32Dist::Val(0),
                    U32Dist::Bits(4),
                    U32Dist::BitsOffset(5, 16),
                    U32Dist::BitsOffset(10, 48),
                ],
                0,
            )?; // name_len
            w.write_bool(false); // alpha_associated
        }
    }
    w.write_bool(false); // xyb_encoded
    write_colour_encoding(w, spec)?;
    if extra_fields {
        w.write_bool(true); // tone_mapping.all_default
    }
    w.write_u64(0); // extensions
    w.write_bool(true); // default_m (default transform data)

    if let ColourSpec::Icc(profile) = &spec.colour {
        write_icc_stream(w, profile)?;
    }
    w.zero_pad_to_byte();
    Ok(())
}

fn write_size_header(w: &mut BitWriter, width: u32, height: u32) -> Result<()> {
    let small =
        width.is_multiple_of(8) && height.is_multiple_of(8) && width <= 256 && height <= 256;
    w.write_bool(small);
    if small {
        w.write_bits(5, height / 8 - 1);
    } else {
        w.write_u32(SIZE_DIST, height)?;
    }
    w.write_bits(3, 0); // ratio: explicit width
    if small {
        w.write_bits(5, width / 8 - 1);
    } else {
        w.write_u32(SIZE_DIST, width)?;
    }
    Ok(())
}

fn write_bit_depth(w: &mut BitWriter, bits: u32) -> Result<()> {
    w.write_bool(false); // float_sample
    w.write_u32(
        [
            U32Dist::Val(8),
            U32Dist::Val(10),
            U32Dist::Val(12),
            U32Dist::BitsOffset(6, 1),
        ],
        bits,
    )
}

/// Table E.5 / E.6 code points from the H.273 values [`ColorInfo`]
/// carries.
fn enum_primaries(primaries: u8) -> u32 {
    match primaries {
        9 => 9,   // kP2100 (BT.2100)
        11 => 11, // kP3
        _ => 1,   // kSRGB
    }
}

fn enum_transfer(transfer: u8) -> u32 {
    match transfer {
        1 => 1,   // k709
        8 => 8,   // kLinear
        16 => 16, // kPQ
        17 => 17, // kDCI
        18 => 18, // kHLG
        _ => 13,  // kSRGB
    }
}

fn write_colour_encoding(w: &mut BitWriter, spec: &EncodeSpec) -> Result<()> {
    match &spec.colour {
        ColourSpec::Icc(_) => {
            w.write_bool(false); // all_default
            w.write_bool(true); // want_icc
            w.write_enum(u32::from(spec.grey))?; // kRGB = 0, kGrey = 1
        }
        ColourSpec::Enumerated {
            primaries,
            transfer,
        } => {
            let prim = enum_primaries(*primaries);
            let tf = enum_transfer(*transfer);
            if !spec.grey && prim == 1 && tf == 13 {
                w.write_bool(true); // all_default: sRGB
                return Ok(());
            }
            w.write_bool(false);
            w.write_bool(false); // want_icc
            w.write_enum(u32::from(spec.grey))?; // colour_space
            w.write_enum(1)?; // white_point kD65
            if !spec.grey {
                w.write_enum(prim)?;
            }
            w.write_bool(false); // have_gamma
            w.write_enum(tf)?;
            w.write_enum(1)?; // rendering_intent kRelative
        }
    }
    Ok(())
}

// ---- ICC ---------------------------------------------------------------------

fn varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 127) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 128);
    }
}

/// The E.4.2 encoded ICC stream for `profile`: header bytes as
/// residuals against the E.4.3 prediction, then — for profiles longer
/// than 128 bytes — an empty tag list (`num_tags = -1`) and one
/// verbatim main-content command carrying the rest, so the decoded
/// profile is byte-identical whatever its tag structure.
pub fn encoded_icc_stream(profile: &[u8]) -> Vec<u8> {
    let size = profile.len();
    let header_len = size.min(128);
    let mut commands = Vec::new();
    if size > 128 {
        varint(&mut commands, 0); // num_tags + 1 = 0 → no tag list
        commands.push(1); // main content: verbatim bytes
        varint(&mut commands, (size - header_len) as u64);
    }
    let mut out = Vec::with_capacity(size + 16);
    varint(&mut out, size as u64);
    varint(&mut out, commands.len() as u64);
    out.extend_from_slice(&commands);
    for i in 0..header_len {
        let p = crate::icc::predict_header_byte(i, &profile[..i], size as u64);
        out.push(profile[i].wrapping_sub(p));
    }
    out.extend_from_slice(&profile[header_len..]);
    out
}

/// E.4.1: `enc_size` as `U64()`, the 41-context prelude and the
/// entropy-coded encoded ICC stream.
pub fn write_icc_stream(w: &mut BitWriter, profile: &[u8]) -> Result<()> {
    if profile.len() < 128 {
        return Err(Error::invalid(
            "JXL encoder: an ICC profile is at least 128 bytes (the header)",
        ));
    }
    let encoded = encoded_icc_stream(profile);
    let mut stats = SymbolStats::single(41, HybridConfig::BYTES)?;
    let mut prev = 0u8;
    let mut prev_prev = 0u8;
    for (i, &b) in encoded.iter().enumerate() {
        stats.add(icc_context(i, prev, prev_prev), u32::from(b));
        prev_prev = prev;
        prev = b;
    }
    let codes = EntropyCodes::build_with(&stats, coding_for(encoded.len()))?;
    w.write_u64(encoded.len() as u64);
    codes.write_prelude(w);
    let mut prev = 0u8;
    let mut prev_prev = 0u8;
    let symbols: Vec<(u32, u32)> = encoded
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let ctx = icc_context(i, prev, prev_prev);
            prev_prev = prev;
            prev = b;
            (ctx, u32::from(b))
        })
        .collect();
    codes.write_symbols(w, symbols.iter().copied());
    Ok(())
}

// ---- MA tree ---------------------------------------------------------------

#[derive(Debug, Clone)]
enum TreeNode {
    Decision { property: u32, value: i32 },
    Leaf { predictor: u32, multiplier: u32 },
}

/// A tree with one leaf per channel, splitting on property 0 (the
/// channel index); `leaf_ctx[c]` is channel `c`'s context.
struct ChannelTree {
    /// Nodes in bitstream (breadth-first) order.
    nodes: Vec<TreeNode>,
    leaf_ctx: Vec<u32>,
}

fn build_channel_tree(predictors: &[u32], multiplier: u32) -> ChannelTree {
    // Build as a binary split of the channel range, then serialise
    // breadth-first (the H.4.2 reading order: children are appended
    // at `tree.size() + nodes_left`).
    #[derive(Clone)]
    enum Tmp {
        Dec(u32, i32, Box<Tmp>, Box<Tmp>),
        Leaf(usize),
    }
    fn split(lo: usize, hi: usize) -> Tmp {
        if lo == hi {
            Tmp::Leaf(lo)
        } else {
            let mid = lo + (hi - lo) / 2;
            // property[0] > mid → channels mid+1..=hi (left branch).
            Tmp::Dec(
                0,
                mid as i32,
                Box::new(split(mid + 1, hi)),
                Box::new(split(lo, mid)),
            )
        }
    }
    let n = predictors.len();
    let root = split(0, n - 1);
    let mut queue: std::collections::VecDeque<Tmp> = std::collections::VecDeque::new();
    queue.push_back(root);
    let mut nodes = Vec::new();
    let mut leaf_ctx = vec![0u32; n];
    let mut ctx = 0u32;
    while let Some(t) = queue.pop_front() {
        match t {
            Tmp::Dec(p, v, l, r) => {
                // The reader places this node's children at
                // `tree.size() + nodes_left + {1, 2}`: exactly the next
                // two queue slots of a breadth-first walk.
                nodes.push(TreeNode::Decision {
                    property: p,
                    value: v,
                });
                queue.push_back(*l);
                queue.push_back(*r);
            }
            Tmp::Leaf(c) => {
                nodes.push(TreeNode::Leaf {
                    predictor: predictors[c],
                    multiplier,
                });
                leaf_ctx[c] = ctx;
                ctx += 1;
            }
        }
    }
    ChannelTree { nodes, leaf_ctx }
}

/// Tree symbols in H.4.2 order as `(context, value)`.
fn tree_symbols(tree: &ChannelTree) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for n in &tree.nodes {
        match n {
            TreeNode::Decision {
                property, value, ..
            } => {
                out.push((1, property + 1));
                out.push((0, pack_signed(*value)));
            }
            TreeNode::Leaf {
                predictor,
                multiplier,
            } => {
                out.push((1, 0));
                out.push((2, *predictor));
                out.push((3, 0)); // offset
                let mul_log = multiplier.trailing_zeros().min(30);
                let mul_bits = (multiplier >> mul_log) - 1;
                out.push((4, mul_log));
                out.push((5, mul_bits));
            }
        }
    }
    out
}

// ---- Modular channel coding -----------------------------------------------

/// Forward YCoCg RCT (`rct_type = 6`, permutation 0) in place — the
/// exact inverse of the decoder's H.6.3 arithmetic.
fn forward_rct(r: &mut [i32], g: &mut [i32], b: &mut [i32]) {
    for i in 0..r.len() {
        let (d, e, f) = (r[i], g[i], b[i]);
        let bb = d.wrapping_sub(f);
        let tmp = f.wrapping_add(bb >> 1);
        let c = e.wrapping_sub(tmp);
        let a = tmp.wrapping_add(c >> 1);
        r[i] = a;
        g[i] = bb;
        b[i] = c;
    }
}

/// H.3 neighbour fetch inside a rectangle of `buf` (`stride` wide):
/// `(W, N, NW, NE)` with the edge rules of Table H.2.
#[inline]
fn neighbours(
    buf: &[i32],
    stride: usize,
    x0: usize,
    y0: usize,
    rw: usize,
    x: usize,
    y: usize,
) -> (i32, i32, i32, i32) {
    let at = |xx: usize, yy: usize| buf[(y0 + yy) * stride + x0 + xx];
    let w = if x > 0 {
        at(x - 1, y)
    } else if y > 0 {
        at(x, y - 1)
    } else {
        0
    };
    let n = if y > 0 { at(x, y - 1) } else { w };
    let nw = if x > 0 && y > 0 { at(x - 1, y - 1) } else { w };
    let ne = if x + 1 < rw && y > 0 {
        at(x + 1, y - 1)
    } else {
        n
    };
    (w, n, nw, ne)
}

#[inline]
fn predict(predictor: u32, w: i32, n: i32, nw: i32, ne: i32) -> i32 {
    match predictor {
        0 => 0,
        1 => w,
        2 => n,
        4 => {
            if (i64::from(n) - i64::from(nw)).abs() < (i64::from(w) - i64::from(nw)).abs() {
                w
            } else {
                n
            }
        }
        5 => w.wrapping_add(n).wrapping_sub(nw).clamp(w.min(n), w.max(n)),
        7 => ne,
        8 => nw,
        _ => unreachable!("encoder predictors are 0, 1, 2, 4, 5, 7, 8"),
    }
}

/// Predictors the encoder chooses among (all free of integer division
/// so no rounding convention is involved).
const CANDIDATE_PREDICTORS: [u32; 4] = [1, 2, 4, 5];

/// The candidate with the least absolute residual over the (unquantised)
/// channel, measured as the decoder will see it (full-image edges).
fn pick_predictor(ch: &[i32], width: usize, height: usize) -> u32 {
    let mut best = (u64::MAX, 5u32);
    for &p in &CANDIDATE_PREDICTORS {
        let mut sum = 0u64;
        for y in 0..height {
            for x in 0..width {
                let (w, n, nw, ne) = neighbours(ch, width, 0, 0, width, x, y);
                let r = i64::from(ch[y * width + x]) - i64::from(predict(p, w, n, nw, ne));
                sum = sum.saturating_add(r.unsigned_abs());
            }
            if sum >= best.0 {
                break;
            }
        }
        if sum < best.0 {
            best = (sum, p);
        }
    }
    best.1
}

/// A group rectangle inside a channel (samples).
#[derive(Debug, Clone, Copy)]
struct Rect {
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
}

/// Code one channel rectangle: residuals (packed, possibly quantised)
/// in raster order; `buf` is updated in place with the reconstruction
/// the decoder will hold (identity at multiplier 1).
fn code_rect(
    buf: &mut [i32],
    stride: usize,
    rect: Rect,
    predictor: u32,
    multiplier: u32,
) -> Vec<u32> {
    let Rect {
        x0,
        y0,
        w: rw,
        h: rh,
    } = rect;
    let mut out = Vec::with_capacity(rw * rh);
    let m = i64::from(multiplier);
    for y in 0..rh {
        for x in 0..rw {
            let (w, n, nw, ne) = neighbours(buf, stride, x0, y0, rw, x, y);
            let pred = predict(predictor, w, n, nw, ne);
            let idx = (y0 + y) * stride + x0 + x;
            let r = i64::from(buf[idx]) - i64::from(pred);
            let q = if m == 1 {
                r
            } else {
                // Round half away from zero.
                let half = m / 2;
                if r >= 0 {
                    (r + half) / m
                } else {
                    -((-r + half) / m)
                }
            };
            let q = q.clamp(i64::from(i32::MIN) + 1, i64::from(i32::MAX));
            out.push(pack_signed(q as i32));
            if m != 1 {
                buf[idx] = (i64::from(pred) + q * m).clamp(i64::from(i32::MIN), i64::from(i32::MAX))
                    as i32;
            }
        }
    }
    out
}

/// Per-section residual symbols: `(context, packed residual)` streams.
struct CodedSection {
    symbols: Vec<(u32, Vec<u32>)>,
}

/// Everything needed to serialise one frame.
struct CodedFrame {
    tree_codes: EntropyCodes,
    tree_syms: Vec<(u32, u32)>,
    leaf_codes: EntropyCodes,
    /// One entry when the picture fits one group (GlobalModular carries
    /// the samples), else one per group in raster order.
    sections: Vec<CodedSection>,
    use_rct: bool,
}

fn code_frame(spec: &EncodeSpec, frame: &FrameInput) -> Result<CodedFrame> {
    let width = spec.width as usize;
    let height = spec.height as usize;
    let n_ch = spec.num_channels();
    if frame.channels.len() != n_ch {
        return Err(Error::invalid(format!(
            "JXL encoder: frame carries {} channel(s), {n_ch} expected",
            frame.channels.len()
        )));
    }
    let max = (1i64 << spec.bits) - 1;
    for (c, ch) in frame.channels.iter().enumerate() {
        if ch.len() != width * height {
            return Err(Error::invalid(format!(
                "JXL encoder: channel {c} has {} samples, {} expected",
                ch.len(),
                width * height
            )));
        }
        if ch.iter().any(|&v| v < 0 || i64::from(v) > max) {
            return Err(Error::invalid(format!(
                "JXL encoder: channel {c} sample outside 0..={max}"
            )));
        }
    }
    let mut chans: Vec<Vec<i32>> = frame.channels.clone();
    let use_rct = !spec.grey;
    if use_rct {
        let (r, rest) = chans.split_at_mut(1);
        let (g, rest) = rest.split_at_mut(1);
        forward_rct(&mut r[0], &mut g[0], &mut rest[0]);
    }
    let multiplier = quantizer_for_quality(spec.quality, spec.bits);
    let predictors: Vec<u32> = chans
        .iter()
        .map(|c| pick_predictor(c, width, height))
        .collect();
    let tree = build_channel_tree(&predictors, multiplier);
    let tree_syms = tree_symbols(&tree);
    let mut tree_stats = SymbolStats::single(6, HybridConfig::RESIDUALS)?;
    for &(c, v) in &tree_syms {
        tree_stats.add(c, v);
    }
    let tree_codes = EntropyCodes::build(&tree_stats);

    let group_dim = spec.group_dim() as usize;
    let single = width <= group_dim && height <= group_dim;
    let mut leaf_stats = SymbolStats::identity(n_ch, HybridConfig::RESIDUALS)?;
    let mut sections = Vec::new();
    let rects: Vec<Rect> = if single {
        vec![Rect {
            x0: 0,
            y0: 0,
            w: width,
            h: height,
        }]
    } else {
        let gx = width.div_ceil(group_dim);
        let gy = height.div_ceil(group_dim);
        let mut v = Vec::with_capacity(gx * gy);
        for j in 0..gy {
            for i in 0..gx {
                let x0 = i * group_dim;
                let y0 = j * group_dim;
                v.push(Rect {
                    x0,
                    y0,
                    w: group_dim.min(width - x0),
                    h: group_dim.min(height - y0),
                });
            }
        }
        v
    };
    for rect in rects {
        let mut symbols = Vec::with_capacity(n_ch);
        for (c, ch) in chans.iter_mut().enumerate() {
            let ctx = tree.leaf_ctx[c];
            let vals = code_rect(ch, width, rect, predictors[c], multiplier);
            for &v in &vals {
                leaf_stats.add(ctx, v);
            }
            symbols.push((ctx, vals));
        }
        sections.push(CodedSection { symbols });
    }
    let leaf_codes = EntropyCodes::build_with(&leaf_stats, coding_for(width * height))?;
    Ok(CodedFrame {
        tree_codes,
        tree_syms,
        leaf_codes,
        sections,
        use_rct,
    })
}

// ---- frame serialisation ----------------------------------------------------

/// The frame-header fields this writer emits (F.2, `kModular`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameShape {
    pub group_size_shift: u32,
    pub num_extra: u32,
    pub have_animation: bool,
    pub duration: u32,
    pub is_last: bool,
}

/// `FrameHeader` (Table F.2) + TOC (F.3) for a plain Modular frame;
/// `toc` holds the section sizes in bytes. Leaves the writer
/// byte-aligned at the first section.
pub fn write_frame_header_and_toc(
    w: &mut BitWriter,
    shape: &FrameShape,
    toc: &[u32],
) -> Result<()> {
    w.write_bool(false); // all_default
    w.write_bits(2, 0); // frame_type kRegularFrame
    w.write_bits(1, 1); // encoding kModular
    w.write_u64(0); // flags
    w.write_bool(false); // do_YCbCr
    w.write_bits(2, 0); // upsampling = 1
    for _ in 0..shape.num_extra {
        w.write_bits(2, 0); // ec_upsampling = 1
    }
    w.write_bits(2, shape.group_size_shift);
    w.write_bits(2, 0); // passes.num_passes = 1
    w.write_bool(false); // have_crop
    w.write_bits(2, 0); // blending_info.mode kReplace
    for _ in 0..shape.num_extra {
        w.write_bits(2, 0); // ec_blending_info kReplace
    }
    if shape.have_animation {
        w.write_u32(
            [
                U32Dist::Val(0),
                U32Dist::Val(1),
                U32Dist::Bits(8),
                U32Dist::Bits(32),
            ],
            shape.duration,
        )?;
    }
    w.write_bool(shape.is_last);
    if !shape.is_last {
        w.write_bits(2, 0); // save_as_reference
        // save_before_ct is present only for a frame that can be
        // referenced (`resets_canvas && can_reference`, Table F.2):
        // with save_as_reference = 0 that is a zero-duration frame.
        if !shape.have_animation || shape.duration == 0 {
            w.write_bool(false); // save_before_ct
        }
    }
    w.write_bits(2, 0); // name_len = 0
    // RestorationFilter (Table J.1): no Gabor, no EPF.
    w.write_bool(false); // all_default
    w.write_bool(false); // gab
    w.write_bits(2, 0); // epf_iters
    w.write_u64(0); // rf extensions
    w.write_u64(0); // frame extensions
    // TOC.
    w.write_bool(false); // permuted_toc
    w.zero_pad_to_byte();
    for &entry in toc {
        w.write_u32(TOC_DIST, entry)?;
    }
    w.zero_pad_to_byte();
    Ok(())
}

fn write_global_modular_header(w: &mut BitWriter, coded: &CodedFrame) -> Result<()> {
    // LfChannelDequantization: all_default.
    w.write_bool(true);
    // GlobalModular: the global tree.
    w.write_bool(true); // use_global_tree
    coded.tree_codes.write_prelude(w);
    coded
        .tree_codes
        .write_symbols(w, coded.tree_syms.iter().copied());
    coded.leaf_codes.write_prelude(w);
    // Inner ModularHeader.
    w.write_bool(true); // use_global_tree
    w.write_bool(true); // wp_params.default_wp
    if coded.use_rct {
        w.write_u32(NB_TRANSFORMS_DIST, 1)?;
        w.write_bits(2, 0); // kRCT
        w.write_u32(
            [
                U32Dist::Bits(3),
                U32Dist::BitsOffset(6, 8),
                U32Dist::BitsOffset(10, 72),
                U32Dist::BitsOffset(13, 1096),
            ],
            0,
        )?; // begin_c
        w.write_u32(
            [
                U32Dist::Val(6),
                U32Dist::Bits(2),
                U32Dist::BitsOffset(4, 2),
                U32Dist::BitsOffset(6, 10),
            ],
            6,
        )?; // rct_type YCoCg
    } else {
        w.write_u32(NB_TRANSFORMS_DIST, 0)?;
    }
    Ok(())
}

/// One section's symbols as one stream (an ANS section carries its own
/// state initialiser, so every section is a stream of its own).
fn write_section_symbols(w: &mut BitWriter, codes: &EntropyCodes, section: &CodedSection) {
    let symbols = section
        .symbols
        .iter()
        .flat_map(|(ctx, vals)| vals.iter().map(move |&v| (*ctx, v)));
    codes.write_symbols(w, symbols);
}

/// Append one frame (header, TOC, sections) to a byte-aligned writer.
fn write_frame(
    w: &mut BitWriter,
    spec: &EncodeSpec,
    frame: &FrameInput,
    is_last: bool,
) -> Result<()> {
    let coded = code_frame(spec, frame)?;
    let single = coded.sections.len() == 1 && spec.num_groups() == 1;
    let mut sections: Vec<Vec<u8>> = Vec::new();
    {
        let mut lf = BitWriter::new();
        write_global_modular_header(&mut lf, &coded)?;
        if single {
            write_section_symbols(&mut lf, &coded.leaf_codes, &coded.sections[0]);
        }
        lf.zero_pad_to_byte();
        sections.push(lf.into_bytes());
    }
    if !single {
        for _ in 0..spec.num_lf_groups() {
            sections.push(Vec::new()); // ModularLfGroup: nothing to code
        }
        sections.push(Vec::new()); // HfGlobal: absent for kModular
        for section in &coded.sections {
            let mut gw = BitWriter::new();
            gw.write_bool(true); // use_global_tree
            gw.write_bool(true); // default_wp
            gw.write_u32(NB_TRANSFORMS_DIST, 0)?;
            write_section_symbols(&mut gw, &coded.leaf_codes, section);
            gw.zero_pad_to_byte();
            sections.push(gw.into_bytes());
        }
    }
    let mut toc = Vec::with_capacity(sections.len());
    for s in &sections {
        if s.len() > MAX_SECTION_BYTES {
            return Err(Error::unsupported(
                "JXL encoder: section exceeds the TOC range",
            ));
        }
        toc.push(s.len() as u32);
    }
    let shape = FrameShape {
        group_size_shift: spec.group_size_shift,
        num_extra: u32::from(spec.alpha),
        have_animation: spec.animation.is_some(),
        duration: frame.duration,
        is_last,
    };
    write_frame_header_and_toc(w, &shape, &toc)?;
    for s in &sections {
        w.write_bytes(s);
    }
    Ok(())
}

/// Encode `frames` as one bare codestream (`FF 0A` …). One frame
/// without an [`AnimationSpec`] is a still; several frames require
/// one.
pub fn encode_codestream(spec: &EncodeSpec, frames: &[FrameInput]) -> Result<Vec<u8>> {
    if frames.is_empty() {
        return Err(Error::invalid("JXL encoder: no frame to encode"));
    }
    if frames.len() > 1 && spec.animation.is_none() {
        return Err(Error::invalid(
            "JXL encoder: several frames need an AnimationSpec (tick rate)",
        ));
    }
    let mut w = BitWriter::new();
    write_prelude(&mut w, spec)?;
    let last = frames.len() - 1;
    for (i, f) in frames.iter().enumerate() {
        write_frame(&mut w, spec, f, i == last)?;
    }
    Ok(w.into_bytes())
}

// ---- container -----------------------------------------------------------------

fn push_box(out: &mut Vec<u8>, kind: &[u8; 4], payload: &[u8]) -> Result<()> {
    let len = payload.len() + 8;
    if len > u32::MAX as usize {
        return Err(Error::unsupported("JXL encoder: box larger than 4 GiB"));
    }
    out.extend_from_slice(&(len as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    Ok(())
}

/// Wrap a codestream in the 18181-2 box file: signature, `ftyp`, then
/// the metadata boxes (`Exif` with a zero TIFF-header offset, `xml `)
/// ahead of the `jxlc` codestream box.
pub fn wrap_container(
    codestream: &[u8],
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(codestream.len() + 64);
    out.extend_from_slice(&[
        0, 0, 0, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A,
    ]);
    out.extend_from_slice(&[
        0, 0, 0, 0x14, b'f', b't', b'y', b'p', b'j', b'x', b'l', b' ', 0, 0, 0, 0, b'j', b'x',
        b'l', b' ',
    ]);
    if let Some(exif) = exif {
        let mut payload = Vec::with_capacity(exif.len() + 4);
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(exif);
        push_box(&mut out, b"Exif", &payload)?;
    }
    if let Some(xmp) = xmp {
        push_box(&mut out, b"xml ", xmp)?;
    }
    push_box(&mut out, b"jxlc", codestream)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantizer_is_monotone_and_lossless_only_at_100() {
        assert_eq!(quantizer_for_quality(100, 8), 1);
        assert_eq!(quantizer_for_quality(100, 16), 1);
        let mut prev = 1;
        for q in (0..100).rev() {
            let m = quantizer_for_quality(q, 8);
            assert!(m >= 2, "q{q} is lossy");
            assert!(m >= prev, "q{q}: {m} < {prev}");
            prev = m;
        }
        assert_eq!(
            quantizer_for_quality(90, 16),
            quantizer_for_quality(90, 8) << 8
        );
        assert_eq!(quantizer_for_quality(90, 8), 2);
        assert_eq!(quantizer_for_quality(50, 8), 18);
    }

    #[test]
    fn rct_is_inverted_by_the_decoder_arithmetic() {
        let src: Vec<[i32; 3]> = vec![
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [17, 200, 99],
            [65535, 1, 65535],
        ];
        let mut r: Vec<i32> = src.iter().map(|p| p[0]).collect();
        let mut g: Vec<i32> = src.iter().map(|p| p[1]).collect();
        let mut b: Vec<i32> = src.iter().map(|p| p[2]).collect();
        forward_rct(&mut r, &mut g, &mut b);
        for i in 0..src.len() {
            // H.6.3 type 6 inverse.
            let (a, bb, c) = (r[i], g[i], b[i]);
            let tmp = a.wrapping_sub(c >> 1);
            let e = c.wrapping_add(tmp);
            let f = tmp.wrapping_sub(bb >> 1);
            let d = f.wrapping_add(bb);
            assert_eq!([d, e, f], src[i]);
        }
    }

    #[test]
    fn channel_tree_serialises_breadth_first() {
        let t = build_channel_tree(&[1, 2, 4, 5], 1);
        // Root: channel > 1 → left (2..=3), right (0..=1); leaves are
        // numbered in reading order.
        assert_eq!(t.nodes.len(), 7);
        // Root: channel > 1 → left (2..=3), right (0..=1); the next two
        // nodes are the second-level decisions, then four leaves.
        assert!(matches!(
            t.nodes[0],
            TreeNode::Decision {
                property: 0,
                value: 1
            }
        ));
        assert!(matches!(t.nodes[1], TreeNode::Decision { value: 2, .. }));
        assert!(matches!(t.nodes[2], TreeNode::Decision { value: 0, .. }));
        assert!(
            t.nodes[3..]
                .iter()
                .all(|n| matches!(n, TreeNode::Leaf { .. }))
        );
        // Leaves 3..6 are channels 3, 2, 1, 0.
        assert_eq!(t.leaf_ctx, vec![3, 2, 1, 0]);
        let one = build_channel_tree(&[5], 3);
        assert_eq!(one.nodes.len(), 1);
        assert_eq!(one.leaf_ctx, vec![0]);
    }

    #[test]
    fn encoded_icc_header_only_profile_is_the_residual_stream() {
        let profile = vec![7u8; 128];
        let enc = encoded_icc_stream(&profile);
        assert_eq!(enc[0], 128);
        assert_eq!(enc[1], 1); // varint(128) = 0x80 0x01
        assert_eq!(enc[2], 0); // commands_size 0
        assert_eq!(enc.len(), 3 + 128);
        assert_eq!(crate::icc::reconstruct_icc_profile(&enc).unwrap(), profile);
        let mut long = vec![0u8; 300];
        for (i, b) in long.iter_mut().enumerate() {
            *b = (i * 31) as u8;
        }
        let enc = encoded_icc_stream(&long);
        assert_eq!(crate::icc::reconstruct_icc_profile(&enc).unwrap(), long);
    }
}
