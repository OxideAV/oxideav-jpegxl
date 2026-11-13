//! VarDCT **encoder** (Annex I, written side): lossy XYB frames.
//!
//! One frame is produced as the list of TOC sections of F.3:
//!
//! * `LfGlobal` — the LF dequantisation weights, the `Quantizer`
//!   (`global_scale`, `quant_lf`), the default HF block-context map,
//!   the default `LfChannelCorrelation`, and `GlobalModular` with the
//!   one MA tree every Modular sub-bitstream of the frame shares;
//! * one `LfGroup` per 2048×2048 area — the quantised LF image as a
//!   Modular sub-bitstream, then the HF metadata (the chroma-from-luma
//!   factors of every 64×64 tile, the `BlockInfo` list of transform
//!   types and quantisation multipliers, the sharpness map);
//! * `HfGlobal` — default dequantisation matrices, one HF preset, the
//!   natural coefficient orders, the clustered coefficient histograms;
//! * one `PassGroup` per 256×256 group — the quantised HF coefficients
//!   with the I.4 context model, then the group's rectangle of the
//!   extra channels as Modular data.
//!
//! The forward path mirrors what the decoder in this crate applies:
//! sRGB → linear → XYB (the inverse of L.2.2), the inverse of the
//! Gabor-like smoothing of J.2 when the frame signals it, a forward
//! transform per varblock (`enc_dct`), quantisation against the default
//! matrices of I.2.5 scaled by `global_scale` and the per-block
//! `HfMul`, with the chroma-from-luma prediction of I.6 taken from the
//! *dequantised* Y coefficients so the decoder reproduces it exactly.
//! Dequantisation, context numbering and matrix values are taken from
//! the decoder's own functions, never re-derived.
//!
//! Every rate decision is this crate's own design, tuned on the corpus
//! described in the README:
//!
//! * **Cost** — everything is decided on a Lagrangian `D + lambda × R`.
//!   `D` is the squared error of the coefficients in the pixel domain
//!   of XYB (the transforms are orthogonal, so this is the error of
//!   the reconstructed samples), each channel in units of its own
//!   reference step, each coefficient weighted by what the decoder's
//!   Gabor-like smoothing leaves of its basis function, each block by
//!   its quantisation-field value. `R` is a four-parameter estimate of
//!   the coefficient bits (a zero inside the scan, a magnitude of one,
//!   the growth per doubling, the block's `non_zeros` symbol) fitted
//!   to the sizes the entropy coder actually produces.
//! * **Quantisation field** — `HfMul` of a block follows the variance
//!   of its luma (in the cube-root domain of XYB) and of its
//!   neighbours: busy blocks mask their own error and take a coarser
//!   step, calm ones a finer one.
//! * **Transform choice** — each 64×64 tile is covered bottom-up:
//!   every aligned 8×8, 16×16, 32×32 and 64×64 area keeps the cheapest
//!   of its sub-coverings and of the transforms that span it, all
//!   three channels counted; an area whose full-size transform leaves
//!   nothing but its LF is accepted without looking further.
//! * **Quantisation** — each coefficient takes the level with the
//!   lowest cost, and the scan is cut after the coefficient beyond
//!   which coding the tail costs more than dropping it.
//! * **Chroma from luma** — the factors of a tile are the least-squares
//!   fit of its X and B coefficients on the dequantised Y coefficients,
//!   in step units.
//! * **Restoration filters** — the frame signals the Gabor-like
//!   transform and the picture is coded sharpened by its inverse; from
//!   moderate distances up it also signals the edge-preserving filter,
//!   whose strength follows each block's quantisation width.

use crate::bitwriter::{BitWriter, pack_signed};
use crate::chroma_from_luma::kx_kb_lf;
use crate::coeff_order::{natural_coeff_order, order_id_for_transform};
use crate::dct_quant_weights::{materialise_default_dequant_set_for_quantizer, slot_for_transform};
use crate::dct_select::TransformType;
use crate::enc_dct::{LfLink, SmallTransform, fdct, forward_small, lf_link, small_transform};
use crate::entropy_write::{EntropyCodes, ValueStats};
use crate::error::{Error, Result};
use crate::lf_dequant::LfMultipliers;
use crate::lf_global::{HfBlockContext, LfChannelCorrelation, LfChannelDequantization, Quantizer};
use crate::metadata_fdis::OpsinInverseMatrix;
use crate::modular_enc::{Chan, LearnParams, Section, Tree, encode_section, learn_tree};
use crate::options::Effort;
use crate::pass_group_hf::{
    block_context, coefficient_context, non_zeros_context, predicted_non_zeros, prev_for_context,
};

/// VarDCT groups are always 256×256 (F.2: `group_size_shift` is only
/// signalled for Modular frames).
const GROUP_DIM: usize = 256;
/// Blocks per group side.
const GROUP_BLOCKS: usize = GROUP_DIM / 8;
/// LF groups cover 2048×2048 pixels.
const LF_GROUP_DIM: usize = GROUP_DIM * 8;
/// Blocks per chroma-from-luma tile side (64×64 pixels).
const TILE_BLOCKS: usize = 8;
/// Pixels of context kept around a group: one block for the activity
/// of the neighbouring blocks, and room for the sharpening iterations.
const MARGIN: usize = 8;

/// Rate parameters of a frame derived from the distance.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RateParams {
    pub global_scale: u32,
    pub quant_lf: u32,
    /// `HfMul` of a block of reference activity (the quantisation
    /// field varies around it).
    pub hf_mul: i32,
    /// `x_qm_scale` / `b_qm_scale` of the frame header (F.2): the HF
    /// steps of X and B are multiplied by `0.8^(scale − 2)`.
    pub x_qm_scale: u32,
    pub b_qm_scale: u32,
    /// LF dequantisation weights `m_x_lf`, `m_y_lf`, `m_b_lf` (G.1.2).
    pub lf_weights: [f32; 3],
    /// The frame signals the Gabor-like transform (J.2); the encoder
    /// codes the picture sharpened by its inverse.
    pub gab: bool,
    /// Iterations of the edge-preserving filter (J.3), `0..=3`.
    pub epf_iters: u32,
    /// `Sharpness` of every block (index into `epf_sharp_lut`).
    pub sharpness: i32,
    /// Lagrangian multiplier: squared step units (times the pixel
    /// count of a coefficient's basis function) per bit.
    pub lambda: f32,
}

/// Precision of the Y channel's lowest HF band at distance 1, as
/// `global_scale × HfMul / 65536` — the reciprocal of the quantisation
/// step in units of the default matrix entry.
const PRECISION_AT_DISTANCE_1: f64 = 0.40;

/// LF step relative to the HF step of the lowest Y band. Block means
/// are the most visible coefficients (a coarse LF shows as blocking),
/// so the LF is kept finer; settled on the corpus PSNR / size curve.
const LF_STEP_RATIO: f64 = 0.55;

/// LF weights `m_x_lf`, `m_y_lf`, `m_b_lf` (G.1.2; the format's
/// defaults are 1/32, 1/4, 1/2). X and B are kept twice as fine as the
/// defaults: at distance 4 the colour photo gains 0.4 dB for 3 % more
/// bytes (0.14 dB is what those bytes buy through the distance), and
/// block-level colour error is what shows first at low rates.
const LF_WEIGHTS: [f32; 3] = [1.0 / 64.0, 0.25, 0.25];

/// `HfMul` of a block of reference activity: large enough that the
/// quantisation field has resolution on both sides.
const BASE_HF_MUL: f64 = 16.0;

/// `x_qm_scale` / `b_qm_scale`: the HF steps of X and B relative to
/// the default matrices are `0.8^(scale − 2)`. On the corpus the X
/// matrices are fine enough as they are (a finer X costs 0.07 dB at
/// equal size), while B at 0.8 gains 0.15 dB.
const X_QM_SCALE: u32 = 2;
const B_QM_SCALE: u32 = 3;

/// Lagrangian multiplier in squared reference steps per bit, per
/// pixel of a basis function. A uniform quantiser at high rate trades
/// `6 dB` per bit, about `0.115` squared steps; at the rates in use
/// most coefficients sit in the dead zone and the slope that serves
/// best is lower. Swept over 0.035 … 0.22 on the corpus (with the
/// distortion weighted by what the decoder's smoothing leaves of
/// it): PSNR at equal size is flat from 0.05 to 0.12 and SSIM keeps
/// rising towards the low end, 0.06 being the last value before PSNR
/// starts to fall.
const LAMBDA_PER_BIT: f32 = 0.06;

/// The edge-preserving filter is signalled from these distances up
/// (one iteration, then two) with this `Sharpness` for every block.
/// Its strength then follows each block's quantisation width (J.4.3).
/// Below, the quantisation noise it would remove is smaller than the
/// texture it removes with it (SSIM falls at equal size); at the
/// thresholds chosen it is neutral to slightly positive on the corpus
/// and takes the ringing off edges.
const EPF_ONE_ITERATION_FROM: f64 = 2.5;
const EPF_TWO_ITERATIONS_FROM: f64 = 6.0;
const EPF_SHARPNESS: i32 = 4;

impl RateParams {
    /// Parameters for a distance (`> 0`; about 1 is visually lossless,
    /// larger is smaller and worse).
    pub(crate) fn for_distance(distance: f32) -> Self {
        let d = f64::from(distance.clamp(0.01, 64.0));
        let precision = PRECISION_AT_DISTANCE_1 / d;
        let product = 65536.0 * precision;
        let mut hf_mul = BASE_HF_MUL;
        let max_scale = 73_728.0; // 8193 + u(16)
        if product / hf_mul > max_scale {
            hf_mul = (product / max_scale).ceil();
        }
        let global_scale = (product / hf_mul).round().clamp(1.0, max_scale) as u32;
        // LF step of Y is 128 / (global_scale × quant_lf); the lowest
        // HF Y band's is 65536 / (560 × global_scale × hf_mul).
        let quant_lf = (128.0 * 560.0 * hf_mul / (65536.0 * LF_STEP_RATIO))
            .round()
            .clamp(1.0, 65_536.0) as u32;
        Self {
            global_scale,
            quant_lf,
            hf_mul: hf_mul as i32,
            x_qm_scale: X_QM_SCALE,
            b_qm_scale: B_QM_SCALE,
            lf_weights: LF_WEIGHTS,
            gab: true,
            epf_iters: if d >= EPF_TWO_ITERATIONS_FROM {
                2
            } else if d >= EPF_ONE_ITERATION_FROM {
                1
            } else {
                0
            },
            sharpness: EPF_SHARPNESS,
            lambda: LAMBDA_PER_BIT * 64.0,
        }
    }
}

/// The distance for an advisory quality in `0..100` (100 is lossless
/// and never reaches this function): `90 → 1.0`, `50 → 5.0`, `0 → 15`.
pub(crate) fn distance_for_quality(quality: u8) -> f32 {
    let q = f32::from(quality.min(99));
    if q >= 50.0 {
        (100.0 - q) / 10.0
    } else {
        5.0 + (50.0 - q) * 0.2
    }
}

// ---- colour ---------------------------------------------------------------

/// sRGB electro-optical transfer (IEC 61966-2-1) for `v` in `0..=1`.
fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn invert3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let mut out = [[0.0; 3]; 3];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, slot) in row.iter_mut().enumerate() {
            let (r1, r2) = ((c + 1) % 3, (c + 2) % 3);
            let (c1, c2) = ((r + 1) % 3, (r + 2) % 3);
            *slot = (m[r1][c1] * m[r2][c2] - m[r1][c2] * m[r2][c1]) / det;
        }
    }
    out
}

/// Forward XYB: the inverse of L.2.2 with the default
/// `OpsinInverseMatrix` and `intensity_target = 255`.
struct Opsin {
    fwd: [[f32; 3]; 3],
    bias: [f32; 3],
    cbrt_bias: [f32; 3],
    /// Linear value of each integer sample.
    lut: Vec<f32>,
}

impl Opsin {
    fn new(bits: u32) -> Self {
        let oim = OpsinInverseMatrix::default();
        let inv = oim.inv_mat.map(|r| r.map(f64::from));
        let bias = oim.opsin_bias.map(f64::from);
        let max = ((1u64 << bits) - 1) as f64;
        Self {
            fwd: invert3(inv).map(|r| r.map(|v| v as f32)),
            bias: bias.map(|v| v as f32),
            cbrt_bias: bias.map(|v| v.cbrt() as f32),
            lut: (0..1usize << bits)
                .map(|v| srgb_to_linear(v as f64 / max) as f32)
                .collect(),
        }
    }

    #[inline]
    fn xyb(&self, r: i32, g: i32, b: i32) -> [f32; 3] {
        let rgb = [
            self.lut[r as usize],
            self.lut[g as usize],
            self.lut[b as usize],
        ];
        let mut gamma = [0.0f32; 3];
        for (i, slot) in gamma.iter_mut().enumerate() {
            let mix = self.fwd[i][0] * rgb[0] + self.fwd[i][1] * rgb[1] + self.fwd[i][2] * rgb[2];
            *slot = (mix - self.bias[i]).max(0.0).cbrt() + self.cbrt_bias[i];
        }
        [
            (gamma[0] - gamma[1]) * 0.5,
            (gamma[0] + gamma[1]) * 0.5,
            gamma[2],
        ]
    }
}

// ---- transforms -----------------------------------------------------------

/// Everything the coder needs to know about one transform type.
struct Kind {
    t: TransformType,
    /// Varblock size in 8×8 cells.
    cols: usize,
    rows: usize,
    /// Coefficient count and the number of LLF cells.
    size: usize,
    num_blocks: usize,
    /// Natural coefficient order (I.3.2): `order[i]` is the index of
    /// the `i`-th scanned coefficient.
    order: Vec<u32>,
    small: Option<&'static SmallTransform>,
    link: LfLink,
    /// `matrix × qm_scale` per channel: the step of a coefficient is
    /// this over `HfMul`.
    step: [Vec<f32>; 3],
    /// The same in scan order: the reciprocal step numerator and the
    /// distortion weight of one squared step unit (the squared pixel
    /// norm of the basis function times the squared ratio of the
    /// coefficient's step to the channel's reference step).
    scan_scale: [Vec<f32>; 3],
    scan_weight: [Vec<f32>; 3],
    /// Per channel: the largest reciprocal step numerator of the
    /// coded coefficients, and the scale and weight of the first one
    /// (the lowest band).
    max_scale: [f32; 3],
    low_scale: [f32; 3],
    low_weight: [f32; 3],
    block_ctx: [u32; 3],
}

/// What a frame is coded with.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VarDctOptions {
    pub distance: f32,
    pub effort: Effort,
    /// Worker budget; the bytes do not depend on it.
    pub threads: usize,
    /// Testing: choose this transform type wherever it fits.
    pub forced: Option<TransformType>,
}

/// How far the transform search goes.
#[derive(Debug, Clone, Copy)]
struct Search {
    /// Largest transform side, in blocks.
    max_blocks: usize,
    /// Largest long side of the rectangular transforms, in blocks:
    /// above 16×8 they earn nothing measurable on the corpus over the
    /// squares and the smaller rectangles, for a sixth of the search
    /// time.
    rect_blocks: usize,
    /// The 8×8 transforms other than the DCT are tried on a block
    /// whose DCT8×8 scan is at least this long: they pay on detail the
    /// DCT spreads over its whole scan (on the corpus, 8 keeps all of
    /// their gain for a third less search time than 0).
    small_min_scan: usize,
    forced: Option<TransformType>,
}

impl Search {
    fn new(opts: &VarDctOptions) -> Self {
        let (max_blocks, rect_blocks, small_min_scan) = match opts.effort {
            Effort::Fast => (1, 1, usize::MAX),
            Effort::Thorough => (8, 8, 0),
            _ => (4, 2, 8),
        };
        Self {
            max_blocks,
            rect_blocks,
            small_min_scan,
            forced: opts.forced,
        }
    }
}

/// The transforms the encoder chooses from: DCT8×8 alone at the
/// fastest effort, everything up to DCT64×64 otherwise (the search
/// depth bounds the sizes actually tried).
fn transforms_for(opts: &VarDctOptions) -> Vec<TransformType> {
    use TransformType as T;
    let mut v = vec![T::Dct8x8];
    if matches!(opts.effort, Effort::Fast) {
        if let Some(t) = opts.forced
            && t != T::Dct8x8
        {
            v.push(t);
        }
        return v;
    }
    v.extend([
        T::Dct16x16,
        T::Dct32x32,
        T::Dct16x8,
        T::Dct8x16,
        T::Dct32x8,
        T::Dct8x32,
        T::Dct32x16,
        T::Dct16x32,
        T::Dct4x4,
        T::Dct4x8,
        T::Dct8x4,
        T::Dct2x2,
        T::Hornuss,
        T::Afv0,
        T::Afv1,
        T::Afv2,
        T::Afv3,
        T::Dct64x64,
        T::Dct64x32,
        T::Dct32x64,
    ]);
    v
}

/// Bits of a coefficient in the rate estimate: a zero inside the
/// coded scan, a magnitude of one, and the growth per doubling.
const BITS_ZERO: f32 = 0.41;
const BITS_ONE: f32 = 3.5;
const BITS_PER_DOUBLING: f32 = 1.21;
/// Bits of the `non_zeros` symbol of a block with coefficients, and
/// of an empty one.
const BITS_BLOCK: f32 = 2.9;
const BITS_EMPTY_BLOCK: f32 = 0.28;

/// A coefficient under this many steps is never worth a level: the
/// gain of coding it as one is below any multiplier in use.
const LEVEL_FLOOR: f32 = 0.3;

/// Estimated bits of a non-zero level `q >= 1`.
#[inline]
fn level_bits(q: i32) -> f32 {
    // `log2` by its integer part and a linear fraction: within 0.09
    // of the true value, far inside the estimate's own error.
    let l = 31 - (q as u32).leading_zeros();
    let frac = (q as f32) / (1u32 << l) as f32 - 1.0;
    BITS_ONE + BITS_PER_DOUBLING * (l as f32 + frac)
}

// ---- the coded frame ------------------------------------------------------

/// A picture for the VarDCT coder: integer sRGB samples.
pub(crate) struct VarDctInput<'a> {
    pub width: usize,
    pub height: usize,
    pub bits: u32,
    /// One (grey) or three (R, G, B) channels.
    pub colour: &'a [Vec<i32>],
    /// Extra channels coded losslessly (alpha).
    pub extra: &'a [Vec<i32>],
}

/// `(context, value)` symbols of one entropy-coded stream.
type Symbols = Vec<(u32, u32)>;

/// Everything the section writers need.
struct Coded {
    rate: RateParams,
    tree: Tree,
    tree_codes: EntropyCodes,
    leaf_codes: EntropyCodes,
    hf_codes: EntropyCodes,
    /// Symbols of the extra channels carried by `GlobalModular`
    /// (`None` without extra channels; empty when they live in groups).
    global_extra: Option<Vec<(u32, u32)>>,
    /// Per LF group: LF-coefficient symbols, `nb_blocks`, HF-metadata
    /// symbols, and the bit width of the `nb_blocks - 1` field.
    lf_groups: Vec<(Symbols, u32, u32, Symbols)>,
    /// Per group: HF symbols and the extra channels' Modular symbols.
    groups: Vec<(Symbols, Symbols)>,
    num_groups: usize,
}

fn ceil_log2(n: u64) -> u32 {
    if n <= 1 {
        0
    } else {
        64 - (n - 1).leading_zeros()
    }
}

/// What one group contributes.
struct GroupOut {
    /// Quantised LF samples (X, Y, B) in raster order of the group's
    /// blocks.
    lf: [Vec<i32>; 3],
    /// Per block: the transform type (Table I.2 value) at the top-left
    /// block of a varblock, `CONTINUATION` elsewhere.
    kind: Vec<u8>,
    /// Per block: `HfMul` of the varblock covering it.
    hf_mul: Vec<i32>,
    /// Per 64×64 tile of the group: `XFromY`, `BFromY`.
    cfl: Vec<[i32; 2]>,
    syms: Symbols,
    stats: ValueStats,
}

/// Marks a block that is not the top-left one of its varblock.
const CONTINUATION: u8 = 255;

/// Per-frame constants every group worker shares.
struct FrameCtx<'a> {
    input: &'a VarDctInput<'a>,
    opsin: Opsin,
    rate: RateParams,
    kinds: Vec<Kind>,
    /// Index into `kinds` of each transform type (Table I.2 value).
    kind_of: [u8; 27],
    lf_mul: LfMultipliers,
    kx_lf: f32,
    kb_lf: f32,
    base_x: f32,
    base_b: f32,
    colour_factor: f32,
    /// `quant_bias` of each channel and `quant_bias_numerator` (I.5.3).
    bias_one: [f32; 3],
    bias_numerator: f32,
    nb_block_ctx: u32,
    bw: usize,
    bh: usize,
    search: Search,
    /// Gabor-like kernel weights (centre, edge, corner), normalised.
    gab_kernel: [f32; 3],
}

impl FrameCtx<'_> {
    fn kind(&self, t: TransformType) -> Option<&Kind> {
        let i = self.kind_of[t.index() as usize];
        (i != CONTINUATION).then(|| &self.kinds[i as usize])
    }
}

/// Mirror a coordinate into `0..n` (edge sample repeated).
#[inline]
fn mirror(mut i: i64, n: usize) -> usize {
    let n = n as i64;
    loop {
        if i < 0 {
            i = -i - 1;
        } else if i >= n {
            i = 2 * n - 1 - i;
        } else {
            return i as usize;
        }
    }
}

/// Reusable buffers of a group worker.
#[derive(Default)]
struct Scratch {
    coef: Vec<f32>,
    tmp: Vec<f32>,
    quant: Vec<i32>,
    q: QuantScratch,
}

/// Per-coefficient work arrays of [`quantise`].
#[derive(Default)]
struct QuantScratch {
    /// Cost of each scan position as coded, and as dropped.
    keep: Vec<f32>,
    drop: Vec<f32>,
    /// The coefficient in step units.
    scaled: Vec<f32>,
}

/// Forward transform of the varblock of `kind` whose top-left pixel is
/// at `origin` in `plane` (row length `stride`).
fn forward(
    kind: &Kind,
    plane: &[f32],
    stride: usize,
    origin: usize,
    out: &mut [f32],
    s: &mut Vec<f32>,
) {
    match kind.small {
        Some(st) => forward_small(st, &plane[origin..], stride, out),
        None => fdct(
            &plane[origin..],
            stride,
            kind.rows * 8,
            kind.cols * 8,
            out,
            s,
        ),
    }
}

/// Quantise one channel of a varblock: `coef` are the coefficients to
/// code (the chroma-from-luma prediction already subtracted), `out`
/// receives the levels (zero at the LLF cells). Returns the Lagrangian
/// cost `D + lambda × R` of the choice and the length of the coded
/// scan (zero for an empty block).
///
/// Each coefficient takes the level minimising its own cost among the
/// nearest level and its neighbours; then the scan is cut after the
/// coefficient at which the cost of the kept part plus the energy of
/// the dropped tail is smallest (trailing zeros are free: the decoder
/// stops when `non_zeros` reaches zero).
#[allow(clippy::too_many_arguments)]
fn quantise(
    ctx: &FrameCtx<'_>,
    kind: &Kind,
    c: usize,
    hf_mul: f32,
    mask: f32,
    coef: &[f32],
    out: &mut [i32],
    s: &mut QuantScratch,
) -> (f32, usize) {
    let lambda = ctx.rate.lambda;
    let (r1, numer) = (ctx.bias_one[c], ctx.bias_numerator);
    // Distortion in units of the reference step at the base `HfMul`.
    // `mask` is the weight of this block's error: the square of its
    // quantisation-field value, so a block coded at its field value
    // sees the same trade-off in step units as any other.
    let dscale = (ctx.rate.hf_mul as f32 / hf_mul) * (ctx.rate.hf_mul as f32 / hf_mul) * mask;
    let (nb, n) = (kind.num_blocks, kind.size);
    if s.keep.len() < n {
        s.keep.resize(n, 0.0);
        s.drop.resize(n, 0.0);
        s.scaled.resize(n, 0.0);
    }
    let (keep, drop, scaled) = (&mut s.keep[..n], &mut s.drop[..n], &mut s.scaled[..n]);
    let order = &kind.order[..n];
    out[..n].fill(0);
    // Every coefficient in step units, and the cost of dropping it.
    // `reach`: one past the last coefficient that can take a level.
    let mut reach = nb;
    let mut energy = 0.0f32;
    for (i, (((t, d), &pos), (&scale, &weight))) in scaled[nb..]
        .iter_mut()
        .zip(&mut drop[nb..])
        .zip(&order[nb..])
        .zip(
            kind.scan_scale[c][nb..]
                .iter()
                .zip(&kind.scan_weight[c][nb..]),
        )
        .enumerate()
    {
        let v = coef[pos as usize] * scale * hf_mul;
        *t = v;
        let e = weight * dscale * v * v;
        *d = e;
        energy += e;
        if v.abs() >= LEVEL_FLOOR {
            reach = nb + i + 1;
        }
    }
    let zero_bits = lambda * BITS_ZERO;
    let mut last = nb; // one past the last non-zero scan index
    for i in nb..reach {
        let t = scaled[i];
        let a = t.abs();
        let zero_cost = drop[i];
        if a < LEVEL_FLOOR {
            keep[i] = zero_cost + zero_bits;
            continue;
        }
        let g = kind.scan_weight[c][i] * dscale;
        let n0 = (a + 0.5) as i32;
        let mut best = (zero_cost + zero_bits, 0i32);
        for q in [n0 - 1, n0, n0 + 1] {
            if q < 1 {
                continue;
            }
            let rec = if q == 1 {
                r1
            } else {
                q as f32 - numer / q as f32
            };
            let cost = g * (a - rec) * (a - rec) + lambda * level_bits(q);
            if cost < best.0 {
                best = (cost, q);
            }
        }
        keep[i] = best.0;
        if best.1 != 0 {
            out[order[i] as usize] = if t < 0.0 { -best.1 } else { best.1 };
            last = i + 1;
        }
    }
    // Cut point: cost(p) = sum keep[..p] + sum drop[p..], over p = one
    // past a non-zero (or the start).
    let mut tail: f32 = (energy - drop[nb..last].iter().sum::<f32>()).max(0.0);
    let mut head: f32 = keep[nb..last].iter().sum();
    let first_bits = if last == nb {
        BITS_EMPTY_BLOCK
    } else {
        BITS_BLOCK
    };
    let mut best = (head + tail + lambda * first_bits, last);
    let mut p = last;
    while p > nb {
        p -= 1;
        head -= keep[p];
        tail += drop[p];
        // A valid cut ends on a non-zero (or is empty).
        if p == nb || out[order[p - 1] as usize] != 0 {
            let block_bits = if p == nb {
                BITS_EMPTY_BLOCK
            } else {
                BITS_BLOCK
            };
            let cost = head + tail + lambda * block_bits;
            if cost < best.0 {
                best = (cost, p);
            }
        }
    }
    for &pos in &order[best.1..last] {
        out[pos as usize] = 0;
    }
    (best.0, best.1 - nb)
}

/// A covering of an area by varblocks, with its cost.
#[derive(Clone, Default)]
struct Lay {
    cost: f32,
    /// `(x, y, kind index, HfMul)`, positions in blocks relative to
    /// the group.
    blocks: Vec<(u8, u8, u8, u8)>,
}

impl Lay {
    fn single(cost: f32, x: usize, y: usize, kind: u8, hf_mul: i32) -> Self {
        Self {
            cost,
            blocks: vec![(x as u8, y as u8, kind, hf_mul as u8)],
        }
    }

    fn join(parts: &[&Lay]) -> Self {
        let mut out = Lay::default();
        for p in parts {
            out.cost += p.cost;
            out.blocks.extend_from_slice(&p.blocks);
        }
        out
    }

    fn take_cheaper(&mut self, other: Lay) {
        if other.cost < self.cost {
            *self = other;
        }
    }
}

/// The best coverings of a 16×16 area: overall, and of each half when
/// split into columns (`tall`) or rows (`wide`).
#[derive(Clone, Default)]
struct Node16 {
    best: Lay,
    tall: [Lay; 2],
    wide: [Lay; 2],
    full: bool,
}

/// One group's view of the picture and its search state.
struct GroupWork<'a> {
    ctx: &'a FrameCtx<'a>,
    /// Planes to code (sharpened when the frame uses the Gabor-like
    /// transform), with `MARGIN` pixels around the group's blocks.
    planes: [Vec<f32>; 3],
    stride: usize,
    gw: usize,
    gh: usize,
    /// Quantisation-field multiplier of each block.
    field: Vec<f32>,
    /// The chroma planes as the search sees them — X and B minus the
    /// luma under the frame's base correlation — and, per block, the
    /// sum and the sum of squares of their samples.
    chroma: [Vec<f32>; 2],
    chroma_stats: [Vec<(f32, f32)>; 2],
    /// Coded scan length of the last evaluated candidate (luma).
    last_scan: usize,
    s: Scratch,
}

impl GroupWork<'_> {
    #[inline]
    fn origin(&self, x: usize, y: usize) -> usize {
        (MARGIN + y * 8) * self.stride + MARGIN + x * 8
    }

    /// The quantisation field averaged over the blocks of a varblock.
    fn field_at(&self, x: usize, y: usize, cols: usize, rows: usize) -> f32 {
        let mut sum = 0.0f32;
        for j in 0..rows {
            for i in 0..cols {
                sum += self.field[(y + j) * self.gw + x + i];
            }
        }
        sum / (cols * rows) as f32
    }

    /// Luma cost of coding the varblock of `t` at block `(x, y)`, with
    /// its `HfMul` and the weight of its error. `None` when the
    /// transform is not in use or does not fit.
    fn eval_luma(&mut self, t: TransformType, x: usize, y: usize) -> Option<(f32, i32, f32)> {
        let kind = self.ctx.kind(t)?;
        if x + kind.cols > self.gw || y + kind.rows > self.gh {
            return None;
        }
        let f = self.field_at(x, y, kind.cols, kind.rows);
        let hf_mul = ((self.ctx.rate.hf_mul as f32 * f).round() as i32).clamp(1, 255);
        let origin = self.origin(x, y);
        self.s.coef.resize(kind.size, 0.0);
        self.s.quant.resize(kind.size, 0);
        forward(
            kind,
            &self.planes[1],
            self.stride,
            origin,
            &mut self.s.coef,
            &mut self.s.tmp,
        );
        let (cost, scan) = quantise(
            self.ctx,
            kind,
            1,
            hf_mul as f32,
            f * f,
            &self.s.coef,
            &mut self.s.quant,
            &mut self.s.q,
        );
        self.last_scan = scan;
        Some((cost, hf_mul, f * f))
    }

    /// The candidate of `t` at `(x, y)` given its luma evaluation: the
    /// two chroma channels are added, coded against the luma under the
    /// base correlation (the fitted factors of the tile are not known
    /// yet).
    fn with_chroma(
        &mut self,
        t: TransformType,
        x: usize,
        y: usize,
        (luma, hf_mul, mask): (f32, i32, f32),
    ) -> Option<Lay> {
        let kind = self.ctx.kind(t)?;
        let mut cost = luma;
        for (k, c) in [(0usize, 0usize), (1, 2)] {
            cost += self.chroma_cost(kind, k, c, x, y, hf_mul as f32, mask);
        }
        // A forced type (tests) wins wherever it fits.
        if self.ctx.search.forced == Some(t) {
            cost *= 1e-4;
        }
        Some(Lay::single(
            cost,
            x,
            y,
            self.ctx.kind_of[t.index() as usize],
            hf_mul,
        ))
    }

    /// Cost of coding the varblock of `t` at block `(x, y)`, all three
    /// channels.
    fn eval(&mut self, t: TransformType, x: usize, y: usize) -> Option<Lay> {
        let luma = self.eval_luma(t, x, y)?;
        let scan = self.last_scan;
        let lay = self.with_chroma(t, x, y, luma);
        self.last_scan = scan;
        lay
    }

    /// Cost of chroma channel `c` (search plane `k`) of a candidate.
    /// When the block's samples deviate from their mean by less than
    /// any coefficient needs to take a level, the channel is empty and
    /// its cost is its energy — no transform is run.
    #[allow(clippy::too_many_arguments)]
    fn chroma_cost(
        &mut self,
        kind: &Kind,
        k: usize,
        c: usize,
        x: usize,
        y: usize,
        hf_mul: f32,
        mask: f32,
    ) -> f32 {
        let (mut sum, mut squares) = (0.0f32, 0.0f32);
        for j in 0..kind.rows {
            for i in 0..kind.cols {
                let (a, b) = self.chroma_stats[k][(y + j) * self.gw + x + i];
                sum += a;
                squares += b;
            }
        }
        let pixels = (kind.cols * kind.rows * 64) as f32;
        let mean = sum / pixels;
        let variance = (squares / pixels - mean * mean).max(0.0);
        // Parseval: the squared coefficients beyond the mean sum to
        // the variance, so none exceeds its square root.
        if variance.sqrt() * hf_mul * kind.max_scale[c] < LEVEL_FLOOR {
            let base = self.ctx.rate.hf_mul as f32 / hf_mul;
            let t = hf_mul * kind.low_scale[c];
            return kind.low_weight[c] * base * base * mask * variance * t * t
                + self.ctx.rate.lambda * BITS_EMPTY_BLOCK;
        }
        let origin = self.origin(x, y);
        forward(
            kind,
            &self.chroma[k],
            self.stride,
            origin,
            &mut self.s.coef,
            &mut self.s.tmp,
        );
        quantise(
            self.ctx,
            kind,
            c,
            hf_mul,
            mask,
            &self.s.coef,
            &mut self.s.quant,
            &mut self.s.q,
        )
        .0
    }

    fn best8(&mut self, x: usize, y: usize) -> Lay {
        use TransformType as T;
        let mut best = self.eval(T::Dct8x8, x, y).unwrap_or_default();
        // The other 8×8 transforms pay on blocks with detail the DCT
        // spreads over its whole scan; a short scan is left alone.
        if self.last_scan < self.ctx.search.small_min_scan {
            return best;
        }
        // The cheapest of them on the luma is the one compared, with
        // its chroma, against the DCT.
        let mut pick: Option<(T, (f32, i32, f32))> = None;
        for t in [
            T::Dct4x4,
            T::Dct4x8,
            T::Dct8x4,
            T::Dct2x2,
            T::Hornuss,
            T::Afv0,
            T::Afv1,
            T::Afv2,
            T::Afv3,
        ] {
            let Some(luma) = self.eval_luma(t, x, y) else {
                continue;
            };
            if self.ctx.search.forced == Some(t) {
                pick = Some((t, luma));
                break;
            }
            if pick.is_none_or(|(_, p)| luma.0 < p.0) {
                pick = Some((t, luma));
            }
        }
        if let Some((t, luma)) = pick
            && let Some(lay) = self.with_chroma(t, x, y, luma)
        {
            best.take_cheaper(lay);
        }
        best
    }

    /// Covering of the blocks of the area that lie inside the group,
    /// for an area that is not entirely inside: its quadrants.
    fn partial(&mut self, x: usize, y: usize, blocks: usize) -> Lay {
        let half = blocks / 2;
        let mut out = Lay::default();
        for (dx, dy) in [(0, 0), (half, 0), (0, half), (half, half)] {
            let (qx, qy) = (x + dx, y + dy);
            if qx >= self.gw || qy >= self.gh {
                continue;
            }
            let part = match half {
                1 => self.best8(qx, qy),
                2 => self.best16(qx, qy).best,
                _ => self.best32(qx, qy),
            };
            out = Lay::join(&[&out, &part]);
        }
        out
    }

    fn best16(&mut self, x: usize, y: usize) -> Node16 {
        use TransformType as T;
        if x + 2 > self.gw || y + 2 > self.gh || self.ctx.search.max_blocks < 2 {
            return Node16 {
                best: self.partial(x, y, 2),
                ..Node16::default()
            };
        }
        // An area its full transform codes with nothing but LF is
        // flat: no covering by smaller blocks is cheaper.
        let whole = self.eval(T::Dct16x16, x, y);
        if let Some(lay) = &whole
            && self.last_scan == 0
        {
            return Node16 {
                best: lay.clone(),
                ..Node16::default()
            };
        }
        let q = [
            self.best8(x, y),
            self.best8(x + 1, y),
            self.best8(x, y + 1),
            self.best8(x + 1, y + 1),
        ];
        let mut tall = [Lay::join(&[&q[0], &q[2]]), Lay::join(&[&q[1], &q[3]])];
        let mut wide = [Lay::join(&[&q[0], &q[1]]), Lay::join(&[&q[2], &q[3]])];
        for i in 0..2 {
            if let Some(lay) = self.eval(T::Dct16x8, x + i, y) {
                tall[i].take_cheaper(lay);
            }
            if let Some(lay) = self.eval(T::Dct8x16, x, y + i) {
                wide[i].take_cheaper(lay);
            }
        }
        let mut best = Lay::join(&[&tall[0], &tall[1]]);
        best.take_cheaper(Lay::join(&[&wide[0], &wide[1]]));
        if let Some(lay) = whole {
            best.take_cheaper(lay);
        }
        Node16 {
            best,
            tall,
            wide,
            full: true,
        }
    }

    fn best32(&mut self, x: usize, y: usize) -> Lay {
        use TransformType as T;
        if x + 4 > self.gw || y + 4 > self.gh || self.ctx.search.max_blocks < 4 {
            return self.partial(x, y, 4);
        }
        let whole = self.eval(T::Dct32x32, x, y);
        if let Some(lay) = &whole
            && self.last_scan == 0
        {
            return lay.clone();
        }
        let q = [
            self.best16(x, y),
            self.best16(x + 2, y),
            self.best16(x, y + 2),
            self.best16(x + 2, y + 2),
        ];
        let mut best = Lay::join(&[&q[0].best, &q[1].best, &q[2].best, &q[3].best]);
        if self.ctx.search.rect_blocks >= 4 {
            // Halves.
            let mut left = Lay::join(&[&q[0].best, &q[2].best]);
            let mut right = Lay::join(&[&q[1].best, &q[3].best]);
            if let Some(lay) = self.eval(T::Dct32x16, x, y) {
                left.take_cheaper(lay);
            }
            if let Some(lay) = self.eval(T::Dct32x16, x + 2, y) {
                right.take_cheaper(lay);
            }
            best.take_cheaper(Lay::join(&[&left, &right]));
            let mut top = Lay::join(&[&q[0].best, &q[1].best]);
            let mut bottom = Lay::join(&[&q[2].best, &q[3].best]);
            if let Some(lay) = self.eval(T::Dct16x32, x, y) {
                top.take_cheaper(lay);
            }
            if let Some(lay) = self.eval(T::Dct16x32, x, y + 2) {
                bottom.take_cheaper(lay);
            }
            best.take_cheaper(Lay::join(&[&top, &bottom]));
            // Strips: four columns or four rows, each one long transform
            // or the best 8-wide pieces of its two halves.
            if q.iter().all(|n| n.full) {
                let mut columns = Lay::default();
                let mut rows = Lay::default();
                for i in 0..4 {
                    let (qa, qb, side) = (&q[i / 2], &q[2 + i / 2], i % 2);
                    let mut column = Lay::join(&[&qa.tall[side], &qb.tall[side]]);
                    if let Some(lay) = self.eval(T::Dct32x8, x + i, y) {
                        column.take_cheaper(lay);
                    }
                    columns = Lay::join(&[&columns, &column]);
                    let (qa, qb) = (&q[2 * (i / 2)], &q[2 * (i / 2) + 1]);
                    let mut row = Lay::join(&[&qa.wide[side], &qb.wide[side]]);
                    if let Some(lay) = self.eval(T::Dct8x32, x, y + i) {
                        row.take_cheaper(lay);
                    }
                    rows = Lay::join(&[&rows, &row]);
                }
                best.take_cheaper(columns);
                best.take_cheaper(rows);
            }
        }
        if let Some(lay) = whole {
            best.take_cheaper(lay);
        }
        best
    }

    fn best64(&mut self, x: usize, y: usize) -> Lay {
        use TransformType as T;
        if x + 8 > self.gw || y + 8 > self.gh || self.ctx.search.max_blocks < 8 {
            return self.partial(x, y, 8);
        }
        let whole = self.eval(T::Dct64x64, x, y);
        if let Some(lay) = &whole
            && self.last_scan == 0
        {
            return lay.clone();
        }
        let q = [
            self.best32(x, y),
            self.best32(x + 4, y),
            self.best32(x, y + 4),
            self.best32(x + 4, y + 4),
        ];
        let mut best = Lay::join(&[&q[0], &q[1], &q[2], &q[3]]);
        if self.ctx.search.rect_blocks >= 8 {
            let mut left = Lay::join(&[&q[0], &q[2]]);
            let mut right = Lay::join(&[&q[1], &q[3]]);
            if let Some(lay) = self.eval(T::Dct64x32, x, y) {
                left.take_cheaper(lay);
            }
            if let Some(lay) = self.eval(T::Dct64x32, x + 4, y) {
                right.take_cheaper(lay);
            }
            best.take_cheaper(Lay::join(&[&left, &right]));
            let mut top = Lay::join(&[&q[0], &q[1]]);
            let mut bottom = Lay::join(&[&q[2], &q[3]]);
            if let Some(lay) = self.eval(T::Dct32x64, x, y) {
                top.take_cheaper(lay);
            }
            if let Some(lay) = self.eval(T::Dct32x64, x, y + 4) {
                bottom.take_cheaper(lay);
            }
            best.take_cheaper(Lay::join(&[&top, &bottom]));
        }
        if let Some(lay) = whole {
            best.take_cheaper(lay);
        }
        best
    }
}

/// A varblock of a group with its quantised coefficients.
struct Placed {
    x: usize,
    y: usize,
    kind: u8,
    hf_mul: i32,
    /// Weight of its error (the squared field value).
    mask: f32,
    /// Offset of its coefficients in the group's per-channel arenas.
    off: usize,
}

/// The XYB planes of a group with `MARGIN` pixels of context, the
/// picture mirrored beyond its edges.
fn group_planes(ctx: &FrameCtx<'_>, bx0: usize, by0: usize, gw: usize, gh: usize) -> [Vec<f32>; 3] {
    let (width, height) = (ctx.input.width, ctx.input.height);
    let (pw, ph) = (gw * 8 + 2 * MARGIN, gh * 8 + 2 * MARGIN);
    let grey = ctx.input.colour.len() == 1;
    let xs: Vec<usize> = (0..pw)
        .map(|x| mirror((bx0 * 8 + x) as i64 - MARGIN as i64, width))
        .collect();
    let mut planes: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; pw * ph]);
    for y in 0..ph {
        let sy = mirror((by0 * 8 + y) as i64 - MARGIN as i64, height);
        let row = sy * width;
        let o = y * pw;
        for (x, &sx) in xs.iter().enumerate() {
            let i = row + sx;
            let v = if grey {
                let g = ctx.input.colour[0][i];
                ctx.opsin.xyb(g, g, g)
            } else {
                ctx.opsin.xyb(
                    ctx.input.colour[0][i],
                    ctx.input.colour[1][i],
                    ctx.input.colour[2][i],
                )
            };
            planes[0][o + x] = v[0];
            planes[1][o + x] = v[1];
            planes[2][o + x] = v[2];
        }
    }
    planes
}

/// Iterations of the fixed-point inversion of the Gabor-like kernel.
const SHARPEN_ITERATIONS: usize = 4;

/// The picture whose Gabor-like smoothing (J.2) is `plane`: the
/// decoder convolves with a 3×3 kernel `K` whose transfer function
/// lies in `(0.44, 1]`, so the fixed-point iteration
/// `x ← x + (plane − K ∗ x)` converges to the inverse; each iteration
/// consumes one pixel of the margin.
fn sharpen(plane: &[f32], pw: usize, ph: usize, kernel: [f32; 3]) -> Vec<f32> {
    let [centre, edge, corner] = kernel;
    let mut cur = plane.to_vec();
    let mut next = plane.to_vec();
    for it in 0..SHARPEN_ITERATIONS {
        let m = it + 1;
        for y in m..ph - m {
            let o = y * pw;
            for x in m..pw - m {
                let i = o + x;
                let blurred = centre * cur[i]
                    + edge * (cur[i - 1] + cur[i + 1] + cur[i - pw] + cur[i + pw])
                    + corner
                        * (cur[i - pw - 1] + cur[i - pw + 1] + cur[i + pw - 1] + cur[i + pw + 1]);
                next[i] = cur[i] + (plane[i] - blurred);
            }
        }
        std::mem::swap(&mut cur, &mut next);
    }
    cur
}

/// The quantisation field: a block's multiplier is
/// `((reference + floor) / (activity + floor)) ^ exponent`, the
/// activity being the variance of the block's luma (in the
/// cube-root domain of XYB, so already relative to brightness), half
/// its own and half the smallest of its 3×3 neighbourhood — a busy
/// block next to a flat one must not spill its coarser step into
/// where it shows. Swept on the corpus: an exponent of 0.2 moves
/// about 0.2 dB of PSNR into 0.3 – 0.5 dB of SSIM at equal size
/// (0.5, the textbook equaliser of SSIM, costs three times the PSNR
/// for little more); the floor is where the activity stops mattering
/// (about four 8-bit codes of standard deviation) and doubles as the
/// reference, the level that keeps the frame's base step.
const AQ_EXPONENT: f32 = 0.2;
const AQ_FLOOR: f32 = 1.0e-3;
const AQ_REFERENCE: f32 = 1.0e-3;
const AQ_OWN_SHARE: f32 = 0.5;
/// Bounds of the field.
const AQ_RANGE: (f32, f32) = (0.4, 2.5);

/// Quantisation-field multiplier of every block of a group from the
/// luma plane (with its margin): blocks busier than the reference take
/// a coarser step, calmer ones a finer step.
fn quant_field(y_plane: &[f32], pw: usize, gw: usize, gh: usize) -> Vec<f32> {
    // Variance of the blocks -1..=gw × -1..=gh.
    let (aw, ah) = (gw + 2, gh + 2);
    let mut act = vec![0.0f32; aw * ah];
    for by in 0..ah {
        for bx in 0..aw {
            let (mut sum, mut sq) = (0.0f32, 0.0f32);
            for r in 0..8 {
                let o = (by * 8 + r) * pw + bx * 8;
                for &v in &y_plane[o..o + 8] {
                    sum += v;
                    sq += v * v;
                }
            }
            let mean = sum / 64.0;
            act[by * aw + bx] = (sq / 64.0 - mean * mean).max(0.0);
        }
    }
    let mut field = vec![1.0f32; gw * gh];
    for by in 0..gh {
        for bx in 0..gw {
            let own = act[(by + 1) * aw + bx + 1];
            let mut low = own;
            for j in 0..3 {
                for i in 0..3 {
                    low = low.min(act[(by + j) * aw + bx + i]);
                }
            }
            let a = AQ_OWN_SHARE * own + (1.0 - AQ_OWN_SHARE) * low;
            let f = ((AQ_REFERENCE + AQ_FLOOR) / (a + AQ_FLOOR)).powf(AQ_EXPONENT);
            field[by * gw + bx] = f.clamp(AQ_RANGE.0, AQ_RANGE.1);
        }
    }
    field
}

/// Colour-convert, transform, quantise and tokenise one group.
fn code_group(ctx: &FrameCtx<'_>, gx: usize, gy: usize) -> Result<GroupOut> {
    let bx0 = gx * GROUP_BLOCKS;
    let by0 = gy * GROUP_BLOCKS;
    let gw = GROUP_BLOCKS.min(ctx.bw - bx0);
    let gh = GROUP_BLOCKS.min(ctx.bh - by0);
    let (pw, ph) = (gw * 8 + 2 * MARGIN, gh * 8 + 2 * MARGIN);
    let source = group_planes(ctx, bx0, by0, gw, gh);
    let field = quant_field(&source[1], pw, gw, gh);
    let planes = if ctx.rate.gab {
        let [x, y, b] = &source;
        [
            sharpen(x, pw, ph, ctx.gab_kernel),
            sharpen(y, pw, ph, ctx.gab_kernel),
            sharpen(b, pw, ph, ctx.gab_kernel),
        ]
    } else {
        source
    };
    // The chroma the search works on, and its per-block moments.
    let chroma: [Vec<f32>; 2] = [
        planes[0]
            .iter()
            .zip(&planes[1])
            .map(|(&x, &y)| x - ctx.base_x * y)
            .collect(),
        planes[2]
            .iter()
            .zip(&planes[1])
            .map(|(&b, &y)| b - ctx.base_b * y)
            .collect(),
    ];
    let chroma_stats: [Vec<(f32, f32)>; 2] = std::array::from_fn(|k| {
        let mut stats = Vec::with_capacity(gw * gh);
        for by in 0..gh {
            for bx in 0..gw {
                let (mut sum, mut squares) = (0.0f32, 0.0f32);
                for r in 0..8 {
                    let o = (MARGIN + by * 8 + r) * pw + MARGIN + bx * 8;
                    for &v in &chroma[k][o..o + 8] {
                        sum += v;
                        squares += v * v;
                    }
                }
                stats.push((sum, squares));
            }
        }
        stats
    });
    let mut work = GroupWork {
        ctx,
        planes,
        stride: pw,
        gw,
        gh,
        field,
        chroma,
        chroma_stats,
        last_scan: 0,
        s: Scratch::default(),
    };

    let n_cells = gw * gh;
    let mut lf: [Vec<i32>; 3] = std::array::from_fn(|_| vec![0i32; n_cells]);
    let mut kind_grid = vec![CONTINUATION; n_cells];
    let mut hf_grid = vec![0i32; n_cells];
    let (tw, th) = (gw.div_ceil(TILE_BLOCKS), gh.div_ceil(TILE_BLOCKS));
    let mut cfl = vec![[0i32; 2]; tw * th];
    let mut placed: Vec<Placed> = Vec::new();
    let mut arena: [Vec<i32>; 3] = std::array::from_fn(|_| Vec::new());
    // Per tile: the coefficients of its varblocks and the dequantised
    // luma they are predicted from.
    let mut coefs: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
    let mut luma: Vec<f32> = Vec::new();
    let mut residual: Vec<f32> = Vec::new();
    let mut lf_samples: Vec<f32> = Vec::new();
    for ty in 0..th {
        for tx in 0..tw {
            let lay = work.best64(tx * TILE_BLOCKS, ty * TILE_BLOCKS);
            let first = placed.len();
            for c in &mut coefs {
                c.clear();
            }
            luma.clear();
            // Luma, and the sums of the chroma-from-luma fit.
            let (mut sxy, mut sx_yy, mut sby, mut sb_yy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for &(x, y, k, hf_mul) in &lay.blocks {
                let kind = &ctx.kinds[k as usize];
                let (x, y) = (x as usize, y as usize);
                let hf_mul = i32::from(hf_mul);
                let f = work.field_at(x, y, kind.cols, kind.rows);
                let mask = f * f;
                let off = arena[0].len();
                for a in &mut arena {
                    a.resize(off + kind.size, 0);
                }
                let toff = luma.len();
                let origin = work.origin(x, y);
                for (c, dst) in coefs.iter_mut().enumerate() {
                    dst.resize(toff + kind.size, 0.0);
                    forward(
                        kind,
                        &work.planes[c],
                        work.stride,
                        origin,
                        &mut dst[toff..],
                        &mut work.s.tmp,
                    );
                }
                quantise(
                    ctx,
                    kind,
                    1,
                    hf_mul as f32,
                    mask,
                    &coefs[1][toff..],
                    &mut arena[1][off..],
                    &mut work.s.q,
                );
                luma.resize(toff + kind.size, 0.0);
                let hm = hf_mul as f32;
                for &pos in &kind.order[kind.num_blocks..] {
                    let pos = pos as usize;
                    let q = arena[1][off + pos];
                    if q == 0 {
                        continue;
                    }
                    let a = q.unsigned_abs() as f32;
                    let rec = if q.abs() == 1 {
                        ctx.bias_one[1]
                    } else {
                        a - ctx.bias_numerator / a
                    };
                    let dy = rec.copysign(q as f32) * kind.step[1][pos] / hm;
                    luma[toff + pos] = dy;
                    let wx = f64::from(hm / kind.step[0][pos]);
                    let wb = f64::from(hm / kind.step[2][pos]);
                    let (dyf, xv, bv) = (
                        f64::from(dy),
                        f64::from(coefs[0][toff + pos]),
                        f64::from(coefs[2][toff + pos]),
                    );
                    sxy += wx * wx * (xv - f64::from(ctx.base_x) * dyf) * dyf;
                    sx_yy += wx * wx * dyf * dyf;
                    sby += wb * wb * (bv - f64::from(ctx.base_b) * dyf) * dyf;
                    sb_yy += wb * wb * dyf * dyf;
                }
                placed.push(Placed {
                    x,
                    y,
                    kind: k,
                    hf_mul,
                    mask,
                    off,
                });
                kind_grid[y * gw + x] = kind.t.index();
                for j in 0..kind.rows {
                    for i in 0..kind.cols {
                        hf_grid[(y + j) * gw + x + i] = hf_mul;
                    }
                }
            }
            // The factors: least squares, shrunk towards the default
            // by a ridge worth a few coefficients at one step.
            let ridge = 16.0f64;
            let cf = f64::from(ctx.colour_factor);
            let fx = ((sxy / (sx_yy + ridge)) * cf).round().clamp(-128.0, 127.0) as i32;
            let fb = ((sby / (sb_yy + ridge)) * cf).round().clamp(-128.0, 127.0) as i32;
            cfl[ty * tw + tx] = [fx, fb];
            let kx = ctx.base_x + fx as f32 / ctx.colour_factor;
            let kb = ctx.base_b + fb as f32 / ctx.colour_factor;

            // Chroma against the dequantised luma, and the LF samples.
            let mut toff = 0usize;
            for p in &placed[first..] {
                let kind = &ctx.kinds[p.kind as usize];
                let n = kind.size;
                for (c, k) in [(0usize, kx), (2usize, kb)] {
                    residual.clear();
                    residual.extend(
                        coefs[c][toff..toff + n]
                            .iter()
                            .zip(&luma[toff..toff + n])
                            .map(|(&v, &dy)| v - k * dy),
                    );
                    quantise(
                        ctx,
                        kind,
                        c,
                        p.hf_mul as f32,
                        p.mask,
                        &residual,
                        &mut arena[c][p.off..],
                        &mut work.s.q,
                    );
                }
                let link = &kind.link;
                let cells = link.cells.len();
                for j in 0..cells {
                    let cell = (p.y + j / link.cols) * gw + p.x + j % link.cols;
                    let mut v = [0.0f32; 3];
                    for (c, slot) in v.iter_mut().enumerate() {
                        lf_samples.clear();
                        lf_samples.extend(link.cells.iter().map(|&i| coefs[c][toff + i]));
                        *slot = link.from_llf[j * cells..(j + 1) * cells]
                            .iter()
                            .zip(&lf_samples)
                            .map(|(m, s)| m * s)
                            .sum();
                    }
                    let qy = (v[1] / ctx.lf_mul.m_y_dc).round();
                    let dy = qy * ctx.lf_mul.m_y_dc;
                    lf[1][cell] = qy as i32;
                    lf[0][cell] = ((v[0] - ctx.kx_lf * dy) / ctx.lf_mul.m_x_dc).round() as i32;
                    lf[2][cell] = ((v[2] - ctx.kb_lf * dy) / ctx.lf_mul.m_b_dc).round() as i32;
                }
                toff += n;
            }
        }
    }

    // Tokens, in the raster order of the varblocks' top-left blocks.
    placed.sort_unstable_by_key(|p| (p.y, p.x));
    let mut nz: [Vec<u32>; 3] = std::array::from_fn(|_| vec![0u32; n_cells]);
    let mut syms: Symbols = Vec::new();
    for p in &placed {
        let kind = &ctx.kinds[p.kind as usize];
        let (num_blocks, size) = (kind.num_blocks as u32, kind.size as u32);
        for c in [1usize, 0, 2] {
            let q = &arena[c][p.off..p.off + kind.size];
            let order = &kind.order;
            let non_zeros = order[kind.num_blocks..]
                .iter()
                .filter(|&&i| q[i as usize] != 0)
                .count() as u32;
            let predicted = predicted_non_zeros(p.x as u32, p.y as u32, |px, py| {
                nz[c][py as usize * gw + px as usize]
            });
            let block_ctx = kind.block_ctx[c];
            syms.push((
                non_zeros_context(predicted, block_ctx, ctx.nb_block_ctx),
                non_zeros,
            ));
            let per_block = non_zeros.div_ceil(num_blocks);
            for j in 0..kind.rows {
                for i in 0..kind.cols {
                    nz[c][(p.y + j) * gw + p.x + i] = per_block;
                }
            }
            let mut left = non_zeros;
            let mut k = num_blocks;
            while k < size && left > 0 {
                let prev = prev_for_context(k, num_blocks, size, left, |kk| {
                    q[order[kk as usize] as usize] != 0
                });
                let cc = coefficient_context(
                    k,
                    left,
                    num_blocks,
                    size,
                    prev,
                    block_ctx,
                    ctx.nb_block_ctx,
                )?;
                let v = q[order[k as usize] as usize];
                syms.push((cc, pack_signed(v)));
                if v != 0 {
                    left -= 1;
                }
                k += 1;
            }
        }
    }
    let mut stats = ValueStats::new(495 * ctx.nb_block_ctx as usize);
    for &(c, v) in &syms {
        stats.add(c, v);
    }
    Ok(GroupOut {
        lf,
        kind: kind_grid,
        hf_mul: hf_grid,
        cfl,
        syms,
        stats,
    })
}

fn build_kinds(
    rate: &RateParams,
    opts: &VarDctOptions,
    hbc_map: &[u8],
    gab_kernel: [f32; 3],
) -> Result<(Vec<Kind>, [u8; 27])> {
    let set = materialise_default_dequant_set_for_quantizer(rate.global_scale)?;
    let qm = [
        0.8f32.powi(rate.x_qm_scale as i32 - 2),
        1.0,
        0.8f32.powi(rate.b_qm_scale as i32 - 2),
    ];
    let no_thresholds: [Vec<i32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    // Reference step of each channel: the lowest AC band of DCT8×8.
    let reference = &set.matrices[slot_for_transform(TransformType::Dct8x8) as usize];
    let mut kinds = Vec::new();
    let mut kind_of = [CONTINUATION; 27];
    for t in transforms_for(opts) {
        let oid = order_id_for_transform(t);
        let order = natural_coeff_order(oid);
        let size = order.len();
        let (cols, rows) = t.block_dims();
        let (cols, rows) = (cols as usize, rows as usize);
        let small = small_transform(t);
        let matrices = &set.matrices[slot_for_transform(t) as usize];
        let mut step: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
        // What the decoder's smoothing leaves of each coefficient's
        // error.
        let gain = rate
            .gab
            .then(|| crate::enc_dct::smoothing_gain(t, gab_kernel));
        let mut scan_scale: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
        let mut scan_weight: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
        let mut block_ctx = [0u32; 3];
        for c in 0..3 {
            if matrices[c].len() != size {
                return Err(Error::invalid(
                    "JXL encoder: dequantisation matrix size mismatch",
                ));
            }
            step[c] = matrices[c].iter().map(|&m| m as f32 * qm[c]).collect();
            let m_ref = reference[c][1] as f32;
            scan_scale[c] = order.iter().map(|&i| 1.0 / step[c][i as usize]).collect();
            scan_weight[c] = order
                .iter()
                .map(|&i| {
                    let i = i as usize;
                    let norm2 = match small {
                        Some(st) => st.norm2[i],
                        None => size as f32,
                    };
                    let rel = matrices[c][i] as f32 / m_ref;
                    norm2 * rel * rel * gain.as_ref().map_or(1.0, |g| g[i])
                })
                .collect();
            block_ctx[c] = block_context(
                c as u32,
                oid.index(),
                rate.hf_mul,
                [0; 3],
                &[],
                &no_thresholds,
                hbc_map,
            )?;
        }
        let nb = cols * rows;
        let max_scale: [f32; 3] =
            std::array::from_fn(|c| scan_scale[c][nb..].iter().copied().fold(0.0f32, f32::max));
        let low_scale: [f32; 3] = std::array::from_fn(|c| scan_scale[c][nb]);
        let low_weight: [f32; 3] = std::array::from_fn(|c| scan_weight[c][nb]);
        kind_of[t.index() as usize] = kinds.len() as u8;
        kinds.push(Kind {
            t,
            cols,
            rows,
            size,
            num_blocks: cols * rows,
            order,
            small,
            link: lf_link(t),
            step,
            scan_scale,
            scan_weight,
            max_scale,
            low_scale,
            low_weight,
            block_ctx,
        });
    }
    Ok((kinds, kind_of))
}

fn code(input: &VarDctInput<'_>, opts: &VarDctOptions, learn: &LearnParams) -> Result<Coded> {
    let threads = opts.threads;
    let search = Search::new(opts);
    let (width, height) = (input.width, input.height);
    let bw = width.div_ceil(8);
    let bh = height.div_ceil(8);
    let rate = RateParams::for_distance(opts.distance);
    let quantizer = Quantizer {
        global_scale: rate.global_scale,
        quant_lf: rate.quant_lf,
    };
    let lf_weights = LfChannelDequantization {
        all_default: rate.lf_weights == [1.0 / 32.0, 0.25, 0.5],
        m_x_lf_unscaled: 128.0 / rate.lf_weights[0],
        m_y_lf_unscaled: 128.0 / rate.lf_weights[1],
        m_b_lf_unscaled: 128.0 / rate.lf_weights[2],
    };
    let cfl = LfChannelCorrelation::default();
    let (kx_lf, kb_lf) = kx_kb_lf(&cfl)?;
    let hbc_map = HfBlockContext::DEFAULT_BLOCK_CTX_MAP;
    let nb_block_ctx = u32::from(*hbc_map.iter().max().unwrap_or(&0)) + 1;
    let oim = OpsinInverseMatrix::default();
    let rf = crate::frame_header::RestorationFilter::default();
    let gab = crate::gaborish::gab_kernel(rf.gab_y_weight1, rf.gab_y_weight2)?;
    let (kinds, kind_of) = build_kinds(&rate, opts, &hbc_map, [gab[4], gab[1], gab[0]])?;
    let ctx = FrameCtx {
        input,
        opsin: Opsin::new(input.bits),
        rate,
        kinds,
        kind_of,
        lf_mul: LfMultipliers::compute(&lf_weights, &quantizer),
        kx_lf,
        kb_lf,
        base_x: cfl.base_correlation_x,
        base_b: cfl.base_correlation_b,
        colour_factor: cfl.colour_factor as f32,
        bias_one: oim.quant_bias,
        bias_numerator: oim.quant_bias_numerator,
        nb_block_ctx,
        bw,
        bh,
        search,
        gab_kernel: [gab[4], gab[1], gab[0]],
    };

    // Groups are independent: colour conversion, transform choice,
    // quantisation and tokenisation run per group.
    let gx_n = width.div_ceil(GROUP_DIM);
    let gy_n = height.div_ceil(GROUP_DIM);
    let num_groups = gx_n * gy_n;
    let n_blocks = bw * bh;
    let mut lf_q: [Vec<i32>; 3] = std::array::from_fn(|_| vec![0i32; n_blocks]);
    let mut kind_grid = vec![CONTINUATION; n_blocks];
    let mut hf_grid = vec![0i32; n_blocks];
    let (ftw, fth) = (bw.div_ceil(TILE_BLOCKS), bh.div_ceil(TILE_BLOCKS));
    let mut cfl_grid = vec![[0i32; 2]; ftw * fth];
    let mut hf_stats = ValueStats::new(495 * nb_block_ctx as usize);
    let mut hf_syms: Vec<Symbols> = Vec::with_capacity(num_groups);
    // A batch of groups at a time, so only a few per-group histograms
    // exist next to the merged one.
    let batch = threads.max(1) * 2;
    let mut first = 0usize;
    while first < num_groups {
        let ids: Vec<usize> = (first..num_groups.min(first + batch)).collect();
        let outs = crate::par::par_map(threads, ids, |_, g| code_group(&ctx, g % gx_n, g / gx_n));
        for (k, out) in outs.into_iter().enumerate() {
            let out = out?;
            let g = first + k;
            let (bx0, by0) = ((g % gx_n) * GROUP_BLOCKS, (g / gx_n) * GROUP_BLOCKS);
            let gw = GROUP_BLOCKS.min(bw - bx0);
            for (dst, src) in lf_q.iter_mut().zip(&out.lf) {
                for (row, line) in src.chunks_exact(gw).enumerate() {
                    let o = (by0 + row) * bw + bx0;
                    dst[o..o + gw].copy_from_slice(line);
                }
            }
            for (row, line) in out.kind.chunks_exact(gw).enumerate() {
                let o = (by0 + row) * bw + bx0;
                kind_grid[o..o + gw].copy_from_slice(line);
            }
            for (row, line) in out.hf_mul.chunks_exact(gw).enumerate() {
                let o = (by0 + row) * bw + bx0;
                hf_grid[o..o + gw].copy_from_slice(line);
            }
            let tw = gw.div_ceil(TILE_BLOCKS);
            for (row, line) in out.cfl.chunks_exact(tw).enumerate() {
                let o = (by0 / TILE_BLOCKS + row) * ftw + bx0 / TILE_BLOCKS;
                cfl_grid[o..o + tw].copy_from_slice(line);
            }
            hf_stats.merge(&out.stats);
            hf_syms.push(out.syms);
        }
        first += batch;
    }
    let hf_codes = EntropyCodes::build_clustered(&hf_stats, num_groups, 128)?;

    // Modular sections: LF coefficients and HF metadata per LF group,
    // the extra channels globally or per group.
    let lx_n = width.div_ceil(LF_GROUP_DIM);
    let ly_n = height.div_ceil(LF_GROUP_DIM);
    let num_lf_groups = lx_n * ly_n;
    let mut sections: Vec<Section> = Vec::new();
    let mut lf_meta: Vec<(u32, u32)> = Vec::with_capacity(num_lf_groups);
    for ly in 0..ly_n {
        for lx in 0..lx_n {
            let lg = ly * lx_n + lx;
            let bx0 = lx * (LF_GROUP_DIM / 8);
            let by0 = ly * (LF_GROUP_DIM / 8);
            let gbw = (LF_GROUP_DIM / 8).min(bw - bx0);
            let gbh = (LF_GROUP_DIM / 8).min(bh - by0);
            // LF coefficients in Modular order: Y, X, B.
            let lf_chans: Vec<Chan> = [1usize, 0, 2]
                .iter()
                .map(|&c| {
                    let mut data = Vec::with_capacity(gbw * gbh);
                    for y in 0..gbh {
                        let o = (by0 + y) * bw + bx0;
                        data.extend_from_slice(&lf_q[c][o..o + gbw]);
                    }
                    Chan::new(gbw, gbh, data)
                })
                .collect();
            sections.push(Section {
                stream_index: 1 + lg as i32,
                chans: lf_chans,
            });
            // HF metadata: XFromY, BFromY, BlockInfo, Sharpness.
            let px_w = (LF_GROUP_DIM).min(width - lx * LF_GROUP_DIM);
            let px_h = (LF_GROUP_DIM).min(height - ly * LF_GROUP_DIM);
            let (tw, th) = (px_w.div_ceil(64), px_h.div_ceil(64));
            let (tx0, ty0) = (bx0 / TILE_BLOCKS, by0 / TILE_BLOCKS);
            let mut x_from_y = Vec::with_capacity(tw * th);
            let mut b_from_y = Vec::with_capacity(tw * th);
            for ty in 0..th {
                for tx in 0..tw {
                    let f = cfl_grid[(ty0 + ty) * ftw + tx0 + tx];
                    x_from_y.push(f[0]);
                    b_from_y.push(f[1]);
                }
            }
            // BlockInfo: the varblocks in raster order of their
            // top-left blocks — first the types, then `HfMul - 1`.
            let mut types = Vec::new();
            let mut muls = Vec::new();
            for y in 0..gbh {
                for x in 0..gbw {
                    let i = (by0 + y) * bw + bx0 + x;
                    if kind_grid[i] != CONTINUATION {
                        types.push(i32::from(kind_grid[i]));
                        muls.push(hf_grid[i] - 1);
                    }
                }
            }
            let nb_blocks = types.len();
            types.extend_from_slice(&muls);
            let meta_chans = vec![
                Chan::new(tw, th, x_from_y),
                Chan::new(tw, th, b_from_y),
                Chan::new(nb_blocks, 2, types),
                Chan::new(gbw, gbh, vec![rate.sharpness; gbw * gbh]),
            ];
            sections.push(Section {
                stream_index: (1 + 2 * num_lf_groups + lg) as i32,
                chans: meta_chans,
            });
            lf_meta.push((nb_blocks as u32, ceil_log2((gbw as u64) * (gbh as u64))));
        }
    }
    let extra_in_groups = width > GROUP_DIM || height > GROUP_DIM;
    let n_fixed = sections.len();
    if !input.extra.is_empty() {
        if extra_in_groups {
            for gy in 0..gy_n {
                for gx in 0..gx_n {
                    let (x0, y0) = (gx * GROUP_DIM, gy * GROUP_DIM);
                    let rw = GROUP_DIM.min(width - x0);
                    let rh = GROUP_DIM.min(height - y0);
                    sections.push(Section {
                        stream_index: (1 + 3 * num_lf_groups + 17 + gy * gx_n + gx) as i32,
                        chans: input
                            .extra
                            .iter()
                            .map(|c| Chan::new(width, height, c.clone()).rect(x0, y0, rw, rh))
                            .collect(),
                    });
                }
            }
        } else {
            sections.push(Section {
                stream_index: 0,
                chans: input
                    .extra
                    .iter()
                    .map(|c| Chan::new(width, height, c.clone()))
                    .collect(),
            });
        }
    }
    // The LF image and the block metadata are a small part of the
    // file; below the largest effort the tree is learned on half the
    // usual sample with a third of the leaves, which costs about 1 %
    // of the file and a fifth of the encoding time less.
    let lf_learn = if opts.effort == Effort::Thorough {
        learn.clone()
    } else {
        LearnParams {
            max_samples: learn.max_samples.min(160_000),
            max_leaves: learn.max_leaves.min(256),
            ..learn.clone()
        }
    };
    let tree = learn_tree(&sections, &lf_learn);
    let tree_syms = tree.symbols();
    let mut tree_stats = ValueStats::new(6);
    for &(c, v) in &tree_syms {
        tree_stats.add(c, v);
    }
    let tree_codes = EntropyCodes::build_clustered(&tree_stats, 1, 6)?;
    let mut leaf_stats = ValueStats::new(tree.num_ctx);
    let mut coded: Vec<Symbols> = Vec::with_capacity(sections.len());
    for syms in crate::par::par_map(threads, sections, |_, mut section| {
        encode_section(&mut section, &tree)
    }) {
        for &(c, v) in &syms {
            leaf_stats.add(c, v);
        }
        coded.push(syms);
    }
    let n_streams = coded.iter().filter(|s| !s.is_empty()).count().max(1);
    let leaf_codes = EntropyCodes::build_clustered(&leaf_stats, n_streams, 256)?;

    let mut extra_syms: Vec<Vec<(u32, u32)>> = coded.split_off(n_fixed);
    let mut it = coded.into_iter();
    let mut lf_groups = Vec::with_capacity(num_lf_groups);
    for &(nb_blocks, nbits) in &lf_meta {
        let lf = it.next().unwrap_or_default();
        let meta = it.next().unwrap_or_default();
        lf_groups.push((lf, nb_blocks, nbits, meta));
    }
    let global_extra = if input.extra.is_empty() {
        None
    } else if extra_in_groups {
        Some(Vec::new())
    } else {
        Some(extra_syms.pop().unwrap_or_default())
    };
    let mut groups = Vec::with_capacity(num_groups);
    let mut extra_it = extra_syms.into_iter();
    for hf in hf_syms {
        let extra = if extra_in_groups && !input.extra.is_empty() {
            extra_it.next().unwrap_or_default()
        } else {
            Vec::new()
        };
        groups.push((hf, extra));
    }
    Ok(Coded {
        rate,
        tree,
        tree_codes,
        leaf_codes,
        hf_codes,
        global_extra,
        lf_groups,
        groups,
        num_groups,
    })
}

/// `ModularHeader` of a sub-bitstream on the global tree with no
/// transform.
fn write_modular_header(w: &mut BitWriter) {
    w.write_bool(true); // use_global_tree
    w.write_bool(true); // wp_params.default_wp
    w.write_bits(2, 0); // nb_transforms = 0
}

/// One `PassGroup`: the HF coefficients (`hfp` takes no bits with one
/// preset), then the group's Modular data when extra channels live in
/// groups.
fn write_pass_group(w: &mut BitWriter, coded: &Coded, hf: &Symbols, extra: &Symbols) {
    coded.hf_codes.write_symbols(w, hf.iter().copied());
    if !extra.is_empty() {
        write_modular_header(w);
        coded.leaf_codes.write_symbols(w, extra.iter().copied());
    }
}

/// Encode one VarDCT frame: the TOC sections in order (one entry when
/// the frame has a single group, each of the others byte-aligned).
pub(crate) fn encode_frame(
    input: &VarDctInput<'_>,
    opts: &VarDctOptions,
    learn: &LearnParams,
) -> Result<(Vec<Vec<u8>>, RateParams)> {
    if input.width == 0 || input.height == 0 {
        return Err(Error::invalid("JXL encoder: zero image dimension"));
    }
    let threads = opts.threads;
    let coded = code(input, opts, learn)?;
    let single = coded.num_groups == 1;
    let mut sections: Vec<Vec<u8>> = Vec::new();
    let mut w = BitWriter::new();
    let finish = |w: &mut BitWriter, sections: &mut Vec<Vec<u8>>| {
        if !single {
            let done = std::mem::take(w);
            sections.push(done.into_bytes());
        }
    };

    // LfGlobal.
    if coded.rate.lf_weights == [1.0 / 32.0, 0.25, 0.5] {
        w.write_bool(true); // LfChannelDequantization.all_default
    } else {
        w.write_bool(false);
        for m in coded.rate.lf_weights {
            w.write_f16(m)?;
        }
    }
    // Quantizer (I.2.1).
    w.write_u32(
        [
            crate::bitreader::U32Dist::BitsOffset(11, 1),
            crate::bitreader::U32Dist::BitsOffset(11, 2049),
            crate::bitreader::U32Dist::BitsOffset(12, 4097),
            crate::bitreader::U32Dist::BitsOffset(16, 8193),
        ],
        coded.rate.global_scale,
    )?;
    w.write_u32(
        [
            crate::bitreader::U32Dist::Val(16),
            crate::bitreader::U32Dist::BitsOffset(5, 1),
            crate::bitreader::U32Dist::BitsOffset(8, 1),
            crate::bitreader::U32Dist::BitsOffset(16, 1),
        ],
        coded.rate.quant_lf,
    )?;
    w.write_bool(true); // HF block context: the default map
    w.write_bool(true); // LfChannelCorrelation.all_default
    // GlobalModular: the tree every Modular sub-bitstream shares.
    w.write_bool(true); // use_global_tree
    let tree_syms = coded.tree.symbols();
    coded.tree_codes.write_prelude(&mut w);
    coded
        .tree_codes
        .write_symbols(&mut w, tree_syms.iter().copied());
    coded.leaf_codes.write_prelude(&mut w);
    if let Some(extra) = &coded.global_extra {
        write_modular_header(&mut w);
        if !extra.is_empty() {
            coded
                .leaf_codes
                .write_symbols(&mut w, extra.iter().copied());
        }
    }
    finish(&mut w, &mut sections);

    // LfGroup sections.
    for (lf, nb_blocks, nbits, meta) in &coded.lf_groups {
        w.write_bits(2, 0); // extra_precision
        write_modular_header(&mut w);
        coded.leaf_codes.write_symbols(&mut w, lf.iter().copied());
        w.write_bits(*nbits, nb_blocks - 1);
        write_modular_header(&mut w);
        coded.leaf_codes.write_symbols(&mut w, meta.iter().copied());
        finish(&mut w, &mut sections);
    }

    // HfGlobal: default matrices, one preset, natural orders, the
    // coefficient histograms.
    w.write_bool(true);
    w.write_bits(ceil_log2(coded.num_groups as u64), 0); // num_hf_presets - 1
    w.write_bits(2, 2); // used_orders = U32 selector 2 → 0
    coded.hf_codes.write_prelude(&mut w);
    finish(&mut w, &mut sections);

    // PassGroup sections.
    if single {
        for (hf, extra) in &coded.groups {
            write_pass_group(&mut w, &coded, hf, extra);
        }
    } else {
        let groups: Vec<&(Symbols, Symbols)> = coded.groups.iter().collect();
        sections.extend(crate::par::par_map(threads, groups, |_, (hf, extra)| {
            let mut gw = BitWriter::new();
            write_pass_group(&mut gw, &coded, hf, extra);
            gw.into_bytes()
        }));
    }
    if single {
        sections.push(w.into_bytes());
    }
    Ok((sections, coded.rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_xyb_is_inverted_by_the_decoder_colour_transform() {
        let opsin = Opsin::new(8);
        let oim = OpsinInverseMatrix::default();
        for rgb in [[0, 0, 0], [255, 255, 255], [200, 30, 90], [12, 250, 128]] {
            let xyb = opsin.xyb(rgb[0], rgb[1], rgb[2]);
            let tm = crate::metadata_fdis::ToneMapping::default();
            let (r, g, b) = crate::xyb::inverse_xyb_to_rgb(xyb[0], xyb[1], xyb[2], &oim, &tm);
            for (got, want) in [r, g, b].iter().zip(rgb) {
                let lin = srgb_to_linear(f64::from(want) / 255.0) as f32;
                assert!((got - lin).abs() < 2e-4, "{rgb:?}: {got} vs {lin}");
            }
        }
    }

    #[test]
    fn sharpening_inverts_the_gabor_like_smoothing() {
        let rf = crate::frame_header::RestorationFilter::default();
        let k = crate::gaborish::gab_kernel(rf.gab_y_weight1, rf.gab_y_weight2).unwrap();
        let (w, h) = (40usize, 40usize);
        let mut s = 9u32;
        let plane: Vec<f32> = (0..w * h)
            .map(|i| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((i % w) as f32 * 0.3).sin() * 0.2 + (s >> 8) as f32 / (1 << 24) as f32 * 0.05
            })
            .collect();
        let sharp = sharpen(&plane, w, h, [k[4], k[1], k[0]]);
        let mut blurred = sharp.clone();
        crate::gaborish::apply_channel_in_place(
            &mut blurred,
            w,
            h,
            rf.gab_y_weight1,
            rf.gab_y_weight2,
        )
        .unwrap();
        for y in MARGIN..h - MARGIN {
            for x in MARGIN..w - MARGIN {
                let i = y * w + x;
                assert!(
                    (blurred[i] - plane[i]).abs() < 4e-3,
                    "({x},{y}): {} vs {}",
                    blurred[i],
                    plane[i]
                );
            }
        }
    }

    #[test]
    fn rate_parameters_are_monotone_in_the_distance() {
        let mut prev = f64::INFINITY;
        for d in [0.05f32, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0] {
            let r = RateParams::for_distance(d);
            let precision = f64::from(r.global_scale) * f64::from(r.hf_mul);
            assert!(precision < prev, "distance {d}");
            assert!(r.global_scale >= 1 && r.quant_lf >= 1 && r.hf_mul >= 1);
            prev = precision;
        }
        assert_eq!(distance_for_quality(90), 1.0);
        assert!(distance_for_quality(0) > distance_for_quality(50));
    }
}
