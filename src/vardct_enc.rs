//! VarDCT **encoder** (Annex I, written side): lossy XYB frames.
//!
//! One frame is produced as the list of TOC sections of F.3:
//!
//! * `LfGlobal` — default LF dequantisation weights, the `Quantizer`
//!   (`global_scale`, `quant_lf`), the default HF block-context map,
//!   the default `LfChannelCorrelation`, and `GlobalModular` with the
//!   one MA tree every Modular sub-bitstream of the frame shares;
//! * one `LfGroup` per 2048×2048 area — the quantised LF image (the
//!   8×8 block means of X, Y and B, B and X coded against the
//!   dequantised Y through the LF correlation factors) as a Modular
//!   sub-bitstream, then the HF metadata (chroma-from-luma tiles, the
//!   `BlockInfo` list of transform types and quantisation multipliers,
//!   the sharpness map);
//! * `HfGlobal` — default dequantisation matrices, one HF preset, the
//!   natural coefficient orders, the clustered coefficient histograms;
//! * one `PassGroup` per 256×256 group — the quantised HF coefficients
//!   with the I.4 context model (`NonZerosContext`,
//!   `CoefficientContext`), then the group's rectangle of the extra
//!   channels as Modular data.
//!
//! The forward path mirrors what the decoder in this crate applies:
//! sRGB → linear → XYB (the inverse of L.2.2), an 8×8 DCT (the inverse
//! of I.7.3), quantisation against the default matrices of I.2.5
//! scaled by `global_scale` and the per-block `HfMul`, with the
//! chroma-from-luma prediction of I.6 taken from the *dequantised* Y
//! coefficients so the decoder reproduces it exactly. Dequantisation,
//! context numbering and matrix values are taken from the decoder's
//! own functions, never re-derived.
//!
//! All rate decisions (step sizes for a distance, the dead zone, the
//! adaptive quantisation field, the chroma-from-luma fit) are this
//! crate's own design, tuned on the corpus described in the README.

use crate::bitwriter::{BitWriter, pack_signed};
use crate::chroma_from_luma::{kx_kb_hf, kx_kb_lf};
use crate::coeff_order::{natural_coeff_order, order_id_for_transform};
use crate::dct_quant_weights::{materialise_default_dequant_set_for_quantizer, slot_for_transform};
use crate::dct_select::TransformType;
use crate::entropy_write::{EntropyCodes, ValueStats};
use crate::error::{Error, Result};
use crate::hf_dequant::bias_adjust;
use crate::lf_dequant::LfMultipliers;
use crate::lf_global::{HfBlockContext, LfChannelCorrelation, LfChannelDequantization, Quantizer};
use crate::metadata_fdis::OpsinInverseMatrix;
use crate::modular_enc::{Chan, LearnParams, Section, Tree, encode_section, learn_tree};
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

/// Rate parameters of a frame derived from the distance.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RateParams {
    pub global_scale: u32,
    pub quant_lf: u32,
    /// `HfMul` of every block (the uniform part of the quantisation
    /// field).
    pub hf_mul: i32,
    /// Quantised magnitudes below this (in units of the step) become
    /// zero.
    pub zero_threshold: f32,
    /// `x_qm_scale` / `b_qm_scale` of the frame header (F.2): the HF
    /// steps of X and B are multiplied by `0.8^(scale − 2)`.
    pub x_qm_scale: u32,
    pub b_qm_scale: u32,
    /// LF dequantisation weights `m_x_lf`, `m_y_lf`, `m_b_lf` (G.1.2).
    pub lf_weights: [f32; 3],
}

/// Precision of the Y channel's lowest HF band at distance 1, as
/// `global_scale × HfMul / 65536` — the reciprocal of the quantisation
/// step in units of the default matrix entry. Calibrated on the corpus
/// (see the README's rate–distortion table): at this value distance 1
/// lands at the file size a black-box encoder produces for its
/// distance 1 on the colour photo.
const PRECISION_AT_DISTANCE_1: f64 = 0.40;

/// LF step relative to the HF step of the lowest Y band. Block means
/// are the most visible coefficients (a coarse LF shows as blocking),
/// so the LF is kept finer; settled on the corpus PSNR / size curve.
const LF_STEP_RATIO: f64 = 0.55;

/// Dead zone: a coefficient under this many steps is dropped. Swept
/// over 0.50 … 0.75 on the corpus: 0.65 is where the bytes saved stop
/// outrunning the PSNR lost (−9 % bytes for −0.6 dB on the grey photo
/// against −0.87 dB for the same bytes by raising the distance).
const ZERO_THRESHOLD: f32 = 0.65;

/// LF weights `m_x_lf`, `m_y_lf`, `m_b_lf` (G.1.2; the format's
/// defaults are 1/32, 1/4, 1/2). X and B are kept twice as fine as the
/// defaults: at distance 4 the colour photo gains 0.4 dB for 3 % more
/// bytes (0.14 dB is what those bytes buy through the distance), and
/// block-level colour error is what shows first at low rates.
const LF_WEIGHTS: [f32; 3] = [1.0 / 64.0, 0.25, 0.25];

impl RateParams {
    /// Parameters for a distance (`> 0`; about 1 is visually lossless,
    /// larger is smaller and worse).
    pub(crate) fn for_distance(distance: f32) -> Self {
        let d = f64::from(distance.clamp(0.01, 64.0));
        let precision = PRECISION_AT_DISTANCE_1 / d;
        // global_scale × hf_mul = 65536 × precision, with HfMul kept
        // near 8 so an adaptive field has room on both sides.
        let product = 65536.0 * precision;
        let mut hf_mul = 8.0f64;
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
            zero_threshold: ZERO_THRESHOLD,
            x_qm_scale: 3,
            b_qm_scale: 2,
            lf_weights: LF_WEIGHTS,
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
    fwd: [[f64; 3]; 3],
    bias: [f64; 3],
    cbrt_bias: [f64; 3],
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
            fwd: invert3(inv),
            bias,
            cbrt_bias: bias.map(f64::cbrt),
            lut: (0..1usize << bits)
                .map(|v| srgb_to_linear(v as f64 / max) as f32)
                .collect(),
        }
    }

    #[inline]
    fn xyb(&self, r: i32, g: i32, b: i32) -> [f32; 3] {
        let rgb = [
            f64::from(self.lut[r as usize]),
            f64::from(self.lut[g as usize]),
            f64::from(self.lut[b as usize]),
        ];
        let mut gamma = [0.0f64; 3];
        for (i, slot) in gamma.iter_mut().enumerate() {
            let mix = self.fwd[i][0] * rgb[0] + self.fwd[i][1] * rgb[1] + self.fwd[i][2] * rgb[2];
            *slot = (mix - self.bias[i]).max(0.0).cbrt() + self.cbrt_bias[i];
        }
        [
            ((gamma[0] - gamma[1]) * 0.5) as f32,
            ((gamma[0] + gamma[1]) * 0.5) as f32,
            gamma[2] as f32,
        ]
    }
}

// ---- transform ------------------------------------------------------------

/// `cos(pi × k × (2n + 1) / 16) × (k ? sqrt(2) : 1) / 8`: the forward
/// DCT of I.7.2 for eight samples.
fn dct8_table() -> [[f32; 8]; 8] {
    let mut t = [[0.0f32; 8]; 8];
    for (k, row) in t.iter_mut().enumerate() {
        for (n, slot) in row.iter_mut().enumerate() {
            let c = (std::f64::consts::PI * k as f64 * (2 * n + 1) as f64 / 16.0).cos();
            let scale = if k == 0 {
                1.0
            } else {
                std::f64::consts::SQRT_2
            };
            *slot = (c * scale / 8.0) as f32;
        }
    }
    t
}

/// Forward 8×8 DCT of a row-major block; `out[u * 8 + v]` holds the
/// coefficient of horizontal frequency `u` and vertical frequency `v`
/// — I.7.3's `DCT_2D` leaves a square block transposed, and that is the
/// layout the decoder's `idct_for_transform(Dct8x8, ..)` consumes
/// (pinned by a unit test).
#[inline]
fn fdct8x8(table: &[[f32; 8]; 8], block: &[f32; 64], out: &mut [f32; 64]) {
    let mut tmp = [0.0f32; 64];
    for y in 0..8 {
        let row = &block[y * 8..y * 8 + 8];
        for u in 0..8 {
            let t = &table[u];
            let mut s = 0.0f32;
            for n in 0..8 {
                s += row[n] * t[n];
            }
            tmp[y * 8 + u] = s;
        }
    }
    for u in 0..8 {
        for v in 0..8 {
            let t = &table[v];
            let mut s = 0.0f32;
            for n in 0..8 {
                s += tmp[n * 8 + u] * t[n];
            }
            out[u * 8 + v] = s;
        }
    }
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

/// Quantise `t` (a coefficient in units of its step) to the level
/// whose reconstruction — I.5.3's bias-adjusted value — is nearest,
/// with everything under `zero_threshold` dropped.
#[inline]
fn quantise_hf(t: f32, channel: usize, oim: &OpsinInverseMatrix, zero_threshold: f32) -> i32 {
    let a = t.abs();
    if a < zero_threshold {
        return 0;
    }
    let n0 = (a.round() as i32).max(1);
    let mut best = n0;
    let mut best_err = (a - bias_adjust(n0, channel, oim)).abs();
    for n in [n0 - 1, n0 + 1] {
        if n >= 1 {
            let e = (a - bias_adjust(n, channel, oim)).abs();
            if e < best_err {
                best_err = e;
                best = n;
            }
        }
    }
    if t < 0.0 { -best } else { best }
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

/// What one group contributes: its quantised LF samples (X, Y, B, in
/// raster order of the group's blocks), its HF symbols and their
/// statistics.
struct GroupOut {
    lf: [Vec<i32>; 3],
    syms: Symbols,
    stats: ValueStats,
}

/// Per-frame constants every group worker shares.
struct GroupCtx<'a> {
    input: &'a VarDctInput<'a>,
    opsin: Opsin,
    table: [[f32; 8]; 8],
    lf_mul: LfMultipliers,
    kx_lf: f32,
    kb_lf: f32,
    kx_hf: f32,
    kb_hf: f32,
    /// Reciprocal HF step of each coefficient, per channel.
    inv_step: [[f32; 64]; 3],
    step_y: [f32; 64],
    oim: OpsinInverseMatrix,
    zero_threshold: f32,
    order: Vec<u32>,
    block_ctx: [u32; 3],
    nb_block_ctx: u32,
    bw: usize,
    bh: usize,
}

/// Colour-convert, transform, quantise and tokenise one group.
fn code_group(ctx: &GroupCtx<'_>, gx: usize, gy: usize) -> Result<GroupOut> {
    let (width, height) = (ctx.input.width, ctx.input.height);
    let bx0 = gx * GROUP_BLOCKS;
    let by0 = gy * GROUP_BLOCKS;
    let gw = GROUP_BLOCKS.min(ctx.bw - bx0);
    let gh = GROUP_BLOCKS.min(ctx.bh - by0);
    let grey = ctx.input.colour.len() == 1;
    // XYB of the group's pixels, padded to whole blocks by edge
    // replication.
    let (pw, ph) = (gw * 8, gh * 8);
    let mut planes: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; pw * ph]);
    for y in 0..ph {
        let sy = (by0 * 8 + y).min(height - 1);
        for x in 0..pw {
            let sx = (bx0 * 8 + x).min(width - 1);
            let i = sy * width + sx;
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
            let o = y * pw + x;
            planes[0][o] = v[0];
            planes[1][o] = v[1];
            planes[2][o] = v[2];
        }
    }
    let mut lf: [Vec<i32>; 3] = std::array::from_fn(|_| vec![0i32; gw * gh]);
    let mut nz: [Vec<u32>; 3] = std::array::from_fn(|_| vec![0u32; gw * gh]);
    let mut syms: Symbols = Vec::new();
    let mut block = [0.0f32; 64];
    let mut coeffs = [[0.0f32; 64]; 3];
    let mut quant = [[0i32; 64]; 3];
    for y in 0..gh {
        for x in 0..gw {
            for (plane, dst) in planes.iter().zip(coeffs.iter_mut()) {
                for r in 0..8 {
                    let o = (y * 8 + r) * pw + x * 8;
                    block[r * 8..r * 8 + 8].copy_from_slice(&plane[o..o + 8]);
                }
                fdct8x8(&ctx.table, &block, dst);
            }
            // LF: the block means, X and B against the dequantised Y.
            let b = y * gw + x;
            let qy = (coeffs[1][0] / ctx.lf_mul.m_y_dc).round();
            let dy = qy * ctx.lf_mul.m_y_dc;
            lf[1][b] = qy as i32;
            lf[0][b] = ((coeffs[0][0] - ctx.kx_lf * dy) / ctx.lf_mul.m_x_dc).round() as i32;
            lf[2][b] = ((coeffs[2][0] - ctx.kb_lf * dy) / ctx.lf_mul.m_b_dc).round() as i32;
            // HF: Y first, then X and B against the dequantised Y.
            for i in 1..64 {
                let qy = quantise_hf(
                    coeffs[1][i] * ctx.inv_step[1][i],
                    1,
                    &ctx.oim,
                    ctx.zero_threshold,
                );
                quant[1][i] = qy;
                let dy = bias_adjust(qy, 1, &ctx.oim) * ctx.step_y[i];
                quant[0][i] = quantise_hf(
                    (coeffs[0][i] - ctx.kx_hf * dy) * ctx.inv_step[0][i],
                    0,
                    &ctx.oim,
                    ctx.zero_threshold,
                );
                quant[2][i] = quantise_hf(
                    (coeffs[2][i] - ctx.kb_hf * dy) * ctx.inv_step[2][i],
                    2,
                    &ctx.oim,
                    ctx.zero_threshold,
                );
            }
            for c in [1usize, 0, 2] {
                let q = &quant[c];
                let non_zeros = q[1..].iter().filter(|&&v| v != 0).count() as u32;
                let predicted = predicted_non_zeros(x as u32, y as u32, |px, py| {
                    nz[c][py as usize * gw + px as usize]
                });
                syms.push((
                    non_zeros_context(predicted, ctx.block_ctx[c], ctx.nb_block_ctx),
                    non_zeros,
                ));
                nz[c][b] = non_zeros;
                let mut left = non_zeros;
                let mut k = 1u32;
                while k < 64 && left > 0 {
                    let prev = prev_for_context(k, 1, 64, left, |kk| {
                        q[ctx.order[kk as usize] as usize] != 0
                    });
                    let cc = coefficient_context(
                        k,
                        left,
                        1,
                        64,
                        prev,
                        ctx.block_ctx[c],
                        ctx.nb_block_ctx,
                    )?;
                    let v = q[ctx.order[k as usize] as usize];
                    syms.push((cc, pack_signed(v)));
                    if v != 0 {
                        left -= 1;
                    }
                    k += 1;
                }
            }
        }
    }
    let mut stats = ValueStats::new(495 * ctx.nb_block_ctx as usize);
    for &(c, v) in &syms {
        stats.add(c, v);
    }
    Ok(GroupOut { lf, syms, stats })
}

fn code(
    input: &VarDctInput<'_>,
    distance: f32,
    learn: &LearnParams,
    threads: usize,
) -> Result<Coded> {
    let (width, height) = (input.width, input.height);
    let bw = width.div_ceil(8);
    let bh = height.div_ceil(8);
    let rate = RateParams::for_distance(distance);
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
    let (kx_hf, kb_hf) = kx_kb_hf(&cfl, 0, 0)?;
    // The decoder's own matrices: step = matrix × qm / HfMul.
    let set = materialise_default_dequant_set_for_quantizer(rate.global_scale)?;
    let slot = slot_for_transform(TransformType::Dct8x8) as usize;
    let matrices = &set.matrices[slot];
    let qm = [
        0.8f32.powi(rate.x_qm_scale as i32 - 2),
        1.0,
        0.8f32.powi(rate.b_qm_scale as i32 - 2),
    ];
    let hf_mul = rate.hf_mul;
    let mut inv_step = [[0.0f32; 64]; 3];
    let mut step_y = [0.0f32; 64];
    for c in 0..3 {
        for i in 0..64 {
            let step = matrices[c][i] as f32 * qm[c] / hf_mul as f32;
            inv_step[c][i] = 1.0 / step;
            if c == 1 {
                step_y[i] = step;
            }
        }
    }
    let hbc_map = HfBlockContext::DEFAULT_BLOCK_CTX_MAP;
    let nb_block_ctx = u32::from(*hbc_map.iter().max().unwrap_or(&0)) + 1;
    let no_thresholds: [Vec<i32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut block_ctx = [0u32; 3];
    for (c, slot) in block_ctx.iter_mut().enumerate() {
        *slot = block_context(c as u32, 0, hf_mul, [0; 3], &[], &no_thresholds, &hbc_map)?;
    }
    let ctx = GroupCtx {
        input,
        opsin: Opsin::new(input.bits),
        table: dct8_table(),
        lf_mul: LfMultipliers::compute(&lf_weights, &quantizer),
        kx_lf,
        kb_lf,
        kx_hf,
        kb_hf,
        inv_step,
        step_y,
        oim: OpsinInverseMatrix::default(),
        zero_threshold: rate.zero_threshold,
        order: natural_coeff_order(order_id_for_transform(TransformType::Dct8x8)),
        block_ctx,
        nb_block_ctx,
        bw,
        bh,
    };

    // Groups are independent: colour conversion, transform,
    // quantisation and tokenisation run per group.
    let gx_n = width.div_ceil(GROUP_DIM);
    let gy_n = height.div_ceil(GROUP_DIM);
    let num_groups = gx_n * gy_n;
    let group_ids: Vec<usize> = (0..num_groups).collect();
    let outs = crate::par::par_map(threads, group_ids, |_, g| {
        code_group(&ctx, g % gx_n, g / gx_n)
    });
    let n_blocks = bw * bh;
    let mut lf_q: [Vec<i32>; 3] = std::array::from_fn(|_| vec![0i32; n_blocks]);
    let mut hf_stats = ValueStats::new(495 * nb_block_ctx as usize);
    let mut hf_syms: Vec<Symbols> = Vec::with_capacity(num_groups);
    for (g, out) in outs.into_iter().enumerate() {
        let out = out?;
        let (bx0, by0) = ((g % gx_n) * GROUP_BLOCKS, (g / gx_n) * GROUP_BLOCKS);
        let gw = GROUP_BLOCKS.min(bw - bx0);
        for (dst, src) in lf_q.iter_mut().zip(&out.lf) {
            for (row, line) in src.chunks_exact(gw).enumerate() {
                let o = (by0 + row) * bw + bx0;
                dst[o..o + gw].copy_from_slice(line);
            }
        }
        hf_stats.merge(&out.stats);
        hf_syms.push(out.syms);
    }
    let hf_codes = EntropyCodes::build_clustered(&hf_stats, num_groups, 128)?;

    // Modular sections: LF coefficients and HF metadata per LF group,
    // the extra channels globally or per group.
    let lx_n = width.div_ceil(LF_GROUP_DIM);
    let ly_n = height.div_ceil(LF_GROUP_DIM);
    let num_lf_groups = lx_n * ly_n;
    let mut sections: Vec<Section> = Vec::new();
    // Index of each kind of section in `sections`.
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
            let nb_blocks = gbw * gbh;
            let (tw, th) = (px_w.div_ceil(64), px_h.div_ceil(64));
            let mut info = vec![0i32; nb_blocks * 2];
            for slot in info[nb_blocks..].iter_mut() {
                *slot = hf_mul - 1;
            }
            let meta_chans = vec![
                Chan::new(tw, th, vec![0; tw * th]),
                Chan::new(tw, th, vec![0; tw * th]),
                Chan::new(nb_blocks, 2, info),
                Chan::new(gbw, gbh, vec![0; gbw * gbh]),
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
    let tree = learn_tree(&sections, learn);
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
    distance: f32,
    learn: &LearnParams,
    threads: usize,
) -> Result<(Vec<Vec<u8>>, RateParams)> {
    if input.width == 0 || input.height == 0 {
        return Err(Error::invalid("JXL encoder: zero image dimension"));
    }
    let coded = code(input, distance, learn, threads)?;
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
    use crate::idct::idct_for_transform;

    #[test]
    fn forward_dct_is_inverted_by_the_decoder_idct() {
        let table = dct8_table();
        let mut block = [0.0f32; 64];
        let mut s = 12345u32;
        for v in block.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s >> 8) as f32 / (1 << 24) as f32 - 0.5;
        }
        // Asymmetric content so a transposed layout would show.
        for (i, v) in block.iter_mut().enumerate() {
            *v += (i % 8) as f32 * 0.1;
        }
        let mut coeffs = [0.0f32; 64];
        fdct8x8(&table, &block, &mut coeffs);
        let mean: f32 = block.iter().sum::<f32>() / 64.0;
        assert!((coeffs[0] - mean).abs() < 1e-5, "DC is the block mean");
        let back = idct_for_transform(TransformType::Dct8x8, &coeffs).unwrap();
        for i in 0..64 {
            assert!(
                (back[i] - block[i]).abs() < 1e-4,
                "sample {i}: {} vs {}",
                back[i],
                block[i]
            );
        }
    }

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
    fn quantiser_picks_the_nearest_reconstruction() {
        let oim = OpsinInverseMatrix::default();
        assert_eq!(quantise_hf(0.3, 1, &oim, 0.56), 0);
        assert_eq!(quantise_hf(-0.3, 1, &oim, 0.56), 0);
        assert_eq!(quantise_hf(0.9, 1, &oim, 0.56), 1);
        assert_eq!(quantise_hf(-1.3, 1, &oim, 0.56), -1);
        assert_eq!(quantise_hf(7.2, 0, &oim, 0.56), 7);
        assert_eq!(quantise_hf(-7.6, 2, &oim, 0.56), -8);
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
