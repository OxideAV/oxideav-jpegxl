//! Forward transforms of the VarDCT encoder: the inverses of the
//! I.7 / I.8 inverse transforms exactly as the decoder in this crate
//! applies them.
//!
//! * The DCT family (8×8 … 64×64 and the rectangular sizes) is a
//!   separable forward DCT (I.7.2) written into the coefficient layout
//!   the decoder's `idct_for_transform` reads: row-major
//!   `short × long`, the row index being the frequency along the
//!   shorter side of the pixel block.
//! * The 8×8 transforms that are not plain DCTs (DCT4×4, DCT4×8,
//!   DCT8×4, DCT2×2, IDENTITY, AFV0..3) have no closed-form forward
//!   here: each is the numerical inverse of the decoder's own 64×64
//!   synthesis matrix, built once per process.
//! * The LF image samples a varblock asks for are the inverse of the
//!   decoder's LF → LLF conversion (I.8), again inverted numerically
//!   from the decoder's function, so the two can never disagree.

use std::sync::OnceLock;

use crate::coeff_order::{order_id_for_transform, varblock_size_for_order};
use crate::dct_select::TransformType;
use crate::idct::idct_for_transform;
use crate::llf_from_lf::llf_from_lf;

/// Largest DCT side the encoder uses.
pub(crate) const MAX_SIDE: usize = 64;

/// `2 × cos(pi × (2n + 1) / (2N))` for `n < N / 2`, `N = 1 << log2n`:
/// the odd-half multipliers of the recursive DCT.
fn half_cosines(log2n: usize) -> &'static [f32] {
    static TABLES: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
    let tables = TABLES.get_or_init(|| {
        (0..=MAX_SIDE.trailing_zeros() as usize)
            .map(|l| {
                let n = 1usize << l;
                (0..n / 2)
                    .map(|i| {
                        (2.0 * (std::f64::consts::PI * (2 * i + 1) as f64 / (2 * n) as f64).cos())
                            as f32
                    })
                    .collect()
            })
            .collect()
    });
    &tables[log2n]
}

/// Unscaled four-point DCT-II (`k` = [`half_cosines`]`(2)`): one level
/// of the recursion of [`dct_in_place`] over two-point transforms.
#[cfg(test)]
#[inline(always)]
fn dct4(a: [f32; 4], k: &[f32]) -> [f32; 4] {
    const R: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let (s0, s1) = (a[0] + a[3], a[1] + a[2]);
    let (d0, d1) = ((a[0] - a[3]) * k[0], (a[1] - a[2]) * k[1]);
    let x1 = (d0 + d1) * 0.5;
    [s0 + s1, x1, (s0 - s1) * R, (d0 - d1) * R - x1]
}

/// Unscaled DCT-II in place: `x[k] ← sum_n x[n] × cos(pi × (2n + 1) ×
/// k / (2N))`, by the even / odd recursion. With `s[n] = x[n] +
/// x[N-1-n]` and `g[n] = (x[n] − x[N-1-n]) × 2 cos(pi (2n+1) / 2N)`,
/// the even outputs are the half-size transform of `s`, and the
/// half-size transform `G` of `g` satisfies `G[m] = X[2m+1] +
/// X[2m−1]` (product-to-sum), so `X[1] = G[0] / 2` and
/// `X[2m+1] = G[m] − X[2m−1]`. `scratch` holds `x.len()` values.
/// The transforms the encoder runs go through [`dct_lanes`]; this one
/// is its reference.
#[cfg(test)]
fn dct_in_place(x: &mut [f32], scratch: &mut [f32]) {
    let n = x.len();
    if n == 8 {
        // The recursion written out: every larger size ends here.
        let c = half_cosines(3);
        let k = half_cosines(2);
        let s = [x[0] + x[7], x[1] + x[6], x[2] + x[5], x[3] + x[4]];
        let d = [
            (x[0] - x[7]) * c[0],
            (x[1] - x[6]) * c[1],
            (x[2] - x[5]) * c[2],
            (x[3] - x[4]) * c[3],
        ];
        let e = dct4(s, k);
        let g = dct4(d, k);
        let x1 = g[0] * 0.5;
        let x3 = g[1] - x1;
        let x5 = g[2] - x3;
        *x.first_chunk_mut::<8>().expect("eight samples") =
            [e[0], x1, e[1], x3, e[2], x5, e[3], g[3] - x5];
        return;
    }
    if n == 4 {
        let out = dct4([x[0], x[1], x[2], x[3]], half_cosines(2));
        x.copy_from_slice(&out);
        return;
    }
    if n == 1 {
        return;
    }
    if n == 2 {
        let (a, b) = (x[0], x[1]);
        x[0] = a + b;
        x[1] = (a - b) * std::f32::consts::FRAC_1_SQRT_2;
        return;
    }
    let half = n / 2;
    let cos = half_cosines(n.trailing_zeros() as usize);
    let (lo, hi) = scratch.split_at_mut(half);
    for i in 0..half {
        let (a, b) = (x[i], x[n - 1 - i]);
        lo[i] = a + b;
        hi[i] = (a - b) * cos[i];
    }
    {
        let (xa, xb) = x.split_at_mut(half);
        dct_in_place(lo, xa);
        dct_in_place(&mut hi[..half], xb);
    }
    let mut prev = hi[0] * 0.5;
    x[0] = lo[0];
    x[1] = prev;
    for m in 1..half {
        x[2 * m] = lo[m];
        prev = hi[m] - prev;
        x[2 * m + 1] = prev;
    }
}

/// [`dct4`] on `lanes` independent transforms at once: `a[i]` holds
/// sample `i` of every transform.
#[inline(always)]
fn dct4_lanes(a: [&[f32]; 4], out: [&mut [f32]; 4], k: &[f32]) {
    const R: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let [a0, a1, a2, a3] = a;
    let [o0, o1, o2, o3] = out;
    let (k0, k1) = (k[0], k[1]);
    for l in 0..o0.len() {
        let (s0, s1) = (a0[l] + a3[l], a1[l] + a2[l]);
        let (d0, d1) = ((a0[l] - a3[l]) * k0, (a1[l] - a2[l]) * k1);
        let x1 = (d0 + d1) * 0.5;
        o0[l] = s0 + s1;
        o1[l] = x1;
        o2[l] = (s0 - s1) * R;
        o3[l] = (d0 - d1) * R - x1;
    }
}

/// [`dct_in_place`] on `lanes` independent transforms of `n` samples
/// at once: `x[i * lanes + l]` is sample `i` of transform `l`. Every
/// transform sees exactly the operations of the scalar routine, in
/// the same order — the results are bit-identical — but each step
/// runs across the lanes, which is a loop the compiler vectorises.
/// `scratch` holds `x.len()` values.
fn dct_lanes(x: &mut [f32], n: usize, lanes: usize, scratch: &mut [f32]) {
    debug_assert_eq!(x.len(), n * lanes);
    if n == 1 || lanes == 0 {
        return;
    }
    if n == 2 {
        let (a, b) = x.split_at_mut(lanes);
        for (a, b) in a.iter_mut().zip(b) {
            let (p, q) = (*a, *b);
            *a = p + q;
            *b = (p - q) * std::f32::consts::FRAC_1_SQRT_2;
        }
        return;
    }
    if n == 4 {
        let k = half_cosines(2);
        let src = &mut scratch[..4 * lanes];
        src.copy_from_slice(x);
        let (s0, r) = src.split_at(lanes);
        let (s1, r) = r.split_at(lanes);
        let (s2, s3) = r.split_at(lanes);
        let (o0, r) = x.split_at_mut(lanes);
        let (o1, r) = r.split_at_mut(lanes);
        let (o2, o3) = r.split_at_mut(lanes);
        dct4_lanes([s0, s1, s2, s3], [o0, o1, o2, o3], k);
        return;
    }
    if n == 8 {
        // The recursion written out, as in the scalar routine.
        let c = half_cosines(3);
        let k = half_cosines(2);
        const R: f32 = std::f32::consts::FRAC_1_SQRT_2;
        let (c0, c1, c2, c3) = (c[0], c[1], c[2], c[3]);
        let (k0, k1) = (k[0], k[1]);
        let (x0, r) = x.split_at_mut(lanes);
        let (x1, r) = r.split_at_mut(lanes);
        let (x2, r) = r.split_at_mut(lanes);
        let (x3, r) = r.split_at_mut(lanes);
        let (x4, r) = r.split_at_mut(lanes);
        let (x5, r) = r.split_at_mut(lanes);
        let (x6, x7) = r.split_at_mut(lanes);
        for l in 0..lanes {
            let s = [x0[l] + x7[l], x1[l] + x6[l], x2[l] + x5[l], x3[l] + x4[l]];
            let d = [
                (x0[l] - x7[l]) * c0,
                (x1[l] - x6[l]) * c1,
                (x2[l] - x5[l]) * c2,
                (x3[l] - x4[l]) * c3,
            ];
            // dct4(s)
            let (ss0, ss1) = (s[0] + s[3], s[1] + s[2]);
            let (sd0, sd1) = ((s[0] - s[3]) * k0, (s[1] - s[2]) * k1);
            let e1 = (sd0 + sd1) * 0.5;
            let e = [ss0 + ss1, e1, (ss0 - ss1) * R, (sd0 - sd1) * R - e1];
            // dct4(d)
            let (ds0, ds1) = (d[0] + d[3], d[1] + d[2]);
            let (dd0, dd1) = ((d[0] - d[3]) * k0, (d[1] - d[2]) * k1);
            let g1 = (dd0 + dd1) * 0.5;
            let g = [ds0 + ds1, g1, (ds0 - ds1) * R, (dd0 - dd1) * R - g1];
            let y1 = g[0] * 0.5;
            let y3 = g[1] - y1;
            let y5 = g[2] - y3;
            x0[l] = e[0];
            x1[l] = y1;
            x2[l] = e[1];
            x3[l] = y3;
            x4[l] = e[2];
            x5[l] = y5;
            x6[l] = e[3];
            x7[l] = g[3] - y5;
        }
        return;
    }
    let half = n / 2;
    let cos = half_cosines(n.trailing_zeros() as usize);
    let (lo, hi) = scratch[..n * lanes].split_at_mut(half * lanes);
    for i in 0..half {
        let ci = cos[i];
        let a = &x[i * lanes..(i + 1) * lanes];
        let b = &x[(n - 1 - i) * lanes..(n - i) * lanes];
        let lo = &mut lo[i * lanes..(i + 1) * lanes];
        let hi = &mut hi[i * lanes..(i + 1) * lanes];
        for l in 0..lanes {
            lo[l] = a[l] + b[l];
            hi[l] = (a[l] - b[l]) * ci;
        }
    }
    {
        let (xa, xb) = x.split_at_mut(half * lanes);
        dct_lanes(lo, half, lanes, xa);
        dct_lanes(hi, half, lanes, xb);
    }
    // x[0] = lo[0]; x[1] = hi[0] / 2; x[2m] = lo[m];
    // x[2m + 1] = hi[m] − x[2m − 1].
    x[..lanes].copy_from_slice(&lo[..lanes]);
    for l in 0..lanes {
        x[lanes + l] = hi[l] * 0.5;
    }
    for m in 1..half {
        x[2 * m * lanes..(2 * m + 1) * lanes].copy_from_slice(&lo[m * lanes..(m + 1) * lanes]);
        let (before, after) = x.split_at_mut((2 * m + 1) * lanes);
        let prev = &before[(2 * m - 1) * lanes..2 * m * lanes];
        let hi = &hi[m * lanes..(m + 1) * lanes];
        for l in 0..lanes {
            after[l] = hi[l] - prev[l];
        }
    }
}

/// Forward DCT of the `rows × cols` pixel block at `px[r * stride + c]`
/// into `out` (`rows × cols` coefficients in the decoder's layout),
/// scaled as I.7.2: coefficient 0 is the block mean, the others carry
/// a factor `sqrt(2)` per non-zero frequency. `tmp` is scratch space.
///
/// Both passes run every one-dimensional transform of the block side
/// by side ([`dct_lanes`]): first along the rows, all rows at once,
/// then along the columns, all columns at once.
pub(crate) fn fdct(
    px: &[f32],
    stride: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
    tmp: &mut Vec<f32>,
) {
    debug_assert!(rows.is_power_of_two() && cols.is_power_of_two());
    debug_assert!(rows <= MAX_SIDE && cols <= MAX_SIDE);
    let n = rows * cols;
    tmp.clear();
    tmp.resize(3 * n, 0.0);
    let (a, rest) = tmp.split_at_mut(n);
    let (b, scratch) = rest.split_at_mut(n);
    // Along the rows: `a[i * rows + r]` is sample `i` of row `r`, and
    // afterwards coefficient `i` of row `r`.
    for r in 0..rows {
        let line = &px[r * stride..r * stride + cols];
        for (i, &v) in line.iter().enumerate() {
            a[i * rows + r] = v;
        }
    }
    dct_lanes(a, cols, rows, scratch);
    // Along the columns: `b[r * cols + u]` is sample `r` of column
    // `u`, and afterwards coefficient `r` of column `u`.
    for u in 0..cols {
        let column = &a[u * rows..(u + 1) * rows];
        for (r, &v) in column.iter().enumerate() {
            b[r * cols + u] = v;
        }
    }
    dct_lanes(b, rows, cols, scratch);
    // Written at the decoder's index: `fs × long + fl`, `fs` the
    // frequency along the shorter side (the horizontal one for a
    // square block).
    let long = rows.max(cols);
    let wide = cols > rows;
    let norm = 1.0 / (rows * cols) as f32;
    for u in 0..cols {
        let su = if u == 0 {
            norm
        } else {
            norm * std::f32::consts::SQRT_2
        };
        for v in 0..rows {
            let s = if v == 0 {
                su
            } else {
                su * std::f32::consts::SQRT_2
            };
            let idx = if wide { v * long + u } else { u * long + v };
            out[idx] = b[v * cols + u] * s;
        }
    }
}

/// The forward DCT one transform at a time: the reference [`fdct`] is
/// checked against, bit for bit. Of the `rows × cols` pixel block at
/// `px[r * stride + c]` into `out` (`rows × cols` coefficients in the decoder's layout),
/// scaled as I.7.2: coefficient 0 is the block mean, the others carry
/// a factor `sqrt(2)` per non-zero frequency. `tmp` is scratch space.
#[cfg(test)]
fn fdct_scalar(
    px: &[f32],
    stride: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
    tmp: &mut Vec<f32>,
) {
    debug_assert!(rows.is_power_of_two() && cols.is_power_of_two());
    debug_assert!(rows <= MAX_SIDE && cols <= MAX_SIDE);
    tmp.clear();
    tmp.resize(rows * cols, 0.0);
    let mut line = [0.0f32; MAX_SIDE];
    let mut scratch = [0.0f32; MAX_SIDE];
    // Rows, stored column-major (`tmp[u * rows + r]`) so the second
    // pass works on contiguous columns.
    for r in 0..rows {
        line[..cols].copy_from_slice(&px[r * stride..r * stride + cols]);
        dct_in_place(&mut line[..cols], &mut scratch[..cols]);
        for u in 0..cols {
            tmp[u * rows + r] = line[u];
        }
    }
    // Columns, written at the decoder's index: `fs × long + fl`, `fs`
    // the frequency along the shorter side (the horizontal one for a
    // square block).
    let long = rows.max(cols);
    let wide = cols > rows;
    let norm = 1.0 / (rows * cols) as f32;
    for u in 0..cols {
        let column = &mut tmp[u * rows..(u + 1) * rows];
        dct_in_place(column, &mut scratch[..rows]);
        let su = if u == 0 {
            norm
        } else {
            norm * std::f32::consts::SQRT_2
        };
        for (v, &value) in column.iter().enumerate() {
            let s = if v == 0 {
                su
            } else {
                su * std::f32::consts::SQRT_2
            };
            let idx = if wide { v * long + u } else { u * long + v };
            out[idx] = value * s;
        }
    }
}

/// Invert the `n × n` row-major matrix `m` (Gauss–Jordan with partial
/// pivoting, in `f64`). `None` when singular.
fn invert(m: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = m.to_vec();
    let mut inv = vec![0.0f64; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for col in 0..n {
        let mut pivot = col;
        for r in col + 1..n {
            if a[r * n + col].abs() > a[pivot * n + col].abs() {
                pivot = r;
            }
        }
        if a[pivot * n + col].abs() < 1e-12 {
            return None;
        }
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
                inv.swap(col * n + k, pivot * n + k);
            }
        }
        let d = 1.0 / a[col * n + col];
        for k in 0..n {
            a[col * n + k] *= d;
            inv[col * n + k] *= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r * n + col];
            if f == 0.0 {
                continue;
            }
            for k in 0..n {
                a[r * n + k] -= f * a[col * n + k];
                inv[r * n + k] -= f * inv[col * n + k];
            }
        }
    }
    Some(inv)
}

/// An 8×8 transform that is not a plain DCT, as matrices.
pub(crate) struct SmallTransform {
    /// The forward matrix, sparse: coefficient `k` is the sum of
    /// `weight × pixel[index]` over `terms[starts[k]..starts[k + 1]]`
    /// (these transforms act on sub-blocks, so most of the dense
    /// 64 × 64 matrix is zero).
    terms: Vec<(u8, f32)>,
    starts: [u16; 65],
    /// Squared pixel-domain norm of each coefficient's basis function.
    pub norm2: [f32; 64],
}

/// The 8×8 transforms handled through [`SmallTransform`], in the order
/// of their slots.
pub(crate) const SMALL_TRANSFORMS: [TransformType; 9] = [
    TransformType::Hornuss,
    TransformType::Dct2x2,
    TransformType::Dct4x4,
    TransformType::Dct4x8,
    TransformType::Dct8x4,
    TransformType::Afv0,
    TransformType::Afv1,
    TransformType::Afv2,
    TransformType::Afv3,
];

/// The forward matrix of a non-DCT 8×8 transform (`None` for the DCT
/// family).
pub(crate) fn small_transform(t: TransformType) -> Option<&'static SmallTransform> {
    static SMALL: OnceLock<Vec<SmallTransform>> = OnceLock::new();
    let slot = SMALL_TRANSFORMS.iter().position(|&s| s == t)?;
    let all = SMALL.get_or_init(|| {
        SMALL_TRANSFORMS
            .iter()
            .map(|&t| {
                // Synthesis matrix: column k is the decoder's output
                // for the k-th unit coefficient vector.
                let mut synth = vec![0.0f64; 64 * 64];
                let mut norm2 = [0.0f32; 64];
                let mut unit = [0.0f32; 64];
                for k in 0..64 {
                    unit[k] = 1.0;
                    let px = idct_for_transform(t, &unit).unwrap_or_else(|_| vec![0.0; 64]);
                    unit[k] = 0.0;
                    let mut n2 = 0.0f64;
                    for (p, &v) in px.iter().enumerate().take(64) {
                        synth[p * 64 + k] = f64::from(v);
                        n2 += f64::from(v) * f64::from(v);
                    }
                    norm2[k] = n2 as f32;
                }
                let forward = invert(&synth, 64).unwrap_or_else(|| vec![0.0f64; 64 * 64]);
                let mut terms = Vec::new();
                let mut starts = [0u16; 65];
                for k in 0..64 {
                    for p in 0..64 {
                        let v = forward[k * 64 + p];
                        if v.abs() > 1e-9 {
                            terms.push((p as u8, v as f32));
                        }
                    }
                    starts[k + 1] = terms.len() as u16;
                }
                SmallTransform {
                    terms,
                    starts,
                    norm2,
                }
            })
            .collect()
    });
    Some(&all[slot])
}

/// How much of the energy of each coefficient's basis function
/// survives the symmetric 3×3 smoothing kernel `[centre, edge,
/// corner]` (J.2): the squared norm of the smoothed basis function
/// over that of the function itself, the block mirrored at its edges.
/// For the DCT family this is the squared transfer function of the
/// kernel at the coefficient's frequencies; the other 8×8 transforms
/// are measured on their basis images.
pub(crate) fn smoothing_gain(t: TransformType, kernel: [f32; 3]) -> Vec<f32> {
    let [centre, edge, corner] = kernel.map(f64::from);
    if let Some((rows, cols)) = crate::idct::dct_pixel_dims(t) {
        let long = rows.max(cols);
        let wide = cols > rows;
        return (0..rows * cols)
            .map(|idx| {
                let (a, b) = (idx / long, idx % long);
                let (u, v) = if wide { (b, a) } else { (a, b) };
                let cx = (std::f64::consts::PI * u as f64 / cols as f64).cos();
                let cy = (std::f64::consts::PI * v as f64 / rows as f64).cos();
                let h = centre + 2.0 * edge * (cx + cy) + 4.0 * corner * cx * cy;
                (h * h) as f32
            })
            .collect();
    }
    let mirror = |i: i32| -> usize { i.clamp(0, 7) as usize };
    let mut unit = [0.0f32; 64];
    (0..64)
        .map(|k| {
            unit[k] = 1.0;
            let px = idct_for_transform(t, &unit).unwrap_or_else(|_| vec![0.0; 64]);
            unit[k] = 0.0;
            let at = |x: i32, y: i32| f64::from(px[mirror(y) * 8 + mirror(x)]);
            let (mut before, mut after) = (0.0f64, 0.0f64);
            for y in 0..8i32 {
                for x in 0..8i32 {
                    let s = centre * at(x, y)
                        + edge * (at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1))
                        + corner
                            * (at(x - 1, y - 1)
                                + at(x + 1, y - 1)
                                + at(x - 1, y + 1)
                                + at(x + 1, y + 1));
                    before += at(x, y) * at(x, y);
                    after += s * s;
                }
            }
            if before > 0.0 {
                (after / before) as f32
            } else {
                1.0
            }
        })
        .collect()
}

/// Forward transform of an 8×8 block through a [`SmallTransform`].
pub(crate) fn forward_small(st: &SmallTransform, px: &[f32], stride: usize, out: &mut [f32]) {
    let mut block = [0.0f32; 64];
    for r in 0..8 {
        block[r * 8..r * 8 + 8].copy_from_slice(&px[r * stride..r * stride + 8]);
    }
    for (k, slot) in out.iter_mut().enumerate().take(64) {
        let row = &st.terms[usize::from(st.starts[k])..usize::from(st.starts[k + 1])];
        let mut s = 0.0f32;
        for &(p, weight) in row {
            s += weight * block[usize::from(p)];
        }
        *slot = s;
    }
}

/// The geometry of a transform's coefficient block and its LF link.
pub(crate) struct LfLink {
    /// Varblock width in 8×8 cells: LF sample `j` is at column
    /// `j % cols`, row `j / cols` of the varblock.
    pub cols: usize,
    /// Index of each LLF coefficient in the coefficient block.
    pub cells: Vec<usize>,
    /// `lf[j] = sum_i from_llf[j * n + i] × coefficient[cells[i]]`,
    /// `lf` being the `rows × cols` LF samples of the varblock in
    /// image order.
    pub from_llf: Vec<f32>,
}

/// Build the LF link of `t`: invert the decoder's LF → LLF map.
pub(crate) fn lf_link(t: TransformType) -> LfLink {
    let (cols, rows) = t.block_dims();
    let (cols, rows) = (cols as usize, rows as usize);
    let n = cols * rows;
    let (bwidth, bheight) = varblock_size_for_order(order_id_for_transform(t));
    let (bwidth, bheight) = (bwidth as usize, bheight as usize);
    let (wcx, wcy) = (bwidth / 8, bheight / 8);
    let mut cells = Vec::with_capacity(n);
    for y in 0..wcy {
        for x in 0..wcx {
            cells.push(y * bwidth + x);
        }
    }
    if n == 1 {
        return LfLink {
            cols,
            cells,
            from_llf: vec![1.0],
        };
    }
    // Forward map: column j is the LLF block of the j-th unit LF
    // sample vector, read the way the decoder places it.
    let mut fwd = vec![0.0f64; n * n];
    let mut unit = vec![0.0f32; n];
    for j in 0..n {
        unit[j] = 1.0;
        let llf = llf_from_lf(&unit, t).unwrap_or_else(|_| vec![0.0; n]);
        unit[j] = 0.0;
        // Placed exactly as the decoder places it.
        let mut block = vec![0.0f32; bwidth * bheight];
        if crate::block_dequant::merge_llf_into_block(&mut block, t, &llf).is_err() {
            continue;
        }
        for (i, &cell) in cells.iter().enumerate() {
            fwd[i * n + j] = f64::from(block[cell]);
        }
    }
    let from_llf = invert(&fwd, n)
        .map(|inv| inv.iter().map(|&v| v as f32).collect())
        .unwrap_or_else(|| vec![0.0f32; n * n]);
    LfLink {
        cols,
        cells,
        from_llf,
    }
}

/// Pixel dimensions `(rows, cols)` of a transform's block.
#[cfg(test)]
pub(crate) fn pixel_dims(t: TransformType) -> (usize, usize) {
    crate::idct::dct_pixel_dims(t).unwrap_or((8, 8))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|i| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / (1 << 24) as f32 - 0.5 + (i % 7) as f32 * 0.05
            })
            .collect()
    }

    const DCTS: [TransformType; 12] = [
        TransformType::Dct8x8,
        TransformType::Dct16x16,
        TransformType::Dct32x32,
        TransformType::Dct16x8,
        TransformType::Dct8x16,
        TransformType::Dct32x8,
        TransformType::Dct8x32,
        TransformType::Dct32x16,
        TransformType::Dct16x32,
        TransformType::Dct64x64,
        TransformType::Dct64x32,
        TransformType::Dct32x64,
    ];

    #[test]
    fn forward_dct_is_inverted_by_the_decoder_for_every_size() {
        let mut tmp = Vec::new();
        for (i, &t) in DCTS.iter().enumerate() {
            let (rows, cols) = pixel_dims(t);
            let px = noise(rows * cols, 77 + i as u32);
            let mut coeffs = vec![0.0f32; rows * cols];
            fdct(&px, cols, rows, cols, &mut coeffs, &mut tmp);
            let mean = px.iter().sum::<f32>() / px.len() as f32;
            assert!((coeffs[0] - mean).abs() < 1e-4, "{t:?}: DC is the mean");
            let back = idct_for_transform(t, &coeffs).unwrap();
            for (p, (&a, &b)) in back.iter().zip(&px).enumerate() {
                assert!((a - b).abs() < 2e-3, "{t:?} sample {p}: {a} vs {b}");
            }
        }
    }

    /// The lane-batched DCT is the scalar one bit for bit, at every
    /// block shape the encoder uses.
    #[test]
    fn batched_dct_is_bit_identical_to_the_scalar_one() {
        let sides = [1usize, 2, 4, 8, 16, 32, 64];
        let (mut tmp, mut tmp2) = (Vec::new(), Vec::new());
        for &rows in &sides {
            for &cols in &sides {
                for seed in 0..3u32 {
                    let stride = cols + 5;
                    let px = noise(rows * stride, 7 + seed + (rows * 131 + cols) as u32);
                    let mut fast = vec![0.0f32; rows * cols];
                    let mut slow = vec![0.0f32; rows * cols];
                    fdct(&px, stride, rows, cols, &mut fast, &mut tmp);
                    fdct_scalar(&px, stride, rows, cols, &mut slow, &mut tmp2);
                    for (i, (f, s)) in fast.iter().zip(&slow).enumerate() {
                        assert_eq!(f.to_bits(), s.to_bits(), "{rows}x{cols} coefficient {i}");
                    }
                }
            }
        }
    }

    #[test]
    fn small_transforms_are_inverted_by_the_decoder() {
        for (i, &t) in SMALL_TRANSFORMS.iter().enumerate() {
            let st = small_transform(t).unwrap();
            let px = noise(64, 5 + i as u32);
            let mut coeffs = vec![0.0f32; 64];
            forward_small(st, &px, 8, &mut coeffs);
            let back = idct_for_transform(t, &coeffs).unwrap();
            for (p, (&a, &b)) in back.iter().zip(&px).enumerate() {
                assert!((a - b).abs() < 2e-3, "{t:?} sample {p}: {a} vs {b}");
            }
            let mean = px.iter().sum::<f32>() / 64.0;
            assert!(
                (coeffs[0] - mean).abs() < 2e-3,
                "{t:?}: coefficient 0 is the block mean ({} vs {mean})",
                coeffs[0]
            );
        }
    }

    #[test]
    fn lf_link_inverts_the_decoder_llf_map() {
        for &t in &DCTS {
            let link = lf_link(t);
            let n = link.cells.len();
            let llf_in = noise(n, 3);
            let mut lf = vec![0.0f32; n];
            for (j, slot) in lf.iter_mut().enumerate() {
                *slot = (0..n).map(|i| link.from_llf[j * n + i] * llf_in[i]).sum();
            }
            let llf = llf_from_lf(&lf, t).unwrap();
            let (rows, cols) = pixel_dims(t);
            let mut block = vec![0.0f32; rows * cols];
            crate::block_dequant::merge_llf_into_block(&mut block, t, &llf).unwrap();
            for i in 0..n {
                let got = block[link.cells[i]];
                assert!(
                    (got - llf_in[i]).abs() < 1e-3,
                    "{t:?} cell {i}: {got} vs {}",
                    llf_in[i]
                );
            }
        }
    }

    /// A varblock coded as its exact forward coefficients, the LLF
    /// cells going through the LF link, reconstructs the pixels: the
    /// LF samples the link asks for regenerate the LLF cells.
    #[test]
    fn llf_cells_round_trip_through_the_lf_image() {
        let mut tmp = Vec::new();
        for &t in &DCTS {
            let (rows, cols) = pixel_dims(t);
            let px = noise(rows * cols, 41);
            let mut coeffs = vec![0.0f32; rows * cols];
            fdct(&px, cols, rows, cols, &mut coeffs, &mut tmp);
            let link = lf_link(t);
            let n = link.cells.len();
            let mut lf = vec![0.0f32; n];
            for (j, slot) in lf.iter_mut().enumerate() {
                *slot = (0..n)
                    .map(|i| link.from_llf[j * n + i] * coeffs[link.cells[i]])
                    .sum();
            }
            let llf = llf_from_lf(&lf, t).unwrap();
            let mut block = coeffs.clone();
            for &c in &link.cells {
                block[c] = 0.0;
            }
            crate::block_dequant::merge_llf_into_block(&mut block, t, &llf).unwrap();
            let back = idct_for_transform(t, &block).unwrap();
            for (p, (&a, &b)) in back.iter().zip(&px).enumerate() {
                assert!((a - b).abs() < 3e-3, "{t:?} sample {p}: {a} vs {b}");
            }
        }
    }
}
