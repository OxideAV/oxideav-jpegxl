//! Modular transform election (H.6, written side): which reversible
//! colour transform and which palettes go in front of the channel
//! coder.
//!
//! * A **palette over all channels** when the picture has few distinct
//!   colours: the pixels become indices into a colour table sorted by
//!   brightness, which the MA tree then models from its neighbours.
//! * A **channel palette** (a one-channel palette) for a channel that
//!   uses only a sparse subset of its value range: the samples become
//!   ranks, so residuals shrink by the sparsity.
//! * Otherwise the **RCT** whose three output channels are cheapest.
//!
//! Choices are made on an estimate — the zero-order entropy of
//! gradient-predicted residuals over a sample of rows — and are this
//! crate's own design.

use std::collections::HashMap;

use crate::bitwriter::pack_signed;
use crate::enc_source::Pixels;
use crate::modular_enc::Chan;

/// A transform of the Modular header (H.6), as written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Transform {
    Rct {
        begin_c: u32,
        rct_type: u32,
    },
    /// A palette without delta entries.
    Palette {
        begin_c: u32,
        num_c: u32,
        nb_colours: u32,
    },
}

/// Forward RCT (H.6.3) for any `rct_type` in `0..42`, in place — the
/// exact inverse of the decoder's arithmetic.
pub(crate) fn forward_rct(c0: &mut [i32], c1: &mut [i32], c2: &mut [i32], rct_type: u32) {
    let perm = (rct_type / 7) as usize;
    let typ = rct_type % 7;
    let (i0, i1, i2) = (
        perm % 3,
        (perm + 1 + perm / 3) % 3,
        (perm + 2 - perm / 3) % 3,
    );
    for i in 0..c0.len() {
        let v = [c0[i], c1[i], c2[i]];
        let (a, b, c) = rct_forward_pixel(v[i0], v[i1], v[i2], typ);
        c0[i] = a;
        c1[i] = b;
        c2[i] = c;
    }
}

/// `(d, e, f)` (the permuted input) to the coded `(A, B, C)`.
#[inline]
fn rct_forward_pixel(d: i32, e: i32, f: i32, typ: u32) -> (i32, i32, i32) {
    if typ == 6 {
        let bb = d.wrapping_sub(f);
        let tmp = f.wrapping_add(bb >> 1);
        let c = e.wrapping_sub(tmp);
        let a = tmp.wrapping_add(c >> 1);
        (a, bb, c)
    } else {
        let a = d;
        let c = if typ & 1 != 0 { f.wrapping_sub(a) } else { f };
        let b = match typ >> 1 {
            1 => e.wrapping_sub(a),
            2 => e.wrapping_sub(a.wrapping_add(f) >> 1),
            _ => e,
        };
        (a, b, c)
    }
}

/// Estimated bits of a hybrid-coded value (`split 16`, two MSBs in the
/// token): the entropy part is added by the caller from the token
/// histogram.
#[inline]
fn estimate_token(v: u32) -> (usize, u32) {
    if v < 16 {
        (v as usize, 0)
    } else {
        let l = 31 - v.leading_zeros();
        ((16 + ((l - 4) << 2) + ((v >> (l - 2)) & 3)) as usize, l - 2)
    }
}

/// Pick the RCT whose three output channels have the cheapest
/// gradient-predicted residuals on a sample of rows (0 = none).
fn choose_rct(src: &Pixels<'_>, width: usize, height: usize) -> u32 {
    if width < 2 || height < 2 {
        return 0;
    }
    let row_step = (height / 48).max(1);
    let col_step = (width / 1024).max(1);
    // The sampled rows with the row above each, three channels.
    let mut rows: Vec<[Vec<i32>; 6]> = Vec::new();
    let mut y = 1;
    while y < height {
        let mut pair: [Vec<i32>; 6] = std::array::from_fn(|_| vec![0i32; width]);
        for c in 0..3 {
            src.row(c, y - 1, 0, &mut pair[c]);
            src.row(c, y, 0, &mut pair[3 + c]);
        }
        rows.push(pair);
        y += row_step;
    }
    let mut best = (f64::INFINITY, 0u32);
    for rct_type in 0..42u32 {
        let perm = (rct_type / 7) as usize;
        let typ = rct_type % 7;
        let (i0, i1, i2) = (
            perm % 3,
            (perm + 1 + perm / 3) % 3,
            (perm + 2 - perm / 3) % 3,
        );
        let mut hist = [[0u32; 128]; 3];
        let mut extra = 0u64;
        let mut n = 0u32;
        for pair in &rows {
            let at = |x: usize, below: usize| -> [i32; 3] {
                let v = [pair[below][x], pair[below + 1][x], pair[below + 2][x]];
                let (a, bb, c) = rct_forward_pixel(v[i0], v[i1], v[i2], typ);
                [a, bb, c]
            };
            let mut x = 1;
            while x < width {
                let (c, w, nn, nw) = (at(x, 3), at(x - 1, 3), at(x, 0), at(x - 1, 0));
                for k in 0..3 {
                    let p = w[k]
                        .wrapping_add(nn[k])
                        .wrapping_sub(nw[k])
                        .clamp(w[k].min(nn[k]), w[k].max(nn[k]));
                    let (t, e) = estimate_token(pack_signed(c[k].wrapping_sub(p)));
                    hist[k][t] += 1;
                    extra += u64::from(e);
                }
                n += 1;
                x += col_step;
            }
        }
        let mut cost = extra as f64;
        for h in &hist {
            cost += f64::from(n) * f64::from(n.max(1)).log2();
            for &c in h {
                if c > 1 {
                    cost -= f64::from(c) * f64::from(c).log2();
                }
            }
        }
        // Strictly better only: ties keep the lower type.
        if cost < best.0 * 0.999 {
            best = (cost, rct_type);
        }
    }
    best.1
}

/// Estimated coded bits of `n_ch` channels of the same geometry:
/// zero-order entropy of the gradient-predicted residuals of a sample
/// of rows, scaled to the whole channel. `fill(y, rows)` writes row `y`
/// of every channel.
fn quick_cost(
    n_ch: usize,
    width: usize,
    height: usize,
    mut fill: impl FnMut(usize, &mut [Vec<i32>]),
) -> f64 {
    if width < 2 || height < 2 {
        return (width * height * n_ch) as f64 * 8.0;
    }
    let row_step = (height / 96).max(1);
    let mut hist = vec![[0u32; 128]; n_ch];
    let mut extra = vec![0u64; n_ch];
    let mut n = 0u32;
    let mut above: Vec<Vec<i32>> = vec![vec![0i32; width]; n_ch];
    let mut here: Vec<Vec<i32>> = vec![vec![0i32; width]; n_ch];
    let mut y = 1;
    while y < height {
        fill(y - 1, &mut above);
        fill(y, &mut here);
        for (c, (up, d)) in above.iter().zip(&here).enumerate() {
            for x in 1..width {
                let (w, nn, nw) = (d[x - 1], up[x], up[x - 1]);
                let p = w
                    .wrapping_add(nn)
                    .wrapping_sub(nw)
                    .clamp(w.min(nn), w.max(nn));
                let (t, e) = estimate_token(pack_signed(d[x].wrapping_sub(p)));
                hist[c][t] += 1;
                extra[c] += u64::from(e);
            }
        }
        n += (width - 1) as u32;
        y += row_step;
    }
    let mut total = 0.0;
    for (h, &e) in hist.iter().zip(&extra) {
        let mut cost = e as f64 + f64::from(n) * f64::from(n.max(1)).log2();
        for &c in h {
            if c > 1 {
                cost -= f64::from(c) * f64::from(c).log2();
            }
        }
        total += cost * (width * height) as f64 / f64::from(n.max(1));
    }
    total
}

/// The distinct pixels of a picture (at most four channels of at most
/// 16 bits) with their counts, or `None` beyond `limit` colours.
fn distinct_colours(
    src: &Pixels<'_>,
    width: usize,
    height: usize,
    limit: usize,
) -> Option<HashMap<u64, u32>> {
    let n_ch = src.num_channels();
    let mut map: HashMap<u64, u32> = HashMap::new();
    let mut last = u64::MAX;
    let mut rows: Vec<Vec<i32>> = vec![vec![0i32; width]; n_ch];
    for y in 0..height {
        for (c, row) in rows.iter_mut().enumerate() {
            src.row(c, y, 0, row);
        }
        for x in 0..width {
            let mut key = 0u64;
            for row in &rows {
                key = (key << 16) | (row[x] as u64 & 0xFFFF);
            }
            if key == last {
                continue;
            }
            last = key;
            let len = map.len();
            let slot = map.entry(key).or_insert(0);
            *slot += 1;
            if len >= limit && map.len() > limit {
                return None;
            }
        }
    }
    Some(map)
}

/// A palette over every channel: the colour table as a meta channel
/// (sorted by brightness) and the index of every colour key.
fn full_palette(colours: &HashMap<u64, u32>, num_c: usize) -> (Chan, HashMap<u64, i32>) {
    let component = |key: u64, c: usize| ((key >> (16 * (num_c - 1 - c))) & 0xFFFF) as i32;
    let mut table: Vec<u64> = colours.keys().copied().collect();
    // Brightness first so neighbouring indices are similar colours;
    // the key itself breaks ties deterministically.
    let luma = |key: u64| -> i64 {
        if num_c >= 3 {
            2 * i64::from(component(key, 0))
                + 5 * i64::from(component(key, 1))
                + i64::from(component(key, 2))
        } else {
            i64::from(component(key, 0))
        }
    };
    table.sort_unstable_by_key(|&k| (luma(k), k));
    let index: HashMap<u64, i32> = table
        .iter()
        .enumerate()
        .map(|(i, &k)| (k, i as i32))
        .collect();
    let nb = table.len();
    let mut meta = Vec::with_capacity(nb * num_c);
    for c in 0..num_c {
        meta.extend(table.iter().map(|&k| component(k, c)));
    }
    let mut meta = Chan::new(nb, num_c, meta);
    meta.hshift = -1;
    meta.vshift = -1;
    (meta, index)
}

/// The pointwise map from a picture's channels to the channels the
/// Modular coder sees after the elected transforms. Being pointwise,
/// it applies to any rectangle on its own: a group is fetched from the
/// source and mapped when a worker codes it.
#[derive(Debug, Clone, Default)]
pub(crate) struct PixelMap {
    /// Every channel replaced by one index channel.
    palette: Option<HashMap<u64, i32>>,
    /// RCT in front of the first three channels (0 = none).
    rct_type: u32,
    /// Per channel: `(lowest value, rank of each value from there)`
    /// when a channel palette compacts it.
    ranks: Vec<Option<(i32, Vec<i32>)>>,
    n_src: usize,
}

impl PixelMap {
    /// The map that changes nothing: `n_src` channels coded as they
    /// are.
    pub(crate) fn identity(n_src: usize) -> Self {
        Self {
            palette: None,
            rct_type: 0,
            ranks: vec![None; n_src],
            n_src,
        }
    }

    /// Channels the coder sees per pixel.
    pub(crate) fn num_coded(&self) -> usize {
        if self.palette.is_some() {
            1
        } else {
            self.n_src
        }
    }

    /// Rows `y` of every source channel after the RCT (no palette).
    fn rct_rows(&self, src: &Pixels<'_>, y: usize, rows: &mut [Vec<i32>]) {
        for (c, row) in rows.iter_mut().enumerate() {
            src.row(c, y, 0, row);
        }
        if self.rct_type != 0 {
            let (r, rest) = rows.split_at_mut(1);
            let (g, rest) = rest.split_at_mut(1);
            forward_rct(&mut r[0], &mut g[0], &mut rest[0], self.rct_type);
        }
    }

    /// The coded channels of the `rw × rh` rectangle at `(x0, y0)`.
    pub(crate) fn rect(
        &self,
        src: &Pixels<'_>,
        x0: usize,
        y0: usize,
        rw: usize,
        rh: usize,
    ) -> Vec<Chan> {
        let mut planes: Vec<Vec<i32>> = (0..self.n_src)
            .map(|c| src.rect(c, x0, y0, rw, rh))
            .collect();
        if let Some(index) = &self.palette {
            let mut idx = Vec::with_capacity(rw * rh);
            let mut last = (u64::MAX, 0i32);
            for i in 0..rw * rh {
                let mut key = 0u64;
                for p in &planes {
                    key = (key << 16) | (p[i] as u64 & 0xFFFF);
                }
                if key != last.0 {
                    last = (key, index[&key]);
                }
                idx.push(last.1);
            }
            return vec![Chan::new(rw, rh, idx)];
        }
        if self.rct_type != 0 {
            let (r, rest) = planes.split_at_mut(1);
            let (g, rest) = rest.split_at_mut(1);
            forward_rct(&mut r[0], &mut g[0], &mut rest[0], self.rct_type);
        }
        for (plane, rank) in planes.iter_mut().zip(&self.ranks) {
            if let Some((lo, rank)) = rank {
                for v in plane.iter_mut() {
                    *v = rank[(*v - lo) as usize];
                }
            }
        }
        planes.into_iter().map(|p| Chan::new(rw, rh, p)).collect()
    }
}

/// The elected transforms of a frame.
pub(crate) struct Planned {
    pub transforms: Vec<Transform>,
    /// The meta channels (palette tables), in final order.
    pub metas: Vec<Chan>,
    /// How a rectangle of the picture becomes coded channels.
    pub map: PixelMap,
}

/// Elect the transforms for a frame of `n_colour` colour channels (1
/// or 3) followed by the extra channels.
pub(crate) fn plan_transforms(
    src: &Pixels<'_>,
    w: usize,
    h: usize,
    n_colour: usize,
    bits: u32,
    lossless: bool,
) -> Planned {
    let n_src = src.num_channels();
    let pixels = w * h;
    let mut transforms = Vec::new();
    let rct_type = if n_colour == 3 {
        choose_rct(src, w, h)
    } else {
        0
    };
    let mut map = PixelMap {
        palette: None,
        rct_type,
        ranks: vec![None; n_src],
        n_src,
    };
    let palettes = lossless && pixels >= 64 && n_src <= 4 && bits <= 16;

    // A palette over every channel.
    if palettes && n_src > 1 {
        let limit = (pixels / 8).clamp(2, 4096);
        if let Some(colours) = distinct_colours(src, w, h, limit) {
            let plain_cost = quick_cost(n_src, w, h, |y, rows| map.rct_rows(src, y, rows));
            let table_bits = (colours.len() * n_src) as f64 * f64::from(bits);
            let (meta, index) = full_palette(&colours, n_src);
            let mut raw: Vec<Vec<i32>> = vec![vec![0i32; w]; n_src];
            // The index channel is modelled from its neighbours far
            // better than the gradient estimate says; the factor was
            // settled on the corpus (screen content and pixel art).
            let cost =
                0.75 * quick_cost(1, w, h, |y, rows| {
                    for (c, row) in raw.iter_mut().enumerate() {
                        src.row(c, y, 0, row);
                    }
                    for (x, slot) in rows[0].iter_mut().enumerate() {
                        let mut key = 0u64;
                        for row in &raw {
                            key = (key << 16) | (row[x] as u64 & 0xFFFF);
                        }
                        *slot = index[&key];
                    }
                }) + table_bits;
            if colours.len() <= 64 || cost < plain_cost {
                let nb_colours = colours.len() as u32;
                return Planned {
                    transforms: vec![Transform::Palette {
                        begin_c: 0,
                        num_c: n_src as u32,
                        nb_colours,
                    }],
                    metas: vec![meta],
                    map: PixelMap {
                        palette: Some(index),
                        rct_type: 0,
                        ranks: Vec::new(),
                        n_src,
                    },
                };
            }
        }
    }

    if rct_type != 0 {
        transforms.push(Transform::Rct {
            begin_c: 0,
            rct_type,
        });
    }

    // Channel palettes on what the RCT left: a channel whose values
    // are a sparse subset of their span is coded as ranks.
    let mut metas: Vec<Chan> = Vec::new();
    if palettes {
        // Which values each channel takes: after an RCT they lie
        // within `±2^(bits + 1)`.
        let offset = 1i32 << (bits + 1);
        let mut seen: Vec<Vec<bool>> = vec![vec![false; 2 * offset as usize]; n_src];
        let mut rows: Vec<Vec<i32>> = vec![vec![0i32; w]; n_src];
        for y in 0..h {
            map.rct_rows(src, y, &mut rows);
            for (row, seen) in rows.iter().zip(&mut seen) {
                for &v in row {
                    seen[(v + offset) as usize] = true;
                }
            }
        }
        for (c, seen) in seen.iter().enumerate() {
            let vals: Vec<i32> = (0..seen.len())
                .filter(|&i| seen[i])
                .map(|i| i as i32 - offset)
                .collect();
            let Some((&lo, &hi)) = vals.first().zip(vals.last()) else {
                continue;
            };
            let span = (hi - lo + 1) as usize;
            let count = vals.len();
            // Worth it only when at most half of the spanned range is
            // used, there is something to compact, and the table is
            // small next to the channel.
            if span < 4 || count < 2 || count * 2 > span || count * 16 > pixels || count > 60_000 {
                continue;
            }
            let mut rank = vec![0i32; span];
            for (i, &v) in vals.iter().enumerate() {
                rank[(v - lo) as usize] = i as i32;
            }
            // The table is a sorted list: a few bits per entry.
            let table_bits = vals.len() as f64 * 8.0;
            let mut all: Vec<Vec<i32>> = vec![vec![0i32; w]; n_src];
            let plain = quick_cost(1, w, h, |y, out| {
                map.rct_rows(src, y, &mut all);
                out[0].copy_from_slice(&all[c]);
            });
            let compact = quick_cost(1, w, h, |y, out| {
                map.rct_rows(src, y, &mut all);
                for (slot, &v) in out[0].iter_mut().zip(&all[c]) {
                    *slot = rank[(v - lo) as usize];
                }
            });
            if compact + table_bits >= plain {
                continue;
            }
            // Applied to the channel at its current position: every
            // earlier palette put one meta channel in front.
            transforms.push(Transform::Palette {
                begin_c: (c + metas.len()) as u32,
                num_c: 1,
                nb_colours: vals.len() as u32,
            });
            let mut meta = Chan::new(vals.len(), 1, vals);
            meta.hshift = -1;
            meta.vshift = -1;
            metas.insert(0, meta);
            map.ranks[c] = Some((lo, rank));
        }
    }
    Planned {
        transforms,
        metas,
        map,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_rct_type_is_inverted_by_the_decoder() {
        use crate::modular_fdis::{ChannelDesc, ModularImage, inverse_rct};
        let src: Vec<[i32; 3]> = vec![
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [17, 200, 99],
            [65535, 1, 65535],
        ];
        for rct_type in 0..42 {
            let mut r: Vec<i32> = src.iter().map(|p| p[0]).collect();
            let mut g: Vec<i32> = src.iter().map(|p| p[1]).collect();
            let mut b: Vec<i32> = src.iter().map(|p| p[2]).collect();
            forward_rct(&mut r, &mut g, &mut b, rct_type);
            let desc = ChannelDesc {
                width: src.len() as u32,
                height: 1,
                hshift: 0,
                vshift: 0,
            };
            let mut img = ModularImage {
                channels: vec![r, g, b],
                descs: vec![desc; 3],
            };
            inverse_rct(&mut img, 0, rct_type).unwrap();
            for (i, px) in src.iter().enumerate() {
                assert_eq!(
                    [img.channels[0][i], img.channels[1][i], img.channels[2][i]],
                    *px,
                    "rct_type {rct_type}"
                );
            }
        }
    }
}
