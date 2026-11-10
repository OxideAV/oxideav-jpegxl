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

/// The channel list after the elected transforms.
pub(crate) struct Planned {
    pub transforms: Vec<Transform>,
    /// Leading meta channels (palettes), in final order.
    pub n_meta: usize,
    pub chans: Vec<Chan>,
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
fn choose_rct(r: &[i32], g: &[i32], b: &[i32], width: usize, height: usize) -> u32 {
    if width < 2 || height < 2 {
        return 0;
    }
    let row_step = (height / 48).max(1);
    let col_step = (width / 1024).max(1);
    let mut best = (f64::INFINITY, 0u32);
    for rct_type in 0..42u32 {
        let perm = (rct_type / 7) as usize;
        let typ = rct_type % 7;
        let (i0, i1, i2) = (
            perm % 3,
            (perm + 1 + perm / 3) % 3,
            (perm + 2 - perm / 3) % 3,
        );
        let at = |x: usize, y: usize| -> [i32; 3] {
            let i = y * width + x;
            let v = [r[i], g[i], b[i]];
            let (a, bb, c) = rct_forward_pixel(v[i0], v[i1], v[i2], typ);
            [a, bb, c]
        };
        let mut hist = [[0u32; 128]; 3];
        let mut extra = 0u64;
        let mut n = 0u32;
        let mut y = 1;
        while y < height {
            let mut x = 1;
            while x < width {
                let (c, w, nn, nw) = (at(x, y), at(x - 1, y), at(x, y - 1), at(x - 1, y - 1));
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
            y += row_step;
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

/// Estimated coded bits of `chans` (same geometry): zero-order entropy
/// of the gradient-predicted residuals of a sample of rows, scaled to
/// the whole channel.
fn quick_cost(chans: &[&[i32]], width: usize, height: usize) -> f64 {
    if width < 2 || height < 2 {
        return (width * height * chans.len()) as f64 * 8.0;
    }
    let row_step = (height / 96).max(1);
    let mut total = 0.0;
    for d in chans {
        let mut hist = [0u32; 128];
        let mut extra = 0u64;
        let mut n = 0u32;
        let mut y = 1;
        while y < height {
            let row = y * width;
            for x in 1..width {
                let (w, nn, nw) = (d[row + x - 1], d[row - width + x], d[row - width + x - 1]);
                let p = w
                    .wrapping_add(nn)
                    .wrapping_sub(nw)
                    .clamp(w.min(nn), w.max(nn));
                let (t, e) = estimate_token(pack_signed(d[row + x].wrapping_sub(p)));
                hist[t] += 1;
                extra += u64::from(e);
                n += 1;
            }
            y += row_step;
        }
        let mut cost = extra as f64 + f64::from(n) * f64::from(n.max(1)).log2();
        for &c in &hist {
            if c > 1 {
                cost -= f64::from(c) * f64::from(c).log2();
            }
        }
        total += cost * (width * height) as f64 / f64::from(n.max(1));
    }
    total
}

/// The distinct pixels of `chans` (at most four channels of at most 16
/// bits) with their counts, or `None` beyond `limit` colours.
fn distinct_colours(chans: &[Chan], limit: usize) -> Option<HashMap<u64, u32>> {
    let n = chans[0].data.len();
    let mut map: HashMap<u64, u32> = HashMap::new();
    let mut last = u64::MAX;
    for i in 0..n {
        let mut key = 0u64;
        for c in chans {
            key = (key << 16) | (c.data[i] as u64 & 0xFFFF);
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
    Some(map)
}

/// Replace every channel by one index channel over a brightness-sorted
/// colour table (placed in front as a meta channel).
fn apply_full_palette(chans: &mut Vec<Chan>, colours: &HashMap<u64, u32>) -> Transform {
    let num_c = chans.len();
    let (w, h) = (chans[0].w, chans[0].h);
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
    let mut idx = Vec::with_capacity(w * h);
    let mut last = (u64::MAX, 0i32);
    for i in 0..w * h {
        let mut key = 0u64;
        for c in chans.iter() {
            key = (key << 16) | (c.data[i] as u64 & 0xFFFF);
        }
        if key != last.0 {
            last = (key, index[&key]);
        }
        idx.push(last.1);
    }
    let nb = table.len();
    let mut meta = Vec::with_capacity(nb * num_c);
    for c in 0..num_c {
        meta.extend(table.iter().map(|&k| component(k, c)));
    }
    let mut meta = Chan::new(nb, num_c, meta);
    meta.hshift = -1;
    meta.vshift = -1;
    chans.clear();
    chans.push(meta);
    chans.push(Chan::new(w, h, idx));
    Transform::Palette {
        begin_c: 0,
        num_c: num_c as u32,
        nb_colours: nb as u32,
    }
}

/// The sorted distinct values of a channel when they are sparse enough
/// for a channel palette to be worth trying.
fn sparse_values(ch: &Chan) -> Option<Vec<i32>> {
    let (mut lo, mut hi) = (i32::MAX, i32::MIN);
    for &v in &ch.data {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let span = (i64::from(hi) - i64::from(lo) + 1) as usize;
    if !(4..=1 << 20).contains(&span) {
        return None;
    }
    let mut seen = vec![false; span];
    let mut count = 0usize;
    for &v in &ch.data {
        let slot = &mut seen[(v - lo) as usize];
        if !*slot {
            *slot = true;
            count += 1;
        }
    }
    // Worth it only when at most half of the spanned range is used,
    // there is something to compact, and the table is small next to
    // the channel.
    if count < 2 || count * 2 > span || count * 16 > ch.data.len() || count > 60_000 {
        return None;
    }
    Some(
        (0..span)
            .filter(|&v| seen[v])
            .map(|v| v as i32 + lo)
            .collect(),
    )
}

/// Elect and apply the transforms for a frame of `n_colour` colour
/// channels (1 or 3) followed by the extra channels.
pub(crate) fn plan_transforms(
    mut chans: Vec<Chan>,
    n_colour: usize,
    bits: u32,
    lossless: bool,
) -> Planned {
    let (w, h) = (chans[0].w, chans[0].h);
    let pixels = w * h;
    let mut transforms = Vec::new();
    let rct_type = if n_colour == 3 {
        choose_rct(&chans[0].data, &chans[1].data, &chans[2].data, w, h)
    } else {
        0
    };
    let palettes = lossless && pixels >= 64 && chans.len() <= 4 && bits <= 16;

    // A palette over every channel.
    if palettes && chans.len() > 1 {
        let limit = (pixels / 8).clamp(2, 4096);
        if let Some(colours) = distinct_colours(&chans, limit) {
            let mut plain: Vec<Vec<i32>> = chans.iter().map(|c| c.data.clone()).collect();
            if rct_type != 0 {
                let (r, rest) = plain.split_at_mut(1);
                let (g, rest) = rest.split_at_mut(1);
                forward_rct(&mut r[0], &mut g[0], &mut rest[0], rct_type);
            }
            let refs: Vec<&[i32]> = plain.iter().map(|v| v.as_slice()).collect();
            let plain_cost = quick_cost(&refs, w, h);
            let table_bits = (colours.len() * chans.len()) as f64 * f64::from(bits);
            let mut trial = chans.clone();
            let t = apply_full_palette(&mut trial, &colours);
            // The index channel is modelled from its neighbours far
            // better than the gradient estimate says; the factor was
            // settled on the corpus (screen content and pixel art).
            let cost = 0.75 * quick_cost(&[&trial[1].data], w, h) + table_bits;
            if colours.len() <= 64 || cost < plain_cost {
                return Planned {
                    transforms: vec![t],
                    n_meta: 1,
                    chans: trial,
                };
            }
        }
    }

    if rct_type != 0 {
        let (r, rest) = chans.split_at_mut(1);
        let (g, rest) = rest.split_at_mut(1);
        forward_rct(&mut r[0].data, &mut g[0].data, &mut rest[0].data, rct_type);
        transforms.push(Transform::Rct {
            begin_c: 0,
            rct_type,
        });
    }

    // Channel palettes on what the RCT left: a channel whose values
    // are a sparse subset of their span is coded as ranks.
    let mut metas: Vec<Chan> = Vec::new();
    if palettes {
        for (c, chan) in chans.iter_mut().enumerate() {
            let Some(vals) = sparse_values(chan) else {
                continue;
            };
            let lo = vals[0];
            let span = (vals[vals.len() - 1] - lo + 1) as usize;
            let mut rank = vec![0i32; span];
            for (i, &v) in vals.iter().enumerate() {
                rank[(v - lo) as usize] = i as i32;
            }
            let compact: Vec<i32> = chan.data.iter().map(|&v| rank[(v - lo) as usize]).collect();
            // The table is a sorted list: a few bits per entry.
            let table_bits = vals.len() as f64 * 8.0;
            if quick_cost(&[&compact], w, h) + table_bits >= quick_cost(&[&chan.data], w, h) {
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
            chan.data = compact;
        }
    }
    let n_meta = metas.len();
    metas.append(&mut chans);
    Planned {
        transforms,
        n_meta,
        chans: metas,
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
