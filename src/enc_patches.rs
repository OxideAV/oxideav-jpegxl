//! Patch dictionary election (K.2, written side): repeated glyph-like
//! regions of a picture are coded once, in a reference frame, and
//! placed wherever they occur.
//!
//! What a decoder does with a patch is fixed by the format: it copies
//! (`kReplace`) a rectangle of a previously decoded reference frame
//! over the frame at each listed position. How the encoder finds
//! rectangles worth listing is this crate's own design:
//!
//! 1. **Runs.** Every row is cut into runs of one colour. Content that
//!    does not look like screen content — few long runs — is left
//!    alone before anything else is computed.
//! 2. **Background.** Runs of one colour that touch across rows form
//!    regions; a region larger than a glyph can be in either direction
//!    is background (page, panel, rule, the inside of a large shape).
//! 3. **Glyphs.** What is not background, grouped with its
//!    neighbours in eight directions, is a candidate when its bounding
//!    box is glyph-sized: a letter with its anti-aliased rim and its
//!    counters, an icon, a bullet.
//! 4. **Repeats.** The pixels of each bounding box are hashed; boxes
//!    with identical content are one dictionary entry. An entry is
//!    kept when placing it again is cheaper than coding its pixels
//!    again.
//! 5. **The frame without them.** Every kept occurrence is painted
//!    over with the colour around it, so the frame itself codes a flat
//!    patch there; the decoder's copy restores every pixel of the box.
//!
//! The entries are packed on shelves into the reference picture
//! (`atlas`), which the caller codes as a `kReferenceOnly` frame.

use std::collections::HashMap;

use crate::enc_source::Pixels;

/// Largest glyph bounding box side, in pixels: a region of one colour
/// wider or taller than this is background, and a component is a
/// candidate only when its box fits.
const MAX_GLYPH: usize = 48;
/// Smallest candidate box area: smaller specks cost more to place
/// than to code.
const MIN_BOX_AREA: usize = 12;
/// Screen content has runs: at most this share of the pixels may
/// start one for the search to go on (dense anti-aliased text sits
/// near 0.45, photographs and noise above 0.8).
const MAX_RUN_SHARE: f64 = 0.60;
/// Estimated bits of one more position of a patch (two coordinate
/// deltas and a blend mode).
const POSITION_BITS: f64 = 14.0;
/// Estimated bits of a dictionary entry beyond its pixels (reference,
/// origin, size, count).
const ENTRY_BITS: f64 = 34.0;
/// Estimated bits of coding one foreground pixel of a glyph in place
/// (and in the reference picture).
const PIXEL_BITS: f64 = 1.6;
/// Widest reference picture.
const MAX_ATLAS_WIDTH: usize = 2048;

/// One dictionary entry: a `w × h` rectangle at `(ax, ay)` of the
/// reference picture, copied to each of `positions` (top-left corners
/// in the frame, raster order).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PatchEntry {
    pub ax: usize,
    pub ay: usize,
    pub w: usize,
    pub h: usize,
    pub positions: Vec<(usize, usize)>,
}

/// A rectangle of the frame repainted in one colour.
#[derive(Debug, Clone, Copy)]
struct Fill {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    colour: [i32; 3],
}

/// The rectangles the frame is coded without, indexed by row band.
#[derive(Debug, Clone, Default)]
pub(crate) struct Fills {
    fills: Vec<Fill>,
    /// Indices into `fills` of the rectangles touching each band of
    /// [`Fills::BAND`] rows.
    bands: Vec<Vec<u32>>,
    /// Channels that are colour: the others are left as they are.
    n_colour: usize,
}

impl Fills {
    const BAND: usize = 32;

    fn new(fills: Vec<Fill>, height: usize, n_colour: usize) -> Self {
        let mut bands = vec![Vec::new(); height.div_ceil(Self::BAND)];
        for (i, f) in fills.iter().enumerate() {
            for band in &mut bands[f.y / Self::BAND..=(f.y + f.h - 1) / Self::BAND] {
                band.push(i as u32);
            }
        }
        Self {
            fills,
            bands,
            n_colour,
        }
    }

    /// Repaint the part of row `y` of channel `c` that `out` holds
    /// (columns from `x0`); extra channels are not touched.
    #[inline]
    pub(crate) fn apply(&self, c: usize, y: usize, x0: usize, out: &mut [i32]) {
        if c >= self.n_colour {
            return;
        }
        let Some(band) = self.bands.get(y / Self::BAND) else {
            return;
        };
        for &i in band {
            let f = &self.fills[i as usize];
            if y < f.y || y >= f.y + f.h {
                continue;
            }
            let lo = f.x.max(x0);
            let hi = (f.x + f.w).min(x0 + out.len());
            if lo < hi {
                out[lo - x0..hi - x0].fill(f.colour[c]);
            }
        }
    }
}

/// What the search found.
#[derive(Debug, Clone)]
pub(crate) struct PatchPlan {
    pub entries: Vec<PatchEntry>,
    /// The reference picture: one plane per colour channel.
    pub atlas: Vec<Vec<i32>>,
    pub atlas_width: usize,
    pub atlas_height: usize,
    /// The frame without its patches.
    pub fills: Fills,
}

#[cfg(test)]
impl PatchPlan {
    /// Positions over all entries.
    pub(crate) fn num_positions(&self) -> usize {
        self.entries.iter().map(|e| e.positions.len()).sum()
    }
}

/// A run of one colour on a row.
#[derive(Clone, Copy)]
struct Run {
    x: u32,
    len: u32,
    key: u64,
}

/// Union–find with path halving.
struct Sets {
    parent: Vec<u32>,
}

impl Sets {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
        }
    }

    fn find(&mut self, mut a: u32) -> u32 {
        while self.parent[a as usize] != a {
            let p = self.parent[a as usize];
            self.parent[a as usize] = self.parent[p as usize];
            a = self.parent[a as usize];
        }
        a
    }

    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            // The lower index stays the root: deterministic labels.
            let (lo, hi) = (ra.min(rb), ra.max(rb));
            self.parent[hi as usize] = lo;
        }
    }
}

/// A bounding box grown one run at a time.
#[derive(Clone, Copy)]
struct Bounds {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    area: u32,
}

impl Bounds {
    const EMPTY: Self = Self {
        x0: u32::MAX,
        y0: u32::MAX,
        x1: 0,
        y1: 0,
        area: 0,
    };

    fn add(&mut self, y: u32, run: &Run) {
        self.x0 = self.x0.min(run.x);
        self.y0 = self.y0.min(y);
        self.x1 = self.x1.max(run.x + run.len);
        self.y1 = self.y1.max(y + 1);
        self.area += run.len;
    }

    fn width(&self) -> usize {
        (self.x1 - self.x0) as usize
    }

    fn height(&self) -> usize {
        (self.y1 - self.y0) as usize
    }
}

/// The colour key of each pixel of a row span.
fn row_keys(src: &Pixels<'_>, n_colour: usize, y: usize, x0: usize, keys: &mut [u64]) {
    let mut line = vec![0i32; keys.len()];
    keys.fill(0);
    for c in 0..n_colour {
        src.row(c, y, x0, &mut line);
        for (k, &v) in keys.iter_mut().zip(&line) {
            *k = (*k << 16) | (v as u64 & 0xFFFF);
        }
    }
}

fn hash_keys(w: usize, h: usize, keys: &[u64]) -> u64 {
    let mut s = 0x9E37_79B9_7F4A_7C15u64 ^ ((w as u64) << 32) ^ h as u64;
    for &k in keys {
        s = (s ^ k).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        s ^= s >> 32;
    }
    s
}

/// Search `src` (its first `n_colour` channels are the colour) for
/// repeated glyph-like rectangles. `None` when the picture is not
/// screen content or nothing repeats enough to pay.
pub(crate) fn find_patches(
    src: &Pixels<'_>,
    width: usize,
    height: usize,
    n_colour: usize,
) -> Option<PatchPlan> {
    if width < 16 || height < 16 || n_colour == 0 || n_colour > 3 {
        return None;
    }
    let pixels = width * height;
    let run_budget = (pixels as f64 * MAX_RUN_SHARE) as usize;

    // 1. Runs, row by row.
    let mut runs: Vec<Run> = Vec::new();
    let mut row_start: Vec<u32> = Vec::with_capacity(height + 1);
    let mut keys = vec![0u64; width];
    for y in 0..height {
        row_start.push(runs.len() as u32);
        row_keys(src, n_colour, y, 0, &mut keys);
        let mut x = 0usize;
        while x < width {
            let key = keys[x];
            let mut end = x + 1;
            while end < width && keys[end] == key {
                end += 1;
            }
            runs.push(Run {
                x: x as u32,
                len: (end - x) as u32,
                key,
            });
            x = end;
        }
        // Not screen content: give up as soon as that is certain.
        if runs.len() > run_budget {
            return None;
        }
    }
    row_start.push(runs.len() as u32);
    let n_runs = runs.len();
    let row_runs = |y: usize| row_start[y] as usize..row_start[y + 1] as usize;

    // 2. Regions of one colour (runs touching across rows), and which
    // of them are background.
    let mut regions = Sets::new(n_runs);
    for y in 1..height {
        let (mut a, a_end) = (row_start[y - 1] as usize, row_start[y] as usize);
        for b in row_runs(y) {
            let (bx0, bx1) = (runs[b].x, runs[b].x + runs[b].len);
            while a < a_end && runs[a].x + runs[a].len <= bx0 {
                a += 1;
            }
            let mut k = a;
            while k < a_end && runs[k].x < bx1 {
                if runs[k].key == runs[b].key {
                    regions.union(k as u32, b as u32);
                }
                k += 1;
            }
        }
    }
    let mut region_bounds = vec![Bounds::EMPTY; n_runs];
    for y in 0..height {
        for i in row_runs(y) {
            let root = regions.find(i as u32) as usize;
            region_bounds[root].add(y as u32, &runs[i]);
        }
    }
    let background: Vec<bool> = (0..n_runs)
        .map(|i| {
            let b = &region_bounds[regions.find(i as u32) as usize];
            b.width() > MAX_GLYPH || b.height() > MAX_GLYPH
        })
        .collect();
    if !background.iter().any(|&b| b) {
        return None;
    }

    // 3. Components of what is not background, joined in eight
    // directions.
    let mut comps = Sets::new(n_runs);
    for y in 0..height {
        let range = row_runs(y);
        for i in range.clone().skip(1) {
            if !background[i] && !background[i - 1] {
                comps.union((i - 1) as u32, i as u32);
            }
        }
        if y == 0 {
            continue;
        }
        let (mut a, a_end) = (row_start[y - 1] as usize, row_start[y] as usize);
        for b in range {
            if background[b] {
                continue;
            }
            let (bx0, bx1) = (runs[b].x, runs[b].x + runs[b].len);
            // Diagonal contact counts: one pixel of slack either side.
            while a < a_end && runs[a].x + runs[a].len < bx0 {
                a += 1;
            }
            let mut k = a;
            while k < a_end && runs[k].x <= bx1 {
                if !background[k] {
                    comps.union(k as u32, b as u32);
                }
                k += 1;
            }
        }
    }
    let mut comp_bounds = vec![Bounds::EMPTY; n_runs];
    for y in 0..height {
        for i in row_runs(y) {
            if !background[i] {
                let root = comps.find(i as u32) as usize;
                comp_bounds[root].add(y as u32, &runs[i]);
            }
        }
    }
    drop(runs);

    // 4. Identical boxes. Candidates in raster order of their boxes so
    // the dictionary does not depend on anything but the picture.
    let mut boxes: Vec<Bounds> = comp_bounds
        .into_iter()
        .filter(|b| {
            b.area > 0
                && b.width() <= MAX_GLYPH
                && b.height() <= MAX_GLYPH
                && b.width() * b.height() >= MIN_BOX_AREA
        })
        .collect();
    if boxes.len() < 2 {
        return None;
    }
    boxes.sort_unstable_by_key(|b| (b.y0, b.x0, b.y1, b.x1));
    struct Found {
        w: usize,
        h: usize,
        keys: Vec<u64>,
        foreground: u32,
        positions: Vec<(usize, usize)>,
    }
    let mut found: Vec<Found> = Vec::new();
    let mut by_hash: HashMap<u64, Vec<u32>> = HashMap::new();
    let mut content: Vec<u64> = Vec::new();
    for b in &boxes {
        let (w, h) = (b.width(), b.height());
        content.clear();
        content.resize(w * h, 0);
        for (row, line) in content.chunks_exact_mut(w).enumerate() {
            row_keys(src, n_colour, b.y0 as usize + row, b.x0 as usize, line);
        }
        let hash = hash_keys(w, h, &content);
        let slot = by_hash.entry(hash).or_default();
        let at = (b.x0 as usize, b.y0 as usize);
        match slot
            .iter()
            .find(|&&i| {
                let f = &found[i as usize];
                f.w == w && f.h == h && f.keys == content
            })
            .copied()
        {
            Some(i) => found[i as usize].positions.push(at),
            None => {
                slot.push(found.len() as u32);
                found.push(Found {
                    w,
                    h,
                    keys: content.clone(),
                    foreground: b.area,
                    positions: vec![at],
                });
            }
        }
    }
    drop(by_hash);

    // An entry pays when the pixels it saves outweigh its positions:
    // its pixels are coded once (in the reference picture) instead of
    // once per occurrence.
    found.retain(|f| {
        let n = f.positions.len() as f64;
        let pixels_bits = PIXEL_BITS * f64::from(f.foreground) + 6.0;
        n >= 2.0 && (n - 1.0) * pixels_bits > n * POSITION_BITS + ENTRY_BITS
    });
    if found.is_empty() {
        return None;
    }

    // Shelves: tallest first, left to right.
    let mut order: Vec<usize> = (0..found.len()).collect();
    order.sort_by_key(|&i| {
        (
            std::cmp::Reverse(found[i].h),
            std::cmp::Reverse(found[i].w),
            i,
        )
    });
    let total: usize = found.iter().map(|f| f.w * f.h).sum();
    let widest = found.iter().map(|f| f.w).max().unwrap_or(1);
    let atlas_width = ((total as f64).sqrt() * 1.5).ceil() as usize;
    let atlas_width = atlas_width.clamp(widest, MAX_ATLAS_WIDTH.max(widest));
    let mut place = vec![(0usize, 0usize); found.len()];
    let (mut x, mut y, mut shelf) = (0usize, 0usize, 0usize);
    for &i in &order {
        let f = &found[i];
        if x + f.w > atlas_width {
            y += shelf;
            x = 0;
            shelf = 0;
        }
        place[i] = (x, y);
        x += f.w;
        shelf = shelf.max(f.h);
    }
    let atlas_height = y + shelf;
    let mut atlas: Vec<Vec<i32>> = vec![vec![0i32; atlas_width * atlas_height]; n_colour];
    for (f, &(ax, ay)) in found.iter().zip(&place) {
        for (row, line) in f.keys.chunks_exact(f.w).enumerate() {
            let o = (ay + row) * atlas_width + ax;
            for (c, plane) in atlas.iter_mut().enumerate() {
                let shift = 16 * (n_colour - 1 - c);
                for (slot, &k) in plane[o..o + f.w].iter_mut().zip(line) {
                    *slot = ((k >> shift) & 0xFFFF) as i32;
                }
            }
        }
    }

    // 5. The frame without them: each occurrence repainted with the
    // commonest colour of the ring around it.
    let mut fills = Vec::new();
    let mut ring: HashMap<u64, u32> = HashMap::new();
    let mut span = vec![0u64; MAX_GLYPH + 2];
    for f in &found {
        for &(px, py) in &f.positions {
            ring.clear();
            let x_lo = px.saturating_sub(1);
            let x_hi = (px + f.w + 1).min(width);
            let span = &mut span[..x_hi - x_lo];
            for y in py.saturating_sub(1)..(py + f.h + 1).min(height) {
                row_keys(src, n_colour, y, x_lo, span);
                let inside_rows = y >= py && y < py + f.h;
                for (i, &k) in span.iter().enumerate() {
                    let x = x_lo + i;
                    if !(inside_rows && x >= px && x < px + f.w) {
                        *ring.entry(k).or_insert(0) += 1;
                    }
                }
            }
            // The commonest ring colour, the smallest key on a tie; a
            // box that fills the frame keeps its own first pixel.
            let key = ring
                .iter()
                .max_by_key(|&(&k, &n)| (n, std::cmp::Reverse(k)))
                .map_or(f.keys[0], |(&k, _)| k);
            let mut colour = [0i32; 3];
            for (c, slot) in colour.iter_mut().enumerate().take(n_colour) {
                *slot = ((key >> (16 * (n_colour - 1 - c))) & 0xFFFF) as i32;
            }
            fills.push(Fill {
                x: px,
                y: py,
                w: f.w,
                h: f.h,
                colour,
            });
        }
    }
    let entries = found
        .into_iter()
        .zip(&place)
        .map(|(f, &(ax, ay))| PatchEntry {
            ax,
            ay,
            w: f.w,
            h: f.h,
            positions: f.positions,
        })
        .collect();
    Some(PatchPlan {
        entries,
        atlas,
        atlas_width,
        atlas_height,
        fills: Fills::new(fills, height, n_colour),
    })
}

/// The patch dictionary as the `(context, value)` symbols of its
/// ten-context entropy stream (the patch-reading listing of the
/// LfGlobal section): every entry reads `reference` slot, is copied
/// with `kReplace` over the colour channels and leaves the
/// `num_extra` extra channels alone (`kNone`).
pub(crate) fn dictionary_symbols(
    entries: &[PatchEntry],
    reference: u32,
    num_extra: usize,
) -> Vec<(u32, u32)> {
    use crate::bitwriter::pack_signed;
    const REPLACE: u32 = 1;
    const NONE: u32 = 0;
    let mut out = vec![(0u32, entries.len() as u32)];
    for e in entries {
        out.push((1, reference));
        out.push((3, e.ax as u32));
        out.push((3, e.ay as u32));
        out.push((2, e.w as u32 - 1));
        out.push((2, e.h as u32 - 1));
        out.push((7, e.positions.len() as u32 - 1));
        let mut last = (0i64, 0i64);
        for (j, &(x, y)) in e.positions.iter().enumerate() {
            let (x, y) = (x as i64, y as i64);
            if j == 0 {
                out.push((4, x as u32));
                out.push((4, y as u32));
            } else {
                out.push((6, pack_signed((x - last.0) as i32)));
                out.push((6, pack_signed((y - last.1) as i32)));
            }
            last = (x, y);
            out.push((5, REPLACE));
            for _ in 0..num_extra {
                out.push((5, NONE));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A page of repeated 6×8 "letters" on white, with a grey panel.
    fn page(width: usize, height: usize) -> Vec<Vec<i32>> {
        let mut p = vec![255i32; width * height];
        for y in height - 20..height {
            for x in 0..width {
                p[y * width + x] = 200;
            }
        }
        let glyph = |k: usize, gx: usize, gy: usize| -> i32 {
            // Deterministic bitmap per letter, with a soft rim.
            let bit = ((k * 2654435761 + gx * 40503 + gy * 9973) >> 3) & 3;
            match bit {
                0 => 0,
                1 => 90,
                _ => 255,
            }
        };
        let mut col = 0usize;
        let mut k = 0usize;
        for line in 0..(height - 30) / 12 {
            let y0 = 4 + line * 12;
            col = (col * 7 + 3) % 5;
            let mut x0 = 4 + col;
            while x0 + 8 < width {
                let letter = (k * k + line) % 7;
                k += 1;
                for gy in 0..8 {
                    for gx in 0..6 {
                        p[(y0 + gy) * width + x0 + gx] = glyph(letter, gx, gy);
                    }
                }
                x0 += 9;
            }
        }
        vec![p]
    }

    #[test]
    fn repeated_glyphs_become_dictionary_entries() {
        let (w, h) = (200usize, 120usize);
        let planes = page(w, h);
        let src = Pixels::Planes {
            planes: &planes,
            width: w,
        };
        let plan = find_patches(&src, w, h, 1).expect("a page of text has patches");
        assert!(plan.entries.len() <= 12, "{} entries", plan.entries.len());
        assert!(
            plan.num_positions() > 50,
            "{} positions",
            plan.num_positions()
        );
        // Every position holds exactly the entry's pixels, and the
        // frame without the patches is flat there.
        for e in &plan.entries {
            assert!(e.positions.len() >= 2);
            for &(px, py) in &e.positions {
                for y in 0..e.h {
                    let mut line = vec![0i32; e.w];
                    src.row(0, py + y, px, &mut line);
                    let o = (e.ay + y) * plan.atlas_width + e.ax;
                    assert_eq!(line, &plan.atlas[0][o..o + e.w], "entry at ({px},{py})");
                    plan.fills.apply(0, py + y, px, &mut line);
                    assert!(line.iter().all(|&v| v == 255), "fill at ({px},{py})");
                }
            }
        }
        // A fill only touches its own rectangle.
        let e = &plan.entries[0];
        let (px, py) = e.positions[0];
        let mut line = vec![7i32; w];
        plan.fills.apply(0, py, 0, &mut line);
        assert_eq!((line[px - 1], line[px]), (7, 255));
        // The dictionary symbols start with the entry count.
        let syms = dictionary_symbols(&plan.entries, 1, 0);
        assert_eq!(syms[0], (0, plan.entries.len() as u32));
        assert_eq!(syms[1], (1, 1));
    }

    #[test]
    fn photographic_content_is_left_alone() {
        let (w, h) = (96usize, 96usize);
        let mut s = 1u32;
        let plane: Vec<i32> = (0..w * h)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 24) as i32
            })
            .collect();
        let planes = vec![plane];
        let src = Pixels::Planes {
            planes: &planes,
            width: w,
        };
        assert!(find_patches(&src, w, h, 1).is_none());
        // A flat picture has background and nothing on it.
        let planes = vec![vec![9i32; w * h]];
        let src = Pixels::Planes {
            planes: &planes,
            width: w,
        };
        assert!(find_patches(&src, w, h, 1).is_none());
    }
}
