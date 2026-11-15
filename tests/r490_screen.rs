//! Screen-content tools on a synthesised screenshot corpus: text set
//! in a procedural anti-aliased font, panels of flat colour, a
//! gradient bar and a patch of photographic noise.
//!
//! Each tool is measured as bytes saved against the same encoder with
//! the tool off; every stream round-trips through this crate's decoder
//! and — when `djxl` is installed — decodes to the same pixels in the
//! black-box decoder. With `cjxl` installed the sizes of a black-box
//! encoder are printed next to them.

use std::path::PathBuf;
use std::process::Command;

use oxideav_jpegxl::{Effort, EncodeOptions, JxlImage, PixelFormat, decode, encode};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
    fn unit(&mut self) -> f32 {
        (self.next() & 0xFFFF) as f32 / 65535.0
    }
}

const CELL_W: usize = 7;
const CELL_H: usize = 10;

/// A font of `n` glyphs: per glyph the ink coverage (0..=255) of a
/// `CELL_W × CELL_H` cell — a few strokes with soft edges, kept a
/// pixel inside the cell so neighbouring letters do not touch.
fn font(n: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| {
            let strokes: Vec<[f32; 4]> = (0..3 + rng.below(3))
                .map(|_| {
                    let mut p = || {
                        [
                            1.5 + rng.unit() * (CELL_W as f32 - 3.0),
                            1.5 + rng.unit() * (CELL_H as f32 - 3.0),
                        ]
                    };
                    let (a, b) = (p(), p());
                    [a[0], a[1], b[0], b[1]]
                })
                .collect();
            let mut cell = vec![0u8; CELL_W * CELL_H];
            for y in 0..CELL_H {
                for x in 0..CELL_W {
                    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                    let mut best = f32::INFINITY;
                    for s in &strokes {
                        let (dx, dy) = (s[2] - s[0], s[3] - s[1]);
                        let len2 = (dx * dx + dy * dy).max(1e-6);
                        let t = (((px - s[0]) * dx + (py - s[1]) * dy) / len2).clamp(0.0, 1.0);
                        let (qx, qy) = (s[0] + t * dx - px, s[1] + t * dy - py);
                        best = best.min((qx * qx + qy * qy).sqrt());
                    }
                    // A stroke 1.1 px wide with a soft rim.
                    let cover = (1.05 - best).clamp(0.0, 1.0);
                    cell[y * CELL_W + x] = (cover * 255.0).round() as u8;
                }
            }
            cell
        })
        .collect()
}

struct Canvas {
    w: usize,
    h: usize,
    rgb: Vec<u8>,
}

impl Canvas {
    fn new(w: usize, h: usize, colour: [u8; 3]) -> Self {
        let mut rgb = Vec::with_capacity(w * h * 3);
        for _ in 0..w * h {
            rgb.extend_from_slice(&colour);
        }
        Self { w, h, rgb }
    }

    fn rect(&mut self, x0: usize, y0: usize, w: usize, h: usize, colour: [u8; 3]) {
        for y in y0..(y0 + h).min(self.h) {
            for x in x0..(x0 + w).min(self.w) {
                self.rgb[(y * self.w + x) * 3..][..3].copy_from_slice(&colour);
            }
        }
    }

    fn glyph(&mut self, cell: &[u8], x0: usize, y0: usize, ink: [u8; 3]) {
        for gy in 0..CELL_H {
            for gx in 0..CELL_W {
                let (x, y) = (x0 + gx, y0 + gy);
                if x >= self.w || y >= self.h {
                    continue;
                }
                let a = u32::from(cell[gy * CELL_W + gx]);
                let px = &mut self.rgb[(y * self.w + x) * 3..][..3];
                for (p, &i) in px.iter_mut().zip(&ink) {
                    *p = ((u32::from(*p) * (255 - a) + u32::from(i) * a + 127) / 255) as u8;
                }
            }
        }
    }

    /// Lines of words in the box, letters on an 8-pixel advance.
    fn text(&mut self, font: &[Vec<u8>], rng: &mut Rng, area: [usize; 4], ink: [u8; 3]) {
        let [x0, y0, w, h] = area;
        let mut y = y0 + 4;
        while y + CELL_H + 2 <= y0 + h {
            let mut x = x0 + 6 + rng.below(3) as usize * 8;
            // Some short lines, like paragraph ends.
            let limit = x0 + w - if rng.below(4) == 0 { w / 2 } else { 8 };
            while x + CELL_W < limit {
                let letters = 2 + rng.below(8) as usize;
                for _ in 0..letters {
                    if x + CELL_W >= limit {
                        break;
                    }
                    // A skewed letter distribution, like a language.
                    let pick = (rng.below(font.len() as u32) * rng.below(font.len() as u32))
                        / font.len() as u32;
                    self.glyph(&font[pick as usize], x, y, ink);
                    x += 8;
                }
                x += 5;
            }
            y += 14;
        }
    }

    fn image(&self) -> JxlImage {
        JxlImage::from_rgb8(self.w as u32, self.h as u32, self.rgb.clone()).unwrap()
    }
}

/// A document window: title bar, side panel, a page of text, a
/// gradient bar and a noisy thumbnail.
fn screenshot(w: usize, h: usize, seed: u64) -> Canvas {
    let glyphs = font(48, 0xF0A7 ^ seed);
    let mut rng = Rng(seed);
    let mut c = Canvas::new(w, h, [255, 255, 255]);
    c.rect(0, 0, w, 26, [45, 62, 80]);
    c.text(&glyphs, &mut rng, [8, 4, w / 2, 20], [236, 240, 241]);
    c.rect(0, 26, 150, h - 26, [236, 240, 241]);
    c.text(&glyphs, &mut rng, [4, 34, 142, h - 60], [52, 73, 94]);
    c.rect(150, 26, 1, h - 26, [189, 195, 199]);
    c.text(&glyphs, &mut rng, [158, 34, w - 170, h - 130], [20, 20, 20]);
    // A gradient bar.
    for y in h - 80..h - 56 {
        for x in 170..w - 20 {
            let t = ((x - 170) * 255 / (w - 190)) as u8;
            c.rgb[(y * w + x) * 3..][..3].copy_from_slice(&[t, 128, 255 - t]);
        }
    }
    // A thumbnail of noise.
    for y in h - 50..h - 6 {
        for x in 170..250 {
            let v = (rng.next() >> 24) as u8;
            c.rgb[(y * w + x) * 3..][..3].copy_from_slice(&[v, v / 2 + 60, 200 - v / 2]);
        }
    }
    // Buttons.
    for k in 0..4 {
        let x = 270 + k * 96;
        if x + 84 < w {
            c.rect(x, h - 44, 84, 26, [52, 152, 219]);
            c.text(&glyphs, &mut rng, [x + 2, h - 42, 80, 22], [255, 255, 255]);
        }
    }
    c
}

/// Text only: black on white, the plainest case.
fn page(w: usize, h: usize, seed: u64) -> Canvas {
    let glyphs = font(40, 0xBEEF ^ seed);
    let mut rng = Rng(seed);
    let mut c = Canvas::new(w, h, [255, 255, 255]);
    c.text(&glyphs, &mut rng, [10, 8, w - 20, h - 16], [0, 0, 0]);
    c
}

/// A terminal: light text on a dark ground, dense.
fn terminal(w: usize, h: usize, seed: u64) -> Canvas {
    let glyphs = font(64, 0x7E11 ^ seed);
    let mut rng = Rng(seed);
    let mut c = Canvas::new(w, h, [12, 12, 16]);
    c.text(&glyphs, &mut rng, [0, 0, w, h], [200, 220, 200]);
    c
}

fn corpus() -> Vec<(&'static str, Canvas)> {
    vec![
        ("window 800x600", screenshot(800, 600, 11)),
        ("page 640x480", page(640, 480, 22)),
        ("terminal 720x400", terminal(720, 400, 33)),
        ("window 1280x800", screenshot(1280, 800, 44)),
    ]
}

fn tool(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("oxideav-jxl-r490-screen-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn ppm(c: &Canvas) -> Vec<u8> {
    let mut out = format!("P6\n{} {}\n255\n", c.w, c.h).into_bytes();
    out.extend_from_slice(&c.rgb);
    out
}

/// The black-box decoder's pixels for `jxl` (RGB8), when installed.
fn black_box_rgb(jxl: &[u8], tag: &str) -> Option<Vec<u8>> {
    if !tool("djxl") {
        return None;
    }
    let (src, dst) = (
        scratch(&format!("{tag}.jxl")),
        scratch(&format!("{tag}.ppm")),
    );
    std::fs::write(&src, jxl).unwrap();
    let out = Command::new("djxl").arg(&src).arg(&dst).output().unwrap();
    // A refused stream is left in place for inspection.
    assert!(
        out.status.success(),
        "{tag}: the black-box decoder refused the stream (kept at {}): {}",
        src.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    let data = std::fs::read(&dst).unwrap();
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&dst);
    // "P6\n<w> <h>\n255\n" then the samples.
    let header = data
        .iter()
        .enumerate()
        .filter(|&(_, &b)| b == b'\n')
        .nth(2)
        .map(|(i, _)| i + 1)
        .unwrap();
    Some(data[header..].to_vec())
}

/// The black-box encoder's lossless size at the given effort.
fn black_box_lossless(c: &Canvas, effort: u32) -> Option<usize> {
    if !tool("cjxl") {
        return None;
    }
    let (src, dst) = (scratch("ref.ppm"), scratch("ref.jxl"));
    std::fs::write(&src, ppm(c)).unwrap();
    let out = Command::new("cjxl")
        .arg(&src)
        .arg(&dst)
        .args(["-d", "0", "-e", &effort.to_string()])
        .output()
        .unwrap();
    assert!(out.status.success());
    let n = std::fs::metadata(&dst).unwrap().len() as usize;
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&dst);
    Some(n)
}

#[test]
fn patches_save_bytes_on_lossless_screen_content() {
    let mut total = (0usize, 0usize);
    for (name, canvas) in corpus() {
        let img = canvas.image();
        let plain = encode(&img, &EncodeOptions::new().with_patches(false)).unwrap();
        let patched = encode(&img, &EncodeOptions::new()).unwrap();
        // Never larger: the dictionary is kept only when it wins.
        assert!(patched.len() <= plain.len(), "{name}");
        for (label, bytes) in [("plain", &plain), ("patches", &patched)] {
            let back = decode(bytes).unwrap();
            assert_eq!(back.format, PixelFormat::Rgb24, "{name} {label}");
            assert_eq!(back.as_bytes().unwrap(), &canvas.rgb[..], "{name} {label}");
            if let Some(theirs) = black_box_rgb(bytes, &format!("{label}-{}", canvas.w)) {
                assert!(
                    theirs == canvas.rgb,
                    "{name} {label}: the black-box decoder reads other pixels"
                );
            }
        }
        let reference = [7, 9]
            .iter()
            .filter_map(|&e| black_box_lossless(&canvas, e).map(|n| format!("e{e} {n}")))
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "{name}: {} bytes without patches, {} with ({:+.1} %){}",
            plain.len(),
            patched.len(),
            (patched.len() as f64 / plain.len() as f64 - 1.0) * 100.0,
            if reference.is_empty() {
                String::new()
            } else {
                format!("; black-box encoder {reference}")
            }
        );
        total.0 += plain.len();
        total.1 += patched.len();
    }
    println!(
        "corpus: {} -> {} bytes ({:+.1} %)",
        total.0,
        total.1,
        (total.1 as f64 / total.0 as f64 - 1.0) * 100.0
    );
    // Text repeats: the dictionary must pay on the corpus as a whole.
    assert!(total.1 * 100 < total.0 * 90, "{} -> {}", total.0, total.1);
}

#[test]
fn lz77_saves_bytes_where_symbols_repeat() {
    let base = EncodeOptions::new().with_patches(false);
    let mut total = (0usize, 0usize);
    for (name, canvas) in corpus() {
        let img = canvas.image();
        let off = encode(&img, &base.clone().with_lz77(false)).unwrap();
        let on = encode(&img, &base).unwrap();
        // Elected on an estimate: allow it to be wrong by a hair.
        assert!(on.len() <= off.len() + off.len() / 200, "{name}");
        let back = decode(&on).unwrap();
        assert_eq!(back.as_bytes().unwrap(), &canvas.rgb[..], "{name}");
        if let Some(theirs) = black_box_rgb(&on, &format!("lz-{}", canvas.w)) {
            assert!(
                theirs == canvas.rgb,
                "{name}: the black-box decoder reads other pixels"
            );
        }
        // With the patches taken out, what is left is mostly flat.
        let both = encode(&img, &EncodeOptions::new()).unwrap();
        let patches_only = encode(&img, &EncodeOptions::new().with_lz77(false)).unwrap();
        println!(
            "{name}: LZ77 {} -> {} bytes ({:+.1} %); with patches {} -> {} ({:+.1} %)",
            off.len(),
            on.len(),
            (on.len() as f64 / off.len() as f64 - 1.0) * 100.0,
            patches_only.len(),
            both.len(),
            (both.len() as f64 / patches_only.len() as f64 - 1.0) * 100.0
        );
        total.0 += off.len();
        total.1 += on.len();
    }
    println!(
        "corpus: {} -> {} bytes ({:+.1} %)",
        total.0,
        total.1,
        (total.1 as f64 / total.0 as f64 - 1.0) * 100.0
    );
    assert!(total.1 < total.0, "{} -> {}", total.0, total.1);
}

#[test]
fn patches_leave_alpha_and_photographs_alone() {
    // RGBA: the colour is patched, the alpha plane is coded as it is.
    let canvas = page(320, 200, 5);
    let mut rgba = Vec::with_capacity(canvas.rgb.len() / 3 * 4);
    for (i, px) in canvas.rgb.as_chunks::<3>().0.iter().enumerate() {
        rgba.extend_from_slice(px);
        rgba.push(((i % 320) * 255 / 319) as u8);
    }
    let img = JxlImage::from_rgba8(320, 200, rgba.clone()).unwrap();
    let plain = encode(&img, &EncodeOptions::new().with_patches(false)).unwrap();
    let patched = encode(&img, &EncodeOptions::new()).unwrap();
    assert!(patched.len() < plain.len());
    assert_eq!(decode(&patched).unwrap().as_bytes().unwrap(), &rgba[..]);
    // Grey.
    let grey: Vec<u8> = canvas.rgb.as_chunks::<3>().0.iter().map(|p| p[1]).collect();
    let img = JxlImage::packed(320, 200, PixelFormat::Gray8, grey.clone()).unwrap();
    let patched = encode(&img, &EncodeOptions::new()).unwrap();
    assert_eq!(decode(&patched).unwrap().as_bytes().unwrap(), &grey[..]);
    // Noise has nothing to place: the same bytes either way.
    let mut rng = Rng(9);
    let noise: Vec<u8> = (0..160 * 120 * 3)
        .map(|_| (rng.next() >> 24) as u8)
        .collect();
    let img = JxlImage::from_rgb8(160, 120, noise).unwrap();
    for effort in [Effort::Fast, Effort::Balanced] {
        let opts = EncodeOptions::new().with_effort(effort);
        assert_eq!(
            encode(&img, &opts).unwrap(),
            encode(&img, &opts.clone().with_patches(false)).unwrap()
        );
    }
}
