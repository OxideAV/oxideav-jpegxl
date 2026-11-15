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

/// The corpus the README numbers come from, in an optimised build. A
/// build with debug assertions — where an encode is thirty to fifty
/// times slower — runs the same tests on two small pictures of the
/// same kinds.
fn corpus() -> Vec<(&'static str, Canvas)> {
    if cfg!(debug_assertions) {
        return vec![
            ("window 320x240", screenshot(320, 240, 11)),
            ("page 256x160", page(256, 160, 22)),
        ];
    }
    vec![
        ("window 800x600", screenshot(800, 600, 11)),
        ("page 640x480", page(640, 480, 22)),
        ("terminal 720x400", terminal(720, 400, 33)),
        ("window 1280x800", screenshot(1280, 800, 44)),
    ]
}

/// Side lengths for the single-picture tests: smaller under debug
/// assertions.
fn small() -> (usize, usize) {
    if cfg!(debug_assertions) {
        (160, 120)
    } else {
        (320, 200)
    }
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

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sse: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum();
    if sse == 0.0 {
        return 99.0;
    }
    10.0 * (255.0 * 255.0 * a.len() as f64 / sse).log10()
}

/// The black-box encoder's lossy stream at `distance`, decoded by this
/// crate: `(bytes, PSNR)`.
fn black_box_lossy(c: &Canvas, distance: f32) -> Option<(usize, f64)> {
    if !tool("cjxl") {
        return None;
    }
    let (src, dst) = (scratch("lossy-ref.ppm"), scratch("lossy-ref.jxl"));
    std::fs::write(&src, ppm(c)).unwrap();
    let out = Command::new("cjxl")
        .arg(&src)
        .arg(&dst)
        .args(["-d", &distance.to_string()])
        .output()
        .unwrap();
    assert!(out.status.success());
    let bytes = std::fs::read(&dst).unwrap();
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&dst);
    let back = decode(&bytes).ok()?;
    Some((bytes.len(), psnr(&back.to_rgb8(), &c.rgb)))
}

/// The lossy codestream of an RGB8 canvas with each screen-content
/// tool on or off: `patches` (the dictionary on the VarDCT frame) and
/// `exact` (keep the lossless codestream when it is no larger).
fn lossy_codestream(canvas: &Canvas, distance: f32, patches: bool, exact: bool) -> Vec<u8> {
    let img = canvas.image();
    lossy_codestream_of(&img, distance, patches, exact)
}

/// [`lossy_codestream`] for any packed 8-bit picture.
fn lossy_codestream_of(img: &JxlImage, distance: f32, patches: bool, exact: bool) -> Vec<u8> {
    use oxideav_jpegxl::encoder::{ColourSpec, EncodeSpec, FrameInput, encode_codestream};
    let n = img.format.channels();
    let spec = EncodeSpec {
        width: img.width,
        height: img.height,
        bits: 8,
        grey: img.format.is_gray(),
        alpha: img.format.has_alpha(),
        colour: ColourSpec::Enumerated {
            primaries: 1,
            transfer: 13,
        },
        animation: None,
        quality: 90,
        group_size_shift: 1,
        effort: Effort::Balanced,
        vardct_distance: Some(distance),
        threads: 1,
        vardct_transform: None,
        patches,
        lz77: true,
        exact_when_smaller: exact,
    };
    let packed = img.as_bytes().unwrap();
    let channels = (0..n)
        .map(|c| {
            packed
                .iter()
                .skip(c)
                .step_by(n)
                .map(|&v| i32::from(v))
                .collect()
        })
        .collect();
    let frame = FrameInput {
        channels,
        duration: 0,
    };
    encode_codestream(&spec, &[frame]).unwrap()
}

#[test]
fn patches_sharpen_lossy_screen_content() {
    for distance in [1.0f32, 3.0] {
        // Per variant: VarDCT alone, VarDCT with patches, and what the
        // public entry point writes (the smaller of that and the
        // lossless codestream).
        let mut total = [(0usize, 0.0f64); 3];
        for (name, canvas) in corpus() {
            let streams = [
                ("plain", lossy_codestream(&canvas, distance, false, false)),
                ("patches", lossy_codestream(&canvas, distance, true, false)),
                ("default", lossy_codestream(&canvas, distance, true, true)),
            ];
            let opts = EncodeOptions::new().with_distance(distance);
            assert_eq!(streams[2].1, encode(&canvas.image(), &opts).unwrap());
            assert_eq!(
                streams[0].1,
                encode(&canvas.image(), &opts.with_patches(false)).unwrap()
            );
            let mut quality = [0.0f64; 3];
            for (k, (label, bytes)) in streams.iter().enumerate() {
                let back = decode(bytes).unwrap();
                assert_eq!(back.format, PixelFormat::Rgb24, "{name} {label}");
                let ours = back.as_bytes().unwrap();
                quality[k] = psnr(ours, &canvas.rgb);
                total[k].0 += bytes.len();
                total[k].1 += quality[k];
                let tag = format!("lossy-{label}-{}-{distance}", canvas.w);
                if let Some(theirs) = black_box_rgb(bytes, &tag) {
                    let worst = ours
                        .iter()
                        .zip(&theirs)
                        .map(|(&a, &b)| a.abs_diff(b))
                        .max()
                        .unwrap_or(0);
                    assert!(
                        worst <= 2,
                        "{name} {label} d{distance}: the black-box decoder differs by {worst}"
                    );
                }
            }
            let [plain, patched, default] = [&streams[0].1, &streams[1].1, &streams[2].1];
            // Each tool is kept only when the file is no larger.
            assert!(patched.len() <= plain.len(), "{name} d{distance}");
            assert!(default.len() <= patched.len(), "{name} d{distance}");
            assert!(
                quality[1] >= quality[0] - 0.1,
                "{name} d{distance}: {:.2} dB with patches, {:.2} without",
                quality[1],
                quality[0]
            );
            assert!(quality[2] >= quality[1] - 0.1, "{name} d{distance}");
            let reference = black_box_lossy(&canvas, distance)
                .map(|(n, q)| format!("; black-box encoder {n} / {q:.2}"))
                .unwrap_or_default();
            println!(
                "{name} d{distance}: VarDCT {} bytes / {:.2} dB, with patches {} / {:.2}, \
                 default {} / {:.2}{reference}",
                plain.len(),
                quality[0],
                patched.len(),
                quality[1],
                default.len(),
                quality[2]
            );
        }
        let n = corpus().len() as f64;
        println!(
            "corpus d{distance}: {} bytes / {:.2} dB -> {} / {:.2} -> {} / {:.2}",
            total[0].0,
            total[0].1 / n,
            total[1].0,
            total[1].1 / n,
            total[2].0,
            total[2].1 / n
        );
        // Text is where the lossy coder spends its bits: the
        // dictionary has to pay in both size and quality.
        assert!(total[1].0 * 100 < total[0].0 * 85, "d{distance}: {total:?}");
        assert!(total[1].1 > total[0].1 + 2.0 * n, "d{distance}: {total:?}");
    }
}

/// A lossy request on flat-colour content is answered with the
/// lossless codestream when that is smaller; a photograph is not
/// touched.
#[test]
fn lossy_requests_on_flat_content_come_back_exact_when_smaller() {
    // Pixel art: 12 colours in 8-pixel cells.
    let (w, h) = (256usize, 192usize);
    let mut rng = Rng(77);
    let palette: Vec<[u8; 3]> = (0..12)
        .map(|_| {
            [
                rng.below(256) as u8,
                rng.below(256) as u8,
                rng.below(256) as u8,
            ]
        })
        .collect();
    let cells: Vec<usize> = (0..(w / 8) * (h / 8))
        .map(|_| rng.below(12) as usize)
        .collect();
    let mut c = Canvas::new(w, h, [0, 0, 0]);
    for y in 0..h {
        for x in 0..w {
            let colour = palette[cells[(y / 8) * (w / 8) + x / 8]];
            c.rgb[(y * w + x) * 3..][..3].copy_from_slice(&colour);
        }
    }
    let img = c.image();
    let lossless = encode(&img, &EncodeOptions::new()).unwrap();
    for distance in [0.5f32, 1.0, 4.0] {
        let opts = EncodeOptions::new().with_distance(distance);
        let lossy = encode(&img, &opts.clone().with_patches(false)).unwrap();
        let default = encode(&img, &opts).unwrap();
        println!(
            "pixel art d{distance}: VarDCT {} bytes, default {} bytes (lossless {})",
            lossy.len(),
            default.len(),
            lossless.len()
        );
        assert!(default.len() <= lossy.len() && default.len() <= lossless.len());
        assert_eq!(
            default, lossless,
            "d{distance}: the exact codestream is smaller"
        );
        assert_eq!(decode(&default).unwrap().as_bytes().unwrap(), &c.rgb[..]);
        if let Some(theirs) = black_box_rgb(&default, &format!("exact-{distance}")) {
            assert!(theirs == c.rgb);
        }
    }
    // Noise: no runs, the lossy stream is what was asked for.
    let noise: Vec<u8> = (0..160 * 120 * 3)
        .map(|_| (rng.next() >> 24) as u8)
        .collect();
    let img = JxlImage::from_rgb8(160, 120, noise).unwrap();
    let opts = EncodeOptions::new().with_distance(2.0);
    assert_eq!(
        encode(&img, &opts).unwrap(),
        encode(&img, &opts.clone().with_patches(false)).unwrap()
    );
    assert!(
        oxideav_jpegxl::info(&encode(&img, &opts).unwrap())
            .unwrap()
            .xyb_encoded
    );
}

/// Lossy patches next to an alpha channel, and on grey: the reference
/// frame is a Modular frame of an XYB image that carries the extra
/// channel too.
#[test]
fn lossy_patches_keep_alpha_exact() {
    let (pw, ph) = small();
    let canvas = page(pw, ph, 5);
    let (w, h) = (canvas.w as u32, canvas.h as u32);
    let mut rgba = Vec::with_capacity(canvas.rgb.len() / 3 * 4);
    for (i, px) in canvas.rgb.as_chunks::<3>().0.iter().enumerate() {
        rgba.extend_from_slice(px);
        rgba.push(((i % pw) * 255 / (pw - 1)) as u8);
    }
    let ya: Vec<u8> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[1], p[3]])
        .collect();
    let grey: Vec<u8> = canvas.rgb.as_chunks::<3>().0.iter().map(|p| p[1]).collect();
    let pictures = [
        ("rgba", JxlImage::from_rgba8(w, h, rgba).unwrap()),
        ("ya", JxlImage::packed(w, h, PixelFormat::Ya8, ya).unwrap()),
        (
            "grey",
            JxlImage::packed(w, h, PixelFormat::Gray8, grey).unwrap(),
        ),
    ];
    for (name, img) in &pictures {
        // The VarDCT coder with its patches, the lossless codestream
        // not considered: this is about the pair of frames.
        let plain = lossy_codestream_of(img, 1.0, false, false);
        let patched = lossy_codestream_of(img, 1.0, true, false);
        assert!(
            oxideav_jpegxl::info(&patched).unwrap().xyb_encoded,
            "{name}"
        );
        assert!(
            patched.len() * 10 < plain.len() * 7,
            "{name}: {} bytes against {}",
            patched.len(),
            plain.len()
        );
        let back = decode(&patched).unwrap();
        assert_eq!(back.format, img.format, "{name}");
        let (src, got) = (img.as_bytes().unwrap(), back.as_bytes().unwrap());
        let channels = img.format.channels();
        let alpha = img.format.has_alpha().then_some(channels - 1);
        let (mut sse, mut n) = (0.0f64, 0usize);
        for (i, (&a, &b)) in src.iter().zip(got).enumerate() {
            if alpha == Some(i % channels) {
                assert_eq!(a, b, "{name}: alpha sample {i}");
            } else {
                let d = f64::from(a) - f64::from(b);
                sse += d * d;
                n += 1;
            }
        }
        let quality = 10.0 * (255.0 * 255.0 * n as f64 / sse.max(1e-9)).log10();
        println!(
            "{name} d1: {} -> {} bytes, {quality:.2} dB",
            plain.len(),
            patched.len()
        );
        assert!(quality > 45.0, "{name}: {quality:.2} dB");
        // The black-box decoder reads the pair of frames too.
        if tool("djxl") {
            let (jxl, out) = (
                scratch(&format!("pa-{name}.jxl")),
                scratch(&format!("pa-{name}.pam")),
            );
            std::fs::write(&jxl, &patched).unwrap();
            let run = Command::new("djxl").arg(&jxl).arg(&out).output().unwrap();
            assert!(
                run.status.success(),
                "{name}: the black-box decoder refused the stream: {}",
                String::from_utf8_lossy(&run.stderr)
            );
            let data = std::fs::read(&out).unwrap();
            let theirs = &data[data.len() - got.len()..];
            let worst = got
                .iter()
                .zip(theirs)
                .map(|(&a, &b)| a.abs_diff(b))
                .max()
                .unwrap_or(0);
            assert!(
                worst <= 2,
                "{name}: the black-box decoder differs by {worst}"
            );
            let _ = std::fs::remove_file(&jxl);
            let _ = std::fs::remove_file(&out);
        }
    }
}

/// A lossy Modular stream of a black-box encoder with an alpha channel
/// (an XYB image whose Modular frame carries the extra channel).
#[test]
fn a_black_box_lossy_modular_stream_with_alpha_decodes() {
    if !tool("cjxl") || !tool("djxl") {
        eprintln!("cjxl / djxl not installed: skipped");
        return;
    }
    let canvas = screenshot(200, 160, 3);
    let mut pam = format!(
        "P7\nWIDTH {}\nHEIGHT {}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n",
        canvas.w, canvas.h
    )
    .into_bytes();
    let mut alpha = Vec::new();
    for (i, px) in canvas.rgb.as_chunks::<3>().0.iter().enumerate() {
        pam.extend_from_slice(px);
        let a = ((i / canvas.w) * 255 / (canvas.h - 1)) as u8;
        pam.push(a);
        alpha.push(a);
    }
    let (src, jxl, out) = (scratch("bb.pam"), scratch("bb.jxl"), scratch("bb-out.pam"));
    std::fs::write(&src, &pam).unwrap();
    let run = Command::new("cjxl")
        .arg(&src)
        .arg(&jxl)
        .args(["-m", "1", "-d", "1"])
        .output()
        .unwrap();
    assert!(run.status.success());
    let bytes = std::fs::read(&jxl).unwrap();
    let back = decode(&bytes).unwrap();
    assert_eq!(back.format, PixelFormat::Rgba);
    let got = back.as_bytes().unwrap();
    let run = Command::new("djxl").arg(&jxl).arg(&out).output().unwrap();
    assert!(run.status.success());
    let data = std::fs::read(&out).unwrap();
    let theirs = &data[data.len() - got.len()..];
    let mut worst = 0u8;
    for (i, (&a, &b)) in got.iter().zip(theirs).enumerate() {
        if i % 4 == 3 {
            assert_eq!(a, b, "alpha sample {}", i / 4);
            assert_eq!(a, alpha[i / 4]);
        } else {
            worst = worst.max(a.abs_diff(b));
        }
    }
    println!("black-box lossy Modular RGBA: colour within {worst} of the black-box decode");
    assert!(worst <= 2, "colour differs by {worst}");
    for f in [src, jxl, out] {
        let _ = std::fs::remove_file(f);
    }
}

#[test]
fn patches_leave_alpha_and_photographs_alone() {
    // RGBA: the colour is patched, the alpha plane is coded as it is.
    let (pw, ph) = small();
    let canvas = page(pw, ph, 5);
    let (w, h) = (pw as u32, ph as u32);
    let mut rgba = Vec::with_capacity(canvas.rgb.len() / 3 * 4);
    for (i, px) in canvas.rgb.as_chunks::<3>().0.iter().enumerate() {
        rgba.extend_from_slice(px);
        rgba.push(((i % pw) * 255 / (pw - 1)) as u8);
    }
    let img = JxlImage::from_rgba8(w, h, rgba.clone()).unwrap();
    let plain = encode(&img, &EncodeOptions::new().with_patches(false)).unwrap();
    let patched = encode(&img, &EncodeOptions::new()).unwrap();
    assert!(patched.len() < plain.len());
    assert_eq!(decode(&patched).unwrap().as_bytes().unwrap(), &rgba[..]);
    // Grey.
    let grey: Vec<u8> = canvas.rgb.as_chunks::<3>().0.iter().map(|p| p[1]).collect();
    let img = JxlImage::packed(w, h, PixelFormat::Gray8, grey.clone()).unwrap();
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
