//! Round 487 — Modular density: the learned MA tree (property splits,
//! all fourteen predictors including the self-correcting one), the
//! general cluster map, the prefix / ANS election and the RCT election
//! round-trip bit-exactly through this crate's decoder at every effort
//! and geometry class, beat the one-leaf-per-channel coder on
//! structured content, and — where a black-box decoder is installed —
//! decode to the same pixels there.

use std::process::Command;

use oxideav_jpegxl::{Effort, EncodeOptions, JxlImage, PixelFormat, decode, encode};

const EFFORTS: [Effort; 3] = [Effort::Fast, Effort::Balanced, Effort::Thorough];

fn hash(seed: &mut u32) -> u32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed
}

/// Photo-like content: smooth structure, edges and texture whose
/// strength varies over the picture (so context modelling pays).
fn structured(width: u32, height: u32, format: PixelFormat, bits: u8, seed: u32) -> JxlImage {
    let channels = format.channels();
    let bytes = format.bytes_per_sample();
    let max = (1u32 << bits) - 1;
    let mut s = seed.wrapping_mul(2654435761) | 1;
    let mut data = Vec::with_capacity((width * height) as usize * channels * bytes);
    for y in 0..height {
        for x in 0..width {
            let fx = x as f64 / width.max(2) as f64;
            let fy = y as f64 / height.max(2) as f64;
            let base = 0.5 + 0.3 * (fx * 9.0).sin() * (fy * 7.0).cos();
            let edge = if (x / 37 + y / 23) % 2 == 0 {
                0.12
            } else {
                -0.08
            };
            let rough = if fx > 0.5 { 0.10 } else { 0.01 };
            for c in 0..channels {
                let noise = (hash(&mut s) % 1000) as f64 / 1000.0 - 0.5;
                let v = if c == 3 || (channels == 2 && c == 1) {
                    // Alpha: mostly opaque with a soft disc.
                    let d = ((fx - 0.5).powi(2) + (fy - 0.5).powi(2)).sqrt();
                    (1.2 - d * 2.0).clamp(0.0, 1.0)
                } else {
                    base + edge + rough * noise + 0.05 * c as f64
                };
                let q = (v.clamp(0.0, 1.0) * f64::from(max)).round() as u32;
                if bytes == 1 {
                    data.push(q as u8);
                } else {
                    data.extend_from_slice(&(q as u16).to_le_bytes());
                }
            }
        }
    }
    JxlImage::packed(width, height, format, data)
        .unwrap()
        .with_bits_per_sample(bits)
}

fn round_trip(img: &JxlImage, opts: &EncodeOptions, what: &str) -> Vec<u8> {
    let bytes = encode(img, opts).unwrap_or_else(|e| panic!("{what}: encode: {e}"));
    let back = decode(&bytes).unwrap_or_else(|e| panic!("{what}: decode: {e}"));
    assert_eq!((back.width, back.height), (img.width, img.height), "{what}");
    assert_eq!(back.format, img.format, "{what}: layout");
    assert_eq!(
        back.as_bytes().unwrap(),
        img.as_bytes().unwrap(),
        "{what}: samples"
    );
    bytes
}

#[test]
fn every_effort_round_trips_every_layout() {
    for (format, bits) in [
        (PixelFormat::Gray8, 8u8),
        (PixelFormat::Ya8, 8),
        (PixelFormat::Rgb24, 8),
        (PixelFormat::Rgba, 8),
        (PixelFormat::Gray16Le, 16),
        (PixelFormat::Rgb48Le, 12),
        (PixelFormat::Rgba64Le, 16),
        (PixelFormat::Gray8, 1),
    ] {
        for effort in EFFORTS {
            let img = structured(97, 61, format, bits, 11);
            let opts = EncodeOptions::new().with_effort(effort);
            round_trip(&img, &opts, &format!("{format:?}/{bits} {effort:?}"));
        }
    }
}

#[test]
fn degenerate_and_extreme_geometries_round_trip() {
    for (w, h) in [
        (1u32, 1u32),
        (1, 2),
        (2, 1),
        (1, 700),
        (700, 1),
        (3, 3),
        (257, 2),
        (2, 513),
        (2051, 3),
        (128, 128),
        (129, 127),
    ] {
        // One effort per geometry, rotating: every effort sees every
        // class of geometry across the list.
        let effort = EFFORTS[(w + h) as usize % 3];
        for format in [PixelFormat::Gray8, PixelFormat::Rgba] {
            let img = structured(w, h, format, 8, w * 31 + h);
            let opts = EncodeOptions::new().with_effort(effort);
            round_trip(&img, &opts, &format!("{w}x{h} {format:?} {effort:?}"));
        }
    }
}

#[test]
fn every_group_size_round_trips_a_multi_group_picture() {
    // 2×2 groups of 128, 2×1 of 256, one of 512 / 1024.
    let img = structured(260, 140, PixelFormat::Rgb24, 8, 5);
    for shift in 0..=3u8 {
        let opts = EncodeOptions::new().with_group_size_shift(shift);
        round_trip(&img, &opts, &format!("group shift {shift}"));
    }
}

#[test]
fn the_learned_tree_is_denser_than_the_small_one_on_structured_content() {
    let img = structured(256, 192, PixelFormat::Rgb24, 8, 3);
    let fast = round_trip(
        &img,
        &EncodeOptions::new().with_effort(Effort::Fast),
        "fast",
    );
    let balanced = round_trip(&img, &EncodeOptions::new(), "balanced");
    assert!(
        balanced.len() < fast.len(),
        "balanced {} bytes, fast {}",
        balanced.len(),
        fast.len()
    );
}

#[test]
fn flat_and_constant_pictures_cost_almost_nothing() {
    for format in [PixelFormat::Gray8, PixelFormat::Rgba, PixelFormat::Rgb48Le] {
        let channels = format.channels() * format.bytes_per_sample();
        let img = JxlImage::packed(600, 400, format, vec![0x5A; 600 * 400 * channels]).unwrap();
        let bytes = round_trip(&img, &EncodeOptions::new(), &format!("flat {format:?}"));
        assert!(bytes.len() < 400, "flat {format:?}: {} bytes", bytes.len());
    }
}

#[test]
fn quantised_modular_stays_decodable_at_every_effort() {
    let img = structured(150, 140, PixelFormat::Rgb24, 8, 9);
    for effort in EFFORTS {
        for q in [10u8, 60, 95] {
            let opts = EncodeOptions::new()
                .with_quality(q)
                .with_modular(true)
                .with_effort(effort);
            let bytes = encode(&img, &opts).unwrap();
            let back = decode(&bytes).unwrap();
            assert_eq!((back.width, back.height), (150, 140));
        }
    }
}

/// Blocky picture of `colours` distinct colours (pixel-art-like).
fn few_colours(width: u32, height: u32, format: PixelFormat, colours: u32, seed: u32) -> JxlImage {
    let channels = format.channels();
    let bytes = format.bytes_per_sample();
    let mut s = seed | 1;
    let table: Vec<Vec<u16>> = (0..colours)
        .map(|_| {
            (0..channels)
                .map(|_| (hash(&mut s) % 65536) as u16)
                .collect()
        })
        .collect();
    let cells_w = width.div_ceil(6);
    let cells: Vec<u32> = (0..cells_w * height.div_ceil(5))
        .map(|_| hash(&mut s) % colours)
        .collect();
    let mut data = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let colour = &table[cells[((y / 5) * cells_w + x / 6) as usize] as usize];
            for &v in colour {
                if bytes == 1 {
                    data.push((v >> 8) as u8);
                } else {
                    data.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
    JxlImage::packed(width, height, format, data).unwrap()
}

#[test]
fn few_colour_pictures_use_a_palette_and_round_trip() {
    for (w, h) in [(96u32, 80u32), (300, 60), (9, 9)] {
        for format in [
            PixelFormat::Rgb24,
            PixelFormat::Rgba,
            PixelFormat::Ya8,
            PixelFormat::Rgb48Le,
            PixelFormat::Rgba64Le,
        ] {
            // The multi-group geometry on two layouts only.
            if w > 256 && !matches!(format, PixelFormat::Rgb24 | PixelFormat::Rgba64Le) {
                continue;
            }
            for colours in [1u32, 2, 13, 200] {
                let img = few_colours(w, h, format, colours, w + colours);
                round_trip(
                    &img,
                    &EncodeOptions::new(),
                    &format!("{w}x{h} {format:?} {colours} colours"),
                );
            }
        }
    }
    // Thirteen colours in 6000 random cells of 600x300 RGB: the cell
    // choices alone are 6000 x log2(13) bits (2 776 bytes); the coded
    // picture stays within about twice that.
    let img = few_colours(600, 300, PixelFormat::Rgb24, 13, 5);
    let bytes = round_trip(&img, &EncodeOptions::new(), "palette size");
    assert!(bytes.len() < 6000, "{} bytes", bytes.len());
}

#[test]
fn sparse_sample_values_use_a_channel_palette_and_round_trip() {
    // 8-bit content in a 16-bit container (multiples of 257), with
    // texture: the ranks cost about what the 8-bit picture costs.
    let base = structured(300, 200, PixelFormat::Gray8, 8, 21);
    let wide: Vec<u8> = base
        .as_bytes()
        .unwrap()
        .iter()
        .flat_map(|&v| (u16::from(v) * 257).to_le_bytes())
        .collect();
    let img16 = JxlImage::packed(300, 200, PixelFormat::Gray16Le, wide).unwrap();
    let b16 = round_trip(&img16, &EncodeOptions::new(), "sparse grey16");
    let b8 = round_trip(&base, &EncodeOptions::new(), "grey8");
    assert!(
        b16.len() < b8.len() + b8.len() / 8 + 400,
        "16-bit container {} bytes vs 8-bit {}",
        b16.len(),
        b8.len()
    );
    // Colour: every channel sparse, across a multi-group geometry.
    let base = structured(300, 130, PixelFormat::Rgba, 8, 22);
    let wide: Vec<u8> = base
        .as_bytes()
        .unwrap()
        .iter()
        .flat_map(|&v| (u16::from(v) * 257).to_le_bytes())
        .collect();
    let img16 = JxlImage::packed(300, 130, PixelFormat::Rgba64Le, wide).unwrap();
    round_trip(&img16, &EncodeOptions::new(), "sparse rgba64");
}

#[test]
fn the_decode_worker_budget_never_changes_the_pixels() {
    use oxideav_jpegxl::{DecodeOptions, decode_with};
    // 3×2 groups of 128: the per-group decode runs on the budget.
    let img = structured(300, 200, PixelFormat::Rgba, 8, 41);
    let opts = EncodeOptions::new()
        .with_group_size_shift(0)
        .with_effort(Effort::Fast);
    let bytes = encode(&img, &opts).unwrap();
    for threads in [1usize, 2, 5, 64] {
        let back = decode_with(&bytes, &DecodeOptions::default().with_threads(threads)).unwrap();
        assert_eq!(
            back.as_bytes().unwrap(),
            img.as_bytes().unwrap(),
            "{threads} workers"
        );
    }
}

fn djxl_available() -> bool {
    Command::new("djxl").arg("--version").output().is_ok()
}

#[test]
fn black_box_decoder_agrees_on_learned_trees() {
    if !djxl_available() {
        eprintln!("skip: no black-box decoder installed");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-jxl-r487-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cases = [
        (
            structured(301, 277, PixelFormat::Rgb24, 8, 1),
            Effort::Balanced,
        ),
        (
            structured(520, 300, PixelFormat::Rgba, 8, 2),
            Effort::Thorough,
        ),
        (
            structured(190, 130, PixelFormat::Gray8, 8, 3),
            Effort::Balanced,
        ),
        (structured(190, 130, PixelFormat::Rgb24, 8, 4), Effort::Fast),
        // Palettes: one group and several, with and without alpha.
        (
            few_colours(120, 90, PixelFormat::Rgb24, 13, 5),
            Effort::Balanced,
        ),
        (
            few_colours(600, 300, PixelFormat::Rgba, 40, 6),
            Effort::Balanced,
        ),
        (
            few_colours(300, 280, PixelFormat::Rgb24, 700, 7),
            Effort::Balanced,
        ),
    ];
    for (i, (img, effort)) in cases.iter().enumerate() {
        let bytes = round_trip(img, &EncodeOptions::new().with_effort(*effort), "djxl case");
        let jxl = dir.join(format!("c{i}.jxl"));
        let out = dir.join(format!("c{i}.ppm"));
        std::fs::write(&jxl, &bytes).unwrap();
        let ext = if img.format == PixelFormat::Rgba {
            "pam"
        } else if img.format == PixelFormat::Gray8 {
            "pgm"
        } else {
            "ppm"
        };
        let out = out.with_extension(ext);
        let status = Command::new("djxl").arg(&jxl).arg(&out).output().unwrap();
        assert!(
            status.status.success(),
            "black-box decoder refused case {i}: {}",
            String::from_utf8_lossy(&status.stderr)
        );
        let raw = std::fs::read(&out).unwrap();
        let want = img.as_bytes().unwrap();
        // The sample payload is the tail of every netpbm flavour.
        assert!(raw.len() > want.len(), "case {i}: short output");
        assert_eq!(&raw[raw.len() - want.len()..], want, "case {i}: pixels");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
