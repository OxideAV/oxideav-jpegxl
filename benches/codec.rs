//! Encode / decode timing on a deterministic photo-like picture.
//!
//! ```text
//! cargo bench --bench codec -- <megapixels> <mode> <threads> [what] [runs]
//!   megapixels  1 | 12 (or any number: a 4:3 picture of about that size)
//!   mode        lossless | lossy
//!   threads     worker budget for the encoder and the decoder
//!   what        both (default) | encode | decode
//!   runs        repetitions the median is taken over (default 3; 1
//!               makes the process's peak resident set that of one
//!               operation — freed pages of earlier runs stay resident)
//! ```
//!
//! One process per measurement, so the peak resident set of the run
//! (`/usr/bin/time -l` on macOS, `-v` on Linux) is the peak of that
//! one operation; `decode` reads the file a previous `encode` run left
//! in the directory named by `JXL_BENCH_DIR` (the system temp
//! directory by default). Prints one line per operation with the
//! median of three runs, and the peak of live heap bytes during the
//! operation (a counting allocator: the picture or file it works on is
//! included, pages the allocator keeps after a free are not).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use oxideav_jpegxl::{DecodeOptions, EncodeOptions, JxlImage, decode_with, encode};

/// The system allocator with a running total and its high-water mark,
/// so a run can report the peak of live heap bytes whatever the host's
/// resident-set accounting does with freed pages.
struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn note(grown: usize) {
    let now = LIVE.fetch_add(grown, Ordering::Relaxed) + grown;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

// SAFETY: every call is forwarded unchanged to the system allocator;
// the counters only observe sizes.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // A block that moves exists twice while it is copied.
        note(new_size);
        let moved = unsafe { System.realloc(ptr, layout, new_size) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        moved
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn lattice(x: i64, y: i64, seed: u64) -> f32 {
    let mut h = (x as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
        .wrapping_add(seed.wrapping_mul(0x1656_67B1_9E37_79F9));
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h & 0xFFFF) as f32 / 65535.0
}

/// Bilinear value noise at one scale.
fn noise(x: f32, y: f32, seed: u64) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let (ix, iy) = (x0 as i64, y0 as i64);
    let a = lattice(ix, iy, seed);
    let b = lattice(ix + 1, iy, seed);
    let c = lattice(ix, iy + 1, seed);
    let d = lattice(ix + 1, iy + 1, seed);
    a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy
}

/// A picture with the statistics of a photograph: 1/f structure over
/// eight octaves, hard-edged regions, correlated colour, sensor noise.
fn photo(width: u32, height: u32) -> JxlImage {
    let mut data = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            let (fx, fy) = (x as f32, y as f32);
            let mut luma = 0.0f32;
            let mut amp = 0.36f32;
            let mut scale = 1.0 / 300.0f32;
            for o in 0..8u64 {
                luma += amp * noise(fx * scale, fy * scale, o);
                amp *= 0.68;
                scale *= 2.0;
            }
            // Hard edges: a region map at a coarse scale.
            let region = noise(fx / 190.0, fy / 190.0, 77);
            if region > 0.55 {
                luma = luma * 0.6 + 0.3;
            }
            let cb = noise(fx / 420.0, fy / 420.0, 101) - 0.5;
            let cr = noise(fx / 360.0, fy / 360.0, 202) - 0.5;
            let grain = |c: u64| (lattice(x as i64, y as i64, 900 + c) - 0.5) * 0.03;
            let r = luma + 0.7 * cr + grain(0);
            let g = luma - 0.25 * cb - 0.35 * cr + grain(1);
            let b = luma + 0.9 * cb + grain(2);
            for v in [r, g, b] {
                data.push((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            }
        }
    }
    JxlImage::from_rgb8(width, height, data).expect("valid geometry")
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| !a.starts_with("--"))
        .collect();
    let mp: f64 = args.first().and_then(|a| a.parse().ok()).unwrap_or(1.0);
    let mode = args.get(1).map(String::as_str).unwrap_or("lossless");
    let threads: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(1);
    let what = args.get(3).map(String::as_str).unwrap_or("both");
    let height = ((mp * 1.0e6 * 3.0 / 4.0).sqrt().round() as u32).max(8);
    let width = height * 4 / 3;
    let runs: usize = args
        .get(4)
        .and_then(|a| a.parse().ok())
        .filter(|&r| r > 0)
        .unwrap_or(3);
    let dir = std::env::var("JXL_BENCH_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    let path = dir.join(format!("oxideav-jxl-bench-{width}x{height}-{mode}.jxl"));
    let opts = match mode {
        "lossy" => EncodeOptions::new().with_distance(1.0),
        _ => EncodeOptions::new(),
    }
    .with_threads(threads);
    let label = format!("{width}x{height} {mode} threads={threads}");
    if what != "decode" {
        let img = photo(width, height);
        let mut times = Vec::new();
        let mut bytes = Vec::new();
        let mut peaks = Vec::new();
        for _ in 0..runs {
            drop(std::mem::take(&mut bytes));
            PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
            let t = Instant::now();
            bytes = encode(&img, &opts).expect("encode");
            times.push(t.elapsed().as_secs_f64() * 1000.0);
            peaks.push(PEAK.load(Ordering::Relaxed) as f64 / (1 << 20) as f64);
        }
        std::fs::write(&path, &bytes).expect("write");
        println!(
            "{label} encode {:.0} ms {} bytes, peak heap {:.0} MiB (picture included)",
            median(times),
            bytes.len(),
            median(peaks)
        );
    }
    if what != "encode" {
        let bytes = std::fs::read(&path).expect("run the encode step first");
        let dopts = DecodeOptions::default().with_threads(threads);
        let mut times = Vec::new();
        let mut peaks = Vec::new();
        for _ in 0..runs {
            PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
            let t = Instant::now();
            let img = decode_with(&bytes, &dopts).expect("decode");
            times.push(t.elapsed().as_secs_f64() * 1000.0);
            peaks.push(PEAK.load(Ordering::Relaxed) as f64 / (1 << 20) as f64);
            assert_eq!((img.width, img.height), (width, height));
        }
        println!(
            "{label} decode {:.0} ms, peak heap {:.0} MiB (file included)",
            median(times),
            median(peaks)
        );
    }
}
