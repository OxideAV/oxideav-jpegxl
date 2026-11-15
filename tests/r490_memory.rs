//! Peak heap of a lossless encode: the coder reads each group from the
//! caller's picture when it codes it, so beyond the picture itself it
//! holds the learner's sample, three bytes per coded symbol and the
//! output — not a plane copy of the picture.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use oxideav_jpegxl::{Effort, EncodeOptions, JxlImage, decode, encode};

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

/// Smooth structure plus noise: residuals of a few bits, like a photo.
fn picture(width: u32, height: u32) -> Vec<u8> {
    let mut s = 0x1234_5678u32;
    let mut data = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let n = (s >> 28) as i32 - 8;
            let base = ((x / 3 + y / 5) % 200) as i32 + 20;
            for c in 0..3 {
                data.push((base + n + c * 7).clamp(0, 255) as u8);
            }
        }
    }
    data
}

// One test in this binary: the counters are process-wide.
#[test]
fn lossless_encode_holds_no_copy_of_the_picture() {
    let (width, height) = (1024u32, 768u32);
    let rgb = picture(width, height);
    let samples = rgb.len();
    let img = JxlImage::from_rgb8(width, height, rgb.clone()).unwrap();
    let opts = EncodeOptions::new().with_effort(Effort::Fast);
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let bytes = encode(&img, &opts).unwrap();
    let extra = PEAK.load(Ordering::Relaxed) - before;
    // While the groups are coded: three bytes per symbol so far, the
    // sections and then the file, one group in flight (its planes,
    // its symbols as pairs, the rolling predictor state). The
    // learner's sample (60 000 samples of 127 bytes at this effort) is
    // gone by then. Integer planes of the picture alone would be four
    // more bytes per sample.
    let budget = 3 * samples + 2 * bytes.len() + (6 << 20);
    assert!(
        extra <= budget,
        "encode peaked {extra} bytes above its input, budget {budget} ({samples} samples, {} \
         bytes out)",
        bytes.len()
    );
    let back = decode(&bytes).unwrap();
    assert_eq!(back.as_bytes().unwrap(), &rgb[..]);
}
