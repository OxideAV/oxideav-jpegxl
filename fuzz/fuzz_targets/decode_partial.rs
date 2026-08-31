#![no_main]

//! Panic-free fuzz target for progressive / partial-input behaviour:
//! the capped multi-frame decode over a fuzz-chosen prefix of the
//! input. A JXL byte stream can end anywhere — a network fetch cut
//! mid-TOC, mid-ANS-stream, mid-section — and the decoder must
//! surface a clean `Err` from every truncation point, never a panic,
//! an out-of-bounds read past the prefix, or a reservation sized by
//! bytes that never arrived.
//!
//! The last input byte selects the truncation fraction
//! (`keep = len × b / 255`), so corpus entries seeded from valid
//! streams explore every prefix depth as the fuzzer mutates that
//! byte. Geometry caps are identical to `decode_full`.

use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 256 * 1024;
const MAX_AREA: u64 = 1 << 20;
const MAX_EXTRA: u32 = 4;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES || data.is_empty() {
        return;
    }
    let (&frac, body) = data.split_last().unwrap();
    let keep = body.len() * usize::from(frac) / 255;
    let prefix = &body[..keep];
    let Ok(headers) = oxideav_jpegxl::probe_fdis(prefix) else {
        return;
    };
    let area = u64::from(headers.size.width) * u64::from(headers.size.height);
    if area > MAX_AREA || headers.metadata.num_extra_channels > MAX_EXTRA {
        return;
    }
    let _ = oxideav_jpegxl::decode_all_frames(prefix, None);
});
