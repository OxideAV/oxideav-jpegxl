#![no_main]

//! Panic-free fuzz target for the end-to-end multi-frame decode
//! ([`oxideav_jpegxl::decode_all_frames`]): container strip, preamble
//! + ICC, then the §C.1 frame loop — FrameHeader (both
//! RestorationFilter editions), permuted/LZ77 TOCs, the Modular and
//! VarDCT paths, §C.2 composition/blending, reference frames, and the
//! §K image features.
//!
//! ## Geometry caps (documented)
//!
//! A JXL SizeHeader can declare up to 2^30-px dimensions while the
//! actual input is a handful of bytes, and the decoder allocates
//! per-pixel plane buffers, so the harness pre-probes the preamble
//! and skips inputs that declare:
//!
//! * pixel area > 2^20 samples (`MAX_AREA`),
//! * more than 4 extra channels (`MAX_EXTRA`).
//!
//! Below those caps the library must fail cleanly on its own: any
//! panic, OOM (rss-limited run) or overflow inside the cap is a
//! library bug, not a harness artefact.
//!
//! ## Input cap
//!
//! 256 KiB raw input.

use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 256 * 1024;
const MAX_AREA: u64 = 1 << 20;
const MAX_EXTRA: u32 = 4;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    let Ok(headers) = oxideav_jpegxl::probe_fdis(data) else {
        return;
    };
    let area = u64::from(headers.size.width) * u64::from(headers.size.height);
    if area > MAX_AREA || headers.metadata.num_extra_channels > MAX_EXTRA {
        return;
    }
    let _ = oxideav_jpegxl::decode_all_frames(data, None);
});
