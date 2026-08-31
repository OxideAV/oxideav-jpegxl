#![no_main]

//! Structure-aware fuzz target for the self-contained Modular
//! single-channel decode
//! ([`oxideav_jpegxl::modular::decode_single_channel`]): channel
//! header, MA decision-tree decode, and the per-pixel
//! predictor + entropy loop — the deepest attacker-reachable Modular
//! surface below the frame layer, exercised here without needing a
//! whole well-formed codestream around it.
//!
//! The first four fuzz bytes choose the channel geometry
//! (`width = 1 + b0 % 256` weighted by `b1`, same for height, both
//! capped at 512), so the tree/pixel loop runs against arbitrary
//! `(geometry, bitstream)` pairings — deep or degenerate MA trees,
//! header ranges that disagree with the payload, truncated ABRAC
//! streams. The tree size and every channel reservation must stay
//! fenced against the actual bytes; any panic or unbounded
//! allocation is a library bug.
//!
//! ## Input cap
//!
//! 64 KiB (the surface is per-pixel work over at most 512×512).

use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 64 * 1024;
const MAX_DIM: u32 = 512;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES || data.len() < 4 {
        return;
    }
    let w = 1 + (u32::from(data[0]) | (u32::from(data[1] & 1) << 8)) % MAX_DIM;
    let h = 1 + (u32::from(data[2]) | (u32::from(data[3] & 1) << 8)) % MAX_DIM;
    let payload = &data[4..];
    let _ = oxideav_jpegxl::modular::decode_single_channel(payload, w, h, 4);
});
