#![no_main]

//! Panic-free fuzz target for the first-frame decode entry
//! ([`oxideav_jpegxl::decode_vardct_frame_from_codestream`]) — the
//! VarDCT pipeline surface: LfGlobal (Quantizer / HfBlockContext /
//! LfChannelCorrelation), LfGroup LF coefficients + HfMetadata, the
//! §C.6/§C.7 HfGlobal section (dequant matrices, per-preset
//! coefficient orders with §C.3.2 permutation streams, §C.7.2
//! histogram block), the per-(pass, group) §C.8.3 entropy decode,
//! dequant + IDCT + chroma-from-luma + §J restoration filters.
//!
//! The entry also accepts Modular frames (shared dispatch); seeding
//! this target with VarDCT streams is what points the exploration at
//! the VarDCT stack. Geometry caps are identical to `decode_full`
//! (see that harness's notes): declared pixel area ≤ 2^20 samples,
//! ≤ 4 extra channels — below the caps every failure must be a clean
//! `Err`.

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
    let _ = oxideav_jpegxl::decode_vardct_frame_from_codestream(data, None);
});
