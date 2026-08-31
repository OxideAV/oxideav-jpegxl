#![no_main]

//! Panic-free fuzz target for the codestream-preamble probes:
//! [`oxideav_jpegxl::probe`] (committee-draft `SizeHeader` +
//! `ImageMetadata` layout) and [`oxideav_jpegxl::probe_fdis`] (the
//! full FDIS Table A.3 + Table A.16 bundle, including the BitDepth /
//! ExtraChannelInfo / ColourEncoding / ToneMapping / extensions tail
//! and the Table A.16 `default_transform` gating).
//!
//! Both probes run signature detection first (raw `FF 0A` codestream
//! vs the 18181-2 box form), extract the codestream from `jxlc` /
//! `jxlp` boxes when wrapped, and then bit-parse the preamble. None of
//! the parses allocate per-pixel buffers, so no geometry cap is
//! needed — the assertion is purely "returns `Result`, never panics".
//!
//! ## Input cap
//!
//! 256 KiB: the ISOBMFF extraction path concatenates `jxlp` payloads
//! into an owned buffer, which is linear in the input size.

use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 256 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    let _ = oxideav_jpegxl::probe(data);
    let _ = oxideav_jpegxl::probe_fdis(data);
});
