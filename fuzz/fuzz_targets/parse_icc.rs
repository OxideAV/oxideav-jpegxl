#![no_main]

//! Panic-free fuzz target for the Annex B / Annex E ICC surface:
//!
//! * [`oxideav_jpegxl::icc::decode_encoded_icc_stream`] — the
//!   `enc_size = U64()` read plus the 41-distribution §D.3 entropy
//!   stream that yields the *encoded* ICC byte stream (E.4.1), driven
//!   from an arbitrary bit position;
//! * [`oxideav_jpegxl::icc::reconstruct_icc_profile`] — the
//!   E.4.2..E.4.5 command/data-stream reconstruction (varint sizes,
//!   predicted header, tag list, command interpretation), fed the raw
//!   fuzz bytes directly as an "encoded" stream.
//!
//! Both layers carry attacker-controlled sizes (`enc_size`,
//! `output_size`, `commands_size`, per-command varints); the library
//! fences each reservation against its documented cap and the actual
//! remaining bytes, and this harness asserts those fences hold —
//! every call returns a `Result` rather than panicking or allocating
//! an unbounded buffer.
//!
//! ## Input cap
//!
//! 256 KiB — both surfaces are linear in the input once the caps
//! hold.

use libfuzzer_sys::fuzz_target;
use oxideav_jpegxl::bitreader::BitReader;
use oxideav_jpegxl::icc;

const MAX_INPUT_BYTES: usize = 256 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    let mut br = BitReader::new(data);
    if let Ok(encoded) = icc::decode_encoded_icc_stream(&mut br) {
        let _ = icc::reconstruct_icc_profile(&encoded);
    }
    // The raw bytes as an already-decoded "encoded stream" — this is
    // the layer the container path reaches after the entropy decode.
    let _ = icc::reconstruct_icc_profile(data);
});
