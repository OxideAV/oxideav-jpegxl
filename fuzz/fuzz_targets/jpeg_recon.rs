#![no_main]

//! Panic-free fuzz target for the Annex A JPEG bitstream
//! reconstruction ([`oxideav_jpegxl::jpeg_reconstruct::reconstruct_jpeg`]):
//! the 18181-2 box walk to the `jbrd` box, its Brotli-compressed
//! reconstruction-data parse (A.4 varint fields, marker sequence,
//! quant/Huffman table metadata, padding-bit stream), the RAW-mode
//! dequant-matrix + §C.8.3 coefficient decode of the transcoded
//! VarDCT frame, and the 10918-1 re-encode (sequential and
//! progressive, restart markers, MCU padding, ICC APP2 re-chunking).
//! Round 473 added the pixel path on the same inputs: the contract
//! `decode` of a `do_YCbCr` VarDCT frame runs the identical coefficient
//! decode and then the per-lattice dequant + IDCT, the J.2 chroma
//! upsampling of `to_rgb8` and the planar `YuvJ*` / `Yuv440P` image
//! assembly — every `jpeg_upsampling` lattice (4:4:4 / 4:2:0 / 4:2:2 /
//! 4:4:0, and the luma-subsampled / asymmetric values the decoder must
//! refuse) is attacker-controlled through the frame header, and the
//! corpus is seeded with the committed transcode fixtures of every
//! sampling class.
//!
//! Every field in the `jbrd` payload and the codestream is
//! attacker-controlled; declared JPEG dimensions, component counts,
//! table ids and scan scripts can all disagree with the embedded
//! frame. The harness asserts the reconstruction returns a `Result`
//! rather than panicking, overflowing (debug) or allocating from a
//! declared-but-unbacked size.
//!
//! ## Input + geometry caps
//!
//! 512 KiB raw input (the staged transcode fixtures are ~100 KiB).
//! The reconstruction allocates per-block coefficient canvases from
//! the declared frame geometry, so — like `decode_full` — the harness
//! pre-probes the preamble and skips inputs declaring a pixel area
//! above 2^20 samples; below the cap every failure must be a clean
//! `Err`.

use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512 * 1024;
const MAX_AREA: u64 = 1 << 20;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    let Ok(headers) = oxideav_jpegxl::probe_fdis(data) else {
        return;
    };
    let area = u64::from(headers.size.width) * u64::from(headers.size.height);
    if area > MAX_AREA {
        return;
    }
    let _ = oxideav_jpegxl::jpeg_reconstruct::reconstruct_jpeg(data);
    // The pixel path over the same transcode: planar native decode,
    // then the J.2 + §L.3 conversion of the planar image.
    if let Ok(img) = oxideav_jpegxl::decode(data) {
        let _ = img.try_to_rgb8();
    }
});
