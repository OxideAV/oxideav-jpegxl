#![no_main]

//! Panic-free fuzz target for the 18181-2 box file format
//! (`oxideav_jpegxl::container`):
//!
//! * the Table 4 box layout walk ([`container::BoxIter`]) — `LBox` /
//!   `TBox` / extended `XLBox` length arithmetic, including the
//!   `LBox = 0` "until EOF" form;
//! * the validated whole-file parse ([`container::JxlFile::parse`]) —
//!   Clause 9 "shall" ordering (signature, ftyp, at-most-one jxll,
//!   jxlc XOR ordered jxlp sequence, zero-or-one jxli), the §9.8
//!   Frame Index parse, and `jbrd` presence rules;
//! * the §9.7 `brob` unwrap ([`container::MetadataBox::content`])
//!   with a decompression-bomb output cap;
//! * the codestream extraction ([`container::extract_codestream`])
//!   that concatenates `jxlp` partial payloads in index order.
//!
//! Every length field is attacker-controlled; the harness asserts the
//! walk returns `Result`s rather than panicking, overflowing (debug),
//! or allocating from a declared-but-unbacked size.
//!
//! ## Input + output caps
//!
//! Input 512 KiB; `brob` decompressed content capped at 1 MiB per box.

use libfuzzer_sys::fuzz_target;
use oxideav_jpegxl::container;

const MAX_INPUT_BYTES: usize = 512 * 1024;
const MAX_BROB_OUTPUT: usize = 1024 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    let _ = container::detect(data);
    for item in container::BoxIter::new(data) {
        if item.is_err() {
            break;
        }
    }
    if let Ok(file) = container::JxlFile::parse(data) {
        for md in &file.metadata {
            let _ = md.content(MAX_BROB_OUTPUT);
        }
    }
    let _ = container::extract_codestream(data);
});
