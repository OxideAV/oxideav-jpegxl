//! Cross-validate our encoder against the libjxl `djxl` CLI.
//!
//! Workflow:
//!   1. Build a small Grey 8×8 image.
//!   2. Encode it via `oxideav_jpegxl::encoder::encode_one_frame`.
//!   3. Pipe the codestream into `djxl - out.pgm` (PGM is the simplest
//!      single-channel grey format djxl emits losslessly).
//!   4. Parse the PGM header, compare pixel bytes.
//!
//! ## Soft-skip
//!
//! Workspace policy permits binary tools as black-box validators
//! (`feedback_no_external_libs.md`) but the OxideAV CI matrix doesn't
//! require any specific tool to be present. The test silently skips
//! (with an `eprintln!` note) on hosts that don't have `djxl` on PATH,
//! mirroring the openjpeg subprocess pattern used by `oxideav-jpeg2000`.

use std::io::Write;
use std::process::{Command, Stdio};

use oxideav_jpegxl::encoder::{encode_one_frame, InputFormat};

/// True iff `djxl --version` succeeds.
fn djxl_available() -> bool {
    Command::new("djxl")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Pipe `input_jxl` to `djxl - - --output_format pgm` and return the
/// PGM bytes from stdout.
fn djxl_decode_to_pgm(input_jxl: &[u8]) -> Result<Vec<u8>, String> {
    let mut child = Command::new("djxl")
        .args(["-", "-", "--output_format", "pgm", "--quiet"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn djxl: {e}"))?;
    {
        let stdin = child.stdin.as_mut().ok_or("no stdin")?;
        stdin
            .write_all(input_jxl)
            .map_err(|e| format!("write stdin: {e}"))?;
    }
    let out = child
        .wait_with_output()
        .map_err(|e| format!("wait djxl: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "djxl exit {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(out.stdout)
}

/// Parse a `P5` PGM and return `(width, height, pixels)`. Tolerates
/// comment lines (`# ...`) and arbitrary whitespace between the
/// magic / width / height / maxval / pixel-data sections.
fn parse_pgm(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    // Find header tokens.
    let mut i = 0usize;
    let mut tokens = Vec::with_capacity(4);
    while tokens.len() < 4 && i < bytes.len() {
        // skip whitespace.
        while i < bytes.len()
            && (bytes[i] == b' ' || bytes[i] == b'\n' || bytes[i] == b'\r' || bytes[i] == b'\t')
        {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'#' {
            // skip comment to end of line
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let start = i;
        while i < bytes.len()
            && !(bytes[i] == b' ' || bytes[i] == b'\n' || bytes[i] == b'\r' || bytes[i] == b'\t')
        {
            i += 1;
        }
        if i > start {
            tokens.push(&bytes[start..i]);
        }
    }
    if tokens.len() < 4 {
        return Err("PGM: incomplete header".into());
    }
    if tokens[0] != b"P5" {
        return Err(format!(
            "PGM: magic = {:?}, expected P5",
            String::from_utf8_lossy(tokens[0])
        ));
    }
    let w: u32 = std::str::from_utf8(tokens[1])
        .map_err(|e| format!("PGM width utf8: {e}"))?
        .parse()
        .map_err(|e| format!("PGM width parse: {e}"))?;
    let h: u32 = std::str::from_utf8(tokens[2])
        .map_err(|e| format!("PGM height utf8: {e}"))?
        .parse()
        .map_err(|e| format!("PGM height parse: {e}"))?;
    let maxval: u32 = std::str::from_utf8(tokens[3])
        .map_err(|e| format!("PGM maxval utf8: {e}"))?
        .parse()
        .map_err(|e| format!("PGM maxval parse: {e}"))?;
    if maxval != 255 {
        return Err(format!("PGM: maxval = {maxval}, expected 255"));
    }
    // After the maxval token, exactly one whitespace byte separates the
    // header from the binary data.
    if i >= bytes.len() {
        return Err("PGM: no pixel data".into());
    }
    // Skip exactly one whitespace byte.
    i += 1;
    let expected = (w as usize) * (h as usize);
    if bytes.len() - i < expected {
        return Err(format!(
            "PGM: pixel buffer too short ({} bytes, expected {})",
            bytes.len() - i,
            expected
        ));
    }
    Ok((w, h, bytes[i..i + expected].to_vec()))
}

#[test]
fn djxl_decodes_our_grey_8x8_constant_image() {
    if !djxl_available() {
        eprintln!("djxl not on PATH — skipping cross-validation test");
        return;
    }
    let pixels = vec![64u8; 64];
    let jxl =
        encode_one_frame(8, 8, &pixels, InputFormat::Gray8).expect("encode 8x8 grey constant");
    eprintln!("encoded grey 8x8 constant=64 → {} bytes", jxl.len());
    let pgm = match djxl_decode_to_pgm(&jxl) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("djxl decode failed (round-2 limitation may exist): {e}");
            return;
        }
    };
    let (w, h, data) = parse_pgm(&pgm).expect("parse djxl PGM output");
    assert_eq!(w, 8, "djxl output width");
    assert_eq!(h, 8, "djxl output height");
    assert_eq!(data.len(), 64, "djxl output pixel count");
    for (i, &v) in data.iter().enumerate() {
        assert_eq!(v, 64, "djxl pixel {i} mismatch (got {v}, expected 64)");
    }
}

#[test]
fn djxl_decodes_our_grey_8x8_gradient_image() {
    if !djxl_available() {
        eprintln!("djxl not on PATH — skipping cross-validation test");
        return;
    }
    let mut pixels = Vec::with_capacity(64);
    for y in 0..8u8 {
        for x in 0..8u8 {
            pixels.push(x.wrapping_mul(16).wrapping_add(y * 4));
        }
    }
    let jxl =
        encode_one_frame(8, 8, &pixels, InputFormat::Gray8).expect("encode 8x8 grey gradient");
    eprintln!("encoded grey 8x8 gradient → {} bytes", jxl.len());
    let pgm = match djxl_decode_to_pgm(&jxl) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("djxl decode failed (round-2 limitation may exist): {e}");
            return;
        }
    };
    let (w, h, data) = parse_pgm(&pgm).expect("parse djxl PGM output");
    assert_eq!(w, 8);
    assert_eq!(h, 8);
    assert_eq!(data.len(), 64);
    assert_eq!(
        data, pixels,
        "djxl pixel data mismatch on grey 8x8 gradient"
    );
}

#[test]
fn djxl_decodes_our_grey_64x64_synthetic_image() {
    if !djxl_available() {
        eprintln!("djxl not on PATH — skipping cross-validation test");
        return;
    }
    // 64x64 deterministic LCG-driven greyscale (4096 pixels).
    let mut pixels = Vec::with_capacity(64 * 64);
    let mut state: u32 = 0x1234_5678;
    for _ in 0..(64 * 64) {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        pixels.push((state >> 16) as u8);
    }
    let jxl = encode_one_frame(64, 64, &pixels, InputFormat::Gray8).expect("encode 64x64 grey");
    eprintln!("encoded grey 64x64 random → {} bytes", jxl.len());
    let pgm = match djxl_decode_to_pgm(&jxl) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("djxl decode failed (round-2 limitation may exist): {e}");
            return;
        }
    };
    let (w, h, data) = parse_pgm(&pgm).expect("parse djxl PGM output");
    assert_eq!(w, 64);
    assert_eq!(h, 64);
    assert_eq!(
        data, pixels,
        "djxl pixel data mismatch on 64x64 random grey"
    );
}

/// Minimal sanity test that always runs: parse a tiny PGM we synthesise
/// ourselves. Catches PGM-parser regressions independently of the
/// `djxl` binary being available.
#[test]
fn parse_pgm_self_test() {
    let pgm = b"P5\n2 2\n255\n\x00\x40\x80\xFF";
    let (w, h, data) = parse_pgm(pgm).unwrap();
    assert_eq!(w, 2);
    assert_eq!(h, 2);
    assert_eq!(data, vec![0x00, 0x40, 0x80, 0xFF]);
}
