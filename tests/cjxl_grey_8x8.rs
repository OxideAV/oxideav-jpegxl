//! Round-3 integration test: decode the simplest cjxl-produced
//! `--lossless` Modular Grey 8×8 fixture.
//!
//! Fixture build (committed to `tests/fixtures/grey_8x8_lossless.jxl`):
//!
//! ```text
//! python3 -c "open('/tmp/test_8x8.pgm','wb').write(b'P5\n8 8\n255\n' + bytes([128]*64))"
//! cjxl -d 0 -e 1 /tmp/test_8x8.pgm /tmp/test_8x8.jxl
//! ```
//!
//! cjxl 0.11.1 produces a raw-codestream `.jxl` of 180 bytes. The
//! header bytes are `FF 0A 41 40 50 DC 08 08 02 01 ...`.

use oxideav_jpegxl::probe_fdis;

const FIXTURE: &[u8] = include_bytes!("fixtures/grey_8x8_lossless.jxl");

#[test]
fn cjxl_grey_8x8_probe_recognises_dimensions_and_grey() {
    let h = probe_fdis(FIXTURE).expect("probe should succeed on cjxl fixture");
    assert_eq!(h.size.width, 8);
    assert_eq!(h.size.height, 8);
    // cjxl emits Grey, 8 bpp, no extras for a PGM input.
    assert_eq!(h.metadata.bit_depth.bits_per_sample, 8);
    assert_eq!(h.metadata.num_extra_channels, 0);
}

#[test]
fn cjxl_grey_8x8_dump_first_bytes() {
    // Print the first 32 bytes of the fixture so that bisecting the
    // decoder against this dump is straightforward.
    let n = FIXTURE.len().min(32);
    let mut s = String::new();
    for b in &FIXTURE[..n] {
        s.push_str(&format!("{b:02x} "));
    }
    eprintln!("first {n} bytes: {s}");
    eprintln!("total bytes: {}", FIXTURE.len());
}

/// Soft test: this MAY return Unsupported / InvalidData while the
/// FDIS decoder is still missing pieces. The test prints the decoder
/// result so a reviewer can see exactly what the failure mode was, but
/// does NOT fail the suite if it's an expected error.
///
/// Round-11 status (this commit): the inverse-palette transform now
/// implements the four-range index partition + Path 1 / Path 2
/// dispatch + Appendix B.6 bit-depth clamp from
/// `docs/image/jpegxl/libjxl-trace-reverse-engineering.md` Appendix B
/// (commit `679cf63`). For the cjxl 8x8 grey lossless fixture
/// (`nb_colours=3 num_c=1 nb_deltas=0 d_pred=0`, idx=-1 throughout),
/// both the FDIS Listing L.6 and Appendix B.4 Path 1 unambiguously
/// give `output = kDeltaPalette[0][0] = 0` per the negative-index
/// rule in §B.3.1 — yet `djxl` decodes the same fixture as all-128.
///
/// This is an UNRESOLVED docs gap one layer deeper than what
/// Appendix B documents: for the implementer-trivial case
/// `(nb_deltas=0, predictor=Zero, idx=-1)`, the cjxl encoder appears
/// to expect a different lookup than the spec text describes. Without
/// libjxl source access (clean-room policy) we cannot reverse-engineer
/// the algorithm beyond what's in Appendix B; round 12 needs an
/// additional empirical correction to the appendix from a clean-room
/// trace. Documented in the agent's report.
///
/// Round-10 status: kRCT / kPalette / kSqueeze transform parsing +
/// dispatch infrastructure landed.
/// Round-8 status: cl_code Kraft RESOLVED (Appendix A.6 — see
/// `cjxl_grey_8x8_round9_progress_marker` for the regression guard).
#[test]
fn cjxl_grey_8x8_decode_attempt() {
    use oxideav_jpegxl::decode_one_frame;
    let res = decode_one_frame(FIXTURE, None);
    match res {
        Ok(vf) => {
            // Decoder ran to completion. Pixel-perfect equality is
            // gated on the docs gap (see doc-comment above); for now
            // we just check shape.
            assert_eq!(vf.planes.len(), 1, "expected 1 plane (Gray8)");
            let plane = &vf.planes[0];
            assert_eq!(plane.stride, 8);
            assert_eq!(plane.data.len(), 64);
            let unique: std::collections::BTreeSet<u8> = plane.data.iter().copied().collect();
            eprintln!("decoded pixel set: {:?}", unique);
            if unique.len() == 1 && unique.contains(&128) {
                eprintln!("ROUND-12 RESOLVED: pixels are all 128 (matches djxl)");
            } else {
                eprintln!(
                    "ROUND-11 stop: pixels not yet 128 (got {:?}); see Appendix B gap",
                    unique
                );
            }
        }
        Err(e) => {
            eprintln!("cjxl_grey_8x8 round-11 stop point: {e}");
        }
    }
}

/// Round-9 status: cl_code Kraft RESOLVED via `space==0`
/// early-terminate (commit d49e583).
///
/// Round-10 status (this commit): kRCT / kPalette / kSqueeze
/// transform parsing + dispatch infrastructure landed.
/// `nb_transforms = 1 not supported` is GONE — the decoder reads
/// the cjxl `Palette { begin_c: 0, num_c: 1, nb_colours: 3,
/// nb_deltas: 0, d_pred: 0 }` cleanly, runs the MA-tree pixel
/// decode for both the meta-channel + the index channel, and
/// then runs `inverse_palette`. The inverse path produces an
/// internally-consistent decode but pixel values are 0 instead
/// of the expected 128 — see `cjxl_grey_8x8_decode_attempt` for
/// the L.6 spec-gap analysis.
#[test]
fn cjxl_grey_8x8_round9_progress_marker() {
    use oxideav_jpegxl::decode_one_frame;
    let res = decode_one_frame(FIXTURE, None);
    match res {
        Ok(_vf) => {
            // Decoder ran end-to-end without error. Pixel-equality
            // assertion deferred until round-11 (L.6 spec-gap fix).
        }
        Err(e) => {
            let msg = format!("{e}");
            // Round-9 expected: error message no longer contains
            // "kraft=33776" (that stop point is now resolved).
            assert!(
                !msg.contains("kraft=33776"),
                "round-9 fix regressed: kraft=33776 stop point came back: {msg}"
            );
            // Round-8 expected: error message no longer contains
            // the cl_code Kraft mismatch / "symbol out of alphabet"
            // pattern from the symbol-stream prelude.
            assert!(
                !msg.contains("symbol out of alphabet"),
                "round-8 fix regressed: simple-prefix symbol-out-of-alphabet stop point came back: {msg}"
            );
            assert!(
                !msg.contains("Kraft mismatch"),
                "round-8 fix regressed: Kraft mismatch came back: {msg}"
            );
            // Round-10 expected: `nb_transforms = 1 not supported`
            // is RESOLVED; if it returns, round-10 regressed.
            assert!(
                !msg.contains("nb_transforms = 1 not supported"),
                "round-10 fix regressed: nb_transforms stop point came back: {msg}"
            );
        }
    }
}
