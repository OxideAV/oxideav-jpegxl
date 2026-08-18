//! Round 444 — the §C.8.3 entropy layer and the VarDCT reconstruction
//! chain, root-caused against the wire and pinned on locally generated
//! fixtures (independent encoder, black-box invocations; every
//! `_expected.png` is the black-box reference decode of the committed
//! stream).
//!
//! ## What round 444 fixed (each pinned below)
//!
//! 1. **§C.8.3 reads are D.3.6 hybrid-integer reads.** The
//!    histogram-backed path returned raw entropy tokens, silently
//!    truncating every value ≥ the cluster's `split` and skipping its
//!    raw bits (desync on impulse-heavy blocks — the round-437/441
//!    "synthetic-content VarDCT accuracy deficiency").
//! 2. **Per-section stream lifecycle.** Each PassGroup section is its
//!    own entropy stream (D.3.3): fresh `u(32)` ANS init per section
//!    (previously silently skipped for sections ≥ 1 on ANS streams),
//!    terminal-state check per section (recorded via
//!    `section_closure_failures`), fresh D.3.6 LZ77/window state, and
//!    a recorded `walk_underruns` diagnostic when a §C.8.3 walk ends
//!    with declared non-zeros missing — neither is silent any more.
//! 3. **Listing I.4 square/tall IDCT orientation.** The pre-transpose
//!    branch ran for every shape; square and tall blocks decoded with
//!    the coefficient axes swapped (masked inside the sub-1/255 band
//!    on photo content, fatal on single-basis/impulse content).
//! 4. **§F.3 / §C.6.2 global quantization scale.** The dequantization
//!    matrices carry `2^16 / global_scale`, omitted by the FDIS text
//!    for the HF path (its LF sibling is explicit in Listing C.1).
//!    Wire-fit on five probe streams spanning `global_scale`
//!    1022..10223: reference amplitude ratio ≡ `65536 / global_scale`
//!    (±2..9 %, the Listing F.2 bias adjustment).
//! 5. **Listing I.16 LLF normalisation carries the Listing I.15
//!    `C(c, 8c, u)` boundary term per axis** over the I.2.1-normalised
//!    DCT (refining the round-385 "no factor" reading, which had been
//!    arbitrated atop defects 3 + 4).
//! 6. **§C.7.1 per-channel permutation assignment is channel-index
//!    order X, Y, B** (round-437 assumed the §C.8.3 decode order
//!    Y, X, B; the assignment is invisible to bit-position oracles and
//!    only an impulse specimen with a signalled OrderId-1 permutation
//!    arbitrates it).
//!
//! ## Fixtures
//!
//! * `r444_onedot.jxl` — 64×64 mid-grey, single white pixel at
//!   (35, 29); distance 1. The round-441 standalone reproducer shape:
//!   the dot block's declared NonZeros exceeded the decoded count
//!   before fix 1. Now inside the ±1/255 reference band.
//! * `r444_minidots.jxl` — 64×64 grey with a 24-px dot lattice,
//!   distance 1, encoder restoration filters off. Signals
//!   `used_orders = 0x2` (a §C.3.2 permutation for OrderId 1) and
//!   codes the dots as Hornuss blocks — the fix-6 arbitration
//!   specimen. Bit-band ±1/255.
//! * `r444_basis32.jxl` — 32×32 single-varblock stream whose content
//!   is two DCT basis functions (horizontal×vertical frequency pairs
//!   (3,2)/(2,3), amplitudes 50/25); distance 1. Arbitrates fixes
//!   3, 4, 5 (a transposed IDCT moves the components to the mirrored
//!   cells; a wrong scale/LLF factor changes their amplitude).
//!   ±1/255.
//! * `r444_basis64.jxl` — 64×64 single-Dct64x64 stream, sixteen
//!   bin-aligned basis components (569 declared non-zeros, |q| up to
//!   ~500); distance 1, filters off. Exercises large-token hybrid
//!   completion through the full §C.8.3 walk with D.3.3 closure.
//!   ±1/255.
//! * `r444_wave64.jxl` — 64×64 single-Dct64x64 stream of a
//!   non-bin-aligned sine/cosine product (spectral leakage: many
//!   small coefficients, near-uniform non-dyadic §C.7.2 histograms);
//!   distance 1. The **known open deficiency**: its section decodes
//!   with a D.3.3 terminal-state miss (counted, not silent) and a
//!   bounded pixel residual. The test pins BOTH the failure count and
//!   the residual band so any change in either direction is loud.

use oxideav_jpegxl::decode_all_frames;
use oxideav_jpegxl::hf_coefficient_histograms::{
    reset_section_closure_failures, section_closure_failures,
};
use oxideav_jpegxl::pass_group_hf::{reset_walk_underruns, walk_underruns};
use std::io::Cursor;

fn png_rgb(bytes: &[u8]) -> (usize, usize, Vec<u8>) {
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let mut reader = decoder.read_info().expect("png header");
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).expect("png frame");
    assert_eq!(info.color_type, png::ColorType::Rgb);
    buf.truncate(info.buffer_size());
    (info.width as usize, info.height as usize, buf)
}

/// Per-channel (MAD, max) against a reference decode.
fn compare_rgb(
    frame: &oxideav_core::VideoFrame,
    w: usize,
    h: usize,
    want: &[u8],
) -> [(f64, u32); 3] {
    assert_eq!(frame.planes.len(), 3, "expected RGB planes");
    let mut out = [(0f64, 0u32); 3];
    for c in 0..3 {
        let plane = &frame.planes[c];
        let mut sum = 0u64;
        let mut max = 0u32;
        for y in 0..h {
            for x in 0..w {
                let got = plane.data[y * plane.stride + x] as i32;
                let exp = want[(y * w + x) * 3 + c] as i32;
                let d = (got - exp).unsigned_abs();
                sum += d as u64;
                if d > max {
                    max = d;
                }
            }
        }
        out[c] = (sum as f64 / (w * h) as f64, max);
    }
    out
}

fn assert_reference_band(jxl: &[u8], png: &[u8], max_allowed: u32, name: &str) {
    let (w, h, want) = png_rgb(png);
    reset_section_closure_failures();
    reset_walk_underruns();
    let frames = decode_all_frames(jxl, None).expect("stream decodes");
    assert_eq!(
        section_closure_failures(),
        0,
        "{name}: every §C.8.3 section must close on the D.3.3 terminal state"
    );
    assert_eq!(
        walk_underruns(),
        0,
        "{name}: no §C.8.3 walk may end with declared non-zeros missing"
    );
    assert_eq!(frames.len(), 1);
    let stats = compare_rgb(&frames[0], w, h, &want);
    for (c, &(mad, max)) in stats.iter().enumerate() {
        assert!(
            max <= max_allowed,
            "{name} channel {c}: max {max} > {max_allowed} (MAD {mad})"
        );
    }
}

/// Fix 1 + 3 + 4: the round-441 impulse reproducer — a single-pixel
/// dot on a flat field — decodes inside the ±1/255 reference band
/// (r441 measured the dot vanishing entirely, with
/// `remaining_non_zeros > 0` silently accepted on its block).
#[test]
fn round444_single_impulse_reference_band() {
    assert_reference_band(
        include_bytes!("fixtures/r444_onedot.jxl"),
        include_bytes!("fixtures/r444_onedot_expected.png"),
        1,
        "r444_onedot",
    );
}

/// Fix 6 (+1): custom coefficient orders with impulse content. The
/// OrderId-1 §C.3.2 permutations are assigned X, Y, B by channel
/// index; under the round-437 Y-first assignment the Y-channel
/// Hornuss corner coefficient lands on the wrong cell and the dots
/// vanish (round-441's `dots` reproducer class).
#[test]
fn round444_custom_orders_impulse_lattice_reference_band() {
    assert_reference_band(
        include_bytes!("fixtures/r444_minidots.jxl"),
        include_bytes!("fixtures/r444_minidots_expected.png"),
        1,
        "r444_minidots",
    );
}

/// Fix 3 + 4 + 5: two known basis functions in one Dct32x32 varblock
/// reconstruct at the right cells with the right amplitude. Before
/// round 444 this fixture decoded transposed AND ~10× attenuated
/// (`2^16/global_scale` missing) with the LLF boundary term dropped.
#[test]
fn round444_basis32_orientation_and_scale_reference_band() {
    assert_reference_band(
        include_bytes!("fixtures/r444_basis32.jxl"),
        include_bytes!("fixtures/r444_basis32_expected.png"),
        1,
        "r444_basis32",
    );
}

/// Fix 1 at scale: a Dct64x64 walk with 569 declared non-zeros and
/// quantised values up to ~±500 (hybrid-completion tokens far past
/// every cluster split) decodes bit-band exact with D.3.3 closure.
#[test]
fn round444_basis64_large_token_walk_reference_band() {
    assert_reference_band(
        include_bytes!("fixtures/r444_basis64.jxl"),
        include_bytes!("fixtures/r444_basis64_expected.png"),
        1,
        "r444_basis64",
    );
}

/// The round-444 "wave-leakage desync" — CLOSED in round 448. The
/// deficiency was never a histogram-prelude problem: the 2021 FDIS
/// Listing C.14 prints `non_zeros > size/16 → prev = 1` for a block's
/// FIRST coefficient read while the 2024 IS (I.4) prints `→ prev = 0`;
/// this crate implemented the 2021 text. Blocks opening with
/// 1 ≤ non_zeros ≤ size/16 routed their first read to the wrong
/// cluster whenever the stream's cluster map splits the two contexts —
/// exactly the near-uniform non-dyadic histograms this fixture
/// produces. With the 2024 reading (wire-arbitrated byte-exactly
/// against original-JPEG oracles in `round448_jpeg_reconstruct`) the
/// stream closes and the pixels land in the reference band, per the
/// round-444 pin's own tightening instruction.
#[test]
fn round444_wave64_closure_deficiency_closed() {
    let jxl = include_bytes!("fixtures/r444_wave64.jxl");
    let (w, h, want) = png_rgb(include_bytes!("fixtures/r444_wave64_expected.png"));
    reset_section_closure_failures();
    reset_walk_underruns();
    let frames = decode_all_frames(jxl, None).expect("stream decodes");
    assert_eq!(
        section_closure_failures(),
        0,
        "the round-448 prev fix closed this stream; a regression reopened it"
    );
    assert_eq!(frames.len(), 1);
    let stats = compare_rgb(&frames[0], w, h, &want);
    for (c, &(mad, max)) in stats.iter().enumerate() {
        assert!(
            mad < 1.0 && max <= 2,
            "channel {c}: MAD {mad} (bound 1.0), max {max} (bound 2)"
        );
    }
}
