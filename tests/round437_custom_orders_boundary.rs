//! Round 437 → round 454 — the multi-preset §C.7 boundary is CLOSED.
//!
//! Round 437 resolved the §C.3.2/§C.7.1 chain for single-preset
//! single-pass streams (`round437_custom_orders_decode`) and left the
//! multi-preset (`num_hf_presets > 1`) / multi-pass §C.7 slice as a
//! loud refusal pinned here. Round 454 found the remaining wire
//! divergence: the FDIS §C.7.1 lead-in "read `num_hf_presets` times"
//! is SUPERSEDED by ISO/IEC 18181-1:2024 §I.3.1, which reads the HF
//! coefficient orders `order[p][b][c]` ONCE PER PASS with no preset
//! dimension at all — `num_hf_presets` multiplies only the §C.7.2 /
//! I.3.3 histogram count and the I.4 `hfp` histogram-offset
//! selection. With one order bundle per pass the staged
//! `progressive-ac-multipass` fixture (2560×1440, 60 groups, 3
//! passes × 2 presets, `used_orders = 0x5F`) consumes its HfGlobal
//! section to the byte-padded end (slack 6 bits) with every D.3.3
//! ANS closure passing, and the frame decodes END TO END — the first
//! full multi-pass pixel decode.
//!
//! This file now pins that decode:
//!
//! 1. Pixel ratchet against the staged black-box reference decode
//!    (`progressive_ac_multipass_expected.png`).
//! 2. The §C.8.3 loudness diagnostics stay ZERO across the decode
//!    (no silently-accepted section desync or walk underrun).
//! 3. The round-437 Part 8.3 survivor `PermStreamConfig` remains the
//!    shipped default.
use std::io::Cursor;

use oxideav_jpegxl::coeff_order::{PermPrevContext, PermStreamConfig};

const FIXTURE: &[u8] = include_bytes!("fixtures/progressive_ac_multipass.jxl");
const EXPECTED: &[u8] = include_bytes!("fixtures/progressive_ac_multipass_expected.png");

fn png_rgb(bytes: &[u8]) -> (usize, usize, Vec<u8>) {
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    assert_eq!(info.color_type, png::ColorType::Rgb);
    (info.width as usize, info.height as usize, buf)
}

/// The multi-preset multi-pass fixture decodes end to end within a
/// pixel ratchet, with zero §C.8.3 loudness diagnostics.
///
/// Ratchet: measured at MAD 1.97 / 1.36 / 0.68 on landing (round
/// 454) — the first multi-pass decode; the residual (vs the sub-1
/// single-pass band) is cross-pass accumulation accuracy, a
/// follow-up, bounded here at 3.0 so a regression to the old
/// misparse (MAD ≈ tens) can never pass.
#[test]
fn multipass_fixture_decodes_within_ratchet() {
    oxideav_jpegxl::hf_coefficient_histograms::reset_section_closure_failures();
    oxideav_jpegxl::pass_group_hf::reset_walk_underruns();

    let (w, h, reference) = png_rgb(EXPECTED);
    let frame = oxideav_jpegxl::decode_one_frame(FIXTURE, None)
        .expect("round 454: the multi-preset multi-pass fixture decodes (2024 I.3.1 layout)");
    assert_eq!(frame.planes.len(), 3);

    assert_eq!(
        oxideav_jpegxl::hf_coefficient_histograms::section_closure_failures(),
        0,
        "no PassGroup section may end off its D.3.3 final state"
    );
    assert_eq!(
        oxideav_jpegxl::pass_group_hf::walk_underruns(),
        0,
        "no varblock walk may under-deliver its declared NonZeros"
    );

    for c in 0..3usize {
        let plane = &frame.planes[c];
        let mut sum = 0u64;
        for y in 0..h {
            for x in 0..w {
                let d = plane.data[y * plane.stride + x].abs_diff(reference[(y * w + x) * 3 + c]);
                sum += d as u64;
            }
        }
        let mad = sum as f64 / (w * h) as f64;
        assert!(
            mad < 3.0,
            "channel {c}: MAD {mad:.3} exceeds the round-454 landing ratchet 3.0"
        );
    }
}

#[test]
fn default_config_is_the_documented_survivor() {
    // The shipped default is the combination that closes both round-437
    // oracles under the per-channel layout (see PermStreamConfig docs).
    let d = PermStreamConfig::default();
    assert_eq!(d.prev_context, PermPrevContext::GetContextOfValue);
    assert_eq!(d.num_dists, 8);
}
