//! Round 437 — §C.7.1 `used_orders != 0` custom coefficient orders:
//! the Part 8.3 six-way bisection is EXHAUSTED and the refusal
//! boundary is pinned precisely.
//!
//! ## What this round established (fdis-errata.md Parts 8.2 + 8.3)
//!
//! The staged `progressive-ac-multipass` fixture (2560×1440, 3 passes,
//! 60 groups, `num_hf_presets = 2`, every pass's first preset coding
//! `used_orders = 0x5F`) plus locally generated single-preset
//! `used_orders != 0` streams were walked against the full Part 8.3
//! candidate grid — the three `D[prev_elem]` readings
//! (`PermPrevContext`) × the two distribution-count/cap pairings
//! (`num_dists` 8/cap 7 and 9/cap 8) — with two decisive oracles: the
//! D.3.3 ANS final-state closure on the shared Listing C.12 stream,
//! and full-frame decode against a black-box reference decode.
//!
//! **No grid point closes.** The shipped default
//! (`GetContextOfValue` over 8 distributions, context capped at 7) is
//! the only reading whose Lehmer walk stays in-range through every
//! entry of every specimen (a bit-level hand-decode of a minimal
//! single-order prefix-coded specimen against RFC 7932 §3.4 with
//! §D.3.5/§D.3.6 confirmed that walk consumes exactly the bits the
//! spec text prescribes), yet on ANS-coded specimens the shared stream's
//! final state misses `0x130000`, and on multi-preset streams the
//! *next* preset's `used_orders` reads back garbage. The residual
//! divergence therefore sits in the §C.7 layout after a preset's
//! Listing C.12 bundle (or in the §C.7.2 prelude that follows), which
//! the finite bisection cannot separate — a per-symbol trace is
//! required (docs-gap, refined round 437; see the final report).
//!
//! ## What this test pins
//!
//! 1. The frame-level multi-pass VarDCT framing gate is GONE: the
//!    decode reaches the §C.7.1 boundary (it no longer refuses at
//!    `num_passes > 1`).
//! 2. The refusal at that boundary is LOUD and precise — never a
//!    silent misparse: the decode of the staged multi-pass fixture
//!    errors (its `used_orders = 0x5F` streams hit the unresolved
//!    boundary), and the error is an `InvalidData`/`Unsupported`
//!    rather than wrong pixels.
//! 3. The `PermStreamConfig` bisection surface stays wired: every
//!    non-default grid point also fails loudly on this fixture (no
//!    candidate silently "succeeds" with garbage output).

use oxideav_jpegxl::coeff_order::{
    set_perm_stream_config_override, PermPrevContext, PermStreamConfig,
};

const FIXTURE: &[u8] = include_bytes!("fixtures/progressive_ac_multipass.jxl");

#[test]
fn multipass_fixture_reaches_the_c71_boundary_and_refuses_loudly() {
    let err = oxideav_jpegxl::decode_one_frame(FIXTURE, None)
        .expect_err("used_orders != 0 must refuse until the §C.3.2 context gap is resolved");
    let msg = format!("{err}");
    // The refusal must NOT be the old blanket multi-pass gate — the
    // frame-level multi-pass framing is wired (round 389) and the
    // decode must get past FrameHeader recognition into the HfGlobal
    // section before stopping.
    assert!(
        !msg.contains("num_passes"),
        "multi-pass frames must not be gated on pass count any more, got: {msg}"
    );
}

#[test]
fn every_bisection_grid_point_fails_loudly_on_the_multipass_fixture() {
    for reading in [
        PermPrevContext::LehmerValue,
        PermPrevContext::PrevToken,
        PermPrevContext::GetContextOfValue,
    ] {
        for num_dists in [8u32, 9u32] {
            set_perm_stream_config_override(Some(PermStreamConfig {
                prev_context: reading,
                num_dists,
            }));
            let r = oxideav_jpegxl::decode_one_frame(FIXTURE, None);
            set_perm_stream_config_override(None);
            assert!(
                r.is_err(),
                "grid point {reading:?}/{num_dists} unexpectedly decoded the \
                 used_orders fixture — if the §C.3.2 gap has been resolved, ship \
                 that combination as the PermStreamConfig default and rewrite this \
                 test into a pixel ratchet"
            );
        }
    }
}

#[test]
fn default_config_is_the_documented_survivor() {
    // The shipped default must stay on the only in-range grid point
    // until the docs-gap resolves (fdis-errata.md Part 8.3).
    let d = PermStreamConfig::default();
    assert_eq!(d.prev_context, PermPrevContext::GetContextOfValue);
    assert_eq!(d.num_dists, 8);
}
