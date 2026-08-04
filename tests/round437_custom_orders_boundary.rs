//! Round 437 — the residual `used_orders != 0` refusal boundary:
//! multi-preset / multi-pass §C.7 streams.
//!
//! The round-437 forensics resolved the §C.3.2/§C.7.1 chain for
//! single-preset single-pass streams (see
//! `round437_custom_orders_decode`): the per-entry context question of
//! fdis-errata.md Part 8.3 was walked as prescribed (no grid point
//! closes under the printed one-permutation-per-bit layout), and the
//! actual erratum turned out to be the LAYOUT — Listing C.12 carries
//! THREE `DecodePermutation()` per set bit (one per colour channel in
//! §C.8.3 Y, X, B sequence), after which the shipped
//! `PermStreamConfig` default closes both oracles (the patches-fixture
//! trace bit-boundaries and the D.3.3 ANS final state).
//!
//! What still refuses — loudly, at a precise position — is the
//! multi-preset (`num_hf_presets > 1`) and/or multi-pass §C.7 slice
//! walk exercised by the staged `progressive-ac-multipass` fixture
//! (3 passes × 2 presets, `used_orders = 0x5F` / `0xE40`-class
//! bundles): after preset 0's per-channel bundles the next preset's
//! fields still misparse, so either the per-preset repetition or the
//! per-pass slice layout hides one more wire divergence. This test
//! pins that boundary:
//!
//! 1. The frame-level multi-pass gate stays GONE (the decode reaches
//!    the §C.7 walk rather than refusing at `num_passes > 1`).
//! 2. The refusal is loud — never a silent misparse — under the
//!    shipped default AND under every Part 8.3 grid point (no
//!    candidate silently "succeeds" with garbage output).
use oxideav_jpegxl::coeff_order::{
    set_perm_stream_config_override, PermPrevContext, PermStreamConfig,
};

const FIXTURE: &[u8] = include_bytes!("fixtures/progressive_ac_multipass.jxl");

#[test]
fn multipass_fixture_reaches_the_c71_boundary_and_refuses_loudly() {
    let err = oxideav_jpegxl::decode_one_frame(FIXTURE, None)
        .expect_err("multi-preset multi-pass used_orders fixture must refuse until the §C.7 preset/pass slice layout is resolved");
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
                 multi-preset multi-pass fixture — if the remaining §C.7 slice \
                 layout has been resolved, rewrite this test into a pixel ratchet"
            );
        }
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
