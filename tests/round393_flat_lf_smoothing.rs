//! Round 393 — the `flat-content-lf-smoothing` fixture (docs commit
//! 8554081, filed against FDIS erratum candidate 4) arbitrated TWO
//! FDIS readings externally:
//!
//! 1. **§F.3 HfMul role (new erratum candidate).** The fixture's 16
//!    uniform DCT64×64 varblocks carry HfMul = 13 and a near-empty HF
//!    band (|quant| ≤ 8, all low-frequency). The literal F.3 reading
//!    ("the resulting quant is then **multiplied** by ... the value of
//!    HfMul") blew the decoded HF ~169× past the reference — ±30-code
//!    low-frequency garbage in exactly the varblocks holding |quant|
//!    5..8, per-channel sRGB MAD 2.67/2.39/2.42 — while dividing lands
//!    MAD ≈ 0.20 on all three channels. HfMul is the per-varblock
//!    quantisation-PRECISION multiplier (§C.8.3 `qf`), so the decoder
//!    divides. Confirmed on every staged VarDCT fixture (d1
//!    3.42/1.99/2.10 → 0.66/0.47/0.61 — closing the long-pinned
//!    round-385 "d1 HF accuracy tail" — d3 and large-d2 also improve).
//!    See `hf_dequant::dequant_hf_coefficient`.
//!
//! 2. **§F.2 adaptive-LF-smoothing ramp (erratum candidate 4,
//!    RESOLVED).** The FDIS prose says the smoothed value is
//!    `(s − wa) × max(0, 3 − 4 × gap) + wa`; round 385 proposed the
//!    sign-flipped clamped ramp `clamp(4 × gap − 3, 0, 1)` but photo
//!    content could not distinguish them (all gaps > 1 → both are
//!    near-no-ops). This fixture is nearly flat, so 674/900 interior LF
//!    samples sit at the `gap = 0.5` floor where the two readings take
//!    OPPOSITE values (literal keeps the sample, corrected replaces it
//!    with the weighted average). With the HfMul fix in place the
//!    corrected ramp beats the literal reading on every channel
//!    (MAD 0.2045/0.2044/0.2036 vs 0.2290/0.2287/0.2276) and matches
//!    ~740 more pixels exactly. The corrected ramp is the conformant
//!    reading.
//!
//! Clean-room: `input.jxl` / `expected.png` are black-box validator
//! artefacts staged under `docs/image/jpegxl/fixtures/`; behaviour is
//! derived from the ISO/IEC 18181-1 FDIS + the staged errata notes. No
//! external implementation source is consulted. The per-sample LF trace
//! and per-varblock HF-coefficient captures exercised here are the
//! crate's own instrumentation (the #168 notes explicitly leave them to
//! the clean-room crate).

use std::io::Cursor;

use oxideav_jpegxl::lf_dequant::{
    set_lf_smooth_trace_armed, set_lf_smoothing_literal_ramp, LF_SMOOTH_TRACE,
};
use oxideav_jpegxl::{set_vardct_hf_coeff_capture_armed, VARDCT_HF_COEFF_CAPTURE};

const JXL: &[u8] = include_bytes!("fixtures/flat_content_lf_smoothing.jxl");
const REF_PNG: &[u8] = include_bytes!("fixtures/flat_content_lf_smoothing_expected.png");

/// Decode the committed reference PNG into interleaved RGB pixels.
fn ref_rgb() -> (u32, u32, Vec<[u8; 3]>) {
    let dec = png::Decoder::new(Cursor::new(REF_PNG));
    let mut reader = dec.read_info().expect("png read_info");
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap_or(0)];
    let info = reader.next_frame(&mut buf).expect("png next_frame");
    assert_eq!(info.bit_depth, png::BitDepth::Eight, "8-bit reference");
    let data = &buf[..info.buffer_size()];
    let ch = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => panic!("unexpected reference colour type {other:?}"),
    };
    let mut px = Vec::with_capacity((info.width * info.height) as usize);
    for c in data.chunks_exact(ch) {
        px.push([c[0], c[1], c[2]]);
    }
    (info.width, info.height, px)
}

/// Decode the fixture through the integrated VarDCT path and return
/// interleaved RGB pixels, with the §F.2 ramp selected by `literal`.
fn decode_rgb(literal: bool) -> (u32, u32, Vec<[u8; 3]>) {
    set_lf_smoothing_literal_ramp(literal);
    let r = oxideav_jpegxl::decode_vardct_frame_from_codestream(JXL, None);
    set_lf_smoothing_literal_ramp(false);
    let frame = r.expect("integrated VarDCT decode of flat-content fixture");
    assert!(
        frame.planes.len() >= 3,
        "expected >= 3 planes, got {}",
        frame.planes.len()
    );
    let w = frame.planes[0].stride as u32;
    let h = (frame.planes[0].data.len() / frame.planes[0].stride) as u32;
    let mut px = Vec::with_capacity((w * h) as usize);
    for y in 0..h as usize {
        for x in 0..w as usize {
            let r = frame.planes[0].data[y * frame.planes[0].stride + x];
            let g = frame.planes[1].data[y * frame.planes[1].stride + x];
            let b = frame.planes[2].data[y * frame.planes[2].stride + x];
            px.push([r, g, b]);
        }
    }
    (w, h, px)
}

/// Per-channel mean-absolute-difference between two same-size images.
fn mad(a: &[[u8; 3]], b: &[[u8; 3]]) -> [f64; 3] {
    assert_eq!(a.len(), b.len());
    let mut sum = [0u64; 3];
    for (pa, pb) in a.iter().zip(b) {
        for k in 0..3 {
            sum[k] += (pa[k] as i64 - pb[k] as i64).unsigned_abs();
        }
    }
    let n = a.len() as f64;
    [sum[0] as f64 / n, sum[1] as f64 / n, sum[2] as f64 / n]
}

/// Accuracy ratchet — the external pin of the §F.3 HfMul-divides
/// erratum. Under the pre-393 multiply reading this fixture decodes at
/// per-channel MAD 2.67/2.39/2.42 (±30-code garbage in the strong-HF
/// varblocks); with the division it lands at ≈ 0.2045/0.2044/0.2036.
#[test]
fn flat_content_decode_tracks_reference() {
    let (rw, rh, refpx) = ref_rgb();
    assert_eq!((rw, rh), (256, 256), "reference fixture is 256×256");
    let (w, h, ours) = decode_rgb(false);
    assert_eq!((w, h), (rw, rh));
    let m = mad(&ours, &refpx);
    eprintln!("flat-content MAD vs reference = {m:?}");
    for (k, &v) in m.iter().enumerate() {
        assert!(
            v < 0.35,
            "channel {k} MAD {v:.4} exceeds the round-393 flat-content ratchet \
             (0.35/255; measured ≈ 0.205). The §F.3 HfMul-multiply misreading \
             alone pushes this to ≈ 2.7."
        );
    }
}

/// §F.2 ramp arbitration (erratum candidate 4): decode under both
/// readings and compare each against the reference. The corrected ramp
/// `clamp(4·gap − 3, 0, 1)` must beat the literal `max(0, 3 − 4·gap)`
/// on every channel AND on the exact-match count over the pixels where
/// the two ramps disagree.
#[test]
fn f2_ramp_arbitration_corrected_ramp_matches_reference() {
    let (rw, rh, refpx) = ref_rgb();
    let (w1, h1, corrected) = decode_rgb(false);
    assert_eq!((w1, h1), (rw, rh));
    let (w2, h2, literal) = decode_rgb(true);
    assert_eq!((w2, h2), (rw, rh));

    // The two readings must actually diverge on this fixture — that is
    // its entire reason for existing. If they ever stop diverging the
    // fixture no longer arbitrates and this test must be revisited.
    assert_ne!(
        corrected, literal,
        "flat-content fixture must distinguish the two §F.2 ramps"
    );

    let mad_corrected = mad(&corrected, &refpx);
    let mad_literal = mad(&literal, &refpx);
    eprintln!("corrected ramp clamp(4g-3,0,1) MAD = {mad_corrected:?}");
    eprintln!("literal   ramp max(0,3-4g)    MAD = {mad_literal:?}");
    for k in 0..3 {
        assert!(
            mad_corrected[k] < mad_literal[k],
            "channel {k}: corrected ramp MAD {:.4} must beat literal ramp MAD {:.4}",
            mad_corrected[k],
            mad_literal[k]
        );
    }

    // Sharper criterion: over the pixels where the two ramps disagree,
    // the corrected ramp must match the reference exactly on more
    // pixels than the literal ramp does (measured ≈ +740).
    let mut exact_corrected = 0u64;
    let mut exact_literal = 0u64;
    let mut differing = 0u64;
    for i in 0..refpx.len() {
        if corrected[i] != literal[i] {
            differing += 1;
            if corrected[i] == refpx[i] {
                exact_corrected += 1;
            }
            if literal[i] == refpx[i] {
                exact_literal += 1;
            }
        }
    }
    eprintln!(
        "ramp-differing pixels: {differing}; exact-vs-ref corrected {exact_corrected} / \
         literal {exact_literal}"
    );
    assert!(differing > 500, "fixture must exercise the ramp broadly");
    assert!(
        exact_corrected > exact_literal,
        "on ramp-differing pixels the corrected ramp must match the reference \
         more often (corrected {exact_corrected} vs literal {exact_literal})"
    );
}

/// Per-sample LF-smoothing trace (the crate-side instrumentation the
/// fixture notes prescribe): the §F.2 smoothing must be ACTIVE on this
/// stream and the gap distribution must sit overwhelmingly at the
/// `gap = 0.5` floor — the design property that makes this fixture an
/// arbiter (the two candidate ramps take opposite values there).
#[test]
fn lf_smooth_trace_pins_gap_distribution() {
    LF_SMOOTH_TRACE.with(|s| *s.borrow_mut() = None);
    set_lf_smooth_trace_armed(true);
    let _ = decode_rgb(false);
    set_lf_smooth_trace_armed(false);
    let trace = LF_SMOOTH_TRACE
        .with(|s| s.borrow_mut().take())
        .expect("F.2 smoothing ran and the trace hook captured it");
    assert_eq!(
        (trace.width, trace.height),
        (32, 32),
        "256×256 frame → 32×32 LF grid"
    );
    let w = trace.width as usize;
    let h = trace.height as usize;
    let mut at_floor = 0u32;
    let mut interior = 0u32;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            interior += 1;
            if trace.gap[y * w + x] < 0.55 {
                at_floor += 1;
            }
        }
    }
    eprintln!("gap at 0.5 floor: {at_floor} / {interior}");
    assert!(
        at_floor * 3 > interior * 2,
        "flat content must put >2/3 of interior LF samples at the gap floor \
         (got {at_floor}/{interior})"
    );
    // Edge samples are never smoothed: factor stays at its 1.0 fill.
    assert_eq!(trace.factor[0], 1.0);
    // The trace planes are complete pre/post snapshots.
    for c in 0..3 {
        assert_eq!(trace.pre[c].len(), w * h);
        assert_eq!(trace.post[c].len(), w * h);
    }
}

/// Per-varblock HF-coefficient capture (the #168 ask-(a) crate-side
/// instrumentation): the fixture is a single group of 16 DCT64×64
/// varblocks whose HF band is sparse and small — pinned so the capture
/// hook and the entropy decode stay observable.
#[test]
fn hf_coeff_capture_pins_varblock_structure() {
    VARDCT_HF_COEFF_CAPTURE.with(|s| *s.borrow_mut() = None);
    set_vardct_hf_coeff_capture_armed(true);
    let _ = decode_rgb(false);
    set_vardct_hf_coeff_capture_armed(false);
    let cap = VARDCT_HF_COEFF_CAPTURE
        .with(|s| s.borrow_mut().take())
        .expect("HF coefficient capture populated");
    assert_eq!(cap.len(), 1, "single 256×256 group");
    let (gidx, stacks) = &cap[0];
    assert_eq!(*gidx, 0);
    assert_eq!(stacks.len(), 1, "single pass");
    let pass = &stacks[0];
    assert_eq!(pass.len(), 16, "16 DCT64×64 varblocks");
    for (vb, blocks, _) in pass {
        assert_eq!(
            vb.transform,
            oxideav_jpegxl::dct_select::TransformType::Dct64x64
        );
        assert_eq!(vb.hf_mul, 13, "uniform HfMul on this fixture");
        for (c, blk) in blocks.iter().enumerate() {
            assert_eq!(blk.coeffs.len(), 64 * 64);
            let maxa = blk.coeffs.iter().map(|v| v.abs()).max().unwrap_or(0);
            assert!(
                maxa <= 8,
                "channel {c} of vb({}, {}) must stay within the fixture's \
                 measured |quant| <= 8 envelope (got {maxa})",
                vb.x,
                vb.y
            );
        }
    }
}
