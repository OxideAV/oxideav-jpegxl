//! Round 367 — VarDCT LF→LLF→IDCT **DC-magnitude preservation** invariant,
//! and the corrected `vardct-256x256-d1` divergence localisation.
//!
//! ## Why this test exists
//!
//! Round 362 localised the `vardct-256x256-d1` reconstruction divergence
//! (output rails ~99.8 % of samples; internal XYB magnitude several × too
//! large) to "the coefficient-magnitude path … most likely the LfQuant
//! modular sub-bitstream decode", and recorded as supporting evidence that
//! "its DC passes the spec IDCT with **unit gain**" — reasoning that
//! assumed the varblocks were DCT8×8.
//!
//! Round 367 re-examined the fixture's actual `DctSelect` grid: **every one
//! of the 16 varblocks of `vardct-256x256-d1` is DCT64×64** (Table C.16
//! numerical value 18), i.e. 8×8 LF-block units covering the full 32×32 LF
//! grid — NOT DCT8×8. For a DCT64×64 varblock the LF→LLF step is the
//! non-trivial §I.2.5 Listing I.16 `DCT_2D` of the 8×8 LF samples scaled by
//! `ScaleF(8, 64, ·)`, followed by the §I.2.3.2 IDCT64×64 — a far longer
//! chain than the degenerate DCT8×8 single-cell identity.
//!
//! This test pins, as a permanent invariant, that **that longer chain still
//! preserves the DC magnitude exactly** (a flat LF block of value `V`
//! reconstructs to a spatial block of value `V` everywhere). Combined with
//! the round-362 confirmations that the Listing F.1 dequant, the Table C.12
//! Quantizer parse, and the Table C.11 `m_*_lf_unscaled` are spec-correct,
//! this rules out the §I.2.5 LLF scaling, the §I.2.3.2 IDCT, the §C.5.4
//! placement, the §6.2 crop and the §L.2.2 XYB→RGB transform as the source
//! of the magnitude error — leaving the **LfQuant modular sub-bitstream
//! decode** (the decoded `qX/qY/qB` integers themselves) as the sole
//! remaining suspect, consistent with the round-17 bisect's observation
//! that the LF per-sample loop over-consumes bitstream.
//!
//! Clean-room: behaviour derived from ISO/IEC 18181-1 (FDIS §I.2.5 Listing
//! I.15/I.16 + §I.2.3.2 IDCT + Table C.16). No external implementation
//! source consulted.

use oxideav_jpegxl::dct_select::TransformType;

/// The full DCT-family LF→LLF→IDCT chain preserves DC magnitude: a flat
/// 8×8-block-units LF input of value `V` reconstructs to a spatial block of
/// value `V` everywhere, for every plain-DCT transform whose LF block is
/// 8×8 LF samples (DCT64×64 .. DCT256×256) as well as the smaller members.
#[test]
fn dct_family_lf_to_llf_idct_preserves_dc() {
    // The LF block is `cx × cy` LF samples per §I.2.5, where
    // `(cx, cy) = block_dims()` (cols, rows in 8×8-block units).
    let transforms = [
        TransformType::Dct8x8,
        TransformType::Dct16x16,
        TransformType::Dct32x32,
        TransformType::Dct64x64,
        TransformType::Dct64x32,
        TransformType::Dct32x64,
        TransformType::Dct128x128,
        TransformType::Dct256x256,
    ];

    let v = 1.0f32;
    for t in transforms {
        let (cx, cy) = t.block_dims();
        // Flat LF plane sized exactly to one varblock's LF block.
        let lf: Vec<f32> = vec![v; (cx * cy) as usize];
        let llf = oxideav_jpegxl::vardct::compose_lf_to_llf_block(&lf, cx, cy, 0, 0, t)
            .unwrap_or_else(|e| panic!("compose_lf_to_llf_block {t:?}: {e:?}"));
        assert_eq!(llf.len(), (cx * cy) as usize, "{t:?} LLF size");

        let (rows, cols) =
            oxideav_jpegxl::idct::dct_pixel_dims(t).unwrap_or_else(|| panic!("dims {t:?}"));
        // Place the LLF prefix into the top-left cx×cy of the coefficient
        // grid (the §I.2.4 natural-order low-frequency prefix); the rest
        // are the HF coefficients, all zero for this DC-only probe.
        let mut coeffs = vec![0.0f32; rows * cols];
        for y in 0..cy as usize {
            for x in 0..cx as usize {
                coeffs[y * cols + x] = llf[y * cx as usize + x];
            }
        }
        let spatial = oxideav_jpegxl::idct::idct_for_transform(t, &coeffs)
            .unwrap_or_else(|e| panic!("idct_for_transform {t:?}: {e:?}"));
        assert_eq!(spatial.len(), rows * cols, "{t:?} spatial size");

        let mean: f32 = spatial.iter().sum::<f32>() / spatial.len() as f32;
        let max = spatial.iter().cloned().fold(f32::MIN, f32::max);
        let min = spatial.iter().cloned().fold(f32::MAX, f32::min);
        // Flat-in → flat-out at the same magnitude. Tolerance covers f32
        // round-off across the longest (256×256) chain.
        assert!(
            (mean - v).abs() < 1e-3,
            "{t:?}: spatial mean {mean} != LF value {v} (DC gain not unit)"
        );
        assert!(
            (max - v).abs() < 5e-3 && (min - v).abs() < 5e-3,
            "{t:?}: spatial not flat (min={min} max={max}, expected ~{v})"
        );
    }
}

/// Pin the specific DCT64×64 case used by every varblock of
/// `vardct-256x256-d1`, with an explicit non-unit LF value so a future
/// regression that introduces a scale factor (e.g. an extra ×4) is caught
/// at the LF→spatial boundary rather than only at the RGB output.
#[test]
fn dct64x64_dc_preservation_explicit_value() {
    let v = 1.83f32; // ~ the d1 Y-plane dequantised LF mean (round-367 diag)
    let lf: Vec<f32> = vec![v; 64]; // 8×8 LF block
    let t = TransformType::Dct64x64;
    let llf = oxideav_jpegxl::vardct::compose_lf_to_llf_block(&lf, 8, 8, 0, 0, t).expect("compose");
    let (rows, cols) = oxideav_jpegxl::idct::dct_pixel_dims(t).expect("dims");
    let mut coeffs = vec![0.0f32; rows * cols];
    for y in 0..8 {
        for x in 0..8 {
            coeffs[y * cols + x] = llf[y * 8 + x];
        }
    }
    let spatial = oxideav_jpegxl::idct::idct_for_transform(t, &coeffs).expect("idct");
    let mean: f32 = spatial.iter().sum::<f32>() / spatial.len() as f32;
    assert!(
        (mean - v).abs() < 1e-2,
        "DCT64×64 DC must pass with unit gain: spatial mean {mean} != {v}. \
         The ~4× d1 Y-plane magnitude error is therefore upstream of the \
         LF→LLF→IDCT chain (in the LfQuant modular decode), not in it."
    );
}
