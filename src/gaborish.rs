//! Gabor-like restoration filter — ISO/IEC FDIS 18181-1:2021 Annex J.2
//! ("Gabor-like transform", page 85).
//!
//! ## Scope (round 141)
//!
//! Round 141 lands the **per-channel 3×3 symmetric-convolution
//! restoration filter** specified by §J.2: given a per-channel plane
//! of samples (the output of the §I.2.5 LLF/HF reconstruction +
//! Annex G chroma-from-luma chain) and the per-channel weights
//! `gab_C_weight1` / `gab_C_weight2` carried by
//! [`crate::frame_header::RestorationFilter`], this module produces
//! a new plane in which every sample is the convolution
//!
//! ```text
//!    [ w2  w1  w2 ]
//!    [ w1   1  w1 ]   normalised so all nine entries sum to 1
//!    [ w2  w1  w2 ]
//! ```
//!
//! applied with §6.5 `Mirror1D` boundary handling on out-of-image
//! references.
//!
//! This is a **pure-math primitive** in exactly the same shape as
//! the round-89 [`crate::dct_quant_weights`], round-95
//! [`crate::hf_dequant`], round-121 [`crate::llf_from_lf`], and
//! round-138 [`crate::chroma_from_luma`] steps already landed: a
//! self-contained function that takes already-decoded structures
//! and float planes, performs the bit-arithmetic step the spec
//! prescribes, and returns float planes. **No bit reading. No
//! interaction with the rest of the pipeline.** A future round
//! wiring §J.2 into the per-frame pipeline can drop these helpers
//! in without re-deriving the kernel or the mirror semantics.
//!
//! ## Spec listing (FDIS page 85 — Annex J.2, normative)
//!
//! > In this subclause, C denotes x, y, or b according to the
//! > channel currently being processed. The decoder applies a
//! > convolution to each channel of the entire image with the
//! > following symmetric 3×3 kernel: the unnormalized weight for
//! > the center is 1, its four neighbours (top, bottom, left,
//! > right) are `restoration_filter.gab_C_weight1` and the four
//! > corners (top-left, top-right, bottom-left, bottom-right) are
//! > `restoration_filter.gab_C_weight2`. These weights are rescaled
//! > uniformly before convolution, such that the nine kernel
//! > weights sum to 1.
//! >
//! > When the convolution references input pixels with coordinates
//! > `cx`, `cy` outside the bounds of the original image, the
//! > decoder instead accesses the pixel at coordinates
//! > `Mirror(cx, cy)` (6.5).
//!
//! And §6.5 (Listing 6.1):
//!
//! ```text
//! Mirror1D(coord, size) {
//!   if (coord < 0) return Mirror1D(-coord - 1, size);
//!   else if (coord >= size) return Mirror1D(2 × size - 1 - coord, size);
//!   else return coord;
//! }
//! ```
//!
//! ## Implementation notes
//!
//! * **Plane representation.** Each channel is a flat
//!   `&[f32]` of length `width * height` in row-major order, matching
//!   the convention used by [`crate::chroma_from_luma`]'s
//!   `apply_lf_plane_inplace` / `apply_hf_plane_inplace`.
//! * **Out-of-place.** The convolution is necessarily out-of-place
//!   (overwriting a sample before its neighbours are read would
//!   corrupt the kernel). [`apply_channel`] takes an immutable
//!   `input` plane and writes the filtered samples into a separate
//!   pre-allocated `output` plane; for callers that want in-place
//!   semantics, [`apply_channel_in_place`] swaps the input into a
//!   fresh buffer first.
//! * **Mirror with non-trivial recursion.** Listing 6.1 is recursive
//!   but bounded: a single `Mirror1D` call recurses at most ~`coord
//!   / size` times. For Gaborish we always pass `coord ∈ {-1, x,
//!   width}` with `x ∈ 0..width`, so the recursion depth is 0 or
//!   1 (`-1 → 0` and `width → width-1`). We implement it
//!   iteratively to keep stack pressure trivial; [`mirror1d`] is
//!   exposed for the test suite and for any future caller that
//!   needs the §6.5 semantics directly.
//! * **One-pixel-wide / -tall images.** `width == 1`, `height == 1`,
//!   and even `width == 0` are accepted: every kernel reference
//!   falls back to the single sample (or is unreachable for `0 ×
//!   N`). The defensive-bounds early-return short-circuits the
//!   zero-area cases.
//! * **Weight choice.** The default weights from §Table C.9 are
//!   `w1 = 0.115169525`, `w2 = 0.061248592` (all three channels).
//!   These give a normalization sum
//!   `1.0 + 4·w1 + 4·w2 ≈ 1.705672468` and a centre tap of
//!   `≈ 0.5862780`. The kernel is computed each call (cheap — 9
//!   floats); callers that need to apply the same kernel to many
//!   planes can use [`gab_kernel`] to materialise the normalized
//!   weights once.
//!
//! ## What this module does NOT do
//!
//! * It does not drive the per-channel loop from a real frame —
//!   the per-frame wiring is a follow-up round's responsibility.
//! * It does not implement the edge-preserving filter (§J.3); that
//!   is a separate primitive (round-142 or later).
//! * It does not check the [`crate::frame_header::RestorationFilter`]
//!   `gab` boolean. The caller is responsible for the skip; this
//!   module unconditionally applies the convolution on the inputs
//!   it is given.
//! * It does not adjust for chroma subsampling. Per §J.2 the
//!   convolution is "to each channel of the entire image" — the
//!   caller passes one plane per channel at the channel's native
//!   resolution.

use oxideav_core::{Error, Result};

/// FDIS Listing 6.1 `Mirror1D(coord, size)`. Returns the in-bounds
/// `usize` index that an out-of-bounds reference resolves to.
///
/// Mathematically `Mirror1D` is defined recursively but every
/// realistic call to Gaborish passes `coord ∈ {-1, x, size}` with
/// `x ∈ 0..size`, so the recursion bottoms out in at most one
/// reflection. We implement it iteratively because it is trivial
/// and avoids any concern about pathological caller-supplied
/// coordinates triggering deep recursion.
///
/// `size == 0` is rejected — the spec does not specify mirroring
/// for an empty axis (and the recursive listing would loop forever).
pub fn mirror1d(mut coord: i64, size: usize) -> Result<usize> {
    if size == 0 {
        return Err(Error::InvalidData(
            "JXL Gaborish: Mirror1D called with size == 0".into(),
        ));
    }
    let size_i = size as i64;
    // Bounded iteration: each step either accepts the value or
    // reflects it once. We cap at a high iteration count just to
    // ensure termination in the face of pathological inputs (the
    // listing is mathematically guaranteed to converge for any
    // finite integer input, but a defensive cap stops us from ever
    // hanging the decoder).
    for _ in 0..64 {
        if coord < 0 {
            coord = -coord - 1;
        } else if coord >= size_i {
            coord = 2 * size_i - 1 - coord;
        } else {
            return Ok(coord as usize);
        }
    }
    // Unreachable in practice — the listing converges in O(coord /
    // size) steps and we bound by 64 reflections, which would
    // require |coord| > 64·size to exhaust.
    Err(Error::InvalidData(format!(
        "JXL Gaborish: Mirror1D did not converge for coord with size {size}"
    )))
}

/// Materialise the normalized 3×3 Gaborish kernel from the
/// unnormalized `(weight1, weight2)` pair carried by
/// [`crate::frame_header::RestorationFilter`]. Per §J.2 the centre
/// tap is `1`, the four edge taps are `weight1`, the four corner
/// taps are `weight2`, and all nine entries are rescaled uniformly
/// so they sum to 1.
///
/// Returned in row-major order:
///
/// ```text
///   [ k[0]=w2  k[1]=w1  k[2]=w2 ]
///   [ k[3]=w1  k[4]=1   k[5]=w1 ]
///   [ k[6]=w2  k[7]=w1  k[8]=w2 ]
/// ```
///
/// `Err(InvalidData)` if the unnormalized sum is non-finite or zero
/// (would imply caller-supplied weights of `NaN` / `inf` or `1 +
/// 4·w1 + 4·w2 == 0`, neither of which the spec produces — defaults
/// are `(0.115169525, 0.061248592)` with sum `≈ 1.7056725`).
pub fn gab_kernel(weight1: f32, weight2: f32) -> Result<[f32; 9]> {
    let unnormalized_sum = 1.0_f32 + 4.0 * weight1 + 4.0 * weight2;
    if !unnormalized_sum.is_finite() || unnormalized_sum == 0.0 {
        return Err(Error::InvalidData(format!(
            "JXL Gaborish: kernel sum ({unnormalized_sum}) not in (0, +inf)"
        )));
    }
    let inv = 1.0_f32 / unnormalized_sum;
    let w1 = weight1 * inv;
    let w2 = weight2 * inv;
    let cc = 1.0_f32 * inv;
    Ok([w2, w1, w2, w1, cc, w1, w2, w1, w2])
}

/// Fetch one sample from a row-major `width × height` plane at
/// integer coordinates `(x, y)`, applying §6.5 Mirror1D on both
/// axes for out-of-bounds references. Returns
/// `Err(InvalidData)` only if the underlying plane buffer is the
/// wrong length, the dimensions overflow `i64`, or [`mirror1d`]
/// fails (zero-size axis).
#[inline]
pub fn sample_mirror(plane: &[f32], width: usize, height: usize, x: i64, y: i64) -> Result<f32> {
    if plane.len() != width * height {
        return Err(Error::InvalidData(format!(
            "JXL Gaborish: sample_mirror plane length {} != {}*{} = {}",
            plane.len(),
            width,
            height,
            width * height
        )));
    }
    let mx = mirror1d(x, width)?;
    let my = mirror1d(y, height)?;
    Ok(plane[my * width + mx])
}

/// Apply the §J.2 Gaborish 3×3 symmetric convolution to a single
/// channel plane, writing the filtered samples to a pre-allocated
/// `output` buffer of the same dimensions.
///
/// `input.len()` and `output.len()` must both equal `width *
/// height` (`Err(InvalidData)` otherwise). Out-of-bounds kernel
/// references are resolved by [`mirror1d`].
///
/// The convolution is necessarily out-of-place; see the module-level
/// notes. If `width == 0 || height == 0` the call is a no-op
/// (both buffers must still have matching length 0).
pub fn apply_channel(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    weight1: f32,
    weight2: f32,
) -> Result<()> {
    let expected = width * height;
    if input.len() != expected {
        return Err(Error::InvalidData(format!(
            "JXL Gaborish: input plane length {} != {}*{} = {}",
            input.len(),
            width,
            height,
            expected
        )));
    }
    if output.len() != expected {
        return Err(Error::InvalidData(format!(
            "JXL Gaborish: output plane length {} != {}*{} = {}",
            output.len(),
            width,
            height,
            expected
        )));
    }
    if expected == 0 {
        return Ok(());
    }
    let k = gab_kernel(weight1, weight2)?;
    // Fast path for the "all interior pixel" range — every sample
    // for which the entire 3×3 neighbourhood lies inside the image
    // bounds — avoids the mirror lookup per sample. Edge pixels
    // (first/last row, first/last column) fall back to the
    // mirroring path.
    for y in 0..height {
        for x in 0..width {
            let s = if x >= 1 && x + 1 < width && y >= 1 && y + 1 < height {
                // Interior pixel — direct row-major fetch.
                let row_above = (y - 1) * width;
                let row = y * width;
                let row_below = (y + 1) * width;
                k[0] * input[row_above + (x - 1)]
                    + k[1] * input[row_above + x]
                    + k[2] * input[row_above + (x + 1)]
                    + k[3] * input[row + (x - 1)]
                    + k[4] * input[row + x]
                    + k[5] * input[row + (x + 1)]
                    + k[6] * input[row_below + (x - 1)]
                    + k[7] * input[row_below + x]
                    + k[8] * input[row_below + (x + 1)]
            } else {
                // Edge / corner pixel — at least one reference
                // falls outside the image. Walk the 3×3 with
                // Mirror1D on both axes.
                let xi = x as i64;
                let yi = y as i64;
                let mut acc = 0.0_f32;
                for (slot, (cy, cx)) in [
                    (yi - 1, xi - 1),
                    (yi - 1, xi),
                    (yi - 1, xi + 1),
                    (yi, xi - 1),
                    (yi, xi),
                    (yi, xi + 1),
                    (yi + 1, xi - 1),
                    (yi + 1, xi),
                    (yi + 1, xi + 1),
                ]
                .iter()
                .enumerate()
                {
                    let mx = mirror1d(*cx, width)?;
                    let my = mirror1d(*cy, height)?;
                    acc += k[slot] * input[my * width + mx];
                }
                acc
            };
            output[y * width + x] = s;
        }
    }
    Ok(())
}

/// In-place sibling of [`apply_channel`]. Allocates a single
/// `Vec<f32>` of length `width * height`, copies `plane` into it,
/// then calls [`apply_channel`] back into `plane`.
///
/// Cheap convenience for callers that don't already own a scratch
/// buffer; the per-channel cost is one extra `width * height` heap
/// allocation per call. Callers in a hot loop should prefer
/// [`apply_channel`] with a reused scratch buffer.
pub fn apply_channel_in_place(
    plane: &mut [f32],
    width: usize,
    height: usize,
    weight1: f32,
    weight2: f32,
) -> Result<()> {
    let scratch = plane.to_vec();
    apply_channel(&scratch, plane, width, height, weight1, weight2)
}

/// Apply the §J.2 Gaborish convolution to all three XYB-pipeline
/// channels in one call, using the per-channel
/// `gab_C_weight1` / `gab_C_weight2` weights from a
/// [`crate::frame_header::RestorationFilter`].
///
/// `x_plane`, `y_plane`, and `b_plane` are each `width * height`
/// f32 buffers in row-major order at the channel's native
/// resolution. All three are filtered in place (each via
/// [`apply_channel_in_place`]).
///
/// Per §J.2 the convolution is applied "to each channel of the
/// entire image" — there is no inter-channel coupling. The
/// per-channel `(weight1, weight2)` pairs come from
/// `rf.gab_x_weight1` / `gab_x_weight2`,
/// `rf.gab_y_weight1` / `gab_y_weight2`,
/// `rf.gab_b_weight1` / `gab_b_weight2`.
///
/// The caller is responsible for honouring `rf.gab` (skip the call
/// when `gab == false`).
pub fn apply_xyb_planes_in_place(
    x_plane: &mut [f32],
    y_plane: &mut [f32],
    b_plane: &mut [f32],
    width: usize,
    height: usize,
    rf: &crate::frame_header::RestorationFilter,
) -> Result<()> {
    apply_channel_in_place(x_plane, width, height, rf.gab_x_weight1, rf.gab_x_weight2)?;
    apply_channel_in_place(y_plane, width, height, rf.gab_y_weight1, rf.gab_y_weight2)?;
    apply_channel_in_place(b_plane, width, height, rf.gab_b_weight1, rf.gab_b_weight2)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- §6.5 Mirror1D ----

    #[test]
    fn mirror1d_in_bounds_is_identity() {
        for size in 1..16 {
            for coord in 0..size as i64 {
                assert_eq!(mirror1d(coord, size).unwrap(), coord as usize);
            }
        }
    }

    /// `coord == -1` reflects to `0` per `Mirror1D(-coord - 1, size)
    /// = Mirror1D(0, size) = 0` for any `size >= 1`.
    #[test]
    fn mirror1d_minus_one_reflects_to_zero() {
        for size in 1..16 {
            assert_eq!(mirror1d(-1, size).unwrap(), 0);
        }
    }

    /// `coord == size` reflects to `size - 1` per
    /// `Mirror1D(2·size - 1 - size, size) = Mirror1D(size - 1, size)
    /// = size - 1`.
    #[test]
    fn mirror1d_size_reflects_to_last() {
        for size in 1..16_usize {
            assert_eq!(mirror1d(size as i64, size).unwrap(), size - 1);
        }
    }

    /// `coord == -2` for `size >= 2` reflects to
    /// `Mirror1D(1, size) = 1`.
    #[test]
    fn mirror1d_minus_two_reflects_to_one() {
        for size in 2..16 {
            assert_eq!(mirror1d(-2, size).unwrap(), 1);
        }
    }

    /// Single-row / single-column images collapse every mirror
    /// reference back to `0`.
    #[test]
    fn mirror1d_size_one_collapses_all() {
        for coord in -8..8 {
            assert_eq!(mirror1d(coord, 1).unwrap(), 0);
        }
    }

    /// `size == 0` is rejected per the module-level note.
    #[test]
    fn mirror1d_size_zero_is_error() {
        assert!(mirror1d(0, 0).is_err());
    }

    // ---- §J.2 kernel materialisation ----

    /// Default weights produce a kernel whose 9 entries sum to 1.0
    /// (within f32 round-off) per the spec's "rescaled uniformly
    /// before convolution, such that the nine kernel weights sum to
    /// 1" clause.
    #[test]
    fn gab_kernel_default_weights_sum_to_one() {
        let k = gab_kernel(0.115_169_525, 0.061_248_592).unwrap();
        let s: f32 = k.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-6,
            "default-weight kernel sum {s} differs from 1.0"
        );
    }

    /// At default weights, the centre tap equals `1 / (1 + 4·w1 +
    /// 4·w2)`.
    #[test]
    fn gab_kernel_default_centre_tap() {
        let w1 = 0.115_169_525_f32;
        let w2 = 0.061_248_592_f32;
        let k = gab_kernel(w1, w2).unwrap();
        let expected_center = 1.0 / (1.0 + 4.0 * w1 + 4.0 * w2);
        assert!((k[4] - expected_center).abs() < 1e-7);
    }

    /// `weight1 == weight2 == 0` is the identity kernel: every
    /// neighbour weight is 0 and the centre tap is 1.
    #[test]
    fn gab_kernel_zero_weights_is_identity() {
        let k = gab_kernel(0.0, 0.0).unwrap();
        assert_eq!(k, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    }

    /// Symmetry: the four edge taps are equal and the four corner
    /// taps are equal (regardless of weight choice).
    #[test]
    fn gab_kernel_is_symmetric() {
        let k = gab_kernel(0.115_169_525, 0.061_248_592).unwrap();
        // Edges: k[1] (top), k[3] (left), k[5] (right), k[7]
        // (bottom).
        assert_eq!(k[1], k[3]);
        assert_eq!(k[3], k[5]);
        assert_eq!(k[5], k[7]);
        // Corners: k[0] k[2] k[6] k[8].
        assert_eq!(k[0], k[2]);
        assert_eq!(k[2], k[6]);
        assert_eq!(k[6], k[8]);
    }

    /// Non-finite or zero `(1 + 4·w1 + 4·w2)` is rejected. Pick
    /// `w1 = w2 = -1/8` so the unnormalized sum is exactly 0.
    #[test]
    fn gab_kernel_zero_sum_is_error() {
        assert!(gab_kernel(-0.125, -0.125).is_err());
    }

    /// `NaN` weights also reject (the resulting sum is `NaN`,
    /// which is not finite).
    #[test]
    fn gab_kernel_nan_weights_is_error() {
        assert!(gab_kernel(f32::NAN, 0.0).is_err());
    }

    // ---- §J.2 convolution ----

    /// Constant plane → constant plane (any unit-sum kernel preserves
    /// a constant). Pin this for a 7×5 plane at the default weights.
    #[test]
    fn apply_channel_constant_is_invariant() {
        let w = 7_usize;
        let h = 5_usize;
        let input = vec![3.5_f32; w * h];
        let mut output = vec![0.0_f32; w * h];
        apply_channel(&input, &mut output, w, h, 0.115_169_525, 0.061_248_592).unwrap();
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - 3.5).abs() < 1e-5,
                "sample {i} = {v} drifted from constant 3.5"
            );
        }
    }

    /// Identity kernel (zero weights) is the identity transform.
    #[test]
    fn apply_channel_identity_kernel_is_identity() {
        let input: Vec<f32> = (0..15).map(|v| v as f32).collect();
        let mut output = vec![0.0_f32; 15];
        apply_channel(&input, &mut output, 5, 3, 0.0, 0.0).unwrap();
        assert_eq!(output, input);
    }

    /// Mirror semantics on a 1×1 plane: every kernel reference
    /// resolves to the single sample, so the convolution returns
    /// that sample (sum of normalised kernel × sample = sample).
    #[test]
    fn apply_channel_one_by_one_is_pass_through() {
        let input = vec![7.5_f32];
        let mut output = vec![0.0_f32; 1];
        apply_channel(&input, &mut output, 1, 1, 0.115_169_525, 0.061_248_592).unwrap();
        assert!((output[0] - 7.5).abs() < 1e-5);
    }

    /// A single non-zero sample at the centre of a 3×3 plane,
    /// filtered with the default-weight kernel, distributes by
    /// exactly the kernel weights to its 8 neighbours and keeps the
    /// centre tap on itself.
    #[test]
    fn apply_channel_impulse_3x3_distributes_kernel() {
        let mut input = vec![0.0_f32; 9];
        input[4] = 1.0; // centre impulse
        let mut output = vec![0.0_f32; 9];
        let w1 = 0.115_169_525_f32;
        let w2 = 0.061_248_592_f32;
        apply_channel(&input, &mut output, 3, 3, w1, w2).unwrap();
        let k = gab_kernel(w1, w2).unwrap();

        // For the centre output, all 9 references hit the centre
        // input via Mirror; for edge outputs only some references
        // hit it. But the property that *every output is a linear
        // combination of the impulse with kernel weights* still
        // holds: the centre output equals the centre tap and the
        // four edge outputs each equal an edge tap (a single
        // mirror reference into the centre).
        //
        // Layout of 3×3 plane (indices 0..9 row-major):
        //   0 1 2
        //   3 4 5
        //   6 7 8
        //
        // For output[4] (centre): centre input is the centre
        // reference k[4].
        assert!(
            (output[4] - k[4]).abs() < 1e-7,
            "centre output {} != k[4] {}",
            output[4],
            k[4]
        );
        // For output[1] (top edge), the centre input is the
        // bottom-centre reference (kernel slot k[7]); the top
        // reference mirrors to the top-edge sample (which is 0),
        // etc. So output[1] should equal k[7].
        assert!(
            (output[1] - k[7]).abs() < 1e-7,
            "top-edge output {} != k[7] {}",
            output[1],
            k[7]
        );
        // For output[3] (left edge), the centre input is the
        // right-centre reference (k[5]).
        assert!((output[3] - k[5]).abs() < 1e-7);
        // For output[5] (right edge), the centre input is the
        // left-centre reference (k[3]).
        assert!((output[5] - k[3]).abs() < 1e-7);
        // For output[7] (bottom edge), the centre input is the
        // top-centre reference (k[1]).
        assert!((output[7] - k[1]).abs() < 1e-7);
    }

    /// `apply_channel_in_place` produces the same output as
    /// `apply_channel` would on a fresh buffer.
    #[test]
    fn apply_channel_in_place_matches_out_of_place() {
        let input: Vec<f32> = (0..6 * 4).map(|v| (v as f32) * 0.1).collect();
        let mut a = input.clone();
        apply_channel_in_place(&mut a, 6, 4, 0.115_169_525, 0.061_248_592).unwrap();

        let mut b = vec![0.0_f32; 24];
        apply_channel(&input, &mut b, 6, 4, 0.115_169_525, 0.061_248_592).unwrap();

        for i in 0..24 {
            assert!(
                (a[i] - b[i]).abs() < 1e-7,
                "sample {i}: in-place={} out-of-place={}",
                a[i],
                b[i]
            );
        }
    }

    /// Wrong-length input / output buffers reject defensively.
    #[test]
    fn apply_channel_wrong_input_length_is_error() {
        let input = vec![0.0_f32; 10];
        let mut output = vec![0.0_f32; 12];
        let r = apply_channel(&input, &mut output, 4, 3, 0.0, 0.0);
        assert!(r.is_err());
    }

    #[test]
    fn apply_channel_wrong_output_length_is_error() {
        let input = vec![0.0_f32; 12];
        let mut output = vec![0.0_f32; 10];
        let r = apply_channel(&input, &mut output, 4, 3, 0.0, 0.0);
        assert!(r.is_err());
    }

    /// Zero-area planes are accepted as a no-op.
    #[test]
    fn apply_channel_zero_area_is_noop() {
        let input: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        apply_channel(&input, &mut output, 0, 0, 0.0, 0.0).unwrap();
        apply_channel(&input, &mut output, 5, 0, 0.0, 0.0).unwrap();
        apply_channel(&input, &mut output, 0, 5, 0.0, 0.0).unwrap();
    }

    /// `apply_xyb_planes_in_place` delegates per-channel to
    /// `apply_channel_in_place` with the right weight pair for each
    /// channel.
    #[test]
    fn apply_xyb_planes_in_place_uses_per_channel_weights() {
        use crate::frame_header::RestorationFilter;
        // Pick distinct per-channel weights so we can detect a
        // channel-mix-up in the delegating call.
        let rf = RestorationFilter {
            gab_x_weight1: 0.10,
            gab_x_weight2: 0.05,
            gab_y_weight1: 0.20,
            gab_y_weight2: 0.10,
            gab_b_weight1: 0.30,
            gab_b_weight2: 0.15,
            ..RestorationFilter::default()
        };

        let w = 4_usize;
        let h = 4_usize;
        let plane: Vec<f32> = (0..w * h).map(|v| v as f32).collect();
        let mut x = plane.clone();
        let mut y = plane.clone();
        let mut b = plane.clone();
        apply_xyb_planes_in_place(&mut x, &mut y, &mut b, w, h, &rf).unwrap();

        // Cross-check against per-channel direct calls.
        let mut x_ref = plane.clone();
        let mut y_ref = plane.clone();
        let mut b_ref = plane.clone();
        apply_channel_in_place(&mut x_ref, w, h, 0.10, 0.05).unwrap();
        apply_channel_in_place(&mut y_ref, w, h, 0.20, 0.10).unwrap();
        apply_channel_in_place(&mut b_ref, w, h, 0.30, 0.15).unwrap();

        assert_eq!(x, x_ref);
        assert_eq!(y, y_ref);
        assert_eq!(b, b_ref);
    }

    /// `sample_mirror` short-circuits to in-bounds direct indexing
    /// for in-bounds coordinates and uses §6.5 Mirror1D for
    /// out-of-bounds coordinates.
    #[test]
    fn sample_mirror_in_and_out_of_bounds() {
        let plane: Vec<f32> = (0..3 * 2).map(|v| v as f32).collect();
        // Layout 3 wide × 2 tall:
        //   0 1 2
        //   3 4 5
        assert_eq!(sample_mirror(&plane, 3, 2, 0, 0).unwrap(), 0.0);
        assert_eq!(sample_mirror(&plane, 3, 2, 2, 1).unwrap(), 5.0);
        // (-1, 0) mirrors to (0, 0) = 0.
        assert_eq!(sample_mirror(&plane, 3, 2, -1, 0).unwrap(), 0.0);
        // (3, 0) mirrors to (2, 0) = 2.
        assert_eq!(sample_mirror(&plane, 3, 2, 3, 0).unwrap(), 2.0);
        // (0, -1) mirrors to (0, 0) = 0.
        assert_eq!(sample_mirror(&plane, 3, 2, 0, -1).unwrap(), 0.0);
        // (0, 2) mirrors to (0, 1) = 3.
        assert_eq!(sample_mirror(&plane, 3, 2, 0, 2).unwrap(), 3.0);
        // (-1, -1) mirrors to (0, 0) = 0.
        assert_eq!(sample_mirror(&plane, 3, 2, -1, -1).unwrap(), 0.0);
    }

    #[test]
    fn sample_mirror_wrong_plane_length_is_error() {
        let plane = vec![0.0_f32; 5];
        // 3 * 2 = 6 ≠ 5 → reject.
        assert!(sample_mirror(&plane, 3, 2, 0, 0).is_err());
    }
}
