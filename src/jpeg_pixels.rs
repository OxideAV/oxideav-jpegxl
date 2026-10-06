//! Pixel reconstruction of JPEG-recompressed (`do_YCbCr` VarDCT) frames.
//!
//! A losslessly recompressed JPEG stores the JPEG's quantized DCT
//! coefficients in a VarDCT frame whose channels sit on the JPEG's
//! sampling lattices (`jpeg_upsampling`, F.2) and whose `RAW` dequant
//! matrices carry the JPEG quantization tables. The exact integer
//! coefficient decode of [`crate::jpeg_reconstruct`] — the one Annex A
//! bitstream reconstruction pins byte-exact — is therefore also the
//! right front end for pixels: this module takes those coefficients and
//! runs the JPEG sample pipeline on them, channel by channel on each
//! channel's own lattice:
//!
//! 1. dequantization (`coefficient × Q`) and the 8×8 inverse DCT of
//!    ISO/IEC 10918-1 A.3.3 (a separable float IDCT), level shift
//!    `+128`, round, clamp to `0 ..= 255` (10918-1 A.3.1 / F.2.1.5);
//! 2. the §6.2 crop of each channel's padded block grid to its own
//!    sample extent `ceil(width / h) × ceil(height / v)` (10918-1
//!    A.1.1);
//! 3. for RGB output, ISO/IEC 18181-1 J.2 "simple upsampling" of the
//!    subsampled channels (the `0.25 · A + 0.75 · B` / `0.75 · B + 0.25 · C`
//!    triangle filter with edge replication, width capped to the image)
//!    followed by the §L.3 inverse YCbCr (`R = Y + 1.402 Cr`, `G = Y −
//!    0.344136 Cb − 0.714136 Cr`, `B = Y + 1.772 Cb`, chroma centred on
//!    128 — the T.871 §7 constants).
//!
//! Step 1 clamps to 8-bit samples **before** the colour transform, as
//! every JPEG decoder does (and as the planar `YuvJ*` native layouts
//! require); the reference JPEG XL decoder keeps the float pipeline
//! unclamped until RGB, which is why it lands up to ±3 (smooth content)
//! or ±15 (noisy 4:4:4 content with out-of-range YCbCr samples) from
//! `djpeg`, while this path tracks `djpeg` to ±3 with the same triangle
//! upsampling. The 4:4:4 path and the subsampled paths share every
//! kernel; only the lattice differs.
//!
//! Frames that reach here: `encoding == kVarDCT && do_YCbCr`, single
//! pass, `RAW` dequant matrices, no extra channels — exactly the
//! transcode class [`crate::jpeg_reconstruct::decode_transcoded_frame`]
//! accepts. The §J restoration filters are not applied: a transcode
//! frame signals `gab = false, epf_iters = 0` (the JPEG has no such
//! filtering to reproduce) and the coefficient path refuses anything
//! else loudly.

use crate::error::{Error, Result};
use crate::image::{Plane, RawFrame};
use crate::jpeg_reconstruct::TranscodedCoefficients;

/// One decoded sample plane of a transcode frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SamplePlane {
    /// Samples per row.
    pub width: usize,
    /// Rows.
    pub height: usize,
    /// Row-major, stride == `width`.
    pub samples: Vec<u8>,
}

/// The decoded planes of a transcode frame in the JPEG sample domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TranscodedPlanes {
    /// Luma extent in pixels (the frame's logical size).
    pub width: u32,
    pub height: u32,
    /// Planes in JPEG component order: `[Y, Cb, Cr]`.
    pub planes: [SamplePlane; 3],
    /// `(hshift, vshift)` of the chroma planes relative to luma — the
    /// `jpeg_upsampling` lattice (0 or 1 per axis).
    pub chroma_shift: (u32, u32),
}

/// `C(u) · cos((2x + 1) u π / 16)` for the 8-point IDCT of ISO/IEC
/// 10918-1 A.3.3, indexed `[x][u]`.
fn idct_basis() -> [[f32; 8]; 8] {
    let mut t = [[0f32; 8]; 8];
    for (x, row) in t.iter_mut().enumerate() {
        for (u, v) in row.iter_mut().enumerate() {
            let cu = if u == 0 {
                std::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            let angle = ((2 * x + 1) as f64) * (u as f64) * std::f64::consts::PI / 16.0;
            *v = (cu * angle.cos()) as f32;
        }
    }
    t
}

/// Dequantize one 8×8 block (10918-1 raster order) and inverse-DCT it
/// into level-shifted 8-bit samples.
fn idct_block(coeffs: &[i32], quant: &[i32], basis: &[[f32; 8]; 8], out: &mut [u8; 64]) {
    // Dequantize into float.
    let mut f = [0f32; 64];
    for (k, v) in f.iter_mut().enumerate() {
        *v = (coeffs[k] as f32) * (quant[k] as f32);
    }
    // Rows: for each row v (vertical frequency), transform along u.
    let mut tmp = [0f32; 64];
    for v in 0..8 {
        let row = &f[v * 8..v * 8 + 8];
        if row[1..].iter().all(|&c| c == 0.0) {
            // DC-only row: constant along x.
            let dc = row[0] * basis[0][0];
            for x in 0..8 {
                tmp[v * 8 + x] = dc;
            }
            continue;
        }
        for x in 0..8 {
            let b = &basis[x];
            let mut acc = 0f32;
            for u in 0..8 {
                acc += b[u] * row[u];
            }
            tmp[v * 8 + x] = acc;
        }
    }
    // Columns: for each x, transform along v.
    for x in 0..8 {
        for y in 0..8 {
            let b = &basis[y];
            let mut acc = 0f32;
            for v in 0..8 {
                acc += b[v] * tmp[v * 8 + x];
            }
            // 10918-1 A.3.3: f(x, y) = 1/4 Σ Σ C(u) C(v) F(u, v) cos cos;
            // A.3.1 level shift by 2^(P−1) = 128; F.2.1.5 clamp.
            let s = acc * 0.25 + 128.0;
            out[y * 8 + x] = (s + 0.5).floor().clamp(0.0, 255.0) as u8;
        }
    }
}

/// Reconstruct the sample planes of a transcode frame from its exact
/// coefficients. `width` / `height` are the frame's logical pixel
/// dimensions (the coefficient canvases are MCU-padded).
pub(crate) fn planes_from_coefficients(
    tc: &TranscodedCoefficients,
    width: u32,
    height: u32,
) -> Result<TranscodedPlanes> {
    if width == 0 || height == 0 {
        return Err(Error::InvalidData(
            "JXL jpeg_pixels: zero-sized transcode frame".into(),
        ));
    }
    let shifts = {
        // `jpeg_upsampling` is in JXL channel order (Cb, Y, Cr); derive
        // the per-channel lattice the coefficient decoder used.
        let f = |ju: u32| -> (u32, u32) {
            match ju {
                1 => (2, 2),
                2 => (2, 1),
                3 => (1, 2),
                _ => (1, 1),
            }
        };
        let factors = [
            f(tc.jpeg_upsampling[0]),
            f(tc.jpeg_upsampling[1]),
            f(tc.jpeg_upsampling[2]),
        ];
        let hmax = factors.iter().map(|&(h, _)| h).max().unwrap_or(1);
        let vmax = factors.iter().map(|&(_, v)| v).max().unwrap_or(1);
        let sh = |c: usize| -> (u32, u32) {
            (
                (hmax / factors[c].0).trailing_zeros(),
                (vmax / factors[c].1).trailing_zeros(),
            )
        };
        [sh(0), sh(1), sh(2)]
    };
    if shifts[1] != (0, 0) {
        return Err(Error::Unsupported(format!(
            "JXL jpeg_pixels: luma stored subsampled (jpeg_upsampling {:?}); only chroma \
             subsampling has a JPEG sample layout",
            tc.jpeg_upsampling
        )));
    }
    if shifts[0] != shifts[2] {
        return Err(Error::Unsupported(format!(
            "JXL jpeg_pixels: Cb / Cr on different lattices (jpeg_upsampling {:?})",
            tc.jpeg_upsampling
        )));
    }
    let basis = idct_basis();
    let mut planes: Vec<SamplePlane> = Vec::with_capacity(3);
    for (c, &(hs, vs)) in shifts.iter().enumerate() {
        let (cw, ch) = tc.cdims[c];
        let pw = (width as usize).div_ceil(1 << hs);
        let ph = (height as usize).div_ceil(1 << vs);
        if cw * 8 < pw || ch * 8 < ph {
            return Err(Error::InvalidData(format!(
                "JXL jpeg_pixels: channel {c} block canvas {cw}×{ch} does not cover its \
                 {pw}×{ph} sample extent"
            )));
        }
        let coeffs = &tc.coeffs[c];
        let quant = &tc.quant[c];
        if coeffs.len() != cw * ch * 64 || quant.len() != 64 {
            return Err(Error::InvalidData(
                "JXL jpeg_pixels: coefficient / quant table size mismatch".into(),
            ));
        }
        let mut samples = vec![0u8; pw * ph];
        let mut block = [0u8; 64];
        for by in 0..ch {
            let y0 = by * 8;
            if y0 >= ph {
                break;
            }
            for bx in 0..cw {
                let x0 = bx * 8;
                if x0 >= pw {
                    break;
                }
                let base = (by * cw + bx) * 64;
                idct_block(&coeffs[base..base + 64], quant, &basis, &mut block);
                let rows = (ph - y0).min(8);
                let cols = (pw - x0).min(8);
                for r in 0..rows {
                    let dst = (y0 + r) * pw + x0;
                    samples[dst..dst + cols].copy_from_slice(&block[r * 8..r * 8 + cols]);
                }
            }
        }
        planes.push(SamplePlane {
            width: pw,
            height: ph,
            samples,
        });
    }
    // JXL channel order is (Cb, Y, Cr); emit JPEG order (Y, Cb, Cr).
    let cr = planes.pop().expect("three planes");
    let y = planes.pop().expect("three planes");
    let cb = planes.pop().expect("three planes");
    Ok(TranscodedPlanes {
        width,
        height,
        planes: [y, cb, cr],
        chroma_shift: shifts[0],
    })
}

/// ISO/IEC 18181-1 J.2 simple upsampling along one axis: every sample
/// `B` with neighbours `A` (before) and `C` (after) becomes the pair
/// `0.25·A + 0.75·B`, `0.75·B + 0.25·C`; missing edge neighbours
/// replicate the edge sample; the result is capped to `out_len`.
fn upsample_axis_j2(src: &[f32], out: &mut [f32]) {
    let n = src.len();
    let out_len = out.len();
    if n == 0 {
        return;
    }
    for (i, &b) in src.iter().enumerate() {
        let a = if i == 0 { b } else { src[i - 1] };
        let c = if i + 1 >= n { b } else { src[i + 1] };
        let lo = 2 * i;
        if lo < out_len {
            out[lo] = 0.25 * a + 0.75 * b;
        }
        if lo + 1 < out_len {
            out[lo + 1] = 0.75 * b + 0.25 * c;
        }
    }
}

/// J.2 upsampling of a `pw × ph` chroma plane to the luma extent
/// `w × h`, horizontally when `hshift == 1` and vertically when
/// `vshift == 1` (either, both or neither). `stride` is the source
/// plane's row stride in bytes.
fn upsample_plane_j2(
    src: &[u8],
    stride: usize,
    pw: usize,
    ph: usize,
    (hshift, vshift): (u32, u32),
    w: usize,
    h: usize,
) -> Vec<f32> {
    // Horizontal pass → `w × ph`.
    let mut horiz = vec![0f32; w * ph];
    let mut row_in = vec![0f32; pw];
    for y in 0..ph {
        let row = &src[y * stride..y * stride + pw];
        for (d, &s) in row_in.iter_mut().zip(row) {
            *d = f32::from(s);
        }
        let out = &mut horiz[y * w..(y + 1) * w];
        if hshift == 1 {
            upsample_axis_j2(&row_in, out);
        } else {
            let n = pw.min(w);
            out[..n].copy_from_slice(&row_in[..n]);
            // A 4:4:4 chroma plane always spans the luma width; this
            // only guards a malformed geometry.
            for v in out[n..].iter_mut() {
                *v = row_in[pw - 1];
            }
        }
    }
    if vshift != 1 {
        if ph == h {
            return horiz;
        }
        let mut out = vec![0f32; w * h];
        for y in 0..h {
            let sy = y.min(ph - 1);
            out[y * w..(y + 1) * w].copy_from_slice(&horiz[sy * w..(sy + 1) * w]);
        }
        return out;
    }
    // Vertical pass → `w × h`.
    let mut out = vec![0f32; w * h];
    let mut col_in = vec![0f32; ph];
    let mut col_out = vec![0f32; h];
    for x in 0..w {
        for y in 0..ph {
            col_in[y] = horiz[y * w + x];
        }
        upsample_axis_j2(&col_in, &mut col_out);
        for y in 0..h {
            out[y * w + x] = col_out[y];
        }
    }
    out
}

/// §L.3 / T.871 §7 inverse YCbCr for one pixel (chroma centred on 128),
/// rounded to nearest and clamped.
#[inline]
pub(crate) fn ycbcr_to_rgb8(y: f32, cb: f32, cr: f32) -> [u8; 3] {
    let cb = cb - 128.0;
    let cr = cr - 128.0;
    let r = y + 1.402 * cr;
    let g = y - 0.344_136 * cb - 0.714_136 * cr;
    let b = y + 1.772 * cb;
    let q = |v: f32| -> u8 { (v + 0.5).floor().clamp(0.0, 255.0) as u8 };
    [q(r), q(g), q(b)]
}

/// Convert planar 8-bit YCbCr (`Y` at `width × height`, `Cb` / `Cr`
/// at `ceil(width >> hshift) × ceil(height >> vshift)`) into packed RGB
/// (`out_channels == 3`) or RGBA with opaque alpha (`out_channels ==
/// 4`): J.2 upsampling then §L.3.
#[allow(clippy::too_many_arguments)]
pub(crate) fn planar_ycbcr_to_rgb(
    width: usize,
    height: usize,
    y: (&[u8], usize),
    cb: (&[u8], usize),
    cr: (&[u8], usize),
    chroma_shift: (u32, u32),
    out_channels: usize,
) -> Result<Vec<u8>> {
    let pw = width.div_ceil(1 << chroma_shift.0);
    let ph = height.div_ceil(1 << chroma_shift.1);
    let check = |name: &str, (data, stride): (&[u8], usize), w: usize, h: usize| -> Result<()> {
        if stride < w || data.len() < stride.saturating_mul(h.saturating_sub(1)).saturating_add(w) {
            return Err(Error::InvalidData(format!(
                "JxlImage: {name} plane geometry (stride {stride}, {} bytes) does not cover \
                 {w}×{h}",
                data.len()
            )));
        }
        Ok(())
    };
    check("Y", y, width, height)?;
    check("Cb", cb, pw, ph)?;
    check("Cr", cr, pw, ph)?;
    let cb_up = upsample_plane_j2(cb.0, cb.1, pw, ph, chroma_shift, width, height);
    let cr_up = upsample_plane_j2(cr.0, cr.1, pw, ph, chroma_shift, width, height);
    let mut out = Vec::with_capacity(width * height * out_channels);
    for row in 0..height {
        let yrow = &y.0[row * y.1..row * y.1 + width];
        let base = row * width;
        for x in 0..width {
            let rgb = ycbcr_to_rgb8(f32::from(yrow[x]), cb_up[base + x], cr_up[base + x]);
            out.extend_from_slice(&rgb);
            if out_channels == 4 {
                out.push(255);
            }
        }
    }
    Ok(out)
}

impl TranscodedPlanes {
    /// The three planes as a planar frame in `[Y, Cb, Cr]` order (each
    /// plane tightly packed).
    pub(crate) fn into_planar_frame(self, pts: Option<i64>) -> RawFrame {
        RawFrame {
            pts,
            planes: self
                .planes
                .into_iter()
                .map(|p| Plane::new(p.width, p.samples))
                .collect(),
        }
    }

    /// J.2 + §L.3 to three `width × height` planes `[R, G, B]` for the
    /// generic (composition / orientation) frame pipeline.
    pub(crate) fn to_rgb_frame(&self, pts: Option<i64>) -> Result<RawFrame> {
        let w = self.width as usize;
        let h = self.height as usize;
        let [y, cb, cr] = &self.planes;
        let packed = planar_ycbcr_to_rgb(
            w,
            h,
            (&y.samples, y.width),
            (&cb.samples, cb.width),
            (&cr.samples, cr.width),
            self.chroma_shift,
            3,
        )?;
        let mut r = Vec::with_capacity(w * h);
        let mut g = Vec::with_capacity(w * h);
        let mut b = Vec::with_capacity(w * h);
        for px in packed.chunks_exact(3) {
            r.push(px[0]);
            g.push(px[1]);
            b.push(px[2]);
        }
        Ok(RawFrame {
            pts,
            planes: vec![Plane::new(w, r), Plane::new(w, g), Plane::new(w, b)],
        })
    }

    /// The luma plane alone (greyscale transcodes: the chroma planes
    /// decode to the neutral 128).
    pub(crate) fn into_luma_frame(mut self, pts: Option<i64>) -> RawFrame {
        let y = std::mem::replace(
            &mut self.planes[0],
            SamplePlane {
                width: 0,
                height: 0,
                samples: Vec::new(),
            },
        );
        RawFrame {
            pts,
            planes: vec![Plane::new(y.width, y.samples)],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_only_block_is_flat() {
        let basis = idct_basis();
        let mut coeffs = [0i32; 64];
        coeffs[0] = 8; // DC 8 × Q 2 = 16 → 16 / 8 = +2 per sample.
        let quant = [2i32; 64];
        let mut out = [0u8; 64];
        idct_block(&coeffs, &quant, &basis, &mut out);
        assert!(out.iter().all(|&v| v == 130), "{out:?}");
    }

    #[test]
    fn idct_clamps_and_rounds() {
        let basis = idct_basis();
        let mut coeffs = [0i32; 64];
        coeffs[0] = -2000;
        let quant = [1i32; 64];
        let mut out = [0u8; 64];
        idct_block(&coeffs, &quant, &basis, &mut out);
        assert!(out.iter().all(|&v| v == 0));
        coeffs[0] = 2000;
        idct_block(&coeffs, &quant, &basis, &mut out);
        assert!(out.iter().all(|&v| v == 255));
    }

    #[test]
    fn j2_axis_is_the_triangle_filter_with_edge_replication() {
        let src = [0f32, 100.0, 200.0];
        let mut out = [0f32; 6];
        upsample_axis_j2(&src, &mut out);
        assert_eq!(out, [0.0, 25.0, 75.0, 125.0, 175.0, 200.0]);
        // Width cap: an odd luma width drops the trailing half-sample.
        let mut out5 = [0f32; 5];
        upsample_axis_j2(&src, &mut out5);
        assert_eq!(out5, [0.0, 25.0, 75.0, 125.0, 175.0]);
    }

    #[test]
    fn ycbcr_neutral_chroma_is_grey() {
        assert_eq!(ycbcr_to_rgb8(77.0, 128.0, 128.0), [77, 77, 77]);
        // 255 + 1.402·127 clamps; 255 + 0.344136·128 − 0.714136·127 = 208.35;
        // 255 − 1.772·128 = 28.18.
        assert_eq!(ycbcr_to_rgb8(255.0, 0.0, 255.0), [255, 208, 28]);
        // 0 − 1.402·128 clamps; 0 − 0.344136·127 + 0.714136·128 = 47.70;
        // 0 + 1.772·127 = 225.04.
        assert_eq!(ycbcr_to_rgb8(0.0, 255.0, 0.0), [0, 48, 225]);
    }

    #[test]
    fn planar_444_round_trips_the_kernel() {
        let y = [10u8, 20, 30, 40];
        let cb = [128u8; 4];
        let cr = [128u8; 4];
        let rgb = planar_ycbcr_to_rgb(2, 2, (&y, 2), (&cb, 2), (&cr, 2), (0, 0), 3).unwrap();
        assert_eq!(rgb, vec![10, 10, 10, 20, 20, 20, 30, 30, 30, 40, 40, 40]);
        let rgba = planar_ycbcr_to_rgb(2, 2, (&y, 2), (&cb, 2), (&cr, 2), (0, 0), 4).unwrap();
        assert_eq!(rgba.len(), 16);
        assert!(rgba.iter().skip(3).step_by(4).all(|&a| a == 255));
    }

    #[test]
    fn planar_420_flat_chroma_upsamples_flat() {
        // 3×3 luma, 2×2 chroma (ceil), constant chroma → constant colour cast.
        let y = [100u8; 9];
        let cb = [128u8; 4];
        let cr = [200u8; 4];
        let rgb = planar_ycbcr_to_rgb(3, 3, (&y, 3), (&cb, 2), (&cr, 2), (1, 1), 3).unwrap();
        let expect = ycbcr_to_rgb8(100.0, 128.0, 200.0);
        for px in rgb.chunks_exact(3) {
            assert_eq!(px, &expect);
        }
    }

    #[test]
    fn planar_geometry_is_validated() {
        let y = [0u8; 4];
        let short = [0u8; 1];
        assert!(planar_ycbcr_to_rgb(2, 2, (&y, 2), (&short, 1), (&short, 1), (0, 0), 3).is_err());
        assert!(planar_ycbcr_to_rgb(2, 2, (&y, 2), (&short, 1), (&short, 1), (1, 1), 3).is_ok());
    }
}
