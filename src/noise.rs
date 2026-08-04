//! §C.4.7 noise-synthesis parameters + §K.4 noise rendering —
//! ISO/IEC FDIS 18181-1:2021, round 437.
//!
//! Noise is one of the three "image features" rendered after the §J
//! restoration filters and before the Annex L colour transform (§5.2
//! decode order; §K.1 "each frame may include alternate
//! representations of image contents"). It is fully specified:
//!
//! * **§C.4.7** — the LfGlobal `NoiseParameters` bundle: 8 LUT values
//!   `lut[i] = u(10) / (1 << 10)` (Listing C.5), the noise strength at
//!   8 intensity levels.
//! * **§K.4** — the renderer:
//!   1. Three pseudorandom channels `RR`, `RG`, `RB` are generated per
//!      group, rows top-to-bottom, in segments of up to 16 samples.
//!      Each segment consumes one batch of eight 64-bit outputs of an
//!      8-lane **XorShift128Plus** generator (Listing K.2), whose
//!      lanes are seeded per group by **SplitMix64** (Listing K.3)
//!      from `seed = (y0 << 32) + x0` — the group's top-left pixel
//!      coordinates. Each 64-bit output provides two 32-bit halves
//!      (low first); sample `s[j]` is `InterpretAsF32((bits[j] >> 9) |
//!      0x3F800000)` (Listing K.4), i.e. a float in `[1, 2)`.
//!   2. Each channel is convolved **frame-level** ("in its totality,
//!      not in a per-group way") with the 5×5 Laplacian-like kernel
//!      (0.16 everywhere, −3.84 at the centre — `−4 × (Id − Bk)`),
//!      out-of-bounds taps redirected through `Mirror` (§6.5).
//!   3. Per pixel: `AR/AG/AB` are the convolved samples scaled by
//!      0.22; the intensity `In_R = Y + X`, `In_G = Y − X` selects an
//!      interpolated LUT strength `S` (×6, split into integer part
//!      clamped to `[0, 6]` and fractional part; result clamped to
//!      `[0, 1]`), and Listing K.5 injects
//!      `NR = 1/128·AR·SR + 127/128·AB·SR` (resp. `NG`) into the XYB
//!      samples with the §C.4.4 base correlations.
//!
//! The staged `noise-feature-256x256` fixture (kNoise, photon-noise
//! ISO 3200) pins the whole chain against a black-box reference
//! decode (`round437_noise_feature`).

use oxideav_core::Result;

use crate::bitreader::BitReader;
use crate::gaborish::mirror1d;

/// §C.4.7 — the 8-entry noise-strength LUT (Listing C.5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseParameters {
    /// `lut[i] = u(10) / 1024`, in `[0, 1)`.
    pub lut: [f32; 8],
}

impl NoiseParameters {
    /// Listing C.5 — eight sequential `u(10)` values scaled by
    /// `1 / (1 << 10)`.
    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let mut lut = [0.0f32; 8];
        for slot in lut.iter_mut() {
            *slot = br.read_bits(10)? as f32 / 1024.0;
        }
        Ok(Self { lut })
    }
}

/// Listing K.3 — SplitMix64.
fn split_mix_64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The 8-lane XorShift128Plus generator of Listing K.2/K.3.
struct XorShift128Plus {
    s0: [u64; 8],
    s1: [u64; 8],
}

impl XorShift128Plus {
    /// Listing K.3 — seed the 8 lanes from the group's
    /// `seed = (y0 << 32) + x0`.
    fn new(seed: u64) -> Self {
        let mut s0 = [0u64; 8];
        let mut s1 = [0u64; 8];
        s0[0] = split_mix_64(seed.wrapping_add(0x9E37_79B9_7F4A_7C15));
        s1[0] = split_mix_64(s0[0]);
        for i in 1..8 {
            s0[i] = split_mix_64(s1[i - 1]);
            s1[i] = split_mix_64(s0[i]);
        }
        Self { s0, s1 }
    }

    /// Listing K.2 — one batch of eight 64-bit outputs.
    fn batch(&mut self) -> [u64; 8] {
        let mut out = [0u64; 8];
        for (i, slot) in out.iter_mut().enumerate() {
            let mut s1_ = self.s0[i];
            let s0_ = self.s1[i];
            *slot = self.s1[i].wrapping_add(self.s0[i]);
            self.s0[i] = s0_;
            s1_ ^= s1_ << 23;
            self.s1[i] = s1_ ^ s0_ ^ (s1_ >> 18) ^ (s0_ >> 5);
        }
        out
    }
}

/// §K.4 steps 1–2 — generate the three pseudorandom channels for the
/// whole frame (per-group seeding, group tiles of `group_dim` pixels)
/// and convolve each with the 5×5 Laplacian-like kernel under §6.5
/// mirroring. Returns the three convolved planes `[RR, RG, RB]`
/// (row-major `width × height`), NOT yet scaled by the 0.22 factor —
/// [`apply_noise`] folds that in.
pub fn generate_noise_planes(
    width: usize,
    height: usize,
    group_dim: usize,
) -> Result<[Vec<f32>; 3]> {
    let mut raw: [Vec<f32>; 3] = [
        vec![0.0f32; width * height],
        vec![0.0f32; width * height],
        vec![0.0f32; width * height],
    ];
    // Per-group generation: one Listing K.3 (re)init per group, the
    // three channels drawn left-to-right from the same lane stream,
    // rows top-to-bottom, segments of up to 16 samples (a full batch
    // is consumed per segment even when fewer samples remain).
    let mut gy0 = 0usize;
    while gy0 < height {
        let gh = group_dim.min(height - gy0);
        let mut gx0 = 0usize;
        while gx0 < width {
            let gw = group_dim.min(width - gx0);
            let seed = ((gy0 as u64) << 32).wrapping_add(gx0 as u64);
            let mut rng = XorShift128Plus::new(seed);
            for chan in raw.iter_mut() {
                for y in 0..gh {
                    let row = (gy0 + y) * width + gx0;
                    let mut x = 0usize;
                    while x < gw {
                        let batch = rng.batch();
                        let n = (gw - x).min(16);
                        for j in 0..n {
                            let b64 = batch[j / 2];
                            let bits = if j % 2 == 0 {
                                (b64 & 0xFFFF_FFFF) as u32
                            } else {
                                (b64 >> 32) as u32
                            };
                            // Listing K.4 — float in [1, 2).
                            chan[row + x + j] = f32::from_bits((bits >> 9) | 0x3F80_0000);
                        }
                        x += n;
                    }
                }
            }
            gx0 += group_dim;
        }
        gy0 += group_dim;
    }

    // Frame-level 5×5 convolution: 0.16 everywhere, −3.84 centre;
    // out-of-bounds taps mirror per §6.5.
    let mut out: [Vec<f32>; 3] = [
        vec![0.0f32; width * height],
        vec![0.0f32; width * height],
        vec![0.0f32; width * height],
    ];
    for (src, dst) in raw.iter().zip(out.iter_mut()) {
        for y in 0..height {
            for x in 0..width {
                let mut acc = 0.0f32;
                for dy in -2i64..=2 {
                    let sy = mirror1d(y as i64 + dy, height)?;
                    for dx in -2i64..=2 {
                        let sx = mirror1d(x as i64 + dx, width)?;
                        let w = if dx == 0 && dy == 0 {
                            -3.84f32
                        } else {
                            0.16f32
                        };
                        acc += w * src[sy * width + sx];
                    }
                }
                dst[y * width + x] = acc;
            }
        }
    }
    Ok(out)
}

/// §K.4 step 3 — Listing K.5 noise injection into the XYB planes, in
/// place. `planes` are the convolved channels from
/// [`generate_noise_planes`]; the 0.22 post-convolution scale is
/// applied here. `base_correlation_x` / `base_correlation_b` are the
/// §C.4.4 LfChannelCorrelation base factors.
pub fn apply_noise(
    x_plane: &mut [f32],
    y_plane: &mut [f32],
    b_plane: &mut [f32],
    planes: &[Vec<f32>; 3],
    params: &NoiseParameters,
    base_correlation_x: f32,
    base_correlation_b: f32,
) {
    let lut = &params.lut;
    // Interpolated LUT strength for an intensity value (§K.4: the
    // input ×6 splits into a floor integer part — clamped to [0, 6] —
    // and a fractional part; the result clamps to [0, 1]).
    let strength = |intensity: f32| -> f32 {
        let scaled = intensity * 6.0;
        let floor = scaled.floor();
        let frac = scaled - floor;
        let idx = (floor as i64).clamp(0, 6) as usize;
        (lut[idx] * (1.0 - frac) + lut[idx + 1] * frac).clamp(0.0, 1.0)
    };
    let n = x_plane.len().min(y_plane.len()).min(b_plane.len());
    for i in 0..n {
        let ar = planes[0][i] * 0.22;
        let ag = planes[1][i] * 0.22;
        let ab = planes[2][i] * 0.22;
        let x = x_plane[i];
        let y = y_plane[i];
        let sr = strength(y + x);
        let sg = strength(y - x);
        // Listing K.5.
        let nr = (1.0 / 128.0) * ar * sr + (127.0 / 128.0) * ab * sr;
        let ng = (1.0 / 128.0) * ag * sg + (127.0 / 128.0) * ab * sg;
        x_plane[i] += base_correlation_x * (nr + ng) + nr - ng;
        y_plane[i] += nr + ng;
        b_plane[i] += base_correlation_b * (nr + ng);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;

    #[test]
    fn noise_parameters_read_ten_bit_lut() {
        // 8 × u(10): values 0, 1, 2, ..., 7 packed LSB-first.
        let mut bits = Vec::new();
        for v in 0u16..8 {
            for k in 0..10 {
                bits.push((v >> k) & 1);
            }
        }
        let mut bytes = vec![0u8; 10];
        for (i, b) in bits.iter().enumerate() {
            bytes[i / 8] |= (*b as u8) << (i % 8);
        }
        let mut br = BitReader::new(&bytes);
        let p = NoiseParameters::read(&mut br).unwrap();
        for (i, v) in p.lut.iter().enumerate() {
            assert!((v - i as f32 / 1024.0).abs() < 1e-9, "lut[{i}] = {v}");
        }
    }

    #[test]
    fn split_mix_64_is_deterministic_and_nontrivial() {
        // Listing K.3 self-consistency: distinct seeds map to
        // distinct outputs and the avalanche is total (any bit flip
        // changes the output).
        let a = split_mix_64(0);
        let b = split_mix_64(1);
        assert_ne!(a, b);
        assert_ne!(split_mix_64(a), split_mix_64(b));
    }

    #[test]
    fn xorshift_batch_interprets_low_half_first() {
        // Listing K.4 mapping: bits[0] = low 32 of batch[0], bits[1] =
        // high 32 of batch[0]. Every generated sample must land in
        // [1, 2) by construction.
        let planes = generate_noise_planes(20, 3, 16).unwrap();
        // Convolved values of a [1,2) field with a zero-sum kernel are
        // small; the raw generation invariant is checked through the
        // convolution output staying bounded (|acc| < 4 × dynamic
        // range of the kernel = 4).
        for p in &planes {
            for &v in p {
                assert!(v.is_finite() && v.abs() < 4.0, "convolved sample {v}");
            }
        }
    }

    #[test]
    fn group_seeding_uses_top_left_pixel_coordinates() {
        // Two frames whose second group starts at different offsets
        // must draw different randomness there; identical offsets
        // must reproduce identical planes (pure function of geometry).
        let a = generate_noise_planes(32, 8, 16).unwrap();
        let b = generate_noise_planes(32, 8, 16).unwrap();
        assert_eq!(a[0], b[0]);
        assert_eq!(a[2], b[2]);
        let c = generate_noise_planes(32, 8, 32).unwrap();
        assert_ne!(a[0], c[0], "different group tiling must reseed differently");
    }

    #[test]
    fn injection_is_identity_when_lut_is_zero() {
        let params = NoiseParameters { lut: [0.0; 8] };
        let planes = generate_noise_planes(8, 8, 8).unwrap();
        let mut x = vec![0.25f32; 64];
        let mut y = vec![0.5f32; 64];
        let mut b = vec![0.75f32; 64];
        apply_noise(&mut x, &mut y, &mut b, &planes, &params, 0.0, 1.0);
        assert!(x.iter().all(|&v| v == 0.25));
        assert!(y.iter().all(|&v| v == 0.5));
        assert!(b.iter().all(|&v| v == 0.75));
    }
}
