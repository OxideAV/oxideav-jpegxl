//! Where the encoder reads a picture from.
//!
//! The coders work a group at a time, so they never need the picture
//! as planes of their own: [`Pixels`] hands out rows and rectangles of
//! one channel straight from the caller's storage — either the packed
//! interleaved bytes of a [`crate::JxlImage`] or integer planes — and
//! the only copies that exist are the rectangles in flight.

use crate::error::{Error, Result};

/// A read-only view of one picture's integer samples.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Pixels<'a> {
    /// One plane per channel, `width` samples per row.
    Planes {
        planes: &'a [Vec<i32>],
        width: usize,
    },
    /// Interleaved samples, `bytes` (1 or 2, little-endian) each,
    /// `channels` per pixel, `stride` bytes per row.
    Packed {
        data: &'a [u8],
        stride: usize,
        channels: usize,
        bytes: usize,
    },
}

impl Pixels<'_> {
    /// Number of channels.
    pub(crate) fn num_channels(&self) -> usize {
        match *self {
            Pixels::Planes { planes, .. } => planes.len(),
            Pixels::Packed { channels, .. } => channels,
        }
    }

    /// `out.len()` samples of channel `c` on row `y`, from column `x0`.
    #[inline]
    pub(crate) fn row(&self, c: usize, y: usize, x0: usize, out: &mut [i32]) {
        match *self {
            Pixels::Planes { planes, width } => {
                let o = y * width + x0;
                out.copy_from_slice(&planes[c][o..o + out.len()]);
            }
            Pixels::Packed {
                data,
                stride,
                channels,
                bytes,
            } => {
                let step = channels * bytes;
                let o = y * stride + x0 * step + c * bytes;
                let src = &data[o..];
                if bytes == 1 {
                    for (slot, px) in out.iter_mut().zip(src.chunks(step)) {
                        *slot = i32::from(px[0]);
                    }
                } else {
                    for (slot, px) in out.iter_mut().zip(src.chunks(step)) {
                        *slot = i32::from(u16::from_le_bytes([px[0], px[1]]));
                    }
                }
            }
        }
    }

    /// The `rw × rh` rectangle of channel `c` at `(x0, y0)`, row-major.
    pub(crate) fn rect(&self, c: usize, x0: usize, y0: usize, rw: usize, rh: usize) -> Vec<i32> {
        let mut out = vec![0i32; rw * rh];
        if rw > 0 {
            for (y, line) in out.chunks_exact_mut(rw).enumerate() {
                self.row(c, y0 + y, x0, line);
            }
        }
        out
    }

    /// Check the geometry and that every sample lies in `0..=max`.
    pub(crate) fn validate(
        &self,
        width: usize,
        height: usize,
        n_ch: usize,
        max: i64,
    ) -> Result<()> {
        if self.num_channels() != n_ch {
            return Err(Error::invalid(format!(
                "JXL encoder: frame carries {} channel(s), {n_ch} expected",
                self.num_channels()
            )));
        }
        match *self {
            Pixels::Planes { planes, .. } => {
                for (c, ch) in planes.iter().enumerate() {
                    if ch.len() != width * height {
                        return Err(Error::invalid(format!(
                            "JXL encoder: channel {c} has {} samples, {} expected",
                            ch.len(),
                            width * height
                        )));
                    }
                    if ch.iter().any(|&v| v < 0 || i64::from(v) > max) {
                        return Err(Error::invalid(format!(
                            "JXL encoder: channel {c} sample outside 0..={max}"
                        )));
                    }
                }
            }
            Pixels::Packed {
                data,
                stride,
                channels,
                bytes,
            } => {
                let row = width * channels * bytes;
                if height == 0 || stride < row || data.len() < stride * (height - 1) + row {
                    return Err(Error::invalid(
                        "JXL encoder: plane too short for the geometry",
                    ));
                }
                // Storage wider than the declared depth can hold a
                // sample the codestream cannot.
                if max < (1i64 << (8 * bytes)) - 1 {
                    for y in 0..height {
                        let line = &data[y * stride..y * stride + row];
                        let bad = if bytes == 1 {
                            line.iter().any(|&v| i64::from(v) > max)
                        } else {
                            line.as_chunks::<2>()
                                .0
                                .iter()
                                .any(|p| i64::from(u16::from_le_bytes(*p)) > max)
                        };
                        if bad {
                            return Err(Error::invalid(format!(
                                "JXL encoder: sample outside 0..={max} on row {y}"
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_and_planar_views_agree() {
        let (w, h, n) = (5usize, 4usize, 3usize);
        let planes: Vec<Vec<i32>> = (0..n)
            .map(|c| {
                (0..w * h)
                    .map(|i| ((i * 37 + c * 101) % 1000) as i32)
                    .collect()
            })
            .collect();
        let mut packed = Vec::new();
        for i in 0..w * h {
            for p in &planes {
                packed.extend_from_slice(&(p[i] as u16).to_le_bytes());
            }
        }
        // A stride with padding.
        let stride = w * n * 2 + 3;
        let mut padded = vec![0xAAu8; stride * h];
        for y in 0..h {
            padded[y * stride..y * stride + w * n * 2]
                .copy_from_slice(&packed[y * w * n * 2..(y + 1) * w * n * 2]);
        }
        let a = Pixels::Planes {
            planes: &planes,
            width: w,
        };
        let b = Pixels::Packed {
            data: &padded,
            stride,
            channels: n,
            bytes: 2,
        };
        for (c, plane) in planes.iter().enumerate() {
            assert_eq!(a.rect(c, 1, 1, 3, 2), b.rect(c, 1, 1, 3, 2));
            assert_eq!(&a.rect(c, 0, 0, w, h), plane);
            assert_eq!(&b.rect(c, 0, 0, w, h), plane);
        }
        let max = i64::from(*planes.iter().flatten().max().unwrap());
        assert!(a.validate(w, h, n, max).is_ok());
        assert!(b.validate(w, h, n, max).is_ok());
        assert!(a.validate(w, h, n, max - 1).is_err());
        assert!(b.validate(w, h, n, max - 1).is_err());
        assert!(b.validate(w, h + 1, n, 65535).is_err());
    }
}
