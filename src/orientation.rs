//! §A.6 orientation transform — ISO/IEC FDIS 18181-1:2021 Table A.17.
//!
//! `ImageMetadata.orientation` (1–8, Exif-compatible per the Table A.17
//! NOTE) names the transform the decoder applies **after** decoding the
//! image. The codestream's SizeHeader dimensions describe the *sample
//! grid* ("width and height are interpreted as the dimensions of the
//! sample grid, not its rotated/mirrored interpretation"), so for the
//! transposing orientations 5–8 the presented image has swapped
//! dimensions. All decode-side geometry (groups, frames, composition)
//! runs in sample-grid space; this module is the final presentation
//! step.
//!
//! | orientation | transform (applied to the coded image)            |
//! |-------------|----------------------------------------------------|
//! | 1           | none                                               |
//! | 2           | flip horizontally                                  |
//! | 3           | rotate 180°                                        |
//! | 4           | flip vertically                                    |
//! | 5           | transpose (rotate 90° CW, then flip horizontally)  |
//! | 6           | rotate 90° clockwise                               |
//! | 7           | flip horizontally, then rotate 90° clockwise       |
//! | 8           | rotate 90° counterclockwise                        |

use oxideav_core::{Error, Result, VideoFrame, VideoPlane};

/// Apply the Table A.17 orientation transform to a decoded frame whose
/// planes use the crate byte layout (`bytes_per_sample` ∈ {1, 2}).
/// Orientation 1 returns the frame unchanged.
pub fn apply_orientation(
    frame: VideoFrame,
    orientation: u8,
    bytes_per_sample: usize,
) -> Result<VideoFrame> {
    if orientation <= 1 {
        return Ok(frame);
    }
    if !(2..=8).contains(&orientation) {
        return Err(Error::InvalidData(format!(
            "JXL orientation: value {orientation} outside 1..=8"
        )));
    }
    let planes = frame
        .planes
        .into_iter()
        .map(|p| orient_plane(p, orientation, bytes_per_sample))
        .collect::<Result<Vec<_>>>()?;
    Ok(VideoFrame {
        pts: frame.pts,
        planes,
    })
}

fn orient_plane(p: VideoPlane, orientation: u8, bytes: usize) -> Result<VideoPlane> {
    if p.stride == 0 || p.stride % bytes != 0 || p.data.len() % p.stride != 0 {
        return Err(Error::InvalidData(format!(
            "JXL orientation: plane geometry (stride {}, {} bytes) not sample-aligned",
            p.stride,
            p.data.len()
        )));
    }
    let w = p.stride / bytes;
    let h = p.data.len() / p.stride;
    // Presented dimensions: orientations 5–8 transpose.
    let (ow, oh) = if orientation >= 5 { (h, w) } else { (w, h) };
    let mut out = vec![0u8; p.data.len()];
    for oy in 0..oh {
        for ox in 0..ow {
            // Map the presented coordinate back to the coded grid.
            let (cx, cy) = match orientation {
                2 => (w - 1 - ox, oy),
                3 => (w - 1 - ox, h - 1 - oy),
                4 => (ox, h - 1 - oy),
                // 5: transpose.
                5 => (oy, ox),
                // 6: rotate 90° CW — coded top row becomes the
                // presented right column.
                6 => (oy, h - 1 - ox),
                // 7: flip horizontally then rotate 90° CW
                // (anti-transpose).
                7 => (w - 1 - oy, h - 1 - ox),
                // 8: rotate 90° CCW.
                8 => (w - 1 - oy, ox),
                _ => unreachable!("gated above"),
            };
            let src = (cy * w + cx) * bytes;
            let dst = (oy * ow + ox) * bytes;
            out[dst..dst + bytes].copy_from_slice(&p.data[src..src + bytes]);
        }
    }
    Ok(VideoPlane {
        stride: ow * bytes,
        data: out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2×3 coded plane:
    /// ```text
    /// 1 2
    /// 3 4
    /// 5 6
    /// ```
    fn plane_2x3() -> VideoPlane {
        VideoPlane {
            stride: 2,
            data: vec![1, 2, 3, 4, 5, 6],
        }
    }

    fn oriented(orientation: u8) -> (usize, Vec<u8>) {
        let f = VideoFrame {
            pts: None,
            planes: vec![plane_2x3()],
        };
        let out = apply_orientation(f, orientation, 1).unwrap();
        (out.planes[0].stride, out.planes[0].data.clone())
    }

    #[test]
    fn identity_and_flips() {
        assert_eq!(oriented(1), (2, vec![1, 2, 3, 4, 5, 6]));
        // 2: flip horizontally.
        assert_eq!(oriented(2), (2, vec![2, 1, 4, 3, 6, 5]));
        // 3: rotate 180°.
        assert_eq!(oriented(3), (2, vec![6, 5, 4, 3, 2, 1]));
        // 4: flip vertically.
        assert_eq!(oriented(4), (2, vec![5, 6, 3, 4, 1, 2]));
    }

    #[test]
    fn transposing_orientations() {
        // 5: transpose — presented 3×2.
        assert_eq!(oriented(5), (3, vec![1, 3, 5, 2, 4, 6]));
        // 6: rotate 90° CW — coded top row = presented right column,
        // i.e. first presented row reads the coded first column
        // bottom-up.
        assert_eq!(oriented(6), (3, vec![5, 3, 1, 6, 4, 2]));
        // 7: flip horizontally then rotate 90° CW.
        assert_eq!(oriented(7), (3, vec![6, 4, 2, 5, 3, 1]));
        // 8: rotate 90° CCW.
        assert_eq!(oriented(8), (3, vec![2, 4, 6, 1, 3, 5]));
    }

    #[test]
    fn two_byte_samples_move_as_units() {
        let f = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![1, 10, 2, 20, 3, 30, 4, 40],
            }],
        };
        let out = apply_orientation(f, 3, 2).unwrap();
        assert_eq!(out.planes[0].stride, 4);
        assert_eq!(out.planes[0].data, vec![4, 40, 3, 30, 2, 20, 1, 10]);
    }

    #[test]
    fn out_of_range_orientation_rejected() {
        let f = VideoFrame {
            pts: None,
            planes: vec![plane_2x3()],
        };
        assert!(apply_orientation(f, 9, 1).is_err());
    }
}
