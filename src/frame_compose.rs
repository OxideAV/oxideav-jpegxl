//! Frame composition + reference-frame semantics — ISO/IEC FDIS
//! 18181-1:2021 §C.2 (Table C.7 `BlendingInfo`, Table C.8 `BlendMode`,
//! the `save_as_reference` / `Reference[…]` prose, and the crop-frame
//! "updates the rectangle of the previous frame" rule).
//!
//! ## Scope (round 389)
//!
//! The multi-frame walk (`decode_all_frames`) previously emitted each
//! decoded frame raw — correct only for the full-frame `kReplace`
//! chains the animation fixture exercises. This module supplies the
//! §C.2 composition state machine those frames plug into:
//!
//! * **Source frame** — *"All blending operations consider as
//!   'previous sample' the sample at the corresponding coordinates in
//!   the source frame, which is the frame that was previously stored
//!   in `Reference[source]` — if no frame was previously stored, the
//!   source frame is assumed to have all sample values set to
//!   zeroes."*
//! * **Crop rule** — *"If `have_crop`, the decoder considers the
//!   current frame to have dimensions width × height, and updates the
//!   rectangle of the previous frame with top-left corner x0, y0 with
//!   the current frame using the given blend_mode."* The composed
//!   output therefore equals the source frame outside the rectangle
//!   and the blended samples inside it.
//! * **Reference recording** — *"If `save_as_reference != 0`, the
//!   samples of the decoded frame are recorded as
//!   `Reference[save_as_reference]` … Blending is performed before
//!   recording the reference frame."* This module records the
//!   post-colour-transform samples (the `save_before_ct == false`
//!   case); a frame that asks for pre-CT recording surfaces a precise
//!   [`Error::Unsupported`] rather than recording the wrong domain.
//! * **Presentation** — *"If duration is zero and `!is_last`, the
//!   decoder does not present the current frame, but the frame may be
//!   composed together with the next frames."*
//!
//! Blend modes implemented on the three colour channels per Table C.8:
//! `kReplace` (`sample = new_sample`), `kAdd` (`sample = old_sample +
//! new_sample`), `kMul` (`sample = old_sample × new_sample`).
//! `kBlend` / `kAlphaWeightedAdd` consume the alpha extra channel,
//! which the multi-frame walk does not yet thread — they surface a
//! precise [`Error::Unsupported`].
//!
//! Blending is specified on the post-inverse-CT samples ("the blending
//! is done in the colour space after inverse colour transforms from
//! Annex L have been applied"); this module operates on the decoded
//! 8-bit planes, mapping through `[0, 1]` floats for the arithmetic
//! modes and re-quantising with round-half-up — exact for `kReplace`,
//! and within one quantisation step for `kAdd` / `kMul`.

use oxideav_core::{Error, Result, VideoFrame, VideoPlane};

use crate::frame_header::BlendMode;

/// Composition inputs of one decoded frame — the §C.2 fields the
/// multi-frame walk extracts from the `FrameHeader`.
#[derive(Debug, Clone, Copy)]
pub struct FrameComposeMeta {
    /// Crop rectangle top-left (0, 0 when `!have_crop`).
    pub x0: u32,
    pub y0: u32,
    /// Colour-channel blending (Table C.7 `blending_info`).
    pub mode: BlendMode,
    /// `Reference[source]` selector (0 when not on the wire).
    pub source: u32,
    /// `Reference[save_as_reference]` recording slot (0 = none).
    pub save_as_reference: u32,
    /// Whether the reference recording is requested pre-colour-transform.
    pub save_before_ct: bool,
    /// Animation tick duration (0 = not presented unless `is_last`).
    pub duration: u32,
    pub is_last: bool,
}

/// §C.2 composition state across a frame array: the image-sized canvas
/// dimensions and the `Reference[1..=3]` slots (slot 0 exists for
/// `source == 0` lookups but is never recorded — recording is gated on
/// `save_as_reference != 0`).
#[derive(Debug)]
pub struct ComposeState {
    width: usize,
    height: usize,
    reference: [Option<[Vec<u8>; 3]>; 4],
}

impl ComposeState {
    /// New state for a `width × height` image. Unstored reference
    /// slots read as all-zero source frames per §C.2.
    pub fn new(width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidData(
                "JXL frame composition: zero-dimension image".into(),
            ));
        }
        Ok(Self {
            width,
            height,
            reference: [None, None, None, None],
        })
    }

    /// Compose one decoded RGB frame per §C.2 and return the composed
    /// image-sized frame. Also records `Reference[save_as_reference]`
    /// when requested (post-blend, per the spec's "blending is
    /// performed before recording the reference frame").
    pub fn compose(&mut self, decoded: &VideoFrame, meta: &FrameComposeMeta) -> Result<VideoFrame> {
        if decoded.planes.len() != 3 {
            return Err(Error::Unsupported(format!(
                "jxl frame composition: {}-plane frame — only 3-plane RGB composition is \
                 wired (extra-channel blending is a follow-up)",
                decoded.planes.len()
            )));
        }
        let fw = decoded.planes[0].stride;
        let fh = decoded.planes[0].data.len().checked_div(fw).unwrap_or(0);
        let (x0, y0) = (meta.x0 as usize, meta.y0 as usize);
        if x0 + fw > self.width || y0 + fh > self.height {
            return Err(Error::InvalidData(format!(
                "JXL frame composition: frame rect ({x0}, {y0})+({fw}×{fh}) exceeds image \
                 {}×{}",
                self.width, self.height
            )));
        }
        let source = meta.source as usize;
        if source >= self.reference.len() {
            return Err(Error::InvalidData(format!(
                "JXL frame composition: source {source} out of range"
            )));
        }

        // Base canvas = Reference[source] (zeros when unstored).
        let n = self.width * self.height;
        let mut planes: [Vec<u8>; 3] = match &self.reference[source] {
            Some(saved) => saved.clone(),
            None => [vec![0u8; n], vec![0u8; n], vec![0u8; n]],
        };

        // Blend the frame rectangle per Table C.8.
        for (c, plane) in planes.iter_mut().enumerate() {
            let new = &decoded.planes[c];
            for fy in 0..fh {
                let src_row = fy * fw;
                let dst_row = (y0 + fy) * self.width + x0;
                match meta.mode {
                    BlendMode::Replace => {
                        plane[dst_row..dst_row + fw]
                            .copy_from_slice(&new.data[src_row..src_row + fw]);
                    }
                    BlendMode::Add => {
                        for fx in 0..fw {
                            let old = plane[dst_row + fx] as f32 / 255.0;
                            let nv = new.data[src_row + fx] as f32 / 255.0;
                            plane[dst_row + fx] =
                                ((old + nv).clamp(0.0, 1.0) * 255.0).round() as u8;
                        }
                    }
                    BlendMode::Mul => {
                        for fx in 0..fw {
                            let old = plane[dst_row + fx] as f32 / 255.0;
                            let nv = new.data[src_row + fx] as f32 / 255.0;
                            plane[dst_row + fx] =
                                ((old * nv).clamp(0.0, 1.0) * 255.0).round() as u8;
                        }
                    }
                    BlendMode::Blend | BlendMode::AlphaWeightedAdd => {
                        return Err(Error::Unsupported(format!(
                            "jxl frame composition: blend mode {:?} needs the alpha extra \
                             channel threaded through the multi-frame walk — follow-up",
                            meta.mode
                        )));
                    }
                }
            }
        }

        // Record Reference[save_as_reference] post-blend.
        if meta.save_as_reference != 0 {
            if meta.save_before_ct {
                return Err(Error::Unsupported(
                    "jxl frame composition: save_before_ct reference recording (pre-colour-\
                     transform domain) is not wired — follow-up"
                        .into(),
                ));
            }
            let slot = meta.save_as_reference as usize;
            if slot >= self.reference.len() {
                return Err(Error::InvalidData(format!(
                    "JXL frame composition: save_as_reference {slot} out of range"
                )));
            }
            self.reference[slot] = Some(planes.clone());
        }

        let [r, g, b] = planes;
        Ok(VideoFrame {
            pts: decoded.pts,
            planes: vec![
                VideoPlane {
                    stride: self.width,
                    data: r,
                },
                VideoPlane {
                    stride: self.width,
                    data: g,
                },
                VideoPlane {
                    stride: self.width,
                    data: b,
                },
            ],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb_frame(w: usize, h: usize, r: u8, g: u8, b: u8) -> VideoFrame {
        VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: w,
                    data: vec![r; w * h],
                },
                VideoPlane {
                    stride: w,
                    data: vec![g; w * h],
                },
                VideoPlane {
                    stride: w,
                    data: vec![b; w * h],
                },
            ],
        }
    }

    fn meta_replace() -> FrameComposeMeta {
        FrameComposeMeta {
            x0: 0,
            y0: 0,
            mode: BlendMode::Replace,
            source: 0,
            save_as_reference: 0,
            save_before_ct: false,
            duration: 1,
            is_last: false,
        }
    }

    #[test]
    fn full_frame_replace_passes_through() {
        let mut st = ComposeState::new(4, 3).unwrap();
        let f = rgb_frame(4, 3, 10, 20, 30);
        let out = st.compose(&f, &meta_replace()).unwrap();
        assert_eq!(out.planes[0].data, vec![10u8; 12]);
        assert_eq!(out.planes[1].data, vec![20u8; 12]);
        assert_eq!(out.planes[2].data, vec![30u8; 12]);
    }

    #[test]
    fn cropped_replace_updates_rect_over_zero_source() {
        let mut st = ComposeState::new(4, 4).unwrap();
        let f = rgb_frame(2, 2, 200, 100, 50);
        let mut m = meta_replace();
        (m.x0, m.y0) = (1, 2);
        let out = st.compose(&f, &m).unwrap();
        // Outside the rect: the zero source frame.
        assert_eq!(out.planes[0].data[0], 0);
        // Inside: the new samples.
        assert_eq!(out.planes[0].data[2 * 4 + 1], 200);
        assert_eq!(out.planes[1].data[3 * 4 + 2], 100);
    }

    #[test]
    fn add_and_mul_follow_table_c8() {
        let mut st = ComposeState::new(1, 1).unwrap();
        // Store 100/255 into Reference[1] via an Add over zeros.
        let f1 = rgb_frame(1, 1, 100, 100, 100);
        let mut m1 = meta_replace();
        m1.save_as_reference = 1;
        st.compose(&f1, &m1).unwrap();

        // Add 50/255 on source 1 → 150.
        let f2 = rgb_frame(1, 1, 50, 50, 50);
        let m2 = FrameComposeMeta {
            mode: BlendMode::Add,
            source: 1,
            ..meta_replace()
        };
        let out = st.compose(&f2, &m2).unwrap();
        assert_eq!(out.planes[0].data[0], 150);

        // Mul: (100/255) × (128/255) × 255 ≈ 50.
        let f3 = rgb_frame(1, 1, 128, 128, 128);
        let m3 = FrameComposeMeta {
            mode: BlendMode::Mul,
            source: 1,
            ..meta_replace()
        };
        let out = st.compose(&f3, &m3).unwrap();
        assert_eq!(out.planes[0].data[0], 50);
    }

    #[test]
    fn add_saturates_at_255() {
        let mut st = ComposeState::new(1, 1).unwrap();
        let f1 = rgb_frame(1, 1, 200, 200, 200);
        let mut m1 = meta_replace();
        m1.save_as_reference = 2;
        st.compose(&f1, &m1).unwrap();
        let f2 = rgb_frame(1, 1, 100, 100, 100);
        let m2 = FrameComposeMeta {
            mode: BlendMode::Add,
            source: 2,
            ..meta_replace()
        };
        let out = st.compose(&f2, &m2).unwrap();
        assert_eq!(out.planes[0].data[0], 255);
    }

    #[test]
    fn unstored_source_reads_as_zeros() {
        let mut st = ComposeState::new(2, 1).unwrap();
        let f = rgb_frame(2, 1, 7, 8, 9);
        let m = FrameComposeMeta {
            mode: BlendMode::Add,
            source: 3,
            ..meta_replace()
        };
        let out = st.compose(&f, &m).unwrap();
        assert_eq!(out.planes[0].data, vec![7, 7]);
    }

    #[test]
    fn alpha_modes_and_pre_ct_saves_surface_precisely() {
        let mut st = ComposeState::new(1, 1).unwrap();
        let f = rgb_frame(1, 1, 1, 2, 3);
        let m = FrameComposeMeta {
            mode: BlendMode::Blend,
            ..meta_replace()
        };
        assert!(matches!(st.compose(&f, &m), Err(Error::Unsupported(_))));

        let m2 = FrameComposeMeta {
            save_as_reference: 1,
            save_before_ct: true,
            ..meta_replace()
        };
        assert!(matches!(st.compose(&f, &m2), Err(Error::Unsupported(_))));
    }

    #[test]
    fn out_of_bounds_rect_rejected() {
        let mut st = ComposeState::new(2, 2).unwrap();
        let f = rgb_frame(2, 2, 0, 0, 0);
        let mut m = meta_replace();
        m.x0 = 1;
        assert!(matches!(st.compose(&f, &m), Err(Error::InvalidData(_))));
    }
}
