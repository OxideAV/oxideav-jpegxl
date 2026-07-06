//! Frame composition + reference-frame semantics — ISO/IEC FDIS
//! 18181-1:2021 §C.2 (Table C.7 `BlendingInfo`, Table C.8 `BlendMode`,
//! the `save_as_reference` / `Reference[…]` prose, and the crop-frame
//! "updates the rectangle of the previous frame" rule).
//!
//! ## Scope (round 389, extended round 393)
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
//! Blend modes implemented per Table C.8: `kReplace`
//! (`sample = new_sample`), `kAdd` (`sample = old_sample +
//! new_sample`), `kMul` (`sample = old_sample × new_sample`), and —
//! round 393, for frames carrying an alpha plane — `kBlend` and
//! `kAlphaWeightedAdd`:
//!
//! * `kBlend`, premultiplied alpha (`alpha_associated == true`):
//!   `sample = new_sample + old_sample × (1 − new_alpha)`.
//! * `kBlend`, straight alpha: `sample = (new_alpha × new_sample +
//!   old_alpha × old_sample × (1 − new_alpha)) / alpha`, where `alpha`
//!   is the post-blend alpha below (0 when `alpha` is 0 — both
//!   numerator terms vanish with it).
//! * `kAlphaWeightedAdd`: `sample = old_sample + alpha × new_sample`.
//!   The §C.2 definitions paragraph binds the bare `alpha` to "the
//!   value after blending" (parallel to `sample`), i.e. the same
//!   post-blend alpha as `kBlend`'s denominator.
//! * The alpha channel itself always blends as
//!   `alpha = old_alpha + new_alpha × (1 − old_alpha)` under both
//!   modes.
//!
//! Table C.7's `clamp` requests alpha values be clamped to `[0, 1]`
//! before blending — a structural no-op here because the 8-bit planes
//! already map into `[0, 1]` exactly.
//!
//! Blending is specified on the post-inverse-CT samples ("the blending
//! is done in the colour space after inverse colour transforms from
//! Annex L have been applied"); this module operates on the decoded
//! 8-bit planes, mapping through `[0, 1]` floats for the arithmetic
//! modes and re-quantising with round-half-up — exact for `kReplace`,
//! and within one quantisation step for the arithmetic modes.

use oxideav_core::{Error, Result, VideoFrame, VideoPlane};

use crate::frame_header::BlendMode;

/// Composition inputs of one decoded frame — the §C.2 fields the
/// multi-frame walk extracts from the `FrameHeader` + `ImageMetadata`.
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
    /// Plane index of the alpha channel in the decoded frame (colour
    /// planes first, extra channels after — Annex G.1.3 order), when
    /// the frame carries one. `None` for pure-RGB frames.
    pub alpha_plane: Option<usize>,
    /// `ExtraChannelInfo.alpha_associated` for that alpha channel
    /// (premultiplied semantics — selects the Table C.8 `kBlend`
    /// branch).
    pub alpha_associated: bool,
    /// The alpha extra channel's own blend mode (Table C.7
    /// `ec_blending_info[alpha_channel].mode`).
    pub alpha_mode: BlendMode,
}

impl FrameComposeMeta {
    /// Compose metadata for a plain RGB frame (no alpha channel).
    /// Mirrors the Table C.7 field order — the §C.2 bundle simply has
    /// this many fields.
    #[allow(clippy::too_many_arguments)]
    pub fn rgb(
        x0: u32,
        y0: u32,
        mode: BlendMode,
        source: u32,
        save_as_reference: u32,
        save_before_ct: bool,
        duration: u32,
        is_last: bool,
    ) -> Self {
        Self {
            x0,
            y0,
            mode,
            source,
            save_as_reference,
            save_before_ct,
            duration,
            is_last,
            alpha_plane: None,
            alpha_associated: false,
            alpha_mode: BlendMode::Replace,
        }
    }
}

/// §C.2 composition state across a frame array: the image-sized canvas
/// dimensions and the `Reference[1..=3]` slots (slot 0 exists for
/// `source == 0` lookups but is never recorded — recording is gated on
/// `save_as_reference != 0`). Each slot stores the full plane stack
/// (colour planes + any extra channels) so alpha survives across the
/// blend chain.
#[derive(Debug)]
pub struct ComposeState {
    width: usize,
    height: usize,
    reference: [Option<Vec<Vec<u8>>>; 4],
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

    /// Compose one decoded frame per §C.2 and return the composed
    /// image-sized frame. Also records `Reference[save_as_reference]`
    /// when requested (post-blend, per the spec's "blending is
    /// performed before recording the reference frame").
    ///
    /// The frame may be 3-plane RGB or carry extra channels; the
    /// alpha-consuming modes (`kBlend` / `kAlphaWeightedAdd`) need
    /// `meta.alpha_plane` to point at the alpha plane and reject
    /// alpha-less frames precisely.
    pub fn compose(&mut self, decoded: &VideoFrame, meta: &FrameComposeMeta) -> Result<VideoFrame> {
        let n_planes = decoded.planes.len();
        if n_planes < 3 {
            return Err(Error::Unsupported(format!(
                "jxl frame composition: {n_planes}-plane frame — need at least the three \
                 colour planes"
            )));
        }
        if let Some(a) = meta.alpha_plane {
            if a < 3 || a >= n_planes {
                return Err(Error::InvalidData(format!(
                    "JXL frame composition: alpha plane index {a} out of range for a \
                     {n_planes}-plane frame"
                )));
            }
        }
        let needs_alpha = matches!(meta.mode, BlendMode::Blend | BlendMode::AlphaWeightedAdd);
        if needs_alpha && meta.alpha_plane.is_none() {
            return Err(Error::Unsupported(format!(
                "jxl frame composition: blend mode {:?} needs an alpha extra channel but \
                 the frame carries none",
                meta.mode
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

        // Base canvas = Reference[source] (zeros when unstored). Zero-
        // pad the plane stack when the stored reference has fewer
        // planes than the current frame (e.g. an RGB frame stored
        // before an RGBA one); a stored alpha with no current-frame
        // counterpart keeps its stored samples outside the rect.
        let n = self.width * self.height;
        let mut planes: Vec<Vec<u8>> = match &self.reference[source] {
            Some(saved) => {
                let mut p = saved.clone();
                while p.len() < n_planes {
                    p.push(vec![0u8; n]);
                }
                p
            }
            None => (0..n_planes).map(|_| vec![0u8; n]).collect(),
        };

        // Pre-blend alpha snapshot: the colour-channel formulas read
        // old_alpha from the SOURCE frame, so it must be captured
        // before the alpha plane itself is blended.
        let old_alpha_plane = meta.alpha_plane.map(|a| planes[a].clone());

        // Blend the colour channels over the frame rectangle per
        // Table C.8.
        for (c, plane) in planes.iter_mut().enumerate().take(3) {
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
                        // Presence checked above.
                        let a_idx = meta.alpha_plane.expect("alpha presence gated above");
                        let new_a = &decoded.planes[a_idx];
                        let old_a = old_alpha_plane
                            .as_ref()
                            .expect("old alpha captured when alpha_plane is set");
                        for fx in 0..fw {
                            let old = plane[dst_row + fx] as f32 / 255.0;
                            let nv = new.data[src_row + fx] as f32 / 255.0;
                            // Table C.7 `clamp` asks for a [0, 1] clamp
                            // on alpha before blending; 8-bit planes are
                            // already in range, so the clamp is exact
                            // here either way.
                            let new_alpha = new_a.data[src_row + fx] as f32 / 255.0;
                            let old_alpha = old_a[dst_row + fx] as f32 / 255.0;
                            // Post-blend alpha (Table C.8, both modes):
                            let alpha = old_alpha + new_alpha * (1.0 - old_alpha);
                            let out = match meta.mode {
                                BlendMode::Blend => {
                                    if meta.alpha_associated {
                                        // Premultiplied semantics.
                                        nv + old * (1.0 - new_alpha)
                                    } else if alpha == 0.0 {
                                        // Straight alpha; both numerator
                                        // terms vanish with alpha == 0.
                                        0.0
                                    } else {
                                        (new_alpha * nv + old_alpha * old * (1.0 - new_alpha))
                                            / alpha
                                    }
                                }
                                _ => old + alpha * nv,
                            };
                            plane[dst_row + fx] = (out.clamp(0.0, 1.0) * 255.0).round() as u8;
                        }
                    }
                }
            }
        }

        // Blend the extra-channel planes. The alpha plane follows the
        // Table C.8 "the blending on the alpha channel itself" formula
        // when its ec mode is kBlend / kAlphaWeightedAdd; the standard
        // formulas otherwise. Non-alpha extra channels only support
        // kReplace for now (no staged fixture carries a second extra
        // channel through a blend chain).
        for (c, plane) in planes.iter_mut().enumerate().skip(3) {
            let is_alpha = meta.alpha_plane == Some(c);
            let mode = if is_alpha {
                meta.alpha_mode
            } else {
                BlendMode::Replace
            };
            let new = &decoded.planes[c];
            for fy in 0..fh {
                let src_row = fy * fw;
                let dst_row = (y0 + fy) * self.width + x0;
                match mode {
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
                        for fx in 0..fw {
                            let old_alpha = plane[dst_row + fx] as f32 / 255.0;
                            let new_alpha = new.data[src_row + fx] as f32 / 255.0;
                            let alpha = old_alpha + new_alpha * (1.0 - old_alpha);
                            plane[dst_row + fx] = (alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
                        }
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

        Ok(VideoFrame {
            pts: decoded.pts,
            planes: planes
                .into_iter()
                .map(|data| VideoPlane {
                    stride: self.width,
                    data,
                })
                .collect(),
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

    fn rgba_frame(w: usize, h: usize, r: u8, g: u8, b: u8, a: u8) -> VideoFrame {
        let mut f = rgb_frame(w, h, r, g, b);
        f.planes.push(VideoPlane {
            stride: w,
            data: vec![a; w * h],
        });
        f
    }

    fn meta_replace() -> FrameComposeMeta {
        FrameComposeMeta::rgb(0, 0, BlendMode::Replace, 0, 0, false, 1, false)
    }

    fn meta_alpha(mode: BlendMode, source: u32) -> FrameComposeMeta {
        FrameComposeMeta {
            mode,
            source,
            alpha_plane: Some(3),
            alpha_mode: BlendMode::Blend,
            ..meta_replace()
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
    fn alpha_less_kblend_and_pre_ct_saves_surface_precisely() {
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

    /// kBlend, straight (non-premultiplied) alpha over an opaque
    /// source: sample = (na·new + oa·old·(1−na)) / (oa + na·(1−oa)).
    /// With old = 100/255 @ alpha 1.0, new = 200/255 @ alpha 0.5:
    /// alpha = 1.0; sample = 0.5·(200/255) + 1.0·(100/255)·0.5 = 150/255.
    #[test]
    fn kblend_straight_alpha_over_opaque_source() {
        let mut st = ComposeState::new(1, 1).unwrap();
        let f1 = rgba_frame(1, 1, 100, 100, 100, 255);
        let mut m1 = meta_alpha(BlendMode::Replace, 0);
        m1.save_as_reference = 1;
        st.compose(&f1, &m1).unwrap();

        let f2 = rgba_frame(1, 1, 200, 200, 200, 128);
        let m2 = meta_alpha(BlendMode::Blend, 1);
        let out = st.compose(&f2, &m2).unwrap();
        // 128/255 ≈ 0.50196; sample ≈ 0.50196·(200/255) + (100/255)·0.49804
        // ≈ 0.58900 → 150.2 → 150.
        assert_eq!(out.planes[0].data[0], 150);
        // Alpha plane: 1 + 0.502·(1−1) = 1 → stays opaque.
        assert_eq!(out.planes[3].data[0], 255);
    }

    /// kBlend with premultiplied semantics:
    /// sample = new + old·(1 − new_alpha).
    #[test]
    fn kblend_premultiplied_alpha() {
        let mut st = ComposeState::new(1, 1).unwrap();
        let f1 = rgba_frame(1, 1, 100, 100, 100, 255);
        let mut m1 = meta_alpha(BlendMode::Replace, 0);
        m1.save_as_reference = 1;
        st.compose(&f1, &m1).unwrap();

        let f2 = rgba_frame(1, 1, 50, 50, 50, 128);
        let mut m2 = meta_alpha(BlendMode::Blend, 1);
        m2.alpha_associated = true;
        let out = st.compose(&f2, &m2).unwrap();
        // 50/255 + (100/255)·(1 − 128/255) = (50 + 100·0.49804)/255
        // ≈ 99.8/255 → 100.
        assert_eq!(out.planes[0].data[0], 100);
    }

    /// kBlend over an UNSTORED source: old_alpha = 0 → alpha = new_alpha
    /// and sample = new_sample (straight-alpha formula reduces exactly).
    #[test]
    fn kblend_over_zero_source_passes_new_samples() {
        let mut st = ComposeState::new(1, 1).unwrap();
        let f = rgba_frame(1, 1, 77, 88, 99, 128);
        let m = meta_alpha(BlendMode::Blend, 0);
        let out = st.compose(&f, &m).unwrap();
        assert_eq!(out.planes[0].data[0], 77);
        assert_eq!(out.planes[1].data[0], 88);
        assert_eq!(out.planes[2].data[0], 99);
        // Alpha: 0 + 0.502·(1 − 0) = new_alpha.
        assert_eq!(out.planes[3].data[0], 128);
    }

    /// kBlend with a fully transparent new frame leaves the source
    /// untouched (straight alpha: sample = old, alpha = old_alpha).
    #[test]
    fn kblend_transparent_frame_is_identity() {
        let mut st = ComposeState::new(1, 1).unwrap();
        let f1 = rgba_frame(1, 1, 60, 70, 80, 255);
        let mut m1 = meta_alpha(BlendMode::Replace, 0);
        m1.save_as_reference = 1;
        st.compose(&f1, &m1).unwrap();

        let f2 = rgba_frame(1, 1, 200, 210, 220, 0);
        let m2 = meta_alpha(BlendMode::Blend, 1);
        let out = st.compose(&f2, &m2).unwrap();
        assert_eq!(out.planes[0].data[0], 60);
        assert_eq!(out.planes[3].data[0], 255);
    }

    /// kAlphaWeightedAdd: sample = old + alpha·new with the post-blend
    /// alpha.
    #[test]
    fn kalpha_weighted_add_uses_post_blend_alpha() {
        let mut st = ComposeState::new(1, 1).unwrap();
        let f1 = rgba_frame(1, 1, 100, 100, 100, 0);
        let mut m1 = meta_alpha(BlendMode::Replace, 0);
        m1.save_as_reference = 1;
        st.compose(&f1, &m1).unwrap();

        let f2 = rgba_frame(1, 1, 100, 100, 100, 128);
        let m2 = meta_alpha(BlendMode::AlphaWeightedAdd, 1);
        let out = st.compose(&f2, &m2).unwrap();
        // old_alpha = 0 → alpha = new_alpha ≈ 0.502;
        // sample = 100/255 + 0.502·(100/255) ≈ 150.2/255 → 150.
        assert_eq!(out.planes[0].data[0], 150);
        // Alpha plane blends to new_alpha.
        assert_eq!(out.planes[3].data[0], 128);
    }

    /// An RGB reference stored before an RGBA frame zero-pads its alpha
    /// (old_alpha = 0 inside the blend).
    #[test]
    fn rgb_reference_zero_pads_alpha_for_rgba_blend() {
        let mut st = ComposeState::new(1, 1).unwrap();
        let f1 = rgb_frame(1, 1, 100, 100, 100);
        let mut m1 = meta_replace();
        m1.save_as_reference = 1;
        st.compose(&f1, &m1).unwrap();

        let f2 = rgba_frame(1, 1, 30, 30, 30, 255);
        let m2 = meta_alpha(BlendMode::Blend, 1);
        let out = st.compose(&f2, &m2).unwrap();
        // Opaque new frame over anything = new samples.
        assert_eq!(out.planes[0].data[0], 30);
        assert_eq!(out.planes.len(), 4);
    }

    /// Cropped kBlend only touches the rectangle; the source's alpha
    /// outside the rect is preserved.
    #[test]
    fn cropped_kblend_preserves_outside_rect() {
        let mut st = ComposeState::new(2, 1).unwrap();
        let f1 = rgba_frame(2, 1, 10, 10, 10, 200);
        let mut m1 = meta_alpha(BlendMode::Replace, 0);
        m1.save_as_reference = 1;
        st.compose(&f1, &m1).unwrap();

        let f2 = rgba_frame(1, 1, 250, 250, 250, 255);
        let mut m2 = meta_alpha(BlendMode::Blend, 1);
        m2.x0 = 1;
        let out = st.compose(&f2, &m2).unwrap();
        assert_eq!(out.planes[0].data[0], 10, "outside the rect: source");
        assert_eq!(out.planes[0].data[1], 250, "inside: blended (opaque)");
        assert_eq!(out.planes[3].data[0], 200);
        assert_eq!(out.planes[3].data[1], 255);
    }
}
