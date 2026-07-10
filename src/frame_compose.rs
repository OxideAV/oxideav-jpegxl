//! Frame composition + reference-frame semantics — ISO/IEC FDIS
//! 18181-1:2021 §C.2 (Table C.7 `BlendingInfo`, Table C.8 `BlendMode`,
//! the `save_as_reference` / `Reference[…]` prose, and the crop-frame
//! "updates the rectangle of the previous frame" rule).
//!
//! ## Scope (round 389, extended rounds 393 / 406)
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
//! * `kAlphaWeightedAdd`: `sample = old_sample + new_alpha ×
//!   new_sample`, and the alpha channel itself is left **unchanged**.
//!   The FDIS prints `old + alpha × new` with the §C.2 definitions
//!   paragraph's post-blend alpha and a kBlend-style alpha rule — both
//!   contradicted field-exactly by the 18181-3 `blendmodes` stream
//!   (round 406, fixture-measured; the post-blend reading errs by
//!   hundreds of codes and the reference's composed alpha equals the
//!   pre-frame value everywhere including under `new_alpha = 1`).
//! * Under `kBlend` the alpha channel itself blends as
//!   `alpha = old_alpha + new_alpha × (1 − old_alpha)`.
//!
//! Table C.7's `clamp` requests alpha values be clamped to `[0, 1]`
//! before blending (`kBlend` / `kAlphaWeightedAdd`); the FDIS does not
//! define what `clamp` means for `kMul`, so that combination surfaces
//! a precise [`Error::Unsupported`] (no staged stream sets it).
//!
//! ## Float-domain, unclamped intermediate state (round 406)
//!
//! Blending is specified on the post-inverse-CT samples ("the blending
//! is done in the colour space after inverse colour transforms from
//! Annex L have been applied") — a *float* domain with nominal range
//! `[0, 1]` but **no clamping between composition steps**: Table C.8
//! defines pure arithmetic, and ISO/IEC 18181-3 §A.3 clamps only at
//! comparison time. This matters in practice: the `blendmodes`
//! conformance stream chains `kBlend → kAdd → kMul → kAlphaWeightedAdd`
//! over `Reference[1]`, and the `kAdd` stage drives samples above 1.0
//! that the later stages *bring back into range* — clamping after each
//! stage (what rounds ≤ 405 did on the 8-bit integer canvas) destroys
//! those samples irrecoverably. The composition state therefore keeps
//! every canvas and `Reference[…]` slot as **unclamped `f32` planes**
//! (normalised so 1.0 = `2^bits − 1`); the decoded integer frames are
//! lifted to float on entry, and quantisation (clamp to `[0, 1]`,
//! scale, round-half-up) happens only when a composed frame is
//! materialised for presentation.
//!
//! ## Crop rectangles outside the image (round 406)
//!
//! The FDIS prose asserts `x0 + width <= size.width` — but the
//! ISO/IEC 18181-3 conformance corpus (`sunset_logo`) carries a crop
//! frame whose *signed* offsets place it partially outside the image:
//! `x0` decodes through `UnpackSigned` to −662 with −662 + 2048 = 1386
//! (exactly the image width, and likewise for y), so the frame in fact
//! *covers* the image while extending past its top-left corner. The
//! §3.5.1 grid definition explicitly permits addressing outside the
//! bounding rectangle. Composition therefore **clips** the frame rect
//! to the canvas: samples outside the image area are discarded, never
//! an error.

use oxideav_core::{Error, Result, VideoFrame, VideoPlane};

use crate::frame_header::BlendMode;

/// Composition inputs of one decoded frame — the §C.2 fields the
/// multi-frame walk extracts from the `FrameHeader` + `ImageMetadata`.
#[derive(Debug, Clone, Copy)]
pub struct FrameComposeMeta {
    /// Crop rectangle top-left (0, 0 when `!have_crop`). Signed:
    /// `UnpackSigned` offsets may place the frame partially outside
    /// the image (the out-of-canvas samples are clipped away).
    pub x0: i32,
    pub y0: i32,
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
    /// Table C.7 `clamp` of the colour-channel `blending_info`: clamp
    /// the alpha values feeding the colour blend to `[0, 1]`.
    pub clamp: bool,
    /// Table C.7 `clamp` of the alpha channel's own
    /// `ec_blending_info` entry.
    pub alpha_clamp: bool,
}

impl FrameComposeMeta {
    /// Compose metadata for a plain RGB frame (no alpha channel).
    /// Mirrors the Table C.7 field order — the §C.2 bundle simply has
    /// this many fields.
    #[allow(clippy::too_many_arguments)]
    pub fn rgb(
        x0: i32,
        y0: i32,
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
            clamp: false,
            alpha_clamp: false,
        }
    }
}

/// §C.2 composition state across a frame array: the image-sized canvas
/// dimensions and the `Reference[1..=3]` slots (slot 0 exists for
/// `source == 0` lookups but is never recorded — recording is gated on
/// `save_as_reference != 0`). Each slot stores the full plane stack
/// (colour planes + any extra channels) as **unclamped normalised
/// `f32`** so alpha and out-of-range intermediates survive across the
/// blend chain (see module docs).
#[derive(Debug)]
pub struct ComposeState {
    width: usize,
    height: usize,
    /// Image `bits_per_sample` — selects the integer plane byte layout
    /// (1 byte per sample through 8 bits, little-endian 2 bytes for
    /// 9–16) and the `2^bits − 1` float normalisation denominator.
    bits_per_sample: u32,
    reference: [Option<Vec<Vec<f32>>>; 4],
}

impl ComposeState {
    /// New state for a `width × height` 8-bit image. Unstored reference
    /// slots read as all-zero source frames per §C.2.
    pub fn new(width: usize, height: usize) -> Result<Self> {
        Self::new_with_depth(width, height, 8)
    }

    /// New state for a `width × height` image whose planes carry
    /// `bits_per_sample`-bit integer samples (1–16; > 8 selects the
    /// little-endian 2-byte plane layout).
    pub fn new_with_depth(width: usize, height: usize, bits_per_sample: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidData(
                "JXL frame composition: zero-dimension image".into(),
            ));
        }
        if bits_per_sample == 0 || bits_per_sample > 16 {
            return Err(Error::Unsupported(format!(
                "jxl frame composition: bits_per_sample {bits_per_sample} outside the \
                 1–16 integer envelope"
            )));
        }
        Ok(Self {
            width,
            height,
            bits_per_sample,
            reference: [None, None, None, None],
        })
    }

    /// Bytes per sample of the integer plane layout (1 or 2).
    fn bytes_per_sample(&self) -> usize {
        if self.bits_per_sample > 8 {
            2
        } else {
            1
        }
    }

    /// Compose one decoded frame per §C.2 and return the composed
    /// image-sized frame (quantised back to the crate's integer plane
    /// layout). Also records `Reference[save_as_reference]` when
    /// requested (post-blend, per the spec's "blending is performed
    /// before recording the reference frame").
    ///
    /// The frame may be 3-plane RGB or carry extra channels; the
    /// alpha-consuming modes (`kBlend` / `kAlphaWeightedAdd`) need
    /// `meta.alpha_plane` to point at the alpha plane and reject
    /// alpha-less frames precisely.
    pub fn compose(&mut self, decoded: &VideoFrame, meta: &FrameComposeMeta) -> Result<VideoFrame> {
        let bytes = self.bytes_per_sample();
        let two = bytes == 2;
        let max = ((1u32 << self.bits_per_sample) - 1) as f32;
        let stride = decoded.planes[0].stride;
        if stride == 0 || stride % bytes != 0 {
            return Err(Error::InvalidData(format!(
                "JXL frame composition: plane stride {stride} is not a multiple of the \
                 {bytes}-byte sample layout"
            )));
        }
        let fw = stride / bytes;
        let fh = decoded.planes[0].data.len() / stride;
        for (c, p) in decoded.planes.iter().enumerate() {
            if p.stride != stride || p.data.len() < stride * fh {
                return Err(Error::InvalidData(format!(
                    "JXL frame composition: plane {c} geometry ({} × {} bytes) disagrees \
                     with plane 0 ({stride} × {fh})",
                    p.stride,
                    p.data.len()
                )));
            }
        }
        // Lift the decoded integer planes into the normalised float
        // domain the blending arithmetic runs in.
        let inv_max = 1.0f32 / max;
        let newf: Vec<Vec<f32>> = decoded
            .planes
            .iter()
            .map(|p| {
                (0..fw * fh)
                    .map(|i| get_sample(&p.data, i, two) as f32 * inv_max)
                    .collect()
            })
            .collect();
        let mut out = self.compose_planes(&newf, fw, fh, meta)?;
        out.pts = decoded.pts;
        Ok(out)
    }

    /// Float-domain core of [`Self::compose`]: `newf` holds the
    /// frame's planes as normalised (1.0 = `2^bits − 1`) **unclamped**
    /// `f32` samples, `fw × fh` each. The unclamped entry point matters
    /// beyond precision: a Modular frame's decoded samples may lie
    /// outside `[0, 2^bits − 1]` (negative or above-range residual
    /// space is legal), and those out-of-range values are meaningful
    /// inputs to the Table C.8 arithmetic — the `blendmodes`
    /// conformance stream's base frame carries a *negative* red plane
    /// that later `kBlend` / `kAlphaWeightedAdd` stages weight back
    /// into range. Clamping the frame before composition (what the
    /// integer [`Self::compose`] wrapper necessarily does) is only
    /// correct for frames whose samples are in range.
    pub fn compose_planes(
        &mut self,
        newf: &[Vec<f32>],
        fw: usize,
        fh: usize,
        meta: &FrameComposeMeta,
    ) -> Result<VideoFrame> {
        let n_planes = newf.len();
        if n_planes < 3 {
            return Err(Error::Unsupported(format!(
                "jxl frame composition: {n_planes}-plane frame — need at least the three \
                 colour planes"
            )));
        }
        for (c, p) in newf.iter().enumerate() {
            if p.len() < fw * fh {
                return Err(Error::InvalidData(format!(
                    "JXL frame composition: plane {c} has {} samples, expected {fw}×{fh}",
                    p.len()
                )));
            }
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
        if meta.mode == BlendMode::Mul && meta.clamp {
            return Err(Error::Unsupported(
                "jxl frame composition: clamp semantics for kMul are not defined by the \
                 FDIS text and no staged stream exercises them"
                    .into(),
            ));
        }
        let bytes = self.bytes_per_sample();
        let two = bytes == 2;
        let max = ((1u32 << self.bits_per_sample) - 1) as f32;
        // §C.2 + §3.5.1: the (signed) frame rect may extend beyond the
        // canvas on any side; clip to the intersection and discard the
        // out-of-image samples. An empty intersection composes to the
        // bare source frame.
        let (x0, y0) = (meta.x0 as i64, meta.y0 as i64);
        let dx0 = x0.max(0);
        let dy0 = y0.max(0);
        let dx1 = (x0 + fw as i64).min(self.width as i64);
        let dy1 = (y0 + fh as i64).min(self.height as i64);
        let (cw, ch) = ((dx1 - dx0).max(0) as usize, (dy1 - dy0).max(0) as usize);
        // Empty intersection: the whole frame is off-canvas; nothing
        // blends.
        let ch = if cw == 0 { 0 } else { ch };
        // Frame-local coordinates of the clipped rect's origin
        // (`dx0 >= x0` by construction, so the differences are >= 0).
        let sx0 = (dx0 - x0) as usize;
        let sy0 = (dy0 - y0) as usize;
        let (dx0, dy0) = (dx0 as usize, dy0 as usize);
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
        let mut planes: Vec<Vec<f32>> = match &self.reference[source] {
            Some(saved) => {
                let mut p = saved.clone();
                while p.len() < n_planes {
                    p.push(vec![0.0f32; n]);
                }
                p
            }
            None => (0..n_planes).map(|_| vec![0.0f32; n]).collect(),
        };

        // Pre-blend alpha snapshot: the colour-channel formulas read
        // old_alpha from the SOURCE frame, so it must be captured
        // before the alpha plane itself is blended.
        let old_alpha_plane = meta.alpha_plane.map(|a| planes[a].clone());

        // Table C.7 clamp: alpha values feeding the blend are clamped
        // to [0, 1] first.
        let clamp01 = |v: f32, on: bool| if on { v.clamp(0.0, 1.0) } else { v };

        // Blend the colour channels over the frame rectangle per
        // Table C.8 — pure float arithmetic, NO clamping of results
        // (out-of-range intermediates are meaningful; see module docs).
        for (c, plane) in planes.iter_mut().enumerate().take(3) {
            let new = &newf[c];
            for fy in 0..ch {
                let src_row = (sy0 + fy) * fw + sx0;
                let dst_row = (dy0 + fy) * self.width + dx0;
                match meta.mode {
                    BlendMode::Replace => {
                        plane[dst_row..dst_row + cw].copy_from_slice(&new[src_row..src_row + cw]);
                    }
                    BlendMode::Add => {
                        for fx in 0..cw {
                            plane[dst_row + fx] += new[src_row + fx];
                        }
                    }
                    BlendMode::Mul => {
                        for fx in 0..cw {
                            plane[dst_row + fx] *= new[src_row + fx];
                        }
                    }
                    BlendMode::Blend | BlendMode::AlphaWeightedAdd => {
                        // Presence checked above.
                        let a_idx = meta.alpha_plane.expect("alpha presence gated above");
                        let new_a = &newf[a_idx];
                        let old_a = old_alpha_plane
                            .as_ref()
                            .expect("old alpha captured when alpha_plane is set");
                        for fx in 0..cw {
                            let old = plane[dst_row + fx];
                            let nv = new[src_row + fx];
                            let new_alpha = clamp01(new_a[src_row + fx], meta.clamp);
                            let old_alpha = clamp01(old_a[dst_row + fx], meta.clamp);
                            plane[dst_row + fx] = match meta.mode {
                                BlendMode::Blend => {
                                    // Post-blend alpha (Table C.8):
                                    let alpha = old_alpha + new_alpha * (1.0 - old_alpha);
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
                                // kAlphaWeightedAdd: the FDIS prints
                                // `sample = old + alpha × new` with the
                                // §C.2 definitions paragraph's post-blend
                                // alpha — but the `blendmodes` conformance
                                // stream pins the published semantics as
                                // the frame's own NEW alpha weighting the
                                // addition (fixture-measured, exact to
                                // ±1/4095 field-wide; the post-blend
                                // reading errs by hundreds of codes).
                                _ => old + new_alpha * nv,
                            };
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
            let new = &newf[c];
            for fy in 0..ch {
                let src_row = (sy0 + fy) * fw + sx0;
                let dst_row = (dy0 + fy) * self.width + dx0;
                match mode {
                    BlendMode::Replace => {
                        plane[dst_row..dst_row + cw].copy_from_slice(&new[src_row..src_row + cw]);
                    }
                    BlendMode::Add => {
                        for fx in 0..cw {
                            plane[dst_row + fx] += new[src_row + fx];
                        }
                    }
                    BlendMode::Mul => {
                        for fx in 0..cw {
                            plane[dst_row + fx] *= new[src_row + fx];
                        }
                    }
                    BlendMode::Blend => {
                        for fx in 0..cw {
                            let old_alpha = clamp01(plane[dst_row + fx], meta.alpha_clamp);
                            let new_alpha = clamp01(new[src_row + fx], meta.alpha_clamp);
                            plane[dst_row + fx] = old_alpha + new_alpha * (1.0 - old_alpha);
                        }
                    }
                    // kAlphaWeightedAdd leaves the alpha channel itself
                    // UNCHANGED — the FDIS prints the same
                    // `old + new×(1−old)` rule as kBlend, but the
                    // `blendmodes` conformance stream pins the published
                    // semantics: its final AWA frame carries new_alpha
                    // = 1.0 where the composed alpha stays at the
                    // pre-frame value (exact keep-old on every pixel).
                    BlendMode::AlphaWeightedAdd => {}
                }
            }
        }

        // Record Reference[save_as_reference] post-blend (unclamped
        // float domain — later frames may bring out-of-range samples
        // back into range).
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

        // Materialise the presented frame: clamp to the nominal range
        // and quantise to the crate's integer plane layout.
        Ok(VideoFrame {
            pts: None,
            planes: planes
                .into_iter()
                .map(|data| {
                    let mut out = vec![0u8; data.len() * bytes];
                    for (i, v) in data.into_iter().enumerate() {
                        let q = (v.clamp(0.0, 1.0) * max).round() as u32;
                        put_sample(&mut out, i, two, q);
                    }
                    VideoPlane {
                        stride: self.width * bytes,
                        data: out,
                    }
                })
                .collect(),
        })
    }
}

/// Read sample `idx` of a plane in the crate-wide byte layout (1 byte
/// per sample, or little-endian 2 bytes when `two_byte`).
#[inline]
fn get_sample(data: &[u8], idx: usize, two_byte: bool) -> u32 {
    if two_byte {
        u16::from_le_bytes([data[2 * idx], data[2 * idx + 1]]) as u32
    } else {
        data[idx] as u32
    }
}

/// Write sample `idx` of a plane in the crate-wide byte layout.
#[inline]
fn put_sample(data: &mut [u8], idx: usize, two_byte: bool, v: u32) {
    if two_byte {
        data[2 * idx..2 * idx + 2].copy_from_slice(&(v as u16).to_le_bytes());
    } else {
        data[idx] = v as u8;
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
    fn add_saturates_at_255_on_presentation() {
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

    /// The intermediate composition state is NOT clamped: an over-range
    /// kAdd result multiplied back below 1.0 by a later kMul frame must
    /// reflect the true (unclamped) product — pinned by the
    /// `blendmodes` conformance stream (see module docs).
    #[test]
    fn intermediate_over_range_survives_across_frames() {
        let mut st = ComposeState::new(1, 1).unwrap();
        // Reference[1] = 200/255 + 100/255 ≈ 1.176 (over range).
        let f1 = rgb_frame(1, 1, 200, 200, 200);
        let mut m1 = meta_replace();
        m1.save_as_reference = 1;
        st.compose(&f1, &m1).unwrap();
        let f2 = rgb_frame(1, 1, 100, 100, 100);
        let m2 = FrameComposeMeta {
            mode: BlendMode::Add,
            source: 1,
            save_as_reference: 1,
            ..meta_replace()
        };
        let out2 = st.compose(&f2, &m2).unwrap();
        assert_eq!(out2.planes[0].data[0], 255, "presented frame clamps");

        // kMul by 128/255 over Reference[1]: 1.176 × 0.502 ≈ 0.5906 →
        // 151 — NOT the 128 a clamped intermediate (1.0 × 0.502) gives.
        let f3 = rgb_frame(1, 1, 128, 128, 128);
        let m3 = FrameComposeMeta {
            mode: BlendMode::Mul,
            source: 1,
            ..meta_replace()
        };
        let out3 = st.compose(&f3, &m3).unwrap();
        assert_eq!(out3.planes[0].data[0], 151);
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

    /// §C.2 + §3.5.1: frame rects extending beyond the canvas are
    /// clipped — the in-canvas intersection composes, the outside
    /// samples are discarded (pinned by the `sunset_logo` conformance
    /// stream, whose signed crop origin is (−662, −100)).
    #[test]
    fn out_of_canvas_rect_is_clipped() {
        // Overhang on the right: only column 0 of the frame lands.
        let mut st = ComposeState::new(2, 2).unwrap();
        let f = rgb_frame(2, 2, 9, 9, 9);
        let mut m = meta_replace();
        m.x0 = 1;
        let out = st.compose(&f, &m).unwrap();
        assert_eq!(out.planes[0].data, vec![0, 9, 0, 9]);

        // Negative origin covering the whole canvas: the frame's
        // bottom-right quadrant is what lands on the image.
        let mut st2 = ComposeState::new(2, 2).unwrap();
        let mut f2 = rgb_frame(4, 4, 0, 0, 0);
        // Distinct quadrants; bottom-right = 5.
        for y in 2..4 {
            for x in 2..4 {
                f2.planes[0].data[y * 4 + x] = 5;
            }
        }
        let mut m2 = meta_replace();
        m2.x0 = -2;
        m2.y0 = -2;
        let out2 = st2.compose(&f2, &m2).unwrap();
        assert_eq!(out2.planes[0].data, vec![5; 4]);

        // Fully off-canvas: composes to the bare (zero) source frame.
        let mut st3 = ComposeState::new(2, 2).unwrap();
        let f3 = rgb_frame(2, 2, 7, 7, 7);
        let mut m3 = meta_replace();
        m3.x0 = -5;
        let out3 = st3.compose(&f3, &m3).unwrap();
        assert_eq!(out3.planes[0].data, vec![0; 4]);
    }

    /// 12-bit planes: little-endian 2-byte samples, `2^12 − 1`
    /// normalisation (the alpha conformance streams' layout).
    #[test]
    fn twelve_bit_planes_blend_with_4095_denominator() {
        fn plane16(w: usize, h: usize, v: u16) -> VideoPlane {
            let mut data = Vec::with_capacity(w * h * 2);
            for _ in 0..w * h {
                data.extend_from_slice(&v.to_le_bytes());
            }
            VideoPlane {
                stride: w * 2,
                data,
            }
        }
        let mut st = ComposeState::new_with_depth(1, 1, 12).unwrap();
        let f1 = VideoFrame {
            pts: None,
            planes: (0..3).map(|_| plane16(1, 1, 3000)).collect(),
        };
        let mut m1 = meta_replace();
        m1.save_as_reference = 1;
        let out = st.compose(&f1, &m1).unwrap();
        assert_eq!(out.planes[0].stride, 2);
        assert_eq!(
            u16::from_le_bytes([out.planes[0].data[0], out.planes[0].data[1]]),
            3000
        );

        // kAdd saturates at 4095 on presentation, not 65535 or 255.
        let f2 = VideoFrame {
            pts: None,
            planes: (0..3).map(|_| plane16(1, 1, 2000)).collect(),
        };
        let m2 = FrameComposeMeta {
            mode: BlendMode::Add,
            source: 1,
            ..meta_replace()
        };
        let out2 = st.compose(&f2, &m2).unwrap();
        assert_eq!(
            u16::from_le_bytes([out2.planes[0].data[0], out2.planes[0].data[1]]),
            4095
        );

        // kBlend straight alpha at 12 bits: opaque new frame wins.
        let mut f3 = VideoFrame {
            pts: None,
            planes: (0..3).map(|_| plane16(1, 1, 1234)).collect(),
        };
        f3.planes.push(plane16(1, 1, 4095));
        let m3 = FrameComposeMeta {
            mode: BlendMode::Blend,
            source: 1,
            alpha_plane: Some(3),
            alpha_mode: BlendMode::Blend,
            ..meta_replace()
        };
        let out3 = st.compose(&f3, &m3).unwrap();
        assert_eq!(
            u16::from_le_bytes([out3.planes[0].data[0], out3.planes[0].data[1]]),
            1234
        );
        assert_eq!(
            u16::from_le_bytes([out3.planes[3].data[0], out3.planes[3].data[1]]),
            4095
        );
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

    /// kAlphaWeightedAdd: sample = old + new_alpha·new, and the alpha
    /// channel itself keeps its old value (both fixture-measured on
    /// the `blendmodes` conformance stream — the FDIS prints a
    /// post-blend-alpha weight and a kBlend-style alpha rule, but the
    /// published semantics differ; see the compose_planes comments).
    #[test]
    fn kalpha_weighted_add_weights_by_new_alpha_and_keeps_old_alpha() {
        let mut st = ComposeState::new(1, 1).unwrap();
        let f1 = rgba_frame(1, 1, 100, 100, 100, 64);
        let mut m1 = meta_alpha(BlendMode::Replace, 0);
        m1.save_as_reference = 1;
        st.compose(&f1, &m1).unwrap();

        let f2 = rgba_frame(1, 1, 100, 100, 100, 128);
        let mut m2 = meta_alpha(BlendMode::AlphaWeightedAdd, 1);
        m2.alpha_mode = BlendMode::AlphaWeightedAdd;
        let out = st.compose(&f2, &m2).unwrap();
        // sample = 100/255 + (128/255)·(100/255) ≈ 150.2/255 → 150 —
        // weighted by the frame's OWN alpha, independent of old_alpha.
        assert_eq!(out.planes[0].data[0], 150);
        // Alpha plane keeps the old value.
        assert_eq!(out.planes[3].data[0], 64);
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
