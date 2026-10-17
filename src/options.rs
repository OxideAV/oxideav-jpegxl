//! [`DecodeOptions`] and [`EncodeOptions`].

use crate::api::AnimationInfo;

/// Decode limits and behaviour switches.
///
/// Limits are checked against the codestream's `SizeHeader` /
/// `ImageMetadata` **before** any pixel buffer is allocated; a hit
/// returns [`Error::LimitExceeded`](crate::Error::LimitExceeded).
/// `None` means unlimited.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct DecodeOptions {
    /// Maximum image width in pixels.
    pub max_width: Option<u32>,
    /// Maximum image height in pixels.
    pub max_height: Option<u32>,
    /// Maximum `width × height`.
    pub max_pixels: Option<u64>,
    /// Maximum size of the decoded output image in bytes
    /// (`width × height × bytes_per_pixel` of the native layout). The
    /// decoder's working set is larger — every channel is held as
    /// `i32` (or `f32`) samples while decoding, roughly four bytes per
    /// sample per channel — so budget accordingly.
    pub max_bytes: Option<u64>,
    /// Reject trailing bytes after the last frame of a raw codestream
    /// (an ISOBMFF file's box structure is always validated). Default
    /// `false`: trailing bytes are ignored.
    pub strict: bool,
    /// Whether [`decode_all`](crate::decode_all) composes frames onto
    /// the image canvas per ISO/IEC 18181-1 §C.2 (blending, crops,
    /// reference frames) and returns only the *presented* frames.
    /// With `false` every decoded frame is returned as-is at its own
    /// frame-rectangle geometry, un-blended, including zero-duration
    /// layers. Default `true`.
    pub coalesce: bool,
}

impl DecodeOptions {
    /// Default `max_bytes`: 1 GiB of decoded output.
    pub const DEFAULT_MAX_BYTES: u64 = 1 << 30;
    /// Default `max_width` / `max_height`: 2^30 − 1 (the largest
    /// `SizeHeader` dimension the `u(30)` field can express).
    pub const DEFAULT_MAX_DIMENSION: u32 = (1 << 30) - 1;

    /// Same as [`Default::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set `max_width` (`None` = unlimited).
    pub fn with_max_width(mut self, max_width: impl Into<Option<u32>>) -> Self {
        self.max_width = max_width.into();
        self
    }

    /// Builder: set `max_height` (`None` = unlimited).
    pub fn with_max_height(mut self, max_height: impl Into<Option<u32>>) -> Self {
        self.max_height = max_height.into();
        self
    }

    /// Builder: set `max_pixels` (`None` = unlimited).
    pub fn with_max_pixels(mut self, max_pixels: impl Into<Option<u64>>) -> Self {
        self.max_pixels = max_pixels.into();
        self
    }

    /// Builder: set `max_bytes` (`None` = unlimited).
    pub fn with_max_bytes(mut self, max_bytes: impl Into<Option<u64>>) -> Self {
        self.max_bytes = max_bytes.into();
        self
    }

    /// Builder: set `strict`.
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Builder: set `coalesce`.
    pub fn with_coalesce(mut self, coalesce: bool) -> Self {
        self.coalesce = coalesce;
        self
    }

    /// Remove every limit (`strict` and `coalesce` unchanged).
    pub fn unlimited(mut self) -> Self {
        self.max_width = None;
        self.max_height = None;
        self.max_pixels = None;
        self.max_bytes = None;
        self
    }
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            max_width: Some(Self::DEFAULT_MAX_DIMENSION),
            max_height: Some(Self::DEFAULT_MAX_DIMENSION),
            max_pixels: Some(Self::DEFAULT_MAX_BYTES),
            max_bytes: Some(Self::DEFAULT_MAX_BYTES),
            strict: false,
            coalesce: true,
        }
    }
}

/// Encoder options.
///
/// `#[non_exhaustive]`: build with [`new`](Self::new) / `Default` and
/// the `with_*` builders.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct EncodeOptions {
    /// Advisory quality `0..=100` (default `100`). `100` is lossless
    /// Modular; below it the Modular residuals are quantised with the
    /// step [`crate::encoder::quantizer_for_quality`] documents
    /// (`round(2^((100 − q) / 12))` at 8 bits, scaled with the depth),
    /// so a lower quality never produces a larger file on textured
    /// content.
    pub quality: u8,
    /// Always write the 18181-2 box file (signature, `ftyp`, `jxlc`).
    /// Default `false`: a bare codestream unless an `Exif` / `xml `
    /// box is needed for the image's metadata.
    pub container: bool,
    /// `group_dim = 128 << group_size_shift` (F.2), `0..=3`; default
    /// `1` (256-pixel groups).
    pub group_size_shift: u8,
    /// Tick rate and loop count for [`encode_all`](crate::encode_all):
    /// frame durations are then the frames' `ticks`. `None` (default)
    /// writes a 1000 Hz tick and takes each frame's `delay` rounded to
    /// milliseconds (`100` ms when a frame has none).
    pub animation: Option<AnimationInfo>,
}

impl EncodeOptions {
    /// Same as [`Default::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set `quality` (clamped to `100`).
    pub fn with_quality(mut self, quality: u8) -> Self {
        self.quality = quality.min(100);
        self
    }

    /// Builder: set `container`.
    pub fn with_container(mut self, container: bool) -> Self {
        self.container = container;
        self
    }

    /// Builder: set `group_size_shift` (clamped to `3`).
    pub fn with_group_size_shift(mut self, shift: u8) -> Self {
        self.group_size_shift = shift.min(3);
        self
    }

    /// Builder: set `animation`.
    pub fn with_animation(mut self, animation: impl Into<Option<AnimationInfo>>) -> Self {
        self.animation = animation.into();
        self
    }

    /// `true` when the residuals are quantised (`quality < 100`).
    pub fn is_lossy(&self) -> bool {
        self.quality < 100
    }
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            quality: 100,
            container: false,
            group_size_shift: 1,
            animation: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_finite() {
        let d = DecodeOptions::default();
        assert!(d.max_width.is_some() && d.max_height.is_some());
        assert_eq!(d.max_bytes, Some(1 << 30));
        assert!(!d.strict);
        assert!(d.coalesce);
        let u = d.unlimited();
        assert!(u.max_width.is_none() && u.max_bytes.is_none() && u.max_pixels.is_none());
    }

    #[test]
    fn encode_defaults_are_lossless_bare_codestream() {
        let e = EncodeOptions::default();
        assert_eq!(e.quality, 100);
        assert!(!e.container && !e.is_lossy());
        assert_eq!(e.group_size_shift, 1);
        assert!(e.animation.is_none());
        let e = e
            .with_quality(200)
            .with_group_size_shift(9)
            .with_container(true);
        assert_eq!((e.quality, e.group_size_shift), (100, 3));
        assert!(e.container);
        assert!(EncodeOptions::new().with_quality(50).is_lossy());
    }
}
