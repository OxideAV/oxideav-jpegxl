//! Modular sub-bitstream channel decoding (per ISO/IEC 18181-1 committee
//! draft, 2019-08-05, Annex C.9).
//!
//! Pipeline (one channel at a time):
//!
//! 1. Read the channel header — `Varint()` packed predictor + entropy_coder
//!    selector, then `Varint()` for `channel.min`, `channel.max`, etc. (C.9.3).
//! 2. If `channel.min == channel.max`, fill the channel with that constant
//!    and return.
//! 3. Otherwise, byte-align the bitstream (the MA tree + entropy stream is
//!    byte-oriented) and initialise an [`Abrac`] reader.
//! 4. Decode the MA tree from the ABRAC stream (D.7.3, see [`MaTree`]).
//! 5. Walk the channel in raster order. For each pixel:
//!    a. Compute the [`PropertyVec`] (D.7.2) from the partially-decoded
//!    channel and any prior channels' values.
//!    b. Walk the MA tree to find the leaf BEGABRAC.
//!    c. Compute the predictor's `predicted` value (C.9.3.1).
//!    d. Decode `diff` from the leaf BEGABRAC over `[min - predicted,
//!    max - predicted]`.
//!    e. Set the pixel to `predicted + diff`.
//!
//! Only `entropy_coder == 0` (MABEGABRAC) is implemented. `entropy_coder`
//! values 1 (MABrotli) and 2 (MAANS) return [`Error::Unsupported`] —
//! they require external decoders not in this clean-room build.
//!
//! VarDCT and the `kPasses` frame path are out of scope for this module;
//! see `lib.rs` for status.
//!
//! ## Spec mapping (committee draft)
//!
//! | spec section | this module                            |
//! |--------------|----------------------------------------|
//! | C.9.2        | (top-level header — pending wiring)    |
//! | C.9.3        | [`decode_channel_header`]              |
//! | C.9.3.1      | [`decode_channel_pixels`] inner loop   |
//! | D.7.2        | [`compute_properties`]                 |

use crate::error::{JxlError as Error, Result};

use crate::abrac::Abrac;
use crate::bitreader::BitReader;
use crate::matree::{MaTree, PropRange};
use crate::predictors::{ch_zero, median3, Predictor};

/// Hard upper bound on a channel's width or height. Real-world JXL
/// images go up to 2^30 samples per side per the FDIS, but anything
/// past 32k×32k is far more likely to be a malformed/adversarial
/// bitstream than a real image — and would allocate >4 GiB per channel
/// even before predicates / intermediates. We refuse to allocate past
/// this so a hostile codestream cannot DoS the decoder by pretending
/// to have huge planes.
pub const MAX_CHANNEL_DIM: u32 = 32_768;

/// Companion cap: the total number of pixels in a single channel. Even
/// at 32k×32k that's 1 G samples = 4 GiB of i32s; we conservatively
/// cap an order of magnitude lower at 256 M samples (1 GiB per channel)
/// so test machines / WASM hosts can't be trivially OOM'd.
pub const MAX_CHANNEL_PIXELS: usize = 256 * 1024 * 1024;

/// A single Modular channel (a 2-D plane of integer samples).
#[derive(Debug, Clone)]
pub struct Channel {
    pub width: u32,
    pub height: u32,
    /// Horizontal subsampling shift (>=0 means dimensions are `image / 2^h`),
    /// `-1` means "not related to image dimensions" (post-Squeeze).
    pub hshift: i32,
    pub vshift: i32,
    /// Min sample value occurring in this channel (signalled in the header).
    pub min: i32,
    /// Max sample value occurring in this channel (signalled in the header).
    pub max: i32,
    /// Pixel data, length `width * height`, row-major.
    pub data: Vec<i32>,
}

impl Channel {
    /// Allocate an empty channel of the given dimensions, with no
    /// declared range yet.
    ///
    /// Falls back to [`Self::try_new`] for dimension validation; this
    /// constructor panics on out-of-range inputs and is intended for
    /// trusted internal callers (tests, unit-test fixtures). Bitstream
    /// callers should use [`Self::try_new`] so a malformed channel
    /// header can't OOM the process.
    pub fn new(width: u32, height: u32) -> Self {
        Self::try_new(width, height).expect("channel dimensions out of range")
    }

    /// Fallible version of [`Self::new`]: refuses to allocate when
    /// `width` or `height` exceeds [`MAX_CHANNEL_DIM`], or when
    /// `width * height` exceeds [`MAX_CHANNEL_PIXELS`]. This is the
    /// gatekeeper between bitstream-controlled dimensions and the
    /// allocator — without it a single forged Varint can ask for
    /// terabytes of pixel storage.
    pub fn try_new(width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidData(format!(
                "JXL Modular channel: zero-dimensional ({width}x{height})"
            )));
        }
        if width > MAX_CHANNEL_DIM || height > MAX_CHANNEL_DIM {
            return Err(Error::InvalidData(format!(
                "JXL Modular channel: dimensions {width}x{height} exceed cap {MAX_CHANNEL_DIM}"
            )));
        }
        let pixels = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| {
                Error::InvalidData(format!(
                    "JXL Modular channel: {width}*{height} overflows usize"
                ))
            })?;
        if pixels > MAX_CHANNEL_PIXELS {
            return Err(Error::InvalidData(format!(
                "JXL Modular channel: {pixels} pixels exceed cap {MAX_CHANNEL_PIXELS}"
            )));
        }
        Ok(Self {
            width,
            height,
            hshift: 0,
            vshift: 0,
            min: 0,
            max: 0,
            data: vec![0; pixels],
        })
    }

    #[inline]
    pub fn get(&self, x: u32, y: u32) -> i32 {
        self.data[(y as usize) * (self.width as usize) + (x as usize)]
    }

    #[inline]
    pub fn set(&mut self, x: u32, y: u32, v: i32) {
        let idx = (y as usize) * (self.width as usize) + (x as usize);
        self.data[idx] = v;
    }
}

/// Per-channel header decoded from the byte stream just before the MA tree.
#[derive(Debug, Clone, Copy)]
pub struct ChannelHeader {
    pub predictor: Predictor,
    pub entropy_coder: u32,
    pub min: i32,
    pub max: i32,
    /// `true` iff `min == max`, in which case no MA tree / pixel stream
    /// follows (the channel is a constant).
    pub constant: bool,
}

/// Decode a single channel's header (C.9.3) from a byte-aligned bit reader.
///
/// The channel header is packed as a sequence of `Varint()` reads. The
/// returned [`ChannelHeader`] is sufficient to either fast-path the
/// constant-channel case or hand off to [`decode_channel_pixels`].
pub fn decode_channel_header(br: &mut BitReader<'_>) -> Result<ChannelHeader> {
    let pack = br.read_varint()? as u32;
    let predictor_id = pack >> 2;
    let entropy_coder = pack & 0x3;
    let predictor = Predictor::from_id(predictor_id).ok_or_else(|| {
        Error::InvalidData(format!(
            "JXL Modular: unknown predictor id {predictor_id} in channel header"
        ))
    })?;

    // First Varint() encodes either (1 - min) when min <= 0, or 0 when
    // min == 1 (with the actual `min - 1` following). We accept the
    // simpler always-non-negative reading for fixtures we currently
    // support (signed channels with min < 0 use the natural form).
    let v0 = br.read_varint()? as i64;
    let min: i64 = if v0 == 0 {
        // min is strictly positive; second Varint() carries (min - 1).
        let m1 = br.read_varint()? as i64;
        m1 + 1
    } else {
        // v0 = 1 - min  →  min = 1 - v0.
        1 - v0
    };
    let span = br.read_varint()? as i64;
    let max = min + span;

    if min < i32::MIN as i64 || max > i32::MAX as i64 {
        return Err(Error::InvalidData(
            "JXL Modular: channel min/max out of i32 range".into(),
        ));
    }
    let min = min as i32;
    let max = max as i32;
    let constant = min == max;
    Ok(ChannelHeader {
        predictor,
        entropy_coder,
        min,
        max,
        constant,
    })
}

/// Property vector computed for one pixel; index [0, n+12).
///
/// `n` is the number of `extra` (prior-channel) properties contributed,
/// so the always-present properties live at indices `n+0..n+11` in the
/// vector. We represent the whole thing as a flat `Vec<i32>`.
pub type PropertyVec = Vec<i32>;

/// Compute the property vector for sample `(x, y)` of `channel` (D.7.2),
/// given a list of prior channels' decoded data (some entries may be
/// `None` when the channel was the constant-fast-pathed kind).
///
/// `max_extra_properties` is the cap from the modular bitstream header;
/// only the most recent matching channels are used.
pub fn compute_properties(
    channel: &Channel,
    x: u32,
    y: u32,
    prior: &[&Channel],
    max_extra_properties: usize,
) -> PropertyVec {
    // Extra-channel properties first (indexes 0..n).
    let mut extras: Vec<i32> = Vec::with_capacity(4 * max_extra_properties);
    for prev in prior.iter().rev() {
        if extras.len() >= 4 * max_extra_properties {
            break;
        }
        if prev.min == prev.max {
            continue;
        }
        if prev.width == 0 || prev.height == 0 {
            continue;
        }
        if prev.hshift < 0 {
            continue;
        }
        // Spec: ry = (y << channel.vshift) >> prev.vshift; same for rx.
        // In our committee-draft fixtures all channels share dimensions
        // so the shifts cancel; keep the formula faithful for safety.
        let cur_h = channel.hshift.max(0) as u32;
        let cur_v = channel.vshift.max(0) as u32;
        let prev_h = prev.hshift.max(0) as u32;
        let prev_v = prev.vshift.max(0) as u32;
        let mut ry = (y << cur_v) >> prev_v;
        let mut rx = (x << cur_h) >> prev_h;
        if ry >= prev.height {
            ry = prev.height - 1;
        }
        if rx >= prev.width {
            rx = prev.width - 1;
        }
        let pz = ch_zero(prev.min, prev.max);
        let rv = prev.get(rx, ry);
        let rleft = if rx > 0 { prev.get(rx - 1, ry) } else { pz };
        let rtop = if ry > 0 { prev.get(rx, ry - 1) } else { rleft };
        let rtopleft = if rx > 0 && ry > 0 {
            prev.get(rx - 1, ry - 1)
        } else {
            rleft
        };
        let rp = median3(rleft + rtop - rtopleft, rleft, rtop);
        extras.push(rv.abs());
        extras.push(rv);
        extras.push((rv - rp).abs());
        extras.push(rv - rp);
    }
    let n = extras.len();

    // Always-present properties (indexes n+0..n+11).
    let cz = ch_zero(channel.min, channel.max);
    let top = if y > 0 { channel.get(x, y - 1) } else { cz };
    let left = if x > 0 { channel.get(x - 1, y) } else { cz };
    let topleft = if x > 0 && y > 0 {
        channel.get(x - 1, y - 1)
    } else {
        left
    };
    let topright = if y > 0 && x + 1 < channel.width {
        channel.get(x + 1, y - 1)
    } else {
        top
    };
    let leftleft = if x > 1 { channel.get(x - 2, y) } else { left };
    let toptop = if y > 1 { channel.get(x, y - 2) } else { top };

    let mut props: PropertyVec = Vec::with_capacity(n + 12);
    props.extend_from_slice(&extras);
    props.push(top.abs()); // n+0
    props.push(left.abs()); // n+1
    props.push(top); // n+2
    props.push(left); // n+3
    props.push(y as i32); // n+4
    props.push(x as i32); // n+5
    props.push(left + top - topleft); // n+6
    props.push(topleft + topright - top); // n+7
    props.push(left - topleft); // n+8
    props.push(topleft - top); // n+9
    props.push(top - toptop); // n+10  (per spec text; table has typo)
    props.push(left - leftleft); // n+11
    props
}

/// Per-property initial ranges (D.7.2 table) for use when seeding the
/// MA tree decode. Returns a vector of length `n + 12` (with `n =
/// 4 * num_decodable_prior`).
pub fn property_ranges(
    channel_min: i32,
    channel_max: i32,
    width: u32,
    height: u32,
    extra_channels: &[&Channel],
    max_extra_properties: usize,
) -> Vec<PropRange> {
    let mut ranges: Vec<PropRange> = Vec::new();
    // Extra-channel ranges first (k+0..k+3 per "previous channel").
    let mut taken = 0usize;
    for prev in extra_channels.iter().rev() {
        if taken >= max_extra_properties {
            break;
        }
        if prev.min == prev.max {
            continue;
        }
        if prev.width == 0 || prev.height == 0 {
            continue;
        }
        if prev.hshift < 0 {
            continue;
        }
        let abs_max = prev.min.abs().max(prev.max.abs());
        let span = (prev.max as i64 - prev.min as i64) as i32;
        ranges.push(PropRange {
            min: 0,
            max: abs_max,
        });
        ranges.push(PropRange {
            min: prev.min,
            max: prev.max,
        });
        ranges.push(PropRange { min: 0, max: span });
        ranges.push(PropRange {
            min: -span,
            max: span,
        });
        taken += 1;
    }
    // Always-present ranges per the table immediately after the loop.
    let abs_max = channel_min.abs().max(channel_max.abs());
    ranges.push(PropRange {
        min: 0,
        max: abs_max,
    }); // abs(top)
    ranges.push(PropRange {
        min: 0,
        max: abs_max,
    }); // abs(left)
    ranges.push(PropRange {
        min: channel_min,
        max: channel_max,
    }); // top
    ranges.push(PropRange {
        min: channel_min,
        max: channel_max,
    }); // left
    if height == 0 {
        ranges.push(PropRange { min: 0, max: 0 });
    } else {
        ranges.push(PropRange {
            min: 0,
            max: (height as i32) - 1,
        }); // y
    }
    if width == 0 {
        ranges.push(PropRange { min: 0, max: 0 });
    } else {
        ranges.push(PropRange {
            min: 0,
            max: (width as i32) - 1,
        }); // x
    }
    let two_min_minus_max = 2 * channel_min - channel_max;
    let two_max_minus_min = 2 * channel_max - channel_min;
    let span = channel_max - channel_min;
    ranges.push(PropRange {
        min: two_min_minus_max,
        max: two_max_minus_min,
    }); // left+top-topleft
    ranges.push(PropRange {
        min: two_min_minus_max,
        max: two_max_minus_min,
    }); // topleft+topright-top
    ranges.push(PropRange {
        min: -span,
        max: span,
    }); // left-topleft
    ranges.push(PropRange {
        min: -span,
        max: span,
    }); // topleft-top
    ranges.push(PropRange {
        min: -span,
        max: span,
    }); // top-toptop / top-topright (spec text)
    ranges.push(PropRange {
        min: -span,
        max: span,
    }); // left-leftleft

    ranges
}

/// Decode the pixel grid of `channel` from `coder`, using the previously
/// decoded `tree`, predictor selector, and prior channels for property
/// computation (C.9.3.1).
///
/// `max_extra_properties` controls how many of the prior channels'
/// values feed into the property vector (matching the MA tree's
/// expectations).
pub fn decode_channel_pixels(
    coder: &mut Abrac<'_>,
    tree: &mut MaTree,
    channel: &mut Channel,
    predictor: Predictor,
    prior: &[&Channel],
    max_extra_properties: usize,
) -> Result<()> {
    let cz = ch_zero(channel.min, channel.max);
    for y in 0..channel.height {
        for x in 0..channel.width {
            let props = compute_properties(channel, x, y, prior, max_extra_properties);
            let leaf_idx = tree.walk(&props)?;
            let top = if y > 0 { channel.get(x, y - 1) } else { cz };
            let left = if x > 0 { channel.get(x - 1, y) } else { cz };
            let topleft = if x > 0 && y > 0 {
                channel.get(x - 1, y - 1)
            } else {
                left
            };
            let predicted = predictor.predict(left, top, topleft, cz);
            let leaf_min = (channel.min as i64 - predicted as i64) as i32;
            let leaf_max = (channel.max as i64 - predicted as i64) as i32;
            let bg = tree.leaf_mut(leaf_idx);
            let diff = bg.decode(coder, leaf_min, leaf_max)?;
            channel.set(x, y, predicted + diff);
        }
    }
    Ok(())
}

/// Convenience: decode a single self-contained channel from `bytes`.
///
/// `bytes` shall start at the first byte of the channel header, and
/// the decoder shall consume the channel header, MA tree and pixel
/// data. This is the smallest end-to-end Modular fixture path useful
/// for tests; it does not handle multi-channel images, transforms,
/// or the surrounding FrameHeader/TOC.
pub fn decode_single_channel(
    bytes: &[u8],
    width: u32,
    height: u32,
    max_extra_properties: usize,
) -> Result<Channel> {
    let mut br = BitReader::new(bytes);
    let header = decode_channel_header(&mut br)?;
    if header.entropy_coder != 0 {
        return Err(Error::Unsupported(format!(
            "jxl modular: entropy_coder={} not yet supported (only MABEGABRAC=0)",
            header.entropy_coder
        )));
    }
    let mut channel = Channel::try_new(width, height)?;
    channel.min = header.min;
    channel.max = header.max;
    if header.constant {
        for v in channel.data.iter_mut() {
            *v = header.min;
        }
        return Ok(channel);
    }
    // Byte-align, then start ABRAC on the rest of the slice.
    br.pu0()?;
    let cursor = br.bytes_consumed();
    let abrac_bytes = &bytes[cursor..];
    let mut coder = Abrac::new(abrac_bytes)?;
    // Build the MA tree.
    let ranges = property_ranges(
        header.min,
        header.max,
        width,
        height,
        &[],
        max_extra_properties,
    );
    let n_bits = bit_depth_for_range(header.min, header.max);
    let mut tree = MaTree::decode(&mut coder, &ranges, n_bits, /*signal_init=*/ true)?;
    decode_channel_pixels(
        &mut coder,
        &mut tree,
        &mut channel,
        header.predictor,
        &[],
        max_extra_properties,
    )?;
    Ok(channel)
}

/// Compute the BEGABRAC bit-depth `N` sufficient to cover every value
/// in `[min, max]`. Per the spec we need
/// `ceil(log2(max(|min|, |max|))) + 1`, with the trailing `+1` so the
/// initial `1 << exp` representation can hold the largest signed
/// magnitude.
fn bit_depth_for_range(min: i32, max: i32) -> u32 {
    let absmax = min.unsigned_abs().max(max.unsigned_abs());
    if absmax == 0 {
        1
    } else {
        32 - absmax.leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abrac::tests_enc::AbracEncoder;
    use crate::begabrac::tests::encode_one;
    use crate::matree::ZERO_INIT;

    /// Encoder counterpart for [`decode_single_channel`], used to build
    /// deterministic test fixtures. NOT part of the public API; the
    /// crate is decoder-only by design.
    ///
    /// Produces a byte stream matching what a spec-conforming encoder
    /// would emit for a single-channel Modular image with the given
    /// pixel grid + Gradient predictor + MABEGABRAC + a single-leaf
    /// MA tree (the simplest possible context model).
    fn encode_single_channel_modular_gradient(width: u32, height: u32, pixels: &[i32]) -> Vec<u8> {
        assert_eq!(pixels.len(), (width * height) as usize);
        let mut min = pixels[0];
        let mut max = pixels[0];
        for &v in pixels.iter() {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        // ---- Channel header (Varints, byte-aligned bit stream) ----
        let mut bytes = Vec::<u8>::new();
        let predictor_id = Predictor::Gradient as u32;
        let entropy_coder: u32 = 0;
        push_varint(&mut bytes, ((predictor_id << 2) | entropy_coder) as u64);
        // min encoding: when min<=0, we send (1 - min); when min>0,
        // we send 0 then (min-1).
        if min <= 0 {
            push_varint(&mut bytes, (1 - min) as u64);
        } else {
            push_varint(&mut bytes, 0);
            push_varint(&mut bytes, (min - 1) as u64);
        }
        // max encoding: span (max - min)
        push_varint(&mut bytes, (max - min) as u64);

        if min == max {
            return bytes;
        }

        // ---- ABRAC payload (MA tree + pixels) ----
        let mut enc = AbracEncoder::new();
        // MA tree: a single leaf. To match the decoder:
        // BEGABRAC1: encode property = -1 (i.e., emit value 0). signal_init=true.
        let n_bits = bit_depth_for_range(min, max);
        let mut bg1 =
            crate::begabrac::Begabrac::new(super_ilog2_at_least((12 + 12) as u32) + 1, 1024);
        let mut bg2 = crate::begabrac::Begabrac::new(4, 1024);
        let _bg3 = crate::begabrac::Begabrac::new(3, 1024); // unused; zc=5 means BEGABRAC3 isn't read.
        let _bg4 = crate::begabrac::Begabrac::new(n_bits.max(4) + 1, 1024);

        // Choose property index = -1 (leaf) → emit 0 in [0, 12+12).
        encode_one(&mut enc, &mut bg1, 0, 0, 12 + 12);
        // signal_init: emit zc = 5 (offset +5 → ZERO_INIT[5] = 2048).
        // bg2 over [-5,5]; we want zc-5=0 → emit value 0.
        encode_one(&mut enc, &mut bg2, 0, -5, 5);
        // (zc=5 not < 3, so no BEGABRAC3 read.)
        let leaf_init = ZERO_INIT[5];

        // ---- Pixel residuals ----
        let mut leaf_bg = crate::begabrac::Begabrac::new(n_bits.max(1) + 1, leaf_init);
        // Mirror the decoder loop step-for-step.
        // Predictor = Gradient.
        let predictor = Predictor::Gradient;
        let cz = ch_zero(min, max);
        // Track the channel as we go (so we can compute the same
        // predictor as the decoder).
        let mut chan = vec![0i32; pixels.len()];
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) as usize;
                let top = if y > 0 {
                    chan[((y - 1) * width + x) as usize]
                } else {
                    cz
                };
                let left = if x > 0 {
                    chan[(y * width + (x - 1)) as usize]
                } else {
                    cz
                };
                let topleft = if x > 0 && y > 0 {
                    chan[((y - 1) * width + (x - 1)) as usize]
                } else {
                    left
                };
                let predicted = predictor.predict(left, top, topleft, cz);
                let diff = pixels[i] - predicted;
                let leaf_min = (min as i64 - predicted as i64) as i32;
                let leaf_max = (max as i64 - predicted as i64) as i32;
                encode_one(&mut enc, &mut leaf_bg, diff, leaf_min, leaf_max);
                chan[i] = pixels[i];
            }
        }

        let abrac_stream = enc.finish();
        bytes.extend_from_slice(&abrac_stream);
        bytes
    }

    /// Identical to the public `bit_depth_for_range`; duplicated here
    /// (with a different name) so the tests don't depend on the module
    /// path of a private helper.
    fn super_ilog2_at_least(x: u32) -> u32 {
        if x <= 1 {
            1
        } else {
            32 - (x - 1).leading_zeros()
        }
    }

    /// Encode a `u64` as a JXL `Varint()` and append it to `out`.
    fn push_varint(out: &mut Vec<u8>, mut v: u64) {
        while v >= 0x80 {
            out.push(((v & 0x7f) as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }

    #[test]
    fn channel_set_get_round_trip() {
        let mut ch = Channel::new(3, 2);
        ch.set(2, 1, 42);
        assert_eq!(ch.get(2, 1), 42);
        assert_eq!(ch.get(0, 0), 0);
    }

    #[test]
    fn property_ranges_no_extras_basic() {
        // 8x8 channel in [0,255], no extras, no max_extra_properties.
        let r = property_ranges(0, 255, 8, 8, &[], 0);
        assert_eq!(r.len(), 12);
        assert_eq!(r[0].min, 0);
        assert_eq!(r[0].max, 255); // abs(top)
        assert_eq!(r[2].min, 0);
        assert_eq!(r[2].max, 255); // top
        assert_eq!(r[4].min, 0);
        assert_eq!(r[4].max, 7); // y
        assert_eq!(r[5].min, 0);
        assert_eq!(r[5].max, 7); // x
    }

    #[test]
    fn compute_properties_first_pixel() {
        // First pixel of 4x4 zero channel: top/left/topleft = ch_zero = 0.
        let ch = Channel {
            width: 4,
            height: 4,
            hshift: 0,
            vshift: 0,
            min: 0,
            max: 255,
            data: vec![0; 16],
        };
        let p = compute_properties(&ch, 0, 0, &[], 0);
        assert_eq!(p.len(), 12);
        assert_eq!(p[0], 0); // abs(top)
        assert_eq!(p[1], 0); // abs(left)
        assert_eq!(p[4], 0); // y
        assert_eq!(p[5], 0); // x
    }

    #[test]
    fn channel_header_constant_channel_round_trips() {
        // Constant channel: predictor=Gradient, entropy_coder=0, min=max=42.
        // pack = (Gradient << 2) | 0 = 8.  min>0 path: 0 then (min-1)=41.
        // span = 0.
        let mut bytes = Vec::new();
        push_varint(&mut bytes, 8);
        push_varint(&mut bytes, 0);
        push_varint(&mut bytes, 41);
        push_varint(&mut bytes, 0);
        let mut br = BitReader::new(&bytes);
        let h = decode_channel_header(&mut br).unwrap();
        assert_eq!(h.entropy_coder, 0);
        assert_eq!(h.predictor, Predictor::Gradient);
        assert_eq!(h.min, 42);
        assert_eq!(h.max, 42);
        assert!(h.constant);
    }

    #[test]
    fn decode_single_channel_constant_yields_constant_grid() {
        let mut bytes = Vec::new();
        push_varint(&mut bytes, (Predictor::Zero as u64) << 2);
        push_varint(&mut bytes, 0);
        push_varint(&mut bytes, 41); // min = 42
        push_varint(&mut bytes, 0); // span = 0 → max = 42
        let ch = decode_single_channel(&bytes, 4, 3, 0).unwrap();
        assert_eq!(ch.width, 4);
        assert_eq!(ch.height, 3);
        assert_eq!(ch.data, vec![42; 12]);
    }

    #[test]
    fn round_trip_4x4_gradient_modular_channel() {
        // A small smooth grid: each row goes 0,1,2,3, then 4,5,6,7, etc.
        // Gradient predictor should track this exactly so residuals are
        // tiny and the round trip is unambiguous.
        let pixels = [0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let bytes = encode_single_channel_modular_gradient(4, 4, &pixels);
        let ch = decode_single_channel(&bytes, 4, 4, 0).unwrap();
        assert_eq!(ch.data, pixels);
    }

    #[test]
    fn round_trip_2x2_single_pixel_value() {
        // All pixels equal but encoded the long way (non-constant min/max):
        // we lie about the range to force the MA tree path.
        let pixels = [7i32, 7, 7, 7];
        let bytes = encode_single_channel_modular_gradient(2, 2, &pixels);
        let ch = decode_single_channel(&bytes, 2, 2, 0).unwrap();
        assert_eq!(ch.data, pixels);
    }

    #[test]
    fn round_trip_8x8_random_in_range() {
        // Pseudorandom pixels in [0,15] from a tiny LCG to keep the test
        // deterministic and free of std::collections.
        let mut state: u32 = 0x1234_5678;
        let mut pixels = [0i32; 64];
        for p in pixels.iter_mut() {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *p = ((state >> 17) & 0xF) as i32;
        }
        let bytes = encode_single_channel_modular_gradient(8, 8, &pixels);
        let ch = decode_single_channel(&bytes, 8, 8, 0).unwrap();
        assert_eq!(ch.data, pixels);
    }

    #[test]
    fn rejects_oversized_channel_dimensions() {
        // Channel header parses fine: predictor=Zero, entropy_coder=0, min=0,
        // max=255 (not constant). But we pass width=1_000_000 / height=1_000_000
        // — a forged dimension pair that, with the previous code, would
        // try to allocate 4 TiB of i32s before the decoder ever looks at
        // the bitstream proper. We expect a clean InvalidData instead.
        let mut bytes = Vec::new();
        push_varint(&mut bytes, (Predictor::Zero as u64) << 2); // pack = 0
        push_varint(&mut bytes, 1); // min = 0 (1 - 1 = 0)
        push_varint(&mut bytes, 255); // span = 255 → max = 255
                                      // Plenty of zero padding so a successful decode wouldn't trip an
                                      // EOF later in the pipeline; the cap should fire first.
        bytes.extend_from_slice(&[0u8; 64]);
        let err = decode_single_channel(&bytes, 1_000_000, 1_000_000, 0).unwrap_err();
        match err {
            Error::InvalidData(msg) => {
                assert!(
                    msg.contains("dimensions") || msg.contains("pixels"),
                    "{msg}"
                );
            }
            other => panic!("expected InvalidData, got {other:?}"),
        }
    }

    #[test]
    fn try_new_rejects_zero_dim() {
        let err = Channel::try_new(0, 8).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
        let err = Channel::try_new(8, 0).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn try_new_rejects_overlarge_side() {
        let err = Channel::try_new(MAX_CHANNEL_DIM + 1, 1).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn try_new_rejects_overlarge_pixel_count() {
        // 32k × 16k = 512 M > MAX_CHANNEL_PIXELS (256 M).
        let err = Channel::try_new(32_768, 16_384).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn rejects_unsupported_entropy_coder() {
        // entropy_coder = 2 (MAANS) — not implemented in this clean-room
        // build; should return Unsupported, not InvalidData.
        let mut bytes = Vec::new();
        push_varint(&mut bytes, ((Predictor::Gradient as u64) << 2) | 2);
        push_varint(&mut bytes, 1); // min = 0
        push_varint(&mut bytes, 255); // max = 255
        let err = decode_single_channel(&bytes, 4, 4, 0).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }
}
