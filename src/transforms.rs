//! Modular transforms — FDIS 18181-1 §C.9.4 + Tables C.24..C.27 + Annexes
//! L.4 (RCT), L.5 (Palette), I.3 (Squeeze).
//!
//! Each `TransformInfo` (Table C.26) describes one of three operations
//! applied to the modular channel list:
//!
//!   * **kRCT** (id 0) — Reversible Colour Transform on three channels
//!     starting at `begin_c`. Selected by `rct_type ∈ [0, 41]`. Channel
//!     count and dimensions are unchanged.
//!   * **kPalette** (id 1) — replaces `num_c` consecutive channels with
//!     one meta-channel (palette table) inserted at index 0 + one
//!     index/data channel at the original position. `nb_meta_channels`
//!     increments by one.
//!   * **kSqueeze** (id 2) — Haar-like wavelet split. `num_sq` steps
//!     either splitting one channel into a smaller "average" channel
//!     plus a residual channel (default params kick in when `num_sq=0`).
//!
//! ## Round 1 scope
//!
//! * Parse all three transform variants from the bitstream per Table C.26.
//! * Apply forward effect on the channel-list shape so pixel decode sees
//!   the right number of channels with the right dimensions.
//! * Implement the three inverse transforms (Listing L.3 RCT, Listing L.6
//!   palette, Listings I.18..I.22 squeeze + tendency).
//!
//! Allocation: `nb_transforms` is capped at `MAX_TRANSFORMS`, `num_sq`
//! at `MAX_SQUEEZE_STEPS`, `nb_colours+nb_deltas` at `MAX_PALETTE_SIZE`.
//! Channel count is bounded by [`crate::modular_fdis::MAX_CHANNELS`].

use crate::error::{JxlError as Error, Result};

use crate::bitreader::{BitReader, U32Dist};
use crate::modular_fdis::{ChannelDesc, ModularImage, MAX_CHANNELS, MAX_DIM};

/// Hard cap on `nb_transforms`. The spec's U32 distribution maxes at
/// `BitsOffset(8, 18) = 273`, but a real-world frame uses at most a
/// handful (cjxl's gradient_64x64 fixture: 4). 64 is generous.
pub const MAX_TRANSFORMS: usize = 64;

/// Hard cap on the number of squeeze steps inside one Squeeze transform.
/// FDIS U32 maxes at `BitsOffset(8, 41) = 296`; cap at 128.
pub const MAX_SQUEEZE_STEPS: usize = 128;

/// Hard cap on `nb_colours + nb_deltas` for a single palette transform.
/// FDIS U32 maxes at `BitsOffset(16, 5377) = 70912` for nb_colours and
/// similar for nb_deltas. Cap at 65536 to bound the meta-channel
/// allocation.
pub const MAX_PALETTE_SIZE: u32 = 1 << 16;

/// Discriminator from Table C.24.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformId {
    Rct = 0,
    Palette = 1,
    Squeeze = 2,
}

impl TransformId {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Self::Rct),
            1 => Ok(Self::Palette),
            2 => Ok(Self::Squeeze),
            _ => Err(Error::InvalidData(format!(
                "JXL TransformInfo: tr={v} not in [0, 2]"
            ))),
        }
    }
}

/// FDIS §9.2.8 `Enum(TransformId)` — same distribution as `Enum(...)`
/// elsewhere in the spec. Caps at 63 then validates.
fn read_enum_transform_id(br: &mut BitReader<'_>) -> Result<TransformId> {
    let v = br.read_u32([
        U32Dist::Val(0),
        U32Dist::Val(1),
        U32Dist::BitsOffset(4, 2),
        U32Dist::BitsOffset(6, 18),
    ])?;
    if v > 63 {
        return Err(Error::InvalidData(
            "JXL Enum(TransformId): value > 63".into(),
        ));
    }
    TransformId::from_u32(v)
}

/// `begin_c` distribution (Table C.25 / C.26). Identical between
/// SqueezeParams and TransformInfo for non-Squeeze.
fn read_begin_c(br: &mut BitReader<'_>) -> Result<u32> {
    br.read_u32([
        U32Dist::Bits(3),
        U32Dist::BitsOffset(6, 8),
        U32Dist::BitsOffset(10, 72),
        U32Dist::BitsOffset(13, 1096),
    ])
}

/// One squeeze step from Table C.25.
#[derive(Debug, Clone, Copy)]
pub struct SqueezeParams {
    pub horizontal: bool,
    pub in_place: bool,
    pub begin_c: u32,
    pub num_c: u32,
}

impl SqueezeParams {
    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let horizontal = br.read_bool()?;
        let in_place = br.read_bool()?;
        let begin_c = read_begin_c(br)?;
        let num_c = br.read_u32([
            U32Dist::Val(1),
            U32Dist::Val(2),
            U32Dist::Val(3),
            U32Dist::BitsOffset(4, 4),
        ])?;
        Ok(Self {
            horizontal,
            in_place,
            begin_c,
            num_c,
        })
    }
}

/// One TransformInfo bundle from Table C.26.
#[derive(Debug, Clone)]
pub enum TransformInfo {
    Rct {
        begin_c: u32,
        rct_type: u32,
    },
    Palette {
        begin_c: u32,
        num_c: u32,
        nb_colours: u32,
        nb_deltas: u32,
        d_pred: u32,
    },
    /// `params` empty ⇒ default Squeeze parameters per Listing I.19.
    Squeeze {
        params: Vec<SqueezeParams>,
    },
}

impl TransformInfo {
    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let tr = read_enum_transform_id(br)?;
        match tr {
            TransformId::Rct => {
                let begin_c = read_begin_c(br)?;
                let rct_type = br.read_u32([
                    U32Dist::Val(6),
                    U32Dist::Bits(2),
                    U32Dist::BitsOffset(4, 2),
                    U32Dist::BitsOffset(6, 10),
                ])?;
                if rct_type > 41 {
                    return Err(Error::InvalidData(format!(
                        "JXL TransformInfo: rct_type {rct_type} > 41 (Table C.26 maximum)"
                    )));
                }
                Ok(Self::Rct { begin_c, rct_type })
            }
            TransformId::Palette => {
                let begin_c = read_begin_c(br)?;
                let num_c = br.read_u32([
                    U32Dist::Val(1),
                    U32Dist::Val(3),
                    U32Dist::Val(4),
                    U32Dist::BitsOffset(13, 1),
                ])?;
                let nb_colours = br.read_u32([
                    U32Dist::BitsOffset(8, 1),
                    U32Dist::BitsOffset(10, 257),
                    U32Dist::BitsOffset(12, 1281),
                    U32Dist::BitsOffset(16, 5377),
                ])?;
                let nb_deltas = br.read_u32([
                    U32Dist::Val(0),
                    U32Dist::BitsOffset(8, 1),
                    U32Dist::BitsOffset(10, 257),
                    U32Dist::BitsOffset(16, 1281),
                ])?;
                let d_pred = br.read_bits(4)?;
                if nb_colours > MAX_PALETTE_SIZE {
                    return Err(Error::InvalidData(format!(
                        "JXL Palette: nb_colours {nb_colours} > cap {MAX_PALETTE_SIZE}"
                    )));
                }
                if nb_deltas > MAX_PALETTE_SIZE {
                    return Err(Error::InvalidData(format!(
                        "JXL Palette: nb_deltas {nb_deltas} > cap {MAX_PALETTE_SIZE}"
                    )));
                }
                if d_pred > 13 {
                    return Err(Error::InvalidData(format!(
                        "JXL Palette: d_pred {d_pred} > 13 (predictor range)"
                    )));
                }
                Ok(Self::Palette {
                    begin_c,
                    num_c,
                    nb_colours,
                    nb_deltas,
                    d_pred,
                })
            }
            TransformId::Squeeze => {
                let num_sq = br.read_u32([
                    U32Dist::Val(0),
                    U32Dist::BitsOffset(4, 1),
                    U32Dist::BitsOffset(6, 9),
                    U32Dist::BitsOffset(8, 41),
                ])?;
                if num_sq as usize > MAX_SQUEEZE_STEPS {
                    return Err(Error::InvalidData(format!(
                        "JXL Squeeze: num_sq {num_sq} > cap {MAX_SQUEEZE_STEPS}"
                    )));
                }
                let mut params = Vec::with_capacity(num_sq as usize);
                for _ in 0..num_sq {
                    params.push(SqueezeParams::read(br)?);
                }
                Ok(Self::Squeeze { params })
            }
        }
    }
}

/// Read the `nb_transforms` U32 + the array per Table C.22.
pub fn read_transforms(br: &mut BitReader<'_>) -> Result<Vec<TransformInfo>> {
    let nb = br.read_u32([
        U32Dist::Val(0),
        U32Dist::Val(1),
        U32Dist::BitsOffset(4, 2),
        U32Dist::BitsOffset(8, 18),
    ])?;
    if nb as usize > MAX_TRANSFORMS {
        return Err(Error::InvalidData(format!(
            "JXL Modular: nb_transforms {nb} > cap {MAX_TRANSFORMS}"
        )));
    }
    let mut v = Vec::with_capacity(nb as usize);
    for _ in 0..nb {
        v.push(TransformInfo::read(br)?);
    }
    Ok(v)
}

/// Apply the forward channel-list effect of a sequence of transforms,
/// producing the channel descriptions that will be decoded from the
/// bitstream. Mirrors Listing I.17 + Tables C.27 + L.5.
///
/// `nb_meta_channels_in` is the initial meta-channel count (always 0
/// per C.9.1; transforms may increment it). The returned tuple is
/// `(transformed_descs, nb_meta_channels)`.
pub fn apply_transforms_forward(
    initial: &[ChannelDesc],
    transforms: &[TransformInfo],
) -> Result<(Vec<ChannelDesc>, u32)> {
    let mut channels: Vec<ChannelDesc> = initial.to_vec();
    let mut nb_meta: u32 = 0;
    for tr in transforms {
        apply_forward_one(&mut channels, &mut nb_meta, tr)?;
        if channels.len() > MAX_CHANNELS {
            return Err(Error::InvalidData(format!(
                "JXL Modular: channel count {} after transform exceeds cap {MAX_CHANNELS}",
                channels.len()
            )));
        }
    }
    Ok((channels, nb_meta))
}

fn apply_forward_one(
    channels: &mut Vec<ChannelDesc>,
    nb_meta: &mut u32,
    tr: &TransformInfo,
) -> Result<()> {
    match tr {
        TransformInfo::Rct { begin_c, .. } => {
            let b = *begin_c as usize;
            if b + 3 > channels.len() {
                return Err(Error::InvalidData(format!(
                    "JXL RCT: begin_c {b} + 3 > channel count {}",
                    channels.len()
                )));
            }
            let (w, h) = (channels[b].width, channels[b].height);
            for i in 0..3 {
                if channels[b + i].width != w || channels[b + i].height != h {
                    return Err(Error::InvalidData(
                        "JXL RCT: channels at begin_c..+3 must share dimensions".into(),
                    ));
                }
            }
            // No change to channel list / meta count.
        }
        TransformInfo::Palette {
            begin_c,
            num_c,
            nb_colours,
            ..
        } => {
            let b = *begin_c as usize;
            let n = *num_c as usize;
            if n == 0 {
                return Err(Error::InvalidData("JXL Palette: num_c == 0".into()));
            }
            if b + n > channels.len() {
                return Err(Error::InvalidData(format!(
                    "JXL Palette: begin_c {b} + num_c {n} > channel count {}",
                    channels.len()
                )));
            }
            // Inserted meta channel: width=nb_colours, height=num_c,
            // hshift=-1, vshift=-1.
            let meta = ChannelDesc {
                width: *nb_colours,
                height: *num_c,
                hshift: -1,
                vshift: -1,
            };
            // Remove channels [b+1 .. b+n] (keep one as the index/data channel
            // at position `b`).
            for _ in 1..n {
                channels.remove(b + 1);
            }
            channels.insert(0, meta);
            *nb_meta = nb_meta.checked_add(1).ok_or_else(|| {
                Error::InvalidData("JXL Palette: nb_meta_channels overflow".into())
            })?;
        }
        TransformInfo::Squeeze { params } => {
            let owned: Vec<SqueezeParams>;
            let sp_slice: &[SqueezeParams] = if params.is_empty() {
                owned = default_squeeze_params(channels, *nb_meta)?;
                &owned
            } else {
                params
            };
            for sp in sp_slice {
                squeeze_step_forward(channels, sp)?;
            }
        }
    }
    Ok(())
}

/// Listing I.17 forward squeeze step (channel-list update).
///
/// **Spec note**: Listing I.17 in the FDIS PDF uses the loop variable `i`
/// for both the outer and the inner; the inner loop is over `c ∈ [begin,
/// end]`, the outer is over `sp[i]`. We apply ONE step here.
fn squeeze_step_forward(channels: &mut Vec<ChannelDesc>, sp: &SqueezeParams) -> Result<()> {
    let begin = sp.begin_c as usize;
    let end = begin + sp.num_c as usize - 1;
    if sp.num_c == 0 {
        return Err(Error::InvalidData("JXL Squeeze: num_c == 0".into()));
    }
    if end >= channels.len() {
        return Err(Error::InvalidData(format!(
            "JXL Squeeze: end {end} >= channel count {}",
            channels.len()
        )));
    }
    // Per Listing I.17, residuals are inserted starting at:
    //   r = sp.in_place ? end + 1 : channel.size()
    // For each c in [begin, end], the channel is shrunk and a residual
    // copy is inserted at index r + (c - begin).
    let mut r = if sp.in_place { end + 1 } else { channels.len() };
    for c in begin..=end {
        let mut residu = channels[c];
        if sp.horizontal {
            let w = channels[c].width;
            channels[c].width = w.div_ceil(2);
            channels[c].hshift = channels[c].hshift.saturating_add(1);
            residu.width = w / 2;
        } else {
            let h = channels[c].height;
            channels[c].height = h.div_ceil(2);
            channels[c].vshift = channels[c].vshift.saturating_add(1);
            residu.height = h / 2;
        }
        let insert_at = r + (c - begin);
        if insert_at > channels.len() {
            return Err(Error::InvalidData(format!(
                "JXL Squeeze: residu insertion index {insert_at} > channel count {}",
                channels.len()
            )));
        }
        channels.insert(insert_at, residu);
        // After in_place insertion the next residu goes to r+1, which
        // (c-begin+1)+r already gives — no extra adjustment needed.
        if !sp.in_place {
            // Inserting at the end shifts `r` only when our insert_at
            // equals the prior list length; since we always insert at
            // r + (c - begin) and the list grew by 1 each iteration,
            // r stays valid.
            r = r.saturating_add(0);
        }
    }
    Ok(())
}

/// Listing I.19 default squeeze params.
fn default_squeeze_params(channels: &[ChannelDesc], nb_meta: u32) -> Result<Vec<SqueezeParams>> {
    let first = nb_meta as usize;
    if first >= channels.len() {
        return Err(Error::InvalidData(
            "JXL default squeeze: no non-meta channels".into(),
        ));
    }
    let last = channels.len() - 1;
    // Spec writes `count = first - last + 1`; that's a sign flip in the
    // PDF; the obvious meaning (and how implementations behave) is
    // `count = last - first + 1`.
    let count = last - first + 1;
    let mut sp: Vec<SqueezeParams> = Vec::new();
    let mut w = channels[first].width;
    let mut h = channels[first].height;
    if count > 2 && channels[first + 1].width == w && channels[first + 1].height == h {
        sp.push(SqueezeParams {
            horizontal: true,
            in_place: false,
            begin_c: (first + 1) as u32,
            num_c: 2,
        });
        sp.push(SqueezeParams {
            horizontal: false,
            in_place: false,
            begin_c: (first + 1) as u32,
            num_c: 2,
        });
    }
    if h >= w && h > 8 {
        sp.push(SqueezeParams {
            horizontal: false,
            in_place: true,
            begin_c: first as u32,
            num_c: count as u32,
        });
        h = h.div_ceil(2);
    }
    while w > 8 || h > 8 {
        if w > 8 {
            sp.push(SqueezeParams {
                horizontal: true,
                in_place: true,
                begin_c: first as u32,
                num_c: count as u32,
            });
            w = w.div_ceil(2);
        }
        if h > 8 {
            sp.push(SqueezeParams {
                horizontal: false,
                in_place: true,
                begin_c: first as u32,
                num_c: count as u32,
            });
            h = h.div_ceil(2);
        }
        if sp.len() > MAX_SQUEEZE_STEPS {
            return Err(Error::InvalidData(format!(
                "JXL default squeeze: step count exceeds cap {MAX_SQUEEZE_STEPS}"
            )));
        }
    }
    Ok(sp)
}

/// Apply the inverse transforms to a decoded modular image, in REVERSE
/// order per C.9.2 final paragraph. Mutates `img` so that `img.channels`
/// + `img.descs` reflect the original (pre-transform) channel layout.
pub fn apply_transforms_inverse(
    img: &mut ModularImage,
    transforms: &[TransformInfo],
    bit_depth: u32,
) -> Result<()> {
    for tr in transforms.iter().rev() {
        apply_inverse_one(img, tr, bit_depth)?;
    }
    Ok(())
}

fn apply_inverse_one(img: &mut ModularImage, tr: &TransformInfo, bit_depth: u32) -> Result<()> {
    match tr {
        TransformInfo::Rct { begin_c, rct_type } => inverse_rct(img, *begin_c as usize, *rct_type),
        TransformInfo::Palette {
            begin_c,
            num_c,
            nb_colours,
            nb_deltas,
            d_pred,
        } => inverse_palette(
            img,
            *begin_c as usize,
            *num_c as usize,
            *nb_colours as i64,
            *nb_deltas as i64,
            *d_pred,
            bit_depth,
        ),
        TransformInfo::Squeeze { params } => {
            // We only know the channel list as it currently is; if
            // params were defaulted at parse time, replay defaults
            // against the *post-decode* channel state would be wrong
            // (defaults depend on pre-squeeze shape). For the round-1
            // common case (defaults were used to derive the forward
            // shape; we stored the *expanded* forward-derived params
            // separately). To handle this cleanly, the caller passes
            // the EXPANDED params (no empties) at this point — see
            // `apply_transforms_inverse_expanded`.
            if params.is_empty() {
                return Err(Error::Unsupported(
                    "JXL inverse Squeeze: default params must be expanded \
                     before inverse — call apply_transforms_inverse_expanded"
                        .into(),
                ));
            }
            for sp in params.iter().rev() {
                inverse_squeeze_step(img, sp)?;
            }
            Ok(())
        }
    }
}

/// Like [`apply_transforms_inverse`] but if a Squeeze transform has
/// empty params, the caller must have already replaced them with the
/// expanded default sequence (computed against the *initial* channel
/// list, not the post-decode one). Returns the bit_depth-aware Palette
/// dispatch.
pub fn apply_transforms_inverse_expanded(
    img: &mut ModularImage,
    transforms: &[TransformInfo],
    bit_depth: u32,
) -> Result<()> {
    apply_transforms_inverse(img, transforms, bit_depth)
}

/// Listing L.3 inverse RCT.
fn inverse_rct(img: &mut ModularImage, begin_c: usize, rct_type: u32) -> Result<()> {
    if begin_c + 3 > img.channels.len() {
        return Err(Error::InvalidData(format!(
            "JXL inverse RCT: begin_c {begin_c} + 3 > channel count {}",
            img.channels.len()
        )));
    }
    let d0 = img.descs[begin_c];
    let d1 = img.descs[begin_c + 1];
    let d2 = img.descs[begin_c + 2];
    if d0.width != d1.width
        || d0.height != d1.height
        || d0.width != d2.width
        || d0.height != d2.height
    {
        return Err(Error::InvalidData(
            "JXL inverse RCT: three channels must share dimensions".into(),
        ));
    }
    let permutation = (rct_type / 7) as usize;
    let kind = rct_type % 7;
    let n = (d0.width as usize).saturating_mul(d0.height as usize);
    if n == 0 {
        return Ok(());
    }
    // Read each pixel, decorrelate, write back into the right
    // permutation slot.
    let mut a_buf = img.channels[begin_c].clone();
    let mut b_buf = img.channels[begin_c + 1].clone();
    let mut c_buf = img.channels[begin_c + 2].clone();
    if a_buf.len() != n || b_buf.len() != n || c_buf.len() != n {
        return Err(Error::InvalidData(format!(
            "JXL inverse RCT: channel sizes ({}, {}, {}) inconsistent with dim ({}x{})",
            a_buf.len(),
            b_buf.len(),
            c_buf.len(),
            d0.width,
            d0.height
        )));
    }
    for i in 0..n {
        let a = a_buf[i] as i64;
        let mut b = b_buf[i] as i64;
        let mut c = c_buf[i] as i64;
        let (d, e, f);
        if kind == 6 {
            // YCgCo per Listing L.3:
            //   tmp = A - (C >> 1);
            //   E = C + tmp;
            //   F = tmp - (B >> 1);
            //   D = F + B;
            let tmp = a.wrapping_sub(c >> 1);
            e = c.wrapping_add(tmp);
            f = tmp.wrapping_sub(b >> 1);
            d = f.wrapping_add(b);
        } else {
            if (kind & 1) == 1 {
                c = c.wrapping_add(a);
            }
            if (kind >> 1) == 1 {
                b = b.wrapping_add(a);
            }
            if (kind >> 1) == 2 {
                let avg = a.wrapping_add(c) >> 1;
                b = b.wrapping_add(avg);
            }
            d = a;
            e = b;
            f = c;
        }
        // Pack into V[] per Listing L.3:
        //   V[permutation Umod 3] = D
        //   V[(permutation+1+(permutation Idiv 3)) Umod 3] = E
        //   V[(permutation+2-(permutation Idiv 3)) Umod 3] = F
        let v_idx_d = permutation % 3;
        let v_idx_e = (permutation + 1 + (permutation / 3)) % 3;
        let v_idx_f = (permutation + 2 - (permutation / 3)) % 3;
        let mut v = [0i64; 3];
        v[v_idx_d] = d;
        v[v_idx_e] = e;
        v[v_idx_f] = f;
        a_buf[i] = clamp_i32(v[0])?;
        b_buf[i] = clamp_i32(v[1])?;
        c_buf[i] = clamp_i32(v[2])?;
    }
    img.channels[begin_c] = a_buf;
    img.channels[begin_c + 1] = b_buf;
    img.channels[begin_c + 2] = c_buf;
    Ok(())
}

fn clamp_i32(v: i64) -> Result<i32> {
    if (i32::MIN as i64..=i32::MAX as i64).contains(&v) {
        Ok(v as i32)
    } else {
        Err(Error::InvalidData(format!(
            "JXL inverse transform: sample value {v} out of i32 range"
        )))
    }
}

/// Listing L.6 — inverse palette transform.
///
/// `begin_c` is the position of the index channel after the palette
/// transform was applied (the meta-channel sits at index 0 of the
/// channel list; `begin_c` is the original `begin_c+1` shifted by the
/// meta-channel insertion, i.e. `begin_c_in_decoded = begin_c + 1`).
/// Per the spec we have `first = begin_c + 1` where `begin_c` is the
/// original transform parameter, and the meta-channel is at index 0.
fn inverse_palette(
    img: &mut ModularImage,
    begin_c: usize,
    num_c: usize,
    nb_colours: i64,
    nb_deltas: i64,
    d_pred: u32,
    bit_depth: u32,
) -> Result<()> {
    // Per L.5: the meta-channel is at the very beginning of the channel
    // list (channel[0]). The index channel is at `begin_c + 1` in the
    // decoded list (because the meta-channel insertion shifted
    // everything by one).
    let first = begin_c + 1;
    if first >= img.channels.len() {
        return Err(Error::InvalidData(format!(
            "JXL inverse Palette: first index {first} >= channel count {}",
            img.channels.len()
        )));
    }
    if num_c == 0 {
        return Err(Error::InvalidData("JXL inverse Palette: num_c == 0".into()));
    }
    if img.descs[0].width as i64 != nb_colours || img.descs[0].height as i64 != num_c as i64 {
        return Err(Error::InvalidData(format!(
            "JXL inverse Palette: meta-channel dims {}x{} mismatch nb_colours={} num_c={}",
            img.descs[0].width, img.descs[0].height, nb_colours, num_c
        )));
    }
    let idx_desc = img.descs[first];
    let w = idx_desc.width;
    let h = idx_desc.height;

    // Snapshot the meta-channel so we can mutate the data channels
    // without overlapping borrows.
    let meta = img.channels[0].clone();
    let meta_w = img.descs[0].width as usize;
    if meta.len() != meta_w * (num_c) {
        return Err(Error::InvalidData(format!(
            "JXL inverse Palette: meta-channel size {} != nb_colours*num_c = {}",
            meta.len(),
            meta_w * num_c
        )));
    }
    let index_channel = img.channels[first].clone();

    // Insert (num_c - 1) copies of the index channel right after `first`.
    for _ in 1..num_c {
        img.channels.insert(first + 1, index_channel.clone());
        img.descs.insert(first + 1, idx_desc);
    }

    // Reconstruct each channel in [first .. first + num_c).
    let predict_needed = nb_deltas > 0;
    // clippy::needless_range_loop fires because we use `c` to index
    // K_DELTA_PALETTE[row][c]; that's the spec's c-channel selector,
    // not a sequential iterator over the palette table.
    #[allow(clippy::needless_range_loop)]
    for c in 0..num_c {
        let target_idx = first + c;
        // Snapshot dims (descs index unchanged for this channel).
        let dw = img.descs[target_idx].width;
        let dh = img.descs[target_idx].height;
        if dw != w || dh != h {
            return Err(Error::InvalidData(
                "JXL inverse Palette: per-channel dim mismatch after duplication".into(),
            ));
        }
        // We must compute one channel at a time, with prediction looking
        // at already-written samples in this same channel.
        let mut out = vec![0i32; (w as usize) * (h as usize)];
        for y in 0..h {
            for x in 0..w {
                let pix = (y as usize) * (w as usize) + (x as usize);
                let index = index_channel[pix] as i64;
                let is_delta = index < nb_deltas;
                let value: i64 = if (0..nb_colours).contains(&index) {
                    // channel[0](index, c) — meta-channel sample at (index, c).
                    let mi = (c) * meta_w + (index as usize);
                    meta[mi] as i64
                } else if index >= nb_colours {
                    let mut k = index - nb_colours;
                    if k < 64 {
                        let bd = bit_depth.max(1);
                        let max_val = (1i64 << bd) - 1;
                        let term = ((k >> (2 * c)).rem_euclid(4)) * max_val / 4;
                        let bias = 1i64 << bd.saturating_sub(3);
                        term + bias
                    } else {
                        k -= 64;
                        for _ in 0..c {
                            k /= 5;
                        }
                        let bd = bit_depth.max(1);
                        let max_val = (1i64 << bd) - 1;
                        (k.rem_euclid(5)) * max_val / 4
                    }
                } else if c < 3 {
                    // index < 0 branch
                    let neg_index = (-index - 1).rem_euclid(143);
                    let row = ((neg_index + 1) >> 1) as usize;
                    let mut v = K_DELTA_PALETTE[row][c] as i64;
                    if (neg_index & 1) == 0 {
                        v = -v;
                    }
                    if bit_depth > 8 {
                        v <<= bit_depth - 8;
                    }
                    v
                } else {
                    0
                };
                let mut sample = value;
                if is_delta && predict_needed {
                    // Need prediction(x, y, d_pred) on the OUTPUT
                    // channel under construction. We pass a temporary
                    // ModularImage view containing only this channel
                    // built so far.
                    let pred = predict_palette(&out, w, h, x as i32, y as i32, d_pred)?;
                    sample = sample.wrapping_add(pred);
                }
                let s32 = clamp_i32(sample)?;
                out[pix] = s32;
            }
        }
        img.channels[target_idx] = out;
    }

    // Remove the meta-channel.
    img.channels.remove(0);
    img.descs.remove(0);

    Ok(())
}

/// Listing C.16 prediction operating on a freshly-constructed channel
/// during palette inverse. Out-of-range reads return 0.
fn predict_palette(out: &[i32], w: u32, h: u32, x: i32, y: i32, predictor: u32) -> Result<i64> {
    let get = |xi: i32, yi: i32| -> i32 {
        if xi < 0 || yi < 0 || (xi as u32) >= w || (yi as u32) >= h {
            0
        } else {
            out[(yi as usize) * (w as usize) + (xi as usize)]
        }
    };
    let left = if x > 0 {
        get(x - 1, y)
    } else if y > 0 {
        get(x, y - 1)
    } else {
        0
    };
    let top = if y > 0 { get(x, y - 1) } else { left };
    let topleft = if x > 0 && y > 0 {
        get(x - 1, y - 1)
    } else {
        left
    };
    let topright = if (x + 1) < w as i32 && y > 0 {
        get(x + 1, y - 1)
    } else {
        top
    };
    let topright2 = if (x + 2) < w as i32 && y > 0 {
        get(x + 2, y - 1)
    } else {
        topright
    };
    let leftleft = if x > 1 { get(x - 2, y) } else { left };
    let toptop = if y > 1 { get(x, y - 2) } else { top };
    let grad = top.wrapping_add(left).wrapping_sub(topleft);
    let v: i64 = match predictor {
        0 => 0,
        1 => left as i64,
        2 => top as i64,
        3 => ((left as i64) + (top as i64)).div_euclid(2),
        4 => {
            if (grad - left).abs() < (grad - top).abs() {
                left as i64
            } else {
                top as i64
            }
        }
        5 => median3(grad, left, top) as i64,
        6 => {
            return Err(Error::Unsupported(
                "JXL inverse Palette: d_pred=6 (weighted predictor) not yet supported".into(),
            ));
        }
        7 => topright as i64,
        8 => topleft as i64,
        9 => leftleft as i64,
        10 => ((left as i64) + (topleft as i64)).div_euclid(2),
        11 => ((topleft as i64) + (top as i64)).div_euclid(2),
        12 => ((top as i64) + (topright as i64)).div_euclid(2),
        13 => (6 * top as i64 - 2 * toptop as i64
            + 7 * left as i64
            + leftleft as i64
            + topright2 as i64
            + 3 * topright as i64
            + 8)
        .div_euclid(16),
        _ => {
            return Err(Error::InvalidData(format!(
                "JXL inverse Palette: d_pred {predictor} out of range"
            )));
        }
    };
    Ok(v)
}

fn median3(a: i32, b: i32, c: i32) -> i32 {
    if (a <= b && b <= c) || (c <= b && b <= a) {
        b
    } else if (b <= a && a <= c) || (c <= a && a <= b) {
        a
    } else {
        c
    }
}

/// Listing I.18 — inverse Squeeze step. Replaces channel[c] (the
/// "average") + channel[r] (the "residual") with one reconstructed
/// channel of doubled extent in the squeeze direction.
fn inverse_squeeze_step(img: &mut ModularImage, sp: &SqueezeParams) -> Result<()> {
    let begin = sp.begin_c as usize;
    let end = begin + sp.num_c as usize - 1;
    if sp.num_c == 0 {
        return Err(Error::InvalidData("JXL inverse Squeeze: num_c == 0".into()));
    }
    if end >= img.channels.len() {
        return Err(Error::InvalidData(format!(
            "JXL inverse Squeeze: end {end} >= channel count {}",
            img.channels.len()
        )));
    }
    // Per Listing I.18: r = sp.in_place ? end+1 : channel.size() + begin - end - 1.
    // We process c = begin..=end in order; after each iteration we
    // remove one residual at index r (which in the in_place case is
    // always end+1 — re-evaluated each iteration since the list shrinks
    // by one and end shifts). For the !in_place case, the residual sits
    // at the end of the list at offset (begin - end - 1) from
    // channel.size() — i.e., for num_c residuals appended at the tail.
    for c in begin..=end {
        let r = if sp.in_place {
            end + 1
        } else {
            img.channels.len() + begin - end - 1
        };
        if r >= img.channels.len() {
            return Err(Error::InvalidData(format!(
                "JXL inverse Squeeze: residual idx {r} >= channel count {}",
                img.channels.len()
            )));
        }
        let avg_desc = img.descs[c];
        let res_desc = img.descs[r];
        let (out_desc, out_data) = if sp.horizontal {
            // Output dims: width = avg.width + res.width, height = avg.height.
            if avg_desc.height != res_desc.height {
                return Err(Error::InvalidData(format!(
                    "JXL inverse Squeeze (horiz): avg.h {} != res.h {}",
                    avg_desc.height, res_desc.height
                )));
            }
            if !(avg_desc.width == res_desc.width || avg_desc.width == res_desc.width + 1) {
                return Err(Error::InvalidData(format!(
                    "JXL inverse Squeeze (horiz): avg.w {} not res.w {} or res.w+1",
                    avg_desc.width, res_desc.width
                )));
            }
            let new_w = avg_desc.width + res_desc.width;
            let new_h = avg_desc.height;
            if new_w > MAX_DIM || new_h > MAX_DIM {
                return Err(Error::InvalidData(format!(
                    "JXL inverse Squeeze: new dim {new_w}x{new_h} exceeds cap {MAX_DIM}"
                )));
            }
            let data = horiz_isqueeze(
                &img.channels[c],
                avg_desc.width,
                &img.channels[r],
                res_desc.width,
                new_h,
            );
            let mut new_desc = avg_desc;
            new_desc.width = new_w;
            // Decrement hshift back toward 0 — minimum -1 to allow the
            // palette meta-channel sentinel to remain unchanged
            // elsewhere; squeeze should never push hshift below 0 from a
            // ≥0 starting point.
            new_desc.hshift = avg_desc.hshift.saturating_sub(1);
            (new_desc, data)
        } else {
            if avg_desc.width != res_desc.width {
                return Err(Error::InvalidData(format!(
                    "JXL inverse Squeeze (vert): avg.w {} != res.w {}",
                    avg_desc.width, res_desc.width
                )));
            }
            if !(avg_desc.height == res_desc.height || avg_desc.height == res_desc.height + 1) {
                return Err(Error::InvalidData(format!(
                    "JXL inverse Squeeze (vert): avg.h {} not res.h {} or res.h+1",
                    avg_desc.height, res_desc.height
                )));
            }
            let new_w = avg_desc.width;
            let new_h = avg_desc.height + res_desc.height;
            if new_w > MAX_DIM || new_h > MAX_DIM {
                return Err(Error::InvalidData(format!(
                    "JXL inverse Squeeze: new dim {new_w}x{new_h} exceeds cap {MAX_DIM}"
                )));
            }
            let data = vert_isqueeze(
                &img.channels[c],
                avg_desc.height,
                &img.channels[r],
                res_desc.height,
                new_w,
            );
            let mut new_desc = avg_desc;
            new_desc.height = new_h;
            new_desc.vshift = avg_desc.vshift.saturating_sub(1);
            (new_desc, data)
        };
        img.channels[c] = out_data;
        img.descs[c] = out_desc;
        // Remove residual at r.
        img.channels.remove(r);
        img.descs.remove(r);
    }
    Ok(())
}

/// Listing I.20 — horizontal inverse squeeze.
fn horiz_isqueeze(input_1: &[i32], w1: u32, input_2: &[i32], w2: u32, h: u32) -> Vec<i32> {
    let out_w = w1 + w2;
    let mut out = vec![0i32; (out_w as usize) * (h as usize)];
    let row_in1 = w1 as usize;
    let row_in2 = w2 as usize;
    let row_out = out_w as usize;
    for y in 0..h as usize {
        for x in 0..w2 as usize {
            let avg = input_1[y * row_in1 + x] as i64;
            let residu = input_2[y * row_in2 + x] as i64;
            let next_avg = if (x as u32) + 1 < w1 {
                input_1[y * row_in1 + x + 1] as i64
            } else {
                avg
            };
            let left = if x > 0 {
                out[y * row_out + (x << 1) - 1] as i64
            } else {
                avg
            };
            let diff = residu.wrapping_add(tendency(left, avg, next_avg));
            let sign_diff = diff.signum();
            let first = (2i64 * avg + diff - sign_diff * (diff & 1)) >> 1;
            let second = first - diff;
            out[y * row_out + 2 * x] = first as i32;
            out[y * row_out + 2 * x + 1] = second as i32;
        }
        if w1 > w2 {
            // Final column: copy input_1[w2, y] to output[2*w2, y].
            out[y * row_out + 2 * w2 as usize] = input_1[y * row_in1 + w2 as usize];
        }
    }
    out
}

/// Listing I.22 — vertical inverse squeeze.
fn vert_isqueeze(input_1: &[i32], h1: u32, input_2: &[i32], h2: u32, w: u32) -> Vec<i32> {
    let out_h = h1 + h2;
    let mut out = vec![0i32; (out_h as usize) * (w as usize)];
    let row = w as usize;
    for y in 0..h2 as usize {
        for x in 0..w as usize {
            let avg = input_1[y * row + x] as i64;
            let residu = input_2[y * row + x] as i64;
            let next_avg = if (y as u32) + 1 < h1 {
                input_1[(y + 1) * row + x] as i64
            } else {
                avg
            };
            let top = if y > 0 {
                out[((y << 1) - 1) * row + x] as i64
            } else {
                avg
            };
            let diff = residu.wrapping_add(tendency(top, avg, next_avg));
            let sign_diff = diff.signum();
            let first = (2i64 * avg + diff - sign_diff * (diff & 1)) >> 1;
            let second = first - diff;
            out[(2 * y) * row + x] = first as i32;
            out[(2 * y + 1) * row + x] = second as i32;
        }
    }
    if h1 > h2 {
        for x in 0..w as usize {
            out[(2 * h2 as usize) * row + x] = input_1[(h2 as usize) * row + x];
        }
    }
    out
}

/// Listing I.21 — `tendency(A, B, C)` used by both squeeze directions.
fn tendency(a: i64, b: i64, c: i64) -> i64 {
    let mut x = (4 * a - 3 * c - b + 6).div_euclid(12);
    if a >= b && b >= c {
        if x - (x & 1) > 2 * (a - b) {
            x = 2 * (a - b) + 1;
        }
        if x + (x & 1) > 2 * (b - c) {
            x = 2 * (b - c);
        }
        x
    } else if a <= b && b <= c {
        if x + (x & 1) < 2 * (a - b) {
            x = 2 * (a - b) - 1;
        }
        if x - (x & 1) < 2 * (b - c) {
            x = 2 * (b - c);
        }
        x
    } else {
        0
    }
}

/// Annex L.5 `kDeltaPalette[72][3]`.
#[allow(clippy::needless_range_loop)]
const K_DELTA_PALETTE: [[i32; 3]; 72] = [
    [0, 0, 0],
    [4, 4, 4],
    [11, 0, 0],
    [0, 0, -13],
    [0, -12, 0],
    [-10, -10, -10],
    [-18, -18, -18],
    [-27, -27, -27],
    [-18, -18, 0],
    [0, 0, -32],
    [-32, 0, 0],
    [-37, -37, -37],
    [0, -32, -32],
    [24, 24, 45],
    [50, 50, 50],
    [-45, -24, -24],
    [-24, -45, -45],
    [0, -24, -24],
    [-34, -34, 0],
    [-24, 0, -24],
    [-45, -45, -24],
    [64, 64, 64],
    [-32, 0, -32],
    [0, -32, 0],
    [-32, 0, 32],
    [-24, -45, -24],
    [45, 24, 45],
    [24, -24, -45],
    [-45, -24, 24],
    [80, 80, 80],
    [64, 0, 0],
    [0, 0, -64],
    [0, -64, -64],
    [-24, -24, 45],
    [96, 96, 96],
    [64, 64, 0],
    [45, -24, -24],
    [34, -34, 0],
    [112, 112, 112],
    [24, -45, -45],
    [45, 45, -24],
    [0, -32, 32],
    [24, -24, 45],
    [0, 96, 96],
    [45, -24, 24],
    [24, -45, -24],
    [-24, -45, 24],
    [0, -64, 0],
    [96, 0, 0],
    [128, 128, 128],
    [64, 0, 64],
    [144, 144, 144],
    [96, 96, 0],
    [-36, -36, 36],
    [45, -24, -45],
    [45, -45, -24],
    [0, 0, -96],
    [0, 128, 128],
    [0, 96, 0],
    [45, 24, -45],
    [-128, 0, 0],
    [24, -45, 24],
    [-45, 24, -45],
    [64, 0, -64],
    [64, -64, -64],
    [96, 0, 96],
    [45, -45, 24],
    [24, 45, -45],
    [64, 64, -64],
    [128, 128, 0],
    [0, 0, -128],
    [-24, 45, -45],
];

#[cfg(test)]
mod tests {
    use super::*;

    fn ch(w: u32, h: u32) -> ChannelDesc {
        ChannelDesc {
            width: w,
            height: h,
            hshift: 0,
            vshift: 0,
        }
    }

    #[test]
    fn rct_type_zero_identity_after_inverse() {
        // type=0, permutation=0:
        //   if (0&1)==0 ⇒ no change to C
        //   (0>>1)==0 ⇒ no change to B
        //   D=A, E=B, F=C
        //   V[0]=D, V[1]=E, V[2]=F → identity
        let mut img = ModularImage {
            channels: vec![vec![10, 20], vec![30, 40], vec![50, 60]],
            descs: vec![ch(2, 1), ch(2, 1), ch(2, 1)],
        };
        inverse_rct(&mut img, 0, 0).unwrap();
        assert_eq!(img.channels[0], vec![10, 20]);
        assert_eq!(img.channels[1], vec![30, 40]);
        assert_eq!(img.channels[2], vec![50, 60]);
    }

    #[test]
    fn squeeze_horiz_identity_on_constant_input() {
        // 4x1 channel of all-128 → squeeze (avg=128, residu=0)  →
        // inverse should reconstruct 128, 128, 128, 128.
        let avg = vec![128i32, 128];
        let res = vec![0i32, 0];
        let out = horiz_isqueeze(&avg, 2, &res, 2, 1);
        assert_eq!(out, vec![128, 128, 128, 128]);
    }

    #[test]
    fn squeeze_vert_identity_on_constant_input() {
        // 1x4 channel of all-128 split into two 1x2 channels (avg + res).
        // input_1 (h1=2, w=1) and input_2 (h2=2, w=1) → out (h=4, w=1).
        let avg = vec![128i32, 128];
        let res = vec![0i32, 0];
        let out = vert_isqueeze(&avg, 2, &res, 2, 1);
        assert_eq!(out, vec![128, 128, 128, 128]);
    }

    #[test]
    fn read_transforms_zero_returns_empty() {
        // 2-bit selector = 0 ⇒ Val(0) ⇒ nb_transforms = 0.
        let bytes = vec![0x00u8];
        let mut br = BitReader::new(&bytes);
        let v = read_transforms(&mut br).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn read_transforms_one_rct_identity() {
        // nb_transforms: selector=1 ⇒ Val(1).
        // tr (Enum): selector=0 ⇒ Val(0) = kRCT.
        // begin_c (U32 with Bits(3) at sel=0): selector=0 ⇒ 3 bits = 0.
        // rct_type (U32 with Val(6) at sel=0): selector=0 ⇒ Val(6).
        //
        // Bit layout LSB-first:
        //   nb_transforms_sel=01 (2 bits)
        //   tr_sel=00 (2 bits) — selector then no extra (Val(0))
        //   begin_c_sel=00 (2) + 3 bits payload 000
        //   rct_type_sel=00 (2)
        //   total = 2+2+2+3+2 = 11 bits → "10000000 0000000" packed.
        // bit0..1 = "10" (sel=1) → byte0 bit0=1, bit1=0 ⇒ 0b00000001
        // bit2..3 = "00" (sel=0)
        // bit4..5 = "00"
        // bit6..8 = "000" (3-bit begin_c)
        // bit9..10 = "00"
        // → byte 0 = 0x01, byte 1 = 0x00.
        let bytes = vec![0x01u8, 0x00u8];
        let mut br = BitReader::new(&bytes);
        let v = read_transforms(&mut br).unwrap();
        assert_eq!(v.len(), 1);
        match &v[0] {
            TransformInfo::Rct { begin_c, rct_type } => {
                assert_eq!(*begin_c, 0);
                assert_eq!(*rct_type, 6);
            }
            _ => panic!("expected RCT"),
        }
    }

    #[test]
    fn apply_forward_rct_no_change() {
        let initial = vec![ch(8, 8), ch(8, 8), ch(8, 8)];
        let trs = vec![TransformInfo::Rct {
            begin_c: 0,
            rct_type: 0,
        }];
        let (out, nb_meta) = apply_transforms_forward(&initial, &trs).unwrap();
        assert_eq!(nb_meta, 0);
        assert_eq!(out.len(), 3);
        for c in &out {
            assert_eq!(c.width, 8);
            assert_eq!(c.height, 8);
        }
    }

    #[test]
    fn apply_forward_palette_inserts_meta() {
        let initial = vec![ch(8, 8)];
        let trs = vec![TransformInfo::Palette {
            begin_c: 0,
            num_c: 1,
            nb_colours: 4,
            nb_deltas: 0,
            d_pred: 0,
        }];
        let (out, nb_meta) = apply_transforms_forward(&initial, &trs).unwrap();
        assert_eq!(nb_meta, 1);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].width, 4);
        assert_eq!(out[0].height, 1);
        assert_eq!(out[0].hshift, -1);
        assert_eq!(out[1].width, 8);
        assert_eq!(out[1].height, 8);
    }

    #[test]
    fn inverse_palette_constant_grey() {
        // 1-channel grey image of all-0 indices into a 1-colour palette
        // (palette[0] = 128). Inverse should produce all-128.
        let mut img = ModularImage {
            channels: vec![vec![128i32], vec![0i32; 64]],
            descs: vec![
                ChannelDesc {
                    width: 1,
                    height: 1,
                    hshift: -1,
                    vshift: -1,
                },
                ch(8, 8),
            ],
        };
        inverse_palette(&mut img, 0, 1, 1, 0, 0, 8).unwrap();
        assert_eq!(img.channels.len(), 1);
        assert_eq!(img.channels[0].len(), 64);
        for v in &img.channels[0] {
            assert_eq!(*v, 128);
        }
    }
}
