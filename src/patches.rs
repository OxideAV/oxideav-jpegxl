//! Patches image feature — ISO/IEC FDIS 18181-1:2021 §C.4.5 (patch
//! dictionary decode) and §K.2 (rendering).
//!
//! A JXL frame whose `frame_header.flags` sets `kPatches` (§C.2.6,
//! `0x02`) carries a dictionary of small rectangular patches, each taken
//! from a previously stored reference frame (`Reference[ref]`, §C.2) and
//! blended onto the current frame at one or more positions — after the
//! restoration filters (Annex J) and before splines (§K.3) and noise
//! (§K.4), in the colour space **before** the inverse colour transforms
//! of Annex L but after upsampling (§K.2).
//!
//! ## Wire structure (§C.4.5)
//!
//! The bundle is a single §D.3 entropy stream with **10 clustered
//! distributions**; `ReadHybridVarLenUint(x)` reads one hybrid var-len
//! integer with distribution `D[x]`. `num_patches` comes first
//! (context 0), then Listing C.2 per patch: `ref` (1), `x0` / `y0` (3),
//! `width` / `height` (2, biased by +1), `count` (7, biased by +1), the
//! first position (4), subsequent positions as `UnpackSigned` deltas
//! (5), and per position one blend-info per (colour, extra-channel)
//! plane group: `mode`, `alpha_channel` (8, only when the image has
//! more than one alpha channel and the mode consumes alpha), `clamp`
//! (9, only for the modes Table K.1 defines clamping for).
//!
//! ### Listing C.2 `mode` context — open question (round 441)
//!
//! The printed listing reads `mode = ReadHybridVarLenUint(5)` — the same
//! context as the position deltas — which leaves context 6 of the
//! declared 10 distributions entirely unused; the natural suspicion is
//! a typo for context 6. The question is **unarbitrable on the
//! available wire evidence**: every locally generated patches
//! codestream (several sizes and dictionary shapes, all from an
//! independent encoder invoked black-box) carries the §D.3.5 cluster
//! map `[0, 1, 2, 3, 0, 2, 2, 4, 2, 2]`, which assigns contexts 5 and 6
//! to the *same* cluster — the two readings decode identically, and
//! both close the §D.3.3 ANS final-state invariant. The printed reading
//! (context 5) is therefore followed; [`set_patch_mode_ctx_override`]
//! keeps the alternative reachable and a CI test pins the equivalence
//! on the committed fixture so a future specimen that separates the
//! clusters arbitrates immediately.
//!
//! ## Rendering (§K.2)
//!
//! For every patch `i`, position `j` and plane group `c` in
//! `[0, num_extra_channels]`, `new_sample` (the reference-frame sample)
//! is blended over `old_sample` (the current frame) per Table K.1;
//! `c == 0` covers the colour channels, `c >= 1` the extra channel
//! `c - 1`. The blending happens in the pre-colour-transform domain, so
//! the reference frames consumed here are the **pre-CT** recordings
//! (§C.2 `save_before_ct`; see [`ReferenceFrames`]).

use crate::bitreader::{unpack_signed, BitReader};
use oxideav_core::{Error, Result};
use std::cell::Cell;

/// §C.4.5: "The decoder reads a set of 10 clustered distributions".
pub const PATCH_NUM_CONTEXTS: usize = 10;

/// Table K.1 — `PatchBlendMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchBlendMode {
    /// `sample = old_sample`.
    None,
    /// `sample = new_sample` (Table C.8 kReplace).
    Replace,
    /// `sample = old_sample + new_sample` (Table C.8 kAdd).
    Add,
    /// `sample = old_sample × new_sample` (Table C.8 kMul).
    Mul,
    /// Table C.8 kBlend — new over old.
    BlendAbove,
    /// Table C.8 kBlend with the roles of new/old swapped.
    BlendBelow,
    /// Table C.8 kAlphaWeightedAdd.
    AlphaWeightedAddAbove,
    /// Table C.8 kAlphaWeightedAdd with the roles swapped.
    AlphaWeightedAddBelow,
}

impl PatchBlendMode {
    /// Decode the Table K.1 value; `[[mode < 8]]` per Listing C.2.
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => PatchBlendMode::None,
            1 => PatchBlendMode::Replace,
            2 => PatchBlendMode::Add,
            3 => PatchBlendMode::Mul,
            4 => PatchBlendMode::BlendAbove,
            5 => PatchBlendMode::BlendBelow,
            6 => PatchBlendMode::AlphaWeightedAddAbove,
            7 => PatchBlendMode::AlphaWeightedAddBelow,
            _ => {
                return Err(Error::InvalidData(format!(
                    "JXL patches: blend mode {v} out of range (Listing C.2: mode < 8)"
                )))
            }
        })
    }

    /// Listing C.2: the modes whose blend consumes an alpha channel
    /// (kBlendAbove / kBlendBelow / kAlphaWeightedAddAbove /
    /// kAlphaWeightedAddBelow).
    pub fn uses_alpha(self) -> bool {
        matches!(
            self,
            PatchBlendMode::BlendAbove
                | PatchBlendMode::BlendBelow
                | PatchBlendMode::AlphaWeightedAddAbove
                | PatchBlendMode::AlphaWeightedAddBelow
        )
    }

    /// Listing C.2: the modes that carry a `clamp` field (the alpha
    /// modes plus kMul).
    pub fn has_clamp(self) -> bool {
        self.uses_alpha() || self == PatchBlendMode::Mul
    }
}

/// One per-(position, plane-group) blend descriptor (Listing C.2).
#[derive(Debug, Clone, Copy)]
pub struct PatchBlending {
    pub mode: PatchBlendMode,
    /// Extra-channel index of "the alpha channel" (Table K.1); 0 when
    /// not on the wire.
    pub alpha_channel: u32,
    /// Clamp alpha to `[0, 1]` before blending (Table K.1 prose).
    pub clamp: bool,
}

/// One decoded patch (§C.4.5).
#[derive(Debug, Clone)]
pub struct Patch {
    /// `Reference[ref]` slot the patch samples come from.
    pub reference: u32,
    /// Top-left corner of the patch in the reference frame.
    pub x0: u32,
    pub y0: u32,
    /// Patch dimensions (wire value + 1).
    pub width: u32,
    pub height: u32,
    /// The `count` positions the patch is blended at (frame
    /// coordinates; §C.4.5 requires the patch rectangle at each to be
    /// fully contained within the frame).
    pub positions: Vec<(i64, i64)>,
    /// `positions.len()` rows of `num_extra_channels + 1` blend
    /// descriptors — index 0 is the colour-channel group, index `k >= 1`
    /// the extra channel `k - 1`.
    pub blendings: Vec<Vec<PatchBlending>>,
}

thread_local! {
    /// Arbitration hook for the Listing C.2 `mode` context (see the
    /// module-level open-question note). `None` → the printed default
    /// (context 5).
    static PATCH_MODE_CTX_OVERRIDE: Cell<Option<u32>> = const { Cell::new(None) };
}

/// The Listing C.2 `mode` context as printed: context **5**. See the
/// module-level open-question note — every available specimen clusters
/// contexts 5 and 6 together, so the printed reading stands until a
/// stream separates them.
pub const PATCH_MODE_CTX_DEFAULT: u32 = 5;

/// Force the Listing C.2 `mode` context for the current thread
/// (`None` restores the printed default). Arbitration hook for the
/// module-level open question; CI pins the 5-vs-6 equivalence on the
/// committed fixture.
#[doc(hidden)] // internal: per-thread arbitration hook for the round-441 mode-context question
pub fn set_patch_mode_ctx_override(ctx: Option<u32>) {
    PATCH_MODE_CTX_OVERRIDE.with(|c| c.set(ctx));
}

fn patch_mode_ctx() -> u32 {
    PATCH_MODE_CTX_OVERRIDE
        .with(|c| c.get())
        .unwrap_or(PATCH_MODE_CTX_DEFAULT)
}

/// FDIS §C.4.5 — decode the patch dictionary from an abstract
/// `ReadHybridVarLenUint(ctx)` source (Listing C.2).
///
/// `num_extra_channels` sizes each position's blend-descriptor row;
/// `num_alpha_channels` gates the `alpha_channel` field ("if there is
/// more than 1 alpha channel"). `frame_width` / `frame_height` validate
/// the §C.4.5 containment assertion on every position.
pub fn decode_patches_with<F>(
    mut read_uint: F,
    num_extra_channels: u32,
    num_alpha_channels: usize,
    frame_width: u32,
    frame_height: u32,
) -> Result<Vec<Patch>>
where
    F: FnMut(u32) -> Result<u32>,
{
    let mode_ctx = patch_mode_ctx();
    let num_patches = read_uint(0)? as usize;
    let mut out: Vec<Patch> = Vec::with_capacity(num_patches.min(1024));
    for _ in 0..num_patches {
        let reference = read_uint(1)?;
        if reference >= 4 {
            return Err(Error::InvalidData(format!(
                "JXL patches: reference slot {reference} out of range (Reference[0..4])"
            )));
        }
        let x0 = read_uint(3)?;
        let y0 = read_uint(3)?;
        let width = read_uint(2)?
            .checked_add(1)
            .ok_or_else(|| Error::InvalidData("JXL patches: width overflow".into()))?;
        let height = read_uint(2)?
            .checked_add(1)
            .ok_or_else(|| Error::InvalidData("JXL patches: height overflow".into()))?;
        if width > frame_width || height > frame_height {
            return Err(Error::InvalidData(format!(
                "JXL patches: {width}×{height} patch exceeds the {frame_width}×{frame_height} frame"
            )));
        }
        let count = read_uint(7)? as usize + 1;
        let mut positions = Vec::with_capacity(count.min(1 << 16));
        let mut blendings = Vec::with_capacity(count.min(1 << 16));
        let (mut last_x, mut last_y): (i64, i64) = (0, 0);
        for j in 0..count {
            let (x, y) = if j == 0 {
                (read_uint(4)? as i64, read_uint(4)? as i64)
            } else {
                (
                    unpack_signed(read_uint(5)?) as i64 + last_x,
                    unpack_signed(read_uint(5)?) as i64 + last_y,
                )
            };
            // §C.4.5: "the width × height rectangle with top-left
            // coordinates (x, y) is fully contained within the frame".
            if x < 0
                || y < 0
                || x + width as i64 > frame_width as i64
                || y + height as i64 > frame_height as i64
            {
                return Err(Error::InvalidData(format!(
                    "JXL patches: {width}×{height} patch at ({x}, {y}) not contained \
                     within the {frame_width}×{frame_height} frame"
                )));
            }
            positions.push((x, y));
            last_x = x;
            last_y = y;
            let mut row = Vec::with_capacity(num_extra_channels as usize + 1);
            for _k in 0..=num_extra_channels {
                let mode = PatchBlendMode::from_u32(read_uint(mode_ctx)?)?;
                let alpha_channel = if mode.uses_alpha() && num_alpha_channels > 1 {
                    read_uint(8)?
                } else {
                    0
                };
                let clamp = if mode.has_clamp() {
                    read_uint(9)? != 0
                } else {
                    false
                };
                row.push(PatchBlending {
                    mode,
                    alpha_channel,
                    clamp,
                });
            }
            blendings.push(row);
        }
        out.push(Patch {
            reference,
            x0,
            y0,
            width,
            height,
            positions,
            blendings,
        });
    }
    Ok(out)
}

/// FDIS §C.4.5 — decode the patch dictionary from the codestream: the
/// §D.3 ten-distribution prelude, its ANS-state init (§C.3.2), the
/// Listing C.2 structure, then the §D.3.3 ANS final-state invariant as
/// a misparse guard (prefix-coded streams carry no ANS state and skip
/// the check).
pub fn decode_patches(
    br: &mut BitReader<'_>,
    num_extra_channels: u32,
    num_alpha_channels: usize,
    frame_width: u32,
    frame_height: u32,
) -> Result<Vec<Patch>> {
    use crate::modular_fdis::{decode_uint_in_with_dist_pub, EntropyStream};
    let mut entropy = EntropyStream::read(br, PATCH_NUM_CONTEXTS)?;
    entropy.read_ans_state_init(br)?;
    let mut hybrid = crate::ans::hybrid::HybridUintState::new(entropy.lz77, entropy.lz_len_conf);
    let patches = decode_patches_with(
        |ctx| decode_uint_in_with_dist_pub(&mut hybrid, &mut entropy, br, ctx, 0),
        num_extra_channels,
        num_alpha_channels,
        frame_width,
        frame_height,
    )?;
    if let Some(dec) = entropy.ans_state.as_ref() {
        if !dec.final_state() {
            return Err(Error::InvalidData(
                "JXL patches: ANS final-state invariant (D.3.3) failed after the \
                 §C.4.5 stream — misparse"
                    .into(),
            ));
        }
    }
    Ok(patches)
}

/// A frame recorded in the **pre-colour-transform** domain (§C.2
/// `save_before_ct`), as consumed by the §K.2 patch renderer: row-major
/// `f32` planes in the frame's pre-CT space (float XYB for
/// `xyb_encoded` frames; samples normalised to `[0, 1]` for integer
/// Modular frames with no colour transform).
#[derive(Debug, Clone)]
pub struct PreCtFrame {
    pub width: usize,
    pub height: usize,
    /// Colour planes first (1 for Grey, 3 otherwise), then extra
    /// channels in Annex G.1.3 order.
    pub planes: Vec<Vec<f32>>,
}

/// The four §C.2 `Reference[0..4]` slots in their pre-CT recordings.
/// Slots the stream never recorded (or recorded post-CT) are `None`;
/// a patch referencing such a slot is a precise error, never a silent
/// zero-fill (§C.4.5 requires the referenced samples to exist).
#[derive(Debug, Clone, Default)]
pub struct ReferenceFrames {
    pub slots: [Option<PreCtFrame>; 4],
}

/// FDIS §K.2 — blend every patch onto the current frame's pre-CT
/// colour planes.
///
/// `color` holds the frame's colour planes (1 for Grey, 3 otherwise),
/// each a row-major `width × height` buffer in the same pre-CT domain
/// the reference frames were recorded in. Extra-channel plane groups
/// (`c >= 1`) are accepted only with mode kNone — blending an extra
/// channel needs the frame's extra planes threaded through, which no
/// current caller carries; a stream that asks for it errs precisely.
/// Alpha-consuming colour modes likewise need an alpha plane and err
/// precisely (no staged or locally generated specimen exercises them).
pub fn render_patches(
    patches: &[Patch],
    refs: &ReferenceFrames,
    color: &mut [&mut [f32]],
    width: usize,
    height: usize,
) -> Result<()> {
    for (pi, patch) in patches.iter().enumerate() {
        let src = refs.slots[patch.reference as usize]
            .as_ref()
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "jxl patches: patch {pi} references Reference[{}] which holds no \
                 pre-colour-transform recording",
                    patch.reference
                ))
            })?;
        let (pw, ph) = (patch.width as usize, patch.height as usize);
        if patch.x0 as usize + pw > src.width || patch.y0 as usize + ph > src.height {
            return Err(Error::InvalidData(format!(
                "JXL patches: patch {pi} rect {}x{} at ({}, {}) exceeds the {}x{} \
                 reference frame",
                pw, ph, patch.x0, patch.y0, src.width, src.height
            )));
        }
        if src.planes.len() < color.len() {
            return Err(Error::InvalidData(format!(
                "JXL patches: reference frame carries {} plane(s), frame has {} colour \
                 plane(s)",
                src.planes.len(),
                color.len()
            )));
        }
        for (j, &(px, py)) in patch.positions.iter().enumerate() {
            let (px, py) = (px as usize, py as usize);
            if px + pw > width || py + ph > height {
                return Err(Error::InvalidData(format!(
                    "JXL patches: patch {pi} position {j} ({px}, {py}) not contained \
                     within the {width}x{height} frame"
                )));
            }
            let row = &patch.blendings[j];
            // Colour plane group (c == 0): one mode covers every colour
            // channel (§K.2 "d iterates over the three colour channels").
            let blend = row[0];
            match blend.mode {
                PatchBlendMode::None => {}
                PatchBlendMode::Replace | PatchBlendMode::Add | PatchBlendMode::Mul => {
                    for (d, plane) in color.iter_mut().enumerate() {
                        let sp = &src.planes[d];
                        for iy in 0..ph {
                            let s_row = (patch.y0 as usize + iy) * src.width + patch.x0 as usize;
                            let d_row = (py + iy) * width + px;
                            for ix in 0..pw {
                                let new_sample = sp[s_row + ix];
                                let old = &mut plane[d_row + ix];
                                *old = match blend.mode {
                                    PatchBlendMode::Replace => new_sample,
                                    PatchBlendMode::Add => *old + new_sample,
                                    PatchBlendMode::Mul => *old * new_sample,
                                    _ => unreachable!(),
                                };
                            }
                        }
                    }
                }
                m => {
                    return Err(Error::Unsupported(format!(
                        "jxl patches: colour blend mode {m:?} needs an alpha plane, \
                         which this decode path does not carry yet"
                    )));
                }
            }
            // Extra-channel plane groups: only kNone is representable
            // until extra planes are threaded through.
            for (k, b) in row.iter().enumerate().skip(1) {
                if b.mode != PatchBlendMode::None {
                    return Err(Error::Unsupported(format!(
                        "jxl patches: extra-channel {} blend mode {:?} — extra-channel \
                         patch blending is not wired yet",
                        k - 1,
                        b.mode
                    )));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scripted-source decode: two patches, the second with two
    /// positions, no extra channels.
    #[test]
    fn scripted_decode_follows_listing_c2() {
        // Token script in read order (mode ctx = the printed default 5):
        // num_patches=2;
        // patch0: ref=1, x0=3, y0=4, w-1=1, h-1=2, count-1=0,
        //         pos0=(5,6), mode=2 (kAdd; no clamp field)
        // patch1: ref=0, x0=0, y0=0, w-1=0, h-1=0, count-1=1,
        //         pos0=(1,1), mode=1 (kReplace),
        //         pos1 deltas UnpackSigned(2)=1,UnpackSigned(4)=2 → (2,3),
        //         mode=3 (kMul; clamp=0)
        let script: Vec<(u32, u32)> = vec![
            (0, 2),
            (1, 1),
            (3, 3),
            (3, 4),
            (2, 1),
            (2, 2),
            (7, 0),
            (4, 5),
            (4, 6),
            (5, 2),
            (1, 0),
            (3, 0),
            (3, 0),
            (2, 0),
            (2, 0),
            (7, 1),
            (4, 1),
            (4, 1),
            (5, 1),
            (5, 2),
            (5, 4),
            (5, 3),
            (9, 0),
        ];
        let mut idx = 0usize;
        let patches = decode_patches_with(
            |ctx| {
                let (want_ctx, v) = script[idx];
                assert_eq!(ctx, want_ctx, "context mismatch at token {idx}");
                idx += 1;
                Ok(v)
            },
            0,
            0,
            64,
            64,
        )
        .expect("scripted patch decode");
        assert_eq!(idx, script.len(), "every scripted token consumed");
        assert_eq!(patches.len(), 2);
        assert_eq!(patches[0].reference, 1);
        assert_eq!((patches[0].x0, patches[0].y0), (3, 4));
        assert_eq!((patches[0].width, patches[0].height), (2, 3));
        assert_eq!(patches[0].positions, vec![(5, 6)]);
        assert_eq!(patches[0].blendings[0][0].mode, PatchBlendMode::Add);
        assert_eq!(patches[1].positions, vec![(1, 1), (2, 3)]);
        assert_eq!(patches[1].blendings[0][0].mode, PatchBlendMode::Replace);
        assert_eq!(patches[1].blendings[1][0].mode, PatchBlendMode::Mul);
        assert!(!patches[1].blendings[1][0].clamp);
    }

    #[test]
    fn out_of_frame_position_is_rejected() {
        // One patch of 4×4 at (62, 0) in a 64×64 frame → x + w > 64.
        let script: Vec<u32> = vec![1, 0, 0, 0, 3, 3, 0, 62, 0];
        let mut idx = 0usize;
        let err = decode_patches_with(
            |_ctx| {
                let v = script[idx];
                idx += 1;
                Ok(v)
            },
            0,
            0,
            64,
            64,
        )
        .unwrap_err();
        assert!(err.to_string().contains("not contained"));
    }

    #[test]
    fn render_add_and_replace() {
        let mut refs = ReferenceFrames::default();
        refs.slots[2] = Some(PreCtFrame {
            width: 4,
            height: 2,
            planes: vec![vec![1.0; 8], vec![2.0; 8], vec![3.0; 8]],
        });
        let patch = Patch {
            reference: 2,
            x0: 1,
            y0: 0,
            width: 2,
            height: 2,
            positions: vec![(3, 1), (0, 0)],
            blendings: vec![
                vec![PatchBlending {
                    mode: PatchBlendMode::Add,
                    alpha_channel: 0,
                    clamp: false,
                }],
                vec![PatchBlending {
                    mode: PatchBlendMode::Replace,
                    alpha_channel: 0,
                    clamp: false,
                }],
            ],
        };
        let (w, h) = (6usize, 4usize);
        let mut p0 = vec![0.5f32; w * h];
        let mut p1 = vec![0.5f32; w * h];
        let mut p2 = vec![0.5f32; w * h];
        {
            let mut color: [&mut [f32]; 3] = [&mut p0, &mut p1, &mut p2];
            render_patches(&[patch], &refs, &mut color, w, h).unwrap();
        }
        // kAdd at (3, 1): 0.5 + ref.
        assert_eq!(p0[w + 3], 1.5);
        assert_eq!(p1[w + 3], 2.5);
        assert_eq!(p2[2 * w + 4], 3.5);
        // kReplace at (0, 0): ref value.
        assert_eq!(p0[0], 1.0);
        assert_eq!(p1[w + 1], 2.0);
        // Untouched samples keep the old value.
        assert_eq!(p0[3 * w + 5], 0.5);
    }

    #[test]
    fn missing_reference_slot_errs_precisely() {
        let refs = ReferenceFrames::default();
        let patch = Patch {
            reference: 1,
            x0: 0,
            y0: 0,
            width: 1,
            height: 1,
            positions: vec![(0, 0)],
            blendings: vec![vec![PatchBlending {
                mode: PatchBlendMode::Replace,
                alpha_channel: 0,
                clamp: false,
            }]],
        };
        let mut p0 = vec![0.0f32; 4];
        let mut color: [&mut [f32]; 1] = [&mut p0];
        let err = render_patches(&[patch], &refs, &mut color, 2, 2).unwrap_err();
        assert!(err.to_string().contains("pre-colour-transform"));
    }
}
