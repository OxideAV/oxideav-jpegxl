//! JPEG XL **encoder** — a Modular (Annex H) codestream writer.
//!
//! What it writes, in codestream order:
//!
//! * the signature, `SizeHeader` (D.2) and `ImageMetadata` (D.3) —
//!   integer samples at 1..=16 bits, grey or RGB, an optional alpha
//!   extra channel, the enumerated `ColourEncoding` (E.2) or an
//!   embedded ICC profile (E.4, written as one verbatim command so
//!   the profile bytes round-trip exactly), the `AnimationHeader`
//!   (D.3.4) for multi-frame input;
//! * one `kModular` frame per picture (F.2): no upsampling, no crop,
//!   `kReplace` blending, no restoration filter, `group_size_shift`
//!   from the options; one global MA tree (H.4.2) learned from the
//!   picture ([`crate::modular_enc`]: property splits, any of the
//!   fourteen predictors per leaf, the leaf `multiplier` as the
//!   quality quantiser — `1` = lossless) and the RCT (H.6.3) whose
//!   output is cheapest in front of three colour channels;
//! * the channel residuals (H.3), closed-loop quantised when
//!   `multiplier > 1`, entropy coded ([`crate::entropy_write`]) with
//!   clustered contexts and whichever of prefix codes and ANS is
//!   smaller; the whole picture in `GlobalModular` when it fits one
//!   group, else one `PassGroup` section per group (G.4.2) sharing the
//!   global tree.
//!
//! Everything produced here is decodable by [`crate::decode`] and
//! round-trips bit-exactly at quality 100 (the integration tests pin
//! that on every layout, and against a black-box decoder where one is
//! installed).

use crate::bitreader::U32Dist;
use crate::bitwriter::BitWriter;
use crate::enc_transforms::{Transform, plan_transforms};
use crate::entropy_write::{Coding, EntropyCodes, HybridConfig, SymbolStats, ValueStats};
use crate::error::{Error, Result};
use crate::icc::icc_context;
use crate::image::ColorInfo;
use crate::modular_enc::{Chan, LearnParams, Section, encode_section, learn_tree};
pub use crate::options::Effort;

/// `U32(1 + u(9), 1 + u(13), 1 + u(18), 1 + u(30))` — SizeHeader dims.
const SIZE_DIST: [U32Dist; 4] = [
    U32Dist::BitsOffset(9, 1),
    U32Dist::BitsOffset(13, 1),
    U32Dist::BitsOffset(18, 1),
    U32Dist::BitsOffset(30, 1),
];

/// `U32(u(10), 1024 + u(14), 17408 + u(22), 4211712 + u(30))` — TOC.
const TOC_DIST: [U32Dist; 4] = [
    U32Dist::Bits(10),
    U32Dist::BitsOffset(14, 1024),
    U32Dist::BitsOffset(22, 17408),
    U32Dist::BitsOffset(30, 4211712),
];

/// `U32(0, 1, 2 + u(4), 18 + u(8))` — `nb_transforms`.
const NB_TRANSFORMS_DIST: [U32Dist; 4] = [
    U32Dist::Val(0),
    U32Dist::Val(1),
    U32Dist::BitsOffset(4, 2),
    U32Dist::BitsOffset(8, 18),
];

/// `U32(u(3), 8 + u(6), 72 + u(10), 1096 + u(13))` — `begin_c`.
const BEGIN_C_DIST: [U32Dist; 4] = [
    U32Dist::Bits(3),
    U32Dist::BitsOffset(6, 8),
    U32Dist::BitsOffset(10, 72),
    U32Dist::BitsOffset(13, 1096),
];

/// Largest section the TOC can describe.
const MAX_SECTION_BYTES: usize = 4211712 + (1 << 30) - 1;

/// Streams with at least this many symbols are ANS coded (fractional
/// bits per symbol); shorter ones keep prefix codes, whose histograms
/// are cheaper than the 32-bit ANS state initialiser.
const ANS_MIN_SYMBOLS: usize = 64;

fn coding_for(num_symbols: usize) -> Coding {
    if num_symbols >= ANS_MIN_SYMBOLS {
        Coding::Ans
    } else {
        Coding::Prefix
    }
}

/// Colour signalling of the codestream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColourSpec {
    /// The enumerated E.2 description: H.273 primaries / transfer code
    /// points (anything unmapped falls back to sRGB).
    Enumerated { primaries: u8, transfer: u8 },
    /// An embedded ICC profile (E.4).
    Icc(Vec<u8>),
}

impl ColourSpec {
    /// From the contract's [`ColorInfo`].
    pub fn from_color_info(c: &ColorInfo) -> Self {
        Self::Enumerated {
            primaries: c.primaries,
            transfer: c.transfer,
        }
    }
}

/// `AnimationHeader` fields (D.3.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnimationSpec {
    pub tps_numerator: u32,
    pub tps_denominator: u32,
    pub num_loops: u32,
}

/// The image-level parameters of one codestream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodeSpec {
    pub width: u32,
    pub height: u32,
    /// Integer sample depth of the colour channels and the alpha
    /// channel, 1..=16.
    pub bits: u32,
    /// One colour channel instead of three.
    pub grey: bool,
    /// An alpha extra channel follows the colour channels.
    pub alpha: bool,
    pub colour: ColourSpec,
    /// `Some` writes `have_animation` and per-frame durations.
    pub animation: Option<AnimationSpec>,
    /// 0..=100; 100 is lossless (see [`quantizer_for_quality`]).
    pub quality: u8,
    /// `group_dim = 128 << group_size_shift`, 0..=3.
    pub group_size_shift: u32,
    /// Speed / density trade-off of the Modular coder.
    pub effort: Effort,
}

impl EncodeSpec {
    fn num_channels(&self) -> usize {
        (if self.grey { 1 } else { 3 }) + usize::from(self.alpha)
    }
    fn group_dim(&self) -> u32 {
        128 << self.group_size_shift
    }
    fn num_lf_groups(&self) -> u64 {
        let g = u64::from(self.group_dim()) * 8;
        u64::from(self.width).div_ceil(g) * u64::from(self.height).div_ceil(g)
    }
}

/// One picture to encode: `channels` in codestream order (grey or R,
/// G, B, then alpha), each `width × height` samples in raster order,
/// already in the integer range of `bits`.
#[derive(Debug, Clone)]
pub struct FrameInput {
    pub channels: Vec<Vec<i32>>,
    /// Frame duration in animation ticks (ignored without an
    /// `AnimationSpec`).
    pub duration: u32,
}

fn learn_params_for(effort: Effort) -> LearnParams {
    match effort {
        Effort::Fast => LearnParams {
            max_samples: 60_000,
            max_leaves: 48,
            use_wp: false,
            ..LearnParams::default()
        },
        Effort::Thorough => LearnParams {
            max_samples: 1_200_000,
            max_leaves: 2048,
            ..LearnParams::default()
        },
        _ => LearnParams::default(),
    }
}

/// The Modular residual quantiser for `quality` at `bits` per sample:
/// `1` (lossless) at 100, otherwise `round(2^((100 − q) / 12))` scaled
/// by `2^(bits − 8)` so the step is the same fraction of the sample
/// range at every depth. Monotone: a lower quality never yields a
/// smaller step.
pub fn quantizer_for_quality(quality: u8, bits: u32) -> u32 {
    let q = quality.min(100);
    if q == 100 {
        return 1;
    }
    let base = (2f64.powf(f64::from(100 - q) / 12.0)).round() as u32;
    let base = base.max(2);
    let scaled = if bits > 8 {
        u64::from(base) << (bits - 8)
    } else if bits < 8 {
        (u64::from(base) >> (8 - bits)).max(1)
    } else {
        u64::from(base)
    };
    scaled.min(1 << 24) as u32
}

// ---- headers ---------------------------------------------------------------

/// Signature + `SizeHeader` + `ImageMetadata` (+ ICC stream), then
/// `ZeroPadToByte` — what precedes the first frame.
pub fn write_prelude(w: &mut BitWriter, spec: &EncodeSpec) -> Result<()> {
    if spec.width == 0 || spec.height == 0 {
        return Err(Error::invalid("JXL encoder: zero image dimension"));
    }
    if spec.width > (1 << 30) || spec.height > (1 << 30) {
        return Err(Error::unsupported(
            "JXL encoder: dimensions beyond the SizeHeader range (2^30)",
        ));
    }
    if spec.bits == 0 || spec.bits > 16 {
        return Err(Error::unsupported(format!(
            "JXL encoder: {} bits per sample (1..=16 supported)",
            spec.bits
        )));
    }
    if spec.group_size_shift > 3 {
        return Err(Error::invalid("JXL encoder: group_size_shift > 3"));
    }
    w.write_bits(8, 0xFF);
    w.write_bits(8, 0x0A);
    write_size_header(w, spec.width, spec.height)?;

    // ImageMetadata (Table D.3).
    w.write_bool(false); // all_default
    let extra_fields = spec.animation.is_some();
    w.write_bool(extra_fields);
    if extra_fields {
        w.write_bits(3, 0); // orientation 1
        w.write_bool(false); // have_intr_size
        w.write_bool(false); // have_preview
        w.write_bool(true); // have_animation
        let a = spec.animation.expect("extra_fields implies animation");
        if a.tps_numerator == 0 || a.tps_denominator == 0 {
            return Err(Error::invalid(
                "JXL encoder: animation ticks per second is not a rate",
            ));
        }
        w.write_u32(
            [
                U32Dist::Val(100),
                U32Dist::Val(1000),
                U32Dist::BitsOffset(10, 1),
                U32Dist::BitsOffset(30, 1),
            ],
            a.tps_numerator,
        )?;
        w.write_u32(
            [
                U32Dist::Val(1),
                U32Dist::Val(1001),
                U32Dist::BitsOffset(8, 1),
                U32Dist::BitsOffset(10, 1),
            ],
            a.tps_denominator,
        )?;
        w.write_u32(
            [
                U32Dist::Val(0),
                U32Dist::Bits(3),
                U32Dist::Bits(16),
                U32Dist::Bits(32),
            ],
            a.num_loops,
        )?;
        w.write_bool(false); // have_timecodes
    }
    write_bit_depth(w, spec.bits)?;
    // modular_16bit_buffers: signed 16-bit holds every sample and
    // inverse-RCT intermediate only up to 8-bit input.
    w.write_bool(spec.bits <= 8);
    w.write_u32(
        [
            U32Dist::Val(0),
            U32Dist::Val(1),
            U32Dist::BitsOffset(4, 2),
            U32Dist::BitsOffset(12, 1),
        ],
        u32::from(spec.alpha),
    )?;
    if spec.alpha {
        // ExtraChannelInfo (Table D.8): the all-default alpha channel is
        // 8-bit; any other depth spells the bundle out.
        if spec.bits == 8 {
            w.write_bool(true);
        } else {
            w.write_bool(false);
            w.write_enum(0)?; // kAlpha
            write_bit_depth(w, spec.bits)?;
            w.write_u32(
                [
                    U32Dist::Val(0),
                    U32Dist::Val(3),
                    U32Dist::Val(4),
                    U32Dist::BitsOffset(3, 1),
                ],
                0,
            )?; // dim_shift
            w.write_u32(
                [
                    U32Dist::Val(0),
                    U32Dist::Bits(4),
                    U32Dist::BitsOffset(5, 16),
                    U32Dist::BitsOffset(10, 48),
                ],
                0,
            )?; // name_len
            w.write_bool(false); // alpha_associated
        }
    }
    w.write_bool(false); // xyb_encoded
    write_colour_encoding(w, spec)?;
    if extra_fields {
        w.write_bool(true); // tone_mapping.all_default
    }
    w.write_u64(0); // extensions
    w.write_bool(true); // default_m (default transform data)

    if let ColourSpec::Icc(profile) = &spec.colour {
        write_icc_stream(w, profile)?;
    }
    w.zero_pad_to_byte();
    Ok(())
}

fn write_size_header(w: &mut BitWriter, width: u32, height: u32) -> Result<()> {
    let small =
        width.is_multiple_of(8) && height.is_multiple_of(8) && width <= 256 && height <= 256;
    w.write_bool(small);
    if small {
        w.write_bits(5, height / 8 - 1);
    } else {
        w.write_u32(SIZE_DIST, height)?;
    }
    w.write_bits(3, 0); // ratio: explicit width
    if small {
        w.write_bits(5, width / 8 - 1);
    } else {
        w.write_u32(SIZE_DIST, width)?;
    }
    Ok(())
}

fn write_bit_depth(w: &mut BitWriter, bits: u32) -> Result<()> {
    w.write_bool(false); // float_sample
    w.write_u32(
        [
            U32Dist::Val(8),
            U32Dist::Val(10),
            U32Dist::Val(12),
            U32Dist::BitsOffset(6, 1),
        ],
        bits,
    )
}

/// Table E.5 / E.6 code points from the H.273 values [`ColorInfo`]
/// carries.
fn enum_primaries(primaries: u8) -> u32 {
    match primaries {
        9 => 9,   // kP2100 (BT.2100)
        11 => 11, // kP3
        _ => 1,   // kSRGB
    }
}

fn enum_transfer(transfer: u8) -> u32 {
    match transfer {
        1 => 1,   // k709
        8 => 8,   // kLinear
        16 => 16, // kPQ
        17 => 17, // kDCI
        18 => 18, // kHLG
        _ => 13,  // kSRGB
    }
}

fn write_colour_encoding(w: &mut BitWriter, spec: &EncodeSpec) -> Result<()> {
    match &spec.colour {
        ColourSpec::Icc(_) => {
            w.write_bool(false); // all_default
            w.write_bool(true); // want_icc
            w.write_enum(u32::from(spec.grey))?; // kRGB = 0, kGrey = 1
        }
        ColourSpec::Enumerated {
            primaries,
            transfer,
        } => {
            let prim = enum_primaries(*primaries);
            let tf = enum_transfer(*transfer);
            if !spec.grey && prim == 1 && tf == 13 {
                w.write_bool(true); // all_default: sRGB
                return Ok(());
            }
            w.write_bool(false);
            w.write_bool(false); // want_icc
            w.write_enum(u32::from(spec.grey))?; // colour_space
            w.write_enum(1)?; // white_point kD65
            if !spec.grey {
                w.write_enum(prim)?;
            }
            w.write_bool(false); // have_gamma
            w.write_enum(tf)?;
            w.write_enum(1)?; // rendering_intent kRelative
        }
    }
    Ok(())
}

// ---- ICC ---------------------------------------------------------------------

fn varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 127) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 128);
    }
}

/// The E.4.2 encoded ICC stream for `profile`: header bytes as
/// residuals against the E.4.3 prediction, then — for profiles longer
/// than 128 bytes — an empty tag list (`num_tags = -1`) and one
/// verbatim main-content command carrying the rest, so the decoded
/// profile is byte-identical whatever its tag structure.
pub fn encoded_icc_stream(profile: &[u8]) -> Vec<u8> {
    let size = profile.len();
    let header_len = size.min(128);
    let mut commands = Vec::new();
    if size > 128 {
        varint(&mut commands, 0); // num_tags + 1 = 0 → no tag list
        commands.push(1); // main content: verbatim bytes
        varint(&mut commands, (size - header_len) as u64);
    }
    let mut out = Vec::with_capacity(size + 16);
    varint(&mut out, size as u64);
    varint(&mut out, commands.len() as u64);
    out.extend_from_slice(&commands);
    for i in 0..header_len {
        let p = crate::icc::predict_header_byte(i, &profile[..i], size as u64);
        out.push(profile[i].wrapping_sub(p));
    }
    out.extend_from_slice(&profile[header_len..]);
    out
}

/// E.4.1: `enc_size` as `U64()`, the 41-context prelude and the
/// entropy-coded encoded ICC stream.
pub fn write_icc_stream(w: &mut BitWriter, profile: &[u8]) -> Result<()> {
    if profile.len() < 128 {
        return Err(Error::invalid(
            "JXL encoder: an ICC profile is at least 128 bytes (the header)",
        ));
    }
    let encoded = encoded_icc_stream(profile);
    let mut stats = SymbolStats::single(41, HybridConfig::BYTES)?;
    let mut prev = 0u8;
    let mut prev_prev = 0u8;
    for (i, &b) in encoded.iter().enumerate() {
        stats.add(icc_context(i, prev, prev_prev), u32::from(b));
        prev_prev = prev;
        prev = b;
    }
    let codes = EntropyCodes::build_with(&stats, coding_for(encoded.len()))?;
    w.write_u64(encoded.len() as u64);
    codes.write_prelude(w);
    let mut prev = 0u8;
    let mut prev_prev = 0u8;
    let symbols: Vec<(u32, u32)> = encoded
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let ctx = icc_context(i, prev, prev_prev);
            prev_prev = prev;
            prev = b;
            (ctx, u32::from(b))
        })
        .collect();
    codes.write_symbols(w, symbols.iter().copied());
    Ok(())
}

// ---- Modular frame coding --------------------------------------------------

/// Everything needed to serialise one frame.
struct CodedFrame {
    tree_codes: EntropyCodes,
    tree_syms: Vec<(u32, u32)>,
    leaf_codes: EntropyCodes,
    transforms: Vec<Transform>,
    /// Symbols of the channels carried by `GlobalModular` (empty when
    /// it carries none).
    global: Vec<(u32, u32)>,
    /// One entry per group in raster order (empty for a one-group
    /// picture).
    groups: Vec<Vec<(u32, u32)>>,
}

fn code_frame(spec: &EncodeSpec, frame: &FrameInput) -> Result<CodedFrame> {
    let width = spec.width as usize;
    let height = spec.height as usize;
    let n_ch = spec.num_channels();
    if frame.channels.len() != n_ch {
        return Err(Error::invalid(format!(
            "JXL encoder: frame carries {} channel(s), {n_ch} expected",
            frame.channels.len()
        )));
    }
    let max = (1i64 << spec.bits) - 1;
    for (c, ch) in frame.channels.iter().enumerate() {
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
    let chans: Vec<Chan> = frame
        .channels
        .iter()
        .map(|c| Chan::new(width, height, c.clone()))
        .collect();
    let multiplier = quantizer_for_quality(spec.quality, spec.bits);
    let planned = plan_transforms(
        chans,
        if spec.grey { 1 } else { 3 },
        spec.bits,
        multiplier == 1,
    );
    let transforms = planned.transforms;
    let n_meta = planned.n_meta;
    let mut chans = planned.chans;

    // Split into independently decoded sections: the meta channels
    // and — when the picture fits one group — everything else go into
    // GlobalModular; otherwise each group carries its rectangle of
    // every non-meta channel.
    let group_dim = spec.group_dim() as usize;
    let single = width <= group_dim && height <= group_dim;
    let mut sections: Vec<Section> = Vec::new();
    if single {
        sections.push(Section {
            stream_index: 0,
            chans,
        });
    } else {
        let image: Vec<Chan> = chans.split_off(n_meta);
        sections.push(Section {
            stream_index: 0,
            chans,
        });
        let gx = width.div_ceil(group_dim);
        let gy = height.div_ceil(group_dim);
        let base = 1 + 3 * spec.num_lf_groups() as i64 + 17;
        for j in 0..gy {
            for i in 0..gx {
                let x0 = i * group_dim;
                let y0 = j * group_dim;
                let rw = group_dim.min(width - x0);
                let rh = group_dim.min(height - y0);
                let stream_index = i32::try_from(base + (j * gx + i) as i64).map_err(|_| {
                    Error::unsupported("JXL encoder: too many groups for a stream index")
                })?;
                sections.push(Section {
                    stream_index,
                    chans: image.iter().map(|c| c.rect(x0, y0, rw, rh)).collect(),
                });
            }
        }
    }

    let params = LearnParams {
        multiplier,
        ..learn_params_for(spec.effort)
    };
    let tree = learn_tree(&sections, &params);
    let tree_syms = tree.symbols();
    let mut tree_stats = ValueStats::new(6);
    for &(c, v) in &tree_syms {
        tree_stats.add(c, v);
    }
    let tree_codes = EntropyCodes::build_clustered(&tree_stats, 1, 6)?;

    let mut leaf_stats = ValueStats::new(tree.num_ctx);
    let mut coded: Vec<Vec<(u32, u32)>> = Vec::with_capacity(sections.len());
    for mut section in sections {
        let syms = encode_section(&mut section, &tree);
        for &(c, v) in &syms {
            leaf_stats.add(c, v);
        }
        coded.push(syms);
    }
    let n_streams = coded.iter().filter(|s| !s.is_empty()).count().max(1);
    let leaf_codes = EntropyCodes::build_clustered(&leaf_stats, n_streams, 256)?;
    let mut it = coded.into_iter();
    let global = it.next().unwrap_or_default();
    Ok(CodedFrame {
        tree_codes,
        tree_syms,
        leaf_codes,
        transforms,
        global,
        groups: it.collect(),
    })
}

// ---- frame serialisation ----------------------------------------------------

/// The frame-header fields this writer emits (F.2, `kModular`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameShape {
    pub group_size_shift: u32,
    pub num_extra: u32,
    pub have_animation: bool,
    pub duration: u32,
    pub is_last: bool,
}

/// `FrameHeader` (Table F.2) + TOC (F.3) for a plain Modular frame;
/// `toc` holds the section sizes in bytes. Leaves the writer
/// byte-aligned at the first section.
pub fn write_frame_header_and_toc(
    w: &mut BitWriter,
    shape: &FrameShape,
    toc: &[u32],
) -> Result<()> {
    w.write_bool(false); // all_default
    w.write_bits(2, 0); // frame_type kRegularFrame
    w.write_bits(1, 1); // encoding kModular
    w.write_u64(0); // flags
    w.write_bool(false); // do_YCbCr
    w.write_bits(2, 0); // upsampling = 1
    for _ in 0..shape.num_extra {
        w.write_bits(2, 0); // ec_upsampling = 1
    }
    w.write_bits(2, shape.group_size_shift);
    w.write_bits(2, 0); // passes.num_passes = 1
    w.write_bool(false); // have_crop
    w.write_bits(2, 0); // blending_info.mode kReplace
    for _ in 0..shape.num_extra {
        w.write_bits(2, 0); // ec_blending_info kReplace
    }
    if shape.have_animation {
        w.write_u32(
            [
                U32Dist::Val(0),
                U32Dist::Val(1),
                U32Dist::Bits(8),
                U32Dist::Bits(32),
            ],
            shape.duration,
        )?;
    }
    w.write_bool(shape.is_last);
    if !shape.is_last {
        w.write_bits(2, 0); // save_as_reference
        // save_before_ct is present only for a frame that can be
        // referenced (`resets_canvas && can_reference`, Table F.2):
        // with save_as_reference = 0 that is a zero-duration frame.
        if !shape.have_animation || shape.duration == 0 {
            w.write_bool(false); // save_before_ct
        }
    }
    w.write_bits(2, 0); // name_len = 0
    // RestorationFilter (Table J.1): no Gabor, no EPF.
    w.write_bool(false); // all_default
    w.write_bool(false); // gab
    w.write_bits(2, 0); // epf_iters
    w.write_u64(0); // rf extensions
    w.write_u64(0); // frame extensions
    // TOC.
    w.write_bool(false); // permuted_toc
    w.zero_pad_to_byte();
    for &entry in toc {
        w.write_u32(TOC_DIST, entry)?;
    }
    w.zero_pad_to_byte();
    Ok(())
}

fn write_transforms(w: &mut BitWriter, transforms: &[Transform]) -> Result<()> {
    w.write_u32(NB_TRANSFORMS_DIST, transforms.len() as u32)?;
    for t in transforms {
        match *t {
            Transform::Palette {
                begin_c,
                num_c,
                nb_colours,
            } => {
                w.write_bits(2, 1); // kPalette
                w.write_u32(BEGIN_C_DIST, begin_c)?;
                w.write_u32(
                    [
                        U32Dist::Val(1),
                        U32Dist::Val(3),
                        U32Dist::Val(4),
                        U32Dist::BitsOffset(13, 1),
                    ],
                    num_c,
                )?;
                w.write_u32(
                    [
                        U32Dist::BitsOffset(8, 0),
                        U32Dist::BitsOffset(10, 256),
                        U32Dist::BitsOffset(12, 1280),
                        U32Dist::BitsOffset(16, 5376),
                    ],
                    nb_colours,
                )?;
                w.write_u32(
                    [
                        U32Dist::Val(0),
                        U32Dist::BitsOffset(8, 1),
                        U32Dist::BitsOffset(10, 257),
                        U32Dist::BitsOffset(16, 1281),
                    ],
                    0,
                )?; // nb_deltas
                w.write_bits(4, 0); // d_pred
            }
            Transform::Rct { begin_c, rct_type } => {
                w.write_bits(2, 0); // kRCT
                w.write_u32(BEGIN_C_DIST, begin_c)?;
                w.write_u32(
                    [
                        U32Dist::Val(6),
                        U32Dist::Bits(2),
                        U32Dist::BitsOffset(4, 2),
                        U32Dist::BitsOffset(6, 10),
                    ],
                    rct_type,
                )?;
            }
        }
    }
    Ok(())
}

fn write_global_modular_header(w: &mut BitWriter, coded: &CodedFrame) -> Result<()> {
    // LfChannelDequantization: all_default.
    w.write_bool(true);
    // GlobalModular: the global tree.
    w.write_bool(true); // use_global_tree
    coded.tree_codes.write_prelude(w);
    coded
        .tree_codes
        .write_symbols(w, coded.tree_syms.iter().copied());
    coded.leaf_codes.write_prelude(w);
    // Inner ModularHeader.
    w.write_bool(true); // use_global_tree
    w.write_bool(true); // wp_params.default_wp
    write_transforms(w, &coded.transforms)
}

/// Append one frame (header, TOC, sections) to a byte-aligned writer.
fn write_frame(
    w: &mut BitWriter,
    spec: &EncodeSpec,
    frame: &FrameInput,
    is_last: bool,
) -> Result<()> {
    let coded = code_frame(spec, frame)?;
    let single = coded.groups.is_empty();
    let mut sections: Vec<Vec<u8>> = Vec::new();
    {
        let mut lf = BitWriter::new();
        write_global_modular_header(&mut lf, &coded)?;
        if !coded.global.is_empty() {
            coded
                .leaf_codes
                .write_symbols(&mut lf, coded.global.iter().copied());
        }
        lf.zero_pad_to_byte();
        sections.push(lf.into_bytes());
    }
    if !single {
        for _ in 0..spec.num_lf_groups() {
            sections.push(Vec::new()); // ModularLfGroup: nothing to code
        }
        sections.push(Vec::new()); // HfGlobal: absent for kModular
        for group in &coded.groups {
            let mut gw = BitWriter::new();
            gw.write_bool(true); // use_global_tree
            gw.write_bool(true); // default_wp
            gw.write_u32(NB_TRANSFORMS_DIST, 0)?;
            coded
                .leaf_codes
                .write_symbols(&mut gw, group.iter().copied());
            gw.zero_pad_to_byte();
            sections.push(gw.into_bytes());
        }
    }
    let mut toc = Vec::with_capacity(sections.len());
    for s in &sections {
        if s.len() > MAX_SECTION_BYTES {
            return Err(Error::unsupported(
                "JXL encoder: section exceeds the TOC range",
            ));
        }
        toc.push(s.len() as u32);
    }
    let shape = FrameShape {
        group_size_shift: spec.group_size_shift,
        num_extra: u32::from(spec.alpha),
        have_animation: spec.animation.is_some(),
        duration: frame.duration,
        is_last,
    };
    write_frame_header_and_toc(w, &shape, &toc)?;
    for s in &sections {
        w.write_bytes(s);
    }
    Ok(())
}

/// Encode `frames` as one bare codestream (`FF 0A` …). One frame
/// without an [`AnimationSpec`] is a still; several frames require
/// one.
pub fn encode_codestream(spec: &EncodeSpec, frames: &[FrameInput]) -> Result<Vec<u8>> {
    if frames.is_empty() {
        return Err(Error::invalid("JXL encoder: no frame to encode"));
    }
    if frames.len() > 1 && spec.animation.is_none() {
        return Err(Error::invalid(
            "JXL encoder: several frames need an AnimationSpec (tick rate)",
        ));
    }
    let mut w = BitWriter::new();
    write_prelude(&mut w, spec)?;
    let last = frames.len() - 1;
    for (i, f) in frames.iter().enumerate() {
        write_frame(&mut w, spec, f, i == last)?;
    }
    Ok(w.into_bytes())
}

// ---- container -----------------------------------------------------------------

fn push_box(out: &mut Vec<u8>, kind: &[u8; 4], payload: &[u8]) -> Result<()> {
    let len = payload.len() + 8;
    if len > u32::MAX as usize {
        return Err(Error::unsupported("JXL encoder: box larger than 4 GiB"));
    }
    out.extend_from_slice(&(len as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    Ok(())
}

/// Wrap a codestream in the 18181-2 box file: signature, `ftyp`, then
/// the metadata boxes (`Exif` with a zero TIFF-header offset, `xml `)
/// ahead of the `jxlc` codestream box.
pub fn wrap_container(
    codestream: &[u8],
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(codestream.len() + 64);
    out.extend_from_slice(&[
        0, 0, 0, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A,
    ]);
    out.extend_from_slice(&[
        0, 0, 0, 0x14, b'f', b't', b'y', b'p', b'j', b'x', b'l', b' ', 0, 0, 0, 0, b'j', b'x',
        b'l', b' ',
    ]);
    if let Some(exif) = exif {
        let mut payload = Vec::with_capacity(exif.len() + 4);
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(exif);
        push_box(&mut out, b"Exif", &payload)?;
    }
    if let Some(xmp) = xmp {
        push_box(&mut out, b"xml ", xmp)?;
    }
    push_box(&mut out, b"jxlc", codestream)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantizer_is_monotone_and_lossless_only_at_100() {
        assert_eq!(quantizer_for_quality(100, 8), 1);
        assert_eq!(quantizer_for_quality(100, 16), 1);
        let mut prev = 1;
        for q in (0..100).rev() {
            let m = quantizer_for_quality(q, 8);
            assert!(m >= 2, "q{q} is lossy");
            assert!(m >= prev, "q{q}: {m} < {prev}");
            prev = m;
        }
        assert_eq!(
            quantizer_for_quality(90, 16),
            quantizer_for_quality(90, 8) << 8
        );
        assert_eq!(quantizer_for_quality(90, 8), 2);
        assert_eq!(quantizer_for_quality(50, 8), 18);
    }

    #[test]
    fn encoded_icc_header_only_profile_is_the_residual_stream() {
        let profile = vec![7u8; 128];
        let enc = encoded_icc_stream(&profile);
        assert_eq!(enc[0], 128);
        assert_eq!(enc[1], 1); // varint(128) = 0x80 0x01
        assert_eq!(enc[2], 0); // commands_size 0
        assert_eq!(enc.len(), 3 + 128);
        assert_eq!(crate::icc::reconstruct_icc_profile(&enc).unwrap(), profile);
        let mut long = vec![0u8; 300];
        for (i, b) in long.iter_mut().enumerate() {
            *b = (i * 31) as u8;
        }
        let enc = encoded_icc_stream(&long);
        assert_eq!(crate::icc::reconstruct_icc_profile(&enc).unwrap(), long);
    }
}
