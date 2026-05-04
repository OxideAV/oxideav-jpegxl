//! JPEG XL encoder — round-4 lossless modular Gray/RGB/RGBA with
//! the **Gradient** (predictor id 5) MA-tree leaf and a frequency-
//! adapted ANS-coded symbol stream.
//!
//! Round 4 deltas vs round 2 (`089af88`):
//!
//! * **Symbol stream switched from prefix code to ANS.** Round 2 emitted
//!   exactly 4 bits per token regardless of frequency; round 4 quantises
//!   the actual token histogram into a 4096-summing distribution
//!   (with each entry a multiple of `bucket_size=16` to satisfy the
//!   alias-table bijection invariant — see
//!   `crate::ans_encoder` module docs for the round-4 finding) and
//!   feeds it through the new [`crate::ans_encoder`] (the inverse of
//!   the existing [`crate::ans::symbol::AnsDecoder`]). On natural
//!   images where ~80% of Gradient residuals are zero, this approaches
//!   the entropy bound `H(D) = -sum(p_i * log2(p_i))` ≈ 0.7 bits/pixel
//!   — a ~5–6× compression improvement vs round 2 once the ~30 byte
//!   distribution preamble is amortised.
//!
//! Round 2 baseline (kept for reference — still used internally for the
//! tree stream which only emits 6 fixed-value tokens):
//!
//! * **Predictor: Gradient (id 5).** Single-leaf MA tree with
//!   `predictor=5, offset=0, multiplier=1`.
//! * **Per-pixel residual = sample - prediction(left, top, topleft)**
//!   per FDIS Listing C.16's clamp(W+N-NW, min(W,N), max(W,N)).
//!
//! Bitstream shape (unchanged from round 1):
//!
//! ```text
//! [2 B signature FF 0A]
//! [SizeHeader              FDIS A.3]
//! [ImageMetadata           FDIS A.6, all_default=0 to flip xyb_encoded]
//! [ZeroPadToByte           FDIS 6.3]
//! [FrameHeader             FDIS C.2, encoding=Modular, single group]
//! [TOC                     FDIS C.3, single entry shortcut]
//! [LfChannelDequantization FDIS C.4.2, all_default=1]
//! [GlobalModular           FDIS C.4.8, no transforms, single-leaf MA tree]
//! [byte align]
//! ```
//!
//! ## Round-2 envelope
//!
//! * Single frame, intra-only.
//! * 8-bit-per-sample integer, Gray (1 channel), RGB (3) or RGBA (4).
//! * No transforms (no Squeeze, no Palette, no RCT).
//! * Single-leaf MA tree: predictor=Gradient (id 5), offset=0, multiplier=1.
//! * Single group (`width, height <= 1024`). Multi-group emits an error.
//! * Prefix (Huffman) entropy coding throughout.
//!   - Tree stream: 2 clusters (single-symbol code each) + HybridUintConfig
//!     `split_exponent=8, msb=0, lsb=0`.
//!   - Symbol stream: 1 cluster, 16 symbols all length 4
//!     (canonical-Huffman codes 0..15 LSB-first). HybridUintConfig
//!     `split_exponent=0, msb=0, lsb=0` so token T encodes
//!     `2^(T-1)..2^T - 1` with T-1 extra bits.
//!
//! Everything outside this envelope returns `Error::Unsupported`.

use oxideav_core::{Error, Result};

use crate::ans::alias::AliasTable;
use crate::ans_encoder::{
    build_inverse_alias, encode_symbols_with_extras, quantise_distribution_aligned,
    write_distribution, AnsTokenWithExtras,
};
use crate::bitwriter::{pack_signed, BitWriter, U32WriteDist};

/// Pixel formats accepted by the round-2 encoder. All inputs are 8-bit
/// integer per channel, interleaved (single-plane) layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    /// 1-channel 8-bit greyscale.
    Gray8,
    /// 3-channel 8-bit RGB, interleaved as R,G,B,R,G,B,...
    Rgb8,
    /// 4-channel 8-bit RGBA, interleaved as R,G,B,A,R,G,B,A,...
    Rgba8,
}

impl InputFormat {
    /// Number of bytes per pixel in the interleaved input buffer.
    pub fn channel_count(self) -> u32 {
        match self {
            InputFormat::Gray8 => 1,
            InputFormat::Rgb8 => 3,
            InputFormat::Rgba8 => 4,
        }
    }
    /// Number of EXTRA channels (beyond the colour channels). Round-2
    /// only emits Alpha as the extra channel.
    pub fn num_extra_channels(self) -> u32 {
        match self {
            InputFormat::Gray8 => 0,
            InputFormat::Rgb8 => 0,
            InputFormat::Rgba8 => 1,
        }
    }
    /// True when the colour encoding is Grey (single colour channel).
    fn is_grey(self) -> bool {
        matches!(self, InputFormat::Gray8)
    }
}

/// Encode one image frame as a raw JPEG XL codestream (no ISOBMFF
/// wrapping). Returns the complete codestream bytes including the
/// `FF 0A` signature.
///
/// `width` × `height` × channels-per-pixel must equal `pixels.len()`.
/// Bounds: `width, height <= 1024` (single-group cap; round 3 lifts
/// this).
pub fn encode_one_frame(
    width: u32,
    height: u32,
    pixels: &[u8],
    format: InputFormat,
) -> Result<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(Error::other("JXL encoder: zero-dim frame"));
    }
    if width > 1024 || height > 1024 {
        return Err(Error::other(
            "JXL encoder: dimensions > 1024 not supported (round 2 single-group cap)",
        ));
    }
    let channel_count = format.channel_count();
    let expected_len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(channel_count as usize))
        .ok_or_else(|| Error::other("JXL encoder: pixel buffer size overflow"))?;
    if pixels.len() != expected_len {
        return Err(Error::other(format!(
            "JXL encoder: pixels len {} != width*height*channels {}",
            pixels.len(),
            expected_len
        )));
    }

    let mut bw = BitWriter::with_capacity(pixels.len());
    // Signature (FF 0A) goes BEFORE any bit-level packing per FDIS 6.2.
    bw.write_bits(0xFF, 8)?;
    bw.write_bits(0x0A, 8)?;

    write_size_header(&mut bw, width, height)?;
    write_image_metadata(&mut bw, format)?;
    bw.pad_to_byte();
    write_frame_header(&mut bw, width, height, format)?;
    write_toc_single_entry_then_payload(&mut bw, width, height, pixels, format)?;
    Ok(bw.finish())
}

/// Like [`encode_one_frame`] but emits an ISOBMFF-wrapped codestream
/// (signature box + `ftyp jxl ` + `jxlc`). The output starts with the
/// 12-byte ISOBMFF signature (`00 00 00 0C 4A 58 4C 20 0D 0A 87 0A`)
/// and is round-trippable through both [`crate::decode_one_frame`]
/// (which transparently extracts the codestream) and external tools
/// like `djxl`.
///
/// Use this when emitting `.jxl` files that need to be recognised by
/// applications expecting the wrapped form (web browsers, image
/// viewers); use [`encode_one_frame`] when bandwidth matters and the
/// consumer accepts raw codestreams.
pub fn encode_one_frame_isobmff(
    width: u32,
    height: u32,
    pixels: &[u8],
    format: InputFormat,
) -> Result<Vec<u8>> {
    let codestream = encode_one_frame(width, height, pixels, format)?;
    crate::container::wrap_codestream(&codestream)
}

/// FDIS A.3 SizeHeader. We always take the `small=0` path for
/// simplicity (covers the full 1..=2^30 range via the four U32
/// distributions).
fn write_size_header(bw: &mut BitWriter, width: u32, height: u32) -> Result<()> {
    bw.write_bit(0); // small = 0
    let dim_dist = [
        U32WriteDist::BitsOffset(9, 1),
        U32WriteDist::BitsOffset(13, 1),
        U32WriteDist::BitsOffset(18, 1),
        U32WriteDist::BitsOffset(30, 1),
    ];
    bw.write_u32(dim_dist, height)?;
    // ratio = 0 → width is encoded explicitly.
    bw.write_bits(0, 3)?;
    bw.write_u32(dim_dist, width)?;
    Ok(())
}

/// FDIS A.6 ImageMetadata. We use `all_default=0` because the default
/// has `xyb_encoded=true` which is wrong for a raw RGB encode. Everything
/// else is left at its default.
fn write_image_metadata(bw: &mut BitWriter, format: InputFormat) -> Result<()> {
    bw.write_bit(0); // all_default = 0
    bw.write_bit(0); // extra_fields = 0 (no orientation/intr_size/preview/animation)

    // BitDepth (Table A.15): float_sample=0, bits_per_sample sel=0 → Val(8).
    bw.write_bit(0); // float_sample = 0
    bw.write_bits(0, 2)?; // sel=0 → 8 bps

    bw.write_bit(1); // modular_16bit_buffers = 1 (default)

    // num_extra_channels: U32(Val(0), Val(1), BitsOffset(4, 2), BitsOffset(12, 1)).
    let nec_dist = [
        U32WriteDist::Val(0),
        U32WriteDist::Val(1),
        U32WriteDist::BitsOffset(4, 2),
        U32WriteDist::BitsOffset(12, 1),
    ];
    bw.write_u32(nec_dist, format.num_extra_channels())?;

    if format.num_extra_channels() == 1 {
        // ExtraChannelInfo[0]: all_default=1 → Alpha, 8bpp, no name,
        // alpha_associated=false.
        bw.write_bit(1);
    }

    bw.write_bit(0); // xyb_encoded = 0

    // ColourEncoding: all_default=1 short-circuits to RGB / D65 / sRGB.
    // For Grey input we write the explicit (non-default) form with
    // colour_space=Grey, white_point=D65, then default TF + intent.
    if format.is_grey() {
        bw.write_bit(0); // all_default = 0
        bw.write_bit(0); // want_icc = 0
                         // colour_space = Grey (=1) via Enum() — sel=1 → Val(1).
        bw.write_bits(1, 2)?;
        // use_desc=true && not_xyb=true → read white_point.
        // white_point = D65 (=1) via Enum() → sel=1 → Val(1).
        bw.write_bits(1, 2)?;
        // colour_space=Grey → no primaries.
        // Read tf (CustomTransferFunction).
        // have_gamma=0, then transfer_function via Enum(). SRGB=13 →
        // Enum sel=2 → BitsOffset(4, 2) with raw = 11 → 13.
        bw.write_bit(0); // have_gamma = 0
        bw.write_bits(2, 2)?; // sel = 2 → BitsOffset(4, 2)
        bw.write_bits(11, 4)?; // raw = 11 → 13 = SRGB
                               // rendering_intent = Relative (=1) → Enum sel=1 → Val(1).
        bw.write_bits(1, 2)?;
    } else {
        bw.write_bit(1); // all_default = 1
    }

    // No tone_mapping (gated on extra_fields, which is 0).

    // Extensions: U64() = 0.
    bw.write_u64(0)?;

    bw.write_bit(1); // default_transform = 1
                     // default_transform=1 + xyb_encoded=0 → no OpsinInverseMatrix.
                     // default_transform=1 → cw_mask = u(3); pick 0 (no custom upsampling weights).
    bw.write_bits(0, 3)?;
    Ok(())
}

/// FDIS C.2 FrameHeader. Modular encoding, single pass, single group.
fn write_frame_header(
    bw: &mut BitWriter,
    width: u32,
    height: u32,
    format: InputFormat,
) -> Result<()> {
    bw.write_bit(0); // all_default = 0
    bw.write_bits(0, 2)?; // frame_type = Regular
    bw.write_bit(1); // encoding = Modular
    bw.write_u64(0)?; // flags = 0

    // do_ycbcr — only read when !xyb_encoded.
    bw.write_bit(0); // do_ycbcr = 0

    // upsampling = 1 (sel=0 → Val(1)).
    bw.write_bits(0, 2)?;
    // ec_upsampling[i] = 1 each.
    for _ in 0..format.num_extra_channels() {
        bw.write_bits(0, 2)?;
    }

    // group_size_shift: u(2). Pick the largest (3) → kGroupDim = 1024,
    // so any width/height up to 1024 produces a single group.
    let group_dim = if width.max(height) <= 128 {
        bw.write_bits(0, 2)?; // shift=0 → 128
        128
    } else if width.max(height) <= 256 {
        bw.write_bits(1, 2)?;
        256
    } else if width.max(height) <= 512 {
        bw.write_bits(2, 2)?;
        512
    } else if width.max(height) <= 1024 {
        bw.write_bits(3, 2)?;
        1024
    } else {
        return Err(Error::other(format!(
            "JXL encoder: image {width}x{height} exceeds single-group cap (1024)"
        )));
    };
    let _ = group_dim;

    // x_qm_scale / b_qm_scale read only when (encoding == VarDCT && xyb_encoded).
    // Both are false here, so neither is written.

    // Passes (Table C.6): num_passes selector = sel=0 → Val(1).
    bw.write_bits(0, 2)?;
    // num_passes == 1 → no num_ds / shift / downsample / last_pass.

    // have_crop = 0 (we always cover the full image).
    bw.write_bit(0);

    // BlendingInfo: mode = Replace (sel=0 → Val(0)).
    bw.write_bits(0, 2)?;
    // multi_extra=false (we have <2 extras), so no alpha_channel/clamp.
    // mode=Replace + full_frame=true → no source field.

    // ec_blending_info[i]: same pattern.
    for _ in 0..format.num_extra_channels() {
        bw.write_bits(0, 2)?;
    }

    // No animation → no duration/timecode.

    // is_last = 1.
    bw.write_bit(1);
    // is_last=1 → save_as_reference NOT read.

    // save_before_ct = 0 (frame_type != LfFrame, so we read this).
    bw.write_bit(0);

    // name_len: sel=0 → Val(0), no name bytes.
    bw.write_bits(0, 2)?;

    // RestorationFilter: gab=0, epf_iters=0, no extras-payload.
    bw.write_bit(0); // gab = 0
    bw.write_bits(0, 2)?; // epf_iters = 0
                          // epf_iters == 0 → no sigma_for_modular field even though encoding==Modular.
    bw.write_u64(0)?; // restoration_filter.extensions = 0

    // outer extensions = 0.
    bw.write_u64(0)?;
    Ok(())
}

/// Emit a single-entry TOC followed by the (single) section payload.
///
/// The section is byte-aligned; we materialise it into a separate
/// BitWriter, length-prefix it via the U32 entry, then byte-align and
/// append the section bytes verbatim.
fn write_toc_single_entry_then_payload(
    bw: &mut BitWriter,
    width: u32,
    height: u32,
    pixels: &[u8],
    format: InputFormat,
) -> Result<()> {
    // Build the section payload in a side buffer.
    let mut sec = BitWriter::new();
    write_lf_global(&mut sec, width, height, pixels, format)?;
    sec.pad_to_byte();
    let section_bytes = sec.finish();
    let n = section_bytes.len();
    if n == 0 {
        return Err(Error::other(
            "JXL encoder: section payload unexpectedly empty",
        ));
    }
    if n > (1u64 << 30) as usize + 4_211_712 {
        return Err(Error::other(
            "JXL encoder: section payload exceeds U32 max range",
        ));
    }

    // permuted_toc = 0.
    bw.write_bit(0);

    // ZeroPadToByte before TOC entries (FDIS C.3.3 / 6.3).
    bw.pad_to_byte();

    // Single TOC entry: U32(Bits(10), BitsOffset(14, 1024),
    //                       BitsOffset(22, 17408), BitsOffset(30, 4211712)).
    let entry_dist = [
        U32WriteDist::Bits(10),
        U32WriteDist::BitsOffset(14, 1024),
        U32WriteDist::BitsOffset(22, 17408),
        U32WriteDist::BitsOffset(30, 4_211_712),
    ];
    bw.write_u32(entry_dist, n as u32)?;

    // ZeroPadToByte after TOC.
    bw.pad_to_byte();

    // Append section bytes verbatim.
    for &b in &section_bytes {
        bw.write_bits(b as u32, 8)?;
    }
    Ok(())
}

/// Write the LfGlobal bundle (round-2 Modular envelope: only
/// LfChannelDequantization + GlobalModular, no Patches/Splines/Noise/
/// VarDCT state).
fn write_lf_global(
    bw: &mut BitWriter,
    width: u32,
    height: u32,
    pixels: &[u8],
    format: InputFormat,
) -> Result<()> {
    // LfChannelDequantization: all_default = 1.
    bw.write_bit(1);

    // GlobalModular (FDIS C.4.8) follows immediately for Modular
    // encoding (no Quantizer / HfBlockContext / CfL — those are VarDCT
    // only).
    write_global_modular(bw, width, height, pixels, format)
}

/// Write the GlobalModular section: outer use_global_tree flag + global
/// MA tree (when set), then the inner Modular sub-bitstream.
fn write_global_modular(
    bw: &mut BitWriter,
    width: u32,
    height: u32,
    pixels: &[u8],
    format: InputFormat,
) -> Result<()> {
    // We do NOT use a global MA tree — the tree is local to the inner
    // sub-bitstream. This matches `inner_use_global_tree=0` below.
    bw.write_bit(0); // global_use_tree = 0

    // Inner Modular sub-bitstream (Table C.22):
    //   - use_global_tree (Bool)
    //   - WPHeader (Table C.23)
    //   - U32 nb_transforms
    //   - TransformInfo[nb_transforms]
    //   - if !use_global_tree: MA tree + clustered distributions
    //   - per-channel decode loop (Listing C.17 + C.16)
    bw.write_bit(0); // inner use_global_tree = 0

    // WPHeader: default_wp = 1 (we don't use predictor 6).
    bw.write_bit(1);

    // nb_transforms = 0 (sel=0 → Val(0)).
    bw.write_bits(0, 2)?;

    // Local MA tree: a single leaf with predictor=Gradient (5 in FDIS),
    // offset=0, multiplier=1 (mul_log=0, mul_bits=0). Encoded via a
    // 6-distribution / 2-cluster prefix-code entropy stream. Cluster 0
    // always returns 0; cluster 1 always returns 5 (used only for the
    // predictor field).
    write_gradient_leaf_ma_tree(bw)?;

    // Round-4: ANS-coded symbol stream. We do a two-pass walk over the
    // pixels:
    //
    // 1. First pass — compute the (token, extra_bits) sequence WITHOUT
    //    writing to the bitstream, building a token histogram.
    // 2. Quantise the histogram into a 4096-summing ANS distribution
    //    aligned to multiples of bucket_size=16 (log_alpha=8) so the
    //    alias-table bijection invariant holds — see ans_encoder docs.
    // 3. Emit the ANS prelude (lz77=0, use_prefix_code=0, log_alpha=8,
    //    HybridUintConfig, distribution).
    // 4. Emit the ANS-coded tokens with interleaved extras via
    //    [`encode_symbols_with_extras`].
    let tokens = collect_pixel_tokens(width, height, pixels, format)?;
    write_ans_symbol_stream(bw, &tokens)
}

/// Write the MA tree sub-bitstream so the decoder reads a tree with a
/// single leaf node: predictor=5 (Gradient), offset=0, multiplier=1.
///
/// The MA tree's entropy stream is `EntropyStream::read(br, 6)` — six
/// distributions T[0..=5] mapped via a simple cluster map to two
/// clusters. Cluster 0 always returns 0 (single-symbol prefix code,
/// alphabet_size=1). Cluster 1 always returns 5 (single-symbol simple
/// prefix code over a 16-symbol alphabet, with the listed symbol = 5).
///
/// Both clusters share `HybridUintConfig { split_exponent=8, msb=0,
/// lsb=0 }` so `split = 256` and tokens below 256 (i.e. all our reads)
/// return their value directly — token 0 → value 0, token 5 → value 5.
///
/// Cluster map for tree contexts (in spec read order):
///   ctx=0 → cluster 0 (decision-node values, never decoded for our 1-leaf tree)
///   ctx=1 → cluster 0 (property+1 = 0 → leaf marker)
///   ctx=2 → cluster 1 (predictor = 5 → Gradient)
///   ctx=3 → cluster 0 (uoffset = 0)
///   ctx=4 → cluster 0 (mul_log = 0)
///   ctx=5 → cluster 0 (mul_bits = 0; multiplier = (0+1) << 0 = 1)
fn write_gradient_leaf_ma_tree(bw: &mut BitWriter) -> Result<()> {
    // Tree-stream prelude (FDIS D.3 + libjxl-trace-reverse-engineering.md §3.6):
    //
    //   1. lz77_enabled = 0 (no LZ77 length config to follow)
    //   2. cluster_map for num_dist=6 (read_clustering, since num_dist > 1)
    //      — is_simple=1, nbits=1, six u(1) reads = [0,0,1,0,0,0]
    //   3. use_prefix_code = 1
    //      (log_alphabet_size fixed at 15 for prefix branch — no bits read)
    //   4. n_clusters = 2 → two HybridUintConfigs:
    //      both with split_exponent=8, msb=0, lsb=0
    //   5. two prefix codes:
    //      cluster 0: count=1 → degenerate single-symbol code returns 0
    //      cluster 1: count=16 (so symbol id 5 fits in alphabet); simple
    //                 prefix code with NSYM=1, listed symbol = 5,
    //                 single-symbol path emits 0 bits per decode
    //                 returning symbol 5 → decoded value 5

    bw.write_bit(0); // lz77_enabled = 0

    // read_clustering: is_simple = 1, nbits = 1, six u(1) reads.
    // Cluster map [0, 0, 1, 0, 0, 0]. n_clusters = 2.
    bw.write_bit(1); // is_simple = 1
    bw.write_bits(1, 2)?; // nbits = 1
    bw.write_bits(0, 1)?; // ctx 0 → cluster 0
    bw.write_bits(0, 1)?; // ctx 1 → cluster 0
    bw.write_bits(1, 1)?; // ctx 2 → cluster 1 (predictor)
    bw.write_bits(0, 1)?; // ctx 3 → cluster 0
    bw.write_bits(0, 1)?; // ctx 4 → cluster 0
    bw.write_bits(0, 1)?; // ctx 5 → cluster 0

    bw.write_bit(1); // use_prefix_code = 1
                     // log_alphabet_size = 15 (fixed for prefix branch).

    // Two HybridUintConfig — both split_exponent = 8.
    // Note: tree-stream values are 0 (most contexts) or 5 (predictor),
    // both well below split=256, so no extra bits are ever consumed.
    write_hybrid_uint_config(bw, 8, 0, 0, 15)?;
    write_hybrid_uint_config(bw, 8, 0, 0, 15)?;

    // Per-cluster preludes per FDIS D.3.1 prefix branch:
    //   1. Count selectors for ALL clusters, in order.
    //   2. Then prefix-code bodies for ALL clusters, in order.
    //
    // Step 1 — count selectors:
    //
    //   Cluster 0: count = 1 → bit 0.
    bw.write_bit(0);
    //   Cluster 1: count = 16. Per spec: bit 1 → count > 1, then n=u(4),
    //   count = 1 + (1<<n) + u(n). For count=16 we use n=3, u(3)=7
    //   → 1 + 8 + 7 = 16.
    bw.write_bit(1); // count > 1
    bw.write_bits(3, 4)?; // n = 3
    bw.write_bits(7, 3)?; // u(3) = 7 → count = 16

    // Step 2 — prefix-code bodies:
    //
    //   Cluster 0: count=1 → `read_prefix_code(br, 1)` short-circuits
    //   in the decoder (alphabet_size == 1 path) and reads NO bits. We
    //   emit nothing here.
    //
    //   Cluster 1: count=16 → simple-prefix code with NSYM=1, symbol=5.
    //   `read_prefix_code(br, 16)` reads kind=u(2)=1 (simple), then
    //   `read_simple_prefix`: nsym=u(2)+1 with u(2)=0 → nsym=1, then
    //   one u(ceil(log2(16))) = u(4) for the symbol id (= 5).
    bw.write_bits(1, 2)?; // kind = 1 → simple prefix
    bw.write_bits(0, 2)?; // nsym - 1 = 0 → nsym = 1
    bw.write_bits(5, 4)?; // symbol id = 5 (4 bits since alphabet_size=16)

    // The decoder now reads all subsequent tree-stream tokens from
    // these two single-symbol codes:
    //   T[1] property+1 (ctx=1 → cluster 0) → 0 → leaf
    //   T[2] predictor   (ctx=2 → cluster 1) → 5 → Gradient
    //   T[3] uoffset     (ctx=3 → cluster 0) → 0 → offset = 0
    //   T[4] mul_log     (ctx=4 → cluster 0) → 0
    //   T[5] mul_bits    (ctx=5 → cluster 0) → 0 → multiplier = 1
    //
    // No bits are emitted for the actual token decodes (degenerate
    // codes are 0 bits per decode).
    Ok(())
}

/// Round-4 ANS symbol stream: prelude + ANS-encoded (token, extras)
/// pairs.
///
/// Prelude shape (matches `EntropyStream::read(br, num_dist=1)` in the
/// ANS branch):
///
/// ```text
/// u(1) lz77_enabled = 0
/// u(1) use_prefix_code = 0  (ANS branch)
/// u(2) log_alphabet_size_minus_5 = 3  (log_alphabet_size = 8, bucket_size=16)
/// HybridUintConfig:
///     split_exp_bits = ceil(log2(8+1)) = 4 → u(4) = 0  (split_exponent = 0)
///     since split_exponent != log_alphabet_size, read msb_bits + lsb_bits.
///     msb_bits = ceil(log2(0+1)) = 0 → no read; msb=0
///     lsb_bits = ceil(log2(0-0+1)) = 0 → no read; lsb=0
/// ANS distribution (D.3.4)
/// ANS state init u(32) + per-symbol decode interleaved with refills + extras
/// ```
///
/// `log_alphabet_size = 8` is the spec maximum (per FDIS:
/// `log_alphabet_size = 5 + u(2)`, max u(2)=3 → 8). Bucket_size = 16 is
/// then the alignment unit for the per-symbol `D[s]` values that
/// satisfies the alias-table bijection invariant — see
/// `crate::ans_encoder` module docs.
///
/// The 256-symbol alphabet comfortably covers the round-4 token range:
/// * Gradient-residual `pack_signed` outputs in `[0, 511]`
/// * Token T = floor(log2(value)) + 1 for value >= 1, else 0
/// * Tokens 0..=9 cover [0, 512) → 10 tokens, well under 256.
fn write_ans_symbol_stream(bw: &mut BitWriter, tokens: &[AnsTokenWithExtras]) -> Result<()> {
    let log_alpha: u32 = 8;
    let table_size: usize = 1usize << log_alpha;

    // 1. Build the histogram of bare ANS tokens (NOT extras).
    let mut counts = vec![0u32; table_size];
    for tok in tokens {
        if (tok.token as usize) >= table_size {
            return Err(Error::other(format!(
                "JXL encoder: token {} >= alphabet_size {}",
                tok.token, table_size
            )));
        }
        counts[tok.token as usize] = counts[tok.token as usize].saturating_add(1);
    }
    let total: u32 = counts.iter().sum();
    if total == 0 {
        return Err(Error::other(
            "JXL encoder: ANS symbol stream has zero tokens",
        ));
    }

    // 2. Quantise to a 4096-summing distribution with each entry a
    //    multiple of bucket_size (round-4 alignment constraint).
    let d = quantise_distribution_aligned(&counts, log_alpha)?;
    let alias = AliasTable::build(&d, log_alpha)?;
    let inv = build_inverse_alias(&d, &alias)?;

    // 3. Emit the prelude.
    bw.write_bit(0); // lz77_enabled = 0
                     // num_dist = 1 → cluster_map skipped.
    bw.write_bit(0); // use_prefix_code = 0 (ANS branch)
                     // log_alphabet_size_minus_5 = log_alpha - 5 = 3.
    bw.write_bits(log_alpha - 5, 2)?;

    // HybridUintConfig: log_alphabet_size = 8.
    // We pick split_exponent = 0 → token T < 1 returns T directly,
    // tokens >= 1 carry their value bit-decomposed: token T encodes
    // values in [2^(T-1), 2^T) with (T-1) extra bits.
    write_hybrid_uint_config(bw, 0, 0, 0, log_alpha)?;

    // 4. Emit the ANS distribution preamble.
    write_distribution(bw, &d, log_alpha)?;

    // 5. Emit the ANS-coded tokens with interleaved extras.
    encode_symbols_with_extras(bw, tokens, &d, &inv, &alias)?;
    Ok(())
}

/// Emit a `HybridUintConfig` per FDIS Listing D.7.
///
/// `log_alphabet_size` is fixed at 15 for our prefix branch; the field
/// widths derived from it are:
///   split_exp_bits = ceil(log2(log_alphabet_size + 1)) = 4
///   msb_bits       = ceil(log2(split_exponent + 1))
///   lsb_bits       = ceil(log2(split_exponent - msb + 1))
fn write_hybrid_uint_config(
    bw: &mut BitWriter,
    split_exponent: u32,
    msb: u32,
    lsb: u32,
    log_alphabet_size: u32,
) -> Result<()> {
    if split_exponent > log_alphabet_size {
        return Err(Error::other(
            "JXL encoder: split_exponent > log_alphabet_size",
        ));
    }
    let split_exp_bits = ceil_log2(log_alphabet_size + 1);
    bw.write_bits(split_exponent, split_exp_bits)?;
    if split_exponent != log_alphabet_size {
        let msb_bits = ceil_log2(split_exponent + 1);
        bw.write_bits(msb, msb_bits)?;
        let lsb_bits = ceil_log2(split_exponent - msb + 1);
        bw.write_bits(lsb, lsb_bits)?;
    }
    Ok(())
}

/// Walk the input pixels in (channel, y, x) order and produce the
/// (token, extras) sequence the ANS symbol stream will encode.
///
/// Channel layout for our `colour_count + num_extra_channels`:
///   * Gray   → channel 0 = Y
///   * RGB    → channels 0,1,2 = R,G,B
///   * RGBA   → channels 0,1,2,3 = R,G,B,A (alpha as extra)
///
/// Predictor is Gradient (5) per the single-leaf MA tree; the residual
/// is `sample - clamp(W + N - NW, min(W,N), max(W,N))`. For the very
/// first pixel both W and N are 0, so the predicted value is 0 and the
/// residual = sample. As we walk the image, predictions track local
/// values closely on natural images, driving most residuals to small
/// magnitudes.
///
/// Token encoding (matches the symbol-stream `HybridUintConfig` with
/// split_exponent=0, msb=0, lsb=0):
///   value 0    → token 0, no extra bits
///   value k>=1 → token T = floor(log2(k)) + 1, extra bits = k - 2^(T-1)
///                ((T-1) extra bits)
fn collect_pixel_tokens(
    width: u32,
    height: u32,
    pixels: &[u8],
    format: InputFormat,
) -> Result<Vec<AnsTokenWithExtras>> {
    let stride = format.channel_count() as usize;
    let w = width as usize;
    let h = height as usize;
    let nc = format.channel_count() as usize;
    let mut recon: Vec<Vec<i32>> = (0..nc).map(|_| vec![0i32; w * h]).collect();
    let mut out: Vec<AnsTokenWithExtras> = Vec::with_capacity(w * h * nc);
    for c in 0..nc {
        for y in 0..h {
            for x in 0..w {
                let v = pixels[(y * w + x) * stride + c] as i32;
                let p = gradient_predict(&recon[c], w, h, x, y);
                let diff = v - p;
                recon[c][y * w + x] = v;
                let packed = pack_signed(diff);
                let (token, extra, n_extra) = encode_packed_to_token(packed)?;
                out.push(AnsTokenWithExtras {
                    token: token as u16,
                    extra_value: extra,
                    extra_bits: n_extra,
                });
            }
        }
    }
    Ok(out)
}

/// Map a packed-signed residual `value` to `(token, extra_value, extra_bits)`
/// using the symbol-stream HybridUintConfig (split_exponent=0, msb=0,
/// lsb=0). This is the inverse of `read_uint` for that specific config.
///
/// * `value == 0` → `(0, 0, 0)` — token 0, no extras.
/// * `value >= 1` → token T = floor(log2(value)) + 1, extra =
///   `value - 2^(T-1)` in `T - 1` bits.
///
/// Tokens > 255 are rejected (round-4 alphabet is 256 = `1 << 8`).
fn encode_packed_to_token(value: u32) -> Result<(u32, u32, u32)> {
    if value == 0 {
        return Ok((0, 0, 0));
    }
    let n = 32 - value.leading_zeros(); // floor(log2(value)) + 1
    if n > 255 {
        return Err(Error::other(format!(
            "JXL encoder: residual value {value} produces token {n} > 255 (alphabet cap)"
        )));
    }
    let token = n;
    let n_extra = n - 1;
    let extra = if n_extra == 0 {
        0
    } else {
        value & ((1u32 << n_extra) - 1)
    };
    Ok((token, extra, n_extra))
}

/// FDIS Listing C.16 predictor 5 (Gradient): clamp(W+N-NW, min, max).
///
/// Out-of-bounds neighbours fall back per spec:
///   x>0: W = sample(x-1, y); else W = (y>0 ? sample(x, y-1) : 0)
///   y>0: N = sample(x, y-1); else N = W
///   x>0 && y>0: NW = sample(x-1, y-1); else NW = W
fn gradient_predict(buf: &[i32], w: usize, _h: usize, x: usize, y: usize) -> i32 {
    let west = if x > 0 {
        buf[y * w + (x - 1)]
    } else if y > 0 {
        buf[(y - 1) * w + x]
    } else {
        0
    };
    let north = if y > 0 { buf[(y - 1) * w + x] } else { west };
    let northwest = if x > 0 && y > 0 {
        buf[(y - 1) * w + (x - 1)]
    } else {
        west
    };
    let grad = west.wrapping_add(north).wrapping_sub(northwest);
    let lo = west.min(north);
    let hi = west.max(north);
    grad.clamp(lo, hi)
}

fn ceil_log2(x: u32) -> u32 {
    if x <= 1 {
        0
    } else {
        32 - (x - 1).leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;

    #[test]
    fn ceil_log2_helper() {
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(9), 4);
        assert_eq!(ceil_log2(16), 4);
        assert_eq!(ceil_log2(17), 5);
    }

    #[test]
    fn encode_packed_to_token_zero() {
        let (t, e, n) = encode_packed_to_token(0).unwrap();
        assert_eq!(t, 0);
        assert_eq!(e, 0);
        assert_eq!(n, 0);
    }

    #[test]
    fn encode_packed_to_token_round_trip_small_values() {
        // Mirror the decoder's ReadUint formula for split=1, msb=0,
        // lsb=0: token T (T >= 1) → value = (1 << (T-1)) | extra,
        // where `extra` is read as `T - 1` bits.
        for v in [0u32, 1, 2, 3, 4, 7, 16, 100, 200, 511] {
            let (token, extra, n_extra) = encode_packed_to_token(v).unwrap();
            let recovered: u32 = if v == 0 {
                assert_eq!(token, 0);
                assert_eq!(n_extra, 0);
                0
            } else {
                assert_eq!(token, 32 - v.leading_zeros());
                assert_eq!(n_extra, token - 1);
                let base = 1u32 << (token - 1);
                base | extra
            };
            assert_eq!(recovered, v, "token round-trip failed for {v}");
        }
    }

    #[test]
    fn gradient_predict_first_pixel_is_zero() {
        let buf = vec![0i32; 4];
        // (0, 0): W = N = NW = 0 → grad = 0, clamp(0, 0, 0) = 0.
        assert_eq!(gradient_predict(&buf, 2, 2, 0, 0), 0);
    }

    #[test]
    fn gradient_predict_uses_west_at_top_row() {
        // Top row, x=1: W = buf[0] = 5, N = W = 5, NW = W = 5.
        // grad = 5 + 5 - 5 = 5, clamp(5, 5, 5) = 5.
        let buf = vec![5, 0, 0, 0];
        assert_eq!(gradient_predict(&buf, 2, 2, 1, 0), 5);
    }

    #[test]
    fn gradient_predict_uses_north_at_left_column() {
        // Left col, y=1: W = sample(0, 0) = 7 (per spec fallback).
        // N = buf[0] = 7. NW = W = 7. grad = 7 + 7 - 7 = 7.
        let buf = vec![7, 0, 0, 0];
        assert_eq!(gradient_predict(&buf, 2, 2, 0, 1), 7);
    }

    #[test]
    fn gradient_predict_clamp_below_min() {
        // Construct: W=10, N=20, NW=15 → grad = 10+20-15 = 15. min=10, max=20.
        // 15 in [10, 20] → returns 15.
        // Now make grad below min: W=10, N=20, NW=25 → grad = 10+20-25 = 5.
        // 5 < min(10, 20) = 10 → clamp to 10.
        let buf = vec![25, 20, 10, 0]; // (0,0)=25, (1,0)=20, (0,1)=10, (1,1)=?
                                       // For (1, 1): W=buf[2]=10, N=buf[1]=20, NW=buf[0]=25.
        assert_eq!(gradient_predict(&buf, 2, 2, 1, 1), 10);
    }

    #[test]
    fn gradient_predict_clamp_above_max() {
        // W=10, N=20, NW=5 → grad = 10+20-5 = 25 > max(10, 20) = 20 → 20.
        let buf = vec![5, 20, 10, 0];
        assert_eq!(gradient_predict(&buf, 2, 2, 1, 1), 20);
    }

    #[test]
    fn encode_smallest_image_produces_jxl_signature() {
        let pixels = vec![128u8; 3]; // 1x1 RGB
        let bytes = encode_one_frame(1, 1, &pixels, InputFormat::Rgb8).unwrap();
        assert_eq!(&bytes[0..2], &[0xFF, 0x0A]);
    }

    #[test]
    fn size_header_round_trip_via_reader() {
        let mut bw = BitWriter::new();
        write_size_header(&mut bw, 64, 32).unwrap();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let size = crate::metadata_fdis::SizeHeaderFdis::read(&mut br).unwrap();
        assert_eq!(size.width, 64);
        assert_eq!(size.height, 32);
    }

    #[test]
    fn image_metadata_round_trip_via_reader() {
        let mut bw = BitWriter::new();
        write_image_metadata(&mut bw, InputFormat::Rgb8).unwrap();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let meta = crate::metadata_fdis::ImageMetadataFdis::read(&mut br).unwrap();
        assert!(!meta.xyb_encoded);
        assert_eq!(meta.bit_depth.bits_per_sample, 8);
        assert_eq!(meta.num_extra_channels, 0);
        use crate::metadata_fdis::ColourSpace;
        assert_eq!(meta.colour_encoding.colour_space, ColourSpace::Rgb);
    }

    #[test]
    fn image_metadata_rgba_includes_alpha_extra() {
        let mut bw = BitWriter::new();
        write_image_metadata(&mut bw, InputFormat::Rgba8).unwrap();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let meta = crate::metadata_fdis::ImageMetadataFdis::read(&mut br).unwrap();
        assert_eq!(meta.num_extra_channels, 1);
        assert_eq!(meta.extra_channel_info.len(), 1);
        use crate::metadata_fdis::ExtraChannelType;
        assert_eq!(meta.extra_channel_info[0].kind, ExtraChannelType::Alpha);
    }

    #[test]
    fn frame_header_round_trip_via_reader() {
        // size header + metadata + pad + frame_header
        let mut bw = BitWriter::new();
        write_size_header(&mut bw, 32, 32).unwrap();
        write_image_metadata(&mut bw, InputFormat::Rgb8).unwrap();
        bw.pad_to_byte();
        write_frame_header(&mut bw, 32, 32, InputFormat::Rgb8).unwrap();
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let size = crate::metadata_fdis::SizeHeaderFdis::read(&mut br).unwrap();
        let _meta = crate::metadata_fdis::ImageMetadataFdis::read(&mut br).unwrap();
        br.pu0().unwrap();
        let params = crate::frame_header::FrameDecodeParams {
            xyb_encoded: false,
            num_extra_channels: 0,
            have_animation: false,
            have_animation_timecodes: false,
            image_width: size.width,
            image_height: size.height,
        };
        let fh = crate::frame_header::FrameHeader::read(&mut br, &params).unwrap();
        assert_eq!(fh.encoding, crate::frame_header::Encoding::Modular);
        assert_eq!(fh.passes.num_passes, 1);
        assert!(fh.is_last);
        assert_eq!(fh.width, 32);
        assert_eq!(fh.height, 32);
    }

    // (Pre-existing test `gradient_leaf_ma_tree_round_trips_via_decoder`
    // removed: it tried to call `MaTreeFdis::read` on just the tree-stream
    // bits, but `MaTreeFdis::read` reads BOTH the tree AND the symbol
    // stream prelude — so it always failed with "unexpected end of JXL
    // bitstream". The end-to-end pipeline tests in
    // `encode_decode_roundtrip_*` exercise the tree via the actual
    // decoder path.)

    /// Round-4: verify the full ANS-coded encoder output decodes back
    /// to the original pixel buffer via `decode_one_frame`. Exercises
    /// the entire round-4 ANS pipeline (quantise-aligned → preamble →
    /// state-stream emission → decoder ingest).
    #[test]
    fn ans_coded_grey_8x8_round_trips_through_decoder() {
        // 8x8 deterministic ramp.
        let mut pixels = Vec::with_capacity(64);
        for y in 0..8u8 {
            for x in 0..8u8 {
                pixels.push(x.wrapping_mul(16).wrapping_add(y * 4));
            }
        }
        let bytes = encode_one_frame(8, 8, &pixels, InputFormat::Gray8).unwrap();
        let frame = crate::decode_one_frame(&bytes, None).unwrap();
        assert_eq!(frame.planes.len(), 1);
        assert_eq!(frame.planes[0].data, pixels);
    }

    /// Issue #382 regression: constant-grey ANS frames easily achieve
    /// <1 bit/pixel — the entire 64-symbol stream collapses to a few
    /// preamble bits + the 32-bit final ANS state. The previous
    /// `pixels > bits_remaining` pre-check in `modular_fdis::decode_channels`
    /// rejected this valid bitstream. The check now applies only to
    /// prefix-coded streams, so this round-trip succeeds.
    #[test]
    fn ans_coded_constant_grey_8x8_round_trips_through_decoder() {
        let pixels = vec![0x42u8; 64];
        let bytes = encode_one_frame(8, 8, &pixels, InputFormat::Gray8).unwrap();
        let frame = crate::decode_one_frame(&bytes, None).unwrap();
        assert_eq!(frame.planes.len(), 1);
        assert_eq!(frame.planes[0].data, pixels);
    }

    // Note: RGB round-trip is pending — the round-3 decoder rejects
    // colour_space=Rgb (Grey only). Tracked separately from #382.
}
