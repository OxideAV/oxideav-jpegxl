//! JPEG XL encoder — round-2 lossless modular Gray/RGB/RGBA with
//! the **Gradient** (predictor id 5) MA-tree leaf and a frequency-
//! optimised single-cluster symbol stream.
//!
//! Round 2 deltas vs round 1 (`a53e041`):
//!
//! * **Predictor switched from Zero (0) to Gradient (5).** The MA tree
//!   is now a single-leaf tree carrying `predictor=5, offset=0,
//!   multiplier=1`. The on-wire tree-token entropy stream is now
//!   2-cluster: cluster 0 always returns 0 (used for property+1, offset,
//!   mul_log, mul_bits, decision-node values), cluster 1 always returns 5
//!   (used only for the predictor field). Cluster map for the 6
//!   tree-stream contexts is `[0, 0, 1, 0, 0, 0]`.
//!
//! * **Per-pixel residual = sample - prediction(left, top, topleft)**
//!   per FDIS Listing C.16's clamp(W+N-NW, min(W,N), max(W,N)). Natural
//!   images compress ~3× better since most residuals fall into token 0
//!   (value 0).
//!
//! * **Symbol stream remains a single-cluster uniform 16×4-bit prefix
//!   code** for simplicity. Round 3 should switch to a frequency-based
//!   complex-prefix code (variable code lengths) — this is the next
//!   biggest win after Gradient and is a pure entropy-coder change with
//!   no decoder side effects.
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

    // Symbol stream — round-2 design uses a single-cluster
    // prefix-coded stream. The MA tree's single leaf has ctx=0; so
    // num_ctx=1 and there's only one cluster.
    write_symbol_stream_prelude(bw)?;

    // Per-channel pixel decode (C.16/C.17). Walk pixels in
    // (channel, y, x) order; emit (token, extra_bits) for each
    // gradient-predicted residual.
    write_pixel_data(bw, width, height, pixels, format)
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

/// Emit the symbol-stream prelude (one cluster, complex prefix code
/// over 16 uniform-length-4 token symbols, hybrid config split_exponent=0).
///
/// This stream is parsed by `EntropyStream::read(br, num_dist=1)`. With
/// num_dist=1 the cluster map step is skipped per D.3.1; n_clusters=1.
fn write_symbol_stream_prelude(bw: &mut BitWriter) -> Result<()> {
    bw.write_bit(0); // lz77_enabled = 0
                     // num_dist=1 → cluster_map skipped.

    bw.write_bit(1); // use_prefix_code = 1
                     // log_alphabet_size = 15 (fixed for prefix branch).

    // Hybrid uint config for the single cluster:
    //   split_exponent = 0, msb = 0, lsb = 0 → split = 1
    // Token T <  1: returns T directly. (Only T==0 takes this branch
    //                → returns 0.)
    // Token T >= 1: ReadUint formula. With msb=lsb=0:
    //   above = T - 1
    //   n_extra = above >> 0 = T - 1
    //   n = split_exponent + n_extra = 0 + (T - 1) = T - 1
    //   tok = (T >> 0) & 0 | (1 << 0) = 1
    //   shifted = 1 << n = 1 << (T - 1)
    //   combined = (shifted | extra) << 0 = (1 << (T-1)) | extra
    //   value = combined | 0 = (1 << (T-1)) | extra
    // So token T encodes values in [2^(T-1), 2^T) with (T-1) extra bits.
    //
    //   T=0 → value = 0 (no extra)
    //   T=1 → value = 1 (no extra)
    //   T=2 → values 2..3 (1 extra bit)
    //   T=3 → values 4..7 (2 extra bits)
    //   ...
    //   T=k → values 2^(k-1)..2^k - 1 ((k-1) extra bits)
    //
    // Max packed-signed value for 8-bit Gradient residuals: residuals
    // can be in [-256, 255], so packed values in [0, 511]. 511 fits in
    // T=9 (covers 256..511). So tokens 0..=9 suffice. We round up the
    // alphabet to 16 (next pow2) so every symbol has length 4 in the
    // canonical-Huffman code below.
    write_hybrid_uint_config(bw, 0, 0, 0, 15)?;

    // Prefix code: complex format, 16 symbols all length 4.
    // Count selector: u(1)=1 (count > 1), then n=u(4), count = 1 + (1<<n) + u(n).
    // We want count = 16. Try n = 3: 1 + 8 + u(3) = 9 + u(3). Max value
    // with n=3 is 9 + 7 = 16. So n=3, u(3)=7 → count=16.
    bw.write_bit(1); // count > 1
    bw.write_bits(3, 4)?; // n = 3
    bw.write_bits(7, 3)?; // u(3) = 7 → count = 16
                          //
                          // Then read_prefix_code(br, 16) takes the kind branch:
                          //   kind = u(2) (0 → complex HSKIP=0, 2 → complex HSKIP=2,
                          //                3 → complex HSKIP=3, 1 → simple).
                          // We use kind=0 (complex, HSKIP=0).
    bw.write_bits(0, 2)?; // kind = 0 → complex HSKIP=0

    // Emit 18 cl_code-length values via the CLCL_VL_TABLE codes:
    //   CLCL_VL_TABLE entries (sym, code, len):
    //     (0, 0b00, 2)     LSB-first integer 0b00 = 0  (2 bits)
    //     (3, 0b01, 2)     LSB-first integer 0b10 = 2  (2 bits)
    //     (4, 0b10, 2)     LSB-first integer 0b01 = 1  (2 bits)
    //     (2, 0b110, 3)    LSB-first integer 0b011 = 3 (3 bits)
    //     (1, 0b1110, 4)   LSB-first integer 0b0111 = 7 (4 bits)
    //     (5, 0b1111, 4)   LSB-first integer 0b1111 = 15 (4 bits)
    //
    // We want only clcl[4] = 1 (single non-zero entry → degenerate
    // cl_code that always returns sym 4 with 0 bits, satisfying the
    // RFC 7932 §3.5 special case the decoder handles in
    // read_complex_prefix).
    //
    // K_CODE_LENGTH_CODE_ORDER = [1, 2, 3, 4, 0, 5, 17, 6, 16, ...].
    // Index 3 → clcl[4]. Indices 0..2, 4..17 → other clcl positions.
    //
    // So we emit 17 zeros + 1 one in the right slot.
    for i in 0..18 {
        // Position in K_CODE_LENGTH_CODE_ORDER: i. Target clcl index =
        // K_CODE_LENGTH_CODE_ORDER[i]. We want clcl[4] = 1, all others = 0.
        let want_one = i == 3; // K_CODE_LENGTH_CODE_ORDER[3] == 4
        if want_one {
            // Emit cl-symbol "1" (length 1 in cl_code → goes through clcl[4] = 1).
            // The clcl array holds the LENGTH of the cl-code's prefix
            // codeword for cl-symbol i. We want clcl[4] = 1 so that
            // cl-symbol 4 has a length-1 codeword. The CLCL_VL_TABLE
            // entry for the literal length value 1 is (1, 0b1110, 4).
            // LSB-first bits to emit: bit_reverse(0b1110, 4) = 0b0111.
            bw.write_bits(0b0111, 4)?;
        } else {
            // Emit cl-symbol "0" (cl-code length 0). CLCL_VL_TABLE entry
            // (0, 0b00, 2). LSB-first: 0b00.
            bw.write_bits(0b00, 2)?;
        }
    }

    // Now the cl_code (with single non-zero clcl[4]=1) is the
    // degenerate single-symbol cl_code that returns cl-symbol 4 with
    // 0 bits per decode. The decoder then reads `count`=16 cl-symbol
    // decodes to populate the per-symbol code-length array. Each
    // returns 4, so all 16 symbols get length 4. We emit zero bits
    // here.

    // Done with prelude.
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

/// Walk the input pixels in (channel, y, x) order and emit one symbol
/// per pixel using the symbol-stream prefix code (token T = packed
/// gradient residual, with extra bits when packed >= 1).
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
fn write_pixel_data(
    bw: &mut BitWriter,
    width: u32,
    height: u32,
    pixels: &[u8],
    format: InputFormat,
) -> Result<()> {
    let stride = format.channel_count() as usize;
    let w = width as usize;
    let h = height as usize;
    // Maintain a per-channel reconstruction buffer so each prediction
    // sees the SAME values the decoder will see. (Since we round-trip
    // exactly, the reconstruction == the input — but mirroring the
    // decode pattern keeps any future quantisation honest.)
    let nc = format.channel_count() as usize;
    let mut recon: Vec<Vec<i32>> = (0..nc).map(|_| vec![0i32; w * h]).collect();
    for c in 0..nc {
        for y in 0..h {
            for x in 0..w {
                let v = pixels[(y * w + x) * stride + c] as i32;
                let p = gradient_predict(&recon[c], w, h, x, y);
                let diff = v - p;
                recon[c][y * w + x] = v;
                let packed = pack_signed(diff);
                write_token_with_extras(bw, packed)?;
            }
        }
    }
    Ok(())
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

/// Encode one packed-signed residual using the symbol stream's token
/// distribution:
///   value 0    → token 0, no extra bits
///   value k>=1 → token T = floor(log2(k)) + 1, extra bits = k - 2^(T-1)
///                (T-1 bits)
///
/// The token is then encoded with the symbol-stream prefix code. Round
/// 2 uses a uniform 4-bit code over 16 symbols, so each token costs
/// exactly 4 bits regardless of value (canonical-Huffman LSB-first
/// representation).
fn write_token_with_extras(bw: &mut BitWriter, value: u32) -> Result<()> {
    let (token, extra, n_extra) = if value == 0 {
        (0u32, 0u32, 0u32)
    } else {
        let n = 32 - value.leading_zeros(); // floor(log2(value)) + 1
        if n > 16 {
            return Err(Error::other(format!(
                "JXL encoder: residual value {value} exceeds token alphabet"
            )));
        }
        let token = n;
        let n_extra = n - 1;
        let extra = if n_extra == 0 {
            0
        } else {
            value & ((1u32 << n_extra) - 1)
        };
        (token, extra, n_extra)
    };
    if token > 15 {
        return Err(Error::other(format!(
            "JXL encoder: token {token} exceeds 16-symbol alphabet"
        )));
    }
    // Emit the token's prefix-code codeword. Canonical Huffman with all
    // 16 symbols at length 4 produces codes 0..15 in symbol-id order,
    // and JXL reads bits LSB-first → the lookup-table index is the
    // BIT-REVERSED canonical code. So the bits we emit are
    // bit_reverse(token, 4).
    let lsb_first = bit_reverse_4(token);
    bw.write_bits(lsb_first, 4)?;
    if n_extra > 0 {
        bw.write_bits(extra, n_extra)?;
    }
    Ok(())
}

fn bit_reverse_4(x: u32) -> u32 {
    let mut out = 0u32;
    for i in 0..4 {
        if (x >> i) & 1 != 0 {
            out |= 1 << (3 - i);
        }
    }
    out
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
    fn bit_reverse_4_known_values() {
        assert_eq!(bit_reverse_4(0b0000), 0b0000);
        assert_eq!(bit_reverse_4(0b0001), 0b1000);
        assert_eq!(bit_reverse_4(0b1000), 0b0001);
        assert_eq!(bit_reverse_4(0b1100), 0b0011);
        assert_eq!(bit_reverse_4(0b1111), 0b1111);
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
        let pixels = vec![128u8; 1 * 1 * 3];
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
}
