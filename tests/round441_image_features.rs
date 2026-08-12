//! Round 441 — Patches (§C.4.5 + §K.2) and Splines (§C.4.6 + §K.3)
//! wired into the registered decode path, pinned against black-box
//! reference decodes.
//!
//! ## Fixtures
//!
//! * `patches_dots_256x256.jxl` — locally generated (independent
//!   encoder, black-box invocation) **lossless Modular** stream whose
//!   dot content forced a patch dictionary: frame 0 is a Table C.3
//!   kReferenceOnly dictionary sheet saved pre-CT into `Reference[3]`,
//!   frame 1 blends 60 single-position 3×3/3×2 patches (kAdd). The
//!   lossless coding makes the reference decode **bit-exact**.
//! * `patches_glyphs_256x256.jxl` — same construction from a
//!   repeated-glyph image: one 9×5 dictionary sheet, non-square
//!   patches, multi-position (`count > 1`) delta-coded placements.
//!   Also bit-exact.
//! * `patches_vardct_256x256.jxl` — the lossy VarDCT sibling (dots on
//!   a smooth gradient, distance 1.0): patch dictionary is a Modular
//!   **XYB** kReferenceOnly frame consumed by a VarDCT main frame in
//!   the pre-CT float-XYB domain.
//! * `modular_xyb_256x256.jxl` — lossy **Modular XYB** stream (no
//!   features) pinning the §L.2 kModular rescale `/128` erratum
//!   (see `xyb::modular_xyb_rescale`).
//! * `spline_synth_64x64.jxl` — hand-assembled per the FDIS bit
//!   layouts by `build_spline_codestream()` below (no encoder
//!   produces spline streams): a 64×64 all-zero Modular XYB frame
//!   whose kSplines dictionary carries one 3-control-point horizontal
//!   spline (Y DC +4 → 0.30, σ DC +16 → 5.33). The committed `.jxl`
//!   was decoded by the reference decoder binary (black-box) to
//!   produce `spline_synth_64x64_expected.png`; the builder is
//!   asserted byte-identical to the committed stream so the fixture
//!   provenance stays reproducible.
//!
//! Every `_expected.png` is the black-box reference decode of the
//! committed stream.

use oxideav_jpegxl::decode_all_frames;
use std::io::Cursor;

fn png_rgb(bytes: &[u8]) -> (usize, usize, Vec<u8>) {
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let mut reader = decoder.read_info().expect("png header");
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).expect("png frame");
    assert_eq!(info.color_type, png::ColorType::Rgb);
    buf.truncate(info.buffer_size());
    (info.width as usize, info.height as usize, buf)
}

/// Per-channel (MAD, max) against a reference decode.
fn compare_rgb(
    frame: &oxideav_core::VideoFrame,
    w: usize,
    h: usize,
    want: &[u8],
) -> [(f64, u32); 3] {
    assert_eq!(frame.planes.len(), 3, "expected RGB planes");
    let mut out = [(0f64, 0u32); 3];
    for c in 0..3 {
        let plane = &frame.planes[c];
        assert_eq!(plane.stride, w, "plane {c} stride");
        let mut sum = 0f64;
        let mut max = 0u32;
        for y in 0..h {
            for x in 0..w {
                let got = plane.data[y * w + x] as i32;
                let exp = want[(y * w + x) * 3 + c] as i32;
                let d = (got - exp).unsigned_abs();
                sum += d as f64;
                max = max.max(d);
            }
        }
        out[c] = (sum / (w * h) as f64, max);
    }
    out
}

// ---------------------------------------------------------------------------
// Patches — lossless Modular (bit-exact oracle)
// ---------------------------------------------------------------------------

#[test]
fn round441_patches_dots_lossless_bit_exact() {
    let jxl = include_bytes!("fixtures/patches_dots_256x256.jxl");
    let (w, h, want) = png_rgb(include_bytes!("fixtures/patches_dots_256x256_expected.png"));
    let frames = decode_all_frames(jxl, None).expect("patches stream decodes");
    assert_eq!(frames.len(), 1, "one presented frame (dict frame skipped)");
    let stats = compare_rgb(&frames[0], w, h, &want);
    for (c, &(mad, max)) in stats.iter().enumerate() {
        assert_eq!(max, 0, "channel {c} must be bit-exact, MAD {mad}");
    }
}

#[test]
fn round441_patches_glyphs_multiposition_bit_exact() {
    let jxl = include_bytes!("fixtures/patches_glyphs_256x256.jxl");
    let (w, h, want) = png_rgb(include_bytes!(
        "fixtures/patches_glyphs_256x256_expected.png"
    ));
    let frames = decode_all_frames(jxl, None).expect("glyph patches stream decodes");
    assert_eq!(frames.len(), 1);
    let stats = compare_rgb(&frames[0], w, h, &want);
    for (c, &(mad, max)) in stats.iter().enumerate() {
        assert_eq!(max, 0, "channel {c} must be bit-exact, MAD {mad}");
    }
}

/// The Listing C.2 `mode`-context open question (see
/// `patches::PATCH_MODE_CTX_DEFAULT`): every available specimen
/// clusters contexts 5 and 6 together, so both readings must decode
/// identically. If a future stream separates the clusters this test's
/// premise breaks and the context is arbitrated for real.
#[test]
fn round441_patches_mode_ctx_5_vs_6_equivalent_on_wire() {
    let jxl = include_bytes!("fixtures/patches_dots_256x256.jxl");
    oxideav_jpegxl::patches::set_patch_mode_ctx_override(Some(5));
    let a = decode_all_frames(jxl, None).expect("ctx 5 decode");
    oxideav_jpegxl::patches::set_patch_mode_ctx_override(Some(6));
    let b = decode_all_frames(jxl, None).expect("ctx 6 decode");
    oxideav_jpegxl::patches::set_patch_mode_ctx_override(None);
    assert_eq!(a.len(), b.len());
    for (fa, fb) in a.iter().zip(b.iter()) {
        for (pa, pb) in fa.planes.iter().zip(fb.planes.iter()) {
            assert_eq!(pa.data, pb.data, "mode ctx 5 vs 6 diverged");
        }
    }
}

// ---------------------------------------------------------------------------
// Patches — VarDCT main frame, Modular-XYB dictionary (pre-CT domain)
// ---------------------------------------------------------------------------

/// The dots render (kAdd in the float-XYB pre-CT domain, from the
/// Reference[3] kReferenceOnly dictionary). The residual band is the
/// known VarDCT synthetic-content deficiency (round 437; the dot
/// blocks are Hornuss / DCT2×2 impulses whose declared NonZeros exceed
/// the decoded count — see the round-441 report), NOT patch error:
/// the same content without the patches feature shows the same band.
#[test]
fn round441_patches_vardct_xyb_ratchet() {
    let jxl = include_bytes!("fixtures/patches_vardct_256x256.jxl");
    let (w, h, want) = png_rgb(include_bytes!(
        "fixtures/patches_vardct_256x256_expected.png"
    ));
    let frames = decode_all_frames(jxl, None).expect("vardct patches stream decodes");
    assert_eq!(frames.len(), 1);
    let stats = compare_rgb(&frames[0], w, h, &want);
    // Round-441 measured: MAD 1.73 / 0.79 / 0.73, max 15 / 13 / 8.
    let bounds = [(2.2, 40), (1.2, 40), (1.2, 40)];
    for (c, (&(mad, max), &(mad_b, max_b))) in stats.iter().zip(bounds.iter()).enumerate() {
        assert!(
            mad < mad_b && max <= max_b,
            "channel {c}: MAD {mad} (bound {mad_b}), max {max} (bound {max_b})"
        );
    }
}

// ---------------------------------------------------------------------------
// §L.2 kModular XYB rescale — the /128 erratum
// ---------------------------------------------------------------------------

/// Lossy Modular-XYB stream (no features): under the literal FDIS §L.2
/// reading (`X = X' × m_x_lf_unscaled`) every sample saturates (the
/// planes come out ≈128× too large); with the /128 the decode sits in
/// the ±1 rounding band of the reference decode. See
/// `xyb::modular_xyb_rescale`.
#[test]
fn round441_modular_xyb_rescale_erratum() {
    let jxl = include_bytes!("fixtures/modular_xyb_256x256.jxl");
    let (w, h, want) = png_rgb(include_bytes!("fixtures/modular_xyb_256x256_expected.png"));
    let frames = decode_all_frames(jxl, None).expect("modular-xyb stream decodes");
    assert_eq!(frames.len(), 1);
    let stats = compare_rgb(&frames[0], w, h, &want);
    for (c, &(mad, max)) in stats.iter().enumerate() {
        assert!(
            mad < 0.35 && max <= 1,
            "channel {c}: MAD {mad}, max {max} — expected the ±1 rounding band"
        );
    }
}

// ---------------------------------------------------------------------------
// Splines — hand-assembled wire fixture
// ---------------------------------------------------------------------------

/// LSB-first bit writer mirroring `BitReader` (the JXL bit order).
struct BitWriter {
    bytes: Vec<u8>,
    bit: u32,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter {
            bytes: Vec::new(),
            bit: 0,
        }
    }
    fn push(&mut self, value: u32, nbits: u32) {
        for i in 0..nbits {
            let b = (value >> i) & 1;
            if self.bit == 0 {
                self.bytes.push(0);
            }
            let last = self.bytes.len() - 1;
            self.bytes[last] |= (b as u8) << self.bit;
            self.bit = (self.bit + 1) % 8;
        }
    }
    fn pad_to_byte(&mut self) {
        self.bit = 0;
    }
}

/// Find the `(pattern, len)` a prefix code assigns to `symbol` by trial
/// decode — robust against the D.2 canonical-ordering details.
fn code_for_symbol(es_bytes: &[u8], num_dist: usize, cluster_ctx: u32, symbol: u32) -> (u32, u32) {
    use oxideav_jpegxl::bitreader::BitReader;
    use oxideav_jpegxl::modular_fdis::EntropyStream;
    for len in 1..=15u32 {
        for pattern in 0..(1u32 << len) {
            // Fresh stream parse per trial (prefix streams carry no
            // cross-symbol state).
            let mut br = BitReader::new(es_bytes);
            let es = EntropyStream::read(&mut br, num_dist).expect("prelude parses");
            let prelude_bits = br.bits_read();
            // Append the candidate bits after the prelude.
            let mut w = BitWriter::new();
            let mut rebits = BitReader::new(es_bytes);
            for _ in 0..prelude_bits {
                w.push(rebits.read_bits(1).unwrap(), 1);
            }
            w.push(pattern, len);
            // Extra padding so short reads never hit EOF.
            w.push(0, 24);
            let buf = w.bytes.clone();
            let mut br2 = BitReader::new(&buf);
            let mut es2 = EntropyStream::read(&mut br2, num_dist).expect("prelude reparses");
            let before = br2.bits_read();
            let _ = &es;
            if let Ok(sym) = es2.decode_symbol(&mut br2, cluster_ctx) {
                let used = br2.bits_read() - before;
                if sym == symbol && used == len as usize {
                    return (pattern, len);
                }
            }
        }
    }
    panic!("no code found for symbol {symbol}");
}

/// Hand-assemble the 64×64 kSplines codestream described in the module
/// docs. Every field mirrors the FDIS bundle tables (A.3, A.6, C.2,
/// C.3.3, Table C.10, §C.4.6, D.2/D.3, C.4.8/C.9, D.4.2).
fn build_spline_codestream() -> Vec<u8> {
    let mut w = BitWriter::new();

    // SizeHeader (Table A.3): small=1, height u(5)=7 → 64, ratio=1 (1:1).
    w.push(1, 1);
    w.push(7, 5);
    w.push(1, 3);

    // ImageMetadata (Table A.16): all_default=1; the unconditional
    // `default_transform` tail bit = 1 (defaults, nothing follows).
    w.push(1, 1);
    w.push(1, 1);

    // Byte-align before the frame array (§6.3).
    w.pad_to_byte();

    // FrameHeader (Table C.2), xyb_encoded image, no extra channels:
    w.push(0, 1); // all_default = 0
    w.push(0, 2); // frame_type = kRegular
    w.push(1, 1); // encoding = kModular
    w.push(1, 2); // flags U64 selector 1 (1..16)
    w.push(15, 4); //   payload 15 → flags = 16 = kSplines
    w.push(0, 2); // upsampling U32 selector 0 → 1
    w.push(1, 2); // group_size_shift = 1 (group_dim 256)
    w.push(0, 2); // passes.num_passes U32 selector 0 → 1
    w.push(0, 1); // have_crop = 0
    w.push(0, 2); // blending_info.mode U32 selector 0 → kReplace
    w.push(1, 1); // is_last = 1
    w.push(0, 2); // name_len U32 selector 0 → 0
                  // RestorationFilter (2024 Table J.1): all_default=0, gab=0,
                  // epf_iters=0, extensions=0 — filters explicitly off.
    w.push(0, 1);
    w.push(0, 1);
    w.push(0, 2);
    w.push(0, 2); // rf.extensions U64 selector 0
    w.push(0, 2); // frame extensions U64 selector 0

    // TOC (C.3.3): single entry (num_groups = 1, num_passes = 1).
    w.push(0, 1); // permuted = 0
    w.pad_to_byte();
    let toc_entry_pos = w.bytes.len(); // patched after the section is built
    w.push(0, 2); // entry U32 selector 0 → u(10)
    w.push(0, 10); // placeholder length
    w.pad_to_byte();

    // ---- LfGlobal section (single TOC entry) ----
    let section_start = w.bytes.len();

    // §C.4.6 Splines bundle: one §D.3 stream with 6 distributions.
    // Prelude: lz77=0; clustering simple, nbits=1, ctx0..4 → cluster 0,
    // ctx5 → cluster 1; prefix codes; per-cluster HybridUintConfig with
    // split_exponent = 15 (= log_alphabet_size → token IS the value);
    // per-cluster alphabet size 33; simple prefix codes:
    //   cluster 0 (structure): symbols {0, 2, 8, 32}, 4×2-bit codes
    //   cluster 1 (coefficients): symbols {0, 8, 32}, lengths {1, 2, 2}
    let mut es = BitWriter::new();
    es.push(0, 1); // lz77 disabled
    es.push(1, 1); // clustering: is_simple
    es.push(1, 2); // nbits = 1
    for ctx in 0..6u32 {
        es.push(u32::from(ctx == 5), 1);
    }
    es.push(1, 1); // use_prefix_code
    es.push(15, 4); // cluster 0 config: split_exponent = 15
    es.push(15, 4); // cluster 1 config: split_exponent = 15
    for _ in 0..2 {
        // alphabet size: 1 + (1 << 5) + u(5)=0 → 33
        es.push(1, 1);
        es.push(5, 4);
        es.push(0, 5);
    }
    // cluster 0 prefix histogram: simple, nsym=4, tree_select=0
    // (alphabet_bits(33) = 6).
    es.push(1, 2); // kind = simple
    es.push(3, 2); // nsym - 1 = 3
    es.push(0, 6);
    es.push(2, 6);
    es.push(8, 6);
    es.push(32, 6);
    es.push(0, 1); // tree_select = 0 → lengths {2, 2, 2, 2}
                   // cluster 1 prefix histogram: simple, nsym=3 → lengths {1, 2, 2}.
    es.push(1, 2);
    es.push(2, 2); // nsym - 1 = 2
    es.push(0, 6);
    es.push(8, 6);
    es.push(32, 6);
    let es_prelude = es.bytes.clone();

    // Token codes by trial decode against the prelude itself.
    let c0 = |sym: u32| code_for_symbol(&es_prelude, 6, 0, sym); // any ctx in cluster 0
    let c5 = |sym: u32| code_for_symbol(&es_prelude, 6, 5, sym);
    let (p0_0, l0_0) = c0(0);
    let (p0_2, l0_2) = c0(2);
    let (p0_8, l0_8) = c0(8);
    let (p0_32, l0_32) = c0(32);
    let (p5_0, l5_0) = c5(0);
    let (p5_8, l5_8) = c5(8);
    let (p5_32, l5_32) = c5(32);

    // Splices the entropy prelude into the section bit stream.
    {
        let mut re = oxideav_jpegxl::bitreader::BitReader::new(&es_prelude);
        let es_check = oxideav_jpegxl::modular_fdis::EntropyStream::read(&mut re, 6)
            .expect("spline prelude parses");
        assert!(es_check.use_prefix_code);
        assert_eq!(es_check.entropies.len(), 2);
        let prelude_bits = re.bits_read();
        let mut rb = oxideav_jpegxl::bitreader::BitReader::new(&es_prelude);
        for _ in 0..prelude_bits {
            w.push(rb.read_bits(1).unwrap(), 1);
        }
    }

    // Token sequence per §C.4.6 with the round-441 Listing C.3 order
    // erratum (`quant_adjust` follows the start-point loop):
    //   ctx2 num_splines-1 = 0
    //   ctx1 sp_x = 8, sp_y = 32
    //   ctx0 quant_adjust  = 0
    //   ctx3 num_control_points-1 = 2
    //   ctx4 interleaved second-order deltas: (32→+16, 0), (0, 0)
    //   ctx5 4×32 coefficients: X zeros; Y DC=8 (→ +4); B zeros;
    //        σ DC=32 (→ +16)
    w.push(p0_0, l0_0); // num_splines - 1
    w.push(p0_8, l0_8); // sp_x = 8
    w.push(p0_32, l0_32); // sp_y = 32
    w.push(p0_0, l0_0); // quant_adjust (after the start loop — round-441 erratum)
    w.push(p0_2, l0_2); // num_control_points - 1 = 2
    w.push(p0_32, l0_32); // dx1 raw 32 → UnpackSigned +16
    w.push(p0_0, l0_0); // dy1
    w.push(p0_0, l0_0); // dx2
    w.push(p0_0, l0_0); // dy2
                        // X channel: 32 zeros.
    for _ in 0..32 {
        w.push(p5_0, l5_0);
    }
    // Y channel: DC token 8 (UnpackSigned → +4) then 31 zeros.
    w.push(p5_8, l5_8);
    for _ in 0..31 {
        w.push(p5_0, l5_0);
    }
    // B channel: 32 zeros.
    for _ in 0..32 {
        w.push(p5_0, l5_0);
    }
    // σ channel: DC token 32 (UnpackSigned → +16) then 31 zeros.
    w.push(p5_32, l5_32);
    for _ in 0..31 {
        w.push(p5_0, l5_0);
    }

    // LfChannelDequantization (Table C.11): all_default = 1.
    w.push(1, 1);

    // GlobalModular (C.4.8 / Table C.22): all-zero 64×64×3 image.
    w.push(0, 1); // global use_global_tree = 0
    w.push(0, 1); // inner use_global_tree = 0
    w.push(1, 1); // WPHeader: default_wp = 1
    w.push(0, 2); // nb_transforms U32 selector 0 → 0
                  // MA tree stream (D.4.2): 6 distributions, single cluster, prefix,
                  // 1-symbol alphabet → every token is 0 at zero bits. The tree is a
                  // single leaf: property+1=0 → leaf(predictor 0, offset 0, mul_log 0,
                  // mul_bits 0).
    w.push(0, 1); // lz77 = 0
    w.push(1, 1); // clustering: is_simple
    w.push(0, 2); // nbits = 0 → all zero cluster map
    w.push(1, 1); // use_prefix_code
    w.push(15, 4); // config split_exponent = 15
    w.push(0, 1); // alphabet count bit = 0 → count 1 (0-bit tokens)
                  // Symbol stream for num_ctx = 1: same degenerate shape (no
                  // clustering read when num_dist == 1).
    w.push(0, 1); // lz77 = 0
    w.push(1, 1); // use_prefix_code
    w.push(15, 4); // config split_exponent = 15
    w.push(0, 1); // count bit = 0 → 1-symbol alphabet
                  // Channel samples: 3 × 64×64 tokens, all from the 1-symbol
                  // alphabet — zero bits on the wire.

    w.pad_to_byte();
    let section_len = w.bytes.len() - section_start;

    // Patch the TOC entry (selector 0 → u(10) length in bytes).
    assert!(section_len < 1024, "section fits the u(10) TOC selector");
    let mut fixed = BitWriter::new();
    fixed.push(0, 2);
    fixed.push(section_len as u32, 10);
    fixed.pad_to_byte();
    w.bytes[toc_entry_pos] = fixed.bytes[0];
    w.bytes[toc_entry_pos + 1] = fixed.bytes[1];

    // Raw-codestream signature.
    let mut out = vec![0xFF, 0x0A];
    out.extend_from_slice(&w.bytes);
    out
}

/// The builder must reproduce the committed fixture byte-for-byte —
/// the committed bytes are what the reference decoder (black-box)
/// verified and decoded into the expected PNG.
#[test]
fn round441_spline_synth_builder_matches_committed_fixture() {
    let built = build_spline_codestream();
    let committed = include_bytes!("fixtures/spline_synth_64x64.jxl");
    assert_eq!(
        built,
        committed.to_vec(),
        "hand-assembled stream drifted from the committed fixture"
    );
}

/// End-to-end §C.4.6 parse + §K.3 render on the registered path,
/// against the reference decoder's pixels for the same bytes.
#[test]
fn round441_spline_synth_renders_reference_exact_band() {
    let jxl = include_bytes!("fixtures/spline_synth_64x64.jxl");
    let (w, h, want) = png_rgb(include_bytes!("fixtures/spline_synth_64x64_expected.png"));
    let frames = decode_all_frames(jxl, None).expect("spline stream decodes");
    assert_eq!(frames.len(), 1);
    let stats = compare_rgb(&frames[0], w, h, &want);
    // The spline brush is continuous math (erf + arc-length resampling
    // + DCT32 evaluation); small float ordering differences against
    // the reference land within a couple of 8-bit codes.
    for (c, &(mad, max)) in stats.iter().enumerate() {
        assert!(
            mad < 0.30 && max <= 3,
            "channel {c}: MAD {mad}, max {max} — spline render out of band"
        );
    }
    // And the spline must actually be there: the canvas is not flat.
    let y_plane = &frames[0].planes[1];
    let on_spline = y_plane.data[32 * w + 24];
    let off_spline = y_plane.data[8 * w + 24];
    assert!(
        on_spline > off_spline.saturating_add(20),
        "spline streak missing: on {on_spline} vs off {off_spline}"
    );
}
