//! Round-4 pixel-correctness tests against committed `expected.png`
//! reference images for the small lossless docs fixtures.
//!
//! Each test:
//! 1. Decodes the committed `<fixture>.jxl` via `decode_one_frame`.
//! 2. Decodes the committed `<fixture>_expected.png` via the `png` crate.
//! 3. Asserts pixel-for-pixel equality on every plane.
//!
//! `png` is added as a dev-dep — a tiny single-purpose PNG decoder with
//! no codec-semantics overlap with JPEG XL. Round 3 used first-16-pixels
//! plus histogram statistics; round 4 graduates to byte-exact match.

use oxideav_jpegxl::decode_one_frame;
use png::ColorType;
use std::io::Cursor;

const PIXEL_1X1_JXL: &[u8] = include_bytes!("fixtures/pixel_1x1.jxl");

const GRAY_64X64_JXL: &[u8] = include_bytes!("fixtures/gray_64x64_lossless.jxl");
const GRAY_64X64_PNG: &[u8] = include_bytes!("fixtures/gray_64x64_expected.png");

const GRADIENT_JXL: &[u8] = include_bytes!("fixtures/gradient_64x64_lossless.jxl");
const GRADIENT_PNG: &[u8] = include_bytes!("fixtures/gradient_64x64_lossless_expected.png");

const PALETTE_JXL: &[u8] = include_bytes!("fixtures/palette_32x32.jxl");
const PALETTE_PNG: &[u8] = include_bytes!("fixtures/palette_32x32_expected.png");

/// Decode a PNG and return `(width, height, planes_in_R-G-B[-A]_or_grey_order)`.
/// Strips alpha if present and panics on unsupported bit depths (only 8-bit
/// here, matching our Modular decoder's round-3 envelope).
fn png_to_planes(bytes: &[u8]) -> (u32, u32, Vec<Vec<u8>>) {
    let dec = png::Decoder::new(Cursor::new(bytes));
    let mut reader = dec.read_info().expect("read png info");
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap_or(0)];
    let info = reader.next_frame(&mut buf).expect("png next_frame");
    let (w, h) = (info.width, info.height);
    let bytes = &buf[..info.buffer_size()];
    assert_eq!(info.bit_depth, png::BitDepth::Eight, "expect 8-bit PNG");
    let planes: Vec<Vec<u8>> = match info.color_type {
        ColorType::Grayscale => vec![bytes.to_vec()],
        ColorType::Rgb => {
            let n = (w * h) as usize;
            let (mut r, mut g, mut b) = (
                Vec::with_capacity(n),
                Vec::with_capacity(n),
                Vec::with_capacity(n),
            );
            for px in bytes.chunks_exact(3) {
                r.push(px[0]);
                g.push(px[1]);
                b.push(px[2]);
            }
            vec![r, g, b]
        }
        ColorType::GrayscaleAlpha => {
            let n = (w * h) as usize;
            let mut grey = Vec::with_capacity(n);
            for px in bytes.chunks_exact(2) {
                grey.push(px[0]);
            }
            vec![grey]
        }
        ColorType::Rgba => {
            let n = (w * h) as usize;
            let (mut r, mut g, mut b) = (
                Vec::with_capacity(n),
                Vec::with_capacity(n),
                Vec::with_capacity(n),
            );
            for px in bytes.chunks_exact(4) {
                r.push(px[0]);
                g.push(px[1]);
                b.push(px[2]);
            }
            vec![r, g, b]
        }
        other => panic!("unsupported PNG color type {:?}", other),
    };
    (w, h, planes)
}

fn assert_planes_equal(label: &str, ours: &[Vec<u8>], theirs: &[Vec<u8>], w: u32, h: u32) {
    assert_eq!(
        ours.len(),
        theirs.len(),
        "{label}: plane count mismatch (ours={} theirs={})",
        ours.len(),
        theirs.len(),
    );
    for (idx, (a, b)) in ours.iter().zip(theirs.iter()).enumerate() {
        assert_eq!(
            a.len(),
            (w * h) as usize,
            "{label}: plane[{idx}] len mismatch ({} vs {})",
            a.len(),
            w * h
        );
        assert_eq!(
            b.len(),
            (w * h) as usize,
            "{label}: ref plane[{idx}] len mismatch",
        );
        if a != b {
            // Find first divergence for diagnostic.
            for (px_idx, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
                if av != bv {
                    let x = (px_idx as u32) % w;
                    let y = (px_idx as u32) / w;
                    panic!("{label}: plane[{idx}] mismatch at ({x}, {y}): ours={av} theirs={bv}",);
                }
            }
        }
    }
}

#[test]
fn pixel_1x1_decodes_to_red_rgb() {
    let vf = decode_one_frame(PIXEL_1X1_JXL, None).expect("pixel-1x1 must decode");
    assert_eq!(vf.planes.len(), 3, "pixel-1x1: expected 3 RGB planes");
    assert_eq!(vf.planes[0].data, vec![255u8], "R plane");
    assert_eq!(vf.planes[1].data, vec![0u8], "G plane");
    assert_eq!(vf.planes[2].data, vec![0u8], "B plane");
}

#[test]
fn gray_64x64_pixel_correct_vs_expected_png() {
    let vf = decode_one_frame(GRAY_64X64_JXL, None).expect("gray-64x64 must decode");
    let (w, h, ref_planes) = png_to_planes(GRAY_64X64_PNG);
    assert_eq!((w, h), (64, 64));
    let ours: Vec<Vec<u8>> = vf.planes.iter().map(|p| p.data.clone()).collect();
    assert_planes_equal("gray-64x64", &ours, &ref_planes, w, h);
}

#[test]
fn gradient_64x64_lossless_pixel_correct_vs_expected_png() {
    let vf = decode_one_frame(GRADIENT_JXL, None).expect("gradient-64x64 must decode");
    let (w, h, ref_planes) = png_to_planes(GRADIENT_PNG);
    assert_eq!((w, h), (64, 64));
    let ours: Vec<Vec<u8>> = vf.planes.iter().map(|p| p.data.clone()).collect();
    assert_planes_equal("gradient-64x64", &ours, &ref_planes, w, h);
}

#[test]
fn palette_32x32_pixel_correct_vs_expected_png() {
    let vf = decode_one_frame(PALETTE_JXL, None).expect("palette-32x32 must decode");
    let (w, h, ref_planes) = png_to_planes(PALETTE_PNG);
    assert_eq!((w, h), (32, 32));
    let ours: Vec<Vec<u8>> = vf.planes.iter().map(|p| p.data.clone()).collect();
    assert_planes_equal("palette-32x32", &ours, &ref_planes, w, h);
}

/// Diagnostic: per-channel/per-sample bit-position trace for palette.
/// We replicate decode_channels directly so we can print bit positions
/// at every milestone and identify where we over-read.
#[test]
fn palette_invasive_pixel_decode() {
    use oxideav_jpegxl::bitreader::BitReader;
    use oxideav_jpegxl::container;
    use oxideav_jpegxl::frame_header::{FrameDecodeParams, FrameHeader};
    use oxideav_jpegxl::lf_global::LfChannelDequantization;
    use oxideav_jpegxl::metadata_fdis::{ImageMetadataFdis, SizeHeaderFdis};
    use oxideav_jpegxl::modular_fdis::{
        decode_uint_in_with_dist_pub, evaluate_tree, get_properties, MaTreeFdis, TransformInfo,
        WpHeader,
    };
    use oxideav_jpegxl::toc::Toc;

    let sig = container::detect(PALETTE_JXL).unwrap();
    let codestream: Vec<u8> = match sig {
        container::Signature::RawCodestream => PALETTE_JXL[2..].to_vec(),
        container::Signature::Isobmff => {
            container::extract_codestream(PALETTE_JXL).unwrap().to_vec()
        }
    };
    eprintln!("palette codestream {} bits", codestream.len() * 8);
    let mut br = BitReader::new(&codestream);
    let size = SizeHeaderFdis::read(&mut br).unwrap();
    let metadata = ImageMetadataFdis::read(&mut br).unwrap();
    br.pu0().unwrap();
    let fh_params = FrameDecodeParams {
        xyb_encoded: metadata.xyb_encoded,
        num_extra_channels: metadata.num_extra_channels,
        have_animation: metadata.have_animation,
        have_animation_timecodes: metadata
            .animation
            .map(|a| a.have_timecodes)
            .unwrap_or(false),
        image_width: size.width,
        image_height: size.height,
    };
    let fh = FrameHeader::read(&mut br, &fh_params).unwrap();
    let _toc = Toc::read(&mut br, &fh).unwrap();
    let dc_global_begin = br.bits_read();
    let _lf = LfChannelDequantization::read(&mut br).unwrap();
    let _g = br.read_bool().unwrap();
    let mut tree = MaTreeFdis::read(&mut br).unwrap();
    eprintln!("MaTree end: rel {}", br.bits_read() - dc_global_begin);
    let _inner = br.read_bool().unwrap();
    let _wp = WpHeader::read(&mut br).unwrap();
    let nb_transforms = br
        .read_u32([
            oxideav_jpegxl::bitreader::U32Dist::Val(0),
            oxideav_jpegxl::bitreader::U32Dist::Val(1),
            oxideav_jpegxl::bitreader::U32Dist::BitsOffset(4, 2),
            oxideav_jpegxl::bitreader::U32Dist::BitsOffset(8, 18),
        ])
        .unwrap();
    for _ in 0..nb_transforms {
        TransformInfo::read(&mut br).unwrap();
    }
    eprintln!("after header: rel {}", br.bits_read() - dc_global_begin);
    tree.entropy.read_ans_state_init(&mut br).unwrap();
    eprintln!(
        "after ans_state_init: rel {}",
        br.bits_read() - dc_global_begin
    );

    use oxideav_jpegxl::modular_fdis::{ChannelDesc, ModularImage};
    let descs = vec![
        ChannelDesc {
            width: 8,
            height: 3,
            hshift: -1,
            vshift: -1,
        },
        ChannelDesc {
            width: 32,
            height: 32,
            hshift: 0,
            vshift: 0,
        },
    ];
    let dist_multiplier = 32u32;
    let mut img = ModularImage {
        channels: descs
            .iter()
            .map(|d| vec![0i32; (d.width * d.height) as usize])
            .collect(),
        descs: descs.clone(),
    };

    let mut total_tokens = 0usize;
    'outer: for (i, desc) in descs.iter().enumerate() {
        for y in 0..desc.height {
            for x in 0..desc.width {
                let pre = br.bits_read();
                let props = get_properties(&img, i, x as i32, y as i32, 0, 0);
                let leaf = match evaluate_tree(&tree.nodes, &props) {
                    Ok(l) => *l,
                    Err(e) => {
                        eprintln!("eval_tree fail: ch={i} x={x} y={y}: {e}");
                        break 'outer;
                    }
                };
                let token = match decode_uint_in_with_dist_pub(
                    &mut tree.hybrid,
                    &mut tree.entropy,
                    &mut br,
                    leaf.ctx,
                    dist_multiplier,
                ) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!(
                            "EOF at ch={i} x={x} y={y} (token #{total_tokens}) ctx={} pre_bits={pre} (rel {}) cur_bits={}: {e}",
                            leaf.ctx,
                            pre - dc_global_begin,
                            br.bits_read()
                        );
                        break 'outer;
                    }
                };
                let diff = oxideav_jpegxl::bitreader::unpack_signed(token);
                let val = diff as i64 * leaf.multiplier as i64 + leaf.offset as i64;
                let v = val as i32; // simplistic since predictor=5 gradient on first samples is 0
                img.channels[i][(y * desc.width + x) as usize] = v;
                total_tokens += 1;
                if total_tokens <= 4 || total_tokens % 64 == 0 {
                    eprintln!(
                        "  tok #{total_tokens} ch={i} ({x},{y}) ctx={} token={token} (pre {} → {}, +{} bits)",
                        leaf.ctx,
                        pre,
                        br.bits_read(),
                        br.bits_read() - pre
                    );
                }
            }
        }
    }
    eprintln!(
        "decoded {} tokens, ended at bit {} (rel {})",
        total_tokens,
        br.bits_read(),
        br.bits_read() - dc_global_begin
    );
}

/// Diagnostic: walk decode_channels manually for palette-32x32 to find
/// where the EOF happens.
#[test]
fn palette_diagnostic_walk() {
    use oxideav_jpegxl::bitreader::BitReader;
    use oxideav_jpegxl::container;
    use oxideav_jpegxl::frame_header::{FrameDecodeParams, FrameHeader};
    use oxideav_jpegxl::lf_global::LfGlobal;
    use oxideav_jpegxl::metadata_fdis::{ImageMetadataFdis, SizeHeaderFdis};
    use oxideav_jpegxl::toc::Toc;

    let sig = container::detect(PALETTE_JXL).unwrap();
    let codestream: Vec<u8> = match sig {
        container::Signature::RawCodestream => PALETTE_JXL[2..].to_vec(),
        container::Signature::Isobmff => {
            container::extract_codestream(PALETTE_JXL).unwrap().to_vec()
        }
    };
    eprintln!(
        "palette codestream {} bytes ({} bits)",
        codestream.len(),
        codestream.len() * 8
    );
    let mut br = BitReader::new(&codestream);
    let size = SizeHeaderFdis::read(&mut br).expect("size");
    let metadata = ImageMetadataFdis::read(&mut br).expect("metadata");
    br.pu0().expect("align");
    let fh_params = FrameDecodeParams {
        xyb_encoded: metadata.xyb_encoded,
        num_extra_channels: metadata.num_extra_channels,
        have_animation: metadata.have_animation,
        have_animation_timecodes: metadata
            .animation
            .map(|a| a.have_timecodes)
            .unwrap_or(false),
        image_width: size.width,
        image_height: size.height,
    };
    let fh = FrameHeader::read(&mut br, &fh_params).expect("frame_header");
    let _toc = Toc::read(&mut br, &fh).expect("toc");
    let dc_global_begin = br.bits_read();
    eprintln!("DC_GLOBAL_BEGIN at bit {}", dc_global_begin);
    match LfGlobal::read(&mut br, &fh, &metadata) {
        Ok(lfg) => {
            eprintln!(
                "LfGlobal OK at bit {} (rel {}): {} channels {} transforms",
                br.bits_read(),
                br.bits_read() - dc_global_begin,
                lfg.global_modular.image.channels.len(),
                lfg.global_modular.nb_transforms
            );
        }
        Err(e) => {
            eprintln!(
                "LfGlobal FAIL at bit {} (rel {}): {e}",
                br.bits_read(),
                br.bits_read() - dc_global_begin
            );
        }
    }
}
