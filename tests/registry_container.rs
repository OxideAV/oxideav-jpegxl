//! The `jpegxl` framework container (demuxer only): probe → demuxer →
//! decoder yields the Layer 1 frames byte for byte on every committed
//! layout, animations come out one packet per frame with the animation
//! header's timing, and there is deliberately no muxer.

#![cfg(feature = "registry")]

use std::io::Cursor;

use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, NullCodecResolver, Packet, PixelFormat, ProbeData,
    ReadSeek, RuntimeContext, StreamInfo, TimeBase, VideoFrame, WriteSeek,
};
use oxideav_jpegxl::demux::{open_demuxer, probe as probe_fn, CONTAINER_NAME};
use oxideav_jpegxl::registry::{OPTION_PACING, PACING_PACKET};
use oxideav_jpegxl::{decode_all, info, make_decoder, register, Signature, CODEC_ID_STR};

const ANIMATION: &[u8] = include_bytes!("fixtures/animation_3frame.jxl");
const ALPHA_64: &[u8] = include_bytes!("fixtures/alpha_64x64.jxl");
const BIT_DEPTH_16: &[u8] = include_bytes!("fixtures/bit_depth_16.jxl");
const GRAY_64: &[u8] = include_bytes!("fixtures/gray_64x64_lossless.jxl");
const GRADIENT_64: &[u8] = include_bytes!("fixtures/gradient_64x64_lossless.jxl");
const ICC: &[u8] = include_bytes!("fixtures/r451_icc.jxl");
const PALETTE_32: &[u8] = include_bytes!("fixtures/palette_32x32.jxl");
const JPEG_TRANSCODE: &[u8] = include_bytes!("fixtures/jpeg_transcode.jxl");
const SUNSET: &[u8] = include_bytes!("fixtures/conformance_sunset_logo.jxl");
const GRAY_ICC: &[u8] = include_bytes!("fixtures/conformance_grayscale.jxl");

const FIXTURES: &[(&str, &[u8])] = &[
    ("animation_3frame.jxl", ANIMATION),
    ("alpha_64x64.jxl", ALPHA_64),
    ("bit_depth_16.jxl", BIT_DEPTH_16),
    ("gray_64x64_lossless.jxl", GRAY_64),
    ("gradient_64x64_lossless.jxl", GRADIENT_64),
    ("r451_icc.jxl", ICC),
    ("palette_32x32.jxl", PALETTE_32),
    ("conformance_sunset_logo.jxl", SUNSET),
    ("conformance_grayscale.jxl", GRAY_ICC),
];

fn ctx() -> RuntimeContext {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    ctx
}

fn reader(bytes: &[u8]) -> Box<dyn ReadSeek> {
    Box::new(Cursor::new(bytes.to_vec()))
}

fn probe(ctx: &RuntimeContext, bytes: &[u8], ext: Option<&str>) -> oxideav_core::Result<String> {
    let mut cur = Cursor::new(bytes.to_vec());
    ctx.containers.probe_input(&mut cur, ext)
}

/// Pump every packet of `bytes` through the registry the way a gateway
/// does (send, drain until `NeedMore`, flush at `Eof`); returns the
/// stream, the packets and the frames in order.
fn demux_decode(ctx: &RuntimeContext, bytes: &[u8]) -> (StreamInfo, Vec<Packet>, Vec<VideoFrame>) {
    let name = probe(ctx, bytes, None).expect("probe");
    assert_eq!(name, CONTAINER_NAME);
    let mut demux = ctx
        .containers
        .open_demuxer(&name, reader(bytes), &ctx.codecs)
        .expect("open_demuxer");
    assert_eq!(demux.format_name(), CONTAINER_NAME);
    let stream = demux.streams()[0].clone();
    let mut dec = ctx
        .codecs
        .first_decoder(&stream.params)
        .expect("first_decoder");
    let mut packets = Vec::new();
    let mut frames = Vec::new();
    loop {
        match demux.next_packet() {
            Ok(pkt) => {
                dec.send_packet(&pkt).expect("send_packet");
                let mut got = 0;
                loop {
                    match dec.receive_frame() {
                        Ok(Frame::Video(vf)) => {
                            frames.push(vf);
                            got += 1;
                        }
                        Ok(_) => panic!("video frame expected"),
                        Err(Error::NeedMore) => break,
                        Err(e) => panic!("receive_frame: {e}"),
                    }
                }
                assert_eq!(got, 1, "paced decoding releases one frame per packet");
                packets.push(pkt);
            }
            Err(Error::Eof) => break,
            Err(e) => panic!("next_packet: {e}"),
        }
    }
    dec.flush().unwrap();
    assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
    (stream, packets, frames)
}

#[test]
fn probe_names_the_container_for_both_signatures() {
    let ctx = ctx();
    let mut raw = 0;
    let mut boxed = 0;
    for (name, bytes) in FIXTURES {
        match info(bytes).unwrap().signature {
            Signature::RawCodestream => raw += 1,
            Signature::Isobmff => boxed += 1,
        }
        assert_eq!(probe(&ctx, bytes, None).unwrap(), CONTAINER_NAME, "{name}");
        assert_eq!(
            probe(&ctx, bytes, Some("jxl")).unwrap(),
            CONTAINER_NAME,
            "{name} + ext"
        );
        assert_eq!(
            probe(&ctx, bytes, Some("png")).unwrap(),
            CONTAINER_NAME,
            "{name}: magic beats a misleading extension"
        );
    }
    assert!(raw > 0 && boxed > 0, "both signatures covered");
    assert_eq!(
        probe(&ctx, b"not a picture at all", Some("jxl")).unwrap(),
        CONTAINER_NAME
    );
    assert!(ctx
        .containers
        .open_demuxer(CONTAINER_NAME, reader(b"not a picture at all"), &ctx.codecs)
        .is_err());
    for foreign in [
        &b"farbfeld\0\0\0\x01\0\0\0\x01\0\0\0\0\0\0\0\0"[..],
        &b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR"[..],
        &b"\xff\xd8\xff\xe0\0\x10JFIF"[..],
        &b"\0\0\0\x0cjP  \r\n\x87\n"[..], // JP2, not JXL
        &[][..],
    ] {
        assert!(
            matches!(probe(&ctx, foreign, None), Err(Error::FormatNotFound(_))),
            "{foreign:?}"
        );
    }
    for n in 0..16 {
        let _ = probe_fn(&ProbeData {
            buf: &BIT_DEPTH_16[..n],
            ext: None,
        });
    }
}

#[test]
fn demuxer_declares_the_layer1_stream() {
    let ctx = ctx();
    let mut formats = Vec::new();
    for (name, bytes) in FIXTURES {
        let i = info(bytes).unwrap();
        formats.push(i.format);
        let (stream, packets, _) = demux_decode(&ctx, bytes);
        let p = &stream.params;
        assert_eq!(p.codec_id, CodecId::new(CODEC_ID_STR), "{name}");
        assert_eq!(p.width, Some(i.width), "{name} width");
        assert_eq!(p.height, Some(i.height), "{name} height");
        assert_eq!(
            p.pixel_format,
            Some(PixelFormat::from(i.format)),
            "{name} format"
        );
        assert_eq!(
            !p.color_signal.is_unspecified(),
            i.color.is_specified(),
            "{name}: colour signal iff the ColourEncoding is enumerated"
        );
        if i.color.is_specified() {
            assert_eq!(p.color_signal.primaries.0, i.color.primaries, "{name}");
            assert_eq!(p.color_signal.transfer.0, i.color.transfer, "{name}");
        }
        assert_eq!(p.options.get(OPTION_PACING), Some(PACING_PACKET), "{name}");
        assert_eq!(
            packets.len(),
            i.frames as usize,
            "{name}: one packet per frame"
        );
        assert_eq!(
            packets[0].data, *bytes,
            "{name}: the first packet is the file"
        );
        assert!(packets[0].flags.keyframe);
        if i.animation.is_none() {
            assert_eq!(stream.time_base, TimeBase::new(1, 1), "{name}");
            assert_eq!(packets[0].pts, Some(0));
            assert_eq!(stream.duration, None);
        }
    }
    for want in [
        oxideav_jpegxl::PixelFormat::Gray8,
        oxideav_jpegxl::PixelFormat::Rgb24,
        oxideav_jpegxl::PixelFormat::Rgba,
        oxideav_jpegxl::PixelFormat::Rgb48Le,
    ] {
        assert!(formats.contains(&want), "{want:?} not covered");
    }
}

#[test]
fn registry_frames_are_byte_identical_to_layer1() {
    let ctx = ctx();
    for (name, bytes) in FIXTURES {
        let layer1 = decode_all(bytes).unwrap();
        let (stream, packets, frames) = demux_decode(&ctx, bytes);
        assert_eq!(frames.len(), layer1.len(), "{name} frame count");
        for (k, (vf, f)) in frames.iter().zip(&layer1).enumerate() {
            let planes = vf.image_planes();
            assert_eq!(planes.len(), f.image.planes.len(), "{name} frame {k}");
            for (a, b) in planes.iter().zip(&f.image.planes) {
                assert_eq!(a.stride, b.stride, "{name} frame {k} stride");
                assert_eq!(a.data, b.data, "{name} frame {k} samples");
            }
            assert_eq!(
                vf.pts, packets[k].pts,
                "{name} frame {k} takes its packet's pts"
            );
            assert_eq!(
                vf.color_signal().is_some(),
                !stream.params.color_signal.is_unspecified(),
                "{name}"
            );
        }
    }
}

#[test]
fn animation_is_one_packet_per_frame_in_the_animation_tick() {
    let ctx = ctx();
    let i = info(ANIMATION).unwrap();
    let anim = i.animation.expect("animated fixture");
    let layer1 = decode_all(ANIMATION).unwrap();
    assert_eq!(layer1.len(), 3);
    let (stream, packets, frames) = demux_decode(&ctx, ANIMATION);
    assert_eq!(frames.len(), 3);
    assert_eq!(packets.len(), 3);
    let (num, den) = anim.ticks_per_second;
    assert_eq!(
        stream.time_base,
        TimeBase::new(i64::from(den), i64::from(num))
    );
    let mut pts = 0i64;
    for (k, (pkt, f)) in packets.iter().zip(&layer1).enumerate() {
        assert_eq!(pkt.time_base, stream.time_base);
        assert_eq!(pkt.pts, Some(pts), "frame {k} pts is the running sum");
        assert_eq!(pkt.duration, Some(i64::from(f.ticks)), "frame {k} duration");
        assert_eq!(pkt.flags.keyframe, k == 0);
        assert_eq!(
            pkt.data.is_empty(),
            k > 0,
            "only the first packet carries bytes"
        );
        pts += i64::from(f.ticks);
    }
    assert_eq!(stream.duration, Some(pts));
    // The frame's display delay through the time base equals Layer 1's.
    for (pkt, f) in packets.iter().zip(&layer1) {
        let secs = pkt.duration.unwrap() as f64 * f64::from(den) / f64::from(num);
        assert!((secs - f.delay.unwrap().as_secs_f64()).abs() < 1e-9);
    }
    let name = probe(&ctx, ANIMATION, None).unwrap();
    let demux = ctx
        .containers
        .open_demuxer(&name, reader(ANIMATION), &ctx.codecs)
        .unwrap();
    assert!(demux
        .metadata()
        .contains(&("loop_count".to_owned(), anim.loops.to_string())));
}

#[test]
fn demuxer_flags_embedded_metadata() {
    let i = info(ICC).unwrap();
    assert!(i.has_icc, "fixture premise");
    let demux = open_demuxer(reader(ICC), &NullCodecResolver).unwrap();
    assert!(demux
        .metadata()
        .contains(&("icc".to_owned(), "present".to_owned())));
    // An ICC-described image has no enumerated colour: the stream carries
    // what the decoder frame carries — the (always full) range only.
    let sig = demux.streams()[0].params.color_signal;
    assert_eq!(sig.range, oxideav_core::ColorRange::Full);
    assert_eq!(sig.primaries.0, 2);
    assert_eq!(sig.transfer.0, 2);
    let plain = open_demuxer(reader(GRAY_64), &NullCodecResolver).unwrap();
    assert!(plain.metadata().is_empty());
}

#[test]
fn jpeg_transcode_decodes_planar_on_both_paths() {
    // A losslessly recompressed 4:2:0 JPEG decodes to the JPEG's own
    // planar YCbCr on Layer 1; the registry publishes the same label on
    // the stream and the decoder frame carries the same three planes
    // byte for byte.
    let ctx = ctx();
    let frames = decode_all(JPEG_TRANSCODE).unwrap();
    assert_eq!(frames.len(), 1);
    let img = &frames[0].image;
    assert_eq!(img.format, oxideav_jpegxl::PixelFormat::YuvJ420P);
    assert_eq!(img.planes.len(), 3);
    assert_eq!(img, &oxideav_jpegxl::decode(JPEG_TRANSCODE).unwrap());
    assert_eq!(
        oxideav_jpegxl::info(JPEG_TRANSCODE).unwrap().format,
        oxideav_jpegxl::PixelFormat::YuvJ420P
    );
    let name = probe(&ctx, JPEG_TRANSCODE, None).unwrap();
    let mut demux = ctx
        .containers
        .open_demuxer(&name, reader(JPEG_TRANSCODE), &ctx.codecs)
        .unwrap();
    let stream = demux.streams()[0].clone();
    assert_eq!(stream.params.pixel_format, Some(PixelFormat::YuvJ420P));
    let pkt = demux.next_packet().unwrap();
    let mut dec = ctx.codecs.first_decoder(&stream.params).unwrap();
    dec.send_packet(&pkt).unwrap();
    let Frame::Video(vf) = dec.receive_frame().unwrap() else {
        panic!("video frame expected");
    };
    let planes = vf.image_planes();
    assert_eq!(planes.len(), 3);
    for (c, p) in planes.iter().enumerate() {
        assert_eq!(p.stride, img.planes[c].stride, "plane {c} stride");
        assert_eq!(p.data, img.planes[c].data, "plane {c} bytes");
    }
    let sig = vf.color_signal().expect("JPEG = sYCC colour signal");
    assert_eq!(sig.range, oxideav_core::ColorRange::Full);
    assert_eq!(sig.matrix.0, 5);
    // Through the generic bridge the planar frame rebuilds the image.
    let back = oxideav_jpegxl::JxlImage::from_video_frame(&vf, &stream.params).unwrap();
    assert_eq!(back.planes, img.planes);
}

#[test]
fn decoder_pacing_option_is_additive() {
    // Default: every frame of the file comes out of its packet.
    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).unwrap();
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 1000), ANIMATION.to_vec()))
        .unwrap();
    let mut n = 0;
    while let Ok(Frame::Video(_)) = dec.receive_frame() {
        n += 1;
    }
    assert_eq!(n, 3);
    // `pacing = packet`: one per packet, the rest at flush.
    let mut paced = params.clone();
    paced.options.insert(OPTION_PACING, PACING_PACKET);
    let mut dec = make_decoder(&paced).unwrap();
    // A pacing packet with nothing decoded is a no-op, not an error.
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 1000), Vec::new()))
        .unwrap();
    assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));
    let mut file = Packet::new(0, TimeBase::new(1, 1000), ANIMATION.to_vec());
    file.pts = Some(10);
    dec.send_packet(&file).unwrap();
    let Frame::Video(f0) = dec.receive_frame().unwrap() else {
        panic!()
    };
    assert_eq!(f0.pts, Some(10));
    assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));
    // A second file while frames are held is refused, as before.
    assert!(dec.send_packet(&file).is_err());
    let mut pace = Packet::new(0, TimeBase::new(1, 1000), Vec::new());
    pace.pts = Some(20);
    dec.send_packet(&pace).unwrap();
    let Frame::Video(f1) = dec.receive_frame().unwrap() else {
        panic!()
    };
    assert_eq!(f1.pts, Some(20), "the pacing packet lends its pts");
    assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));
    dec.flush().unwrap();
    let Frame::Video(f2) = dec.receive_frame().unwrap() else {
        panic!()
    };
    assert_eq!(
        f2.pts, None,
        "released by the flush, no packet to borrow from"
    );
    assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
    // Unknown values are rejected at the factory.
    let mut bad = params.clone();
    bad.options.insert(OPTION_PACING, "sometimes");
    assert!(make_decoder(&bad).is_err());
}

#[test]
fn register_installs_codec_and_demuxer_but_no_muxer() {
    let ctx = ctx();
    let id = CodecId::new(CODEC_ID_STR);
    assert!(ctx.codecs.has_decoder(&id));
    assert!(ctx.containers.demuxer_names().any(|n| n == CONTAINER_NAME));
    assert!(
        ctx.containers.muxer_names().all(|n| n != CONTAINER_NAME),
        "decoder-only crate registers no muxer"
    );
    let out: Box<dyn WriteSeek> = Box::new(Cursor::new(Vec::new()));
    assert!(matches!(
        ctx.containers.open_muxer(CONTAINER_NAME, out, &[]),
        Err(Error::FormatNotFound(_))
    ));
    assert_eq!(
        ctx.containers.container_for_extension("JXL"),
        Some(CONTAINER_NAME)
    );
    let mut ctx2 = RuntimeContext::new();
    oxideav_jpegxl::__oxideav_entry(&mut ctx2);
    assert!(ctx2.codecs.has_decoder(&id));
    assert_eq!(probe(&ctx2, GRAY_64, None).unwrap(), CONTAINER_NAME);
    assert_eq!(probe(&ctx2, BIT_DEPTH_16, None).unwrap(), CONTAINER_NAME);
}

#[test]
fn hostile_input_never_panics() {
    let ctx = ctx();
    for (_, bytes) in FIXTURES {
        for cut in [0usize, 1, 2, 3, 4, 11, 12, 13, 20, 40, 60, bytes.len() / 2] {
            let cut = cut.min(bytes.len());
            if let Ok(mut d) = open_demuxer(reader(&bytes[..cut]), &NullCodecResolver) {
                let p = d.streams()[0].params.clone();
                let mut dec = ctx.codecs.first_decoder(&p).unwrap();
                while let Ok(pkt) = d.next_packet() {
                    let _ = dec.send_packet(&pkt);
                    while dec.receive_frame().is_ok() {}
                }
            }
        }
    }
    assert!(open_demuxer(reader(&[]), &NullCodecResolver).is_err());
    assert!(open_demuxer(reader(&[0xff, 0x0a]), &NullCodecResolver).is_err());
    // A zero-length packet to an unpaced decoder is an error, not a panic.
    let mut dec = make_decoder(&CodecParameters::video(CodecId::new(CODEC_ID_STR))).unwrap();
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), Vec::new()))
        .unwrap();
    assert!(dec.receive_frame().is_err());
}
