//! Round 477 — the registry path: the encoder is live under `jpegxl`
//! with declared layouts and the `quality` schema, metadata blobs
//! travel demuxer → stream params / decoder → frame and frame / stream
//! params → encoder → file, the `jpegxl` muxer splices one packet per
//! picture into an animation, and a byte-exact JPEG reconstruction's
//! APP1 / APP2 payloads are the same blobs the decoder reports.

#![cfg(feature = "registry")]

use std::io::Cursor;

use oxideav_core::{
    BlobKind, CodecId, CodecOptions, CodecParameters, Error, Frame, MetadataBlob,
    NullCodecResolver, Packet, PixelFormat, ReadSeek, RuntimeContext, StreamInfo, TimeBase,
    VideoFrame, WriteSeek,
};
use oxideav_jpegxl::demux::{open_demuxer, open_muxer, CONTAINER_NAME};
use oxideav_jpegxl::registry::ACCEPTED_PIXEL_FORMATS;
use oxideav_jpegxl::{decode, decode_all, info, register, JxlImage, Signature, CODEC_ID_STR};

const ICC_FIXTURE: &[u8] = include_bytes!("fixtures/r451_icc.jxl");
const EXIF_XMP_TRANSCODE: &[u8] = include_bytes!("fixtures/r477_exif_xmp.jxl");
const EXIF_XMP_JPEG: &[u8] = include_bytes!("fixtures/r477_exif_xmp.jpg");
const ANIMATION: &[u8] = include_bytes!("fixtures/animation_3frame.jxl");

fn ctx() -> RuntimeContext {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    ctx
}

fn reader(bytes: &[u8]) -> Box<dyn ReadSeek> {
    Box::new(Cursor::new(bytes.to_vec()))
}

fn textured(width: u32, height: u32, channels: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height) as usize * channels);
    let mut h = seed.wrapping_mul(2654435761) | 1;
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels as u32 {
                h ^= h << 13;
                h ^= h >> 17;
                h ^= h << 5;
                out.push(((x * 2 + y * 3 + c * 50) % 200 + h % 56) as u8);
            }
        }
    }
    out
}

fn params(w: u32, h: u32, fmt: PixelFormat) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    p.width = Some(w);
    p.height = Some(h);
    p.pixel_format = Some(fmt);
    p
}

/// Encode one picture on a fresh registry encoder; returns the packet.
fn encode_one(ctx: &RuntimeContext, params: &CodecParameters, frame: VideoFrame) -> Packet {
    let mut enc = ctx.codecs.first_encoder(params).expect("first_encoder");
    enc.send_frame(&Frame::Video(frame)).expect("send_frame");
    enc.flush().unwrap();
    let pkt = enc.receive_packet().expect("one packet");
    assert!(matches!(enc.receive_packet(), Err(Error::Eof)));
    pkt
}

/// Demux + decode through the registry; returns the stream and frames.
fn demux_decode(ctx: &RuntimeContext, bytes: &[u8]) -> (StreamInfo, Vec<VideoFrame>) {
    let mut demux = ctx
        .containers
        .open_demuxer(CONTAINER_NAME, reader(bytes), &ctx.codecs)
        .expect("open_demuxer");
    let stream = demux.streams()[0].clone();
    let mut dec = ctx.codecs.first_decoder(&stream.params).expect("decoder");
    let mut frames = Vec::new();
    loop {
        match demux.next_packet() {
            Ok(pkt) => {
                dec.send_packet(&pkt).unwrap();
                while let Ok(Frame::Video(vf)) = dec.receive_frame() {
                    frames.push(vf);
                }
            }
            Err(Error::Eof) => break,
            Err(e) => panic!("{e}"),
        }
    }
    dec.flush().unwrap();
    while let Ok(Frame::Video(vf)) = dec.receive_frame() {
        frames.push(vf);
    }
    (stream, frames)
}

#[test]
fn encoder_is_registered_with_declared_layouts_and_quality_schema() {
    let ctx = ctx();
    let id = CodecId::new(CODEC_ID_STR);
    assert!(ctx.codecs.has_encoder(&id));
    let caps = &ctx.codecs.implementations(&id)[0].caps;
    assert_eq!(caps.accepted_pixel_formats, ACCEPTED_PIXEL_FORMATS);
    assert_eq!(ACCEPTED_PIXEL_FORMATS.len(), 8);
    assert!(caps.lossless && caps.lossy);
    let schema = ctx.codecs.encoder_options_schema(&id).unwrap();
    let names: Vec<&str> = schema.iter().map(|f| f.name).collect();
    assert_eq!(names, ["quality", "container", "group_size_shift"]);
    assert!(ctx.containers.muxer_names().any(|n| n == CONTAINER_NAME));
    assert_eq!(
        ctx.containers.container_for_extension("jxl"),
        Some(CONTAINER_NAME)
    );
    // Every declared layout encodes and comes back under the same label.
    for &fmt in ACCEPTED_PIXEL_FORMATS {
        let (w, h) = (9u32, 6u32);
        let jfmt: oxideav_jpegxl::PixelFormat = fmt.try_into().unwrap();
        let data: Vec<u8> = textured(w, h, jfmt.bytes_per_pixel(), 3);
        let img = JxlImage::packed(w, h, jfmt, data.clone()).unwrap();
        let vf: VideoFrame = (&img).into();
        let pkt = encode_one(&ctx, &params(w, h, fmt), vf);
        let back = decode(&pkt.data).unwrap();
        assert_eq!(PixelFormat::from(back.format), fmt, "{fmt:?}");
        assert_eq!(back.as_bytes().unwrap(), &data[..], "{fmt:?}");
        let (stream, frames) = demux_decode(&ctx, &pkt.data);
        assert_eq!(stream.params.pixel_format, Some(fmt));
        assert_eq!(frames.len(), 1);
        assert_eq!(
            frames[0].planes[0].data, data,
            "{fmt:?} through the registry"
        );
    }
    // Undeclared layouts are refused by type, never converted.
    for fmt in [
        PixelFormat::Yuv420P,
        PixelFormat::YuvJ444P,
        PixelFormat::Gray10Le,
    ] {
        assert!(
            matches!(
                ctx.codecs.first_encoder(&params(4, 4, fmt)),
                Err(Error::Unsupported(_))
            ),
            "{fmt:?}"
        );
    }
}

#[test]
fn quality_option_is_honoured_through_the_registry() {
    let ctx = ctx();
    let (w, h) = (96u32, 64u32);
    let data = textured(w, h, 3, 11);
    let mut sizes = Vec::new();
    for q in ["50", "90", "100"] {
        let mut p = params(w, h, PixelFormat::Rgb24);
        p.options = CodecOptions::new().set("quality", q);
        let img = JxlImage::from_rgb8(w, h, data.clone()).unwrap();
        let pkt = encode_one(&ctx, &p, (&img).into());
        let back = decode(&pkt.data).unwrap();
        if q == "100" {
            assert_eq!(back.as_bytes().unwrap(), &data[..]);
        } else {
            assert_ne!(back.as_bytes().unwrap(), &data[..], "q{q} is lossy");
        }
        sizes.push(pkt.data.len());
    }
    assert!(
        sizes[0] < sizes[1] && sizes[1] < sizes[2],
        "ladder {sizes:?}"
    );
    // `container` forces the box file.
    let mut p = params(w, h, PixelFormat::Rgb24);
    p.options = CodecOptions::new().set("container", "true");
    let img = JxlImage::from_rgb8(w, h, data).unwrap();
    let pkt = encode_one(&ctx, &p, (&img).into());
    assert_eq!(info(&pkt.data).unwrap().signature, Signature::Isobmff);
}

fn exif_blob(tag: u8) -> Vec<u8> {
    let mut v = b"II*\0".to_vec();
    v.extend_from_slice(&8u32.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes());
    v.extend_from_slice(&0x0112u16.to_le_bytes());
    v.extend_from_slice(&3u16.to_le_bytes());
    v.extend_from_slice(&1u32.to_le_bytes());
    v.extend_from_slice(&u16::from(tag).to_le_bytes());
    v.extend_from_slice(&0u16.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    v
}

#[test]
fn blobs_travel_both_ways_through_the_registry() {
    let ctx = ctx();
    // Demuxer: the ICC fixture's profile is a stream blob; the decoder
    // repeats it on the first frame.
    let (stream, frames) = demux_decode(&ctx, ICC_FIXTURE);
    let icc = stream
        .params
        .blob(&BlobKind::ICC)
        .expect("stream ICC blob")
        .data
        .clone();
    assert_eq!(
        decode(ICC_FIXTURE).unwrap().metadata.icc.as_deref(),
        Some(&icc[..])
    );
    assert_eq!(
        frames[0].blob(&BlobKind::ICC).map(|b| b.data),
        Some(icc.clone())
    );
    assert!(stream.params.blob(&BlobKind::EXIF).is_none());

    // Encoder: stream blobs on the parameters, a frame blob overriding
    // the stream blob of the same kind (Exif), ICC into the codestream,
    // Exif / XMP into boxes.
    let (w, h) = (10u32, 7u32);
    let data = textured(w, h, 4, 5);
    let mut p = params(w, h, PixelFormat::Rgba);
    p.blobs = vec![
        MetadataBlob::new(BlobKind::ICC, icc.clone()),
        MetadataBlob::new(BlobKind::EXIF, exif_blob(6)),
        MetadataBlob::new(
            BlobKind::XMP,
            b"<x:xmpmeta xmlns:x='adobe:ns:meta/'/>".to_vec(),
        ),
        MetadataBlob::new(BlobKind::IPTC, vec![0x1C, 0x02]), // not a JPEG XL payload: dropped
    ];
    let img = JxlImage::from_rgba8(w, h, data.clone()).unwrap();
    let mut vf: VideoFrame = (&img).into();
    vf.set_blobs(vec![MetadataBlob::new(BlobKind::EXIF, exif_blob(3))]);
    let pkt = encode_one(&ctx, &p, vf);
    let i = info(&pkt.data).unwrap();
    assert_eq!(i.signature, Signature::Isobmff);
    assert!(i.has_icc && i.has_exif && i.has_xmp);
    let back = decode(&pkt.data).unwrap();
    assert_eq!(back.as_bytes().unwrap(), &data[..]);
    assert_eq!(back.metadata.icc.as_deref(), Some(&icc[..]));
    assert_eq!(
        back.metadata.exif.as_deref(),
        Some(&exif_blob(3)[..]),
        "frame Exif wins"
    );
    assert_eq!(
        back.metadata.xmp.as_deref(),
        Some(&b"<x:xmpmeta xmlns:x='adobe:ns:meta/'/>"[..])
    );
    // And back through the demuxer: three stream blobs, in ICC / Exif /
    // XMP order, repeated on the frame.
    let (stream, frames) = demux_decode(&ctx, &pkt.data);
    let kinds: Vec<BlobKind> = stream.params.blobs.iter().map(|b| b.kind.clone()).collect();
    assert_eq!(kinds, [BlobKind::ICC, BlobKind::EXIF, BlobKind::XMP]);
    assert_eq!(
        stream.params.blob(&BlobKind::EXIF).unwrap().data,
        exif_blob(3)
    );
    assert_eq!(frames[0].blobs(), stream.params.blobs);
}

#[test]
fn reconstructed_jpeg_app_segments_are_the_decoder_blobs() {
    let ctx = ctx();
    let (stream, frames) = demux_decode(&ctx, EXIF_XMP_TRANSCODE);
    assert_eq!(stream.params.pixel_format, Some(PixelFormat::YuvJ420P));
    let exif = stream
        .params
        .blob(&BlobKind::EXIF)
        .expect("Exif blob")
        .data
        .clone();
    let xmp = stream
        .params
        .blob(&BlobKind::XMP)
        .expect("XMP blob")
        .data
        .clone();
    assert_eq!(frames[0].blob(&BlobKind::EXIF).unwrap().data, exif);
    assert!(exif.starts_with(b"II*\0"), "TIFF-structured Exif payload");
    assert!(xmp.starts_with(b"<?xpacket"));
    // Byte-exact reconstruction, then the APP1 segments of the JPEG.
    let jpeg = oxideav_jpegxl::jpeg_reconstruct::reconstruct_jpeg(EXIF_XMP_TRANSCODE).unwrap();
    assert_eq!(jpeg, EXIF_XMP_JPEG);
    let mut pos = 2usize;
    let mut found_exif = None;
    let mut found_xmp = None;
    while pos + 4 <= jpeg.len() && jpeg[pos] == 0xFF {
        let marker = jpeg[pos + 1];
        let len = usize::from(u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]));
        let payload = &jpeg[pos + 4..pos + 2 + len];
        if marker == 0xE1 {
            if let Some(rest) = payload.strip_prefix(b"Exif\0\0") {
                found_exif = Some(rest.to_vec());
            } else if let Some(rest) = payload.strip_prefix(b"http://ns.adobe.com/xap/1.0/\0") {
                found_xmp = Some(rest.to_vec());
            }
        }
        if marker == 0xDA {
            break;
        }
        pos += 2 + len;
    }
    assert_eq!(
        found_exif.as_deref(),
        Some(&exif[..]),
        "APP1 Exif == EXIF blob"
    );
    assert_eq!(found_xmp.as_deref(), Some(&xmp[..]), "APP1 XMP == XMP blob");
    // The ICC transcode: its APP2 chunks reassemble to the ICC blob.
    let (stream, _) = demux_decode(&ctx, ICC_FIXTURE);
    let icc = stream.params.blob(&BlobKind::ICC).unwrap().data.clone();
    let jpeg = oxideav_jpegxl::jpeg_reconstruct::reconstruct_jpeg(ICC_FIXTURE).unwrap();
    let mut pos = 2usize;
    let mut chunks: Vec<(u8, Vec<u8>)> = Vec::new();
    while pos + 4 <= jpeg.len() && jpeg[pos] == 0xFF && jpeg[pos + 1] != 0xDA {
        let marker = jpeg[pos + 1];
        let len = usize::from(u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]));
        let payload = &jpeg[pos + 4..pos + 2 + len];
        if marker == 0xE2 {
            if let Some(rest) = payload.strip_prefix(b"ICC_PROFILE\0") {
                chunks.push((rest[0], rest[2..].to_vec()));
            }
        }
        pos += 2 + len;
    }
    chunks.sort_by_key(|c| c.0);
    let joined: Vec<u8> = chunks.into_iter().flat_map(|c| c.1).collect();
    assert_eq!(joined, icc, "APP2 ICC_PROFILE chunks == ICC blob");
}

/// A `WriteSeek` sink whose bytes stay readable after the muxer owns it.
#[derive(Clone, Default)]
struct Shared(std::sync::Arc<std::sync::Mutex<Cursor<Vec<u8>>>>);

impl std::io::Write for Shared {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().write(b)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl std::io::Seek for Shared {
    fn seek(&mut self, s: std::io::SeekFrom) -> std::io::Result<u64> {
        self.0.lock().unwrap().seek(s)
    }
}

/// Drive the `jpegxl` muxer over `packets` and return the file.
fn mux(time_base: TimeBase, packets: &[Packet]) -> oxideav_core::Result<Vec<u8>> {
    let stream = StreamInfo {
        index: 0,
        params: params(1, 1, PixelFormat::Rgb24),
        time_base,
        start_time: Some(0),
        duration: None,
    };
    let sink = Shared::default();
    let mut muxer = open_muxer(Box::new(sink.clone()), std::slice::from_ref(&stream))?;
    muxer.write_header()?;
    for p in packets {
        muxer.write_packet(p)?;
    }
    muxer.write_trailer()?;
    drop(muxer);
    let bytes = sink.0.lock().unwrap().get_ref().clone();
    Ok(bytes)
}

#[test]
fn muxer_splices_one_packet_per_picture_into_an_animation() {
    let ctx = ctx();
    let (w, h) = (40u32, 24u32);
    let tb = TimeBase::new(1, 100); // ticks of 10 ms
    let mut packets = Vec::new();
    let mut sources = Vec::new();
    for i in 0..3u32 {
        let data = textured(w, h, 3, 20 + i);
        let img = JxlImage::from_rgb8(w, h, data.clone()).unwrap();
        let mut vf: VideoFrame = (&img).into();
        vf.pts = Some(i64::from(i) * 5);
        let mut pkt = encode_one(&ctx, &params(w, h, PixelFormat::Rgb24), vf);
        pkt.time_base = tb;
        pkt.duration = Some(5);
        packets.push(pkt);
        sources.push(data);
    }
    let bytes = mux(tb, &packets).expect("mux");
    let i = info(&bytes).unwrap();
    assert_eq!(i.frames, 3);
    assert_eq!(i.animation.unwrap().ticks_per_second, (100, 1));
    let frames = decode_all(&bytes).unwrap();
    for (k, f) in frames.iter().enumerate() {
        assert_eq!(f.image.as_bytes().unwrap(), &sources[k][..], "frame {k}");
        assert_eq!(f.ticks, 5);
        assert_eq!(f.delay, Some(std::time::Duration::from_millis(50)));
    }
    // Through the registry demuxer too: three packets in the animation tick.
    let (stream, vframes) = demux_decode(&ctx, &bytes);
    assert_eq!(stream.time_base, TimeBase::new(1, 100));
    assert_eq!(vframes.len(), 3);
    assert_eq!(vframes[2].pts, Some(10));
    // One packet is written verbatim.
    let one = mux(tb, &packets[..1]).unwrap();
    assert_eq!(one, packets[0].data);
    // Durations fall back to pts gaps when the packets carry none.
    let mut gapped = packets.clone();
    for p in &mut gapped {
        p.duration = None;
    }
    let frames = decode_all(&mux(tb, &gapped).unwrap()).unwrap();
    assert!(frames.iter().all(|f| f.ticks == 5));
    // Metadata boxes of the first packet are kept.
    let mut boxed = packets.clone();
    let mut p = params(w, h, PixelFormat::Rgb24);
    p.blobs = vec![MetadataBlob::new(BlobKind::XMP, b"<x/>".to_vec())];
    let img = JxlImage::from_rgb8(w, h, sources[0].clone()).unwrap();
    let mut first = encode_one(&ctx, &p, (&img).into());
    first.time_base = tb;
    first.duration = Some(5);
    boxed[0] = first;
    let bytes = mux(tb, &boxed).unwrap();
    let i = info(&bytes).unwrap();
    assert!(i.has_xmp && i.frames == 3);
    assert_eq!(
        decode(&bytes).unwrap().metadata.xmp.as_deref(),
        Some(&b"<x/>"[..])
    );
}

#[test]
fn muxer_refuses_what_it_cannot_retime() {
    let ctx = ctx();
    let tb = TimeBase::new(1, 100);
    let a = encode_one(
        &ctx,
        &params(8, 8, PixelFormat::Rgb24),
        (&JxlImage::from_rgb8(8, 8, textured(8, 8, 3, 1)).unwrap()).into(),
    );
    let b = encode_one(
        &ctx,
        &params(8, 9, PixelFormat::Rgb24),
        (&JxlImage::from_rgb8(8, 9, textured(8, 9, 3, 2)).unwrap()).into(),
    );
    match mux(tb, &[a.clone(), b]) {
        Err(Error::Unsupported(m)) => assert!(m.contains("8×9") && m.contains("8×8"), "{m}"),
        other => panic!("expected Unsupported, got {:?}", other.map(|v| v.len())),
    }
    // A foreign animation (not a still) cannot be spliced.
    let mut foreign = a.clone();
    foreign.data = ANIMATION.to_vec();
    assert!(matches!(
        mux(tb, &[a.clone(), foreign]),
        Err(Error::Unsupported(_))
    ));
    // Garbage is InvalidData, an empty packet is refused up front.
    let mut junk = a.clone();
    junk.data = b"not a jxl".to_vec();
    assert!(matches!(
        mux(tb, &[a.clone(), junk]),
        Err(Error::InvalidData(_))
    ));
    let mut empty = a.clone();
    empty.data.clear();
    assert!(matches!(mux(tb, &[a, empty]), Err(Error::InvalidData(_))));
    // Two streams, or a non-jpegxl stream, are refused at open.
    let sink: Box<dyn WriteSeek> = Box::new(Cursor::new(Vec::new()));
    let s = StreamInfo {
        index: 0,
        params: CodecParameters::video(CodecId::new("png")),
        time_base: tb,
        start_time: None,
        duration: None,
    };
    assert!(matches!(
        open_muxer(sink, std::slice::from_ref(&s)),
        Err(Error::Unsupported(_))
    ));
    let sink: Box<dyn WriteSeek> = Box::new(Cursor::new(Vec::new()));
    let js = StreamInfo {
        params: params(1, 1, PixelFormat::Rgb24),
        ..s
    };
    assert!(matches!(
        open_muxer(sink, &[js.clone(), js]),
        Err(Error::Unsupported(_))
    ));
    let _ = open_demuxer(reader(ANIMATION), &NullCodecResolver).unwrap();
}

/// Round-477 `mux_splice` fuzz find: a byte flip in a spliced still made
/// the decoder index past a one-entry TOC (LfGroup slot on a frame whose
/// GlobalModular did not finish) — a typed error now, never a panic.
#[test]
fn flipped_spliced_still_fails_cleanly() {
    let tb = TimeBase::new(1, 100);
    let (w, h) = (10u32, 18u32);
    let body = [0xf9u8, 0x11, 0x11];
    let n = (w * h * 3) as usize;
    let mut packets = Vec::new();
    for k in 0..2u8 {
        let px: Vec<u8> = (0..n)
            .map(|i| body[(i + usize::from(k)) % body.len()])
            .collect();
        let img = JxlImage::from_rgb8(w, h, px).unwrap();
        let mut bytes =
            oxideav_jpegxl::encode(&img, &oxideav_jpegxl::EncodeOptions::default()).unwrap();
        if k == 1 {
            let at = 0x11 % bytes.len();
            bytes[at] ^= body[2] | 1;
        }
        let mut p = Packet::new(0, tb, bytes);
        p.pts = Some(i64::from(k) * 7);
        p.duration = Some(7);
        packets.push(p);
    }
    if let Ok(out) = mux(tb, &packets) {
        let _ = decode_all(&out);
        let _ = decode(&out);
    }
    // Every byte flip over the whole spliced file, too.
    let clean: Vec<Packet> = packets
        .iter()
        .enumerate()
        .map(|(k, p)| {
            let img =
                JxlImage::from_rgb8(w, h, (0..n).map(|i| body[(i + k) % 3]).collect()).unwrap();
            let mut q = p.clone();
            q.data =
                oxideav_jpegxl::encode(&img, &oxideav_jpegxl::EncodeOptions::default()).unwrap();
            q
        })
        .collect();
    let out = mux(tb, &clean).unwrap();
    for at in 0..out.len() {
        let mut flipped = out.clone();
        flipped[at] ^= 0x11;
        let _ = decode_all(&flipped);
    }
}
