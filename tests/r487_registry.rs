//! Round 487 — the lossy coder through the registry: the `quality` /
//! `distance` / `modular` / `effort` options reach the encoder, a lossy
//! packet is a VarDCT still, and the `jpegxl` muxer splices VarDCT
//! stills into a lossy animation (mixing coders is refused).

#![cfg(feature = "registry")]

use std::io::{Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};

use oxideav_core::{
    CodecId, CodecOptions, CodecParameters, Frame, Packet, PixelFormat, RuntimeContext, StreamInfo,
    TimeBase, VideoFrame,
};
use oxideav_jpegxl::demux::open_muxer;
use oxideav_jpegxl::{CODEC_ID_STR, JxlImage, decode, decode_all, info, register};

#[derive(Clone, Default)]
struct Shared(Arc<Mutex<std::io::Cursor<Vec<u8>>>>);

impl Write for Shared {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Seek for Shared {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.0.lock().unwrap().seek(pos)
    }
}

fn picture(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3u32 {
                let v = 128.0
                    + 80.0
                        * ((f64::from(x) / 9.0 + f64::from(c + seed)).sin()
                            * (f64::from(y) / 13.0).cos());
                out.push(v.clamp(0.0, 255.0) as u8);
            }
        }
    }
    out
}

fn params(w: u32, h: u32, options: CodecOptions) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    p.width = Some(w);
    p.height = Some(h);
    p.pixel_format = Some(PixelFormat::Rgb24);
    p.options = options;
    p
}

fn encode_one(ctx: &RuntimeContext, params: &CodecParameters, data: &[u8]) -> Packet {
    let (w, h) = (params.width.unwrap(), params.height.unwrap());
    let img = JxlImage::from_rgb8(w, h, data.to_vec()).unwrap();
    let vf: VideoFrame = (&img).into();
    let mut enc = ctx.codecs.first_encoder(params).expect("first_encoder");
    enc.send_frame(&Frame::Video(vf)).expect("send_frame");
    enc.flush().unwrap();
    enc.receive_packet().expect("one packet")
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let sse: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum();
    10.0 * (255.0f64 * 255.0 / (sse / a.len() as f64).max(1e-9)).log10()
}

fn mux(time_base: TimeBase, packets: &[Packet]) -> oxideav_core::Result<Vec<u8>> {
    let stream = StreamInfo {
        index: 0,
        params: params(1, 1, CodecOptions::new()),
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
    Ok(sink.0.lock().unwrap().get_ref().clone())
}

#[test]
fn lossy_options_reach_the_encoder() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let (w, h) = (96u32, 64u32);
    let src = picture(w, h, 1);
    let lossless = encode_one(&ctx, &params(w, h, CodecOptions::new()), &src);
    assert!(!info(&lossless.data).unwrap().xyb_encoded);
    assert_eq!(
        decode(&lossless.data).unwrap().as_bytes().unwrap(),
        &src[..]
    );

    let q80 = encode_one(
        &ctx,
        &params(w, h, CodecOptions::new().set("quality", "80")),
        &src,
    );
    assert!(info(&q80.data).unwrap().xyb_encoded, "quality 80 is VarDCT");
    assert!(q80.data.len() < lossless.data.len());
    assert!(psnr(&src, decode(&q80.data).unwrap().as_bytes().unwrap()) > 34.0);

    let d4 = encode_one(
        &ctx,
        &params(w, h, CodecOptions::new().set("distance", "4")),
        &src,
    );
    assert!(info(&d4.data).unwrap().xyb_encoded);
    assert!(d4.data.len() < q80.data.len());

    let modular = encode_one(
        &ctx,
        &params(
            w,
            h,
            CodecOptions::new()
                .set("quality", "80")
                .set("modular", "true")
                .set("effort", "fast"),
        ),
        &src,
    );
    assert!(!info(&modular.data).unwrap().xyb_encoded);

    for bad in [("effort", "extreme"), ("distance", "-1")] {
        let p = params(w, h, CodecOptions::new().set(bad.0, bad.1));
        assert!(ctx.codecs.first_encoder(&p).is_err(), "{bad:?} is refused");
    }
}

#[test]
fn the_muxer_splices_vardct_stills_into_a_lossy_animation() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let (w, h) = (300u32, 280u32); // four groups per frame
    let tb = TimeBase::new(1, 50);
    let lossy = params(w, h, CodecOptions::new().set("distance", "1.5"));
    let mut packets = Vec::new();
    let mut sources = Vec::new();
    for i in 0..3u32 {
        let src = picture(w, h, 10 + i);
        let mut pkt = encode_one(&ctx, &lossy, &src);
        pkt.time_base = tb;
        pkt.pts = Some(i64::from(i) * 4);
        pkt.duration = Some(4);
        packets.push(pkt);
        sources.push(src);
    }
    let bytes = mux(tb, &packets).expect("mux");
    let i = info(&bytes).unwrap();
    assert!(i.xyb_encoded && i.frames == 3);
    let frames = decode_all(&bytes).unwrap();
    assert_eq!(frames.len(), 3);
    for (k, f) in frames.iter().enumerate() {
        assert_eq!(f.ticks, 4, "frame {k}");
        // The spliced frame is the still's frame, bit for bit.
        let still = decode(&packets[k].data).unwrap();
        assert_eq!(f.image.as_bytes().unwrap(), still.as_bytes().unwrap());
        assert!(psnr(&sources[k], f.image.as_bytes().unwrap()) > 34.0);
    }
    // A lossless (Modular) still does not splice with VarDCT ones.
    let mut odd = encode_one(&ctx, &params(w, h, CodecOptions::new()), &sources[0]);
    odd.time_base = tb;
    odd.duration = Some(4);
    let mixed = [packets[0].clone(), odd];
    assert!(mux(tb, &mixed).is_err());
}
