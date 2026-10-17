#![no_main]

//! Muxer robustness target: arbitrary bytes split into packets go
//! through the `jpegxl` muxer's splice path (`parse_still` on every
//! packet, header rewrite, section copy). The muxer must return a
//! `Result` on hostile packets, never panic; when every packet is a
//! still this crate's encoder produced, the spliced animation must
//! decode to the same pictures.

use std::io::Cursor;

use libfuzzer_sys::fuzz_target;
use oxideav_core::{Packet, PixelFormat, StreamInfo, TimeBase, WriteSeek, CodecId, CodecParameters};
use oxideav_jpegxl::demux::open_muxer;
use oxideav_jpegxl::{decode_all, encode, EncodeOptions, JxlImage, CODEC_ID_STR};

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

fn mux(packets: &[Packet]) -> oxideav_core::Result<Vec<u8>> {
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(1);
    params.height = Some(1);
    params.pixel_format = Some(PixelFormat::Rgb24);
    let stream = StreamInfo {
        index: 0,
        params,
        time_base: TimeBase::new(1, 100),
        start_time: Some(0),
        duration: None,
    };
    let sink = Shared::default();
    let mut m = open_muxer(Box::new(sink.clone()), std::slice::from_ref(&stream))?;
    m.write_header()?;
    for p in packets {
        m.write_packet(p)?;
    }
    m.write_trailer()?;
    drop(m);
    let bytes = sink.0.lock().unwrap().get_ref().clone();
    Ok(bytes)
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let mode = data[0] % 3;
    let body = &data[1..];
    if mode == 0 {
        // Hostile: the bytes themselves as two to four packets.
        let n = usize::from(data[1] % 3) + 2;
        let chunk = body.len() / n;
        if chunk == 0 {
            return;
        }
        let packets: Vec<Packet> = (0..n)
            .map(|i| {
                let mut p = Packet::new(0, TimeBase::new(1, 100), body[i * chunk..(i + 1) * chunk].to_vec());
                p.pts = Some(i as i64 * 5);
                p.duration = Some(5);
                p
            })
            .collect();
        let _ = mux(&packets);
        return;
    }
    // Genuine stills from the encoder (geometry from the data), the
    // second one possibly corrupted by a byte flip (mode 2).
    let width = u32::from(data[1]) % 40 + 1;
    let height = u32::from(data[2]) % 20 + 1;
    let n = (width * height * 3) as usize;
    let mut packets = Vec::new();
    let mut sources = Vec::new();
    for k in 0..2u8 {
        let px: Vec<u8> = (0..n).map(|i| body.get((i + usize::from(k)) % body.len().max(1)).copied().unwrap_or(k)).collect();
        let img = JxlImage::from_rgb8(width, height, px.clone()).unwrap();
        let mut bytes = encode(&img, &EncodeOptions::default()).unwrap();
        if mode == 2 && k == 1 && !bytes.is_empty() {
            let at = usize::from(data[3]) % bytes.len();
            bytes[at] ^= body.get(2).copied().unwrap_or(0xFF) | 1;
        }
        let mut p = Packet::new(0, TimeBase::new(1, 100), bytes);
        p.pts = Some(i64::from(k) * 7);
        p.duration = Some(7);
        packets.push(p);
        sources.push(px);
    }
    match mux(&packets) {
        Ok(out) => {
            if mode == 1 {
                let frames = decode_all(&out).expect("spliced animation decodes");
                assert_eq!(frames.len(), 2);
                for (k, f) in frames.iter().enumerate() {
                    assert_eq!(f.image.as_bytes().unwrap(), &sources[k][..]);
                    assert_eq!(f.ticks, 7);
                }
            } else {
                // A corrupted still that still parsed: the splice must at
                // least be a decodable-or-rejected file, never a panic.
                let _ = decode_all(&out);
            }
        }
        Err(_) => assert_ne!(mode, 1, "genuine stills always splice"),
    }
});
