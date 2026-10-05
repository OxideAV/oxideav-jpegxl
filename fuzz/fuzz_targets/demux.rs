#![no_main]

//! Framework-container robustness target: arbitrary bytes through
//! `demux::open_demuxer` (a bare codestream or the box file, the
//! signature decides), every packet through the paced registry
//! decoder, nothing may panic. Headers can declare huge dimensions, so
//! inputs whose header-declared area exceeds 1 Mi pixels are skipped
//! after the header walk (the library must still fail cleanly below
//! that cap).

use libfuzzer_sys::fuzz_target;
use oxideav_core::{Error, NullCodecResolver, ReadSeek};
use oxideav_jpegxl::demux::open_demuxer;

fuzz_target!(|data: &[u8]| {
    let input: Box<dyn ReadSeek> = Box::new(std::io::Cursor::new(data.to_vec()));
    let Ok(mut demux) = open_demuxer(input, &NullCodecResolver) else {
        return;
    };
    let stream = demux.streams()[0].clone();
    let w = u64::from(stream.params.width.unwrap_or(0));
    let h = u64::from(stream.params.height.unwrap_or(0));
    assert!(w > 0 && h > 0);
    let decode = w * h <= 1 << 20;
    let mut dec = oxideav_jpegxl::make_decoder(&stream.params).expect("paced decoder");
    let mut packets = 0usize;
    loop {
        match demux.next_packet() {
            Ok(pkt) => {
                packets += 1;
                assert!(packets <= data.len().max(1));
                if packets == 1 {
                    assert_eq!(pkt.data.len(), data.len());
                } else {
                    assert!(pkt.data.is_empty());
                }
                if decode {
                    if dec.send_packet(&pkt).is_ok() {
                        while dec.receive_frame().is_ok() {}
                    }
                }
            }
            Err(Error::Eof) => break,
            Err(_) => unreachable!("the demuxer only ends with Eof"),
        }
    }
    let _ = dec.flush();
    while dec.receive_frame().is_ok() {}
});
