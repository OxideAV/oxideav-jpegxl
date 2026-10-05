//! The `jpegxl` framework container (demuxer only): a bare codestream
//! (`FF 0A`) or the 18181-2 box file (`00 00 00 0C JXL␣ …`), extension
//! `.jxl`, opened through `ctx.containers.probe_input` → `open_demuxer`
//! → `first_decoder` the way `oxideav-image` (Layer 2) reads any image.
//!
//! The demuxer declares one video stream the way [`crate::info`]
//! describes the file — width / height, the **native** contract layout
//! as `pixel_format`, the §A.6 `ColourEncoding` as `color_signal` (JPEG
//! XL always signals one; with an ICC profile the enumerated fields
//! are unspecified and nothing is stamped), and the decoder option
//! `pacing = packet` ([`crate::registry::PACING_PACKET`]) so frames
//! come out one per packet. An image whose channel set has no contract
//! layout still opens with `pixel_format = None`; the decoder then
//! reports `Unsupported`.
//!
//! **Stills** are one keyframe packet holding the whole file (time
//! base 1/1, `pts` 0). **Animations** are one packet per presented
//! frame in file order (the Layer 2 timing rule): the stream time base
//! is the animation header's tick, `tps_denominator / tps_numerator`
//! seconds, each packet's `duration` is its frame's FrameHeader
//! `duration` in ticks and `pts` the running sum. A JPEG XL frame is
//! not decodable on its own (frames reference and blend over earlier
//! ones), so only the first packet carries bytes — the whole file —
//! and the following packets are zero-length pacing packets that
//! release the next frame from the decoder; `metadata()` carries
//! `("loop_count", n)` (`0` = forever) plus `("icc" | "exif" | "xmp",
//! "present")` flags (the blobs have no framework carriage yet).
//!
//! **No muxer**: the crate is decoder-only by ruling (every `encode*`
//! returns `Unsupported`), so there is nothing a muxer could receive.
//!
//! Lives behind the `registry` feature.

use std::io::{Read, SeekFrom};

use oxideav_core::{
    CodecId, CodecParameters, CodecResolver, ContainerRegistry, Demuxer, Error, Packet,
    PixelFormat, ProbeData, ProbeScore, ReadSeek, Result, StreamInfo, TimeBase, MAX_PROBE_SCORE,
    PROBE_SCORE_EXTENSION,
};

use crate::api::describe;
use crate::registry::{to_color_signal, OPTION_PACING, PACING_PACKET};
use crate::CODEC_ID_STR;

/// The container name (both signatures).
pub const CONTAINER_NAME: &str = "jpegxl";

/// Register the demuxer, the probe and the `.jxl` extension.
pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer(CONTAINER_NAME, open_demuxer);
    reg.register_extension("jxl", CONTAINER_NAME);
    reg.register_probe(CONTAINER_NAME, probe);
}

/// Either JXL signature scores full marks, the `.jxl` extension alone
/// the conventional weak score.
pub fn probe(data: &ProbeData) -> ProbeScore {
    if crate::container::detect(data.buf).is_some() {
        return MAX_PROBE_SCORE;
    }
    if data.ext == Some("jxl") {
        PROBE_SCORE_EXTENSION
    } else {
        0
    }
}

/// Open a JXL file as a one-stream container (see the module docs for
/// the packet layout). The headers of every frame are walked eagerly
/// (no section decode) so the stream carries accurate geometry and
/// timing before the decoder runs.
pub fn open_demuxer(
    mut input: Box<dyn ReadSeek>,
    _codecs: &dyn CodecResolver,
) -> Result<Box<dyn Demuxer>> {
    input.seek(SeekFrom::Start(0))?;
    let mut buf = Vec::new();
    input.read_to_end(&mut buf)?;
    drop(input);
    if crate::container::detect(&buf).is_none() {
        return Err(Error::invalid("jpegxl demuxer: no JXL signature"));
    }
    let d = describe(&buf)?;
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(d.width);
    params.height = Some(d.height);
    params.pixel_format = d.format.map(PixelFormat::from);
    if d.color.is_specified() {
        params.color_signal = to_color_signal(&d.color);
    }
    params.options.insert(OPTION_PACING, PACING_PACKET);

    // Timing: the animation tick as the stream time base, else 1/1.
    let (time_base, ticks): (TimeBase, Vec<i64>) = match d.animation {
        Some(a) => {
            let (num, den) = a.ticks_per_second;
            if num == 0 || den == 0 {
                return Err(Error::invalid(format!(
                    "jpegxl demuxer: animation header ticks per second {num}/{den} is not a rate"
                )));
            }
            // One tick is den / num seconds.
            (
                TimeBase::new(den as i64, num as i64),
                d.durations.iter().map(|&t| i64::from(t)).collect(),
            )
        }
        None => (TimeBase::new(1, 1), vec![0]),
    };
    let total: i64 = ticks.iter().sum();
    let mut metadata = Vec::new();
    if let Some(a) = d.animation {
        metadata.push(("loop_count".to_owned(), a.loops.to_string()));
    }
    for (flag, present) in [("icc", d.has_icc), ("exif", d.has_exif), ("xmp", d.has_xmp)] {
        if present {
            metadata.push((flag.to_owned(), "present".to_owned()));
        }
    }
    let stream = StreamInfo {
        index: 0,
        params,
        time_base,
        start_time: Some(0),
        duration: (ticks.len() > 1).then_some(total),
    };
    Ok(Box::new(JxlDemuxer {
        streams: vec![stream],
        metadata,
        time_base,
        ticks,
        next: 0,
        pts: 0,
        data: Some(buf),
    }))
}

struct JxlDemuxer {
    streams: Vec<StreamInfo>,
    metadata: Vec<(String, String)>,
    time_base: TimeBase,
    /// Per-frame durations in ticks.
    ticks: Vec<i64>,
    /// Next frame to emit.
    next: usize,
    /// Running presentation time in ticks.
    pts: i64,
    /// The whole file, carried by the first packet only.
    data: Option<Vec<u8>>,
}

impl Demuxer for JxlDemuxer {
    fn format_name(&self) -> &str {
        CONTAINER_NAME
    }
    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }
    fn metadata(&self) -> &[(String, String)] {
        &self.metadata
    }
    fn next_packet(&mut self) -> Result<Packet> {
        let Some(&dur) = self.ticks.get(self.next) else {
            return Err(Error::Eof);
        };
        // The first packet carries the file; later ones only pace the
        // decoder (see the module docs).
        let data = self.data.take().unwrap_or_default();
        let mut pkt = Packet::new(0, self.time_base, data);
        pkt.pts = Some(self.pts);
        pkt.dts = Some(self.pts);
        pkt.duration = (dur > 0).then_some(dur);
        pkt.flags.keyframe = self.next == 0;
        self.pts = self.pts.saturating_add(dur);
        self.next += 1;
        Ok(pkt)
    }
}
