//! The `jpegxl` framework container (demuxer + muxer): a bare codestream
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
//! "present")` flags, and the stream's `CodecParameters::blobs` carry
//! the codestream ICC profile and the `Exif` / `xml ` box payloads as
//! `BlobKind::ICC` / `EXIF` / `XMP` (the decoder repeats them on the
//! first frame).
//!
//! **Muxer** ([`open_muxer`]): one video stream of codec `jpegxl`. The
//! registry encoder emits one complete still file per picture; one
//! packet is written as it is, and several packets are spliced into one
//! animated codestream — the image headers are rewritten with an
//! `AnimationHeader` whose tick is the stream time base, every frame
//! header is re-emitted with the packet's `duration` in ticks (the
//! last packet becomes `is_last`), and the frames' TOCs and section
//! bytes are copied verbatim. Metadata boxes of the first packet are
//! kept. Only frames of this crate's encoder shape (plain `kModular`,
//! `kReplace`, no crop / upsampling / restoration filter) can be
//! re-timed; anything else is `Unsupported`.
//!
//! Lives behind the `registry` feature.

use std::io::{Read, SeekFrom, Write};

use oxideav_core::{
    CodecId, CodecParameters, CodecResolver, ContainerRegistry, Demuxer, Error, MAX_PROBE_SCORE,
    MediaType, Muxer, PROBE_SCORE_EXTENSION, Packet, PixelFormat, ProbeData, ProbeScore, ReadSeek,
    Result, StreamInfo, TimeBase, WriteSeek,
};

use crate::CODEC_ID_STR;
use crate::api::describe;
use crate::bitwriter::BitWriter;
use crate::encoder::{AnimationSpec, ColourSpec, EncodeSpec, FrameShape};
use crate::frame_header::{BlendMode, Encoding, FrameType};
use crate::metadata_fdis::{ColourSpace, ExtraChannelType};
use crate::registry::{OPTION_PACING, PACING_PACKET, blobs_of_metadata, to_color_signal};

/// The container name (both signatures).
pub const CONTAINER_NAME: &str = "jpegxl";

/// Register the demuxer, the muxer, the probe and the `.jxl` extension.
pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer(CONTAINER_NAME, open_demuxer);
    reg.register_muxer(CONTAINER_NAME, open_muxer);
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
    params.blobs = blobs_of_metadata(&d.metadata);

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

// ---- muxer ----------------------------------------------------------------------

/// Open the `jpegxl` muxer on `output` for exactly one video stream of
/// codec `jpegxl` (see the module docs for what it accepts).
pub fn open_muxer(output: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Box<dyn Muxer>> {
    let [stream] = streams else {
        return Err(Error::Unsupported(format!(
            "jpegxl muxer: exactly one stream is written, {} given",
            streams.len()
        )));
    };
    if stream.params.media_type != MediaType::Video
        || stream.params.codec_id.as_str() != CODEC_ID_STR
    {
        return Err(Error::Unsupported(format!(
            "jpegxl muxer: stream codec {} is not {CODEC_ID_STR}",
            stream.params.codec_id
        )));
    }
    Ok(Box::new(JxlMuxer {
        out: Some(output),
        time_base: stream.time_base,
        packets: Vec::new(),
    }))
}

struct JxlMuxer {
    out: Option<Box<dyn WriteSeek>>,
    time_base: TimeBase,
    packets: Vec<Packet>,
}

impl Muxer for JxlMuxer {
    fn format_name(&self) -> &str {
        CONTAINER_NAME
    }

    fn write_header(&mut self) -> Result<()> {
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.out.is_none() {
            return Err(Error::invalid(
                "jpegxl muxer: write_packet after the trailer",
            ));
        }
        if packet.data.is_empty() {
            return Err(Error::invalid("jpegxl muxer: empty packet"));
        }
        self.packets.push(packet.clone());
        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        let Some(mut out) = self.out.take() else {
            return Ok(());
        };
        let bytes = match self.packets.len() {
            0 => return Err(Error::invalid("jpegxl muxer: no packet to write")),
            1 => std::mem::take(&mut self.packets[0].data),
            _ => splice_animation(&self.packets, self.time_base)?,
        };
        out.write_all(&bytes)?;
        out.flush()?;
        Ok(())
    }
}

/// One parsed still: its headers and the frame's TOC + section bytes.
#[derive(Clone)]
struct StillParts {
    spec: EncodeSpec,
    group_size_shift: u32,
    toc: Vec<u32>,
    sections: Vec<u8>,
    exif: Option<Vec<u8>>,
    xmp: Option<Vec<u8>>,
}

fn parse_still(bytes: &[u8]) -> Result<StillParts> {
    let opened = crate::open(bytes)?;
    let prelude = crate::read_prelude_at_level(&opened.codestream, opened.level())?;
    let md = &prelude.metadata;
    if md.have_animation {
        return Err(Error::Unsupported(
            "jpegxl muxer: a packet is already an animation; one still per packet is spliced"
                .into(),
        ));
    }
    if md.bit_depth.float_sample || md.xyb_encoded || md.orientation != 1 || md.have_preview {
        return Err(Error::Unsupported(
            "jpegxl muxer: only integer, non-XYB, upright stills without preview can be re-timed"
                .into(),
        ));
    }
    let alpha =
        match md.num_extra_channels {
            0 => false,
            1 if md
                .extra_channel_info
                .first()
                .is_some_and(|ec| ec.kind == ExtraChannelType::Alpha && ec.dim_shift == 0) =>
            {
                true
            }
            _ => return Err(Error::Unsupported(
                "jpegxl muxer: extra channels other than one full-size alpha cannot be re-timed"
                    .into(),
            )),
        };
    let grey = md.colour_encoding.colour_space == ColourSpace::Grey;
    let colour = match &prelude.icc {
        Some(icc) => ColourSpec::Icc(icc.clone()),
        None => ColourSpec::from_color_info(&crate::api::color_info(md)),
    };
    let params = crate::frame_params(&prelude);
    let slice = opened
        .codestream
        .get(prelude.frames_offset..)
        .ok_or_else(|| Error::invalid("jpegxl muxer: frame data start past codestream end"))?;
    let mut br = crate::bitreader::BitReader::new(slice);
    let (fh, toc) = crate::read_frame_header_and_toc(&mut br, &params, slice)?;
    let rf = &fh.restoration_filter;
    let canonical = fh.frame_type == FrameType::Regular
        && fh.encoding == Encoding::Modular
        && fh.flags == 0
        && !fh.do_ycbcr
        && fh.upsampling == 1
        && fh.ec_upsampling.iter().all(|&u| u == 1)
        && fh.passes.num_passes == 1
        && !fh.have_crop
        && fh.blending_info.mode == BlendMode::Replace
        && fh
            .ec_blending_info
            .iter()
            .all(|b| b.mode == BlendMode::Replace)
        && fh.is_last
        && fh.name.is_empty()
        && !rf.gab
        && rf.epf_iters == 0
        && rf.extensions.mask == 0
        && fh.extensions.mask == 0
        && !toc.permuted;
    if !canonical {
        return Err(Error::Unsupported(
            "jpegxl muxer: the frame is not a plain Modular still (kReplace, no crop / upsampling / \
             restoration filter / TOC permutation) and cannot be re-timed"
                .into(),
        ));
    }
    let data_start = br.bytes_consumed();
    let total: usize = toc.entries.iter().map(|&e| e as usize).sum();
    let sections = slice
        .get(data_start..data_start + total)
        .ok_or_else(|| Error::invalid("jpegxl muxer: TOC exceeds the frame data"))?
        .to_vec();
    let mut meta = crate::image::Metadata::new();
    crate::api::container_metadata(opened.file.as_ref(), &mut meta);
    Ok(StillParts {
        spec: EncodeSpec {
            width: prelude.size.width,
            height: prelude.size.height,
            bits: md.bit_depth.bits_per_sample,
            grey,
            alpha,
            colour,
            animation: None,
            quality: 100,
            group_size_shift: fh.group_size_shift,
        },
        group_size_shift: fh.group_size_shift,
        toc: toc.entries.clone(),
        sections,
        exif: meta.exif,
        xmp: meta.xmp,
    })
}

/// Durations in ticks: the packet's own `duration`, else the gap to
/// the next packet's `pts`, else the previous frame's duration (`1`
/// for a lone frame).
fn durations_in_ticks(packets: &[Packet]) -> Vec<u32> {
    let mut out = Vec::with_capacity(packets.len());
    for (i, p) in packets.iter().enumerate() {
        let d = match p.duration {
            Some(d) if d > 0 => d,
            _ => match (p.pts, packets.get(i + 1).and_then(|n| n.pts)) {
                (Some(a), Some(b)) if b > a => b - a,
                _ => out.last().copied().map(i64::from).unwrap_or(1),
            },
        };
        out.push(d.clamp(1, i64::from(u32::MAX)) as u32);
    }
    out
}

/// Splice one still per packet into an animated codestream (boxed when
/// the first packet carried metadata boxes).
fn splice_animation(packets: &[Packet], time_base: TimeBase) -> Result<Vec<u8>> {
    let first = parse_still(&packets[0].data)?;
    let (num, den) = (time_base.0.num, time_base.0.den);
    let (tps_numerator, tps_denominator) =
        if num > 0 && den > 0 && den <= u32::MAX as i64 && num <= u32::MAX as i64 {
            // One tick is num / den seconds, so den / num ticks per second.
            (den as u32, num as u32)
        } else {
            (1000, 1)
        };
    let mut spec = first.spec.clone();
    spec.animation = Some(AnimationSpec {
        tps_numerator,
        tps_denominator,
        num_loops: 0,
    });
    let ticks = durations_in_ticks(packets);
    let mut w = BitWriter::new();
    crate::encoder::write_prelude(&mut w, &spec)?;
    let last = packets.len() - 1;
    for (i, pkt) in packets.iter().enumerate() {
        let parts = if i == 0 {
            first.clone()
        } else {
            parse_still(&pkt.data)?
        };
        let mut other = parts.spec.clone();
        other.animation = spec.animation;
        if other != spec {
            return Err(Error::Unsupported(format!(
                "jpegxl muxer: packet {i} ({}×{}, {} bits, grey={}, alpha={}) differs from packet 0 \
                 ({}×{}, {} bits, grey={}, alpha={}) — an animation has one geometry and layout",
                other.width,
                other.height,
                other.bits,
                other.grey,
                other.alpha,
                spec.width,
                spec.height,
                spec.bits,
                spec.grey,
                spec.alpha
            )));
        }
        let shape = FrameShape {
            group_size_shift: parts.group_size_shift,
            num_extra: u32::from(spec.alpha),
            have_animation: true,
            duration: ticks[i],
            is_last: i == last,
        };
        crate::encoder::write_frame_header_and_toc(&mut w, &shape, &parts.toc)?;
        w.write_bytes(&parts.sections);
    }
    let codestream = w.into_bytes();
    if first.exif.is_some() || first.xmp.is_some() {
        Ok(crate::encoder::wrap_container(
            &codestream,
            first.exif.as_deref(),
            first.xmp.as_deref(),
        )?)
    } else {
        Ok(codestream)
    }
}
