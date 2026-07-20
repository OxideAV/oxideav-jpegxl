//! Round 420 — coded-domain Squeeze residual oracle.
//!
//! Reconstructs the exact coded-domain channel pyramid (averages +
//! residuals) by running the FORWARD Squeeze transform over the
//! reference decode, then compares it sample-by-sample against the
//! decoder's fully-assembled pre-inverse Modular image (captured via
//! `MODULAR_PRE_INVERSE_CAPTURE`). This pins the multi-group decode of
//! every channel — GlobalModular-decoded tops of the pyramid AND the
//! per-LfGroup / per-PassGroup residual slices — in the coded domain,
//! where a divergence points at the exact channel/sample instead of a
//! smeared pixel neighbourhood after the inverse.
//!
//! The forward transform here is derived purely from the inverse
//! listings (Annex H.6.2 / Listing I.20-I.21 + the tendency errata):
//! the inverse maps `(avg, residu)` -> `(first, second)` with
//! `diff = residu + tendency(left, avg, next_avg)`,
//! `first = (2*avg + diff - sign(diff)*(diff&1)) >> 1`,
//! `second = first - diff`; solving for the unique integer preimage
//! gives `diff = first - second` and
//! `avg = (first+second) >> 1` rounded TOWARD `first` when `diff` is
//! odd. A round-trip self-check through the crate's own
//! `horiz_isqueeze` / `vert_isqueeze` guards the derivation.
//!
//! Because each inverse step is a bijection, `forward(reference
//! decode) == the coded pyramid` holds EXACTLY for any stream whose
//! output is the raw inverse-transform result — i.e. lossless or
//! lossy alike, PROVIDED no out-of-gamut clamping happened at the
//! output quantiser and no §J restoration filters ran afterwards.
//! The committed assertions therefore run on lossless
//! unfiltered fixtures only. (The lossy + gab/EPF
//! `grayscale_public_university` stream is instead entropy-verified:
//! all 87 of its modular sub-bitstreams end on the D.3.3 ANS
//! final-state invariant, pinning the pyramid decode in sync; its
//! output accuracy is ratcheted in `round408_squeeze_multilf`.)
//!
//! Round-420 catches pinned by these oracles:
//! * Listing I.21 tendency: exact negative half-ties (`4A - 3C - B ≡
//!   6 mod 12`, ascending branch) round HALF-AWAY-FROM-ZERO — the
//!   round-408 biased-floor reading was off by one exactly there,
//!   which WAS the whole "multi-group Squeeze residual tail".
//! * D.4.2 MA-tree size: real encoder output exceeds the old
//!   1024-node working cap (the 2880×320 weighted-predictor stream
//!   signals a larger global tree); the cap now matches the spec's
//!   `tree.size() <= (1 << 26)`.
//! * Listing I.18 in-place inverse pairing (`r` constant through the
//!   c-loop) — regression-guarded here, pinned end-to-end by the
//!   3-channel XYB layouts the pairing fix unblocked.

use std::io::Cursor;

use oxideav_jpegxl::modular_fdis::{
    horiz_isqueeze, squeeze_tendency_pub, vert_isqueeze, ChannelDesc, TransformId,
};

fn png_grey(bytes: &[u8]) -> (usize, usize, Vec<u8>) {
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    assert_eq!(info.color_type, png::ColorType::Grayscale);
    (info.width as usize, info.height as usize, buf)
}

/// avg for a coded pair, rounded toward `first` when the diff is odd
/// (the unique integer preimage of the inverse listing).
fn pair_avg(first: i32, second: i32) -> i32 {
    let s = first as i64 + second as i64;
    let diff = first as i64 - second as i64;
    let a = if diff & 1 == 0 {
        s >> 1
    } else if diff > 0 {
        (s + 1) >> 1
    } else {
        (s - 1) >> 1
    };
    a as i32
}

/// Per-residual-sample tendency inputs recorded by the forward pass:
/// `(a=left/top, b=avg, c=next_avg, diff)`.
type TendencyLog = Vec<(i32, i32, i32, i32)>;

/// Forward horizontal Squeeze of one channel: returns (avg, residu,
/// per-residual tendency-input log) with widths (ceil(w/2), floor(w/2)).
fn horiz_fsqueeze(data: &[i32], w: usize, h: usize) -> (Vec<i32>, Vec<i32>, TendencyLog) {
    let w1 = w.div_ceil(2);
    let w2 = w / 2;
    let mut avg = vec![0i32; w1 * h];
    let mut res = vec![0i32; w2 * h];
    let mut log: TendencyLog = vec![(0, 0, 0, 0); w2 * h];
    for y in 0..h {
        for x in 0..w2 {
            avg[y * w1 + x] = pair_avg(data[y * w + 2 * x], data[y * w + 2 * x + 1]);
        }
        if w1 > w2 {
            avg[y * w1 + w2] = data[y * w + w - 1];
        }
        for x in 0..w2 {
            let a = avg[y * w1 + x];
            let diff = data[y * w + 2 * x].wrapping_sub(data[y * w + 2 * x + 1]);
            let next_avg = if x + 1 < w1 { avg[y * w1 + x + 1] } else { a };
            let left = if x > 0 { data[y * w + 2 * x - 1] } else { a };
            res[y * w2 + x] = diff.wrapping_sub(squeeze_tendency_pub(left, a, next_avg));
            log[y * w2 + x] = (left, a, next_avg, diff);
        }
    }
    (avg, res, log)
}

/// Forward vertical Squeeze of one channel: returns (avg, residu,
/// per-residual tendency-input log) with heights (ceil(h/2), floor(h/2)).
fn vert_fsqueeze(data: &[i32], w: usize, h: usize) -> (Vec<i32>, Vec<i32>, TendencyLog) {
    let h1 = h.div_ceil(2);
    let h2 = h / 2;
    let mut avg = vec![0i32; w * h1];
    let mut res = vec![0i32; w * h2];
    let mut log: TendencyLog = vec![(0, 0, 0, 0); w * h2];
    for x in 0..w {
        for y in 0..h2 {
            avg[y * w + x] = pair_avg(data[2 * y * w + x], data[(2 * y + 1) * w + x]);
        }
        if h1 > h2 {
            avg[h2 * w + x] = data[(h - 1) * w + x];
        }
        for y in 0..h2 {
            let a = avg[y * w + x];
            let diff = data[2 * y * w + x].wrapping_sub(data[(2 * y + 1) * w + x]);
            let next_avg = if y + 1 < h1 { avg[(y + 1) * w + x] } else { a };
            let top = if y > 0 { data[(2 * y - 1) * w + x] } else { a };
            res[y * w + x] = diff.wrapping_sub(squeeze_tendency_pub(top, a, next_avg));
            log[y * w + x] = (top, a, next_avg, diff);
        }
    }
    (avg, res, log)
}

/// Build the ground-truth coded-domain channel list by applying the
/// captured transform sequence FORWARD over the reference image.
fn forward_pyramid(
    reference: Vec<i32>,
    w: usize,
    h: usize,
    transforms: &[oxideav_jpegxl::modular_fdis::TransformInfo],
) -> (Vec<ChannelDesc>, Vec<Vec<i32>>, Vec<TendencyLog>) {
    let mut descs = vec![ChannelDesc {
        width: w as u32,
        height: h as u32,
        hshift: 0,
        vshift: 0,
    }];
    let mut chans = vec![reference];
    let mut logs: Vec<TendencyLog> = vec![Vec::new()];
    for t in transforms {
        assert_eq!(
            t.tr,
            TransformId::Squeeze,
            "oracle only handles Squeeze transform chains"
        );
        for sp in &t.squeeze_params {
            let begin = sp.begin_c as usize;
            let end = begin + sp.num_c as usize - 1;
            let r_base = if sp.in_place { end + 1 } else { chans.len() };
            for (k, c) in (begin..=end).enumerate() {
                let d = descs[c];
                let (cw, ch) = (d.width as usize, d.height as usize);
                let (avg, res, tlog, ad, rd) = if sp.horizontal {
                    let (avg, res, tlog) = horiz_fsqueeze(&chans[c], cw, ch);
                    (
                        avg,
                        res,
                        tlog,
                        ChannelDesc {
                            width: cw.div_ceil(2) as u32,
                            height: ch as u32,
                            hshift: d.hshift + 1,
                            vshift: d.vshift,
                        },
                        ChannelDesc {
                            width: (cw / 2) as u32,
                            height: ch as u32,
                            hshift: d.hshift + 1,
                            vshift: d.vshift,
                        },
                    )
                } else {
                    let (avg, res, tlog) = vert_fsqueeze(&chans[c], cw, ch);
                    (
                        avg,
                        res,
                        tlog,
                        ChannelDesc {
                            width: cw as u32,
                            height: ch.div_ceil(2) as u32,
                            hshift: d.hshift,
                            vshift: d.vshift + 1,
                        },
                        ChannelDesc {
                            width: cw as u32,
                            height: (ch / 2) as u32,
                            hshift: d.hshift,
                            vshift: d.vshift + 1,
                        },
                    )
                };
                // Round-trip self-check through the crate's inverse.
                let (merged, _) = if sp.horizontal {
                    horiz_isqueeze(&avg, ad.width, &res, rd.width, ad.height).unwrap()
                } else {
                    vert_isqueeze(&avg, ad.height, &res, rd.height, ad.width).unwrap()
                };
                assert_eq!(
                    merged, chans[c],
                    "forward/inverse Squeeze round-trip failed (channel {c})"
                );
                chans[c] = avg;
                descs[c] = ad;
                logs[c] = Vec::new();
                let insert_at = if sp.in_place { r_base + k } else { chans.len() };
                chans.insert(insert_at, res);
                descs.insert(insert_at, rd);
                logs.insert(insert_at, tlog);
            }
        }
    }
    (descs, chans, logs)
}

/// True when the reference pixels feeding pyramid sample (x, y) of a
/// channel at (hshift, vshift) — plus a guard margin — contain any
/// 0/255 value (a potential PNG clamp, which poisons the forward
/// oracle for lossy streams).
fn clamp_tainted(
    reference: &[u8],
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    hshift: i32,
    vshift: i32,
) -> bool {
    let sx = 1usize << hshift.clamp(0, 30);
    let sy = 1usize << vshift.clamp(0, 30);
    let x0 = (x * sx).saturating_sub(4 * sx);
    let y0 = (y * sy).saturating_sub(4 * sy);
    let x1 = ((x + 5) * sx).min(w);
    let y1 = ((y + 5) * sy).min(h);
    for yy in y0..y1 {
        for xx in x0..x1 {
            let v = reference[yy * w + xx];
            if v == 0 || v == 255 {
                return true;
            }
        }
    }
    false
}

fn run_oracle(jxl: &[u8], expected_png: &[u8], group_dim: u32) -> usize {
    oxideav_jpegxl::set_modular_pre_inverse_capture_armed(true);
    let frame = oxideav_jpegxl::decode_one_frame(jxl, None).expect("decode");
    oxideav_jpegxl::set_modular_pre_inverse_capture_armed(false);
    let captured = oxideav_jpegxl::MODULAR_PRE_INVERSE_CAPTURE
        .with(|s| s.borrow_mut().take())
        .expect("pre-inverse capture populated");
    let _ = &frame;
    let (w, h, reference) = png_grey(expected_png);
    let ref_i32: Vec<i32> = reference.iter().map(|&b| b as i32).collect();
    let (descs, chans, transforms) = captured;
    let (want_descs, want_chans, tlogs) = forward_pyramid(ref_i32, w, h, &transforms);
    assert_eq!(descs.len(), want_descs.len(), "channel count mismatch");
    let mut total_bad = 0usize;
    for (i, (d, wd)) in descs.iter().zip(want_descs.iter()).enumerate() {
        assert_eq!(
            (d.width, d.height, d.hshift, d.vshift),
            (wd.width, wd.height, wd.hshift, wd.vshift),
            "channel {i} desc mismatch"
        );
        let got = &chans[i];
        let want = &want_chans[i];
        let mut bad = 0usize;
        let mut bad_clean = 0usize;
        let mut first: Vec<(usize, usize, i32, i32)> = Vec::new();
        for y in 0..d.height as usize {
            for x in 0..d.width as usize {
                let g = got[y * d.width as usize + x];
                let t = want[y * d.width as usize + x];
                if g != t {
                    bad += 1;
                    let tainted = clamp_tainted(&reference, w, h, x, y, d.hshift, d.vshift);
                    if !tainted {
                        bad_clean += 1;
                        if first.len() < 12 {
                            first.push((x, y, g, t));
                        }
                        // Optional CSV dump of clean tendency ground
                        // truth (decode is ANS-invariant-verified, so
                        // `diff - got` is the encoder's tendency).
                        if let Ok(path) = std::env::var("R420_TENDENCY_CSV") {
                            if let Some((ta, tb, tc, tdiff)) =
                                tlogs[i].get(y * d.width as usize + x).copied()
                            {
                                use std::io::Write;
                                let mut f = std::fs::OpenOptions::new()
                                    .create(true)
                                    .append(true)
                                    .open(path)
                                    .unwrap();
                                writeln!(
                                    f,
                                    "{i},{x},{y},{ta},{tb},{tc},{tdiff},{},{g},{t}",
                                    tdiff.wrapping_sub(g)
                                )
                                .unwrap();
                            }
                        }
                    }
                }
            }
        }
        if bad > 0 {
            eprintln!("channel {i}: {bad_clean} clamp-free of {bad} divergent");
        }
        if bad > 0 {
            let hs = d.hshift.max(0) as u32;
            let vs = d.vshift.max(0) as u32;
            let gx = (group_dim >> hs).max(1);
            let gy = (group_dim >> vs).max(1);
            eprintln!(
                "channel {i} ({}x{} shift {},{}): {bad} divergent samples \
                 (group grid {}x{} samples/group)",
                d.width, d.height, d.hshift, d.vshift, gx, gy
            );
            for (x, y, g, t) in &first {
                // Tendency-input forensics: if the entropy decode is in
                // sync and only the tendency rounding diverges, the
                // decoder's residual implies tendency_true = diff - got.
                let tinfo = tlogs[i]
                    .get(y * d.width as usize + x)
                    .copied()
                    .unwrap_or((0, 0, 0, 0));
                let (ta, tb, tc, tdiff) = tinfo;
                let t_mine = squeeze_tendency_pub(ta, tb, tc);
                let t_true = tdiff.wrapping_sub(*g);
                eprintln!(
                    "  ({x:4},{y:4}) group ({},{}) in-group ({:3},{:3}): got {g} want {t} \
                     (delta {}) | a={ta} b={tb} c={tc} diff={tdiff} t_mine={t_mine} t_true={t_true}",
                    *x as u32 / gx,
                    *y as u32 / gy,
                    *x as u32 % gx,
                    *y as u32 % gy,
                    g - t
                );
            }
        }
        total_bad += bad;
    }
    total_bad
}

/// Env-driven variant for local bisects: R420_JXL / R420_PNG point at
/// an externally staged (fixture, reference-decode) pair. No-op when
/// the env vars are absent so CI never depends on local files.
#[test]
fn external_coded_domain_oracle() {
    let (Ok(jxl_path), Ok(png_path)) = (std::env::var("R420_JXL"), std::env::var("R420_PNG"))
    else {
        return;
    };
    let jxl = std::fs::read(jxl_path).unwrap();
    let png = std::fs::read(png_path).unwrap();
    let bad = run_oracle(&jxl, &png, 256);
    eprintln!("external fixture total divergent coded-domain samples: {bad}");
    assert_eq!(bad, 0, "coded-domain divergence: {bad} samples");
}

/// Env-driven output-domain divergence-geometry probe (local bisects
/// only; no-op without R420_GEO_JXL / R420_GEO_PNG).
#[test]
fn external_output_geometry_probe() {
    let (Ok(jxl_path), Ok(png_path)) =
        (std::env::var("R420_GEO_JXL"), std::env::var("R420_GEO_PNG"))
    else {
        return;
    };
    let jxl = std::fs::read(jxl_path).unwrap();
    let png = std::fs::read(png_path).unwrap();
    let headers = oxideav_jpegxl::probe_fdis(&jxl).expect("probe");
    eprintln!(
        "probe: {}x{} bps={} float={} xyb={} colour_space={:?} extra={}",
        headers.size.width,
        headers.size.height,
        headers.metadata.bit_depth.bits_per_sample,
        headers.metadata.bit_depth.float_sample,
        headers.metadata.xyb_encoded,
        headers.metadata.colour_encoding.colour_space,
        headers.metadata.num_extra_channels,
    );
    oxideav_jpegxl::set_modular_pre_inverse_capture_armed(true);
    let decoded = oxideav_jpegxl::decode_one_frame(&jxl, None);
    oxideav_jpegxl::set_modular_pre_inverse_capture_armed(false);
    if let Some((descs, chans, transforms)) =
        oxideav_jpegxl::MODULAR_PRE_INVERSE_CAPTURE.with(|s| s.borrow_mut().take())
    {
        eprintln!("pre-inverse channel layout ({} channels):", descs.len());
        for (i, d) in descs.iter().enumerate() {
            eprintln!(
                "  ch{i}: {}x{} shift ({},{})",
                d.width, d.height, d.hshift, d.vshift
            );
        }
        for (ti, t) in transforms.iter().enumerate() {
            eprintln!("  transform {ti}: {:?}", t.tr);
            for (si, sp) in t.squeeze_params.iter().enumerate() {
                eprintln!(
                    "    step {si}: horizontal={} in_place={} begin_c={} num_c={}",
                    sp.horizontal, sp.in_place, sp.begin_c, sp.num_c
                );
            }
        }
        // Unclamped-range census: run the crate inverse over the
        // captured pyramid and count out-of-gamut samples. When ~0,
        // PNG clamping cannot explain a coded-domain oracle
        // divergence on this stream.
        let mut img = oxideav_jpegxl::modular_fdis::ModularImage {
            channels: chans,
            descs,
        };
        if oxideav_jpegxl::global_modular::apply_inverse_transforms(&mut img, &transforms, 8)
            .is_ok()
        {
            let mut lo = i32::MAX;
            let mut hi = i32::MIN;
            let mut oob = 0usize;
            for v in img.channels[0].iter() {
                lo = lo.min(*v);
                hi = hi.max(*v);
                if *v < 0 || *v > 255 {
                    oob += 1;
                }
            }
            eprintln!(
                "unclamped inverse of decoded pyramid: min {lo} max {hi}, {oob} out-of-gamut of {}",
                img.channels[0].len()
            );
        }
    }
    if let Some(info) =
        oxideav_jpegxl::MODULAR_PRE_INVERSE_CAPTURE_INFO.with(|s| s.borrow_mut().take())
    {
        eprintln!("{info}");
    }
    let frame = decoded.expect("decode");
    let (w, h, reference) = png_grey(&png);
    let plane = &frame.planes[0];
    let mut by_row: Vec<usize> = vec![0; h];
    let mut by_col: Vec<usize> = vec![0; w];
    let mut hist = std::collections::BTreeMap::<i32, usize>::new();
    let mut first: Vec<(usize, usize, u8, u8)> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let g = plane.data[y * plane.stride + x];
            let r = reference[y * w + x];
            if g != r {
                by_row[y] += 1;
                by_col[x] += 1;
                *hist.entry(g as i32 - r as i32).or_default() += 1;
                if first.len() < 20 {
                    first.push((x, y, g, r));
                }
            }
        }
    }
    let total: usize = by_row.iter().sum();
    eprintln!("geometry probe: {total} divergent pixels of {}", w * h);
    eprintln!("delta histogram: {hist:?}");
    let bad_rows: Vec<(usize, usize)> = by_row
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(i, &c)| (i, c))
        .take(30)
        .collect();
    eprintln!("first 30 rows with divergence: {bad_rows:?}");
    let col_left: usize = by_col[..w.min(2048)].iter().sum();
    let col_right: usize = by_col[w.min(2048)..].iter().sum();
    eprintln!("divergence left of x=2048: {col_left}, right: {col_right}");
    eprintln!("first divergent samples (x, y, got, want): {first:?}");
}

#[test]
fn sq_512_coded_domain_oracle() {
    let bad = run_oracle(
        include_bytes!("fixtures/sq_512.jxl"),
        include_bytes!("fixtures/sq_512_expected.png"),
        256,
    );
    assert_eq!(bad, 0, "sq_512 coded-domain divergence: {bad} samples");
}

/// 2880×320 lossless Squeeze, weighted predictor forced at encode
/// time: 2 LfGroups (so the shift-(3,3) residual channel decodes
/// through the §C.5.2 ModularLfGroup walk), 12 PassGroups, an
/// all-predictor-6 global MA tree larger than 1024 nodes, and WP
/// state exercised on every group-slice seam. Coded-domain exact ⟺
/// every per-group residual sample decodes bit-exactly.
#[test]
fn sq_2880x320_wp_multilfgroup_coded_domain_oracle() {
    let bad = run_oracle(
        include_bytes!("fixtures/sq_2880x320_wp.jxl"),
        include_bytes!("fixtures/sq_2880x320_wp_expected.png"),
        256,
    );
    assert_eq!(
        bad, 0,
        "sq_2880x320_wp coded-domain divergence: {bad} samples"
    );
}

/// Output-domain bit-exactness for the same stream (the oracle above
/// pins the coded domain; this pins the inverse-transform walk and
/// output quantisation on top).
#[test]
fn sq_2880x320_wp_multilfgroup_bit_exact() {
    let frame =
        oxideav_jpegxl::decode_one_frame(include_bytes!("fixtures/sq_2880x320_wp.jxl"), None)
            .expect("decode");
    let (w, h, reference) = png_grey(include_bytes!("fixtures/sq_2880x320_wp_expected.png"));
    let plane = &frame.planes[0];
    let mut max = 0u8;
    for y in 0..h {
        for x in 0..w {
            max = max.max(plane.data[y * plane.stride + x].abs_diff(reference[y * w + x]));
        }
    }
    assert_eq!(max, 0, "multi-LfGroup WP Squeeze must be bit-exact");
}
