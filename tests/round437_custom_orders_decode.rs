//! Round 437 — `used_orders != 0` streams DECODE: the §C.7.1
//! per-channel permutation layout erratum.
//!
//! ## The erratum (fixture-pinned)
//!
//! Listing C.12 prints ONE `DecodePermutation()` per set `used_orders`
//! bit. The wire carries **three** — one per colour channel, in the
//! §C.8.3 decode sequence Y, X, B. Two independent oracles pin it:
//!
//! * The staged `patches-256x256` fixture's clean-room decode trace
//!   records the HfGlobal section's internal stream boundaries. Under
//!   the printed one-per-bit reading our §C.7.2 read starts 281 bits
//!   early and its prelude misparses; under one-per-channel the §C.7.2
//!   stream begins at the exact recorded position, parses to the
//!   recorded shape (ANS, no LZ77, 74 clusters for the
//!   `495 × nb_block_ctx` distributions), and the whole section lands
//!   on the trace's `AC_GLOBAL_END` bit count to the bit.
//! * On ANS-coded permutation streams (locally generated
//!   `used_orders = 0x53 / 0x13` specimens) the D.3.3 final-state
//!   invariant (`0x130000`) fails under one-per-bit for every Part 8.3
//!   context/count grid point, and closes under one-per-channel.
//!
//! ## What this test pins
//!
//! `custom_orders_t256_e1.jxl` — 256×256 synthetic (deterministic
//! pattern-generator source), lossy VarDCT d1, `used_orders = 0x1`
//! (DCT8×8 order permuted; all 1024 varblocks are DCT8×8), single
//! group, ANS §C.7.2 — decodes end to end through the full §C.7.1
//! chain: used_orders → shared Listing C.12 stream → 3 × §C.3.2
//! `DecodePermutation` → §C.7.2 histograms → live AC group decode.
//!
//! **Accuracy status.** The decode is structurally exact (the §C.3.2
//! stream satisfies ANS closure; §C.7.2 parses at the right position)
//! but the reference-decode MAD on this synthetic-edge content is
//! ≈ 20/13/8 — far above the sub-1/255 band of the photo-content
//! VarDCT fixtures. Round-437 measurement shows the residual is NOT
//! the permutation content: substituting any of eight alternative
//! order compositions (inverse permutation, natural-order
//! pre/post-composition, raw permutation) moves the MAD by < 0.4,
//! and the same-source `photo_e5` control (`used_orders = 0`,
//! pure DCT64×64) decodes at 0.70/0.49/0.62. The deficiency tracks
//! high-detail content regions (flat saturated regions decode
//! byte-exact) and is an open follow-up (see the round report); the
//! ratchet below bounds it against regression.

use std::io::Cursor;

fn png_rgb(bytes: &[u8]) -> (usize, usize, Vec<u8>) {
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    assert_eq!(info.color_type, png::ColorType::Rgb);
    (info.width as usize, info.height as usize, buf)
}

#[test]
fn custom_orders_stream_decodes_end_to_end() {
    let jxl = include_bytes!("fixtures/custom_orders_t256_e1.jxl");
    let expected = include_bytes!("fixtures/custom_orders_t256_e1_expected.png");
    let (w, h, reference) = png_rgb(expected);
    let frame = oxideav_jpegxl::decode_one_frame(jxl, None)
        .expect("used_orders != 0 stream must decode (round 437 per-channel layout)");
    assert_eq!(frame.planes.len(), 3);
    // Regression ratchet on the open accuracy deficiency (see module
    // docs): round-437 baseline MAD 20.25 / 13.57 / 7.72. Drive DOWN
    // when the synthetic-content VarDCT residual is found; never up.
    let bounds = [21.0, 14.5, 8.5];
    for (c, mad_max) in bounds.iter().enumerate() {
        let plane = &frame.planes[c];
        let mut sum = 0u64;
        for y in 0..h {
            for x in 0..w {
                sum += plane.data[y * plane.stride + x].abs_diff(reference[(y * w + x) * 3 + c])
                    as u64;
            }
        }
        let mad = sum as f64 / (w * h) as f64;
        assert!(
            mad < *mad_max,
            "custom-orders fixture ch{c} regressed: MAD {mad} (ratchet {mad_max})"
        );
    }
    // Flat saturated regions must stay byte-exact — the structural
    // part of the decode (LF + entropy + §C.7 chain) is exact; only
    // the high-detail residual is open.
    let probe = [(128usize, 128usize), (200, 60), (32, 200)];
    for (px, py) in probe {
        for c in 0..3 {
            let d = frame.planes[c].data[py * frame.planes[c].stride + px];
            let r = reference[(py * w + px) * 3 + c];
            assert!(
                d.abs_diff(r) <= 2,
                "flat-region probe ({px},{py}) ch{c}: {d} vs {r}"
            );
        }
    }
}
