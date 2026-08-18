//! Round 448 — ISO/IEC 18181-2:2024 container surface: the typed box
//! walk ([`oxideav_jpegxl::container::JxlFile`]) and the §9.11 JPEG
//! Bitstream Reconstruction Data parse
//! ([`oxideav_jpegxl::jpeg_bitstream::JpegBitstreamData`]), pinned on a
//! real lossless JPEG→JXL transcode (`tests/fixtures/jpeg_transcode.jxl`,
//! encoded from `jpeg_transcode_original.jpg`) whose original JPEG bytes
//! are committed alongside as the arbitration oracle.

use oxideav_jpegxl::container::JxlFile;
use oxideav_jpegxl::jpeg_bitstream::JpegBitstreamData;

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/tests/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

/// Walk the metadata segments of a JPEG (up to its first SOS) and return
/// the marker byte + whole-segment slices.
fn jpeg_segments(jpg: &[u8]) -> Vec<(u8, Vec<u8>)> {
    assert_eq!(&jpg[..2], &[0xFF, 0xD8], "fixture must start with SOI");
    let mut out = Vec::new();
    let mut pos = 2usize;
    while pos + 4 <= jpg.len() {
        assert_eq!(jpg[pos], 0xFF, "expected marker at {pos}");
        let m = jpg[pos + 1];
        if m == 0xDA {
            break;
        }
        let len = u16::from_be_bytes([jpg[pos + 2], jpg[pos + 3]]) as usize;
        out.push((m, jpg[pos..pos + 2 + len].to_vec()));
        pos += 2 + len;
    }
    out
}

#[test]
fn container_walk_of_real_transcode() {
    let jxl = fixture("jpeg_transcode.jxl");
    let f = JxlFile::parse(&jxl).unwrap();
    // No Level box in this file: the §9.3 default applies.
    assert_eq!(f.level, 5);
    // The codestream is carried in a single jxlc box and starts with the
    // raw-codestream signature.
    assert_eq!(&f.codestream[..2], &[0xFF, 0x0A]);
    assert!(f.frame_index.is_none());
    assert!(f.metadata.is_empty());
    assert!(f.jbrd.is_some(), "transcode carries reconstruction data");
}

#[test]
fn jbrd_bundle_matches_original_jpeg() {
    let jxl = fixture("jpeg_transcode.jxl");
    let jpg = fixture("jpeg_transcode_original.jpg");
    let f = JxlFile::parse(&jxl).unwrap();
    let d = JpegBitstreamData::parse(f.jbrd.unwrap()).unwrap();

    // The marker array reproduces the original marker sequence up to the
    // scan, then SOS + EOI.
    let segs = jpeg_segments(&jpg);
    let seg_markers: Vec<u8> = segs.iter().map(|(m, _)| *m).collect();
    let mut expected = seg_markers.clone();
    expected.extend_from_slice(&[0xDA, 0xD9]);
    assert_eq!(d.markers, expected);
    assert!(!d.is_grey);

    // The single APPn marker (JFIF APP0) is an unknown-type app marker
    // whose verbatim bytes (after the 0xFF) travel in the Brotli tail.
    assert_eq!(d.app_markers.len(), 1);
    assert_eq!(d.app_markers[0].kind, 0);
    let app0 = &segs[0].1;
    assert_eq!(
        d.trailing.app_data[0].as_deref(),
        Some(&app0[1..]),
        "app_data must be the APP0 segment bytes after its 0xFF byte"
    );

    // Component layout: YCbCr ids {1,2,3}, luma table 0, chroma table 1.
    assert_eq!(d.comp_type, 1);
    assert_eq!(d.component_ids, vec![1, 2, 3]);
    assert_eq!(d.component_q_idx, vec![0, 1, 1]);

    // Two 8-bit quant tables, each alone in its DQT segment.
    assert_eq!(d.quant_tables.len(), 2);
    for (i, qt) in d.quant_tables.iter().enumerate() {
        assert_eq!(qt.precision, 0);
        assert_eq!(qt.index, i as u32);
        assert!(qt.is_last);
    }

    // Four Huffman codes (DC0, AC0, DC1, AC1), each alone in its DHT
    // segment. Their symbol totals reproduce the original DHT payload
    // sizes once the A.3 sentinel drop (-1 symbol, last non-zero count
    // decremented) is applied: DHT segment length = 2 + 1 + 16 +
    // (sum(counts) - 1).
    assert_eq!(d.huffman_codes.len(), 4);
    let dht_segs: Vec<&Vec<u8>> = segs
        .iter()
        .filter(|(m, _)| *m == 0xC4)
        .map(|(_, s)| s)
        .collect();
    assert_eq!(dht_segs.len(), 4);
    let expect_sig = [(false, 0u32), (true, 0), (false, 1), (true, 1)];
    for (i, hc) in d.huffman_codes.iter().enumerate() {
        assert!(hc.is_last);
        assert_eq!((hc.is_ac, hc.id), expect_sig[i]);
        let total: u32 = hc.counts.iter().sum();
        assert_eq!(hc.values.len() as u32, total);
        // The sentinel (value 256) is the single symbol dropped on DHT
        // serialization.
        assert_eq!(
            hc.values.iter().filter(|&&v| v == 256).count(),
            1,
            "table {i} carries exactly one sentinel symbol"
        );
        let dht_payload_len = dht_segs[i].len() - 2; // minus FF C4
        assert_eq!(dht_payload_len, 2 + 1 + 16 + (total as usize - 1));
        // And the emitted L_i / V_j bytes match the original segment.
        let mut l = [0u8; 16];
        for (j, &c) in hc.counts[1..].iter().enumerate() {
            l[j] = c as u8;
        }
        let last_nz = hc.counts.iter().rposition(|&c| c != 0).unwrap();
        assert!(last_nz >= 1);
        l[last_nz - 1] -= 1;
        assert_eq!(&dht_segs[i][5..21], &l, "table {i} L_i bytes");
        let vals: Vec<u8> = hc
            .values
            .iter()
            .filter(|&&v| v != 256)
            .map(|&v| v as u8)
            .collect();
        assert_eq!(&dht_segs[i][21..], &vals[..], "table {i} V_j bytes");
    }

    // One baseline sequential scan over all three components.
    assert_eq!(d.scan_infos.len(), 1);
    let si = &d.scan_infos[0];
    assert_eq!((si.ss, si.se, si.al, si.ah), (0, 63, 0, 0));
    assert_eq!(si.components.len(), 3);

    // No DRI, no encoder quirks, no trailing garbage.
    assert_eq!(d.restart_interval, 0);
    assert!(d.scan_more_infos[0].reset_points.is_empty());
    assert!(d.scan_more_infos[0].extra_zero_runs.is_empty());
    assert!(d.intermarker_lengths.is_empty());
    assert_eq!(d.tail_data_length, 0);
    assert!(!d.has_padding);
    assert!(d.trailing.tail_data.is_empty());
}
