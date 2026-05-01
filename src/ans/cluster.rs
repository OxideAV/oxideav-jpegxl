//! Distribution clustering — FDIS Annex D.3.5 (Listing D.5, p. 65).
//!
//! Maps each of `num_distributions` non-clustered distributions to one
//! of `num_clusters` shared distributions. Two encodings are supported
//! by the bitstream:
//!
//! 1. **Simple** — one fixed-width field per distribution.
//! 2. **General** — each distribution's index is itself decoded from a
//!    one-distribution ANS stream (D.3.6 + D.3.4), optionally followed
//!    by an inverse move-to-front transform (Listing D.5).
//!
//! Round 1 shipped the *simple* path + MTF helper; round 2 adds the
//! *general* path, which reads each cluster index via a one-distribution
//! ANS sub-stream (`is_simple == 0`). The sub-stream's own clustering
//! step is skipped per D.3.1's "unless only one distribution is to be
//! decoded" clause, leaving exactly one
//! ANS distribution + one [`crate::ans::hybrid::HybridUintConfig`]
//! to drive [`HybridUintState::decode`] across `num_distributions`
//! integer reads.

use oxideav_core::{Error, Result};

use crate::ans::alias::AliasTable;
use crate::ans::distribution::read_distribution;
use crate::ans::hybrid::{HybridUintState, Lz77Params};
use crate::ans::hybrid_config::HybridUintConfig;
use crate::ans::symbol::AnsDecoder;
use crate::bitreader::BitReader;

/// `MTF(v[256], index)` per FDIS Listing D.5.
///
/// Pulls `v[index]` to the front of `v`, shifting positions
/// `0..index` down by one.
pub fn mtf(v: &mut [u32; 256], index: usize) {
    if index == 0 || index >= 256 {
        return;
    }
    let value = v[index];
    for i in (1..=index).rev() {
        v[i] = v[i - 1];
    }
    v[0] = value;
}

/// `InverseMoveToFrontTransform(clusters)` per FDIS Listing D.5.
pub fn inverse_mtf(clusters: &mut [u32]) {
    let mut v: [u32; 256] = [0; 256];
    for (i, slot) in v.iter_mut().enumerate() {
        *slot = i as u32;
    }
    for slot in clusters.iter_mut() {
        let index = *slot as usize;
        if index >= 256 {
            // Spec assumes indices fit in [0, 256). Out of range is
            // a malformed bitstream; we leave the value alone in
            // release builds and let downstream validation catch it.
            continue;
        }
        *slot = v[index];
        if index != 0 {
            mtf(&mut v, index);
        }
    }
}

/// Read a *simple* clustering map (the `is_simple == 1` branch of
/// D.3.5). Returns the `num_distributions`-element cluster index array.
///
/// The caller has already gated on `num_distributions > 1`; for
/// `num_distributions == 1` the spec says skip D.3.5 entirely.
pub fn read_simple_clustering(
    br: &mut BitReader<'_>,
    num_distributions: usize,
) -> Result<Vec<u32>> {
    if num_distributions == 0 {
        return Ok(Vec::new());
    }
    if num_distributions > super::distribution::ALPHABET_SIZE_MAX {
        return Err(Error::InvalidData(
            "JXL clustering: num_distributions absurdly large".into(),
        ));
    }
    let nbits = br.read_bits(2)?;
    let mut clusters = Vec::with_capacity(num_distributions);
    for _ in 0..num_distributions {
        let v = br.read_bits(nbits)?;
        clusters.push(v);
    }
    Ok(clusters)
}

/// Compute `num_clusters` (= 1 + max value in the cluster array, or 0
/// for empty input). Used by the caller to decide how many ANS
/// histograms to read after the clustering map.
pub fn num_clusters(clusters: &[u32]) -> u32 {
    clusters.iter().copied().max().map(|m| m + 1).unwrap_or(0)
}

/// Read the general clustering map (the `is_simple == 0` branch of
/// D.3.5).
///
/// Per D.3.5, each cluster index is read via a one-distribution ANS
/// sub-stream:
///
/// * one `HybridUintConfig` (D.3.7),
/// * one ANS distribution (D.3.4),
/// * one alias table (D.3.2),
/// * one `AnsDecoder` state init (D.3.3),
/// * `num_distributions` calls to `DecodeHybridVarLenUint` (D.3.6).
///
/// `use_mtf` toggles the inverse-MTF post-pass.
///
/// LZ77 is **not enabled** in the cluster sub-stream: the spec doesn't
/// allow LZ77 in a one-distribution stream because there is no
/// distinct "distance" context. The implementation therefore feeds
/// `Lz77Params { enabled: false, .. }` to `HybridUintState::new`.
pub fn read_general_clustering(
    br: &mut BitReader<'_>,
    num_distributions: usize,
) -> Result<Vec<u32>> {
    if num_distributions == 0 {
        return Ok(Vec::new());
    }
    if num_distributions > super::distribution::ALPHABET_SIZE_MAX {
        return Err(Error::InvalidData(
            "JXL clustering: num_distributions absurdly large".into(),
        ));
    }
    if num_distributions > br.bits_remaining() {
        // Each cluster-index decode reads at least one bit; refuse if
        // the input could not even supply trivial reads.
        return Err(Error::InvalidData(
            "JXL clustering: num_distributions exceeds remaining input".into(),
        ));
    }

    let use_mtf = br.read_bit()? == 1;

    // D.3.5 sub-stream is a one-distribution ANS stream. Per D.3.1, the
    // sub-stream itself begins with LZ77Params; if that signals
    // lz77.enabled then num_dist becomes 2 and D.3.5 would be invoked
    // recursively — a hostile-input attack vector. Reject the recursive
    // case rather than risk an unbounded recursion.
    let lz77_enabled = br.read_bit()? == 1;
    if lz77_enabled {
        return Err(Error::InvalidData(
            "JXL D.3.5 general clustering: LZ77-enabled sub-stream not supported (recursive clustering disallowed)".into(),
        ));
    }

    // num_dist == 1 → D.3.5 is skipped for the sub-stream. Proceed
    // straight to use_prefix_code + HybridUintConfig + distribution.
    let use_prefix_code = br.read_bit()? == 1;
    let log_alphabet_size = if use_prefix_code {
        5 + br.read_bits(2)?
    } else {
        15
    };

    let cfg = HybridUintConfig::read(br, log_alphabet_size)?;

    if use_prefix_code {
        // Sub-stream uses prefix codes (D.2). For the cluster-index
        // case in practice we don't see this branch on real codestreams
        // — the spec permits it but it requires a full prefix-code
        // histogram on top of the HybridUintConfig. Defer to a
        // follow-up round: error out cleanly so callers don't get
        // silently wrong data.
        return Err(Error::Unsupported(
            "JXL D.3.5 general clustering: prefix-coded sub-stream not yet supported".into(),
        ));
    }

    // ANS sub-stream: read one distribution, build alias table, init
    // state, decode num_distributions integers.
    let dist = read_distribution(br, log_alphabet_size)?;
    let alias = AliasTable::build(&dist, log_alphabet_size)?;
    let mut ans = AnsDecoder::new(br)?;
    let mut state = HybridUintState::new(
        Lz77Params {
            enabled: false,
            min_symbol: 224,
            min_length: 3,
        },
        cfg,
    );

    let mut clusters = Vec::with_capacity(num_distributions);
    for _ in 0..num_distributions {
        let value = state.decode(
            br,
            0,
            0,
            0,
            |br_inner, _ctx| Ok(ans.decode_symbol(br_inner, &dist, &alias)? as u32),
            |_ctx| cfg,
        )?;
        clusters.push(value);
    }

    if use_mtf {
        inverse_mtf(&mut clusters);
    }

    // Per D.3.5: "All integers in [0, num_clusters) are present in this
    // array." We don't enforce surjectivity here (it would force a
    // double pass over potentially large arrays); the caller relies on
    // num_clusters() returning max+1 which is correct for any cluster
    // map regardless of whether intermediate values are skipped.
    Ok(clusters)
}

/// Top-level entry point for D.3.5 clustering map reading. Dispatches
/// between the simple (`is_simple == 1`) and general paths.
///
/// `num_distributions == 1` skips D.3.5 entirely per D.3.1, and the
/// caller should not invoke this routine in that case.
pub fn read_clustering(br: &mut BitReader<'_>, num_distributions: usize) -> Result<Vec<u32>> {
    if num_distributions <= 1 {
        return Err(Error::InvalidData(
            "JXL clustering: read_clustering called with num_distributions <= 1 (caller must skip D.3.5)".into(),
        ));
    }
    let is_simple = br.read_bit()? == 1;
    if is_simple {
        read_simple_clustering(br, num_distributions)
    } else {
        read_general_clustering(br, num_distributions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ans::test_helpers::pack_lsb;

    #[test]
    fn mtf_zero_index_is_noop() {
        let mut v: [u32; 256] = [0; 256];
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = i as u32;
        }
        let copy = v;
        mtf(&mut v, 0);
        assert_eq!(v, copy);
    }

    #[test]
    fn mtf_pulls_value_to_front() {
        let mut v: [u32; 256] = [0; 256];
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = i as u32;
        }
        mtf(&mut v, 5);
        assert_eq!(v[0], 5);
        assert_eq!(v[1], 0);
        assert_eq!(v[2], 1);
        assert_eq!(v[3], 2);
        assert_eq!(v[4], 3);
        assert_eq!(v[5], 4);
        assert_eq!(v[6], 6);
    }

    #[test]
    fn inverse_mtf_round_trip_against_naive_mtf_encoder() {
        // Encode an arbitrary cluster sequence with the forward MTF
        // (which the spec doesn't show explicitly, but is well-defined:
        // for each output value, find its current position in the alphabet
        // table, emit that position, then move-to-front).
        let original: [u32; 5] = [3, 1, 4, 1, 5];
        let mut alphabet: [u32; 256] = [0; 256];
        for (i, slot) in alphabet.iter_mut().enumerate() {
            *slot = i as u32;
        }
        let mut encoded: Vec<u32> = Vec::with_capacity(original.len());
        for &value in original.iter() {
            let pos = alphabet
                .iter()
                .position(|&x| x == value)
                .expect("alphabet missing value");
            encoded.push(pos as u32);
            if pos != 0 {
                mtf(&mut alphabet, pos);
            }
        }
        // Inverse must recover the original.
        let mut decoded = encoded.clone();
        inverse_mtf(&mut decoded);
        assert_eq!(&decoded[..], &original[..]);
    }

    #[test]
    fn read_simple_clustering_two_bit_field() {
        // nbits = 2 (binary 10 → bit0=0, bit1=1). 3 distributions, then
        // values 1, 2, 3 each as u(2).
        // bit0=0, bit1=1,            // nbits = 2
        // u(2) = 1 → bits 1,0
        // u(2) = 2 → bits 0,1
        // u(2) = 3 → bits 1,1
        let bytes = pack_lsb(&[(2, 2), (1, 2), (2, 2), (3, 2)]);
        let mut br = BitReader::new(&bytes);
        let clusters = read_simple_clustering(&mut br, 3).unwrap();
        assert_eq!(clusters, vec![1, 2, 3]);
        assert_eq!(num_clusters(&clusters), 4);
    }

    #[test]
    fn read_simple_clustering_zero_distributions() {
        let bytes = vec![0u8; 1];
        let mut br = BitReader::new(&bytes);
        let clusters = read_simple_clustering(&mut br, 0).unwrap();
        assert!(clusters.is_empty());
        assert_eq!(num_clusters(&clusters), 0);
    }
}
