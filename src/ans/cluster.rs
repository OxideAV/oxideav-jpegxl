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
//! Round-1 ships only the *simple* path and the MTF helper. The
//! general path needs a fully wired ANS decoder over a sub-distribution
//! and is decoded inline by future rounds (it requires
//! `DecodeHybridVarLenUint` from [`crate::ans::hybrid`] which expects a
//! prebuilt ANS context array — exactly what this routine constructs).
//! The MTF transform itself is exercised end-to-end by the unit tests
//! below so that round-2 only has to wire it up.

use oxideav_core::{Error, Result};

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
