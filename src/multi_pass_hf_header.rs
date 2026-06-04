//! Per-LfGroup multi-pass driver with per-pass HF-header `hfp` reads —
//! ISO/IEC FDIS 18181-1:2021 §C.8.3 first paragraph.
//!
//! ## Scope (round 232)
//!
//! Round 232 lands the next §C.8.3 sub-procedure above the round-228
//! per-LfGroup multi-pass three-channel varblock decode driver
//! ([`crate::multi_pass_decode::decode_multi_pass_three_channels_with_resolver`])
//! — a typed primitive that for each pass `p ∈ [0, num_passes)`:
//!
//! 1. reads a [`crate::pass_group_hf::PassGroupHfHeader`] per the
//!    §C.8.3 first-paragraph prose
//!    `hfp = u(ceil(log2(num_hf_presets)))`,
//! 2. derives `histogram_offset = 495 × nb_block_ctx × hfp` (the
//!    spec's `offset` term in `D[NonZerosContext(...) + offset]` and
//!    `D[CoefficientContext(...) + offset]`),
//! 3. threads `(p, histogram_offset)` into the per-pass histogram-
//!    routing closures wrapping the round-228 inner driver.
//!
//! ## FDIS prose anchor
//!
//! §C.8.3, first paragraph (FDIS p. 55) reads:
//!
//! > The decoder read `hfp = u(ceil(log2(num_hf_presets)))`, which
//! > indicates the coefficient order to be used for this group as
//! > well as the offset in the histogram, which is given by
//! > `offset = 495 × nb_block_ctx × hfp`.
//!
//! The 495 factor is the per-block per-context histogram dimensioning
//! that §C.7.2 sizes (`495 × num_hf_presets × nb_block_ctx`); the per-
//! pass `hfp` selects which 495-aligned window the per-pass HF reads
//! consult.
//!
//! ## Scope boundary
//!
//! This driver is a pure-control-flow layer above the round-228
//! per-LfGroup multi-pass driver — no histogram materialisation, no
//! ANS state setup, no coefficient-order resolution. The per-pass
//! `histogram_offset` is exposed to the caller's closure as an
//! additive `u64` so the caller can route it into whichever
//! [`crate::modular_fdis::EntropyStream`] (or equivalent) holds the
//! `495 × num_hf_presets × nb_block_ctx` clustered distributions read
//! by the deferred §C.7.2 step (#799 DOCS-GAP for the §C.7.2 entropy
//! histogram bundle).
//!
//! Per-pass coefficient-order selection (`PassGroupHfHeader::select_pass`
//! against the per-pass [`crate::hf_pass::HfPass`] array) is a caller
//! concern — the round-232 driver exposes the per-pass `hfp` so the
//! caller can perform the lookup once per pass without re-reading
//! `hfp`.
//!
//! Same pure-control-flow primitive shape as round-89
//! [`crate::dct_quant_weights`], round-95 [`crate::hf_dequant`],
//! round-121 [`crate::llf_from_lf`], round-138
//! [`crate::chroma_from_luma`], round-141 [`crate::gaborish`],
//! round-144 [`crate::epf`], round-147 [`crate::afv::afv_idct`],
//! round-159 / 164 [`crate::pass_group_hf`], round-177
//! [`crate::non_zeros_grid`], round-183
//! [`crate::per_channel_non_zeros`], round-190
//! [`crate::per_pass_non_zeros`], round-208 [`crate::varblock_walk`],
//! round-214 [`crate::block_context_resolver`], round-221, and
//! round-228 [`crate::multi_pass_decode`].

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::block_context_resolver::BlockContextResolver;
use crate::dct_select::DctSelectGrid;
use crate::multi_pass_decode::{
    decode_multi_pass_three_channels_with_resolver, MultiPassThreeChannelOutput,
};
use crate::pass_group_hf::PassGroupHfHeader;
use crate::per_pass_non_zeros::PerPassNonZerosGrids;
use crate::varblock_walk::Varblock;

/// Typed per-LfGroup container of one [`PassGroupHfHeader`] per pass.
///
/// `headers[p]` is the §C.8.3 first-paragraph header read at the
/// start of pass `p`'s PassGroup payload. The container records the
/// per-pass `hfp` selector and the derived
/// `histogram_offset = 495 × nb_block_ctx × hfp` so the round-232
/// driver can route the offset into the per-pass histogram closures
/// without re-reading any bits.
///
/// Built via [`PerPassHfHeaders::read`] (consume `num_passes`
/// consecutive headers from a [`BitReader`]) or
/// [`PerPassHfHeaders::from_headers`] (construct from a pre-built
/// `Vec` for unit tests / callers that hand-wire the per-pass header
/// sequence).
#[derive(Debug, Clone)]
pub struct PerPassHfHeaders {
    /// Per-pass headers in pass order. `headers.len() == num_passes`.
    headers: Vec<PassGroupHfHeader>,
}

impl PerPassHfHeaders {
    /// Read `num_passes` consecutive [`PassGroupHfHeader`] values
    /// from `br`.
    ///
    /// Each per-pass read consumes
    /// `ceil(log2(num_hf_presets))` bits for `hfp` and derives the
    /// per-pass `histogram_offset = 495 × nb_block_ctx × hfp`.
    ///
    /// Returns [`Error::InvalidData`] when [`PassGroupHfHeader::read`]
    /// rejects (the per-pass invariant `hfp < num_hf_presets` is
    /// enforced inside the inner reader — a non-power-of-two
    /// `num_hf_presets` can otherwise allow an out-of-range `hfp`
    /// value to slip past the `u(nbits)` width).
    pub fn read(
        br: &mut BitReader<'_>,
        num_passes: u32,
        num_hf_presets: u32,
        nb_block_ctx: u32,
    ) -> Result<Self> {
        let mut headers = Vec::with_capacity(num_passes as usize);
        for _ in 0..num_passes {
            headers.push(PassGroupHfHeader::read(br, num_hf_presets, nb_block_ctx)?);
        }
        Ok(Self { headers })
    }

    /// Construct from a pre-built `Vec<PassGroupHfHeader>` — useful
    /// for unit tests that want to pin a per-pass header sequence
    /// without going through a [`BitReader`].
    pub fn from_headers(headers: Vec<PassGroupHfHeader>) -> Self {
        Self { headers }
    }

    /// Pass count = `headers.len()` (matches the caller's
    /// `num_passes` argument to [`Self::read`]).
    pub fn num_passes(&self) -> u32 {
        self.headers.len() as u32
    }

    /// Per-pass header lookup. Returns [`Error::InvalidData`] when
    /// `p >= num_passes`.
    pub fn get(&self, p: u32) -> Result<&PassGroupHfHeader> {
        self.headers.get(p as usize).ok_or_else(|| {
            Error::InvalidData(format!(
                "JXL multi_pass_hf_header: pass index {p} out of {} per-pass headers",
                self.headers.len()
            ))
        })
    }

    /// Per-pass `histogram_offset` lookup =
    /// `495 × nb_block_ctx × headers[p].hfp`.
    pub fn histogram_offset(&self, p: u32) -> Result<u64> {
        Ok(self.get(p)?.histogram_offset)
    }

    /// Per-pass `hfp` selector lookup. Provided as a convenience so
    /// callers can perform the per-pass [`crate::hf_pass::HfPass`]
    /// array lookup (`hfp` indexes the per-pass coefficient-order
    /// presets) without dereferencing the header.
    pub fn hfp(&self, p: u32) -> Result<u32> {
        Ok(self.get(p)?.hfp)
    }

    /// Borrow the underlying per-pass slice. Provided so callers can
    /// iterate per-pass headers without going through repeated
    /// [`Self::get`] calls — primarily for read-only diagnostics.
    pub fn as_slice(&self) -> &[PassGroupHfHeader] {
        &self.headers
    }
}

/// Per-LfGroup multi-pass three-channel varblock decode driver with
/// per-pass `histogram_offset` routing — round 232's outer-pass loop
/// above the round-228 driver.
///
/// Behaves exactly like
/// [`decode_multi_pass_three_channels_with_resolver`] (same per-pass
/// raster walk, same per-varblock per-channel sweep, same
/// [`PerPassNonZerosGrids`] writeback) — the only addition is that
/// the per-pass `histogram_offset` (looked up off `headers`) is
/// passed as the 4th argument to the `read_non_zeros` and
/// `decode_symbol` closures.
///
/// The closure signatures match the §C.8.3 prose
/// `D[NonZerosContext(predicted) + offset]` and
/// `D[CoefficientContext(...) + offset]` precisely — the caller
/// receives `(pass, channel, context, offset)` and returns the
/// decoded symbol. The driver does not interpret the offset; it
/// purely routes it through.
///
/// `num_passes` is taken from `headers.num_passes()` (the per-pass
/// header container is the authoritative pass-count source); it
/// must match `nz.num_passes()` (the per-pass per-channel non-zeros
/// container constructed by the caller). A mismatch returns
/// [`Error::InvalidData`].
///
/// On any per-pass error the driver propagates the error
/// immediately and discards in-flight partial output, matching the
/// round-228 outer-loop error semantics.
#[allow(clippy::too_many_arguments)]
pub fn decode_multi_pass_with_hf_headers<Q, F, G>(
    grid: &DctSelectGrid,
    headers: &PerPassHfHeaders,
    nz: &mut PerPassNonZerosGrids,
    resolver: &BlockContextResolver<'_>,
    mut qdc_at: Q,
    mut read_non_zeros: F,
    mut decode_symbol: G,
) -> Result<MultiPassThreeChannelOutput>
where
    Q: FnMut(u32, &Varblock) -> Result<[i32; 3]>,
    F: FnMut(u32, u32, u32, u64) -> Result<u32>,
    G: FnMut(u32, u32, u32, u64) -> Result<u32>,
{
    if headers.num_passes() != nz.num_passes() {
        return Err(Error::InvalidData(format!(
            "JXL multi_pass_hf_header: headers.num_passes()={} != nz.num_passes()={}",
            headers.num_passes(),
            nz.num_passes()
        )));
    }
    // Pre-resolve per-pass offsets once so the inner closures don't
    // re-lookup the per-pass header on every (varblock, channel)
    // call. The round-228 driver's `read_non_zeros` / `decode_symbol`
    // closures are invoked once per (pass, varblock, channel); the
    // per-pass `histogram_offset` is constant across the inner walk.
    let offsets: Vec<u64> = (0..headers.num_passes())
        .map(|p| {
            headers
                .histogram_offset(p)
                .expect("p < num_passes by construction")
        })
        .collect();
    decode_multi_pass_three_channels_with_resolver(
        grid,
        nz,
        resolver,
        |p, vb| qdc_at(p, vb),
        |p, c, predicted| {
            let offset = offsets[p as usize];
            read_non_zeros(p, c, predicted, offset)
        },
        |p, c, coeff_ctx| {
            let offset = offsets[p as usize];
            decode_symbol(p, c, coeff_ctx, offset)
        },
    )
}

/// Convenience helper: per-LfGroup multi-pass three-channel decode
/// driver that reads the per-pass [`PassGroupHfHeader`] sequence
/// inline from `br` before invoking
/// [`decode_multi_pass_with_hf_headers`].
///
/// The bit-position on entry must be at the start of pass-0's
/// `hfp` field (§C.8.3 first paragraph); the `nz.num_passes()`,
/// `num_hf_presets`, and `nb_block_ctx` arguments are the inherited
/// per-frame invariants from
/// [`crate::frame_header::FrameHeader::passes`] (Table C.6),
/// [`crate::hf_global::HfGlobal::num_hf_presets`] (§I.2.6), and
/// [`crate::lf_global::HfBlockContext::nb_block_ctx`] (§I.2.2).
#[allow(clippy::too_many_arguments)]
pub fn read_and_decode_multi_pass_with_hf_headers<Q, F, G>(
    br: &mut BitReader<'_>,
    grid: &DctSelectGrid,
    nz: &mut PerPassNonZerosGrids,
    resolver: &BlockContextResolver<'_>,
    num_hf_presets: u32,
    nb_block_ctx: u32,
    qdc_at: Q,
    read_non_zeros: F,
    decode_symbol: G,
) -> Result<(PerPassHfHeaders, MultiPassThreeChannelOutput)>
where
    Q: FnMut(u32, &Varblock) -> Result<[i32; 3]>,
    F: FnMut(u32, u32, u32, u64) -> Result<u32>,
    G: FnMut(u32, u32, u32, u64) -> Result<u32>,
{
    let num_passes = nz.num_passes();
    let headers = PerPassHfHeaders::read(br, num_passes, num_hf_presets, nb_block_ctx)?;
    let out = decode_multi_pass_with_hf_headers(
        grid,
        &headers,
        nz,
        resolver,
        qdc_at,
        read_non_zeros,
        decode_symbol,
    )?;
    Ok((headers, out))
}

/// Per-LfGroup multi-pass output companion type — a per-pass
/// `(hfp, histogram_offset)` digest. Provided so callers wiring the
/// next round (per-pass [`crate::hf_pass::HfPass`] coefficient-order
/// lookup + §C.7.2 entropy histogram routing) can read the per-pass
/// selectors back without re-walking the
/// [`PerPassHfHeaders::as_slice`] container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PassHfDigest {
    /// `headers[p].hfp` — the §C.8.3 first-paragraph selector.
    pub hfp: u32,
    /// `headers[p].histogram_offset = 495 × nb_block_ctx × hfp`.
    pub histogram_offset: u64,
}

impl PerPassHfHeaders {
    /// Collect a per-pass `(hfp, histogram_offset)` digest vector.
    /// `out[p]` corresponds to the `p`-th pass.
    pub fn digest(&self) -> Vec<PassHfDigest> {
        self.headers
            .iter()
            .map(|h| PassHfDigest {
                hfp: h.hfp,
                histogram_offset: h.histogram_offset,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;
    use crate::dct_select::{derive_dct_select, TransformType};
    use crate::lf_global::HfBlockContext;
    use crate::lf_group::HfMetadata;

    fn make_hf(block_info: Vec<i32>, nb_blocks: u32, info_w: u32) -> HfMetadata {
        HfMetadata {
            nb_blocks,
            x_from_y: vec![0],
            b_from_y: vec![0],
            block_info,
            sharpness: vec![0],
            channel_widths: [1, 1, info_w, 1],
            channel_heights: [1, 1, 2, 1],
        }
    }

    fn default_hbc() -> HfBlockContext {
        // Matches the round-214 / round-221 / round-228 default —
        // empty thresholds collapse the qf / qdc knobs, default
        // 39-entry block_ctx_map, nb_block_ctx = 15.
        HfBlockContext {
            used_default: true,
            qf_thresholds: vec![],
            lf_thresholds: [vec![], vec![], vec![]],
            block_ctx_map: vec![
                7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 9, 9, 10, 11, 12, 13, 14, 0, 0, 0, 0,
                7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            nb_block_ctx: 15,
        }
    }

    // ---------- PerPassHfHeaders tests ----------

    #[test]
    fn r232_per_pass_headers_read_two_passes_two_presets() {
        // num_hf_presets = 2 → nbits = 1 per `ceil(log2(2)) = 1`.
        // Pass 0 reads hfp = 0; pass 1 reads hfp = 1.
        // Bit layout: 0 | 1 = 0b10 = byte 0x02 (low bit first).
        let data = [0b0000_0010u8];
        let mut br = BitReader::new(&data);
        let headers = PerPassHfHeaders::read(&mut br, 2, 2, 15).unwrap();
        assert_eq!(headers.num_passes(), 2);
        assert_eq!(headers.hfp(0).unwrap(), 0);
        assert_eq!(headers.hfp(1).unwrap(), 1);
        // histogram_offset = 495 × 15 × hfp = 7425 × hfp.
        assert_eq!(headers.histogram_offset(0).unwrap(), 0);
        assert_eq!(headers.histogram_offset(1).unwrap(), 7425);
    }

    #[test]
    fn r232_per_pass_headers_single_preset_zero_bits() {
        // num_hf_presets = 1 → nbits = 0; every per-pass hfp is 0
        // without consuming any bits.
        let data = [0u8];
        let mut br = BitReader::new(&data);
        let headers = PerPassHfHeaders::read(&mut br, 3, 1, 15).unwrap();
        assert_eq!(headers.num_passes(), 3);
        for p in 0..3 {
            assert_eq!(headers.hfp(p).unwrap(), 0);
            assert_eq!(headers.histogram_offset(p).unwrap(), 0);
        }
    }

    #[test]
    fn r232_per_pass_headers_get_out_of_range_errors() {
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let r = headers.get(1);
        assert!(r.is_err());
    }

    #[test]
    fn r232_per_pass_headers_digest_matches_per_pass_state() {
        let headers = PerPassHfHeaders::from_headers(vec![
            PassGroupHfHeader {
                hfp: 0,
                histogram_offset: 0,
            },
            PassGroupHfHeader {
                hfp: 2,
                histogram_offset: 7425 * 2,
            },
        ]);
        let digest = headers.digest();
        assert_eq!(
            digest,
            vec![
                PassHfDigest {
                    hfp: 0,
                    histogram_offset: 0
                },
                PassHfDigest {
                    hfp: 2,
                    histogram_offset: 14850
                },
            ]
        );
    }

    #[test]
    fn r232_per_pass_headers_zero_passes_returns_empty() {
        let data = [0u8];
        let mut br = BitReader::new(&data);
        let headers = PerPassHfHeaders::read(&mut br, 0, 4, 15).unwrap();
        assert_eq!(headers.num_passes(), 0);
        assert!(headers.as_slice().is_empty());
    }

    #[test]
    fn r232_per_pass_headers_rejects_zero_num_hf_presets() {
        // PassGroupHfHeader::read rejects num_hf_presets == 0 per
        // §I.2.6 invariant.
        let data = [0u8; 4];
        let mut br = BitReader::new(&data);
        let r = PerPassHfHeaders::read(&mut br, 1, 0, 15);
        assert!(r.is_err());
    }

    #[test]
    fn r232_per_pass_headers_four_presets_two_bits_per_pass() {
        // num_hf_presets = 4 → nbits = 2.
        // Three passes encoding hfp = 1, 3, 2 → little-endian bits:
        // 01 | 11 | 10 = 0b101101 = 0x2D.
        let data = [0b0010_1101u8];
        let mut br = BitReader::new(&data);
        let headers = PerPassHfHeaders::read(&mut br, 3, 4, 15).unwrap();
        assert_eq!(headers.hfp(0).unwrap(), 1);
        assert_eq!(headers.hfp(1).unwrap(), 3);
        assert_eq!(headers.hfp(2).unwrap(), 2);
        // 495 × 15 = 7425.
        assert_eq!(headers.histogram_offset(0).unwrap(), 7425);
        assert_eq!(headers.histogram_offset(1).unwrap(), 22275);
        assert_eq!(headers.histogram_offset(2).unwrap(), 14850);
    }

    // ---------- decode_multi_pass_with_hf_headers tests ----------

    #[test]
    fn r232_driver_single_pass_routes_zero_offset() {
        // num_hf_presets = 1, num_passes = 1 → histogram_offset = 0.
        // The closure observes offset = 0 across every call.
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![0, 0], 1, 1);
        let grid = derive_dct_select(&hf, 8, 8).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 1, 1).unwrap();
        let headers = PerPassHfHeaders::from_headers(vec![PassGroupHfHeader {
            hfp: 0,
            histogram_offset: 0,
        }]);
        let mut observed_offsets: Vec<u64> = Vec::new();
        let out = decode_multi_pass_with_hf_headers(
            &grid,
            &headers,
            &mut nz,
            &resolver,
            |_p, _vb| Ok([0, 0, 0]),
            |_p, _c, _pred, offset| {
                observed_offsets.push(offset);
                Ok(0)
            },
            |_p, _c, _coef, _offset| Ok(0),
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 1);
        assert_eq!(out[0][0].0.transform, TransformType::Dct8x8);
        // Per-pass per-channel call: 1 pass × 3 channels = 3 calls.
        assert_eq!(observed_offsets, vec![0, 0, 0]);
    }

    #[test]
    fn r232_driver_two_pass_distinct_offsets() {
        // num_passes = 2 with hfp = (0, 1), nb_block_ctx = 15 →
        // histogram_offset = (0, 7425). Pass-0 closure observes 0,
        // pass-1 closure observes 7425.
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![0, 0], 1, 1);
        let grid = derive_dct_select(&hf, 8, 8).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(2, 3, 1, 1).unwrap();
        let headers = PerPassHfHeaders::from_headers(vec![
            PassGroupHfHeader {
                hfp: 0,
                histogram_offset: 0,
            },
            PassGroupHfHeader {
                hfp: 1,
                histogram_offset: 7425,
            },
        ]);
        let mut per_pass_offsets: Vec<u64> = vec![u64::MAX, u64::MAX];
        let _ = decode_multi_pass_with_hf_headers(
            &grid,
            &headers,
            &mut nz,
            &resolver,
            |_p, _vb| Ok([0, 0, 0]),
            |p, _c, _pred, offset| {
                per_pass_offsets[p as usize] = offset;
                Ok(0)
            },
            |_p, _c, _coef, _offset| Ok(0),
        )
        .unwrap();
        assert_eq!(per_pass_offsets, vec![0, 7425]);
    }

    #[test]
    fn r232_driver_offsets_threaded_through_both_closures() {
        // Verify that BOTH read_non_zeros and decode_symbol receive
        // the same per-pass offset. The round-221 inner driver
        // invokes decode_symbol `(size - num_blocks)` times per
        // channel per varblock (per §C.8.3 — the k-loop runs k in
        // [num_blocks, size)). For DCT8×8: size = 64, num_blocks = 1
        // → 63 decode_symbol calls per channel.
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![0, 0], 1, 1);
        let grid = derive_dct_select(&hf, 8, 8).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(2, 3, 1, 1).unwrap();
        let headers = PerPassHfHeaders::from_headers(vec![
            PassGroupHfHeader {
                hfp: 0,
                histogram_offset: 0,
            },
            PassGroupHfHeader {
                hfp: 2,
                histogram_offset: 14850,
            },
        ]);
        let mut decode_symbol_pass_offsets: Vec<(u32, u64)> = Vec::new();
        let _ = decode_multi_pass_with_hf_headers(
            &grid,
            &headers,
            &mut nz,
            &resolver,
            |_p, _vb| Ok([0, 0, 0]),
            |_p, _c, _pred, _offset| Ok(1),
            |p, _c, _coef, offset| {
                decode_symbol_pass_offsets.push((p, offset));
                Ok(0)
            },
        )
        .unwrap();
        // 2 passes × 3 channels × 63 decode_symbol calls per channel
        // = 378. Pass 0 has offset 0; pass 1 has offset 14850.
        // Each per-pass per-channel decode_symbol invocation must
        // observe the per-pass histogram offset.
        assert_eq!(decode_symbol_pass_offsets.len(), 378);
        let per_pass = 3 * 63;
        for (i, &(p, off)) in decode_symbol_pass_offsets.iter().enumerate() {
            let expected_pass = if i < per_pass { 0 } else { 1 };
            let expected_off = if expected_pass == 0 { 0 } else { 14850 };
            assert_eq!(p, expected_pass, "i={i}");
            assert_eq!(off, expected_off, "i={i}");
        }
    }

    #[test]
    fn r232_driver_rejects_num_passes_mismatch() {
        // headers.num_passes() = 2, nz.num_passes() = 3 → reject.
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![0, 0], 1, 1);
        let grid = derive_dct_select(&hf, 8, 8).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(3, 3, 1, 1).unwrap();
        let headers = PerPassHfHeaders::from_headers(vec![
            PassGroupHfHeader {
                hfp: 0,
                histogram_offset: 0,
            },
            PassGroupHfHeader {
                hfp: 0,
                histogram_offset: 0,
            },
        ]);
        let r = decode_multi_pass_with_hf_headers(
            &grid,
            &headers,
            &mut nz,
            &resolver,
            |_p, _vb| Ok([0, 0, 0]),
            |_p, _c, _pred, _offset| Ok(0),
            |_p, _c, _coef, _offset| Ok(0),
        );
        assert!(r.is_err());
    }

    #[test]
    fn r232_driver_per_pass_error_propagates() {
        // Pass 1's read_non_zeros closure errors mid-walk; the
        // driver propagates and aborts.
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![0, 0], 1, 1);
        let grid = derive_dct_select(&hf, 8, 8).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(2, 3, 1, 1).unwrap();
        let headers = PerPassHfHeaders::from_headers(vec![
            PassGroupHfHeader {
                hfp: 0,
                histogram_offset: 0,
            },
            PassGroupHfHeader {
                hfp: 1,
                histogram_offset: 7425,
            },
        ]);
        let r = decode_multi_pass_with_hf_headers(
            &grid,
            &headers,
            &mut nz,
            &resolver,
            |_p, _vb| Ok([0, 0, 0]),
            |p, _c, _pred, _offset| {
                if p == 1 {
                    Err(Error::InvalidData("r232: pass-1 closure failure".into()))
                } else {
                    Ok(0)
                }
            },
            |_p, _c, _coef, _offset| Ok(0),
        );
        assert!(r.is_err());
    }

    #[test]
    fn r232_driver_pass_distinct_qdc_threading() {
        // qdc closure receives the pass index per the round-228
        // contract; round-232 must preserve that argument shape
        // unchanged.
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![0, 0], 1, 1);
        let grid = derive_dct_select(&hf, 8, 8).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(3, 3, 1, 1).unwrap();
        let headers = PerPassHfHeaders::from_headers(vec![
            PassGroupHfHeader {
                hfp: 0,
                histogram_offset: 0,
            },
            PassGroupHfHeader {
                hfp: 1,
                histogram_offset: 7425,
            },
            PassGroupHfHeader {
                hfp: 0,
                histogram_offset: 0,
            },
        ]);
        let mut qdc_passes: Vec<u32> = Vec::new();
        let _ = decode_multi_pass_with_hf_headers(
            &grid,
            &headers,
            &mut nz,
            &resolver,
            |p, _vb| {
                qdc_passes.push(p);
                Ok([0, 0, 0])
            },
            |_p, _c, _pred, _offset| Ok(0),
            |_p, _c, _coef, _offset| Ok(0),
        )
        .unwrap();
        assert_eq!(qdc_passes, vec![0, 1, 2]);
    }

    #[test]
    fn r232_driver_offset_per_channel_uniform_within_pass() {
        // Within a single pass, the offset is constant across all
        // three channels (X / Y / B). Verify by recording the
        // (channel, offset) pairs for a 2-pass run.
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![0, 0], 1, 1);
        let grid = derive_dct_select(&hf, 8, 8).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(2, 3, 1, 1).unwrap();
        let headers = PerPassHfHeaders::from_headers(vec![
            PassGroupHfHeader {
                hfp: 0,
                histogram_offset: 0,
            },
            PassGroupHfHeader {
                hfp: 1,
                histogram_offset: 7425,
            },
        ]);
        let mut observed: Vec<(u32, u32, u64)> = Vec::new();
        let _ = decode_multi_pass_with_hf_headers(
            &grid,
            &headers,
            &mut nz,
            &resolver,
            |_p, _vb| Ok([0, 0, 0]),
            |p, c, _pred, offset| {
                observed.push((p, c, offset));
                Ok(0)
            },
            |_p, _c, _coef, _offset| Ok(0),
        )
        .unwrap();
        // Pass 0: 3 calls (channels 0, 1, 2) with offset 0.
        // Pass 1: 3 calls with offset 7425.
        assert_eq!(observed.len(), 6);
        assert!(observed[..3].iter().all(|&(p, _, off)| p == 0 && off == 0));
        assert!(observed[3..]
            .iter()
            .all(|&(p, _, off)| p == 1 && off == 7425));
    }

    // ---------- read_and_decode_multi_pass_with_hf_headers ----------

    #[test]
    fn r232_inline_read_and_decode_round_trip() {
        // End-to-end: read the per-pass header sequence from bits +
        // decode the per-LfGroup multi-pass varblock walk.
        // num_passes = 2, num_hf_presets = 2 → nbits = 1.
        // Pass-0 hfp = 0, pass-1 hfp = 1 → bits 0, 1 → byte 0b10 = 0x02.
        let data = [0b0000_0010u8];
        let mut br = BitReader::new(&data);
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![0, 0], 1, 1);
        let grid = derive_dct_select(&hf, 8, 8).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(2, 3, 1, 1).unwrap();

        let mut closure_offsets: Vec<u64> = Vec::new();
        let (headers, out) = read_and_decode_multi_pass_with_hf_headers(
            &mut br,
            &grid,
            &mut nz,
            &resolver,
            2,
            15,
            |_p, _vb| Ok([0, 0, 0]),
            |_p, _c, _pred, offset| {
                closure_offsets.push(offset);
                Ok(0)
            },
            |_p, _c, _coef, _offset| Ok(0),
        )
        .unwrap();
        assert_eq!(headers.num_passes(), 2);
        assert_eq!(headers.hfp(0).unwrap(), 0);
        assert_eq!(headers.hfp(1).unwrap(), 1);
        assert_eq!(out.len(), 2);
        // 2 passes × 3 channels = 6 read_non_zeros calls;
        // pass 0 → offset 0, pass 1 → offset 7425.
        assert_eq!(closure_offsets.len(), 6);
        assert!(closure_offsets[..3].iter().all(|&o| o == 0));
        assert!(closure_offsets[3..].iter().all(|&o| o == 7425));
        // Bits consumed = 2 × 1 = 2 (the two per-pass `u(1)` reads).
        assert_eq!(br.bits_read(), 2);
    }

    #[test]
    fn r232_inline_read_propagates_inner_bit_read_error() {
        // num_hf_presets = 2 → 1 bit per pass. Empty data → first
        // bit read fails (out-of-data).
        let data = [];
        let mut br = BitReader::new(&data);
        let hbc = default_hbc();
        let resolver = BlockContextResolver::new(&hbc);
        let hf = make_hf(vec![0, 0], 1, 1);
        let grid = derive_dct_select(&hf, 8, 8).unwrap();
        let mut nz = PerPassNonZerosGrids::new_uniform(1, 3, 1, 1).unwrap();
        let r = read_and_decode_multi_pass_with_hf_headers(
            &mut br,
            &grid,
            &mut nz,
            &resolver,
            2,
            15,
            |_p, _vb| Ok([0, 0, 0]),
            |_p, _c, _pred, _offset| Ok(0),
            |_p, _c, _coef, _offset| Ok(0),
        );
        assert!(r.is_err());
    }
}
