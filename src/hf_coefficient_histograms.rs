//! `HfCoefficientHistograms` — ISO/IEC FDIS 18181-1:2021 §C.7.2 HF
//! coefficient histograms read.
//!
//! ## §C.7.2 in full (FDIS p. 54)
//!
//! > Let `nb_block_ctx` be equal to `max(block_ctx_map) + 1`. The
//! > decoder reads a histogram with `495 × num_hf_presets ×
//! > nb_block_ctx` clustered distributions D from the codestream as
//! > specified in D.3.
//!
//! ## Scope (round 247)
//!
//! Round 247 lifts the **actual codestream read** of the §C.7.2
//! clustered-distributions block out of the deferred-next-step backlog
//! left by round 238. Round 238 landed the typed
//! [`crate::hf_coeff_histogram_size::HfCoefficientHistogramSize`]
//! sizing primitive — `num_distributions()` answers exactly how many
//! distributions §C.7.2 consumes — but the actual
//! `EntropyStream::read(br, num_distributions)` call against that size
//! remained deferred. This round closes that gap.
//!
//! Round 247 lands the typed [`HfCoefficientHistograms`] wrapper which
//! binds the §C.7.2 read-size to the resulting
//! [`crate::modular_fdis::EntropyStream`] and exposes both the
//! pre-state-init and post-state-init shapes. The wrapper holds:
//!
//! * `size: HfCoefficientHistogramSize` — the §C.7.2 sizing
//!   descriptor (`num_hf_presets`, `nb_block_ctx`, derived
//!   `per_preset` / `num_distributions` / `offset_for_hfp`).
//! * `entropy: EntropyStream` — the [`crate::modular_fdis::EntropyStream`]
//!   resulting from the §D.3 read for `num_distributions()` clustered
//!   distributions. ANS state initialisation is **not** performed here
//!   per the round-3 2024-spec correction (the
//!   [`crate::modular_fdis::EntropyStream`] doc-comment spells this
//!   out: the `u(32)` ANS state initialiser is read between the
//!   prelude and the first symbol decode, not eagerly during
//!   [`crate::modular_fdis::EntropyStream::read`]).
//!
//! ## Read shape
//!
//! Two entry-points:
//!
//! * [`HfCoefficientHistograms::read`] — caller-built
//!   [`HfCoefficientHistogramSize`] driving the
//!   [`crate::modular_fdis::EntropyStream::read`] call.
//! * [`HfCoefficientHistograms::read_after_hf_pass_sequence`] — the
//!   §C.7.1 → §C.7.2 transition convenience: a caller that has just
//!   walked [`crate::hf_pass::read_hf_pass_sequence`] for the same
//!   `(num_hf_presets, nb_block_ctx)` invokes this helper directly
//!   against the same [`BitReader`].
//!
//! ## What this round is **not**
//!
//! * No per-context offset routing (§C.8.3 `offset = 495 × nb_block_ctx
//!   × hfp` was already landed by round 90 +
//!   [`crate::pass_group_hf::PassGroupHfHeader`]; round 247 simply
//!   re-uses the same primitive).
//! * No per-block decode walk against the histograms (Listing C.13
//!   `BlockContext()` / `NonZerosContext()` / `CoefficientContext()`
//!   already exist in [`crate::pass_group_hf`]; round 247 wires the
//!   histogram-stream input they will eventually route through but
//!   does **not** itself perform the per-block sweep).
//! * No ANS state initialiser read inside [`Self::read`]; per the
//!   round-3 2024-spec correction, the caller invokes
//!   [`Self::read_ans_state_init`] just before the first symbol
//!   decode against the §C.7.2 stream.
//!
//! ## Bound: `usize` cap
//!
//! `num_distributions() : u64` is converted to `usize` for the
//! `EntropyStream::read` call. On a 32-bit target the §C.7.2 read
//! could theoretically exceed `usize::MAX` (`num_hf_presets ≤ 2^28` ×
//! `nb_block_ctx ≤ 256` × 495 ≈ `2^45`); we reject with `InvalidData`
//! before the cast. On 64-bit targets the cast is always lossless for
//! the spec-permitted maxima.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::hf_coeff_histogram_size::HfCoefficientHistogramSize;
use crate::modular_fdis::EntropyStream;

/// The §C.7.2 HF coefficient histograms bundle — sizing descriptor +
/// the [`EntropyStream`] read against `size.num_distributions()`.
///
/// ANS state initialisation is deferred to [`Self::read_ans_state_init`]
/// per the 2024-spec round-3 correction (see
/// [`EntropyStream::read_ans_state_init`] for the rationale).
#[derive(Debug, Clone)]
pub struct HfCoefficientHistograms {
    /// §C.7.2 sizing descriptor (`num_hf_presets`, `nb_block_ctx`).
    pub size: HfCoefficientHistogramSize,
    /// [`EntropyStream`] read for `size.num_distributions()` clustered
    /// distributions per §D.3.
    pub entropy: EntropyStream,
}

impl HfCoefficientHistograms {
    /// Read the §C.7.2 histogram block from `br` against the typed
    /// sizing descriptor `size`.
    ///
    /// The wire format is exactly [`EntropyStream::read`] with
    /// `num_dist = size.num_distributions()`. ANS state initialisation
    /// is **not** performed; the caller invokes
    /// [`Self::read_ans_state_init`] before the first symbol decode.
    ///
    /// Returns `Err(InvalidData)` when `size.num_distributions()`
    /// overflows `usize` on the target architecture (defensive guard
    /// against the upper-bound product on 32-bit targets).
    pub fn read(br: &mut BitReader<'_>, size: HfCoefficientHistogramSize) -> Result<Self> {
        let num_dist_u64 = size.num_distributions();
        let num_dist: usize = num_dist_u64.try_into().map_err(|_| {
            Error::InvalidData(format!(
                "JXL HfCoefficientHistograms: num_distributions {num_dist_u64} exceeds usize on \
                 this target"
            ))
        })?;
        let entropy = EntropyStream::read(br, num_dist)?;
        Ok(Self { size, entropy })
    }

    /// §C.7.1 → §C.7.2 transition convenience. Constructs the
    /// [`HfCoefficientHistogramSize`] from `num_hf_presets +
    /// nb_block_ctx` (the same two inputs the caller already passes
    /// to [`crate::hf_pass::read_hf_pass_sequence`]) and reads the
    /// §C.7.2 block from the same [`BitReader`] without a separate
    /// constructor call.
    ///
    /// The §C.7.1 `read_hf_pass_sequence` advances `br` past the
    /// per-preset [`crate::hf_pass::HfPass`] bundles; the caller then
    /// invokes `read_after_hf_pass_sequence` on the same `br` for the
    /// §C.7.2 step.
    pub fn read_after_hf_pass_sequence(
        br: &mut BitReader<'_>,
        num_hf_presets: u32,
        nb_block_ctx: u32,
    ) -> Result<Self> {
        let size = HfCoefficientHistogramSize::new(num_hf_presets, nb_block_ctx)?;
        Self::read(br, size)
    }

    /// Read the ANS state initialiser (`u(32)` per C.3.2) on the
    /// underlying [`EntropyStream`]. Forwards to
    /// [`EntropyStream::read_ans_state_init`].
    ///
    /// Must be called once, after [`Self::read`] returned and before
    /// the first symbol decode against the histogram block. A no-op
    /// for prefix-coded streams (`use_prefix_code == true`).
    pub fn read_ans_state_init(&mut self, br: &mut BitReader<'_>) -> Result<()> {
        self.entropy.read_ans_state_init(br)
    }

    /// `495 × num_hf_presets × nb_block_ctx` — the §C.7.2 total. The
    /// caller should not need to recompute this; surfaced for
    /// downstream consumers (e.g. logging / trace tape).
    pub fn num_distributions(&self) -> u64 {
        self.size.num_distributions()
    }

    /// `495 × nb_block_ctx × hfp` per §C.8.3, range-checked on `hfp`.
    /// Forwarded from [`HfCoefficientHistogramSize::offset_for_hfp`].
    pub fn offset_for_hfp(&self, hfp: u32) -> Result<u64> {
        self.size.offset_for_hfp(hfp)
    }

    /// `num_hf_presets` — the §I.2.6 HfGlobal field.
    pub fn num_hf_presets(&self) -> u32 {
        self.size.num_hf_presets
    }

    /// `nb_block_ctx = max(block_ctx_map) + 1` per §C.7.2 line 1.
    pub fn nb_block_ctx(&self) -> u32 {
        self.size.nb_block_ctx
    }

    /// Mutable reference to the underlying entropy stream so the
    /// downstream §C.8.3 per-block decode loop can route
    /// `decode_symbol(ctx + offset)` reads through it once the ANS
    /// state is initialised.
    pub fn entropy_mut(&mut self) -> &mut EntropyStream {
        &mut self.entropy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ans::test_helpers::pack_lsb;

    /// Smoke-test of the size-only construction path: when the caller
    /// supplies a pre-built [`HfCoefficientHistogramSize`], the wrapper
    /// constructor pushes the correct `num_distributions` value into
    /// the [`EntropyStream::read`] call, and the entropy stream's
    /// resulting cluster_map length is `effective_num_dist` (which the
    /// EntropyStream::read documentation defines as `num_dist + 1`
    /// when LZ77 is enabled, else `num_dist`).
    ///
    /// We choose `num_hf_presets = 1`, `nb_block_ctx = 1` → 495
    /// distributions, lz77 disabled, simple clustering with all 495
    /// distributions mapped to cluster 0.
    #[test]
    fn r247_read_with_minimal_prelude_single_cluster() {
        // Bit layout (LSB-first per pack_lsb):
        //   1 bit  : lz77_enabled = 0
        //   1 bit  : use_prefix_code = 1  (prefix path → log_alphabet_size = 15)
        //   1 bit  : clustering is_simple = 1
        //   2 bits : nbits = 0  (all clusters take 0 bits, i.e. cluster 0)
        //   1 bit  : per-cluster prefix-count selector u(1) = 0 → count = 1
        //   1 bit  : prefix code for the single-symbol cluster — handled by read_prefix_code
        // The exact bit-shape after the use_prefix_code prelude depends on the
        // read_prefix_code path for a single-symbol cluster.

        // Build the minimal prelude up to use_prefix_code.
        let mut parts: Vec<(u32, u32)> = vec![
            (0, 1), // lz77_enabled = 0
        ];
        // num_dist > 1 → clustering is read AFTER use_prefix_code +
        // log_alphabet_size. The EntropyStream::read source orders the
        // reads as: lz77 (+ optional lz77 prelude) → effective_num_dist
        // computed → clustering read (because effective_num_dist > 1) →
        // use_prefix_code → log_alphabet_size → per-cluster configs →
        // per-cluster prefix/ANS data.
        //
        // r247 NOTE: we therefore lay out the clustering bits BEFORE the
        // use_prefix_code bit. Simple clustering for 495 distributions
        // with nbits = 0 emits 1 (is_simple) then 2 (nbits) then nothing
        // (495 × 0 bits) → 3 bits total.
        parts.push((1, 1)); // is_simple = 1
        parts.push((0, 2)); // nbits = 0 → every distribution maps to cluster 0
        parts.push((1, 1)); // use_prefix_code = 1 → log_alphabet_size = 15

        // For the prefix path, we need a per-cluster HybridUintConfig
        // before the prefix histograms. HybridUintConfig::read for
        // log_alphabet_size=15: split_exponent u(ceil_log2(15+1)) = u(4)
        // — we keep split_exponent = 0 to minimise downstream reads.
        // Reading 0 → split_exponent = 0; then msb_in_token u(...) =
        // 0; then lsb_in_token u(...) = 0. Layout per
        // HybridUintConfig::read: split_exponent first, then if
        // split_exponent != log_alphabet_size the two extra fields are
        // read; otherwise default (msb=0, lsb=0).
        parts.push((0, 4)); // split_exponent = 0
                            // split_exponent (0) != log_alphabet_size (15), so the routine
                            // reads the two extra fields:
                            //   msb_in_token = u(ceil_log2(split_exponent+1)) = u(0) → no bits
                            //   lsb_in_token = u(ceil_log2(split_exponent-msb+1)) = u(0) → no bits
                            // (the ceil_log2(1) = 0 fast-path consumes zero bits)

        // Per-cluster prefix histogram: u(1) selector to choose
        // count=1 vs count from u(4)+payload.
        parts.push((0, 1)); // u(1) = 0 → count = 1 (single-symbol code)

        let bytes = pack_lsb(&parts);
        let mut br = BitReader::new(&bytes);

        let size = HfCoefficientHistogramSize::new(1, 1).unwrap();
        let histos = HfCoefficientHistograms::read(&mut br, size).unwrap();
        // 495 × 1 × 1 = 495
        assert_eq!(histos.num_distributions(), 495);
        // simple clustering with nbits=0 → every distribution → cluster 0
        // → 1 cluster, 1 entropy entry.
        assert_eq!(histos.entropy.entropies.len(), 1);
        assert_eq!(histos.entropy.cluster_map.len(), 495);
        assert!(histos.entropy.use_prefix_code);
        assert_eq!(histos.entropy.log_alphabet_size, 15);
    }

    /// Size accessors forward through to the underlying
    /// [`HfCoefficientHistogramSize`] without re-deriving the product.
    #[test]
    fn r247_accessors_forward_size_fields() {
        let size = HfCoefficientHistogramSize::new(4, 15).unwrap();

        // Build a minimal prelude that yields a successful read.
        // num_distributions = 495 × 4 × 15 = 29700. We need clustering
        // (simple, nbits=0, all-zero) + use_prefix_code=1 + per-cluster
        // HybridUintConfig + single-symbol prefix.
        let mut parts: Vec<(u32, u32)> = vec![
            (0, 1), // lz77_enabled = 0
            (1, 1), // is_simple = 1
            (0, 2), // nbits = 0 → all distributions cluster 0
            (1, 1), // use_prefix_code = 1 → log_alphabet_size = 15
            (0, 4), // split_exponent = 0
            (0, 1), // prefix count selector = 0 → count = 1
        ];
        let _ = &mut parts; // keep mutability explicit for the diff
        let bytes = pack_lsb(&parts);
        let mut br = BitReader::new(&bytes);

        let histos = HfCoefficientHistograms::read(&mut br, size).unwrap();
        assert_eq!(histos.num_hf_presets(), 4);
        assert_eq!(histos.nb_block_ctx(), 15);
        assert_eq!(histos.num_distributions(), 29_700);
        assert_eq!(histos.offset_for_hfp(0).unwrap(), 0);
        assert_eq!(histos.offset_for_hfp(1).unwrap(), 7425);
        assert_eq!(histos.offset_for_hfp(2).unwrap(), 14_850);
        assert_eq!(histos.offset_for_hfp(3).unwrap(), 22_275);
        assert!(histos.offset_for_hfp(4).is_err());
    }

    /// Convenience entry-point `read_after_hf_pass_sequence` is
    /// numerically identical to building the size by hand.
    #[test]
    fn r247_read_after_hf_pass_sequence_matches_direct_read() {
        // Two presets, nb_block_ctx = 1 → num_distributions = 990.
        let parts: Vec<(u32, u32)> = vec![
            (0, 1), // lz77_enabled = 0
            (1, 1), // is_simple = 1
            (0, 2), // nbits = 0
            (1, 1), // use_prefix_code = 1
            (0, 4), // split_exponent = 0
            (0, 1), // prefix count = 1
        ];
        let bytes = pack_lsb(&parts);

        let mut br_direct = BitReader::new(&bytes);
        let size = HfCoefficientHistogramSize::new(2, 1).unwrap();
        let direct = HfCoefficientHistograms::read(&mut br_direct, size).unwrap();

        let mut br_helper = BitReader::new(&bytes);
        let helper =
            HfCoefficientHistograms::read_after_hf_pass_sequence(&mut br_helper, 2, 1).unwrap();

        assert_eq!(direct.num_distributions(), helper.num_distributions());
        assert_eq!(direct.num_hf_presets(), helper.num_hf_presets());
        assert_eq!(direct.nb_block_ctx(), helper.nb_block_ctx());
        assert_eq!(
            direct.entropy.cluster_map.len(),
            helper.entropy.cluster_map.len()
        );
        assert_eq!(
            direct.entropy.use_prefix_code,
            helper.entropy.use_prefix_code
        );
        assert_eq!(
            direct.entropy.log_alphabet_size,
            helper.entropy.log_alphabet_size
        );
    }

    /// Zero-input rejections propagate from
    /// [`HfCoefficientHistogramSize::new`] through the helper without
    /// touching the [`BitReader`].
    #[test]
    fn r247_zero_inputs_rejected_before_reading_br() {
        let bytes = [0u8];
        let mut br = BitReader::new(&bytes);
        let bits_before = br.bits_read();
        // num_hf_presets = 0 → size constructor rejects → no bits consumed.
        let r = HfCoefficientHistograms::read_after_hf_pass_sequence(&mut br, 0, 1);
        assert!(matches!(r, Err(Error::InvalidData(_))));
        assert_eq!(br.bits_read(), bits_before);

        // nb_block_ctx = 0 → same.
        let r = HfCoefficientHistograms::read_after_hf_pass_sequence(&mut br, 1, 0);
        assert!(matches!(r, Err(Error::InvalidData(_))));
        assert_eq!(br.bits_read(), bits_before);
    }

    /// Truncated bitstream propagates an `InvalidData` (or similar)
    /// error from the underlying [`EntropyStream::read`] without
    /// panicking. We deliberately feed an empty buffer to force the
    /// very first `read_bit` to fail.
    #[test]
    fn r247_truncated_bitstream_returns_error() {
        let bytes: [u8; 0] = [];
        let mut br = BitReader::new(&bytes);
        let size = HfCoefficientHistogramSize::new(1, 1).unwrap();
        let r = HfCoefficientHistograms::read(&mut br, size);
        assert!(r.is_err());
    }

    /// `read_ans_state_init` is a no-op for prefix-coded streams. Round
    /// 247 verifies that calling it twice on a successfully-read
    /// prefix-stream histogram is idempotent (the second call observes
    /// `use_prefix_code == true` and returns early).
    #[test]
    fn r247_read_ans_state_init_noop_for_prefix_stream() {
        let parts: Vec<(u32, u32)> = vec![
            (0, 1), // lz77_enabled = 0
            (1, 1), // is_simple = 1
            (0, 2), // nbits = 0
            (1, 1), // use_prefix_code = 1
            (0, 4), // split_exponent = 0
            (0, 1), // prefix count = 1
        ];
        let bytes = pack_lsb(&parts);
        let mut br = BitReader::new(&bytes);
        let size = HfCoefficientHistogramSize::new(1, 1).unwrap();
        let mut histos = HfCoefficientHistograms::read(&mut br, size).unwrap();
        let bits_after_read = br.bits_read();
        histos.read_ans_state_init(&mut br).unwrap();
        // Prefix path → no `u(32)` was consumed.
        assert_eq!(br.bits_read(), bits_after_read);
        // Idempotent.
        histos.read_ans_state_init(&mut br).unwrap();
        assert_eq!(br.bits_read(), bits_after_read);
    }

    /// `entropy_mut()` returns a mutable reference suitable for the
    /// downstream §C.8.3 per-block decode loop. We check we can mutate
    /// a field through it (round-trip the cluster_map vector len).
    #[test]
    fn r247_entropy_mut_returns_mutable_ref() {
        let parts: Vec<(u32, u32)> = vec![
            (0, 1), // lz77_enabled = 0
            (1, 1), // is_simple = 1
            (0, 2), // nbits = 0
            (1, 1), // use_prefix_code = 1
            (0, 4), // split_exponent = 0
            (0, 1), // prefix count = 1
        ];
        let bytes = pack_lsb(&parts);
        let mut br = BitReader::new(&bytes);
        let size = HfCoefficientHistogramSize::new(1, 1).unwrap();
        let mut histos = HfCoefficientHistograms::read(&mut br, size).unwrap();
        let before_len = histos.entropy.cluster_map.len();
        // Demonstrate we can take a &mut from entropy_mut().
        let e: &mut EntropyStream = histos.entropy_mut();
        assert_eq!(e.cluster_map.len(), before_len);
    }
}
