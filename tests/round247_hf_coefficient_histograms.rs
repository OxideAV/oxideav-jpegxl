//! Round 247 integration coverage —
//! [`oxideav_jpegxl::hf_coefficient_histograms::HfCoefficientHistograms`].
//!
//! ISO/IEC FDIS 18181-1:2021 §C.7.2 HF coefficient histograms — the
//! actual `EntropyStream::read(br, num_distributions)` call against the
//! §C.7.2 read-size that round 238 computed but left deferred.
//!
//! Round 247 binds the §C.7.2 read-size derived from the
//! [`oxideav_jpegxl::hf_coeff_histogram_size::HfCoefficientHistogramSize`]
//! sizing primitive to the resulting
//! [`oxideav_jpegxl::modular_fdis::EntropyStream`], and ties the
//! [`oxideav_jpegxl::hf_pass::read_hf_pass_sequence`] → §C.7.2
//! transition together via
//! [`HfCoefficientHistograms::read_after_hf_pass_sequence`].
//!
//! These integration tests pin the public-surface invariants:
//!
//! * `HfCoefficientHistograms::read` consumes the §D.3 prelude.
//! * Sizing accessors (`num_distributions` / `offset_for_hfp` /
//!   `num_hf_presets` / `nb_block_ctx`) forward through the underlying
//!   [`oxideav_jpegxl::hf_coeff_histogram_size::HfCoefficientHistogramSize`]
//!   and match the §C.7.2 / §C.8.3 spec arithmetic.
//! * Zero-input rejections do not advance the
//!   [`oxideav_jpegxl::bitreader::BitReader`].
//! * `read_ans_state_init` for a prefix-coded stream is a no-op and
//!   idempotent.
//! * `read_after_hf_pass_sequence` is numerically identical to
//!   building the sizing descriptor by hand and invoking `read`.

use oxideav_jpegxl::bitreader::BitReader;
use oxideav_jpegxl::hf_coeff_histogram_size::HfCoefficientHistogramSize;
use oxideav_jpegxl::hf_coefficient_histograms::HfCoefficientHistograms;

/// Minimal §D.3 prelude that reads cleanly for `num_dist = N` with a
/// single cluster, prefix-coded, count = 1.
///
/// Layout (LSB-first):
///   bit 0       : lz77_enabled = 0
///   bit 1       : is_simple = 1            (clustering D.3.5 simple path)
///   bits 2-3    : nbits = 0                (every distribution → cluster 0)
///   bit 4       : use_prefix_code = 1      (log_alphabet_size = 15)
///   bits 5-8    : split_exponent = 0       (HybridUintConfig::read u(4))
///   bit 9       : prefix count selector = 0 → count = 1 for the single cluster
///
/// Total = 10 bits → 2 bytes. The exact byte values for an all-zero
/// payload except bits 1 and 4 are: byte 0 = 0b00010010 = 0x12,
/// byte 1 = 0x00.
fn minimal_prefix_prelude_bytes() -> [u8; 2] {
    // bit 0: 0  (lz77_enabled)
    // bit 1: 1  (is_simple)
    // bit 2: 0  (nbits low)
    // bit 3: 0  (nbits high)
    // bit 4: 1  (use_prefix_code)
    // bit 5: 0  (split_exponent bit 0)
    // bit 6: 0
    // bit 7: 0
    // bit 8: 0  (split_exponent bit 3)
    // bit 9: 0  (prefix count selector)
    [0b0001_0010, 0b0000_0000]
}

#[test]
fn r247_integration_read_with_minimal_prelude() {
    let bytes = minimal_prefix_prelude_bytes();
    let mut br = BitReader::new(&bytes);
    let size = HfCoefficientHistogramSize::new(1, 1).unwrap();
    let histos = HfCoefficientHistograms::read(&mut br, size).unwrap();
    assert_eq!(histos.num_distributions(), 495);
    assert_eq!(histos.num_hf_presets(), 1);
    assert_eq!(histos.nb_block_ctx(), 1);
    assert_eq!(histos.offset_for_hfp(0).unwrap(), 0);
    assert!(histos.offset_for_hfp(1).is_err());
    // Single cluster → one entropy entry, 495-long cluster map.
    assert_eq!(histos.entropy.cluster_map.len(), 495);
    assert_eq!(histos.entropy.entropies.len(), 1);
    assert!(histos.entropy.use_prefix_code);
    assert_eq!(histos.entropy.log_alphabet_size, 15);
}

#[test]
fn r247_integration_read_after_hf_pass_sequence_helper_matches_direct() {
    let bytes = minimal_prefix_prelude_bytes();

    let mut br_direct = BitReader::new(&bytes);
    let size = HfCoefficientHistogramSize::new(1, 1).unwrap();
    let direct = HfCoefficientHistograms::read(&mut br_direct, size).unwrap();
    let direct_bits = br_direct.bits_read();

    let mut br_helper = BitReader::new(&bytes);
    let helper =
        HfCoefficientHistograms::read_after_hf_pass_sequence(&mut br_helper, 1, 1).unwrap();
    let helper_bits = br_helper.bits_read();

    // Exact same bit budget consumed.
    assert_eq!(direct_bits, helper_bits);
    assert_eq!(direct.num_distributions(), helper.num_distributions());
    assert_eq!(direct.entropy.cluster_map, helper.entropy.cluster_map);
}

#[test]
fn r247_integration_size_accessors_match_primitive() {
    // num_hf_presets = 2 → num_distributions = 990. We re-use the
    // minimal-prelude bytes for the §D.3 read (same shape works for
    // any num_dist > 1 with the simple-clustering + single-cluster
    // single-symbol prefix layout).
    let bytes = minimal_prefix_prelude_bytes();
    let mut br = BitReader::new(&bytes);
    let size = HfCoefficientHistogramSize::new(2, 1).unwrap();
    let expected_total = size.num_distributions();
    let expected_offset_1 = size.offset_for_hfp(1).unwrap();
    let histos = HfCoefficientHistograms::read(&mut br, size).unwrap();
    assert_eq!(histos.num_distributions(), expected_total);
    assert_eq!(histos.offset_for_hfp(1).unwrap(), expected_offset_1);
    assert_eq!(histos.num_hf_presets(), 2);
    assert_eq!(histos.nb_block_ctx(), 1);
}

#[test]
fn r247_integration_zero_inputs_dont_advance_br() {
    let bytes = [0xFFu8; 4];
    let mut br = BitReader::new(&bytes);
    let bits_before = br.bits_read();
    let r = HfCoefficientHistograms::read_after_hf_pass_sequence(&mut br, 0, 1);
    assert!(r.is_err());
    assert_eq!(br.bits_read(), bits_before);

    let r = HfCoefficientHistograms::read_after_hf_pass_sequence(&mut br, 1, 0);
    assert!(r.is_err());
    assert_eq!(br.bits_read(), bits_before);
}

#[test]
fn r247_integration_truncated_bitstream_is_error_not_panic() {
    let bytes: [u8; 0] = [];
    let mut br = BitReader::new(&bytes);
    let size = HfCoefficientHistogramSize::new(1, 1).unwrap();
    let r = HfCoefficientHistograms::read(&mut br, size);
    assert!(r.is_err());
}

#[test]
fn r247_integration_read_ans_state_init_noop_for_prefix() {
    let bytes = minimal_prefix_prelude_bytes();
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
