//! `HfGlobal` bundle — ISO/IEC 18181-1:2024 Annex I.2.4 + I.2.6.
//!
//! ## Round 13 baseline
//!
//! The default-encoding fast path (`u(1) == 1`) selected all 17
//! dequant matrices from Table I.6 defaults. The non-default branch
//! returned `Error::Unsupported`.
//!
//! ## Round 14 — encoding-modes parse (I.2.4 / Table I.5 / Listing C.10)
//!
//! Round 14 wires the **non-default-encoding parse** of the HfGlobal
//! dequantization-matrix bundle:
//!
//! * 17 sets of `encoding_mode = u(3)` (Table I.5).
//! * Per mode, the corresponding parameter blocks are read from the
//!   bitstream (Listing C.10):
//!   - **Library (0)** — no parameters.
//!   - **Hornuss (1)** — 3 × 3 F16 matrix, each element ×64.
//!   - **DCT2 (2)**   — 3 × 6 F16 matrix, each element ×64.
//!   - **DCT4 (3)**   — 3 × 2 F16 matrix (×64) + `ReadDctParams()`.
//!   - **DCT4x8 (4)** — 3 × 1 F16 matrix + `ReadDctParams()`.
//!   - **AFV (5)**    — 3 × 9 F16 matrix (cols 0..5 ×64) +
//!     `ReadDctParams()` + `ReadDctParams()` (the dct4x4 set).
//!   - **DCT (6)**    — `ReadDctParams()` only.
//!   - **RAW (7)**    — `denominator = F16()` then a modular
//!     sub-bitstream of the same shape as the target quant matrix.
//!     Round 14 keeps RAW under `Error::Unsupported` because the
//!     "same shape as the required quant matrix" pulls in per-Table-I.4
//!     dimension lookup AND a fresh modular sub-bitstream tied to the
//!     stream_index Table H.4; that lands in round 15+ alongside the
//!     IDCT dispatcher that consumes the matrices.
//!
//! Validity: `encoding_mode` must lie in the valid-index list of
//! Table I.5 for the specific matrix slot. Round 14 enforces this.
//!
//! `GetDCTQuantWeights()` (Listing C.10's continuation) is the matrix
//! materialisation step — it consumes the parsed `params` arrays and
//! produces dequantization matrices via the `Interpolate` formula. The
//! matrices are not materialised in round 14: HF coefficient decode +
//! IDCT (the only consumers) defer to round 15+. The parsed
//! `EncodingModeParams` are stored on `HfGlobal` so the next round can
//! pick them up directly.
//!
//! ## C.6.4 / I.2.6 num_hf_presets
//!
//! Always read after the dequant-matrix bundle:
//! `num_hf_presets_minus_1 = u(ceil(log2(num_groups)))`, where
//! `num_hf_presets = num_hf_presets_minus_1 + 1`. For single-group
//! frames the field uses 0 bits.

use oxideav_core::{Error, Result};

use crate::bitreader::{BitReader, U32Dist};
use crate::global_modular::{apply_inverse_transforms, apply_transforms_to_channel_layout};
use crate::modular_fdis::{
    decode_channels_at_stream, ChannelDesc, MaTreeFdis, TransformInfo, WpHeader,
};

/// Context required to decode `RAW`-mode dequantization matrices: their
/// Table I.4 modular sub-bitstreams reference the frame's LF-group count
/// (stream index `1 + 3 × num_lf_groups + table index` per Table D.2
/// property 1) and may use the GlobalModular MA tree.
pub struct RawDequantContext<'a> {
    /// `frame_header.num_lf_groups()`.
    pub num_lf_groups: u64,
    /// The GlobalModular global MA tree, if the frame decoded one.
    pub global_tree: Option<&'a MaTreeFdis>,
}

/// Encoding-mode discriminator per Table I.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingMode {
    Library = 0,
    Hornuss = 1,
    Dct2 = 2,
    Dct4 = 3,
    Dct4x8 = 4,
    Afv = 5,
    Dct = 6,
    Raw = 7,
}

impl EncodingMode {
    fn from_u3(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::Library,
            1 => Self::Hornuss,
            2 => Self::Dct2,
            3 => Self::Dct4,
            4 => Self::Dct4x8,
            5 => Self::Afv,
            6 => Self::Dct,
            7 => Self::Raw,
            _ => {
                return Err(Error::InvalidData(format!(
                    "JXL HfGlobal: encoding_mode {v} out of range [0, 7]"
                )));
            }
        })
    }
}

/// Valid `encoding_mode` indices per Table I.5 for each of the 17
/// dequantization-matrix slots (Table I.4 ordering).
///
/// Each row lists which mode discriminators are allowed for that slot.
/// `0` (Library) and `6` (DCT) and `7` (RAW) are listed as `all` in
/// Table I.5; the per-mode `Valid index` column constrains the others.
pub const VALID_ENCODING_MODES: [&[u32]; 17] = [
    // 0  DCT8x8 — DCT, RAW, Library only.
    &[0, 6, 7],
    // 1  Hornuss — Hornuss only (1) plus Library/DCT/RAW.
    &[0, 1, 6, 7],
    // 2  DCT2x2 — DCT2 (2) plus Library/DCT/RAW.
    &[0, 2, 6, 7],
    // 3  DCT4x4 — DCT4 (3) plus Library/DCT/RAW.
    &[0, 3, 6, 7],
    // 4  DCT16x16 — DCT, Library, RAW only.
    &[0, 6, 7],
    // 5  DCT32x32 — DCT, Library, RAW only.
    &[0, 6, 7],
    // 6  DCT16x8/DCT8x16 — DCT, Library, RAW only.
    &[0, 6, 7],
    // 7  DCT32x8/DCT8x32 — DCT, Library, RAW only.
    &[0, 6, 7],
    // 8  DCT16x32/DCT32x16 — DCT, Library, RAW only.
    &[0, 6, 7],
    // 9  DCT4x8/DCT8x4 — DCT4x8 (4) plus Library/DCT/RAW.
    &[0, 4, 6, 7],
    // 10 AFV0..3 — AFV (5) plus Library/DCT/RAW.
    &[0, 5, 6, 7],
    // 11 DCT64x64 — DCT, Library, RAW.
    &[0, 6, 7],
    // 12 DCT32x64/DCT64x32 — DCT, Library, RAW.
    &[0, 6, 7],
    // 13 DCT128x128 — DCT, Library, RAW.
    &[0, 6, 7],
    // 14 DCT64x128/DCT128x64 — DCT, Library, RAW.
    &[0, 6, 7],
    // 15 DCT256x256 — DCT, Library, RAW.
    &[0, 6, 7],
    // 16 DCT128x256/DCT256x128 — DCT, Library, RAW.
    &[0, 6, 7],
];

/// Parsed parameters for a single dequantization matrix slot per Listing
/// C.10. Stored as `f32` because each F16() field decodes via the
/// crate's existing F16-to-f32 helper.
#[derive(Debug, Clone)]
pub struct DequantMatrixParams {
    /// The encoding-mode discriminator for this slot.
    pub mode: EncodingMode,
    /// The base parameter matrix (`3 × N` row-major, where N depends on
    /// the mode). Empty for `Library`, `Dct`, and `Raw`.
    pub params: Vec<f32>,
    /// Number of columns in `params` (3 for Hornuss, 6 for DCT2, 2 for
    /// DCT4, 1 for DCT4x8, 9 for AFV; 0 otherwise).
    pub params_cols: u32,
    /// `dct_params` from `ReadDctParams()` — present for `Dct`,
    /// `Dct4`, `Dct4x8`, `Afv`. `dct_params` is `3 × num_params` row-
    /// major.
    pub dct_params: Vec<f32>,
    pub dct_params_cols: u32,
    /// `dct4x4_params` from a second `ReadDctParams()` — present only
    /// for `Afv`.
    pub dct4x4_params: Vec<f32>,
    pub dct4x4_params_cols: u32,
    /// RAW-mode denominator (F16). Present only for `Raw`.
    pub raw_denominator: f32,
    /// RAW-mode quantization factors: the three per-channel integer
    /// matrices decoded from the Table I.4 modular sub-bitstream, each
    /// row-major of the slot's matrix shape. Empty for every other
    /// mode. Per I.2.4 the dequantization matrix is the element-wise
    /// `raw_params × raw_denominator`; for a lossless JPEG transcode
    /// these integers ARE the original JPEG quantization table entries
    /// (18181-2 A.7).
    pub raw_params: Vec<Vec<i32>>,
}

impl Default for DequantMatrixParams {
    fn default() -> Self {
        Self {
            mode: EncodingMode::Library,
            params: Vec::new(),
            params_cols: 0,
            dct_params: Vec::new(),
            dct_params_cols: 0,
            dct4x4_params: Vec::new(),
            dct4x4_params_cols: 0,
            raw_denominator: 0.0,
            raw_params: Vec::new(),
        }
    }
}

/// Read a single 3 × `cols` matrix in raster order (`3 * cols` F16s)
/// where the first column (col 0) of every row is multiplied by 64
/// per Listing C.10's `params(i, 0) *= 64` post-pass.
///
/// `multiply_first_col_by_64` reflects Listing C.10's per-matrix-mode
/// post-processing rule. For Hornuss / DCT2 the listing multiplies all
/// elements by 64; for DCT4, AFV (cols 0..5), DCT4x8 only column 0;
/// for ReadDctParams it multiplies col 0 only.
fn read_3xn_f16_matrix(
    br: &mut BitReader<'_>,
    cols: u32,
    scale_all_by_64: bool,
) -> Result<Vec<f32>> {
    if cols == 0 {
        return Ok(Vec::new());
    }
    let total = (3 * cols) as usize;
    let mut v = Vec::with_capacity(total);
    for i in 0..3u32 {
        for j in 0..cols {
            let f = br.read_f16()?;
            let scaled = if scale_all_by_64 || j == 0 {
                f * 64.0
            } else {
                f
            };
            let _ = i;
            v.push(scaled);
        }
    }
    Ok(v)
}

/// Read an AFV-mode 3 × 9 matrix where cols 0..=5 (inclusive) are
/// multiplied by 64 (per Listing C.10's AFV branch:
/// `for (i = 0; i < 3; i++) for (j = 0; j < 6; j++) params(i, j) *= 64`).
fn read_afv_matrix(br: &mut BitReader<'_>) -> Result<Vec<f32>> {
    let mut v = Vec::with_capacity(3 * 9);
    for _i in 0..3 {
        for j in 0..9u32 {
            let f = br.read_f16()?;
            let scaled = if j < 6 { f * 64.0 } else { f };
            v.push(scaled);
        }
    }
    Ok(v)
}

/// `ReadDctParams()` per Listing C.10:
///
/// ```text
/// num_params = u(4) + 1;
/// vals = /* read 3 x num_params matrix in raster order */;
/// for (i = 0; i < 3; i++) vals(i, 0) *= 64;
/// return vals;
/// ```
fn read_dct_params(br: &mut BitReader<'_>) -> Result<(Vec<f32>, u32)> {
    let num_params = br.read_bits(4)? + 1;
    if num_params > 16 {
        // u(4) ranges 0..=15 → num_params ≤ 16 by construction; defensive.
        return Err(Error::InvalidData(format!(
            "JXL HfGlobal ReadDctParams: num_params {num_params} > 16"
        )));
    }
    // Standard 3 × num_params, only col 0 scaled by 64.
    let v = read_3xn_f16_matrix(br, num_params, false)?;
    Ok((v, num_params))
}

/// Decode a single dequantization-matrix slot's parameters per Listing
/// C.10. `slot_index` (0..17) gates the `encoding_mode` against
/// [`VALID_ENCODING_MODES`].
fn read_one_dequant_matrix(
    br: &mut BitReader<'_>,
    slot_index: usize,
    raw_ctx: Option<&RawDequantContext<'_>>,
) -> Result<DequantMatrixParams> {
    let raw = br.read_bits(3)?;
    let mode = EncodingMode::from_u3(raw)?;

    // Validate against per-slot allowed modes per Table I.5.
    if !VALID_ENCODING_MODES[slot_index].contains(&raw) {
        return Err(Error::InvalidData(format!(
            "JXL HfGlobal: encoding_mode {raw} not in valid set for matrix slot {slot_index}: \
             {:?}",
            VALID_ENCODING_MODES[slot_index]
        )));
    }

    let mut out = DequantMatrixParams {
        mode,
        ..Default::default()
    };

    match mode {
        EncodingMode::Library => {
            // No params; the slot uses the default from Table I.6.
        }
        EncodingMode::Hornuss => {
            // 3 × 3 F16 matrix, ALL elements multiplied by 64.
            out.params = read_3xn_f16_matrix(br, 3, true)?;
            out.params_cols = 3;
        }
        EncodingMode::Dct2 => {
            // 3 × 6 F16 matrix, ALL elements multiplied by 64.
            out.params = read_3xn_f16_matrix(br, 6, true)?;
            out.params_cols = 6;
        }
        EncodingMode::Dct4 => {
            // 3 × 2 F16 matrix, only col 0 scaled. Then ReadDctParams.
            out.params = read_3xn_f16_matrix(br, 2, false)?;
            out.params_cols = 2;
            let (dct, n) = read_dct_params(br)?;
            out.dct_params = dct;
            out.dct_params_cols = n;
        }
        EncodingMode::Dct4x8 => {
            // 3 × 1 F16 matrix; ReadDctParams.
            out.params = read_3xn_f16_matrix(br, 1, false)?;
            out.params_cols = 1;
            let (dct, n) = read_dct_params(br)?;
            out.dct_params = dct;
            out.dct_params_cols = n;
        }
        EncodingMode::Afv => {
            // 3 × 9 F16 matrix, cols 0..5 × 64; ReadDctParams; ReadDctParams (dct4x4).
            out.params = read_afv_matrix(br)?;
            out.params_cols = 9;
            let (dct, n) = read_dct_params(br)?;
            out.dct_params = dct;
            out.dct_params_cols = n;
            let (dct4x4, n2) = read_dct_params(br)?;
            out.dct4x4_params = dct4x4;
            out.dct4x4_params_cols = n2;
        }
        EncodingMode::Dct => {
            let (dct, n) = read_dct_params(br)?;
            out.dct_params = dct;
            out.dct_params_cols = n;
        }
        EncodingMode::Raw => {
            // RAW mode (Listing C.10): `denominator = F16()`, then
            // `ZeroPadToByte()`, then a 3-channel modular sub-bitstream
            // (C.9) of the same shape as the required quant matrix
            // (Table I.4 dims for this slot). Its MA-tree stream index
            // is `1 + 3 × num_lf_groups + table index` (Table D.2
            // property 1, "RAW dequantization tables"), with the table
            // index being this slot's index.
            let ctx = raw_ctx.ok_or_else(|| {
                Error::Unsupported(format!(
                    "JXL HfGlobal: dequant-matrix slot {slot_index} uses RAW encoding mode but \
                     the caller supplied no RawDequantContext (frame-level decode required)"
                ))
            })?;
            out.raw_denominator = br.read_f16()?;
            // Listing C.10 prints a ZeroPadToByte() here, but the wire
            // carries NONE: on a stream whose cursor is unaligned at
            // this point, skipping to the byte boundary misparses the
            // modular sub-bitstream header (nb_transforms reads 7 and
            // the first transform reads as a Squeeze with end 1679),
            // while reading the ModularHeader at the very next bit
            // parses cleanly and yields the correct quantization
            // tables. Wire-arbitrated on two independently generated
            // transcode streams; an aligned stream parses identically
            // under both readings. Recorded as an FDIS erratum
            // candidate.

            let (x_dim, y_dim) =
                crate::dct_quant_weights::weights_matrix_dims_for_slot(slot_index as u32)?;
            let descs: Vec<ChannelDesc> = (0..3)
                .map(|_| ChannelDesc {
                    width: x_dim,
                    height: y_dim,
                    hshift: 0,
                    vshift: 0,
                })
                .collect();

            // Inner ModularHeader (Table H.1) per Annex H.2.
            let use_global_tree = br.read_bool()?;
            let wp_header = WpHeader::read(br)?;
            let nb_transforms = br.read_u32([
                U32Dist::Val(0),
                U32Dist::Val(1),
                U32Dist::BitsOffset(4, 2),
                U32Dist::BitsOffset(8, 18),
            ])?;
            const MAX_TRANSFORMS: u32 = 274;
            if nb_transforms > MAX_TRANSFORMS {
                return Err(Error::InvalidData(format!(
                    "JXL HfGlobal: RAW dequant slot {slot_index}: nb_transforms {nb_transforms} \
                     exceeds {MAX_TRANSFORMS}"
                )));
            }
            let mut transforms: Vec<TransformInfo> = Vec::with_capacity(nb_transforms as usize);
            for _ in 0..nb_transforms {
                transforms.push(TransformInfo::read(br)?);
            }
            let mut tree = if use_global_tree {
                ctx.global_tree
                    .ok_or_else(|| {
                        Error::InvalidData(format!(
                            "JXL HfGlobal: RAW dequant slot {slot_index} wants the global MA \
                             tree but GlobalModular decoded none"
                        ))
                    })?
                    .cloned_with_fresh_state()
            } else {
                MaTreeFdis::read(br)?
            };
            let stream_index = (1 + 3 * ctx.num_lf_groups + slot_index as u64) as i32;
            let mut descs = descs;
            if !transforms.is_empty() {
                descs = apply_transforms_to_channel_layout(descs, &mut transforms)?;
            }
            let mut img =
                decode_channels_at_stream(br, &descs, &mut tree, &wp_header, stream_index)?;
            if !transforms.is_empty() {
                // The RAW params are quantization factors; the modular
                // bit_depth argument only affects paths these tiny
                // integer images do not exercise.
                apply_inverse_transforms(&mut img, &transforms, 8)?;
            }
            if img.channels.len() != 3 {
                return Err(Error::InvalidData(format!(
                    "JXL HfGlobal: RAW dequant slot {slot_index} decoded {} channels (want 3)",
                    img.channels.len()
                )));
            }
            out.raw_params = img.channels;
        }
    }

    Ok(out)
}

/// `HfGlobal` bundle — Table C.17. For a `kVarDCT` frame the bundle
/// contains:
///
/// * **Dequantization matrices** (I.2.4). Round 14 supports both the
///   `u(1) == 1` default-fast-path AND the non-default-encoding branch
///   (17 × Listing C.10 `encoding_mode` reads). RAW mode within the
///   non-default branch defers to round 15+.
/// * **Number of HF decoding presets** (I.2.6). Always read.
#[derive(Debug, Clone)]
pub struct HfGlobal {
    /// `true` when the codestream signaled the I.2.4 default-encoding
    /// fast path (`u(1) == 1`). `false` when 17 per-matrix encoding
    /// modes were parsed.
    pub dequant_default: bool,
    /// Per-slot encoded parameters when `dequant_default == false`. Empty
    /// when `dequant_default == true` (every slot uses Table I.6
    /// defaults).
    pub dequant_matrices: Vec<DequantMatrixParams>,
    /// `num_hf_presets` per I.2.6. The codestream encodes
    /// `num_hf_presets - 1` so this value is at least 1.
    pub num_hf_presets: u32,
}

impl HfGlobal {
    /// Decode the HfGlobal bundle (Table C.17). The caller has positioned
    /// `br` at the start of the HfGlobal TOC slot AND verified that
    /// `frame_header.encoding == kVarDCT` (the bundle is empty for
    /// `kModular`).
    ///
    /// `num_groups` parameterises the bit-count of `num_hf_presets - 1`
    /// per I.2.6: `u(ceil(log2(num_groups)))`. For single-group frames
    /// `num_groups == 1` and the field uses 0 bits (legal value: 0 →
    /// `num_hf_presets = 1`).
    pub fn read(br: &mut BitReader<'_>, num_groups: u64) -> Result<Self> {
        Self::read_with_raw(br, num_groups, None)
    }

    /// [`HfGlobal::read`] with the context that lets `RAW`-mode
    /// dequantization matrices decode their Table I.4 modular
    /// sub-bitstreams. Without a context (`None`, the [`HfGlobal::read`]
    /// behaviour) a RAW slot refuses loudly.
    pub fn read_with_raw(
        br: &mut BitReader<'_>,
        num_groups: u64,
        raw_ctx: Option<&RawDequantContext<'_>>,
    ) -> Result<Self> {
        // I.2.4 first sentence: read u(1). When 1, all dequant matrices
        // take their default encoding from I.2.5 / Table I.6.
        let dequant_default = br.read_bool()?;
        let dequant_matrices = if dequant_default {
            Vec::new()
        } else {
            // 17 sets of encoding_mode + per-mode parameters per
            // Listing C.10.
            let mut v = Vec::with_capacity(17);
            for slot in 0..17 {
                v.push(read_one_dequant_matrix(br, slot, raw_ctx)?);
            }
            v
        };

        // I.2.6: num_hf_presets_minus_1 = u(ceil(log2(num_groups))).
        // For num_groups == 0 the spec implicitly forbids the case (a
        // VarDCT frame must have at least one group); be defensive.
        if num_groups == 0 {
            return Err(Error::InvalidData(
                "JXL HfGlobal: num_groups = 0 (a VarDCT frame must have at least one group)".into(),
            ));
        }
        let nbits = ceil_log2_u64(num_groups);
        let num_hf_presets_minus_1 = if nbits == 0 { 0 } else { br.read_bits(nbits)? };
        // num_hf_presets = num_hf_presets_minus_1 + 1.
        let num_hf_presets = num_hf_presets_minus_1
            .checked_add(1)
            .ok_or_else(|| Error::InvalidData("JXL HfGlobal: num_hf_presets overflow".into()))?;
        if (num_hf_presets as u64) > num_groups {
            return Err(Error::InvalidData(format!(
                "JXL HfGlobal: num_hf_presets {num_hf_presets} exceeds num_groups {num_groups}"
            )));
        }
        Ok(Self {
            dequant_default,
            dequant_matrices,
            num_hf_presets,
        })
    }
}

/// `ceil(log2(n))` for `n >= 1`. `0` when `n == 1`.
fn ceil_log2_u64(n: u64) -> u32 {
    if n <= 1 {
        return 0;
    }
    64 - (n - 1).leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ans::test_helpers::pack_lsb;

    #[test]
    fn hf_global_default_fast_path_one_group() {
        // u(1) = 1 (default), num_groups = 1 → no bits for
        // num_hf_presets_minus_1, value = 0 → num_hf_presets = 1.
        let bytes = pack_lsb(&[(1, 1)]);
        let mut br = BitReader::new(&bytes);
        let hf = HfGlobal::read(&mut br, 1).unwrap();
        assert!(hf.dequant_default);
        assert_eq!(hf.num_hf_presets, 1);
        assert!(hf.dequant_matrices.is_empty());
    }

    #[test]
    fn hf_global_default_fast_path_four_groups() {
        let bytes = pack_lsb(&[(1, 1), (2, 2)]);
        let mut br = BitReader::new(&bytes);
        let hf = HfGlobal::read(&mut br, 4).unwrap();
        assert!(hf.dequant_default);
        assert_eq!(hf.num_hf_presets, 3);
    }

    #[test]
    fn hf_global_non_default_all_library_one_group() {
        // u(1) = 0 → non-default. 17 matrices, every one uses
        // encoding_mode = Library (0) which is `u(3) = 0`. Total bits =
        // 1 + 17*3 = 52. num_groups = 1 → no preset bits.
        let mut bits: Vec<(u32, u32)> = vec![(0, 1)];
        for _ in 0..17 {
            bits.push((0, 3)); // Library
        }
        let bytes = pack_lsb(&bits);
        let mut br = BitReader::new(&bytes);
        let hf = HfGlobal::read(&mut br, 1).unwrap();
        assert!(!hf.dequant_default);
        assert_eq!(hf.dequant_matrices.len(), 17);
        for m in &hf.dequant_matrices {
            assert_eq!(m.mode, EncodingMode::Library);
            assert!(m.params.is_empty());
            assert!(m.dct_params.is_empty());
        }
        assert_eq!(hf.num_hf_presets, 1);
    }

    #[test]
    fn hf_global_non_default_dct_mode_for_dct8x8() {
        // u(1) = 0; slot 0 (DCT8x8) uses encoding_mode = DCT (6).
        // ReadDctParams: num_params = 0 + 1 = 1, then 3 × 1 F16 values.
        // For testing: F16(0.0) = 0x0000, F16(1.0) = 0x3C00, F16(2.0) = 0x4000.
        // Following 16 slots use Library to keep test small.
        let mut bits: Vec<(u32, u32)> = vec![
            (0, 1),       // u(1) = 0 → non-default
            (6, 3),       // slot 0: DCT
            (0, 4),       // num_params - 1 = 0 → num_params = 1
            (0x3C00, 16), // F16(1.0)
            (0x4000, 16), // F16(2.0)
            (0x4400, 16), // F16(4.0)
        ];
        for _ in 1..17 {
            bits.push((0, 3));
        }
        let bytes = pack_lsb(&bits);
        let mut br = BitReader::new(&bytes);
        let hf = HfGlobal::read(&mut br, 1).unwrap();
        let m0 = &hf.dequant_matrices[0];
        assert_eq!(m0.mode, EncodingMode::Dct);
        assert_eq!(m0.dct_params_cols, 1);
        // 3 × 1 = 3 elements; col 0 ×64 each → 64.0, 128.0, 256.0.
        assert_eq!(m0.dct_params, vec![64.0, 128.0, 256.0]);
    }

    #[test]
    fn hf_global_non_default_invalid_mode_for_slot() {
        // Slot 0 (DCT8x8) accepts only DCT/Library/RAW. Try Hornuss (1)
        // → invalid for slot 0.
        let bits: Vec<(u32, u32)> = vec![
            (0, 1), // u(1) = 0 → non-default
            (1, 3), // slot 0: Hornuss → invalid
        ];
        let bytes = pack_lsb(&bits);
        let mut br = BitReader::new(&bytes);
        let r = HfGlobal::read(&mut br, 1);
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn hf_global_non_default_raw_mode_unsupported() {
        // Slot 0 (DCT8x8) with RAW (7) — defers to round 15+.
        let bits: Vec<(u32, u32)> = vec![
            (0, 1),
            (7, 3), // slot 0: RAW → Unsupported (round 15+)
        ];
        let bytes = pack_lsb(&bits);
        let mut br = BitReader::new(&bytes);
        let r = HfGlobal::read(&mut br, 1);
        assert!(matches!(r, Err(Error::Unsupported(_))));
    }

    #[test]
    fn hf_global_default_fast_path_three_groups() {
        let bytes = pack_lsb(&[(1, 1), (0, 2)]);
        let mut br = BitReader::new(&bytes);
        let hf = HfGlobal::read(&mut br, 3).unwrap();
        assert_eq!(hf.num_hf_presets, 1);
    }

    #[test]
    fn hf_global_num_groups_zero_rejected() {
        let bytes = pack_lsb(&[(1, 1)]);
        let mut br = BitReader::new(&bytes);
        let r = HfGlobal::read(&mut br, 0);
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn hf_global_num_hf_presets_exceeds_num_groups_rejected() {
        let bytes = pack_lsb(&[(1, 1), (3, 2)]);
        let mut br = BitReader::new(&bytes);
        let r = HfGlobal::read(&mut br, 3);
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn ceil_log2_edges() {
        assert_eq!(ceil_log2_u64(0), 0);
        assert_eq!(ceil_log2_u64(1), 0);
        assert_eq!(ceil_log2_u64(2), 1);
        assert_eq!(ceil_log2_u64(3), 2);
        assert_eq!(ceil_log2_u64(4), 2);
        assert_eq!(ceil_log2_u64(8), 3);
        assert_eq!(ceil_log2_u64(9), 4);
    }

    #[test]
    fn hf_global_non_default_hornuss_for_slot_1() {
        // slot 1 (Hornuss) accepts Hornuss (1).
        // Hornuss reads 3 × 3 F16, all multiplied by 64.
        // Use F16(1.0) for everything → 1.0 × 64 = 64.0 per element.
        let mut bits: Vec<(u32, u32)> = vec![(0, 1)];
        bits.push((0, 3)); // slot 0: Library
        bits.push((1, 3)); // slot 1: Hornuss
        for _ in 0..9 {
            bits.push((0x3C00, 16)); // F16(1.0)
        }
        for _ in 2..17 {
            bits.push((0, 3)); // remaining slots: Library
        }
        let bytes = pack_lsb(&bits);
        let mut br = BitReader::new(&bytes);
        let hf = HfGlobal::read(&mut br, 1).unwrap();
        let m1 = &hf.dequant_matrices[1];
        assert_eq!(m1.mode, EncodingMode::Hornuss);
        assert_eq!(m1.params_cols, 3);
        assert_eq!(m1.params, vec![64.0; 9]);
    }
}
