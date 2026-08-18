//! JPEG bitstream reconstruction — ISO/IEC 18181-2:2024 Annex A.
//!
//! A losslessly recompressed JPEG XL file (`jbrd` box present) carries
//! everything needed to regenerate the *original* JPEG file byte for
//! byte: the [`crate::jpeg_bitstream::JpegBitstreamData`] side data plus
//! the codestream itself, whose VarDCT frame stores the JPEG's quantized
//! DCT coefficients (DC in the §C.5.3 quantized-LF channels, AC in the
//! §C.8.3 HF-coefficient streams) and whose §I.2.4 `RAW`-mode
//! dequantization matrices store the JPEG quantization tables.
//!
//! [`decode_transcoded_coefficients`] is the coefficient-level decode
//! driver: it walks the frame sections exactly like the pixel decoder
//! (LfGlobal → LfGroups → HfGlobal section → per-group §C.8.3 entropy
//! decode) but stops at the quantized integers — no dequantization, no
//! IDCT, no colour transform. [`reconstruct_jpeg`] then runs the Annex A
//! procedure: the implicit SOI, one segment per element of the `jbrd`
//! marker array (A.2 SOF, A.3 DHT, A.4 RSTn, A.5 EOI, A.6 SOS with the
//! ISO/IEC 10918-1 entropy re-encode, A.7 DQT, A.8 DRI, A.9 APPn, A.10
//! COM, A.11 unrecognized data), and byte-exactness is the only accepted
//! outcome — every committed fixture pins the output against the
//! original JPEG bytes.
//!
//! Current scope: 3-component YCbCr without chroma subsampling
//! (`jpeg_upsampling == [0, 0, 0]`), single-pass frames, sequential DCT
//! scans. Everything else refuses loudly.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::block_context_resolver::BlockContextResolver;
use crate::container::{JxlFile, MetadataKind};
use crate::dct_select::TransformType;
use crate::frame_header::{Encoding, FrameDecodeParams};
use crate::jpeg_bitstream::{HuffmanCode, JpegBitstreamData, ScanInfo, ScanMoreInfo};
use crate::lf_global::LfGlobal;
use crate::metadata_fdis::{ImageMetadataFdis, SizeHeaderFdis};
use crate::multi_pass_hf_header::PerPassHfHeaders;
use crate::per_pass_non_zeros::PerPassNonZerosGrids;

/// The zig-zag scan order of ISO/IEC 10918-1 (Figure 5): `ZIGZAG[k]`
/// is the raster index (`row * 8 + col`) of zig-zag position `k`.
const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, //
    17, 24, 32, 25, 18, 11, 4, 5, //
    12, 19, 26, 33, 40, 48, 41, 34, //
    27, 20, 13, 6, 7, 14, 21, 28, //
    35, 42, 49, 56, 57, 50, 43, 36, //
    29, 22, 15, 23, 30, 37, 44, 51, //
    58, 59, 52, 45, 38, 31, 39, 46, //
    53, 60, 61, 54, 47, 55, 62, 63,
];

/// Quantized DCT coefficients recovered from a lossless JPEG→JXL
/// transcode, plus the JPEG quantization tables the codestream carries.
#[derive(Debug)]
pub struct TranscodedCoefficients {
    /// Image width / height in pixels (SizeHeader).
    pub width: u32,
    pub height: u32,
    /// FrameHeader `jpeg_upsampling` (channel order Cb, Y, Cr).
    pub jpeg_upsampling: [u32; 3],
    /// Full-resolution 8×8-block grid dimensions.
    pub bw: usize,
    pub bh: usize,
    /// Per JXL channel `(width, height)` in blocks — equal to
    /// `(bw, bh)` for a 4:4:4 frame, halved per axis for a channel
    /// subsampled by `jpeg_upsampling`.
    pub cdims: [(usize, usize); 3],
    /// Per JXL channel (0 = X/Cb, 1 = Y/luma, 2 = B/Cr): `bw*bh` blocks
    /// in raster order, 64 coefficients per block in ISO/IEC 10918-1
    /// raster `(row, col)` order (transposed from the JXL cell layout,
    /// see the decode driver), DC at index 0, chroma re-correlated
    /// (the stored CfL residuals are already undone).
    pub coeffs: [Vec<i32>; 3],
    /// Per JXL channel: the 64 RAW quantization factors (10918-1
    /// raster order) from the §I.2.4 RAW dequant matrices (18181-2
    /// A.7's `Q_k`).
    pub quant: [Vec<i32>; 3],
    /// §C.5.4 chroma-from-luma tile factors (64×64-px tiles), frame
    /// level, raster order.
    pub x_from_y: Vec<i32>,
    pub b_from_y: Vec<i32>,
    /// Tile-grid width.
    pub tw: usize,
    /// §C.4.4 colour correlation constants.
    pub colour_factor: u32,
}

/// Decode the quantized DCT coefficients of a single-frame VarDCT
/// lossless JPEG transcode. `codestream` starts AFTER the 2-byte
/// `FF 0A` signature.
pub fn decode_transcoded_coefficients(codestream: &[u8]) -> Result<TranscodedCoefficients> {
    let mut br = BitReader::new(codestream);
    let size = SizeHeaderFdis::read(&mut br)?;
    let metadata = ImageMetadataFdis::read(&mut br)?;
    if metadata.xyb_encoded {
        return Err(Error::Unsupported(
            "JXL jpeg_reconstruct: frame is xyb_encoded (not a JPEG transcode)".into(),
        ));
    }
    // Headers → frame boundary is byte-aligned (§6.3).
    br.pu()?;

    let fh_params = FrameDecodeParams {
        xyb_encoded: metadata.xyb_encoded,
        num_extra_channels: metadata.num_extra_channels,
        have_animation: metadata.have_animation,
        have_animation_timecodes: metadata
            .animation
            .map(|a| a.have_timecodes)
            .unwrap_or(false),
        image_width: size.width,
        image_height: size.height,
    };
    let (fh, toc) = crate::read_frame_header_and_toc(&mut br, &fh_params, codestream)?;

    if fh.encoding != Encoding::VarDct || !fh.do_ycbcr {
        return Err(Error::Unsupported(
            "JXL jpeg_reconstruct: transcode frame must be VarDCT + YCbCr".into(),
        ));
    }
    if !fh.is_last || fh.have_crop || fh.frame_type != crate::frame_header::FrameType::Regular {
        return Err(Error::Unsupported(
            "JXL jpeg_reconstruct: transcode must be a single full regular frame".into(),
        ));
    }
    if fh.passes.num_passes != 1 {
        return Err(Error::Unsupported(format!(
            "JXL jpeg_reconstruct: {} passes (only single-pass transcodes handled)",
            fh.passes.num_passes
        )));
    }
    let shifts = fh.jpeg_upsampling_shifts();
    let subsampled = shifts.iter().any(|&(h, v)| h != 0 || v != 0);
    if fh.upsampling != 1 {
        return Err(Error::Unsupported(
            "JXL jpeg_reconstruct: upsampling != 1 in a transcode frame".into(),
        ));
    }

    let num_groups = fh.num_groups();
    let num_lf_groups = fh.num_lf_groups();
    let frame_data_start = br.bytes_consumed();
    let frame_bytes = &codestream[frame_data_start..];
    let total_frame_len: u64 = toc.entries.iter().map(|&e| e as u64).sum();
    if total_frame_len > frame_bytes.len() as u64 {
        return Err(Error::InvalidData(
            "JXL jpeg_reconstruct: TOC overruns codestream".into(),
        ));
    }
    let section_bytes = |idx: usize| -> Result<&[u8]> {
        let start = *toc.group_offsets.get(idx).ok_or_else(|| {
            Error::InvalidData(format!("JXL jpeg_reconstruct: TOC slot {idx} out of range"))
        })? as usize;
        let len = toc.entries[idx] as usize;
        frame_bytes.get(start..start + len).ok_or_else(|| {
            Error::InvalidData(format!(
                "JXL jpeg_reconstruct: section {idx} range exceeds frame bytes"
            ))
        })
    };

    let single_toc = toc.entries.len() == 1
        && num_groups == 1
        && fh.passes.num_passes == 1
        && num_lf_groups == 1;

    // Frame-level canvases. Per channel: the full-resolution block
    // grid divided (rounding up) by the channel's subsampling lattice.
    let fbw = fh.width.div_ceil(8) as usize;
    let fbh = fh.height.div_ceil(8) as usize;
    let cdims: [(usize, usize); 3] = [
        (
            fbw.div_ceil(1 << shifts[0].0),
            fbh.div_ceil(1 << shifts[0].1),
        ),
        (
            fbw.div_ceil(1 << shifts[1].0),
            fbh.div_ceil(1 << shifts[1].1),
        ),
        (
            fbw.div_ceil(1 << shifts[2].0),
            fbh.div_ceil(1 << shifts[2].1),
        ),
    ];
    let mut lf_quant: [Vec<i32>; 3] = [
        vec![0i32; cdims[0].0 * cdims[0].1],
        vec![0i32; cdims[1].0 * cdims[1].1],
        vec![0i32; cdims[2].0 * cdims[2].1],
    ];
    let mut cells = vec![crate::dct_select::DctSelectCell::Empty; fbw * fbh];
    let mut hf_mul_grid = vec![0i32; fbw * fbh];
    let ftw = fh.width.div_ceil(64) as usize;
    let fth = fh.height.div_ceil(64) as usize;
    let mut x_from_y = vec![0i32; ftw * fth];
    let mut b_from_y = vec![0i32; ftw * fth];

    // Section walk (mirrors the pixel decoder's layout logic).
    let (lf_global, lf_groups, mut hf_section, pass_group_readers) = if single_toc {
        let mut shared = BitReader::new_section(section_bytes(0)?);
        let lf_global = LfGlobal::read(&mut shared, &fh, &metadata)?;
        let lf_group = crate::lf_group::LfGroup::read(&mut shared, &fh, &lf_global, &metadata, 0)?;
        let nb_block_ctx = require_hbc(&lf_global)?.nb_block_ctx;
        let raw_ctx = crate::hf_global::RawDequantContext {
            num_lf_groups,
            global_tree: lf_global.global_modular.global_tree.as_ref(),
        };
        let hf_section = crate::hf_global_section::HfGlobalSection::read_with_raw(
            &mut shared,
            num_groups,
            nb_block_ctx,
            1,
            Some(&raw_ctx),
        )?;
        (lf_global, vec![lf_group], hf_section, vec![vec![shared]])
    } else {
        let mut lf_br = BitReader::new_section(section_bytes(0)?);
        let lf_global = LfGlobal::read(&mut lf_br, &fh, &metadata)?;
        let mut lf_groups = Vec::with_capacity(num_lf_groups as usize);
        for lg in 0..num_lf_groups {
            let mut lg_br = BitReader::new_section(section_bytes(1 + lg as usize)?);
            lf_groups.push(crate::lf_group::LfGroup::read(
                &mut lg_br, &fh, &lf_global, &metadata, lg as u32,
            )?);
        }
        let nb_block_ctx = require_hbc(&lf_global)?.nb_block_ctx;
        let raw_ctx = crate::hf_global::RawDequantContext {
            num_lf_groups,
            global_tree: lf_global.global_modular.global_tree.as_ref(),
        };
        let mut hg_br = BitReader::new_section(section_bytes(1 + num_lf_groups as usize)?);
        let hf_section = crate::hf_global_section::HfGlobalSection::read_with_raw(
            &mut hg_br,
            num_groups,
            nb_block_ctx,
            1,
            Some(&raw_ctx),
        )?;
        let mut readers = Vec::with_capacity(num_groups as usize);
        for g in 0..num_groups {
            readers.push(vec![BitReader::new_section(section_bytes(
                2 + num_lf_groups as usize + g as usize,
            )?)]);
        }
        (lf_global, lf_groups, hf_section, readers)
    };

    // RAW quant tables: slot 0 (DCT8×8) carries the JPEG tables.
    let hg = &hf_section.hf_global;
    if hg.dequant_default {
        return Err(Error::Unsupported(
            "JXL jpeg_reconstruct: dequant matrices are defaults, not RAW JPEG tables".into(),
        ));
    }
    let slot0 = hg.dequant_matrices.first().ok_or_else(|| {
        Error::InvalidData("JXL jpeg_reconstruct: no dequant matrix slots".into())
    })?;
    if slot0.raw_params.len() != 3 || slot0.raw_params.iter().any(|c| c.len() != 64) {
        return Err(Error::Unsupported(
            "JXL jpeg_reconstruct: DCT8×8 dequant slot is not a RAW 3×64 table".into(),
        ));
    }
    let quant: [Vec<i32>; 3] = [
        slot0.raw_params[0].clone(),
        slot0.raw_params[1].clone(),
        slot0.raw_params[2].clone(),
    ];

    // CfL must be neutral: reconstruction needs the stored quantized
    // values to BE the JPEG coefficients (chroma-from-luma would fold
    // luma into the stored chroma residuals).
    let cfl = lf_global.lf_channel_correlation.ok_or_else(|| {
        Error::InvalidData("JXL jpeg_reconstruct: LfChannelCorrelation missing".into())
    })?;
    if cfl.base_correlation_x != 0.0 || cfl.base_correlation_b != 0.0 {
        return Err(Error::Unsupported(
            "JXL jpeg_reconstruct: non-zero base chroma-from-luma correlation".into(),
        ));
    }

    // Assemble frame-level quantized LF (= JPEG DC) + DctSelect grids.
    let group_dim_blocks = (fh.group_dim() as usize) * 8 / 8; // group_dim px / 8 px per block
    let lf_cols = fh.width.div_ceil(fh.group_dim() * 8) as usize;
    for (lg_idx, lf_group) in lf_groups.iter().enumerate() {
        let gx = lg_idx % lf_cols;
        let gy = lg_idx / lf_cols;
        let bx0 = gx * group_dim_blocks * 8;
        let by0 = gy * group_dim_blocks * 8;
        let lf_coeff = lf_group.lf_coeff.as_ref().ok_or_else(|| {
            Error::InvalidData("JXL jpeg_reconstruct: LfGroup without LfCoefficients".into())
        })?;
        if lf_coeff.extra_precision != 0 {
            return Err(Error::Unsupported(format!(
                "JXL jpeg_reconstruct: extra_precision {} (JPEG DC must be plain integers)",
                lf_coeff.extra_precision
            )));
        }
        // LfCoefficients channels arrive in modular order (Y, X, B);
        // reorder to channel index order (X, Y, B). Each channel pastes
        // into its own (possibly subsampled) frame-level canvas at the
        // LfGroup offset shifted by that channel's lattice.
        let order = [1usize, 0, 2];
        for c in 0..3 {
            let gcw = lf_coeff.lf_quant_widths[order[c]] as usize;
            let gch = lf_coeff.lf_quant_heights[order[c]] as usize;
            let (hs, vs) = shifts[c];
            let (cw, _ch) = cdims[c];
            let cx0 = bx0 >> hs;
            let cy0 = by0 >> vs;
            let src = &lf_coeff.lf_quant[order[c]];
            if src.len() != gcw * gch {
                return Err(Error::InvalidData(
                    "JXL jpeg_reconstruct: LF channel size mismatch".into(),
                ));
            }
            for row in 0..gch {
                let s = row * gcw;
                let d = (cy0 + row) * cw + cx0;
                lf_quant[c][d..d + gcw].copy_from_slice(&src[s..s + gcw]);
            }
        }
        let hf_meta = lf_group.hf_meta.as_ref().ok_or_else(|| {
            Error::InvalidData("JXL jpeg_reconstruct: LfGroup without HfMetadata".into())
        })?;
        let lf_w = (fh.width - (bx0 as u32) * 8).min(fh.group_dim() * 8 * 8);
        let lf_h = (fh.height - (by0 as u32) * 8).min(fh.group_dim() * 8 * 8);
        // CfL tiles (§C.5.4): paste into the frame-level tile canvases;
        // the transcode's chroma channels are stored CfL-decorrelated
        // and are re-correlated in the integer domain below.
        let gtw = (lf_w.div_ceil(64)) as usize;
        let gth = (lf_h.div_ceil(64)) as usize;
        if hf_meta.x_from_y.len() != gtw * gth || hf_meta.b_from_y.len() != gtw * gth {
            return Err(Error::InvalidData(
                "JXL jpeg_reconstruct: CfL tile plane size mismatch".into(),
            ));
        }
        let tx0 = bx0 / 8;
        let ty0 = by0 / 8;
        for row in 0..gth {
            let s = row * gtw;
            let dcol = (ty0 + row) * ftw + tx0;
            x_from_y[dcol..dcol + gtw].copy_from_slice(&hf_meta.x_from_y[s..s + gtw]);
            b_from_y[dcol..dcol + gtw].copy_from_slice(&hf_meta.b_from_y[s..s + gtw]);
        }
        let g_grid = crate::dct_select::derive_dct_select(hf_meta, lf_w, lf_h)?;
        let gw = g_grid.width_blocks as usize;
        let gh = g_grid.height_blocks as usize;
        for row in 0..gh {
            let s = row * gw;
            let d = (by0 + row) * fbw + bx0;
            cells[d..d + gw].copy_from_slice(&g_grid.cells[s..s + gw]);
            hf_mul_grid[d..d + gw].copy_from_slice(&g_grid.hf_mul[s..s + gw]);
        }
    }
    let dct_grid = crate::dct_select::DctSelectGrid {
        cells,
        hf_mul: hf_mul_grid,
        width_blocks: fbw as u32,
        height_blocks: fbh as u32,
    };

    // §C.8.3 per-group HF decode at the quantized level.
    let hbc = require_hbc(&lf_global)?.clone();
    let resolver = BlockContextResolver::new(&hbc);
    let num_hf_presets = hf_section.num_hf_presets();
    let group_rects =
        crate::group_rect::group_rects_in_blocks(fh.width, fh.height, fh.group_dim())?;
    if group_rects.len() != pass_group_readers.len() {
        return Err(Error::InvalidData(
            "JXL jpeg_reconstruct: group count mismatch".into(),
        ));
    }

    let mut coeffs: [Vec<i32>; 3] = [
        vec![0i32; cdims[0].0 * cdims[0].1 * 64],
        vec![0i32; cdims[1].0 * cdims[1].1 * 64],
        vec![0i32; cdims[2].0 * cdims[2].1 * 64],
    ];

    for (rect, per_pass) in group_rects.into_iter().zip(pass_group_readers) {
        let sub_grid = crate::group_rect::slice_dct_select_rect(&dct_grid, &rect)?;
        let mut gbr = per_pass.into_iter().next().ok_or_else(|| {
            Error::InvalidData("JXL jpeg_reconstruct: missing PassGroup reader".into())
        })?;
        let headers = PerPassHfHeaders::read(&mut gbr, 1, num_hf_presets, hbc.nb_block_ctx)?;
        let pass_data = hf_section.pass_data_mut(0)?;
        pass_data.histograms.begin_section(&mut gbr)?;
        let mut ctx = pass_data.single_pass_context(&headers)?;
        // Group-lattice invariant: every channel's subsampling lattice
        // must align with the group origin (group_dim is a multiple of
        // every lattice in practice — 256 px = 32 blocks).
        for &(hs, vs) in &shifts {
            if rect.bx0 % (1 << hs) != 0 || rect.by0 % (1 << vs) != 0 {
                return Err(Error::Unsupported(
                    "JXL jpeg_reconstruct: group origin off the subsampling lattice".into(),
                ));
            }
        }
        let per_channel_dims: [(u32, u32); 3] = [
            (
                sub_grid.width_blocks.div_ceil(1 << shifts[0].0),
                sub_grid.height_blocks.div_ceil(1 << shifts[0].1),
            ),
            (
                sub_grid.width_blocks.div_ceil(1 << shifts[1].0),
                sub_grid.height_blocks.div_ceil(1 << shifts[1].1),
            ),
            (
                sub_grid.width_blocks.div_ceil(1 << shifts[2].0),
                sub_grid.height_blocks.div_ceil(1 << shifts[2].1),
            ),
        ];
        let mut nz = PerPassNonZerosGrids::new(&[&per_channel_dims])?;

        // Per-varblock decoded blocks: uniform driver for 4:4:4, the
        // lattice-skipping driver otherwise. Output normalised to
        // `(vb, [Option<DecodedHfBlock>; 3])`.
        let varblocks: Vec<(
            crate::varblock_walk::Varblock,
            [Option<crate::pass_group_hf::DecodedHfBlock>; 3],
        )> = if subsampled {
            let qdc_at = |c: u32, cx: u32, cy: u32| -> Result<i32> {
                let (hs, vs) = shifts[c as usize];
                let (cw, _) = cdims[c as usize];
                let gx = ((rect.bx0 >> hs) + cx) as usize;
                let gy = ((rect.by0 >> vs) + cy) as usize;
                Ok(lf_quant[c as usize][gy * cw + gx])
            };
            ctx.decode_lf_group_single_pass_subsampled(
                &mut gbr, &sub_grid, &mut nz, &resolver, shifts, qdc_at,
            )?
        } else {
            let qdc_at = |_p: u32, vb: &crate::varblock_walk::Varblock| -> Result<[i32; 3]> {
                let bx = (rect.bx0 + vb.x) as usize;
                let by = (rect.by0 + vb.y) as usize;
                let idx = by * fbw + bx;
                Ok([lf_quant[0][idx], lf_quant[1][idx], lf_quant[2][idx]])
            };
            let mut out = ctx.decode_lf_group_multi_pass_three_channels(
                &mut gbr, &sub_grid, &mut nz, &resolver, qdc_at,
            )?;
            let vbs = out.pop().ok_or_else(|| {
                Error::InvalidData("JXL jpeg_reconstruct: empty per-pass decode output".into())
            })?;
            vbs.into_iter()
                .map(|(vb, [d0, d1, d2], _raw)| (vb, [Some(d0), Some(d1), Some(d2)]))
                .collect()
        };
        hf_section.pass_data_mut(0)?.histograms.finish_section()?;

        for (vb, blocks) in varblocks {
            if vb.transform != TransformType::Dct8x8 {
                return Err(Error::Unsupported(format!(
                    "JXL jpeg_reconstruct: varblock transform {:?} at ({}, {}) — a JPEG \
                     transcode contains only DCT8×8 varblocks",
                    vb.transform, vb.x, vb.y
                )));
            }
            for (c, slot) in blocks.iter().enumerate() {
                let Some(dhb) = slot else { continue };
                let (hs, vs) = shifts[c];
                let cx = ((rect.bx0 + vb.x) >> hs) as usize;
                let cy = ((rect.by0 + vb.y) >> vs) as usize;
                let (cw, _) = cdims[c];
                if dhb.remaining_non_zeros != 0 {
                    return Err(Error::InvalidData(format!(
                        "JXL jpeg_reconstruct: varblock ({cx}, {cy}) channel {c} decoded with \
                         {} undelivered NonZeros — refusing to reconstruct from a desynced \
                         stream",
                        dhb.remaining_non_zeros
                    )));
                }
                let base = (cy * cw + cx) * 64;
                coeffs[c][base..base + 64].copy_from_slice(&dhb.coeffs);
                coeffs[c][base] = lf_quant[c][cy * cw + cx];
            }
        }
    }

    // Integer chroma-from-luma inversion (wire-arbitrated, see the
    // module notes): the stored X / B AC coefficients are residuals
    // after the CfL prediction `round(k × y × qY[pos] / q{X,B}[pos])`
    // with `k = base_correlation + tile / colour_factor` — the raw
    // integer relation, with NO Listing F.2 quant bias and NO
    // 0.8^(qm_scale − 2) factor (pinned by the exact tile fit
    // kX = −48/84 on the minimal edge fixture). DC is excluded: the
    // LF-side factors are `{x,b}_factor_lf − 128`, required neutral
    // (128) above.
    if cfl.x_factor_lf != 128 || cfl.b_factor_lf != 128 {
        return Err(Error::Unsupported(
            "JXL jpeg_reconstruct: non-neutral LF chroma-from-luma factors".into(),
        ));
    }
    // I.6: "This clause is skipped if any channel is subsampled" — no
    // CfL inversion for 4:2:0 / 4:2:2 frames.
    let cfl_blocks = if subsampled { 0 } else { fbh };
    let cf = cfl.colour_factor as f64;
    for by in 0..cfl_blocks {
        for bx in 0..fbw {
            let tile = (by / 8).min(fth - 1) * ftw + (bx / 8).min(ftw - 1);
            let kx = cfl.base_correlation_x as f64 + (x_from_y[tile] as f64) / cf;
            let kb = cfl.base_correlation_b as f64 + (b_from_y[tile] as f64) / cf;
            let base = (by * fbw + bx) * 64;
            for pos in 1..64usize {
                let y = coeffs[1][base + pos];
                if y == 0 {
                    continue;
                }
                let dy = (y as f64) * (quant[1][pos] as f64);
                let px = (kx * dy / (quant[0][pos] as f64)).round() as i32;
                let pb = (kb * dy / (quant[2][pos] as f64)).round() as i32;
                coeffs[0][base + pos] += px;
                coeffs[2][base + pos] += pb;
            }
        }
    }

    // The JXL cell layout is the TRANSPOSE of the JPEG raster (pinned
    // on the edge fixture: a pure-horizontal-frequency JPEG block
    // stores its run down our first column, and the RAW quant table
    // rows reproduce the JPEG table's columns). Emit both the
    // coefficients and the quant tables in JPEG (row, col) raster so
    // every consumer downstream speaks 10918-1 coordinates.
    let transpose64 = |v: &mut [i32]| {
        for r in 0..8 {
            for c in (r + 1)..8 {
                v.swap(r * 8 + c, c * 8 + r);
            }
        }
    };
    let mut quant = quant;
    for c in 0..3 {
        let (cw, ch) = cdims[c];
        for b in 0..(cw * ch) {
            transpose64(&mut coeffs[c][b * 64..b * 64 + 64]);
        }
        transpose64(&mut quant[c]);
    }

    Ok(TranscodedCoefficients {
        width: size.width,
        height: size.height,
        jpeg_upsampling: fh.jpeg_upsampling,
        bw: fbw,
        bh: fbh,
        cdims,
        coeffs,
        quant,
        x_from_y,
        b_from_y,
        tw: ftw,
        colour_factor: cfl.colour_factor,
    })
}

fn require_hbc(lf_global: &LfGlobal) -> Result<&crate::lf_global::HfBlockContext> {
    lf_global
        .hf_block_context
        .as_ref()
        .ok_or_else(|| Error::InvalidData("JXL jpeg_reconstruct: HfBlockContext missing".into()))
}

// ---------------------------------------------------------------------
// Annex A — the JPEG writer.
// ---------------------------------------------------------------------

/// MSB-first JPEG bit emitter with ISO/IEC 10918-1 B.1.1.5 byte
/// stuffing (a `0x00` byte after every emitted `0xFF`).
struct JpegBitWriter {
    out: Vec<u8>,
    acc: u32,
    nbits: u32,
}

impl JpegBitWriter {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            acc: 0,
            nbits: 0,
        }
    }

    fn put_bits(&mut self, value: u32, len: u32) {
        debug_assert!(len <= 24);
        if len == 0 {
            return;
        }
        self.acc = (self.acc << len) | (value & ((1u32 << len) - 1));
        self.nbits += len;
        while self.nbits >= 8 {
            let byte = ((self.acc >> (self.nbits - 8)) & 0xFF) as u8;
            self.out.push(byte);
            if byte == 0xFF {
                self.out.push(0x00);
            }
            self.nbits -= 8;
        }
        self.acc &= (1u32 << self.nbits) - 1;
    }

    /// Pad to the next byte boundary per 18181-2 A.6: when the jbrd
    /// signalled `has_padding`, the next bits come from `bbit`;
    /// otherwise the padding bits are zero.
    fn pad_to_byte(&mut self, padding: &mut PaddingBits<'_>) -> Result<()> {
        while self.nbits % 8 != 0 {
            let bit = padding.next()?;
            self.put_bits(bit as u32, 1);
        }
        Ok(())
    }
}

/// The A.6 padding-bit source: `bbit` entries when `has_padding`,
/// zeros otherwise.
struct PaddingBits<'a> {
    bits: &'a [bool],
    pos: usize,
    has_padding: bool,
}

impl PaddingBits<'_> {
    fn next(&mut self) -> Result<bool> {
        if !self.has_padding {
            // A.6 prints "otherwise these padding bits are zero", but
            // the wire says otherwise: fixtures whose original JPEGs
            // pad entropy segments with 1-bits (the common encoder
            // convention) carry has_padding = false, and the black-box
            // reference reconstruction reproduces the 1-bits. The
            // default padding bit is therefore ONE; bbit exists for
            // encoders that padded with something else.
            return Ok(true);
        }
        let b = *self.bits.get(self.pos).ok_or_else(|| {
            Error::InvalidData("JXL jpeg_reconstruct: bbit padding bits exhausted".into())
        })?;
        self.pos += 1;
        Ok(b)
    }
}

/// A built Huffman table: `codes[symbol] = (code, length)`.
struct BuiltHuffman {
    codes: Vec<Option<(u32, u32)>>,
}

impl BuiltHuffman {
    /// Canonical ISO/IEC 10918-1 Annex C code assignment over the jbrd
    /// counts/values (including the sentinel symbol 256, which receives
    /// a code but is never emitted).
    fn build(hc: &HuffmanCode) -> Result<Self> {
        if hc.counts[0] != 0 {
            return Err(Error::Unsupported(
                "JXL jpeg_reconstruct: length-0 (single-symbol) Huffman code".into(),
            ));
        }
        let mut codes: Vec<Option<(u32, u32)>> = vec![None; 257];
        let mut code = 0u32;
        let mut vi = 0usize;
        for len in 1..=16u32 {
            for _ in 0..hc.counts[len as usize] {
                let sym = *hc.values.get(vi).ok_or_else(|| {
                    Error::InvalidData(
                        "JXL jpeg_reconstruct: Huffman values shorter than counts".into(),
                    )
                })? as usize;
                vi += 1;
                if sym > 256 || codes[sym].is_some() {
                    return Err(Error::InvalidData(format!(
                        "JXL jpeg_reconstruct: invalid / duplicate Huffman symbol {sym}"
                    )));
                }
                codes[sym] = Some((code, len));
                code += 1;
            }
            code <<= 1;
        }
        Ok(Self { codes })
    }

    fn emit(&self, bw: &mut JpegBitWriter, symbol: u32) -> Result<()> {
        let (code, len) = self
            .codes
            .get(symbol as usize)
            .copied()
            .flatten()
            .ok_or_else(|| {
                Error::InvalidData(format!(
                "JXL jpeg_reconstruct: symbol {symbol:#x} has no code in the signalled Huffman \
                 table"
            ))
            })?;
        bw.put_bits(code, len);
        Ok(())
    }
}

/// Magnitude category (10918-1 F.1.2.1.1): number of bits of `|v|`.
fn category(v: i32) -> u32 {
    32 - (v.unsigned_abs()).leading_zeros()
}

/// The `S` low-order bits appended after a DC/AC Huffman symbol
/// (10918-1 F.1.2.1.3): `v` when positive, `v - 1` in two's complement
/// (equivalently `v + (1 << s) - 1`) when negative.
fn value_bits(v: i32, s: u32) -> u32 {
    if v >= 0 {
        v as u32
    } else {
        (v + (1i32 << s) - 1) as u32
    }
}

struct FrameComponent {
    /// JXL channel index (0 = X/Cb, 1 = Y, 2 = B/Cr).
    jxl_channel: usize,
    /// JPEG sampling factors.
    h: u32,
    v: u32,
}

/// Map the A.2 component order: JPEG components are Y, Cb, Cr while
/// the JXL channels are (X=Cb, Y=luma, B=Cr). Greyscale is a single
/// luma component.
fn frame_components(
    d: &JpegBitstreamData,
    jpeg_upsampling: [u32; 3],
) -> Result<Vec<FrameComponent>> {
    let factors = |ju: u32| -> (u32, u32) {
        // F.2: 0 → {1,1}, 1 → {2,2}, 2 → {2,1}, 3 → {1,2}.
        match ju {
            1 => (2, 2),
            2 => (2, 1),
            3 => (1, 2),
            _ => (1, 1),
        }
    };
    if d.is_grey {
        return Err(Error::Unsupported(
            "JXL jpeg_reconstruct: greyscale JPEG reconstruction not yet handled".into(),
        ));
    }
    if d.component_ids.len() != 3 {
        return Err(Error::InvalidData(format!(
            "JXL jpeg_reconstruct: {} components for a colour JPEG",
            d.component_ids.len()
        )));
    }
    // jpeg_upsampling is in (Cb, Y, Cr) order (A.2 note); JPEG
    // component order is Y, Cb, Cr.
    let ju_for_comp = [jpeg_upsampling[1], jpeg_upsampling[0], jpeg_upsampling[2]];
    let jxl_channel_for_comp = [1usize, 0, 2];
    Ok((0..3)
        .map(|i| {
            let (h, v) = factors(ju_for_comp[i]);
            FrameComponent {
                jxl_channel: jxl_channel_for_comp[i],
                h,
                v,
            }
        })
        .collect())
}

/// Reconstruct the original JPEG file from a box-structured JPEG XL
/// file carrying a `jbrd` box (18181-2 Annex A).
pub fn reconstruct_jpeg(file: &[u8]) -> Result<Vec<u8>> {
    let parsed = JxlFile::parse(file)?;
    let jbrd_payload = parsed.jbrd.ok_or_else(|| {
        Error::InvalidData(
            "JXL jpeg_reconstruct: file carries no JPEG Bitstream Reconstruction Data box".into(),
        )
    })?;
    let d = JpegBitstreamData::parse(jbrd_payload)?;
    let cs = &parsed.codestream;
    if cs.len() < 2 || cs[0] != 0xFF || cs[1] != 0x0A {
        return Err(Error::InvalidData(
            "JXL jpeg_reconstruct: codestream missing FF 0A signature".into(),
        ));
    }
    let tc = decode_transcoded_coefficients(&cs[2..])?;
    reconstruct_from_parts(&d, &tc, &parsed)
}

/// Annex A segment loop over already-decoded parts.
pub fn reconstruct_from_parts(
    d: &JpegBitstreamData,
    tc: &TranscodedCoefficients,
    file: &JxlFile<'_>,
) -> Result<Vec<u8>> {
    let comps = frame_components(d, tc.jpeg_upsampling)?;

    let mut out: Vec<u8> = Vec::new();
    // A.1: implicit SOI.
    out.extend_from_slice(&[0xFF, 0xD8]);

    // "next"-style iterators (A.1: every element used exactly once, in
    // order of increasing index).
    let mut app_it = 0usize;
    let mut com_it = 0usize;
    let mut inter_it = 0usize;
    let mut quant_it = 0usize;
    let mut huff_it = 0usize;
    let mut scan_it = 0usize;
    let mut exif_boxes = file
        .metadata
        .iter()
        .filter(|m| m.kind == MetadataKind::Exif);
    let mut xml_boxes = file.metadata.iter().filter(|m| m.kind == MetadataKind::Xml);
    let mut padding = PaddingBits {
        bits: &d.padding_bits,
        pos: 0,
        has_padding: d.has_padding,
    };

    // Entropy-coding state that persists across segments.
    // dc/ac tables by (class, id).
    let mut dc_tables: [Option<BuiltHuffman>; 4] = [None, None, None, None];
    let mut ac_tables: [Option<BuiltHuffman>; 4] = [None, None, None, None];
    let mut restart_interval_active = 0u32;

    for &marker in &d.markers {
        match marker {
            0xC0 | 0xC1 | 0xC2 | 0xC9 | 0xCA => {
                // A.2 SOF.
                if marker == 0xC2 || marker == 0xCA {
                    return Err(Error::Unsupported(
                        "JXL jpeg_reconstruct: progressive JPEG reconstruction not yet handled"
                            .into(),
                    ));
                }
                let nc = comps.len();
                let len = 8 + 3 * nc;
                out.extend_from_slice(&[0xFF, marker, (len >> 8) as u8, (len & 0xFF) as u8, 8]);
                out.extend_from_slice(&(tc.height as u16).to_be_bytes());
                out.extend_from_slice(&(tc.width as u16).to_be_bytes());
                out.push(nc as u8);
                for (i, c) in comps.iter().enumerate() {
                    out.push(d.component_ids[i] as u8);
                    out.push(((c.h << 4) | c.v) as u8);
                    out.push(d.component_q_idx[i] as u8);
                }
            }
            0xC4 => {
                // A.3 DHT: HuffmanCode entities until is_last.
                let start = huff_it;
                loop {
                    let hc = d.huffman_codes.get(huff_it).ok_or_else(|| {
                        Error::InvalidData(
                            "JXL jpeg_reconstruct: huffman_code iterator exhausted".into(),
                        )
                    })?;
                    huff_it += 1;
                    if hc.is_last {
                        break;
                    }
                }
                let entities = &d.huffman_codes[start..huff_it];
                let mut payload: Vec<u8> = Vec::new();
                for hc in entities {
                    let total: u32 = hc.counts.iter().sum();
                    if total == 0 {
                        return Err(Error::InvalidData(
                            "JXL jpeg_reconstruct: empty Huffman code".into(),
                        ));
                    }
                    payload.push((u8::from(hc.is_ac) << 4) | hc.id as u8);
                    let mut l = [0u8; 16];
                    for (j, &cnt) in hc.counts[1..].iter().enumerate() {
                        l[j] = cnt as u8;
                    }
                    let last_nz = hc.counts.iter().rposition(|&cnt| cnt != 0).unwrap_or(0);
                    if last_nz == 0 {
                        return Err(Error::Unsupported(
                            "JXL jpeg_reconstruct: length-0 Huffman code".into(),
                        ));
                    }
                    l[last_nz - 1] -= 1;
                    payload.extend_from_slice(&l);
                    // Values minus the sentinel (256).
                    for &v in hc.values.iter().filter(|&&v| v != 256) {
                        payload.push(v as u8);
                    }
                    // Register the table for subsequent scans.
                    let built = BuiltHuffman::build(hc)?;
                    let slot = hc.id as usize;
                    if slot >= 4 {
                        return Err(Error::InvalidData(
                            "JXL jpeg_reconstruct: Huffman table id out of range".into(),
                        ));
                    }
                    if hc.is_ac {
                        ac_tables[slot] = Some(built);
                    } else {
                        dc_tables[slot] = Some(built);
                    }
                }
                let len = 2 + payload.len();
                out.extend_from_slice(&[0xFF, 0xC4, (len >> 8) as u8, (len & 0xFF) as u8]);
                out.extend_from_slice(&payload);
            }
            0xD0..=0xD7 => {
                // A.4 free-standing RSTn.
                out.extend_from_slice(&[0xFF, marker]);
            }
            0xD9 => {
                // A.5 EOI + tail data.
                out.extend_from_slice(&[0xFF, 0xD9]);
                out.extend_from_slice(&d.trailing.tail_data);
            }
            0xDA => {
                // A.6 SOS + entropy-coded data.
                let si = d.scan_infos.get(scan_it).ok_or_else(|| {
                    Error::InvalidData("JXL jpeg_reconstruct: scan_info iterator exhausted".into())
                })?;
                let smi = &d.scan_more_infos[scan_it];
                scan_it += 1;
                let ncomps = si.components.len();
                let len = 6 + 2 * ncomps;
                out.extend_from_slice(&[0xFF, 0xDA, (len >> 8) as u8, (len & 0xFF) as u8]);
                out.push(ncomps as u8);
                for csi in &si.components {
                    out.push(*d.component_ids.get(csi.comp_idx as usize).ok_or_else(|| {
                        Error::InvalidData(
                            "JXL jpeg_reconstruct: scan comp_idx out of range".into(),
                        )
                    })? as u8);
                    out.push(((csi.dc_tbl_idx << 4) | csi.ac_tbl_idx) as u8);
                }
                out.push(si.ss as u8);
                out.push(si.se as u8);
                out.push(((si.ah << 4) | si.al) as u8);

                let mut bw = JpegBitWriter::new();
                encode_sequential_scan(
                    &mut bw,
                    si,
                    smi,
                    &comps,
                    tc,
                    &dc_tables,
                    &ac_tables,
                    restart_interval_active,
                    &mut padding,
                )?;
                bw.pad_to_byte(&mut padding)?;
                debug_assert_eq!(bw.nbits, 0);
                out.extend_from_slice(&bw.out);
            }
            0xDB => {
                // A.7 DQT: QuantTable entities until is_last.
                let mut payload: Vec<u8> = Vec::new();
                loop {
                    let qt = d.quant_tables.get(quant_it).ok_or_else(|| {
                        Error::InvalidData("JXL jpeg_reconstruct: quant iterator exhausted".into())
                    })?;
                    quant_it += 1;
                    payload.push(((qt.precision << 4) | qt.index) as u8);
                    // Q_k from the codestream: the component whose
                    // component_q_idx names this table index supplies
                    // the channel.
                    let comp_i = d
                        .component_q_idx
                        .iter()
                        .position(|&q| q == qt.index)
                        .ok_or_else(|| {
                            Error::Unsupported(format!(
                                "JXL jpeg_reconstruct: quant table {} unused by any component \
                                 (previous-table copy not yet handled)",
                                qt.index
                            ))
                        })?;
                    let channel = comps
                        .get(comp_i)
                        .map(|c| c.jxl_channel)
                        .unwrap_or(comp_i.min(2));
                    let q = &tc.quant[channel];
                    for k in 0..64 {
                        let val = q[ZIGZAG[k]];
                        if qt.precision == 1 {
                            payload.extend_from_slice(&(val as u16).to_be_bytes());
                        } else {
                            if !(0..=255).contains(&val) {
                                return Err(Error::InvalidData(format!(
                                    "JXL jpeg_reconstruct: 8-bit quant factor {val} out of range"
                                )));
                            }
                            payload.push(val as u8);
                        }
                    }
                    if qt.is_last {
                        break;
                    }
                }
                let len = 2 + payload.len();
                out.extend_from_slice(&[0xFF, 0xDB, (len >> 8) as u8, (len & 0xFF) as u8]);
                out.extend_from_slice(&payload);
            }
            0xDD => {
                // A.8 DRI.
                out.extend_from_slice(&[0xFF, 0xDD, 0x00, 0x04]);
                out.extend_from_slice(&(d.restart_interval as u16).to_be_bytes());
                restart_interval_active = d.restart_interval;
            }
            0xE0..=0xEF => {
                // A.9 APPn.
                let am = d.app_markers.get(app_it).ok_or_else(|| {
                    Error::InvalidData("JXL jpeg_reconstruct: app_marker iterator exhausted".into())
                })?;
                let data = d.trailing.app_data[app_it].as_deref();
                app_it += 1;
                out.push(0xFF);
                match am.kind {
                    0 => {
                        out.extend_from_slice(data.ok_or_else(|| {
                            Error::InvalidData("JXL jpeg_reconstruct: missing app_data".into())
                        })?);
                    }
                    2 | 3 => {
                        let len_field = am.length - 1;
                        out.push(0xE1);
                        out.extend_from_slice(&(len_field as u16).to_be_bytes());
                        if am.kind == 2 {
                            out.extend_from_slice(b"Exif\0\0");
                            let ex = exif_boxes.next().ok_or_else(|| {
                                Error::InvalidData(
                                    "JXL jpeg_reconstruct: Exif app marker but no Exif box".into(),
                                )
                            })?;
                            let content = ex.content(crate::jpeg_bitstream::MAX_BROTLI_OUTPUT)?;
                            if content.len() < 4 {
                                return Err(Error::InvalidData(
                                    "JXL jpeg_reconstruct: Exif box too short".into(),
                                ));
                            }
                            // Exif box payload minus the 4-byte tiff
                            // header offset (A.9).
                            out.extend_from_slice(&content[4..]);
                        } else {
                            out.extend_from_slice(b"http://ns.adobe.com/xap/1.0/\0");
                            let xb = xml_boxes.next().ok_or_else(|| {
                                Error::InvalidData(
                                    "JXL jpeg_reconstruct: XMP app marker but no XML box".into(),
                                )
                            })?;
                            let content = xb.content(crate::jpeg_bitstream::MAX_BROTLI_OUTPUT)?;
                            out.extend_from_slice(&content);
                        }
                    }
                    1 => {
                        return Err(Error::Unsupported(
                            "JXL jpeg_reconstruct: ICC-profile APP2 reconstruction not yet \
                             handled"
                                .into(),
                        ));
                    }
                    k => {
                        return Err(Error::InvalidData(format!(
                            "JXL jpeg_reconstruct: unknown app marker kind {k}"
                        )));
                    }
                }
            }
            0xFE => {
                // A.10 COM.
                let data = d.trailing.com_data.get(com_it).ok_or_else(|| {
                    Error::InvalidData("JXL jpeg_reconstruct: com_data iterator exhausted".into())
                })?;
                com_it += 1;
                out.extend_from_slice(&[0xFF, 0xFE]);
                out.extend_from_slice(data);
            }
            0xFF => {
                // A.11 unrecognized data (raw bytes, no marker prefix).
                let data = d.trailing.intermarker_data.get(inter_it).ok_or_else(|| {
                    Error::InvalidData(
                        "JXL jpeg_reconstruct: intermarker iterator exhausted".into(),
                    )
                })?;
                inter_it += 1;
                out.extend_from_slice(data);
            }
            m => {
                return Err(Error::InvalidData(format!(
                    "JXL jpeg_reconstruct: marker {m:#04x} matches no Annex A instruction \
                     (ill-formed codestream)"
                )));
            }
        }
    }

    Ok(out)
}

/// ISO/IEC 10918-1 sequential-DCT Huffman entropy encoding of one scan
/// (Annexes F.1.2 + B.2.3), with the 18181-2 A.6 amendments (extra ZRL
/// symbols, recorded padding bits).
#[allow(clippy::too_many_arguments)]
fn encode_sequential_scan(
    bw: &mut JpegBitWriter,
    si: &ScanInfo,
    smi: &ScanMoreInfo,
    comps: &[FrameComponent],
    tc: &TranscodedCoefficients,
    dc_tables: &[Option<BuiltHuffman>; 4],
    ac_tables: &[Option<BuiltHuffman>; 4],
    restart_interval: u32,
    padding: &mut PaddingBits<'_>,
) -> Result<()> {
    if si.ss != 0 || si.se != 63 || si.ah != 0 || si.al != 0 {
        return Err(Error::Unsupported(
            "JXL jpeg_reconstruct: spectral-selection / successive-approximation scan in a \
             sequential frame"
                .into(),
        ));
    }
    if !smi.reset_points.is_empty() {
        return Err(Error::Unsupported(
            "JXL jpeg_reconstruct: reset points in a sequential scan".into(),
        ));
    }

    let h_max = comps.iter().map(|c| c.h).max().unwrap_or(1);
    let v_max = comps.iter().map(|c| c.v).max().unwrap_or(1);

    // Resolve per-scan-component state.
    struct ScanComp<'t> {
        jxl_channel: usize,
        h: u32,
        v: u32,
        width_blocks: usize,
        height_blocks: usize,
        dc: &'t BuiltHuffman,
        ac: &'t BuiltHuffman,
        pred: i32,
    }
    let mut scomps: Vec<ScanComp<'_>> = Vec::with_capacity(si.components.len());
    for csi in &si.components {
        let fc = comps.get(csi.comp_idx as usize).ok_or_else(|| {
            Error::InvalidData("JXL jpeg_reconstruct: scan comp_idx out of range".into())
        })?;
        let dc = dc_tables[csi.dc_tbl_idx as usize].as_ref().ok_or_else(|| {
            Error::InvalidData(format!(
                "JXL jpeg_reconstruct: DC table {} not defined before the scan",
                csi.dc_tbl_idx
            ))
        })?;
        let ac = ac_tables[csi.ac_tbl_idx as usize].as_ref().ok_or_else(|| {
            Error::InvalidData(format!(
                "JXL jpeg_reconstruct: AC table {} not defined before the scan",
                csi.ac_tbl_idx
            ))
        })?;
        let (cw, ch) = tc.cdims[fc.jxl_channel];
        scomps.push(ScanComp {
            jxl_channel: fc.jxl_channel,
            h: fc.h,
            v: fc.v,
            width_blocks: cw,
            height_blocks: ch,
            dc,
            ac,
            pred: 0,
        });
    }

    // Extra-zero-run lookup by block index in the current scan.
    let mut ezr_it = smi.extra_zero_runs.iter().peekable();

    // MCU geometry (B.2.2): with 4:4:4 the MCU is one block per scan
    // component; a single-component scan walks its own block grid.
    let (mcus_x, mcus_y) = if scomps.len() == 1 {
        (scomps[0].width_blocks, scomps[0].height_blocks)
    } else {
        (
            (tc.width as usize).div_ceil(8 * h_max as usize),
            (tc.height as usize).div_ceil(8 * v_max as usize),
        )
    };

    let mut block_idx_in_scan: u32 = 0;
    let mut mcu_count: u32 = 0;
    let mut rst_m: u8 = 0;

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            if restart_interval != 0 && mcu_count == restart_interval {
                // B.2.1: byte-align, emit RSTm, reset predictions.
                bw.pad_to_byte(padding)?;
                bw.out.extend_from_slice(&[0xFF, 0xD0 + rst_m]);
                rst_m = (rst_m + 1) & 7;
                mcu_count = 0;
                for sc in scomps.iter_mut() {
                    sc.pred = 0;
                }
            }
            #[allow(clippy::needless_range_loop)]
            for sc_i in 0..scomps.len() {
                let (h, v) = (scomps[sc_i].h as usize, scomps[sc_i].v as usize);
                for by_in in 0..v {
                    for bx_in in 0..h {
                        let bx = mcu_x * h + bx_in;
                        let by = mcu_y * v + by_in;
                        let sc = &mut scomps[sc_i];
                        if bx >= sc.width_blocks || by >= sc.height_blocks {
                            return Err(Error::Unsupported(
                                "JXL jpeg_reconstruct: MCU padding blocks (image dims not a \
                                 multiple of the MCU size) not yet handled"
                                    .into(),
                            ));
                        }
                        let base = (by * sc.width_blocks + bx) * 64;
                        let block = &tc.coeffs[sc.jxl_channel][base..base + 64];

                        // Extra ZRL data for this block, if any.
                        let mut extra_runs = 0u32;
                        if let Some(ezr) = ezr_it.peek() {
                            if ezr.block_idx == block_idx_in_scan {
                                extra_runs = ezr.num_runs;
                                ezr_it.next();
                            }
                        }

                        let mut pred = sc.pred;
                        encode_sequential_block(bw, sc.dc, sc.ac, &mut pred, block, extra_runs)?;
                        sc.pred = pred;
                        block_idx_in_scan += 1;
                    }
                }
            }
            mcu_count += 1;
        }
    }
    if ezr_it.peek().is_some() {
        return Err(Error::InvalidData(
            "JXL jpeg_reconstruct: unused extra-zero-run entries after the scan".into(),
        ));
    }
    Ok(())
}

/// Encode one 8×8 block, sequential DCT (10918-1 F.1.2 + F.2 figures,
/// with the A.6 extra-ZRL / EOB amendment).
fn encode_sequential_block(
    bw: &mut JpegBitWriter,
    dc_table: &BuiltHuffman,
    ac_table: &BuiltHuffman,
    pred: &mut i32,
    block: &[i32],
    extra_runs: u32,
) -> Result<()> {
    // DC: category + amplitude of the prediction difference.
    let dc = block[0];
    let diff = dc - *pred;
    *pred = dc;
    let s = if diff == 0 { 0 } else { category(diff) };
    dc_table.emit(bw, s)?;
    if s > 0 {
        bw.put_bits(value_bits(diff, s), s);
    }

    // AC: run-length of zeros + category, zig-zag order.
    let mut k_last = 0usize; // zig-zag index of the last non-zero coefficient
    for k in (1..64).rev() {
        if block[ZIGZAG[k]] != 0 {
            k_last = k;
            break;
        }
    }
    let mut run = 0u32;
    for k in 1..=k_last {
        let v = block[ZIGZAG[k]];
        if v == 0 {
            run += 1;
            continue;
        }
        while run > 15 {
            ac_table.emit(bw, 0xF0)?; // ZRL
            run -= 16;
        }
        let s = category(v);
        ac_table.emit(bw, (run << 4) | s)?;
        bw.put_bits(value_bits(v, s), s);
        run = 0;
    }
    // Tail: A.6 extra ZRL symbols, then EOB only when zero
    // coefficients remain.
    let mut remaining = 63i32 - k_last as i32;
    for _ in 0..extra_runs {
        ac_table.emit(bw, 0xF0)?;
        remaining -= 16;
        if remaining < 0 {
            return Err(Error::InvalidData(
                "JXL jpeg_reconstruct: extra zero runs exceed the block's trailing zeros".into(),
            ));
        }
    }
    if remaining > 0 {
        ac_table.emit(bw, 0x00)?; // EOB
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_is_a_permutation() {
        let mut seen = [false; 64];
        for &z in &ZIGZAG {
            assert!(z < 64);
            seen[z] = true;
        }
        assert!(seen.iter().all(|&s| s), "ZIGZAG must cover all 64 cells");
    }

    #[test]
    fn categories() {
        assert_eq!(category(1), 1);
        assert_eq!(category(-1), 1);
        assert_eq!(category(2), 2);
        assert_eq!(category(-3), 2);
        assert_eq!(category(255), 8);
        assert_eq!(category(-1024), 11);
    }

    #[test]
    fn value_bits_ones_complement() {
        assert_eq!(value_bits(5, 3), 5);
        assert_eq!(value_bits(-5, 3), 2); // 5 → 101; -5 → 010
        assert_eq!(value_bits(-1, 1), 0);
        assert_eq!(value_bits(1, 1), 1);
    }
}
