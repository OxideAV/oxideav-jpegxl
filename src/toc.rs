//! `TOC` (Table of Contents) — FDIS 18181-1 §C.3.
//!
//! Three sub-procedures:
//!
//! * §C.3.1 — `permuted_toc = u(1)` selector + the entry array
//!   structure (LfGlobal, LfGroups, HfGlobal+HfPasses, PassGroups).
//! * §C.3.2 — Lehmer-code decoder: reads `end + (end - skip) integers
//!   per D.3.6` against an 8-cluster ANS context, then converts the
//!   Lehmer sequence into a permutation via the `temp` shuffling
//!   procedure.
//! * §C.3.3 — Per-entry `U32` size decode (byte-aligned), followed by
//!   `group_offsets` running sum and the optional permutation.
//!
//! Allocation bound: every per-entry `Vec::with_capacity` is sized
//! against the [`num_toc_entries`] total computed up-front from
//! `frame_header.num_groups()` × `passes.num_passes`. The total never
//! exceeds `1 + num_lf_groups + 1 + num_passes + num_groups *
//! num_passes`. We cap that derived total against the bit reader's
//! remaining input length — a malicious frame header that claimed
//! billions of groups would already have been rejected by the
//! `width × height` check in [`crate::frame_header::FrameHeader`].

use oxideav_core::{Error, Result};

use crate::ans::hybrid::HybridUintState;
use crate::bitreader::{BitReader, U32Dist};
use crate::frame_header::{Encoding, FrameHeader};
use crate::modular_fdis::EntropyStream;

/// Decoded `TOC` per FDIS C.3.
#[derive(Debug, Clone)]
pub struct Toc {
    /// True if the optional permutation was applied.
    pub permuted: bool,
    /// Per-entry size in bytes (post-permutation order, in the order
    /// the caller will consume them — so element `i` is the size of
    /// section `i`'s on-wire data).
    pub entries: Vec<u32>,
    /// Running sum: `group_offsets[0] = 0`, `group_offsets[i] =
    /// sum(entries[0..i])`. Permutation already applied if `permuted`.
    pub group_offsets: Vec<u64>,
}

/// Compute the number of TOC entries per ISO/IEC 18181-1:2024 §F.3.1
/// from `(num_groups, num_passes, encoding)`.
///
/// Returns 1 if `num_groups == 1 && num_passes == 1` (the
/// "single TOC entry" shortcut), otherwise the full layout count.
///
/// **Round 9 fix**: the 2024 spec lists `HfGlobal` UNCONDITIONALLY (one
/// section, regardless of `encoding`). Per the spec NOTE 1: "Some of
/// these sections may be empty in case `encoding == kModular`, i.e.
/// their size in bytes is zero" — i.e. for kModular, HfGlobal has
/// 0-byte size but the slot still occupies a TOC entry. Round 8's
/// reading skipped HfGlobal for kModular which off-by-oned every
/// PassGroup index in multi-group kModular frames (synth_320 then
/// took entries[2] = HfGlobal as PassGroup[0][0] and saw a bogus
/// 0-byte first PassGroup). Also, HfPass is part of the HfGlobal
/// section per Annex G.3 Table G.4 — it does NOT contribute additional
/// TOC entries.
pub fn num_toc_entries(num_groups: u64, num_passes: u32, _encoding: Encoding) -> u64 {
    if num_groups == 1 && num_passes == 1 {
        return 1;
    }
    let num_lf_groups_term = num_groups; // caller passes num_lf_groups, see [`from_frame_header`]
    let mut count: u64 = 1; // LfGlobal
    count += num_lf_groups_term; // LfGroup[num_lf_groups]
    count += 1; // HfGlobal (unconditional; 0-byte for kModular)
    count += num_groups * num_passes as u64; // PassGroup[num_groups × num_passes]
    count
}

/// FDIS hard cap on TOC entries — the 30-bit `BitsOffset(30, 4211712)`
/// distribution can encode up to ~2^30 + 4MiB bytes per entry, but we
/// must also cap the *count* of entries themselves: we accept up to
/// 2^24 entries which is 16M sections per frame, far beyond any
/// realistic codestream.
pub const MAX_TOC_ENTRIES: u64 = 1 << 24;

impl Toc {
    /// Decode a `TOC` for the given `FrameHeader`. Per FDIS C.3.3 the
    /// TOC entries are byte-aligned (an implicit `ZeroPadToByte()` runs
    /// before the first entry); the caller must arrange for `br` to be
    /// at a byte boundary or invoke `pu0()` first.
    ///
    /// Note: per the FDIS phrasing in C.1, FrameHeader is byte-aligned
    /// at its end, so the natural position after FrameHeader::read +
    /// `pu0()` is the right entry point for this routine.
    pub fn read(br: &mut BitReader<'_>, fh: &FrameHeader) -> Result<Self> {
        let num_groups = fh.num_groups();
        let num_lf_groups = fh.num_lf_groups();
        let num_passes = fh.passes.num_passes;
        // Round 9 fix: HfGlobal is unconditional (one TOC entry, 0-byte
        // for kModular per F.3.1 NOTE 1). HfPass is part of the
        // HfGlobal section per Annex G.3 Table G.4, NOT separate TOC
        // entries.
        let total = if num_groups == 1 && num_passes == 1 {
            1u64
        } else {
            let mut count: u64 = 1; // LfGlobal
            count += num_lf_groups; // LfGroup[num_lf_groups]
            count += 1; // HfGlobal (unconditional; 0-byte for kModular)
            count += num_groups * num_passes as u64;
            count
        };

        if total == 0 {
            return Err(Error::InvalidData(
                "JXL TOC: zero TOC entries (frame has no groups)".into(),
            ));
        }
        if total > MAX_TOC_ENTRIES {
            return Err(Error::InvalidData(format!(
                "JXL TOC: {total} entries exceeds cap {MAX_TOC_ENTRIES}"
            )));
        }
        // Each TOC entry costs at least 12 bits (U32 selector + smallest
        // representation 10 bits in distribution 0). Cap against
        // remaining input.
        if total.saturating_mul(12) > br.bits_remaining() as u64 {
            return Err(Error::InvalidData(
                "JXL TOC: declared entries exceed remaining input".into(),
            ));
        }
        let total_usize = total as usize;

        let permuted = br.read_bit()? == 1;
        let permutation = if permuted {
            decode_permutation(br, total_usize)?
        } else {
            Vec::new()
        };

        // C.3.3: entries are byte-aligned. ZeroPadToByte() runs before
        // the first entry.
        br.pu0()?;
        let entry_dist = [
            U32Dist::Bits(10),
            U32Dist::BitsOffset(14, 1024),
            U32Dist::BitsOffset(22, 17408),
            U32Dist::BitsOffset(30, 4211712),
        ];
        let mut entries: Vec<u32> = Vec::with_capacity(total_usize);
        for _ in 0..total_usize {
            // Per C.3.3 / F.3 entries may be 0 (an empty LfGroup or
            // empty PassGroup is legal; for example a Modular frame
            // whose channels all have hshift>=3 vshift>=3 leaves the
            // ModularGroup sub-bitstream empty). Round 6 over-strictly
            // rejected zero; round 7 accepts.
            let v = br.read_u32(entry_dist)?;
            entries.push(v);
        }
        // ZeroPadToByte() after the last TOC entry per C.3.3 / 6.3.
        br.pu0()?;

        // Compute group_offsets (running sum of entries).
        let mut group_offsets: Vec<u64> = Vec::with_capacity(total_usize);
        let mut acc: u64 = 0;
        for &e in &entries {
            group_offsets.push(acc);
            acc = acc
                .checked_add(e as u64)
                .ok_or_else(|| Error::InvalidData("JXL TOC: group_offsets overflow".into()))?;
        }

        // If permuted, reorder group_offsets so that group_offsets[i] =
        // (old) group_offsets[permutation[i]]. We mirror the same
        // reordering on `entries` so callers can access them in
        // permuted order.
        let (entries, group_offsets) = if permuted {
            if permutation.len() != total_usize {
                return Err(Error::InvalidData(
                    "JXL TOC: permutation length mismatch".into(),
                ));
            }
            let mut new_entries = Vec::with_capacity(total_usize);
            let mut new_offsets = Vec::with_capacity(total_usize);
            for &p in &permutation {
                let pi = p as usize;
                if pi >= total_usize {
                    return Err(Error::InvalidData(
                        "JXL TOC: permutation index out of range".into(),
                    ));
                }
                new_entries.push(entries[pi]);
                new_offsets.push(group_offsets[pi]);
            }
            (new_entries, new_offsets)
        } else {
            (entries, group_offsets)
        };

        Ok(Self {
            permuted,
            entries,
            group_offsets,
        })
    }
}

/// FDIS C.3.2 Lehmer-code permutation decoder (TOC call site,
/// `skip = 0` per C.3.1).
///
/// Reads the §C.3.1 permutation sub-stream prelude
/// (`PermStreamConfig::num_dists` clustered distributions per D.3 —
/// see `crate::coeff_order` for the Part 8.3 context/count
/// resolution) and delegates the Lehmer walk to the shared
/// [`decode_permutation_from_stream`] so the TOC and Listing C.12
/// call sites cannot drift apart.
///
/// Round 393: the sub-stream is a FULL D.3 entropy stream — cjxl's
/// large-image progressive TOC permutations ship with LZ77 enabled —
/// so the hand-rolled ANS-only prelude was replaced by the shared
/// [`EntropyStream`] + [`HybridUintState`] pipeline (LZ77 window,
/// prefix-or-ANS, dedicated distance context and all).
fn decode_permutation(br: &mut BitReader<'_>, size: usize) -> Result<Vec<u32>> {
    if size == 0 {
        return Ok(Vec::new());
    }
    if size > MAX_TOC_ENTRIES as usize {
        return Err(Error::InvalidData(
            "JXL permutation: size exceeds TOC cap".into(),
        ));
    }
    if size > br.bits_remaining() {
        return Err(Error::InvalidData(
            "JXL permutation: size exceeds remaining input".into(),
        ));
    }
    if size == 1 {
        // Single-entry frame: permutation is trivial.
        return Ok(vec![0]);
    }

    // C.3.1: "a single entropy coded stream with 8 clustered
    // distributions, as specified in D.3" (count per the Part 8.3
    // resolution — see `PermStreamConfig`). The full D.3 prelude —
    // LZ77Params (+1 distance context when enabled), clustering,
    // use_prefix_code, per-cluster configs + distributions — then the
    // ANS state init (u(32), a no-op for prefix streams) directly
    // before the first symbol.
    let cfg = crate::coeff_order::perm_stream_config();
    let mut stream = EntropyStream::read(br, cfg.num_dists as usize)?;
    stream.read_ans_state_init(br)?;
    let mut hybrid = HybridUintState::new(stream.lz77, stream.lz_len_conf);
    crate::coeff_order::decode_permutation_from_stream(br, &mut stream, &mut hybrid, size, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_toc_entries_single_group_single_pass() {
        assert_eq!(num_toc_entries(1, 1, Encoding::Modular), 1);
    }

    #[test]
    fn num_toc_entries_modular_path() {
        // num_lf_groups passed in via num_groups param (see fn comment);
        // use the formula with num_groups=4, num_passes=1, Modular →
        // 1 (LfGlobal) + 4 (LfGroup) + 1 (HfGlobal, 0-byte for kModular)
        // + 4*1 (PassGroup) = 10. Round 9 fix: HfGlobal is unconditional
        // per ISO/IEC 18181-1:2024 §F.3.1.
        assert_eq!(num_toc_entries(4, 1, Encoding::Modular), 10);
    }

    /// Build a Toc from an unpermuted, byte-aligned byte string.
    /// Uses the pack helper to match the FDIS U32 distribution for
    /// entries.
    #[test]
    fn unpermuted_toc_single_entry_round_trip() {
        // Frame: width=128, height=128 → kGroupDim=256 → num_groups=1.
        // num_passes=1 (default Passes). Modular encoding (so no
        // HfGlobal/HfPass). num_lf_groups = 1. Total = 1 entry.
        let fh = build_test_frame_header(128, 128);
        // Bit budget: 1 bit permuted_toc=0, then ZeroPad to byte (7
        // bits zero), then U32 entry. Pick distribution selector 0
        // (Bits(10)) with raw value = 5 → entry size = 5 bytes.
        // After entry, ZeroPad again.
        // bit 0: permuted_toc = 0; bits 1..=7: zero pad.
        // U32 entry: selector u(2) = 0 (Bits(10)) → bit0,bit1 = 0,0
        // followed by u(10) = 5 → bits 1,0,1,0,0,0,0,0,0,0
        // packed LSB-first across 12 bits: bits 0..1 selector=00,
        // bits 2..11 = 5 LSB-first
        // So byte1 = bit0=sel0, bit1=sel1, bits2..7 = u(10) bits 0..5
        // byte2 = u(10) bits 6..9 followed by zero-pad
        // We want u(10) value 5 = 0000000101 LSB-first → bits: 1,0,1,0,0,0,0,0,0,0
        // byte1 bits: 0,0,1,0,1,0,0,0  → 0b0001_0100 = 0x14
        // byte2 bits: 0,0,0,0, 0,0,0,0 (4 leftover u(10) bits = 0, 4 zero pad) → 0x00
        let bytes = vec![0u8, 0x14, 0x00];
        let mut br = BitReader::new(&bytes);
        let toc = Toc::read(&mut br, &fh).unwrap();
        assert!(!toc.permuted);
        assert_eq!(toc.entries, vec![5]);
        assert_eq!(toc.group_offsets, vec![0]);
    }

    fn build_test_frame_header(w: u32, h: u32) -> FrameHeader {
        let params = crate::frame_header::FrameDecodeParams {
            xyb_encoded: true,
            num_extra_channels: 0,
            have_animation: false,
            have_animation_timecodes: false,
            image_width: w,
            image_height: h,
        };
        // Use the `default_with` pattern via a single all_default=1 read.
        let bytes = crate::ans::test_helpers::pack_lsb(&[(1, 1)]);
        let mut br = BitReader::new(&bytes);
        FrameHeader::read(&mut br, &params).unwrap()
    }

    #[test]
    fn accepts_zero_entry_size() {
        // Round 7: a TOC entry value of 0 is legal — an empty LfGroup
        // / PassGroup section is allowed when no channel matches that
        // section's criterion.
        let fh = build_test_frame_header(128, 128);
        // permuted=0 + zero pad, then U32 sel=0 (Bits(10)) value 0.
        let bytes = vec![0u8, 0x00, 0x00];
        let mut br = BitReader::new(&bytes);
        let toc = Toc::read(&mut br, &fh).unwrap();
        assert_eq!(toc.entries, vec![0]);
    }

    #[test]
    fn rejects_overflowing_entry_count() {
        // Build a frame header by hand whose num_groups would create
        // a TOC count > MAX_TOC_ENTRIES. We forge a FrameHeader since
        // FrameHeader::read won't accept such absurd dimensions.
        let mut fh = build_test_frame_header(128, 128);
        fh.width = 1 << 30;
        fh.height = 1 << 30;
        fh.group_size_shift = 0; // kGroupDim=128 → num_groups huge
        let bytes = vec![0u8; 4];
        let mut br = BitReader::new(&bytes);
        let res = Toc::read(&mut br, &fh);
        assert!(res.is_err(), "expected overflow rejection");
    }

    #[test]
    fn group_offsets_running_sum_correct() {
        // Build a FrameHeader with num_groups=2 to force multiple TOC
        // entries. Easiest: width=512, height=128 with
        // group_size_shift=1 → kGroupDim=256 → 2x1=2 groups, plus
        // num_lf_groups=1 (kGroupDim*8=2048 > 512). Total entries
        // (round-9 §F.3.1 fix): 1 (LfGlobal) + 1 (LfGroup) + 1 (HfGlobal,
        // 0-byte for kModular) + 2 (PassGroup) = 5 entries.
        let mut fh = build_test_frame_header(512, 128);
        fh.encoding = Encoding::Modular;
        // Build 5 entries with sizes 7, 11, 0, 13, 17 (HfGlobal=0; all
        // < 1024 → use distribution 0 (Bits(10))).
        let mut bw = TestBw::new();
        bw.w(0, 1); // permuted_toc = 0
                    // ZeroPad to byte
        bw.pad();
        for v in [7u32, 11, 0, 13, 17] {
            bw.w(0, 2); // sel = 0 → Bits(10)
            bw.w(v, 10);
        }
        bw.pad();
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let toc = Toc::read(&mut br, &fh).unwrap();
        assert_eq!(toc.entries, vec![7, 11, 0, 13, 17]);
        assert_eq!(toc.group_offsets, vec![0, 7, 18, 18, 31]);
    }

    /// Tiny LSB-first bit writer for tests, with byte-pad helper.
    struct TestBw {
        out: Vec<u8>,
        bit_pos: u8,
    }
    impl TestBw {
        fn new() -> Self {
            Self {
                out: Vec::new(),
                bit_pos: 0,
            }
        }
        fn w(&mut self, value: u32, n: u32) {
            for i in 0..n {
                if self.bit_pos == 0 {
                    self.out.push(0);
                }
                let bit = ((value >> i) & 1) as u8;
                let last = self.out.len() - 1;
                self.out[last] |= bit << self.bit_pos;
                self.bit_pos = (self.bit_pos + 1) % 8;
            }
        }
        fn pad(&mut self) {
            while self.bit_pos != 0 {
                self.w(0, 1);
            }
        }
        fn into_bytes(self) -> Vec<u8> {
            self.out
        }
    }
}
