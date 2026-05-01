//! ANS entropy coder + companion lookup tables for the FDIS-2021
//! revision of JPEG XL (ISO/IEC FDIS 18181-1:2021, Annex D).
//!
//! This module is **additive**: nothing in the existing
//! `abrac` / `begabrac` / `matree` / `modular` committee-draft pipeline
//! is rewired here. Future rounds will replace the committee-draft
//! Modular entry point with the FDIS path that drives this ANS coder
//! through `FrameHeader` + `TOC`.
//!
//! The submodules implement, top-down, FDIS Annex D:
//!
//! * [`prefix`]  — D.2.1 / D.2.2: RFC 7932 §3.4 + §3.5 prefix codes
//!   (the histogram preamble used when `use_prefix_code == 1`).
//! * [`alias`]   — D.3.2: alias mapping initialisation (Listing D.1) +
//!   lookup (Listing D.2).
//! * [`symbol`]  — D.3.3: the 32-bit-state ANS symbol decoder
//!   (Listing D.3).
//! * [`distribution`] — D.3.4: ANS distribution decoding (Listing D.4)
//!   plus the verbatim 128×2 `kLogCountLut` table.
//! * [`cluster`] — D.3.5: distribution clustering + the inverse
//!   move-to-front transform (Listing D.5).
//! * [`hybrid`]  — D.3.6: hybrid integer coding (Listing D.6) including
//!   the 120×2 `kSpecialDistances` table and LZ77 windowing.
//! * [`hybrid_config`] — D.3.7: `HybridUintConfig` decoder
//!   (Listing D.7).
//!
//! Tests in each submodule build their input by hand from the spec text
//! so a single failing module is locally diagnosable.
//!
//! ## Bound checks
//!
//! Every allocation in this module is sized against the input length
//! (number of bits remaining behind the [`crate::bitreader::BitReader`]
//! cursor) before any `Vec::with_capacity`. The intent is that a 10-byte
//! malicious codestream cannot cause the decoder to allocate gigabytes:
//! see [`distribution::ALPHABET_SIZE_MAX`],
//! [`hybrid::WINDOW_SIZE`], and the explicit cap on
//! `log_alphabet_size <= 15` in [`hybrid_config`].
//!
//! ## Spec audit trail
//!
//! Each transcribed lookup table carries a comment with its FDIS
//! Listing/Table number and the PDF page it appears on. The two big
//! tables are:
//!
//! * `kLogCountLut`  — `distribution::K_LOG_COUNT_LUT`  (FDIS D.3.4
//!   Listing D.4, p. 64).
//! * `kSpecialDistances` — `hybrid::K_SPECIAL_DISTANCES` (FDIS D.3.6
//!   Listing D.6, p. 66).

pub mod alias;
pub mod cluster;
pub mod distribution;
pub mod hybrid;
pub mod hybrid_config;
pub mod prefix;
pub mod symbol;

#[cfg(test)]
pub(crate) mod test_helpers {
    /// Pack a sequence of `(value, n_bits)` into LSB-first bytes
    /// suitable for [`crate::bitreader::BitReader`]. Used by every
    /// `ans::*` test module to construct an exact bitstream from spec
    /// listings without needing a reference encoder.
    pub fn pack_lsb(parts: &[(u32, u32)]) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        let mut byte: u32 = 0;
        let mut bits: u32 = 0;
        for &(v, n) in parts {
            assert!(n <= 32);
            for i in 0..n {
                let bit = (v >> i) & 1;
                byte |= bit << bits;
                bits += 1;
                if bits == 8 {
                    out.push(byte as u8);
                    byte = 0;
                    bits = 0;
                }
            }
        }
        if bits > 0 {
            out.push(byte as u8);
        }
        out
    }
}
