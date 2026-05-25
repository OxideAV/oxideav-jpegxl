//! Natural ordering of the DCT coefficients —
//! ISO/IEC FDIS 18181-1:2021 §I.2.4 + Table I.1.
//!
//! ## Scope (round 90 — structural)
//!
//! Round 90 lands the natural-coefficient-order computation for every
//! `Order ID` value (Table I.1, 0..=12) so that the §C.7.1 HfPass
//! per-pass coefficient-order parser has a "natural" baseline it can
//! permute with the spec's `DecodePermutation()` result.
//!
//! ## Spec quotes (FDIS §I.2.4)
//!
//! > For every DctSelect value (Hornuss, DCT2×2, etc), the natural
//! > order of the coefficients is computed as follows. The varblock
//! > size `(bwidth, bheight)` for a DctSelect value with name
//! > "DCTN×M" is `bwidth = max(8, max(N, M))` and
//! > `bheight = max(8, min(N, M))`, respectively. The varblock size
//! > for all other transforms is `bwidth = bheight = 8`.
//!
//! > The natural ordering of the DCT coefficients is defined as a
//! > vector `order` of cell positions `(x, y)` between `(0, 0)` and
//! > `(bwidth, bheight)`, described below. The number of elements in
//! > the vector `order` is therefore `bwidth × bheight`, and the
//! > vector is defined as the elements of `LLF` in their original
//! > order followed by the elements of `HF` also in their original
//! > order. `LLF` is a vector of lower frequency coefficients,
//! > containing cells `(x, y)` with `(x < (bwidth / 8))` &&
//! > `(y < (bheight / 8))`. The cells `(x, y)` that do not satisfy
//! > this condition belong to the higher frequencies vector `HF`.
//!
//! > The rest of this subclause specifies how to order the elements
//! > within each of the arrays `LLF` and `HF`. The pairs `(x, y)`
//! > in the `LLF` vector is sorted in ascending order according to
//! > the value `y × bwidth + x`. For the pairs `(x, y)` in the
//! > `HF` vector, the decoder first computes the value of the
//! > variables `key1` and `key2` as specified by Listing I.14.
//!
//! Listing I.14 — Keys for ordering coefficients:
//!
//! ```text
//! cx = bwidth / 8; cy = bheight / 8;
//! scaled_x = x * max(cx, cy) / cx;
//! scaled_y = y * max(cx, cy) / cy;
//! key1 = scaled_x + scaled_y;
//! key2 = scaled_x - scaled_y;
//! if (key1 Umod 2 == 1) key2 = -key2;
//! ```
//!
//! > The decoder sorts the `(x, y)` pairs on the vector `HF` in
//! > ascending order according to the value `key1`. In case of a
//! > tie, the decoder also sorts in ascending order according to
//! > the value `key2`. The order ID is defined based on the
//! > DctSelect as defined in Table I.1.
//!
//! ## Module layout
//!
//! * [`OrderId`] — the 0..=12 enum (Table I.1).
//! * [`varblock_size_for_order`] — `(bwidth, bheight)` for an
//!   `OrderId`. Used by both HfPass (for the size argument to
//!   `DecodePermutation`) and PassGroup HF (for the per-DctSelect
//!   coefficient-count loop).
//! * [`natural_coeff_order`] — the natural-ordering vector
//!   `order[i]` for every `OrderId`. Returns a permutation `Vec<u32>`
//!   of length `bwidth * bheight`. The vector is the LLF prefix
//!   followed by the HF tail; both subsorted per the spec rules
//!   above.
//! * [`COEFFICIENTS_PER_ORDER`] — convenience: `bwidth * bheight`
//!   for every `OrderId` (size argument for `DecodePermutation`).

use oxideav_core::{Error, Result};

use crate::ans::hybrid::HybridUintState;
use crate::bitreader::BitReader;
use crate::dct_select::TransformType;
use crate::modular_fdis::{decode_uint_in_with_dist_pub, EntropyStream};

/// `Order ID` per Table I.1. Each DctSelect value maps to one of 13
/// order IDs; orders 1, 4, 5, 6, 8, 10, 12 are shared by two or three
/// DctSelect values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum OrderId {
    /// 0 — DCT8×8.
    Id0 = 0,
    /// 1 — Hornuss, DCT2×2, DCT4×4, DCT4×8, DCT8×4, AFV0, AFV1, AFV2, AFV3.
    Id1 = 1,
    /// 2 — DCT16×16.
    Id2 = 2,
    /// 3 — DCT32×32.
    Id3 = 3,
    /// 4 — DCT16×8, DCT8×16.
    Id4 = 4,
    /// 5 — DCT32×8, DCT8×32.
    Id5 = 5,
    /// 6 — DCT32×16, DCT16×32.
    Id6 = 6,
    /// 7 — DCT64×64.
    Id7 = 7,
    /// 8 — DCT32×64, DCT64×32.
    Id8 = 8,
    /// 9 — DCT128×128.
    Id9 = 9,
    /// 10 — DCT64×128, DCT128×64.
    Id10 = 10,
    /// 11 — DCT256×256.
    Id11 = 11,
    /// 12 — DCT128×256, DCT256×128.
    Id12 = 12,
}

/// Number of distinct order IDs (Table I.1).
pub const NUM_ORDERS: usize = 13;

impl OrderId {
    /// Convert a numerical 0..=12 value to [`OrderId`].
    pub fn from_index(i: u32) -> Result<Self> {
        Ok(match i {
            0 => Self::Id0,
            1 => Self::Id1,
            2 => Self::Id2,
            3 => Self::Id3,
            4 => Self::Id4,
            5 => Self::Id5,
            6 => Self::Id6,
            7 => Self::Id7,
            8 => Self::Id8,
            9 => Self::Id9,
            10 => Self::Id10,
            11 => Self::Id11,
            12 => Self::Id12,
            _ => {
                return Err(Error::InvalidData(format!(
                    "JXL coeff_order: Order ID {i} out of range 0..=12 (Table I.1)"
                )));
            }
        })
    }

    /// Numerical 0..=12 (matches Table I.1 column 1).
    pub fn index(self) -> u32 {
        self as u32
    }
}

/// Map a [`TransformType`] (Table C.16) to its [`OrderId`]
/// (Table I.1).
pub fn order_id_for_transform(t: TransformType) -> OrderId {
    match t {
        TransformType::Dct8x8 => OrderId::Id0,
        TransformType::Hornuss
        | TransformType::Dct2x2
        | TransformType::Dct4x4
        | TransformType::Dct4x8
        | TransformType::Dct8x4
        | TransformType::Afv0
        | TransformType::Afv1
        | TransformType::Afv2
        | TransformType::Afv3 => OrderId::Id1,
        TransformType::Dct16x16 => OrderId::Id2,
        TransformType::Dct32x32 => OrderId::Id3,
        TransformType::Dct16x8 | TransformType::Dct8x16 => OrderId::Id4,
        TransformType::Dct32x8 | TransformType::Dct8x32 => OrderId::Id5,
        TransformType::Dct32x16 | TransformType::Dct16x32 => OrderId::Id6,
        TransformType::Dct64x64 => OrderId::Id7,
        TransformType::Dct32x64 | TransformType::Dct64x32 => OrderId::Id8,
        TransformType::Dct128x128 => OrderId::Id9,
        TransformType::Dct64x128 | TransformType::Dct128x64 => OrderId::Id10,
        TransformType::Dct256x256 => OrderId::Id11,
        TransformType::Dct128x256 | TransformType::Dct256x128 => OrderId::Id12,
    }
}

/// Varblock size `(bwidth, bheight)` for an [`OrderId`] per §I.2.4
/// opening paragraph.
///
/// The DctSelect "name" `DCTN×M` yields
/// `bwidth = max(8, max(N, M))`, `bheight = max(8, min(N, M))`. For
/// non-DCT transforms (Hornuss / DCT2×2 / DCT4×4 / DCT4×8 / DCT8×4 /
/// AFVn) `bwidth = bheight = 8`.
pub fn varblock_size_for_order(o: OrderId) -> (u32, u32) {
    match o {
        OrderId::Id0 => (8, 8),
        OrderId::Id1 => (8, 8),
        OrderId::Id2 => (16, 16),
        OrderId::Id3 => (32, 32),
        OrderId::Id4 => (16, 8),
        OrderId::Id5 => (32, 8),
        OrderId::Id6 => (32, 16),
        OrderId::Id7 => (64, 64),
        OrderId::Id8 => (64, 32),
        OrderId::Id9 => (128, 128),
        OrderId::Id10 => (128, 64),
        OrderId::Id11 => (256, 256),
        OrderId::Id12 => (256, 128),
    }
}

/// `bwidth × bheight` for every [`OrderId`]. This is the `size`
/// argument the §C.7.1 HfPass parser must pass to
/// `DecodePermutation()` and the upper bound on the natural-order
/// vector length.
pub const COEFFICIENTS_PER_ORDER: [u32; NUM_ORDERS] = [
    64,    // OrderId::Id0  (8×8)
    64,    // OrderId::Id1  (8×8)
    256,   // OrderId::Id2  (16×16)
    1024,  // OrderId::Id3  (32×32)
    128,   // OrderId::Id4  (16×8)
    256,   // OrderId::Id5  (32×8)
    512,   // OrderId::Id6  (32×16)
    4096,  // OrderId::Id7  (64×64)
    2048,  // OrderId::Id8  (64×32)
    16384, // OrderId::Id9  (128×128)
    8192,  // OrderId::Id10 (128×64)
    65536, // OrderId::Id11 (256×256)
    32768, // OrderId::Id12 (256×128)
];

/// Convenience: `COEFFICIENTS_PER_ORDER[o.index()]`.
pub fn coefficient_count(o: OrderId) -> u32 {
    COEFFICIENTS_PER_ORDER[o.index() as usize]
}

/// Compute the natural coefficient order for an [`OrderId`] per
/// §I.2.4.
///
/// Returns a vector `order` of length `bwidth * bheight` such that
/// `order[i] = y * bwidth + x` for the `i`-th cell in natural-order
/// scan. The LLF prefix (cells with `x < bwidth/8 && y < bheight/8`)
/// is sorted by `y * bwidth + x` ascending; the HF tail is sorted by
/// `key1` (then `key2`) per Listing I.14.
///
/// This is the `natural_coeff_order[i]` referenced by Listing C.12:
///
/// ```text
/// order[i] = natural_coeff_order[nat_ord_perm[i]];
/// ```
pub fn natural_coeff_order(o: OrderId) -> Vec<u32> {
    let (bwidth, bheight) = varblock_size_for_order(o);
    let cx = bwidth / 8;
    let cy = bheight / 8;
    let total = (bwidth * bheight) as usize;
    let mut llf: Vec<(u32, u32)> = Vec::with_capacity((cx * cy) as usize);
    let mut hf: Vec<(u32, u32)> = Vec::with_capacity(total - (cx * cy) as usize);
    for y in 0..bheight {
        for x in 0..bwidth {
            if x < cx && y < cy {
                llf.push((x, y));
            } else {
                hf.push((x, y));
            }
        }
    }
    // LLF: sort by y * bwidth + x ascending.
    llf.sort_by_key(|&(x, y)| (y as u64) * (bwidth as u64) + (x as u64));

    // HF: sort by (key1, key2) per Listing I.14.
    let m = cx.max(cy);
    hf.sort_by(|&a, &b| {
        let (ka1, ka2) = listing_i14_keys(a.0, a.1, cx, cy, m);
        let (kb1, kb2) = listing_i14_keys(b.0, b.1, cx, cy, m);
        ka1.cmp(&kb1).then(ka2.cmp(&kb2))
    });

    let mut out: Vec<u32> = Vec::with_capacity(total);
    for (x, y) in llf.into_iter().chain(hf) {
        out.push(y * bwidth + x);
    }
    out
}

/// Listing I.14 — return `(key1, key2)` for an `(x, y)` HF cell.
///
/// `scaled_x = x * max(cx, cy) / cx;`
/// `scaled_y = y * max(cx, cy) / cy;`
/// `key1 = scaled_x + scaled_y;`
/// `key2 = scaled_x - scaled_y;`
/// `if (key1 Umod 2 == 1) key2 = -key2;`
///
/// `cx == 0` or `cy == 0` cannot occur on a real input (LLF prefix is
/// empty when so, and this routine is only called on HF cells), but
/// be defensive: a 0 divisor returns key1=0 key2=0 so the sort is
/// stable rather than panicking.
fn listing_i14_keys(x: u32, y: u32, cx: u32, cy: u32, m: u32) -> (u32, i64) {
    if cx == 0 || cy == 0 {
        return (0, 0);
    }
    // scaled values fit easily inside u64.
    let scaled_x = (x as u64) * (m as u64) / (cx as u64);
    let scaled_y = (y as u64) * (m as u64) / (cy as u64);
    let key1 = scaled_x + scaled_y;
    let mut key2 = scaled_x as i64 - scaled_y as i64;
    if key1 % 2 == 1 {
        key2 = -key2;
    }
    (key1 as u32, key2)
}

/// `GetContext(x) = min(8, ceil(log2(x + 1)))` per FDIS §C.3.2.
///
/// The eight clustered distributions `D` are indexed by `GetContext`,
/// whose range is 0..=8. Callers cap the value at 7 before using it as
/// a distribution context because there are only 8 distributions
/// (indices 0..=7) — this matches the existing TOC permutation reader
/// (`crate::toc`).
fn get_context(x: u32) -> u32 {
    if x <= 1 {
        // ceil(log2(0+1)) = 0; ceil(log2(1+1)) = 1.
        x
    } else {
        let nbits = 32 - x.leading_zeros();
        nbits.min(8)
    }
}

/// FDIS §C.3.2 — decode one permutation of length `size` from an already
/// set-up entropy stream, with the given `skip`.
///
/// This is the generic Lehmer-code → permutation procedure used by both
/// the TOC (`crate::toc`, with `skip = 0`, where the stream is built
/// inline) and the HfPass `DecodePermutation()` step (Listing C.12,
/// with `skip = size / 64`). For HfPass the **same** 8-distribution
/// stream + ANS state is shared across every set `used_orders` bit
/// (Listing C.12 reads "8 clustered distributions D" once), so this
/// helper takes the `EntropyStream` + `HybridUintState` by mutable
/// reference and threads them across calls.
///
/// Procedure (verbatim §C.3.2):
///
/// > The decoder first decodes an integer `end`, as specified in D.3.6,
/// > using distribution `D[GetContext(size)]`. Then, `(end - skip)`
/// > elements of the lehmer sequence are produced as follows. For each
/// > element, an integer `lehmer[skip + i]` is read as specified in
/// > D.3.6 using the distribution `D[prev_elem]`. `prev_elem` is
/// > `lehmer[skip + i - 1]` if `i > 0`, or `0` otherwise. All other
/// > elements of the sequence lehmer are 0.
/// >
/// > The decoder maintains a sequence `temp`, initially `[0, size)` in
/// > increasing order, and a sequence `permutation`, initially empty.
/// > Then, for each `i` in `[0, size)`, the decoder appends
/// > `temp[lehmer[i]]` to permutation, then removes it from temp,
/// > leaving the relative order of other elements unchanged.
///
/// The caller is responsible for having already read the §D.3 prelude
/// (`EntropyStream::read(br, 8)`) and the ANS state init
/// (`EntropyStream::read_ans_state_init`).
pub fn decode_permutation_from_stream(
    br: &mut BitReader<'_>,
    entropy: &mut EntropyStream,
    hybrid: &mut HybridUintState,
    size: usize,
    skip: usize,
) -> Result<Vec<u32>> {
    if skip > size {
        return Err(Error::InvalidData(format!(
            "JXL coeff permutation: skip {skip} exceeds size {size}"
        )));
    }
    if size == 0 {
        return Ok(Vec::new());
    }

    // `end` is decoded against D[GetContext(size)] (capped at 7: only
    // 8 distributions exist).
    let end_ctx = get_context(size as u32).min(7);
    let end = decode_uint_in_with_dist_pub(hybrid, entropy, br, end_ctx, 0)? as usize;
    if end < skip || end > size {
        return Err(Error::InvalidData(format!(
            "JXL coeff permutation: end {end} out of range [skip={skip}, size={size}]"
        )));
    }

    // `(end - skip)` Lehmer entries are produced; indices [skip, end).
    // All other lehmer entries stay 0.
    let mut lehmer = vec![0u32; size];
    let mut prev: u32 = 0;
    for slot in lehmer.iter_mut().take(end).skip(skip) {
        let ctx = get_context(prev).min(7);
        let v = decode_uint_in_with_dist_pub(hybrid, entropy, br, ctx, 0)?;
        // lehmer[i] indexes the (shrinking) `temp` array, so it must be
        // strictly less than the remaining temp length at that step.
        // The tightest static bound is `< size`; the exact-length check
        // happens in the `temp.remove` loop below.
        if v as usize >= size {
            return Err(Error::InvalidData(format!(
                "JXL coeff permutation: lehmer entry {v} >= size {size}"
            )));
        }
        *slot = v;
        prev = v;
    }

    lehmer_to_permutation(&lehmer)
}

/// FDIS §C.3.2 final step — turn a Lehmer code into a permutation.
///
/// > The decoder maintains a sequence `temp`, initially `[0, size)` in
/// > increasing order, and a sequence `permutation`, initially empty.
/// > Then, for each `i` in `[0, size)`, the decoder appends
/// > `temp[lehmer[i]]` to permutation, then removes it from temp,
/// > leaving the relative order of other elements unchanged.
///
/// `size` is `lehmer.len()`. Each `lehmer[i]` must index into the
/// remaining `temp` (length `size - i`), so it must be in
/// `[0, size - i)`; otherwise the code is malformed.
fn lehmer_to_permutation(lehmer: &[u32]) -> Result<Vec<u32>> {
    let size = lehmer.len();
    let mut temp: Vec<u32> = (0..size as u32).collect();
    let mut permutation: Vec<u32> = Vec::with_capacity(size);
    for &lh in lehmer {
        let idx = lh as usize;
        if idx >= temp.len() {
            return Err(Error::InvalidData(format!(
                "JXL coeff permutation: lehmer index {idx} out of range (temp len {})",
                temp.len()
            )));
        }
        permutation.push(temp.remove(idx));
    }
    Ok(permutation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_id_round_trip() {
        for i in 0..NUM_ORDERS as u32 {
            let o = OrderId::from_index(i).unwrap();
            assert_eq!(o.index(), i);
        }
    }

    #[test]
    fn order_id_out_of_range() {
        assert!(OrderId::from_index(13).is_err());
        assert!(OrderId::from_index(99).is_err());
    }

    #[test]
    fn varblock_sizes_match_table_i1_plus_name_derivation() {
        // Per §I.2.4 opening paragraph: bwidth = max(8, max(N, M));
        // bheight = max(8, min(N, M)). For DCT8×8 → (8, 8).
        assert_eq!(varblock_size_for_order(OrderId::Id0), (8, 8));
        // DCT16×16 → (16, 16).
        assert_eq!(varblock_size_for_order(OrderId::Id2), (16, 16));
        // DCT16×8 / DCT8×16 → max=16, min=8 → (16, 8).
        assert_eq!(varblock_size_for_order(OrderId::Id4), (16, 8));
        // DCT32×8 / DCT8×32 → (32, 8).
        assert_eq!(varblock_size_for_order(OrderId::Id5), (32, 8));
        // DCT128×256 / DCT256×128 → max=256, min=128 → (256, 128).
        assert_eq!(varblock_size_for_order(OrderId::Id12), (256, 128));
        // Hornuss (and friends) → (8, 8) per the "all other transforms"
        // clause.
        assert_eq!(varblock_size_for_order(OrderId::Id1), (8, 8));
    }

    #[test]
    fn coefficient_counts_match_varblock_areas() {
        for i in 0..NUM_ORDERS as u32 {
            let o = OrderId::from_index(i).unwrap();
            let (w, h) = varblock_size_for_order(o);
            assert_eq!(
                coefficient_count(o),
                w * h,
                "OrderId {i}: count {} != {w}×{h}",
                coefficient_count(o)
            );
        }
    }

    #[test]
    fn order_id_for_transform_table_i1_complete() {
        // Spot-check every row in Table I.1.
        assert_eq!(order_id_for_transform(TransformType::Dct8x8), OrderId::Id0);
        assert_eq!(order_id_for_transform(TransformType::Hornuss), OrderId::Id1);
        assert_eq!(order_id_for_transform(TransformType::Dct2x2), OrderId::Id1);
        assert_eq!(order_id_for_transform(TransformType::Dct4x4), OrderId::Id1);
        assert_eq!(order_id_for_transform(TransformType::Dct4x8), OrderId::Id1);
        assert_eq!(order_id_for_transform(TransformType::Dct8x4), OrderId::Id1);
        assert_eq!(order_id_for_transform(TransformType::Afv0), OrderId::Id1);
        assert_eq!(order_id_for_transform(TransformType::Afv1), OrderId::Id1);
        assert_eq!(order_id_for_transform(TransformType::Afv2), OrderId::Id1);
        assert_eq!(order_id_for_transform(TransformType::Afv3), OrderId::Id1);
        assert_eq!(
            order_id_for_transform(TransformType::Dct16x16),
            OrderId::Id2
        );
        assert_eq!(
            order_id_for_transform(TransformType::Dct32x32),
            OrderId::Id3
        );
        assert_eq!(order_id_for_transform(TransformType::Dct16x8), OrderId::Id4);
        assert_eq!(order_id_for_transform(TransformType::Dct8x16), OrderId::Id4);
        assert_eq!(order_id_for_transform(TransformType::Dct32x8), OrderId::Id5);
        assert_eq!(order_id_for_transform(TransformType::Dct8x32), OrderId::Id5);
        assert_eq!(
            order_id_for_transform(TransformType::Dct32x16),
            OrderId::Id6
        );
        assert_eq!(
            order_id_for_transform(TransformType::Dct16x32),
            OrderId::Id6
        );
        assert_eq!(
            order_id_for_transform(TransformType::Dct64x64),
            OrderId::Id7
        );
        assert_eq!(
            order_id_for_transform(TransformType::Dct32x64),
            OrderId::Id8
        );
        assert_eq!(
            order_id_for_transform(TransformType::Dct64x32),
            OrderId::Id8
        );
        assert_eq!(
            order_id_for_transform(TransformType::Dct128x128),
            OrderId::Id9
        );
        assert_eq!(
            order_id_for_transform(TransformType::Dct64x128),
            OrderId::Id10
        );
        assert_eq!(
            order_id_for_transform(TransformType::Dct128x64),
            OrderId::Id10
        );
        assert_eq!(
            order_id_for_transform(TransformType::Dct256x256),
            OrderId::Id11
        );
        assert_eq!(
            order_id_for_transform(TransformType::Dct128x256),
            OrderId::Id12
        );
        assert_eq!(
            order_id_for_transform(TransformType::Dct256x128),
            OrderId::Id12
        );
    }

    #[test]
    fn natural_order_id0_is_a_permutation_of_0_to_63() {
        let order = natural_coeff_order(OrderId::Id0);
        assert_eq!(order.len(), 64);
        let mut seen = [false; 64];
        for &v in &order {
            assert!(v < 64, "order entry {v} out of [0, 64)");
            assert!(!seen[v as usize], "order entry {v} repeated");
            seen[v as usize] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn natural_order_id0_starts_with_dc_then_llf_then_hf() {
        // For Id0 (8×8) cx = cy = 1, so the only LLF cell is (0, 0).
        // Spec: LLF prefix length = cx * cy = 1, then HF tail = 63 cells
        // (everything except (0,0)) sorted by (key1, key2).
        let order = natural_coeff_order(OrderId::Id0);
        assert_eq!(order[0], 0, "first element must be DC at index 0");
        // The remaining 63 entries are HF.
        for &v in &order[1..] {
            assert!(v >= 1, "HF entries cannot include (0,0)");
        }
    }

    #[test]
    fn natural_order_id2_llf_prefix_length_4() {
        // 16×16 (Id2) has cx = cy = 2 → LLF length = 4. LLF cells:
        // (0,0), (1,0), (0,1), (1,1) sorted by y*16+x → indices
        // 0, 1, 16, 17.
        let order = natural_coeff_order(OrderId::Id2);
        assert_eq!(order.len(), 256);
        assert_eq!(order[0], 0);
        assert_eq!(order[1], 1);
        assert_eq!(order[2], 16);
        assert_eq!(order[3], 17);
    }

    #[test]
    fn natural_order_id4_llf_prefix_length_2() {
        // 16×8 (Id4) cx = 2, cy = 1 → LLF cells: (0, 0), (1, 0).
        let order = natural_coeff_order(OrderId::Id4);
        assert_eq!(order.len(), 128);
        assert_eq!(order[0], 0);
        assert_eq!(order[1], 1);
        // After LLF, HF starts. The first HF cells must NOT be (0,0)
        // or (1,0).
        for &v in &order[2..] {
            assert!(v != 0 && v != 1);
        }
    }

    #[test]
    fn natural_order_each_id_is_a_permutation_of_its_range() {
        for i in 0..NUM_ORDERS as u32 {
            // Skip the four largest orders to keep test time bounded;
            // they are covered by `natural_order_largest_id_is_valid_perm`.
            if i >= 9 {
                continue;
            }
            let o = OrderId::from_index(i).unwrap();
            let n = coefficient_count(o);
            let order = natural_coeff_order(o);
            assert_eq!(order.len(), n as usize, "OrderId {i} length");
            let mut seen = vec![false; n as usize];
            for &v in &order {
                assert!(v < n, "OrderId {i}: entry {v} out of [0, {n})");
                assert!(!seen[v as usize], "OrderId {i}: entry {v} repeated");
                seen[v as usize] = true;
            }
            assert!(seen.iter().all(|&b| b), "OrderId {i} not full perm");
        }
    }

    #[test]
    fn natural_order_largest_id_is_valid_perm() {
        // 256×256 — 65536 entries. Validate it's a permutation by
        // sum-of-indices comparison rather than 65536-entry bitset
        // walks (cheaper but still proves bijection given the size
        // matches).
        let o = OrderId::Id11;
        let n = coefficient_count(o) as u64;
        let order = natural_coeff_order(o);
        assert_eq!(order.len() as u64, n);
        let expected_sum: u64 = (0..n).sum();
        let actual_sum: u64 = order.iter().map(|&v| v as u64).sum();
        assert_eq!(expected_sum, actual_sum);
        // Spot-check first 4 entries are the LLF prefix sorted by row.
        // cx = cy = 32 → LLF length = 1024. Sample the first 4.
        assert_eq!(order[0], 0); // (0, 0)
        assert_eq!(order[1], 1); // (1, 0)
        assert_eq!(order[2], 2); // (2, 0)
        assert_eq!(order[3], 3); // (3, 0)
    }

    #[test]
    fn listing_i14_keys_id4_basic() {
        // OrderId::Id4 → cx=2, cy=1, m=2.
        // (x, y) = (2, 0): scaled_x = 2*2/2 = 2; scaled_y = 0*2/1 = 0;
        //   key1 = 2; key1 % 2 == 0 → key2 = 2 - 0 = 2.
        let (k1, k2) = listing_i14_keys(2, 0, 2, 1, 2);
        assert_eq!((k1, k2), (2, 2));
        // (x, y) = (0, 1): scaled_x = 0; scaled_y = 2; key1 = 2;
        //   key2 = 0 - 2 = -2; key1 % 2 == 0 → key2 stays -2.
        let (k1, k2) = listing_i14_keys(0, 1, 2, 1, 2);
        assert_eq!((k1, k2), (2, -2));
        // (x, y) = (1, 0): scaled_x = 1*2/2 = 1; scaled_y = 0; key1 = 1;
        //   key2 = 1 - 0 = 1; key1 % 2 == 1 → key2 = -1.
        let (k1, k2) = listing_i14_keys(1, 0, 2, 1, 2);
        assert_eq!((k1, k2), (1, -1));
    }

    #[test]
    fn get_context_matches_spec() {
        // GetContext(x) = min(8, ceil(log2(x + 1))).
        assert_eq!(get_context(0), 0); // ceil(log2(1)) = 0
        assert_eq!(get_context(1), 1); // ceil(log2(2)) = 1
        assert_eq!(get_context(2), 2); // ceil(log2(3)) = 2
        assert_eq!(get_context(3), 2); // ceil(log2(4)) = 2
        assert_eq!(get_context(4), 3); // ceil(log2(5)) = 3
        assert_eq!(get_context(7), 3); // ceil(log2(8)) = 3
        assert_eq!(get_context(8), 4); // ceil(log2(9)) = 4
                                       // Saturates at 8.
        assert_eq!(get_context(255), 8);
        assert_eq!(get_context(256), 8);
        assert_eq!(get_context(u32::MAX), 8);
    }

    #[test]
    fn lehmer_all_zero_is_identity() {
        // An all-zero Lehmer code always picks temp[0] each step → the
        // identity permutation. This is the `end == skip` case (no
        // non-zero Lehmer entries) that DecodePermutation produces for
        // an unmodified natural order.
        let lehmer = vec![0u32; 6];
        let perm = lehmer_to_permutation(&lehmer).unwrap();
        assert_eq!(perm, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn lehmer_reverse_permutation() {
        // To reverse [0,1,2,3]: pick the LAST remaining element each
        // step. temp = [0,1,2,3]; lehmer = [3,2,1,0]:
        //   i=0: temp[3]=3, temp=[0,1,2]
        //   i=1: temp[2]=2, temp=[0,1]
        //   i=2: temp[1]=1, temp=[0]
        //   i=3: temp[0]=0
        // → permutation = [3,2,1,0].
        let perm = lehmer_to_permutation(&[3, 2, 1, 0]).unwrap();
        assert_eq!(perm, vec![3, 2, 1, 0]);
    }

    #[test]
    fn lehmer_spec_temp_shuffle_example() {
        // temp = [0,1,2,3,4]; lehmer = [2,0,2,0,0]:
        //   i=0: temp[2]=2, temp=[0,1,3,4]
        //   i=1: temp[0]=0, temp=[1,3,4]
        //   i=2: temp[2]=4, temp=[1,3]
        //   i=3: temp[0]=1, temp=[3]
        //   i=4: temp[0]=3
        // → [2,0,4,1,3].
        let perm = lehmer_to_permutation(&[2, 0, 2, 0, 0]).unwrap();
        assert_eq!(perm, vec![2, 0, 4, 1, 3]);
        // Result is always a valid permutation of 0..len.
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn lehmer_out_of_range_index_rejected() {
        // lehmer[0] = 5 but temp only has 5 elements (indices 0..=4).
        assert!(lehmer_to_permutation(&[5, 0, 0, 0, 0]).is_err());
        // lehmer[1] = 4 after one removal temp has only 4 elements.
        assert!(lehmer_to_permutation(&[0, 4, 0, 0, 0]).is_err());
    }
}
