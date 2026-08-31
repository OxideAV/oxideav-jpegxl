//! `HfPass` bundle — ISO/IEC FDIS 18181-1:2021 §C.7.
//!
//! ## Scope (round 90)
//!
//! Round 90 lands the **structural** parse for the HfPass section of
//! the codestream. Round 454 corrects the multiplicity: the FDIS
//! §C.7.1 first sentence ("read `num_hf_presets` times, once for each
//! preset") is SUPERSEDED by ISO/IEC 18181-1:2024 §I.3.1, which reads
//! the coefficient orders `order[p][b][c]` ONCE PER PASS with no
//! preset dimension — `num_hf_presets` multiplies only the §C.7.2 /
//! I.3.3 histogram count and the I.4 `hfp` histogram-offset range.
//! Wire-arbitrated on both staged 2-preset streams (single- and
//! 3-pass): one bundle per pass consumes the HfGlobal section to its
//! byte-padded end with every D.3.3 ANS closure passing; every
//! per-preset repetition variant misparses.
//!
//! ### Per-preset wire format (§C.7.1 Listing C.12)
//!
//! ```text
//! used_orders = U32(Val(0x5F), Val(0x13), Val(0), Bits(13));  // 13-bit mask
//! if (used_orders != 0)
//!   [[read 8 clustered distributions D according to subclause D.3]]
//! for (b = 0; b < 13; b++)
//!   if ((used_orders & (1 << b)) != 0) {
//!     nat_ord_perm = DecodePermutation();
//!     for [[each i]]
//!       order[i] = natural_coeff_order[nat_ord_perm[i]];
//!   } else {
//!     for [[each i]]
//!       order[i] = natural_coeff_order[i];
//!   }
//! ```
//!
//! ### Per-pass histogram (§C.7.2)
//!
//! After Listing C.12 the decoder reads
//! `495 × num_hf_presets × nb_block_ctx` clustered distributions per
//! the §D.3 ANS-distribution machinery. The `495` factor comes from
//! the per-block per-context histogram dimensioning that §C.8.3
//! consumes (the `CoefficientContext` k argument has a 64-element
//! range while `BlockContext()` returns up to `nb_block_ctx` values,
//! plus the various `NonZerosContext` offsets — see Listing C.13).
//!
//! ### Envelope (round 90 + round 133)
//!
//! Round 90 implemented the `used_orders == 0` fast path end-to-end
//! (all 13 orders take their natural-coefficient-order per
//! [`crate::coeff_order::natural_coeff_order`]).
//!
//! **Round 133** lands the `used_orders != 0` path. Listing C.12's
//! "read 8 clustered distributions D" is wired to a single shared
//! [`crate::modular_fdis::EntropyStream`] (`num_dist = 8`), read once;
//! every set `used_orders` bit then runs `DecodePermutation()` (§C.3.2,
//! [`crate::coeff_order::decode_permutation_from_stream`]) against that
//! same stream + ANS state. §C.7.1 fixes the parameters: `size` is the
//! coefficient count covered by the order's `dcts`, and
//! `skip = size / 64`. The final order is
//! `order[i] = natural_coeff_order[nat_ord_perm[i]]`.
//!
//! The §C.7.2 histogram read remains deferred — the contract still
//! computes and exposes [`HfPass::num_histogram_distributions`] so the
//! next round knows exactly how many clustered distributions to read.
//!
//! ### What this parser exposes
//!
//! * [`HfPass`] — ONE per pass (round-454 erratum; the FDIS's
//!   one-per-preset lead-in is superseded by 2024 §I.3.1); stores
//!   `used_orders`, the 13 per-channel final coefficient orders
//!   (natural or permuted), and the histogram-size invariants the
//!   §C.7.2 read consumes.

use oxideav_core::{Error, Result};

use crate::ans::hybrid::HybridUintState;
use crate::bitreader::{BitReader, U32Dist};
use crate::coeff_order::{
    coefficient_count, decode_permutation_from_stream, natural_coeff_order, OrderId,
    COEFFICIENTS_PER_ORDER, NUM_ORDERS,
};
use crate::modular_fdis::EntropyStream;

/// `HfPass` bundle for a single PASS (§C.7.1 / 2024 §I.3.1 + §C.7.2)
/// — shared by every preset of that pass (round-454 erratum; see the
/// module notes).
#[derive(Debug, Clone)]
pub struct HfPass {
    /// `used_orders` (Listing C.12, 13-bit mask). Bit `b` set means
    /// order ID `b` carries a permutation; otherwise the order takes
    /// its natural ordering.
    pub used_orders: u32,
    /// Final coefficient order per channel × [`OrderId`] — indexed
    /// `orders[channel][order_id]` with the §C.8.3 channel numbering
    /// (0 = X, 1 = Y, 2 = B). For an order whose `used_orders` bit is
    /// 0, every channel takes [`natural_coeff_order(o)`]; for a set
    /// bit, THREE §C.3.2 `DecodePermutation()` results are decoded
    /// from the shared stream — one per channel in the §C.8.3 decode
    /// sequence **Y, X, B** — and
    /// `orders[c][b][i] = natural_coeff_order[nat_ord_perm_c[i]]`.
    ///
    /// **Round-437 erratum (fixture-pinned).** Listing C.12 prints ONE
    /// `DecodePermutation()` per set bit; the wire carries one per
    /// colour channel. Pinned bit-exactly on the staged
    /// `patches-256x256` fixture, whose clean-room decode trace gives
    /// per-stream ground truth for the HfGlobal section: under the
    /// printed one-per-bit reading the §C.7.2 read starts 281 bits
    /// early and its prelude misparses, while under one-per-channel
    /// the very next bit begins a §C.7.2 whose parse reproduces the
    /// trace exactly (`lz77=0`, ANS, 74 clusters) and the frame's
    /// AC decode comes out reference-accurate.
    pub orders: [[Vec<u32>; NUM_ORDERS]; 3],
    /// Number of clustered distributions the §C.7.2 step reads from
    /// the codestream: `495 × num_hf_presets × nb_block_ctx`. The
    /// histogram bytes themselves are not yet consumed by this
    /// round — the next round consumes them once the shared ANS
    /// stream is wired.
    pub num_histogram_distributions: u64,
}

impl HfPass {
    /// Parse a single per-pass HfPass bundle per Listing C.12 / 2024
    /// §I.3.1 + §C.7.2.
    ///
    /// * `br` must be positioned at the start of the preset's
    ///   wire-format bits.
    /// * `num_hf_presets` and `nb_block_ctx` are inherited from
    ///   HfGlobal (§I.2.6) and HfBlockContext (§I.2.2), respectively;
    ///   the parser uses them only to compute
    ///   `num_histogram_distributions`.
    ///
    /// When `used_orders != 0` the decoder reads the shared 8-cluster
    /// ANS stream (Listing C.12) once and then THREE
    /// `DecodePermutation()` (§C.3.2) per set bit — one per colour
    /// channel in Y, X, B order (round-437 erratum; see [`Self::orders`]) —
    /// building `orders[c][b][i] = natural_coeff_order[nat_ord_perm[i]]`.
    ///
    /// Returns `Err(InvalidData)` when the parsed `used_orders` exceeds
    /// its 13-bit cap (defensive: the
    /// `U32(Val(0x5F), Val(0x13), Val(0), Bits(13))` selector should
    /// never yield a value > `0x1FFF` for the explicit-bits arm or one
    /// of the three sentinel values `0x5F` / `0x13` / `0`).
    pub fn read(br: &mut BitReader<'_>, num_hf_presets: u32, nb_block_ctx: u32) -> Result<Self> {
        let used_orders = br.read_u32([
            U32Dist::Val(0x5F),
            U32Dist::Val(0x13),
            U32Dist::Val(0),
            U32Dist::Bits(13),
        ])?;
        // Cap: the only legal values are 0x5F, 0x13, 0 and any 13-bit
        // word (≤ 0x1FFF). Anything else is a decoder-side bug.
        if used_orders != 0x5F && used_orders != 0x13 && used_orders > 0x1FFF {
            return Err(Error::InvalidData(format!(
                "JXL HfPass: used_orders 0x{used_orders:X} exceeds 13-bit cap and isn't a \
                 sentinel value"
            )));
        }

        let orders = if used_orders != 0 {
            // Listing C.12: "read 8 clustered distributions D according
            // to subclause D.3". The same 8-distribution stream + ANS
            // state is shared by EVERY DecodePermutation() call in the
            // per-bit loop below — it is read ONCE here.
            //
            // Round 437 — the round-408 "exact bit consumption still
            // wrong somewhere" gap is CLOSED: the per-entry context
            // reading + distribution count are resolved by the
            // fdis-errata.md Part 8.3 six-way bisection (see
            // `crate::coeff_order::PermStreamConfig`), with the D.3.3
            // ANS final-state closure below as the decisive per-stream
            // oracle and the staged `progressive-ac-multipass` fixture
            // as the end-to-end pin.
            let cfg = crate::coeff_order::perm_stream_config();
            let mut entropy = EntropyStream::read(br, cfg.num_dists as usize)?;
            entropy.read_ans_state_init(br)?;
            let mut hybrid = HybridUintState::new(entropy.lz77, entropy.lz_len_conf);
            let orders = build_permuted_orders(br, &mut entropy, &mut hybrid, used_orders)?;
            // D.3.3: "the decoder checks that the final state of the
            // ANS decoder is 0x130000" — the shared permutation
            // stream ends with the last DecodePermutation() symbol,
            // so the invariant must hold HERE. A miss means the
            // stream was misparsed; erroring loudly upholds the
            // crate's no-silent-misparse contract. (Prefix-coded
            // streams carry no ANS state; nothing to check.)
            if let Some(ans) = entropy.ans_state.as_ref() {
                if !ans.final_state() {
                    return Err(Error::InvalidData(format!(
                        "JXL HfPass: ANS final-state invariant (0x130000) failed at the end \
                         of the C.3.2 permutation stream (state {:#x}) - used_orders \
                         sub-stream misparsed",
                        ans.state()
                    )));
                }
            }
            orders
        } else {
            // used_orders == 0 → every order is the natural order.
            build_natural_orders()
        };

        // §C.7.2: number of clustered distributions, routed through
        // the typed sizing primitive so the spec constant has one
        // home and the zero-input guards run consistently across
        // every call site that needs the §C.7.2 read size.
        let num_histogram_distributions =
            crate::hf_coeff_histogram_size::HfCoefficientHistogramSize::new(
                num_hf_presets,
                nb_block_ctx,
            )?
            .num_distributions();

        Ok(Self {
            used_orders,
            orders,
            num_histogram_distributions,
        })
    }

    /// Look up the final coefficient order for a colour channel
    /// (§C.8.3 numbering: 0 = X, 1 = Y, 2 = B) and an [`OrderId`].
    pub fn order_for(&self, channel: u32, o: OrderId) -> &[u32] {
        &self.orders[(channel as usize).min(2)][o.index() as usize]
    }
}

/// Build the per-channel 13-element natural-order set (every order =
/// natural, identical across the three channels).
fn build_natural_orders() -> [[Vec<u32>; NUM_ORDERS]; 3] {
    let one = build_natural_orders_one();
    [one.clone(), one.clone(), one]
}

/// Build the 13-element natural-order set (every order = natural).
fn build_natural_orders_one() -> [Vec<u32>; NUM_ORDERS] {
    [
        natural_coeff_order(OrderId::Id0),
        natural_coeff_order(OrderId::Id1),
        natural_coeff_order(OrderId::Id2),
        natural_coeff_order(OrderId::Id3),
        natural_coeff_order(OrderId::Id4),
        natural_coeff_order(OrderId::Id5),
        natural_coeff_order(OrderId::Id6),
        natural_coeff_order(OrderId::Id7),
        natural_coeff_order(OrderId::Id8),
        natural_coeff_order(OrderId::Id9),
        natural_coeff_order(OrderId::Id10),
        natural_coeff_order(OrderId::Id11),
        natural_coeff_order(OrderId::Id12),
    ]
}

/// Build the 13-element order set when `used_orders != 0`, decoding a
/// permutation for each set bit (Listing C.12 + §C.7.1 DecodePermutation).
///
/// Per Listing C.12, for every order ID `b` in 0..13:
///
/// * if `used_orders & (1 << b)` is set, decode `nat_ord_perm =
///   DecodePermutation()` and set `order[i] =
///   natural_coeff_order[nat_ord_perm[i]]`;
/// * otherwise `order[i] = natural_coeff_order[i]` (the natural order).
///
/// §C.7.1 fixes the `DecodePermutation()` parameters: `size` is the
/// number of coefficients covered by the order's `dcts` (i.e.
/// [`coefficient_count`]), and `skip = size / 64`. The `entropy` +
/// `hybrid` state is the single shared 8-distribution stream read once
/// by the caller; it is threaded across every set-bit permutation.
fn build_permuted_orders(
    br: &mut BitReader<'_>,
    entropy: &mut EntropyStream,
    hybrid: &mut HybridUintState,
    used_orders: u32,
) -> Result<[[Vec<u32>; NUM_ORDERS]; 3]> {
    // Start from the all-natural baseline; overwrite the permuted ones.
    let mut orders = build_natural_orders();
    for b in 0..NUM_ORDERS as u32 {
        if (used_orders & (1 << b)) == 0 {
            continue;
        }
        let order_id = OrderId::from_index(b)?;
        let natural = natural_coeff_order(order_id);
        let size = natural.len();
        let skip = size / 64;
        // Round-437 erratum: the wire carries THREE §C.3.2 permutations
        // per set bit — one per colour channel (Listing C.12 prints
        // only one). Pinned bit-exactly by the staged `patches-256x256`
        // clean-room decode trace (the §C.7.2 stream that follows
        // starts exactly at the section position the trace records,
        // and the whole HfGlobal section lands on the trace's
        // AC_GLOBAL_END bit count) and by the D.3.3 ANS final-state
        // closure on ANS-coded specimens, which fails under the
        // printed one-per-bit reading and closes under this one.
        //
        // Round-444 refinement: the per-channel ASSIGNMENT is the
        // Listing C.13 channel-INDEX order X (0), Y (1), B (2) — NOT
        // the §C.8.3 per-varblock decode sequence Y, X, B that round
        // 437 assumed. The assignment is invisible to every
        // bit-position oracle (all three permutations are decoded
        // back to back whatever the naming), so round 437 never
        // actually arbitrated it; a custom-orders impulse specimen
        // does — under Y-first assignment the Y channel's dominant
        // Hornuss corner coefficient (§I.2.3.5's `coefficients(x + 2,
        // y + 2)` sample driver) lands on the wrong cell and the dot
        // vanishes, under index order it lands exactly and the decode
        // matches the reference band.
        for channel in [0usize, 1, 2] {
            let nat_ord_perm = decode_permutation_from_stream(br, entropy, hybrid, size, skip)?;
            if nat_ord_perm.len() != size {
                return Err(Error::InvalidData(format!(
                    "JXL HfPass: DecodePermutation for order {b} returned {} entries (expected {size})",
                    nat_ord_perm.len()
                )));
            }
            // order[i] = natural_coeff_order[nat_ord_perm[i]] (Listing C.12).
            let mut permuted = Vec::with_capacity(size);
            for &p in &nat_ord_perm {
                let pi = p as usize;
                let v = *natural.get(pi).ok_or_else(|| {
                    Error::InvalidData(format!(
                        "JXL HfPass: permutation index {pi} out of range for order {b} (size {size})"
                    ))
                })?;
                permuted.push(v);
            }
            orders[channel][b as usize] = permuted;
        }
    }
    Ok(orders)
}

/// `bwidth × bheight` totals for the 13 orders, re-exported for
/// downstream §C.8.3 callers that iterate over the per-block
/// coefficient count.
pub const ORDER_COEFFICIENT_COUNTS: [u32; NUM_ORDERS] = COEFFICIENTS_PER_ORDER;

/// Round-90 sanity check: every `OrderId`'s natural order has length
/// matching [`COEFFICIENTS_PER_ORDER`]. Re-exported for downstream
/// consumers that want a self-test surface.
pub fn coefficient_count_for_order(o: OrderId) -> u32 {
    coefficient_count(o)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ans::test_helpers::pack_lsb;

    /// Pack a U32 selector value. For HfPass `used_orders` the U32
    /// distribution is:
    ///   - selector 0 → Val(0x5F)   (2-bit selector, no payload)
    ///   - selector 1 → Val(0x13)   (2-bit selector, no payload)
    ///   - selector 2 → Val(0)      (2-bit selector, no payload)
    ///   - selector 3 → Bits(13)    (2-bit selector + 13-bit payload)
    fn pack_used_orders_zero() -> Vec<u8> {
        // selector 2 → Val(0) → 2 bits, value=2 LSB-first.
        pack_lsb(&[(2, 2)])
    }

    fn pack_used_orders_5f() -> Vec<u8> {
        // selector 0 → Val(0x5F) → 2 bits, value=0.
        pack_lsb(&[(0, 2)])
    }

    fn pack_used_orders_arbitrary(mask: u32) -> Vec<u8> {
        // selector 3 → Bits(13) → 2 bits selector value=3, then 13-bit payload.
        pack_lsb(&[(3, 2), (mask, 13)])
    }

    #[test]
    fn hf_pass_used_orders_zero_natural_for_all_ids() {
        let bytes = pack_used_orders_zero();
        let mut br = BitReader::new(&bytes);
        let hp = HfPass::read(&mut br, 1, 15).unwrap();
        assert_eq!(hp.used_orders, 0);
        for i in 0..NUM_ORDERS as u32 {
            let o = OrderId::from_index(i).unwrap();
            let expected = natural_coeff_order(o);
            assert_eq!(hp.order_for(1, o), expected.as_slice(), "order {i}");
        }
        // 495 × 1 × 15 = 7425
        assert_eq!(hp.num_histogram_distributions, 7425);
    }

    #[test]
    fn hf_pass_used_orders_5f_attempts_stream() {
        // used_orders = 0x5F (≠ 0) now takes the DecodePermutation path:
        // the parser reads the shared 8-distribution stream. With only
        // the `used_orders` selector bits packed, the truncated stream
        // produces an error — but NOT the old `Unsupported` deferral.
        let bytes = pack_used_orders_5f();
        let mut br = BitReader::new(&bytes);
        let r = HfPass::read(&mut br, 1, 1);
        assert!(r.is_err());
        assert!(
            !matches!(r, Err(Error::Unsupported(_))),
            "used_orders != 0 must no longer return Unsupported"
        );
    }

    #[test]
    fn hf_pass_used_orders_explicit_bits_attempts_stream() {
        // used_orders = 0x0007 (3 bits set; not 0) → same: the stream
        // read is attempted, no Unsupported deferral.
        let bytes = pack_used_orders_arbitrary(0x0007);
        let mut br = BitReader::new(&bytes);
        let r = HfPass::read(&mut br, 1, 1);
        assert!(r.is_err());
        assert!(!matches!(r, Err(Error::Unsupported(_))));
    }

    #[test]
    fn hf_pass_used_orders_explicit_bits_zero_is_zero() {
        // selector 3, payload = 0 → used_orders = 0. This should hit
        // the "natural order" branch even though the selector is the
        // explicit-bits one (the 13-bit field happens to be all zero).
        let bytes = pack_used_orders_arbitrary(0);
        let mut br = BitReader::new(&bytes);
        let hp = HfPass::read(&mut br, 2, 15).unwrap();
        assert_eq!(hp.used_orders, 0);
        assert_eq!(hp.num_histogram_distributions, 495u64 * 2 * 15);
    }

    #[test]
    fn hf_pass_read_zero_presets_still_sizes_histograms_safely() {
        // num_hf_presets = 0 is rejected by the histogram sizing
        // primitive (HfGlobal §I.2.6 guarantees ≥ 1); HfPass::read
        // surfaces that as InvalidData rather than a zero-sized
        // histogram contract.
        let bytes = pack_lsb(&[(2, 2)]);
        let mut br = BitReader::new(&bytes);
        let r = HfPass::read(&mut br, 0, 1);
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    #[test]
    fn coefficient_count_for_order_matches_table() {
        for i in 0..NUM_ORDERS as u32 {
            let o = OrderId::from_index(i).unwrap();
            assert_eq!(
                coefficient_count_for_order(o),
                COEFFICIENTS_PER_ORDER[i as usize]
            );
        }
    }
}
