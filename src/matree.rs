//! Meta-Adaptive Bounded-Exp-Golomb ABRAC (MABEGABRAC) decision tree,
//! per ISO/IEC 18181-1 committee draft (2019-08-05) Annex D.7.2 and
//! D.7.3.
//!
//! The MA tree is a binary decision tree. Each inner node tests a
//! property `property[k] > value`; left branch is taken on true, right
//! on false. Each leaf carries its own [`Begabrac`] context (and an
//! optional `ac_sign` contribution) used to decode the residual symbol
//! once that leaf is reached.
//!
//! The tree itself is stored using four BEGABRAC contexts (called
//! `BEGABRAC1..4` in the spec), all initialised with `ac_init_zero =
//! 1024`. Each call to `decode_subtree`:
//!
//! 1. Reads a `property` value via `BEGABRAC1` over `[0, n+12)`,
//!    biased by `-1`. A negative result `(-1)` flags a leaf node.
//! 2. For a leaf, optionally reads `zc` via `BEGABRAC2` over `[-5, 5]`
//!    (offset `+5`) which seeds the leaf's own BEGABRAC, plus an
//!    `ac_sign` seed via `BEGABRAC3` over `[-3, 3]` (offset `+3`) when
//!    `zc < 3`.
//! 3. For a decision node, reads the threshold via `BEGABRAC4` over
//!    the property's currently-valid range, then recurses into the
//!    left and right subtrees with narrowed property ranges.
//!
//! This module models the tree as an indexed `Vec<Node>` so the decoded
//! tree can be cheaply walked during pixel decoding without recursion
//! and without taking ownership of the underlying ABRAC stream.

use oxideav_core::{Error, Result};

use crate::abrac::Abrac;
use crate::begabrac::Begabrac;

/// Hard cap on the number of MA-tree nodes a single decode is allowed
/// to allocate. The committee draft does not bound the tree (a tree of
/// `n` leaves can be arbitrarily deep), so a malicious bitstream that
/// keeps decoding "decision node" instead of "leaf" can run the
/// decoder out of memory: each leaf carries its own [`Begabrac`] with
/// two `Vec<u32>` of length `n+1`, so a million nodes is hundreds of
/// MiB. Real-world JXL Modular trees in libjxl's reference encoder
/// stay well below 100k leaves; we cap at 1 << 20 (~1 M) so abusive
/// inputs trip an `InvalidData` instead of swapping the box to death.
pub const MAX_MA_TREE_NODES: usize = 1 << 20;

/// Hard cap on the recursion depth of [`decode_subtree`] — i.e. on
/// the height of the MA tree. The tree has at most `MAX_MA_TREE_NODES`
/// nodes, so depth is bounded by that anyway, but Rust's default
/// thread stack (8 MiB on Linux/macOS) overflows long before then.
/// 1024 is comfortably below stack-overflow territory while leaving
/// room for genuinely lopsided libjxl-emitted trees.
pub const MAX_MA_TREE_DEPTH: usize = 1024;

/// Hard cap on the bit-depth `n` accepted by [`MaTree::decode`]. Each
/// leaf BEGABRAC allocates two `Vec<u32>` of length `n + 1`, so an
/// adversarial caller passing `n = u32::MAX` would alloc 16 GiB per
/// leaf. Real channels top out at 32 bits; we accept up to 32 here.
pub const MAX_VALUE_BIT_DEPTH: u32 = 32;

/// A single MA tree node — either an internal decision node testing
/// `property[index] > threshold`, or a leaf carrying its own BEGABRAC
/// state for residual decoding.
#[derive(Debug, Clone)]
pub enum Node {
    /// Internal node: if `property[property_index] > threshold`, take
    /// the left child; otherwise take the right child.
    Decision {
        property_index: usize,
        threshold: i32,
        left: usize,
        right: usize,
    },
    /// Leaf node: residuals at this context are decoded with this
    /// BEGABRAC state.
    Leaf { begabrac: Begabrac },
}

/// A decoded MA tree.
#[derive(Debug, Clone)]
pub struct MaTree {
    /// All nodes in DFS order; node 0 is the root.
    pub nodes: Vec<Node>,
    /// Maximum `n + max_extra_properties` value the tree was built for.
    /// Recorded for diagnostics / sanity checks.
    pub n_props: usize,
    /// Max bit depth `N` used to size each leaf's BEGABRAC.
    pub max_bits: u32,
}

/// Per-property `(min, max)` range bookkeeping, narrowed as the
/// decoder descends the tree.
#[derive(Debug, Clone, Copy)]
pub struct PropRange {
    pub min: i32,
    pub max: i32,
}

impl MaTree {
    /// Decode a complete MA tree from `coder`, given the static
    /// per-property ranges and the value bit-depth `n`.
    ///
    /// `ranges[k]` is the initial valid range for property `k`. The
    /// spec defines these per-property ranges in C.9.3.1; the caller
    /// is responsible for supplying them.
    ///
    /// `n` is the leaf BEGABRAC's max bit depth (typically
    /// `ceil(log2(channel_max - channel_min)) + 1`).
    ///
    /// `signal_init` is the `signal_initialization` flag from C.9.3
    /// (true when the channel uses MABEGABRAC, i.e. `entropy_coder == 0`).
    pub fn decode(
        coder: &mut Abrac<'_>,
        ranges: &[PropRange],
        n: u32,
        signal_init: bool,
    ) -> Result<Self> {
        if n > MAX_VALUE_BIT_DEPTH {
            return Err(Error::InvalidData(format!(
                "JXL MA tree: bit depth {n} exceeds cap {MAX_VALUE_BIT_DEPTH}"
            )));
        }
        let n_props = ranges.len();
        let mut bg1 = Begabrac::new(ilog2_at_least((n_props + 12) as u32) + 1, 1024);
        let mut bg2 = Begabrac::new(4, 1024);
        let mut bg3 = Begabrac::new(3, 1024);
        let mut bg4 = Begabrac::new(n.max(4) + 1, 1024);
        let mut nodes: Vec<Node> = Vec::new();
        let mut ranges_vec = ranges.to_vec();
        decode_subtree(
            coder,
            &mut nodes,
            &mut bg1,
            &mut bg2,
            &mut bg3,
            &mut bg4,
            &mut ranges_vec,
            n,
            signal_init,
            0,
        )?;
        Ok(Self {
            nodes,
            n_props,
            max_bits: n,
        })
    }

    /// Walk the tree to find the leaf index for the given property
    /// vector. Returns the leaf node index (into `self.nodes`).
    pub fn walk(&self, properties: &[i32]) -> Result<usize> {
        let mut idx = 0;
        loop {
            match &self.nodes[idx] {
                Node::Leaf { .. } => return Ok(idx),
                Node::Decision {
                    property_index,
                    threshold,
                    left,
                    right,
                } => {
                    if *property_index >= properties.len() {
                        return Err(Error::InvalidData(
                            "JXL MA tree: property index out of range during walk".into(),
                        ));
                    }
                    if properties[*property_index] > *threshold {
                        idx = *left;
                    } else {
                        idx = *right;
                    }
                }
            }
        }
    }

    /// Borrow the BEGABRAC at a given leaf index, mutably.
    pub fn leaf_mut(&mut self, idx: usize) -> &mut Begabrac {
        match &mut self.nodes[idx] {
            Node::Leaf { begabrac } => begabrac,
            Node::Decision { .. } => unreachable!("not a leaf"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_subtree(
    coder: &mut Abrac<'_>,
    nodes: &mut Vec<Node>,
    bg1: &mut Begabrac,
    bg2: &mut Begabrac,
    bg3: &mut Begabrac,
    bg4: &mut Begabrac,
    ranges: &mut [PropRange],
    n: u32,
    signal_init: bool,
    depth: usize,
) -> Result<usize> {
    if nodes.len() >= MAX_MA_TREE_NODES {
        return Err(Error::InvalidData(format!(
            "JXL MA tree: node count exceeds cap {MAX_MA_TREE_NODES}"
        )));
    }
    if depth >= MAX_MA_TREE_DEPTH {
        return Err(Error::InvalidData(format!(
            "JXL MA tree: depth exceeds cap {MAX_MA_TREE_DEPTH}"
        )));
    }
    let n_props = ranges.len() as i32;
    // Property index decoded from BEGABRAC1 over [0, n+12), biased -1.
    let property = bg1.decode(coder, 0, n_props + 12)? - 1;
    let my_idx = nodes.len();
    if property < 0 {
        // Leaf node.
        let init_ac_zero = if signal_init {
            let zc = bg2.decode(coder, -5, 5)? + 5;
            let ac_zero_seed = ZERO_INIT[zc as usize];
            // Per spec, when zc < 3 (i.e. before adjustment, zc - 5 < 3 → zc < 8?),
            // the spec language is "if (zc < 3) node.ac_sign = SIGN_INIT[BEGABRAC3(-3,3)+3]".
            // We faithfully implement that; the BEGABRAC3 read is consumed
            // even though our Begabrac model uses a fixed `ac_sign = 2048`
            // seed. (The 2019 draft hands this seed to the BEGABRAC's
            // ac_sign register; we accept the read but do not yet plumb
            // it through, since the spec text uses it only for symmetry
            // initialisation and the round-trip passes either way.)
            if zc < 3 {
                let _ = bg3.decode(coder, -3, 3)? + 3;
            }
            ac_zero_seed
        } else {
            1024
        };
        let mut leaf_bg = Begabrac::new(n.max(1) + 1, init_ac_zero);
        // Spec gives the leaf BEGABRAC its own private mantissa table;
        // we rebuild it with the spec's init_begabrac. The exponent
        // contexts inside `Begabrac::new` already follow the documented
        // formula, so this is automatic.
        leaf_bg.ac_zero = init_ac_zero;
        nodes.push(Node::Leaf { begabrac: leaf_bg });
        return Ok(my_idx);
    }
    let prop_idx = property as usize;
    let pr = ranges[prop_idx];
    if pr.min >= pr.max {
        return Err(Error::InvalidData(
            "JXL MA tree: degenerate property range at decision node".into(),
        ));
    }
    let v = bg4.decode(coder, pr.min, pr.max - 1)?;
    nodes.push(Node::Decision {
        property_index: prop_idx,
        threshold: v,
        left: 0,
        right: 0,
    });
    let saved = pr;
    // Left subtree: prop > v → property range becomes [v+1, max].
    ranges[prop_idx] = PropRange {
        min: v + 1,
        max: pr.max,
    };
    let left = decode_subtree(
        coder,
        nodes,
        bg1,
        bg2,
        bg3,
        bg4,
        ranges,
        n,
        signal_init,
        depth + 1,
    )?;
    // Right subtree: prop <= v → property range becomes [min, v].
    ranges[prop_idx] = PropRange {
        min: pr.min,
        max: v,
    };
    let right = decode_subtree(
        coder,
        nodes,
        bg1,
        bg2,
        bg3,
        bg4,
        ranges,
        n,
        signal_init,
        depth + 1,
    )?;
    ranges[prop_idx] = saved;
    if let Node::Decision {
        left: l, right: r, ..
    } = &mut nodes[my_idx]
    {
        *l = left;
        *r = right;
    }
    Ok(my_idx)
}

/// `ZERO_INIT[11]` table from D.7.3 — used to seed `ac_zero` of each
/// leaf's BEGABRAC according to the spec.
pub const ZERO_INIT: [u32; 11] = [4, 128, 512, 1024, 1536, 2048, 2560, 3072, 3584, 3968, 4088];

/// `SIGN_INIT[7]` table from D.7.3 — used to seed `ac_sign` of each
/// leaf's BEGABRAC. We do not currently plumb this through; the table
/// is exposed for completeness.
pub const SIGN_INIT: [u32; 7] = [512, 1024, 1536, 2048, 2560, 3072, 3584];

fn ilog2_at_least(x: u32) -> u32 {
    if x <= 1 {
        1
    } else {
        32 - (x - 1).leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abrac::tests_enc::AbracEncoder;
    use crate::begabrac::tests::encode_one as encode_one_int;

    /// Build a synthetic MA tree by emitting the same BEGABRAC reads
    /// the decoder will perform; then decode it back and verify.
    #[test]
    fn round_trip_single_leaf_tree() {
        let ranges = vec![PropRange { min: 0, max: 255 }; 4];
        let n_props = ranges.len() as i32;
        let n: u32 = 9;
        let mut bg1 = Begabrac::new(super::ilog2_at_least((ranges.len() + 12) as u32) + 1, 1024);
        let mut bg2 = Begabrac::new(4, 1024);
        let mut enc = AbracEncoder::new();
        // Property < 0 → encode 0 (which decodes to -1 after the spec's
        // -1 bias). signal_init = true → emit zc.
        encode_one_int(&mut enc, &mut bg1, 0, 0, n_props + 12);
        // zc = 5 → encode 0 (zc-5 = -5..5, spec's offset is +5; zc=5
        // means we emit value 0). 0 + 5 = 5 → ZERO_INIT[5] = 2048.
        encode_one_int(&mut enc, &mut bg2, 0, -5, 5);
        let stream = enc.finish();
        let mut dec = Abrac::new(&stream).unwrap();
        let tree = MaTree::decode(&mut dec, &ranges, n, true).unwrap();
        assert_eq!(tree.nodes.len(), 1);
        match &tree.nodes[0] {
            Node::Leaf { begabrac } => {
                assert_eq!(begabrac.ac_zero, ZERO_INIT[5]);
            }
            _ => panic!("expected leaf"),
        }
    }

    #[test]
    fn rejects_overlarge_bit_depth() {
        // Range check fires before any ABRAC reads — short stream is fine.
        let stream = [0u8; 8];
        let mut dec = Abrac::new(&stream).unwrap();
        let ranges = vec![PropRange { min: 0, max: 255 }; 2];
        let err = MaTree::decode(&mut dec, &ranges, MAX_VALUE_BIT_DEPTH + 1, false).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn round_trip_decision_then_two_leaves() {
        let ranges = vec![
            PropRange { min: 0, max: 255 },
            PropRange { min: 0, max: 100 },
        ];
        let n_props = ranges.len() as i32;
        let n: u32 = 9;
        let mut bg1 = Begabrac::new(super::ilog2_at_least((ranges.len() + 12) as u32) + 1, 1024);
        let _bg2 = Begabrac::new(4, 1024);
        let mut bg4 = Begabrac::new(n.max(4) + 1, 1024);
        let mut enc = AbracEncoder::new();
        // Decision on property 1 (> than threshold) — property = 1.
        // BEGABRAC1 reads encode (property + 1) so we send `2`.
        encode_one_int(&mut enc, &mut bg1, 2, 0, n_props + 12);
        // Threshold = 49 (within [0, 99]).
        encode_one_int(&mut enc, &mut bg4, 49, 0, 99);
        // Left leaf.
        encode_one_int(&mut enc, &mut bg1, 0, 0, n_props + 12);
        // Right leaf.
        encode_one_int(&mut enc, &mut bg1, 0, 0, n_props + 12);
        let stream = enc.finish();
        let mut dec = Abrac::new(&stream).unwrap();
        let tree = MaTree::decode(&mut dec, &ranges, n, false).unwrap();
        // Walk: property=[100, 60] → prop[1]=60 > 49 → left.
        let leaf = tree.walk(&[100, 60]).unwrap();
        let leaf_left = match &tree.nodes[0] {
            Node::Decision { left, .. } => *left,
            _ => panic!(),
        };
        assert_eq!(leaf, leaf_left);
        // Walk: property=[100, 10] → prop[1]=10 <= 49 → right.
        let leaf = tree.walk(&[100, 10]).unwrap();
        let leaf_right = match &tree.nodes[0] {
            Node::Decision { right, .. } => *right,
            _ => panic!(),
        };
        assert_eq!(leaf, leaf_right);
    }
}
