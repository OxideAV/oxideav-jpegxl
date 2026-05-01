//! Modular image sub-bitstream — FDIS 18181-1 §C.9 (FDIS-2021 path).
//!
//! This module is the **FDIS-2021** Modular decoder, written from scratch
//! against the published spec rather than the 2019 committee draft. The
//! committee-draft pipeline (BEGABRAC + matree + modular) lives in the
//! sibling `modular` / `matree` / `begabrac` modules and is **kept in
//! place** for round 3 — round 4 will gate or remove it.
//!
//! Key differences from the committee-draft path:
//!
//! * Entropy coder is **ANS** (FDIS Annex D.3), not BEGABRAC.
//! * The MA tree (D.4.1) is decoded via a 6-cluster ANS sub-stream
//!   (D.4.2), not via BEGABRAC.
//! * Per-channel symbols are decoded as `UnpackSigned(integer)` × leaf
//!   multiplier + leaf offset, then added to a Listing-C.16 prediction.
//!
//! ## Round 3 scope
//!
//! Implements the minimum needed to decode the first cjxl-produced
//! `--lossless` Modular `.jxl` fixture: single Grey channel, no
//! transforms, no Squeeze, no Palette, no RCT. Specifically:
//!
//! * `WPHeader` decoded but Annex E predictor is rejected (predictor
//!   == 6) since the round-3 fixture uses simpler predictors.
//! * `nb_transforms == 0` is the only accepted shape; non-zero → returns
//!   `Error::Unsupported`.
//! * Channel decoding loop implements Listing C.17 + Listing C.16
//!   (predictors 0..=5 and 7..=13). Predictor 6 (Annex E weighted
//!   predictor) is rejected.
//! * MA tree must contain a single leaf (no decision nodes) — the
//!   default MA tree that cjxl emits for trivial images. Multi-node
//!   trees parse cleanly but property evaluation is left to round 4.
//!
//! Allocation bound: every `Vec::with_capacity` is sized against either
//! a per-channel `width * height` pre-validated count or the bit
//! reader's remaining input length. Channels are capped at the
//! decoder-supplied `(width, height, num_channels)` from C.4.8 — none
//! of which are read from the bitstream in this module.

use oxideav_core::{Error, Result};

use crate::ans::alias::AliasTable;
use crate::ans::cluster::{num_clusters, read_clustering};
use crate::ans::distribution::read_distribution;
use crate::ans::hybrid::{HybridUintState, Lz77Params};
use crate::ans::hybrid_config::HybridUintConfig;
use crate::ans::prefix::{read_prefix_code, PrefixCode};
use crate::ans::symbol::AnsDecoder;
use crate::bitreader::{unpack_signed, BitReader, U32Dist};

/// Maximum channels per modular sub-bitstream we accept. The FDIS does
/// not impose an explicit limit, but real frames carry at most a
/// handful of channels (3 colour + a few extras). A bound of 64 is
/// generous; anything larger is almost certainly malicious.
pub const MAX_CHANNELS: usize = 64;

/// Maximum width or height per channel. The FDIS implicitly caps this
/// via the size_header (1<<30); we cap further at 65536 because the
/// decoder allocates a `Vec<i32>` of `width * height` per channel.
pub const MAX_DIM: u32 = 65536;

/// `WPHeader` per FDIS Table C.23 — Weighted Predictor parameters used
/// only when a leaf node selects predictor 6. We *parse* the header for
/// every Modular sub-bitstream (even when no leaf uses predictor 6) per
/// the spec, then ignore the values if they're never consulted.
#[derive(Debug, Clone, Copy)]
pub struct WpHeader {
    pub default_wp: bool,
    pub p1: u32,
    pub p2: u32,
    pub p3a: u32,
    pub p3b: u32,
    pub p3c: u32,
    pub p3d: u32,
    pub p3e: u32,
    pub w0: u32,
    pub w1: u32,
    pub w2: u32,
    pub w3: u32,
}

impl Default for WpHeader {
    fn default() -> Self {
        Self {
            default_wp: true,
            p1: 16,
            p2: 10,
            p3a: 7,
            p3b: 7,
            p3c: 7,
            p3d: 0,
            p3e: 0,
            w0: 13,
            w1: 12,
            w2: 12,
            w3: 12,
        }
    }
}

impl WpHeader {
    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let default_wp = br.read_bool()?;
        if default_wp {
            return Ok(Self::default());
        }
        Ok(Self {
            default_wp: false,
            p1: br.read_bits(5)?,
            p2: br.read_bits(5)?,
            p3a: br.read_bits(5)?,
            p3b: br.read_bits(5)?,
            p3c: br.read_bits(5)?,
            p3d: br.read_bits(5)?,
            p3e: br.read_bits(5)?,
            w0: br.read_bits(4)?,
            w1: br.read_bits(4)?,
            w2: br.read_bits(4)?,
            w3: br.read_bits(4)?,
        })
    }
}

/// Single MA-tree leaf node — what `MA(properties)` resolves to per
/// D.4.1. For round 3 we only support trees with a single leaf node;
/// `properties` are not evaluated.
#[derive(Debug, Clone, Copy)]
pub struct MaLeaf {
    pub ctx: u32,
    pub predictor: u32,
    pub offset: i32,
    pub multiplier: u32,
}

/// MA-tree node — either a decision node or a leaf. We decode the
/// whole tree into a flat `Vec<MaNode>` per Listing D.9.
#[derive(Debug, Clone, Copy)]
pub enum MaNode {
    Decision {
        property: u32,
        value: i32,
        left_child: u32,
        right_child: u32,
    },
    Leaf(MaLeaf),
}

/// Per-cluster entropy state — either an ANS pair (distribution +
/// alias table) or a prefix code. Both modes share the per-cluster
/// `HybridUintConfig` carried alongside.
#[derive(Debug)]
pub enum ClusterEntropy {
    Ans { dist: Vec<u16>, alias: AliasTable },
    Prefix { code: PrefixCode },
}

/// One full entropy stream as defined in FDIS D.3 (one prelude — LZ77,
/// clustering, use_prefix_code, per-cluster configs, per-cluster
/// distributions/codes — followed by the ANS state init OR no prelude
/// for prefix mode).
#[derive(Debug)]
pub struct EntropyStream {
    pub use_prefix_code: bool,
    pub log_alphabet_size: u32,
    /// Per-cluster `HybridUintConfig` (length `n_clusters`).
    pub configs: Vec<HybridUintConfig>,
    /// Per-cluster entropy state (length `n_clusters`).
    pub entropies: Vec<ClusterEntropy>,
    /// LZ77 settings for the symbol stream.
    pub lz77: Lz77Params,
    /// Hybrid uint config for LZ77 length symbols.
    pub lz_len_conf: HybridUintConfig,
    /// Cluster map: `cluster_map[ctx] = cluster index`.
    pub cluster_map: Vec<u32>,
    /// ANS state — only meaningful when `use_prefix_code == false`.
    pub ans_state: Option<AnsDecoder>,
}

impl EntropyStream {
    /// Read the FDIS D.3 prelude for `num_dist` distributions, then —
    /// for the ANS branch — read the `u(32)` state initialiser.
    pub fn read(br: &mut BitReader<'_>, num_dist: usize) -> Result<Self> {
        let lz77_enabled = br.read_bit()? == 1;
        let lz77 = if lz77_enabled {
            let min_symbol = br.read_u32([
                U32Dist::Val(224),
                U32Dist::Val(512),
                U32Dist::Val(4096),
                U32Dist::BitsOffset(15, 8),
            ])?;
            let min_length = br.read_u32([
                U32Dist::Val(3),
                U32Dist::Val(4),
                U32Dist::BitsOffset(2, 5),
                U32Dist::BitsOffset(8, 9),
            ])?;
            Lz77Params {
                enabled: true,
                min_symbol,
                min_length,
            }
        } else {
            Lz77Params::default()
        };

        let lz_len_conf = if lz77_enabled {
            HybridUintConfig::read(br, 8)?
        } else {
            HybridUintConfig {
                split_exponent: 8,
                msb_in_token: 0,
                lsb_in_token: 0,
                split: 256,
            }
        };

        // num_dist for the entropy stream: spec adds 1 if LZ77 is on.
        let effective_num_dist = if lz77_enabled { num_dist + 1 } else { num_dist };

        let cluster_map = if effective_num_dist > 1 {
            read_clustering(br, effective_num_dist)?
        } else {
            vec![0u32; effective_num_dist]
        };
        let n_clusters = if effective_num_dist > 1 {
            num_clusters(&cluster_map) as usize
        } else {
            1
        };
        if n_clusters == 0 || n_clusters > effective_num_dist {
            return Err(Error::InvalidData(format!(
                "JXL EntropyStream: invalid cluster count {n_clusters} for num_dist {effective_num_dist}"
            )));
        }

        let use_prefix_code = br.read_bit()? == 1;
        let log_alphabet_size = if use_prefix_code {
            5 + br.read_bits(2)?
        } else {
            15
        };
        let _alphabet_size_max = 1u32 << log_alphabet_size;

        // Per-cluster HybridUintConfig.
        let mut configs: Vec<HybridUintConfig> = Vec::with_capacity(n_clusters);
        for _ in 0..n_clusters {
            configs.push(HybridUintConfig::read(br, log_alphabet_size)?);
        }

        let mut entropies: Vec<ClusterEntropy> = Vec::with_capacity(n_clusters);
        if use_prefix_code {
            // For each cluster, read the symbol count then the prefix
            // histogram. Per D.3.1: "if u(1) is 0, count is 1, otherwise
            // n = u(4), count = 1 + (1<<n) + u(n). The symbol count is
            // at most 1<<15 for any distribution. The decoder then
            // proceeds to read the clustered distribution’s histograms
            // as specified in D.2.1."
            let mut counts = Vec::with_capacity(n_clusters);
            for _ in 0..n_clusters {
                let count = if br.read_bit()? == 0 {
                    1u32
                } else {
                    let n = br.read_bits(4)?;
                    if n > 14 {
                        return Err(Error::InvalidData(format!(
                            "JXL EntropyStream: prefix count n {n} > 14"
                        )));
                    }
                    1 + (1 << n) + br.read_bits(n)?
                };
                if count > (1 << 15) {
                    return Err(Error::InvalidData(format!(
                        "JXL EntropyStream: prefix symbol count {count} > 1<<15"
                    )));
                }
                // Note: spec does NOT cap count at alphabet_size_max
                // (alphabet_size_max governs ANS table sizing only). The
                // prefix-coded path may have a count larger than
                // 1<<log_alphabet_size — that's normal when log_alphabet_size
                // is small (== 5).
                counts.push(count);
            }
            for &count in &counts {
                let code = read_prefix_code(br, count)?;
                entropies.push(ClusterEntropy::Prefix { code });
            }
            Ok(Self {
                use_prefix_code: true,
                log_alphabet_size,
                configs,
                entropies,
                lz77,
                lz_len_conf,
                cluster_map,
                ans_state: None,
            })
        } else {
            // ANS path: per-cluster distribution + alias.
            for _ in 0..n_clusters {
                let dist = read_distribution(br, log_alphabet_size)?;
                let alias = AliasTable::build(&dist, log_alphabet_size)?;
                entropies.push(ClusterEntropy::Ans { dist, alias });
            }
            // ANS state init.
            let ans_state = AnsDecoder::new(br)?;
            Ok(Self {
                use_prefix_code: false,
                log_alphabet_size,
                configs,
                entropies,
                lz77,
                lz_len_conf,
                cluster_map,
                ans_state: Some(ans_state),
            })
        }
    }

    /// Decode a symbol from the underlying entropy stream against the
    /// cluster mapped by `ctx`. Used internally by `decode_uint`.
    pub fn decode_symbol(&mut self, br: &mut BitReader<'_>, ctx: u32) -> Result<u32> {
        let cluster = self
            .cluster_map
            .get(ctx as usize)
            .copied()
            .ok_or_else(|| Error::InvalidData("JXL EntropyStream: ctx out of range".into()))?
            as usize;
        if cluster >= self.entropies.len() {
            return Err(Error::InvalidData(
                "JXL EntropyStream: cluster index out of range".into(),
            ));
        }
        match &mut self.entropies[cluster] {
            ClusterEntropy::Ans { dist, alias } => {
                let ans = self.ans_state.as_mut().ok_or_else(|| {
                    Error::InvalidData("JXL EntropyStream: missing ANS state".into())
                })?;
                Ok(ans.decode_symbol(br, dist, alias)? as u32)
            }
            ClusterEntropy::Prefix { code } => code.decode(br),
        }
    }

    /// Hybrid uint config for the cluster mapped by `ctx`.
    pub fn config_for_ctx(&self, ctx: u32) -> HybridUintConfig {
        let cluster = self.cluster_map.get(ctx as usize).copied().unwrap_or(0) as usize;
        self.configs[cluster.min(self.configs.len().saturating_sub(1))]
    }
}

/// MA-tree as decoded by D.4.2 (Listing D.9), plus the entropy stream
/// used to decode per-channel symbols.
#[derive(Debug)]
pub struct MaTreeFdis {
    pub nodes: Vec<MaNode>,
    /// Number of distinct contexts (= number of leaves).
    pub num_ctx: usize,
    /// Symbol-stream entropy state.
    pub entropy: EntropyStream,
    /// Hybrid uint state for the *symbol* stream (LZ77 + windowing).
    pub hybrid: HybridUintState,
}

impl MaTreeFdis {
    /// Decode an MA tree per FDIS D.4.2 Listing D.9, then read the
    /// per-context clustered distributions for the symbol stream.
    ///
    /// Hard caps:
    /// * tree size is capped at 1024 nodes (spec gives 1 << 26 — way
    ///   above what any realistic image needs). 1024 covers every
    ///   real-world cjxl emission.
    /// * `mul_log` is bounded at 30, `mul_bits` at `(1 << (31 -
    ///   mul_log)) - 2`, both per the spec.
    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        // The tree sub-stream uses 6 distributions (T[0..=5]).
        let mut tree_stream = EntropyStream::read(br, 6)?;

        // Listing D.9 — decode the tree.
        let mut nodes: Vec<MaNode> = Vec::new();
        let mut nodes_left: u32 = 1;
        let mut ctx_id: u32 = 0;
        const MAX_NODES: usize = 1024;

        // Local hybrid state for the tree sub-stream (LZ77 + window).
        let mut tree_hybrid = HybridUintState::new(tree_stream.lz77, tree_stream.lz_len_conf);

        while nodes_left > 0 {
            if nodes.len() >= MAX_NODES {
                return Err(Error::InvalidData(format!(
                    "JXL MA tree: {} nodes exceeds round-3 cap {}",
                    nodes.len(),
                    MAX_NODES
                )));
            }
            let property_plus_1 = decode_uint_in(&mut tree_hybrid, &mut tree_stream, br, 1)?;
            let property = property_plus_1 as i64 - 1;
            if property < 0 {
                // Leaf.
                let predictor = decode_uint_in(&mut tree_hybrid, &mut tree_stream, br, 2)?;
                if predictor > 13 {
                    return Err(Error::InvalidData(format!(
                        "JXL MA tree: predictor {predictor} out of range [0, 13]"
                    )));
                }
                let uoffset = decode_uint_in(&mut tree_hybrid, &mut tree_stream, br, 3)?;
                let offset = unpack_signed(uoffset);
                let mul_log = decode_uint_in(&mut tree_hybrid, &mut tree_stream, br, 4)?;
                if mul_log > 30 {
                    return Err(Error::InvalidData(format!(
                        "JXL MA tree: mul_log {mul_log} > 30"
                    )));
                }
                let mul_bits = decode_uint_in(&mut tree_hybrid, &mut tree_stream, br, 5)?;
                let mul_bits_max = (1u64 << (31 - mul_log)) - 2;
                if mul_bits as u64 > mul_bits_max {
                    return Err(Error::InvalidData(format!(
                        "JXL MA tree: mul_bits {mul_bits} > {mul_bits_max}"
                    )));
                }
                let multiplier = (mul_bits + 1) << mul_log;
                nodes.push(MaNode::Leaf(MaLeaf {
                    ctx: ctx_id,
                    predictor,
                    offset,
                    multiplier,
                }));
                ctx_id += 1;
                nodes_left -= 1;
            } else {
                if property > 256 {
                    return Err(Error::InvalidData(format!(
                        "JXL MA tree: property {property} too large"
                    )));
                }
                let uvalue = decode_uint_in(&mut tree_hybrid, &mut tree_stream, br, 0)?;
                let value = unpack_signed(uvalue);
                let nodes_now = nodes.len() as u32;
                let left_child = nodes_now + nodes_left;
                let right_child = nodes_now + nodes_left + 1;
                nodes.push(MaNode::Decision {
                    property: property as u32,
                    value,
                    left_child,
                    right_child,
                });
                nodes_left += 2;
                nodes_left -= 1;
            }
        }

        let num_ctx = ctx_id as usize;
        let expected_ctx = nodes.len().div_ceil(2);
        if num_ctx != expected_ctx {
            return Err(Error::InvalidData(format!(
                "JXL MA tree: ctx_id {num_ctx} != (tree.size()+1)/2 = {expected_ctx}"
            )));
        }

        // Symbol stream — independent D.3 prelude.
        let entropy = EntropyStream::read(br, num_ctx)?;
        let hybrid = HybridUintState::new(entropy.lz77, entropy.lz_len_conf);

        Ok(Self {
            nodes,
            num_ctx,
            entropy,
            hybrid,
        })
    }
}

/// Decode one unsigned integer using the hybrid var-len uint stream
/// configured by `entropy` and the LZ77 / window state in `hybrid`. The
/// `ctx` selects which cluster to use.
///
/// This is the spec's `DecodeHybridVarLenUint(ctx)` from Listing D.6.
fn decode_uint_in(
    hybrid: &mut HybridUintState,
    entropy: &mut EntropyStream,
    br: &mut BitReader<'_>,
    ctx: u32,
) -> Result<u32> {
    // Split the borrows: `cluster_map` + `configs` are immutably
    // captured by the `configs` closure; `entropies` + `ans_state` +
    // `cluster_map` are mutably captured by `read_token`. Rust's
    // borrow checker won't allow `entropy.config_for_ctx(c)` and
    // `entropy.decode_symbol(...)` to coexist as closures, so we open
    // the struct here.
    let EntropyStream {
        cluster_map,
        configs,
        entropies,
        ans_state,
        ..
    } = entropy;
    let cluster_map_ref: &Vec<u32> = cluster_map;
    let configs_ref: &Vec<HybridUintConfig> = configs;
    let cfg_for = |c: u32| -> HybridUintConfig {
        let cl = cluster_map_ref.get(c as usize).copied().unwrap_or(0) as usize;
        configs_ref[cl.min(configs_ref.len().saturating_sub(1))]
    };
    let n_entropies = entropies.len();
    let read_token = |br_inner: &mut BitReader<'_>, c: u32| -> Result<u32> {
        let cluster = cluster_map_ref
            .get(c as usize)
            .copied()
            .ok_or_else(|| Error::InvalidData("JXL EntropyStream: ctx out of range".into()))?
            as usize;
        if cluster >= n_entropies {
            return Err(Error::InvalidData(
                "JXL EntropyStream: cluster index out of range".into(),
            ));
        }
        match &mut entropies[cluster] {
            ClusterEntropy::Ans { dist, alias } => {
                let ans = ans_state.as_mut().ok_or_else(|| {
                    Error::InvalidData("JXL EntropyStream: missing ANS state".into())
                })?;
                Ok(ans.decode_symbol(br_inner, dist, alias)? as u32)
            }
            ClusterEntropy::Prefix { code } => code.decode(br_inner),
        }
    };
    hybrid.decode(br, ctx, ctx, 0, read_token, cfg_for)
}

/// Description of a single channel to be decoded.
#[derive(Debug, Clone, Copy)]
pub struct ChannelDesc {
    pub width: u32,
    pub height: u32,
    pub hshift: i32,
    pub vshift: i32,
}

/// Decoded modular image — one `Vec<i32>` per channel.
#[derive(Debug, Clone)]
pub struct ModularImage {
    pub channels: Vec<Vec<i32>>,
    pub descs: Vec<ChannelDesc>,
}

impl ModularImage {
    /// Look up a sample at `(x, y)` in channel `i`, returning 0 for
    /// out-of-bounds reads (matching FDIS Listing C.16's predictor
    /// behaviour at image borders).
    pub fn get(&self, i: usize, x: i32, y: i32) -> i32 {
        if i >= self.channels.len() {
            return 0;
        }
        let desc = self.descs[i];
        if x < 0 || y < 0 || (x as u32) >= desc.width || (y as u32) >= desc.height {
            return 0;
        }
        self.channels[i][(y as u32 * desc.width + x as u32) as usize]
    }

    fn set(&mut self, i: usize, x: u32, y: u32, v: i32) {
        let desc = self.descs[i];
        self.channels[i][(y * desc.width + x) as usize] = v;
    }
}

/// Apply FDIS Listing C.16 — `prediction(x, y, predictor)` for sample
/// at `(x, y)` in channel `i`.
fn predict(img: &ModularImage, i: usize, x: i32, y: i32, predictor: u32) -> Result<i32> {
    let left = if x > 0 {
        img.get(i, x - 1, y)
    } else if y > 0 {
        img.get(i, x, y - 1)
    } else {
        0
    };
    let top = if y > 0 { img.get(i, x, y - 1) } else { left };
    let topleft = if x > 0 && y > 0 {
        img.get(i, x - 1, y - 1)
    } else {
        left
    };
    let desc = img.descs[i];
    let topright = if (x + 1) < desc.width as i32 && y > 0 {
        img.get(i, x + 1, y - 1)
    } else {
        top
    };
    let topright2 = if (x + 2) < desc.width as i32 && y > 0 {
        img.get(i, x + 2, y - 1)
    } else {
        topright
    };
    let leftleft = if x > 1 { img.get(i, x - 2, y) } else { left };
    let toptop = if y > 1 { img.get(i, x, y - 2) } else { top };
    let grad = top.wrapping_add(left).wrapping_sub(topleft);
    let v = match predictor {
        0 => 0,
        1 => left,
        2 => top,
        3 => left.wrapping_add(top).wrapping_div_euclid(2),
        4 => {
            if (grad - left).abs() < (grad - top).abs() {
                left
            } else {
                top
            }
        }
        5 => median3(grad, left, top),
        6 => {
            return Err(Error::Unsupported(
                "JXL Modular: weighted predictor (6) not yet supported (round 4)".into(),
            ));
        }
        7 => topright,
        8 => topleft,
        9 => leftleft,
        10 => left.wrapping_add(topleft).wrapping_div_euclid(2),
        11 => topleft.wrapping_add(top).wrapping_div_euclid(2),
        12 => top.wrapping_add(topright).wrapping_div_euclid(2),
        13 => {
            // (6*top - 2*toptop + 7*left + leftleft + topright2 + 3*topright + 8) Idiv 16
            let s = 6i64 * top as i64 - 2 * toptop as i64
                + 7 * left as i64
                + leftleft as i64
                + topright2 as i64
                + 3 * topright as i64
                + 8;
            s.div_euclid(16) as i32
        }
        _ => {
            return Err(Error::InvalidData(format!(
                "JXL Modular: predictor {predictor} out of range"
            )));
        }
    };
    Ok(v)
}

fn median3(a: i32, b: i32, c: i32) -> i32 {
    if (a <= b && b <= c) || (c <= b && b <= a) {
        b
    } else if (b <= a && a <= c) || (c <= a && a <= b) {
        a
    } else {
        c
    }
}

/// Round-3 channel decode loop — Listing C.17.
///
/// Constraints: the MA tree must be a single leaf (no decision nodes).
/// With a single leaf the per-pixel context is constant; per the spec
/// `D[leaf_node.ctx]` is a single distribution. We avoid the property
/// computation entirely.
pub fn decode_channels(
    br: &mut BitReader<'_>,
    descs: &[ChannelDesc],
    tree: &mut MaTreeFdis,
) -> Result<ModularImage> {
    if descs.is_empty() {
        return Ok(ModularImage {
            channels: Vec::new(),
            descs: Vec::new(),
        });
    }
    if descs.len() > MAX_CHANNELS {
        return Err(Error::InvalidData(format!(
            "JXL Modular: {} channels exceeds cap {}",
            descs.len(),
            MAX_CHANNELS
        )));
    }
    if tree.nodes.len() != 1 {
        return Err(Error::Unsupported(format!(
            "JXL Modular: MA tree with {} nodes not supported (round 3 needs single-leaf tree)",
            tree.nodes.len()
        )));
    }
    let leaf = match tree.nodes[0] {
        MaNode::Leaf(l) => l,
        MaNode::Decision { .. } => {
            return Err(Error::InvalidData(
                "JXL Modular: single-node tree is decision, not leaf".into(),
            ));
        }
    };
    if (leaf.ctx as usize) >= tree.num_ctx {
        return Err(Error::InvalidData(
            "JXL Modular: leaf ctx out of bounds".into(),
        ));
    }

    // Pre-validate channel sizes.
    for (i, d) in descs.iter().enumerate() {
        if d.width == 0 || d.height == 0 {
            return Err(Error::InvalidData(format!(
                "JXL Modular: channel {i} has zero dim ({}x{})",
                d.width, d.height
            )));
        }
        if d.width > MAX_DIM || d.height > MAX_DIM {
            return Err(Error::InvalidData(format!(
                "JXL Modular: channel {i} dim {}x{} exceeds cap {MAX_DIM}",
                d.width, d.height
            )));
        }
        let pixels = (d.width as u64).saturating_mul(d.height as u64);
        // Each pixel's smallest possible decode is one entropy-coded
        // symbol = at minimum 1 bit. Reject if input could not even
        // supply one bit per pixel.
        if pixels > br.bits_remaining() as u64 {
            return Err(Error::InvalidData(format!(
                "JXL Modular: channel {i} pixel count {pixels} exceeds remaining input bits"
            )));
        }
    }

    let mut channels: Vec<Vec<i32>> = Vec::with_capacity(descs.len());
    for d in descs.iter() {
        let n = (d.width as usize).saturating_mul(d.height as usize);
        channels.push(vec![0i32; n]);
    }
    let mut img = ModularImage {
        channels,
        descs: descs.to_vec(),
    };

    for (i, desc) in descs.iter().enumerate() {
        for y in 0..desc.height {
            for x in 0..desc.width {
                let token = decode_uint_in(&mut tree.hybrid, &mut tree.entropy, br, leaf.ctx)?;
                let diff = unpack_signed(token);
                let p = predict(&img, i, x as i32, y as i32, leaf.predictor)?;
                let val = (diff as i64)
                    .saturating_mul(leaf.multiplier as i64)
                    .saturating_add(leaf.offset as i64)
                    .saturating_add(p as i64);
                if !(i32::MIN as i64..=i32::MAX as i64).contains(&val) {
                    return Err(Error::InvalidData(format!(
                        "JXL Modular: decoded sample value {val} out of i32 range"
                    )));
                }
                img.set(i, x, y, val as i32);
            }
        }
    }

    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wp_header_default_values() {
        let bytes = vec![0x01u8]; // bit 0 = 1 → default_wp
        let mut br = BitReader::new(&bytes);
        let wp = WpHeader::read(&mut br).unwrap();
        assert!(wp.default_wp);
        assert_eq!(wp.p1, 16);
        assert_eq!(wp.w0, 13);
    }

    #[test]
    fn predict_zero_is_zero() {
        let img = ModularImage {
            channels: vec![vec![10, 20, 30, 40]],
            descs: vec![ChannelDesc {
                width: 2,
                height: 2,
                hshift: 0,
                vshift: 0,
            }],
        };
        assert_eq!(predict(&img, 0, 0, 0, 0).unwrap(), 0);
        assert_eq!(predict(&img, 0, 1, 1, 0).unwrap(), 0);
    }

    #[test]
    fn predict_left_returns_left_neighbour() {
        let img = ModularImage {
            channels: vec![vec![5, 7, 11, 13]],
            descs: vec![ChannelDesc {
                width: 2,
                height: 2,
                hshift: 0,
                vshift: 0,
            }],
        };
        // (1, 0) → left = sample at (0, 0) = 5
        assert_eq!(predict(&img, 0, 1, 0, 1).unwrap(), 5);
    }

    #[test]
    fn predict_top_returns_top_neighbour() {
        let img = ModularImage {
            channels: vec![vec![5, 7, 11, 13]],
            descs: vec![ChannelDesc {
                width: 2,
                height: 2,
                hshift: 0,
                vshift: 0,
            }],
        };
        // (0, 1) → top = sample at (0, 0) = 5
        assert_eq!(predict(&img, 0, 0, 1, 2).unwrap(), 5);
    }

    #[test]
    fn predict_average_left_top() {
        let img = ModularImage {
            channels: vec![vec![10, 0, 30, 0]],
            descs: vec![ChannelDesc {
                width: 2,
                height: 2,
                hshift: 0,
                vshift: 0,
            }],
        };
        // sample at (1, 1): left = 30 (sample at (0,1)), top = 0 (sample at (1,0))
        // (30 + 0) / 2 = 15
        assert_eq!(predict(&img, 0, 1, 1, 3).unwrap(), 15);
    }

    #[test]
    fn predict_predictor_6_rejected() {
        let img = ModularImage {
            channels: vec![vec![0; 4]],
            descs: vec![ChannelDesc {
                width: 2,
                height: 2,
                hshift: 0,
                vshift: 0,
            }],
        };
        assert!(predict(&img, 0, 1, 1, 6).is_err());
    }

    #[test]
    fn median3_works() {
        assert_eq!(median3(1, 2, 3), 2);
        assert_eq!(median3(3, 1, 2), 2);
        assert_eq!(median3(2, 3, 1), 2);
        assert_eq!(median3(5, 5, 5), 5);
    }
}
