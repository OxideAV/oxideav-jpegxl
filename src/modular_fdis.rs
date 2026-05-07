//! Modular image sub-bitstream — ISO/IEC 18181-1:2024 Annex H.
//!
//! Implements the 2024-published Modular decoder (formerly FDIS-2021
//! §C.9). The committee-draft pipeline (BEGABRAC + matree + modular)
//! lives in the sibling `modular` / `matree` / `begabrac` modules and is
//! retained as reference scaffolding only — it is not on the live
//! decode path.
//!
//! ## 2024-spec correspondence
//!
//! * `H.2` Image decoding — `ModularHeader` bundle (Table H.1) read by
//!   [`crate::global_modular::GlobalModular::read`].
//! * `H.3` Channel decoding — predictors per Table H.3, neighbours per
//!   Table H.2; implemented in [`predict`] + [`decode_channels`].
//! * `H.4` Meta-adaptive context model — properties per Table H.4 +
//!   tree traversal in [`evaluate_tree`] / [`get_properties`].
//! * `H.4.2` MA tree decoding — implemented in [`MaTreeFdis::read`].
//! * `H.5` Self-correcting predictor (predictor 6) — DEFERRED (round 2).
//! * `H.6` Transformations (RCT / Palette / Squeeze) — `H.6.3` RCT
//!   parsed + applied (round 1); `H.6.4` Palette + `H.6.2` Squeeze
//!   parsed but inverse-application errors (round 2 work).
//!
//! ## Round 1 (2024-spec) scope
//!
//! * `WPHeader` decoded but predictor 6 rejected at decode time —
//!   simpler predictors cover round-1 fixtures.
//! * Multi-leaf MA trees evaluated end-to-end (decision-node
//!   `property[k] > value` traversal per H.4.1). 16 base properties
//!   from Table H.4 plus per-previous-channel properties.
//! * Multi-channel decode (Grey 1ch + RGB 3ch).
//! * RCT inverse applied (H.6.3).
//! * Palette / Squeeze parsed but error out at inverse — round 2.
//!
//! Allocation bound: every `Vec::with_capacity` is sized against either
//! a per-channel `width * height` pre-validated count or the bit
//! reader's remaining input length. Channels are capped at the
//! decoder-supplied `(width, height, num_channels)` from G.1.3 — none
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

/// 2024-spec Table H.6 — Modular transform identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformId {
    /// Reversible Colour Transform (H.6.3).
    Rct,
    /// (Delta-)Palette (H.6.4).
    Palette,
    /// Modified Haar transform / Squeeze (H.6.2).
    Squeeze,
}

impl TransformId {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(TransformId::Rct),
            1 => Ok(TransformId::Palette),
            2 => Ok(TransformId::Squeeze),
            _ => Err(Error::InvalidData(format!(
                "JXL TransformId: invalid value {v}"
            ))),
        }
    }
}

/// 2024-spec Table H.7 — `TransformInfo` bundle. Per-transform fields
/// are made `Option<…>` because each transform kind only populates a
/// subset (e.g. RCT uses `rct_type`; Palette uses `num_c`/`nb_colours`).
#[derive(Debug, Clone)]
pub struct TransformInfo {
    pub tr: TransformId,
    pub begin_c: Option<u32>,
    pub rct_type: Option<u32>,
    pub num_c: Option<u32>,
    pub nb_colours: Option<u32>,
    pub nb_deltas: Option<u32>,
    pub d_pred: Option<u32>,
    pub num_sq: Option<u32>,
    pub squeeze_params: Vec<SqueezeParam>,
}

impl TransformInfo {
    /// 2024-spec Table H.7. The first u(2) selects the transform kind
    /// (kRCT=0, kPalette=1, kSqueeze=2). Fields that follow depend on
    /// the kind.
    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let tr_raw = br.read_bits(2)?;
        let tr = TransformId::from_u32(tr_raw)?;

        let begin_c = if tr != TransformId::Squeeze {
            Some(br.read_u32([
                U32Dist::Bits(3),
                U32Dist::BitsOffset(6, 8),
                U32Dist::BitsOffset(10, 72),
                U32Dist::BitsOffset(13, 1096),
            ])?)
        } else {
            None
        };

        let rct_type = if tr == TransformId::Rct {
            Some(br.read_u32([
                U32Dist::Val(6),
                U32Dist::Bits(2),
                U32Dist::BitsOffset(4, 2),
                U32Dist::BitsOffset(6, 10),
            ])?)
        } else {
            None
        };

        let (num_c, nb_colours, nb_deltas, d_pred) = if tr == TransformId::Palette {
            let num_c = br.read_u32([
                U32Dist::Val(1),
                U32Dist::Val(3),
                U32Dist::Val(4),
                U32Dist::BitsOffset(13, 1),
            ])?;
            let nb_colours = br.read_u32([
                U32Dist::BitsOffset(8, 0),
                U32Dist::BitsOffset(10, 256),
                U32Dist::BitsOffset(12, 1280),
                U32Dist::BitsOffset(16, 5376),
            ])?;
            let nb_deltas = br.read_u32([
                U32Dist::Val(0),
                U32Dist::BitsOffset(8, 1),
                U32Dist::BitsOffset(10, 257),
                U32Dist::BitsOffset(16, 1281),
            ])?;
            let d_pred = br.read_bits(4)?;
            (Some(num_c), Some(nb_colours), Some(nb_deltas), Some(d_pred))
        } else {
            (None, None, None, None)
        };

        let (num_sq, squeeze_params) = if tr == TransformId::Squeeze {
            let num_sq = br.read_u32([
                U32Dist::Val(0),
                U32Dist::BitsOffset(4, 1),
                U32Dist::BitsOffset(6, 9),
                U32Dist::BitsOffset(8, 41),
            ])?;
            // Bound: cap the number of squeeze steps to prevent absurd
            // allocations on malicious input.
            const MAX_SQUEEZE: u32 = 49 + 256;
            if num_sq > MAX_SQUEEZE {
                return Err(Error::InvalidData(format!(
                    "JXL Modular Squeeze: num_sq {num_sq} exceeds {MAX_SQUEEZE}"
                )));
            }
            let mut sps: Vec<SqueezeParam> = Vec::with_capacity(num_sq as usize);
            for _ in 0..num_sq {
                sps.push(SqueezeParam::read(br)?);
            }
            (Some(num_sq), sps)
        } else {
            (None, Vec::new())
        };

        Ok(Self {
            tr,
            begin_c,
            rct_type,
            num_c,
            nb_colours,
            nb_deltas,
            d_pred,
            num_sq,
            squeeze_params,
        })
    }
}

/// 2024-spec Table H.9 — `SqueezeParams` bundle. Only present when
/// `nb_transforms` includes a Squeeze with non-default parameters.
#[derive(Debug, Clone, Copy)]
pub struct SqueezeParam {
    pub horizontal: bool,
    pub in_place: bool,
    pub begin_c: u32,
    pub num_c: u32,
}

impl SqueezeParam {
    pub fn read(br: &mut BitReader<'_>) -> Result<Self> {
        let horizontal = br.read_bool()?;
        let in_place = br.read_bool()?;
        let begin_c = br.read_u32([
            U32Dist::Bits(3),
            U32Dist::BitsOffset(6, 8),
            U32Dist::BitsOffset(10, 72),
            U32Dist::BitsOffset(13, 1096),
        ])?;
        let num_c = br.read_u32([
            U32Dist::Val(1),
            U32Dist::Val(2),
            U32Dist::Val(3),
            U32Dist::BitsOffset(4, 4),
        ])?;
        Ok(Self {
            horizontal,
            in_place,
            begin_c,
            num_c,
        })
    }
}

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

        // 2024-spec C.2.1: if use_prefix_code is FALSE (ANS path),
        // log_alphabet_size = 5 + u(2). If TRUE (prefix path),
        // log_alphabet_size = 15. The FDIS 2021 text had this swapped
        // (a documented spec typo); the 2024 published edition is the
        // authoritative reading and matches the libjxl reference output
        // observed via cjxl/djxl black-box validation.
        let use_prefix_code = br.read_bit()? == 1;
        let log_alphabet_size = if use_prefix_code {
            15
        } else {
            5 + br.read_bits(2)?
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

/// Neighbour set for prediction + property derivation per
/// 2024-spec Table H.2.
///
/// Names match the spec exactly: `c` is the current sample (not yet
/// decoded), `w` / `n` / `nw` / `ne` / `nn` / `nee` / `ww` are previously
/// decoded neighbours with edge-fallback rules from H.3.
#[derive(Debug, Clone, Copy)]
pub struct Neighbours {
    pub w: i32,
    pub n: i32,
    pub nw: i32,
    pub ne: i32,
    pub nn: i32,
    pub nee: i32,
    pub ww: i32,
}

impl Neighbours {
    /// Compute the seven prediction neighbours for sample `(x, y)` in
    /// `channel[i]`, applying the 2024-spec H.3 edge-case fallbacks.
    pub fn at(img: &ModularImage, i: usize, x: i32, y: i32) -> Self {
        let width = img.descs[i].width as i32;
        let w = if x > 0 {
            img.get(i, x - 1, y)
        } else if y > 0 {
            img.get(i, x, y - 1)
        } else {
            0
        };
        let n = if y > 0 { img.get(i, x, y - 1) } else { w };
        let nw = if x > 0 && y > 0 {
            img.get(i, x - 1, y - 1)
        } else {
            w
        };
        let ne = if (x + 1) < width && y > 0 {
            img.get(i, x + 1, y - 1)
        } else {
            n
        };
        let nn = if y > 1 { img.get(i, x, y - 2) } else { n };
        let nee = if (x + 2) < width && y > 0 {
            img.get(i, x + 2, y - 1)
        } else {
            ne
        };
        let ww = if x > 1 { img.get(i, x - 2, y) } else { w };
        Self {
            w,
            n,
            nw,
            ne,
            nn,
            nee,
            ww,
        }
    }
}

/// Apply 2024-spec Table H.3 — `prediction(x, y, k)` for sample at
/// `(x, y)` in channel `i`. Predictor 6 (Self-correcting) is rejected
/// in round 1.
fn predict(img: &ModularImage, i: usize, x: i32, y: i32, predictor: u32) -> Result<i32> {
    let nb = Neighbours::at(img, i, x, y);
    let v = match predictor {
        0 => 0, // Zero
        1 => nb.w,
        2 => nb.n,
        3 => nb.w.wrapping_add(nb.n).wrapping_div_euclid(2), // Avg(W, N)
        4 => {
            // Select: |N - NW| < |W - NW| ? W : N
            // Spec text: abs(N - NW) < abs(W - NW)
            let lhs = (nb.n as i64 - nb.nw as i64).abs();
            let rhs = (nb.w as i64 - nb.nw as i64).abs();
            if lhs < rhs {
                nb.w
            } else {
                nb.n
            }
        }
        5 => {
            // Gradient: clamp(W + N - NW, min(W, N), max(W, N))
            let g = nb.w.wrapping_add(nb.n).wrapping_sub(nb.nw);
            let lo = nb.w.min(nb.n);
            let hi = nb.w.max(nb.n);
            g.clamp(lo, hi)
        }
        6 => {
            // Self-correcting predictor (Annex H.5). Full implementation
            // requires the per-pixel `true_err` and `err[i]` history
            // arrays — round 2 work. For round 1 we satisfy the *trivial*
            // case where the predictor is being asked at a position with
            // no decoded history (no W, N, NW, NE, NN, NEE, WW): all
            // edge-case fallbacks resolve to 0, every sub-predictor of
            // H.5.2 is 0, the weighted sum is 0, and `(prediction + 3)
            // >> 3 == 0`. This is exactly the situation at the (0, 0)
            // pixel of any image; without full WP we cannot extend
            // beyond it. Larger images that genuinely use predictor 6
            // at non-(0, 0) positions error out cleanly.
            if x == 0 && y == 0 {
                0
            } else {
                return Err(Error::Unsupported(
                    "JXL Modular: self-correcting predictor (6) at non-origin position requires full WP (round 2)".into(),
                ));
            }
        }
        7 => nb.ne,                                            // NorthEast
        8 => nb.nw,                                            // NorthWest
        9 => nb.ww,                                            // WestWest
        10 => nb.w.wrapping_add(nb.nw).wrapping_div_euclid(2), // Avg(W, NW)
        11 => nb.n.wrapping_add(nb.nw).wrapping_div_euclid(2), // Avg(N, NW)
        12 => nb.n.wrapping_add(nb.ne).wrapping_div_euclid(2), // Avg(N, NE)
        13 => {
            // AvgAll: (6*N - 2*NN + 7*W + WW + NEE + 3*NE + 8) Idiv 16
            let s = 6i64 * nb.n as i64 - 2 * nb.nn as i64
                + 7 * nb.w as i64
                + nb.ww as i64
                + nb.nee as i64
                + 3 * nb.ne as i64
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

/// Compute a property vector per 2024-spec Table H.4 + H.4.1.
///
/// Properties 0..=15 are the base set (channel/stream indices, position,
/// neighbour-magnitude features, and gradient residuals). Properties
/// 16+ are added for each previously-decoded channel that shares the
/// current channel's exact (width, height, hshift, vshift), in order
/// of decreasing index.
///
/// `stream_index` is 0 for the GlobalModular sub-bitstream (the only
/// case round 1 covers).
pub fn get_properties(img: &ModularImage, i: usize, x: i32, y: i32, stream_index: i32) -> Vec<i32> {
    let nb = Neighbours::at(img, i, x, y);
    // Property 8: x > 0 ? (W at (x-1) - gradient_at_(x-1)) : N.
    // gradient_at_(x-1) = (the value of property 9 at (x-1, y))
    //                   = W' + N' - NW' for sample (x-1, y).
    // For (x-1, y): W' = (x>1 ? img(x-2, y) : if y>0 img(x-1, y-1) else 0)
    //               N' = if y>0 img(x-1, y-1) else W'
    //               NW' = if x>1 && y>0 img(x-2, y-1) else W'
    let prop8 = if x > 0 {
        let wm1 = nb.w; // value at (x-1, y), already decoded (was W of (x,y))
                        // Re-derive W'/N'/NW' for the previous sample.
        let w_prev = if x > 1 {
            img.get(i, x - 2, y)
        } else if y > 0 {
            img.get(i, x - 1, y - 1)
        } else {
            0
        };
        let n_prev = if y > 0 {
            img.get(i, x - 1, y - 1)
        } else {
            w_prev
        };
        let nw_prev = if x > 1 && y > 0 {
            img.get(i, x - 2, y - 1)
        } else {
            w_prev
        };
        let grad_prev = w_prev.wrapping_add(n_prev).wrapping_sub(nw_prev);
        wm1.wrapping_sub(grad_prev)
    } else {
        nb.n
    };

    let mut props = vec![
        i as i32,                                    // 0: i (channel index)
        stream_index,                                // 1: stream index
        y,                                           // 2: y
        x,                                           // 3: x
        nb.n.unsigned_abs() as i32, // 4: abs(N) — keep as i32, may overflow on i32::MIN
        nb.w.unsigned_abs() as i32, // 5: abs(W)
        nb.n,                       // 6: N
        nb.w,                       // 7: W
        prop8,                      // 8
        nb.w.wrapping_add(nb.n).wrapping_sub(nb.nw), // 9: W + N - NW (gradient)
        nb.w.wrapping_sub(nb.nw),   // 10: W - NW
        nb.nw.wrapping_sub(nb.n),   // 11: NW - N
        nb.n.wrapping_sub(nb.ne),   // 12: N - NE
        nb.n.wrapping_sub(nb.nn),   // 13: N - NN
        nb.w.wrapping_sub(nb.ww),   // 14: W - WW
        0,                          // 15: max_error (predictor 6) — round 2
    ];

    // Properties 16+: scan previous channels with matching dims/shifts.
    // For each j from i-1 down to 0 with matching dims:
    //   property[k++] = abs(rC)         (16)
    //   property[k++] = rC              (17)
    //   property[k++] = abs(rC - rG)    (18)
    //   property[k++] = rC - rG         (19)
    // where rC = channel[j](x, y), rW/rN/rNW are j's neighbours,
    // rG = clamp(rW + rN - rNW, min(rW, rN), max(rW, rN)).
    let cur = img.descs[i];
    for j in (0..i).rev() {
        let d = img.descs[j];
        if d.width != cur.width
            || d.height != cur.height
            || d.hshift != cur.hshift
            || d.vshift != cur.vshift
        {
            continue;
        }
        let r_c = img.get(j, x, y);
        let r_w = if x > 0 {
            img.get(j, x - 1, y)
        } else if y > 0 {
            img.get(j, x, y - 1)
        } else {
            0
        };
        let r_n = if y > 0 { img.get(j, x, y - 1) } else { r_w };
        let r_nw = if x > 0 && y > 0 {
            img.get(j, x - 1, y - 1)
        } else {
            r_w
        };
        let g = r_w.wrapping_add(r_n).wrapping_sub(r_nw);
        let r_g = g.clamp(r_w.min(r_n), r_w.max(r_n));
        props.push(r_c.unsigned_abs() as i32);
        props.push(r_c);
        props.push(r_c.wrapping_sub(r_g).unsigned_abs() as i32);
        props.push(r_c.wrapping_sub(r_g));
    }

    props
}

/// Walk the MA tree per H.4.1: from `tree[0]`, for each decision node
/// `d` test `property[d.property] > d.value`. True → `d.left_child`.
/// False → `d.right_child`. Repeat until a leaf is reached.
pub fn evaluate_tree<'a>(nodes: &'a [MaNode], properties: &[i32]) -> Result<&'a MaLeaf> {
    if nodes.is_empty() {
        return Err(Error::InvalidData("JXL MA tree: empty tree".into()));
    }
    let mut cursor: usize = 0;
    // Bound: at most nodes.len() decision-node steps (the tree is acyclic
    // by construction since left/right_child are forward-only).
    for _ in 0..=nodes.len() {
        match &nodes[cursor] {
            MaNode::Leaf(l) => return Ok(l),
            MaNode::Decision {
                property,
                value,
                left_child,
                right_child,
            } => {
                let p_idx = *property as usize;
                let pv = properties.get(p_idx).copied().unwrap_or(0);
                let next = if pv > *value {
                    *left_child as usize
                } else {
                    *right_child as usize
                };
                if next >= nodes.len() {
                    return Err(Error::InvalidData(format!(
                        "JXL MA tree: child index {next} out of range {}",
                        nodes.len()
                    )));
                }
                cursor = next;
            }
        }
    }
    Err(Error::InvalidData(
        "JXL MA tree: traversal exceeded node count (cycle or malformed tree)".into(),
    ))
}

/// 2024-spec Annex H.3 — channel decode loop.
///
/// Decodes every channel in `descs` using the supplied MA tree. For
/// each sample:
/// 1. Compute properties (Table H.4).
/// 2. Walk the MA tree to find a leaf (`MA(properties)`).
/// 3. Decode an integer via the leaf's context ANS / prefix entropy.
/// 4. `diff = UnpackSigned(integer) * leaf.multiplier + leaf.offset`.
/// 5. `channel[i](x, y) = diff + prediction(x, y, leaf.predictor)`.
///
/// `dist_multiplier` is set per H.3 to "the largest channel width
/// amongst all channels that are to be decoded".
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
    if tree.nodes.is_empty() {
        return Err(Error::InvalidData("JXL Modular: empty MA tree".into()));
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

    // dist_multiplier per H.3 — largest channel width amongst channels
    // to be decoded. Used by the LZ77 special-distance branch.
    let dist_multiplier = descs.iter().map(|d| d.width).max().unwrap_or(0);

    let stream_index = 0i32; // GlobalModular only in round 1.

    for (i, desc) in descs.iter().enumerate() {
        for y in 0..desc.height {
            for x in 0..desc.width {
                let props = get_properties(&img, i, x as i32, y as i32, stream_index);
                let leaf = *evaluate_tree(&tree.nodes, &props)?;
                if (leaf.ctx as usize) >= tree.num_ctx {
                    return Err(Error::InvalidData(format!(
                        "JXL Modular: leaf ctx {} out of bounds {}",
                        leaf.ctx, tree.num_ctx
                    )));
                }
                let token = decode_uint_in_with_dist(
                    &mut tree.hybrid,
                    &mut tree.entropy,
                    br,
                    leaf.ctx,
                    dist_multiplier,
                )?;
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

/// Variant of `decode_uint_in` that propagates a non-zero
/// `dist_multiplier` to the LZ77 special-distance branch (H.3 prescribes
/// this for the channel-decode hybrid uint stream).
fn decode_uint_in_with_dist(
    hybrid: &mut HybridUintState,
    entropy: &mut EntropyStream,
    br: &mut BitReader<'_>,
    ctx: u32,
    dist_multiplier: u32,
) -> Result<u32> {
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
    hybrid.decode(br, ctx, ctx, dist_multiplier, read_token, cfg_for)
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
    fn evaluate_tree_single_leaf() {
        let leaf = MaLeaf {
            ctx: 0,
            predictor: 1,
            offset: 0,
            multiplier: 1,
        };
        let nodes = vec![MaNode::Leaf(leaf)];
        let result = evaluate_tree(&nodes, &[]).unwrap();
        assert_eq!(result.ctx, 0);
        assert_eq!(result.predictor, 1);
    }

    #[test]
    fn evaluate_tree_decision_walks_correctly() {
        // Tree: root (property=3 (x), value=10)
        //   true branch (x > 10) → leaf ctx=1
        //   false branch       → leaf ctx=2
        let nodes = vec![
            MaNode::Decision {
                property: 3,
                value: 10,
                left_child: 1,
                right_child: 2,
            },
            MaNode::Leaf(MaLeaf {
                ctx: 1,
                predictor: 0,
                offset: 0,
                multiplier: 1,
            }),
            MaNode::Leaf(MaLeaf {
                ctx: 2,
                predictor: 0,
                offset: 0,
                multiplier: 1,
            }),
        ];
        // properties[3] = 5 → 5 > 10 false → right (ctx=2)
        let mut props = vec![0i32; 16];
        props[3] = 5;
        let leaf = evaluate_tree(&nodes, &props).unwrap();
        assert_eq!(leaf.ctx, 2);
        // properties[3] = 100 → 100 > 10 true → left (ctx=1)
        props[3] = 100;
        let leaf = evaluate_tree(&nodes, &props).unwrap();
        assert_eq!(leaf.ctx, 1);
    }

    #[test]
    fn evaluate_tree_rejects_out_of_range_child() {
        let nodes = vec![MaNode::Decision {
            property: 0,
            value: 0,
            left_child: 99, // out of range
            right_child: 1,
        }];
        // properties[0] = 1 → property > value (0) → take left → out of range
        let props = vec![1i32];
        assert!(evaluate_tree(&nodes, &props).is_err());
    }

    #[test]
    fn get_properties_first_pixel_grey_image() {
        // 2x2 grey channel, all zero → first pixel (0, 0): all neighbours
        // collapse to 0; props[2] = y = 0; props[3] = x = 0.
        let img = ModularImage {
            channels: vec![vec![0i32; 4]],
            descs: vec![ChannelDesc {
                width: 2,
                height: 2,
                hshift: 0,
                vshift: 0,
            }],
        };
        let p = get_properties(&img, 0, 0, 0, 0);
        assert_eq!(p[0], 0); // channel index
        assert_eq!(p[1], 0); // stream index
        assert_eq!(p[2], 0); // y
        assert_eq!(p[3], 0); // x
        assert!(p.len() >= 16);
    }
}
