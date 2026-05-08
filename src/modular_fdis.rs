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

/// 2024-spec Annex H.6.4 (FDIS L.5) — `kDeltaPalette` table for
/// implicit delta-palette entries (used when `index < 0`). 72 RGB
/// triples covering common signed-delta neighbourhoods.
#[rustfmt::skip]
pub const K_DELTA_PALETTE: [[i32; 3]; 72] = [
    [0, 0, 0], [4, 4, 4], [11, 0, 0], [0, 0, -13], [0, -12, 0], [-10, -10, -10],
    [-18, -18, -18], [-27, -27, -27], [-18, -18, 0], [0, 0, -32], [-32, 0, 0],
    [-37, -37, -37], [0, -32, -32], [24, 24, 45], [50, 50, 50], [-45, -24, -24],
    [-24, -45, -45], [0, -24, -24], [-34, -34, 0], [-24, 0, -24], [-45, -45, -24],
    [64, 64, 64], [-32, 0, -32], [0, -32, 0], [-32, 0, 32], [-24, -45, -24],
    [45, 24, 45], [24, -24, -45], [-45, -24, 24], [80, 80, 80], [64, 0, 0],
    [0, 0, -64], [0, -64, -64], [-24, -24, 45], [96, 96, 96], [64, 64, 0],
    [45, -24, -24], [34, -34, 0], [112, 112, 112], [24, -45, -45], [45, 45, -24],
    [0, -32, 32], [24, -24, 45], [0, 96, 96], [45, -24, 24], [24, -45, -24],
    [-24, -45, 24], [0, -64, 0], [96, 0, 0], [128, 128, 128], [64, 0, 64],
    [144, 144, 144], [96, 96, 0], [-36, -36, 36], [45, -24, -45], [45, -45, -24],
    [0, 0, -96], [0, 128, 128], [0, 96, 0], [45, 24, -45], [-128, 0, 0],
    [24, -45, 24], [-45, 24, -45], [64, 0, -64], [64, -64, -64], [96, 0, 96],
    [45, -45, 24], [24, 45, -45], [64, 64, -64], [128, 128, 0], [0, 0, -128],
    [-24, 45, -45],
];

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
#[derive(Debug, Clone)]
pub enum ClusterEntropy {
    Ans { dist: Vec<u16>, alias: AliasTable },
    Prefix { code: PrefixCode },
}

/// One full entropy stream as defined in FDIS D.3 (one prelude — LZ77,
/// clustering, use_prefix_code, per-cluster configs, per-cluster
/// distributions/codes — followed by the ANS state init OR no prelude
/// for prefix mode).
#[derive(Debug, Clone)]
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
    /// Read the FDIS D.3 prelude for `num_dist` distributions.
    ///
    /// **2024-spec correction (round 3)**: the `u(32)` ANS state
    /// initialiser specified by C.3.2 ("Upon initialization of a new
    /// ANS stream") is **NOT** read here. cjxl 0.12.0 traces show that
    /// the state initialiser is emitted between the entropy stream's
    /// prelude and the FIRST `DecodeHybridVarLenUint` call against the
    /// stream — typically AFTER the consumer's intervening bundle reads
    /// (ModularHeader.use_global_tree / WPHeader / nb_transforms /
    /// transforms in the GlobalModular path). Round 1+2 read the
    /// state init eagerly inside this routine, which off-aligned every
    /// downstream field by 32 bits and ultimately mis-decoded the
    /// inner Modular ModularHeader's `use_global_tree` flag.
    ///
    /// Callers must invoke [`Self::read_ans_state_init`] just before
    /// the first symbol decode against an ANS stream.
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
        // authoritative reading.
        //
        // Round 2 also moves the read of `use_prefix_code` BEFORE the
        // clustering map (D.3.5). Round 1 had clustering before, but
        // black-box validation against cjxl 0.12.0 small-fixture traces
        // shows clustering follows use_prefix_code+log_alphabet_size.
        // 2024-spec C.2.1: "If use_prefix_code is false, the decoder
        // sets log_alphabet_size to 5 + u(2); otherwise, it sets
        // log_alphabet_size to 15." This matches the small-fixture
        // tests against cjxl 0.11.1.
        // 2024-spec C.2.1: "If use_prefix_code is false, the decoder
        // sets log_alphabet_size to 5 + u(2); otherwise, it sets
        // log_alphabet_size to 15." This matches the small-fixture
        // tests against cjxl 0.11.1.
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
            // 2024-spec round-3 fix: ANS state init `u(32)` is read by
            // the consumer just before the first symbol decode, NOT
            // here. See `read_ans_state_init`.
            Ok(Self {
                use_prefix_code: false,
                log_alphabet_size,
                configs,
                entropies,
                lz77,
                lz_len_conf,
                cluster_map,
                ans_state: None,
            })
        }
    }

    /// Read the ANS state initialiser (`u(32)` per spec C.3.2) for an
    /// ANS-path entropy stream. This is a no-op if the stream uses a
    /// prefix code or has already been initialised. Must be called
    /// after the prelude (`Self::read`) and BEFORE the first symbol
    /// decode for that stream.
    pub fn read_ans_state_init(&mut self, br: &mut BitReader<'_>) -> Result<()> {
        if self.use_prefix_code {
            return Ok(());
        }
        if self.ans_state.is_some() {
            return Ok(());
        }
        self.ans_state = Some(AnsDecoder::new(br)?);
        Ok(())
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
#[derive(Debug, Clone)]
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
    /// Return a fresh copy with the entropy-stream state and hybrid
    /// state reset. Used when reusing a "global" tree across multiple
    /// per-section sub-bitstreams (Annex H.2 — clustered distributions
    /// shared, but the entropy-coded stream itself is fresh per
    /// section). Allocates new sliding-window buffers.
    pub fn cloned_with_fresh_state(&self) -> Self {
        let mut entropy = self.entropy.clone();
        // Reset the ANS state — re-read from the bitstream when the
        // first symbol decode is about to happen.
        entropy.ans_state = None;
        let hybrid = HybridUintState::new(entropy.lz77, entropy.lz_len_conf);
        Self {
            nodes: self.nodes.clone(),
            num_ctx: self.num_ctx,
            entropy,
            hybrid,
        }
    }
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
        // For an ANS-path tree-stream, the state initialiser is read
        // before the first MA-tree decode call. Tree-streams observed
        // in cjxl 0.12.0 fixtures uniformly use prefix codes, but the
        // spec permits ANS for the tree as well.
        tree_stream.read_ans_state_init(br)?;

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
                // FDIS Listing D.9: left = tree.size + nodes_left + 1
                // right = tree.size + nodes_left + 2. The listing's
                // `nodes_left` reflects the count AFTER decrementing
                // for the current iteration; equivalent to our
                // (nodes_left - 1) at the time of the formula. We
                // expand the formula directly for clarity.
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

        // Symbol stream — independent D.3 prelude. The ANS state init
        // (if any) is deferred to just before the first symbol decode
        // — see comments on `EntropyStream::read`.
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

/// Per-channel state for the Self-correcting Weighted Predictor
/// (Annex H.5). Holds `true_err` and `sub_err[0..4]` for every
/// previously-decoded sample of one channel.
#[derive(Debug, Clone)]
pub struct WpState {
    pub width: u32,
    pub height: u32,
    /// `true_err[y * width + x]`. Stored in 8x scale (sample << 3).
    pub true_err: Vec<i32>,
    /// `sub_err[i][y * width + x]`. 1x scale.
    pub sub_err: [Vec<i32>; 4],
    /// `max_error` for the most recently predicted sample.
    pub last_max_error: i32,
}

impl WpState {
    pub fn new(width: u32, height: u32) -> Self {
        let n = (width as usize).saturating_mul(height as usize);
        Self {
            width,
            height,
            true_err: vec![0i32; n],
            sub_err: [vec![0i32; n], vec![0i32; n], vec![0i32; n], vec![0i32; n]],
            last_max_error: 0,
        }
    }

    fn at(&self, buf: &[i32], x: i32, y: i32) -> i32 {
        if x < 0 || y < 0 || (x as u32) >= self.width || (y as u32) >= self.height {
            return 0;
        }
        buf[(y as u32 * self.width + x as u32) as usize]
    }

    fn true_err_at(&self, x: i32, y: i32) -> i32 {
        self.at(&self.true_err, x, y)
    }

    fn sub_err_at(&self, i: usize, x: i32, y: i32) -> i32 {
        self.at(&self.sub_err[i], x, y)
    }

    fn set_true_err(&mut self, x: u32, y: u32, v: i32) {
        self.true_err[(y * self.width + x) as usize] = v;
    }

    fn set_sub_err(&mut self, i: usize, x: u32, y: u32, v: i32) {
        self.sub_err[i][(y * self.width + x) as usize] = v;
    }
}

/// Annex H.5.2 sub-predictor + final-prediction computation.
///
/// Returns `(prediction_in_8x_scale, [predictioni_in_8x_scale; 4],
/// max_error)`.  All inputs (N, W, NW, NE, NN, WW) are pre-shifted by 3.
/// The caller is responsible for the `(prediction + 3) >> 3` rounding
/// before using the value as the predictor 6 result.
fn wp_predict(
    state: &WpState,
    nb: &Neighbours,
    x: i32,
    y: i32,
    wp: &WpHeader,
) -> (i32, [i32; 4], i32) {
    // Inputs in 8x scale.
    let n8 = nb.n.wrapping_shl(3);
    let w8 = nb.w.wrapping_shl(3);
    let nw8 = nb.nw.wrapping_shl(3);
    let ne8 = nb.ne.wrapping_shl(3);
    let nn8 = nb.nn.wrapping_shl(3);
    let _ww8 = nb.ww.wrapping_shl(3);

    // true_err neighbours (already in 8x scale, stored that way).
    let te_w = state.true_err_at(x - 1, y);
    let te_n = state.true_err_at(x, y - 1);
    let te_nw = state.true_err_at(x - 1, y - 1);
    let te_ne_raw = if (x + 1) < state.width as i32 && y > 0 {
        state.true_err_at(x + 1, y - 1)
    } else {
        te_n
    };
    let te_ne = te_ne_raw;

    // Sub-predictions per Listing E.1, in 8x scale.
    let p0 = w8.wrapping_add(ne8).wrapping_sub(n8);
    let p1 =
        n8.wrapping_sub((((te_w as i64 + te_n as i64 + te_ne as i64) * wp.p1 as i64) >> 5) as i32);
    let p2 =
        w8.wrapping_sub((((te_w as i64 + te_n as i64 + te_nw as i64) * wp.p2 as i64) >> 5) as i32);
    // 2024-spec Listing H.5.2 subpred[3] — sign is `N3 - (...)`, not `N3 + (...)`.
    // Round 3 had `wrapping_add`; corrected to `wrapping_sub` here.
    let p3 = n8.wrapping_sub(
        ((te_nw as i64 * wp.p3a as i64
            + te_n as i64 * wp.p3b as i64
            + te_ne as i64 * wp.p3c as i64
            + (nn8 as i64 - n8 as i64) * wp.p3d as i64
            + (nw8 as i64 - w8 as i64) * wp.p3e as i64)
            >> 5) as i32,
    );
    let preds = [p0, p1, p2, p3];

    // Sum sub_err over the 5 neighbours: N, W, NW, WW, NE.
    let weights = {
        // err_sum_i = sum of sub_err[i] over (N, W, NW, WW, NE).
        let mut weights = [0u64; 4];
        let weights_cfg = [wp.w0, wp.w1, wp.w2, wp.w3];
        for k in 0..4 {
            // Edge cases per H.5.2:
            //  - if W, N or WW does not exist, the value 0 is used instead;
            //  - if NW or NE does not exist, the value of N is used instead.
            let n_se = if y > 0 {
                state.sub_err_at(k, x, y - 1)
            } else {
                0
            };
            let w_se = if x > 0 {
                state.sub_err_at(k, x - 1, y)
            } else {
                0
            };
            let nw_se = if x > 0 && y > 0 {
                state.sub_err_at(k, x - 1, y - 1)
            } else {
                n_se
            };
            let ww_se = if x > 1 {
                state.sub_err_at(k, x - 2, y)
            } else {
                0
            };
            let ne_se = if (x + 1) < state.width as i32 && y > 0 {
                state.sub_err_at(k, x + 1, y - 1)
            } else {
                n_se
            };
            // err_sum is `Umod (1 << 32)` per spec; we hold non-negative
            // u32 since each err[i] is `(abs(...) + 3) >> 3 >= 0`.
            let mut err_sum =
                (n_se as i64 + w_se as i64 + nw_se as i64 + ww_se as i64 + ne_se as i64) as u64
                    & 0xFFFF_FFFF;
            // Special case: if x == width - 1, err_sum[i] += err[i]_W.
            if x == (state.width as i32 - 1) {
                err_sum = (err_sum + w_se as u64) & 0xFFFF_FFFF;
            }
            // error2weight(err_sum, maxweight):
            //   shift = floor(log2(err_sum + 1)) - 5; if shift < 0 shift = 0;
            //   return 4 + ((maxweight * ((1 << 24) Idiv ((err_sum >> shift) + 1))) >> shift);
            let err_sum32 = err_sum as u32;
            let bits = 32u32 - (err_sum32 + 1).leading_zeros();
            let shift = (bits.saturating_sub(1)).saturating_sub(5);
            let denom = ((err_sum32 >> shift) as u64) + 1;
            let inner = (weights_cfg[k] as u64 * (1u64 << 24) / denom) >> shift;
            let weight = 4u64 + inner;
            weights[k] = weight;
        }
        weights
    };

    // Listing E.3 — final prediction.
    let sum_weights_pre = weights[0] + weights[1] + weights[2] + weights[3];
    // log_weight = floor(log2(sum_weights)) + 1
    let log_weight = if sum_weights_pre == 0 {
        1
    } else {
        // floor(log2(x))+1 == bit-position of MSB in 1-indexed terms.
        64u32 - (sum_weights_pre).leading_zeros()
    };
    // shift = log_weight - 5 (saturating).
    let mut shifted = [0u64; 4];
    let sh = log_weight.saturating_sub(5);
    for (k, w) in weights.iter().enumerate() {
        shifted[k] = w >> sh;
    }
    let sum_weights = shifted[0] + shifted[1] + shifted[2] + shifted[3];
    if sum_weights == 0 {
        // Degenerate; should never happen in well-formed bitstreams.
        // Fall back to the NW gradient.
        let g = w8.wrapping_add(n8).wrapping_sub(nw8);
        return (g, preds, 0);
    }
    // 2024-spec Listing H.5.2 — `s = (sum_weights >> 1) - 1`. Round 3
    // missed the `- 1`.
    let s_init: i64 = (sum_weights >> 1) as i64 - 1;
    let s = (0..4).fold(s_init, |acc, k| acc + preds[k] as i64 * shifted[k] as i64);
    let mut prediction = ((s * (1i64 << 24).div_euclid(sum_weights as i64)) >> 24) as i32;

    // If true_errN, true_errW, true_errNW have the same sign,
    // clamp to [min(W, N, NE), max(W, N, NE)] (in 8x scale).
    if (te_n ^ te_w) | (te_n ^ te_nw) <= 0 {
        let lo = w8.min(n8).min(ne8);
        let hi = w8.max(n8).max(ne8);
        prediction = prediction.clamp(lo, hi);
    }

    // Listing E.4 — max_error.
    let mut max_error = te_w;
    if te_n.unsigned_abs() > max_error.unsigned_abs() {
        max_error = te_n;
    }
    if te_nw.unsigned_abs() > max_error.unsigned_abs() {
        max_error = te_nw;
    }
    if te_ne.unsigned_abs() > max_error.unsigned_abs() {
        max_error = te_ne;
    }

    (prediction, preds, max_error)
}

/// Apply 2024-spec Table H.3 — `prediction(x, y, k)` for sample at
/// `(x, y)` in channel `i`. Predictor 6 (Self-correcting) requires the
/// caller to pass the per-channel `WpState` and `WpHeader`.
fn predict(
    img: &ModularImage,
    i: usize,
    x: i32,
    y: i32,
    predictor: u32,
    wp: &WpHeader,
    wp_state: Option<&WpState>,
) -> Result<(i32, [i32; 4], i32)> {
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
            // Self-correcting predictor (Annex H.5).
            let state = wp_state.ok_or_else(|| {
                Error::InvalidData("JXL Modular: predictor 6 used but WP state missing".into())
            })?;
            let (pred8, preds, max_err) = wp_predict(state, &nb, x, y, wp);
            // Round (prediction + 3) >> 3 → arithmetic shift, but the
            // spec uses unsigned-style >> for non-negative values; for
            // signed we use (pred8 + 3) >> 3 with arith shift.
            let v = (pred8.wrapping_add(3)) >> 3;
            return Ok((v, preds, max_err));
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
    Ok((v, [0; 4], 0))
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
/// case round 1 covers). `max_error` is property[15] from the
/// self-correcting predictor's last call (Listing H.5/E.4).
pub fn get_properties(
    img: &ModularImage,
    i: usize,
    x: i32,
    y: i32,
    stream_index: i32,
    max_error: i32,
) -> Vec<i32> {
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
        max_error,                  // 15: max_error (predictor 6, Annex H.5)
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
/// amongst all channels that are to be decoded, excluding meta-channels".
pub fn decode_channels(
    br: &mut BitReader<'_>,
    descs: &[ChannelDesc],
    tree: &mut MaTreeFdis,
    wp: &WpHeader,
) -> Result<ModularImage> {
    decode_channels_at_stream(br, descs, tree, wp, 0)
}

/// Identical to [`decode_channels`] but takes the `stream_index` to be
/// embedded in MA-tree property[1] (Table H.4). The 2024 spec defines
/// six stream-index families:
///
/// * `0` — GlobalModular sub-bitstream.
/// * `1 + lf_group_idx` — LfCoefficients (kVarDCT only).
/// * `1 + num_lf_groups + lf_group_idx` — ModularLfGroup.
/// * `1 + 2*num_lf_groups + lf_group_idx` — HFMetadata (kVarDCT only).
/// * `1 + 3*num_lf_groups + parameters_index` — RAW dequant tables (kVarDCT only).
/// * `1 + 3*num_lf_groups + 17 + num_groups * pass_idx + group_idx` — ModularGroup.
///
/// Properties 0..=15 use property[1] = `stream_index`. The MA tree may
/// branch on this property to pick a different leaf depending on which
/// section the sub-bitstream belongs to.
pub fn decode_channels_at_stream(
    br: &mut BitReader<'_>,
    descs: &[ChannelDesc],
    tree: &mut MaTreeFdis,
    wp: &WpHeader,
    stream_index: i32,
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
    }

    // 2024-spec C.3.2 round-3 fix: read the ANS state initialiser for
    // the symbol stream (if any) RIGHT HERE, just before the first
    // symbol decode. cjxl 0.12.0 traces show the `u(32)` is emitted
    // AFTER the inner ModularHeader (use_global_tree / WPHeader /
    // nb_transforms / transforms) and IMMEDIATELY before the first
    // pixel-decode call against the stream.
    tree.entropy.read_ans_state_init(br)?;

    let mut channels: Vec<Vec<i32>> = Vec::with_capacity(descs.len());
    for d in descs.iter() {
        let n = (d.width as usize).saturating_mul(d.height as usize);
        channels.push(vec![0i32; n]);
    }
    let mut img = ModularImage {
        channels,
        descs: descs.to_vec(),
    };

    // dist_multiplier per H.3 — largest channel width amongst all
    // channels that are to be decoded (incl. meta-channels). The
    // ROUND-2 reading "excluding meta-channels" was wrong — re-read
    // 2024 H.3 paragraph 3 ("...largest channel width amongst all
    // channels that are to be decoded.") which makes no exclusion.
    let dist_multiplier = descs.iter().map(|d| d.width).max().unwrap_or(0);

    // `stream_index` is supplied by the caller; it threads through
    // property[1] of the MA tree per Table H.4 and can change per
    // sub-bitstream (GlobalModular = 0; per-PassGroup ModularGroup =
    // 1 + 3*num_lf_groups + 17 + num_groups * pass_idx + group_idx).

    // Per-channel WP state. Allocate only when the MA tree has any
    // leaf with predictor 6 (or property[15] reads — the Self-correcting
    // max_error). Otherwise the state is irrelevant.
    let needs_wp_state = tree.nodes.iter().any(|n| match n {
        MaNode::Leaf(l) => l.predictor == 6,
        MaNode::Decision { property, .. } => *property == 15,
    });
    let mut wp_states: Vec<Option<WpState>> = if needs_wp_state {
        descs
            .iter()
            .map(|d| Some(WpState::new(d.width, d.height)))
            .collect()
    } else {
        descs.iter().map(|_| None).collect()
    };

    for (i, desc) in descs.iter().enumerate() {
        for y in 0..desc.height {
            for x in 0..desc.width {
                // 2024-spec H.5.1: WP is invoked "for each sample (including
                // samples that use a different predictor)". So compute the
                // weighted-predictor result first when WP state exists; the
                // resulting `max_error` is property[15] for the MA-tree
                // decision that picks this sample's leaf.
                let (wp_pred8, wp_subpreds, wp_max_error) =
                    if let Some(state) = wp_states[i].as_ref() {
                        let nb = Neighbours::at(&img, i, x as i32, y as i32);
                        wp_predict(state, &nb, x as i32, y as i32, wp)
                    } else {
                        (0, [0; 4], 0)
                    };

                let props = get_properties(&img, i, x as i32, y as i32, stream_index, wp_max_error);
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

                // Compute prediction value for `leaf.predictor`. Predictor 6
                // reuses the wp_predict result we already have (rounded).
                let p = if leaf.predictor == 6 {
                    if wp_states[i].is_none() {
                        return Err(Error::InvalidData(
                            "JXL Modular: leaf predictor 6 but no WP state".into(),
                        ));
                    }
                    (wp_pred8.wrapping_add(3)) >> 3
                } else {
                    let (v, _, _) = predict(
                        &img,
                        i,
                        x as i32,
                        y as i32,
                        leaf.predictor,
                        wp,
                        wp_states[i].as_ref(),
                    )?;
                    v
                };

                let val = (diff as i64)
                    .saturating_mul(leaf.multiplier as i64)
                    .saturating_add(leaf.offset as i64)
                    .saturating_add(p as i64);
                if !(i32::MIN as i64..=i32::MAX as i64).contains(&val) {
                    return Err(Error::InvalidData(format!(
                        "JXL Modular: decoded sample value {val} out of i32 range"
                    )));
                }
                let v = val as i32;
                img.set(i, x, y, v);

                // 2024-spec H.5.1: AFTER decoding the sample, compute and
                // store true_err + err[i] for the current sample so that
                // future samples (which see this position as a neighbour)
                // get the correct history. We use the ALREADY-COMPUTED
                // wp_pred8 / wp_subpreds from the BEFORE-decode call —
                // they were computed against neighbour state and are the
                // same prediction values the spec stores against.
                if let Some(state) = wp_states[i].as_mut() {
                    state.last_max_error = wp_max_error;
                    // true_err = NarrowToI32(prediction - (true_value << 3))
                    let te = wp_pred8.wrapping_sub(v.wrapping_shl(3));
                    state.set_true_err(x, y, te);
                    // err[i] = (abs(subpred[i] - (true_value << 3)) + 3) >> 3
                    let tv8 = v.wrapping_shl(3);
                    for (k, p_i) in wp_subpreds.iter().enumerate() {
                        let diff_i = p_i.wrapping_sub(tv8);
                        let se = (diff_i.unsigned_abs().wrapping_add(3)) >> 3;
                        state.set_sub_err(k, x, y, se as i32);
                    }
                }
            }
        }
    }

    Ok(img)
}

/// Public diagnostic re-export of `decode_uint_in_with_dist` for use by
/// integration-test bisects. Identical to the internal function used in
/// `decode_channels`; round-4 bisects need it to step through pixel
/// decode token-by-token while printing bit positions.
pub fn decode_uint_in_with_dist_pub(
    hybrid: &mut HybridUintState,
    entropy: &mut EntropyStream,
    br: &mut BitReader<'_>,
    ctx: u32,
    dist_multiplier: u32,
) -> Result<u32> {
    decode_uint_in_with_dist(hybrid, entropy, br, ctx, dist_multiplier)
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

// ----------------------------------------------------------------------
// Inverse transforms — Annex H.6
// ----------------------------------------------------------------------

/// 2024-spec Annex H.6.3 — Inverse RCT (Reversible Colour Transform).
///
/// Operates on three channels starting at `begin_c`. For every pixel
/// `(x, y)` in those channels, the values `(A, B, C)` are replaced by
/// `(V[0], V[1], V[2])`. `rct_type` ∈ [0, 41] selects one of 42
/// permutation × type combinations; `permutation = rct_type / 7` and
/// `type = rct_type % 7`.
pub fn inverse_rct(img: &mut ModularImage, begin_c: usize, rct_type: u32) -> Result<()> {
    if begin_c + 3 > img.channels.len() {
        return Err(Error::InvalidData(format!(
            "JXL Modular RCT: begin_c {} + 3 exceeds channel count {}",
            begin_c,
            img.channels.len()
        )));
    }
    let permutation = (rct_type / 7) as usize;
    let typ = rct_type % 7;
    if permutation > 5 {
        return Err(Error::InvalidData(format!(
            "JXL Modular RCT: invalid rct_type {rct_type} (permutation {permutation} > 5)"
        )));
    }

    let d0 = img.descs[begin_c];
    let d1 = img.descs[begin_c + 1];
    let d2 = img.descs[begin_c + 2];
    if d0.width != d1.width
        || d0.width != d2.width
        || d0.height != d1.height
        || d0.height != d2.height
    {
        return Err(Error::InvalidData(
            "JXL Modular RCT: three channels must share dimensions".into(),
        ));
    }
    let w = d0.width as usize;
    let h = d0.height as usize;
    let n = w * h;

    // Take the channels out for in-place mutation.
    // We need three independent &mut slices.
    let (a_buf, b_buf, c_buf) = {
        // Split-borrow the three channels.
        let chans = &mut img.channels;
        let (head, tail) = chans.split_at_mut(begin_c + 1);
        let a = &mut head[begin_c]; // index begin_c
        let (b, rest) = tail.split_first_mut().expect("begin_c+1 exists");
        let c = &mut rest[0]; // index begin_c+2
        (a, b, c)
    };
    if a_buf.len() < n || b_buf.len() < n || c_buf.len() < n {
        return Err(Error::InvalidData(
            "JXL Modular RCT: channel buffer length mismatch".into(),
        ));
    }

    for idx in 0..n {
        let a = a_buf[idx];
        let mut b = b_buf[idx];
        let mut c = c_buf[idx];
        let (d, e, f);
        if typ == 6 {
            // YCgCo
            let tmp = a.wrapping_sub(c >> 1);
            e = c.wrapping_add(tmp);
            f = tmp.wrapping_sub(b >> 1);
            d = f.wrapping_add(b);
        } else {
            if typ & 1 != 0 {
                c = c.wrapping_add(a);
            }
            if (typ >> 1) == 1 {
                b = b.wrapping_add(a);
            }
            if (typ >> 1) == 2 {
                b = b.wrapping_add((a.wrapping_add(c)) >> 1);
            }
            d = a;
            e = b;
            f = c;
        }
        // V[permutation Umod 3] = D
        // V[(permutation + 1 + permutation/3) Umod 3] = E
        // V[(permutation + 2 - permutation/3) Umod 3] = F
        let mut v = [0i32; 3];
        v[permutation % 3] = d;
        v[(permutation + 1 + permutation / 3) % 3] = e;
        v[(permutation + 2 - permutation / 3) % 3] = f;
        a_buf[idx] = v[0];
        b_buf[idx] = v[1];
        c_buf[idx] = v[2];
    }
    Ok(())
}

/// Helper for the Palette inverse transform: prediction(x, y, d_pred)
/// against the *current state* of the indices channel. Uses the same
/// `predict` code path as the main decode loop. Predictor 6 (WP) is
/// rejected here — palette delta-prediction with WP is not exercised
/// by the round-2 small fixtures.
fn palette_predict_for_inverse(
    img: &ModularImage,
    chan_idx: usize,
    x: i32,
    y: i32,
    d_pred: u32,
) -> Result<i32> {
    if d_pred == 6 {
        return Err(Error::Unsupported(
            "JXL Modular Palette: d_pred=6 (WP) for delta-palette not yet supported".into(),
        ));
    }
    let wp = WpHeader::default();
    let (v, _, _) = predict(img, chan_idx, x, y, d_pred, &wp, None)?;
    Ok(v)
}

/// 2024-spec Annex H.6.4 — Inverse Palette transform.
///
/// `begin_c` is the channel index (in the channel list **at time of
/// inverse**, i.e. AFTER all prior transforms have been applied) at
/// which the palette indices live. `num_c` original channels are to be
/// reconstructed; `nb_colours` is the explicit palette size. The meta-
/// channel containing the palette table is at index 0 and has
/// dimensions `nb_colours × num_c`.
pub fn inverse_palette(
    img: &mut ModularImage,
    begin_c: usize,
    num_c: u32,
    nb_colours: u32,
    nb_deltas: u32,
    d_pred: u32,
    bit_depth: u32,
) -> Result<()> {
    let num_c = num_c as usize;
    let _nb_colours_signed = nb_colours as i32;
    if num_c == 0 {
        return Err(Error::InvalidData("JXL Palette: num_c must be >= 1".into()));
    }
    if img.channels.is_empty() {
        return Err(Error::InvalidData(
            "JXL Palette: no meta-channel for palette".into(),
        ));
    }
    // After channel-list adjustment, indices live at index `begin_c + 1`
    // (the `+1` is because the meta-channel is at position 0). But the
    // caller passes `begin_c` referring to the BITSTREAM begin_c, so
    // the meta channel is at 0 and indices at `begin_c + 1`.
    let first = begin_c + 1;
    let last = first + num_c - 1;
    if last >= img.channels.len() + (num_c - 1) {
        // We will be inserting copies; check that first is in range.
        if first >= img.channels.len() {
            return Err(Error::InvalidData(format!(
                "JXL Palette: indices channel {first} out of range {}",
                img.channels.len()
            )));
        }
    }
    if img.channels.is_empty() {
        return Err(Error::InvalidData(
            "JXL Palette: missing meta-channel at index 0".into(),
        ));
    }

    let idx_desc = img.descs[first];
    let w = idx_desc.width;
    let h = idx_desc.height;
    let _ = idx_desc;

    // Insert num_c-1 copies of channel[first] at positions [first+1..=last].
    for i in (first + 1)..=last {
        let copy = img.channels[first].clone();
        let desc = img.descs[first];
        if i <= img.channels.len() {
            img.channels.insert(i, copy);
            img.descs.insert(i, desc);
        } else {
            img.channels.push(copy);
            img.descs.push(desc);
        }
    }

    // Reconstruct each channel independently. We need the meta-channel
    // table at index 0 to remain intact during reconstruction (read
    // only) and we mutate channels [first..=last].
    let bitdepth = bit_depth.max(1);
    let one_shifted = if bitdepth >= 32 {
        i32::MAX
    } else {
        ((1i64 << bitdepth) - 1) as i32
    };
    let saturate_div_4 = one_shifted / 4;
    let small_offset = if bitdepth > 3 {
        1i32 << (bitdepth - 3)
    } else {
        1
    };

    // Take a snapshot of the meta-channel for read-only use.
    let palette_table = img.channels[0].clone();
    let palette_w = img.descs[0].width as i32;

    #[allow(clippy::needless_range_loop)]
    // c indexes spec equations; iterator form would obscure them.
    for c in 0..num_c {
        let chan_index = first + c;
        // For non-delta entries we read indices directly from the
        // partially-reconstructed channel BEFORE this iteration's
        // sample is written; for delta entries we add a prediction
        // that uses already-decoded NEIGHBOURS (never (x,y) itself).
        for y in 0..h {
            for x in 0..w {
                // Read the index from channel[chan_index] BEFORE we
                // overwrite it.
                let pos = (y * w + x) as usize;
                let index = img.channels[chan_index][pos];
                let is_delta = index < nb_deltas as i32;

                let mut value;
                if index >= 0 && index < nb_colours as i32 {
                    // value = channel[0](index, c)
                    let pidx = index;
                    let pv = palette_table
                        .get((c as i32 * palette_w + pidx) as usize)
                        .copied()
                        .ok_or_else(|| {
                            Error::InvalidData(format!(
                                "JXL Palette: index {pidx} out of palette {palette_w}"
                            ))
                        })?;
                    value = pv;
                } else if index >= nb_colours as i32 {
                    // implicit colour from numeric index extrapolation
                    let mut idx = index - nb_colours as i32;
                    if idx < 64 {
                        // value = ((idx >> (2*c)) % 4) * ((1 << bd) - 1) / 4
                        //       + (1 << max(0, bd - 3))
                        let v_part = (idx >> (2 * c as i32)).rem_euclid(4);
                        value = v_part
                            .wrapping_mul(saturate_div_4)
                            .wrapping_add(small_offset);
                    } else {
                        idx -= 64;
                        for _ in 0..c {
                            idx /= 5;
                        }
                        let v_part = idx.rem_euclid(5);
                        value = v_part.wrapping_mul(saturate_div_4);
                    }
                } else if c < 3 {
                    // delta-palette via kDeltaPalette table
                    let neg_idx = (-index - 1).rem_euclid(143);
                    let row = ((neg_idx + 1) >> 1) as usize;
                    let row = row.min(K_DELTA_PALETTE.len().saturating_sub(1));
                    let mut v = K_DELTA_PALETTE[row][c];
                    if (neg_idx & 1) == 0 {
                        v = -v;
                    }
                    if bitdepth > 8 {
                        v <<= bitdepth - 8;
                    }
                    value = v;
                } else {
                    value = 0;
                }
                // Per H.6.4: assign value, then if delta-palette, add
                // `prediction(x, y, d_pred)` against the already-decoded
                // neighbours of (x, y) in this channel. Neighbour lookup
                // never reads (x, y) itself, so the pre-write step is
                // safe even though `predict` borrows `img` immutably.
                if is_delta {
                    let p =
                        palette_predict_for_inverse(img, chan_index, x as i32, y as i32, d_pred)?;
                    value = value.wrapping_add(p);
                }
                img.channels[chan_index][pos] = value;
            }
        }
    }

    // Remove the meta-channel.
    img.channels.remove(0);
    img.descs.remove(0);

    Ok(())
}

/// 2024-spec Annex H.6.2 — Tendency function.
fn squeeze_tendency(a: i32, b: i32, c: i32) -> i32 {
    let mut x = (4i64 * a as i64 - 3 * c as i64 - b as i64 + 6).div_euclid(12) as i32;
    if a >= b && b >= c {
        if x.wrapping_sub(x & 1) > 2i32.wrapping_mul(a.wrapping_sub(b)) {
            x = 2i32.wrapping_mul(a.wrapping_sub(b)).wrapping_add(1);
        }
        if x.wrapping_add(x & 1) > 2i32.wrapping_mul(b.wrapping_sub(c)) {
            x = 2i32.wrapping_mul(b.wrapping_sub(c));
        }
        x
    } else if a <= b && b <= c {
        if x.wrapping_add(x & 1) < 2i32.wrapping_mul(a.wrapping_sub(b)) {
            x = 2i32.wrapping_mul(a.wrapping_sub(b)).wrapping_sub(1);
        }
        if x.wrapping_sub(x & 1) < 2i32.wrapping_mul(b.wrapping_sub(c)) {
            x = 2i32.wrapping_mul(b.wrapping_sub(c));
        }
        x
    } else {
        0
    }
}

/// 2024-spec Annex H.6.2 — Horizontal inverse squeeze step. Combines
/// `input_1` (W1 × H, "averages") and `input_2` (W2 × H, "residuals")
/// into one output channel of dimensions `(W1 + W2) × H`. Either
/// W1 == W2 or W1 == W2 + 1.
pub fn horiz_isqueeze(
    input_1: &[i32],
    w1: u32,
    input_2: &[i32],
    w2: u32,
    h: u32,
) -> Result<(Vec<i32>, u32)> {
    if !(w1 == w2 || w1 == w2 + 1) {
        return Err(Error::InvalidData(format!(
            "JXL Squeeze (horiz): w1={w1} w2={w2} not pair-compatible"
        )));
    }
    let out_w = w1 + w2;
    let mut out = vec![0i32; (out_w as usize) * (h as usize)];
    for y in 0..h {
        for x in 0..w2 {
            let avg = input_1[(y * w1 + x) as usize];
            let residu = input_2[(y * w2 + x) as usize];
            let next_avg = if x + 1 < w1 {
                input_1[(y * w1 + x + 1) as usize]
            } else {
                avg
            };
            let left = if x > 0 {
                out[(y * out_w + (x << 1) - 1) as usize]
            } else {
                avg
            };
            let diff = residu.wrapping_add(squeeze_tendency(left, avg, next_avg));
            // first = (2*avg + diff - sign(diff) * (diff & 1)) >> 1
            let sgn = diff.signum();
            let first = (2i64 * avg as i64 + diff as i64 - sgn as i64 * (diff & 1) as i64) >> 1;
            let first = first as i32;
            out[(y * out_w + 2 * x) as usize] = first;
            out[(y * out_w + 2 * x + 1) as usize] = first.wrapping_sub(diff);
        }
        if w1 > w2 {
            out[(y * out_w + 2 * w2) as usize] = input_1[(y * w1 + w2) as usize];
        }
    }
    Ok((out, out_w))
}

/// 2024-spec Annex H.6.2 — Vertical inverse squeeze step.
pub fn vert_isqueeze(
    input_1: &[i32],
    h1: u32,
    input_2: &[i32],
    h2: u32,
    w: u32,
) -> Result<(Vec<i32>, u32)> {
    if !(h1 == h2 || h1 == h2 + 1) {
        return Err(Error::InvalidData(format!(
            "JXL Squeeze (vert): h1={h1} h2={h2} not pair-compatible"
        )));
    }
    let out_h = h1 + h2;
    let mut out = vec![0i32; (w as usize) * (out_h as usize)];
    for y in 0..h2 {
        for x in 0..w {
            let avg = input_1[(y * w + x) as usize];
            let residu = input_2[(y * w + x) as usize];
            let next_avg = if y + 1 < h1 {
                input_1[((y + 1) * w + x) as usize]
            } else {
                avg
            };
            let top = if y > 0 {
                out[(((y << 1) - 1) * w + x) as usize]
            } else {
                avg
            };
            let diff = residu.wrapping_add(squeeze_tendency(top, avg, next_avg));
            let sgn = diff.signum();
            let first = (2i64 * avg as i64 + diff as i64 - sgn as i64 * (diff & 1) as i64) >> 1;
            let first = first as i32;
            out[(2 * y * w + x) as usize] = first;
            out[((2 * y + 1) * w + x) as usize] = first.wrapping_sub(diff);
        }
    }
    if h1 > h2 {
        for x in 0..w {
            out[((2 * h2) * w + x) as usize] = input_1[(h2 * w + x) as usize];
        }
    }
    Ok((out, out_h))
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
        let wp = WpHeader::default();
        assert_eq!(predict(&img, 0, 0, 0, 0, &wp, None).unwrap().0, 0);
        assert_eq!(predict(&img, 0, 1, 1, 0, &wp, None).unwrap().0, 0);
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
        let wp = WpHeader::default();
        // (1, 0) → left = sample at (0, 0) = 5
        assert_eq!(predict(&img, 0, 1, 0, 1, &wp, None).unwrap().0, 5);
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
        let wp = WpHeader::default();
        // (0, 1) → top = sample at (0, 0) = 5
        assert_eq!(predict(&img, 0, 0, 1, 2, &wp, None).unwrap().0, 5);
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
        let wp = WpHeader::default();
        // sample at (1, 1): left = 30 (sample at (0,1)), top = 0 (sample at (1,0))
        // (30 + 0) / 2 = 15
        assert_eq!(predict(&img, 0, 1, 1, 3, &wp, None).unwrap().0, 15);
    }

    #[test]
    fn predict_predictor_6_at_origin_returns_zero() {
        // At (0, 0) all neighbours collapse to 0; sub-predictors are 0;
        // (prediction + 3) >> 3 == 0.
        let img = ModularImage {
            channels: vec![vec![0; 4]],
            descs: vec![ChannelDesc {
                width: 2,
                height: 2,
                hshift: 0,
                vshift: 0,
            }],
        };
        let wp = WpHeader::default();
        let state = WpState::new(2, 2);
        let (v, _, _) = predict(&img, 0, 0, 0, 6, &wp, Some(&state)).unwrap();
        assert_eq!(v, 0);
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

    /// Round-2 inverse-transform unit tests — Annex H.6.
    #[test]
    fn inverse_rct_type_0_identity_after_no_op() {
        // type==0 means no add-back operations; D = A, E = B, F = C.
        // permutation=0 → V[0]=D=A, V[1]=E=B, V[2]=F=C — exact identity.
        let mut img = ModularImage {
            channels: vec![vec![10i32], vec![20i32], vec![30i32]],
            descs: vec![
                ChannelDesc {
                    width: 1,
                    height: 1,
                    hshift: 0,
                    vshift: 0,
                };
                3
            ],
        };
        inverse_rct(&mut img, 0, 0).unwrap();
        assert_eq!(img.channels[0][0], 10);
        assert_eq!(img.channels[1][0], 20);
        assert_eq!(img.channels[2][0], 30);
    }

    #[test]
    fn inverse_rct_type_6_ycgco_round_trip() {
        // YCgCo (type 6, permutation 0) maps (Y, Co, Cg) → (G, R, B) per
        // FDIS L.4. With Y=128, Co=0, Cg=0: tmp = 128 - 0 = 128; E = 0 + 128 = 128;
        // F = 128 - 0 = 128; D = 128 + 0 = 128. So all three V are 128.
        let mut img = ModularImage {
            channels: vec![vec![128i32], vec![0i32], vec![0i32]],
            descs: vec![
                ChannelDesc {
                    width: 1,
                    height: 1,
                    hshift: 0,
                    vshift: 0,
                };
                3
            ],
        };
        inverse_rct(&mut img, 0, 6).unwrap();
        assert_eq!(img.channels[0][0], 128);
        assert_eq!(img.channels[1][0], 128);
        assert_eq!(img.channels[2][0], 128);
    }

    #[test]
    fn inverse_palette_explicit_colour_lookup() {
        // 2x2 indices channel + 4-colour palette meta channel.
        // Layout: meta @ idx 0 (4×1), indices @ idx 1 (2×2).
        // num_c=1, nb_colours=4, nb_deltas=0, d_pred=0.
        // Palette: [10, 20, 30, 40].
        // Indices: [0, 1; 2, 3] → expected output [10, 20; 30, 40].
        let mut img = ModularImage {
            channels: vec![vec![10, 20, 30, 40], vec![0, 1, 2, 3]],
            descs: vec![
                ChannelDesc {
                    width: 4,
                    height: 1,
                    hshift: -1,
                    vshift: -1,
                },
                ChannelDesc {
                    width: 2,
                    height: 2,
                    hshift: 0,
                    vshift: 0,
                },
            ],
        };
        inverse_palette(&mut img, 0, 1, 4, 0, 0, 8).unwrap();
        // After inverse: meta is removed, channel 0 holds the colours.
        assert_eq!(img.channels.len(), 1);
        assert_eq!(img.channels[0], vec![10, 20, 30, 40]);
    }

    #[test]
    fn squeeze_tendency_zero_when_neighbours_disagree() {
        // tendency returns 0 when neither A>=B>=C nor A<=B<=C.
        assert_eq!(squeeze_tendency(10, 5, 8), 0);
        // Strictly monotone: A>=B>=C: A=20, B=10, C=5.
        // x = (4*20 - 3*5 - 10 + 6) / 12 = (80-15-10+6)/12 = 61/12 = 5
        // first guard: x - (x&1) = 4 vs 2*(20-10)=20. 4 > 20? no.
        // second guard: x + (x&1) = 6 vs 2*(10-5)=10. 6 > 10? no.
        // So x = 5.
        assert_eq!(squeeze_tendency(20, 10, 5), 5);
    }

    #[test]
    fn horiz_isqueeze_simple_pair() {
        // input_1 = [10] (averages, 1×1), input_2 = [0] (residuals, 1×1),
        // output dims = 2×1. With residu=0 and tendency=0 (degenerate
        // single-element case), diff=0; first = (2*10 + 0 - 0) >> 1 = 10;
        // out = [10, 10].
        let (out, w) = horiz_isqueeze(&[10], 1, &[0], 1, 1).unwrap();
        assert_eq!(w, 2);
        assert_eq!(out, vec![10, 10]);
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
        let p = get_properties(&img, 0, 0, 0, 0, 0);
        assert_eq!(p[0], 0); // channel index
        assert_eq!(p[1], 0); // stream index
        assert_eq!(p[2], 0); // y
        assert_eq!(p[3], 0); // x
        assert!(p.len() >= 16);
    }
}
