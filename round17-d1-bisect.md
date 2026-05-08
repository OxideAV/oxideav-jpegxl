# Round-17 d1 bit-position-drift bisect (Auditor mode)

**Date**: 2026-05-09  
**Fixture**: `crates/oxideav-jpegxl/tests/fixtures/vardct_256x256_d1.jxl` (256x256 VarDCT lossy at distance 1.0)  
**Input symptom (round 16)**: `InvalidData("JXL Modular Squeeze: end 40 >= channel count 4")` raised by `apply_transforms_to_channel_layout` when the HfMetadata sub-bitstream's first decoded `SqueezeParam.begin_c == 39`. Round 16 hypothesised an upstream bit-position drift; round 17 confirms it.

## Ground truth (cjxl/djxl black-box trace)

The conformance fixture ships with a hand-curated bit-position trace at  
`docs/image/jpegxl/fixtures/vardct-256x256-d1/trace.txt` (19 lines, written by the docs collaborator from `cjxl --debug` output). The relevant bit-position checkpoints:

| Checkpoint | Trace bit_pos | Notes |
|---|---|---|
| `DC_GLOBAL_BEGIN` | 0 | (relative to first byte after TOC; codestream bit 72) |
| `DC_GLOBAL_END` | **1026** | LfGlobal section ends here |
| `DC_GROUP_BEGIN` | 1026 | LfGroup section starts here |
| `DC_GROUP_END` | **12754** | LfGroup section ends here (= LfCoefficients + ModularLfGroup + HfMetadata) |
| `AC_GLOBAL_BEGIN` | 12754 | HfGlobal section starts here |
| `AC_GLOBAL_END` | 13061 | HfGlobal section ends here |

So the cjxl-derived budget for the entire `LfGroup` bundle (G.2) is **11728 bits** (= 12754 − 1026).

## Round-17 trace: oxideav-jpegxl positions

Captured by `cargo test --release d1_bit_position_walk_round_17 -- --nocapture` (test in `tests/round17_d1_bit_trace.rs`):

```
[r17-trace] LF GLOBAL
  after LfChannelDequant   = 1
  after Quantizer          = 22
  after HfBlockContext     = 109   (consumed 87 bits, used_default=false, nb_block_ctx=3)
  after LfChannelCorr      = 110
  after GlobalModular      = 1026  (consumed 916 bits, fully_decoded=true, tree_present=true)
LF GLOBAL END = 1026 bits   ← MATCHES cjxl trace exactly
[r17-trace] LF GROUP (lf_dim=256x256)
  LfCoeff ModularHeader: iugt=true default_wp=true nb_transforms=0 (6 bits)
  after LfCoefficients     = 13021 (consumed 11995 bits)
  HfMetadata starts at     = 13021
  HfMetadata ERR @ bit 13254 (consumed 233 bits) — InvalidData("JXL Modular Squeeze: end 40 >= channel count 4")
```

**Key delta:** our `LfCoefficients::read` alone consumes **11995 bits** — already 267 bits more than the cjxl-traced total budget for the entire `LfGroup` bundle (11728 bits). With `ModularLfGroup` being a no-op (zero channels) and `HfMetadata` yet to run, we are clearly over-consuming inside the LfCoefficients per-channel decode loop.

## Bit-precise breakdown

`LfCoefficients::read` proper:

| Step | Bits | Cumulative |
|---|---|---|
| `extra_precision = u(2)` | 2 | 2 |
| ModularHeader (`iugt=true` + default WP + `nb_transforms=Val(0)`) | 4 | 6 |
| ANS state init `u(32)` (read at first symbol decode time per round-3 fix) | 32 | 38 |
| Per-sample loop, 3 channels × 32 × 32 = **3072 samples** | 11957 | 11995 |

3072 samples × **3.89 bits / sample** = 11957 bits.

Inferred cjxl budget for the same per-sample loop (subtracting HfMetadata's plausible ~6800-bit budget from the 11728-bit DC_GROUP total):

`11728 − 38 (LfCoeff header + state init) − 6800 (HfMetadata estimate) ≈ 4890 bits` → **1.6 bits / sample**.

So we are over-consuming by roughly **2.3 bits per LF sample** (3.89 − 1.6 ≈ 2.3) — about **7000 bits total** across all 3072 samples. The decoded sample values look plausible (smooth, monotonic gradients in ch0, small chrominance values in ch1/ch2), suggesting our entropy decoder produces "real" tokens, but consumes too many trailing extra bits per token.

## Suspect ranking (revised against r16 hypothesis)

The r16 hypothesis ranked HfBlockContext custom branch as HIGH and HfGlobal/LfCoefficients-prelude as MEDIUM. Round-17 evidence flips the picture:

* **HIGH (CONFIRMED)** — `decode_channels_at_stream` per-sample drift in LfCoefficients. The per-sample loop reads ≈2.3 bits more than the spec demands, accumulating ~7000 bits across 3072 samples. Likely cause: wrong **`HybridUintConfig`** sourcing per cluster, OR wrong **`dist_multiplier`** application (unlikely since LZ77 is OFF in this stream), OR a stray **post-token tail-bits** read.
* **HIGH (CONFIRMED RULED OUT)** — HfBlockContext is the WRONG suspect. The trace shows HfBlockContext consumes 87 bits in the custom branch with `nb_lf_thr=[0,0,0] nb_qf_thr=0` (smallest legal custom path). LfGlobal correctly ends at bit 1026, matching cjxl exactly.
* **LOW** — LfGlobal/Quantizer/LfChannelCorrelation positions all match cjxl. GlobalModular ends at bit 1026 exactly.
* **LOW** — HfMetadata's parse logic itself is correct per FDIS Table H.7 + H.9 (verified against ISO/IEC 18181-1:2024 page 50). The garbage SqueezeParam values (begin_c=39, num_c=2 → end=40) are a *symptom* of LfCoefficients over-reading into HfMetadata's bit territory.

## Spec corroboration (ISO/IEC 18181-1:2024)

* **G.2.4 HF metadata** (page 43): `nb_blocks = 1 + u(ceil(log2(ceil(width / 8) * ceil(height / 8))))` — our code reads `u(nbits)` then computes `nb_blocks = stored + 1`, equivalent. **Not a bug.**
* **Table H.7 TransformInfo** + **Table H.9 SqueezeParams** (page 50/51): `tr == kSqueeze: U32(0, 1+u(4), 9+u(6), 41+u(8))` for `num_sq`; `U32(u(3), 8+u(6), 72+u(10), 1096+u(13))` for `begin_c`; `U32(1, 2, 3, 4+u(4))` for `num_c`. **Our `TransformInfo::read` and `SqueezeParam::read` match the 2024 spec exactly.**
* **C.9.1 trivial case** (page ~45 of FDIS / cross-checked in 2024 page 47): "In the trivial case where N is zero, the decoder takes no action." Round 15's `prelim_descs.is_empty()` gate in GlobalModular is consistent with this. **Not a bug.**
* **Annex H.5 / Table H.5 WPHeader**: 11 fields after `default_wp=false` (7×u(5) + 4×u(4) = 51 bits). Our `WpHeader::read` matches. **Not a bug.** (The HfMetadata trace shows a non-default WPHeader because we are decoding garbage bits from inside LfCoefficients's tail — that is a *consequence*, not a cause.)

## r18 candidate (specific module + line targets)

* **PRIMARY**: `crates/oxideav-jpegxl/src/modular_fdis.rs::decode_channels_at_stream` and the `decode_uint_in_with_dist` it calls. Specifically, the **hybrid-uint extra-bits path** (`HybridUintConfig::decode_extra_bits` in `src/ans/hybrid.rs`) when the leaf is reached via the **global tree** (`MaTreeFdis::cloned_with_fresh_state`).
* **Bisect strategy for r18**: Add per-token bit accounting inside `decode_uint_in_with_dist` (gated behind `cfg(test)`). Run d1 → expect to see a per-token over-consumption of ~2.3 bits in the LF channel loop. Compare against ground-truth reading with the spec's Listing D.6 (`DecodeHybridVarLenUint`) — likely a stray `u(extra_bits)` outside its expected gate, OR a wrong `HybridUintConfig.split_exponent / msb_in_token / lsb_in_token` triple read inside `EntropyStream::read` for the leaf-level stream.

A secondary line of investigation: the trace shows leaf-level entropy stream `num_contexts=16 num_histograms=5 log_alpha_size=6 bits=602`. Our `EntropyStream::read` produces a stream with the same num_contexts/num_histograms (verifiable via debug-print inside `MaTreeFdis::read`). If the prelude bit count itself is correct (~602), the drift MUST be downstream — i.e. in per-token decode. If our prelude consumes more than 602 bits, the drift is in the prelude.

## Round-17 fix landed

**No fix landed.** The bug class is "wrong per-token extra-bits accounting in hybrid uint path", which is a deeper algorithmic divergence — not an off-by-one in a U32 distribution or a wrong condition. Per Auditor-mode rules, round-17 ships the diagnostic only and defers the fix to round 18.

## Sentinels

The five small lossless fixtures (pixel_1x1, gray_64x64, gradient_64x64, palette_32x32, grey_8x8) all stay green. Round 11–16 sentinel tests stay green. The new `d1_bit_position_walk_round_17` test stays green (it never asserts; it just emits the trace under `--nocapture`).
