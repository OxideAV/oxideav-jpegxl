# Blocked: ISO/IEC 18181-1 (JPEG XL) bit-level spec not in docs/

**Status:** Blocked — no further pixel-decoder work (Modular MA-tree
path or VarDCT path) until the normative ISO/IEC 18181-1 document is
added to `docs/image/jxl/`.

## Context

A planned round aimed to start the **Modular sub-bitstream pixel decode
path** in this crate (per ISO/IEC 18181-1 §H "Modular sub-bitstream"):
frame header + TOC + GroupHeader for Modular frames (§E + §F), MA-tree
(§H.5), the JXL-flavoured ANS entropy coder (§D), the named predictors
(§H.4), and enough of the squeeze transform (§H.6) to land a flat
single-channel fixture decoding end-to-end against a `cjxl`-encoded
reference.

Workspace policy (HARD rule from the round brief, mirroring the
codebase-wide clean-room policy) explicitly forbids reading third-party
source: **libjxl, jxlatte, jxl-rs, FUIF, brunsli, or any other JXL
implementation** are off-limits. `cjxl` / `djxl` may only be used as
external-process validators that produce reference bytes — never as a
source to read.

The ONLY acceptable bit-level reference is the ISO/IEC 18181-1 PDF
present in `docs/image/jxl/`.

## Documents checked in `docs/image/`

`docs/image/jxl/` does **not exist** in the workspace as of this round
(2026-04-30). The only image-codec specs present are:

| Directory | What's inside |
| --- | --- |
| `docs/image/jpeg/` | T.81 (1992), T.871 (JFIF, 2011), T.872 (float, 2012). |
| `docs/image/motion-jpeg/` | RFC 2435 (Motion JPEG over RTP). |
| `docs/image/avif/` | AOM AV1 Image File Format HTML. |
| `docs/image/heif/` | ISO/IEC 23008-12:2017 (1st ed.). |
| `docs/image/jpeg2000/` | T.800 (Part 1, 2019), T.814 (Motion JP2K). |
| `docs/image/jpegxr/` | T.832 (core, 2019), T.833 (motion, 2010), T.834 (conformance, 2014). |
| `docs/image/png/` | RFC 2083, W3C PNG 3rd Edition. |

`docs/README.md` already lists ISO/IEC 18181 explicitly under
**"Known gaps — ISO/IEC (paid)"** alongside ISO/IEC 21122 (JPEG-XS),
ISO/IEC 14496-15 newer editions, ISO/IEC 23001-7 (CENC), and similar
items. The standard is published — ISO/IEC 18181-1:2024 (2nd edition,
~91 pp, July 2024) and ISO/IEC 18181-1:2022 (1st edition) — but only
behind ISO / ANSI / accuristech paywalls. No freely redistributable
draft is available; the JPEG committee's `ds.jpeg.org` whitepaper is
informational only and contains no codeword tables, no MA-tree node
syntax, no ANS distribution layout, no predictor formulas, and no
bundle bit layout.

Workspace-wide search confirms no in-repo copy exists:
`find /Users/magicaltux/projects/oxideav-workspace -iname "*18181*"`
returns only build-artifact filenames (cargo hash collisions); no `.pdf`
or `.txt` ISO document is present.

## What this crate does today (commit `b0c97e7` baseline)

Header-only, correct as far as it goes, written from the ISO 18181-1
text the original implementer had access to before the workspace
policy tightened:

- `container::{detect, extract_codestream, Signature}` — raw `FF 0A`
  signature + ISOBMFF wrapper detection, codestream assembled from
  `jxlc` / `jxlp` boxes (large-size headers handled, jxlp index prefix
  stripped).
- `bitreader::{BitReader, U32Dist}` — LSB-first bit reader and the
  JXL `U32` 2-bit-selector `{Val, Bits, BitsOffset}` distribution.
- `metadata::parse_headers` — `SizeHeader` (small + large + 7-entry
  fixed aspect ratio table, all four `U32` selectors), then
  `ImageMetadata` up to `num_extra_channels`: `all_default` shortcut,
  `extra_fields` (orientation + intrinsic size + preview / animation
  flag presence detection), the `BitDepth` sub-bundle (integer
  1..=31 with range checking + IEEE float with valid-exponent /
  valid-mantissa range checks), and the
  `modular_16_bit_buffer_sufficient` flag.
- `make_decoder` returns `Error::Unsupported("jxl decode not yet
  implemented")`; `make_encoder` likewise.

The crate identifies a `.jxl` file and reports its dimensions and
sample format, but cannot produce a pixel.

## Why this is genuinely blocked, not deferrable

The JXL Modular pixel pipeline is a stack of interlocking subsystems,
and none of them are reconstructable from non-spec material:

- **MA-tree (§H.5)** — a binary decision tree where each node tests a
  property (channel index, position, neighbour values, gradient, ...)
  using one of 16+ named property functions, with thresholds drawn
  from a context distribution. The tree itself is encoded against a
  meta-distribution. Both the property catalogue and its bit-level
  layout are normative-only.
- **ANS / Brotli entropy (§D)** — JXL uses a specific 8-bit ANS variant
  (precision 12, with a custom alphabet-size encoding, prefix-coded
  symbol distributions, and "lz77" symbol slots that defer back into
  the same stream). Implementing "an ANS that decodes one of the JXL
  reference images" requires the exact distribution-table syntax, the
  exact state-update rule, and the symbol-side-info conventions.
- **Predictors (§H.4)** — 14 named predictors (Zero, Left, Top, Average,
  Select, Gradient, Weighted, ...). The Weighted predictor maintains
  four sub-predictor states with adaptive weights updated per pixel
  using a documented error-feedback formula. Reproducing it without
  the spec's exact tables and shift counts will be wrong-by-a-bit.
- **Squeeze transform (§H.6)** — the residual / subsampling preprocess
  uses a Haar-like horizontal+vertical decomposition with documented
  rounding behaviour at borders.
- **Frame header + TOC (§E + §F)** — group geometry, pass / restoration
  filter flags, RCT / Patches / Splines / Noise sub-bundles, all using
  the same `Bundle::AllDefault` shortcut bits the metadata parser
  already does — but the field layouts are spec-normative.

Implementing "something plausible" without the spec would not interop
with `djxl` and would mislead users who see `Error::Unsupported`
disappear.

## Unblock procedure

1. Add the ISO/IEC 18181-1 PDF (preferably the 2024 2nd edition,
   `ISO_IEC_18181-1_2024_Information_technology_-_JPEG_XL_image_coding_system_-_Part_1_Core_coding_system.pdf`,
   ~91 pp) to `docs/image/jxl/`. Source: ISO webstore, ANSI webstore,
   or accuristech.
2. Optionally add ISO/IEC 18181-2 (file format, 2024) so the ISOBMFF
   `jbrd` / `Exif` / `xml ` / `jumb` boxes can be promoted from
   "ignored" to "diagnostic".
3. Optionally add ISO/IEC 18181-3 (conformance, 2022) to source the
   reference images for golden tests.
4. Re-run this round. The work plan is:
   - Finish `parse_image_metadata` — currently stops at
     `num_extra_channels`. Add `ExtraChannelInfo` + `ColorEncoding` +
     `ToneMapping` + `PreviewHeader` + `AnimationHeader` decode.
   - Add `frame::FrameHeader` + `frame::Toc` + `frame::GroupHeader`
     parsing for Modular-only frames (skip VarDCT branches with
     `Error::Unsupported`).
   - Implement the JXL ANS decoder against §D, starting with a
     single-context flat distribution to land a smoke test.
   - Implement the MA-tree walker against §H.5 with the property
     functions enumerated by the spec, plus the Zero + Gradient
     predictors from §H.4 (sufficient for a flat single-channel
     fixture).
   - Add `tests/cjxl_interop.rs` that runs `cjxl` on a 1×8 grayscale
     gradient → decodes with this crate → compares pixels exactly.

## Remaining gaps (unchanged)

- Pixel decoding (entire Modular path, entire VarDCT path).
- Encoder (out of scope for this crate).
- Animation, preview, intrinsic-size sub-bundle decoding.
- `ColorEncoding`, `ToneMapping`, `ExtraChannelInfo`.
- `FrameHeader`, TOC, GroupHeader.

None of the above can advance without the ISO/IEC 18181-1 normative
text in `docs/image/jxl/`.
