//! Crate-local error type used by `oxideav-jpegxl`'s standalone (no
//! `oxideav-core`) public API.
//!
//! Defined as a small std-only enum so the crate can be built with the
//! default `registry` feature off — i.e. without depending on
//! `oxideav-core` at all. When the `registry` feature is on a
//! `From<JxlError> for oxideav_core::Error` impl is enabled in
//! [`crate::registry`] so the `Decoder` / `Encoder` trait surface still
//! interoperates cleanly.
//!
//! The variants mirror the subset of `oxideav_core::Error` that the JPEG
//! XL decoder/encoder pipeline actually produces.

use core::fmt;

/// `Result` alias scoped to `oxideav-jpegxl`. Standalone (no
/// `oxideav-core`) callers see this; framework callers convert via the
/// gated `From<JxlError> for oxideav_core::Error` impl.
pub type Result<T> = core::result::Result<T, JxlError>;

/// Crate-local error type for the JPEG XL decoder / encoder pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JxlError {
    /// The bitstream is malformed (bad signature, truncated header, etc.).
    InvalidData(String),
    /// The bitstream uses a feature this decoder does not implement, or
    /// the encoder was asked to emit a frame format it does not support.
    Unsupported(String),
    /// End of stream — no more packets / frames forthcoming.
    Eof,
    /// More input is required before another frame can be produced
    /// (decoder) or another packet can be flushed (encoder).
    NeedMore,
    /// Catch-all for everything else — invalid params, plane stride
    /// mismatch, etc.
    Other(String),
}

impl JxlError {
    /// Construct a [`JxlError::InvalidData`] from a stringy message.
    pub fn invalid(msg: impl Into<String>) -> Self {
        Self::InvalidData(msg.into())
    }

    /// Construct a [`JxlError::Unsupported`] from a stringy message.
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Self::Unsupported(msg.into())
    }

    /// Construct a [`JxlError::Other`] from a stringy message.
    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other(msg.into())
    }
}

impl fmt::Display for JxlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidData(s) => write!(f, "invalid data: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported: {s}"),
            Self::Eof => write!(f, "end of stream"),
            Self::NeedMore => write!(f, "need more data"),
            Self::Other(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for JxlError {}
