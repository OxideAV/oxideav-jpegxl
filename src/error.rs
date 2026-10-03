//! Crate-local error type of the standalone (no `oxideav-core`) API.
//!
//! Every decode stage in this crate returns [`Result`]; the registry
//! adapter (behind the `registry` feature) maps [`JxlError`] onto
//! `oxideav_core::Error` so the framework `Decoder` can keep returning
//! `oxideav_core::Result<T>` while the decoding code stays
//! framework-free.

use std::fmt;

/// `Result` alias scoped to `oxideav-jpegxl`.
pub type Result<T> = std::result::Result<T, JxlError>;

/// The contract name for [`JxlError`].
pub type Error = JxlError;

/// Error variants returned by `oxideav-jpegxl`.
///
/// `InvalidData`, `Unsupported`, `LimitExceeded` and `Io` are the
/// image-crate contract floor; `Eof` / `NeedMore` / `Other` exist so
/// the framework trait surface can be expressed in the same type.
#[derive(Debug)]
#[non_exhaustive]
pub enum JxlError {
    /// The codestream or container violates ISO/IEC 18181 (bad
    /// signature, truncated bundle, inconsistent TOC, …).
    InvalidData(String),
    /// Valid input exercising a feature this decoder does not
    /// implement, or an encode request (the crate is decoder-only).
    Unsupported(String),
    /// A [`crate::DecodeOptions`] limit (dimensions / pixels / bytes)
    /// would be exceeded; nothing was allocated.
    LimitExceeded(String),
    /// A read / write on a caller-supplied stream failed
    /// ([`crate::decode_from`] / [`crate::encode_to`]).
    Io(std::io::Error),
    /// End of stream — no more frames forthcoming (framework surface).
    Eof,
    /// More input is required before another frame can be produced
    /// (framework surface).
    NeedMore,
    /// Caller protocol violations and anything that fits no other
    /// variant.
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

    /// Construct a [`JxlError::LimitExceeded`] from a stringy message.
    pub fn limit(msg: impl Into<String>) -> Self {
        Self::LimitExceeded(msg.into())
    }

    /// Construct a [`JxlError::Other`] from a stringy message.
    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other(msg.into())
    }
}

impl From<std::io::Error> for JxlError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl fmt::Display for JxlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidData(s) => write!(f, "invalid data: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported: {s}"),
            Self::LimitExceeded(s) => write!(f, "limit exceeded: {s}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Eof => write!(f, "end of stream"),
            Self::NeedMore => write!(f, "need more data"),
            Self::Other(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for JxlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_prefixes_match_variants() {
        assert_eq!(
            JxlError::invalid("x").to_string(),
            "invalid data: x".to_string()
        );
        assert_eq!(JxlError::unsupported("y").to_string(), "unsupported: y");
        assert_eq!(JxlError::limit("z").to_string(), "limit exceeded: z");
        assert_eq!(JxlError::Eof.to_string(), "end of stream");
    }

    #[test]
    fn io_error_converts_and_sources() {
        let e: JxlError = std::io::Error::other("disk").into();
        assert!(matches!(e, JxlError::Io(_)));
        assert!(std::error::Error::source(&e).is_some());
    }
}
