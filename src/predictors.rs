//! Modular sub-bitstream pixel predictors, per ISO/IEC 18181-1
//! committee draft (2019-08-05) Annex C.9.3.1.
//!
//! Five predictors are defined for Modular channel decoding:
//!
//! | id | name      | formula                                  |
//! |----|-----------|------------------------------------------|
//! | 0  | Zero      | `ch_zero` (constant per-channel zero)    |
//! | 1  | Average   | `(left + top) Idiv 2`                    |
//! | 2  | Gradient  | `median(left + top - topleft, left, top)`|
//! | 3  | Left      | `left`                                   |
//! | 4  | Top       | `top`                                    |
//!
//! Border behaviour (also from C.9.3.1):
//!
//! * `top    = y > 0           ? channel[i](x, y - 1) : ch_zero`
//! * `left   = x > 0           ? channel[i](x - 1, y) : ch_zero`
//! * `topleft = (x>0 && y>0)   ? channel[i](x - 1, y - 1) : left`
//!
//! `ch_zero` is `max(channel[i].min, min(channel[i].max, 0))`. For an
//! unsigned channel with `min == 0` this is `0`; for a channel that
//! straddles zero this is `0`; for one strictly positive this is `min`
//! and strictly negative is `max`.

/// The five modular predictor ids enumerated by C.9.3.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Predictor {
    Zero = 0,
    Average = 1,
    Gradient = 2,
    Left = 3,
    Top = 4,
}

impl Predictor {
    /// Map a predictor id (0..=4) to the corresponding [`Predictor`].
    /// Returns `None` for any other id (the spec leaves predictor ids
    /// 5..=13 reserved in this clause; the Weighted predictor lives in
    /// E.3 and is not enumerated here).
    pub fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::Zero),
            1 => Some(Self::Average),
            2 => Some(Self::Gradient),
            3 => Some(Self::Left),
            4 => Some(Self::Top),
            _ => None,
        }
    }

    /// Evaluate this predictor given the in-bounds `top` / `left` /
    /// `topleft` neighbour values and the channel's `ch_zero`.
    ///
    /// Border replacement (top/left = ch_zero, topleft = left when
    /// out-of-bounds) is the caller's responsibility; this function
    /// only computes the raw formula.
    pub fn predict(self, left: i32, top: i32, topleft: i32, ch_zero: i32) -> i32 {
        match self {
            Self::Zero => ch_zero,
            Self::Average => (left + top).div_euclid(2),
            Self::Gradient => median3(left + top - topleft, left, top),
            Self::Left => left,
            Self::Top => top,
        }
    }
}

/// Compute the channel's `ch_zero` value, per C.9.3.1:
/// `max(min, min(max, 0))`.
pub fn ch_zero(channel_min: i32, channel_max: i32) -> i32 {
    channel_min.max(channel_max.min(0))
}

/// Three-element median, used by the Gradient predictor.
pub fn median3(a: i32, b: i32, c: i32) -> i32 {
    // Equivalent to median = max(min(a,b), min(max(a,b), c))
    let lo = a.min(b);
    let hi = a.max(b);
    hi.min(c).max(lo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_basic() {
        assert_eq!(median3(1, 2, 3), 2);
        assert_eq!(median3(3, 1, 2), 2);
        assert_eq!(median3(2, 2, 2), 2);
        assert_eq!(median3(-1, 5, 3), 3);
        assert_eq!(median3(10, 0, -5), 0);
    }

    #[test]
    fn ch_zero_unsigned_channel() {
        assert_eq!(ch_zero(0, 255), 0);
    }

    #[test]
    fn ch_zero_strictly_positive_channel() {
        assert_eq!(ch_zero(5, 100), 5);
    }

    #[test]
    fn ch_zero_strictly_negative_channel() {
        assert_eq!(ch_zero(-100, -5), -5);
    }

    #[test]
    fn ch_zero_straddles_zero() {
        assert_eq!(ch_zero(-128, 127), 0);
    }

    #[test]
    fn predict_zero_returns_ch_zero() {
        assert_eq!(Predictor::Zero.predict(50, 60, 40, 7), 7);
    }

    #[test]
    fn predict_left_top_obvious() {
        assert_eq!(Predictor::Left.predict(11, 22, 33, 0), 11);
        assert_eq!(Predictor::Top.predict(11, 22, 33, 0), 22);
    }

    #[test]
    fn predict_average_rounds_floor() {
        // (3 + 8) / 2 = 5 (floor div), not 5.5 → 5.
        assert_eq!(Predictor::Average.predict(3, 8, 0, 0), 5);
        // Negative values: (-1 + -3) / 2 = -2.
        assert_eq!(Predictor::Average.predict(-1, -3, 0, 0), -2);
    }

    #[test]
    fn predict_gradient_clamps_via_median() {
        // Smooth gradient: left=10, top=20, topleft=15 →
        // raw = 10 + 20 - 15 = 15; median(15, 10, 20) = 15.
        assert_eq!(Predictor::Gradient.predict(10, 20, 15, 0), 15);
        // Edge: left=10, top=20, topleft=0 →
        // raw = 30; median(30, 10, 20) = 20 (clamped to top).
        assert_eq!(Predictor::Gradient.predict(10, 20, 0, 0), 20);
        // Reverse edge: left=10, top=20, topleft=100 →
        // raw = -70; median(-70, 10, 20) = 10 (clamped to left).
        assert_eq!(Predictor::Gradient.predict(10, 20, 100, 0), 10);
    }

    #[test]
    fn from_id_round_trips() {
        for (id, p) in [
            (0, Predictor::Zero),
            (1, Predictor::Average),
            (2, Predictor::Gradient),
            (3, Predictor::Left),
            (4, Predictor::Top),
        ] {
            assert_eq!(Predictor::from_id(id), Some(p));
        }
        assert_eq!(Predictor::from_id(5), None);
        assert_eq!(Predictor::from_id(99), None);
    }
}
