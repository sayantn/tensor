//! Associative, commutative reduction operators ([`Monoid`]) for SIMD-friendly
//! 16-lane tensor reductions.

use core::ops::{Add, Mul};
use num_traits::{One, Zero};

#[cfg(feature = "std")]
use num_traits::Float;

/// An associative, commutative reduction operator with an identity element.
///
/// Reductions over [`Monoid`] operators evaluate elements across 16
/// independent accumulators before combining lanes, breaking loop-carried
/// dependency chains so LLVM can emit SIMD vector instructions under strict
/// IEEE-754 floating-point rules.
///
/// # Laws
///
/// Implementations must satisfy for all `a`, `b`, `c`:
/// - **Identity**: `self.combine(self.identity(), a) == a`
/// - **Associativity**: `self.combine(self.combine(a, b), c) == self.combine(a, self.combine(b, c))`
/// - **Commutativity**: `self.combine(a, b) == self.combine(b, a)`
///
/// *(Note: Floating-point addition and multiplication are only approximately
/// associative; lane-wise evaluation produces a deterministic, fixed-tree
/// reduction order without `-ffast-math`.)*
pub trait Monoid<T>: Copy {
    /// Returns the two-sided identity element of the reduction.
    fn identity(self) -> T;

    /// Combines an accumulated lane value `acc` with a new element `val`.
    fn combine(self, acc: T, val: T) -> T;
}

/// Additive reduction monoid (`0`, `+`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Sum;

impl<T: Copy + Zero + Add<Output = T>> Monoid<T> for Sum {
    #[inline]
    fn identity(self) -> T {
        T::zero()
    }

    #[inline]
    fn combine(self, acc: T, val: T) -> T {
        acc + val
    }
}

/// Multiplicative reduction monoid (`1`, `*`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Prod;

impl<T: Copy + One + Mul<Output = T>> Monoid<T> for Prod {
    #[inline]
    fn identity(self) -> T {
        T::one()
    }

    #[inline]
    fn combine(self, acc: T, val: T) -> T {
        acc * val
    }
}

/// Minimum reduction monoid (`+∞` / `T::max_value()`, `min`).
///
/// Uses IEEE-754 [`Float::min`] (`f32::min` / `f64::min`), which propagates
/// non-`NaN` values when one operand is `NaN` and maps directly to hardware
/// vector minimum instructions (`vminps` / `fmin`).
#[cfg(feature = "std")]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Min;

#[cfg(feature = "std")]
impl<T: Float> Monoid<T> for Min {
    #[inline]
    fn identity(self) -> T {
        T::infinity()
    }

    #[inline]
    fn combine(self, acc: T, val: T) -> T {
        acc.min(val)
    }
}

/// Maximum reduction monoid (`-∞` / `T::min_value()`, `max`).
///
/// Uses IEEE-754 [`Float::max`] (`f32::max` / `f64::max`), which propagates
/// non-`NaN` values when one operand is `NaN` and maps directly to hardware
/// vector maximum instructions (`vmaxps` / `fmax`).
#[cfg(feature = "std")]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Max;

#[cfg(feature = "std")]
impl<T: Float> Monoid<T> for Max {
    #[inline]
    fn identity(self) -> T {
        T::neg_infinity()
    }

    #[inline]
    fn combine(self, acc: T, val: T) -> T {
        acc.max(val)
    }
}
