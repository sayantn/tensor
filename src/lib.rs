//! A minimal, `#![no_std]` row-major dense tensor library with zero-copy views
//! and pluggable Level-1/2/3 linear algebra backends.
//!
//! # Core Types
#![cfg_attr(
    feature = "alloc",
    doc = "- [`Tensor<T, N>`]: Heap-allocated (`alloc::boxed::Box<[T]>`) row-major\n  owned tensor of rank `N`."
)]
//! - [`TensorView<'a, T, N>`]: Immutable, `Copy`, zero-allocation borrow over a
//!   contiguous row-major slice.
//! - [`TensorViewMut<'a, T, N>`]: Exclusive mutable borrow over a contiguous
//!   row-major slice.
//!
//! Convenience rank aliases are provided for rank-0 ([`ScalarView`],
//! [`ScalarViewMut`]), rank-1 ([`VectorView`], [`VectorViewMut`]), and rank-2
//! ([`MatrixView`], [`MatrixViewMut`]) views
#![cfg_attr(
    feature = "alloc",
    doc = "as well as their owned counterparts ([`Scalar`], [`Vector`], and [`Matrix`])."
)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![no_std]
#![warn(
    unsafe_op_in_unsafe_fn,
    missing_docs,
    clippy::pedantic,
    clippy::undocumented_unsafe_blocks,
    clippy::multiple_unsafe_ops_per_block,
    clippy::as_ptr_cast_mut,
    clippy::float_cmp,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    clippy::missing_inline_in_public_items,
    clippy::dbg_macro,
    clippy::todo,
    clippy::unimplemented
)]
#![allow(
    clippy::len_without_is_empty,
    clippy::many_single_char_names,
    clippy::needless_pass_by_value
)]

#[cfg(feature = "alloc")]
extern crate alloc;

use core::fmt::{self, Display, Formatter};
use core::iter::zip;
use core::ops::{AddAssign, Bound, DivAssign, MulAssign, Range, RangeBounds, SubAssign};

#[cfg(feature = "alloc")]
mod tensor;
mod view;
mod view_mut;

#[cfg(feature = "alloc")]
pub use tensor::*;
pub use view::*;
pub use view_mut::*;

pub mod linalg;

mod reduce;
pub use reduce::*;

// ============================================================================
// Internal Helpers
// ============================================================================

#[inline]
const unsafe fn unchecked_shape_len(shape: &[usize]) -> usize {
    let mut prod = 1usize;
    let mut i = 0;
    while i < shape.len() {
        // SAFETY: Caller guarantees the product of `shape` does not overflow `usize`.
        prod = unsafe { prod.unchecked_mul(shape[i]) };
        i += 1;
    }
    prod
}

#[inline]
#[track_caller]
fn shape_len(shape: &[usize]) -> usize {
    shape
        .iter()
        .fold(1, |acc, &x| acc.checked_mul(x).expect("shape overflow"))
}

#[inline]
#[track_caller]
fn validate_shape(buffer_len: usize, shape: &[usize]) {
    assert!(
        !shape.contains(&0),
        "tensor dimensions must be non-zero, got {shape:?}",
    );
    assert_eq!(
        buffer_len,
        shape_len(shape),
        "buffer len ({buffer_len}) does not match shape {shape:?}",
    );
}

#[inline]
#[track_caller]
fn linear_offset<const N: usize>(shape: &[usize; N], index: &[usize; N]) -> usize {
    zip(shape, index).fold(0, |offset, (&len, &idx)| {
        assert!(idx < len, "index {idx} out of bounds for axis len {len}");
        offset * len + idx
    })
}

#[inline]
fn linear_offset_checked<const N: usize>(shape: &[usize; N], index: &[usize; N]) -> Option<usize> {
    zip(shape, index).try_fold(0, |offset, (&len, &idx)| {
        (idx < len).then_some(offset * len + idx)
    })
}

#[inline]
#[track_caller]
fn resolve_range<R: RangeBounds<usize>>(range: R, axis_len: usize) -> Range<usize> {
    let start = match range.start_bound() {
        Bound::Included(&s) => s,
        Bound::Excluded(&s) => s.checked_add(1).expect("range start overflow"),
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&e) => e.checked_add(1).expect("range end overflow"),
        Bound::Excluded(&e) => e,
        Bound::Unbounded => axis_len,
    };
    assert!(start < end, "slice range {start}..{end} must be non-empty");
    assert!(
        end <= axis_len,
        "slice end {end} exceeds axis len {axis_len}"
    );
    start..end
}

macro_rules! impl_assign_ops {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T: Copy + $trait<T>, const N: usize, const M: usize> $trait<TensorView<'_, T, M>>
            for TensorViewMut<'_, T, N>
        {
            #[inline]
            #[track_caller]
            fn $method(&mut self, rhs: TensorView<'_, T, M>) {
                const {
                    assert!(
                        M <= N,
                        "cannot broadcast higher-rank tensor into lower-rank tensor"
                    );
                }
                assert_eq!(self.shape()[N - M..], rhs.shape(), "broadcast shape mismatch");

                self.view_mut().chunks::<M>().for_each(|chunk| {
                    for (a, &b) in zip(chunk.into_mut_slice(), rhs.as_slice()) {
                        *a $op b;
                    }
                })
            }
        }

        impl<T: Copy + $trait<T>, const N: usize> $trait<T> for TensorViewMut<'_, T, N> {
            #[inline]
            fn $method(&mut self, rhs: T) {
                self.$method(ScalarView::scalar(&rhs))
            }
        }
    };
}

impl_assign_ops!(AddAssign, add_assign, +=);
impl_assign_ops!(SubAssign, sub_assign, -=);
impl_assign_ops!(MulAssign, mul_assign, *=);
impl_assign_ops!(DivAssign, div_assign, /=);

macro_rules! impl_owned_binary_ops {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident) => {
        #[cfg(feature = "alloc")]
        // Tensor<T, N> + RHS (reuses `self` buffer in place; supports broadcasting M <= N)
        impl<T: Copy + $assign_trait<T>, const N: usize, const M: usize>
            core::ops::$trait<TensorView<'_, T, M>> for Tensor<T, N>
        {
            type Output = Tensor<T, N>;

            #[inline]
            #[track_caller]
            fn $method(mut self, rhs: TensorView<'_, T, M>) -> Self::Output {
                self.view_mut().$assign_method(rhs);
                self
            }
        }

        #[cfg(feature = "alloc")]
        impl<T: Copy + $assign_trait<T>, const N: usize> core::ops::$trait<T> for Tensor<T, N> {
            type Output = Tensor<T, N>;

            #[inline]
            fn $method(mut self, rhs: T) -> Self::Output {
                self.view_mut().$assign_method(rhs);
                self
            }
        }

        #[cfg(feature = "alloc")]
        // Borrowed LHS (rank M <= N) + owned Tensor<T, N> (reuses `rhs` buffer in place!)
        impl<T: Copy + core::ops::$trait<T, Output = T>, const N: usize, const M: usize>
            core::ops::$trait<Tensor<T, N>> for TensorView<'_, T, M>
        {
            type Output = Tensor<T, N>;

            #[inline]
            #[track_caller]
            fn $method(self, mut rhs: Tensor<T, N>) -> Self::Output {
                const {
                    assert!(
                        M <= N,
                        "cannot broadcast higher-rank tensor into lower-rank tensor"
                    );
                }
                assert_eq!(
                    self.shape(),
                    rhs.shape()[N - M..],
                    "broadcast shape mismatch"
                );

                let lhs_slice = self.as_slice();
                rhs.view_mut().chunks::<M>().for_each(|chunk| {
                    for (&a, b) in zip(lhs_slice, chunk.into_mut_slice()) {
                        *b = a.$method(*b);
                    }
                });
                rhs
            }
        }
    };
}

impl_owned_binary_ops!(Add, add, AddAssign, add_assign);
impl_owned_binary_ops!(Sub, sub, SubAssign, sub_assign);
impl_owned_binary_ops!(Mul, mul, MulAssign, mul_assign);
impl_owned_binary_ops!(Div, div, DivAssign, div_assign);

#[cfg(feature = "alloc")]
impl<T: Copy + core::ops::Neg<Output = T>, const N: usize> core::ops::Neg for Tensor<T, N> {
    type Output = Tensor<T, N>;
    #[inline]
    fn neg(mut self) -> Self::Output {
        for x in self.as_mut_slice() {
            *x = -*x;
        }
        self
    }
}

// ============================================================================
// Display
// ============================================================================

fn fmt_nd<T: Display>(
    f: &mut Formatter<'_>,
    data: &[T],
    shape: &[usize],
    depth: usize,
) -> fmt::Result {
    match shape {
        [] => Display::fmt(&data[0], f),
        [len] => {
            write!(f, "[")?;
            for (i, x) in data[..*len].iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                Display::fmt(x, f)?;
            }
            write!(f, "]")
        }
        [outer, rest @ ..] => {
            let chunk = shape_len(rest);
            write!(f, "[")?;
            for i in 0..*outer {
                if i > 0 {
                    writeln!(f, ",")?;
                    for _ in 0..=depth {
                        write!(f, " ")?;
                    }
                }
                let start = i * chunk;
                fmt_nd(f, &data[start..start + chunk], rest, depth + 1)?;
            }
            write!(f, "]")
        }
    }
}

// ============================================================================
// Aliases & Rank-Specific Helpers
// ============================================================================

#[cfg(feature = "alloc")]
/// Owned 0-dimensional scalar tensor.
pub type Scalar<T> = Tensor<T, 0>;
/// Immutable 0-dimensional scalar view.
pub type ScalarView<'a, T> = TensorView<'a, T, 0>;
/// Mutable 0-dimensional scalar view.
pub type ScalarViewMut<'a, T> = TensorViewMut<'a, T, 0>;

#[cfg(feature = "alloc")]
/// Owned 1-dimensional vector tensor.
pub type Vector<T> = Tensor<T, 1>;
/// Immutable 1-dimensional vector view.
pub type VectorView<'a, T> = TensorView<'a, T, 1>;
/// Mutable 1-dimensional vector view.
pub type VectorViewMut<'a, T> = TensorViewMut<'a, T, 1>;

#[cfg(feature = "alloc")]
/// Owned 2-dimensional matrix tensor.
pub type Matrix<T> = Tensor<T, 2>;
/// Immutable 2-dimensional matrix view.
pub type MatrixView<'a, T> = TensorView<'a, T, 2>;
/// Mutable 2-dimensional matrix view.
pub type MatrixViewMut<'a, T> = TensorViewMut<'a, T, 2>;
