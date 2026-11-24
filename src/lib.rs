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
    missing_docs,
    missing_debug_implementations,
    unreachable_pub,
    clippy::pedantic,
    clippy::undocumented_unsafe_blocks,
    clippy::multiple_unsafe_ops_per_block,
    clippy::as_ptr_cast_mut,
    clippy::float_cmp,
    clippy::imprecise_flops,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    clippy::missing_inline_in_public_items,
    clippy::dbg_macro,
    clippy::todo,
    clippy::unimplemented,
    clippy::std_instead_of_core,
    clippy::alloc_instead_of_core,
    clippy::std_instead_of_alloc,
    clippy::unnecessary_safety_comment,
    clippy::unnecessary_safety_doc
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
use core::num::NonZeroUsize;
use core::ops::{AddAssign, Bound, DivAssign, MulAssign, RangeBounds, SubAssign};

mod inner;
mod reduce;
#[cfg(feature = "alloc")]
mod tensor;
mod view;
mod view_mut;

pub mod linalg;

pub use inner::*;
pub use reduce::*;
pub use view::*;
pub use view_mut::*;

// ============================================================================
// Rank Aliases
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

// ============================================================================
// Arithmetic Operators
// ============================================================================

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
// Internal Helpers
// ============================================================================

// -- Shape & Indexing --------------------------------------------------------

#[inline]
const fn unpack_shape<const N: usize>(shape: [NonZeroUsize; N]) -> [usize; N] {
    let mut out = [0; N];
    let mut i = 0;
    while i < N {
        out[i] = shape[i].get();
        i += 1;
    }
    out
}

#[inline]
const unsafe fn unchecked_shape_len(shape: &[NonZeroUsize]) -> NonZeroUsize {
    let mut prod = 1usize;
    let mut i = 0;
    while i < shape.len() {
        // SAFETY: Caller guarantees the product of `shape` does not overflow `usize`.
        prod = unsafe { prod.unchecked_mul(shape[i].get()) };
        i += 1;
    }
    // SAFETY: Every element in `shape` is non-zero and the product did not overflow `usize`.
    unsafe { NonZeroUsize::new_unchecked(prod) }
}

#[inline]
#[track_caller]
const fn check_shape<const N: usize>(shape: [usize; N]) -> ([NonZeroUsize; N], usize) {
    let mut out = [NonZeroUsize::MIN; N];
    let mut prod = 1usize;
    let mut i = 0;
    while i < N {
        let Some(dim) = NonZeroUsize::new(shape[i]) else {
            panic!("tensor dimensions must be non-zero");
        };
        let Some(next) = prod.checked_mul(dim.get()) else {
            panic!("shape overflow");
        };
        out[i] = dim;
        prod = next;
        i += 1;
    }
    (out, prod)
}

#[inline]
#[track_caller]
const fn validate_shape<const N: usize>(buffer_len: usize, shape: [usize; N]) -> [NonZeroUsize; N] {
    let (out, prod) = check_shape(shape);
    assert!(buffer_len == prod, "buffer len does not match shape");
    out
}

#[inline]
#[track_caller]
fn linear_offset<const N: usize>(shape: &[NonZeroUsize; N], index: &[usize; N]) -> usize {
    zip(shape, index).fold(0, |offset, (&len, &idx)| {
        assert!(
            idx < len.get(),
            "index {idx} out of bounds for axis len {len}"
        );
        offset * len.get() + idx
    })
}

#[inline]
const fn linear_offset_checked<const N: usize>(
    shape: &[NonZeroUsize; N],
    index: &[usize; N],
) -> Option<usize> {
    let mut offset = 0;
    let mut i = 0;
    while i < N {
        let len = shape[i].get();
        let idx = index[i];
        if idx >= len {
            return None;
        }
        offset = offset * len + idx;
        i += 1;
    }
    Some(offset)
}

#[inline]
#[track_caller]
fn resolve_range<R: RangeBounds<usize>>(range: R, axis_len: NonZeroUsize) -> (usize, NonZeroUsize) {
    let axis_len = axis_len.get();
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
    assert!(
        end <= axis_len,
        "slice end {end} exceeds axis len {axis_len}"
    );
    let len = end
        .checked_sub(start)
        .and_then(NonZeroUsize::new)
        .unwrap_or_else(|| panic!("slice range {start}..{end} must be non-empty"));
    (start, len)
}

// -- Formatting --------------------------------------------------------------

fn fmt_nd<T: Display>(
    f: &mut Formatter<'_>,
    data: &[T],
    shape: &[NonZeroUsize],
    depth: usize,
) -> fmt::Result {
    match shape {
        [] => Display::fmt(&data[0], f),
        [len] => {
            write!(f, "[")?;
            for (i, x) in data[..len.get()].iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                Display::fmt(x, f)?;
            }
            write!(f, "]")
        }
        [outer, rest @ ..] => {
            // SAFETY: `shape` comes from a valid tensor, so any sub-slice product fits in `usize`.
            let chunk = unsafe { unchecked_shape_len(rest).get() };
            write!(f, "[")?;
            for i in 0..outer.get() {
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
