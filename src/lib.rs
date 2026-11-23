//! A minimal, `#![no_std]` row-major dense tensor library with zero-copy views
//! and pluggable Level-1/2/3 linear algebra backends.
//!
//! # Core Types
//! - [`Tensor<T, N>`]: Heap-allocated (`alloc::boxed::Box<[T]>`) row-major
//!   owned tensor of rank `N`.
//! - [`TensorView<'a, T, N>`]: Immutable, `Copy`, zero-allocation borrow over a
//!   contiguous row-major slice.
//! - [`TensorViewMut<'a, T, N>`]: Exclusive mutable borrow over a contiguous
//!   row-major slice.
//!
//! Convenience rank aliases are provided for rank-0 ([`Scalar`]), rank-1
//! ([`Vector`]), and rank-2 ([`Matrix`]) tensors and their view variants.

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
    // clippy::allow_attributes,
    // clippy::allow_attributes_without_reason,
    clippy::dbg_macro,
    clippy::todo,
    clippy::unimplemented
)]
#![allow(
    clippy::len_without_is_empty,
    clippy::many_single_char_names,
    clippy::needless_pass_by_value
)]

extern crate alloc;
use alloc::{boxed::Box, vec, vec::Vec};

use core::array;
use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{DoubleEndedIterator, ExactSizeIterator, FusedIterator, zip};
use core::ops::{
    AddAssign, Bound, DivAssign, Index, IndexMut, MulAssign, Range, RangeBounds, SubAssign,
};
use core::slice::{self, Iter, IterMut};

pub mod linalg;

// ============================================================================
// Core Types
// ============================================================================

/// An owned, heap-allocated, row-major dense tensor of rank `N`.
///
/// All dimensions must be strictly non-zero (`> 0`). Rank-0 scalars are
/// represented as `Tensor<T, 0>` with shape `[]` and a single element.
#[must_use]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor<T, const N: usize> {
    shape: [usize; N],
    data: Box<[T]>,
}

/// An immutable, zero-copy view over a contiguous row-major tensor of rank `N`.
///
/// `TensorView` implements [`Copy`] and [`Clone`] regardless of whether `T`
/// implements them.
#[must_use]
#[derive(Debug, PartialEq, Eq)]
pub struct TensorView<'a, T, const N: usize> {
    shape: [usize; N],
    data: &'a [T],
}

/// A mutable, zero-copy view over a contiguous row-major tensor of rank `N`.
#[must_use]
#[derive(Debug, PartialEq, Eq)]
pub struct TensorViewMut<'a, T, const N: usize> {
    shape: [usize; N],
    data: &'a mut [T],
}

impl<T, const N: usize> Clone for TensorView<'_, T, N> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, const N: usize> Copy for TensorView<'_, T, N> {}

// ============================================================================
// Internal Helpers
// ============================================================================

#[inline]
fn shape_len(shape: &[usize]) -> usize {
    shape.iter().product()
}

#[inline]
fn checked_shape_len(shape: &[usize]) -> usize {
    shape
        .iter()
        .fold(1, |acc, &x| acc.checked_mul(x).expect("shape overflow"))
}

#[inline]
fn validate_shape(buffer_len: usize, shape: &[usize]) {
    assert!(
        !shape.contains(&0),
        "tensor dimensions must be non-zero, got {shape:?}",
    );
    assert_eq!(
        buffer_len,
        checked_shape_len(shape),
        "buffer len ({buffer_len}) does not match shape {shape:?}",
    );
}

#[inline]
fn linear_offset<const N: usize>(shape: &[usize; N], index: &[usize; N]) -> usize {
    zip(shape, index).fold(0, |offset, (&len, &idx)| {
        assert!(idx < len, "index {idx} out of bounds for axis len {len}");
        offset * len + idx
    })
}

#[inline]
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

// ============================================================================
// TensorView (Immutable)
// ============================================================================

impl<'a, T, const N: usize> TensorView<'a, T, N> {
    /// Wraps an existing row-major slice in an immutable [`TensorView`].
    ///
    /// # Panics
    ///
    /// Panics if any dimension in `shape` is zero, if the product of `shape`
    /// overflows `usize`, or if `data.len()` does not equal the product of
    /// `shape`.
    #[inline]
    pub fn from_slice(data: &'a [T], shape: [usize; N]) -> Self {
        validate_shape(data.len(), &shape);
        Self { shape, data }
    }

    /// Returns the dimensions of the tensor view.
    #[inline]
    #[must_use]
    pub const fn shape(&self) -> [usize; N] {
        self.shape
    }

    /// Returns the total number of elements in the tensor view.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns the underlying row-major data as a flat slice.
    #[inline]
    #[must_use]
    pub const fn as_slice(&self) -> &'a [T] {
        self.data
    }

    /// Returns a raw const pointer to the first element in the view.
    #[inline]
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Returns an iterator over all elements in row-major order.
    #[inline]
    pub fn iter(&self) -> Iter<'a, T> {
        self.data.iter()
    }

    /// Reinterprets the view with a new rank `M` and `new_shape` without
    /// copying data.
    ///
    /// # Panics
    ///
    /// Panics if any dimension in `new_shape` is zero, if its product overflows
    /// `usize`, or if the element count differs from `self.len()`.
    #[inline]
    pub fn reshape<const M: usize>(self, new_shape: [usize; M]) -> TensorView<'a, T, M> {
        validate_shape(self.len(), &new_shape);
        TensorView {
            data: self.data,
            shape: new_shape,
        }
    }

    /// Flattens the view into a 1D [`VectorView`] of length `self.len()`.
    #[inline]
    pub const fn flatten(self) -> TensorView<'a, T, 1> {
        TensorView {
            data: self.data,
            shape: [self.len()],
        }
    }

    /// Slices the outermost axis (`axis 0`) by `range`, returning a sub-view
    /// with the same rank `N`.
    ///
    /// # Panics
    ///
    /// Panics at compile time if `N == 0`, or at runtime if `range` resolves
    /// to an empty range or exceeds `self.shape()[0]`.
    #[inline]
    pub fn slice<R: RangeBounds<usize>>(self, range: R) -> TensorView<'a, T, N> {
        const {
            assert!(N >= 1, "cannot slice a 0D scalar tensor");
        }
        let r = resolve_range(range, self.shape[0]);
        let inner = shape_len(&self.shape[1..]);

        let mut new_shape = self.shape;
        new_shape[0] = r.end - r.start;

        TensorView {
            // SAFETY: `r.end <= self.shape[0]` and `self.data.len() == self.shape[0] * inner`,
            // so `r.start * inner .. r.end * inner` is strictly within `self.data`.
            data: unsafe { self.data.get_unchecked(r.start * inner..r.end * inner) },
            shape: new_shape,
        }
    }

    /// Partitions the tensor along its leading axes into contiguous sub-views
    /// of rank `M` (spanning the trailing `M` dimensions).
    ///
    /// # Panics
    ///
    /// Panics at compile time if `M > N`.
    #[inline]
    pub fn chunks<const M: usize>(
        self,
    ) -> impl ExactSizeIterator<Item = TensorView<'a, T, M>> + DoubleEndedIterator + FusedIterator
    {
        const {
            assert!(M <= N, "chunk rank M cannot exceed tensor rank N");
        }
        let sub_shape = array::from_fn(|i| self.shape[N - M + i]);
        let chunk_len = shape_len(&sub_shape);
        self.data
            .chunks_exact(chunk_len)
            .map(move |chunk| TensorView {
                data: chunk,
                shape: sub_shape,
            })
    }

    /// Clones the elements of this view into an owned [`Tensor<T, N>`].
    #[inline]
    pub fn to_tensor(self) -> Tensor<T, N>
    where
        T: Clone,
    {
        Tensor::from_vec(self.data.to_vec(), self.shape)
    }
}

impl<'a, T> TensorView<'a, T, 0> {
    /// Creates a 0-dimensional [`ScalarView`] borrowing a single reference.
    #[inline]
    pub const fn scalar(val: &'a T) -> Self {
        Self {
            data: slice::from_ref(val),
            shape: [],
        }
    }
}

impl<T, const N: usize> Index<[usize; N]> for TensorView<'_, T, N> {
    type Output = T;

    #[inline]
    fn index(&self, index: [usize; N]) -> &Self::Output {
        let off = linear_offset(&self.shape, &index);
        // SAFETY: `linear_offset` bounds-checks every coordinate against `self.shape`,
        // and `self.data.len()` equals the product of `self.shape`.
        unsafe { self.data.get_unchecked(off) }
    }
}

// ============================================================================
// TensorViewMut (Mutable)
// ============================================================================

impl<'a, T, const N: usize> TensorViewMut<'a, T, N> {
    /// Wraps an existing mutable row-major slice in a [`TensorViewMut`].
    ///
    /// # Panics
    ///
    /// Panics if any dimension in `shape` is zero, if the product of `shape`
    /// overflows `usize`, or if `data.len()` does not equal the product of
    /// `shape`.
    #[inline]
    pub fn from_slice(data: &'a mut [T], shape: [usize; N]) -> Self {
        validate_shape(data.len(), &shape);
        Self { shape, data }
    }

    /// Returns the dimensions of the tensor view.
    #[inline]
    #[must_use]
    pub const fn shape(&self) -> [usize; N] {
        self.shape
    }

    /// Returns the total number of elements in the tensor view.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns the underlying row-major data as an immutable flat slice.
    #[inline]
    #[must_use]
    pub const fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Returns a raw const pointer to the first element in the view.
    #[inline]
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Returns the underlying row-major data as a mutable flat slice.
    #[inline]
    #[must_use]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }

    /// Returns a raw mutable pointer to the first element in the view.
    #[inline]
    #[must_use]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    /// Consumes the view and returns the underlying immutable slice with
    /// lifetime `'a`.
    #[inline]
    #[must_use]
    pub const fn into_slice(self) -> &'a [T] {
        self.data
    }

    /// Consumes the view and returns the underlying mutable slice with
    /// lifetime `'a`.
    #[inline]
    #[must_use]
    pub const fn into_mut_slice(self) -> &'a mut [T] {
        self.data
    }

    /// Reborrows this mutable view as an immutable [`TensorView`].
    #[inline]
    pub const fn view(&self) -> TensorView<'_, T, N> {
        TensorView {
            data: self.data,
            shape: self.shape,
        }
    }

    /// Consumes this mutable view and converts it into an immutable
    /// [`TensorView`] with lifetime `'a`.
    #[inline]
    pub const fn into_view(self) -> TensorView<'a, T, N> {
        TensorView {
            data: self.data,
            shape: self.shape,
        }
    }

    /// Reborrows this mutable view with a shorter lifetime.
    #[inline]
    pub const fn view_mut(&mut self) -> TensorViewMut<'_, T, N> {
        TensorViewMut {
            data: self.data,
            shape: self.shape,
        }
    }

    /// Returns an iterator over immutable element references in row-major
    /// order.
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        self.data.iter()
    }

    /// Returns an iterator over mutable element references in row-major order.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        self.data.iter_mut()
    }

    /// Fills every element in the view with clones of `value`.
    #[inline]
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.data.fill(value);
    }

    /// Copies `src` into every trailing rank-`M` slice of `self` (NumPy-style
    /// right-aligned broadcasting).
    ///
    /// # Panics
    ///
    /// Panics at compile time if `M > N`, or at runtime if `src.shape()` does
    /// not match `&self.shape()[N - M..]`.
    #[inline]
    pub fn assign<const M: usize>(&mut self, src: TensorView<'_, T, M>)
    where
        T: Copy,
    {
        const {
            assert!(
                M <= N,
                "cannot broadcast higher-rank tensor into lower-rank tensor"
            );
        }
        let tail = &self.shape[N - M..];
        assert_eq!(tail, src.shape(), "assign shape mismatch");
        let src_slice = src.as_slice();
        for mut chunk in self.view_mut().chunks_mut::<M>() {
            chunk.as_mut_slice().copy_from_slice(src_slice);
        }
    }

    /// Consumes and reinterprets the mutable view with a new rank `M` and
    /// `new_shape` without copying data.
    ///
    /// # Panics
    ///
    /// Panics if any dimension in `new_shape` is zero, if its product overflows
    /// `usize`, or if the element count differs from `self.len()`.
    #[inline]
    pub fn reshape<const M: usize>(self, new_shape: [usize; M]) -> TensorViewMut<'a, T, M> {
        validate_shape(self.len(), &new_shape);
        TensorViewMut {
            data: self.data,
            shape: new_shape,
        }
    }

    /// Consumes and flattens the view into a 1D [`VectorViewMut`] of length
    /// `self.len()`.
    #[inline]
    pub const fn flatten(self) -> TensorViewMut<'a, T, 1> {
        TensorViewMut {
            shape: [self.data.len()],
            data: self.data,
        }
    }

    /// Consumes and slices the outermost axis (`axis 0`) by `range`, returning
    /// a mutable sub-view with the same rank `N`.
    ///
    /// # Panics
    ///
    /// Panics at compile time if `N == 0`, or at runtime if `range` resolves
    /// to an empty range or exceeds `self.shape()[0]`.
    #[inline]
    pub fn slice<R: RangeBounds<usize>>(self, range: R) -> TensorViewMut<'a, T, N> {
        const {
            assert!(N >= 1, "cannot slice a 0D scalar tensor");
        }
        let r = resolve_range(range, self.shape[0]);
        let inner = shape_len(&self.shape[1..]);
        let mut new_shape = self.shape;
        new_shape[0] = r.end - r.start;

        let start = r.start * inner;
        let end = r.end * inner;
        // SAFETY: `r.end <= self.shape[0]` and `self.data.len() == self.shape[0] * inner`,
        // so `start..end` is strictly within `self.data`.
        let new_data = unsafe { self.data.get_unchecked_mut(start..end) };
        TensorViewMut {
            data: new_data,
            shape: new_shape,
        }
    }

    /// Consumes and partitions the tensor into immutable sub-views of rank `M`
    /// spanning the trailing `M` dimensions.
    ///
    /// # Panics
    ///
    /// Panics at compile time if `M > N`.
    #[inline]
    pub fn chunks<const M: usize>(
        self,
    ) -> impl ExactSizeIterator<Item = TensorView<'a, T, M>> + DoubleEndedIterator + FusedIterator
    {
        self.into_view().chunks::<M>()
    }

    /// Consumes and partitions the tensor into mutable sub-views of rank `M`
    /// spanning the trailing `M` dimensions.
    ///
    /// # Panics
    ///
    /// Panics at compile time if `M > N`.
    #[inline]
    pub fn chunks_mut<const M: usize>(
        self,
    ) -> impl ExactSizeIterator<Item = TensorViewMut<'a, T, M>> + DoubleEndedIterator + FusedIterator
    {
        const {
            assert!(M <= N, "chunk rank M cannot exceed tensor rank N");
        }
        let sub_shape = array::from_fn(|i| self.shape[N - M + i]);
        let chunk_len = shape_len(&sub_shape);
        self.data
            .chunks_exact_mut(chunk_len)
            .map(move |chunk| TensorViewMut {
                data: chunk,
                shape: sub_shape,
            })
    }

    /// Clones the elements of this view into an owned [`Tensor<T, N>`].
    #[inline]
    pub fn to_tensor(&self) -> Tensor<T, N>
    where
        T: Clone,
    {
        self.view().to_tensor()
    }
}

impl<'a, T> TensorViewMut<'a, T, 0> {
    /// Creates a 0-dimensional [`ScalarViewMut`] borrowing a single mutable
    /// reference.
    #[inline]
    pub const fn scalar(val: &'a mut T) -> Self {
        Self {
            data: slice::from_mut(val),
            shape: [],
        }
    }
}

impl<T, const N: usize> Index<[usize; N]> for TensorViewMut<'_, T, N> {
    type Output = T;

    #[inline]
    fn index(&self, index: [usize; N]) -> &Self::Output {
        let off = linear_offset(&self.shape, &index);
        // SAFETY: `linear_offset` bounds-checks every coordinate against `self.shape`.
        unsafe { self.data.get_unchecked(off) }
    }
}

impl<T, const N: usize> IndexMut<[usize; N]> for TensorViewMut<'_, T, N> {
    #[inline]
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        let off = linear_offset(&self.shape, &index);
        // SAFETY: `linear_offset` bounds-checks every coordinate against `self.shape`.
        unsafe { self.data.get_unchecked_mut(off) }
    }
}

// ============================================================================
// Tensor (Owned)
// ============================================================================

impl<T, const N: usize> Tensor<T, N> {
    /// Constructs an owned [`Tensor`] from a flat row-major [`Vec`] and
    /// `shape`.
    ///
    /// # Panics
    ///
    /// Panics if any dimension in `shape` is zero, if the product of `shape`
    /// overflows `usize`, or if `data.len()` does not equal the product of
    /// `shape`.
    #[inline]
    pub fn from_vec(data: Vec<T>, shape: [usize; N]) -> Self {
        validate_shape(data.len(), &shape);
        Self {
            data: data.into_boxed_slice(),
            shape,
        }
    }

    /// Allocates a [`Tensor`] of `shape` filled with clones of `val`.
    ///
    /// # Panics
    ///
    /// Panics if any dimension in `shape` is zero or if its product overflows
    /// `usize`.
    #[inline]
    pub fn from_elem(val: T, shape: [usize; N]) -> Self
    where
        T: Clone,
    {
        Self::from_vec(vec![val; checked_shape_len(&shape)], shape)
    }

    /// Allocates a [`Tensor`] of `shape` filled with `T::default()`.
    ///
    /// # Panics
    ///
    /// Panics if any dimension in `shape` is zero or if its product overflows
    /// `usize`.
    #[inline]
    pub fn zeros(shape: [usize; N]) -> Self
    where
        T: Default + Clone,
    {
        Self::from_elem(T::default(), shape)
    }

    /// Consumes the tensor and returns its underlying flat [`Vec`].
    #[inline]
    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        self.data.into_vec()
    }

    /// Returns the dimensions of the tensor.
    #[inline]
    #[must_use]
    pub const fn shape(&self) -> [usize; N] {
        self.shape
    }

    /// Returns the total number of elements in the tensor.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns the underlying row-major data as an immutable flat slice.
    #[inline]
    #[must_use]
    pub const fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns a raw const pointer to the first element in the tensor.
    #[inline]
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Returns the underlying row-major data as a mutable flat slice.
    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Returns a raw mutable pointer to the first element in the tensor.
    #[inline]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    /// Borrows the tensor as an immutable [`TensorView`].
    #[inline]
    pub const fn view(&self) -> TensorView<'_, T, N> {
        TensorView {
            data: &self.data,
            shape: self.shape,
        }
    }

    /// Borrows the tensor as a mutable [`TensorViewMut`].
    #[inline]
    pub const fn view_mut(&mut self) -> TensorViewMut<'_, T, N> {
        TensorViewMut {
            data: &mut self.data,
            shape: self.shape,
        }
    }

    /// Returns an iterator over immutable element references in row-major
    /// order.
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        self.data.iter()
    }

    /// Returns an iterator over mutable element references in row-major order.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        self.data.iter_mut()
    }

    /// Fills every element in the tensor with clones of `value`.
    #[inline]
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.data.fill(value);
    }

    /// Copies `src` into every trailing rank-`M` slice of `self` (NumPy-style
    /// right-aligned broadcasting).
    ///
    /// # Panics
    ///
    /// Panics at compile time if `M > N`, or at runtime if `src.shape()` does
    /// not match `&self.shape()[N - M..]`.
    #[inline]
    pub fn assign<const M: usize>(&mut self, src: TensorView<'_, T, M>)
    where
        T: Copy,
    {
        self.view_mut().assign(src);
    }

    /// Consumes and reshapes the owned tensor to rank `M` and `new_shape`
    /// without reallocating.
    ///
    /// # Panics
    ///
    /// Panics if any dimension in `new_shape` is zero, if its product overflows
    /// `usize`, or if the element count differs from `self.len()`.
    #[inline]
    pub fn reshape<const M: usize>(self, new_shape: [usize; M]) -> Tensor<T, M> {
        validate_shape(self.len(), &new_shape);
        Tensor {
            data: self.data,
            shape: new_shape,
        }
    }

    /// Borrows an immutable sub-view sliced along `axis 0` by `range`.
    ///
    /// # Panics
    ///
    /// Panics at compile time if `N == 0`, or at runtime if `range` resolves
    /// to an empty range or exceeds `self.shape()[0]`.
    #[inline]
    pub fn slice<R: RangeBounds<usize>>(&self, range: R) -> TensorView<'_, T, N> {
        self.view().slice(range)
    }

    /// Borrows a mutable sub-view sliced along `axis 0` by `range`.
    ///
    /// # Panics
    ///
    /// Panics at compile time if `N == 0`, or at runtime if `range` resolves
    /// to an empty range or exceeds `self.shape()[0]`.
    #[inline]
    pub fn slice_mut<R: RangeBounds<usize>>(&mut self, range: R) -> TensorViewMut<'_, T, N> {
        self.view_mut().slice(range)
    }

    /// Returns an iterator over immutable sub-views of rank `M` spanning the
    /// trailing `M` dimensions.
    ///
    /// # Panics
    ///
    /// Panics at compile time if `M > N`.
    #[inline]
    pub fn chunks<const M: usize>(
        &self,
    ) -> impl ExactSizeIterator<Item = TensorView<'_, T, M>> + DoubleEndedIterator + FusedIterator
    {
        self.view().chunks::<M>()
    }

    /// Returns an iterator over mutable sub-views of rank `M` spanning the
    /// trailing `M` dimensions.
    ///
    /// # Panics
    ///
    /// Panics at compile time if `M > N`.
    #[inline]
    pub fn chunks_mut<const M: usize>(
        &mut self,
    ) -> impl ExactSizeIterator<Item = TensorViewMut<'_, T, M>> + DoubleEndedIterator + FusedIterator
    {
        self.view_mut().chunks_mut::<M>()
    }
}

impl<T> Tensor<T, 0> {
    /// Creates an owned 0-dimensional [`Scalar`] tensor containing `val`.
    #[inline]
    pub fn scalar(val: T) -> Self {
        Self::from_vec(vec![val], [])
    }
}

impl<T, const N: usize> Index<[usize; N]> for Tensor<T, N> {
    type Output = T;

    #[inline]
    fn index(&self, index: [usize; N]) -> &Self::Output {
        let off = linear_offset(&self.shape, &index);
        // SAFETY: `linear_offset` bounds-checks every coordinate against `self.shape`.
        unsafe { self.data.get_unchecked(off) }
    }
}

impl<T, const N: usize> IndexMut<[usize; N]> for Tensor<T, N> {
    #[inline]
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        let off = linear_offset(&self.shape, &index);
        // SAFETY: `linear_offset` bounds-checks every coordinate against `self.shape`.
        unsafe { self.data.get_unchecked_mut(off) }
    }
}

// ============================================================================
// IntoIterator
// ============================================================================

impl<'a, T, const N: usize> IntoIterator for TensorView<'a, T, N> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &TensorView<'a, T, N> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for TensorViewMut<'a, T, N> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a TensorViewMut<'_, T, N> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut TensorViewMut<'_, T, N> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a Tensor<T, N> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut Tensor<T, N> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl<T, const N: usize> IntoIterator for Tensor<T, N> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}

// ============================================================================
// Compound Assignment (Broadcasting Views & Scalars)
// ============================================================================

macro_rules! impl_assign_ops {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T: Copy + $trait<T>, const N: usize, const M: usize> $trait<TensorView<'_, T, M>>
            for TensorViewMut<'_, T, N>
        {
            #[inline]
            fn $method(&mut self, rhs: TensorView<'_, T, M>) {
                const {
                    assert!(
                        M <= N,
                        "cannot broadcast higher-rank tensor into lower-rank tensor"
                    );
                }
                let tail = &self.shape[N - M..];
                assert_eq!(tail, &rhs.shape()[..], "broadcast shape mismatch");
                let rhs_slice = rhs.as_slice();
                for mut chunk in self.view_mut().chunks_mut::<M>() {
                    for (a, &b) in zip(chunk.as_mut_slice(), rhs_slice) {
                        *a $op b;
                    }
                }
            }
        }

        impl<T: Copy + $trait<T>, const N: usize, const M: usize> $trait<&Tensor<T, M>>
            for TensorViewMut<'_, T, N>
        {
            #[inline]
            fn $method(&mut self, rhs: &Tensor<T, M>) {
                self.$method(rhs.view());
            }
        }

        impl<T: Copy + $trait<T>, const N: usize, const M: usize> $trait<&TensorViewMut<'_, T, M>>
            for TensorViewMut<'_, T, N>
        {
            #[inline]
            fn $method(&mut self, rhs: &TensorViewMut<'_, T, M>) {
                self.$method(rhs.view());
            }
        }

        impl<T: Copy + $trait<T>, const N: usize> $trait<T> for TensorViewMut<'_, T, N> {
            #[inline]
            fn $method(&mut self, rhs: T) {
                for a in self.as_mut_slice() {
                    *a $op rhs;
                }
            }
        }

        impl<T: Copy + $trait<T>, const N: usize, const M: usize> $trait<TensorView<'_, T, M>>
            for Tensor<T, N>
        {
            #[inline]
            fn $method(&mut self, rhs: TensorView<'_, T, M>) {
                self.view_mut().$method(rhs);
            }
        }

        impl<T: Copy + $trait<T>, const N: usize, const M: usize> $trait<&Tensor<T, M>>
            for Tensor<T, N>
        {
            #[inline]
            fn $method(&mut self, rhs: &Tensor<T, M>) {
                self.$method(rhs.view());
            }
        }

        impl<T: Copy + $trait<T>, const N: usize, const M: usize> $trait<&TensorViewMut<'_, T, M>>
            for Tensor<T, N>
        {
            #[inline]
            fn $method(&mut self, rhs: &TensorViewMut<'_, T, M>) {
                self.$method(rhs.view());
            }
        }

        impl<T: Copy + $trait<T>, const N: usize> $trait<T> for Tensor<T, N> {
            #[inline]
            fn $method(&mut self, rhs: T) {
                self.view_mut().$method(rhs);
            }
        }
    };
}

impl_assign_ops!(AddAssign, add_assign, +=);
impl_assign_ops!(SubAssign, sub_assign, -=);
impl_assign_ops!(MulAssign, mul_assign, *=);
impl_assign_ops!(DivAssign, div_assign, /=);

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

impl<T: Display, const N: usize> Display for TensorView<'_, T, N> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        fmt_nd(f, self.data, &self.shape, 0)
    }
}

impl<T: Display, const N: usize> Display for TensorViewMut<'_, T, N> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.view().fmt(f)
    }
}

impl<T: Display, const N: usize> Display for Tensor<T, N> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.view().fmt(f)
    }
}

// ============================================================================
// Aliases
// ============================================================================

/// Owned 0-dimensional scalar tensor.
pub type Scalar<T> = Tensor<T, 0>;
/// Immutable 0-dimensional scalar view.
pub type ScalarView<'a, T> = TensorView<'a, T, 0>;
/// Mutable 0-dimensional scalar view.
pub type ScalarViewMut<'a, T> = TensorViewMut<'a, T, 0>;

/// Owned 1-dimensional vector tensor.
pub type Vector<T> = Tensor<T, 1>;
/// Immutable 1-dimensional vector view.
pub type VectorView<'a, T> = TensorView<'a, T, 1>;
/// Mutable 1-dimensional vector view.
pub type VectorViewMut<'a, T> = TensorViewMut<'a, T, 1>;

/// Owned 2-dimensional matrix tensor.
pub type Matrix<T> = Tensor<T, 2>;
/// Immutable 2-dimensional matrix view.
pub type MatrixView<'a, T> = TensorView<'a, T, 2>;
/// Mutable 2-dimensional matrix view.
pub type MatrixViewMut<'a, T> = TensorViewMut<'a, T, 2>;

impl<T> Matrix<T> {
    /// Returns the number of rows (`shape()[0]`) in the matrix.
    #[inline]
    #[must_use]
    pub const fn nrows(&self) -> usize {
        self.shape[0]
    }

    /// Returns the number of columns (`shape()[1]`) in the matrix.
    #[inline]
    #[must_use]
    pub const fn ncols(&self) -> usize {
        self.shape[1]
    }
}

impl<T> MatrixView<'_, T> {
    /// Returns the number of rows (`shape()[0]`) in the matrix view.
    #[inline]
    #[must_use]
    pub const fn nrows(&self) -> usize {
        self.shape[0]
    }

    /// Returns the number of columns (`shape()[1]`) in the matrix view.
    #[inline]
    #[must_use]
    pub const fn ncols(&self) -> usize {
        self.shape[1]
    }
}

impl<T> MatrixViewMut<'_, T> {
    /// Returns the number of rows (`shape()[0]`) in the mutable matrix view.
    #[inline]
    #[must_use]
    pub const fn nrows(&self) -> usize {
        self.shape[0]
    }

    /// Returns the number of columns (`shape()[1]`) in the mutable matrix view.
    #[inline]
    #[must_use]
    pub const fn ncols(&self) -> usize {
        self.shape[1]
    }
}
