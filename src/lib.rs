#![cfg_attr(docsrs, feature(doc_cfg))]
#![no_std]
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

#[must_use]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor<T, const N: usize> {
    shape: [usize; N],
    data: Box<[T]>,
}

#[must_use]
#[derive(Debug, PartialEq, Eq)]
pub struct TensorView<'a, T, const N: usize> {
    shape: [usize; N],
    data: &'a [T],
}

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
    #[inline]
    pub fn from_slice(data: &'a [T], shape: [usize; N]) -> Self {
        validate_shape(data.len(), &shape);
        Self { shape, data }
    }

    #[inline]
    #[must_use]
    pub fn shape(&self) -> [usize; N] {
        self.shape
    }

    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &'a [T] {
        self.data
    }

    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    #[inline]
    pub fn iter(&self) -> Iter<'a, T> {
        self.data.iter()
    }

    #[inline]
    pub fn reshape<const M: usize>(self, new_shape: [usize; M]) -> TensorView<'a, T, M> {
        validate_shape(self.len(), &new_shape);
        TensorView {
            data: self.data,
            shape: new_shape,
        }
    }

    #[inline]
    pub fn flatten(self) -> TensorView<'a, T, 1> {
        TensorView {
            data: self.data,
            shape: [self.len()],
        }
    }

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
            data: unsafe { self.data.get_unchecked(r.start * inner..r.end * inner) },
            shape: new_shape,
        }
    }

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

    pub fn to_tensor(self) -> Tensor<T, N>
    where
        T: Clone,
    {
        Tensor::from_vec(self.data.to_vec(), self.shape)
    }
}

impl<'a, T> TensorView<'a, T, 0> {
    #[inline]
    pub fn scalar(val: &'a T) -> Self {
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
        unsafe { self.data.get_unchecked(off) }
    }
}

// ============================================================================
// TensorViewMut (Mutable)
// ============================================================================

impl<'a, T, const N: usize> TensorViewMut<'a, T, N> {
    #[inline]
    pub fn from_slice(data: &'a mut [T], shape: [usize; N]) -> Self {
        validate_shape(data.len(), &shape);
        Self { shape, data }
    }

    #[inline]
    #[must_use]
    pub fn shape(&self) -> [usize; N] {
        self.shape
    }

    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }

    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    #[inline]
    #[must_use]
    pub fn into_slice(self) -> &'a [T] {
        self.data
    }

    #[inline]
    #[must_use]
    pub fn into_mut_slice(self) -> &'a mut [T] {
        self.data
    }

    #[inline]
    pub fn view(&self) -> TensorView<'_, T, N> {
        TensorView {
            data: self.data,
            shape: self.shape,
        }
    }

    #[inline]
    pub fn into_view(self) -> TensorView<'a, T, N> {
        TensorView {
            data: self.data,
            shape: self.shape,
        }
    }

    #[inline]
    pub fn view_mut(&mut self) -> TensorViewMut<'_, T, N> {
        TensorViewMut {
            data: self.data,
            shape: self.shape,
        }
    }

    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        self.data.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        self.data.iter_mut()
    }

    #[inline]
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.data.fill(value);
    }

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

    #[inline]
    pub fn reshape<const M: usize>(self, new_shape: [usize; M]) -> TensorViewMut<'a, T, M> {
        validate_shape(self.len(), &new_shape);
        TensorViewMut {
            data: self.data,
            shape: new_shape,
        }
    }

    #[inline]
    pub fn flatten(self) -> TensorViewMut<'a, T, 1> {
        TensorViewMut {
            shape: [self.data.len()],
            data: self.data,
        }
    }

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
        let new_data = unsafe { self.data.get_unchecked_mut(start..end) };
        TensorViewMut {
            data: new_data,
            shape: new_shape,
        }
    }

    #[inline]
    pub fn chunks<const M: usize>(
        self,
    ) -> impl ExactSizeIterator<Item = TensorView<'a, T, M>> + DoubleEndedIterator + FusedIterator
    {
        self.into_view().chunks::<M>()
    }

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

    pub fn to_tensor(&self) -> Tensor<T, N>
    where
        T: Clone,
    {
        self.view().to_tensor()
    }
}

impl<'a, T> TensorViewMut<'a, T, 0> {
    #[inline]
    pub fn scalar(val: &'a mut T) -> Self {
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
        unsafe { self.data.get_unchecked(off) }
    }
}

impl<T, const N: usize> IndexMut<[usize; N]> for TensorViewMut<'_, T, N> {
    #[inline]
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        let off = linear_offset(&self.shape, &index);
        unsafe { self.data.get_unchecked_mut(off) }
    }
}

// ============================================================================
// Tensor (Owned)
// ============================================================================
impl<T, const N: usize> Tensor<T, N> {
    #[inline]
    pub fn from_vec(data: Vec<T>, shape: [usize; N]) -> Self {
        validate_shape(data.len(), &shape);
        Self {
            data: data.into_boxed_slice(),
            shape,
        }
    }

    #[inline]
    pub fn from_elem(val: T, shape: [usize; N]) -> Self
    where
        T: Clone,
    {
        Self::from_vec(vec![val; checked_shape_len(&shape)], shape)
    }

    #[inline]
    pub fn zeros(shape: [usize; N]) -> Self
    where
        T: Default + Clone,
    {
        Self::from_elem(T::default(), shape)
    }

    #[inline]
    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        self.data.into_vec()
    }

    #[inline]
    #[must_use]
    pub fn shape(&self) -> [usize; N] {
        self.shape
    }

    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    #[inline]
    pub fn view(&self) -> TensorView<'_, T, N> {
        TensorView {
            data: &self.data,
            shape: self.shape,
        }
    }

    #[inline]
    pub fn view_mut(&mut self) -> TensorViewMut<'_, T, N> {
        TensorViewMut {
            data: &mut self.data,
            shape: self.shape,
        }
    }

    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        self.data.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        self.data.iter_mut()
    }

    #[inline]
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.data.fill(value);
    }

    #[inline]
    pub fn assign<const M: usize>(&mut self, src: TensorView<'_, T, M>)
    where
        T: Copy,
    {
        self.view_mut().assign(src);
    }

    #[inline]
    pub fn reshape<const M: usize>(self, new_shape: [usize; M]) -> Tensor<T, M> {
        validate_shape(self.len(), &new_shape);
        Tensor {
            data: self.data,
            shape: new_shape,
        }
    }

    #[inline]
    pub fn slice<R: RangeBounds<usize>>(&self, range: R) -> TensorView<'_, T, N> {
        self.view().slice(range)
    }

    #[inline]
    pub fn slice_mut<R: RangeBounds<usize>>(&mut self, range: R) -> TensorViewMut<'_, T, N> {
        self.view_mut().slice(range)
    }

    #[inline]
    pub fn chunks<const M: usize>(
        &self,
    ) -> impl ExactSizeIterator<Item = TensorView<'_, T, M>> + DoubleEndedIterator + FusedIterator
    {
        self.view().chunks::<M>()
    }

    #[inline]
    pub fn chunks_mut<const M: usize>(
        &mut self,
    ) -> impl ExactSizeIterator<Item = TensorViewMut<'_, T, M>> + DoubleEndedIterator + FusedIterator
    {
        self.view_mut().chunks_mut::<M>()
    }
}
impl<T> Tensor<T, 0> {
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
        unsafe { self.data.get_unchecked(off) }
    }
}
impl<T, const N: usize> IndexMut<[usize; N]> for Tensor<T, N> {
    #[inline]
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        let off = linear_offset(&self.shape, &index);
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
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        fmt_nd(f, self.data, &self.shape, 0)
    }
}

impl<T: Display, const N: usize> Display for TensorViewMut<'_, T, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.view().fmt(f)
    }
}
impl<T: Display, const N: usize> Display for Tensor<T, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.view().fmt(f)
    }
}

// ============================================================================
// Aliases
// ============================================================================

pub type Scalar<T> = Tensor<T, 0>;
pub type ScalarView<'a, T> = TensorView<'a, T, 0>;
pub type ScalarViewMut<'a, T> = TensorViewMut<'a, T, 0>;

pub type Vector<T> = Tensor<T, 1>;
pub type VectorView<'a, T> = TensorView<'a, T, 1>;
pub type VectorViewMut<'a, T> = TensorViewMut<'a, T, 1>;

pub type Matrix<T> = Tensor<T, 2>;
pub type MatrixView<'a, T> = TensorView<'a, T, 2>;
pub type MatrixViewMut<'a, T> = TensorViewMut<'a, T, 2>;

impl<T> Matrix<T> {
    #[inline]
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.shape[0]
    }

    #[inline]
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.shape[1]
    }
}

impl<T> MatrixView<'_, T> {
    #[inline]
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.shape[0]
    }

    #[inline]
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.shape[1]
    }
}

impl<T> MatrixViewMut<'_, T> {
    #[inline]
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.shape[0]
    }

    #[inline]
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.shape[1]
    }
}
