use core::fmt;
use core::fmt::{Display, Formatter};
use core::ops::{Index, IndexMut};
use core::slice::{Iter, IterMut};

use alloc::boxed::Box;
use alloc::{vec, vec::Vec};

use crate::{
    Matrix, Scalar, TensorView, TensorViewMut, Vector, checked_shape_len, linear_offset,
    linear_offset_checked, validate_shape,
};

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

    /// Constructs an owned [`Tensor`] from a flat row-major [`Box<[T]>`](Box) and
    /// `shape`.
    ///
    /// # Panics
    ///
    /// Panics if any dimension in `shape` is zero, if the product of `shape`
    /// overflows `usize`, or if `data.len()` does not equal the product of
    /// `shape`.
    #[inline]
    pub fn from_boxed_slice(data: Box<[T]>, shape: [usize; N]) -> Self {
        validate_shape(data.len(), &shape);
        Self { shape, data }
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

    /// Consumes the tensor and returns its underlying flat [`Box<[T]>`](Box).
    #[inline]
    #[must_use]
    pub fn into_boxed_slice(self) -> Box<[T]> {
        self.data
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
    #[must_use]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Returns a raw mutable pointer to the first element in the tensor.
    #[inline]
    #[must_use]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    /// Borrows the tensor as an immutable [`TensorView`].
    #[inline]
    pub const fn view(&self) -> TensorView<'_, T, N> {
        TensorView::new(self.shape, &self.data)
    }

    /// Borrows the tensor as a mutable [`TensorViewMut`].
    #[inline]
    pub const fn view_mut(&mut self) -> TensorViewMut<'_, T, N> {
        TensorViewMut::new(self.shape, &mut self.data)
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

    /// Consumes and flattens the tensor into a 1D [`Vector`] of length
    /// `self.len()`.
    #[inline]
    pub fn flatten(self) -> Vector<T> {
        Tensor {
            shape: [self.len()],
            data: self.data,
        }
    }

    /// Returns a reference to the element at `index`, or `None` if any
    /// coordinate is out of bounds.
    #[inline]
    #[must_use]
    pub fn get(&self, index: [usize; N]) -> Option<&T> {
        let off = linear_offset_checked(&self.shape, &index)?;
        // SAFETY: `linear_offset_checked` bounds-checks every coordinate against `self.shape`,
        // and `self.data.len()` equals the product of `self.shape`.
        Some(unsafe { self.data.get_unchecked(off) })
    }

    /// Returns a mutable reference to the element at `index`, or `None` if any
    /// coordinate is out of bounds.
    #[inline]
    #[must_use]
    pub fn get_mut(&mut self, index: [usize; N]) -> Option<&mut T> {
        let off = linear_offset_checked(&self.shape, &index)?;
        // SAFETY: `linear_offset_checked` bounds-checks every coordinate against `self.shape`,
        // and `self.data.len()` equals the product of `self.shape`.
        Some(unsafe { self.data.get_unchecked_mut(off) })
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

impl<T: Display, const N: usize> Display for Tensor<T, N> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.view().fmt(f)
    }
}

impl<T> Scalar<T> {
    /// Creates an owned 0-dimensional [`Scalar`] tensor containing `val`.
    #[inline]
    pub fn scalar(val: T) -> Self {
        Self::from_boxed_slice(Box::new([val]), [])
    }

    /// Consumes the scalar tensor and returns its single inner element.
    #[inline]
    #[must_use]
    pub fn into_scalar(self) -> T {
        // SAFETY: A rank-0 `Scalar` has shape `[]`, which guarantees `self.data.len() == 1`.
        unsafe { self.data.into_vec().pop().unwrap_unchecked() }
    }
}

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

impl<T> From<Vec<T>> for Vector<T> {
    #[inline]
    fn from(value: Vec<T>) -> Self {
        let len = value.len();
        Self::from_vec(value, [len])
    }
}

impl<T, const N: usize> From<Tensor<T, N>> for Vec<T> {
    #[inline]
    fn from(value: Tensor<T, N>) -> Self {
        value.into_vec()
    }
}

impl<'a, T, const N: usize> From<&'a Tensor<T, N>> for TensorView<'a, T, N> {
    #[inline]
    fn from(value: &'a Tensor<T, N>) -> Self {
        value.view()
    }
}

impl<'a, T, const N: usize> From<&'a mut Tensor<T, N>> for TensorViewMut<'a, T, N> {
    #[inline]
    fn from(value: &'a mut Tensor<T, N>) -> Self {
        value.view_mut()
    }
}

impl<T, const N: usize> AsRef<[T]> for Tensor<T, N> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, const N: usize> AsMut<[T]> for Tensor<T, N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Default> Default for Scalar<T> {
    #[inline]
    fn default() -> Self {
        Scalar::scalar(T::default())
    }
}

impl<T: PartialEq, const N: usize> PartialEq<Tensor<T, N>> for TensorView<'_, T, N> {
    #[inline]
    fn eq(&self, other: &Tensor<T, N>) -> bool {
        self == &other.view()
    }
}

impl<T: PartialEq, const N: usize> PartialEq<Tensor<T, N>> for TensorViewMut<'_, T, N> {
    #[inline]
    fn eq(&self, other: &Tensor<T, N>) -> bool {
        self.view() == other.view()
    }
}
