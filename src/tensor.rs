use core::fmt::{self, Display, Formatter};
use core::num::NonZeroUsize;
use core::ptr::NonNull;

use alloc::boxed::Box;
use alloc::{vec, vec::Vec};

use crate::{
    Matrix, Scalar, Tensor, TensorView, TensorViewMut, Vector, check_shape, unpack_shape,
    validate_shape,
};

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
    #[track_caller]
    pub fn from_vec(data: Vec<T>, shape: [usize; N]) -> Self {
        let shape = validate_shape(data.len(), shape);
        // SAFETY: `validate_shape` verified `shape` does not overflow and matches `data.len()`.
        unsafe { Self::from_raw_parts(shape, data.into_boxed_slice()) }
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
    #[track_caller]
    pub fn from_boxed_slice(data: Box<[T]>, shape: [usize; N]) -> Self {
        let shape = validate_shape(data.len(), shape);
        // SAFETY: `validate_shape` verified `shape` does not overflow and matches `data.len()`.
        unsafe { Self::from_raw_parts(shape, data) }
    }

    /// Allocates a [`Tensor`] of `shape` filled with clones of `val`.
    ///
    /// # Panics
    ///
    /// Panics if any dimension in `shape` is zero or if its product overflows
    /// `usize`.
    #[inline]
    #[track_caller]
    pub fn from_elem(val: T, shape: [usize; N]) -> Self
    where
        T: Clone,
    {
        let (shape, len) = check_shape(shape);
        // SAFETY: `check_shape` verified `shape` does not overflow and has product `len`.
        unsafe { Self::from_raw_parts(shape, vec![val; len].into_boxed_slice()) }
    }

    /// Allocates a [`Tensor`] of `shape` filled with `T::default()`.
    ///
    /// # Panics
    ///
    /// Panics if any dimension in `shape` is zero or if its product overflows
    /// `usize`.
    #[inline]
    #[track_caller]
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
        self.into_data().into_vec()
    }

    /// Consumes the tensor and returns its underlying flat [`Box<[T]>`](Box).
    #[inline]
    #[must_use]
    pub fn into_boxed_slice(self) -> Box<[T]> {
        self.into_data()
    }

    /// Returns the dimensions of the tensor.
    #[inline]
    #[must_use]
    pub const fn shape(&self) -> [usize; N] {
        unpack_shape(self.shape_raw())
    }

    /// Returns the total number of elements in the tensor.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.data().len()
    }

    /// Returns the underlying row-major data as an immutable flat slice.
    #[inline]
    #[must_use]
    pub const fn as_slice(&self) -> &[T] {
        self.data()
    }

    /// Returns a raw const pointer to the first element in the tensor.
    #[inline]
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.data().as_ptr()
    }

    /// Returns the underlying row-major data as a mutable flat slice.
    #[inline]
    #[must_use]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        self.data_mut()
    }

    /// Returns a raw mutable pointer to the first element in the tensor.
    #[inline]
    #[must_use]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.data_mut().as_mut_ptr()
    }

    /// Borrows the tensor as an immutable [`TensorView`].
    #[inline]
    pub const fn view(&self) -> TensorView<'_, T, N> {
        // SAFETY: `self.data.len()` matches the product of `self.shape` by `Tensor`'s invariant.
        unsafe {
            TensorView::from_raw_parts(self.shape_raw(), NonNull::from_ref(self.as_slice()).cast())
        }
    }

    /// Borrows the tensor as a mutable [`TensorViewMut`].
    #[inline]
    pub const fn view_mut(&mut self) -> TensorViewMut<'_, T, N> {
        // SAFETY: `self.data.len()` matches the product of `self.shape` by `Tensor`'s invariant.
        unsafe {
            TensorViewMut::from_raw_parts(
                self.shape_raw(),
                NonNull::from_mut(self.as_mut_slice()).cast(),
            )
        }
    }

    /// Consumes and reshapes the owned tensor to rank `M` and `new_shape`
    /// without reallocating.
    ///
    /// # Panics
    ///
    /// Panics if any dimension in `new_shape` is zero, if its product overflows
    /// `usize`, or if the element count differs from `self.len()`.
    #[inline]
    #[track_caller]
    pub fn reshape<const M: usize>(self, new_shape: [usize; M]) -> Tensor<T, M> {
        let shape = validate_shape(self.len(), new_shape);
        // SAFETY: `validate_shape` verified `shape` does not overflow and matches `self.data.len()`.
        unsafe { Tensor::from_raw_parts(shape, self.into_data()) }
    }

    /// Consumes and flattens the tensor into a 1D [`Vector`] of length
    /// `self.len()`.
    #[inline]
    pub fn flatten(self) -> Vector<T> {
        // SAFETY: `self.data.len()` equals the product of non-zero dimensions in `self.shape`,
        // so it is strictly non-zero.
        let len = unsafe { NonZeroUsize::new_unchecked(self.len()) };
        // SAFETY: `[len]` has product `self.data.len()`.
        unsafe { Tensor::from_raw_parts([len], self.into_data()) }
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
        unsafe { self.into_vec().pop().unwrap_unchecked() }
    }
}

impl<T> Matrix<T> {
    /// Returns the number of rows (`shape()[0]`) in the matrix.
    #[inline]
    #[must_use]
    pub const fn nrows(&self) -> usize {
        self.shape_raw()[0].get()
    }

    /// Returns the number of columns (`shape()[1]`) in the matrix.
    #[inline]
    #[must_use]
    pub const fn ncols(&self) -> usize {
        self.shape_raw()[1].get()
    }
}

impl<T> From<Vec<T>> for Vector<T> {
    #[inline]
    #[track_caller]
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
