use core::fmt::{Display, Formatter};
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::ops::{Index, RangeBounds};
use core::ptr::NonNull;
use core::slice::Iter;
use core::{array, fmt, slice};

use crate::{
    MatrixView, ScalarView, Tensor, TensorViewMut, VectorView, fmt_nd, linear_offset,
    linear_offset_checked, resolve_range, unchecked_shape_len, validate_shape,
};

/// An immutable, zero-copy view over a contiguous row-major tensor of rank `N`.
///
/// `TensorView` implements [`Copy`] and [`Clone`] regardless of whether `T`
/// implements them.
#[must_use]
#[derive(Debug, Eq)]
pub struct TensorView<'a, T, const N: usize> {
    shape: [usize; N],
    ptr: NonNull<T>,
    _marker: PhantomData<&'a T>,
}

// SAFETY: `TensorView<'a, T, N>` acts identically to `&'a [T]`.
unsafe impl<T: Sync, const N: usize> Send for TensorView<'_, T, N> {}
// SAFETY: `TensorView<'a, T, N>` provides only shared immutable access to `T`.
unsafe impl<T: Sync, const N: usize> Sync for TensorView<'_, T, N> {}

impl<T, const N: usize> Clone for TensorView<'_, T, N> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, const N: usize> Copy for TensorView<'_, T, N> {}

impl<'a, T, const N: usize> TensorView<'a, T, N> {
    #[inline]
    pub(crate) const fn new(shape: [usize; N], data: &'a [T]) -> Self {
        Self {
            shape,
            ptr: NonNull::from_ref(data).cast(),
            _marker: PhantomData,
        }
    }

    /// Creates a [`TensorView`] directly from a `shape` and non-null base pointer.
    ///
    /// # Safety
    ///
    /// - Every dimension in `shape` must be non-zero (`> 0`), and the product of
    ///   `shape` must not overflow `usize`.
    /// - `ptr` must be valid for reads of `shape.iter().product()` contiguous `T`
    ///   elements for the entire lifetime `'a`, properly aligned, and point to
    ///   initialized values of type `T`.
    #[inline]
    pub(crate) const unsafe fn from_raw_parts(shape: [usize; N], ptr: NonNull<T>) -> Self {
        Self {
            shape,
            ptr,
            _marker: PhantomData,
        }
    }

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
        Self::new(shape, data)
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
        // SAFETY: Any valid `TensorView` validates at construction that the product
        // of `self.shape` does not overflow `usize`.
        unsafe { unchecked_shape_len(&self.shape) }
    }

    /// Returns the underlying row-major data as a flat slice.
    #[inline]
    #[must_use]
    pub const fn as_slice(&self) -> &'a [T] {
        // SAFETY: `self.ptr` is valid for reads of `self.len()` initialized elements
        // of type `T` for lifetime `'a` by `TensorView`'s type invariant.
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len()) }
    }

    /// Returns a raw const pointer to the first element in the view.
    #[inline]
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns an iterator over all elements in row-major order.
    #[inline]
    pub fn iter(&self) -> Iter<'a, T> {
        self.as_slice().iter()
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
        // SAFETY: `validate_shape` verified that `new_shape` has non-zero dimensions
        // and the exact same element count as `self.len()`, so `self.ptr` covers it.
        unsafe { TensorView::from_raw_parts(new_shape, self.ptr) }
    }

    /// Flattens the view into a 1D [`VectorView`] of length `self.len()`.
    #[inline]
    pub const fn flatten(self) -> VectorView<'a, T> {
        // SAFETY: `self.len()` is non-zero and matches the element span of `self.ptr`.
        unsafe { TensorView::from_raw_parts([self.len()], self.ptr) }
    }

    /// Slices the outermost axis (`axis 0`) by `range`, returning a sub-view
    /// with the same rank `N`.
    ///
    /// # Panics
    ///
    /// Panics at compile time if `N == 0`, or at runtime if `range` resolves
    /// to an empty range or exceeds `self.shape()[0]`.
    #[inline]
    pub fn slice<R: RangeBounds<usize>>(self, range: R) -> Self {
        const {
            assert!(N >= 1, "cannot slice a 0D scalar tensor");
        }
        let r = resolve_range(range, self.shape[0]);
        // SAFETY: `self.shape[1..]` is a subslice of `self.shape`, whose product fits in `usize`.
        let inner = unsafe { unchecked_shape_len(&self.shape[1..]) };

        let mut new_shape = self.shape;
        new_shape[0] = r.end - r.start;

        // SAFETY: `r.start < r.end <= self.shape[0]`, so `r.start * inner` is strictly
        // within the `self.shape[0] * inner` allocation of `self.ptr`.
        let ptr = unsafe { self.ptr.add(r.start * inner) };

        // SAFETY: `new_shape[0] = r.end - r.start > 0`, and `(r.end - r.start) * inner`
        // elements starting at `ptr` lie strictly within `self`.
        unsafe { TensorView::from_raw_parts(new_shape, ptr) }
    }

    /// Splits the view into two non-empty sub-views along the outermost axis
    /// (`axis 0`) at `position`.
    ///
    /// The first view covers `0..position` and the second covers
    /// `position..self.shape()[0]`.
    ///
    /// # Panics
    ///
    /// Panics at compile time if `N == 0`, or at runtime if `position == 0`
    /// or `position >= self.shape()[0]` (since zero-sized dimensions are
    /// disallowed).
    #[inline]
    pub fn split_at(self, position: usize) -> (Self, Self) {
        const {
            assert!(N >= 1, "cannot split a 0D scalar tensor");
        }
        assert!(position > 0 && position < self.shape[0]);

        let mut shape1 = self.shape;
        shape1[0] = position;

        let mut shape2 = self.shape;
        shape2[0] -= position;

        // SAFETY: `self.shape[1..]` is a subslice of `self.shape`, whose product fits in `usize`.
        let inner = unsafe { unchecked_shape_len(&self.shape[1..]) };

        // SAFETY: `position < self.shape[0]`, so `position * inner` is within `self`.
        let ptr2 = unsafe { self.ptr.add(position * inner) };

        // SAFETY: Both `position > 0` and `self.shape[0] - position > 0`, and the two
        // sub-ranges partition `self.len()` elements without exceeding `self`.
        let first = unsafe { Self::from_raw_parts(shape1, self.ptr) };
        // SAFETY: See above; `ptr2` points to the remaining `(shape[0] - position) * inner` elements.
        let second = unsafe { Self::from_raw_parts(shape2, ptr2) };

        (first, second)
    }

    /// Partitions the tensor along its leading axes into contiguous sub-views
    /// of rank `M` (spanning the trailing `M` dimensions).
    ///
    /// # Panics
    ///
    /// Panics at compile time if `M > N`.
    #[inline]
    pub fn chunks<const M: usize>(self) -> Chunks<'a, T, M> {
        const {
            assert!(M <= N, "chunk rank M cannot exceed tensor rank N");
        }
        let shape = array::from_fn(|i| self.shape[N - M + i]);
        let inner = if M == 0 {
            ChunksInner::Scalar(self.as_slice().iter())
        } else if M == N {
            ChunksInner::Single(Some(self.as_slice()))
        } else {
            // SAFETY: `shape` consists of the trailing `M` dimensions of `self.shape`,
            // so its product cannot overflow `usize` and is non-zero.
            let chunk_len = unsafe { unchecked_shape_len(&shape) };
            ChunksInner::Multi(self.as_slice().chunks_exact(chunk_len))
        };
        Chunks { inner, shape }
    }

    /// Clones the elements of this view into an owned [`Tensor<T, N>`].
    #[inline]
    pub fn to_tensor(self) -> Tensor<T, N>
    where
        T: Clone,
    {
        Tensor::from_vec(self.as_slice().to_vec(), self.shape)
    }

    /// Returns a reference to the element at `index`, or `None` if any
    /// coordinate is out of bounds.
    #[inline]
    #[must_use]
    pub fn get(self, index: [usize; N]) -> Option<&'a T> {
        let off = linear_offset_checked(&self.shape, &index)?;
        // SAFETY: `linear_offset_checked` bounds-checks every coordinate against `self.shape`,
        // so `off < self.len()` and `ptr` stays within the valid read span.
        let ptr = unsafe { self.ptr.add(off) };
        // SAFETY: `ptr` points to a valid, initialized `T` for lifetime `'a`.
        Some(unsafe { ptr.as_ref() })
    }
}

impl<T, const N: usize> Index<[usize; N]> for TensorView<'_, T, N> {
    type Output = T;

    #[inline]
    fn index(&self, index: [usize; N]) -> &Self::Output {
        let off = linear_offset(&self.shape, &index);
        // SAFETY: `linear_offset` bounds-checks every coordinate against `self.shape`,
        // so `off < self.len()` and `ptr` stays within the valid read span.
        let ptr = unsafe { self.ptr.add(off) };
        // SAFETY: `ptr` points to a valid, initialized `T` for lifetime `'a`.
        unsafe { ptr.as_ref() }
    }
}

impl<'a, T, const N: usize> IntoIterator for TensorView<'a, T, N> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &TensorView<'a, T, N> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<T: Display, const N: usize> Display for TensorView<'_, T, N> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        fmt_nd(f, self.as_slice(), &self.shape, 0)
    }
}

impl<'a, T> ScalarView<'a, T> {
    /// Creates a 0-dimensional [`ScalarView`] borrowing a single reference.
    #[inline]
    pub const fn scalar(val: &'a T) -> Self {
        Self::new([], slice::from_ref(val))
    }

    /// Consumes the view and returns the underlying immutable scalar reference.
    #[inline]
    #[must_use]
    pub const fn into_scalar(self) -> &'a T {
        // SAFETY: A rank-0 `ScalarView` has shape `[]` (`len == 1`), so `self.ptr`
        // points to a valid, initialized `T` for lifetime `'a`.
        unsafe { self.ptr.as_ref() }
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

impl<T, const N: usize> AsRef<[T]> for TensorView<'_, T, N> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

#[derive(Debug, Clone)]
enum ChunksInner<'a, T> {
    Scalar(slice::Iter<'a, T>),
    Single(Option<&'a [T]>),
    Multi(slice::ChunksExact<'a, T>),
}

/// An iterator over contiguous rank-`M` sub-views of a [`TensorView`].
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Debug, Clone)]
pub struct Chunks<'a, T, const M: usize> {
    inner: ChunksInner<'a, T>,
    shape: [usize; M],
}

impl<'a, T, const M: usize> Iterator for Chunks<'a, T, M> {
    type Item = TensorView<'a, T, M>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let data = match &mut self.inner {
            ChunksInner::Scalar(iter) => slice::from_ref(iter.next()?),
            ChunksInner::Single(slot) => slot.take()?,
            ChunksInner::Multi(iter) => iter.next()?,
        };
        Some(TensorView::new(self.shape, data))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let data = match &mut self.inner {
            ChunksInner::Scalar(iter) => slice::from_ref(iter.nth(n)?),
            ChunksInner::Single(slot) => {
                let data = slot.take()?;
                if n == 0 {
                    data
                } else {
                    return None;
                }
            }
            ChunksInner::Multi(iter) => iter.nth(n)?,
        };
        Some(TensorView::new(self.shape, data))
    }

    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let shape = self.shape;
        match self.inner {
            ChunksInner::Scalar(iter) => iter.fold(init, |acc, elem| {
                f(acc, TensorView::new(shape, slice::from_ref(elem)))
            }),
            ChunksInner::Single(Some(data)) => f(init, TensorView::new(shape, data)),
            ChunksInner::Single(None) => init,
            ChunksInner::Multi(iter) => {
                iter.fold(init, |acc, data| f(acc, TensorView::new(shape, data)))
            }
        }
    }
}

impl<T, const M: usize> DoubleEndedIterator for Chunks<'_, T, M> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let data = match &mut self.inner {
            ChunksInner::Scalar(iter) => slice::from_ref(iter.next_back()?),
            ChunksInner::Single(slot) => slot.take()?,
            ChunksInner::Multi(iter) => iter.next_back()?,
        };
        Some(TensorView::new(self.shape, data))
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let data = match &mut self.inner {
            ChunksInner::Scalar(iter) => slice::from_ref(iter.nth_back(n)?),
            ChunksInner::Single(slot) => {
                let data = slot.take()?;
                if n == 0 {
                    data
                } else {
                    return None;
                }
            }
            ChunksInner::Multi(iter) => iter.nth_back(n)?,
        };
        Some(TensorView::new(self.shape, data))
    }

    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let shape = self.shape;
        match self.inner {
            ChunksInner::Scalar(iter) => iter.rfold(init, |acc, elem| {
                f(acc, TensorView::new(shape, slice::from_ref(elem)))
            }),
            ChunksInner::Single(Some(data)) => f(init, TensorView::new(shape, data)),
            ChunksInner::Single(None) => init,
            ChunksInner::Multi(iter) => {
                iter.rfold(init, |acc, data| f(acc, TensorView::new(shape, data)))
            }
        }
    }
}

impl<T, const M: usize> ExactSizeIterator for Chunks<'_, T, M> {
    #[inline]
    fn len(&self) -> usize {
        match &self.inner {
            ChunksInner::Scalar(iter) => iter.len(),
            ChunksInner::Single(slot) => usize::from(slot.is_some()),
            ChunksInner::Multi(iter) => iter.len(),
        }
    }
}

impl<T, const M: usize> FusedIterator for Chunks<'_, T, M> {}

impl<T: PartialEq, const N: usize> PartialEq<TensorView<'_, T, N>> for TensorView<'_, T, N> {
    #[inline]
    fn eq(&self, other: &TensorView<'_, T, N>) -> bool {
        self.shape == other.shape && self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq, const N: usize> PartialEq<TensorView<'_, T, N>> for TensorViewMut<'_, T, N> {
    #[inline]
    fn eq(&self, other: &TensorView<'_, T, N>) -> bool {
        &self.view() == other
    }
}

impl<T: PartialEq, const N: usize> PartialEq<TensorView<'_, T, N>> for Tensor<T, N> {
    #[inline]
    fn eq(&self, other: &TensorView<'_, T, N>) -> bool {
        &self.view() == other
    }
}
