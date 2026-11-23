use core::fmt::{Display, Formatter};
use core::iter::FusedIterator;
use core::ops::{Index, IndexMut, RangeBounds};
use core::slice::{Iter, IterMut};
use core::{array, fmt, slice};

use crate::{
    MatrixViewMut, ScalarViewMut, Tensor, TensorView, VectorViewMut, linear_offset,
    linear_offset_checked, resolve_range, shape_len, validate_shape,
};

/// A mutable, zero-copy view over a contiguous row-major tensor of rank `N`.
#[must_use]
#[derive(Debug, PartialEq, Eq)]
pub struct TensorViewMut<'a, T, const N: usize> {
    shape: [usize; N],
    data: &'a mut [T],
}

impl<'a, T, const N: usize> TensorViewMut<'a, T, N> {
    pub(crate) const fn new(shape: [usize; N], data: &'a mut [T]) -> Self {
        Self { shape, data }
    }

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
        TensorView::new(self.shape, self.data)
    }

    /// Consumes this mutable view and converts it into an immutable
    /// [`TensorView`] with lifetime `'a`.
    #[inline]
    pub const fn into_view(self) -> TensorView<'a, T, N> {
        TensorView::new(self.shape, self.data)
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
        self.view_mut()
            .chunks::<M>()
            .for_each(|chunk| chunk.into_mut_slice().copy_from_slice(src_slice));
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
    pub const fn flatten(self) -> VectorViewMut<'a, T> {
        TensorViewMut {
            shape: [self.len()],
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
    pub fn slice<R: RangeBounds<usize>>(self, range: R) -> Self {
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
        Self {
            data: new_data,
            shape: new_shape,
        }
    }

    /// Consumes the mutable view and splits it into two non-empty,
    /// non-overlapping mutable sub-views along the outermost axis (`axis 0`)
    /// at `position`.
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

        // SAFETY: `position < self.shape[0]` and `self.data.len() == self.shape[0] * inner`,
        // so `position * inner` is strictly within `self.data`.
        let (data1, data2) = unsafe {
            self.data
                .split_at_mut_unchecked(position * shape_len(&self.shape[1..]))
        };

        (
            Self {
                data: data1,
                shape: shape1,
            },
            Self {
                data: data2,
                shape: shape2,
            },
        )
    }

    /// Consumes and partitions the mutable view into mutable sub-views of rank `M`
    /// spanning the trailing `M` dimensions.
    ///
    /// # Panics
    ///
    /// Panics at compile time if `M > N`.
    #[inline]
    pub fn chunks<const M: usize>(self) -> ChunksMut<'a, T, M> {
        const {
            assert!(M <= N, "chunk rank M cannot exceed tensor rank N");
        }
        let shape = array::from_fn(|i| self.shape[N - M + i]);
        let inner = if M == N {
            ChunksInnerMut::Single(Some(self.data))
        } else {
            ChunksInnerMut::Multi(self.data.chunks_exact_mut(shape_len(&shape)))
        };
        ChunksMut { inner, shape }
    }

    /// Clones the elements of this view into an owned [`Tensor<T, N>`].
    #[inline]
    pub fn to_tensor(&self) -> Tensor<T, N>
    where
        T: Clone,
    {
        self.view().to_tensor()
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

impl<T: Display, const N: usize> Display for TensorViewMut<'_, T, N> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.view().fmt(f)
    }
}

impl<'a, T> ScalarViewMut<'a, T> {
    /// Creates a 0-dimensional [`ScalarViewMut`] borrowing a single mutable
    /// reference.
    #[inline]
    pub const fn scalar(val: &'a mut T) -> Self {
        Self {
            data: slice::from_mut(val),
            shape: [],
        }
    }

    /// Consumes the view and returns the underlying mutable scalar reference.
    #[inline]
    #[must_use]
    pub const fn into_scalar(self) -> &'a mut T {
        // SAFETY: A rank-0 `ScalarViewMut` has shape `[]`, which guarantees `self.data.len() == 1`.
        unsafe { &mut *self.data.as_mut_ptr() }
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

impl<'a, T, const N: usize> From<TensorViewMut<'a, T, N>> for TensorView<'a, T, N> {
    #[inline]
    fn from(value: TensorViewMut<'a, T, N>) -> Self {
        value.into_view()
    }
}

impl<T, const N: usize> AsRef<[T]> for TensorViewMut<'_, T, N> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, const N: usize> AsMut<[T]> for TensorViewMut<'_, T, N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

#[derive(Debug)]
enum ChunksInnerMut<'a, T> {
    Single(Option<&'a mut [T]>),
    Multi(slice::ChunksExactMut<'a, T>),
}

/// An iterator over contiguous mutable rank-`M` sub-views of a [`TensorViewMut`].
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Debug)]
pub struct ChunksMut<'a, T, const M: usize> {
    inner: ChunksInnerMut<'a, T>,
    shape: [usize; M],
}

impl<'a, T, const M: usize> Iterator for ChunksMut<'a, T, M> {
    type Item = TensorViewMut<'a, T, M>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let data = match &mut self.inner {
            ChunksInnerMut::Single(slot) => slot.take()?,
            ChunksInnerMut::Multi(iter) => iter.next()?,
        };
        Some(TensorViewMut {
            shape: self.shape,
            data,
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let data = match &mut self.inner {
            ChunksInnerMut::Single(slot) => {
                let data = slot.take()?;
                if n == 0 {
                    data
                } else {
                    return None;
                }
            }
            ChunksInnerMut::Multi(iter) => iter.nth(n)?,
        };
        Some(TensorViewMut {
            data,
            shape: self.shape,
        })
    }

    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let shape = self.shape;
        match self.inner {
            ChunksInnerMut::Single(Some(data)) => f(init, TensorViewMut { shape, data }),
            ChunksInnerMut::Single(None) => init,
            ChunksInnerMut::Multi(iter) => {
                iter.fold(init, |acc, data| f(acc, TensorViewMut { shape, data }))
            }
        }
    }
}

impl<T, const M: usize> DoubleEndedIterator for ChunksMut<'_, T, M> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let data = match &mut self.inner {
            ChunksInnerMut::Single(slot) => slot.take()?,
            ChunksInnerMut::Multi(iter) => iter.next_back()?,
        };
        Some(TensorViewMut {
            shape: self.shape,
            data,
        })
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let data = match &mut self.inner {
            ChunksInnerMut::Single(slot) => {
                let data = slot.take()?;
                if n == 0 {
                    data
                } else {
                    return None;
                }
            }
            ChunksInnerMut::Multi(iter) => iter.nth_back(n)?,
        };
        Some(TensorViewMut {
            data,
            shape: self.shape,
        })
    }

    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let shape = self.shape;
        match self.inner {
            ChunksInnerMut::Single(Some(data)) => f(init, TensorViewMut { shape, data }),
            ChunksInnerMut::Single(None) => init,
            ChunksInnerMut::Multi(iter) => {
                iter.rfold(init, |acc, data| f(acc, TensorViewMut { shape, data }))
            }
        }
    }
}

impl<T, const M: usize> ExactSizeIterator for ChunksMut<'_, T, M> {
    #[inline]
    fn len(&self) -> usize {
        match &self.inner {
            ChunksInnerMut::Single(slot) => usize::from(slot.is_some()),
            ChunksInnerMut::Multi(iter) => iter.len(),
        }
    }
}

impl<T, const M: usize> FusedIterator for ChunksMut<'_, T, M> {}

impl<T: PartialEq, const N: usize> PartialEq<TensorViewMut<'_, T, N>> for TensorView<'_, T, N> {
    #[inline]
    fn eq(&self, other: &TensorViewMut<'_, T, N>) -> bool {
        self == &other.view()
    }
}

impl<T: PartialEq, const N: usize> PartialEq<TensorViewMut<'_, T, N>> for Tensor<T, N> {
    #[inline]
    fn eq(&self, other: &TensorViewMut<'_, T, N>) -> bool {
        self.view() == other.view()
    }
}
