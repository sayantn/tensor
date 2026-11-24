use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{FusedIterator, zip};
use core::num::NonZeroUsize;
use core::ops::{Index, RangeBounds};
use core::ptr::NonNull;
use core::slice::{self, Iter};

use crate::{
    MatrixView, Monoid, ScalarView, TensorView, TensorViewMut, VectorView, fmt_nd, linear_offset,
    linear_offset_checked, resolve_range, unchecked_shape_len, unpack_shape, validate_shape,
};

#[cfg(feature = "alloc")]
use crate::Tensor;

// ============================================================================
// TensorView
// ============================================================================

impl<'a, T, const N: usize> TensorView<'a, T, N> {
    // -- Construction & Conversion -------------------------------------------

    /// Wraps an existing row-major slice in an immutable [`TensorView`].
    ///
    /// # Panics
    ///
    /// Panics if any dimension in `shape` is zero, if the product of `shape`
    /// overflows `usize`, or if `data.len()` does not equal the product of
    /// `shape`.
    #[inline]
    #[track_caller]
    pub const fn from_slice(data: &'a [T], shape: [usize; N]) -> Self {
        let shape = validate_shape(data.len(), shape);
        // SAFETY: `validate_shape` verified `shape` matches `data.len()` without overflow,
        // and `data` is valid for reads for lifetime `'a`.
        unsafe { Self::from_raw_parts(shape, NonNull::from_ref(data).cast()) }
    }

    /// Clones the elements of this view into an owned [`Tensor<T, N>`].
    #[cfg(feature = "alloc")]
    #[inline]
    pub fn to_tensor(self) -> Tensor<T, N>
    where
        T: Clone,
    {
        // SAFETY: `self.shape` is valid and the cloned boxed slice has length `self.len()`.
        unsafe {
            Tensor::from_raw_parts(
                self.shape_raw(),
                self.as_slice().to_vec().into_boxed_slice(),
            )
        }
    }

    // -- Metadata & Raw Access -----------------------------------------------

    /// Returns the dimensions of the tensor view.
    #[inline]
    #[must_use]
    pub const fn shape(&self) -> [usize; N] {
        unpack_shape(self.shape_raw())
    }

    /// Returns the total number of elements in the tensor view.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        // SAFETY: Any valid `TensorView` validates at construction that the product
        // of `self.shape` does not overflow `usize`.
        unsafe { unchecked_shape_len(&self.shape_raw()).get() }
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
        self.ptr().as_ptr()
    }

    // -- Indexing & Iteration ------------------------------------------------

    /// Returns a reference to the element at `index`, or `None` if any
    /// coordinate is out of bounds.
    #[inline]
    #[must_use]
    pub const fn get(self, index: [usize; N]) -> Option<&'a T> {
        let Some(off) = linear_offset_checked(&self.shape_raw(), &index) else {
            return None;
        };
        // SAFETY: `linear_offset_checked` bounds-checks every coordinate against `self.shape`,
        // so `off < self.len()` and `ptr` stays within the valid read span.
        let ptr = unsafe { self.ptr().add(off) };
        // SAFETY: `ptr` points to a valid, initialized `T` for lifetime `'a`.
        Some(unsafe { ptr.as_ref() })
    }

    /// Returns an iterator over all elements in row-major order.
    #[inline]
    pub fn iter(&self) -> Iter<'a, T> {
        self.as_slice().iter()
    }

    // -- Layout & Slicing ----------------------------------------------------

    /// Reinterprets the view with a new rank `M` and `new_shape` without
    /// copying data.
    ///
    /// # Panics
    ///
    /// Panics if any dimension in `new_shape` is zero, if its product overflows
    /// `usize`, or if the element count differs from `self.len()`.
    #[inline]
    #[track_caller]
    pub const fn reshape<const M: usize>(self, new_shape: [usize; M]) -> TensorView<'a, T, M> {
        // SAFETY: `validate_shape` verified that `new_shape` has non-zero dimensions
        // and the exact same element count as `self.len()`, so `self.ptr` covers it.
        unsafe { TensorView::from_raw_parts(validate_shape(self.len(), new_shape), self.ptr()) }
    }

    /// Flattens the view into a 1D [`VectorView`] of length `self.len()`.
    #[inline]
    pub const fn flatten(self) -> VectorView<'a, T> {
        // SAFETY: Any valid `TensorView` validates at construction that the product
        // of `self.shape` does not overflow `usize`.
        let len = unsafe { unchecked_shape_len(&self.shape_raw()) };
        // SAFETY: `len` is non-zero and matches the element span of `self.ptr`.
        unsafe { TensorView::from_raw_parts([len], self.ptr()) }
    }

    /// Slices the outermost axis (`axis 0`) by `range`, returning a sub-view
    /// with the same rank `N`.
    ///
    /// # Panics
    ///
    /// Panics at compile time if `N == 0`, or at runtime if `range` resolves
    /// to an empty range or exceeds `self.shape()[0]`.
    #[inline]
    #[track_caller]
    pub fn slice<R: RangeBounds<usize>>(self, range: R) -> Self {
        const {
            assert!(N >= 1, "cannot slice a 0D scalar tensor");
        }

        let shape = self.shape_raw();
        let (start, len) = resolve_range(range, shape[0]);
        // SAFETY: `self.shape[1..]` is a subslice of `self.shape`, whose product fits in `usize`.
        let inner = unsafe { unchecked_shape_len(&shape[1..]) };

        let mut new_shape = shape;
        new_shape[0] = len;

        // SAFETY: `start + len <= self.shape[0]`, so `start * inner` is strictly
        // within the `self.shape[0] * inner` allocation of `self.ptr`.
        let ptr = unsafe { self.ptr().add(start * inner.get()) };

        // SAFETY: `len` is non-zero and `len * inner` elements starting at `ptr`
        // lie strictly within `self`.
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
    #[track_caller]
    pub const fn split_at(self, position: usize) -> (Self, Self) {
        const {
            assert!(N >= 1, "cannot split a 0D scalar tensor");
        }

        let shape = self.shape_raw();
        let Some((&axis0, rest)) = shape.split_first() else {
            unreachable!();
        };
        let Some(first_len) = NonZeroUsize::new(position) else {
            panic!("split position must be non-zero");
        };
        let Some(rem) = axis0.get().checked_sub(position) else {
            panic!("split position out of bounds");
        };
        let Some(second_len) = NonZeroUsize::new(rem) else {
            panic!("split position must be less than axis len");
        };

        let mut shape1 = shape;
        shape1[0] = first_len;

        let mut shape2 = shape;
        shape2[0] = second_len;

        // SAFETY: `rest` is a subslice of `self.shape`, whose product fits in `usize`.
        let inner = unsafe { unchecked_shape_len(rest).get() };

        // SAFETY: `position < axis0`, so `position * inner` is within `self`.
        let ptr2 = unsafe { self.ptr().add(position * inner) };

        // SAFETY: `first_len` and `second_len` are non-zero and partition `axis0`.
        let first = unsafe { Self::from_raw_parts(shape1, self.ptr()) };
        // SAFETY: `ptr2` points to the remaining `second_len * inner` elements.
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
    #[track_caller]
    pub fn chunks<const M: usize>(self) -> Chunks<'a, T, M> {
        const {
            assert!(M <= N, "chunk rank M cannot exceed tensor rank N");
        }
        let shape = *self.shape_raw().last_chunk().unwrap();

        let inner = if M == 0 {
            ChunksInner::Scalar(self.as_slice().iter())
        } else if M == N {
            ChunksInner::Single(Some(self.as_slice()))
        } else {
            // SAFETY: `shape` consists of the trailing `M` dimensions of `self.shape`,
            // so its product cannot overflow `usize` and is non-zero.
            let chunk_len = unsafe { unchecked_shape_len(&shape) };
            ChunksInner::Multi(self.as_slice().chunks_exact(chunk_len.get()))
        };
        Chunks { inner, shape }
    }

    // -- Reductions ----------------------------------------------------------

    /// Maps each element with `f` and reduces the mapped values using the
    /// associative, commutative [`Monoid`] `op` across 16 independent
    /// accumulators for SIMD vectorization.
    #[inline]
    #[must_use]
    pub fn reduce_by<Out: Copy>(self, op: impl Monoid<Out>, mut f: impl FnMut(&T) -> Out) -> Out {
        const LANES: usize = 16;
        let (chunks, rem) = self.as_slice().as_chunks::<LANES>();

        let mut acc = [op.identity(); LANES];
        for chunk in chunks {
            for l in 0..LANES {
                acc[l] = op.combine(acc[l], f(&chunk[l]));
            }
        }

        let mut total = acc[0];
        for &lane in &acc[1..] {
            total = op.combine(total, lane);
        }
        for x in rem {
            total = op.combine(total, f(x));
        }
        total
    }

    /// Reduces all elements using the associative, commutative [`Monoid`] `op`
    /// across 16 independent accumulators.
    #[inline]
    #[must_use]
    pub fn reduce(self, op: impl Monoid<T>) -> T
    where
        T: Copy,
    {
        self.reduce_by(op, |&x| x)
    }

    /// Maps each elementwise pair `(a, b)` from `self` and `rhs` with `f` and
    /// reduces the mapped values using the [`Monoid`] `op` across 16
    /// independent accumulators.
    ///
    /// # Panics
    ///
    /// Panics if `self.shape() != rhs.shape()`.
    #[inline]
    #[must_use]
    #[track_caller]
    pub fn zip_reduce<U, Out: Copy>(
        self,
        rhs: TensorView<'_, U, N>,
        op: impl Monoid<Out>,
        mut f: impl FnMut(&T, &U) -> Out,
    ) -> Out {
        const LANES: usize = 16;
        assert_eq!(self.shape(), rhs.shape(), "tensor shape mismatch");

        let (ac, ar) = self.as_slice().as_chunks::<LANES>();
        let (bc, br) = rhs.as_slice().as_chunks::<LANES>();

        let mut acc = [op.identity(); LANES];
        for (aa, ba) in zip(ac, bc) {
            for l in 0..LANES {
                acc[l] = op.combine(acc[l], f(&aa[l], &ba[l]));
            }
        }

        let mut total = acc[0];
        for &lane in &acc[1..] {
            total = op.combine(total, lane);
        }
        for (a, b) in zip(ar, br) {
            total = op.combine(total, f(a, b));
        }
        total
    }
}

// ============================================================================
// Rank-Specific Helpers
// ============================================================================

impl<'a, T> ScalarView<'a, T> {
    /// Creates a 0-dimensional [`ScalarView`] borrowing a single reference.
    #[inline]
    pub const fn scalar(val: &'a T) -> Self {
        // SAFETY: Shape `[]` has product 1 and `val` points to 1 initialized `T` for `'a`.
        unsafe { Self::from_raw_parts([], NonNull::from_ref(val)) }
    }

    /// Consumes the view and returns the underlying immutable scalar reference.
    #[inline]
    #[must_use]
    pub const fn into_scalar(self) -> &'a T {
        // SAFETY: A rank-0 `ScalarView` has shape `[]` (`len == 1`), so `self.ptr`
        // points to a valid, initialized `T` for lifetime `'a`.
        unsafe { self.ptr().as_ref() }
    }
}

impl<'a, T> VectorView<'a, T> {
    /// Creates a 1-dimensional [`VectorView`] borrowing a slice.
    ///
    /// # Panics
    ///
    /// Panics if `val` is empty.
    #[inline]
    #[track_caller]
    pub const fn vector(val: &'a [T]) -> Self {
        Self::from_slice(val, [val.len()])
    }
}

impl<T> MatrixView<'_, T> {
    /// Returns the number of rows (`shape()[0]`) in the matrix view.
    #[inline]
    #[must_use]
    pub const fn nrows(&self) -> usize {
        self.shape_raw()[0].get()
    }

    /// Returns the number of columns (`shape()[1]`) in the matrix view.
    #[inline]
    #[must_use]
    pub const fn ncols(&self) -> usize {
        self.shape_raw()[1].get()
    }
}

// ============================================================================
// Standard Trait Implementations
// ============================================================================

impl<T, const N: usize> Clone for TensorView<'_, T, N> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, const N: usize> Copy for TensorView<'_, T, N> {}

impl<T, const N: usize> AsRef<[T]> for TensorView<'_, T, N> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, const N: usize> Index<[usize; N]> for TensorView<'_, T, N> {
    type Output = T;

    #[inline]
    #[track_caller]
    fn index(&self, index: [usize; N]) -> &Self::Output {
        let off = linear_offset(&self.shape_raw(), &index);
        // SAFETY: `linear_offset` bounds-checks every coordinate against `self.shape`,
        // so `off < self.len()` and `ptr` stays within the valid read span.
        let ptr = unsafe { self.ptr().add(off) };
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

impl<T: Debug, const N: usize> Debug for TensorView<'_, T, N> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorView")
            .field("shape", &self.shape())
            .field("data", &self.as_slice())
            .finish()
    }
}

impl<T: Display, const N: usize> Display for TensorView<'_, T, N> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        fmt_nd(f, self.as_slice(), &self.shape_raw(), 0)
    }
}

impl<T: PartialEq, const N: usize> PartialEq<TensorView<'_, T, N>> for TensorView<'_, T, N> {
    #[inline]
    fn eq(&self, other: &TensorView<'_, T, N>) -> bool {
        self.shape_raw() == other.shape_raw() && self.as_slice() == other.as_slice()
    }
}

impl<T: Eq, const N: usize> Eq for TensorView<'_, T, N> {}

impl<T: PartialEq, const N: usize> PartialEq<TensorView<'_, T, N>> for TensorViewMut<'_, T, N> {
    #[inline]
    fn eq(&self, other: &TensorView<'_, T, N>) -> bool {
        &self.view() == other
    }
}

#[cfg(feature = "alloc")]
impl<T: PartialEq, const N: usize> PartialEq<TensorView<'_, T, N>> for Tensor<T, N> {
    #[inline]
    fn eq(&self, other: &TensorView<'_, T, N>) -> bool {
        &self.view() == other
    }
}

// ============================================================================
// Chunks Iterator
// ============================================================================

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
    shape: [NonZeroUsize; M],
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
        // SAFETY: `data.len()` matches the product of `self.shape` by `Chunks` construction.
        Some(unsafe { TensorView::from_raw_parts(self.shape, NonNull::from_ref(data).cast()) })
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
        // SAFETY: `data.len()` matches the product of `self.shape` by `Chunks` construction.
        Some(unsafe { TensorView::from_raw_parts(self.shape, NonNull::from_ref(data).cast()) })
    }

    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let shape = self.shape;
        match self.inner {
            ChunksInner::Scalar(iter) => iter.fold(init, |acc, elem| {
                // SAFETY: `shape` is `[]` (len 1) and `elem` is a valid `&'a T`.
                let view = unsafe { TensorView::from_raw_parts(shape, NonNull::from_ref(elem)) };
                f(acc, view)
            }),
            ChunksInner::Single(Some(data)) => {
                // SAFETY: `data.len()` matches the product of `shape` by `Chunks` construction.
                let view =
                    unsafe { TensorView::from_raw_parts(shape, NonNull::from_ref(data).cast()) };
                f(init, view)
            }
            ChunksInner::Single(None) => init,
            ChunksInner::Multi(iter) => iter.fold(init, |acc, data| {
                // SAFETY: `data.len()` matches the product of `shape` by `Chunks` construction.
                let view =
                    unsafe { TensorView::from_raw_parts(shape, NonNull::from_ref(data).cast()) };
                f(acc, view)
            }),
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
        // SAFETY: `data.len()` matches the product of `self.shape` by `Chunks` construction.
        Some(unsafe { TensorView::from_raw_parts(self.shape, NonNull::from_ref(data).cast()) })
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
        // SAFETY: `data.len()` matches the product of `self.shape` by `Chunks` construction.
        Some(unsafe { TensorView::from_raw_parts(self.shape, NonNull::from_ref(data).cast()) })
    }

    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let shape = self.shape;
        match self.inner {
            ChunksInner::Scalar(iter) => iter.rfold(init, |acc, elem| {
                // SAFETY: `shape` is `[]` (len 1) and `elem` is a valid `&'a T`.
                let view = unsafe { TensorView::from_raw_parts(shape, NonNull::from_ref(elem)) };
                f(acc, view)
            }),
            ChunksInner::Single(Some(data)) => {
                // SAFETY: `data.len()` matches the product of `shape` by `Chunks` construction.
                let view =
                    unsafe { TensorView::from_raw_parts(shape, NonNull::from_ref(data).cast()) };
                f(init, view)
            }
            ChunksInner::Single(None) => init,
            ChunksInner::Multi(iter) => iter.rfold(init, |acc, data| {
                // SAFETY: `data.len()` matches the product of `shape` by `Chunks` construction.
                let view =
                    unsafe { TensorView::from_raw_parts(shape, NonNull::from_ref(data).cast()) };
                f(acc, view)
            }),
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
