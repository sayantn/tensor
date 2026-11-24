use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{FusedIterator, zip};
use core::num::NonZeroUsize;
use core::ops::{Index, IndexMut, RangeBounds};
use core::ptr::NonNull;
use core::slice::{self, Iter, IterMut};

use crate::{
    MatrixViewMut, Monoid, ScalarViewMut, TensorView, TensorViewMut, VectorViewMut, linear_offset,
    linear_offset_checked, resolve_range, unchecked_shape_len, unpack_shape, validate_shape,
};

#[cfg(feature = "alloc")]
use crate::Tensor;

// ============================================================================
// TensorViewMut
// ============================================================================

impl<'a, T, const N: usize> TensorViewMut<'a, T, N> {
    // -- Construction & Conversion -------------------------------------------

    /// Wraps an existing mutable row-major slice in a [`TensorViewMut`].
    ///
    /// # Panics
    ///
    /// Panics if any dimension in `shape` is zero, if the product of `shape`
    /// overflows `usize`, or if `data.len()` does not equal the product of
    /// `shape`.
    #[inline]
    #[track_caller]
    pub const fn from_slice(data: &'a mut [T], shape: [usize; N]) -> Self {
        let shape = validate_shape(data.len(), shape);
        // SAFETY: `validate_shape` verified `shape` matches `data.len()` without overflow,
        // and `data` is exclusively valid for reads and writes for lifetime `'a`.
        unsafe { Self::from_raw_parts(shape, NonNull::from_mut(data).cast()) }
    }

    /// Consumes the view and returns the underlying mutable slice with
    /// lifetime `'a`.
    #[inline]
    #[must_use]
    pub const fn into_mut_slice(self) -> &'a mut [T] {
        // SAFETY: Consuming `self` transfers exclusive `'a` access to `self.ptr`
        // across `self.len()` initialized elements of type `T`.
        unsafe { slice::from_raw_parts_mut(self.ptr().as_ptr(), self.len()) }
    }

    /// Clones the elements of this view into an owned [`Tensor<T, N>`].
    #[cfg(feature = "alloc")]
    #[inline]
    pub fn to_tensor(&self) -> Tensor<T, N>
    where
        T: Clone,
    {
        self.view().to_tensor()
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
        // SAFETY: Any valid `TensorViewMut` validates at construction that the product
        // of `self.shape` does not overflow `usize`.
        unsafe { unchecked_shape_len(&self.shape_raw()).get() }
    }

    /// Returns the underlying row-major data as an immutable flat slice.
    #[inline]
    #[must_use]
    pub const fn as_slice(&self) -> &[T] {
        // SAFETY: `self.ptr` is valid for reads of `self.len()` initialized elements
        // of type `T` while `self` is borrowed by `TensorViewMut`'s type invariant.
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len()) }
    }

    /// Returns a raw const pointer to the first element in the view.
    #[inline]
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr().as_ptr()
    }

    /// Returns the underlying row-major data as a mutable flat slice.
    #[inline]
    #[must_use]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `self.ptr` is exclusively valid for reads and writes of `self.len()`
        // initialized elements of type `T` while `self` is mutably borrowed.
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }

    /// Returns a raw mutable pointer to the first element in the view.
    #[inline]
    #[must_use]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr().as_ptr()
    }

    // -- Reborrowing ---------------------------------------------------------

    /// Reborrows this mutable view as an immutable [`TensorView`].
    #[inline]
    pub const fn view(&self) -> TensorView<'_, T, N> {
        // SAFETY: Reborrowing `&self` guarantees shared read access to `self.ptr`
        // over `self.shape` for the reborrow lifetime.
        unsafe { TensorView::from_raw_parts(self.shape_raw(), self.ptr()) }
    }

    /// Consumes this mutable view and converts it into an immutable
    /// [`TensorView`] with lifetime `'a`.
    #[inline]
    pub const fn into_view(self) -> TensorView<'a, T, N> {
        // SAFETY: Consuming `self` downgrades exclusive `'a` access to shared `'a`
        // read access over `self.shape`.
        unsafe { TensorView::from_raw_parts(self.shape_raw(), self.ptr()) }
    }

    /// Reborrows this mutable view with a shorter lifetime.
    #[inline]
    pub const fn view_mut(&mut self) -> TensorViewMut<'_, T, N> {
        // SAFETY: Reborrowing `&mut self` guarantees exclusive read/write access to
        // `self.ptr` over `self.shape` for the reborrow lifetime.
        unsafe { TensorViewMut::from_raw_parts(self.shape_raw(), self.ptr()) }
    }

    // -- Indexing & Iteration ------------------------------------------------

    /// Returns a reference to the element at `index`, or `None` if any
    /// coordinate is out of bounds.
    #[inline]
    #[must_use]
    pub const fn get(&self, index: [usize; N]) -> Option<&T> {
        let Some(off) = linear_offset_checked(&self.shape_raw(), &index) else {
            return None;
        };
        // SAFETY: `linear_offset_checked` bounds-checks every coordinate against `self.shape`,
        // so `off < self.len()` and `ptr` stays within the exclusive span.
        let ptr = unsafe { self.ptr().add(off) };
        // SAFETY: `ptr` points to a valid, initialized, exclusively borrowed `T`.
        Some(unsafe { ptr.as_ref() })
    }

    /// Returns a mutable reference to the element at `index`, or `None` if any
    /// coordinate is out of bounds.
    #[inline]
    #[must_use]
    pub const fn get_mut(&mut self, index: [usize; N]) -> Option<&mut T> {
        let Some(off) = linear_offset_checked(&self.shape_raw(), &index) else {
            return None;
        };
        // SAFETY: `linear_offset_checked` bounds-checks every coordinate against `self.shape`,
        // so `off < self.len()` and `ptr` stays within the exclusive span.
        let mut ptr = unsafe { self.ptr().add(off) };
        // SAFETY: `ptr` points to a valid, initialized, exclusively borrowed `T`.
        Some(unsafe { ptr.as_mut() })
    }

    /// Returns an iterator over immutable element references in row-major
    /// order.
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        self.as_slice().iter()
    }

    /// Returns an iterator over mutable element references in row-major order.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }

    // -- In-Place Mutation ---------------------------------------------------

    /// Fills every element in the view with clones of `value`.
    #[inline]
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.as_mut_slice().fill(value);
    }

    /// Copies `src` into every trailing rank-`M` slice of `self` (NumPy-style
    /// right-aligned broadcasting).
    ///
    /// # Panics
    ///
    /// Panics at compile time if `M > N`, or at runtime if `src.shape()` does
    /// not match `&self.shape()[N - M..]`.
    #[inline]
    #[track_caller]
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
        let tail = &self.shape_raw()[N - M..];
        assert_eq!(tail, src.shape_raw(), "assign shape mismatch");
        let src_slice = src.as_slice();
        self.view_mut()
            .chunks::<M>()
            .for_each(|chunk| chunk.into_mut_slice().copy_from_slice(src_slice));
    }

    // -- Layout & Slicing ----------------------------------------------------

    /// Consumes and reinterprets the mutable view with a new rank `M` and
    /// `new_shape` without copying data.
    ///
    /// # Panics
    ///
    /// Panics if any dimension in `new_shape` is zero, if its product overflows
    /// `usize`, or if the element count differs from `self.len()`.
    #[inline]
    #[track_caller]
    pub const fn reshape<const M: usize>(self, new_shape: [usize; M]) -> TensorViewMut<'a, T, M> {
        // SAFETY: `validate_shape` verified that `new_shape` has non-zero dimensions
        // and the exact same element count as `self.len()`, so `self.ptr` covers it.
        unsafe { TensorViewMut::from_raw_parts(validate_shape(self.len(), new_shape), self.ptr()) }
    }

    /// Consumes and flattens the view into a 1D [`VectorViewMut`] of length
    /// `self.len()`.
    #[inline]
    pub const fn flatten(self) -> VectorViewMut<'a, T> {
        // SAFETY: Any valid `TensorViewMut` validates at construction that the product
        // of `self.shape` does not overflow `usize`.
        let len = unsafe { unchecked_shape_len(&self.shape_raw()) };
        // SAFETY: `len` is non-zero and matches the exclusive element span of `self.ptr`.
        unsafe { TensorViewMut::from_raw_parts([len], self.ptr()) }
    }

    /// Consumes and slices the outermost axis (`axis 0`) by `range`, returning
    /// a mutable sub-view with the same rank `N`.
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
        // lie strictly within `self`'s exclusive span.
        unsafe { TensorViewMut::from_raw_parts(new_shape, ptr) }
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

        // SAFETY: `first_len` and `second_len` are non-zero and partition `axis0`
        // into disjoint exclusive spans.
        let first = unsafe { Self::from_raw_parts(shape1, self.ptr()) };
        // SAFETY: `ptr2` points to the disjoint trailing `second_len * inner` elements.
        let second = unsafe { Self::from_raw_parts(shape2, ptr2) };

        (first, second)
    }

    /// Consumes and partitions the mutable view into mutable sub-views of rank `M`
    /// spanning the trailing `M` dimensions.
    ///
    /// # Panics
    ///
    /// Panics at compile time if `M > N`.
    #[inline]
    #[track_caller]
    pub fn chunks<const M: usize>(self) -> ChunksMut<'a, T, M> {
        const {
            assert!(M <= N, "chunk rank M cannot exceed tensor rank N");
        }
        let shape = *self.shape_raw().last_chunk().unwrap();
        let inner = if M == 0 {
            ChunksInnerMut::Scalar(self.into_mut_slice().iter_mut())
        } else if M == N {
            ChunksInnerMut::Single(Some(self.into_mut_slice()))
        } else {
            // SAFETY: `shape` consists of the trailing `M` dimensions of `self.shape`,
            // so its product cannot overflow `usize` and is non-zero.
            let chunk_len = unsafe { unchecked_shape_len(&shape) };
            ChunksInnerMut::Multi(self.into_mut_slice().chunks_exact_mut(chunk_len.get()))
        };
        ChunksMut { inner, shape }
    }

    // -- Reductions ----------------------------------------------------------

    /// Mutates each element with `f` in place while reducing the returned
    /// values using the [`Monoid`] `op` across 16 independent accumulators.
    #[inline]
    #[must_use]
    pub fn reduce_by_mut<Out: Copy>(
        &mut self,
        op: impl Monoid<Out>,
        mut f: impl FnMut(&mut T) -> Out,
    ) -> Out {
        const LANES: usize = 16;
        let (chunks, rem) = self.as_mut_slice().as_chunks_mut::<LANES>();

        let mut acc = [op.identity(); LANES];
        for chunk in chunks {
            for l in 0..LANES {
                acc[l] = op.combine(acc[l], f(&mut chunk[l]));
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

    /// Mutates each elementwise pair `(a, b)` with `f` in place while reducing
    /// the returned values using `op` across 16 independent accumulators.
    ///
    /// # Panics
    ///
    /// Panics if `self.shape() != rhs.shape()`.
    #[inline]
    #[must_use]
    #[track_caller]
    pub fn zip_reduce_mut<U, Out: Copy>(
        &mut self,
        rhs: TensorView<'_, U, N>,
        op: impl Monoid<Out>,
        mut f: impl FnMut(&mut T, &U) -> Out,
    ) -> Out {
        const LANES: usize = 16;
        assert_eq!(self.shape(), rhs.shape(), "tensor shape mismatch");

        let (ac, ar) = self.as_mut_slice().as_chunks_mut::<LANES>();
        let (bc, br) = rhs.as_slice().as_chunks::<LANES>();

        let mut acc = [op.identity(); LANES];
        for (aa, ba) in zip(ac, bc) {
            for l in 0..LANES {
                acc[l] = op.combine(acc[l], f(&mut aa[l], &ba[l]));
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

impl<'a, T> ScalarViewMut<'a, T> {
    /// Creates a 0-dimensional [`ScalarViewMut`] borrowing a single mutable
    /// reference.
    #[inline]
    pub const fn scalar(val: &'a mut T) -> Self {
        // SAFETY: Shape `[]` has product 1 and `val` points to 1 exclusively borrowed `T` for `'a`.
        unsafe { Self::from_raw_parts([], NonNull::from_mut(val)) }
    }

    /// Consumes the view and returns the underlying mutable scalar reference.
    #[inline]
    #[must_use]
    pub const fn into_scalar(self) -> &'a mut T {
        // SAFETY: A rank-0 `ScalarViewMut` has shape `[]` (`len == 1`), so consuming
        // `self` transfers exclusive `'a` access to `self.ptr`.
        unsafe { self.ptr().as_mut() }
    }
}

impl<'a, T> VectorViewMut<'a, T> {
    /// Creates a 1-dimensional [`VectorViewMut`] borrowing a mutable slice.
    ///
    /// # Panics
    ///
    /// Panics if `val` is empty.
    #[inline]
    #[track_caller]
    pub const fn vector(val: &'a mut [T]) -> Self {
        Self::from_slice(val, [val.len()])
    }
}

impl<T> MatrixViewMut<'_, T> {
    /// Returns the number of rows (`shape()[0]`) in the mutable matrix view.
    #[inline]
    #[must_use]
    pub const fn nrows(&self) -> usize {
        self.shape_raw()[0].get()
    }

    /// Returns the number of columns (`shape()[1]`) in the mutable matrix view.
    #[inline]
    #[must_use]
    pub const fn ncols(&self) -> usize {
        self.shape_raw()[1].get()
    }
}

// ============================================================================
// Standard Trait Implementations
// ============================================================================

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

impl<T, const N: usize> Index<[usize; N]> for TensorViewMut<'_, T, N> {
    type Output = T;

    #[inline]
    #[track_caller]
    fn index(&self, index: [usize; N]) -> &Self::Output {
        let off = linear_offset(&self.shape_raw(), &index);
        // SAFETY: `linear_offset` bounds-checks every coordinate against `self.shape`,
        // so `off < self.len()` and `ptr` stays within the exclusive span.
        let ptr = unsafe { self.ptr().add(off) };
        // SAFETY: `ptr` points to a valid, initialized, exclusively borrowed `T`.
        unsafe { ptr.as_ref() }
    }
}

impl<T, const N: usize> IndexMut<[usize; N]> for TensorViewMut<'_, T, N> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        let off = linear_offset(&self.shape_raw(), &index);
        // SAFETY: `linear_offset` bounds-checks every coordinate against `self.shape`,
        // so `off < self.len()` and `ptr` stays within the exclusive span.
        let mut ptr = unsafe { self.ptr().add(off) };
        // SAFETY: `ptr` points to a valid, initialized, exclusively borrowed `T`.
        unsafe { ptr.as_mut() }
    }
}

impl<'a, T, const N: usize> IntoIterator for TensorViewMut<'a, T, N> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.into_mut_slice().iter_mut()
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

impl<T: Debug, const N: usize> Debug for TensorViewMut<'_, T, N> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorViewMut")
            .field("shape", &self.shape())
            .field("data", &self.as_slice())
            .finish()
    }
}

impl<T: Display, const N: usize> Display for TensorViewMut<'_, T, N> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.view().fmt(f)
    }
}

impl<T: PartialEq, const N: usize> PartialEq<TensorViewMut<'_, T, N>> for TensorViewMut<'_, T, N> {
    #[inline]
    fn eq(&self, other: &TensorViewMut<'_, T, N>) -> bool {
        self.view() == other.view()
    }
}

impl<T: Eq, const N: usize> Eq for TensorViewMut<'_, T, N> {}

impl<T: PartialEq, const N: usize> PartialEq<TensorViewMut<'_, T, N>> for TensorView<'_, T, N> {
    #[inline]
    fn eq(&self, other: &TensorViewMut<'_, T, N>) -> bool {
        self == &other.view()
    }
}

#[cfg(feature = "alloc")]
impl<T: PartialEq, const N: usize> PartialEq<TensorViewMut<'_, T, N>> for Tensor<T, N> {
    #[inline]
    fn eq(&self, other: &TensorViewMut<'_, T, N>) -> bool {
        self.view() == other.view()
    }
}

// ============================================================================
// ChunksMut Iterator
// ============================================================================

#[derive(Debug)]
enum ChunksInnerMut<'a, T> {
    Scalar(slice::IterMut<'a, T>),
    Single(Option<&'a mut [T]>),
    Multi(slice::ChunksExactMut<'a, T>),
}

/// An iterator over contiguous mutable rank-`M` sub-views of a [`TensorViewMut`].
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Debug)]
pub struct ChunksMut<'a, T, const M: usize> {
    inner: ChunksInnerMut<'a, T>,
    shape: [NonZeroUsize; M],
}

impl<'a, T, const M: usize> Iterator for ChunksMut<'a, T, M> {
    type Item = TensorViewMut<'a, T, M>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let data = match &mut self.inner {
            ChunksInnerMut::Scalar(iter) => slice::from_mut(iter.next()?),
            ChunksInnerMut::Single(slot) => slot.take()?,
            ChunksInnerMut::Multi(iter) => iter.next()?,
        };
        // SAFETY: `data.len()` matches the product of `self.shape` by `ChunksMut` construction.
        Some(unsafe { TensorViewMut::from_raw_parts(self.shape, NonNull::from_mut(data).cast()) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let data = match &mut self.inner {
            ChunksInnerMut::Scalar(iter) => slice::from_mut(iter.nth(n)?),
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
        // SAFETY: `data.len()` matches the product of `self.shape` by `ChunksMut` construction.
        Some(unsafe { TensorViewMut::from_raw_parts(self.shape, NonNull::from_mut(data).cast()) })
    }

    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let shape = self.shape;
        match self.inner {
            ChunksInnerMut::Scalar(iter) => iter.fold(init, |acc, elem| {
                // SAFETY: `shape` is `[]` (len 1) and `elem` is a valid `&'a mut T`.
                let view = unsafe { TensorViewMut::from_raw_parts(shape, NonNull::from_mut(elem)) };
                f(acc, view)
            }),
            ChunksInnerMut::Single(Some(data)) => {
                // SAFETY: `data.len()` matches the product of `shape` by `ChunksMut` construction.
                let view =
                    unsafe { TensorViewMut::from_raw_parts(shape, NonNull::from_mut(data).cast()) };
                f(init, view)
            }
            ChunksInnerMut::Single(None) => init,
            ChunksInnerMut::Multi(iter) => iter.fold(init, |acc, data| {
                // SAFETY: `data.len()` matches the product of `shape` by `ChunksMut` construction.
                let view =
                    unsafe { TensorViewMut::from_raw_parts(shape, NonNull::from_mut(data).cast()) };
                f(acc, view)
            }),
        }
    }
}

impl<T, const M: usize> DoubleEndedIterator for ChunksMut<'_, T, M> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let data = match &mut self.inner {
            ChunksInnerMut::Scalar(iter) => slice::from_mut(iter.next_back()?),
            ChunksInnerMut::Single(slot) => slot.take()?,
            ChunksInnerMut::Multi(iter) => iter.next_back()?,
        };
        // SAFETY: `data.len()` matches the product of `self.shape` by `ChunksMut` construction.
        Some(unsafe { TensorViewMut::from_raw_parts(self.shape, NonNull::from_mut(data).cast()) })
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let data = match &mut self.inner {
            ChunksInnerMut::Scalar(iter) => slice::from_mut(iter.nth_back(n)?),
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
        // SAFETY: `data.len()` matches the product of `self.shape` by `ChunksMut` construction.
        Some(unsafe { TensorViewMut::from_raw_parts(self.shape, NonNull::from_mut(data).cast()) })
    }

    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let shape = self.shape;
        match self.inner {
            ChunksInnerMut::Scalar(iter) => iter.rfold(init, |acc, elem| {
                // SAFETY: `shape` is `[]` (len 1) and `elem` is a valid `&'a mut T`.
                let view = unsafe { TensorViewMut::from_raw_parts(shape, NonNull::from_mut(elem)) };
                f(acc, view)
            }),
            ChunksInnerMut::Single(Some(data)) => {
                // SAFETY: `data.len()` matches the product of `shape` by `ChunksMut` construction.
                let view =
                    unsafe { TensorViewMut::from_raw_parts(shape, NonNull::from_mut(data).cast()) };
                f(init, view)
            }
            ChunksInnerMut::Single(None) => init,
            ChunksInnerMut::Multi(iter) => iter.rfold(init, |acc, data| {
                // SAFETY: `data.len()` matches the product of `shape` by `ChunksMut` construction.
                let view =
                    unsafe { TensorViewMut::from_raw_parts(shape, NonNull::from_mut(data).cast()) };
                f(acc, view)
            }),
        }
    }
}

impl<T, const M: usize> ExactSizeIterator for ChunksMut<'_, T, M> {
    #[inline]
    fn len(&self) -> usize {
        match &self.inner {
            ChunksInnerMut::Scalar(iter) => iter.len(),
            ChunksInnerMut::Single(slot) => usize::from(slot.is_some()),
            ChunksInnerMut::Multi(iter) => iter.len(),
        }
    }
}

impl<T, const M: usize> FusedIterator for ChunksMut<'_, T, M> {}
