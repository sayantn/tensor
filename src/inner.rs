use core::marker::PhantomData;
use core::num::NonZeroUsize;
use core::ptr::NonNull;

#[cfg(feature = "alloc")]
use alloc::boxed::Box;

// ============================================================================
// TensorView
// ============================================================================

/// An immutable, zero-copy view over a contiguous row-major tensor of rank `N`.
///
/// `TensorView` implements [`Copy`] and [`Clone`] regardless of whether `T`
/// implements them.
#[must_use]
pub struct TensorView<'a, T, const N: usize> {
    shape: [NonZeroUsize; N],
    ptr: NonNull<T>,
    _marker: PhantomData<&'a T>,
}

// SAFETY: `TensorView<'a, T, N>` acts identically to `&'a [T]`.
unsafe impl<T: Sync, const N: usize> Send for TensorView<'_, T, N> {}
// SAFETY: `TensorView<'a, T, N>` provides only shared immutable access to `T`.
unsafe impl<T: Sync, const N: usize> Sync for TensorView<'_, T, N> {}

impl<T, const N: usize> TensorView<'_, T, N> {
    /// Creates a [`TensorView`] directly from a `shape` and non-null base pointer.
    ///
    /// # Safety
    ///
    /// - The product of `shape` must not overflow `usize`.
    /// - `ptr` must be valid for reads of `shape.iter().product()` contiguous `T`
    ///   elements for the entire lifetime `'a`, properly aligned, and point to
    ///   initialized values of type `T`.
    #[inline]
    pub(crate) const unsafe fn from_raw_parts(shape: [NonZeroUsize; N], ptr: NonNull<T>) -> Self {
        Self {
            shape,
            ptr,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub(crate) const fn shape_raw(&self) -> [NonZeroUsize; N] {
        self.shape
    }

    #[inline]
    pub(crate) const fn ptr(&self) -> NonNull<T> {
        self.ptr
    }
}

// ============================================================================
// TensorViewMut
// ============================================================================

/// A mutable, zero-copy view over a contiguous row-major tensor of rank `N`.
#[must_use]
pub struct TensorViewMut<'a, T, const N: usize> {
    shape: [NonZeroUsize; N],
    ptr: NonNull<T>,
    _marker: PhantomData<&'a mut T>,
}

// SAFETY: `TensorViewMut<'a, T, N>` acts identically to `&'a mut [T]`.
unsafe impl<T: Send, const N: usize> Send for TensorViewMut<'_, T, N> {}
// SAFETY: `TensorViewMut<'a, T, N>` acts identically to `&'a mut [T]`.
unsafe impl<T: Sync, const N: usize> Sync for TensorViewMut<'_, T, N> {}

impl<T, const N: usize> TensorViewMut<'_, T, N> {
    /// Creates a [`TensorViewMut`] directly from a `shape` and non-null base pointer.
    ///
    /// # Safety
    ///
    /// - The product of `shape` must not overflow `usize` (or `isize::MAX` bytes).
    /// - `ptr` must be valid for reads and writes of `shape.iter().product()`
    ///   contiguous `T` elements for the entire lifetime `'a`, properly aligned,
    ///   non-aliased (exclusively borrowed), and point to initialized values of type `T`.
    #[inline]
    pub(crate) const unsafe fn from_raw_parts(shape: [NonZeroUsize; N], ptr: NonNull<T>) -> Self {
        Self {
            shape,
            ptr,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub(crate) const fn shape_raw(&self) -> [NonZeroUsize; N] {
        self.shape
    }

    #[inline]
    pub(crate) const fn ptr(&self) -> NonNull<T> {
        self.ptr
    }
}

// ============================================================================
// Tensor
// ============================================================================

/// An owned, heap-allocated, row-major dense tensor of rank `N`.
///
/// All dimensions must be strictly non-zero (`> 0`). Rank-0 scalars are
/// represented as `Tensor<T, 0>` with shape `[]` and a single element.
#[cfg(feature = "alloc")]
#[must_use]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor<T, const N: usize> {
    shape: [NonZeroUsize; N],
    data: Box<[T]>,
}

#[cfg(feature = "alloc")]
impl<T, const N: usize> Tensor<T, N> {
    /// Creates a [`Tensor`] directly from a `shape` and boxed slice.
    ///
    /// # Safety
    ///
    /// - The product of `shape` must not overflow `usize`.
    /// - `data.len()` must equal the product of `shape`.
    #[inline]
    pub(crate) const unsafe fn from_raw_parts(shape: [NonZeroUsize; N], data: Box<[T]>) -> Self {
        Self { shape, data }
    }

    #[inline]
    pub(crate) const fn shape_raw(&self) -> [NonZeroUsize; N] {
        self.shape
    }

    #[inline]
    pub(crate) const fn data(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub(crate) const fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    #[inline]
    pub(crate) fn into_data(self) -> Box<[T]> {
        self.data
    }
}
