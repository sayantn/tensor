use crate::{MatrixView, MatrixViewMut, VectorView, VectorViewMut};

use core::iter::zip;
use core::ops::{Add, AddAssign, Mul, MulAssign};
use num_traits::{One, Zero};

mod gemm;
pub use gemm::*;

pub trait Element:
    Add<Output = Self> + AddAssign + Mul<Output = Self> + MulAssign + Copy + PartialEq + Zero + One
{
}
impl<
    T: Mul<Output = Self> + MulAssign + Add<Output = Self> + AddAssign + Copy + PartialEq + Zero + One,
> Element for T
{
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MatrixOp {
    NoTrans,
    Trans,
}

#[cfg(feature = "backend-blas")]
impl From<MatrixOp> for cblas_sys::CBLAS_TRANSPOSE {
    fn from(op: MatrixOp) -> Self {
        match op {
            MatrixOp::NoTrans => Self::CblasNoTrans,
            MatrixOp::Trans => Self::CblasTrans,
        }
    }
}

#[inline]
fn op_shape(shape: [usize; 2], op: MatrixOp) -> [usize; 2] {
    match op {
        MatrixOp::NoTrans => shape,
        MatrixOp::Trans => [shape[1], shape[0]],
    }
}

#[inline(always)]
pub(crate) fn axpy_slice<T: Element>(alpha: T, x: &[T], y: &mut [T]) {
    debug_assert_eq!(x.len(), y.len());
    if alpha.is_zero() {
        return;
    }
    for (yi, &xi) in zip(y, x) {
        *yi += alpha * xi;
    }
}

#[inline]
#[track_caller]
pub fn axpy<T: Element>(alpha: T, x: VectorView<'_, T>, mut y: VectorViewMut<'_, T>) {
    assert_eq!(x.len(), y.len(), "vector length mismatch");
    axpy_slice(alpha, x.as_slice(), y.as_mut_slice());
}

#[inline(always)]
pub(crate) fn dot_slice<T: Element>(x: &[T], y: &[T]) -> T {
    const LANES: usize = 16;
    debug_assert_eq!(x.len(), y.len());

    let (xc, xr) = x.as_chunks::<LANES>();
    let (yc, yr) = y.as_chunks::<LANES>();

    let mut acc = [T::zero(); LANES];
    for (xa, ya) in zip(xc, yc) {
        for l in 0..LANES {
            acc[l] += xa[l] * ya[l];
        }
    }

    let mut sum = T::zero();
    for v in acc {
        sum += v;
    }
    for (&xv, &yv) in zip(xr, yr) {
        sum += xv * yv;
    }
    sum
}

/// Dot product with independent accumulators (breaks the add dependency chain and lets LLVM
/// vectorize without needing float reassociation).
#[inline]
#[must_use]
#[track_caller]
pub fn dot<T: Element>(x: VectorView<'_, T>, y: VectorView<'_, T>) -> T {
    assert_eq!(x.len(), y.len(), "vector length mismatch");
    dot_slice(x.as_slice(), y.as_slice())
}

#[inline]
#[track_caller]
pub fn gemv<T: Element>(
    alpha: T,
    (a, ta): (MatrixView<'_, T>, MatrixOp),
    x: VectorView<'_, T>,
    beta: T,
    mut y: VectorViewMut<'_, T>,
) {
    let [m, n] = op_shape(a.shape(), ta);
    assert_eq!(n, x.len(), "inner dimension mismatch");
    assert_eq!(m, y.len(), "output shape mismatch");

    if beta.is_zero() {
        y.fill(T::zero());
    } else if beta != T::one() {
        y *= beta;
    }

    if alpha.is_zero() {
        return;
    }

    let (a_slice, x_slice, y_slice) = (a.as_slice(), x.as_slice(), y.as_mut_slice());

    match ta {
        MatrixOp::NoTrans => {
            for (yi, a_row) in zip(y_slice, a_slice.chunks_exact(a.ncols())) {
                *yi += alpha * dot_slice(a_row, x_slice);
            }
        }
        MatrixOp::Trans => {
            for (&xp, a_row) in zip(x_slice, a_slice.chunks_exact(a.ncols())) {
                axpy_slice(alpha * xp, a_row, y_slice);
            }
        }
    }
}

#[inline]
#[track_caller]
pub fn sum_axis0<T: Element>(x: MatrixView<'_, T>, beta: T, mut y: VectorViewMut<'_, T>) {
    assert_eq!(x.ncols(), y.len(), "output length mismatch");

    if beta.is_zero() {
        y.fill(T::zero());
    } else if beta != T::one() {
        y *= beta;
    }

    let y_slice = y.as_mut_slice();
    for x_row in x.as_slice().chunks_exact(x.ncols()) {
        for (yj, &xij) in zip(&mut *y_slice, x_row) {
            *yj += xij;
        }
    }
}

#[inline]
#[track_caller]
pub fn ger<T: Element>(
    alpha: T,
    x: VectorView<'_, T>,
    y: VectorView<'_, T>,
    mut a: MatrixViewMut<'_, T>,
) {
    assert_eq!(
        [x.len(), y.len()],
        a.shape(),
        "outer product shape mismatch"
    );

    if alpha.is_zero() {
        return;
    }

    let (x_slice, y_slice, a_slice) = (x.as_slice(), y.as_slice(), a.as_mut_slice());
    for (&xi, a_row) in zip(x_slice, a_slice.chunks_exact_mut(y_slice.len())) {
        axpy_slice(alpha * xi, y_slice, a_row);
    }
}
