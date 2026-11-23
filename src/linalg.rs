//! BLAS-style Level-1, Level-2, and Level-3 dense linear algebra routines over
//! [`VectorView`], [`VectorViewMut`], [`MatrixView`], and [`MatrixViewMut`].

use crate::{MatrixView, MatrixViewMut, TensorView, TensorViewMut, VectorView, VectorViewMut};

use core::iter::zip;
use core::ops::{Add, AddAssign, Mul, MulAssign};
use num_traits::{One, Zero};

mod gemm;
pub use gemm::*;

/// Scalar ring-like element trait required by generic linear algebra kernels.
pub trait Element:
    Add<Output = Self> + AddAssign + Mul<Output = Self> + MulAssign + Copy + PartialEq + Zero + One
{
}

impl<
    T: Mul<Output = Self> + MulAssign + Add<Output = Self> + AddAssign + Copy + PartialEq + Zero + One,
> Element for T
{
}

/// Optional transposition applied logically to a [`MatrixView`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MatrixOp {
    /// Use the matrix as stored (`[rows, cols]`).
    NoTrans,
    /// Logically transpose the matrix (`[cols, rows]`).
    Trans,
}

#[inline]
const fn op_shape(shape: [usize; 2], op: MatrixOp) -> [usize; 2] {
    match op {
        MatrixOp::NoTrans => shape,
        MatrixOp::Trans => [shape[1], shape[0]],
    }
}

#[inline(always)]
#[allow(clippy::inline_always)]
pub(crate) fn axpy_slice<T: Element>(alpha: T, x: &[T], y: &mut [T]) {
    debug_assert_eq!(x.len(), y.len());
    if alpha.is_zero() {
        return;
    }
    for (yi, &xi) in zip(y, x) {
        *yi += alpha * xi;
    }
}

/// Computes `y := alpha * x + y` (BLAS Level-1 `axpy`).
///
/// # Panics
///
/// Panics if `x.shape() != y.shape()`.
#[inline]
#[track_caller]
pub fn axpy<T: Element, const N: usize, const M: usize>(
    alpha: T,
    x: TensorView<'_, T, M>,
    y: TensorViewMut<'_, T, N>,
) {
    const {
        assert!(
            M <= N,
            "cannot broadcast higher-rank tensor into lower-rank tensor"
        );
    }
    assert_eq!(x.shape(), y.shape()[N - M..], "tensor shape mismatch");

    if alpha.is_zero() {
        return;
    }

    for mut chunk in y.chunks::<M>() {
        axpy_slice(alpha, x.as_slice(), chunk.as_mut_slice());
    }
}

#[inline(always)]
#[allow(clippy::inline_always)]
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

/// Computes the inner product `x^T y` (BLAS Level-1 `dot`) using 16
/// independent accumulators to break loop-carried dependency chains and
/// enable SIMD vectorization without floating-point reassociation flags.
///
/// # Panics
///
/// Panics if `x.len() != y.len()`.
#[inline]
#[must_use]
#[track_caller]
pub fn dot<T: Element>(x: VectorView<'_, T>, y: VectorView<'_, T>) -> T {
    assert_eq!(x.len(), y.len(), "vector length mismatch");
    dot_slice(x.as_slice(), y.as_slice())
}

/// Computes `y := alpha * op(a) * x + beta * y` (BLAS Level-2 `gemv`).
///
/// # Panics
///
/// Panics if the logical column count of `op(a)` does not equal `x.len()` or
/// if the logical row count of `op(a)` does not equal `y.len()`.
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

/// Accumulates the column-wise sum of `x` into `y`:
/// `y[j] := sum_i x[i, j] + beta * y[j]`.
///
/// Processes rows in unrolled 4-row blocks to improve instruction-level
/// parallelism and reduce write traffic to `y`.
///
/// # Panics
///
/// Panics if `x.ncols() != y.len()`.
#[inline]
#[track_caller]
pub fn sum_axis0<T: Element>(x: MatrixView<'_, T>, beta: T, mut y: VectorViewMut<'_, T>) {
    assert_eq!(x.ncols(), y.len(), "output length mismatch");

    if beta.is_zero() {
        y.fill(T::zero());
    } else if beta != T::one() {
        y *= beta;
    }

    let ncols = x.ncols();
    let y_slice = y.as_mut_slice();
    let mut blocks = x.as_slice().chunks_exact(4 * ncols);

    for block in &mut blocks {
        let (r0, rest) = block.split_at(ncols);
        let (r1, rest) = rest.split_at(ncols);
        let (r2, r3) = rest.split_at(ncols);

        for j in 0..ncols {
            y_slice[j] += (r0[j] + r1[j]) + (r2[j] + r3[j]);
        }
    }

    for row in blocks.remainder().chunks_exact(ncols) {
        for (yj, &xij) in zip(&mut *y_slice, row) {
            *yj += xij;
        }
    }
}

/// Computes the rank-1 outer product update `a := alpha * x * y^T + a` (BLAS
/// Level-2 `ger`).
///
/// # Panics
///
/// Panics if `[x.len(), y.len()] != a.shape()`.
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
