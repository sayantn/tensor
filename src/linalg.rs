//! BLAS-style Level-1, Level-2, and Level-3 dense linear algebra routines over
//! [`VectorView`], [`VectorViewMut`], [`MatrixView`], and [`MatrixViewMut`].

use crate::{
    MatrixView, MatrixViewMut, TensorView, TensorViewMut, VectorView, VectorViewMut,
    unchecked_shape_len,
};

use core::array;
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

    y.chunks::<M>()
        .for_each(|chunk| axpy_slice(alpha, x.as_slice(), chunk.into_mut_slice()));
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
            for (yi, a_row) in zip(y_slice, a.chunks::<1>()) {
                *yi += alpha * dot_slice(a_row.as_slice(), x_slice);
            }
        }
        MatrixOp::Trans => {
            let ncols = a.ncols();
            let (x_blocks, x_rem) = x_slice.as_chunks::<4>();
            let mut a_blocks = a_slice.chunks_exact(4 * ncols);

            for (xb, ab) in zip(x_blocks, &mut a_blocks) {
                let ax0 = alpha * xb[0];
                let ax1 = alpha * xb[1];
                let ax2 = alpha * xb[2];
                let ax3 = alpha * xb[3];

                let (r0, rest) = ab.split_at(ncols);
                let (r1, rest) = rest.split_at(ncols);
                let (r2, r3) = rest.split_at(ncols);

                for (yj, ((&a0, &a1), (&a2, &a3))) in
                    zip(&mut *y_slice, zip(zip(r0, r1), zip(r2, r3)))
                {
                    *yj += (ax0 * a0 + ax1 * a1) + (ax2 * a2 + ax3 * a3);
                }
            }

            for (&xp, a_row) in zip(x_rem, a_blocks.remainder().chunks_exact(ncols)) {
                axpy_slice(alpha * xp, a_row, y_slice);
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::inline_always)]
pub(crate) fn sum_slice<T: Element>(x: &[T]) -> T {
    const LANES: usize = 16;

    let (xc, xr) = x.as_chunks::<LANES>();
    let mut acc = [T::zero(); LANES];
    for xa in xc {
        for l in 0..LANES {
            acc[l] += xa[l];
        }
    }

    let mut sum = T::zero();
    for v in acc {
        sum += v;
    }
    for &xv in xr {
        sum += xv;
    }
    sum
}

#[inline(always)]
#[allow(clippy::inline_always)]
fn sum_rows_into<T: Element>(x: &[T], ncols: usize, y: &mut [T]) {
    debug_assert_eq!(y.len(), ncols);
    debug_assert_eq!(x.len() % ncols, 0);

    let mut blocks = x.chunks_exact(4 * ncols);
    for block in &mut blocks {
        let (r0, rest) = block.split_at(ncols);
        let (r1, rest) = rest.split_at(ncols);
        let (r2, r3) = rest.split_at(ncols);

        for (y, ((&r0, &r1), (&r2, &r3))) in zip(&mut *y, zip(zip(r0, r1), zip(r2, r3))) {
            *y += (r0 + r1) + (r2 + r3);
        }
    }

    for row in blocks.remainder().chunks_exact(ncols) {
        for (yj, &xij) in zip(&mut *y, row) {
            *yj += xij;
        }
    }
}

/// Accumulates the sum of `x` along `axis` into `y`:
/// `y[..., j, ...] := sum_i x[..., i, ...] + beta * y[..., j, ...]`.
///
/// # Panics
///
/// Panics at compile time if `M + 1 != N`, or at runtime if `axis >= N` or if
/// `y.shape()` does not match `x.shape()` with `axis` removed.
#[inline]
#[track_caller]
pub fn sum_axis<T: Element, const N: usize, const M: usize>(
    x: TensorView<'_, T, N>,
    axis: usize,
    beta: T,
    mut y: TensorViewMut<'_, T, M>,
) {
    const {
        assert!(M + 1 == N, "output rank M must equal input rank N - 1");
    }
    assert!(axis < N, "axis {axis} out of bounds for rank {N}");

    let x_shape = x.shape();
    let expected_shape: [usize; M] =
        array::from_fn(|i| if i < axis { x_shape[i] } else { x_shape[i + 1] });
    assert_eq!(y.shape(), expected_shape, "output shape mismatch");

    if beta.is_zero() {
        y.fill(T::zero());
    } else if beta != T::one() {
        y *= beta;
    }

    let axis_len = x_shape[axis];
    // SAFETY: this is a subslice of `x_shape`
    let inner = unsafe { unchecked_shape_len(&x_shape[axis + 1..]) };
    let x_slice = x.as_slice();
    let y_slice = y.as_mut_slice();

    if inner == 1 {
        for (yi, row) in zip(y_slice, x.chunks::<1>()) {
            *yi += sum_slice(row.as_slice());
        }
    } else if axis == 0 {
        sum_rows_into(x_slice, inner, y_slice);
    } else {
        // `axis < N - 1`: each outer block is a `[axis_len, inner]` slice reduced along axis 0.
        for (y_block, x_block) in zip(
            y_slice.chunks_exact_mut(inner),
            x_slice.chunks_exact(axis_len * inner),
        ) {
            sum_rows_into(x_block, inner, y_block);
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
    a: MatrixViewMut<'_, T>,
) {
    assert_eq!(
        [x.len(), y.len()],
        a.shape(),
        "outer product shape mismatch"
    );

    if alpha.is_zero() {
        return;
    }

    for (&xi, a_row) in zip(x.as_slice(), a.chunks::<1>()) {
        axpy_slice(alpha * xi, y.as_slice(), a_row.into_mut_slice());
    }
}
