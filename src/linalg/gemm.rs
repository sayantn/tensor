use super::{Element, MatrixOp, axpy_slice, dot_slice, op_shape};
use crate::{MatrixView, MatrixViewMut};

use core::iter::zip;

/// Backend abstraction for general matrix-matrix multiplication (`GEMM`):
/// `c := alpha * op(a) * op(b) + beta * c`.
pub trait GemmBackend<T> {
    /// Performs `c := alpha * op(a) * op(b) + beta * c`.
    ///
    /// # Panics
    ///
    /// Panics if the inner dimensions of `op(a)` and `op(b)` do not match, or
    /// if `c.shape()` does not match `[m, n]`.
    fn gemm(
        &self,
        alpha: T,
        a: (MatrixView<'_, T>, MatrixOp),
        b: (MatrixView<'_, T>, MatrixOp),
        beta: T,
        c: MatrixViewMut<'_, T>,
    );
}

/// Pure-Rust reference [`GemmBackend`].
///
/// Works for any scalar type implementing [`Element`] in `#![no_std]`
/// environments without external crates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Naive;

/// [`GemmBackend`] backed by system CBLAS (`cblas_sgemm` / `cblas_dgemm`).
#[cfg(feature = "backend-blas")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Blas;

/// [`GemmBackend`] backed by the pure-Rust `matrixmultiply` crate (`f32` and `f64`).
#[cfg(feature = "backend-matrixmultiply")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MatrixMultiplyCrate;

/// [`GemmBackend`] backed by the pure-Rust `gemm` crate (`f32`, `f64`, and
/// optionally `half::f16`).
#[cfg(feature = "backend-gemm")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GemmCrate;

/// Adaptive [`GemmBackend`] that dispatches to `Small` when `m * n * k <= cutoff`
/// and to `Large` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cutoff<Small, Large> {
    #[allow(clippy::struct_field_names)]
    cutoff: usize,
    small: Small,
    large: Large,
}

impl<Small, Large> Cutoff<Small, Large> {
    /// Creates a [`Cutoff`] backend with explicit `small` and `large` backend
    /// instances.
    #[inline]
    #[must_use]
    pub const fn with(cutoff: usize, small: Small, large: Large) -> Self {
        Self {
            cutoff,
            small,
            large,
        }
    }
}

impl<Small: Default, Large: Default> Cutoff<Small, Large> {
    /// Creates a [`Cutoff`] backend using [`Default`] instances of `Small` and
    /// `Large`.
    #[inline]
    #[must_use]
    pub fn new(cutoff: usize) -> Self {
        Self {
            cutoff,
            small: Small::default(),
            large: Large::default(),
        }
    }
}

#[inline]
#[track_caller]
fn check_shapes<T>(
    (a, ta): (MatrixView<'_, T>, MatrixOp),
    (b, tb): (MatrixView<'_, T>, MatrixOp),
    c: &MatrixViewMut<'_, T>,
) -> [usize; 3] {
    let [m, k] = op_shape(a.shape(), ta);
    let [kb, n] = op_shape(b.shape(), tb);
    assert_eq!(k, kb, "inner dimension mismatch");
    assert_eq!([m, n], c.shape(), "output shape mismatch");
    [m, n, k]
}

/// `(row stride, col stride)` of the logical `op(x)` for a row-major `x` with `ld` columns.
#[inline]
#[allow(unused)]
const fn strides(ld: usize, op: MatrixOp) -> (usize, usize) {
    match op {
        MatrixOp::NoTrans => (ld, 1),
        MatrixOp::Trans => (1, ld),
    }
}

macro_rules! impl_for_blas {
    ($ty:ty : $fn:ident) => {
        #[cfg(feature = "backend-blas")]
        impl GemmBackend<$ty> for Blas {
            #[inline]
            #[track_caller]
            fn gemm(
                &self,
                alpha: $ty,
                (a, ta): (MatrixView<'_, $ty>, MatrixOp),
                (b, tb): (MatrixView<'_, $ty>, MatrixOp),
                beta: $ty,
                mut c: MatrixViewMut<'_, $ty>,
            ) {
                let [m, n, k] = check_shapes((a, ta), (b, tb), &c);

                assert!(
                    m | n | k <= core::ffi::c_int::MAX as usize,
                    "Matrix dimension too large for BLAS"
                );

                let ta = match ta {
                    MatrixOp::NoTrans => cblas_sys::CblasNoTrans,
                    MatrixOp::Trans => cblas_sys::CblasTrans,
                };
                let tb = match tb {
                    MatrixOp::NoTrans => cblas_sys::CblasNoTrans,
                    MatrixOp::Trans => cblas_sys::CblasTrans,
                };

                // SAFETY: Shapes and leading dimensions are validated by `check_shapes`
                // and fit within `c_int`.
                #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                unsafe {
                    cblas_sys::$fn(
                        cblas_sys::CblasRowMajor,
                        ta,
                        tb,
                        m as _,
                        n as _,
                        k as _,
                        alpha,
                        a.as_ptr(),
                        a.ncols() as _,
                        b.as_ptr(),
                        b.ncols() as _,
                        beta,
                        c.as_mut_ptr(),
                        c.ncols() as _,
                    );
                }
            }
        }
    };
}

impl_for_blas!(f32: cblas_sgemm);
impl_for_blas!(f64: cblas_dgemm);

macro_rules! impl_for_matrixmultiply_crate {
    ($ty:ty : $fn:ident) => {
        #[cfg(feature = "backend-matrixmultiply")]
        impl GemmBackend<$ty> for MatrixMultiplyCrate {
            #[inline]
            #[track_caller]
            fn gemm(
                &self,
                alpha: $ty,
                (a, ta): (MatrixView<'_, $ty>, MatrixOp),
                (b, tb): (MatrixView<'_, $ty>, MatrixOp),
                beta: $ty,
                mut c: MatrixViewMut<'_, $ty>,
            ) {
                let [m, n, k] = check_shapes((a, ta), (b, tb), &c);

                let (rsa, csa) = strides(a.ncols(), ta);
                let (rsb, csb) = strides(b.ncols(), tb);

                // SAFETY: Shapes and strides match the validated `a`, `b`, and `c` buffers.
                #[allow(clippy::cast_possible_wrap)]
                unsafe {
                    matrixmultiply::$fn(
                        m,
                        k,
                        n,
                        alpha,
                        a.as_ptr(),
                        rsa as _,
                        csa as _,
                        b.as_ptr(),
                        rsb as _,
                        csb as _,
                        beta,
                        c.as_mut_ptr(),
                        c.ncols() as _,
                        1,
                    );
                }
            }
        }
    };
}

impl_for_matrixmultiply_crate!(f32: sgemm);
impl_for_matrixmultiply_crate!(f64: dgemm);

macro_rules! impl_for_gemm_crate {
    (#[$meta:meta] $ty:ty) => {
        #[$meta]
        impl GemmBackend<$ty> for GemmCrate {
            #[inline]
            #[track_caller]
            fn gemm(
                &self,
                alpha: $ty,
                (a, ta): (MatrixView<'_, $ty>, MatrixOp),
                (b, tb): (MatrixView<'_, $ty>, MatrixOp),
                beta: $ty,
                mut c: MatrixViewMut<'_, $ty>,
            ) {
                let [m, n, k] = check_shapes((a, ta), (b, tb), &c);

                let (rsa, csa) = strides(a.ncols(), ta);
                let (rsb, csb) = strides(b.ncols(), tb);

                // SAFETY: Shapes and strides match the validated `a`, `b`, and `c` buffers.
                #[allow(clippy::cast_possible_wrap)]
                unsafe {
                    gemm::gemm(
                        m,
                        n,
                        k,
                        c.as_mut_ptr(),
                        1,
                        c.ncols() as _,
                        !num_traits::Zero::is_zero(&beta),
                        a.as_ptr(),
                        csa as _,
                        rsa as _,
                        b.as_ptr(),
                        csb as _,
                        rsb as _,
                        beta,
                        alpha,
                        false,
                        false,
                        false,
                        gemm::Parallelism::None,
                    );
                }
            }
        }
    };
}

impl_for_gemm_crate!(
    #[cfg(feature = "backend-gemm")]
    f32
);
impl_for_gemm_crate!(
    #[cfg(feature = "backend-gemm")]
    f64
);
impl_for_gemm_crate!(
    #[cfg(feature = "gemm-f16")]
    half::f16
);

impl<T, Small: GemmBackend<T>, Large: GemmBackend<T>> GemmBackend<T> for Cutoff<Small, Large> {
    #[inline]
    #[track_caller]
    fn gemm(
        &self,
        alpha: T,
        a: (MatrixView<'_, T>, MatrixOp),
        b: (MatrixView<'_, T>, MatrixOp),
        beta: T,
        c: MatrixViewMut<'_, T>,
    ) {
        let [m, n, k] = check_shapes(a, b, &c);

        if m.saturating_mul(n).saturating_mul(k) <= self.cutoff {
            self.small.gemm(alpha, a, b, beta, c);
        } else {
            self.large.gemm(alpha, a, b, beta, c);
        }
    }
}

impl<T: Element> GemmBackend<T> for Naive {
    #[inline]
    #[track_caller]
    fn gemm(
        &self,
        alpha: T,
        (a, ta): (MatrixView<'_, T>, MatrixOp),
        (b, tb): (MatrixView<'_, T>, MatrixOp),
        beta: T,
        mut c: MatrixViewMut<'_, T>,
    ) {
        let [m, n, k] = check_shapes((a, ta), (b, tb), &c);

        if beta.is_zero() {
            c.fill(T::zero());
        } else if beta != T::one() {
            c *= beta;
        }

        if alpha.is_zero() {
            return;
        }

        let (a, b, c) = (a.as_slice(), b.as_slice(), c.as_mut_slice());

        match (ta, tb) {
            (MatrixOp::NoTrans, MatrixOp::Trans) => {
                for (c_row, a_row) in zip(c.chunks_exact_mut(n), a.chunks_exact(k)) {
                    for (cij, b_row) in zip(c_row, b.chunks_exact(k)) {
                        *cij += alpha * dot_slice(a_row, b_row);
                    }
                }
            }
            (_, MatrixOp::NoTrans) => {
                let (rsa, csa) = match ta {
                    MatrixOp::NoTrans => (k, 1),
                    MatrixOp::Trans => (1, m),
                };

                for (i, c_row) in c.chunks_exact_mut(n).enumerate() {
                    for (p, b_row) in b.chunks_exact(n).enumerate() {
                        // SAFETY: i < m and p < k, and (rsa, csa) index the m x k logical view of
                        // `a`, which has exactly m * k elements (validated by `check_shapes`).
                        let a_ip = alpha * unsafe { *a.get_unchecked(i * rsa + p * csa) };
                        axpy_slice(a_ip, b_row, c_row);
                    }
                }
            }
            (MatrixOp::Trans, MatrixOp::Trans) => {
                for (i, c_row) in c.chunks_exact_mut(n).enumerate() {
                    for p in 0..k {
                        // SAFETY: i < m, p < k; a has k * m elements.
                        let a_ip = alpha * unsafe { *a.get_unchecked(p * m + i) };
                        for (j, cj) in c_row.iter_mut().enumerate() {
                            // SAFETY: p < k, j < n; b has n * k elements.
                            *cj += a_ip * unsafe { *b.get_unchecked(j * k + p) };
                        }
                    }
                }
            }
        }
    }
}
