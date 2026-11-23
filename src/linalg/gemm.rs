use super::{Element, MatrixOp, axpy_slice, dot_slice, op_shape};
use crate::{MatrixView, MatrixViewMut};

use core::iter::zip;

pub trait GemmBackend<T> {
    fn gemm(
        &self,
        alpha: T,
        a: (MatrixView<'_, T>, MatrixOp),
        b: (MatrixView<'_, T>, MatrixOp),
        beta: T,
        c: MatrixViewMut<'_, T>,
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Naive;

#[cfg(feature = "backend-blas")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Blas;

#[cfg(feature = "backend-matrixmultiply")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MatrixMultiplyCrate;

#[cfg(feature = "backend-gemm")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GemmCrate;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cutoff<Small, Large> {
    cutoff: usize,
    small: Small,
    large: Large,
}

impl<Small, Large> Cutoff<Small, Large> {
    #[inline]
    pub const fn with(cutoff: usize, small: Small, large: Large) -> Self {
        Self {
            cutoff,
            small,
            large,
        }
    }
}

impl<Small: Default, Large: Default> Cutoff<Small, Large> {
    #[inline]
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
fn strides(ld: usize, op: MatrixOp) -> (usize, usize) {
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
                    (m | n | k) <= core::ffi::c_int::MAX as usize,
                    "Matrix dimension too large for BLAS"
                );

                unsafe {
                    cblas_sys::$fn(
                        cblas_sys::CblasRowMajor,
                        ta.into(),
                        tb.into(),
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

                unsafe {
                    gemm::gemm(
                        m,
                        n,
                        k,
                        c.as_mut_ptr(),
                        1,
                        c.ncols() as _,
                        beta != num_traits::Zero::zero(),
                        a.as_ptr(),
                        csa as _,
                        rsa as _,
                        b.as_ptr(),
                        csb as _,
                        rsb as _,
                        alpha,
                        beta,
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
    #[cfg(all(feature = "backend-gemm", feature = "f16"))]
    half::f16
);
impl_for_gemm_crate!(
    #[cfg(all(feature = "backend-gemm", feature = "f16"))]
    half::bf16
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
                        // SAFETY: i < m, p < k, j < n; a has k * m elements, b has n * k.
                        let a_ip = alpha * unsafe { *a.get_unchecked(p * m + i) };
                        for (j, cj) in c_row.iter_mut().enumerate() {
                            *cj += a_ip * unsafe { *b.get_unchecked(j * k + p) };
                        }
                    }
                }
            }
        }
    }
}
