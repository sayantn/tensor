use core::any::TypeId;
use core::mem;

#[cfg(feature = "blas")]
use cblas_sys::{CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE, cblas_dgemm, cblas_sgemm};
#[cfg(feature = "blas")]
use core::ffi::c_int;

#[cfg(feature = "faer")]
use faer::traits::ComplexField;
#[cfg(feature = "faer")]
use faer::{Accum, MatMut, MatRef, Par, linalg::matmul::matmul};

use crate::linalg::{Element, MatrixOp};
use crate::{MatrixView, MatrixViewMut};

/// Performs `C = alpha * op(A) * op(B) + beta * C`
/// Dynamically dispatches to specialized f32/f64 backends if enabled.
pub fn gemm<T: Element>(
    alpha: T,
    (a, trans_a): (MatrixView<'_, T>, MatrixOp),
    (b, trans_b): (MatrixView<'_, T>, MatrixOp),
    beta: T,
    c: MatrixViewMut<'_, T>,
) {
    let (m_a, k_a) = match trans_a {
        MatrixOp::NoTrans => (a.nrows(), a.ncols()),
        MatrixOp::Trans => (a.ncols(), a.nrows()),
    };
    let (k_b, n_b) = match trans_b {
        MatrixOp::NoTrans => (b.nrows(), b.ncols()),
        MatrixOp::Trans => (b.ncols(), b.nrows()),
    };

    assert_eq!(k_a, k_b, "Inner dimension mismatch: {k_a} != {k_b}");
    assert_eq!((m_a, n_b), (c.nrows(), c.ncols()), "Output shape mismatch");

    // Fast-path dispatch for f32
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        unsafe {
            gemm_f32(
                mem::transmute_copy(&alpha),
                (a.transmute(), trans_a),
                (b.transmute(), trans_b),
                mem::transmute_copy(&beta),
                c.transmute_mut(),
            );
        }
        return;
    }

    // Fast-path dispatch for f64
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        unsafe {
            gemm_f64(
                mem::transmute_copy(&alpha),
                (a.transmute(), trans_a),
                (b.transmute(), trans_b),
                mem::transmute_copy(&beta),
                c.transmute_mut(),
            );
        }
        return;
    }

    // Fallback for all other types
    naive_gemm(alpha, (a, trans_a), (b, trans_b), beta, c);
}

// ============================================================================
// Specialized Backends
// ============================================================================

fn gemm_f32(
    alpha: f32,
    (a, trans_a): (MatrixView<'_, f32>, MatrixOp),
    (b, trans_b): (MatrixView<'_, f32>, MatrixOp),
    beta: f32,
    mut c: MatrixViewMut<'_, f32>,
) {
    let (m, n) = (c.nrows(), c.ncols());
    let k = if trans_a == MatrixOp::NoTrans {
        a.ncols()
    } else {
        a.nrows()
    };

    if m * n * k < 16_000 {
        return naive_gemm(alpha, (a, trans_a), (b, trans_b), beta, c);
    }

    #[cfg(feature = "blas")]
    if (m | n | k) <= c_int::MAX as usize {
        unsafe {
            cblas_sgemm(
                CblasRowMajor,
                CBLAS_TRANSPOSE::from(trans_a),
                CBLAS_TRANSPOSE::from(trans_b),
                m as c_int,
                n as c_int,
                k as c_int,
                alpha,
                a.as_slice().as_ptr(),
                a.ncols() as c_int,
                b.as_slice().as_ptr(),
                b.ncols() as c_int,
                beta,
                c.as_mut_slice().as_mut_ptr(),
                c.ncols() as c_int,
            );
        }
        return;
    }

    #[cfg(feature = "faer")]
    {
        run_faer(alpha, (a, trans_a), (b, trans_b), beta, c);
        return;
    }

    #[cfg(all(feature = "matrixmultiply", not(feature = "faer")))]
    {
        let (rsa, csa) = strides(a.ncols(), trans_a);
        let (rsb, csb) = strides(b.ncols(), trans_b);
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                alpha,
                a.as_slice().as_ptr(),
                rsa,
                csa,
                b.as_slice().as_ptr(),
                rsb,
                csb,
                beta,
                c.as_mut_slice().as_mut_ptr(),
                c.ncols() as isize,
                1,
            );
        }
        return;
    }

    #[cfg(not(any(feature = "faer", feature = "matrixmultiply", feature = "blas")))]
    naive_gemm(alpha, (a, trans_a), (b, trans_b), beta, c);
}

fn gemm_f64(
    alpha: f64,
    (a, trans_a): (MatrixView<'_, f64>, MatrixOp),
    (b, trans_b): (MatrixView<'_, f64>, MatrixOp),
    beta: f64,
    mut c: MatrixViewMut<'_, f64>,
) {
    let (m, n) = (c.nrows(), c.ncols());
    let k = if trans_a == MatrixOp::NoTrans {
        a.ncols()
    } else {
        a.nrows()
    };

    if m * n * k < 16_000 {
        return naive_gemm(alpha, (a, trans_a), (b, trans_b), beta, c);
    }

    #[cfg(feature = "blas")]
    if (m | n | k) <= c_int::MAX as usize {
        unsafe {
            cblas_dgemm(
                CblasRowMajor,
                CBLAS_TRANSPOSE::from(trans_a),
                CBLAS_TRANSPOSE::from(trans_b),
                m as c_int,
                n as c_int,
                k as c_int,
                alpha,
                a.as_slice().as_ptr(),
                a.ncols() as c_int,
                b.as_slice().as_ptr(),
                b.ncols() as c_int,
                beta,
                c.as_mut_slice().as_mut_ptr(),
                c.ncols() as c_int,
            );
        }
        return;
    }

    #[cfg(feature = "faer")]
    {
        run_faer(alpha, (a, trans_a), (b, trans_b), beta, c);
        return;
    }

    #[cfg(all(feature = "matrixmultiply", not(feature = "faer")))]
    {
        let (rsa, csa) = strides(a.ncols(), trans_a);
        let (rsb, csb) = strides(b.ncols(), trans_b);
        unsafe {
            matrixmultiply::dgemm(
                m,
                k,
                n,
                alpha,
                a.as_slice().as_ptr(),
                rsa,
                csa,
                b.as_slice().as_ptr(),
                rsb,
                csb,
                beta,
                c.as_mut_slice().as_mut_ptr(),
                c.ncols() as isize,
                1,
            );
        }
        return;
    }

    #[cfg(not(any(feature = "faer", feature = "matrixmultiply", feature = "blas")))]
    naive_gemm(alpha, (a, trans_a), (b, trans_b), beta, c);
}

#[cfg(feature = "faer")]
fn run_faer<T: Element + ComplexField>(
    alpha: T,
    (a, trans_a): (MatrixView<'_, T>, MatrixOp),
    (b, trans_b): (MatrixView<'_, T>, MatrixOp),
    beta: T,
    mut c: MatrixViewMut<'_, T>,
) {
    let (m, n) = (c.nrows(), c.ncols());
    let k = if trans_a == MatrixOp::NoTrans {
        a.ncols()
    } else {
        a.nrows()
    };

    let accum = if beta == T::zero() {
        Accum::Replace
    } else if beta == T::one() {
        Accum::Add
    } else {
        c.as_mut_slice().iter_mut().for_each(|val| *val *= beta);
        Accum::Add
    };

    let a_mat = MatRef::from_row_major_slice(a.as_slice(), a.nrows(), a.ncols());
    let b_mat = MatRef::from_row_major_slice(b.as_slice(), b.nrows(), b.ncols());
    let c_mat = MatMut::from_row_major_slice_mut(c.as_mut_slice(), m, n);

    let a_view = match trans_a {
        MatrixOp::NoTrans => a_mat.as_ref(),
        MatrixOp::Trans => a_mat.transpose(),
    };
    let b_view = match trans_b {
        MatrixOp::NoTrans => b_mat.as_ref(),
        MatrixOp::Trans => b_mat.transpose(),
    };

    matmul(c_mat, accum, a_view, b_view, alpha, Par::Seq);
}

#[cfg(all(feature = "matrixmultiply", not(feature = "faer")))]
#[inline]
fn strides(ld: usize, op: MatrixOp) -> (isize, isize) {
    match op {
        MatrixOp::NoTrans => (ld as isize, 1),
        MatrixOp::Trans => (1, ld as isize),
    }
}

// ============================================================================
// Fallback Backend (High-Performance Generic Slice Implementation)
// ============================================================================

/// 100% slice-based generic fallback GEMM.
/// Uses strict `MatrixView` and cache-friendly iteration, avoiding raw ptr arith.
pub fn naive_gemm<T: Element>(
    alpha: T,
    (a, trans_a): (MatrixView<'_, T>, MatrixOp),
    (b, trans_b): (MatrixView<'_, T>, MatrixOp),
    beta: T,
    mut c: MatrixViewMut<'_, T>,
) {
    let (m, n) = (c.nrows(), c.ncols());
    let k = if trans_a == MatrixOp::NoTrans {
        a.ncols()
    } else {
        a.nrows()
    };
    let ldc = c.ncols();

    let a_slice = a.as_slice();
    let b_slice = b.as_slice();
    let c_slice = c.as_mut_slice();

    // Scale C by beta (handling beta == 0 properly to overwrite NaNs)
    let zero = T::zero();
    for i in 0..m {
        // Safe indexing: dimensions validated before function logic
        let c_row = unsafe { c_slice.get_unchecked_mut(i * ldc..(i + 1) * ldc) };
        if beta == zero {
            c_row.fill(zero);
        } else if beta != T::one() {
            c_row.iter_mut().for_each(|val| *val *= beta);
        }
    }

    if alpha == zero {
        return;
    }

    let (rsa, csa) = match trans_a {
        MatrixOp::NoTrans => (a.ncols(), 1),
        MatrixOp::Trans => (1, a.ncols()),
    };
    let (rsb, csb) = match trans_b {
        MatrixOp::NoTrans => (b.ncols(), 1),
        MatrixOp::Trans => (1, b.ncols()),
    };

    // Cache-friendly row-major accumulation (i -> p -> j)
    for i in 0..m {
        let c_row = unsafe { c_slice.get_unchecked_mut(i * ldc..(i + 1) * ldc) };
        for p in 0..k {
            let a_val = alpha * unsafe { *a_slice.get_unchecked(i * rsa + p * csa) };
            for j in 0..n {
                // Vectorizer-friendly inner loop (bounds checks elided via get_unchecked)
                unsafe {
                    *c_row.get_unchecked_mut(j) +=
                        a_val * *b_slice.get_unchecked(p * rsb + j * csb);
                }
            }
        }
    }
}
