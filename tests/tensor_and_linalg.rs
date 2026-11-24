#![allow(clippy::float_cmp, clippy::many_single_char_names)]
#![cfg(feature = "alloc")]

use tensor::linalg::{Cutoff, GemmBackend, MatrixOp, Naive, axpy, dot, gemv, ger, sum_axis};
use tensor::{Matrix, Scalar, Tensor, TensorViewMut, Vector};

#[cfg(feature = "backend-blas")]
use tensor::linalg::Blas;
#[cfg(feature = "backend-gemm")]
use tensor::linalg::GemmCrate;
#[cfg(feature = "backend-matrixmultiply")]
use tensor::linalg::MatrixMultiplyCrate;

// ============================================================================
// Helpers
// ============================================================================

fn assert_close_f32(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "mismatch at index {i}: actual={a}, expected={e}, diff={}",
            (a - e).abs()
        );
    }
}

fn assert_close_f64(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "mismatch at index {i}: actual={a}, expected={e}, diff={}",
            (a - e).abs()
        );
    }
}

/// Deterministic pseudo-random generator (xorshift64) so tests stay `#![no_std]`-friendly
/// and reproducible without extra dev-dependencies.
struct Rng(u64);

impl Rng {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        // Uniform in [-1.0, 1.0]
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32 / (1u32 << 23) as f32) - 1.0
    }

    fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64 / (1u64 << 52) as f64) - 1.0
    }
}

// ============================================================================
// Core Tensor & View Tests
// ============================================================================

#[test]
fn scalar_0d_operations() {
    let mut s = Scalar::scalar(42_i32);
    assert_eq!(s.shape(), []);
    assert_eq!(s.len(), 1);
    assert_eq!(s[[]], 42);

    s[[]] = 99;
    assert_eq!(s.as_slice(), &[99]);

    let v = s.view();
    assert_eq!(v[[]], 99);
    assert_eq!(format!("{v}"), "99");

    let mut x = 7_i32;
    let mut sv = TensorViewMut::scalar(&mut x);
    sv[[]] += 3;
    assert_eq!(x, 10);
}

#[test]
fn indexing_reshape_and_slicing() {
    // 2 x 3 x 2 tensor = 12 elements
    let data: Vec<i32> = (0..12).collect();
    let t = Tensor::from_vec(data, [2, 3, 2]);

    assert_eq!(t[[0, 0, 0]], 0);
    assert_eq!(t[[0, 2, 1]], 5);
    assert_eq!(t[[1, 1, 0]], 8);
    assert_eq!(t[[1, 2, 1]], 11);

    // Slice axis 0 with inclusive, exclusive, and unbounded ranges
    let s1 = t.view().slice(1..2);
    assert_eq!(s1.shape(), [1, 3, 2]);
    assert_eq!(s1[[0, 0, 0]], 6);

    let s2 = t.view().slice(0..=0);
    assert_eq!(s2.shape(), [1, 3, 2]);
    assert_eq!(s2[[0, 2, 1]], 5);

    // Reshape and flatten
    let m = t.view().reshape([4, 3]);
    assert_eq!(m.nrows(), 4);
    assert_eq!(m.ncols(), 3);
    assert_eq!(m[[2, 1]], 7);

    let flat = m.flatten();
    assert_eq!(flat.shape(), [12]);
    assert_eq!(flat[[11]], 11);
}

#[test]
fn chunks_and_broadcast_assign_ops() {
    let mut batch = Tensor::zeros([3, 2, 2]);
    let bias = Matrix::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], [2, 2]);

    batch.assign(bias.view());
    for chunk in batch.view_mut().chunks::<2>() {
        assert_eq!(chunk.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    let row = Vector::from_vec(vec![10.0_f32, 20.0], [2]);
    batch += row.view();
    assert_eq!(
        batch.view_mut().slice(0..1).as_slice(),
        &[11.0, 22.0, 13.0, 24.0]
    );

    batch *= 2.0;
    batch -= 2.0;
    batch /= 2.0;
    assert_eq!(
        batch.view_mut().slice(2..3).as_slice(),
        &[10.0, 21.0, 12.0, 23.0]
    );
}

#[test]
#[should_panic(expected = "tensor dimensions must be non-zero")]
fn reject_zero_dimension() {
    let _ = Tensor::<f32, 2>::zeros([3, 0]);
}

#[test]
#[should_panic(expected = "does not match shape")]
fn reject_mismatched_buffer_len() {
    let _ = Tensor::from_vec(vec![1, 2, 3], [2, 2]);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn reject_out_of_bounds_index() {
    let m = Matrix::from_vec(vec![1, 2, 3, 4], [2, 2]);
    let _ = m[[1, 2]];
}

#[test]
#[should_panic(expected = "must be non-empty")]
fn reject_empty_slice_range() {
    let m = Matrix::from_vec(vec![1, 2, 3, 4], [2, 2]);
    let _ = m.view().slice(1..1);
}

// ============================================================================
// Level-1 & Level-2 Linear Algebra Tests
// ============================================================================

#[test]
fn axpy_and_dot_across_simd_tails() {
    let mut rng = Rng::new(0x1234_5678_9abc_def0);

    // Test lengths covering 0-chunk tails, exact 16-lane multiples, and remainder elements
    for n in [1, 3, 7, 15, 16, 17, 31, 32, 33, 67] {
        let x_vec: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let y0_vec: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let alpha = -1.75_f32;

        let x = Vector::from_vec(x_vec.clone(), [n]);
        let mut y = Vector::from_vec(y0_vec.clone(), [n]);

        axpy(alpha, x.view(), y.view_mut());
        let expected_axpy: Vec<f32> = x_vec
            .iter()
            .zip(&y0_vec)
            .map(|(&xi, &yi)| alpha * xi + yi)
            .collect();
        assert_close_f32(y.as_slice(), &expected_axpy, 1e-5);

        // alpha == 0 fast path should leave `y` completely untouched (even with NaN)
        let mut y_nan = Vector::from_vec(vec![f32::NAN; n], [n]);
        axpy(0.0, x.view(), y_nan.view_mut());
        assert!(y_nan.iter().all(|v| v.is_nan()));

        // dot product check
        let actual_dot = dot(x.view(), y.view());
        let expected_dot: f32 = x_vec
            .iter()
            .zip(&expected_axpy)
            .map(|(&xi, &yi)| xi * yi)
            .sum();
        assert!(
            (actual_dot - expected_dot).abs() <= 1e-4,
            "dot mismatch at n={n}: actual={actual_dot}, expected={expected_dot}"
        );
    }
}

#[test]
fn gemv_notrans_and_trans_and_nan_overwrite() {
    // A = [[1, 2, 3],
    //      [4, 5, 6]] (2x3)
    let a = Matrix::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let x = Vector::from_vec(vec![10.0, 20.0, 30.0], [3]);

    // beta = 0.0 must overwrite NaNs in `y` rather than producing NaN
    let mut y = Vector::from_vec(vec![f64::NAN, f64::NAN], [2]);
    gemv(
        2.0,
        (a.view(), MatrixOp::NoTrans),
        x.view(),
        0.0,
        y.view_mut(),
    );
    // 2 * [1*10 + 2*20 + 3*30, 4*10 + 5*20 + 6*30] = 2 * [140, 320] = [280, 640]
    assert_close_f64(y.as_slice(), &[280.0, 640.0], 1e-12);

    // Transposed: y := 0.5 * A^T * z + 3.0 * y
    let z = Vector::from_vec(vec![2.0, 4.0], [2]);
    let mut yt = Vector::from_vec(vec![1.0, 2.0, 3.0], [3]);
    gemv(
        0.5,
        (a.view(), MatrixOp::Trans),
        z.view(),
        3.0,
        yt.view_mut(),
    );
    // 0.5 * [1*2 + 4*4, 2*2 + 5*4, 3*2 + 6*4] + [3, 6, 9]
    // = [9, 12, 15] + [3, 6, 9] = [12, 18, 24]
    assert_close_f64(yt.as_slice(), &[12.0, 18.0, 24.0], 1e-12);
}

#[test]
fn sum_axis0_unrolled_and_remainder_rows() {
    // 6 rows x 5 cols -> exercises 1 block of 4 rows + 2 remainder rows
    let data: Vec<f32> = (1..=30).map(|v| v as f32).collect();
    let x = Matrix::from_vec(data, [6, 5]);
    let mut y = Vector::from_vec(vec![10.0; 5], [5]);

    sum_axis(x.view(), 0, 2.0, y.view_mut());

    // Column j has values (j+1) + 5*r for r in 0..6:
    // sum_r = 6*(j+1) + 5*(0+1+2+3+4+5) = 6*(j+1) + 75
    // plus beta * 10 = 20 => 6*(j+1) + 95
    let expected: Vec<f32> = (0..5).map(|j| 6.0 * (j as f32 + 1.0) + 95.0).collect();
    assert_close_f32(y.as_slice(), &expected, 1e-5);
}

#[test]
fn ger_outer_product() {
    let x = Vector::from_vec(vec![2.0_f32, -3.0], [2]);
    let y = Vector::from_vec(vec![4.0_f32, 5.0, 6.0], [3]);
    let mut a = Matrix::from_vec(vec![1.0; 6], [2, 3]);

    ger(0.5, x.view(), y.view(), a.view_mut());
    // 0.5 * [[8, 10, 12], [-12, -15, -18]] + 1
    assert_close_f32(a.as_slice(), &[5.0, 6.0, 7.0, -5.0, -6.5, -8.0], 1e-6);
}

// ============================================================================
// Level-3 GEMM Backend Equivalence Tests
// ============================================================================

fn verify_gemm_backend_f32<B: GemmBackend<f32>>(backend: &B) {
    let mut rng = Rng::new(0xdead_beef_cafe_babe);

    #[cfg(miri)]
    let cases = [(1, 1, 1), (3, 5, 4)];
    #[cfg(not(miri))]
    let cases = [
        (1, 1, 1),
        (3, 5, 4),
        (17, 19, 23),
        (32, 32, 32),
        (37, 29, 41),
    ];

    let ops = [
        (MatrixOp::NoTrans, MatrixOp::NoTrans),
        (MatrixOp::NoTrans, MatrixOp::Trans),
        (MatrixOp::Trans, MatrixOp::NoTrans),
        (MatrixOp::Trans, MatrixOp::Trans),
    ];

    for (m, n, k) in cases {
        for (ta, tb) in ops {
            let a_shape = match ta {
                MatrixOp::NoTrans => [m, k],
                MatrixOp::Trans => [k, m],
            };
            let b_shape = match tb {
                MatrixOp::NoTrans => [k, n],
                MatrixOp::Trans => [n, k],
            };

            let a = Matrix::from_vec(
                (0..a_shape[0] * a_shape[1])
                    .map(|_| rng.next_f32())
                    .collect(),
                a_shape,
            );
            let b = Matrix::from_vec(
                (0..b_shape[0] * b_shape[1])
                    .map(|_| rng.next_f32())
                    .collect(),
                b_shape,
            );
            let c_init = Matrix::from_vec((0..m * n).map(|_| rng.next_f32()).collect(), [m, n]);

            for (alpha, beta) in [(1.0, 0.0), (1.0, 1.0), (-0.75, 0.5), (0.0, 0.25)] {
                let mut c_ref = c_init.clone();
                let mut c_test = c_init.clone();

                Naive.gemm(
                    alpha,
                    (a.view(), ta),
                    (b.view(), tb),
                    beta,
                    c_ref.view_mut(),
                );
                backend.gemm(
                    alpha,
                    (a.view(), ta),
                    (b.view(), tb),
                    beta,
                    c_test.view_mut(),
                );

                assert_close_f32(c_test.as_slice(), c_ref.as_slice(), 1e-3);
            }

            // Verify beta = 0.0 safely overwrites NaNs in destination buffer
            let mut c_nan = Matrix::from_vec(vec![f32::NAN; m * n], [m, n]);
            let mut c_ref = Matrix::zeros([m, n]);
            Naive.gemm(1.25, (a.view(), ta), (b.view(), tb), 0.0, c_ref.view_mut());
            backend.gemm(1.25, (a.view(), ta), (b.view(), tb), 0.0, c_nan.view_mut());
            assert_close_f32(c_nan.as_slice(), c_ref.as_slice(), 1e-3);
        }
    }
}

fn verify_gemm_backend_f64<B: GemmBackend<f64>>(backend: &B) {
    let mut rng = Rng::new(0x0102_0304_0506_0708);

    #[cfg(miri)]
    let (m, n, k) = (3, 4, 2);
    #[cfg(not(miri))]
    let (m, n, k) = (19, 23, 17);

    for (ta, tb) in [
        (MatrixOp::NoTrans, MatrixOp::NoTrans),
        (MatrixOp::NoTrans, MatrixOp::Trans),
        (MatrixOp::Trans, MatrixOp::NoTrans),
        (MatrixOp::Trans, MatrixOp::Trans),
    ] {
        let a_shape = match ta {
            MatrixOp::NoTrans => [m, k],
            MatrixOp::Trans => [k, m],
        };
        let b_shape = match tb {
            MatrixOp::NoTrans => [k, n],
            MatrixOp::Trans => [n, k],
        };

        let a = Matrix::from_vec(
            (0..a_shape[0] * a_shape[1])
                .map(|_| rng.next_f64())
                .collect(),
            a_shape,
        );
        let b = Matrix::from_vec(
            (0..b_shape[0] * b_shape[1])
                .map(|_| rng.next_f64())
                .collect(),
            b_shape,
        );
        let c_init = Matrix::from_vec((0..m * n).map(|_| rng.next_f64()).collect(), [m, n]);

        let mut c_ref = c_init.clone();
        let mut c_test = c_init;

        Naive.gemm(-1.3, (a.view(), ta), (b.view(), tb), 0.7, c_ref.view_mut());
        backend.gemm(-1.3, (a.view(), ta), (b.view(), tb), 0.7, c_test.view_mut());

        assert_close_f64(c_test.as_slice(), c_ref.as_slice(), 1e-11);
    }
}

#[test]
fn gemm_naive_and_cutoff() {
    let cutoff = Cutoff::with(16 * 16 * 16, Naive, Naive);
    verify_gemm_backend_f32(&cutoff);
    verify_gemm_backend_f64(&cutoff);
}

#[cfg(feature = "backend-gemm")]
#[test]
fn gemm_crate_matches_naive() {
    verify_gemm_backend_f32(&GemmCrate);
    verify_gemm_backend_f64(&GemmCrate);

    let hybrid = Cutoff::with(16 * 16 * 16, Naive, GemmCrate);
    verify_gemm_backend_f32(&hybrid);
}

#[cfg(feature = "backend-matrixmultiply")]
#[test]
fn matrixmultiply_crate_matches_naive() {
    verify_gemm_backend_f32(&MatrixMultiplyCrate);
    verify_gemm_backend_f64(&MatrixMultiplyCrate);
}

#[cfg(feature = "backend-blas")]
#[test]
fn blas_matches_naive() {
    verify_gemm_backend_f32(&Blas);
    verify_gemm_backend_f64(&Blas);
}

#[cfg(feature = "gemm-f16")]
#[test]
fn gemm_f16_match_naive() {
    use half::f16;

    let (m, n, k) = (13, 17, 11);
    let a_f16: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32(((i % 7) as f32 - 3.0) * 0.25))
        .collect();
    let b_f16: Vec<f16> = (0..k * n)
        .map(|i| f16::from_f32(((i % 5) as f32 - 2.0) * 0.5))
        .collect();

    let a = Matrix::from_vec(a_f16, [m, k]);
    let b = Matrix::from_vec(b_f16, [k, n]);

    let mut c_ref = Matrix::zeros([m, n]);
    let mut c_gemm = Matrix::zeros([m, n]);

    Naive.gemm(
        f16::from_f32(1.5),
        (a.view(), MatrixOp::NoTrans),
        (b.view(), MatrixOp::NoTrans),
        f16::from_f32(0.0),
        c_ref.view_mut(),
    );
    GemmCrate.gemm(
        f16::from_f32(1.5),
        (a.view(), MatrixOp::NoTrans),
        (b.view(), MatrixOp::NoTrans),
        f16::from_f32(0.0),
        c_gemm.view_mut(),
    );

    for (r, g) in c_ref.iter().zip(c_gemm.iter()) {
        assert!((r.to_f32() - g.to_f32()).abs() <= 5e-2);
    }
}
#[test]
fn split_at_disjoint_mutable_views() {
    let mut m = Matrix::from_vec((0..20).collect(), [5, 4]);

    let (top, bottom) = m.view().split_at(2);
    assert_eq!(top.shape(), [2, 4]);
    assert_eq!(bottom.shape(), [3, 4]);
    assert_eq!(top[[1, 3]], 7);
    assert_eq!(bottom[[0, 0]], 8);

    // Disjoint mutable halves exercised under miri
    let (mut top_mut, mut bottom_mut) = m.view_mut().split_at(2);
    top_mut += 100;
    bottom_mut *= 2;

    assert_eq!(m[[0, 0]], 100);
    assert_eq!(m[[1, 3]], 107);
    assert_eq!(m[[2, 0]], 16);
    assert_eq!(m[[4, 3]], 38);
}

#[test]
fn binary_into_and_owned_ops() {
    let a = Matrix::from_vec(vec![10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0], [2, 3]);
    let b = Vector::from_vec(vec![1.0_f32, 2.0, 3.0], [3]);
    let mut out = Matrix::zeros([2, 3]);

    // Broadcasted *_into ops
    a.view().add_into(b.view(), out.view_mut());
    assert_close_f32(out.as_slice(), &[11.0, 22.0, 33.0, 41.0, 52.0, 63.0], 1e-6);

    a.view().sub_into(b.view(), out.view_mut());
    assert_close_f32(out.as_slice(), &[9.0, 18.0, 27.0, 39.0, 48.0, 57.0], 1e-6);

    a.view().mul_into(b.view(), out.view_mut());
    assert_close_f32(
        out.as_slice(),
        &[10.0, 40.0, 90.0, 40.0, 100.0, 180.0],
        1e-6,
    );

    a.view().div_into(b.view(), out.view_mut());
    assert_close_f32(out.as_slice(), &[10.0, 10.0, 10.0, 40.0, 25.0, 20.0], 1e-6);

    // Owned LHS + borrowed RHS (in-place reuse, with broadcasting)
    let r1 = a.clone() + &b;
    assert_close_f32(r1.as_slice(), &[11.0, 22.0, 33.0, 41.0, 52.0, 63.0], 1e-6);

    // Borrowed LHS (broadcasted) - owned RHS (non-commutative check)
    let r2 = &b - a.clone();
    assert_close_f32(
        r2.as_slice(),
        &[-9.0, -18.0, -27.0, -39.0, -48.0, -57.0],
        1e-6,
    );

    // Scalar LHS and unary Neg
    let r3 = -(100.0_f32 - a);
    assert_close_f32(
        r3.as_slice(),
        &[-90.0, -80.0, -70.0, -60.0, -50.0, -40.0],
        1e-6,
    );

    // Bitwise & unary Not on integers
    let bits = Vector::from_vec(vec![0b1010_u8, 0b1100], [2]);
    let masked = (!bits) & 0b0000_1111_u8;
    assert_eq!(masked.as_slice(), &[0b0101, 0b0011]);
}

#[test]
fn gemv_trans_unrolled_4_rows_and_remainder() {
    // 6x5 matrix -> exercises 1 unrolled 4-row block + 2 remainder rows in MatrixOp::Trans
    let a = Matrix::from_vec((1..=30).map(|v| v as f32).collect(), [6, 5]);
    let x = Vector::from_vec(vec![1.0_f32, -2.0, 3.0, -4.0, 5.0, -6.0], [6]);
    let mut y = Vector::from_vec(vec![0.5_f32; 5], [5]);

    gemv(
        2.0,
        (a.view(), MatrixOp::Trans),
        x.view(),
        4.0,
        y.view_mut(),
    );

    let mut expected = [0.5_f32 * 4.0; 5];
    for (i, &xi) in x.iter().enumerate() {
        for (j, ej) in expected.iter_mut().enumerate() {
            *ej += 2.0 * a[[i, j]] * xi;
        }
    }
    assert_close_f32(y.as_slice(), &expected, 1e-4);
}

#[test]
fn sum_axis_3d_all_axes_and_trailing_one() {
    // 2 x 3 x 4 tensor
    let x = Tensor::from_vec((1..=24).map(|v| v as f32).collect(), [2, 3, 4]);

    // axis = 0 -> [3, 4]
    let mut y0 = Matrix::zeros([3, 4]);
    sum_axis(x.view(), 0, 0.0, y0.view_mut());
    for j in 0..3 {
        for k in 0..4 {
            assert_close_f32(&[y0[[j, k]]], &[x[[0, j, k]] + x[[1, j, k]]], 1e-5);
        }
    }

    // axis = 1 (middle axis, 0 < axis < N - 1) -> [2, 4]
    let mut y1 = Matrix::zeros([2, 4]);
    sum_axis(x.view(), 1, 0.0, y1.view_mut());
    for i in 0..2 {
        for k in 0..4 {
            let exp = x[[i, 0, k]] + x[[i, 1, k]] + x[[i, 2, k]];
            assert_close_f32(&[y1[[i, k]]], &[exp], 1e-5);
        }
    }

    // axis = 2 (innermost axis, inner == 1) -> [2, 3]
    let mut y2 = Matrix::zeros([2, 3]);
    sum_axis(x.view(), 2, 0.0, y2.view_mut());
    for i in 0..2 {
        for j in 0..3 {
            let exp = x[[i, j, 0]] + x[[i, j, 1]] + x[[i, j, 2]] + x[[i, j, 3]];
            assert_close_f32(&[y2[[i, j]]], &[exp], 1e-5);
        }
    }

    // Trailing-1 dimension: [2, 3, 1] summed along axis = 1 -> [2, 1]
    let x1 = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 10.0, 20.0, 30.0], [2, 3, 1]);
    let mut y_t1 = Matrix::zeros([2, 1]);
    sum_axis(x1.view(), 1, 0.0, y_t1.view_mut());
    assert_close_f32(y_t1.as_slice(), &[6.0, 60.0], 1e-5);
}
