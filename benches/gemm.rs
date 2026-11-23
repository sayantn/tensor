#![allow(unused_variables)]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tensor::Matrix;
use tensor::linalg::{Element, GemmBackend, MatrixOp, Naive};

#[cfg(feature = "f16")]
use half::f16;

#[cfg(feature = "backend-blas")]
use tensor::linalg::Blas;
#[cfg(feature = "backend-gemm")]
use tensor::linalg::GemmCrate;
#[cfg(feature = "backend-matrixmultiply")]
use tensor::linalg::MatrixMultiplyCrate;

#[allow(clippy::too_many_arguments)]
fn bench_gemm_sizes<T, B>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    backend: B,
    sizes: &[(usize, usize, usize)],
    ta: MatrixOp,
    tb: MatrixOp,
    val_a: T,
    val_b: T,
) where
    T: Element + Default,
    B: GemmBackend<T>,
{
    for &(m, n, k) in sizes {
        let a_shape = match ta {
            MatrixOp::NoTrans => [m, k],
            MatrixOp::Trans => [k, m],
        };
        let b_shape = match tb {
            MatrixOp::NoTrans => [k, n],
            MatrixOp::Trans => [n, k],
        };

        let a = Matrix::from_elem(val_a, a_shape);
        let b = Matrix::from_elem(val_b, b_shape);
        let mut c = Matrix::zeros([m, n]);

        let flops = (2 * m * n * k) as u64;
        group.throughput(Throughput::Elements(flops));

        let label = format!("{m}x{n}x{k}");
        group.bench_with_input(BenchmarkId::new(name, &label), &label, |bench, _| {
            bench.iter(|| {
                backend.gemm(
                    black_box(T::one()),
                    (a.view(), ta),
                    (b.view(), tb),
                    black_box(T::zero()),
                    c.view_mut(),
                );
                black_box(c.as_ptr());
            });
        });
    }
}

trait OptionalBackends<T: Element> {
    fn run_extra(
        group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
        sizes: &[(usize, usize, usize)],
        ta: MatrixOp,
        tb: MatrixOp,
        val_a: T,
        val_b: T,
    );
}

#[cfg(feature = "f16")]
impl OptionalBackends<f16> for f16 {
    fn run_extra(
        group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
        sizes: &[(usize, usize, usize)],
        ta: MatrixOp,
        tb: MatrixOp,
        val_a: f16,
        val_b: f16,
    ) {
        #[cfg(feature = "backend-gemm")]
        bench_gemm_sizes(group, "GemmCrate", GemmCrate, sizes, ta, tb, val_a, val_b);
    }
}

impl OptionalBackends<f32> for f32 {
    fn run_extra(
        group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
        sizes: &[(usize, usize, usize)],
        ta: MatrixOp,
        tb: MatrixOp,
        val_a: f32,
        val_b: f32,
    ) {
        #[cfg(feature = "backend-gemm")]
        bench_gemm_sizes(group, "GemmCrate", GemmCrate, sizes, ta, tb, val_a, val_b);
        #[cfg(feature = "backend-matrixmultiply")]
        bench_gemm_sizes(
            group,
            "MatrixMultiply",
            MatrixMultiplyCrate,
            sizes,
            ta,
            tb,
            val_a,
            val_b,
        );
        #[cfg(feature = "backend-blas")]
        bench_gemm_sizes(group, "Blas", Blas, sizes, ta, tb, val_a, val_b);
    }
}

impl OptionalBackends<f64> for f64 {
    fn run_extra(
        group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
        sizes: &[(usize, usize, usize)],
        ta: MatrixOp,
        tb: MatrixOp,
        val_a: f64,
        val_b: f64,
    ) {
        #[cfg(feature = "backend-gemm")]
        bench_gemm_sizes(group, "GemmCrate", GemmCrate, sizes, ta, tb, val_a, val_b);
        #[cfg(feature = "backend-matrixmultiply")]
        bench_gemm_sizes(
            group,
            "MatrixMultiply",
            MatrixMultiplyCrate,
            sizes,
            ta,
            tb,
            val_a,
            val_b,
        );
        #[cfg(feature = "backend-blas")]
        bench_gemm_sizes(group, "Blas", Blas, sizes, ta, tb, val_a, val_b);
    }
}

fn run_suite<T: Element + Default + OptionalBackends<T>>(
    c: &mut Criterion,
    dtype_name: &str,
    sizes: &[(usize, usize, usize)],
    val_a: T,
    val_b: T,
) {
    for (op_label, ta, tb) in [
        ("NN", MatrixOp::NoTrans, MatrixOp::NoTrans),
        ("NT", MatrixOp::NoTrans, MatrixOp::Trans),
        ("TN", MatrixOp::Trans, MatrixOp::NoTrans),
        ("TT", MatrixOp::Trans, MatrixOp::Trans),
    ] {
        let mut group = c.benchmark_group(format!("gemm_{dtype_name}_{op_label}"));
        bench_gemm_sizes(&mut group, "Naive", Naive, sizes, ta, tb, val_a, val_b);
        T::run_extra(&mut group, sizes, ta, tb, val_a, val_b);
        group.finish();
    }
}

fn gemm_benchmarks(c: &mut Criterion) {
    let sizes = [
        (4, 4, 4),
        (8, 8, 8),
        (16, 16, 16),
        (24, 24, 24),
        (32, 32, 32),
        (48, 48, 48),
        (64, 64, 64),
        (96, 96, 96),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (64, 128, 784),
        (64, 256, 256),
    ];

    #[cfg(feature = "f16")]
    run_suite(c, "f16", &sizes, f16::from_f32(0.05), f16::from_f32(0.05));
    run_suite(c, "f32", &sizes, 1.001_f32, 0.999_f32);
    run_suite(c, "f64", &sizes, 1.001_f64, 0.999_f64);
}

criterion_group!(benches, gemm_benchmarks);
criterion_main!(benches);
