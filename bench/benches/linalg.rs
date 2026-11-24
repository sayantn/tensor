use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use half::f16;
use std::hint::black_box;
use tensor::linalg::{Element, MatrixOp, axpy, dot, gemv, ger, sum_axis};
use tensor::{Matrix, Vector};

trait BenchElem: Element {
    fn from_f32(v: f32) -> Self;
}

impl BenchElem for f32 {
    #[inline]
    fn from_f32(v: f32) -> Self {
        v
    }
}

impl BenchElem for f64 {
    #[inline]
    fn from_f32(v: f32) -> Self {
        v as f64
    }
}

impl BenchElem for f16 {
    #[inline]
    fn from_f32(v: f32) -> Self {
        f16::from_f32(v)
    }
}

fn fill_vec<T: BenchElem>(len: usize) -> Vector<T> {
    let data = (0..len)
        .map(|i| T::from_f32(((i % 17) as f32 - 8.0) * 0.125))
        .collect();
    Vector::from_vec(data, [len])
}

fn fill_mat<T: BenchElem>(rows: usize, cols: usize) -> Matrix<T> {
    let data = (0..rows * cols)
        .map(|i| T::from_f32(((i % 19) as f32 - 9.0) * 0.0625))
        .collect();
    Matrix::from_vec(data, [rows, cols])
}

fn bench_vec_ops<T: BenchElem>(c: &mut Criterion, dtype: &str) {
    let sizes = [16, 64, 128, 256, 784, 1024, 4096, 16_384, 65_536];
    let alpha = T::from_f32(0.75);

    let mut group = c.benchmark_group(format!("axpy_{dtype}"));
    for &n in &sizes {
        group.throughput(Throughput::Elements((2 * n) as u64));
        let x = fill_vec::<T>(n);
        let mut y = fill_vec::<T>(n);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                axpy(
                    black_box(alpha),
                    black_box(x.view()),
                    black_box(y.view_mut()),
                );
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group(format!("dot_{dtype}"));
    for &n in &sizes {
        group.throughput(Throughput::Elements((2 * n) as u64));
        let x = fill_vec::<T>(n);
        let y = fill_vec::<T>(n);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| black_box(dot(black_box(x.view()), black_box(y.view()))));
        });
    }
    group.finish();
}

fn bench_mat_vec_ops<T: BenchElem>(c: &mut Criterion, dtype: &str) {
    // Includes square sizes + typical NN forward/backward shapes (batch=64, MNIST=784, hidden=256/512)
    let shapes = [
        (16, 16),
        (32, 32),
        (64, 64),
        (64, 128),
        (64, 256),
        (128, 128),
        (128, 784),
        (256, 256),
        (512, 512),
    ];

    let alpha = T::from_f32(0.75);
    let beta = T::zero();

    for op in [MatrixOp::NoTrans, MatrixOp::Trans] {
        let tag = match op {
            MatrixOp::NoTrans => "N",
            MatrixOp::Trans => "T",
        };
        let mut group = c.benchmark_group(format!("gemv_{dtype}_{tag}"));

        for &(m, n) in &shapes {
            group.throughput(Throughput::Elements((2 * m * n) as u64));
            let a = fill_mat::<T>(m, n);
            let (x_len, y_len) = match op {
                MatrixOp::NoTrans => (n, m),
                MatrixOp::Trans => (m, n),
            };
            let x = fill_vec::<T>(x_len);
            let mut y = fill_vec::<T>(y_len);

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{m}x{n}")),
                &(m, n),
                |b, _| {
                    b.iter(|| {
                        gemv(
                            black_box(alpha),
                            (black_box(a.view()), op),
                            black_box(x.view()),
                            black_box(beta),
                            black_box(y.view_mut()),
                        );
                    });
                },
            );
        }
        group.finish();
    }

    let mut group = c.benchmark_group(format!("sum_axis0_{dtype}"));
    for &(m, n) in &shapes {
        group.throughput(Throughput::Elements((m * n) as u64));
        let x = fill_mat::<T>(m, n);
        let mut y = fill_vec::<T>(n);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{m}x{n}")),
            &(m, n),
            |b, _| {
                b.iter(|| {
                    sum_axis(
                        black_box(x.view()),
                        0,
                        black_box(T::zero()),
                        black_box(y.view_mut()),
                    );
                });
            },
        );
    }
    group.finish();

    let mut group = c.benchmark_group(format!("ger_{dtype}"));
    for &(m, n) in &shapes {
        group.throughput(Throughput::Elements((2 * m * n) as u64));
        let x = fill_vec::<T>(m);
        let y = fill_vec::<T>(n);
        let mut a = fill_mat::<T>(m, n);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{m}x{n}")),
            &(m, n),
            |b, _| {
                b.iter(|| {
                    ger(
                        black_box(alpha),
                        black_box(x.view()),
                        black_box(y.view()),
                        black_box(a.view_mut()),
                    );
                });
            },
        );
    }
    group.finish();
}

fn bench_all(c: &mut Criterion) {
    bench_vec_ops::<f16>(c, "f16");
    bench_mat_vec_ops::<f16>(c, "f16");

    bench_vec_ops::<f32>(c, "f32");
    bench_mat_vec_ops::<f32>(c, "f32");

    bench_vec_ops::<f64>(c, "f64");
    bench_mat_vec_ops::<f64>(c, "f64");
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
