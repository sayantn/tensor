#![allow(unused_variables)]

use crate::{MatrixView, MatrixViewMut, VectorView, VectorViewMut};

use core::ops::{Add, AddAssign, Mul, MulAssign};

mod gemm;
pub use gemm::*;
use num_traits::{One, Zero};

pub trait Element:
    'static
    + Mul<Output = Self>
    + MulAssign
    + Add<Output = Self>
    + AddAssign
    + Copy
    + PartialEq
    + Zero
    + One
{
}
impl<
    T: 'static
        + Mul<Output = Self>
        + MulAssign
        + Add<Output = Self>
        + AddAssign
        + Copy
        + PartialEq
        + Zero
        + One,
> Element for T
{
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MatrixOp {
    NoTrans,
    Trans,
}

#[cfg(feature = "blas")]
impl From<MatrixOp> for cblas_sys::CBLAS_TRANSPOSE {
    fn from(op: MatrixOp) -> Self {
        match op {
            MatrixOp::NoTrans => Self::CblasNoTrans,
            MatrixOp::Trans => Self::CblasTrans,
        }
    }
}

pub fn gemv<T>(
    alpha: T,
    a: MatrixView<'_, T>,
    x: VectorView<'_, T>,
    beta: T,
    y: VectorViewMut<'_, T>,
) {
    todo!()
}

pub fn axpy<T>(alpha: T, x: VectorView<'_, T>, y: VectorViewMut<'_, T>) {
    todo!()
}

pub fn dot<T>(x: VectorView<'_, T>, y: VectorView<'_, T>) -> T {
    todo!()
}

pub fn nrm2<T>(x: VectorView<'_, T>) -> T {
    todo!()
}

pub fn sum_axis0<T>(x: MatrixView<'_, T>, beta: T, y: VectorViewMut<'_, T>) {
    todo!()
}

pub fn ger<T>(alpha: T, x: VectorView<'_, T>, y: VectorView<'_, T>, a: MatrixViewMut<'_, T>) {
    todo!()
}
