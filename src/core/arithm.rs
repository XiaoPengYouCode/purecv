/*
 *  arithm.rs
 *  purecv
 *
 *  This file is part of purecv - OpenCV.
 *
 *  purecv is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  purecv is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with purecv.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  As a special exception, the copyright holders of this library give you
 *  permission to link this library with independent modules to produce an
 *  executable, regardless of the license terms of these independent modules, and to
 *  copy and distribute the resulting executable under terms of your choice,
 *  provided that you also meet, for each linked independent module, the terms and
 *  conditions of the license of that module. An independent module is a module
 *  which is neither derived from nor based on this library. If you modify this
 *  library, you may extend this exception to your version of the library, but you
 *  are not obligated to do so. If you do not wish to do so, delete this exception
 *  statement from your version.
 *
 *  Copyright 2026 WebARKit.
 *
 *  Author(s): Walter Perdan @kalwalt https://github.com/kalwalt
 *
 */

use crate::core::error::{PureCvError, Result};
use crate::core::types::Scalar;
use crate::core::Matrix;
use num_traits::{Bounded, FromPrimitive, Num, ToPrimitive};
use std::ops::{BitAnd, BitOr, BitXor, Not};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Internal macro to handle feature-gated loop execution for binary operations.
/// Handles Parallel auto-vectorized, and Sequential auto-vectorized loops.
macro_rules! binary_op {
    ($dst:expr, $src1:expr, $src2:expr, $t_dst:ty, $t_src:ty, |$d:ident, $s1:ident, $s2:ident| $body:expr) => {
        #[cfg(feature = "parallel")]
        {
            $dst.data
                .par_iter_mut()
                .zip($src1.data.par_iter())
                .zip($src2.data.par_iter())
                .for_each(|((d_raw, &s1_raw), &s2_raw)| {
                    let $d: &mut $t_dst = d_raw;
                    let $s1: $t_src = s1_raw;
                    let $s2: $t_src = s2_raw;
                    $body
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            $dst.data
                .iter_mut()
                .zip($src1.data.iter())
                .zip($src2.data.iter())
                .for_each(|((d_raw, &s1_raw), &s2_raw)| {
                    let $d: &mut $t_dst = d_raw;
                    let $s1: $t_src = s1_raw;
                    let $s2: $t_src = s2_raw;
                    $body
                });
        }
    };
}

/// Internal macro to handle feature-gated loop execution for unary operations.
macro_rules! unary_op {
    ($dst:expr, $src:expr, $t_dst:ty, $t_src:ty, |$d:ident, $s:ident| $body:expr) => {
        #[cfg(feature = "parallel")]
        {
            $dst.data
                .par_iter_mut()
                .zip($src.data.par_iter())
                .for_each(|(d_raw, &s_raw)| {
                    let $d: &mut $t_dst = d_raw;
                    let $s: $t_src = s_raw;
                    $body
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            $dst.data
                .iter_mut()
                .zip($src.data.iter())
                .for_each(|(d_raw, &s_raw)| {
                    let $d: &mut $t_dst = d_raw;
                    let $s: $t_src = s_raw;
                    $body
                });
        }
    };
}

/// Calculates the per-element sum of two matrices.
///
/// dst = src1 + src2
pub fn add<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + PartialOrd + Bounded + Default + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| *d = s1 + s2);

    Ok(dst)
}

/// Calculates the per-element difference between two matrices.
///
/// dst = src1 - src2
pub fn subtract<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + PartialOrd + Bounded + Default + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| *d = s1 - s2);

    Ok(dst)
}

/// Calculates the per-element absolute difference between two matrices.
///
/// dst = |src1 - src2|
pub fn absdiff<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + PartialOrd + Bounded + Default + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| *d =
        if s1 > s2 { s1 - s2 } else { s2 - s1 });

    Ok(dst)
}

/// Calculates the per-element product of two matrices.
///
/// dst = src1 * src2
pub fn multiply<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + PartialOrd + Bounded + Default + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| *d = s1 * s2);

    Ok(dst)
}

/// Calculates the per-element quotient of two matrices.
///
/// dst = src1 / src2
pub fn divide<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + PartialOrd + Bounded + Default + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| {
        if !s2.is_zero() {
            *d = s1 / s2;
        } else {
            *d = T::zero();
        }
    });

    Ok(dst)
}

/// Calculates the per-element bit-wise conjunction of two matrices.
///
/// dst = src1 & src2
pub fn bitwise_and<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + BitAnd<Output = T> + Default + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| *d = s1 & s2);

    Ok(dst)
}

/// Calculates the per-element bit-wise disjunction of two matrices.
///
/// dst = src1 | src2
pub fn bitwise_or<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + BitOr<Output = T> + Default + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| *d = s1 | s2);

    Ok(dst)
}

/// Calculates the per-element bit-wise "exclusive or" operation on two matrices.
///
/// dst = src1 ^ src2
pub fn bitwise_xor<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + BitXor<Output = T> + Default + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| *d = s1 ^ s2);

    Ok(dst)
}

/// Inverts every bit of every array element.
///
/// dst = ~src
pub fn bitwise_not<T>(src: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + Not<Output = T> + Default + 'static,
{
    let mut dst = Matrix::<T>::new(src.rows, src.cols, src.channels);

    unary_op!(dst, src, T, T, |d, s| *d = !s);

    Ok(dst)
}

/// Calculates the weighted sum of two matrices.
///
/// dst = src1*alpha + src2*beta + gamma
pub fn add_weighted<T>(
    src1: &Matrix<T>,
    alpha: f64,
    src2: &Matrix<T>,
    beta: f64,
    gamma: f64,
) -> Result<Matrix<T>>
where
    T: Num
        + Copy
        + Send
        + Sync
        + PartialOrd
        + Bounded
        + ToPrimitive
        + FromPrimitive
        + Default
        + 'static,
{
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.channels != src2.channels {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let mut dst = Matrix::<T>::new(src1.rows, src1.cols, src1.channels);

    binary_op!(dst, src1, src2, T, T, |d, s1, s2| {
        let val = s1.to_f64().unwrap_or(0.0) * alpha + s2.to_f64().unwrap_or(0.0) * beta + gamma;
        *d = T::from_f64(val).unwrap_or(T::zero());
    });

    Ok(dst)
}

/// Calculates the square root of every matrix element.
///
/// dst = sqrt(src)
pub fn sqrt<T>(src: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num
        + Copy
        + Send
        + Sync
        + PartialOrd
        + Bounded
        + ToPrimitive
        + FromPrimitive
        + Default
        + 'static,
{
    let mut dst = Matrix::<T>::new(src.rows, src.cols, src.channels);

    unary_op!(dst, src, T, T, |d, s| {
        let val = s.to_f64().unwrap_or(0.0).sqrt();
        *d = T::from_f64(val).unwrap_or(T::zero());
    });

    Ok(dst)
}

/// Calculates the exponent of every matrix element.
///
/// dst = exp(src)
pub fn exp<T>(src: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num
        + Copy
        + Send
        + Sync
        + PartialOrd
        + Bounded
        + ToPrimitive
        + FromPrimitive
        + Default
        + 'static,
{
    let mut dst = Matrix::<T>::new(src.rows, src.cols, src.channels);

    unary_op!(dst, src, T, T, |d, s| {
        let val = s.to_f64().unwrap_or(0.0).exp();
        *d = T::from_f64(val).unwrap_or(T::zero());
    });

    Ok(dst)
}

/// Calculates the natural logarithm of every matrix element.
///
/// dst = log(src)
pub fn log<T>(src: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num
        + Copy
        + Send
        + Sync
        + PartialOrd
        + Bounded
        + ToPrimitive
        + FromPrimitive
        + Default
        + 'static,
{
    let mut dst = Matrix::<T>::new(src.rows, src.cols, src.channels);

    unary_op!(dst, src, T, T, |d, s| {
        let val = s.to_f64().unwrap_or(0.0).ln();
        *d = T::from_f64(val).unwrap_or(T::zero());
    });

    Ok(dst)
}

/// Raises every matrix element to a power.
///
/// dst = src^p
pub fn pow<T>(src: &Matrix<T>, p: f64) -> Result<Matrix<T>>
where
    T: Num
        + Copy
        + Send
        + Sync
        + PartialOrd
        + Bounded
        + ToPrimitive
        + FromPrimitive
        + Default
        + 'static,
{
    let mut dst = Matrix::<T>::new(src.rows, src.cols, src.channels);

    unary_op!(dst, src, T, T, |d, s| {
        let val = s.to_f64().unwrap_or(0.0).powf(p);
        *d = T::from_f64(val).unwrap_or(T::zero());
    });

    Ok(dst)
}

/// Scales, calculates absolute values, and converts the result to 8-bit.
///
/// dst(I) = saturate_cast<u8>(|src(I)*alpha + beta|)
pub fn convert_scale_abs<T>(src: &Matrix<T>, alpha: f64, beta: f64) -> Result<Matrix<u8>>
where
    T: Num + Copy + Send + Sync + ToPrimitive + Default + 'static,
{
    let mut dst = Matrix::<u8>::new(src.rows, src.cols, src.channels);

    unary_op!(dst, src, u8, T, |d, s| {
        let val = (s.to_f64().unwrap_or(0.0) * alpha + beta).abs();
        *d = val.clamp(0.0, 255.0).round() as u8;
    });

    Ok(dst)
}

pub const GEMM_1_T: i32 = 1;
pub const GEMM_2_T: i32 = 2;
pub const GEMM_3_T: i32 = 4;

/// Performs generalized matrix multiplication.
///
/// dst = alpha * op(src1) * op(src2) + beta * op(src3)
/// where op(X) is X or X^T based on flags.
pub fn gemm<T>(
    src1: &Matrix<T>,
    src2: &Matrix<T>,
    alpha: f64,
    src3: &Matrix<T>,
    beta: f64,
    flags: i32,
) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + ToPrimitive + FromPrimitive + Default + 'static,
{
    let trans1 = (flags & GEMM_1_T) != 0;
    let trans2 = (flags & GEMM_2_T) != 0;
    let trans3 = (flags & GEMM_3_T) != 0;

    let (m, k1) = if trans1 {
        (src1.cols, src1.rows)
    } else {
        (src1.rows, src1.cols)
    };
    let (k2, n) = if trans2 {
        (src2.cols, src2.rows)
    } else {
        (src2.rows, src2.cols)
    };

    if k1 != k2 {
        return Err(PureCvError::InvalidDimensions(format!(
            "Incompatible dimensions for GEMM: {}x{} and {}x{}",
            m, k1, k2, n
        )));
    }

    let k = k1;
    let mut dst = Matrix::<T>::new(m, n, src1.channels);

    #[cfg(feature = "parallel")]
    {
        dst.data
            .par_chunks_mut(n * src1.channels)
            .enumerate()
            .for_each(|(i, row_slice)| {
                for j in 0..n {
                    for c in 0..src1.channels {
                        let mut sum = 0.0;
                        for l in 0..k {
                            let idx1 = if trans1 {
                                (l * src1.cols + i) * src1.channels + c
                            } else {
                                (i * src1.cols + l) * src1.channels + c
                            };
                            let idx2 = if trans2 {
                                (j * src2.cols + l) * src2.channels + c
                            } else {
                                (l * src2.cols + j) * src2.channels + c
                            };

                            let v1 = src1.data[idx1].to_f64().unwrap_or(0.0);
                            let v2 = src2.data[idx2].to_f64().unwrap_or(0.0);
                            sum += v1 * v2;
                        }

                        let v3 = if beta != 0.0 && src3.rows > 0 {
                            let (r3, c3) = if trans3 { (j, i) } else { (i, j) };
                            let idx3 = (r3 * src3.cols + c3) * src3.channels + c;
                            src3.data[idx3].to_f64().unwrap_or(0.0)
                        } else {
                            0.0
                        };

                        let final_val = alpha * sum + beta * v3;
                        row_slice[j * src1.channels + c] =
                            T::from_f64(final_val).unwrap_or(T::zero());
                    }
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..m {
            for j in 0..n {
                for c in 0..src1.channels {
                    let mut sum = 0.0;
                    for l in 0..k {
                        let idx1 = if trans1 {
                            (l * src1.cols + i) * src1.channels + c
                        } else {
                            (i * src1.cols + l) * src1.channels + c
                        };
                        let idx2 = if trans2 {
                            (j * src2.cols + l) * src2.channels + c
                        } else {
                            (l * src2.cols + j) * src2.channels + c
                        };

                        let v1 = src1.data[idx1].to_f64().unwrap_or(0.0);
                        let v2 = src2.data[idx2].to_f64().unwrap_or(0.0);
                        sum += v1 * v2;
                    }

                    let v3 = if beta != 0.0 && src3.rows > 0 {
                        let (r3, c3) = if trans3 { (j, i) } else { (i, j) };
                        let idx3 = (r3 * src3.cols + c3) * src3.channels + c;
                        src3.data[idx3].to_f64().unwrap_or(0.0)
                    } else {
                        0.0
                    };

                    let final_val = alpha * sum + beta * v3;
                    dst.set(i, j, c, T::from_f64(final_val).unwrap_or(T::zero()));
                }
            }
        }
    }

    Ok(dst)
}

/// Sets the matrix elements to 0, except for the diagonal which is set to a given value.
///
/// mtx = identity * s
pub fn set_identity<T>(mtx: &mut Matrix<T>, s: Scalar<T>)
where
    T: Num + Copy + Send + Sync + Default + 'static,
{
    mtx.data.fill(T::zero());
    let n = std::cmp::min(mtx.rows, mtx.cols);
    let channels = mtx.channels;
    let cols = mtx.cols;

    for i in 0..n {
        let base_idx = (i * cols + i) * channels;
        for c in 0..channels {
            mtx.data[base_idx + c] = s.v[c];
        }
    }
}

/// Checks if every array element is within a specified range.
pub fn check_range<T>(src: &Matrix<T>, min_val: f64, max_val: f64) -> bool
where
    T: Num + Copy + Send + Sync + ToPrimitive + Default + 'static,
{
    src.data.iter().all(|&val| {
        let v = val.to_f64().unwrap_or(0.0);
        v >= min_val && v <= max_val
    })
}

/// Computes the scalar dot product of two matrices (vectors).
pub fn dot<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<f64>
where
    T: Num + Copy + Send + Sync + ToPrimitive + Default + 'static,
{
    if src1.data.len() != src2.data.len() {
        return Err(PureCvError::InvalidDimensions(
            "Matrices must have the same number of elements".to_string(),
        ));
    }

    #[cfg(feature = "parallel")]
    {
        let sum: f64 = src1
            .data
            .par_iter()
            .zip(src2.data.par_iter())
            .map(|(&v1, &v2)| v1.to_f64().unwrap_or(0.0) * v2.to_f64().unwrap_or(0.0))
            .sum();
        Ok(sum)
    }

    #[cfg(not(feature = "parallel"))]
    {
        let sum: f64 = src1
            .data
            .iter()
            .zip(src2.data.iter())
            .map(|(&v1, &v2)| v1.to_f64().unwrap_or(0.0) * v2.to_f64().unwrap_or(0.0))
            .sum();
        Ok(sum)
    }
}

/// Computes the 3D cross product of two vectors.
pub fn cross<T>(src1: &Matrix<T>, src2: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Num + Copy + Send + Sync + Default + 'static,
{
    let len1 = src1.rows * src1.cols * src1.channels;
    let len2 = src2.rows * src2.cols * src2.channels;

    if len1 != 3 || len2 != 3 {
        return Err(PureCvError::InvalidDimensions(
            "Cross product requires 3-element vectors".to_string(),
        ));
    }

    let min_rows_cols = if src1.rows == 3 {
        (3, 1)
    } else if src1.cols == 3 {
        (1, 3)
    } else {
        (1, 1)
    };

    let v1 = [src1.data[0], src1.data[1], src1.data[2]];
    let v2 = [src2.data[0], src2.data[1], src2.data[2]];

    let mut dst = Matrix::<T>::new(min_rows_cols.0, min_rows_cols.1, src1.channels);
    dst.data[0] = v1[1] * v2[2] - v1[2] * v2[1];
    dst.data[1] = v1[2] * v2[0] - v1[0] * v2[2];
    dst.data[2] = v1[0] * v2[1] - v1[1] * v2[0];

    Ok(dst)
}

/// Returns the sum of diagonal elements of a matrix.
pub fn trace<T>(src: &Matrix<T>) -> Scalar<f64>
where
    T: Num + Copy + Send + Sync + ToPrimitive + Default + 'static,
{
    let n = std::cmp::min(src.rows, src.cols);
    let channels = src.channels;
    let cols = src.cols;
    let mut sum = [0.0; 4];

    for i in 0..n {
        let base_idx = (i * cols + i) * channels;
        for c in 0..channels {
            sum[c] += src.data[base_idx + c].to_f64().unwrap_or(0.0);
        }
    }

    Scalar { v: sum }
}
