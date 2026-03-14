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
/// Initializes a scaled identity matrix.
///
/// The function initializes a scaled identity matrix:
/// `dst(i,j) = s` if `i=j`, else `0`.
///
/// # Arguments
///
/// * `mtx` - Matrix to be initialized as an identity.
/// * `s` - Scalar value to be assigned to diagonal elements.
///
/// # Example
///
/// ```rust
/// use purecv::core::{Matrix, Scalar, set_identity};
///
/// let mut mat = Matrix::<f32>::new(3, 3, 1);
/// set_identity(&mut mat, Scalar::all(1.0));
/// // mat becomes a 3x3 identity matrix
/// ```
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

/// Checks if array elements lie within a specified range.
///
/// The function checks if every element of the input matrix is within the range `[min_val, max_val]`.
///
/// # Arguments
///
/// * `src` - Input matrix.
/// * `min_val` - Lower boundary of the range (inclusive).
/// * `max_val` - Upper boundary of the range (inclusive).
///
/// # Example
///
/// ```rust
/// use purecv::core::{Matrix, check_range};
///
/// let mat = Matrix::from_vec(2, 2, 1, vec![10, 20, 30, 40]);
/// let ok = check_range(&mat, 0.0, 255.0);
/// assert!(ok);
/// ```
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
///
/// The function computes the dot product of two matrices. If the matrices are not vectors,
/// they are treated as vectors by iterating through all elements.
///
/// # Arguments
///
/// * `src1` - First source matrix.
/// * `src2` - Second source matrix.
///
/// # Example
///
/// ```rust
/// use purecv::core::{Matrix, dot};
///
/// let v1 = Matrix::from_vec(1, 3, 1, vec![1.0, 2.0, 3.0]);
/// let v2 = Matrix::from_vec(1, 3, 1, vec![4.0, 5.0, 6.0]);
/// let d = dot(&v1, &v2).unwrap();
/// assert_eq!(d, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
/// ```
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
///
/// The function computes the cross product of two 3-element vectors.
///
/// # Arguments
///
/// * `src1` - First 3-element vector.
/// * `src2` - Second 3-element vector.
///
/// # Example
///
/// ```rust
/// use purecv::core::{Matrix, cross};
///
/// let v1 = Matrix::from_vec(1, 3, 1, vec![1.0, 0.0, 0.0]);
/// let v2 = Matrix::from_vec(1, 3, 1, vec![0.0, 1.0, 0.0]);
/// let v3 = cross(&v1, &v2).unwrap();
/// // v3 should be [0.0, 0.0, 1.0]
/// ```
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
///
/// The function returns the sum of diagonal elements of a matrix (the trace).
///
/// # Arguments
///
/// * `src` - Input matrix.
///
/// # Example
///
/// ```rust
/// use purecv::core::{Matrix, trace};
///
/// let mat = Matrix::from_vec(2, 2, 1, vec![1.0, 2.0, 3.0, 4.0]);
/// let t = trace(&mat);
/// assert_eq!(t.v[0], 5.0); // 1 + 4 = 5
/// ```
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
        for (c, s) in sum.iter_mut().enumerate().take(channels) {
            *s += src.data[base_idx + c].to_f64().unwrap_or(0.0);
        }
    }

    Scalar { v: sum }
}

/// Returns the determinant of a square matrix.
///
/// The function returns the determinant of a square single-channel matrix.
/// For matrices larger than 3x3, LU decomposition is used.
///
/// # Arguments
///
/// * `src` - Input square single-channel matrix.
///
/// # Example
///
/// ```rust
/// use purecv::core::{Matrix, determinant};
///
/// let mat = Matrix::from_vec(2, 2, 1, vec![1.0, 2.0, 3.0, 4.0]);
/// let det = determinant(&mat);
/// assert_eq!(det, -2.0); // 1*4 - 2*3 = -2
/// ```
pub fn determinant<T>(src: &Matrix<T>) -> f64
where
    T: Num + Copy + Send + Sync + ToPrimitive + Default + 'static,
{
    if src.rows != src.cols || src.channels != 1 {
        return 0.0;
    }

    let n = src.rows;
    if n == 0 {
        return 0.0;
    }

    match n {
        1 => src.data[0].to_f64().unwrap_or(0.0),
        2 => {
            let m = &src.data;
            let a = m[0].to_f64().unwrap_or(0.0);
            let b = m[1].to_f64().unwrap_or(0.0);
            let c = m[2].to_f64().unwrap_or(0.0);
            let d = m[3].to_f64().unwrap_or(0.0);
            a * d - b * c
        }
        3 => {
            let m = &src.data;
            let a11 = m[0].to_f64().unwrap_or(0.0);
            let a12 = m[1].to_f64().unwrap_or(0.0);
            let a13 = m[2].to_f64().unwrap_or(0.0);
            let a21 = m[3].to_f64().unwrap_or(0.0);
            let a22 = m[4].to_f64().unwrap_or(0.0);
            let a23 = m[5].to_f64().unwrap_or(0.0);
            let a31 = m[6].to_f64().unwrap_or(0.0);
            let a32 = m[7].to_f64().unwrap_or(0.0);
            let a33 = m[8].to_f64().unwrap_or(0.0);

            a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31)
                + a13 * (a21 * a32 - a22 * a31)
        }
        _ => {
            // LU decomposition for n > 3
            let mut lu = Vec::with_capacity(n * n);
            for val in &src.data {
                lu.push(val.to_f64().unwrap_or(0.0));
            }

            let mut det = 1.0;
            for i in 0..n {
                // Find pivot
                let mut pivot = i;
                for j in (i + 1)..n {
                    if lu[j * n + i].abs() > lu[pivot * n + i].abs() {
                        pivot = j;
                    }
                }

                if lu[pivot * n + i].abs() < 1e-12 {
                    return 0.0;
                }

                if pivot != i {
                    for j in i..n {
                        lu.swap(i * n + j, pivot * n + j);
                    }
                    det = -det;
                }

                let p = lu[i * n + i];
                det *= p;

                for j in (i + 1)..n {
                    let factor = lu[j * n + i] / p;
                    for k in (i + 1)..n {
                        lu[j * n + k] -= factor * lu[i * n + k];
                    }
                }
            }
            det
        }
    }
}

/// Decomposition types for solve and invert.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum DecompTypes {
    /// Gaussian elimination with optimal pivot element chosen.
    DECOMP_LU = 0,
    /// Singular value decomposition (SVD) method.
    DECOMP_SVD = 1,
    /// Eigenvalue decomposition method.
    DECOMP_EIG = 2,
    /// Cholesky decomposition; the matrix must be symmetrical and positively defined.
    DECOMP_CHOLESKY = 3,
    /// QR decomposition.
    DECOMP_QR = 4,
    /// While the other flags are mutually exclusive, this one can be combined with any of them.
    /// It means that the normal equations are solved.
    DECOMP_NORMAL = 16,
}

/// Finds the inverse of a matrix.
///
/// The function inverts the matrix `src` and stores the result in `dst`.
///
/// # Arguments
///
/// * `src` - Input square single-channel matrix.
/// * `dst` - Output matrix of the same size and type as `src`.
/// * `flags` - Inversion method (currently only DECOMP_LU is supported).
///
/// # Returns
///
/// Returns the determinant of `src`.
///
/// # Example
///
/// ```rust
/// use purecv::core::{Matrix, invert, DecompTypes};
///
/// let a = Matrix::from_vec(2, 2, 1, vec![4.0, 7.0, 2.0, 6.0]);
/// let mut inv_a = Matrix::<f64>::new(2, 2, 1);
/// let det = invert(&a, &mut inv_a, DecompTypes::DECOMP_LU).unwrap();
/// ```
pub fn invert<T>(src: &Matrix<T>, dst: &mut Matrix<f64>, flags: DecompTypes) -> Result<f64>
where
    T: Num + Copy + Send + Sync + ToPrimitive + Default + 'static,
{
    if src.rows != src.cols || src.channels != 1 {
        return Err(PureCvError::InvalidDimensions(
            "Inverse only supports single-channel square matrices".to_string(),
        ));
    }

    let n = src.rows;
    let mut identity = Matrix::<f64>::new(n, n, 1);
    set_identity(&mut identity, Scalar::new(1.0, 0.0, 0.0, 0.0));

    if solve(src, &identity, dst, flags)? {
        Ok(determinant(src))
    } else {
        dst.data.fill(0.0);
        Ok(0.0)
    }
}

/// Solves a linear system or least-squares problem.
///
/// The function solves the linear system `src1 * dst = src2`.
///
/// # Arguments
///
/// * `src1` - Input matrix A.
/// * `src2` - Input matrix B.
/// * `dst` - Output matrix X.
/// * `flags` - Solver method (currently only DECOMP_LU is supported).
///
/// # Returns
///
/// Returns `true` if the system was solved successfully.
///
/// # Example
///
/// ```rust
/// use purecv::core::{Matrix, solve, DecompTypes};
///
/// let a = Matrix::from_vec(2, 2, 1, vec![1.0, 1.0, 1.0, -1.0]);
/// let b = Matrix::from_vec(2, 1, 1, vec![2.0, 0.0]);
/// let mut x = Matrix::<f64>::new(2, 1, 1);
/// solve(&a, &b, &mut x, DecompTypes::DECOMP_LU).unwrap();
/// // x should be [1.0, 1.0] (x+y=2, x-y=0)
/// ```
pub fn solve<T, S>(
    src1: &Matrix<T>,
    src2: &Matrix<S>,
    dst: &mut Matrix<f64>,
    flags: DecompTypes,
) -> Result<bool>
where
    T: Num + Copy + Send + Sync + ToPrimitive + Default + 'static,
    S: Num + Copy + Send + Sync + ToPrimitive + Default + 'static,
{
    if src1.rows != src1.cols || src1.channels != 1 || src2.channels != 1 || src1.rows != src2.rows
    {
        return Err(PureCvError::InvalidDimensions(
            "Linear system solver requires compatible single-channel matrices".to_string(),
        ));
    }

    if flags != DecompTypes::DECOMP_LU {
        return Err(PureCvError::NotImplemented(
            "Only DECOMP_LU is currently supported".to_string(),
        ));
    }

    let n = src1.rows;
    let m = src2.cols;

    // Convert to f64 and copy to working buffers
    let mut a = Vec::with_capacity(n * n);
    for val in &src1.data {
        a.push(val.to_f64().unwrap_or(0.0));
    }

    let mut b = Vec::with_capacity(n * m);
    for val in &src2.data {
        b.push(val.to_f64().unwrap_or(0.0));
    }

    // LU decomposition with partial pivoting
    let mut p = (0..n).collect::<Vec<usize>>();
    for i in 0..n {
        let mut max_abs = 0.0;
        let mut pivot = i;
        for j in i..n {
            let abs_val = a[j * n + i].abs();
            if abs_val > max_abs {
                max_abs = abs_val;
                pivot = j;
            }
        }

        if max_abs < 1e-12 {
            return Ok(false); // Singular matrix
        }

        if pivot != i {
            for j in 0..n {
                a.swap(i * n + j, pivot * n + j);
            }
            p.swap(i, pivot);
        }

        for j in (i + 1)..n {
            a[j * n + i] /= a[i * n + i];
            for k in (i + 1)..n {
                a[j * n + k] -= a[j * n + i] * a[i * n + k];
            }
        }
    }

    // Forward and backward substitution for each column of B
    dst.rows = n;
    dst.cols = m;
    dst.channels = 1;
    dst.data.resize(n * m, 0.0);

    for col in 0..m {
        // Forward substitution (LY = PB)
        let mut y = vec![0.0; n];
        for i in 0..n {
            y[i] = b[p[i] * m + col];
            for j in 0..i {
                y[i] -= a[i * n + j] * y[j];
            }
        }

        // Backward substitution (UX = Y)
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            x[i] = y[i];
            for j in (i + 1)..n {
                x[i] -= a[i * n + j] * x[j];
            }
            x[i] /= a[i * n + i];
        }

        for (i, val) in x.iter().enumerate().take(n) {
            dst.data[i * m + col] = *val;
        }
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determinant() {
        let mut mat = Matrix::<f32>::new(2, 2, 1);
        mat.data = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(determinant(&mat), -2.0);

        let mut mat3 = Matrix::<f32>::new(3, 3, 1);
        mat3.data = vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0];
        // det = 1*(0-24) - 2*(0-20) + 3*(0-5) = -24 + 40 - 15 = 1.0
        assert_eq!(determinant(&mat3), 1.0);
    }

    #[test]
    fn test_solve() {
        let mut a = Matrix::<f32>::new(3, 3, 1);
        a.data = vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0];
        let mut b = Matrix::<f32>::new(3, 1, 1);
        b.data = vec![1.0, 2.0, 3.0];

        let mut x = Matrix::<f64>::new(0, 0, 0);
        assert!(solve(&a, &b, &mut x, DecompTypes::DECOMP_LU).unwrap());

        // Check solution: A*X = B
        // Manual calculation for this system: x=27, y=-22, z=6
        // Output for solving with b as 3x1 matrix with 1 column should be in x.data
        assert!(
            (x.data[0] - 27.0).abs() < 1e-10,
            "x failed: expected 27.0, got {}",
            x.data[0]
        );
        assert!(
            (x.data[1] - (-22.0)).abs() < 1e-10,
            "y failed: expected -22.0, got {}",
            x.data[1]
        );
        assert!(
            (x.data[2] - 6.0).abs() < 1e-10,
            "z failed: expected 6.0, got {}",
            x.data[2]
        );
    }

    #[test]
    fn test_invert() {
        let mut a = Matrix::<f32>::new(2, 2, 1);
        a.data = vec![4.0, 7.0, 2.0, 6.0];
        let mut inv_a = Matrix::<f64>::new(0, 0, 0);
        invert(&a, &mut inv_a, DecompTypes::DECOMP_LU).unwrap();

        // det = 4*6 - 7*2 = 24 - 14 = 10
        // inv = 1/10 * [6, -7; -2, 4] = [0.6, -0.7; -0.2, 0.4]
        assert!((inv_a.data[0] - 0.6).abs() < 1e-10);
        assert!((inv_a.data[1] - (-0.7)).abs() < 1e-10);
        assert!((inv_a.data[2] - (-0.2)).abs() < 1e-10);
        assert!((inv_a.data[3] - 0.4).abs() < 1e-10);
    }
}
