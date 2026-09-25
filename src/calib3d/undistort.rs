/*
 *  undistort.rs
 *  purecv
 *
 *  This file is part of purecv - WebARKit.
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

use alloc::{format, string::ToString, vec::Vec};
#[allow(unused_imports)]
use num_traits::Float;

use crate::core::error::{PureCvError, Result};
use crate::core::types::{Point2f, Size2i, TermCriteria, TermType};
use crate::core::Matrix;

use super::linalg::mat3_mul;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Computes the undistortion and rectification transformation map.
///
/// The function computes the joint undistortion and rectification transformation map
/// and stores the result in `map1` and `map2`.
///
/// # Arguments
///
/// * `camera_matrix` - Input camera matrix (3x3).
/// * `dist_coeffs` - Input vector of distortion coefficients (k1, k2, p1, p2, k3, k4, k5, k6).
/// * `r` - Optional rectification transformation in object space (3x3 matrix). If `None`, identity is used.
/// * `new_camera_matrix` - New camera matrix (3x3).
/// * `size` - Undistorted image size.
///
/// # Returns
///
/// Returns a `Result<(Matrix<f32>, Matrix<f32>)>` containing the computed maps `map1` and `map2`.
///
/// # Errors
///
/// Returns an error if:
/// * `camera_matrix`, `new_camera_matrix`, or `r` (if provided) are not 3x3 single-channel matrices.
///
/// # Examples
///
/// ```
/// use purecv::core::{Matrix, types::Size2i};
/// use purecv::calib3d::undistort::init_undistort_rectify_map;
///
/// let mut camera_matrix = Matrix::<f64>::new(3, 3, 1);
/// camera_matrix.data[0] = 800.0; // fx
/// camera_matrix.data[4] = 800.0; // fy
/// camera_matrix.data[2] = 320.0; // cx
/// camera_matrix.data[5] = 240.0; // cy
/// camera_matrix.data[8] = 1.0;
///
/// let dist_coeffs = Matrix::<f64>::new(1, 5, 1); // e.g., k1, k2, p1, p2, k3
/// let size = Size2i::new(640, 480);
///
/// let (map1, map2) = init_undistort_rectify_map(
///     &camera_matrix,
///     &dist_coeffs,
///     None,
///     &camera_matrix,
///     size
/// ).unwrap();
/// ```
pub fn init_undistort_rectify_map(
    camera_matrix: &Matrix<f64>,
    dist_coeffs: &Matrix<f64>,
    r: Option<&Matrix<f64>>,
    new_camera_matrix: &Matrix<f64>,
    size: Size2i,
) -> Result<(Matrix<f32>, Matrix<f32>)> {
    if camera_matrix.rows != 3 || camera_matrix.cols != 3 || camera_matrix.channels != 1 {
        return Err(PureCvError::InvalidDimensions(
            "camera_matrix must be 3x3 single-channel".to_string(),
        ));
    }
    if new_camera_matrix.rows != 3 || new_camera_matrix.cols != 3 || new_camera_matrix.channels != 1
    {
        return Err(PureCvError::InvalidDimensions(
            "new_camera_matrix must be 3x3 single-channel".to_string(),
        ));
    }
    if let Some(r_mat) = r {
        if r_mat.rows != 3 || r_mat.cols != 3 || r_mat.channels != 1 {
            return Err(PureCvError::InvalidDimensions(
                "R must be 3x3 single-channel".to_string(),
            ));
        }
    }

    let fx = camera_matrix.data[0];
    let cx = camera_matrix.data[2];
    let fy = camera_matrix.data[4];
    let cy = camera_matrix.data[5];

    let fx_prime = new_camera_matrix.data[0];
    let cx_prime = new_camera_matrix.data[2];
    let fy_prime = new_camera_matrix.data[4];
    let cy_prime = new_camera_matrix.data[5];

    let mut k1 = 0.0;
    let mut k2 = 0.0;
    let mut p1 = 0.0;
    let mut p2 = 0.0;
    let mut k3 = 0.0;
    let mut k4 = 0.0;
    let mut k5 = 0.0;
    let mut k6 = 0.0;

    let len = dist_coeffs.data.len();
    if len >= 1 {
        k1 = dist_coeffs.data[0];
    }
    if len >= 2 {
        k2 = dist_coeffs.data[1];
    }
    if len >= 3 {
        p1 = dist_coeffs.data[2];
    }
    if len >= 4 {
        p2 = dist_coeffs.data[3];
    }
    if len >= 5 {
        k3 = dist_coeffs.data[4];
    }
    if len >= 6 {
        k4 = dist_coeffs.data[5];
    }
    if len >= 7 {
        k5 = dist_coeffs.data[6];
    }
    if len >= 8 {
        k6 = dist_coeffs.data[7];
    }

    let (r00, r01, r02, r10, r11, r12, r20, r21, r22) = match r {
        Some(r_mat) => (
            r_mat.data[0],
            r_mat.data[3],
            r_mat.data[6], // Row 0 of transpose (Col 0 of R)
            r_mat.data[1],
            r_mat.data[4],
            r_mat.data[7], // Row 1 of transpose (Col 1 of R)
            r_mat.data[2],
            r_mat.data[5],
            r_mat.data[8], // Row 2 of transpose (Col 2 of R)
        ),
        None => (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    };

    let mut map1 = Matrix::<f32>::new(size.height as usize, size.width as usize, 1);
    let mut map2 = Matrix::<f32>::new(size.height as usize, size.width as usize, 1);

    #[cfg(feature = "parallel")]
    {
        map1.data
            .par_chunks_exact_mut(size.width as usize)
            .zip(map2.data.par_chunks_exact_mut(size.width as usize))
            .enumerate()
            .for_each(|(v, (map1_row, map2_row))| {
                let y_val = v as f64;
                for u in 0..size.width as usize {
                    let x_val = u as f64;
                    let x = (x_val - cx_prime) / fx_prime;
                    let y = (y_val - cy_prime) / fy_prime;

                    let rx = r00 * x + r01 * y + r02;
                    let ry = r10 * x + r11 * y + r12;
                    let rz = r20 * x + r21 * y + r22;

                    let (xp, yp) = if rz.abs() > 1e-10 {
                        (rx / rz, ry / rz)
                    } else {
                        (x, y)
                    };

                    let r2 = xp * xp + yp * yp;
                    let radial_num = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
                    let radial_den = 1.0 + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2;
                    let radial = if radial_den.abs() > 1e-10 {
                        radial_num / radial_den
                    } else {
                        radial_num
                    };

                    let dx = 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp);
                    let dy = p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp;

                    let x_dist = xp * radial + dx;
                    let y_dist = yp * radial + dy;

                    map1_row[u] = (fx * x_dist + cx) as f32;
                    map2_row[u] = (fy * y_dist + cy) as f32;
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        let width = size.width as usize;
        let height = size.height as usize;
        for v in 0..height {
            let map1_row = &mut map1.data[v * width..(v + 1) * width];
            let map2_row = &mut map2.data[v * width..(v + 1) * width];
            let y_val = v as f64;
            for u in 0..width {
                let x_val = u as f64;
                let x = (x_val - cx_prime) / fx_prime;
                let y = (y_val - cy_prime) / fy_prime;

                let rx = r00 * x + r01 * y + r02;
                let ry = r10 * x + r11 * y + r12;
                let rz = r20 * x + r21 * y + r22;

                let (xp, yp) = if rz.abs() > 1e-10 {
                    (rx / rz, ry / rz)
                } else {
                    (x, y)
                };

                let r2 = xp * xp + yp * yp;
                let radial_num = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
                let radial_den = 1.0 + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2;
                let radial = if radial_den.abs() > 1e-10 {
                    radial_num / radial_den
                } else {
                    radial_num
                };

                let dx = 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp);
                let dy = p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp;

                let x_dist = xp * radial + dx;
                let y_dist = yp * radial + dy;

                map1_row[u] = (fx * x_dist + cx) as f32;
                map2_row[u] = (fy * y_dist + cy) as f32;
            }
        }
    }

    Ok((map1, map2))
}

// ---------------------------------------------------------------------------
// Public API — undistort_points
// ---------------------------------------------------------------------------

/// Computes the ideal point coordinates from the observed point coordinates.
///
/// Mirrors `cv::undistortPoints` for a sparse set of points: each observed
/// pixel is divided by the intrinsic matrix, the pinhole distortion model is
/// inverted with OpenCV's damped fixed-point iteration, and the result is
/// optionally rectified with `r` and re-projected through `p`.
///
/// The distortion model is OpenCV's pinhole model, truncated to the number of
/// coefficients supplied:
///
/// | `dist_coeffs.len()` | Coefficients |
/// |---------------------|--------------|
/// | `0`   | none (identity) |
/// | `4`   | `k1, k2, p1, p2` |
/// | `5`   | `k1, k2, p1, p2, k3` |
/// | `8`   | `… , k4, k5, k6` (rational radial) |
/// | `12`  | `… , s1, s2, s3, s4` (thin prism) |
/// | `14`  | `… , tau_x, tau_y` (tilted sensor) |
///
/// # Arguments
///
/// * `src`           – Observed points, in **pixel** coordinates.
/// * `camera_matrix` – 3×3 intrinsic matrix `K = [[fx,0,cx],[0,fy,cy],[0,0,1]]`.
/// * `dist_coeffs`   – Distortion coefficients (see table above); an empty slice
///   means zero distortion.
/// * `r`             – Optional 3×3 rectification rotation in object space.
/// * `p`             – Optional new camera matrix (3×3, or 3×4 of which only the
///   left 3×3 block is used, as in OpenCV).  When `None` (or identity) the
///   result is in normalized camera coordinates.
/// * `criteria`      – Termination criteria of the iterative inversion;
///   OpenCV's default is `TermCriteria::new(TermType::Count, 5, 0.01)`.
///
/// # Returns
///
/// The ideal point coordinates.  They are in normalized camera coordinates
/// unless `p` is given, in which case they are `p`-projected pixels.
///
/// # Errors
///
/// Returns [`PureCvError::InvalidDimensions`] when `camera_matrix`, `r` or `p`
/// have an unsupported shape, and [`PureCvError::InvalidInput`] when
/// `dist_coeffs` has an unsupported length or the focal lengths are zero.
///
/// # Divergences from OpenCV
///
/// | OpenCV | purecv |
/// |--------|--------|
/// | Accepts `Mat` (many layouts), returns `Mat` | Accepts `&[Point2f]`, returns `Vec<Point2f>` (always 32-bit) |
/// | `distCoeffs` may be `NULL` | Empty slice means zero distortion |
///
/// # Examples
///
/// ```
/// use purecv::core::{types::{Point2f, TermCriteria, TermType}, Matrix};
/// use purecv::calib3d::undistort::undistort_points;
///
/// let k = Matrix::from_vec(3, 3, 1, vec![800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0]);
/// let dist = [0.1, -0.05, 0.001, 0.0];
/// let points = [Point2f::new(320.0, 240.0)];
///
/// let ideal = undistort_points(
///     &points,
///     &k,
///     &dist,
///     None,
///     None,
///     TermCriteria::new(TermType::Count, 5, 0.01),
/// )
/// .unwrap();
///
/// assert!(ideal[0].x.abs() < 1e-9 && ideal[0].y.abs() < 1e-9);
/// ```
pub fn undistort_points(
    src: &[Point2f],
    camera_matrix: &Matrix<f64>,
    dist_coeffs: &[f64],
    r: Option<&Matrix<f64>>,
    p: Option<&Matrix<f64>>,
    criteria: TermCriteria,
) -> Result<Vec<Point2f>> {
    if camera_matrix.rows != 3 || camera_matrix.cols != 3 || camera_matrix.channels != 1 {
        return Err(PureCvError::InvalidDimensions(
            "camera_matrix must be 3x3 single-channel".to_string(),
        ));
    }
    if let Some(r_mat) = r {
        if r_mat.rows != 3 || r_mat.cols != 3 || r_mat.channels != 1 {
            return Err(PureCvError::InvalidDimensions(
                "R must be 3x3 single-channel".to_string(),
            ));
        }
    }
    if let Some(p_mat) = p {
        if p_mat.rows != 3 || (p_mat.cols != 3 && p_mat.cols != 4) || p_mat.channels != 1 {
            return Err(PureCvError::InvalidDimensions(
                "P must be 3x3 or 3x4 single-channel".to_string(),
            ));
        }
    }

    let k = extract_camera_matrix(camera_matrix)?;
    let model = DistortionModel::new(dist_coeffs)?;
    let ideal = undistort_normalized_points(src, &k, &model, criteria)?;

    // OpenCV applies R (then P) to the ideal normalized coordinates; the
    // translation column of a 3x4 P is ignored, as in OpenCV.
    if r.is_none() && p.is_none() {
        return Ok(ideal);
    }

    let mut rr = match r {
        Some(r_mat) => [
            r_mat.data[0],
            r_mat.data[1],
            r_mat.data[2],
            r_mat.data[3],
            r_mat.data[4],
            r_mat.data[5],
            r_mat.data[6],
            r_mat.data[7],
            r_mat.data[8],
        ],
        None => [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    };
    if let Some(p_mat) = p {
        // Only the left 3x3 block of P is used, as in OpenCV; a 3x4 P has a
        // row stride of 4.
        let stride = p_mat.cols;
        let p33 = [
            p_mat.data[0],
            p_mat.data[1],
            p_mat.data[2],
            p_mat.data[stride],
            p_mat.data[stride + 1],
            p_mat.data[stride + 2],
            p_mat.data[2 * stride],
            p_mat.data[2 * stride + 1],
            p_mat.data[2 * stride + 2],
        ];
        rr = mat3_mul(&p33, &rr);
    }

    Ok(ideal
        .into_iter()
        .map(|point| {
            let x = point.x as f64;
            let y = point.y as f64;
            let xx = rr[0] * x + rr[1] * y + rr[2];
            let yy = rr[3] * x + rr[4] * y + rr[5];
            let ww = 1.0 / (rr[6] * x + rr[7] * y + rr[8]);
            Point2f {
                x: (xx * ww) as f32,
                y: (yy * ww) as f32,
            }
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Distortion model
// ---------------------------------------------------------------------------

/// Number of coefficients of OpenCV's pinhole distortion model:
/// `k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tau_x, tau_y`.
const DIST_COEFF_COUNT: usize = 14;

/// Coefficient counts accepted by OpenCV's `undistortPoints` / `projectPoints`.
const VALID_DIST_COEFF_COUNTS: [usize; 5] = [4, 5, 8, 12, 14];

/// OpenCV's default `undistortPoints` termination criteria:
/// `TermCriteria(TermCriteria::MAX_ITER, 5, 0.01)`.
pub(crate) const DEFAULT_UNDISTORT_CRITERIA: TermCriteria = TermCriteria {
    type_: TermType::Count,
    max_count: 5,
    epsilon: 0.01,
};

/// The pinhole distortion model of `cv::projectPoints`, in normalized
/// coordinates.
///
/// `project` maps ideal (undistorted) normalized coordinates to distorted
/// normalized coordinates; [`DistortionModel::undistort_point`] inverts it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DistortionModel {
    /// `k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tau_x, tau_y`.
    k: [f64; DIST_COEFF_COUNT],
    /// Whether `tau_x`/`tau_y` are non-zero, i.e. the sensor is tilted.
    has_tilt: bool,
    /// Tilt projection matrix `matTilt` (`computeTiltProjectionMatrix`).
    mat_tilt: [f64; 9],
    /// Inverse tilt projection matrix `invMatTilt`.
    inv_mat_tilt: [f64; 9],
}

impl DistortionModel {
    /// Builds a model from a coefficient slice, validating its length the same
    /// way `cv::undistortPoints` does.
    pub(crate) fn new(dist_coeffs: &[f64]) -> Result<Self> {
        let count = dist_coeffs.len();
        if count != 0 && !VALID_DIST_COEFF_COUNTS.contains(&count) {
            return Err(PureCvError::InvalidInput(format!(
                "dist_coeffs must have 4, 5, 8, 12 or 14 elements, got {count}"
            )));
        }

        let mut k = [0.0f64; DIST_COEFF_COUNT];
        k[..count].copy_from_slice(dist_coeffs);

        let has_tilt = count == DIST_COEFF_COUNT && (k[12] != 0.0 || k[13] != 0.0);
        let (mat_tilt, inv_mat_tilt) = if has_tilt {
            tilt_projection_matrices(k[12], k[13])
        } else {
            const IDENTITY: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
            (IDENTITY, IDENTITY)
        };

        Ok(Self {
            k,
            has_tilt,
            mat_tilt,
            inv_mat_tilt,
        })
    }

    /// Applies the distortion model to ideal normalized coordinates.
    pub(crate) fn project(&self, x: f64, y: f64) -> (f64, f64) {
        let (xd0, yd0) = self.distort(x, y);
        self.apply_tilt(xd0, yd0)
    }

    /// Applies the distortion model and returns the 2×2 Jacobian
    /// `[d(xd)/dx, d(xd)/dy, d(yd)/dx, d(yd)/dy]` alongside.
    pub(crate) fn project_with_jacobian(&self, x: f64, y: f64) -> ((f64, f64), [f64; 4]) {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let cdist = 1.0 + self.k[0] * r2 + self.k[1] * r4 + self.k[4] * r4 * r2;
        let icdist2 = 1.0 / (1.0 + self.k[5] * r2 + self.k[6] * r4 + self.k[7] * r4 * r2);

        let (xd0, yd0) = self.distort(x, y);
        let (xd, yd) = self.apply_tilt(xd0, yd0);
        let tilt = self.tilt_jacobian(xd0, yd0);

        let (mx, my) = self.distort_direction(x, y, r2, r4, cdist, icdist2, 1.0, 0.0);
        let j00 = tilt[0] * mx + tilt[1] * my;
        let j10 = tilt[2] * mx + tilt[3] * my;
        let (mx, my) = self.distort_direction(x, y, r2, r4, cdist, icdist2, 0.0, 1.0);
        let j01 = tilt[0] * mx + tilt[1] * my;
        let j11 = tilt[2] * mx + tilt[3] * my;

        ((xd, yd), [j00, j01, j10, j11])
    }

    /// Inverts the distortion model for one point (OpenCV's
    /// `undistortPointsInternal`): distorted pixel `(u, v)` in, ideal
    /// normalized coordinates out.
    fn undistort_point(&self, u: f64, v: f64, k: &[f64; 9], criteria: TermCriteria) -> (f64, f64) {
        let fx = k[0];
        let fy = k[4];
        let cx = k[2];
        let cy = k[5];

        // Normalized coordinates of the observed (distorted) point.
        let px = (u - cx) / fx;
        let py = (v - cy) / fy;

        // Compensate tilt distortion:
        // s·[x''', y''', 1]ᵀ = matTilt·[x'', y'', 1]ᵀ, so the inverse is
        // s·matTilt⁻¹·[x''', y''', 1]ᵀ = [x'', y'', 1]ᵀ.
        let (mut x, mut y) = if self.has_tilt {
            let untilt = mat3_vec3(&self.inv_mat_tilt, px, py);
            let inv_proj = if untilt[2] != 0.0 {
                1.0 / untilt[2]
            } else {
                1.0
            };
            (inv_proj * untilt[0], inv_proj * untilt[1])
        } else {
            (px, py)
        };
        let (x0, y0) = (x, y);

        let check_count = matches!(criteria.type_, TermType::Count | TermType::Both);
        let check_eps = matches!(criteria.type_, TermType::Eps | TermType::Both);
        let max_count = criteria.max_count.max(0) as usize;

        let mut error = f64::MAX;
        let mut prev_error = f64::MAX;
        // Damping factor of the fixed-point iteration, as in OpenCV.
        let mut alpha = 1.0;
        let mut j = 0usize;

        loop {
            if check_count && j >= max_count {
                break;
            }
            if check_eps && error < criteria.epsilon {
                break;
            }

            let r2 = x * x + y * y;
            // icdist = (1 + k4·r² + k5·r⁴ + k6·r⁶) / (1 + k1·r² + k2·r⁴ + k3·r⁶)
            let icdist = (1.0 + ((self.k[7] * r2 + self.k[6]) * r2 + self.k[5]) * r2)
                / (1.0 + ((self.k[4] * r2 + self.k[1]) * r2 + self.k[0]) * r2);
            if icdist < 0.0 {
                // OpenCV falls back to the observed point, unchanged.
                x = px;
                y = py;
                break;
            }

            let delta_x = 2.0 * self.k[2] * x * y
                + self.k[3] * (r2 + 2.0 * x * x)
                + self.k[8] * r2
                + self.k[9] * r2 * r2;
            let delta_y = self.k[2] * (r2 + 2.0 * y * y)
                + 2.0 * self.k[3] * x * y
                + self.k[10] * r2
                + self.k[11] * r2 * r2;

            let new_x = (1.0 - alpha) * x + alpha * (x0 - delta_x) * icdist;
            let new_y = (1.0 - alpha) * y + alpha * (y0 - delta_y) * icdist;

            if check_eps {
                let (xd, yd) = self.project(new_x, new_y);
                let du = xd * fx + cx - u;
                let dv = yd * fy + cy - v;
                error = (du * du + dv * dv).sqrt();
            }

            if error > prev_error {
                alpha *= 0.5;
            } else {
                x = new_x;
                y = new_y;
            }
            prev_error = error;
            j += 1;
        }

        (x, y)
    }

    /// Radial, tangential and thin-prism distortion in normalized coordinates.
    fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let a1 = 2.0 * x * y;
        let a2 = r2 + 2.0 * x * x;
        let a3 = r2 + 2.0 * y * y;
        let cdist = 1.0 + self.k[0] * r2 + self.k[1] * r4 + self.k[4] * r4 * r2;
        let icdist2 = 1.0 / (1.0 + self.k[5] * r2 + self.k[6] * r4 + self.k[7] * r4 * r2);

        (
            x * cdist * icdist2 + self.k[2] * a1 + self.k[3] * a2 + self.k[8] * r2 + self.k[9] * r4,
            y * cdist * icdist2
                + self.k[2] * a3
                + self.k[3] * a1
                + self.k[10] * r2
                + self.k[11] * r4,
        )
    }

    /// Projects distorted normalized coordinates onto the tilt plane.
    fn apply_tilt(&self, xd0: f64, yd0: f64) -> (f64, f64) {
        if !self.has_tilt {
            return (xd0, yd0);
        }
        let vec_tilt = mat3_vec3(&self.mat_tilt, xd0, yd0);
        let inv_proj = if vec_tilt[2] != 0.0 {
            1.0 / vec_tilt[2]
        } else {
            1.0
        };
        (inv_proj * vec_tilt[0], inv_proj * vec_tilt[1])
    }

    /// 2×2 Jacobian of [`DistortionModel::apply_tilt`], scaled by `1/s²`.
    fn tilt_jacobian(&self, xd0: f64, yd0: f64) -> [f64; 4] {
        if !self.has_tilt {
            return [1.0, 0.0, 0.0, 1.0];
        }
        let vec_tilt = mat3_vec3(&self.mat_tilt, xd0, yd0);
        let inv_proj = if vec_tilt[2] != 0.0 {
            1.0 / vec_tilt[2]
        } else {
            1.0
        };
        let s = inv_proj * inv_proj;
        let m = &self.mat_tilt;
        [
            s * (m[0] * vec_tilt[2] - m[6] * vec_tilt[0]),
            s * (m[1] * vec_tilt[2] - m[7] * vec_tilt[0]),
            s * (m[3] * vec_tilt[2] - m[6] * vec_tilt[1]),
            s * (m[4] * vec_tilt[2] - m[7] * vec_tilt[1]),
        ]
    }

    /// Directional derivative of the radial/tangential/thin-prism model along
    /// `(dx, dy)`, i.e. `∂(xd0, yd0)/∂(dx, dy)` in OpenCV's `projectPoints`.
    #[allow(clippy::too_many_arguments)]
    fn distort_direction(
        &self,
        x: f64,
        y: f64,
        r2: f64,
        r4: f64,
        cdist: f64,
        icdist2: f64,
        dx: f64,
        dy: f64,
    ) -> (f64, f64) {
        let dr2 = 2.0 * x * dx + 2.0 * y * dy;
        let dcdist = (self.k[0] + 2.0 * self.k[1] * r2 + 3.0 * self.k[4] * r4) * dr2;
        let dicdist2 =
            -icdist2 * icdist2 * (self.k[5] + 2.0 * self.k[6] * r2 + 3.0 * self.k[7] * r4) * dr2;
        let da1 = 2.0 * (x * dy + y * dx);

        let mx = dx * cdist * icdist2
            + x * dcdist * icdist2
            + x * cdist * dicdist2
            + self.k[2] * da1
            + self.k[3] * (dr2 + 4.0 * x * dx)
            + (self.k[8] + 2.0 * r2 * self.k[9]) * dr2;
        let my = dy * cdist * icdist2
            + y * dcdist * icdist2
            + y * cdist * dicdist2
            + self.k[2] * (dr2 + 4.0 * y * dy)
            + self.k[3] * da1
            + (self.k[10] + 2.0 * r2 * self.k[11]) * dr2;

        (mx, my)
    }
}

/// Applies K⁻¹ and the inverse distortion model to a list of pixels, as
/// `cv::findExtrinsicCameraParams2` does before linear pose estimation.
pub(crate) fn undistort_normalized_points(
    src: &[Point2f],
    k: &[f64; 9],
    model: &DistortionModel,
    criteria: TermCriteria,
) -> Result<Vec<Point2f>> {
    if k[0] == 0.0 || k[4] == 0.0 {
        return Err(PureCvError::InvalidInput(
            "camera_matrix has a zero focal length".to_string(),
        ));
    }

    Ok(src
        .iter()
        .map(|point| {
            let (x, y) = model.undistort_point(point.x as f64, point.y as f64, k, criteria);
            Point2f {
                x: x as f32,
                y: y as f32,
            }
        })
        .collect())
}

/// Reads the 3×3 intrinsic matrix into a flat array.
fn extract_camera_matrix(camera_matrix: &Matrix<f64>) -> Result<[f64; 9]> {
    if camera_matrix.data.len() != 9 {
        return Err(PureCvError::InvalidInput(
            "camera_matrix must contain exactly 9 elements".to_string(),
        ));
    }
    camera_matrix
        .data
        .as_slice()
        .try_into()
        .map_err(|_| PureCvError::InternalError("camera_matrix layout error".to_string()))
}

/// OpenCV's `computeTiltProjectionMatrix`: returns `(matTilt, invMatTilt)` for
/// the sensor tilt angles `tau_x`, `tau_y`.
fn tilt_projection_matrices(tau_x: f64, tau_y: f64) -> ([f64; 9], [f64; 9]) {
    let (c_x, s_x) = (tau_x.cos(), tau_x.sin());
    let (c_y, s_y) = (tau_y.cos(), tau_y.sin());

    let rot_x = [1.0, 0.0, 0.0, 0.0, c_x, s_x, 0.0, -s_x, c_x];
    let rot_y = [c_y, 0.0, -s_y, 0.0, 1.0, 0.0, s_y, 0.0, c_y];
    let rot_xy = mat3_mul(&rot_y, &rot_x);

    let proj_z = [
        rot_xy[8], 0.0, -rot_xy[2], 0.0, rot_xy[8], -rot_xy[5], 0.0, 0.0, 1.0,
    ];
    let mat_tilt = mat3_mul(&proj_z, &rot_xy);

    let inv = 1.0 / rot_xy[8];
    let inv_proj_z = [
        inv,
        0.0,
        inv * rot_xy[2],
        0.0,
        inv,
        inv * rot_xy[5],
        0.0,
        0.0,
        1.0,
    ];
    let rot_xy_t = [
        rot_xy[0], rot_xy[3], rot_xy[6], rot_xy[1], rot_xy[4], rot_xy[7], rot_xy[2], rot_xy[5],
        rot_xy[8],
    ];
    let inv_mat_tilt = mat3_mul(&rot_xy_t, &inv_proj_z);

    (mat_tilt, inv_mat_tilt)
}

/// Multiplies a 3×3 row-major matrix by the homogeneous vector `[x, y, 1]ᵀ`.
fn mat3_vec3(m: &[f64; 9], x: f64, y: f64) -> [f64; 3] {
    [
        m[0] * x + m[1] * y + m[2],
        m[3] * x + m[4] * y + m[5],
        m[6] * x + m[7] * y + m[8],
    ]
}
