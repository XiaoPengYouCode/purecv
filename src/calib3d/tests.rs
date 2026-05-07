/*
 *  tests.rs
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

#[cfg(test)]
mod calib3d_tests {
    use crate::calib3d::{find_homography, rodrigues, solve_pnp, solve_pnp_ransac};
    use crate::calib3d::{HomographyMethod, SolvePnPMethod};
    use crate::core::types::{Point2f, Point3f};
    use crate::core::Matrix;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    fn mat_approx_eq(a: &Matrix<f64>, b: &Matrix<f64>, tol: f64) -> bool {
        a.rows == b.rows
            && a.cols == b.cols
            && a.data
                .iter()
                .zip(b.data.iter())
                .all(|(&x, &y)| approx_eq(x, y, tol))
    }

    // -----------------------------------------------------------------------
    // linalg internals
    // -----------------------------------------------------------------------

    #[test]
    fn test_jacobi_eigen_2x2() {
        use crate::calib3d::linalg::jacobi_eigen;
        // A = [[2, 1], [1, 2]]  eigenvalues: 3, 1
        let mut a = [2.0f64, 1.0, 1.0, 2.0];
        let mut v = [0.0f64; 4];
        jacobi_eigen(&mut a, 2, &mut v);
        // Eigenvalues should be 1 and 3 (order may vary).
        let eigs = [a[0], a[3]];
        assert!(eigs.contains(&3.0) || eigs.iter().any(|&e| approx_eq(e, 3.0, 1e-10)));
        assert!(eigs.iter().any(|&e| approx_eq(e, 1.0, 1e-10)));
    }

    #[test]
    fn test_null_space_vector() {
        use crate::calib3d::linalg::null_space_vector;
        // A = [[1, 0], [0, 0]] — null space is [0, 1].
        let a = [1.0f64, 0.0, 0.0, 0.0];
        let ns = null_space_vector(&a, 2, 2);
        assert!(approx_eq(ns[0].abs(), 0.0, 1e-8));
        assert!(approx_eq(ns[1].abs(), 1.0, 1e-8));
    }

    // -----------------------------------------------------------------------
    // Rodrigues
    // -----------------------------------------------------------------------

    #[test]
    fn test_rodrigues_identity() {
        // Zero rotation vector → identity matrix.
        let src = Matrix::from_vec(3, 1, 1, vec![0.0f64, 0.0, 0.0]);
        let mut dst = Matrix::<f64>::new(1, 1, 1);
        rodrigues(&src, &mut dst).unwrap();
        let eye = Matrix::from_vec(3, 3, 1, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(mat_approx_eq(&dst, &eye, 1e-12));
    }

    #[test]
    fn test_rodrigues_90_deg_x_axis() {
        // Rotation of π/2 around the x-axis.
        use std::f64::consts::FRAC_PI_2;
        let src = Matrix::from_vec(3, 1, 1, vec![FRAC_PI_2, 0.0, 0.0]);
        let mut dst = Matrix::<f64>::new(1, 1, 1);
        rodrigues(&src, &mut dst).unwrap();
        assert_eq!(dst.rows, 3);
        assert_eq!(dst.cols, 3);
        // Expected: [[1,0,0],[0,0,-1],[0,1,0]]
        assert!(approx_eq(dst.data[0], 1.0, 1e-10));
        assert!(approx_eq(dst.data[4], 0.0, 1e-10));
        assert!(approx_eq(dst.data[5], -1.0, 1e-10));
        assert!(approx_eq(dst.data[7], 1.0, 1e-10));
    }

    #[test]
    fn test_rodrigues_roundtrip() {
        // rvec → rmat → rvec should return the original vector.
        let rv_orig = vec![0.3f64, -0.5, 0.8];
        let src = Matrix::from_vec(3, 1, 1, rv_orig.clone());
        let mut rmat = Matrix::<f64>::new(1, 1, 1);
        rodrigues(&src, &mut rmat).unwrap();

        let mut rv_back = Matrix::<f64>::new(1, 1, 1);
        rodrigues(&rmat, &mut rv_back).unwrap();

        for (a, b) in rv_orig.iter().zip(rv_back.data.iter()) {
            assert!(approx_eq(*a, *b, 1e-8), "roundtrip mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_rodrigues_rmat_to_rvec_output_shape() {
        let rmat = Matrix::from_vec(3, 3, 1, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let mut rv = Matrix::<f64>::new(1, 1, 1);
        rodrigues(&rmat, &mut rv).unwrap();
        assert_eq!(rv.rows, 3);
        assert_eq!(rv.cols, 1);
    }

    #[test]
    fn test_rodrigues_invalid_shape() {
        let bad = Matrix::from_vec(2, 2, 1, vec![1.0f64, 0.0, 0.0, 1.0]);
        let mut dst = Matrix::<f64>::new(1, 1, 1);
        assert!(rodrigues(&bad, &mut dst).is_err());
    }

    // -----------------------------------------------------------------------
    // find_homography — minimal point set (exact)
    // -----------------------------------------------------------------------

    /// Build a set of point correspondences that obey a known homography H.
    fn make_homography_pts(h: &[f64; 9]) -> (Vec<Point2f>, Vec<Point2f>) {
        let src: Vec<Point2f> = vec![
            Point2f { x: 0.0, y: 0.0 },
            Point2f { x: 1.0, y: 0.0 },
            Point2f { x: 1.0, y: 1.0 },
            Point2f { x: 0.0, y: 1.0 },
            Point2f { x: 0.5, y: 0.5 },
        ];
        let dst: Vec<Point2f> = src
            .iter()
            .map(|p| {
                let x = p.x as f64;
                let y = p.y as f64;
                let w = h[6] * x + h[7] * y + h[8];
                Point2f {
                    x: ((h[0] * x + h[1] * y + h[2]) / w) as f32,
                    y: ((h[3] * x + h[4] * y + h[5]) / w) as f32,
                }
            })
            .collect();
        (src, dst)
    }

    #[test]
    fn test_find_homography_identity() {
        // H = identity → dst == src.
        let h_known = [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let (src, dst) = make_homography_pts(&h_known);
        let h = find_homography(&src, &dst, HomographyMethod::None, 3.0, None).unwrap();
        // Should be close to identity (up to scale).
        for (computed, &expected) in h.data.iter().zip(h_known.iter()) {
            assert!(
                approx_eq(*computed, expected, 1e-6),
                "{computed} vs {expected}"
            );
        }
    }

    #[test]
    fn test_find_homography_pure_translation() {
        // H = [[1,0,10],[0,1,20],[0,0,1]]
        let h_known = [1.0f64, 0.0, 10.0, 0.0, 1.0, 20.0, 0.0, 0.0, 1.0];
        let (src, dst) = make_homography_pts(&h_known);
        let h = find_homography(&src, &dst, HomographyMethod::None, 3.0, None).unwrap();
        for (computed, &expected) in h.data.iter().zip(h_known.iter()) {
            assert!(
                approx_eq(*computed, expected, 1e-4),
                "{computed} vs {expected}"
            );
        }
    }

    #[test]
    fn test_find_homography_general() {
        // A general projective homography.
        let h_known = [1.2f64, 0.1, 30.0, -0.05, 0.9, 10.0, 0.001, 0.0005, 1.0];
        let (src, dst) = make_homography_pts(&h_known);
        let h = find_homography(&src, &dst, HomographyMethod::None, 3.0, None).unwrap();
        for (computed, &expected) in h.data.iter().zip(h_known.iter()) {
            assert!(
                approx_eq(*computed, expected, 1e-4),
                "{computed} vs {expected}"
            );
        }
    }

    #[test]
    fn test_find_homography_too_few_points() {
        let src = vec![Point2f { x: 0.0, y: 0.0 }, Point2f { x: 1.0, y: 0.0 }];
        let dst = src.clone();
        assert!(find_homography(&src, &dst, HomographyMethod::None, 3.0, None).is_err());
    }

    #[test]
    fn test_find_homography_ransac_no_outliers() {
        // With no outliers, RANSAC should produce the same result as plain DLT.
        let h_known = [1.0f64, 0.0, 5.0, 0.0, 1.0, 5.0, 0.0, 0.0, 1.0];
        let (src, dst) = make_homography_pts(&h_known);
        let mut mask = Vec::new();
        let h =
            find_homography(&src, &dst, HomographyMethod::Ransac, 2.0, Some(&mut mask)).unwrap();
        assert_eq!(mask.len(), src.len());
        // All should be inliers.
        assert!(mask.iter().all(|&m| m == 1));
        for (computed, &expected) in h.data.iter().zip(h_known.iter()) {
            assert!(
                approx_eq(*computed, expected, 1e-3),
                "{computed} vs {expected}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // solve_pnp
    // -----------------------------------------------------------------------

    /// Build a synthetic PnP problem with a known pose.
    fn make_pnp_data(rvec: [f64; 3], tvec: [f64; 3], k: &[f64; 9]) -> (Vec<Point3f>, Vec<Point2f>) {
        use crate::calib3d::geometry::rvec_to_rmat;
        let r = rvec_to_rmat(rvec[0], rvec[1], rvec[2]);
        let obj: Vec<Point3f> = vec![
            Point3f {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            Point3f {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            Point3f {
                x: 1.0,
                y: 1.0,
                z: 0.0,
            },
            Point3f {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            Point3f {
                x: 0.5,
                y: 0.5,
                z: 0.5,
            },
            Point3f {
                x: -0.5,
                y: 0.3,
                z: 0.2,
            },
        ];
        let img: Vec<Point2f> = obj
            .iter()
            .map(|p| {
                let cx = r[0] * p.x as f64 + r[1] * p.y as f64 + r[2] * p.z as f64 + tvec[0];
                let cy = r[3] * p.x as f64 + r[4] * p.y as f64 + r[5] * p.z as f64 + tvec[1];
                let cz = r[6] * p.x as f64 + r[7] * p.y as f64 + r[8] * p.z as f64 + tvec[2];
                let xn = cx / cz;
                let yn = cy / cz;
                let u = k[0] * xn + k[2];
                let v = k[4] * yn + k[5];
                Point2f {
                    x: u as f32,
                    y: v as f32,
                }
            })
            .collect();
        (obj, img)
    }

    fn make_camera_matrix() -> Matrix<f64> {
        Matrix::from_vec(
            3,
            3,
            1,
            vec![800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0],
        )
    }

    #[test]
    fn test_solve_pnp_small_rotation() {
        let k = [800.0f64, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0];
        let true_rv = [0.1f64, 0.05, 0.02];
        let true_tv = [0.0f64, 0.0, 5.0];
        let cam = make_camera_matrix();
        let (obj, img) = make_pnp_data(true_rv, true_tv, &k);

        let mut rvec = Matrix::<f64>::new(1, 1, 1);
        let mut tvec = Matrix::<f64>::new(1, 1, 1);
        let ok = solve_pnp(
            &obj,
            &img,
            &cam,
            None,
            &mut rvec,
            &mut tvec,
            false,
            SolvePnPMethod::Iterative,
        )
        .unwrap();
        assert!(ok);

        // Translation z should be close to 5.
        assert!(
            approx_eq(tvec.data[2], true_tv[2], 0.5),
            "tz={} expected ~5",
            tvec.data[2]
        );
    }

    #[test]
    fn test_solve_pnp_too_few_points() {
        let cam = make_camera_matrix();
        let obj = vec![
            Point3f {
                x: 0.0,
                y: 0.0,
                z: 0.0
            };
            3
        ];
        let img = vec![Point2f { x: 0.0, y: 0.0 }; 3];
        let mut rv = Matrix::<f64>::new(1, 1, 1);
        let mut tv = Matrix::<f64>::new(1, 1, 1);
        assert!(solve_pnp(
            &obj,
            &img,
            &cam,
            None,
            &mut rv,
            &mut tv,
            false,
            SolvePnPMethod::Iterative
        )
        .is_err());
    }

    // -----------------------------------------------------------------------
    // solve_pnp_ransac
    // -----------------------------------------------------------------------

    #[test]
    fn test_solve_pnp_ransac_clean_data() {
        let k = [800.0f64, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0];
        let true_rv = [0.0f64, 0.0, 0.1];
        let true_tv = [0.0f64, 0.0, 6.0];
        let cam = make_camera_matrix();
        let (obj, img) = make_pnp_data(true_rv, true_tv, &k);

        let mut rvec = Matrix::<f64>::new(1, 1, 1);
        let mut tvec = Matrix::<f64>::new(1, 1, 1);
        let mut inliers = Vec::new();
        let ok = solve_pnp_ransac(
            &obj,
            &img,
            &cam,
            None,
            &mut rvec,
            &mut tvec,
            false,
            100,
            2.0,
            0.99,
            Some(&mut inliers),
            SolvePnPMethod::Iterative,
        )
        .unwrap();
        assert!(ok);
        assert!(!inliers.is_empty());
        assert!(
            approx_eq(tvec.data[2], true_tv[2], 1.0),
            "tz={} expected ~6",
            tvec.data[2]
        );
    }
}
