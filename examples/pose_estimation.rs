/*
 *  pose_estimation.rs
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

//! Example demonstrating camera pose estimation using `solve_pnp` and `rodrigues`.

use purecv::calib3d::geometry::rodrigues;
use purecv::calib3d::pose::{solve_pnp, SolvePnPMethod};
use purecv::core::{
    types::{Point2f, Point3f},
    Matrix,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- PureCV Pose Estimation Example ---\n");

    // 1. Data Setup
    // Define a 10x10 planar marker centered at the origin
    let object_points = [
        Point3f {
            x: -5.0,
            y: -5.0,
            z: 0.0,
        },
        Point3f {
            x: 5.0,
            y: -5.0,
            z: 0.0,
        },
        Point3f {
            x: 5.0,
            y: 5.0,
            z: 0.0,
        },
        Point3f {
            x: -5.0,
            y: 5.0,
            z: 0.0,
        },
        Point3f {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        Point3f {
            x: 0.0,
            y: 5.0,
            z: 0.0,
        },
    ];

    // Simulate where these corners might project in the image
    // (Simulating a slight angle)
    let image_points = [
        Point2f { x: 260.0, y: 180.0 },
        Point2f { x: 380.0, y: 185.0 },
        Point2f { x: 390.0, y: 300.0 },
        Point2f { x: 250.0, y: 290.0 },
        Point2f { x: 320.0, y: 240.0 },
        Point2f { x: 320.0, y: 295.0 },
    ];

    // Define Camera Matrix (640x480 resolution, fx=fy=500, cx=320, cy=240)
    let camera_matrix = Matrix::<f64>::from_vec(
        3,
        3,
        1,
        vec![500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0],
    );

    // Initialize rvec and tvec
    let mut rvec = Matrix::<f64>::new(3, 1, 1);
    let mut tvec = Matrix::<f64>::new(3, 1, 1);

    // 2. Solve PnP
    println!("\n[2] Solving PnP (Iterative Method)...");
    let success = solve_pnp(
        &object_points,
        &image_points,
        &camera_matrix,
        None, // No distortion coeffs
        &mut rvec,
        &mut tvec,
        false, // use_extrinsic_guess
        SolvePnPMethod::Iterative,
    )?;

    if success {
        println!("Success! Refined pose:");
        println!(
            "rvec: [{:.4}, {:.4}, {:.4}]",
            rvec.data[0], rvec.data[1], rvec.data[2]
        );
        println!(
            "tvec: [{:.4}, {:.4}, {:.4}]",
            tvec.data[0], tvec.data[1], tvec.data[2]
        );
    } else {
        println!("Failed to solve PnP.");
        return Ok(());
    }

    // 3. Rodrigues Conversion
    println!("\n[3] Converting rvec to 3x3 Rotation Matrix...");
    let mut rmat = Matrix::<f64>::new(3, 3, 1);
    rodrigues(&rvec, &mut rmat)?;

    println!("Rotation Matrix (R):");
    println!(
        "[{:>7.4}, {:>7.4}, {:>7.4}]",
        rmat.data[0], rmat.data[1], rmat.data[2]
    );
    println!(
        "[{:>7.4}, {:>7.4}, {:>7.4}]",
        rmat.data[3], rmat.data[4], rmat.data[5]
    );
    println!(
        "[{:>7.4}, {:>7.4}, {:>7.4}]",
        rmat.data[6], rmat.data[7], rmat.data[8]
    );

    // 4. World Position Calculation
    // C = -R^T * t
    println!("\n[4] Calculating Camera Position in World Space (C = -R^T * t)...");

    // R^T elements (Transpose of R)
    let rt00 = rmat.data[0];
    let rt01 = rmat.data[3];
    let rt02 = rmat.data[6];
    let rt10 = rmat.data[1];
    let rt11 = rmat.data[4];
    let rt12 = rmat.data[7];
    let rt20 = rmat.data[2];
    let rt21 = rmat.data[5];
    let rt22 = rmat.data[8];

    let tx = tvec.data[0];
    let ty = tvec.data[1];
    let tz = tvec.data[2];

    let cx = -(rt00 * tx + rt01 * ty + rt02 * tz);
    let cy = -(rt10 * tx + rt11 * ty + rt12 * tz);
    let cz = -(rt20 * tx + rt21 * ty + rt22 * tz);

    println!("Camera World Position (X, Y, Z):");
    println!("({:.4}, {:.4}, {:.4})", cx, cy, cz);

    Ok(())
}
