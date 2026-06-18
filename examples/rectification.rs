/*
 *  rectification.rs
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

use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use purecv::prelude::*;
use purecv::core::Matrix;
use purecv::version;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- purecv Rectification & Warp Example ---");
    println!("purecv v{}", version::get_version());

    // 1. Load the image
    let img_path = "examples/data/butterfly.jpg";
    if !Path::new(img_path).exists() {
        eprintln!("Error: {} not found. Run from the project root.", img_path);
        return Ok(());
    }

    let img = image::open(img_path)?;
    let (width, height) = img.dimensions();
    println!("Loaded image: {} ({}x{})", img_path, width, height);

    let rgb_img = img.to_rgb8();
    let mat_rgb = Matrix::from_vec(height as usize, width as usize, 3, rgb_img.into_raw());

    std::fs::create_dir_all("examples/data/out")?;

    // 2. Camera Undistortion
    println!("Computing camera undistortion map...");
    let camera_matrix = Matrix::from_vec(3, 3, 1, vec![
        width as f64 * 0.8, 0.0, width as f64 * 0.5,
        0.0, height as f64 * 0.8, height as f64 * 0.5,
        0.0, 0.0, 1.0,
    ]);
    // Strong barrel distortion (k1 = -0.5, k2 = 0.1)
    let dist_coeffs = Matrix::from_vec(1, 4, 1, vec![-0.5, 0.1, 0.0, 0.0]);
    let new_camera_matrix = camera_matrix.clone();

    let t = Instant::now();
    let (map1, map2) = init_undistort_rectify_map(
        &camera_matrix,
        &dist_coeffs,
        None,
        &new_camera_matrix,
        Size2i::new(width as i32, height as i32),
    )?;
    println!("  Map initialized in {:.2?}", t.elapsed());

    println!("Remapping image (camera undistortion)...");
    let t_remap = Instant::now();
    let undistorted = remap(
        &mat_rgb,
        &map1,
        &map2,
        InterpolationFlags::Linear,
        BorderTypes::Constant,
        Scalar::new(0, 0, 0, 0),
    )?;
    println!("  Remapped in {:.2?}", t_remap.elapsed());
    save_matrix_rgb(&undistorted, "examples/data/out/butterfly_undistorted.png")?;
    println!("  Saved: examples/data/out/butterfly_undistorted.png");

    // 3. Perspective Warp
    println!("Applying perspective warp...");
    // Dynamic perspective homography matrix (slight rotation, skew, and zoom)
    let m = Matrix::from_vec(3, 3, 1, vec![
        0.8, 0.1, width as f64 * 0.1,
        -0.1, 0.8, height as f64 * 0.1,
        0.0005, 0.0005, 1.0,
    ]);

    let t_warp = Instant::now();
    let warped = warp_perspective(
        &mat_rgb,
        &m,
        Size2i::new(width as i32, height as i32),
        InterpolationFlags::Linear,
        BorderTypes::Constant,
        Scalar::new(0, 0, 0, 0),
    )?;
    println!("  Perspective warp completed in {:.2?}", t_warp.elapsed());
    save_matrix_rgb(&warped, "examples/data/out/butterfly_warped.png")?;
    println!("  Saved: examples/data/out/butterfly_warped.png");

    println!("\nRectification and Warp operations applied successfully!");
    Ok(())
}

fn save_matrix_rgb(mat: &Matrix<u8>, filename: &str) -> image::ImageResult<()> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(mat.cols as u32, mat.rows as u32, mat.data.clone())
            .expect("Failed to create image buffer");
    DynamicImage::ImageRgb8(img).save(filename)
}
