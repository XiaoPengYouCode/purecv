/*
 *  hough_transform.rs
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

use image::ImageReader;
use purecv::core::{constants::CV_PI, Matrix};
use purecv::imgproc::{
    canny, cvt_color, hough_circles, hough_lines, hough_lines_p, ColorConversionCode,
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("--- Hough Transform Example with a Real Image ---");

    // Load the image from disk
    let img_path = "examples/data/detect_blob.png";
    let img = ImageReader::open(img_path)?.decode()?;
    let img_rgb = img.to_rgb8();

    // Convert the image to a purecv Matrix
    let src_rgb = Matrix::from_vec(
        img.height() as usize,
        img.width() as usize,
        3,
        img_rgb.into_raw(),
    );

    // Convert to grayscale for processing
    let gray_image = cvt_color(&src_rgb, ColorConversionCode::COLOR_RGB2GRAY)?;

    // --- 1. Hough Lines and Hough Lines P Example ---
    println!("\n--- Detecting Lines ---");

    // First, detect edges with Canny
    let edges = canny(&gray_image, 50.0, 150.0, 3, false)?;

    // --- Standard Hough Transform (hough_lines) ---
    let lines = hough_lines(&edges, 1.0, CV_PI / 180.0, 100, 0.0, CV_PI)?;
    println!("\nFound {} lines (Standard Hough Transform):", lines.len());
    for (i, line) in lines.iter().take(10).enumerate() {
        let rho = line[0];
        let theta = line[1];
        println!(
            "  Line {}: rho = {:.2}, theta = {:.2} radians",
            i, rho, theta
        );
    }
    if lines.len() > 10 {
        println!("  (and {} more...)", lines.len() - 10);
    }

    // --- Probabilistic Hough Transform (hough_lines_p) ---
    let segments = hough_lines_p(&edges, 1.0, CV_PI / 180.0, 50, 50.0, 10.0)?;
    println!(
        "\nFound {} line segments (Probabilistic Hough Transform):",
        segments.len()
    );
    for (i, segment) in segments.iter().take(10).enumerate() {
        println!(
            "  Segment {}: [({}, {}), ({}, {})]",
            i, segment[0], segment[1], segment[2], segment[3]
        );
    }
    if segments.len() > 10 {
        println!("  (and {} more...)", segments.len() - 10);
    }

    // --- 2. Hough Circles Example ---
    println!("\n--- Detecting Circles ---");

    // --- Hough Gradient Method for Circles (hough_circles) ---
    // This function takes a grayscale image directly.
    let circles = hough_circles(&gray_image, 1.0, 20.0, 90.0, 40.0, 10, 60)?;
    println!("\nFound {} circles:", circles.len());
    for (i, circle) in circles.iter().enumerate() {
        let center_x = circle[0];
        let center_y = circle[1];
        let radius = circle[2];
        println!(
            "  Circle {}: Center = ({:.2}, {:.2}), Radius = {:.2}",
            i, center_x, center_y, radius
        );
    }

    Ok(())
}
