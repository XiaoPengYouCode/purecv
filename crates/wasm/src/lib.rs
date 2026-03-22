/*
 *  lib.rs
 *  purecv-wasm
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

use wasm_bindgen::prelude::*;

use purecv::core::arithm;
use purecv::core::dynamic::{DynamicData, DynamicMatrix};
use purecv::core::structural;
use purecv::core::types::{BorderTypes, Point2i, Size2i};
use purecv::core::Matrix;
use purecv::imgproc::color::{cvt_color, ColorConversionCode};
use purecv::imgproc::derivatives;
use purecv::imgproc::edge;
use purecv::imgproc::filter;
use purecv::imgproc::threshold::{threshold, ThresholdTypes};
use purecv::version;

// ---------------------------------------------------------------------------
//  Initialization helpers
// ---------------------------------------------------------------------------

/// Returns the current version string of the library.
#[wasm_bindgen]
pub fn get_version() -> String {
    version::get_version().to_string()
}

/// Logs the library name and version to the browser console.
#[wasm_bindgen]
pub fn print_version() {
    let msg = format!("purecv v{}", version::get_version());
    web_sys::console::log_1(&msg.into());
}

/// Initializes the panic hook for better Rust panic messages in the browser console.
#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

/// Initializes the WASM module: installs the panic hook and logs the library version.
#[wasm_bindgen]
pub fn init_purecv() {
    console_error_panic_hook::set_once();
    print_version();
}

// ---------------------------------------------------------------------------
//  Mat — unified opaque wrapper around DynamicMatrix
// ---------------------------------------------------------------------------

/// An opaque wrapper around `DynamicMatrix`, exposed as `Mat` in JavaScript.
///
/// This is the single matrix type used across the entire WASM API.
/// The element depth (`u8`, `f32`, `f64`, etc.) is selected at construction
/// time and can be queried or converted at runtime.
#[wasm_bindgen(js_name = "Mat")]
pub struct Mat {
    inner: DynamicMatrix,
}

/// Internal helper: require f32 depth, return `&Matrix<f32>` or a JsError.
fn require_f32<'a>(mat: &'a Mat, op_name: &str) -> Result<&'a Matrix<f32>, JsError> {
    mat.inner.as_matrix_f32().ok_or_else(|| {
        JsError::new(&format!(
            "{op_name} requires f32 depth, but Mat has depth '{}'",
            mat.inner.depth_name()
        ))
    })
}

/// Internal helper: require u8 depth, return `&Matrix<u8>` or a JsError.
fn require_u8<'a>(mat: &'a Mat, op_name: &str) -> Result<&'a Matrix<u8>, JsError> {
    mat.inner.as_matrix_u8().ok_or_else(|| {
        JsError::new(&format!(
            "{op_name} requires u8 depth, but Mat has depth '{}'",
            mat.inner.depth_name()
        ))
    })
}

#[wasm_bindgen(js_name = "Mat")]
impl Mat {
    // -- Constructors -------------------------------------------------------

    /// Creates a new zero-filled matrix with the given depth.
    ///
    /// * `rows`     – Number of rows (height).
    /// * `cols`     – Number of columns (width).
    /// * `channels` – Number of channels (e.g. 1 for gray, 3 for RGB, 4 for RGBA).
    /// * `depth`    – Element type: `"u8"`, `"i8"`, `"u16"`, `"i16"`, `"i32"`, `"f32"`, `"f64"`.
    ///               Defaults to `"f32"` if omitted or empty.
    #[wasm_bindgen(constructor)]
    pub fn new(
        rows: usize,
        cols: usize,
        channels: usize,
        depth: Option<String>,
    ) -> Result<Mat, JsError> {
        let depth_str = depth.unwrap_or_else(|| "f32".to_string());
        let dm = match depth_str.as_str() {
            "u8" => DynamicMatrix::new_u8(rows, cols, channels, vec![0u8; rows * cols * channels]),
            "i8" => DynamicMatrix::new_i8(rows, cols, channels, vec![0i8; rows * cols * channels]),
            "u16" => {
                DynamicMatrix::new_u16(rows, cols, channels, vec![0u16; rows * cols * channels])
            }
            "i16" => {
                DynamicMatrix::new_i16(rows, cols, channels, vec![0i16; rows * cols * channels])
            }
            "i32" => {
                DynamicMatrix::new_i32(rows, cols, channels, vec![0i32; rows * cols * channels])
            }
            "f32" => {
                DynamicMatrix::new_f32(rows, cols, channels, vec![0f32; rows * cols * channels])
            }
            "f64" => {
                DynamicMatrix::new_f64(rows, cols, channels, vec![0f64; rows * cols * channels])
            }
            other => {
                return Err(JsError::new(&format!(
                    "Unknown depth '{other}'. Use one of: u8, i8, u16, i16, i32, f32, f64"
                )));
            }
        };
        dm.map(|d| Mat { inner: d })
            .map_err(|e| JsError::new(&format!("{e}")))
    }

    /// Creates a Mat from a `Uint8Array`.
    #[wasm_bindgen(js_name = "fromU8Data")]
    pub fn from_u8_data(
        rows: usize,
        cols: usize,
        channels: usize,
        data: &[u8],
    ) -> Result<Mat, JsError> {
        let expected = rows * cols * channels;
        if data.len() != expected {
            return Err(JsError::new(&format!(
                "Data length {} does not match {}×{}×{} = {}",
                data.len(),
                rows,
                cols,
                channels,
                expected
            )));
        }
        let dm = DynamicMatrix::new_u8(rows, cols, channels, data.to_vec())
            .map_err(|e| JsError::new(&format!("{e}")))?;
        Ok(Mat { inner: dm })
    }

    /// Creates a Mat from a `Float32Array`.
    #[wasm_bindgen(js_name = "fromF32Data")]
    pub fn from_f32_data(
        rows: usize,
        cols: usize,
        channels: usize,
        data: &[f32],
    ) -> Result<Mat, JsError> {
        let expected = rows * cols * channels;
        if data.len() != expected {
            return Err(JsError::new(&format!(
                "Data length {} does not match {}×{}×{} = {}",
                data.len(),
                rows,
                cols,
                channels,
                expected
            )));
        }
        let dm = DynamicMatrix::new_f32(rows, cols, channels, data.to_vec())
            .map_err(|e| JsError::new(&format!("{e}")))?;
        Ok(Mat { inner: dm })
    }

    /// Creates a Mat from a `Float64Array`.
    #[wasm_bindgen(js_name = "fromF64Data")]
    pub fn from_f64_data(
        rows: usize,
        cols: usize,
        channels: usize,
        data: &[f64],
    ) -> Result<Mat, JsError> {
        let expected = rows * cols * channels;
        if data.len() != expected {
            return Err(JsError::new(&format!(
                "Data length {} does not match {}×{}×{} = {}",
                data.len(),
                rows,
                cols,
                channels,
                expected
            )));
        }
        let dm = DynamicMatrix::new_f64(rows, cols, channels, data.to_vec())
            .map_err(|e| JsError::new(&format!("{e}")))?;
        Ok(Mat { inner: dm })
    }

    // -- Accessors ----------------------------------------------------------

    /// Returns the number of rows (height).
    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> usize {
        self.inner.rows()
    }

    /// Returns the number of columns (width).
    #[wasm_bindgen(getter)]
    pub fn cols(&self) -> usize {
        self.inner.cols()
    }

    /// Returns the number of channels.
    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> usize {
        self.inner.channels()
    }

    /// Returns the total number of elements (rows × cols × channels).
    #[wasm_bindgen(getter, js_name = "length")]
    pub fn length(&self) -> usize {
        self.inner.total()
    }

    /// Returns the element depth as a string (e.g. `"u8"`, `"f32"`).
    #[wasm_bindgen(getter)]
    pub fn depth(&self) -> String {
        self.inner.depth_name().to_string()
    }

    // -- Data access --------------------------------------------------------

    /// Returns a copy of the underlying data as a `Uint8Array`.
    /// Errors if this Mat is not of depth `u8`.
    #[wasm_bindgen(js_name = "dataU8")]
    pub fn data_u8(&self) -> Result<Vec<u8>, JsError> {
        self.inner.data_u8().map(|s| s.to_vec()).ok_or_else(|| {
            JsError::new(&format!(
                "dataU8() requires u8 depth, but Mat has depth '{}'",
                self.inner.depth_name()
            ))
        })
    }

    /// Returns a copy of the underlying data as a `Float32Array`.
    /// Errors if this Mat is not of depth `f32`.
    #[wasm_bindgen(js_name = "dataF32")]
    pub fn data_f32(&self) -> Result<Vec<f32>, JsError> {
        self.inner.data_f32().map(|s| s.to_vec()).ok_or_else(|| {
            JsError::new(&format!(
                "dataF32() requires f32 depth, but Mat has depth '{}'",
                self.inner.depth_name()
            ))
        })
    }

    /// Returns a copy of the underlying data as a `Float64Array`.
    /// Errors if this Mat is not of depth `f64`.
    #[wasm_bindgen(js_name = "dataF64")]
    pub fn data_f64(&self) -> Result<Vec<f64>, JsError> {
        self.inner.data_f64().map(|s| s.to_vec()).ok_or_else(|| {
            JsError::new(&format!(
                "dataF64() requires f64 depth, but Mat has depth '{}'",
                self.inner.depth_name()
            ))
        })
    }

    /// Sets the underlying data from a `Uint8Array`. Errors if depth is not `u8`.
    #[wasm_bindgen(js_name = "setDataU8")]
    pub fn set_data_u8(&mut self, data: &[u8]) -> Result<(), JsError> {
        match &mut self.inner.data {
            DynamicData::U8(m) => {
                if data.len() != m.data.len() {
                    return Err(JsError::new(&format!(
                        "Data length {} does not match matrix length {}",
                        data.len(),
                        m.data.len()
                    )));
                }
                m.data.copy_from_slice(data);
                Ok(())
            }
            _ => Err(JsError::new(&format!(
                "setDataU8() requires u8 depth, but Mat has depth '{}'",
                self.inner.depth_name()
            ))),
        }
    }

    /// Sets the underlying data from a `Float32Array`. Errors if depth is not `f32`.
    #[wasm_bindgen(js_name = "setDataF32")]
    pub fn set_data_f32(&mut self, data: &[f32]) -> Result<(), JsError> {
        match &mut self.inner.data {
            DynamicData::F32(m) => {
                if data.len() != m.data.len() {
                    return Err(JsError::new(&format!(
                        "Data length {} does not match matrix length {}",
                        data.len(),
                        m.data.len()
                    )));
                }
                m.data.copy_from_slice(data);
                Ok(())
            }
            _ => Err(JsError::new(&format!(
                "setDataF32() requires f32 depth, but Mat has depth '{}'",
                self.inner.depth_name()
            ))),
        }
    }

    /// Sets the underlying data from a `Float64Array`. Errors if depth is not `f64`.
    #[wasm_bindgen(js_name = "setDataF64")]
    pub fn set_data_f64(&mut self, data: &[f64]) -> Result<(), JsError> {
        match &mut self.inner.data {
            DynamicData::F64(m) => {
                if data.len() != m.data.len() {
                    return Err(JsError::new(&format!(
                        "Data length {} does not match matrix length {}",
                        data.len(),
                        m.data.len()
                    )));
                }
                m.data.copy_from_slice(data);
                Ok(())
            }
            _ => Err(JsError::new(&format!(
                "setDataF64() requires f64 depth, but Mat has depth '{}'",
                self.inner.depth_name()
            ))),
        }
    }

    /// Returns the value at (row, col, channel) cast to `f64`.
    /// Returns `undefined` if the coordinates are out of bounds.
    #[wasm_bindgen(js_name = "at")]
    pub fn at(&self, row: i32, col: i32, channel: usize) -> Option<f64> {
        self.inner.at_f64(row, col, channel)
    }

    // -- Type conversion ----------------------------------------------------

    /// Converts this Mat to a new Mat with a different element depth.
    ///
    /// * `depth` – Target depth: `"u8"`, `"i8"`, `"u16"`, `"i16"`, `"i32"`, `"f32"`, `"f64"`.
    #[wasm_bindgen(js_name = "convertTo")]
    pub fn convert_to(&self, depth: &str) -> Result<Mat, JsError> {
        let dm = self
            .inner
            .convert_to(depth)
            .map_err(|e| JsError::new(&format!("{e}")))?;
        Ok(Mat { inner: dm })
    }
}

// ---------------------------------------------------------------------------
//  Arithmetic operations (require f32 depth)
// ---------------------------------------------------------------------------

/// Per-element addition: `dst = a + b`.
#[wasm_bindgen(js_name = "add")]
pub fn add(a: &Mat, b: &Mat) -> Result<Mat, JsError> {
    let ma = require_f32(a, "add")?;
    let mb = require_f32(b, "add")?;
    let result = arithm::add(ma, mb).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Per-element subtraction: `dst = a - b`.
#[wasm_bindgen(js_name = "subtract")]
pub fn subtract(a: &Mat, b: &Mat) -> Result<Mat, JsError> {
    let ma = require_f32(a, "subtract")?;
    let mb = require_f32(b, "subtract")?;
    let result = arithm::subtract(ma, mb).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Per-element multiplication: `dst = a * b`.
#[wasm_bindgen(js_name = "multiply")]
pub fn multiply(a: &Mat, b: &Mat) -> Result<Mat, JsError> {
    let ma = require_f32(a, "multiply")?;
    let mb = require_f32(b, "multiply")?;
    let result = arithm::multiply(ma, mb).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Per-element division: `dst = a / b`.
#[wasm_bindgen(js_name = "divide")]
pub fn divide(a: &Mat, b: &Mat) -> Result<Mat, JsError> {
    let ma = require_f32(a, "divide")?;
    let mb = require_f32(b, "divide")?;
    let result = arithm::divide(ma, mb).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Per-element absolute difference: `dst = |a - b|`.
#[wasm_bindgen(js_name = "absDiff")]
pub fn abs_diff(a: &Mat, b: &Mat) -> Result<Mat, JsError> {
    let ma = require_f32(a, "absDiff")?;
    let mb = require_f32(b, "absDiff")?;
    let result = arithm::abs_diff(ma, mb).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Per-element minimum: `dst(i) = min(a(i), b(i))`.
#[wasm_bindgen(js_name = "min")]
pub fn min(a: &Mat, b: &Mat) -> Result<Mat, JsError> {
    let ma = require_f32(a, "min")?;
    let mb = require_f32(b, "min")?;
    let result = arithm::min(ma, mb).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Per-element maximum: `dst(i) = max(a(i), b(i))`.
#[wasm_bindgen(js_name = "max")]
pub fn max(a: &Mat, b: &Mat) -> Result<Mat, JsError> {
    let ma = require_f32(a, "max")?;
    let mb = require_f32(b, "max")?;
    let result = arithm::max(ma, mb).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

// ---------------------------------------------------------------------------
//  Structural operations (dispatch across all depths)
// ---------------------------------------------------------------------------

/// Flips a matrix around vertical (0), horizontal (1), or both axes (-1).
#[wasm_bindgen(js_name = "flip")]
pub fn flip(src: &Mat, flip_code: i32) -> Result<Mat, JsError> {
    let data = match &src.inner.data {
        DynamicData::U8(m) => DynamicData::U8(
            structural::flip(m, flip_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::I8(m) => DynamicData::I8(
            structural::flip(m, flip_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::U16(m) => DynamicData::U16(
            structural::flip(m, flip_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::I16(m) => DynamicData::I16(
            structural::flip(m, flip_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::I32(m) => DynamicData::I32(
            structural::flip(m, flip_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::F32(m) => DynamicData::F32(
            structural::flip(m, flip_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::F64(m) => DynamicData::F64(
            structural::flip(m, flip_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
    };
    Ok(Mat {
        inner: DynamicMatrix { data },
    })
}

/// Transposes a matrix (swaps rows and columns).
#[wasm_bindgen(js_name = "transpose")]
pub fn transpose(src: &Mat) -> Result<Mat, JsError> {
    let data = match &src.inner.data {
        DynamicData::U8(m) => {
            DynamicData::U8(structural::transpose(m).map_err(|e| JsError::new(&format!("{e}")))?)
        }
        DynamicData::I8(m) => {
            DynamicData::I8(structural::transpose(m).map_err(|e| JsError::new(&format!("{e}")))?)
        }
        DynamicData::U16(m) => {
            DynamicData::U16(structural::transpose(m).map_err(|e| JsError::new(&format!("{e}")))?)
        }
        DynamicData::I16(m) => {
            DynamicData::I16(structural::transpose(m).map_err(|e| JsError::new(&format!("{e}")))?)
        }
        DynamicData::I32(m) => {
            DynamicData::I32(structural::transpose(m).map_err(|e| JsError::new(&format!("{e}")))?)
        }
        DynamicData::F32(m) => {
            DynamicData::F32(structural::transpose(m).map_err(|e| JsError::new(&format!("{e}")))?)
        }
        DynamicData::F64(m) => {
            DynamicData::F64(structural::transpose(m).map_err(|e| JsError::new(&format!("{e}")))?)
        }
    };
    Ok(Mat {
        inner: DynamicMatrix { data },
    })
}

/// Rotates a matrix: 0 = 90° CW, 1 = 180°, 2 = 90° CCW.
#[wasm_bindgen(js_name = "rotate")]
pub fn rotate(src: &Mat, rotate_code: i32) -> Result<Mat, JsError> {
    let data = match &src.inner.data {
        DynamicData::U8(m) => DynamicData::U8(
            structural::rotate(m, rotate_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::I8(m) => DynamicData::I8(
            structural::rotate(m, rotate_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::U16(m) => DynamicData::U16(
            structural::rotate(m, rotate_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::I16(m) => DynamicData::I16(
            structural::rotate(m, rotate_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::I32(m) => DynamicData::I32(
            structural::rotate(m, rotate_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::F32(m) => DynamicData::F32(
            structural::rotate(m, rotate_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
        DynamicData::F64(m) => DynamicData::F64(
            structural::rotate(m, rotate_code).map_err(|e| JsError::new(&format!("{e}")))?,
        ),
    };
    Ok(Mat {
        inner: DynamicMatrix { data },
    })
}

// ---------------------------------------------------------------------------
//  Color conversion (requires u8 depth)
// ---------------------------------------------------------------------------

/// Helper to convert a JS integer into a `ColorConversionCode`.
fn color_code_from_i32(code: i32) -> Result<ColorConversionCode, JsError> {
    match code {
        0 => Ok(ColorConversionCode::COLOR_BGR2GRAY),
        1 => Ok(ColorConversionCode::COLOR_RGB2GRAY),
        2 => Ok(ColorConversionCode::COLOR_BGRA2GRAY),
        3 => Ok(ColorConversionCode::COLOR_RGBA2GRAY),
        4 => Ok(ColorConversionCode::COLOR_GRAY2RGB),
        5 => Ok(ColorConversionCode::COLOR_GRAY2BGR),
        6 => Ok(ColorConversionCode::COLOR_GRAY2RGBA),
        7 => Ok(ColorConversionCode::COLOR_GRAY2BGRA),
        _ => Err(JsError::new(&format!(
            "Unknown color conversion code: {code}"
        ))),
    }
}

/// Converts an 8-bit image from one colour space to another.
///
/// Codes (integer):
///   0 = BGR2GRAY, 1 = RGB2GRAY, 2 = BGRA2GRAY, 3 = RGBA2GRAY,
///   4 = GRAY2RGB, 5 = GRAY2BGR, 6 = GRAY2RGBA, 7 = GRAY2BGRA.
#[wasm_bindgen(js_name = "cvtColor")]
pub fn convert_color(src: &Mat, code: i32) -> Result<Mat, JsError> {
    let m = require_u8(src, "cvtColor")?;
    let cc = color_code_from_i32(code)?;
    let result = cvt_color(m, cc).map_err(|e| JsError::new(e))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::U8(result),
        },
    })
}

// ---------------------------------------------------------------------------
//  Threshold (requires f32 depth)
// ---------------------------------------------------------------------------

/// Helper to convert a JS integer into a `ThresholdTypes`.
fn thresh_type_from_i32(t: i32) -> Result<ThresholdTypes, JsError> {
    match t {
        0 => Ok(ThresholdTypes::THRESH_BINARY),
        1 => Ok(ThresholdTypes::THRESH_BINARY_INV),
        2 => Ok(ThresholdTypes::THRESH_TRUNC),
        3 => Ok(ThresholdTypes::THRESH_TOZERO),
        4 => Ok(ThresholdTypes::THRESH_TOZERO_INV),
        _ => Err(JsError::new(&format!("Unknown threshold type: {t}"))),
    }
}

/// Result of the threshold operation, containing the computed threshold
/// value and the output matrix.
#[wasm_bindgen]
pub struct ThresholdResult {
    thresh_val: f64,
    matrix: Mat,
}

#[wasm_bindgen]
impl ThresholdResult {
    /// The threshold value that was used (relevant for Otsu / Triangle).
    #[wasm_bindgen(getter, js_name = "threshVal")]
    pub fn thresh_val(&self) -> f64 {
        self.thresh_val
    }

    /// The output (thresholded) matrix.  Consumes the result.
    #[wasm_bindgen(js_name = "getMatrix")]
    pub fn get_matrix(self) -> Mat {
        self.matrix
    }
}

/// Applies a fixed-level threshold to every element.
///
/// * `threshold_type`: 0 = BINARY, 1 = BINARY_INV, 2 = TRUNC, 3 = TOZERO, 4 = TOZERO_INV.
#[wasm_bindgen(js_name = "threshold")]
pub fn apply_threshold(
    src: &Mat,
    thresh: f64,
    maxval: f64,
    threshold_type: i32,
) -> Result<ThresholdResult, JsError> {
    let m = require_f32(src, "threshold")?;
    let tt = thresh_type_from_i32(threshold_type)?;
    let (tv, mat) = threshold(m, thresh, maxval, tt).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(ThresholdResult {
        thresh_val: tv,
        matrix: Mat {
            inner: DynamicMatrix {
                data: DynamicData::F32(mat),
            },
        },
    })
}

// ---------------------------------------------------------------------------
//  Edge detection (requires f32 depth)
// ---------------------------------------------------------------------------

/// Helper to convert a JS integer into a `BorderTypes`.
fn border_type_from_i32(bt: i32) -> Result<BorderTypes, JsError> {
    match bt {
        0 => Ok(BorderTypes::Constant),
        1 => Ok(BorderTypes::Replicate),
        2 => Ok(BorderTypes::Reflect),
        3 => Ok(BorderTypes::Wrap),
        4 => Ok(BorderTypes::Reflect101),
        5 => Ok(BorderTypes::Transparent),
        16 => Ok(BorderTypes::Isolated),
        _ => Err(JsError::new(&format!("Unknown border type: {bt}"))),
    }
}

/// Canny edge detection.  Input must be single-channel f32.
/// Returns a u8 edge map.
///
/// * `aperture_size` – Sobel kernel size (default: 3).
/// * `l2_gradient`   – Use L₂ norm (true) or L₁ norm (false).
#[wasm_bindgen(js_name = "canny")]
pub fn canny(
    src: &Mat,
    threshold1: f64,
    threshold2: f64,
    aperture_size: i32,
    l2_gradient: bool,
) -> Result<Mat, JsError> {
    let m = require_f32(src, "canny")?;
    let result = edge::canny(m, threshold1, threshold2, aperture_size, l2_gradient)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::U8(result),
        },
    })
}

/// Sobel derivative filter.
///
/// * `dx`, `dy`      – Order of the derivative in x / y.
/// * `ksize`         – Aperture size (1, 3, 5, or 7; -1 = Scharr).
/// * `scale`, `delta` – Scale factor and optional offset.
/// * `border_type`   – Border interpolation (integer, see `BorderTypes`).
#[wasm_bindgen(js_name = "sobel")]
pub fn sobel(
    src: &Mat,
    dx: i32,
    dy: i32,
    ksize: i32,
    scale: f64,
    delta: f64,
    border_type: i32,
) -> Result<Mat, JsError> {
    let m = require_f32(src, "sobel")?;
    let bt = border_type_from_i32(border_type)?;
    let result = derivatives::sobel(m, dx, dy, ksize, scale, delta, bt)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Scharr derivative filter (equivalent to Sobel with ksize = -1).
#[wasm_bindgen(js_name = "scharr")]
pub fn scharr(
    src: &Mat,
    dx: i32,
    dy: i32,
    scale: f64,
    delta: f64,
    border_type: i32,
) -> Result<Mat, JsError> {
    let m = require_f32(src, "scharr")?;
    let bt = border_type_from_i32(border_type)?;
    let result = derivatives::scharr(m, dx, dy, scale, delta, bt)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Laplacian of an image.
#[wasm_bindgen(js_name = "laplacian")]
pub fn laplacian(
    src: &Mat,
    ksize: i32,
    scale: f64,
    delta: f64,
    border_type: i32,
) -> Result<Mat, JsError> {
    let m = require_f32(src, "laplacian")?;
    let bt = border_type_from_i32(border_type)?;
    let result = derivatives::laplacian(m, ksize, scale, delta, bt)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

// ---------------------------------------------------------------------------
//  Blur / filter operations (require f32 depth)
// ---------------------------------------------------------------------------

/// Box blur (mean filter).
///
/// * `ksize_w`, `ksize_h` – Kernel width and height.
/// * `border_type`        – Border interpolation (integer, see `BorderTypes`).
#[wasm_bindgen(js_name = "blur")]
pub fn blur(src: &Mat, ksize_w: i32, ksize_h: i32, border_type: i32) -> Result<Mat, JsError> {
    let m = require_f32(src, "blur")?;
    let bt = border_type_from_i32(border_type)?;
    let ksize = Size2i::new(ksize_w, ksize_h);
    let anchor = Point2i::new(-1, -1);
    let result = filter::blur(m, ksize, anchor, bt).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Gaussian blur.
///
/// * `ksize_w`, `ksize_h` – Kernel width and height (must be odd).
/// * `sigma1`             – Gaussian σ in X direction (0 = auto).
/// * `sigma2`             – Gaussian σ in Y direction (0 = same as σ₁).
/// * `border_type`        – Border interpolation (integer).
#[wasm_bindgen(js_name = "gaussianBlur")]
pub fn gaussian_blur(
    src: &Mat,
    ksize_w: i32,
    ksize_h: i32,
    sigma1: f64,
    sigma2: f64,
    border_type: i32,
) -> Result<Mat, JsError> {
    let m = require_f32(src, "gaussianBlur")?;
    let bt = border_type_from_i32(border_type)?;
    let ksize = Size2i::new(ksize_w, ksize_h);
    let result = filter::gaussian_blur(m, ksize, sigma1, sigma2, bt)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Median blur.
///
/// * `ksize` – Aperture size (must be odd and > 1).
#[wasm_bindgen(js_name = "medianBlur")]
pub fn median_blur(src: &Mat, ksize: i32) -> Result<Mat, JsError> {
    let m = require_f32(src, "medianBlur")?;
    let result = filter::median_blur(m, ksize).map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

/// Bilateral filter.
///
/// * `d`           – Diameter of pixel neighbourhood (-1 = auto from sigma_space).
/// * `sigma_color` – Filter sigma in the colour space.
/// * `sigma_space` – Filter sigma in the coordinate space.
/// * `border_type` – Border interpolation (integer).
#[wasm_bindgen(js_name = "bilateralFilter")]
pub fn bilateral_filter(
    src: &Mat,
    d: i32,
    sigma_color: f64,
    sigma_space: f64,
    border_type: i32,
) -> Result<Mat, JsError> {
    let m = require_f32(src, "bilateralFilter")?;
    let bt = border_type_from_i32(border_type)?;
    let result = filter::bilateral_filter(m, d, sigma_color, sigma_space, bt)
        .map_err(|e| JsError::new(&format!("{e}")))?;
    Ok(Mat {
        inner: DynamicMatrix {
            data: DynamicData::F32(result),
        },
    })
}

// ---------------------------------------------------------------------------
//  JS-side enum constants (exposed as getter functions)
// ---------------------------------------------------------------------------

// Color conversion codes
#[wasm_bindgen(js_name = "COLOR_BGR2GRAY")]
pub fn color_bgr2gray() -> i32 {
    0
}
#[wasm_bindgen(js_name = "COLOR_RGB2GRAY")]
pub fn color_rgb2gray() -> i32 {
    1
}
#[wasm_bindgen(js_name = "COLOR_BGRA2GRAY")]
pub fn color_bgra2gray() -> i32 {
    2
}
#[wasm_bindgen(js_name = "COLOR_RGBA2GRAY")]
pub fn color_rgba2gray() -> i32 {
    3
}
#[wasm_bindgen(js_name = "COLOR_GRAY2RGB")]
pub fn color_gray2rgb() -> i32 {
    4
}
#[wasm_bindgen(js_name = "COLOR_GRAY2BGR")]
pub fn color_gray2bgr() -> i32 {
    5
}
#[wasm_bindgen(js_name = "COLOR_GRAY2RGBA")]
pub fn color_gray2rgba() -> i32 {
    6
}
#[wasm_bindgen(js_name = "COLOR_GRAY2BGRA")]
pub fn color_gray2bgra() -> i32 {
    7
}

// Threshold types
#[wasm_bindgen(js_name = "THRESH_BINARY")]
pub fn thresh_binary() -> i32 {
    0
}
#[wasm_bindgen(js_name = "THRESH_BINARY_INV")]
pub fn thresh_binary_inv() -> i32 {
    1
}
#[wasm_bindgen(js_name = "THRESH_TRUNC")]
pub fn thresh_trunc() -> i32 {
    2
}
#[wasm_bindgen(js_name = "THRESH_TOZERO")]
pub fn thresh_tozero() -> i32 {
    3
}
#[wasm_bindgen(js_name = "THRESH_TOZERO_INV")]
pub fn thresh_tozero_inv() -> i32 {
    4
}

// Border types
#[wasm_bindgen(js_name = "BORDER_CONSTANT")]
pub fn border_constant() -> i32 {
    0
}
#[wasm_bindgen(js_name = "BORDER_REPLICATE")]
pub fn border_replicate() -> i32 {
    1
}
#[wasm_bindgen(js_name = "BORDER_REFLECT")]
pub fn border_reflect() -> i32 {
    2
}
#[wasm_bindgen(js_name = "BORDER_WRAP")]
pub fn border_wrap() -> i32 {
    3
}
#[wasm_bindgen(js_name = "BORDER_REFLECT_101")]
pub fn border_reflect_101() -> i32 {
    4
}

// Flip codes
#[wasm_bindgen(js_name = "FLIP_VERTICAL")]
pub fn flip_vertical() -> i32 {
    0
}
#[wasm_bindgen(js_name = "FLIP_HORIZONTAL")]
pub fn flip_horizontal() -> i32 {
    1
}
#[wasm_bindgen(js_name = "FLIP_BOTH")]
pub fn flip_both() -> i32 {
    -1
}

// Rotate codes
#[wasm_bindgen(js_name = "ROTATE_90_CLOCKWISE")]
pub fn rotate_90_clockwise() -> i32 {
    0
}
#[wasm_bindgen(js_name = "ROTATE_180")]
pub fn rotate_180() -> i32 {
    1
}
#[wasm_bindgen(js_name = "ROTATE_90_COUNTERCLOCKWISE")]
pub fn rotate_90_counterclockwise() -> i32 {
    2
}
