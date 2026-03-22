/*
 *  dynamic.rs
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
 *  Author(s): Walter Perdan <https://github.com/kalwalt>
 *
 */

use crate::core::error::Result;
use crate::core::matrix::Matrix;

/// An enum bridging type-erased dynamic usage to strongly typed generic `Matrix<T>`.
#[derive(Debug, Clone, PartialEq)]
pub enum DynamicData {
    U8(Matrix<u8>),
    I8(Matrix<i8>),
    U16(Matrix<u16>),
    I16(Matrix<i16>),
    I32(Matrix<i32>),
    F32(Matrix<f32>),
    F64(Matrix<f64>),
}

/// A type-erased matrix that holds any dynamic OpenCV-like depth.
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicMatrix {
    pub data: DynamicData,
}

/// Macro for dispatching dynamic implementations to their underlying generic variants.
#[macro_export]
macro_rules! dispatch_dynamic {
    ($data:expr, $mat:pat => $body:expr) => {
        match $data {
            $crate::core::dynamic::DynamicData::U8($mat) => $body,
            $crate::core::dynamic::DynamicData::I8($mat) => $body,
            $crate::core::dynamic::DynamicData::U16($mat) => $body,
            $crate::core::dynamic::DynamicData::I16($mat) => $body,
            $crate::core::dynamic::DynamicData::I32($mat) => $body,
            $crate::core::dynamic::DynamicData::F32($mat) => $body,
            $crate::core::dynamic::DynamicData::F64($mat) => $body,
        }
    };
}

impl DynamicMatrix {
    pub fn new_u8(rows: usize, cols: usize, channels: usize, data: Vec<u8>) -> Result<Self> {
        let mat = Matrix::from_vec(rows, cols, channels, data);
        Ok(Self {
            data: DynamicData::U8(mat),
        })
    }

    pub fn new_i8(rows: usize, cols: usize, channels: usize, data: Vec<i8>) -> Result<Self> {
        let mat = Matrix::from_vec(rows, cols, channels, data);
        Ok(Self {
            data: DynamicData::I8(mat),
        })
    }

    pub fn new_u16(rows: usize, cols: usize, channels: usize, data: Vec<u16>) -> Result<Self> {
        let mat = Matrix::from_vec(rows, cols, channels, data);
        Ok(Self {
            data: DynamicData::U16(mat),
        })
    }

    pub fn new_i16(rows: usize, cols: usize, channels: usize, data: Vec<i16>) -> Result<Self> {
        let mat = Matrix::from_vec(rows, cols, channels, data);
        Ok(Self {
            data: DynamicData::I16(mat),
        })
    }

    pub fn new_i32(rows: usize, cols: usize, channels: usize, data: Vec<i32>) -> Result<Self> {
        let mat = Matrix::from_vec(rows, cols, channels, data);
        Ok(Self {
            data: DynamicData::I32(mat),
        })
    }

    pub fn new_f32(rows: usize, cols: usize, channels: usize, data: Vec<f32>) -> Result<Self> {
        let mat = Matrix::from_vec(rows, cols, channels, data);
        Ok(Self {
            data: DynamicData::F32(mat),
        })
    }

    pub fn new_f64(rows: usize, cols: usize, channels: usize, data: Vec<f64>) -> Result<Self> {
        let mat = Matrix::from_vec(rows, cols, channels, data);
        Ok(Self {
            data: DynamicData::F64(mat),
        })
    }

    pub fn rows(&self) -> usize {
        dispatch_dynamic!(&self.data, mat => mat.rows)
    }

    pub fn cols(&self) -> usize {
        dispatch_dynamic!(&self.data, mat => mat.cols)
    }

    pub fn channels(&self) -> usize {
        dispatch_dynamic!(&self.data, mat => mat.channels)
    }

    /// Returns a human-readable name of the element depth (e.g. `"u8"`, `"f32"`).
    pub fn depth_name(&self) -> &str {
        match &self.data {
            DynamicData::U8(_) => "u8",
            DynamicData::I8(_) => "i8",
            DynamicData::U16(_) => "u16",
            DynamicData::I16(_) => "i16",
            DynamicData::I32(_) => "i32",
            DynamicData::F32(_) => "f32",
            DynamicData::F64(_) => "f64",
        }
    }

    /// Total number of elements (rows × cols × channels).
    pub fn total(&self) -> usize {
        dispatch_dynamic!(&self.data, mat => mat.data.len())
    }

    // -- Typed data accessors ------------------------------------------------

    pub fn data_u8(&self) -> Option<&[u8]> {
        match &self.data {
            DynamicData::U8(m) => Some(&m.data),
            _ => None,
        }
    }

    pub fn data_f32(&self) -> Option<&[f32]> {
        match &self.data {
            DynamicData::F32(m) => Some(&m.data),
            _ => None,
        }
    }

    pub fn data_f64(&self) -> Option<&[f64]> {
        match &self.data {
            DynamicData::F64(m) => Some(&m.data),
            _ => None,
        }
    }

    // -- Typed matrix borrow -------------------------------------------------

    pub fn as_matrix_u8(&self) -> Option<&Matrix<u8>> {
        match &self.data {
            DynamicData::U8(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_matrix_f32(&self) -> Option<&Matrix<f32>> {
        match &self.data {
            DynamicData::F32(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_matrix_f64(&self) -> Option<&Matrix<f64>> {
        match &self.data {
            DynamicData::F64(m) => Some(m),
            _ => None,
        }
    }

    // -- Read a single element as f64 (for JS interop) -----------------------

    /// Returns the element at `(row, col, channel)` cast to `f64`.
    pub fn at_f64(&self, row: i32, col: i32, channel: usize) -> Option<f64> {
        match &self.data {
            DynamicData::U8(m) => m.at(row, col, channel).map(|v| *v as f64),
            DynamicData::I8(m) => m.at(row, col, channel).map(|v| *v as f64),
            DynamicData::U16(m) => m.at(row, col, channel).map(|v| *v as f64),
            DynamicData::I16(m) => m.at(row, col, channel).map(|v| *v as f64),
            DynamicData::I32(m) => m.at(row, col, channel).map(|v| *v as f64),
            DynamicData::F32(m) => m.at(row, col, channel).map(|v| *v as f64),
            DynamicData::F64(m) => m.at(row, col, channel).copied(),
        }
    }

    // -- Type conversion -----------------------------------------------------

    /// Creates a new DynamicMatrix with a different element depth.
    pub fn convert_to(&self, depth: &str) -> Result<DynamicMatrix> {
        macro_rules! convert_inner {
            ($src_mat:expr, $depth:expr) => {
                match $depth {
                    "u8" => Ok(DynamicMatrix {
                        data: DynamicData::U8($src_mat.convert_to::<u8>()?),
                    }),
                    "i8" => Ok(DynamicMatrix {
                        data: DynamicData::I8($src_mat.convert_to::<i8>()?),
                    }),
                    "u16" => Ok(DynamicMatrix {
                        data: DynamicData::U16($src_mat.convert_to::<u16>()?),
                    }),
                    "i16" => Ok(DynamicMatrix {
                        data: DynamicData::I16($src_mat.convert_to::<i16>()?),
                    }),
                    "i32" => Ok(DynamicMatrix {
                        data: DynamicData::I32($src_mat.convert_to::<i32>()?),
                    }),
                    "f32" => Ok(DynamicMatrix {
                        data: DynamicData::F32($src_mat.convert_to::<f32>()?),
                    }),
                    "f64" => Ok(DynamicMatrix {
                        data: DynamicData::F64($src_mat.convert_to::<f64>()?),
                    }),
                    other => Err(crate::core::error::PureCvError::InvalidInput(format!(
                        "Unknown depth: {other}"
                    ))),
                }
            };
        }
        match &self.data {
            DynamicData::U8(m) => convert_inner!(m, depth),
            DynamicData::I8(m) => convert_inner!(m, depth),
            DynamicData::U16(m) => convert_inner!(m, depth),
            DynamicData::I16(m) => convert_inner!(m, depth),
            DynamicData::I32(m) => convert_inner!(m, depth),
            DynamicData::F32(m) => convert_inner!(m, depth),
            DynamicData::F64(m) => convert_inner!(m, depth),
        }
    }
}
