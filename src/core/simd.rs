/*
 *  simd.rs
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

//! SIMD abstraction layer for PureCV.
//!
//! This module provides a trait-based dispatch system that routes element-wise
//! operations to `pulp`-powered SIMD kernels for supported types (`f32`, `f64`,
//! `u8`) and falls back to a no-op for all other types. The no-op fallback
//! causes the caller's scalar loop to execute instead, at zero cost.
//!
//! The trait [`SimdElement`] is **always compiled** (regardless of the `simd`
//! feature flag) so that generic functions can unconditionally require
//! `T: SimdElement`. When the `simd` feature is disabled, every method is a
//! default no-op and `has_simd()` returns `false`.
//!
//! # Portability
//!
//! `pulp` auto-detects CPU features at runtime (x86 SSE/AVX, ARM NEON,
//! WASM `simd128`). No target-specific `#[cfg(target_arch)]` is needed —
//! the same source compiles for native and `wasm32` targets.

// ---------------------------------------------------------------------------
//  SimdElement trait — compile-time type routing (always available)
// ---------------------------------------------------------------------------

/// Trait for types that *may* have SIMD-accelerated operations.
///
/// Default methods are all no-ops that return immediately.  Concrete
/// implementations for `f32`, `f64`, and `u8` override these methods
/// when the `simd` feature is enabled.
#[allow(dead_code)]
pub trait SimdElement: Copy + Send + Sync + 'static {
    /// Returns `true` if this type has SIMD kernel implementations
    /// *and* the `simd` feature is enabled at compile time.
    fn has_simd() -> bool {
        false
    }

    // -- Binary element-wise operations --
    // Return `true` if the operation was performed, `false` to fall back to scalar.

    fn simd_add(_dst: &mut [Self], _a: &[Self], _b: &[Self]) -> bool {
        false
    }
    fn simd_sub(_dst: &mut [Self], _a: &[Self], _b: &[Self]) -> bool {
        false
    }
    fn simd_mul(_dst: &mut [Self], _a: &[Self], _b: &[Self]) -> bool {
        false
    }
    fn simd_div(_dst: &mut [Self], _a: &[Self], _b: &[Self]) -> bool {
        false
    }
    fn simd_min(_dst: &mut [Self], _a: &[Self], _b: &[Self]) -> bool {
        false
    }
    fn simd_max(_dst: &mut [Self], _a: &[Self], _b: &[Self]) -> bool {
        false
    }

    // -- Unary operations --

    fn simd_sqrt(_dst: &mut [Self], _src: &[Self]) -> bool {
        false
    }
    fn simd_abs(_dst: &mut [Self], _src: &[Self]) -> bool {
        false
    }

    // -- Reductions --

    fn simd_dot(_a: &[Self], _b: &[Self]) -> Option<f64> {
        None
    }
    fn simd_sum(_src: &[Self]) -> Option<f64> {
        None
    }

    // -- Fused operations --

    /// dst = a * alpha + b * beta + gamma
    fn simd_add_weighted(
        _dst: &mut [Self],
        _a: &[Self],
        _b: &[Self],
        _alpha: f64,
        _beta: f64,
        _gamma: f64,
    ) -> bool {
        false
    }

    /// dst = |src * alpha + beta|, clamped to [0, 255] and written as u8.
    fn simd_convert_scale_abs(_dst: &mut [u8], _src: &[Self], _alpha: f64, _beta: f64) -> bool {
        false
    }

    /// dst = sqrt(x*x + y*y)
    fn simd_magnitude(_dst: &mut [Self], _x: &[Self], _y: &[Self]) -> bool {
        false
    }

    /// Apply threshold operation on a flat slice.
    ///
    /// `thresh_type` values: 0 = BINARY, 1 = BINARY_INV, 2 = TRUNC,
    /// 3 = TOZERO, 4 = TOZERO_INV.
    ///
    /// Returns `true` if the operation was handled by SIMD, `false` to
    /// fall back to the scalar loop.
    fn simd_threshold(
        _dst: &mut [Self],
        _src: &[Self],
        _thresh: f64,
        _maxval: f64,
        _thresh_type: u8,
    ) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
//  Default no-op implementations for f32/f64/u8 when simd is DISABLED
// ---------------------------------------------------------------------------

#[cfg(not(feature = "simd"))]
mod no_simd_impls {
    use super::SimdElement;
    impl SimdElement for f32 {}
    impl SimdElement for f64 {}
    impl SimdElement for u8 {}
}

// Types that never have SIMD support regardless of feature flags.
impl SimdElement for i8 {}
impl SimdElement for i16 {}
impl SimdElement for u16 {}
impl SimdElement for i32 {}
impl SimdElement for u32 {}
impl SimdElement for i64 {}
impl SimdElement for u64 {}

// ---------------------------------------------------------------------------
//  SIMD-enabled implementations (only when `simd` feature is active)
// ---------------------------------------------------------------------------

#[cfg(feature = "simd")]
mod simd_impls {
    use super::SimdElement;
    use pulp::Arch;

    // -----------------------------------------------------------------------
    //  f32
    // -----------------------------------------------------------------------

    impl SimdElement for f32 {
        fn has_simd() -> bool {
            true
        }

        fn simd_add(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i] + b[i];
                }
            });
            true
        }

        fn simd_sub(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i] - b[i];
                }
            });
            true
        }

        fn simd_mul(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i] * b[i];
                }
            });
            true
        }

        fn simd_div(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    if b[i] != 0.0 {
                        dst[i] = a[i] / b[i];
                    } else {
                        dst[i] = 0.0;
                    }
                }
            });
            true
        }

        fn simd_min(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = if a[i] < b[i] { a[i] } else { b[i] };
                }
            });
            true
        }

        fn simd_max(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = if a[i] > b[i] { a[i] } else { b[i] };
                }
            });
            true
        }

        fn simd_sqrt(dst: &mut [Self], src: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = src[i].sqrt();
                }
            });
            true
        }

        fn simd_abs(dst: &mut [Self], src: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = src[i].abs();
                }
            });
            true
        }

        fn simd_dot(a: &[Self], b: &[Self]) -> Option<f64> {
            // Reduction: accumulate without dispatch to avoid value-propagation
            // issues with pulp closures. LLVM auto-vectorizes the loop.
            let mut acc = 0.0f64;
            for i in 0..a.len() {
                acc += a[i] as f64 * b[i] as f64;
            }
            Some(acc)
        }

        fn simd_sum(src: &[Self]) -> Option<f64> {
            // Reduction: accumulate directly; dispatch is not needed here.
            let mut acc = 0.0f64;
            for &v in src {
                acc += v as f64;
            }
            Some(acc)
        }

        fn simd_add_weighted(
            dst: &mut [Self],
            a: &[Self],
            b: &[Self],
            alpha: f64,
            beta: f64,
            gamma: f64,
        ) -> bool {
            let alpha_f32 = alpha as f32;
            let beta_f32 = beta as f32;
            let gamma_f32 = gamma as f32;
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i] * alpha_f32 + b[i] * beta_f32 + gamma_f32;
                }
            });
            true
        }

        fn simd_convert_scale_abs(dst: &mut [u8], src: &[Self], alpha: f64, beta: f64) -> bool {
            let alpha_f32 = alpha as f32;
            let beta_f32 = beta as f32;
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    let val = (src[i] * alpha_f32 + beta_f32).abs();
                    dst[i] = val.clamp(0.0, 255.0).round() as u8;
                }
            });
            true
        }

        fn simd_magnitude(dst: &mut [Self], x: &[Self], y: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = (x[i] * x[i] + y[i] * y[i]).sqrt();
                }
            });
            true
        }

        fn simd_threshold(
            dst: &mut [Self],
            src: &[Self],
            thresh: f64,
            maxval: f64,
            thresh_type: u8,
        ) -> bool {
            if thresh_type > 4 {
                return false;
            }
            let t = thresh as f32;
            let m = maxval as f32;
            let arch = Arch::new();
            match thresh_type {
                0 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { m } else { 0.0 };
                    }
                }),
                1 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { 0.0 } else { m };
                    }
                }),
                2 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { t } else { src[i] };
                    }
                }),
                3 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { src[i] } else { 0.0 };
                    }
                }),
                4 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { 0.0 } else { src[i] };
                    }
                }),
                _ => return false,
            }
            true
        }
    }

    // -----------------------------------------------------------------------
    //  f64
    // -----------------------------------------------------------------------

    impl SimdElement for f64 {
        fn has_simd() -> bool {
            true
        }

        fn simd_add(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i] + b[i];
                }
            });
            true
        }

        fn simd_sub(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i] - b[i];
                }
            });
            true
        }

        fn simd_mul(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i] * b[i];
                }
            });
            true
        }

        fn simd_div(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    if b[i] != 0.0 {
                        dst[i] = a[i] / b[i];
                    } else {
                        dst[i] = 0.0;
                    }
                }
            });
            true
        }

        fn simd_min(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = if a[i] < b[i] { a[i] } else { b[i] };
                }
            });
            true
        }

        fn simd_max(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = if a[i] > b[i] { a[i] } else { b[i] };
                }
            });
            true
        }

        fn simd_sqrt(dst: &mut [Self], src: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = src[i].sqrt();
                }
            });
            true
        }

        fn simd_abs(dst: &mut [Self], src: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = src[i].abs();
                }
            });
            true
        }

        fn simd_dot(a: &[Self], b: &[Self]) -> Option<f64> {
            // Reduction: accumulate without dispatch to avoid value-propagation
            // issues with pulp closures. LLVM auto-vectorizes the loop.
            let mut acc = 0.0f64;
            for i in 0..a.len() {
                acc += a[i] * b[i];
            }
            Some(acc)
        }

        fn simd_sum(src: &[Self]) -> Option<f64> {
            // Reduction: accumulate directly; dispatch is not needed here.
            let mut acc = 0.0f64;
            for &v in src {
                acc += v;
            }
            Some(acc)
        }

        fn simd_add_weighted(
            dst: &mut [Self],
            a: &[Self],
            b: &[Self],
            alpha: f64,
            beta: f64,
            gamma: f64,
        ) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i] * alpha + b[i] * beta + gamma;
                }
            });
            true
        }

        fn simd_convert_scale_abs(dst: &mut [u8], src: &[Self], alpha: f64, beta: f64) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    let val = (src[i] * alpha + beta).abs();
                    dst[i] = val.clamp(0.0, 255.0).round() as u8;
                }
            });
            true
        }

        fn simd_magnitude(dst: &mut [Self], x: &[Self], y: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = (x[i] * x[i] + y[i] * y[i]).sqrt();
                }
            });
            true
        }

        fn simd_threshold(
            dst: &mut [Self],
            src: &[Self],
            thresh: f64,
            maxval: f64,
            thresh_type: u8,
        ) -> bool {
            if thresh_type > 4 {
                return false;
            }
            let t = thresh;
            let m = maxval;
            let arch = Arch::new();
            match thresh_type {
                0 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { m } else { 0.0 };
                    }
                }),
                1 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { 0.0 } else { m };
                    }
                }),
                2 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { t } else { src[i] };
                    }
                }),
                3 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { src[i] } else { 0.0 };
                    }
                }),
                4 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { 0.0 } else { src[i] };
                    }
                }),
                _ => return false,
            }
            true
        }
    }

    // -----------------------------------------------------------------------
    //  u8 (limited set of SIMD-friendly operations)
    // -----------------------------------------------------------------------

    impl SimdElement for u8 {
        fn has_simd() -> bool {
            true
        }

        fn simd_add(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i].saturating_add(b[i]);
                }
            });
            true
        }

        fn simd_sub(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i].saturating_sub(b[i]);
                }
            });
            true
        }

        fn simd_min(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i].min(b[i]);
                }
            });
            true
        }

        fn simd_max(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i].max(b[i]);
                }
            });
            true
        }

        fn simd_sum(src: &[Self]) -> Option<f64> {
            // Reduction: accumulate directly; dispatch is not needed here.
            let mut acc = 0u64;
            for &v in src {
                acc += v as u64;
            }
            Some(acc as f64)
        }

        fn simd_threshold(
            dst: &mut [Self],
            src: &[Self],
            thresh: f64,
            maxval: f64,
            thresh_type: u8,
        ) -> bool {
            if thresh_type > 4 {
                return false;
            }
            let t = thresh.clamp(0.0, 255.0) as u8;
            let m = maxval.clamp(0.0, 255.0) as u8;
            let arch = Arch::new();
            match thresh_type {
                0 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { m } else { 0 };
                    }
                }),
                1 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { 0 } else { m };
                    }
                }),
                2 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { t } else { src[i] };
                    }
                }),
                3 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { src[i] } else { 0 };
                    }
                }),
                4 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { 0 } else { src[i] };
                    }
                }),
                _ => return false,
            }
            true
        }
    }
}

// ---------------------------------------------------------------------------
//  Standalone SIMD helpers for color conversion (u8 fixed-point)
// ---------------------------------------------------------------------------

/// Converts RGB u8 pixels to grayscale using fixed-point integer arithmetic.
///
/// Coefficients: R×77 + G×150 + B×29 ≈ R×0.299 + G×0.587 + B×0.114 (×256).
/// `rgb_data` must have length `gray_data.len() * 3`.
#[cfg(feature = "simd")]
pub(crate) fn simd_rgb_to_gray_u8(gray_data: &mut [u8], rgb_data: &[u8]) {
    debug_assert_eq!(rgb_data.len(), gray_data.len() * 3);
    let arch = pulp::Arch::new();
    arch.dispatch(|| {
        for (out, inp) in gray_data.iter_mut().zip(rgb_data.chunks_exact(3)) {
            let r = inp[0] as u16;
            let g = inp[1] as u16;
            let b = inp[2] as u16;
            *out = ((r * 77 + g * 150 + b * 29 + 128) >> 8) as u8;
        }
    });
}

/// Converts BGR u8 pixels to grayscale using fixed-point integer arithmetic.
/// `bgr_data` must have length `gray_data.len() * 3`.
#[cfg(feature = "simd")]
pub(crate) fn simd_bgr_to_gray_u8(gray_data: &mut [u8], bgr_data: &[u8]) {
    debug_assert_eq!(bgr_data.len(), gray_data.len() * 3);
    let arch = pulp::Arch::new();
    arch.dispatch(|| {
        for (out, inp) in gray_data.iter_mut().zip(bgr_data.chunks_exact(3)) {
            let b = inp[0] as u16;
            let g = inp[1] as u16;
            let r = inp[2] as u16;
            *out = ((r * 77 + g * 150 + b * 29 + 128) >> 8) as u8;
        }
    });
}

/// Converts RGBA u8 pixels to grayscale (alpha ignored).
/// `rgba_data` must have length `gray_data.len() * 4`.
#[cfg(feature = "simd")]
pub(crate) fn simd_rgba_to_gray_u8(gray_data: &mut [u8], rgba_data: &[u8]) {
    debug_assert_eq!(rgba_data.len(), gray_data.len() * 4);
    let arch = pulp::Arch::new();
    arch.dispatch(|| {
        for (out, inp) in gray_data.iter_mut().zip(rgba_data.chunks_exact(4)) {
            let r = inp[0] as u16;
            let g = inp[1] as u16;
            let b = inp[2] as u16;
            *out = ((r * 77 + g * 150 + b * 29 + 128) >> 8) as u8;
        }
    });
}

/// Converts BGRA u8 pixels to grayscale (alpha ignored).
/// `bgra_data` must have length `gray_data.len() * 4`.
#[cfg(feature = "simd")]
pub(crate) fn simd_bgra_to_gray_u8(gray_data: &mut [u8], bgra_data: &[u8]) {
    debug_assert_eq!(bgra_data.len(), gray_data.len() * 4);
    let arch = pulp::Arch::new();
    arch.dispatch(|| {
        for (out, inp) in gray_data.iter_mut().zip(bgra_data.chunks_exact(4)) {
            let b = inp[0] as u16;
            let g = inp[1] as u16;
            let r = inp[2] as u16;
            *out = ((r * 77 + g * 150 + b * 29 + 128) >> 8) as u8;
        }
    });
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_simd_default_types() {
        assert!(!i8::has_simd());
        assert!(!i16::has_simd());
        assert!(!u16::has_simd());
        assert!(!i32::has_simd());
        assert!(!u32::has_simd());
        assert!(!i64::has_simd());
        assert!(!u64::has_simd());
    }

    #[cfg(feature = "simd")]
    mod simd_tests {
        use super::super::*;

        #[test]
        fn test_has_simd_for_f32() {
            assert!(f32::has_simd());
        }

        #[test]
        fn test_has_simd_for_f64() {
            assert!(f64::has_simd());
        }

        #[test]
        fn test_has_simd_for_u8() {
            assert!(u8::has_simd());
        }

        #[test]
        fn test_simd_add_f32() {
            let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let b = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
            let mut dst = vec![0.0f32; 8];
            f32::simd_add(&mut dst, &a, &b);
            assert_eq!(dst, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0]);
        }

        #[test]
        fn test_simd_sub_f32() {
            let a = vec![10.0f32, 20.0, 30.0, 40.0];
            let b = vec![1.0f32, 2.0, 3.0, 4.0];
            let mut dst = vec![0.0f32; 4];
            f32::simd_sub(&mut dst, &a, &b);
            assert_eq!(dst, vec![9.0, 18.0, 27.0, 36.0]);
        }

        #[test]
        fn test_simd_mul_f32() {
            let a = vec![2.0f32, 3.0, 4.0, 5.0];
            let b = vec![10.0f32, 10.0, 10.0, 10.0];
            let mut dst = vec![0.0f32; 4];
            f32::simd_mul(&mut dst, &a, &b);
            assert_eq!(dst, vec![20.0, 30.0, 40.0, 50.0]);
        }

        #[test]
        fn test_simd_div_f32() {
            let a = vec![10.0f32, 20.0, 30.0, 0.0];
            let b = vec![2.0f32, 5.0, 10.0, 0.0];
            let mut dst = vec![0.0f32; 4];
            f32::simd_div(&mut dst, &a, &b);
            assert_eq!(dst, vec![5.0, 4.0, 3.0, 0.0]);
        }

        #[test]
        fn test_simd_dot_f32() {
            let a = vec![1.0f32, 2.0, 3.0, 4.0];
            let b = vec![4.0f32, 3.0, 2.0, 1.0];
            let result = f32::simd_dot(&a, &b).unwrap();
            assert!((result - 20.0).abs() < 1e-6);
        }

        #[test]
        fn test_simd_magnitude_f32() {
            let x = vec![3.0f32, 0.0, 5.0];
            let y = vec![4.0f32, 3.0, 12.0];
            let mut dst = vec![0.0f32; 3];
            f32::simd_magnitude(&mut dst, &x, &y);
            assert!((dst[0] - 5.0).abs() < 1e-5);
            assert!((dst[1] - 3.0).abs() < 1e-5);
            assert!((dst[2] - 13.0).abs() < 1e-5);
        }

        #[test]
        fn test_simd_add_weighted_f32() {
            let a = vec![1.0f32, 2.0, 3.0, 4.0];
            let b = vec![10.0f32, 20.0, 30.0, 40.0];
            let mut dst = vec![0.0f32; 4];
            f32::simd_add_weighted(&mut dst, &a, &b, 0.5, 0.5, 1.0);
            assert!((dst[0] - 6.5).abs() < 1e-5);
            assert!((dst[1] - 12.0).abs() < 1e-5);
            assert!((dst[2] - 17.5).abs() < 1e-5);
            assert!((dst[3] - 23.0).abs() < 1e-5);
        }

        #[test]
        fn test_simd_add_f64() {
            let a = vec![1.0f64, 2.0, 3.0, 4.0];
            let b = vec![10.0f64, 20.0, 30.0, 40.0];
            let mut dst = vec![0.0f64; 4];
            f64::simd_add(&mut dst, &a, &b);
            assert_eq!(dst, vec![11.0, 22.0, 33.0, 44.0]);
        }

        #[test]
        fn test_simd_dot_f64() {
            let a = vec![1.0f64, 2.0, 3.0];
            let b = vec![4.0f64, 5.0, 6.0];
            let result = f64::simd_dot(&a, &b).unwrap();
            assert!((result - 32.0).abs() < 1e-12);
        }

        #[test]
        fn test_simd_add_u8_saturating() {
            let a = vec![200u8, 100, 50, 0];
            let b = vec![100u8, 200, 50, 0];
            let mut dst = vec![0u8; 4];
            u8::simd_add(&mut dst, &a, &b);
            assert_eq!(dst, vec![255, 255, 100, 0]);
        }

        #[test]
        fn test_simd_rgb_to_gray() {
            let rgb = vec![255, 0, 0, 0, 255, 0, 0, 0, 255];
            let mut gray = vec![0u8; 3];
            simd_rgb_to_gray_u8(&mut gray, &rgb);
            assert_eq!(gray[0], 77);
            assert_eq!(gray[1], 149);
            assert_eq!(gray[2], 29);
        }

        #[test]
        fn test_simd_threshold_binary_u8() {
            let src = vec![10u8, 100, 128, 200, 255, 50];
            let mut dst = vec![0u8; 6];
            assert!(u8::simd_threshold(&mut dst, &src, 127.0, 255.0, 0));
            assert_eq!(dst, vec![0, 0, 255, 255, 255, 0]);
        }

        #[test]
        fn test_simd_threshold_binary_inv_u8() {
            let src = vec![10u8, 100, 128, 200, 255, 50];
            let mut dst = vec![0u8; 6];
            assert!(u8::simd_threshold(&mut dst, &src, 127.0, 255.0, 1));
            assert_eq!(dst, vec![255, 255, 0, 0, 0, 255]);
        }

        #[test]
        fn test_simd_threshold_trunc_u8() {
            let src = vec![10u8, 100, 128, 200, 255, 50];
            let mut dst = vec![0u8; 6];
            assert!(u8::simd_threshold(&mut dst, &src, 127.0, 255.0, 2));
            assert_eq!(dst, vec![10, 100, 127, 127, 127, 50]);
        }

        #[test]
        fn test_simd_threshold_tozero_u8() {
            let src = vec![10u8, 100, 128, 200, 255, 50];
            let mut dst = vec![0u8; 6];
            assert!(u8::simd_threshold(&mut dst, &src, 127.0, 255.0, 3));
            assert_eq!(dst, vec![0, 0, 128, 200, 255, 0]);
        }

        #[test]
        fn test_simd_threshold_tozero_inv_u8() {
            let src = vec![10u8, 100, 128, 200, 255, 50];
            let mut dst = vec![0u8; 6];
            assert!(u8::simd_threshold(&mut dst, &src, 127.0, 255.0, 4));
            assert_eq!(dst, vec![10, 100, 0, 0, 0, 50]);
        }

        #[test]
        fn test_simd_threshold_binary_f32() {
            let src = vec![0.1f32, 0.4, 0.5, 0.6, 0.9, 0.3];
            let mut dst = vec![0.0f32; 6];
            assert!(f32::simd_threshold(&mut dst, &src, 0.5, 1.0, 0));
            assert_eq!(dst, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
        }

        #[test]
        fn test_simd_threshold_trunc_f32() {
            let src = vec![0.1f32, 0.4, 0.6, 0.9];
            let mut dst = vec![0.0f32; 4];
            assert!(f32::simd_threshold(&mut dst, &src, 0.5, 1.0, 2));
            assert_eq!(dst, vec![0.1, 0.4, 0.5, 0.5]);
        }

        #[test]
        fn test_simd_threshold_invalid_type() {
            let src = vec![10u8; 4];
            let mut dst = vec![0u8; 4];
            assert!(!u8::simd_threshold(&mut dst, &src, 127.0, 255.0, 5));
        }

        #[test]
        fn test_simd_convert_scale_abs_f32() {
            let src = vec![1.0f32, -2.0, 3.0, -4.0];
            let mut dst = vec![0u8; 4];
            f32::simd_convert_scale_abs(&mut dst, &src, 10.0, 0.0);
            assert_eq!(dst, vec![10, 20, 30, 40]);
        }

        // ---------------------------------------------------------------
        //  SIMD vs scalar equivalence tests
        // ---------------------------------------------------------------

        /// Helper: compute element-wise add in pure scalar Rust
        fn scalar_add_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
            a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
        }

        /// Helper: compute element-wise sub in pure scalar Rust
        fn scalar_sub_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
            a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
        }

        /// Helper: compute element-wise mul in pure scalar Rust
        fn scalar_mul_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
            a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
        }

        /// Helper: compute element-wise div in pure scalar Rust (0/0 = 0)
        fn scalar_div_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
            a.iter()
                .zip(b.iter())
                .map(|(&x, &y)| if y != 0.0 { x / y } else { 0.0 })
                .collect()
        }

        /// Helper: compute dot product in pure scalar Rust
        fn scalar_dot_f32(a: &[f32], b: &[f32]) -> f64 {
            a.iter()
                .zip(b.iter())
                .map(|(&x, &y)| x as f64 * y as f64)
                .sum()
        }

        /// Helper: compute sum in pure scalar Rust
        fn scalar_sum_f32(src: &[f32]) -> f64 {
            src.iter().map(|&v| v as f64).sum()
        }

        /// Generate a non-trivial f32 test vector
        fn make_test_vec_f32(len: usize) -> Vec<f32> {
            (0..len).map(|i| (i as f32 * 0.7) - 50.0).collect()
        }

        #[test]
        fn test_simd_vs_scalar_add_f32() {
            let a = make_test_vec_f32(1024);
            let b: Vec<f32> = (0..1024).map(|i| (i as f32) * 1.3 + 2.0).collect();
            let expected = scalar_add_f32(&a, &b);
            let mut simd_result = vec![0.0f32; 1024];
            assert!(f32::simd_add(&mut simd_result, &a, &b));
            for (i, (&e, &s)) in expected.iter().zip(simd_result.iter()).enumerate() {
                assert!(
                    (e - s).abs() < 1e-5,
                    "Mismatch at index {i}: expected {e}, got {s}"
                );
            }
        }

        #[test]
        fn test_simd_vs_scalar_sub_f32() {
            let a = make_test_vec_f32(1024);
            let b: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.3).collect();
            let expected = scalar_sub_f32(&a, &b);
            let mut simd_result = vec![0.0f32; 1024];
            assert!(f32::simd_sub(&mut simd_result, &a, &b));
            for (i, (&e, &s)) in expected.iter().zip(simd_result.iter()).enumerate() {
                assert!(
                    (e - s).abs() < 1e-5,
                    "Mismatch at index {i}: expected {e}, got {s}"
                );
            }
        }

        #[test]
        fn test_simd_vs_scalar_mul_f32() {
            let a = make_test_vec_f32(512);
            let b: Vec<f32> = (0..512).map(|i| (i as f32) * 0.01 + 0.5).collect();
            let expected = scalar_mul_f32(&a, &b);
            let mut simd_result = vec![0.0f32; 512];
            assert!(f32::simd_mul(&mut simd_result, &a, &b));
            for (i, (&e, &s)) in expected.iter().zip(simd_result.iter()).enumerate() {
                assert!(
                    (e - s).abs() < 1e-3,
                    "Mismatch at index {i}: expected {e}, got {s}"
                );
            }
        }

        #[test]
        fn test_simd_vs_scalar_div_f32() {
            let a = make_test_vec_f32(256);
            let mut b: Vec<f32> = (0..256).map(|i| (i as f32) * 0.5 + 1.0).collect();
            b[100] = 0.0; // test division by zero
            let expected = scalar_div_f32(&a, &b);
            let mut simd_result = vec![0.0f32; 256];
            assert!(f32::simd_div(&mut simd_result, &a, &b));
            for (i, (&e, &s)) in expected.iter().zip(simd_result.iter()).enumerate() {
                assert!(
                    (e - s).abs() < 1e-3,
                    "Mismatch at index {i}: expected {e}, got {s}"
                );
            }
        }

        #[test]
        fn test_simd_vs_scalar_dot_f32() {
            let a = make_test_vec_f32(2048);
            let b: Vec<f32> = (0..2048).map(|i| (i as f32) * 0.3 - 100.0).collect();
            let expected = scalar_dot_f32(&a, &b);
            let simd_result = f32::simd_dot(&a, &b).unwrap();
            assert!(
                (expected - simd_result).abs() < 1e-2,
                "Dot mismatch: expected {expected}, got {simd_result}"
            );
        }

        #[test]
        fn test_simd_vs_scalar_sum_f32() {
            let src = make_test_vec_f32(4096);
            let expected = scalar_sum_f32(&src);
            let simd_result = f32::simd_sum(&src).unwrap();
            assert!(
                (expected - simd_result).abs() < 1e-1,
                "Sum mismatch: expected {expected}, got {simd_result}"
            );
        }

        #[test]
        fn test_simd_vs_scalar_sqrt_f32() {
            let src: Vec<f32> = (0..512).map(|i| (i as f32) * 2.0 + 1.0).collect();
            let expected: Vec<f32> = src.iter().map(|&v| v.sqrt()).collect();
            let mut simd_result = vec![0.0f32; 512];
            assert!(f32::simd_sqrt(&mut simd_result, &src));
            for (i, (&e, &s)) in expected.iter().zip(simd_result.iter()).enumerate() {
                assert!(
                    (e - s).abs() < 1e-5,
                    "Sqrt mismatch at index {i}: expected {e}, got {s}"
                );
            }
        }

        #[test]
        fn test_simd_vs_scalar_min_max_f32() {
            let a = make_test_vec_f32(256);
            let b: Vec<f32> = (0..256).map(|i| (i as f32) * 0.5 - 30.0).collect();

            // min
            let expected_min: Vec<f32> = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| if x < y { x } else { y })
                .collect();
            let mut simd_min_result = vec![0.0f32; 256];
            assert!(f32::simd_min(&mut simd_min_result, &a, &b));
            assert_eq!(simd_min_result, expected_min);

            // max
            let expected_max: Vec<f32> = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| if x > y { x } else { y })
                .collect();
            let mut simd_max_result = vec![0.0f32; 256];
            assert!(f32::simd_max(&mut simd_max_result, &a, &b));
            assert_eq!(simd_max_result, expected_max);
        }

        #[test]
        fn test_simd_vs_scalar_magnitude_f32() {
            let x: Vec<f32> = (0..128).map(|i| (i as f32) * 0.5).collect();
            let y: Vec<f32> = (0..128).map(|i| (i as f32) * 0.3 + 1.0).collect();
            let expected: Vec<f32> = x
                .iter()
                .zip(y.iter())
                .map(|(&xv, &yv)| (xv * xv + yv * yv).sqrt())
                .collect();
            let mut simd_result = vec![0.0f32; 128];
            assert!(f32::simd_magnitude(&mut simd_result, &x, &y));
            for (i, (&e, &s)) in expected.iter().zip(simd_result.iter()).enumerate() {
                assert!(
                    (e - s).abs() < 1e-4,
                    "Magnitude mismatch at index {i}: expected {e}, got {s}"
                );
            }
        }

        #[test]
        fn test_simd_vs_scalar_add_weighted_f32() {
            let a = make_test_vec_f32(256);
            let b: Vec<f32> = (0..256).map(|i| (i as f32) * 0.2 + 3.0).collect();
            let alpha = 0.6f64;
            let beta = 0.4f64;
            let gamma = 2.5f64;
            let expected: Vec<f32> = a
                .iter()
                .zip(b.iter())
                .map(|(&av, &bv)| (av as f64 * alpha + bv as f64 * beta + gamma) as f32)
                .collect();
            let mut simd_result = vec![0.0f32; 256];
            assert!(f32::simd_add_weighted(
                &mut simd_result,
                &a,
                &b,
                alpha,
                beta,
                gamma,
            ));
            for (i, (&e, &s)) in expected.iter().zip(simd_result.iter()).enumerate() {
                assert!(
                    (e - s).abs() < 1e-3,
                    "AddWeighted mismatch at index {i}: expected {e}, got {s}"
                );
            }
        }

        #[test]
        fn test_simd_vs_scalar_convert_scale_abs_f32() {
            let src: Vec<f32> = (0..256).map(|i| (i as f32) * 0.5 - 64.0).collect();
            let alpha = 2.0f64;
            let beta = 10.0f64;
            let expected: Vec<u8> = src
                .iter()
                .map(|&v| {
                    let val = ((v as f64) * alpha + beta).abs();
                    val.clamp(0.0, 255.0).round() as u8
                })
                .collect();
            let mut simd_result = vec![0u8; 256];
            assert!(f32::simd_convert_scale_abs(
                &mut simd_result,
                &src,
                alpha,
                beta,
            ));
            assert_eq!(simd_result, expected);
        }

        // --- f64 equivalence tests ---

        #[test]
        fn test_simd_vs_scalar_add_f64() {
            let a: Vec<f64> = (0..512).map(|i| (i as f64) * 0.7 - 100.0).collect();
            let b: Vec<f64> = (0..512).map(|i| (i as f64) * 1.3 + 50.0).collect();
            let expected: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
            let mut simd_result = vec![0.0f64; 512];
            assert!(f64::simd_add(&mut simd_result, &a, &b));
            assert_eq!(simd_result, expected);
        }

        #[test]
        fn test_simd_vs_scalar_dot_f64() {
            let a: Vec<f64> = (0..1024).map(|i| (i as f64) * 0.3 - 100.0).collect();
            let b: Vec<f64> = (0..1024).map(|i| (i as f64) * 0.7 + 20.0).collect();
            let expected: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
            let simd_result = f64::simd_dot(&a, &b).unwrap();
            assert!(
                (expected - simd_result).abs() < 1e-6,
                "f64 dot mismatch: expected {expected}, got {simd_result}"
            );
        }

        #[test]
        fn test_simd_vs_scalar_sum_f64() {
            let src: Vec<f64> = (0..2048).map(|i| (i as f64) * 0.3 - 300.0).collect();
            let expected: f64 = src.iter().sum();
            let simd_result = f64::simd_sum(&src).unwrap();
            assert!(
                (expected - simd_result).abs() < 1e-6,
                "f64 sum mismatch: expected {expected}, got {simd_result}"
            );
        }

        // --- u8 equivalence tests ---

        #[test]
        fn test_simd_vs_scalar_add_u8_equiv() {
            let a: Vec<u8> = (0..256).map(|i| i as u8).collect();
            let b: Vec<u8> = (0..256).map(|i| (255 - i) as u8).collect();
            let expected: Vec<u8> = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| x.saturating_add(y))
                .collect();
            let mut simd_result = vec![0u8; 256];
            assert!(u8::simd_add(&mut simd_result, &a, &b));
            assert_eq!(simd_result, expected);
        }

        #[test]
        fn test_simd_vs_scalar_sub_u8_equiv() {
            let a: Vec<u8> = (0..256).map(|i| i as u8).collect();
            let b: Vec<u8> = (0..256).map(|i| (i / 2) as u8).collect();
            let expected: Vec<u8> = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| x.saturating_sub(y))
                .collect();
            let mut simd_result = vec![0u8; 256];
            assert!(u8::simd_sub(&mut simd_result, &a, &b));
            assert_eq!(simd_result, expected);
        }

        #[test]
        fn test_simd_vs_scalar_sum_u8_equiv() {
            let src: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
            let expected: f64 = src.iter().map(|&v| v as f64).sum();
            let simd_result = u8::simd_sum(&src).unwrap();
            assert!(
                (expected - simd_result).abs() < 1e-10,
                "u8 sum mismatch: expected {expected}, got {simd_result}"
            );
        }
    }

    #[cfg(not(feature = "simd"))]
    mod no_simd_tests {
        use super::super::*;

        #[test]
        fn test_no_simd_f32() {
            assert!(!f32::has_simd());
        }

        #[test]
        fn test_no_simd_f64() {
            assert!(!f64::has_simd());
        }

        #[test]
        fn test_no_simd_u8() {
            assert!(!u8::has_simd());
        }
    }
}
