# Changelog - webarkit/purecv

All notable changes to this project will be documented in this file.

## [Unreleased]

### 🚜 Refactor

- *(simd)* Split imgproc-specific SIMD helpers (`simd_rgb_to_gray_u8`, `simd_bgr_to_gray_u8`, `simd_rgba_to_gray_u8`, `simd_bgra_to_gray_u8`, `simd_deriv_3x3_row_f32`) from `src/core/simd.rs` into new `src/imgproc/simd.rs` for better modularity.

### 📚 Documentation

- Add per-function `# Performance` doc comments to SIMD-accelerated functions documenting measured speedups.
- Add unit tests for `bgr_to_gray`, `rgba_to_gray`, `bgra_to_gray`, and `deriv_3x3_row_f32` SIMD kernels.

## [0.2.0] - 2026-03-21

### 🐛 Bug Fixes

- Fmt issue in wasm rust module

### 🚀 Features

- Introduce a WebAssembly module for purecv, including build scripts and a workspace structure.
- Improves to WebAssembly module with dual (standard and SIMD) builds and packaging infrastructure.
- Add GitHub Actions workflows for continuous integration, release management, and package publishing.
- Add a new WebAssembly image processing demo showcasing various filters on a butterfly image.
- Introduce and document the new WebAssembly package, including installation instructions and badge updates in the main README.

### 🚜 Refactor

- *(wasm)* Fix wasm crate metadata and core re-exports
- Improved GitHub Actions workflow for automated releases, including artifact packaging and publishing to Crates.io and NPM.

## [0.1.4] - 2026-03-19

### 🐛 Bug Fixes

- Correct indentation in rust.yml file and fix for wrong command

### 📚 Documentation

- Update README to indicate project is a work in progress

### 🚀 Features

- *(version)* Add the new src/version.rs file
- Add badges to README and update CI workflow for formatting check

### 🚜 Refactor

- Fix in version.rs file for fmt issue
- Fix in version.rs for test failing issue
- *(chore)* Restructure CI workflow to separate code formatting and build steps

## [0.1.3] - 2026-03-17

### ⚡ Performance

- *(imgproc)* Add bilateral_filter and sobel_f32 SIMD benchmarks

### 🐛 Bug Fixes

- *(simd)* Fix simd_dot/simd_sum returning 0.0 and use bool return in macros
- *(simd)* Allow dead code for SimdElement trait and clean up threshold.rs

### 📚 Documentation

- Update README for PR 2 Color + Threshold SIMD changes
- Update README with performance highlights for sobel and bilateral_filter SIMD optimizations

### 🚀 Features

- *(core,imgproc)* Add SIMD acceleration via pulp and full benchmark suite
- *(imgproc)* Add SIMD acceleration for color conversion and threshold
- *(imgproc)* Add threshold example

## [0.1.2] - 2026-03-15

### 🐛 Bug Fixes

- *(core)* Resolve arithm.rs compilation errors and add magnitude/polar functions

### 📚 Documentation

- Add Copilot instructions and project guidelines
- Fix 3-D to 3D in ndarray method doc comments
- Update README with ndarray feature flag and usage examples
- Fix ndarray example to avoid use-after-move
- *(core)* Add documentation and examples to linear algebra functions
- Update contribution guidelines for pre-commit checks and code quality in agents files
- Update README to enhance feature descriptions and clarify dependencies
- Add AGENTS.md for AI coding agents
- Update README to include new functions in linear algebra, sorting, clustering, transforms, and utilities

### 🚀 Features

- Add ndarray interoperability for Matrix via optional feature flag
- *(core)* Implement gemm, trace, dot, cross, check_range, and set_identity
- *(core)* Add randu, randn, and set_rng_seed
- *(arithm)* Add matrix transformation and perspective transformation functions
- *(core)* Add solve_poly, sort, sort_idx, and kmeans functions

### 🚜 Refactor

- *(core)* Fix formatting violations from cargo fmt
- Fix formatting in core arithm
- Fix clippy needless_range_loop warnings in core arithm
- Apply cargo fmt to fix CI formatting check

## [0.1.1] - 2026-03-14

### ⚙️ Miscellaneous Tasks

- Fix clippy warnings in src/core/matrix.rs
- Update repository URL in Cargo.toml and add changelog and CI release configuration files
- Pushing correct github release script
- Add simd feature to Cargo.toml and update CI workflow for dev branch

### 📚 Documentation

- Add CONTRIBUTING and MAINTAINERS guidelines

### 🚀 Features

- Implement MatType and DataType for OpenCV parity

## [0.1.0] - 2026-03-10

## Overview
This changelog documents all changes that occurred from the initial commit through the release of v0.1.0 for the webarkit/purecv repository - a pure Rust computer vision library focusing on the core and imgproc modules of OpenCV.

---

## 1. Project Foundation & Structure

- ✅ Initialized core project structure with Cargo workspace setup
- ✅ Established LGPL-2.1-or-later license
- ✅ Created comprehensive header documentation across all source files
- ✅ Set up GitHub Actions CI/CD workflows (Rust.yml) with formatting checks, build tests, and clippy linting

---

## 2. Core Module Enhancements

### Matrix Operations

- ✅ Implemented `Matrix<T>` generic 2D matrix with row-major memory layout
- ✅ Added factory methods: `zeros()`, `ones()`, `eye()`, `diag()`
- ✅ Added `from_size()` constructor for convenient size-based matrix creation
- ✅ Implemented `convert_to<U>()` for type casting with precision handling
- ✅ Added accessor methods: `get()`, `at()` with i32 indexing, `get_mut()`, `at_mut()`
- ✅ Implemented `flat_index()` for efficient 1D-2D coordinate conversion

### Arithmetic Operations

- ✅ Implemented core arithmetic: `add()`, `subtract()`, `multiply()`, `divide()`
- ✅ Added mathematical functions: `sqrt()`, `exp()`, `log()`, `pow()`
- ✅ Implemented bitwise operations: `bitwise_and()`, `bitwise_or()`, `bitwise_xor()`, `bitwise_not()`
- ✅ Added `absdiff()` for absolute difference
- ✅ Implemented `add_weighted()` for weighted sums
- ✅ Added `convert_scale_abs()` for scaling and absolute value conversion
- ✅ Feature-gated SIMD support with Rayon for parallel operations

### Data Types

- ✅ Added `BorderTypes` enum with REFLECT_101, REPLICATE, WRAP, CONSTANT, etc.

### Statistical Functions

- ✅ Implemented `sum()` and `mean()` per-channel calculations
- ✅ Added `min_max_loc()` for finding min/max values and their locations
- ✅ Implemented `mean_std_dev()` for standard deviation calculations
- ✅ Implemented `norm()` with support for INF, L1, L2 norms
- ✅ Added `normalize()` for MINMAX and norm-based normalization

### Structural Operations

- ✅ Implemented `flip()` for vertical, horizontal, and both-axes flipping
- ✅ Added `transpose()` for matrix transposition
- ✅ Implemented `split()` for channel separation
- ✅ Added `merge()` for channel combination
- ✅ Implemented `repeat()` for pattern repetition
- ✅ Added `reshape()` for matrix dimension changes
- ✅ Implemented `copy_make_border()` for border padding
- ✅ Added `hconcat()` and `vconcat()` for horizontal/vertical concatenation
- ✅ Implemented `mixChannels()` for advanced channel mixing

---

## 3. Image Processing (imgproc) Module

### Color Conversions

- ✅ Implemented `cvt_color()` as main wrapper function
- ✅ Added color space conversions:
    - RGB ↔ Grayscale
    - BGR ↔ Grayscale
    - RGBA/BGRA ↔ Grayscale
    - Grayscale ↔ RGB/BGR/RGBA/BGRA
- ✅ Added `ColorConversionCode` enum for OpenCV-style API

### Filtering Operations

- ✅ Implemented `blur()` and `box_filter()` for box filtering
- ✅ Added `gaussian_blur()` with kernel generation
- ✅ Implemented `median_blur()` for non-linear filtering
- ✅ Added `bilateral_filter()` for edge-preserving smoothing

### Derivative Operations

- ✅ Implemented `sobel()` for Sobel derivatives
- ✅ Added `scharr()` for Scharr operator
- ✅ Implemented `laplacian()` for Laplacian computation
- ✅ Added `get_sobel_kernels()` and `get_deriv_kernel()` utilities

### Edge Detection

- ✅ Implemented `canny()` edge detector with:
    - Gradient computation using Sobel
    - Non-maximum suppression
    - Hysteresis thresholding

### Image Thresholding

- ✅ Implemented `threshold()` function with types:
    - THRESH_BINARY
    - THRESH_BINARY_INV
    - THRESH_TRUNC
    - THRESH_TOZERO
    - THRESH_TOZERO_INV

---

## 4. Performance & Optimization

### Parallelization

- ✅ Integrated Rayon for multi-core processing
- ✅ Feature-gated parallel execution with fallback to sequential
- ✅ Implemented parallel iterator patterns for memory-efficient batch processing

### Algorithmic Optimizations

- ✅ Added `fast_deriv_3x3()` for optimized 3x3 derivative computation
- ✅ Implemented separated filter passes for efficiency
- ✅ Added interior "fast path" with boundary "slow path" separation
- ✅ Optimized chunk-based processing to reduce allocations

### SIMD Support

- ✅ Optional Pulp integration for portable SIMD
- ✅ Feature-gated SIMD dispatch with fallback
- ✅ Auto-vectorization-friendly code patterns

---

## 5. Utility Functions

- ✅ Implemented `border_interpolate()` with multiple border types
- ✅ Added `get_log_level()` and `set_log_level()` for debugging
- ✅ Implemented `ParIterFallback` trait for seamless feature-gated compilation

---

## 6. Testing & Documentation

- ✅ Comprehensive unit test coverage across all modules
- ✅ Integration tests verifying OpenCV parity
- ✅ Benchmark suite with Criterion for performance evaluation
- ✅ Created example programs:
    - arithmetic
    - color_conversion
    - filters
    - structural_ops
- ✅ Added benchmark results documentation

---

## 7. Build & CI/CD

- ✅ Configured Cargo.toml with proper features:
    - parallel
    - wasm
    - simd
- ✅ Set optimization levels:
    - opt-level=3
    - lto=true
    - codegen-units=1
- ✅ Configured panic=abort for WASM/system safety
- ✅ GitHub Actions workflow for automated testing and linting

---

## 8. Documentation & Community

- ✅ Updated README with philosophy, features, and usage examples
- ✅ Added roadmap indicating Phase 1 (Core Foundation) completion
- ✅ Added CI badge for project health visibility
- ✅ Organized examples and data directories

---

## Summary

This development represents a **complete foundation for a production-grade pure Rust computer vision library**, with emphasis on:

- **Safety**: Leveraging Rust's memory safety guarantees
- **Performance**: Through parallelization, SIMD support, and algorithmic optimizations
- **OpenCV API Compatibility**: Familiar interfaces for developers transitioning from OpenCV
- **Extensibility**: Well-structured codebase for future module additions

**v0.1.0** marks the successful completion of **Phase 1 (Core Foundation)**, establishing the core matrix operations, essential image processing functions, and the architectural foundation for future development.