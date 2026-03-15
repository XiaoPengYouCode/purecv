# PureCV

![PureCv Banner](./assets/purecv_banner.png)

[![Rust CI](https://github.com/webarkit/purecv/actions/workflows/rust.yml/badge.svg)](https://github.com/webarkit/purecv/actions/workflows/rust.yml)

A high-performance, **pure Rust** computer vision library focusing on the `core` and `imgproc` modules of OpenCV. **PureCV** is built from the ground up to be memory-safe, thread-safe, and highly portable without the overhead of C++ FFI.

## 🎯 Philosophy

Unlike existing wrappers, **PureCV** is a native rewrite. It aims to provide:

* **Zero-FFI:** No complex linking or C++ toolchain requirements.
* **Memory Safety:** Elimination of segmentation faults and buffer overflows via Rust's ownership model.
* **Modern Parallelism:** Native integration with **Rayon** for effortless multi-core processing.
* **Portable SIMD:** Optional SIMD acceleration via [`pulp`](https://crates.io/crates/pulp) — auto-detects x86 SSE/AVX, ARM NEON, and WASM `simd128` at runtime. Zero `unsafe`, zero `#[cfg(target_arch)]`.

## ✨ Features

### `purecv-core`
- **Matrix Operations:** Multi-dimensional `Matrix<T>` with support for common arithmetic (`add`, `subtract`, `multiply`, `divide`) and bitwise logic (`bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`). Matrix and scalar variants for all operations.
- **Factory Methods:** Intuitive initialization with `zeros`, `ones`, `eye`, and `diag`.
- **Comparison:** `compare`, `compare_scalar`, `min`, `max`, `abs_diff`, `in_range`.
- **Structural:** `flip`, `rotate`, `transpose`, `repeat`, `reshape`, `hconcat`, `vconcat`, `copy_make_border`, `extract_channel`, `insert_channel`.
- **Math:** `sqrt`, `exp`, `log`, `pow`, `magnitude`, `phase`, `cart_to_polar`, `polar_to_cart`, `convert_scale_abs`.
- **Stats:** `sum`, `mean`, `mean_std_dev`, `min_max_loc`, `norm`, `normalize`, `count_non_zero`, `reduce`.
- **Linear Algebra:** `gemm`, `dot`, `cross`, `trace`, `determinant`, `invert`, `solve`, `solve_poly`, `set_identity`.
- **Sorting:** `sort`, `sort_idx` with configurable row/column and ascending/descending flags.
- **Clustering:** `kmeans` with random, k-means++, and user-supplied initialization strategies.
- **Transforms:** `transform` (per-element matrix transformation), `perspective_transform` (projective / homography mapping).
- **Random Number Generation:** `randu` (uniform distribution), `randn` (normal/Gaussian distribution), `set_rng_seed`.
- **Channel Management:** `split`, `merge`, `mix_channels`.
- **Utilities:** `add_weighted`, `check_range`, `absdiff`, `get_tick_count`, `get_tick_frequency`.
- **ndarray Interop:** Optional, zero-cost conversions to/from `ndarray::Array3` via the `ndarray` feature flag.
- **SIMD Acceleration** (`simd` feature): Trait-based dispatch via `pulp` for `f32`, `f64`, and `u8` types. Accelerated operations include `add`, `sub`, `mul`, `div`, `min`, `max`, `sqrt`, `dot`, `sum`, `add_weighted`, `convert_scale_abs`, and `magnitude`. Falls back to scalar loops at zero cost when disabled.

### `purecv-imgproc`
- **Color Conversions:** High-performance `cvt_color` supporting RGB, BGR, Gray, RGBA, BGRA and more. Up to **6.6× speedup** with Parallel + SIMD.
- **Filtering:** `blur`, `box_filter`, `gaussian_blur`, `median_blur`, `bilateral_filter`.
- **Edge Detection:** `canny`, `sobel`, `scharr`, `laplacian`. Optimized `fast_deriv_3x3` kernel delivers up to **12× speedup** with Parallel.
- **Thresholding:** `threshold` with multiple threshold types.

## 🚀 Getting Started

### Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
purecv = "0.1"
```

### Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `std` | ✅ | Standard library support |
| `parallel` | ✅ | Multi-core parallelism via **Rayon** |
| `ndarray` | ❌ | Interop with the `ndarray` crate (zero-cost views & ownership transfers) |
| `simd` | ❌ | SIMD acceleration via [`pulp`](https://crates.io/crates/pulp) (x86 SSE/AVX, ARM NEON, WASM `simd128`) |
| `wasm` | ❌ | WebAssembly-specific optimizations |

To enable the `ndarray` feature:

```toml
[dependencies]
purecv = { version = "0.1", features = ["ndarray"] }
```

To enable SIMD + Parallel for maximum performance:

```toml
[dependencies]
purecv = { version = "0.1", features = ["parallel", "simd"] }
```

### Usage Example

```rust
use purecv::core::{Matrix, Size, Scalar};
use purecv::imgproc::{cvt_color, ColorConversionCodes};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 3-channel matrix initialized to ones
    let mat = Matrix::<f32>::ones(480, 640, 3);
    
    // Create an identity matrix
    let identity = Matrix::<f32>::eye(3, 3, 1);
    
    println!("Matrix size: {}x{}", mat.cols, mat.rows);
    Ok(())
}
```

### ndarray Interoperability

With the `ndarray` feature enabled, you can convert between `Matrix<T>` and `ndarray::Array3<T>`:

```rust
use purecv::core::Matrix;

// Matrix → ndarray (zero-cost view)
let mat = Matrix::<f32>::ones(480, 640, 3);
let view = mat.as_ndarray_view(); // ArrayView3<f32>, shape (480, 640, 3)

// Matrix → ndarray (ownership transfer)
let mat2 = Matrix::<f32>::ones(480, 640, 3);
let arr = mat2.into_ndarray();

// ndarray → Matrix (guarantees contiguous C-order layout for SIMD/WASM)
let mat3 = Matrix::from_ndarray(arr);

// Also works via the From trait
let arr2 = ndarray::Array3::<f32>::zeros((100, 100, 3));
let mat4: Matrix<f32> = Matrix::from(arr2);
```

### Running Examples

Explore the capabilities of PureCV by running the provided examples:

```bash
# Basic matrix arithmetic
cargo run --example arithmetic

# Structural operations (flip, rotate, split/merge)
cargo run --example structural_ops

# Color conversion (RGB to Grayscale)
cargo run --example color_conversion
```

## 🧪 Testing & Benchmarking

### Running Tests
PureCV uses a comprehensive suite of unit tests to ensure correctness and parity with OpenCV.

```bash
# Run all tests
cargo test
```

### Running Benchmarks
Performance is a core focus. Benchmarks are available for `arithm`, `imgproc`, and `structural` modules across four configurations:

```bash
# Standard (sequential, no SIMD)
cargo bench --no-default-features

# SIMD Only (sequential + auto-vectorization)
RUSTFLAGS="-C target-cpu=native" cargo bench --no-default-features

# Parallel (Rayon multi-threading)
cargo bench --features parallel

# Parallel + SIMD (maximum throughput)
RUSTFLAGS="-C target-cpu=native" cargo bench --features parallel
```

#### Key Performance Highlights (1024×1024 matrices)

| Operation | Standard | Parallel + SIMD | Speedup |
|-----------|----------|-----------------|---------|
| `cvt_color_rgb2gray` | 2.66 ms | **404 µs** | 6.6× |
| `sobel_3x3` | 22.79 ms | **1.87 ms** | 12× |
| `laplacian_3x3` | 45.91 ms | **4.44 ms** | 10.4× |
| `dot` | 997 µs | **157 µs** | 6.4× |
| `gemm_256×256` | 15.71 ms | **4.40 ms** | 3.7× |
| `canny` | 57.61 ms | **12.54 ms** | 4.6× |

> Full results in [`benches/benchmark_results.md`](./benches/benchmark_results.md)

## 🗺 Roadmap

- [x] **Phase 1: Core Foundation** - Matrix types, arithmetic, geometric utilities, and basic structural transforms.
- [x] **Phase 2: Performance** - SIMD acceleration via `pulp`, Rayon parallelism, and Criterion benchmarking across 32 operations.
- [ ] **Phase 3: WebAssembly** - Specialized wrappers and multi-threading for the web.
- [ ] **Phase 4: Image Processing** - Advanced filtering, convolutions, and feature detection.

## 📄 License

This project is licensed under the LGPL-3.0 License.
