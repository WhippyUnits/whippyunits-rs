# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.3] - 2026-07-01

### Added

- `lossy_into` and `lossless_into` methods on `Quantity` for converting the storage type while preserving scale, dimension, and brand
- `LossyFrom` trait with implementations between all primitive numeric types (`f32`, `f64`, `i8`–`i128`, `u8`–`u128`, `isize`, `usize`)
- `type_conversions` example demonstrating `lossy_into` and `lossless_into` usage

## [0.2.2] - 2026-06-17

### Fixed

- Small documentation fixes

## [0.2.1] - 2026-06-17

### Added

- `isize` and `usize` as supported storage types across the library (declarators, `value!` macro, `rescale!` macro, dimensionless/radian erasure)
- `generate_all_dimensionless_cross_type!` proc macro for exhaustive cross-type dimensionless erasure

### Fixed

- rust-analyzer inference via declarative macro refactor

## [0.2.0] - 2026-06-15

### Added

- Stable Rust support (no longer requires nightly `generic_const_exprs`); nightly is still supported behind the `cge` feature flag
- `no_std` and `alloc` feature support
- Arithmetic operations for all primitive numeric storage types (`f32`, `f64`, `i8`–`i128`, `u8`–`u128`)
- `comparison.md` documentation comparing WhippyUnits to other Rust unit libraries

### Fixed

- `no_std` and `serde` feature gating
