# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0]

### Added

- **`whippyalgebra`** — a new companion crate (`0.1.0`) providing zero-cost,
  unit-safe linear algebra
  - `nalgebra` module: newtypes over [nalgebra](https://docs.rs/nalgebra):
    - `MixedUnitMatrix`, where entry `(i, j)` carries the factored unit
      `RowDims[i] / ColDims[j]` (keeping matrix product, transpose, determinant,
      and inverse dimensionally coherent), and `UniformUnitMatrix`, where every
      entry shares a single unit
    - the `dims!`, `mixed_unit_matrix!`, and `uniform_unit_matrix!` declarative
      macros, and the `generic_matrix`/`generic_block` attribute macros for
      writing shape- and unit-generic code
    - unit-checked wrappers over nalgebra's decompositions (LU, full-pivot LU, QR,
      column-pivot QR, SVD, Cholesky, eigen, Schur, bidiagonal, Hessenberg, UDU,
      and their generalized variants)
    - gated behind the default `nalgebra` feature; disable it to depend only on
      the backend-agnostic unit machinery
- Public `Unit<Scale, Dimension>` type: the value-free dimensional signature
  (scale + dimension, with no storage type, brand, or value) used for type-level
  unit reasoning, e.g. the per-entry units of a unit-safe matrix.
- `qty!` macro: builds a concrete `Quantity` *type* from a unit expression
  (previously `unit!`).
- `bytemuck` support: `Quantity` is now `#[repr(transparent)]` over its storage
  type `T` and implements `Pod`, `Zeroable`, and `TransparentWrapper<T>`.

### Changed

- **BREAKING**: `Quantity` now takes a single inner `Unit` parameter —
  `Quantity<Unit<Scale, Dimension>, T = f64, Brand = ()>` — instead of carrying
  scale and dimension as separate parameters (`Quantity<Scale, Dimension, T,
  Brand>`). Code that spells out `Quantity`'s type parameters directly must wrap
  the scale/dimension pair in `Unit<…>`.
- **BREAKING**: the `unit!` macro now expands to a bare `Unit<Scale, Dimension>`
  type rather than a `Quantity` type. Use the new `qty!` macro to build a
  `Quantity` type from a unit expression, and `quantity!` to build a value.

### Removed

- **BREAKING**: the `cge` feature flag (and its nightly `generic_const_exprs`
  code path) has been removed. The typenum-based polyfill — supporting exponents
  in the range -200 to 200 — is now the only backend. If/when CGA is stabilized,
  a full-`i16`-range implementation will be reintroduced.

## [0.2.6] - 2026-07-13

### Added

- Dimension traits for all 20 named composite dimensions (`Area`, `Volume`, `Frequency`, `Force`, `Energy`, `Power`, `Pressure`, `ElectricCharge`, `ElectricPotential`, `Capacitance`, `ElectricResistance`, `ElectricConductance`, `Inductance`, `MagneticField`, `MagneticFlux`, `Illuminance`, `VolumeMassDensity`, `LinearMassDensity`, `DynamicViscosity`, `KinematicViscosity`) — these can now be used as trait bounds in the same way as the atomic dimension traits (`Mass`, `Length`, etc.)
- Hover documentation in `define_generic_dimension!` now works for all named dimensions (atomic and composite), derived automatically from the source-of-truth definitions in `whippyunits-core`
- `DynDimensionExponents::to_symbol_string()` for formatting dimension exponents with unicode superscripts (e.g. `ML²T⁻³I⁻¹`)

### Changed

- `Dimension::symbol` is now `Option<&'static str>` — only atomic/basis dimensions have symbols; composite dimensions no longer carry redundant exponent-formula strings
- `define_generic_dimension!` hover documentation and trait type resolution are now driven dynamically from `Dimension::find_dimension()` instead of hardcoded match arms
- `Dimension::find_dimension()` now ignores spaces in dimension names, so `ElectricPotential` matches `"Electric Potential"`

### Fixed

- `define_generic_dimension!` now correctly resolves multi-word dimension names like `ElectricPotential` (previously failed because Rust identifiers cannot contain spaces)
- **lsp-proxy**: `parse_parameter` no longer treats unresolved const generic names (e.g. `SCALE_P2`) as zero — they are now correctly classified as unresolved, preventing false "dimensionless" formatting in hover tooltips
- **lsp-proxy**: Trait signature simplification (`impl Add for Quantity<...>`) is no longer discarded by the subsequent type formatting pass
- **lsp-proxy**: Fixed off-by-one in test input for wholly-unresolved type formatting (missing closing `>`)

## [0.2.5] - 2026-07-08

### Added

- `amp` as a recognized abbreviation/symbol for ampere (like `s` for second or `hr` for hour)
- Compile-time validation of unit names/symbols in `unit!`, `quantity!`, and `value!` macros — previously, unrecognized units were silently treated as dimensionless; they now produce a `compile_error!` with fuzzy "Did you mean?" suggestions

## [0.2.4] - 2026-07-07

### Fixed

- `quantity!` macro now respects explicit storage type and brand type for nonstorage and affine units (e.g. `quantity!(4.5, inch, f32)` previously produced an `f64` quantity)
- `value!` macro affine path now resolves units by name as well as symbol, and panics at compile time instead of silently falling back to Kelvin when lookup fails

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
