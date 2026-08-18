# Changelog

All notable changes to `whippyalgebra` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.1]

### Added

- `no_std` support: the crate is now `#![no_std]` when the default `std` feature
  is off, mirroring [whippyunits](https://crates.io/crates/whippyunits). Two new
  additive features, both on by default (via `std`):
  - `std` (default) — full standard library; implies `alloc`. Also enables
    nalgebra's `matrixmultiply` fast path and the `std`-only matrix exponential
    (`MixedUnitMatrix::exp`).
  - `alloc` — heap allocation without `std`. Required for matrix `Display`
    (unit-aware pretty-printing), dynamically sized matrices (`DMatrix`/
    `DVector`, `from_dyn`), and the `Vec`/columns/rows uniform constructors.

  With neither feature, the statically sized, unit-checked core — construction,
  arithmetic, decompositions, element access, and `rescale_matrix` — keeps
  working; the allocation-dependent items above are compiled out (matrices then
  print via `Debug` rather than the unit-aware `Display`). The `nalgebra`
  backend always enables nalgebra's `libm` feature, so `f32`/`f64` satisfy
  `ComplexField` in every configuration.

### Changed

- The `default` feature set is now `["std", "nalgebra"]` (was `["nalgebra"]`).
  Behaviour is unchanged for the default build; to depend only on the
  backend-agnostic unit machinery, disable default features and re-enable `std`
  (or `alloc`) as needed.
- `rescale_matrix` no longer allocates: its per-row/per-column factor buffers
  are now stack-backed nalgebra `OVector`s instead of `Vec`s, so it works in
  `no_std`/no-alloc for statically sized matrices. It gained two `Allocator<R>`/
  `Allocator<C>` bounds, both auto-satisfied for `Const`-sized shapes (so
  `generic_matrix`/`generic_block` callers need no new `where` clauses).
- The `whippyunits` and `nalgebra` dependencies are now pulled with
  `default-features = false`, with their `std`/`alloc` features forwarded from
  this crate's own `std`/`alloc` features.

### Feature-gating (only affects `no_std`/no-alloc builds)

- Matrix `Display`, the `DMatrix`/`DVector` re-exports, and the
  `UniformUnitMatrix::{from_vec, from_columns, from_rows}` constructors now
  require `alloc`.
- `MixedUnitMatrix::exp` now requires `std` (nalgebra's matrix exponential is
  itself `std`-only). `pow` and all other operations are unaffected.

## [0.1.0]

### Added

- Initial release: zero-cost, unit-safe linear algebra powered by
  [whippyunits](https://crates.io/crates/whippyunits).
  - `MixedUnitMatrix`, where entry `(i, j)` carries the factored unit
    `RowDims[i] / ColDims[j]` (keeping matrix product, transpose, determinant,
    and inverse dimensionally coherent), and `UniformUnitMatrix`, where every
    entry shares a single unit.
  - the `dims!`, `mixed_unit_matrix!`, `uniform_unit_matrix!`, `block_matrix!`,
    `unblock_matrix!`, `zeros!`, and `gauge!` declarative macros, and the
    `generic_matrix`/`generic_block` attribute macros for writing shape- and
    unit-generic code.
  - unit-checked wrappers over nalgebra's decompositions (LU, full-pivot LU, QR,
    column-pivot QR, SVD, Cholesky, eigen, Schur, bidiagonal, Hessenberg, UDU,
    and their generalized variants).
  - gated behind the default `nalgebra` feature; disable it to depend only on
    the backend-agnostic unit machinery (`dims`, `index`, `entry`).
