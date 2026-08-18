// The `_Θ` dimension marker re-used from whippyunits trips the confusables
// lint; it is intentional, so the lint is allowed crate-wide.
#![allow(mixed_script_confusables)]
// Mirrors whippyunits: the crate is `no_std` unless the (default) `std` feature
// is on. Heap-using paths (matrix `Display`, dynamic-matrix constructors) are
// gated behind `alloc`, which `std` implies.
#![cfg_attr(not(feature = "std"), no_std)]

//! WhippyAlgebra: zero-cost unit-safe linear algebra, powered by [whippyunits](https://docs.rs/whippyunits).
//!
//! WhippyAlgebra wraps existing linear-algebra libraries in transparent newtype
//! wrappers that add compile-time unit safety and compile away to the underlying
//! library types.
//!
//! ```
//! use whippyalgebra::dims;
//! use whippyalgebra::nalgebra::mixed_unit_matrix;
//! use whippyunits::{quantity, qty};
//!
//! type State = dims![m, m / s];
//! let phi = mixed_unit_matrix![State, State;
//!     [1.0, quantity!(0.5, s)],
//!     [quantity!(0.0, 1 / s), 1.0],
//! ];
//!
//! let x0 = mixed_unit_matrix![State, dims![1];
//!     [quantity!(2.0, m)],     // position
//!     [quantity!(3.0, m / s)], // velocity
//! ];
//!
//! let x1 = phi * x0;
//!
//! // x1 = [x0 + v0·Δt, v0] = [3.5 m, 3.0 m/s], read back with enforced units.
//! let pos: qty!(m) = x1.get::<0, 0>();
//! let vel: qty!(m / s) = x1.get::<1, 0>();
//! ```
//!
//! # Supported backends
//!
//! - [nalgebra](https://docs.rs/nalgebra) (enabled by default)
//!
//! Backends are feature-gated; disable the `nalgebra` feature to depend only on
//! the backend-agnostic unit machinery:
//!
//! ```toml
//! [dependencies]
//! whippyalgebra = { version = "0.1.1", default-features = false, features = ["std"] }
//! ```
//!
//! # `no_std`
//!
//! Like [whippyunits](https://docs.rs/whippyunits), this crate is `#![no_std]`
//! when its default `std` feature is turned off. The features are additive:
//!
//! - **`std`** (default) — full standard library. Adds nalgebra's
//!   `matrixmultiply` fast path and the `std`-only matrix exponential
//!   ([`exp`](crate::nalgebra::MixedUnitMatrix::exp)).
//! - **`alloc`** — heap allocation without `std` (implied by `std`). Required
//!   for matrix [`Display`](core::fmt::Display) (each cell is rendered to a
//!   `String`), dynamically sized matrices (`DMatrix`/`DVector`, `from_dyn`),
//!   and the `Vec`/columns/rows uniform constructors.
//! - neither — a `no_std`, no-alloc build. The statically sized, unit-checked
//!   core (construction, arithmetic, decompositions, element access, and
//!   [`rescale_matrix`](crate::nalgebra::rescale_matrix)) keeps working; the
//!   allocation-dependent items above are compiled out, so matrices print via
//!   [`Debug`](core::fmt::Debug) (delegating to the inner matrix) rather than
//!   the unit-aware [`Display`](core::fmt::Display).
//!
//! Floating-point math resolves in every configuration: the `nalgebra` backend
//! always enables nalgebra's `libm` feature, so `f32`/`f64` satisfy
//! `ComplexField` even without `std`.
//!
//! ```toml
//! [dependencies]
//! # no_std, no-alloc:
//! whippyalgebra = { version = "0.1.0", default-features = false, features = ["nalgebra"] }
//! # no_std + alloc (matrix Display, dynamic matrices):
//! whippyalgebra = { version = "0.1.0", default-features = false, features = ["alloc", "nalgebra"] }
//! ```
//!
//! # Matrix types
//!
//! WhippyAlgebra provides two matrix newtypes, both zero-cost wrappers around the
//! backend's own matrix that expose the same interface:
//!
//! - [`UniformUnitMatrix`](crate::nalgebra::UniformUnitMatrix): every entry
//!   shares a single unit `U`. Operations that can't be made statically safe on
//!   a mixed matrix (e.g. reciprocating the whole matrix, or an SVD with
//!   dimensioned singular values) live here.
//! - [`MixedUnitMatrix`](crate::nalgebra::MixedUnitMatrix): carries a *row*
//!   dimension list and a column dimension list; entry `(i, j)` carries the
//!   unit `RowDims[i] / ColDims[j]`.
//!
//! # Why row and column dimensions?
//!
//! `MixedUnitMatrix` tags entry `(i, j)` with the factored unit
//! `RowDims[i] / ColDims[j]` rather than an arbitrary per-cell grid. That
//! structure is what keeps the standard matrix operations dimensionally coherent:
//!
//! - the matrix product contracts a shared inner axis, so `A · B` type-checks
//!   precisely when `A`'s column dims equal `B`'s row dims;
//! - the transpose swaps the two lists;
//! - the determinant has one well-defined unit `∏ RowDims / ∏ ColDims`, shared by
//!   every term of its expansion. An unfactored per-cell grid gives no such
//!   guarantee, so its determinant would be dimensionally meaningless.
//!
//! ```
//! use whippyalgebra::dims;
//! use whippyalgebra::nalgebra::mixed_unit_matrix;
//! use whippyunits::{quantity, qty};
//!
//! type Rows = dims![V, V];
//! type Cols = dims![A, A];
//! let g = mixed_unit_matrix![Rows, Cols;
//!     [quantity!(2.0, V / A), quantity!(0.0, V / A)],
//!     [quantity!(0.0, V / A), quantity!(3.0, V / A)],
//! ];
//!
//! // det has unit (V·V) / (A·A) = Ω², the same for every Leibniz term.
//! let det: qty!(V^2 / A^2) = g.determinant();
//! # let _ = det;
//! ```
//!
//! # Declaring matrices
//!
//! ## Mixed-unit matrices
//!
//! [`mixed_unit_matrix!`](crate::nalgebra::mixed_unit_matrix) leads with the row and column
//! dimension lists (its shape is read from their lengths), then the rows; each
//! entry is checked against its cell unit `RowDims[i] / ColDims[j]`:
//!
//! ```
//! use whippyalgebra::dims;
//! use whippyalgebra::nalgebra::mixed_unit_matrix;
//! use whippyunits::quantity;
//!
//! type Rows = dims![m, m / s]; // row units
//! type Cols = dims![s];        // one column, in seconds
//! // entry (i, 0) = RowDims[i] / s  ⇒  m/s and m/s²
//! let m = mixed_unit_matrix![Rows, Cols;
//!     [quantity!(1.0, m / s)],
//!     [quantity!(2.0, m / s^2)],
//! ];
//! # let _ = m;
//! ```
//!
//! ## Uniform matrices
//!
//! [`uniform_unit_matrix!`](crate::nalgebra::uniform_unit_matrix) leads with a single unit
//! (any `whippyunits::unit!` expression); the shape is counted from the literal:
//!
//! ```
//! use whippyalgebra::nalgebra::uniform_unit_matrix;
//! use whippyunits::quantity;
//!
//! let m = uniform_unit_matrix![m / s;
//!     [quantity!(1.0, m / s), quantity!(2.0, m / s)],
//!     [quantity!(3.0, m / s), quantity!(4.0, m / s)],
//! ];
//! # let _ = m;
//! ```
//!
//! # Prior art
//!
//! For more information on the row-units / column-units decomposition used throughout this crate, see:
//!
//! > George W. Hart, *Multidimensional Analysis: Algebras and Systems for Science
//! > and Engineering*, Springer-Verlag New York, 1995. ISBN 0-387-94417-6.
//! > DOI: [10.1007/978-1-4612-4208-6](https://doi.org/10.1007/978-1-4612-4208-6).
//! > See also the author's [overview page](http://georgehart.com/research/multanal.html).

// In a `no_std + alloc` build we still need the `alloc` crate's `Vec`/`String`/
// `format!` for matrix pretty-printing; under `std` those come from `std`
// instead (see the `alloc` shim module). Not needed when `std` re-exports them.
#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc as alloc_crate;

// Unifies the `alloc`-vs-`std` import path for heap types, so `alloc`-gated code
// (currently the matrix `Display`) can `use crate::alloc::*` regardless of which
// feature supplies them.
#[cfg(feature = "alloc")]
#[doc(hidden)]
mod alloc;

pub mod dims;
pub mod entry;
pub mod index;
pub mod uniformity;

// Backend adapters. Each wraps a concrete linear-algebra library behind the
// shared, backend-agnostic unit machinery (`dims`, `index`, `entry`) and is
// gated behind its own feature so downstreams pull in only what they use.
#[cfg(feature = "nalgebra")]
pub mod nalgebra;

pub use dims::{
    ApplyUnit, Concat, Concatenated, CrossMul, CrossMulled, DCons, DNil, DiagUnit, DimList,
    Dimensionless, DivBy, HasUnit, MapUnits, Mapped, MetricShape, MulBy, PivotDims, Product,
    Producted, Reciprocal, RescaleFactors, ToDimensionless, UniformDiag, Unit, UnitRescale, ZipDiv,
    ZipDivided, ZipMul, ZipMulled, ZipToDimensionless,
};
pub use entry::{ColUnitOf, EntryUnit, EntryUnitOf, FromRaw, IntoEntry, RowUnitOf};
pub use index::{
    Drop, ElemAt, Nat, PowUnit, Repeat, Repeated, ShapeIndex, Sliced, SlicedAt, Take, UnitPow,
    Unsigned,
};
pub use uniformity::{CollapseUniform, GaugeReproduces, IsUniform, IsUniformBit, Uniform};

// Backend-specific items — the matrix newtypes, their decompositions, the
// declarative construction macros, and the `generic_matrix`/`generic_block`
// attributes — are intentionally **not** re-exported at the crate root. They
// live under their backend adapter (e.g. [`crate::nalgebra`]) so that multiple
// backends can be enabled at once without their identically-named types (two
// `MixedUnitMatrix`es, two `mixed_unit_matrix!`s, …) colliding. Import them from
// the adapter: `use whippyalgebra::nalgebra::{MixedUnitMatrix, mixed_unit_matrix, …};`.

#[doc(hidden)]
pub mod __reexport {
    // Backend-agnostic: the units library the `dims!`/`gauge!`/… macros expand
    // against, re-exported so those bounds resolve wherever the macro is used.
    // The backend crates themselves are re-exported by their own adapter
    // (e.g. `crate::nalgebra::__backend`), never here.
    pub use whippyunits;
}
