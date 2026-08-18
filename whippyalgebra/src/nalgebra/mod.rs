//! The [nalgebra](https://docs.rs/nalgebra) backend adapter.
//!
//! Enabled by the `nalgebra` feature (on by default). Provides
//! [`MixedUnitMatrix`] and its vector aliases, wrapping nalgebra matrices with
//! whippyunits row/column dimension vectors. The backend-agnostic unit-entry
//! machinery it builds on lives in [`crate::entry`].
//!
//! # Re-exported storage types
//!
//! This module re-exports a curated set of nalgebra's own types that user code
//! needs in order to spell out matrix storage in its own signatures — the
//! dimension markers ([`Const`], [`Dyn`]), the matrix/vector aliases
//! ([`OMatrix`], [`SMatrix`], …), and the allocator entry point
//! ([`DefaultAllocator`]).
//!
//! # Declarative constructors
//!
//! The [`block_matrix!`], [`mixed_unit_matrix!`], [`uniform_unit_matrix!`],
//! [`zeros!`], [`gauge!`], and [`unblock_matrix!`] macros offer declarative
//! syntax for building and destructuring matrices.
//!
//! # Generic-code attributes
//!
//! The `where`-clause–writing attributes emit nalgebra `Dim`/allocator
//! obligations.
//!
//! - [`macro@generic_matrix`] writes the shape well-formedness
//!   (`SquareDim`/`ShapeIndex`, plus `DimList` for named axes) of a set of matrix
//!   sizes into an item's `where` clause.
//! - [`macro@generic_block`] writes a block assembly's storage (and `exp`)
//!   obligations from the grid shape.

pub mod linalg;
pub mod matrix;
pub mod reduce;
pub mod uniform;

mod macros;

pub use linalg::{
    Cholesky, GeneralizedBidiagonal, GeneralizedColPivQR, GeneralizedQR, GeneralizedSVD,
    GeneralizedSymmetricEigen, Hessenberg, LU, MetricGeneralizedEigen, OpaqueFullPivLU, OpaqueLU,
    Schur, SymmetricEigen, SymmetricTridiagonal, UDU, UniformBidiagonal, UniformCholesky,
    UniformColPivQR, UniformFullPivLU, UniformHessenberg, UniformLU, UniformQR, UniformSVD,
    UniformSchur, UniformSymmetricTridiagonal, UniformUDU,
};
pub use matrix::{
    CountDim, CountedDim, MixedUnitMatrix, SquareDim, UnitRowVector, UnitVector, rescale_matrix,
};
pub use reduce::{AutoReduce, Reduced};
pub use uniform::{UniformUnitMatrix, rescale_uniform_matrix};

// The declarative matrix constructors and reshapers. See the module-level
// "Declarative constructors" section for why these come from the adapter. Each
// macro carries its own documentation, inlined here from `macros.rs`.
#[doc(inline)]
pub use crate::{
    __wa_na_block_matrix as block_matrix, __wa_na_gauge as gauge,
    __wa_na_mixed_unit_matrix as mixed_unit_matrix, __wa_na_unblock_matrix as unblock_matrix,
    __wa_na_uniform_unit_matrix as uniform_unit_matrix, __wa_na_zeros as zeros,
};

// The `where`-clause–writing attributes. See the module-level "Generic-code
// attributes" section for what they do and why they come from the adapter. Each
// attribute carries its own documentation, inlined here from the macro crate.
#[doc(inline)]
pub use whippyalgebra_macros::{generic_block, generic_matrix};

// The curated storage types users need to spell matrix storage in their own
// signatures. See the module-level "Re-exported storage types" section for why
// these come from the adapter rather than the `nalgebra` crate directly.
#[doc(no_inline)]
pub use ::nalgebra::{
    Const, DefaultAllocator, Dyn, Matrix, OMatrix, OVector, SMatrix, SVector, U1,
};
// The dynamically sized aliases are backed by nalgebra's `VecStorage`, which
// only exists with heap allocation, so they follow our `alloc` feature. `Dyn`
// (the dimension marker) is allocation-free and stays available above, so
// generic `Dyn`-shaped signatures still compile without `alloc`.
#[cfg(feature = "alloc")]
#[doc(no_inline)]
pub use ::nalgebra::{DMatrix, DVector};

/// The nalgebra crate itself, re-exported for macro expansions (both the
/// declarative constructors here and the `generic_*` attributes) to name
/// backend paths through `whippyalgebra` — so the emitted code resolves even
/// when the caller does not depend on `nalgebra` directly, or under a different
/// name. Each backend adapter owns its own `__backend`, so nothing collides at
/// the crate root. Not part of the public API.
#[doc(hidden)]
pub use ::nalgebra as __backend;
