//! Unit-safe matrix decompositions, one module per factorization.
//!
//! Each module carries both the [`MixedUnitMatrix`](crate::nalgebra::MixedUnitMatrix)
//! wrapper and its [`UniformUnitMatrix`](crate::nalgebra::UniformUnitMatrix) twin,
//! mirroring nalgebra's `linalg` layout.
//!
//! # Metric (weighted) decompositions
//!
//! The `generalized_*` methods on [`MixedUnitMatrix`](crate::nalgebra::MixedUnitMatrix)
//! — [`generalized_pseudo_inverse`](crate::nalgebra::MixedUnitMatrix::generalized_pseudo_inverse),
//! [`generalized_svd`](crate::nalgebra::MixedUnitMatrix::generalized_svd),
//! [`generalized_qr`](crate::nalgebra::MixedUnitMatrix::generalized_qr),
//! [`generalized_col_piv_qr`](crate::nalgebra::MixedUnitMatrix::generalized_col_piv_qr),
//! [`generalized_bidiagonalize`](crate::nalgebra::MixedUnitMatrix::generalized_bidiagonalize)
//! — are the metric-relative (equivalently, weighted) forms of the classical
//! decompositions. A genuinely mixed-unit matrix has no canonical inner product (the
//! identity metric is not invariant under rescaling the units), so each takes one
//! positive-definite weight matrix per space it touches:
//!
//! - `Gr : ⟨1/RowDims, RowDims⟩` on the codomain (row) space, and
//! - `Gc : ⟨1/ColDims, ColDims⟩` on the domain (column) space.
//!
//! Each defines an inner product `⟨x, y⟩ = xᴴ G y` whose induced norm is a
//! dimensionless scalar; `G` is the one object variously called a weight, metric, or
//! Gram matrix. (A uniform matrix carries its own canonical metric, so its plain
//! decompositions need no weights — reach for the `UniformUnitMatrix` twin.)
//!
//! ## Relation to standard terms of art
//!
//! With `Gr`, `Gc` the weights, these are textbook constructions under a friendlier
//! name:
//! [`generalized_pseudo_inverse`](crate::nalgebra::MixedUnitMatrix::generalized_pseudo_inverse)
//! is the weighted Moore–Penrose inverse `A†_{Gr,Gc}`, and the least-squares solve it
//! (and [`GeneralizedColPivQR::solve`]) provide is weighted / Aitken generalized least
//! squares: minimize the `Gr`-norm of the residual, breaking ties by the `Gc`-norm.
//!
//! ## Not to be confused with
//!
//! - The GSVD of a matrix *pair*. [`GeneralizedSVD`] is the SVD of a single matrix
//!   in the `Gr`/`Gc` inner products (a weighted SVD); it is not the LAPACK / Van
//!   Loan generalized SVD of a pair `(A, B)`.
//! - The bare "generalized inverse": in the wider literature that names any
//!   `{1}`-inverse; here it is specifically the weighted Moore–Penrose inverse.

mod bidiagonal;
mod cholesky;
mod col_piv_qr;
mod determinant;
mod eigen;
mod full_piv_lu;
mod generalized_bidiagonal;
mod generalized_col_piv_qr;
mod generalized_qr;
mod generalized_svd;
mod hessenberg;
mod inverse;
mod lu;
mod qr;
mod schur;
mod solve;
mod svd;
mod symmetric_eigen;
mod symmetric_tridiagonal;
mod udu;

pub use bidiagonal::UniformBidiagonal;
pub use cholesky::{Cholesky, UniformCholesky};
pub use col_piv_qr::UniformColPivQR;
pub use eigen::MetricGeneralizedEigen;
pub use full_piv_lu::{OpaqueFullPivLU, UniformFullPivLU};
pub use generalized_bidiagonal::GeneralizedBidiagonal;
pub use generalized_col_piv_qr::GeneralizedColPivQR;
pub use generalized_qr::GeneralizedQR;
pub use generalized_svd::GeneralizedSVD;
pub use hessenberg::{Hessenberg, UniformHessenberg};
pub use lu::{LU, OpaqueLU, UniformLU};
pub use qr::UniformQR;
pub use schur::{Schur, UniformSchur};
pub use svd::UniformSVD;
pub use symmetric_eigen::{GeneralizedSymmetricEigen, SymmetricEigen};
pub use symmetric_tridiagonal::{SymmetricTridiagonal, UniformSymmetricTridiagonal};
pub use udu::{UDU, UniformUDU};
