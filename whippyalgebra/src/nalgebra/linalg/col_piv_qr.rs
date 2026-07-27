#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use whippyunits::{DivUnit, UnitDiv};

use crate::dims::Dimensionless;
use crate::nalgebra::uniform::UniformUnitMatrix;

// The *typed* mixed column-pivoted QR (exposing `Q`/`R`/`P`) is deliberately
// absent in bare form: like a bare mixed QR/SVD, its orthogonal `Q` is orthonormal
// only in the silent identity metric a mixed matrix cannot name, and — because the
// columns carry *different* units — even the *pivot order* is unit-dependent (the
// column norms it compares are incommensurable). Both leaks are sealed by naming a
// metric on each space: the metric-supplied
// [`generalized_col_piv_qr`](MixedUnitMatrix::generalized_col_piv_qr) makes the
// column norms commensurable (so the pivot is well-defined and rank-revealing) and
// `Q` metric-orthonormal, exposing typed factors and a metric least-squares
// `solve`. A uniform matrix carries its own canonical metric and keeps the twin
// below.

/// The result of a column-pivoted QR decomposition `M P = Q R` of a uniform
/// matrix, mirroring nalgebra's [`ColPivQR`](nalgebra::ColPivQR).
///
/// Returned by [`UniformUnitMatrix::col_piv_qr`]. As with the uniform
/// [`lu`](UniformUnitMatrix::lu), both margins share one unit, so the runtime
/// column permutation is type-invisible and the decomposition is ungated:
/// the orthogonal `Q` and the permutation `P` are dimensionless, while the
/// upper-triangular `R` carries the entry unit `U`.
pub struct UniformColPivQR<U, T, D, Brand = ()>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
{
    /// The orthogonal factor `Q`, dimensionless and orthonormal.
    pub q: UniformUnitMatrix<Dimensionless, nalgebra::OMatrix<T, D, D>, Brand>,
    /// The upper-triangular factor `R`, in the entry unit `U`.
    pub r: UniformUnitMatrix<U, nalgebra::OMatrix<T, D, D>, Brand>,
    /// The column-permutation matrix `P` (dimensionless), with `M·P = Q·R`.
    pub p: UniformUnitMatrix<Dimensionless, nalgebra::OMatrix<T, D, D>, Brand>,
}

impl<U, T, D, Brand> UniformColPivQR<U, T, D, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
{
    /// Reconstructs `M = Q·R·Pᵀ` from the factors, in the entry unit `U` (`Pᵀ`
    /// inverts the column permutation).
    pub fn recompose(&self) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, D, D>, Brand> {
        let m = self.q.nalgebra() * self.r.nalgebra() * self.p.nalgebra().transpose();
        UniformUnitMatrix::from_nalgebra(m)
    }

    /// Solves `M x = b` reusing this factorization (`x = P R⁻¹ Qᴴ b`), in the
    /// quotient unit `Ub / U` — the same signature as
    /// [`UniformUnitMatrix::solve`](crate::nalgebra::UniformUnitMatrix::solve), but without
    /// re-factorizing. Returns `None` if `R` is singular.
    pub fn solve<Ub, C2, SB>(
        &self,
        b: &UniformUnitMatrix<Ub, nalgebra::Matrix<T, D, C2, SB>, Brand>,
    ) -> Option<UniformUnitMatrix<DivUnit<Ub, U>, nalgebra::OMatrix<T, D, C2>, Brand>>
    where
        Ub: UnitDiv<U>,
        C2: nalgebra::Dim,
        SB: nalgebra::storage::Storage<T, D, C2>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, C2>,
    {
        let qt_b = self.q.nalgebra().adjoint() * b.nalgebra();
        let rz = self.r.nalgebra().solve_upper_triangular(&qt_b)?;
        let x = self.p.nalgebra() * rz;
        Some(UniformUnitMatrix::from_nalgebra(x))
    }
}

impl<U, Brand, T, D, S> UniformUnitMatrix<U, nalgebra::Matrix<T, D, D, S>, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::DimMin<D, Output = D>,
    S: nalgebra::storage::Storage<T, D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>
        + nalgebra::allocator::Allocator<D>
        + nalgebra::allocator::Reallocator<T, D, D, D, D>,
{
    /// Column-pivoted QR, `M P = Q R`, returning a [`UniformColPivQR`].
    ///
    /// Always available on a uniform matrix — both margins share one unit, so the
    /// runtime column pivot is type-invisible and the orthogonal factor is
    /// orthonormal in that canonical metric. A mixed matrix has no such metric, so
    /// its column-pivoted QR must be supplied one:
    /// [`generalized_col_piv_qr`](crate::nalgebra::MixedUnitMatrix::generalized_col_piv_qr).
    pub fn col_piv_qr(&self) -> UniformColPivQR<U, T, D, Brand> {
        let (q, r, p) = self.inner.clone_owned().col_piv_qr().unpack();
        let (nr, nc) = self.inner.shape_generic();
        let mut pm = nalgebra::OMatrix::<T, D, D>::identity_generic(nr, nc);
        p.permute_columns(&mut pm);
        UniformColPivQR {
            q: UniformUnitMatrix::from_nalgebra(q),
            r: UniformUnitMatrix::from_nalgebra(r),
            p: UniformUnitMatrix::from_nalgebra(pm),
        }
    }
}
