#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use crate::dims::{Dimensionless, MapUnits, Mapped, MetricShape, Reciprocal, ToDimensionless};
use crate::nalgebra::matrix::MixedUnitMatrix;
use crate::nalgebra::uniform::UniformUnitMatrix;

/// The result of the (real) Schur decomposition `M = Q T Qᵀ` of a metric.
///
/// Returned by [`MixedUnitMatrix::schur`]. It shares the unitary frame `Q … Qᵀ`
/// of the [symmetric eigendecomposition](SymmetricEigen), so it carries the same
/// metric shape `⟨RowDims, 1/RowDims⟩`, but relaxes the diagonal `Λ` to a
/// (quasi-)upper-triangular `T` and so drops the self-adjointness requirement.
/// `T` is dimensionless. This handles a metric-typed but numerically
/// non-symmetric matrix, whose complex spectrum lands as real 2×2 blocks on
/// `T`'s diagonal.
pub struct Schur<RowDims, T, D, Brand = ()>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    RowDims: MapUnits<ToDimensionless>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// The orthonormal Schur vectors `Q`, in `⟨RowDims, [1 … 1]⟩`: a basis of the
    /// row space with dimensionless columns (the same shape as the
    /// [`SymmetricEigen`] eigenvectors and a Cholesky factor).
    pub q: MixedUnitMatrix<
        RowDims,
        Mapped<ToDimensionless, RowDims>,
        nalgebra::OMatrix<T, D, D>,
        Brand,
    >,
    /// The (quasi-)upper-triangular factor `T`, dimensionless; its diagonal holds
    /// the eigenvalues (real 2×2 blocks for complex-conjugate pairs), which are
    /// pure numbers since the spectrum is metric-relative (see
    /// [`symmetric_eigen`]).
    ///
    /// [`symmetric_eigen`]: MixedUnitMatrix::symmetric_eigen
    pub t: UniformUnitMatrix<Dimensionless, nalgebra::OMatrix<T, D, D>, Brand>,
}

impl<RowDims, T, D, Brand> Schur<RowDims, T, D, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    RowDims: MapUnits<ToDimensionless>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// Reconstructs the metric `M = Q T Qᵀ` from its factors, landing back in
    /// `⟨RowDims, 1/RowDims⟩`.
    ///
    /// `Q : ⟨RowDims, [1…1]⟩` contracts the dimensionless `T : ⟨[1…1], [1…1]⟩` on
    /// the pivot, then `Qᵀ : ⟨[1…1], 1/RowDims⟩` — the same round trip as
    /// [`SymmetricEigen::recompose`], just with a triangular filling.
    pub fn recompose(
        &self,
    ) -> MixedUnitMatrix<RowDims, Mapped<Reciprocal, RowDims>, nalgebra::OMatrix<T, D, D>, Brand>
    where
        RowDims: MapUnits<Reciprocal>,
    {
        let q = self.q.nalgebra();
        MixedUnitMatrix::from_nalgebra(q * self.t.nalgebra() * q.adjoint())
    }
}

// Schur `M = Q T Qᵀ` shares the unitary frame of [`symmetric_eigen`], so it
// carries the identical dimensional bound — the input must be metric-shaped
// `ColDims = 1/RowDims`. The generalization over `symmetric_eigen` is purely
// numerical (drop self-adjointness ⇒ `T` triangular, not diagonal), and since
// `T` is dimensionless in either case, that relaxation does not touch the units:
//
//   Q : ⟨RowDims, [1…1]⟩   T : ⟨[1…1], [1…1]⟩   Qᵀ : ⟨[1…1], 1/RowDims⟩
//   Q·T·Qᵀ : ⟨RowDims, 1/RowDims⟩  =  M
//
// (The non-normal *endomorphisms* Schur is classically used for are shape
// `⟨u, u⟩`, which this frame cannot reconstruct unless `u` is dimensionless;
// erase to a uniform matrix and use [`UniformUnitMatrix::schur`] for those.)
impl<RowDims, ColDims, Brand, T, D, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, D, D, S>, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::DimSub<nalgebra::U1>,
    S: nalgebra::storage::Storage<T, D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>
        + nalgebra::allocator::Allocator<D>
        + nalgebra::allocator::Allocator<nalgebra::DimDiff<D, nalgebra::U1>>
        + nalgebra::allocator::Allocator<D, nalgebra::DimDiff<D, nalgebra::U1>>,
{
    /// The (real) Schur decomposition `M = Q T Qᵀ` of a metric, returning a
    /// [`Schur`] with the orthonormal frame `Q` and the dimensionless
    /// (quasi-)triangular `T`.
    ///
    /// Requires the metric shape `ColDims = 1/RowDims` — the same bound as
    /// [`symmetric_eigen`](Self::symmetric_eigen), since both build on the
    /// frame `Q … Qᵀ`. Where `symmetric_eigen` symmetrizes and yields a diagonal
    /// real spectrum, `schur` accepts a metric-typed but non-symmetric matrix and
    /// returns the triangular form (complex eigenvalues as real 2×2 blocks). For
    /// a dimensioned spectrum on a general (endomorphism-shaped) matrix, erase to
    /// [`UniformUnitMatrix::schur`](crate::nalgebra::UniformUnitMatrix::schur).
    pub fn schur(self) -> Schur<RowDims, T, D, Brand>
    where
        RowDims: MetricShape<ColDims> + MapUnits<ToDimensionless>,
    {
        let (q, t) = self.inner.clone_owned().schur().unpack();
        Schur {
            q: MixedUnitMatrix::from_nalgebra(q),
            t: UniformUnitMatrix::from_nalgebra(t),
        }
    }
}

/// The (real) Schur decomposition `M = Q T Qᵀ` of a uniform matrix.
///
/// Returned by [`UniformUnitMatrix::schur`]. Schur relaxes the diagonal `Λ` of
/// the [symmetric eigendecomposition](crate::nalgebra::SymmetricEigen) to a (quasi-)upper-
/// triangular `T` — the same unitary frame, but no self-adjointness required, so
/// it handles any square matrix (complex spectra land as real 2×2 blocks on
/// `T`'s diagonal). A uniform matrix carries one unit `U` on both sides, which is
/// a canonical metric, so the frame closes: `Q` is dimensionless and
/// orthonormal, and `T` carries `U` with the eigenvalues on its diagonal. Like
/// [uniform SVD](crate::nalgebra::UniformSVD) it is ungated — nothing roots the unit.
pub struct UniformSchur<U, T, D, Brand = ()>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
{
    /// The orthonormal Schur vectors `Q`, dimensionless.
    pub q: UniformUnitMatrix<Dimensionless, nalgebra::OMatrix<T, D, D>, Brand>,
    /// The (quasi-)upper-triangular factor `T`, in the entry unit `U`; its
    /// diagonal holds the eigenvalues (real 2×2 blocks for complex pairs).
    pub t: UniformUnitMatrix<U, nalgebra::OMatrix<T, D, D>, Brand>,
}

impl<U, T, D, Brand> UniformSchur<U, T, D, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
{
    /// Reconstructs `M = Q T Qᵀ`, landing back in the entry unit `U`.
    ///
    /// The two dimensionless orthonormal `Q`'s flank the `U`-carrying `T`, so the
    /// product is uniform in `1 · U · 1 = U` — the unit round trip is enforced by
    /// construction.
    pub fn recompose(&self) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, D, D>, Brand> {
        let q = self.q.nalgebra();
        UniformUnitMatrix::from_nalgebra(q * self.t.nalgebra() * q.adjoint())
    }
}

// The Schur frame `Q … Qᵀ` requires a metric on both sides; a uniform matrix's
// single shared unit `U` *is* that canonical metric (exactly as for uniform
// [`svd`](Self::svd)), so `Q` comes out dimensionless and `T` keeps `U`. Unlike
// the mixed [`schur`](crate::nalgebra::MixedUnitMatrix::schur) — which needs the input to
// already be metric-*shaped* `⟨R, 1/R⟩` — the uniform case is unconditional, and
// it is the practical home for the general non-normal endomorphism (erase to
// uniform, then triangularize).
impl<U, Brand, T, D, S> UniformUnitMatrix<U, nalgebra::Matrix<T, D, D, S>, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::DimSub<nalgebra::U1>,
    S: nalgebra::storage::Storage<T, D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>
        + nalgebra::allocator::Allocator<D>
        + nalgebra::allocator::Allocator<nalgebra::DimDiff<D, nalgebra::U1>>
        + nalgebra::allocator::Allocator<D, nalgebra::DimDiff<D, nalgebra::U1>>,
{
    /// The (real) Schur decomposition `M = Q T Qᵀ`, returning a [`UniformSchur`]
    /// with the dimensionless orthonormal frame `Q` and the (quasi-)triangular
    /// `T` in the entry unit `U`.
    ///
    /// This is the dimensioned generalization of the symmetric eigendecomposition
    /// to any uniform matrix: it drops the self-adjointness the
    /// [mixed `symmetric_eigen`](crate::nalgebra::MixedUnitMatrix::symmetric_eigen) needs,
    /// keeping the same unitary frame (which a uniform matrix's single unit makes
    /// a canonical metric). Ungated — nothing roots the unit.
    pub fn schur(&self) -> UniformSchur<U, T, D, Brand> {
        let (q, t) = self.inner.clone_owned().schur().unpack();
        UniformSchur {
            q: UniformUnitMatrix::from_nalgebra(q),
            t: UniformUnitMatrix::from_nalgebra(t),
        }
    }
}
