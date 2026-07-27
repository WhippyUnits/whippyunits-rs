#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use crate::dims::{Dimensionless, MapUnits, Mapped, MetricShape, Reciprocal, ToDimensionless};
use crate::nalgebra::matrix::MixedUnitMatrix;
use crate::nalgebra::uniform::UniformUnitMatrix;

/// The result of a Hessenberg reduction `M = Q H Qᵀ` of a metric.
///
/// Returned by [`MixedUnitMatrix::hessenberg`]. It shares the unitary similarity
/// frame `Q … Qᵀ` of [`Schur`] and [`SymmetricEigen`] (the same `Q` on both
/// sides), so it needs the metric shape `⟨RowDims, 1/RowDims⟩`. It reduces `M`
/// to `H` (upper-triangular but for one subdiagonal), the starting point of the
/// Schur iteration; `H` is dimensionless and `Q : ⟨RowDims, [1…1]⟩`.
pub struct Hessenberg<RowDims, T, D, Brand = ()>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    RowDims: MapUnits<ToDimensionless>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// The orthonormal reducing basis `Q`, in `⟨RowDims, [1 … 1]⟩`.
    pub q: MixedUnitMatrix<
        RowDims,
        Mapped<ToDimensionless, RowDims>,
        nalgebra::OMatrix<T, D, D>,
        Brand,
    >,
    /// The upper-Hessenberg factor `H`, dimensionless.
    pub h: UniformUnitMatrix<Dimensionless, nalgebra::OMatrix<T, D, D>, Brand>,
}

impl<RowDims, T, D, Brand> Hessenberg<RowDims, T, D, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    RowDims: MapUnits<ToDimensionless>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// Reconstructs the metric `M = Q H Qᵀ`, landing back in `⟨RowDims, 1/RowDims⟩`
    /// — the same round trip as [`Schur::recompose`], with a Hessenberg filling.
    pub fn recompose(
        &self,
    ) -> MixedUnitMatrix<RowDims, Mapped<Reciprocal, RowDims>, nalgebra::OMatrix<T, D, D>, Brand>
    where
        RowDims: MapUnits<Reciprocal>,
    {
        let q = self.q.nalgebra();
        MixedUnitMatrix::from_nalgebra(q * self.h.nalgebra() * q.adjoint())
    }
}

// Hessenberg `M = Q H Qᵀ` is a unitary *similarity* (one `Q`, reused as `Qᵀ`), so
// it inherits the metric bound `ColDims = 1/RowDims` shared by every member of
// that family; the reduced `H` is dimensionless. It is the Schur reduction step,
// exposed on its own.
impl<RowDims, ColDims, Brand, T, D, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, D, D, S>, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::DimSub<nalgebra::U1>,
    S: nalgebra::storage::Storage<T, D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>
        + nalgebra::allocator::Allocator<D>
        + nalgebra::allocator::Allocator<nalgebra::DimDiff<D, nalgebra::U1>>,
{
    /// The Hessenberg reduction `M = Q H Qᵀ` of a metric, returning a
    /// [`Hessenberg`] with the orthonormal frame `Q` and the dimensionless
    /// upper-Hessenberg `H`.
    ///
    /// Requires the metric shape `ColDims = 1/RowDims` — the same
    /// similarity-frame bound as [`schur`](Self::schur) and
    /// [`symmetric_eigen`](Self::symmetric_eigen). For a general
    /// (endomorphism-shaped) matrix, erase to
    /// [`UniformUnitMatrix::hessenberg`](crate::nalgebra::UniformUnitMatrix::hessenberg).
    pub fn hessenberg(self) -> Hessenberg<RowDims, T, D, Brand>
    where
        RowDims: MetricShape<ColDims> + MapUnits<ToDimensionless>,
    {
        let (q, h) = self.inner.clone_owned().hessenberg().unpack();
        Hessenberg {
            q: MixedUnitMatrix::from_nalgebra(q),
            h: UniformUnitMatrix::from_nalgebra(h),
        }
    }
}

/// The result of a Hessenberg reduction `M = Q H Qᵀ` of a uniform matrix.
///
/// Returned by [`UniformUnitMatrix::hessenberg`]. The similarity frame `Q … Qᵀ`
/// closes on any uniform matrix (its single unit is a canonical metric), so `Q`
/// is dimensionless and `H` carries `U`. Unconditional, like [`UniformSchur`]
/// whose reduction step it is.
pub struct UniformHessenberg<U, T, D, Brand = ()>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
{
    /// The orthonormal reducing basis `Q`, dimensionless.
    pub q: UniformUnitMatrix<Dimensionless, nalgebra::OMatrix<T, D, D>, Brand>,
    /// The upper-Hessenberg factor `H`, in the entry unit `U`.
    pub h: UniformUnitMatrix<U, nalgebra::OMatrix<T, D, D>, Brand>,
}

impl<U, T, D, Brand> UniformHessenberg<U, T, D, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
{
    /// Reconstructs `M = Q H Qᵀ`, in the entry unit `U`.
    pub fn recompose(&self) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, D, D>, Brand> {
        let q = self.q.nalgebra();
        UniformUnitMatrix::from_nalgebra(q * self.h.nalgebra() * q.adjoint())
    }
}

impl<U, Brand, T, D, S> UniformUnitMatrix<U, nalgebra::Matrix<T, D, D, S>, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::DimMin<D, Output = D> + nalgebra::DimSub<nalgebra::U1>,
    S: nalgebra::storage::Storage<T, D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>
        + nalgebra::allocator::Allocator<D>
        + nalgebra::allocator::Allocator<nalgebra::DimDiff<D, nalgebra::U1>>,
{
    /// The Hessenberg reduction `M = Q H Qᵀ`, with `Q` dimensionless and `H` in
    /// the entry unit `U`. The uniform, unconditional companion of the mixed
    /// [`hessenberg`](crate::nalgebra::MixedUnitMatrix::hessenberg).
    pub fn hessenberg(&self) -> UniformHessenberg<U, T, D, Brand> {
        let (q, h) = self.inner.clone_owned().hessenberg().unpack();
        UniformHessenberg {
            q: UniformUnitMatrix::from_nalgebra(q),
            h: UniformUnitMatrix::from_nalgebra(h),
        }
    }
}
