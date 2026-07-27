#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use crate::dims::{Dimensionless, MapUnits, Mapped, MetricShape, Reciprocal, ToDimensionless};
use crate::nalgebra::matrix::MixedUnitMatrix;
use crate::nalgebra::uniform::UniformUnitMatrix;

/// The result of a symmetric tridiagonalization `M = Q T Qᵀ` of a metric.
///
/// Returned by [`MixedUnitMatrix::symmetric_tridiagonalize`]. Another unitary
/// similarity (same `Q` both sides), so it needs the metric shape
/// `⟨RowDims, 1/RowDims⟩`. It is the reduction step inside
/// [`symmetric_eigen`](MixedUnitMatrix::symmetric_eigen): `T` is the symmetric
/// tridiagonal form (dimensionless), `Q : ⟨RowDims, [1…1]⟩`.
pub struct SymmetricTridiagonal<RowDims, T, D, Brand = ()>
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
    /// The symmetric tridiagonal factor `T`, dimensionless.
    pub t: UniformUnitMatrix<Dimensionless, nalgebra::OMatrix<T, D, D>, Brand>,
}

impl<RowDims, T, D, Brand> SymmetricTridiagonal<RowDims, T, D, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    RowDims: MapUnits<ToDimensionless>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// Reconstructs the metric `M = Q T Qᵀ`, landing back in `⟨RowDims, 1/RowDims⟩`.
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

// Symmetric tridiagonalization `M = Q T Qᵀ` — the reduction feeding
// [`symmetric_eigen`] — is a unitary similarity, hence metric-bound; `T` is
// dimensionless. nalgebra hands back the diagonal/off-diagonal, so we reassemble
// the tridiagonal matrix to expose it as a typed factor.
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
    /// The symmetric tridiagonalization `M = Q T Qᵀ` of a metric, returning a
    /// [`SymmetricTridiagonal`] with the orthonormal frame `Q` and the
    /// dimensionless tridiagonal `T`.
    ///
    /// Requires the metric shape `ColDims = 1/RowDims`, like the rest of the
    /// similarity family. This is the reduction step
    /// [`symmetric_eigen`](Self::symmetric_eigen) runs before iterating.
    pub fn symmetric_tridiagonalize(self) -> SymmetricTridiagonal<RowDims, T, D, Brand>
    where
        RowDims: MetricShape<ColDims> + MapUnits<ToDimensionless>,
    {
        let (q, diag, off) = self.inner.clone_owned().symmetric_tridiagonalize().unpack();
        let mut t = nalgebra::OMatrix::<T, D, D>::from_diagonal(&diag.map(|r| T::from_real(r)));
        for i in 0..off.len() {
            let v = T::from_real(off[i].clone());
            t[(i, i + 1)] = v.clone();
            t[(i + 1, i)] = v;
        }
        SymmetricTridiagonal {
            q: MixedUnitMatrix::from_nalgebra(q),
            t: UniformUnitMatrix::from_nalgebra(t),
        }
    }
}

/// The result of a symmetric tridiagonalization `M = Q T Qᵀ` of a uniform
/// matrix.
///
/// Returned by [`UniformUnitMatrix::symmetric_tridiagonalize`]. As with
/// [`UniformSchur`], the uniform unit supplies the metric, so `Q` is
/// dimensionless and `T` carries `U`.
pub struct UniformSymmetricTridiagonal<U, T, D, Brand = ()>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
{
    /// The orthonormal reducing basis `Q`, dimensionless.
    pub q: UniformUnitMatrix<Dimensionless, nalgebra::OMatrix<T, D, D>, Brand>,
    /// The symmetric tridiagonal factor `T`, in the entry unit `U`.
    pub t: UniformUnitMatrix<U, nalgebra::OMatrix<T, D, D>, Brand>,
}

impl<U, T, D, Brand> UniformSymmetricTridiagonal<U, T, D, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
{
    /// Reconstructs `M = Q T Qᵀ`, in the entry unit `U`.
    pub fn recompose(&self) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, D, D>, Brand> {
        let q = self.q.nalgebra();
        UniformUnitMatrix::from_nalgebra(q * self.t.nalgebra() * q.adjoint())
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
    /// The symmetric tridiagonalization `M = Q T Qᵀ`, with `Q` dimensionless and
    /// `T` in the entry unit `U`.
    pub fn symmetric_tridiagonalize(&self) -> UniformSymmetricTridiagonal<U, T, D, Brand> {
        let (q, diag, off) = self.inner.clone_owned().symmetric_tridiagonalize().unpack();
        let mut t = nalgebra::OMatrix::<T, D, D>::from_diagonal(&diag.map(|r| T::from_real(r)));
        for i in 0..off.len() {
            let v = T::from_real(off[i].clone());
            t[(i, i + 1)] = v.clone();
            t[(i + 1, i)] = v;
        }
        UniformSymmetricTridiagonal {
            q: UniformUnitMatrix::from_nalgebra(q),
            t: UniformUnitMatrix::from_nalgebra(t),
        }
    }
}
