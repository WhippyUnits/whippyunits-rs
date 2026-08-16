#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use crate::dims::{Dimensionless, MapUnits, Mapped, MetricShape, Reciprocal, ToDimensionless};
use crate::nalgebra::matrix::MixedUnitMatrix;
use crate::nalgebra::uniform::UniformUnitMatrix;

/// The result of a `U D Uᵀ` factorization of a symmetric metric, mirroring
/// nalgebra's [`UDU`](nalgebra::UDU) but carrying the whippyunits row/column
/// types on each factor.
///
/// Returned by [`MixedUnitMatrix::udu`]. Like [Cholesky](MixedUnitMatrix::cholesky)
/// it needs the metric shape `ColDims = 1/RowDims` and splits the same way: the
/// unit-upper-triangular `U` lands in `⟨RowDims, [1…1]⟩` (dimensionless columns)
/// and `D` is a dimensionless diagonal. Keeping the root-free diagonal separate
/// means no even-exponent constraint applies to the mixed metric (contrast the
/// uniform [`UniformUDU`](crate::nalgebra::UniformUDU)).
pub struct UDU<RowDims, T, D, Brand = ()>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    RowDims: MapUnits<ToDimensionless>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// The unit-upper-triangular factor `U` (unit diagonal), in
    /// `⟨RowDims, [1 … 1]⟩` — dimensionless columns, the Cholesky-factor shape.
    pub u: MixedUnitMatrix<
        RowDims,
        Mapped<ToDimensionless, RowDims>,
        nalgebra::OMatrix<T, D, D>,
        Brand,
    >,
    /// The diagonal `D` as a dimensionless vector (a metric's pivots are pure
    /// numbers, like the [`SymmetricEigen`] eigenvalues).
    pub d: UniformUnitMatrix<Dimensionless, nalgebra::OVector<T, D>, Brand>,
}

impl<RowDims, T, D, Brand> UDU<RowDims, T, D, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    RowDims: MapUnits<ToDimensionless>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// Reconstructs the metric `M = U · diag(D) · Uᵀ` from its factors, landing
    /// back in `⟨RowDims, 1/RowDims⟩` — the same round trip as
    /// [`SymmetricEigen::recompose`], with a triangular `U` in place of `Q`.
    pub fn recompose(
        &self,
    ) -> MixedUnitMatrix<RowDims, Mapped<Reciprocal, RowDims>, nalgebra::OMatrix<T, D, D>, Brand>
    where
        RowDims: MapUnits<Reciprocal>,
    {
        let u = self.u.nalgebra();
        let diag = nalgebra::OMatrix::<T, D, D>::from_diagonal(self.d.nalgebra());
        MixedUnitMatrix::from_nalgebra(u * diag * u.adjoint())
    }
}

// `U D Uᵀ` lives on a *metric* for exactly the reason Cholesky and the symmetric
// eigendecomposition do: a self-transpose matrix forces `ColDims = 1/RowDims`,
// and then the unit-upper-triangular `U : ⟨RowDims, [1…1]⟩` contracts a
// dimensionless diagonal `D` — `U : ⟨RowDims,[1…1]⟩ · D : ⟨[1…1],[1…1]⟩ · Uᵀ :
// ⟨[1…1],1/RowDims⟩ = ⟨RowDims, 1/RowDims⟩`. No unit is rooted (the `√` of a
// Cholesky is replaced by the separate real diagonal), so no even-exponent
// constraint applies. nalgebra reads only the upper triangle.
impl<RowDims, ColDims, Brand, T, D, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, D, D, S>, Brand>
where
    T: nalgebra::RealField,
    D: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, D, D>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// The `U D Uᵀ` factorization of a metric, returning a [`UDU`] with the
    /// unit-upper-triangular `U` and the dimensionless diagonal `D`. Returns
    /// `None` when the factorization does not exist (a zero pivot).
    ///
    /// Requires the metric shape `ColDims = 1/RowDims` (`M = Mᵀ`) at the type
    /// level, exactly like [`cholesky`](Self::cholesky) and
    /// [`symmetric_eigen`](Self::symmetric_eigen); nalgebra reads only the upper
    /// triangle. See [`UDU`] for the full unit story.
    pub fn udu(self) -> Option<UDU<RowDims, T, D, Brand>>
    where
        RowDims: MetricShape<ColDims> + MapUnits<ToDimensionless>,
    {
        self.inner.udu().map(|udu| UDU {
            u: MixedUnitMatrix::from_nalgebra(udu.u),
            d: UniformUnitMatrix::from_nalgebra(udu.d),
        })
    }
}

/// The result of a `U D Uᵀ` factorization of a symmetric uniform matrix,
/// mirroring nalgebra's [`UDU`](nalgebra::UDU).
///
/// Returned by [`UniformUnitMatrix::udu`]. Unlike the uniform
/// [`cholesky`](UniformUnitMatrix::cholesky) — which must root the shared unit
/// (`L` in `√U`, gated on `U: UnitSqrt`) — `U D Uᵀ` keeps a separate diagonal/// that absorbs the whole unit: the unit-triangular factor `U` is
/// dimensionless and `D` carries `U`. So it is ungated (any `U`), the
/// uniform echo of why the mixed [`UDU`](crate::nalgebra::UDU) needs no even-exponent
/// constraint either.
pub struct UniformUDU<U, T, D, Brand = ()>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// The unit-upper-triangular factor `U` (unit diagonal), dimensionless.
    pub u: UniformUnitMatrix<Dimensionless, nalgebra::OMatrix<T, D, D>, Brand>,
    /// The diagonal `D` as a vector in the entry unit `U` — it carries the whole
    /// unit, so no square root (hence no even-exponent gate) is needed.
    pub d: UniformUnitMatrix<U, nalgebra::OVector<T, D>, Brand>,
}

impl<U, T, D, Brand> UniformUDU<U, T, D, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// Reconstructs `M = U · diag(D) · Uᵀ` from its factors, in the entry unit
    /// `U` (the dimensionless triangular frame leaves `D` carrying the unit).
    pub fn recompose(&self) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, D, D>, Brand> {
        let u = self.u.nalgebra();
        let diag = nalgebra::OMatrix::<T, D, D>::from_diagonal(self.d.nalgebra());
        UniformUnitMatrix::from_nalgebra(u * diag * u.adjoint())
    }
}

impl<U, Brand, T, D, S> UniformUnitMatrix<U, nalgebra::Matrix<T, D, D, S>, Brand>
where
    T: nalgebra::RealField,
    D: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, D, D>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// The `U D Uᵀ` factorization of a symmetric uniform matrix, returning a
    /// [`UniformUDU`] with the dimensionless unit-upper-triangular `U` and
    /// the diagonal `D` in the entry unit `U`. Returns `None` on a zero pivot.
    ///
    /// Ungated for any entry unit — the separate diagonal carries the whole
    /// unit, so nothing is rooted (contrast [`cholesky`](Self::cholesky), whose
    /// single factor needs `U: UnitSqrt`). nalgebra reads only the upper triangle.
    pub fn udu(self) -> Option<UniformUDU<U, T, D, Brand>> {
        self.inner.udu().map(|udu| UniformUDU {
            u: UniformUnitMatrix::from_nalgebra(udu.u),
            d: UniformUnitMatrix::from_nalgebra(udu.d),
        })
    }
}
