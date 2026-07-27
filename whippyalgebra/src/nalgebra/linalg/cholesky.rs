#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use core::marker::PhantomData;

use whippyunits::{DivUnit, InvUnit, SqrtUnit, UnitDiv, UnitInv, UnitSqrt};

use crate::dims::{MapUnits, Mapped, MetricShape, Product, Producted, ToDimensionless};
use crate::entry::{DetUnitOf, FromRaw};
use crate::nalgebra::matrix::MixedUnitMatrix;
use crate::nalgebra::uniform::UniformUnitMatrix;

impl<RowDims, ColDims, Brand, T, D, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, D, D, S>, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, D, D>,
{
    /// Cholesky factor `L` of a symmetric-positive-definite metric, so that
    /// `M = L Lᵀ`. Returns `None` when `M` is not positive-definite.
    ///
    /// Requires the metric shape `ColDims = 1/RowDims` (`M = Mᵀ`) at the type
    /// level. The factor lands in `⟨RowDims, [1 … 1]⟩` — row units `RowDims`,
    /// dimensionless columns — so `L Lᵀ` reconstructs `M`'s `⟨RowDims, 1/RowDims⟩`
    /// type exactly. The square root lives only in the scalar values
    /// (`L(i,i) = √M(i,i)`); `L`'s units are read straight off `RowDims`, never
    /// rooted, so no even-exponent constraint applies here (unlike a uniform
    /// Cholesky, whose single entry unit must be a perfect square).
    pub fn cholesky(self) -> Option<Cholesky<RowDims, ColDims, T, D, Brand>>
    where
        RowDims: MetricShape<ColDims> + MapUnits<ToDimensionless>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
    {
        self.inner.cholesky().map(|c| Cholesky {
            l: MixedUnitMatrix::from_nalgebra(c.l()),
            chol: c,
            _cols: PhantomData,
        })
    }
}

/// The Cholesky factorization `M = L Lᵀ` of a symmetric-positive-definite
/// metric, mirroring nalgebra's [`Cholesky`](nalgebra::Cholesky) but carrying
/// the whippyunits row/column types.
///
/// Returned by [`MixedUnitMatrix::cholesky`]. Holds the factorization for
/// factor-once-solve-many reuse: [`solve`](Self::solve), [`inverse`](Self::inverse)
/// and [`determinant`](Self::determinant) each cost only a back/forward
/// substitution rather than a fresh `O(n³)` factorization. The exposed
/// [`l`](Self::l) is the typed lower factor in `⟨RowDims, [1 … 1]⟩`
/// (dimensionless columns), so `L·Lᵀ` reconstructs `M`'s
/// `⟨RowDims, 1/RowDims⟩` type with no even-exponent constraint (contrast the
/// uniform [`UniformCholesky`](crate::nalgebra::UniformUnitMatrix::cholesky),
/// whose lone unit must be a perfect square).
pub struct Cholesky<RowDims, ColDims, T, D, Brand = ()>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    RowDims: MapUnits<ToDimensionless>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
{
    /// The lower-triangular factor `L`, in `⟨RowDims, [1 … 1]⟩` (rows keep `M`'s
    /// row units, columns collapse to dimensionless), with `M = L·Lᵀ`.
    pub l: MixedUnitMatrix<
        RowDims,
        Mapped<ToDimensionless, RowDims>,
        nalgebra::OMatrix<T, D, D>,
        Brand,
    >,
    chol: nalgebra::Cholesky<T, D>,
    _cols: PhantomData<fn() -> ColDims>,
}

impl<RowDims, ColDims, T, D, Brand> Cholesky<RowDims, ColDims, T, D, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    RowDims: MapUnits<ToDimensionless>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
{
    /// Solves `M x = b` reusing this factorization (forward/back substitution
    /// through `L`/`Lᵀ`): `b` lives in the output space (`RowDims`), the solution
    /// `x` in the input space (`ColDims = 1/RowDims`), carrying `b`'s own column
    /// dims — the same signature as
    /// [`MixedUnitMatrix::solve`](crate::nalgebra::MixedUnitMatrix::solve).
    /// Infallible: a positive-definite metric is always invertible.
    pub fn solve<BCols, C2, SB>(
        &self,
        b: &MixedUnitMatrix<RowDims, BCols, nalgebra::Matrix<T, D, C2, SB>, Brand>,
    ) -> MixedUnitMatrix<ColDims, BCols, nalgebra::OMatrix<T, D, C2>, Brand>
    where
        C2: nalgebra::Dim,
        SB: nalgebra::storage::Storage<T, D, C2>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, C2>,
    {
        MixedUnitMatrix::from_nalgebra(self.chol.solve(&b.inner))
    }

    /// The inverse `M⁻¹`, in the swapped `⟨ColDims, RowDims⟩` — the same type as
    /// [`MixedUnitMatrix::try_inverse`](crate::nalgebra::MixedUnitMatrix::try_inverse), reusing
    /// this factorization. Infallible for a positive-definite metric.
    pub fn inverse(&self) -> MixedUnitMatrix<ColDims, RowDims, nalgebra::OMatrix<T, D, D>, Brand> {
        MixedUnitMatrix::from_nalgebra(self.chol.inverse())
    }

    /// The determinant, in `∏RowDims / ∏ColDims` (real and positive for a metric),
    /// reusing this factorization.
    pub fn determinant(&self) -> DetUnitOf<RowDims, ColDims, T, Brand>
    where
        RowDims: Product,
        ColDims: Product,
        Producted<RowDims>: UnitDiv<Producted<ColDims>>,
        DetUnitOf<RowDims, ColDims, T, Brand>: FromRaw<T>,
    {
        <DetUnitOf<RowDims, ColDims, T, Brand> as FromRaw<T>>::from_raw(T::from_real(
            self.chol.determinant(),
        ))
    }
}

impl<U, Brand, T, D, S> UniformUnitMatrix<U, nalgebra::Matrix<T, D, D, S>, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, D, D>,
{
    /// Cholesky factor `L` of a symmetric-positive-definite uniform matrix, so
    /// that `M = L Lᵀ`. Returns `None` when `M` is not positive-definite.
    ///
    /// Because every entry shares one unit `U` and the uniform transpose keeps
    /// that unit, the product `L Lᵀ` multiplies entry units (`√U · √U = U`), so
    /// the factor is uniform in `√U`. That root exists only when every exponent
    /// of `U` is even (the `U: UnitSqrt` bound); a `U` with an odd exponent (e.g.
    /// `m`) has no uniform Cholesky and is rejected at compile time. The
    /// [mixed metric](crate::nalgebra::MixedUnitMatrix::cholesky) has no such gate — its
    /// columns collapse to dimensionless.
    pub fn cholesky(self) -> Option<UniformCholesky<U, T, D, Brand>>
    where
        U: UnitSqrt,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
    {
        self.inner.cholesky().map(|c| UniformCholesky {
            l: UniformUnitMatrix::from_nalgebra(c.l()),
            chol: c,
        })
    }
}

/// The Cholesky factorization `M = L Lᵀ` of a symmetric-positive-definite uniform
/// matrix, mirroring nalgebra's [`Cholesky`](nalgebra::Cholesky).
///
/// Returned by [`UniformUnitMatrix::cholesky`]. It holds the factorization for
/// factor-once-solve-many reuse: [`solve`](Self::solve) and
/// [`inverse`](Self::inverse) cost only a substitution, not a fresh `O(n³)`
/// factorization. The exposed [`l`](Self::l) is the uniform lower factor in `√U`
/// — which is why the whole decomposition is gated on `U: UnitSqrt` (every
/// exponent of `U` even), unlike the mixed-metric
/// [`Cholesky`](crate::nalgebra::MixedUnitMatrix::cholesky) whose columns collapse to
/// dimensionless.
pub struct UniformCholesky<U, T, D, Brand = ()>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    U: UnitSqrt,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
{
    /// The lower-triangular factor `L`, uniform in `√U`, with `M = L·Lᵀ`.
    pub l: UniformUnitMatrix<SqrtUnit<U>, nalgebra::OMatrix<T, D, D>, Brand>,
    chol: nalgebra::Cholesky<T, D>,
}

impl<U, T, D, Brand> UniformCholesky<U, T, D, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    U: UnitSqrt,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
{
    /// Solves `M x = b` reusing this factorization, in the quotient unit
    /// `Ub / U` — the same signature as [`UniformUnitMatrix::solve`](crate::nalgebra::UniformUnitMatrix::solve),
    /// but without re-factorizing. Infallible: a positive-definite matrix is always
    /// invertible.
    pub fn solve<Ub, C2, SB>(
        &self,
        b: &UniformUnitMatrix<Ub, nalgebra::Matrix<T, D, C2, SB>, Brand>,
    ) -> UniformUnitMatrix<DivUnit<Ub, U>, nalgebra::OMatrix<T, D, C2>, Brand>
    where
        Ub: UnitDiv<U>,
        C2: nalgebra::Dim,
        SB: nalgebra::storage::Storage<T, D, C2>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, C2>,
    {
        UniformUnitMatrix::from_nalgebra(self.chol.solve(&b.inner))
    }

    /// The inverse `M⁻¹`, uniform with the reciprocal unit `1 / U` — the same
    /// type as [`UniformUnitMatrix::try_inverse`](crate::nalgebra::UniformUnitMatrix::try_inverse),
    /// reusing this factorization. Infallible for a positive-definite matrix.
    pub fn inverse(&self) -> UniformUnitMatrix<InvUnit<U>, nalgebra::OMatrix<T, D, D>, Brand>
    where
        U: UnitInv,
    {
        UniformUnitMatrix::from_nalgebra(self.chol.inverse())
    }
}
