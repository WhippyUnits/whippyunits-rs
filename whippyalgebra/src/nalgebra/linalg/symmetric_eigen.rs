#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use whippyunits::{DivUnit, InvUnit, SqrtUnit, UnitDiv, UnitInv, UnitSqrt};

use crate::dims::{Dimensionless, MapUnits, Mapped, MetricShape, Reciprocal, ToDimensionless};
use crate::nalgebra::matrix::MixedUnitMatrix;
use crate::nalgebra::uniform::UniformUnitMatrix;

/// The result of a symmetric (metric) eigendecomposition `M = Q Λ Qᵀ`,
/// mirroring nalgebra's [`SymmetricEigen`](nalgebra::SymmetricEigen) but carrying
/// the whippyunits row/column dimension types on each factor.
///
/// Returned by [`MixedUnitMatrix::symmetric_eigen`]. The two factors are typed so
/// that `Q · diag(Λ) · Qᵀ` reconstructs the original metric `⟨RowDims, 1/RowDims⟩`
/// exactly — see that method for the full unit story.
pub struct SymmetricEigen<RowDims, T, D, Brand = ()>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    RowDims: MapUnits<ToDimensionless>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// The eigenvectors `Q`, in `⟨RowDims, [1 … 1]⟩`: an orthonormal basis of the
    /// row (state) space, whose columns are dimensionless (the same shape as a
    /// Cholesky factor). Column `k` is the eigenvector for `eigenvalues[k]`.
    pub eigenvectors: MixedUnitMatrix<
        RowDims,
        Mapped<ToDimensionless, RowDims>,
        nalgebra::OMatrix<T, D, D>,
        Brand,
    >,
    /// The eigenvalues `Λ` as a dimensionless vector — a metric's eigenvalues
    /// are pure numbers (the diagonal of `Qᵀ M Q`).
    pub eigenvalues: UniformUnitMatrix<Dimensionless, nalgebra::OVector<T::RealField, D>, Brand>,
}

impl<RowDims, T, D, Brand> SymmetricEigen<RowDims, T, D, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    RowDims: MapUnits<ToDimensionless>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// Reconstructs the original metric `M = Q · diag(Λ) · Qᵀ` from its factors —
    /// the inverse of [`MixedUnitMatrix::symmetric_eigen`], mirroring nalgebra's
    /// [`SymmetricEigen::recompose`](nalgebra::SymmetricEigen::recompose).
    ///
    /// The result lands back in `⟨RowDims, 1/RowDims⟩`: `Q : ⟨RowDims, [1…1]⟩`
    /// contracts `diag(Λ) : ⟨[1…1], [1…1]⟩` on the dimensionless pivot, then
    /// `Qᵀ : ⟨[1…1], 1/RowDims⟩`, so the product is exactly the metric type.
    pub fn recompose(
        &self,
    ) -> MixedUnitMatrix<RowDims, Mapped<Reciprocal, RowDims>, nalgebra::OMatrix<T, D, D>, Brand>
    where
        RowDims: MapUnits<Reciprocal>,
    {
        let q = self.eigenvectors.nalgebra();
        // Promote the (real) eigenvalues to `T` on the diagonal, then form the
        // similarity `Q · diag(Λ) · Qᵀ` in the underlying scalar field.
        let diag = nalgebra::OMatrix::<T, D, D>::from_diagonal(
            &self.eigenvalues.nalgebra().map(|r| T::from_real(r)),
        );
        MixedUnitMatrix::from_nalgebra(q * diag * q.adjoint())
    }
}

// The symmetric eigendecomposition lives on a *metric*, not an endomorphism: a
// self-transpose matrix `M = Mᵀ` forces `ColDims = 1/RowDims` (the same shape
// Cholesky needs). Then `M = Q Λ Qᵀ` with the eigenvectors `Q : ⟨RowDims, [1…1]⟩`
// (dimensionless columns, exactly the Cholesky-factor shape) and the eigenvalues
// `Λ` **dimensionless**:
//
//   Q : ⟨RowDims, [1…1]⟩   Λ : ⟨[1…1], [1…1]⟩   Qᵀ : ⟨[1…1], 1/RowDims⟩
//   Q·Λ·Qᵀ : ⟨RowDims, 1/RowDims⟩  =  M
//
// (`QᵀQ = I` is *not* something we type — it contracts `1/RowDims` against
// `RowDims` and so would need a dimensionless space — but the reconstruction
// `QΛQᵀ`, which contracts only on the dimensionless pivot, is well-typed and is
// what we return the factors for.)
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
    /// The symmetric eigendecomposition `M = Q Λ Qᵀ` of a metric, returning a
    /// [`SymmetricEigen`] with the eigenvectors `Q` and the dimensionless
    /// eigenvalues `Λ`.
    ///
    /// Requires the metric shape `ColDims = 1/RowDims` (`M = Mᵀ`) at the type
    /// level, like [`cholesky`](Self::cholesky). The eigenvectors land in
    /// `⟨RowDims, [1…1]⟩` with dimensionless columns, so `Q` re-expresses the row
    /// (state) space in an orthonormal basis, and the eigenvalues are pure
    /// numbers; `Q · diag(Λ) · Qᵀ` reconstructs `M`'s exact type.
    ///
    /// nalgebra reads only the lower triangle (assuming symmetry), so a
    /// metric-typed but numerically non-symmetric matrix is symmetrized by that
    /// convention, as in nalgebra itself.
    pub fn symmetric_eigen(self) -> SymmetricEigen<RowDims, T, D, Brand>
    where
        RowDims: MetricShape<ColDims> + MapUnits<ToDimensionless>,
    {
        let eig = self.inner.symmetric_eigen();
        SymmetricEigen {
            eigenvectors: MixedUnitMatrix::from_nalgebra(eig.eigenvectors),
            eigenvalues: UniformUnitMatrix::from_nalgebra(eig.eigenvalues),
        }
    }
}

/// The result of a generalized symmetric eigendecomposition of a pencil
/// `(K, M)` — the eigenpairs solving `K v = λ M v` with `K` Hermitian and `M`
/// Hermitian-positive-definite — carrying the whippyunits entry units on each
/// factor.
///
/// Returned by [`UniformUnitMatrix::generalized_symmetric_eigen`]. Reducing the
/// pencil to a standard symmetric problem needs a root of the mass metric `M`
/// (`M = L Lᴴ`, then `C = L⁻¹ K L⁻ᴴ`), so — mirroring the
/// [uniform Cholesky](UniformUnitMatrix::cholesky) — it typechecks only when the
/// mass unit `Um` has all-even exponents (`Um: UnitSqrt`). In exchange for that
/// gate you get the guarantees the [ungated
/// path](UniformUnitMatrix::generalized_eigenvalues) cannot: a real spectrum
/// and an `M`-orthonormal eigenvector basis (`Vᴴ M V = I`).
pub struct GeneralizedSymmetricEigen<Uk, Um, T, D, Brand = ()>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    Uk: UnitDiv<Um>,
    Um: UnitSqrt,
    SqrtUnit<Um>: UnitInv,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// The generalized eigenvalues `λ` (real) in the unit `Uk / Um` — e.g. `ω²`
    /// in `1/s²` for a stiffness/mass pencil. Entry `k` pairs with column `k` of
    /// [`eigenvectors`](Self::eigenvectors).
    pub eigenvalues: UniformUnitMatrix<DivUnit<Uk, Um>, nalgebra::OVector<T::RealField, D>, Brand>,
    /// The `M`-orthonormal eigenvectors `V` (`Vᴴ M V = I`), uniform in
    /// `1 / √Um`: they are the columns `v` satisfying `K v = λ M v`, recovered
    /// from the reduced problem as `V = L⁻ᴴ Y`.
    pub eigenvectors: UniformUnitMatrix<InvUnit<SqrtUnit<Um>>, nalgebra::OMatrix<T, D, D>, Brand>,
}

impl<Uk, Um, T, D, Brand> GeneralizedSymmetricEigen<Uk, Um, T, D, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    Uk: UnitDiv<Um>,
    Um: UnitSqrt,
    SqrtUnit<Um>: UnitInv,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// Reconstructs the stiffness `K = M V Λ Vᴴ M` from the factors — the inverse
    /// of [`UniformUnitMatrix::generalized_symmetric_eigen`].
    ///
    /// Unlike the ordinary [`SymmetricEigen::recompose`](crate::nalgebra::SymmetricEigen),
    /// this needs the mass matrix `m` back: a generalized decomposition is
    /// spectral relative to `M`, and it is `M`-orthonormality (`Vᴴ M V = I`),
    /// not `Vᴴ V = I`, that inverts the eigenvector basis — so `V⁻¹ = Vᴴ M` and
    /// `K = M V Λ V⁻¹ = M V Λ Vᴴ M`. The units close the loop back to `Uk`: the
    /// two flanking `M`s (`Um`) and the `Vᴴ … V` (`1/√Um` twice) exactly cancel
    /// the `1/Um` in `Λ`'s `Uk/Um`, leaving `Uk`.
    pub fn recompose<Sm>(
        &self,
        m: &UniformUnitMatrix<Um, nalgebra::Matrix<T, D, D, Sm>, Brand>,
    ) -> UniformUnitMatrix<Uk, nalgebra::OMatrix<T, D, D>, Brand>
    where
        Sm: nalgebra::storage::Storage<T, D, D>,
    {
        let v = self.eigenvectors.nalgebra();
        let m_own = m.nalgebra().clone_owned();
        // Promote the real eigenvalues onto the diagonal in the scalar field `T`.
        let lambda = nalgebra::OMatrix::<T, D, D>::from_diagonal(
            &self.eigenvalues.nalgebra().map(|r| T::from_real(r)),
        );
        let k = &m_own * v * lambda * v.adjoint() * &m_own;
        UniformUnitMatrix::from_nalgebra(k)
    }
}

// The `UnitSqrt`-gated generalized *symmetric* eigendecomposition. Unlike the
// ungated `generalized_eigenvalues` (eigenvalues of `M⁻¹K`, which discards
// symmetry), this reduces `K v = λ M v` to a genuine symmetric standard problem
// by rooting the mass metric — `M = L Lᴴ` (uniform Cholesky, `L : √Um`), then
// `C = L⁻¹ K L⁻ᴴ` (Hermitian, unit `Uk/Um`). Because it forms `L`, it inherits
// exactly Cholesky's constraint: the root `√Um` exists only when every exponent
// of `Um` is even. What it buys is the real spectrum and `M`-orthonormal
// eigenvectors of a symmetric solver.
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
    /// The generalized symmetric eigendecomposition of the pencil `(self, m)`,
    /// solving `self · v = λ · m · v` for a Hermitian `self = K` and a
    /// Hermitian-positive-definite `m = M`. Returns a
    /// [`GeneralizedSymmetricEigen`], or `None` when `M` is not positive-definite.
    ///
    /// The eigenvalues land in `U / Um` (real — `ω²` in `1/s²` for a
    /// stiffness/mass pencil) and the eigenvectors in `1 / √Um`, `M`-orthonormal.
    /// Reducing the pencil roots the mass metric (`M = L Lᴴ`), so — exactly like
    /// [`cholesky`](Self::cholesky) — this requires `Um: UnitSqrt`, i.e. every
    /// exponent of the mass unit is even; an odd-exponent mass (e.g. `kg`) has
    /// no uniform root and is rejected at compile time (use
    /// [`generalized_eigenvalues`](Self::generalized_eigenvalues) for the ungated
    /// eigenvalues-only path).
    pub fn generalized_symmetric_eigen<Um, Sm>(
        self,
        m: &UniformUnitMatrix<Um, nalgebra::Matrix<T, D, D, Sm>, Brand>,
    ) -> Option<GeneralizedSymmetricEigen<U, Um, T, D, Brand>>
    where
        U: UnitDiv<Um>,
        Um: UnitSqrt,
        SqrtUnit<Um>: UnitInv,
        Sm: nalgebra::storage::Storage<T, D, D>,
    {
        // M = L Lᴴ, then reduce to the symmetric standard problem C = L⁻¹ K L⁻ᴴ.
        let l = m.inner.clone_owned().cholesky()?.l();
        let l_inv = l.try_inverse()?;
        let l_inv_adj = l_inv.adjoint();
        let c = l_inv * self.inner.clone_owned() * &l_inv_adj;
        let se = c.symmetric_eigen();
        // Map the reduced eigenvectors back to the pencil: V = L⁻ᴴ Y.
        let eigenvectors = l_inv_adj * se.eigenvectors;
        Some(GeneralizedSymmetricEigen {
            eigenvalues: UniformUnitMatrix::from_nalgebra(se.eigenvalues),
            eigenvectors: UniformUnitMatrix::from_nalgebra(eigenvectors),
        })
    }
}
