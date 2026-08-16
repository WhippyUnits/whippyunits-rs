#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use whippyunits::{DivUnit, UnitDiv};

use crate::dims::{
    DiagUnit, Dimensionless, MapUnits, Mapped, MetricShape, Reciprocal, ToDimensionless,
    UniformDiag,
};
use crate::nalgebra::matrix::MixedUnitMatrix;
use crate::nalgebra::uniform::UniformUnitMatrix;

// Spectral scalars of a (uniform) endomorphism. The characteristic equation
// `det(A − λI) = 0` needs `λI` to carry `A`'s own diagonal unit, so every
// eigenvalue `λ` shares that unit. This is well-defined exactly when the
// diagonal `RowDims[i] / ColDims[i]` is a *single* unit `U` across the matrix —
// the [`UniformDiag`] condition — and the whole spectrum then carries that `U`.
// The dimensionless-endomorphism case (`RowDims = ColDims`) is the special
// instance `U = 1` (e.g. discrete-time stability is `|λ| < 1`); a continuous
// state matrix `⟨C/t, C⟩` is the general one, with `U = 1/t`, so its spectrum is
// the poles in `1/time`. Eigen*vectors* are *not* uniform in general (they live
// back in the state space `⟨Dims, [1]⟩`), and there is no general *orthogonal*
// eigen/`Schur` form here: an orthogonal similarity `A ↦ QᵀAQ` would force `Q` —
// and hence the whole space — dimensionless (`QᵀQ = I` contracts `1/Dims`
// against `Dims`), so it is representable only when `Dims` is already
// dimensionless. The symmetric case is therefore offered on a *metric* (see
// [`symmetric_eigen`](MixedUnitMatrix::symmetric_eigen)) rather than here; the
// uniform *spectrum*, however, is always well-typed.
impl<RowDims, ColDims, Brand, T, D, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, D, D, S>, Brand>
where
    RowDims: UniformDiag<ColDims>,
    T: nalgebra::ComplexField,
    D: nalgebra::DimSub<nalgebra::U1>,
    S: nalgebra::storage::Storage<T, D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>
        + nalgebra::allocator::Allocator<D>
        + nalgebra::allocator::Allocator<nalgebra::DimDiff<D, nalgebra::U1>>
        + nalgebra::allocator::Allocator<D, nalgebra::DimDiff<D, nalgebra::U1>>,
{
    /// The eigenvalues of the (uniform) endomorphism as a vector in the shared
    /// diagonal unit `U = RowDims[i] / ColDims[i]`, or `None` if any eigenvalue
    /// is non-real (they are found via the real Schur form, which succeeds only
    /// for a real spectrum).
    ///
    /// Every eigenvalue shares `A`'s diagonal unit, which the [`UniformDiag`]
    /// bound guarantees is a single `U`. For a dimensionless endomorphism
    /// (`RowDims = ColDims`) that unit is `1` and `get`/`iter` yield `qty!(1)` —
    /// what a spectral-radius / stability test consumes; for a continuous state
    /// matrix `⟨C/t, C⟩` it is `1/t`, so the entries are the poles in `1/time`.
    pub fn eigenvalues(
        &self,
    ) -> Option<UniformUnitMatrix<DiagUnit<RowDims, ColDims>, nalgebra::OVector<T, D>, Brand>> {
        self.inner
            .eigenvalues()
            .map(UniformUnitMatrix::from_nalgebra)
    }

    /// The eigenvalues as a complex vector in the shared diagonal unit `U`
    /// (always defined, unlike [`eigenvalues`](Self::eigenvalues), since complex
    /// conjugate pairs are represented rather than rejected).
    pub fn complex_eigenvalues(
        &self,
    ) -> UniformUnitMatrix<
        DiagUnit<RowDims, ColDims>,
        nalgebra::OVector<nalgebra::Complex<T>, D>,
        Brand,
    >
    where
        T: nalgebra::RealField,
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.complex_eigenvalues())
    }
}

/// The result of a generalized symmetric eigendecomposition of a metric pencil
/// `(K, M)` — the eigenpairs solving `K v = λ M v` where `K` and `M` are both
/// metrics of the same shape `⟨RowDims, 1/RowDims⟩` (with `M` positive-definite)
/// — carrying the whippyunits dimension types on each factor.
///
/// Returned by [`MixedUnitMatrix::generalized_symmetric_eigen`]. It is the
/// mixed-metric companion to the uniform
/// [`GeneralizedSymmetricEigen`](crate::nalgebra::GeneralizedSymmetricEigen), and
/// stands to it as [mixed Cholesky](MixedUnitMatrix::cholesky) stands to the
/// [uniform one](crate::nalgebra::UniformUnitMatrix::cholesky): reducing the
/// pencil roots the mass metric `M = L Lᴴ`, but a metric's Cholesky factor has
/// dimensionless columns, so the root never touches the units and no
/// even-exponent gate applies. The trade is that two same-shape metrics force
/// the generalized eigenvalues to be dimensionless (the uniform variant yields a
/// dimensioned `ω²`, at the cost of the `UnitSqrt` gate). The eigenvectors are
/// `M`-orthonormal (`Vᴴ M V = I`).
pub struct MetricGeneralizedEigen<RowDims, T, D, Brand = ()>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    RowDims: MapUnits<Reciprocal> + MapUnits<ToDimensionless>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// The generalized eigenvalues `Λ` as a dimensionless vector — the ratios of
    /// the two quadratic forms, which share a space and so cancel to pure
    /// numbers. Entry `k` pairs with column `k` of
    /// [`eigenvectors`](Self::eigenvectors).
    pub eigenvalues: UniformUnitMatrix<Dimensionless, nalgebra::OVector<T::RealField, D>, Brand>,
    /// The `M`-orthonormal eigenvectors `V` (`Vᴴ M V = I`), in
    /// `⟨1/RowDims, [1 … 1]⟩`: dimensionless columns (like a Cholesky factor),
    /// with rows in `1/RowDims` — the columns `v` solving `K v = λ M v`,
    /// recovered from the reduced problem as `V = L⁻ᴴ Y`.
    pub eigenvectors: MixedUnitMatrix<
        Mapped<Reciprocal, RowDims>,
        Mapped<ToDimensionless, RowDims>,
        nalgebra::OMatrix<T, D, D>,
        Brand,
    >,
}

impl<RowDims, T, D, Brand> MetricGeneralizedEigen<RowDims, T, D, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    RowDims: MapUnits<Reciprocal> + MapUnits<ToDimensionless>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// Reconstructs the stiffness `K = M V Λ Vᴴ M` from the factors — the inverse
    /// of [`MixedUnitMatrix::generalized_symmetric_eigen`].
    ///
    /// Like the [uniform recompose](crate::nalgebra::GeneralizedSymmetricEigen::recompose)
    /// it needs the mass metric `m` back, since the decomposition is spectral
    /// relative to `M`: `M`-orthonormality gives `V⁻¹ = Vᴴ M`, so
    /// `K = M V Λ V⁻¹ = M V Λ Vᴴ M`. The two flanking `M`s and the dimensionless
    /// `V Λ Vᴴ` between them rebuild `K`'s `⟨RowDims, 1/RowDims⟩` type.
    pub fn recompose<Sm>(
        &self,
        m: &MixedUnitMatrix<
            RowDims,
            Mapped<Reciprocal, RowDims>,
            nalgebra::Matrix<T, D, D, Sm>,
            Brand,
        >,
    ) -> MixedUnitMatrix<RowDims, Mapped<Reciprocal, RowDims>, nalgebra::OMatrix<T, D, D>, Brand>
    where
        Sm: nalgebra::storage::Storage<T, D, D>,
    {
        let v = self.eigenvectors.nalgebra();
        let m_own = m.nalgebra().clone_owned();
        let lambda = nalgebra::OMatrix::<T, D, D>::from_diagonal(
            &self.eigenvalues.nalgebra().map(|r| T::from_real(r)),
        );
        let k = &m_own * v * lambda * v.adjoint() * &m_own;
        MixedUnitMatrix::from_nalgebra(k)
    }
}

// The mixed-metric generalized *symmetric* eigendecomposition. Reduces the
// pencil `K v = λ M v` — both metrics `⟨Dm, 1/Dm⟩` — by rooting the mass metric
// `M = L Lᴴ`. A metric's Cholesky factor has dimensionless columns
// (`L : ⟨Dm, [1]⟩`), so the reduced `C = L⁻¹ K L⁻ᴴ` lands fully dimensionless:
// the generalized eigenvalues are pure numbers and the eigenvectors are
// `⟨1/Dm, [1]⟩`. No even-exponent gate applies (contrast the uniform variant),
// which is the exact analogue of mixed vs. uniform Cholesky.
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
    /// The generalized symmetric eigendecomposition of the metric pencil
    /// `(self, m)`, solving `self · v = λ · m · v` for two metrics `self = K` and
    /// `m = M` of the same shape `⟨RowDims, 1/RowDims⟩` (with `M`
    /// positive-definite). Returns a [`MetricGeneralizedEigen`], or `None` when
    /// `M` is not positive-definite.
    ///
    /// The generalized eigenvalues are dimensionless and the eigenvectors land in
    /// `⟨1/RowDims, [1 … 1]⟩`, `M`-orthonormal. Reducing the pencil roots `M`, but
    /// a metric's Cholesky factor is dimensionless-columned, so — unlike the
    /// uniform [`generalized_symmetric_eigen`](crate::nalgebra::UniformUnitMatrix::generalized_symmetric_eigen)
    /// — no even-exponent constraint applies; the trade is that same-shape
    /// metrics cannot carry a dimensioned spectrum (use the uniform variant, or
    /// the ungated pencil `m.try_inverse() * k` then
    /// [`eigenvalues`](Self::eigenvalues), for `ω²`).
    pub fn generalized_symmetric_eigen<Sm>(
        self,
        m: &MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, D, D, Sm>, Brand>,
    ) -> Option<MetricGeneralizedEigen<RowDims, T, D, Brand>>
    where
        RowDims: MetricShape<ColDims> + MapUnits<Reciprocal> + MapUnits<ToDimensionless>,
        Sm: nalgebra::storage::Storage<T, D, D>,
    {
        // M = L Lᴴ, then reduce to the symmetric standard problem C = L⁻¹ K L⁻ᴴ.
        let l = m.inner.clone_owned().cholesky()?.l();
        let l_inv = l.try_inverse()?;
        let l_inv_adj = l_inv.adjoint();
        let c = &l_inv * self.inner.clone_owned() * &l_inv_adj;
        let se = c.symmetric_eigen();
        // Map the reduced eigenvectors back to the pencil: V = L⁻ᴴ Y.
        let eigenvectors = &l_inv_adj * se.eigenvectors;
        Some(MetricGeneralizedEigen {
            eigenvalues: UniformUnitMatrix::from_nalgebra(se.eigenvalues),
            eigenvectors: MixedUnitMatrix::from_nalgebra(eigenvectors),
        })
    }
}

// Spectral scalars of a uniform (square) matrix. Every entry shares the unit
// `U`, so `det(A − λI) = 0` forces each eigenvalue `λ` to carry that same `U` —
// the fully-uniform instance of the [`MixedUnitMatrix`] endomorphism spectrum. A
// dimensionless `U` recovers the classic "eigenvalues are pure numbers"; a `U`
// of `1/s²` (e.g. the `M⁻¹K` of a mass/stiffness pencil) yields the squared
// modal frequencies `ω²` directly.
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
    /// The eigenvalues as a vector in the shared entry unit `U`, or `None` if any
    /// eigenvalue is non-real (found via the real Schur form).
    pub fn eigenvalues(&self) -> Option<UniformUnitMatrix<U, nalgebra::OVector<T, D>, Brand>> {
        self.inner
            .eigenvalues()
            .map(UniformUnitMatrix::from_nalgebra)
    }

    /// The eigenvalues as a complex vector in the entry unit `U` (always defined,
    /// since complex conjugate pairs are represented rather than rejected).
    pub fn complex_eigenvalues(
        &self,
    ) -> UniformUnitMatrix<U, nalgebra::OVector<nalgebra::Complex<T>, D>, Brand>
    where
        T: nalgebra::RealField,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.complex_eigenvalues())
    }

    /// The generalized eigenvalues of the pencil `(self, m)` — i.e. the `λ`
    /// solving `self · v = λ · m · v` — as a vector in the unit `U / Um`, or
    /// `None` if `m` is singular or the spectrum is non-real.
    ///
    /// Computed as the ordinary eigenvalues of `m⁻¹ · self`, a uniform
    /// endomorphism in `U / Um`. This is the ungated path: it needs no square
    /// root, so any pair of units works — a stiffness `U = N/m` against a mass
    /// `Um = kg` gives `ω² = 1/s²` directly. It does not exploit symmetry, so
    /// the spectrum is only returned when it happens to be real; for the
    /// guaranteed-real, orthonormal-eigenvector version see
    /// [`generalized_symmetric_eigen`](Self::generalized_symmetric_eigen).
    pub fn generalized_eigenvalues<Um, Sm>(
        &self,
        m: &UniformUnitMatrix<Um, nalgebra::Matrix<T, D, D, Sm>, Brand>,
    ) -> Option<UniformUnitMatrix<DivUnit<U, Um>, nalgebra::OVector<T, D>, Brand>>
    where
        U: UnitDiv<Um>,
        Sm: nalgebra::storage::Storage<T, D, D>,
    {
        m.inner
            .clone_owned()
            .try_inverse()
            .and_then(|m_inv| (m_inv * self.inner.clone_owned()).eigenvalues())
            .map(UniformUnitMatrix::from_nalgebra)
    }
}
