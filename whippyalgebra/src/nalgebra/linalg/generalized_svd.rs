#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use crate::dims::{Dimensionless, MapUnits, Mapped, PivotDims, Reciprocal, ZipToDimensionless};
use crate::nalgebra::matrix::MixedUnitMatrix;
use crate::nalgebra::uniform::UniformUnitMatrix;

/// The result of a weighted (generalized) singular value decomposition
/// `M = U Σ Vᴴ Gc` of a mixed matrix `M : ⟨RowDims, ColDims⟩` against a codomain
/// metric `Gr : ⟨1/RowDims, RowDims⟩` and a domain metric `Gc : ⟨1/ColDims, ColDims⟩`
/// (`Gr`, `Gc`: positive-definite weight (metric/Gram) matrices on the row/column
/// space — see the [module overview](crate::nalgebra::linalg)).
/// Returned by [`MixedUnitMatrix::generalized_svd`]. The spectrum `Σ` is
/// dimensionless and the singular vectors are metric-orthonormal
/// (`Uᴴ Gr U = I`, `Vᴴ Gc V = I`), sharing the thin dimensionless pivot of length
/// `min(m, n)` (see [`PivotDims`]): `U` is `m × min`, `V` is `n × min`. This is a
/// weighted SVD — the SVD of `M` in the `Gr`/`Gc` inner products — not the
/// generalized SVD of a matrix pair `(A, B)`.
///
/// If `Gr` and `Gc` are the identity metrics, this is the same as the
/// [`UniformUnitMatrix::svd`](crate::nalgebra::UniformUnitMatrix::svd), whose single
/// shared unit is a canonical metric that additionally keeps the spectrum
/// dimensioned. Demands a metric because a bare mixed SVD would silently pick
/// the identity metric, which is not invariant under rescaling in the mixed-unit case.
pub struct GeneralizedSVD<RowDims, ColDims, T, R, C, Brand = ()>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C>,
    C: nalgebra::Dim,
    RowDims: ZipToDimensionless<ColDims>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<C, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>>,
{
    /// The left singular vectors `U`, `Gr`-orthonormal, in `⟨RowDims, [1 … 1]⟩`: a
    /// codomain basis with dimensionless columns (the Cholesky-factor shape),
    /// `m × min(m, n)`. Column `k` pairs with `singular_values[k]`.
    pub u: MixedUnitMatrix<
        RowDims,
        PivotDims<RowDims, ColDims>,
        nalgebra::OMatrix<T, R, nalgebra::DimMinimum<R, C>>,
        Brand,
    >,
    /// The singular values `Σ` as a dimensionless vector (sorted descending),
    /// `min(m, n)` of them: ratios of two metric-measured lengths, hence pure
    /// numbers.
    pub singular_values: UniformUnitMatrix<
        Dimensionless,
        nalgebra::OVector<T::RealField, nalgebra::DimMinimum<R, C>>,
        Brand,
    >,
    /// The right singular vectors `V`, `Gc`-orthonormal, in `⟨ColDims, [1 … 1]⟩`:
    /// a domain basis with dimensionless columns, `n × min(m, n)`.
    pub v: MixedUnitMatrix<
        ColDims,
        PivotDims<RowDims, ColDims>,
        nalgebra::OMatrix<T, C, nalgebra::DimMinimum<R, C>>,
        Brand,
    >,
}

impl<RowDims, ColDims, T, R, C, Brand> GeneralizedSVD<RowDims, ColDims, T, R, C, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C>,
    C: nalgebra::Dim,
    RowDims: ZipToDimensionless<ColDims>,
    ColDims: MapUnits<Reciprocal>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<C, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, C>
        + nalgebra::allocator::Allocator<R, C>
        + nalgebra::allocator::Allocator<C, C>,
{
    /// Reconstructs `M = U · diag(Σ) · Vᴴ · Gc` from the factors — the inverse of
    /// [`MixedUnitMatrix::generalized_svd`].
    ///
    /// Like the generalized eigenproblems it needs the domain metric `g_c` back:
    /// `V` is `Gc`-orthonormal, so `V⁻¹ = Vᴴ Gc`, and `M = U Σ V⁻¹ = U Σ Vᴴ Gc`.
    /// The units close the loop to `⟨RowDims, ColDims⟩`: `U : ⟨RowDims, [1…1]⟩`
    /// and the dimensionless `Σ` land at `⟨RowDims, [1…1]⟩`, `Vᴴ : ⟨[1…1], 1/ColDims⟩`
    /// takes it to `⟨RowDims, 1/ColDims⟩`, and the metric `Gc : ⟨1/ColDims, ColDims⟩`
    /// contracts on its domain to rebuild `M`'s exact type.
    pub fn recompose<Sc>(
        &self,
        g_c: &MixedUnitMatrix<
            Mapped<Reciprocal, ColDims>,
            ColDims,
            nalgebra::Matrix<T, C, C, Sc>,
            Brand,
        >,
    ) -> MixedUnitMatrix<RowDims, ColDims, nalgebra::OMatrix<T, R, C>, Brand>
    where
        Sc: nalgebra::storage::Storage<T, C, C>,
    {
        let sigma = nalgebra::OMatrix::<T, nalgebra::DimMinimum<R, C>, nalgebra::DimMinimum<R, C>>::from_diagonal(
            &self.singular_values.nalgebra().map(|r| T::from_real(r)),
        );
        let m = self.u.nalgebra() * sigma * self.v.nalgebra().adjoint() * g_c.nalgebra();
        MixedUnitMatrix::from_nalgebra(m)
    }
}

// The generalized SVD whitens `M` by the Cholesky factors of the two supplied
// metrics — `Gr = Lr Lrᴴ`, `Gc = Lc Lcᴴ` — forming `M̂ = Lrᴴ M Lc⁻ᴴ`, which is
// *fully dimensionless* (a metric factor has dimensionless columns, `Lr : ⟨1/R,
// [1]⟩`, so `Lrᴴ : ⟨[1], R⟩` cancels `M`'s rows and `Lc⁻ᴴ : ⟨C, [1]⟩` its
// columns). An ordinary SVD `M̂ = Û Σ V̂ᴴ` then has dimensionless `Σ`, and the
// vectors map back to the metric-orthonormal bases `U = Lr⁻ᴴ Û : ⟨R, [1]⟩`,
// `V = Lc⁻ᴴ V̂ : ⟨C, [1]⟩`. No unit is ever rooted, so — like the mixed metric
// decompositions — the decomposition is ungated; and because a well-typed norm
// forces `Σ` dimensionless, there is deliberately no dimensioned mixed variant
// (that is the uniform SVD's job). The whitening is two-sided but shape-preserving,
// so any `m × n` works, rectangular included.
impl<RowDims, ColDims, Brand, T, R, C, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C> + nalgebra::DimMin<R, Output = R>,
    C: nalgebra::DimMin<C, Output = C>,
    S: nalgebra::storage::Storage<T, R, C>,
    nalgebra::DimMinimum<R, C>: nalgebra::DimSub<nalgebra::U1>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>
        + nalgebra::allocator::Allocator<C>
        + nalgebra::allocator::Allocator<R>
        + nalgebra::allocator::Allocator<R, R>
        + nalgebra::allocator::Allocator<C, C>
        + nalgebra::allocator::Allocator<nalgebra::DimDiff<nalgebra::DimMinimum<R, C>, nalgebra::U1>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, C>
        + nalgebra::allocator::Allocator<R, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<C, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>>,
{
    /// The weighted (generalized) SVD `M = U Σ Vᴴ Gc`, measuring `M`'s
    /// singular values against the codomain metric `g_r : ⟨1/RowDims, RowDims⟩` and
    /// the domain metric `g_c : ⟨1/ColDims, ColDims⟩` (`Gr`, `Gc`: positive-definite
    /// weight (metric/Gram) matrices on the row/column space — see the
    /// [module overview](crate::nalgebra::linalg)). The spectrum is dimensionless
    /// and the singular vectors are metric-orthonormal (`Uᴴ Gr U = I`, `Vᴴ Gc V = I`).
    /// Returns a [`GeneralizedSVD`], or `None` when either metric is not positive-definite.
    /// This is a weighted SVD (of `M` in the `Gr`/`Gc` inner products), not the
    /// generalized SVD of a matrix pair `(A, B)`.
    ///
    /// If `Gr` and `Gc` are the identity metrics, this is the same as the
    /// [`UniformUnitMatrix::svd`](crate::nalgebra::UniformUnitMatrix::svd), whose
    /// single shared unit is a canonical metric that additionally keeps the spectrum
    /// dimensioned. Demands a metric because a bare mixed SVD would silently
    /// pick the identity metric, which is not invariant under rescaling in the
    /// mixed-unit case.
    pub fn generalized_svd<Sr, Sc>(
        self,
        g_r: &MixedUnitMatrix<
            Mapped<Reciprocal, RowDims>,
            RowDims,
            nalgebra::Matrix<T, R, R, Sr>,
            Brand,
        >,
        g_c: &MixedUnitMatrix<
            Mapped<Reciprocal, ColDims>,
            ColDims,
            nalgebra::Matrix<T, C, C, Sc>,
            Brand,
        >,
    ) -> Option<GeneralizedSVD<RowDims, ColDims, T, R, C, Brand>>
    where
        RowDims: MapUnits<Reciprocal> + ZipToDimensionless<ColDims>,
        ColDims: MapUnits<Reciprocal>,
        Sr: nalgebra::storage::Storage<T, R, R>,
        Sc: nalgebra::storage::Storage<T, C, C>,
    {
        // Whiten by the metrics' Cholesky factors: M̂ = Lrᴴ M Lc⁻ᴴ is dimensionless.
        let lr = g_r.inner.clone_owned().cholesky()?.l();
        let lc = g_c.inner.clone_owned().cholesky()?.l();
        let lr_inv_adj = lr.clone().try_inverse()?.adjoint();
        let lc_inv_adj = lc.clone().try_inverse()?.adjoint();
        let m_hat = lr.adjoint() * self.inner.clone_owned() * &lc_inv_adj;

        let svd = m_hat.svd(true, true);
        let u_hat = svd.u.expect("compute_u = true yields Û");
        let vt_hat = svd.v_t.expect("compute_v = true yields V̂ᴴ");

        // Map the whitened vectors back to the metric-orthonormal bases.
        let u = &lr_inv_adj * u_hat;
        let v = &lc_inv_adj * vt_hat.adjoint();
        Some(GeneralizedSVD {
            u: MixedUnitMatrix::from_nalgebra(u),
            singular_values: UniformUnitMatrix::from_nalgebra(svd.singular_values),
            v: MixedUnitMatrix::from_nalgebra(v),
        })
    }
}
