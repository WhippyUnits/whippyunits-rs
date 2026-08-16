#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use crate::dims::{Dimensionless, MapUnits, Mapped, PivotDims, Reciprocal, ZipToDimensionless};
use crate::nalgebra::matrix::MixedUnitMatrix;
use crate::nalgebra::uniform::UniformUnitMatrix;

/// The result of a weighted (generalized) bidiagonalization
/// `M = U B Vᴴ Gc` of a mixed matrix `M : ⟨RowDims, ColDims⟩` against a codomain
/// metric `Gr : ⟨1/RowDims, RowDims⟩` and a domain metric `Gc : ⟨1/ColDims, ColDims⟩`
/// (`Gr`, `Gc`: positive-definite weight (metric/Gram) matrices on the row/column
/// space — see the [module overview](crate::nalgebra::linalg)).
/// Returned by [`MixedUnitMatrix::generalized_bidiagonalize`]; the reduction step of
/// the [`generalized_svd`](MixedUnitMatrix::generalized_svd). The band `B` is
/// dimensionless and the frames are metric-orthonormal (`Uᴴ Gr U = I`, `Vᴴ Gc V = I`),
/// sharing the thin dimensionless pivot of length `min(m, n)` (see [`PivotDims`]):
/// `U` is `m × min`, `B` is `min × min`, `V` is `n × min`.
///
/// If `Gr` and `Gc` are the identity metrics, this is the same as the
/// [`UniformUnitMatrix::bidiagonalize`](crate::nalgebra::UniformUnitMatrix::bidiagonalize),
/// whose single shared unit is a canonical metric. Demands a metric because a
/// bare mixed bidiagonalization would silently pick the identity metric, which is not
/// invariant under rescaling in the mixed-unit case.
pub struct GeneralizedBidiagonal<RowDims, ColDims, T, R, C, Brand = ()>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C>,
    C: nalgebra::Dim,
    RowDims: ZipToDimensionless<ColDims>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<C, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, nalgebra::DimMinimum<R, C>>,
{
    /// The left factor `U`, `Gr`-orthonormal, in `⟨RowDims, [1 … 1]⟩`: a codomain
    /// basis with dimensionless columns (the Cholesky-factor shape), `m × min`.
    pub u: MixedUnitMatrix<
        RowDims,
        PivotDims<RowDims, ColDims>,
        nalgebra::OMatrix<T, R, nalgebra::DimMinimum<R, C>>,
        Brand,
    >,
    /// The bidiagonal band `B`, dimensionless, `min × min` (its lengths are ratios
    /// of two metric-measured lengths, hence pure numbers).
    pub d: UniformUnitMatrix<
        Dimensionless,
        nalgebra::OMatrix<T, nalgebra::DimMinimum<R, C>, nalgebra::DimMinimum<R, C>>,
        Brand,
    >,
    /// The right factor `V`, `Gc`-orthonormal, in `⟨ColDims, [1 … 1]⟩`: a domain
    /// basis with dimensionless columns, `n × min`.
    pub v: MixedUnitMatrix<
        ColDims,
        PivotDims<RowDims, ColDims>,
        nalgebra::OMatrix<T, C, nalgebra::DimMinimum<R, C>>,
        Brand,
    >,
}

impl<RowDims, ColDims, T, R, C, Brand> GeneralizedBidiagonal<RowDims, ColDims, T, R, C, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C>,
    C: nalgebra::Dim,
    RowDims: ZipToDimensionless<ColDims>,
    ColDims: MapUnits<Reciprocal>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<C, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, C>
        + nalgebra::allocator::Allocator<R, C>
        + nalgebra::allocator::Allocator<C, C>,
{
    /// Reconstructs `M = U · B · Vᴴ · Gc` from the factors — the inverse of
    /// [`MixedUnitMatrix::generalized_bidiagonalize`].
    ///
    /// Like the generalized SVD it needs the domain metric `g_c` back: `V` is
    /// `Gc`-orthonormal, so `V⁻¹ = Vᴴ Gc`, and `M = U B V⁻¹ = U B Vᴴ Gc`. The
    /// units close the loop to `⟨RowDims, ColDims⟩`.
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
        let m = self.u.nalgebra() * self.d.nalgebra() * self.v.nalgebra().adjoint() * g_c.nalgebra();
        MixedUnitMatrix::from_nalgebra(m)
    }
}

// The generalized bidiagonalization whitens `M` by the Cholesky factors of the
// two supplied metrics — `Gr = Lr Lrᴴ`, `Gc = Lc Lcᴴ` — forming
// `M̂ = Lrᴴ M Lc⁻ᴴ`, which is fully dimensionless. An ordinary bidiagonalization
// `M̂ = Û B V̂ᴴ` then has a dimensionless band `B`, and the vectors map back to
// the metric-orthonormal bases `U = Lr⁻ᴴ Û : ⟨R, [1]⟩`, `V = Lc⁻ᴴ V̂ : ⟨C, [1]⟩`.
// It is the reduction step of [`generalized_svd`](Self::generalized_svd), sharing
// its exact frame; no unit is ever rooted, so it is ungated. The whitening is
// shape-preserving, so any `m × n` works, rectangular included.
impl<RowDims, ColDims, Brand, T, R, C, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C> + nalgebra::DimMin<R, Output = R>,
    C: nalgebra::DimMin<C, Output = C>,
    S: nalgebra::storage::Storage<T, R, C>,
    nalgebra::DimMinimum<R, C>: nalgebra::DimSub<nalgebra::U1>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>
        + nalgebra::allocator::Allocator<R>
        + nalgebra::allocator::Allocator<C>
        + nalgebra::allocator::Allocator<R, R>
        + nalgebra::allocator::Allocator<C, C>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimDiff<nalgebra::DimMinimum<R, C>, nalgebra::U1>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<R, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<C, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, C>,
{
    /// The weighted (generalized) bidiagonalization `M = U B Vᴴ Gc`,
    /// against the codomain metric `g_r : ⟨1/RowDims, RowDims⟩` and the domain
    /// metric `g_c : ⟨1/ColDims, ColDims⟩` (`Gr`, `Gc`: positive-definite weight
    /// (metric/Gram) matrices on the row/column space — see the
    /// [module overview](crate::nalgebra::linalg)). The band `B` is dimensionless and the
    /// vectors are metric-orthonormal (`Uᴴ Gr U = I`, `Vᴴ Gc V = I`); it is the
    /// reduction step of the [`generalized_svd`](Self::generalized_svd). Returns a
    /// [`GeneralizedBidiagonal`], or `None` when either metric is not positive-definite.
    ///
    /// If `Gr` and `Gc` are the identity metrics, this is the same as the
    /// [`UniformUnitMatrix::bidiagonalize`](crate::nalgebra::UniformUnitMatrix::bidiagonalize),
    /// whose single shared unit is a canonical metric. Demands a metric
    /// because a bare mixed bidiagonalization would silently pick the identity
    /// metric, which is not invariant under rescaling in the mixed-unit case.
    pub fn generalized_bidiagonalize<Sr, Sc>(
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
    ) -> Option<GeneralizedBidiagonal<RowDims, ColDims, T, R, C, Brand>>
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

        let (u_hat, d, vt_hat) = m_hat.bidiagonalize().unpack();

        // Map the whitened vectors back to the metric-orthonormal bases.
        let u = &lr_inv_adj * u_hat;
        let v = &lc_inv_adj * vt_hat.adjoint();
        Some(GeneralizedBidiagonal {
            u: MixedUnitMatrix::from_nalgebra(u),
            d: UniformUnitMatrix::from_nalgebra(d),
            v: MixedUnitMatrix::from_nalgebra(v),
        })
    }
}
