#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use crate::dims::{MapUnits, Mapped, PivotDims, Reciprocal, ZipToDimensionless};
use crate::nalgebra::matrix::MixedUnitMatrix;

/// The result of a weighted (generalized) QR decomposition `M = Q R`
/// of a mixed matrix `M : ⟨RowDims, ColDims⟩` against a codomain metric
/// `Gr : ⟨1/RowDims, RowDims⟩` (`Gr`: a positive-definite weight (metric/Gram)
/// matrix on the row space — see the [module overview](crate::nalgebra::linalg)).
/// Returned by [`MixedUnitMatrix::generalized_qr`].
/// `Q` is metric-orthonormal (`Qᴴ Gr Q = I`) and `R` is upper-triangular, sharing
/// the thin dimensionless pivot of length `min(m, n)` (see [`PivotDims`]): `Q` is
/// `m × min`, `R` is `min × n`, rectangular included.
///
/// If `Gr` is the identity metric, this is the same as the
/// [`UniformUnitMatrix::qr`](crate::nalgebra::UniformUnitMatrix::qr), whose single
/// shared unit is its own canonical metric. Demands a metric because a bare
/// mixed QR would silently pick the identity metric, which is not invariant under
/// rescaling in the mixed-unit case.
pub struct GeneralizedQR<RowDims, ColDims, T, R, C, Brand = ()>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C>,
    C: nalgebra::Dim,
    RowDims: ZipToDimensionless<ColDims>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, C>,
{
    /// The orthogonal factor `Q`, `Gr`-orthonormal, in `⟨RowDims, [1 … 1]⟩`: a
    /// codomain basis with dimensionless columns (the Cholesky-factor shape),
    /// `m × min(m, n)`. `Q` alone carries `M`'s row units.
    pub q: MixedUnitMatrix<
        RowDims,
        PivotDims<RowDims, ColDims>,
        nalgebra::OMatrix<T, R, nalgebra::DimMinimum<R, C>>,
        Brand,
    >,
    /// The upper-triangular factor `R`, in `⟨[1 … 1], ColDims⟩`: it maps the
    /// dimensionless pivot onto the column space, so `R` carries `M`'s column
    /// units, and is `min(m, n) × n`.
    pub r: MixedUnitMatrix<
        PivotDims<RowDims, ColDims>,
        ColDims,
        nalgebra::OMatrix<T, nalgebra::DimMinimum<R, C>, C>,
        Brand,
    >,
}

impl<RowDims, ColDims, T, R, C, Brand> GeneralizedQR<RowDims, ColDims, T, R, C, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C>,
    C: nalgebra::Dim,
    RowDims: ZipToDimensionless<ColDims>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, C>
        + nalgebra::allocator::Allocator<R, C>,
{
    /// Reconstructs `M = Q·R` from the factors, landing back in
    /// `⟨RowDims, ColDims⟩`.
    ///
    /// Unlike the two-sided generalized decompositions, QR is one-sided and needs
    /// no metric to close the loop: `Q R = (Lr⁻ᴴ Q̂) R = Lr⁻ᴴ M̂ = M` on the nose.
    /// `Q : ⟨RowDims, [1…1]⟩` contracts the dimensionless pivot with
    /// `R : ⟨[1…1], ColDims⟩` to rebuild `M`'s exact type.
    pub fn recompose(
        &self,
    ) -> MixedUnitMatrix<RowDims, ColDims, nalgebra::OMatrix<T, R, C>, Brand> {
        let m = self.q.nalgebra() * self.r.nalgebra();
        MixedUnitMatrix::from_nalgebra(m)
    }
}

// The generalized QR whitens `M`'s rows by the Cholesky factor of the codomain
// metric — `Gr = Lr Lrᴴ` — forming `M̂ = Lrᴴ M`, whose rows are dimensionless
// (`Lrᴴ : ⟨[1], R⟩` cancels `M`'s rows). An ordinary QR `M̂ = Q̂ R` then has a
// dimensionless `Q̂`, and the orthogonal factor maps back to the metric-orthonormal
// basis `Q = Lr⁻ᴴ Q̂ : ⟨R, [1]⟩`, while `R` is unchanged. No unit is ever rooted,
// so — like the mixed metric decompositions — it is ungated; and a genuinely mixed
// matrix's `Q` is orthonormal only against a *named* metric (that is the uniform
// QR's job for the canonical case). The row whitening is shape-preserving, so any
// `m × n` works, rectangular included.
impl<RowDims, ColDims, Brand, T, R, C, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C> + nalgebra::DimMin<R, Output = R>,
    C: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, R, C>,
    nalgebra::DimMinimum<R, C>: nalgebra::DimMin<C, Output = nalgebra::DimMinimum<R, C>>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>
        + nalgebra::allocator::Allocator<R>
        + nalgebra::allocator::Allocator<R, R>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<R, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, C>
        + nalgebra::allocator::Reallocator<T, R, C, nalgebra::DimMinimum<R, C>, C>,
{
    /// The weighted (generalized) QR `M = Q R`, orthonormalizing the
    /// columns against the codomain metric `g_r : ⟨1/RowDims, RowDims⟩` (`Gr`: a
    /// positive-definite weight (metric/Gram) matrix on the row space — see the
    /// [module overview](crate::nalgebra::linalg)). `Q` is
    /// `Gr`-orthonormal (`Qᴴ Gr Q = I`) and `R` is upper-triangular. Returns a
    /// [`GeneralizedQR`], or `None` when the metric is not positive-definite.
    ///
    /// If `Gr` is the identity metric, this is the same as the
    /// [`UniformUnitMatrix::qr`](crate::nalgebra::UniformUnitMatrix::qr), whose
    /// single shared unit is its own canonical metric. Demands a metric
    /// because a bare mixed QR would silently pick the identity metric, which is not
    /// invariant under rescaling in the mixed-unit case.
    pub fn generalized_qr<Sr>(
        self,
        g_r: &MixedUnitMatrix<
            Mapped<Reciprocal, RowDims>,
            RowDims,
            nalgebra::Matrix<T, R, R, Sr>,
            Brand,
        >,
    ) -> Option<GeneralizedQR<RowDims, ColDims, T, R, C, Brand>>
    where
        RowDims: MapUnits<Reciprocal> + ZipToDimensionless<ColDims>,
        Sr: nalgebra::storage::Storage<T, R, R>,
    {
        // Whiten the rows by the metric's Cholesky factor: M̂ = Lrᴴ M is
        // row-dimensionless.
        let lr = g_r.inner.clone_owned().cholesky()?.l();
        let lr_inv_adj = lr.clone().try_inverse()?.adjoint();
        let m_hat = lr.adjoint() * self.inner.clone_owned();

        let (q_hat, r) = m_hat.qr().unpack();

        // Map the whitened Q back to the metric-orthonormal basis.
        let q = &lr_inv_adj * q_hat;
        Some(GeneralizedQR {
            q: MixedUnitMatrix::from_nalgebra(q),
            r: MixedUnitMatrix::from_nalgebra(r),
        })
    }
}
