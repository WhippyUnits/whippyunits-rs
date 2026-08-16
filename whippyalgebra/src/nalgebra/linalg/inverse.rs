#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use whippyunits::{InvUnit, UnitInv};

use crate::dims::{MapUnits, Mapped, Reciprocal};
use crate::nalgebra::matrix::MixedUnitMatrix;
use crate::nalgebra::uniform::UniformUnitMatrix;

// A *bare* mixed pseudoinverse is deliberately absent — gated exactly as strictly
// as the mixed QR family. Its *type* `⟨ColDims, RowDims⟩` is Penrose-forced and
// honest, but its *values* are the Euclidean minimum-norm least-squares inverse,
// and that Euclidean structure is a silent identity metric on each space: the
// least-squares residual `‖Ax − b‖²` and the minimum-norm tie-break `‖x‖²` sum
// entries of *different* units whenever the reduced-against margin is non-uniform.
// For a genuinely mixed rectangular/rank-deficient matrix the "best fit" then
// depends on the choice of units — breaking the rescale-invariance the rest of the
// library guarantees — so the mixed case must name a metric on each space:
// [`generalized_pseudo_inverse`](MixedUnitMatrix::generalized_pseudo_inverse) below.
// A square, full-rank matrix is metric-free (`A⁺ = A⁻¹` — see
// [`try_inverse`](MixedUnitMatrix::try_inverse)), and a uniform matrix carries its
// own canonical metric, keeping the bare
// [`pseudo_inverse`](UniformUnitMatrix::pseudo_inverse).
impl<RowDims, ColDims, Brand, T, R, C, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C>,
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
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<C, R>,
{
    /// The weighted (generalized) Moore–Penrose pseudoinverse `A⁺_{Gr,Gc}`, in
    /// `⟨ColDims, RowDims⟩`. `x = A⁺_{Gr,Gc} b` minimizes the
    /// `Gr`-norm of the residual `A x − b`, with minimum-`Gc`-norm in the nullspace
    /// of `A` (`Gr`, `Gc`: positive-definite weight (metric/Gram) matrices on the
    /// row/column space — see the [module overview](crate::nalgebra::linalg)).
    /// 
    /// If `Gr` and `Gc` are the identity metrics, this is the same as the
    /// [`UniformUnitMatrix::pseudo_inverse`](crate::nalgebra::UniformUnitMatrix::pseudo_inverse).
    /// Demands a metric because a bare mixed pseudoinverse would silently pick the
    /// identity metric, which is not invariant under rescaling in the mixed-unit case.
    pub fn generalized_pseudo_inverse<Sr, Sc>(
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
        eps: T::RealField,
    ) -> Result<MixedUnitMatrix<ColDims, RowDims, nalgebra::OMatrix<T, C, R>, Brand>, &'static str>
    where
        RowDims: MapUnits<Reciprocal>,
        ColDims: MapUnits<Reciprocal>,
        Sr: nalgebra::storage::Storage<T, R, R>,
        Sc: nalgebra::storage::Storage<T, C, C>,
    {
        // Whiten both sides so the least-squares geometry is Euclidean:
        // M̂ = Lrᴴ M Lc⁻ᴴ is dimensionless, so its Euclidean pseudoinverse is the
        // honest one. Then A⁺ = Lc⁻ᴴ M̂⁺ Lrᴴ maps back to ⟨ColDims, RowDims⟩.
        let lr = g_r
            .inner
            .clone_owned()
            .cholesky()
            .ok_or("codomain metric is not positive-definite")?
            .l();
        let lc = g_c
            .inner
            .clone_owned()
            .cholesky()
            .ok_or("domain metric is not positive-definite")?
            .l();
        let lr_adj = lr.adjoint();
        let lc_inv_adj = lc
            .try_inverse()
            .ok_or("domain metric is not invertible")?
            .adjoint();
        let m_hat = &lr_adj * self.inner.clone_owned() * &lc_inv_adj;
        let m_hat_pinv = m_hat.pseudo_inverse(eps)?;
        let a_pinv = &lc_inv_adj * m_hat_pinv * &lr_adj;
        Ok(MixedUnitMatrix::from_nalgebra(a_pinv))
    }
}

impl<RowDims, ColDims, Brand, T, D, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, D, D, S>, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, D, D>,
{
    /// Attempts to invert the matrix.
    ///
    /// The inverse maps output back to input, so its row/column dimension lists
    /// are the original column/row lists swapped (entry `(j, i)` has unit
    /// `col_j / row_i`). Returns `None` when the matrix is not invertible.
    pub fn try_inverse(
        self,
    ) -> Option<MixedUnitMatrix<ColDims, RowDims, nalgebra::OMatrix<T, D, D>, Brand>>
    where
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
    {
        self.inner.try_inverse().map(MixedUnitMatrix::from_nalgebra)
    }
}

impl<U, Brand, T, D, S> UniformUnitMatrix<U, nalgebra::Matrix<T, D, D, S>, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, D, D>,
{
    /// Attempts to invert the matrix; the inverse is uniform with the
    /// reciprocal unit `1 / U`, so `M⁻¹ · M` collapses to the dimensionless
    /// identity. Returns `None` when the matrix is singular.
    ///
    /// This is the uniform counterpart of
    /// [`MixedUnitMatrix::try_inverse`](crate::nalgebra::MixedUnitMatrix::try_inverse); a
    /// single shared unit carries no row/column labels to swap, so inversion just
    /// reciprocates it.
    pub fn try_inverse(
        self,
    ) -> Option<UniformUnitMatrix<InvUnit<U>, nalgebra::OMatrix<T, D, D>, Brand>>
    where
        U: UnitInv,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
    {
        self.inner
            .try_inverse()
            .map(UniformUnitMatrix::from_nalgebra)
    }
}

// The Moore-Penrose pseudoinverse, uniform. Like the uniform
// [`try_inverse`](UniformUnitMatrix::try_inverse) it just reciprocates the shared
// unit (`U → 1/U`); a single unit carries no row/column labels to swap. It
// generalizes the inverse to rectangular/rank-deficient matrices. Its *values*
// are the identity-metric ones from the underlying SVD — sound here precisely
// because the single shared unit *is* a canonical metric, so the least-squares
// geometry is well-posed. A genuinely mixed matrix has no such metric and must
// name one via
// [`generalized_pseudo_inverse`](crate::nalgebra::MixedUnitMatrix::generalized_pseudo_inverse).
impl<U, Brand, T, R, C, S> UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C>,
    C: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, R, C>,
    nalgebra::DimMinimum<R, C>: nalgebra::DimSub<nalgebra::U1>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>
        + nalgebra::allocator::Allocator<C>
        + nalgebra::allocator::Allocator<R>
        + nalgebra::allocator::Allocator<nalgebra::DimDiff<nalgebra::DimMinimum<R, C>, nalgebra::U1>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, C>
        + nalgebra::allocator::Allocator<R, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<C, R>,
{
    /// The Moore-Penrose pseudoinverse, uniform in the reciprocal unit `1/U`
    /// (the rectangular generalization of [`try_inverse`](Self::try_inverse), with
    /// which it coincides when the matrix is square and invertible). `eps` is the
    /// singular-value threshold below which a value is treated as zero.
    pub fn pseudo_inverse(
        self,
        eps: T::RealField,
    ) -> Result<UniformUnitMatrix<InvUnit<U>, nalgebra::OMatrix<T, C, R>, Brand>, &'static str>
    where
        U: UnitInv,
    {
        self.inner
            .pseudo_inverse(eps)
            .map(UniformUnitMatrix::from_nalgebra)
    }
}
