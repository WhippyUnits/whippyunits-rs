#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use crate::dims::{Dimensionless, MapUnits, Mapped, PivotDims, Reciprocal, ZipToDimensionless};
use crate::nalgebra::matrix::MixedUnitMatrix;
use crate::nalgebra::uniform::UniformUnitMatrix;

/// The result of a weighted (generalized) column-pivoted QR decomposition
/// `M P = Q R` of a mixed matrix `M : ⟨RowDims, ColDims⟩` against a codomain metric
/// `Gr : ⟨1/RowDims, RowDims⟩` and a domain metric `Gc : ⟨1/ColDims, ColDims⟩`
/// (`Gr`, `Gc`: positive-definite weight (metric/Gram) matrices on the row/column
/// space — see the [module overview](crate::nalgebra::linalg)).
/// Returned by [`MixedUnitMatrix::generalized_col_piv_qr`]. `Q` is metric-orthonormal
/// (`Qᴴ Gr Q = I`), `R` is upper-triangular, and `P` is the rank-revealing column
/// permutation; `Q` and `R` share the thin dimensionless pivot of length `min(m, n)`
/// (see [`PivotDims`]), rectangular included.
///
/// If `Gr` and `Gc` are the identity metrics, this is the same as the
/// [`UniformUnitMatrix::col_piv_qr`](crate::nalgebra::UniformUnitMatrix::col_piv_qr),
/// whose single shared unit is a canonical metric on both spaces. Demands a metric
/// because a bare mixed column-pivoted QR would silently pick the identity
/// metric, which is not invariant under rescaling in the mixed-unit case — and here
/// that metric also orders the pivot, so the revealed rank would be a unit-dependent
/// artifact.
pub struct GeneralizedColPivQR<RowDims, ColDims, T, R, C, Brand = ()>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C>,
    C: nalgebra::Dim,
    RowDims: ZipToDimensionless<ColDims>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, C>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, R>
        + nalgebra::allocator::Allocator<C, C>,
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
    /// The upper-triangular factor `R`, fully dimensionless (`min(m, n) × n`): it
    /// acts on the whitened, pivot-ordered column space, so it carries no unit —
    /// the pivot `P` and the domain metric `Gc` reattach the column units. Its
    /// diagonal (descending in magnitude) is what the pivot reveals the rank from.
    pub r: UniformUnitMatrix<
        Dimensionless,
        nalgebra::OMatrix<T, nalgebra::DimMinimum<R, C>, C>,
        Brand,
    >,
    /// The column-permutation matrix `P` (dimensionless, `n × n`), with
    /// `M P = Q R Pᵀ`-consistent ordering: its leading columns select the
    /// numerically dominant directions of the `Gc`-whitened column space.
    pub p: UniformUnitMatrix<Dimensionless, nalgebra::OMatrix<T, C, C>, Brand>,
    // Projector `Q̂ᴴ Lrᴴ = Qᴴ Gr` (`min × m`): maps a right-hand side in the
    // codomain onto the dimensionless pivot axis, reused by `solve`.
    b_project: nalgebra::OMatrix<T, nalgebra::DimMinimum<R, C>, R>,
    // Domain de-whitening `Lc⁻ᴴ` (`n × n`): lifts a pivot-space solution back into
    // `ColDims`. Reused by `solve`.
    lc_inv_adj: nalgebra::OMatrix<T, C, C>,
    // Domain whitening `Lcᴴ` (`n × n`): reused by `recompose` to reattach the
    // column units.
    lc_adj: nalgebra::OMatrix<T, C, C>,
}

impl<RowDims, ColDims, T, R, C, Brand> GeneralizedColPivQR<RowDims, ColDims, T, R, C, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C>,
    C: nalgebra::Dim,
    RowDims: ZipToDimensionless<ColDims>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, C>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, R>
        + nalgebra::allocator::Allocator<C, C>
        + nalgebra::allocator::Allocator<R, C>,
{
    /// Reconstructs `M = Q R Pᵀ Lcᴴ` from the factors, landing back in
    /// `⟨RowDims, ColDims⟩`.
    ///
    /// Un-pivoting (`Pᵀ`) and re-attaching the domain units (`Lcᴴ`, the stored
    /// Cholesky factor of `Gc`) rebuild `M`'s exact type: `Q : ⟨RowDims, [1…1]⟩`
    /// carries the rows, `R Pᵀ` is dimensionless, and `Lcᴴ : ⟨[1…1], ColDims⟩`
    /// restores the columns.
    pub fn recompose(
        &self,
    ) -> MixedUnitMatrix<RowDims, ColDims, nalgebra::OMatrix<T, R, C>, Brand> {
        let m =
            self.q.nalgebra() * self.r.nalgebra() * self.p.nalgebra().transpose() * &self.lc_adj;
        MixedUnitMatrix::from_nalgebra(m)
    }
}

// The held-factorization reuse back-substitutes against `R` (the thin `min × n`
// factor), which is triangular iff `min = n` — i.e. `m ≥ n`, the tall/square
// (over- or exactly-determined) range. A wide `m < n` cannot reuse it; use
// [`generalized_pseudo_inverse`](crate::nalgebra::MixedUnitMatrix::generalized_pseudo_inverse).
impl<RowDims, ColDims, T, R, C, Brand> GeneralizedColPivQR<RowDims, ColDims, T, R, C, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C, Output = C>,
    C: nalgebra::Dim,
    RowDims: ZipToDimensionless<ColDims>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, C>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, R>
        + nalgebra::allocator::Allocator<C, C>,
{
    /// Solves `M x = b` reusing this factorization, in the metric least-squares
    /// sense: `x` minimizes the `Gr`-norm of the residual `M x − b`, with the
    /// minimum-`Gc`-norm tie-break — the genuinely metric-relative counterpart of
    /// the opaque solve it replaces. `b` lives in the codomain (`RowDims`), the
    /// solution `x` in the domain (`ColDims`), carrying `b`'s own column dims — the
    /// same signature as [`MixedUnitMatrix::solve`](crate::nalgebra::MixedUnitMatrix::solve).
    ///
    /// Available for tall/square `M` (`m ≥ n`); for a square full-rank `M` it is
    /// the exact solve (metric-independent), and for a tall one the `Gr`-weighted
    /// least-squares fit. Returns `None` if `R` is singular (rank-deficient).
    pub fn solve<BCols, C2, SB>(
        &self,
        b: &MixedUnitMatrix<RowDims, BCols, nalgebra::Matrix<T, R, C2, SB>, Brand>,
    ) -> Option<MixedUnitMatrix<ColDims, BCols, nalgebra::OMatrix<T, C, C2>, Brand>>
    where
        C2: nalgebra::Dim,
        SB: nalgebra::storage::Storage<T, R, C2>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<C, C2>
            + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, C2>,
    {
        // Project b onto the dimensionless pivot axis, back-substitute against the
        // triangular R, un-pivot, then lift back into ColDims by de-whitening.
        let z = &self.b_project * b.nalgebra();
        let w = self.r.nalgebra().solve_upper_triangular(&z)?;
        let x_hat = self.p.nalgebra() * w;
        let x = &self.lc_inv_adj * x_hat;
        Some(MixedUnitMatrix::from_nalgebra(x))
    }
}

// The generalized column-pivoted QR whitens `M` by the Cholesky factors of the two
// supplied metrics — `Gr = Lr Lrᴴ`, `Gc = Lc Lcᴴ` — forming `M̂ = Lrᴴ M Lc⁻ᴴ`,
// which is *fully dimensionless*. Only then are the column norms commensurable, so
// nalgebra's column pivot `M̂ P = Q̂ R` is a genuine rank revelation rather than an
// identity-metric artifact. The orthogonal factor maps back to the
// metric-orthonormal basis `Q = Lr⁻ᴴ Q̂ : ⟨R, [1]⟩`, `R` and `P` stay dimensionless,
// and the domain whitening `Lc` is retained to reattach column units in `solve` /
// `recompose`. No unit is ever rooted, so — like the mixed metric decompositions —
// it is ungated. The whitening is two-sided but shape-preserving, so any `m × n`
// works, rectangular included.
impl<RowDims, ColDims, Brand, T, R, C, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C> + nalgebra::DimMin<R, Output = R>,
    C: nalgebra::DimMin<C, Output = C>,
    S: nalgebra::storage::Storage<T, R, C>,
    nalgebra::DimMinimum<R, C>: nalgebra::DimMin<C, Output = nalgebra::DimMinimum<R, C>>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>
        + nalgebra::allocator::Allocator<R>
        + nalgebra::allocator::Allocator<C>
        + nalgebra::allocator::Allocator<R, R>
        + nalgebra::allocator::Allocator<C, C>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<R, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, C>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, R>
        + nalgebra::allocator::Reallocator<T, R, C, nalgebra::DimMinimum<R, C>, C>,
{
    /// The weighted (generalized) column-pivoted QR `M P = Q R`, ranking
    /// and orthonormalizing the columns against the codomain metric
    /// `g_r : ⟨1/RowDims, RowDims⟩` and the domain metric `g_c : ⟨1/ColDims, ColDims⟩`
    /// (`Gr`, `Gc`: positive-definite weight (metric/Gram) matrices on the row/column
    /// space — see the [module overview](crate::nalgebra::linalg)).
    /// `Q` is `Gr`-orthonormal (`Qᴴ Gr Q = I`), `R` is upper-triangular, and `P` is
    /// the rank-revealing column permutation; it powers a metric least-squares
    /// [`solve`](GeneralizedColPivQR::solve). Returns a [`GeneralizedColPivQR`], or
    /// `None` when either metric is not positive-definite.
    ///
    /// If `Gr` and `Gc` are the identity metrics, this is the same as the
    /// [`UniformUnitMatrix::col_piv_qr`](crate::nalgebra::UniformUnitMatrix::col_piv_qr),
    /// whose single shared unit is a canonical metric on both spaces. Demands a
    /// metric because a bare mixed column-pivoted QR would silently pick the
    /// identity metric, which is not invariant under rescaling in the mixed-unit
    /// case — and the metric also orders the pivot, so the revealed rank would
    /// be a unit-dependent artifact.
    pub fn generalized_col_piv_qr<Sr, Sc>(
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
    ) -> Option<GeneralizedColPivQR<RowDims, ColDims, T, R, C, Brand>>
    where
        RowDims: MapUnits<Reciprocal> + ZipToDimensionless<ColDims>,
        ColDims: MapUnits<Reciprocal>,
        Sr: nalgebra::storage::Storage<T, R, R>,
        Sc: nalgebra::storage::Storage<T, C, C>,
    {
        // Whiten both sides so the pivot compares commensurable columns:
        // M̂ = Lrᴴ M Lc⁻ᴴ is dimensionless.
        let lr = g_r.inner.clone_owned().cholesky()?.l();
        let lc = g_c.inner.clone_owned().cholesky()?.l();
        let lr_adj = lr.adjoint();
        let lr_inv_adj = lr.try_inverse()?.adjoint();
        let lc_adj = lc.adjoint();
        let lc_inv_adj = lc.try_inverse()?.adjoint();
        let m_hat = &lr_adj * self.inner.clone_owned() * &lc_inv_adj;

        let (q_hat, r, p) = m_hat.col_piv_qr().unpack();

        // The n × n permutation matrix, so P is exposed as typed data.
        let (_, nc) = self.inner.shape_generic();
        let mut pm = nalgebra::OMatrix::<T, C, C>::identity_generic(nc, nc);
        p.permute_columns(&mut pm);

        // Map the whitened Q back to the metric-orthonormal basis; precompute the
        // right-hand-side projector for `solve`.
        let b_project = q_hat.adjoint() * &lr_adj;
        let q = &lr_inv_adj * q_hat;
        Some(GeneralizedColPivQR {
            q: MixedUnitMatrix::from_nalgebra(q),
            r: UniformUnitMatrix::from_nalgebra(r),
            p: UniformUnitMatrix::from_nalgebra(pm),
            b_project,
            lc_inv_adj,
            lc_adj,
        })
    }
}
