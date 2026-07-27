#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use crate::dims::Dimensionless;
use crate::nalgebra::uniform::UniformUnitMatrix;

/// The result of a singular value decomposition `M = U Σ Vᵀ` of a uniform
/// matrix, mirroring nalgebra's [`SVD`](nalgebra::SVD) but carrying the entry
/// unit through to the singular values.
///
/// Returned by [`UniformUnitMatrix::svd`]. A uniform matrix carries the *same*
/// unit `U` on both the row and the column side, which is a canonical
/// (uniform) metric — the one convention a *mixed* matrix cannot supply, which
/// is why a genuinely mixed matrix has no bare SVD at all (its singular values
/// would be dimensionless artifacts of the silent identity metric, a footgun);
/// to decompose a mixed matrix you must instead name the metrics explicitly via
/// [`generalized_svd`](crate::nalgebra::MixedUnitMatrix::generalized_svd). With
/// the uniform matrix's canonical metric in hand the singular values come out
/// honestly dimensioned in `U`: `M = U Σ Vᵀ` with the orthonormal `U`, `Vᵀ`
/// dimensionless and `Σ` in `U`, and no even-exponent gate (the root lives in
/// the numbers, never in the unit).
pub struct UniformSVD<U, T, R, C, Brand = ()>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C>,
    C: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, C>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>>,
{
    /// The left singular vectors `U`, dimensionless and orthonormal,
    /// `m × min(m, n)`.
    pub u: UniformUnitMatrix<
        Dimensionless,
        nalgebra::OMatrix<T, R, nalgebra::DimMinimum<R, C>>,
        Brand,
    >,
    /// The singular values `Σ`, a vector in the shared entry unit `U` (sorted
    /// descending), `min(m, n)` of them. Entry `k` pairs with column `k` of
    /// [`u`](Self::u) and row `k` of [`v_t`](Self::v_t).
    pub singular_values:
        UniformUnitMatrix<U, nalgebra::OVector<T::RealField, nalgebra::DimMinimum<R, C>>, Brand>,
    /// The right singular vectors transposed `Vᵀ`, dimensionless and
    /// orthonormal, `min(m, n) × n`.
    pub v_t: UniformUnitMatrix<
        Dimensionless,
        nalgebra::OMatrix<T, nalgebra::DimMinimum<R, C>, C>,
        Brand,
    >,
}

impl<U, T, R, C, Brand> UniformSVD<U, T, R, C, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C>,
    C: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, C>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<R, C>,
{
    /// Reconstructs `M = U · diag(Σ) · Vᵀ`, landing back in the entry unit `U`.
    ///
    /// The two dimensionless orthonormal factors flank the `U`-carrying `diag(Σ)`,
    /// so the product is uniform in `1 · U · 1 = U` — the unit round trip is
    /// enforced by construction. For a rectangular input `diag(Σ)` is the thin
    /// `min(m, n) × min(m, n)` core.
    pub fn recompose(&self) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, R, C>, Brand> {
        let sigma = nalgebra::OMatrix::<T, nalgebra::DimMinimum<R, C>, nalgebra::DimMinimum<R, C>>::from_diagonal(
            &self.singular_values.nalgebra().map(|r| T::from_real(r)),
        );
        let m = self.u.nalgebra() * sigma * self.v_t.nalgebra();
        UniformUnitMatrix::from_nalgebra(m)
    }
}

// A uniform matrix `M = U · M̃` carries one unit `U` on every entry, which — read
// as an operator — is a *uniform* metric on both the row and the column space.
// That single shared convention is exactly the canonical metric a mixed matrix
// lacks, so the singular values (the lengths of `M`'s principal axes in it) come
// out honestly dimensioned in `U`: the numeric SVD `M̃ = Û Σ̃ V̂ᵀ` reads its
// orthonormal factors off `M̃` (dimensionless) and its singular values off the
// `U`-scaled magnitudes. No square root of the unit is ever taken — `Σ` inherits
// `U` whole — so, unlike uniform [`cholesky`](Self::cholesky), there is no
// even-exponent gate.
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
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>>,
{
    /// The singular value decomposition `M = U Σ Vᵀ`, returning a [`UniformSVD`]
    /// with the orthonormal (dimensionless) singular vectors and the singular
    /// values in the shared entry unit `U`.
    ///
    /// A uniform matrix is its own metric on both sides, so — unlike a mixed
    /// matrix, which has no bare SVD and must name its metrics via
    /// [`generalized_svd`](crate::nalgebra::MixedUnitMatrix::generalized_svd) —
    /// the singular values keep the unit `U` (no even-exponent gate — see
    /// [`cholesky`](Self::cholesky) for when a root does require one).
    /// Rectangular inputs use the thin factors of pivot length `min(m, n)`.
    pub fn svd(&self) -> UniformSVD<U, T, R, C, Brand> {
        let svd = self.inner.clone_owned().svd(true, true);
        UniformSVD {
            u: UniformUnitMatrix::from_nalgebra(svd.u.expect("compute_u = true yields U")),
            singular_values: UniformUnitMatrix::from_nalgebra(svd.singular_values),
            v_t: UniformUnitMatrix::from_nalgebra(svd.v_t.expect("compute_v = true yields Vᵀ")),
        }
    }
}
