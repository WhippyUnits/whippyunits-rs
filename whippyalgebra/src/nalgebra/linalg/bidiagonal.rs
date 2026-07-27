#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use crate::dims::Dimensionless;
use crate::nalgebra::uniform::UniformUnitMatrix;

// A bare mixed bidiagonalization is deliberately absent: its two-sided
// equivalence frame `U … Vᵀ` is orthonormal only in the silent identity metric a
// mixed matrix cannot name — the same footgun as a bare mixed SVD, whose
// reduction step this is. Supply metrics explicitly with
// [`generalized_bidiagonalize`](crate::nalgebra::MixedUnitMatrix::generalized_bidiagonalize)
// for the mixed case; a uniform matrix carries its own canonical metric and keeps
// `bidiagonalize`.

/// The result of a bidiagonalization `M = U B Vᵀ` of a uniform matrix.
///
/// Returned by [`UniformUnitMatrix::bidiagonalize`]. The equivalence frame
/// `U … Vᵀ` leaves the two orthonormal factors dimensionless and the
/// bidiagonal `B` carrying `U` — the reduction step inside [uniform SVD](UniformSVD).
pub struct UniformBidiagonal<U, T, R, C, Brand = ()>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C>,
    C: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, C>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, nalgebra::DimMinimum<R, C>>,
{
    /// The left orthonormal factor `U`, dimensionless, `m × min(m, n)`.
    pub u: UniformUnitMatrix<
        Dimensionless,
        nalgebra::OMatrix<T, R, nalgebra::DimMinimum<R, C>>,
        Brand,
    >,
    /// The bidiagonal factor `B`, in the entry unit `U`, `min(m, n) × min(m, n)`.
    pub d: UniformUnitMatrix<
        U,
        nalgebra::OMatrix<T, nalgebra::DimMinimum<R, C>, nalgebra::DimMinimum<R, C>>,
        Brand,
    >,
    /// The right orthonormal factor transposed `Vᵀ`, dimensionless,
    /// `min(m, n) × n`.
    pub v_t: UniformUnitMatrix<
        Dimensionless,
        nalgebra::OMatrix<T, nalgebra::DimMinimum<R, C>, C>,
        Brand,
    >,
}

impl<U, T, R, C, Brand> UniformBidiagonal<U, T, R, C, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C>,
    C: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, C>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<R, C>,
{
    /// Reconstructs `M = U B Vᵀ`, in the entry unit `U` (`1 · U · 1 = U`).
    pub fn recompose(&self) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, R, C>, Brand> {
        let m = self.u.nalgebra() * self.d.nalgebra() * self.v_t.nalgebra();
        UniformUnitMatrix::from_nalgebra(m)
    }
}

// Bidiagonalization `M = U B Vᵀ` is the SVD's two-sided *equivalence* frame, so
// it needs no relation between the row and column spaces and works for *any*
// uniform matrix, rectangular included: `U` is `m × min` and `Vᵀ` is `min × n`
// (both dimensionless), `B` is the `min × min` band in the entry unit `U`.
impl<U, Brand, T, R, C, S> UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C>,
    C: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, R, C>,
    nalgebra::DimMinimum<R, C>: nalgebra::DimSub<nalgebra::U1>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>
        + nalgebra::allocator::Allocator<R>
        + nalgebra::allocator::Allocator<C>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimDiff<nalgebra::DimMinimum<R, C>, nalgebra::U1>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<R, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, C>,
{
    /// The bidiagonalization `M = U B Vᵀ`, with `U`/`Vᵀ` dimensionless and the
    /// bidiagonal `B` in the entry unit `U`. A uniform matrix is its own canonical
    /// metric, so this needs no supplied metric — unlike the mixed
    /// [`generalized_bidiagonalize`](crate::nalgebra::MixedUnitMatrix::generalized_bidiagonalize).
    /// Rectangular inputs use the thin pivot of length `min(m, n)`.
    pub fn bidiagonalize(&self) -> UniformBidiagonal<U, T, R, C, Brand> {
        let (u, d, v_t) = self.inner.clone_owned().bidiagonalize().unpack();
        UniformBidiagonal {
            u: UniformUnitMatrix::from_nalgebra(u),
            d: UniformUnitMatrix::from_nalgebra(d),
            v_t: UniformUnitMatrix::from_nalgebra(v_t),
        }
    }
}
