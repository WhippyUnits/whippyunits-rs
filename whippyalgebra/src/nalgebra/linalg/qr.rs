#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use whippyunits::{DivUnit, InvUnit, UnitDiv, UnitInv};

use crate::dims::Dimensionless;
use crate::nalgebra::uniform::UniformUnitMatrix;

// A bare mixed QR is deliberately absent: `M = Q R` orthonormalizes the columns
// of `M`, and on a genuinely mixed matrix that orthonormality is only meaningful
// in the silent identity metric — the same footgun as a bare mixed SVD. Supply a
// codomain metric explicitly with
// [`generalized_qr`](crate::nalgebra::MixedUnitMatrix::generalized_qr) for the
// mixed case; a uniform matrix carries its own canonical metric and keeps `qr`.

// QR reconstructs the entry unit through a dimensionless pivot, so it is sound
// for *any* uniform matrix, rectangular included: `Q` is `m × min(m, n)`
// dimensionless, `R` is `min(m, n) × n` in the entry unit `U`.
impl<U, Brand, T, R, C, S> UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C>,
    C: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, R, C>,
    nalgebra::DimMinimum<R, C>: nalgebra::DimMin<C, Output = nalgebra::DimMinimum<R, C>>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>
        + nalgebra::allocator::Allocator<R>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<R, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, C>
        + nalgebra::allocator::Reallocator<T, R, C, nalgebra::DimMinimum<R, C>, C>,
{
    /// The QR decomposition `M = Q R`, returning a [`UniformQR`].
    ///
    /// Always available on a uniform matrix and ungated (no pivot): the
    /// orthogonal `Q` is dimensionless and the upper-triangular `R` carries the
    /// entry unit `U`, so `Q·R` reconstructs `M`'s unit. The unpivoted counterpart
    /// of [`col_piv_qr`](Self::col_piv_qr). Rectangular inputs use the thin pivot
    /// of length `min(m, n)`.
    pub fn qr(&self) -> UniformQR<U, T, R, C, Brand> {
        let (q, r) = self.inner.clone_owned().qr().unpack();
        UniformQR {
            q: UniformUnitMatrix::from_nalgebra(q),
            r: UniformUnitMatrix::from_nalgebra(r),
        }
    }
}

/// The QR decomposition `M = Q R` of a uniform matrix, mirroring nalgebra's
/// [`QR`](nalgebra::QR).
///
/// Returned by [`UniformUnitMatrix::qr`]. Unpivoted and ungated: the
/// orthogonal `Q` is dimensionless (a raw-coordinate orthonormal basis) and the
/// upper-triangular `R` carries the entry unit `U`, so `Q·R` reconstructs `M`.
/// The column-pivoted variant is [`UniformColPivQR`].
pub struct UniformQR<U, T, R, C, Brand = ()>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C>,
    C: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, C>,
{
    /// The orthogonal factor `Q`, dimensionless and orthonormal,
    /// `m × min(m, n)`.
    pub q: UniformUnitMatrix<
        Dimensionless,
        nalgebra::OMatrix<T, R, nalgebra::DimMinimum<R, C>>,
        Brand,
    >,
    /// The upper-triangular factor `R`, in the entry unit `U`, `min(m, n) × n`.
    pub r: UniformUnitMatrix<U, nalgebra::OMatrix<T, nalgebra::DimMinimum<R, C>, C>, Brand>,
}

impl<U, T, R, C, Brand> UniformQR<U, T, R, C, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C>,
    C: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, nalgebra::DimMinimum<R, C>>
        + nalgebra::allocator::Allocator<nalgebra::DimMinimum<R, C>, C>
        + nalgebra::allocator::Allocator<R, C>,
{
    /// Reconstructs `M = Q·R` from the factors, in the entry unit `U`.
    pub fn recompose(&self) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, R, C>, Brand> {
        let m = self.q.nalgebra() * self.r.nalgebra();
        UniformUnitMatrix::from_nalgebra(m)
    }
}

// The held-factorization reuse back-substitutes against `R` (the thin `min × n`
// factor), which is triangular iff `min = n` — i.e. `m ≥ n`, the tall/square
// (over- or exactly-determined) range. Then `R` is `n × n`, and for full column
// rank `R⁻¹ Qᴴ` is the Moore–Penrose left inverse, so `solve` is the
// least-squares solve and `try_inverse` the pseudo-inverse (the true ones when
// square). A wide `m < n` cannot reuse this factorization — use
// [`pseudo_inverse`](UniformUnitMatrix::pseudo_inverse).
impl<U, T, R, C, Brand> UniformQR<U, T, R, C, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::DimMin<C, Output = C>,
    C: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>
        + nalgebra::allocator::Allocator<C, R>
        + nalgebra::allocator::Allocator<C, C>,
{
    /// Solves `M x = b` reusing this factorization (`x = R⁻¹ Qᴴ b`), in the
    /// quotient unit `Ub / U` — the same signature as
    /// [`UniformUnitMatrix::solve`](crate::nalgebra::UniformUnitMatrix::solve), but without
    /// re-factorizing.
    ///
    /// For a tall `M` (`m > n`) this is the least-squares solve; for a square
    /// `M`, the exact one. Returns `None` if `R` is singular (rank-deficient).
    pub fn solve<Ub, C2, SB>(
        &self,
        b: &UniformUnitMatrix<Ub, nalgebra::Matrix<T, R, C2, SB>, Brand>,
    ) -> Option<UniformUnitMatrix<DivUnit<Ub, U>, nalgebra::OMatrix<T, C, C2>, Brand>>
    where
        Ub: UnitDiv<U>,
        C2: nalgebra::Dim,
        SB: nalgebra::storage::Storage<T, R, C2>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<C, C2>,
    {
        let qt_b = self.q.nalgebra().adjoint() * b.nalgebra();
        let x = self.r.nalgebra().solve_upper_triangular(&qt_b)?;
        Some(UniformUnitMatrix::from_nalgebra(x))
    }

    /// The inverse `M⁻¹` (`= R⁻¹ Qᴴ`), uniform with the reciprocal unit `1 / U`
    /// — the same type as [`UniformUnitMatrix::try_inverse`](crate::nalgebra::UniformUnitMatrix::try_inverse),
    /// reusing this factorization.
    ///
    /// For a tall `M` (`m > n`) this is the left (Moore–Penrose) inverse `M⁺`
    /// with `M⁺ M = I`; for a square `M`, the true inverse. Returns `None` if `R`
    /// is singular.
    pub fn try_inverse(
        &self,
    ) -> Option<UniformUnitMatrix<InvUnit<U>, nalgebra::OMatrix<T, C, R>, Brand>>
    where
        U: UnitInv,
    {
        let qh = self.q.nalgebra().adjoint();
        self.r
            .nalgebra()
            .solve_upper_triangular(&qh)
            .map(UniformUnitMatrix::from_nalgebra)
    }
}
