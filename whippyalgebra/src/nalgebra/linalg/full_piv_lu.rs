#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use core::marker::PhantomData;

use whippyunits::{DivUnit, UnitDiv};

use crate::dims::{Dimensionless, Product, Producted};
use crate::entry::{DetUnitOf, FromRaw};
use crate::nalgebra::matrix::MixedUnitMatrix;
use crate::nalgebra::uniform::UniformUnitMatrix;

impl<RowDims, ColDims, Brand, T, D, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, D, D, S>, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::DimMin<D, Output = D>,
    S: nalgebra::storage::Storage<T, D, D>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// LU with full pivoting (`P·M·Q = L·U`) kept in opaque form — an
    /// [`OpaqueFullPivLU`] — for factor-once-solve-many on any mixed matrix, and the
    /// most numerically robust of the pivoting factorizations.
    ///
    /// Full pivoting permutes both axes at runtime, so a typed variant would need
    /// both margins uniform; this keeps the factors raw, retaining only the
    /// pivot-invariant `⟨RowDims, ColDims⟩` — enough for a unit-safe
    /// [`solve`](OpaqueFullPivLU::solve) /
    /// [`try_inverse`](OpaqueFullPivLU::try_inverse) /
    /// [`determinant`](OpaqueFullPivLU::determinant).
    pub fn full_piv_lu_opaque(&self) -> OpaqueFullPivLU<RowDims, ColDims, T, D, Brand> {
        OpaqueFullPivLU {
            lu: nalgebra::FullPivLU::new(self.inner.clone_owned()),
            _dims: PhantomData,
        }
    }
}

/// A full-pivot LU factorization (`P·M·Q = L·U`) held in opaque form: the
/// triangular factors and both permutations stay as raw nalgebra data, and only
/// the original matrix's `⟨RowDims, ColDims⟩` type — invariant under either pivot
/// — is retained.
///
/// Returned by [`MixedUnitMatrix::full_piv_lu_opaque`]. Full pivoting reorders
/// both axes at runtime, so unit-safe access to the pivoted factors is not
/// possible for a mixed-unit matrix (hence no typed variant); however, the
/// solution of `M x = b`, the inverse, and the determinant do not depend on the
/// pivot order, and remain unit-safe — factor-once-solve-many on any mixed matrix,
/// and the most numerically robust of the pivoting factorizations.
pub struct OpaqueFullPivLU<RowDims, ColDims, T, D, Brand = ()>
where
    T: nalgebra::ComplexField,
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    lu: nalgebra::FullPivLU<T, D, D>,
    _dims: PhantomData<fn() -> (RowDims, ColDims, Brand)>,
}

impl<RowDims, ColDims, T, D, Brand> OpaqueFullPivLU<RowDims, ColDims, T, D, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// Solves `M x = b` reusing this factorization, unit-safe and ungated: `b`
    /// lives in the output space (`RowDims`), the solution `x` in the input space
    /// (`ColDims`), carrying `b`'s own column dims — the same signature as
    /// [`MixedUnitMatrix::solve`](crate::nalgebra::MixedUnitMatrix::solve), but
    /// without re-factorizing on each call and without a uniformity requirement
    /// (both pivot orders the units never saw cancel in the solution). Returns
    /// `None` if `M` is singular.
    pub fn solve<BCols, C2, SB>(
        &self,
        b: &MixedUnitMatrix<RowDims, BCols, nalgebra::Matrix<T, D, C2, SB>, Brand>,
    ) -> Option<MixedUnitMatrix<ColDims, BCols, nalgebra::OMatrix<T, D, C2>, Brand>>
    where
        C2: nalgebra::Dim,
        SB: nalgebra::storage::Storage<T, D, C2>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, C2>,
    {
        self.lu.solve(&b.inner).map(MixedUnitMatrix::from_nalgebra)
    }

    /// The inverse `M⁻¹`, in the swapped `⟨ColDims, RowDims⟩` — the same type as
    /// [`MixedUnitMatrix::try_inverse`](crate::nalgebra::MixedUnitMatrix::try_inverse), reusing
    /// this factorization. Returns `None` if `M` is singular.
    pub fn try_inverse(
        &self,
    ) -> Option<MixedUnitMatrix<ColDims, RowDims, nalgebra::OMatrix<T, D, D>, Brand>> {
        self.lu.try_inverse().map(MixedUnitMatrix::from_nalgebra)
    }

    /// The determinant, in `∏RowDims / ∏ColDims` (the two pivots only flip its
    /// sign), reusing this factorization.
    pub fn determinant(&self) -> DetUnitOf<RowDims, ColDims, T, Brand>
    where
        RowDims: Product,
        ColDims: Product,
        Producted<RowDims>: UnitDiv<Producted<ColDims>>,
        DetUnitOf<RowDims, ColDims, T, Brand>: FromRaw<T>,
    {
        <DetUnitOf<RowDims, ColDims, T, Brand> as FromRaw<T>>::from_raw(self.lu.determinant())
    }

    /// The raw `nalgebra::FullPivLU`, for unit-unsafe access to the factors
    /// (`.l()`, `.u()`, `.p()`, `.q()`, …). The whippy type system attaches no
    /// units to these — both axes are permuted — so any dimensional meaning is
    /// yours to supply.
    pub fn nalgebra(&self) -> &nalgebra::FullPivLU<T, D, D> {
        &self.lu
    }

    /// Consumes the wrapper for the owned raw `nalgebra::FullPivLU` (see
    /// [`nalgebra`](Self::nalgebra)).
    pub fn into_nalgebra(self) -> nalgebra::FullPivLU<T, D, D> {
        self.lu
    }
}

/// The result of an LU decomposition with full pivoting, `P·M·Q = L·U`, of a
/// uniform matrix.
///
/// Returned by [`UniformUnitMatrix::full_piv_lu`]. Full pivoting permutes both
/// rows and columns at runtime; a uniform matrix is uniform on both margins, so
/// both permutations are type-invisible. As with [`UniformLU`], `L` is
/// dimensionless, `U` carries `U`, and the permutations `P`, `Q` are
/// dimensionless.
pub struct UniformFullPivLU<U, T, D, Brand = ()>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
{
    /// The unit-lower-triangular factor `L` (unit diagonal), dimensionless.
    pub l: UniformUnitMatrix<Dimensionless, nalgebra::OMatrix<T, D, D>, Brand>,
    /// The upper-triangular factor `U`, in the entry unit `U`.
    pub u: UniformUnitMatrix<U, nalgebra::OMatrix<T, D, D>, Brand>,
    /// The row-permutation matrix `P` (dimensionless).
    pub p: UniformUnitMatrix<Dimensionless, nalgebra::OMatrix<T, D, D>, Brand>,
    /// The column-permutation matrix `Q` (dimensionless), with `P·M·Q = L·U`.
    pub q: UniformUnitMatrix<Dimensionless, nalgebra::OMatrix<T, D, D>, Brand>,
}

impl<U, T, D, Brand> UniformFullPivLU<U, T, D, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
{
    /// Reconstructs `M = Pᵀ·L·U·Qᵀ` from the factors, in the entry unit `U`.
    pub fn recompose(&self) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, D, D>, Brand> {
        let m = self.p.nalgebra().transpose()
            * self.l.nalgebra()
            * self.u.nalgebra()
            * self.q.nalgebra().transpose();
        UniformUnitMatrix::from_nalgebra(m)
    }

    /// Solves `M x = b` reusing this full-pivot factorization — no
    /// re-factorization.
    ///
    /// From `P·M·Q = L·U`, `x = Q U⁻¹ L⁻¹ P b`: permute by `P`, forward/back
    /// substitute `L`/`U`, then undo the column permutation with `Q`. Same `Ub / U`
    /// unit as [`UniformUnitMatrix::solve`](crate::nalgebra::UniformUnitMatrix::solve).
    /// Returns `None` if `U` is singular.
    pub fn solve<Ub, C2, SB>(
        &self,
        b: &UniformUnitMatrix<Ub, nalgebra::Matrix<T, D, C2, SB>, Brand>,
    ) -> Option<UniformUnitMatrix<DivUnit<Ub, U>, nalgebra::OMatrix<T, D, C2>, Brand>>
    where
        Ub: UnitDiv<U>,
        C2: nalgebra::Dim,
        SB: nalgebra::storage::Storage<T, D, C2>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, C2>,
    {
        let pb = self.p.nalgebra() * b.nalgebra();
        let y = self.l.nalgebra().solve_lower_triangular(&pb)?;
        let z = self.u.nalgebra().solve_upper_triangular(&y)?;
        let x = self.q.nalgebra() * z;
        Some(UniformUnitMatrix::from_nalgebra(x))
    }
}

impl<U, Brand, T, D, S> UniformUnitMatrix<U, nalgebra::Matrix<T, D, D, S>, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::DimMin<D, Output = D>,
    S: nalgebra::storage::Storage<T, D, D>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// LU with full pivoting, `P·M·Q = L·U`, returning a [`UniformFullPivLU`].
    ///
    /// Both row and column pivots are permitted because a uniform matrix is
    /// uniform on both margins; the extra numerical robustness of full pivoting
    /// comes for free unit-wise.
    pub fn full_piv_lu(&self) -> UniformFullPivLU<U, T, D, Brand> {
        let lu = self.inner.clone_owned().full_piv_lu();
        let (nr, nc) = self.inner.shape_generic();
        let mut pm = nalgebra::OMatrix::<T, D, D>::identity_generic(nr, nc);
        let mut qm = nalgebra::OMatrix::<T, D, D>::identity_generic(nr, nc);
        lu.p().permute_rows(&mut pm);
        lu.q().permute_columns(&mut qm);
        UniformFullPivLU {
            l: UniformUnitMatrix::from_nalgebra(lu.l()),
            u: UniformUnitMatrix::from_nalgebra(lu.u()),
            p: UniformUnitMatrix::from_nalgebra(pm),
            q: UniformUnitMatrix::from_nalgebra(qm),
        }
    }
}
