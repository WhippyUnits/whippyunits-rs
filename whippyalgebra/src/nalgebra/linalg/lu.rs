#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use core::marker::PhantomData;

use whippyunits::{DivUnit, UnitDiv};

use crate::dims::{Dimensionless, Product, Producted};
use crate::entry::{DetUnitOf, FromRaw};
use crate::nalgebra::matrix::MixedUnitMatrix;
use crate::nalgebra::uniform::UniformUnitMatrix;
use crate::uniformity::CollapseUniform;

/// The result of an LU decomposition with partial (row) pivoting, `P·M = L·U`,
/// of a mixed matrix whose row space is uniform.
///
/// Returned by [`MixedUnitMatrix::lu`]. Uniform rows (`RowDims = [r … r]`) are
/// what keep the runtime pivot type-invisible and its down-column magnitude
/// comparison commensurable. `L` and `P` are dimensionless (`r/r`); `U` keeps
/// `M`'s full `⟨RowDims, ColDims⟩`, its diagonal pivots `r / ColDims[i]`.
pub struct LU<RowDims, ColDims, T, D, Brand = ()>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
{
    /// The row-permutation matrix `P` (dimensionless, `⟨RowDims, RowDims⟩`), with
    /// `P·M = L·U`.
    pub p: MixedUnitMatrix<RowDims, RowDims, nalgebra::OMatrix<T, D, D>, Brand>,
    /// The unit-lower-triangular factor `L` (unit diagonal), dimensionless in
    /// `⟨RowDims, RowDims⟩`.
    pub l: MixedUnitMatrix<RowDims, RowDims, nalgebra::OMatrix<T, D, D>, Brand>,
    /// The upper-triangular factor `U`, in `M`'s `⟨RowDims, ColDims⟩`; its
    /// diagonal holds the pivots.
    pub u: MixedUnitMatrix<RowDims, ColDims, nalgebra::OMatrix<T, D, D>, Brand>,
}

impl<RowDims, ColDims, T, D, Brand> LU<RowDims, ColDims, T, D, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
{
    /// Reconstructs `M = Pᵀ·L·U` from the factors, landing back in
    /// `⟨RowDims, ColDims⟩` (`Pᵀ` inverts the row permutation; the dimensionless
    /// `Pᵀ L` leaves `U` carrying the units).
    pub fn recompose(
        &self,
    ) -> MixedUnitMatrix<RowDims, ColDims, nalgebra::OMatrix<T, D, D>, Brand> {
        let m = self.p.nalgebra().transpose() * self.l.nalgebra() * self.u.nalgebra();
        MixedUnitMatrix::from_nalgebra(m)
    }

    /// Solves `M x = b` reusing this factorization: from `P·M = L·U`,
    /// `x = U⁻¹ L⁻¹ P b` in two `O(n²)` triangular solves on the stored factors, so
    /// many right-hand sides cost one `O(n³)` [`lu`](MixedUnitMatrix::lu) plus
    /// `O(n²)` each, versus re-factorizing inside the one-shot
    /// [`MixedUnitMatrix::solve`](crate::nalgebra::MixedUnitMatrix::solve). 
    /// Returns `None` if `U` is singular.
    pub fn solve<BCols, C2, SB>(
        &self,
        b: &MixedUnitMatrix<RowDims, BCols, nalgebra::Matrix<T, D, C2, SB>, Brand>,
    ) -> Option<MixedUnitMatrix<ColDims, BCols, nalgebra::OMatrix<T, D, C2>, Brand>>
    where
        C2: nalgebra::Dim,
        SB: nalgebra::storage::Storage<T, D, C2>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, C2>,
    {
        let pb = self.p.nalgebra() * b.nalgebra();
        let y = self.l.nalgebra().solve_lower_triangular(&pb)?;
        let x = self.u.nalgebra().solve_upper_triangular(&y)?;
        Some(MixedUnitMatrix::from_nalgebra(x))
    }
}

impl<RowDims, ColDims, Brand, T, D, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, D, D, S>, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::DimMin<D, Output = D>,
    S: nalgebra::storage::Storage<T, D, D>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// LU with partial (row) pivoting, `P·M = L·U`, returning an [`LU`].
    ///
    /// Requires the row space to be uniform (`RowDims: CollapseUniform`): only
    /// then is the runtime row permutation type-invisible and the pivot's
    /// down-column magnitude comparison commensurable (see [`LU`]). Columns may
    /// carry any mixed units. For the pivot-agnostic linear solve, which needs no
    /// uniformity, use [`solve`](Self::solve); for a fully-uniform matrix, LU is
    /// unconditional on [`UniformUnitMatrix::lu`](crate::nalgebra::UniformUnitMatrix::lu).
    pub fn lu(&self) -> LU<RowDims, ColDims, T, D, Brand>
    where
        RowDims: CollapseUniform,
    {
        let lu = self.inner.clone_owned().lu();
        let (nr, nc) = self.inner.shape_generic();
        let mut pm = nalgebra::OMatrix::<T, D, D>::identity_generic(nr, nc);
        lu.p().permute_rows(&mut pm);
        LU {
            p: MixedUnitMatrix::from_nalgebra(pm),
            l: MixedUnitMatrix::from_nalgebra(lu.l()),
            u: MixedUnitMatrix::from_nalgebra(lu.u()),
        }
    }

    /// LU with partial pivoting kept in opaque form — an [`OpaqueLU`] — for
    /// factor-once-solve-many on any mixed matrix, with no uniform-rows gate.
    ///
    /// Unlike [`lu`](Self::lu), the pivot and triangular factors stay raw (no typed
    /// `P`/`L`/`U`), keeping only the pivot-invariant `⟨RowDims, ColDims⟩` — enough
    /// for a unit-safe [`solve`](OpaqueLU::solve) /
    /// [`try_inverse`](OpaqueLU::try_inverse) /
    /// [`determinant`](OpaqueLU::determinant). Use it when your rows are not uniform.
    pub fn lu_opaque(&self) -> OpaqueLU<RowDims, ColDims, T, D, Brand> {
        OpaqueLU {
            lu: self.inner.clone_owned().lu(),
            _dims: PhantomData,
        }
    }
}

/// An LU factorization (partial pivoting, `P·M = L·U`) held in opaque form: the
/// pivot and triangular factors stay as raw nalgebra data, and only the original
/// matrix's `⟨RowDims, ColDims⟩` type — invariant under pivoting — is retained.
///
/// Returned by [`MixedUnitMatrix::lu_opaque`]. Pivoting reorders rows at runtime,
/// so unit-safe access to the pivoted factors is not possible for a mixed-unit matrix;
/// however, the solution of `M x = b` does not depend on the pivot order, and remains unit-safe.
pub struct OpaqueLU<RowDims, ColDims, T, D, Brand = ()>
where
    T: nalgebra::ComplexField,
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    lu: nalgebra::LU<T, D, D>,
    _dims: PhantomData<fn() -> (RowDims, ColDims, Brand)>,
}

impl<RowDims, ColDims, T, D, Brand> OpaqueLU<RowDims, ColDims, T, D, Brand>
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
    /// (the pivot order the units never saw cancels in the solution). Returns
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

    /// The determinant, in `∏RowDims / ∏ColDims` (the pivot only flips its sign),
    /// reusing this factorization.
    pub fn determinant(&self) -> DetUnitOf<RowDims, ColDims, T, Brand>
    where
        RowDims: Product,
        ColDims: Product,
        Producted<RowDims>: UnitDiv<Producted<ColDims>>,
        DetUnitOf<RowDims, ColDims, T, Brand>: FromRaw<T>,
    {
        <DetUnitOf<RowDims, ColDims, T, Brand> as FromRaw<T>>::from_raw(self.lu.determinant())
    }

    /// The raw `nalgebra::LU`, for unit-unsafe access to the pivoted factors
    /// (`.l()`, `.u()`, `.p()`, …). The whippy type system attaches no units to
    /// these — the row axis is permuted — so any dimensional meaning is yours to
    /// supply.
    pub fn nalgebra(&self) -> &nalgebra::LU<T, D, D> {
        &self.lu
    }

    /// Consumes the wrapper for the owned raw `nalgebra::LU` (see
    /// [`nalgebra`](Self::nalgebra)).
    pub fn into_nalgebra(self) -> nalgebra::LU<T, D, D> {
        self.lu
    }
}

/// The result of an LU decomposition with partial (row) pivoting, `P·M = L·U`,
/// of a uniform matrix.
///
/// Returned by [`UniformUnitMatrix::lu`]. Uniform on both margins, so the runtime
/// row pivot is type-invisible and compares commensurable magnitudes down each
/// column (every entry shares `U`). The factors split the one unit cleanly: `L`
/// and `P` are dimensionless, `U` carries `U` with the pivots on its diagonal.
pub struct UniformLU<U, T, D, Brand = ()>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
{
    /// The unit-lower-triangular factor `L` (unit diagonal), dimensionless.
    pub l: UniformUnitMatrix<Dimensionless, nalgebra::OMatrix<T, D, D>, Brand>,
    /// The upper-triangular factor `U`, in the entry unit `U`; its diagonal holds
    /// the pivots.
    pub u: UniformUnitMatrix<U, nalgebra::OMatrix<T, D, D>, Brand>,
    /// The row-permutation matrix `P` (dimensionless), with `P·M = L·U`.
    pub p: UniformUnitMatrix<Dimensionless, nalgebra::OMatrix<T, D, D>, Brand>,
}

impl<U, T, D, Brand> UniformLU<U, T, D, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
{
    /// Reconstructs `M = Pᵀ·L·U` from the factors, landing back in the entry unit
    /// `U` (`Pᵀ` inverts the row permutation; the dimensionless `Pᵀ L` leaves `U`
    /// carrying the unit).
    pub fn recompose(&self) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, D, D>, Brand> {
        let m = self.p.nalgebra().transpose() * self.l.nalgebra() * self.u.nalgebra();
        UniformUnitMatrix::from_nalgebra(m)
    }

    /// Solves `M x = b` reusing this factorization — no re-factorization.
    ///
    /// From `P·M = L·U`, `x = U⁻¹ L⁻¹ P b`: permute by `P`, forward-substitute the
    /// unit-lower `L`, back-substitute the upper `U`, each `O(n²)` on the stored
    /// factors. The unit divides as in [`UniformUnitMatrix::solve`](crate::nalgebra::UniformUnitMatrix::solve):
    /// `b`'s `Ub` over the matrix's `U`. Returns `None` if `U` is singular.
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
        let x = self.u.nalgebra().solve_upper_triangular(&y)?;
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
    /// LU with partial (row) pivoting, `P·M = L·U`, returning a [`UniformLU`].
    ///
    /// Always available on a uniform matrix — both margins are uniform, so the
    /// runtime row pivot is both type-safe and a comparison of commensurable
    /// magnitudes. The mixed counterpart
    /// [`MixedUnitMatrix::lu`](crate::nalgebra::MixedUnitMatrix::lu) needs an explicit
    /// uniform-rows gate for the same reason.
    pub fn lu(&self) -> UniformLU<U, T, D, Brand> {
        let lu = self.inner.clone_owned().lu();
        let (nr, nc) = self.inner.shape_generic();
        let mut pm = nalgebra::OMatrix::<T, D, D>::identity_generic(nr, nc);
        lu.p().permute_rows(&mut pm);
        UniformLU {
            l: UniformUnitMatrix::from_nalgebra(lu.l()),
            u: UniformUnitMatrix::from_nalgebra(lu.u()),
            p: UniformUnitMatrix::from_nalgebra(pm),
        }
    }
}
