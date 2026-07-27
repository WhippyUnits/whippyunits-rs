#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use whippyunits::{DivUnit, UnitDiv};

use crate::nalgebra::matrix::MixedUnitMatrix;
use crate::nalgebra::uniform::UniformUnitMatrix;

impl<RowDims, ColDims, Brand, T, D, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, D, D, S>, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, D, D>,
{
    /// Solves the linear system `A x = b` for `x`, where `A` is `self`, using an
    /// LU factorization. Returns `None` when `A` is not invertible.
    ///
    /// `b` lives in the output space (row dims `RowDims`, matching `A`) and the
    /// solution `x` in the input space (row dims `ColDims`), carrying `b`'s own
    /// column dims — the same signature as `A.try_inverse() * b`, but without
    /// forming the inverse. The shared `RowDims` on `A` and `b` is what makes the
    /// system well-posed, enforced at the type level.
    pub fn solve<BCols, C2, SB>(
        self,
        b: &MixedUnitMatrix<RowDims, BCols, nalgebra::Matrix<T, D, C2, SB>, Brand>,
    ) -> Option<MixedUnitMatrix<ColDims, BCols, nalgebra::OMatrix<T, D, C2>, Brand>>
    where
        C2: nalgebra::Dim,
        SB: nalgebra::storage::Storage<T, D, C2>,
        D: nalgebra::DimMin<D, Output = D>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>
            + nalgebra::allocator::Allocator<D, C2>
            + nalgebra::allocator::Allocator<nalgebra::DimMinimum<D, D>>,
    {
        self.inner
            .lu()
            .solve(&b.inner)
            .map(MixedUnitMatrix::from_nalgebra)
    }

    /// Solves `A x = b` assuming `A` (`self`) is lower-triangular, by forward
    /// substitution. Returns `None` if a diagonal entry is zero.
    ///
    /// Triangularity is invisible to the units, so the signature matches the
    /// general [`solve`](Self::solve): `b` in the output space (`RowDims`), `x`
    /// in the input space (`ColDims`), carrying `b`'s own column dims. Cheaper
    /// than `solve` (`O(n²)`, no factorization).
    pub fn solve_lower_triangular<BCols, C2, SB>(
        &self,
        b: &MixedUnitMatrix<RowDims, BCols, nalgebra::Matrix<T, D, C2, SB>, Brand>,
    ) -> Option<MixedUnitMatrix<ColDims, BCols, nalgebra::OMatrix<T, D, C2>, Brand>>
    where
        C2: nalgebra::Dim,
        SB: nalgebra::storage::Storage<T, D, C2>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, C2>,
    {
        self.inner
            .solve_lower_triangular(&b.inner)
            .map(MixedUnitMatrix::from_nalgebra)
    }

    /// Solves `A x = b` assuming `A` (`self`) is upper-triangular, by back
    /// substitution. Returns `None` if a diagonal entry is zero. Same signature as
    /// [`solve`](Self::solve); see
    /// [`solve_lower_triangular`](Self::solve_lower_triangular).
    pub fn solve_upper_triangular<BCols, C2, SB>(
        &self,
        b: &MixedUnitMatrix<RowDims, BCols, nalgebra::Matrix<T, D, C2, SB>, Brand>,
    ) -> Option<MixedUnitMatrix<ColDims, BCols, nalgebra::OMatrix<T, D, C2>, Brand>>
    where
        C2: nalgebra::Dim,
        SB: nalgebra::storage::Storage<T, D, C2>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, C2>,
    {
        self.inner
            .solve_upper_triangular(&b.inner)
            .map(MixedUnitMatrix::from_nalgebra)
    }
}

// Linear solves on a square uniform matrix. Solving `M x = b` divides the two
// entry units: `M(U)·x = b(Ub)` forces `x` into `Ub / U` (the uniform analogue
// of the mixed [`solve`](crate::nalgebra::MixedUnitMatrix::solve), whose `RowDims` cancel
// against `b`'s). Both operands may carry any unit; only the quotient `Ub / U`
// need exist, i.e. `Ub: UnitDiv<U>`.
impl<U, Brand, T, D, S> UniformUnitMatrix<U, nalgebra::Matrix<T, D, D, S>, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, D, D>,
{
    /// Solves the linear system `M x = b` for `x` via an LU factorization, in the
    /// quotient unit `Ub / U`. Returns `None` when `M` is singular.
    pub fn solve<Ub, C2, SB>(
        &self,
        b: &UniformUnitMatrix<Ub, nalgebra::Matrix<T, D, C2, SB>, Brand>,
    ) -> Option<UniformUnitMatrix<DivUnit<Ub, U>, nalgebra::OMatrix<T, D, C2>, Brand>>
    where
        Ub: UnitDiv<U>,
        C2: nalgebra::Dim,
        SB: nalgebra::storage::Storage<T, D, C2>,
        D: nalgebra::DimMin<D, Output = D>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>
            + nalgebra::allocator::Allocator<D, C2>
            + nalgebra::allocator::Allocator<nalgebra::DimMinimum<D, D>>,
    {
        self.inner
            .clone_owned()
            .lu()
            .solve(&b.inner)
            .map(UniformUnitMatrix::from_nalgebra)
    }

    /// Solves `M x = b` assuming `M` is lower-triangular, by forward
    /// substitution (`O(n²)`, no factorization). Same `Ub / U` unit as
    /// [`solve`](Self::solve). Returns `None` if a diagonal entry is zero.
    pub fn solve_lower_triangular<Ub, C2, SB>(
        &self,
        b: &UniformUnitMatrix<Ub, nalgebra::Matrix<T, D, C2, SB>, Brand>,
    ) -> Option<UniformUnitMatrix<DivUnit<Ub, U>, nalgebra::OMatrix<T, D, C2>, Brand>>
    where
        Ub: UnitDiv<U>,
        C2: nalgebra::Dim,
        SB: nalgebra::storage::Storage<T, D, C2>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, C2>,
    {
        self.inner
            .solve_lower_triangular(&b.inner)
            .map(UniformUnitMatrix::from_nalgebra)
    }

    /// Solves `M x = b` assuming `M` is upper-triangular, by back substitution.
    /// Same `Ub / U` unit as [`solve`](Self::solve). Returns `None` if a diagonal
    /// entry is zero.
    pub fn solve_upper_triangular<Ub, C2, SB>(
        &self,
        b: &UniformUnitMatrix<Ub, nalgebra::Matrix<T, D, C2, SB>, Brand>,
    ) -> Option<UniformUnitMatrix<DivUnit<Ub, U>, nalgebra::OMatrix<T, D, C2>, Brand>>
    where
        Ub: UnitDiv<U>,
        C2: nalgebra::Dim,
        SB: nalgebra::storage::Storage<T, D, C2>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, C2>,
    {
        self.inner
            .solve_upper_triangular(&b.inner)
            .map(UniformUnitMatrix::from_nalgebra)
    }
}
