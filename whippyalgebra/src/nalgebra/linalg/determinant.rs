#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use whippyunits::UnitDiv;
use whippyunits::quantity::Quantity;

use crate::dims::{Product, Producted};
use crate::entry::{DetUnitOf, FromRaw};
use crate::index::{PowUnit, ShapeIndex, UnitPow};
use crate::nalgebra::matrix::MixedUnitMatrix;
use crate::nalgebra::uniform::UniformUnitMatrix;

impl<RowDims, ColDims, Brand, T, D, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, D, D, S>, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, D, D>,
{
    /// Returns the determinant, in `∏ RowDims / ∏ ColDims`.
    ///
    /// Every term of the determinant selects one entry from each row and each
    /// column, so they all share that unit and no homogeneity constraint is
    /// needed (contrast the [`trace`](Self::trace), which sums the diagonal
    /// entries and so requires them to be commensurable).
    pub fn determinant(&self) -> DetUnitOf<RowDims, ColDims, T, Brand>
    where
        RowDims: Product,
        ColDims: Product,
        Producted<RowDims>: UnitDiv<Producted<ColDims>>,
        DetUnitOf<RowDims, ColDims, T, Brand>: FromRaw<T>,
        D: nalgebra::DimMin<D, Output = D>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>
            + nalgebra::allocator::Allocator<nalgebra::DimMinimum<D, D>>,
    {
        <DetUnitOf<RowDims, ColDims, T, Brand> as FromRaw<T>>::from_raw(self.inner.determinant())
    }
}

impl<U, Brand, T, D, S> UniformUnitMatrix<U, nalgebra::Matrix<T, D, D, S>, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::DimMin<D, Output = D> + ShapeIndex,
    S: nalgebra::storage::Storage<T, D, D>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// The determinant of an `n × n` uniform matrix, in `Uⁿ` (`n` the square
    /// dimension). The value is nalgebra's ordinary determinant; the unit is
    /// `U` raised to the entry count of one Leibniz term.
    pub fn determinant(&self) -> Quantity<PowUnit<U, <D as ShapeIndex>::Nat>, T, Brand>
    where
        U: UnitPow<<D as ShapeIndex>::Nat>,
        Quantity<PowUnit<U, <D as ShapeIndex>::Nat>, T, Brand>: FromRaw<T>,
    {
        FromRaw::from_raw(self.inner.determinant())
    }
}
