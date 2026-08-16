//! Automatic reduction of a block to its lightest representation.
//!
//! When a block is sliced out of a [`MixedUnitMatrix`],
//! its row and column sublists might each turn out to be uniform — in which
//! case the block has a single shared entry unit and is really a
//! [`UniformUnitMatrix`]. [`AutoReduce`] performs that
//! decision entirely at the type level, so block extraction can hand back a
//! `UniformUnitMatrix` exactly when the block is uniform and a `MixedUnitMatrix`
//! otherwise — with no annotation at the call site.
//!
//! The decision is driven by [`IsUniform`], which is total: every list yields
//! a [`typenum::Bit`], so a single impl can branch on "both sublists uniform"
//! and select the output type accordingly. The uniform branch then uses the
//! partial [`CollapseUniform`] to name the shared unit — sound precisely
//! because the flag guarantees uniformity in that branch.

use core::ops::BitAnd;

use typenum::{B0, B1};
use whippyunits::{DivUnit, UnitDiv};

use super::matrix::MixedUnitMatrix;
use super::uniform::UniformUnitMatrix;
use crate::uniformity::{CollapseUniform, IsUniform, Uniform};

/// The `typenum::Bit` that is `B1` iff both dimension lists are uniform.
type BothUniform<RowDims, ColDims> =
    <<RowDims as IsUniform>::Out as BitAnd<<ColDims as IsUniform>::Out>>::Output;

/// Wraps a raw nalgebra block `M` (tagged with row list `RowDims` and column
/// list `ColDims`, brand `Brand`) into the lightest whippyalgebra matrix: a
/// [`UniformUnitMatrix`] when both lists are uniform, else a
/// [`MixedUnitMatrix`].
///
/// Implemented once, for the `(RowDims, ColDims)` pair; it computes the
/// uniformity flag and defers the actual choice to [`ReduceWith`], keyed on that
/// flag. [`Out`](AutoReduce::Out) is the selected matrix type and
/// [`wrap`](AutoReduce::wrap) builds it from the block.
pub trait AutoReduce<M, Brand> {
    /// The selected matrix type (`Uniform…` or `Mixed…`).
    type Out;
    /// Wraps the raw block into the selected type.
    fn wrap(inner: M) -> Self::Out;
}

impl<RowDims, ColDims, M, Brand> AutoReduce<M, Brand> for (RowDims, ColDims)
where
    RowDims: IsUniform,
    ColDims: IsUniform,
    <RowDims as IsUniform>::Out: BitAnd<<ColDims as IsUniform>::Out>,
    (): ReduceWith<BothUniform<RowDims, ColDims>, RowDims, ColDims, M, Brand>,
{
    type Out = <() as ReduceWith<BothUniform<RowDims, ColDims>, RowDims, ColDims, M, Brand>>::Out;

    fn wrap(inner: M) -> Self::Out {
        <() as ReduceWith<BothUniform<RowDims, ColDims>, RowDims, ColDims, M, Brand>>::wrap(inner)
    }
}

/// Flag-keyed selector behind [`AutoReduce`]: the two impls (for the `B1` and
/// `B0` [`typenum`] bits) pick the uniform and mixed representations
/// respectively. Keying on the bit type is what lets a single `AutoReduce` impl
/// choose between two different output types without overlap.
#[doc(hidden)]
pub trait ReduceWith<Flag, RowDims, ColDims, M, Brand> {
    /// The selected matrix type for this flag.
    type Out;
    /// Wraps the raw block into the selected type.
    fn wrap(inner: M) -> Self::Out;
}

// Both lists uniform → collapse to the single shared entry unit `Ru / Cu`.
impl<RowDims, ColDims, M, Brand> ReduceWith<B1, RowDims, ColDims, M, Brand> for ()
where
    RowDims: CollapseUniform,
    ColDims: CollapseUniform,
    Uniform<RowDims>: UnitDiv<Uniform<ColDims>>,
{
    type Out = UniformUnitMatrix<DivUnit<Uniform<RowDims>, Uniform<ColDims>>, M, Brand>;

    fn wrap(inner: M) -> Self::Out {
        UniformUnitMatrix::from_nalgebra(inner)
    }
}

// Otherwise → keep the full per-entry mixed representation.
impl<RowDims, ColDims, M, Brand> ReduceWith<B0, RowDims, ColDims, M, Brand> for () {
    type Out = MixedUnitMatrix<RowDims, ColDims, M, Brand>;

    fn wrap(inner: M) -> Self::Out {
        MixedUnitMatrix::from_nalgebra(inner)
    }
}

/// The reduced matrix type for a block with the given row/column sublists,
/// storage `M`, and brand: [`UniformUnitMatrix`] if both are uniform, else
/// [`MixedUnitMatrix`]. This is the type block extraction reports.
pub type Reduced<RowDims, ColDims, M, Brand> = <(RowDims, ColDims) as AutoReduce<M, Brand>>::Out;
