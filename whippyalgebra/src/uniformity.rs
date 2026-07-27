//! Detecting when a dimension list is uniform — every entry the same unit.
//!
//! A [`MixedUnitMatrix`](crate::nalgebra::MixedUnitMatrix) whose row list and column list
//! are each uniform has a single shared entry unit, so it can be represented as
//! the lighter [`UniformUnitMatrix`](crate::nalgebra::UniformUnitMatrix). This module
//! provides the two type-level predicates that drive that reduction:
//!
//! - [`IsUniform`] — a total predicate: every list maps to a [`typenum::Bit`]
//!   (`B1` iff all entries are equal). Being total, it can select between the
//!   uniform and mixed representations for any list, which is what lets block
//!   extraction hand back a `UniformUnitMatrix` automatically when — and only
//!   when — the block is uniform.
//! - [`CollapseUniform`] — a partial projection: it yields the shared
//!   [`Unit`](whippyunits::unit::Unit) of a uniform list (and simply has no impl
//!   for a non-uniform one). This is the extractor used once uniformity is
//!   known.
//!
//! Both rest on whippyunits' type-level unit equality
//! ([`UnitEq`]): `IsUniform` AND-folds the adjacent-pair
//! equalities (all adjacent equal ⟺ all equal), while `CollapseUniform` uses
//! ordinary associated-type unification to require each tail entry to be the
//! head.

use core::ops::BitAnd;

use typenum::B1;
use whippyunits::UnitEq;

use crate::dims::{DCons, DNil};

/// A total predicate: [`Out`](IsUniform::Out) is a [`typenum::Bit`], `B1` iff
/// every entry of the list is the same unit.
///
/// The empty and singleton lists are vacuously uniform. A longer list is
/// uniform iff its first two entries are equal and its tail is uniform;
/// AND-folding these adjacent equalities is equivalent to "all entries equal"
/// by transitivity, and yields a single `Bit` verdict.
pub trait IsUniform {
    /// The [`typenum::Bit`] `B1` (uniform) or `B0` (mixed).
    type Out;
}

impl IsUniform for DNil {
    type Out = B1;
}

impl<H> IsUniform for DCons<H, DNil> {
    type Out = B1;
}

impl<H, H2, T> IsUniform for DCons<H, DCons<H2, T>>
where
    H: UnitEq<H2>,
    DCons<H2, T>: IsUniform,
    <H as UnitEq<H2>>::Output: BitAnd<<DCons<H2, T> as IsUniform>::Out>,
{
    type Out = <<H as UnitEq<H2>>::Output as BitAnd<<DCons<H2, T> as IsUniform>::Out>>::Output;
}

/// The [`typenum::Bit`] verdict of whether the dimension list `L` is uniform.
pub type IsUniformBit<L> = <L as IsUniform>::Out;

/// A partial projection: [`Unit`](CollapseUniform::Unit) is the single unit
/// shared by every entry of a uniform list.
///
/// It is deliberately implemented only for uniform lists — a non-uniform list
/// has no impl, so requesting its shared unit is a compile error. Uniformity is
/// enforced by unification: each recursive step requires the tail to collapse to
/// the same unit as the head. (The empty list has no shared unit and so has no
/// impl either.)
pub trait CollapseUniform {
    /// The shared unit of every entry.
    type Unit;
}

impl<H> CollapseUniform for DCons<H, DNil> {
    type Unit = H;
}

impl<H, H2, T> CollapseUniform for DCons<H, DCons<H2, T>>
where
    DCons<H2, T>: CollapseUniform<Unit = H>,
{
    type Unit = H;
}

/// The shared unit of a uniform dimension list `L`.
pub type Uniform<L> = <L as CollapseUniform>::Unit;

/// Marker that a candidate entry unit `Q` reproduces a target unit `U` (i.e.
/// `Q == U` at the type level).
///
/// This is the soundness condition for gauging a uniform block into a mixed
/// one: a [`UniformUnitMatrix`](crate::nalgebra::UniformUnitMatrix) carries a single entry
/// unit `U`, and placing it as a `MixedUnitMatrix` over `<RowDims, ColDims>`
/// tags entry `(i, j)` with `RowDims[i] / ColDims[j]`. For every such tag to
/// still be `U`, the lists must be uniform and their quotient must be `U` —
/// exactly `Q = Uniform(RowDims) / Uniform(ColDims)` satisfying `Q: GaugeReproduces<U>`.
///
/// It is stated as its own trait (rather than a bare `UnitEq<_, Output = B1>`
/// bound) purely for diagnostics: an inconsistent gauge then reports *this*
/// obligation, with the actionable message below, instead of a raw `B0 != B1`
/// type mismatch.
#[diagnostic::on_unimplemented(
    message = "the gauge tag does not reproduce the uniform block's entry unit",
    label = "this row/column gauge is inconsistent with the block's unit",
    note = "A uniform block carries a single entry unit `U`. Placing it as a mixed \
            block over `<RowDims, ColDims>` requires both lists to be uniform and \
            their entry-unit quotient to equal `U`: \
            `Uniform(RowDims) / Uniform(ColDims) == U`.",
    note = "Fix: choose row/column dimension lists whose quotient is the block's \
            unit — any common scale shared by both is free, and that is the gauge."
)]
pub trait GaugeReproduces<U> {}

// `do_not_recommend` keeps rustc from drilling into this impl's `UnitEq`
// prerequisite when the gauge is inconsistent, so the failure reports *as*
// `GaugeReproduces` (with the message above) rather than as a bare
// `<_ as UnitEq<_>>::Output == B1` mismatch.
#[diagnostic::do_not_recommend]
impl<Q, U> GaugeReproduces<U> for Q where Q: UnitEq<U, Output = B1> {}
