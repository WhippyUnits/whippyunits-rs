//! Compile-time indexing of dimension lists.
//!
//! Unit-safe element access is only meaningful at a statically-known index: the
//! unit of entry `(i, j)` is `RowDims[i] / ColDims[j]`, which is a *type*, so
//! `i` and `j` must be known at compile time. We therefore index the cons-list
//! with a const generic, bridged to a type-level natural so the list can be
//! walked at the type level.
//!
//! The naturals are [`typenum`]'s — the same machinery the rest of the stack
//! already depends on — rather than a parallel Peano encoding: a `usize` const
//! `N` becomes the type-level [`Unsigned`] `Nat<N>` (a `typenum::U{N}`). The
//! const-generic bridge that produces it is keyed on a single `Const`: with
//! the `nalgebra` feature it is `nalgebra::Const<N>` (bridged via nalgebra's
//! `ToTypenum`), so the one `Const` a user names for sizing is the same one they
//! name for indexing; without a backend it falls back to typenum's `Const`. Only
//! the list operations ([`ElemAt`], [`Take`], [`Drop`]) are ours, since they
//! walk our unit cons-list; each step peels one node and decrements the index by
//! one (`Sub1`), bottoming out at [`UTerm`] (zero).
//!
//! Indexing or slicing out of range is a compile error: [`ElemAt`] is only
//! implemented for [`DCons`], never for
//! [`DNil`], and the `Take`/`Drop` recursions likewise run
//! off the end of the list and fail to resolve.

use core::ops::Sub;

use typenum::{B1, Bit, Sub1, UInt, UTerm};

pub use typenum::Unsigned;

use crate::dims::{DCons, DNil, Product};

/// A shape (`Const<N>`) that has a type-level-natural representation — i.e. one
/// usable as a compile-time index or size by the indexing/slicing operations.
///
/// The carrier is the backend's const-generic dimension: with the `nalgebra`
/// feature it is `nalgebra::Const<N>`, whose associated
/// [`Nat`](ShapeIndex::Nat) is `<nalgebra::Const<N> as ToTypenum>::Typenum` —
/// the same `typenum::U{N}` typenum's own bridge would produce, so a single
/// `Const` (nalgebra's) serves both matrix sizing and list indexing. Without a
/// backend it falls back to typenum's `Const<N>`/`ToUInt`.
///
/// The crate routes every const-generic index/size through it (via the [`Nat`]
/// alias) so that the one obligation the type-nat encoding leaks reports as
/// `ShapeIndex` — carrying the actionable [`on_unimplemented`] message — instead
/// of a bare `ToTypenum`/`ToUInt` miss.
///
/// The obligation holds for every concrete shape the backend maps (nalgebra:
/// `Const<0..=127>`), so ordinary code never mentions it — it only surfaces when
/// a function is generic over a `const N: usize` and indexes or slices by it.
///
/// [`on_unimplemented`]: https://doc.rust-lang.org/reference/attributes/diagnostics.html
#[diagnostic::on_unimplemented(
    message = "the shape `{Self}` can't be used as a compile-time index or size",
    label = "no type-level representation for this shape",
    note = "Indexing or slicing a dimension list bridges a `usize` shape to a \
            type-level natural. Every concrete shape (up to nalgebra's mapped \
            maximum of 127) satisfies this automatically; it only needs stating \
            when a function is generic over the shape.",
    note = "Fix: add `{Self}: ShapeIndex` to the enclosing `where` clause — for a \
            function generic over `const N: usize`, that is \
            `nalgebra::Const<N>: ShapeIndex` (or, for a square dimension, the \
            bundled `nalgebra::Const<N>: SquareDim`).",
    note = "(A shape above the backend's mapped maximum also lands here.)"
)]
pub trait ShapeIndex {
    /// The type-level natural (an [`Unsigned`]) this shape maps to.
    ///
    /// Bounded [`Unsigned`] so a shape routed through `ShapeIndex` can be used
    /// as a type-level offset (added to other offsets, walked by [`Drop`]) — see
    /// [`SlicedAt`] and the block-extraction offsets it feeds.
    type Nat: Unsigned;
}

// `do_not_recommend` stops rustc from drilling into this impl's bridge
// prerequisite (`ToTypenum`/`ToUInt`) when the bound is unmet, so a missing
// `Const<N>: ShapeIndex` reports *as* `ShapeIndex` (naming the documented trait,
// with its message) instead of dumping the bridge's hundred-plus impls.

// With a backend, `nalgebra::Const<N>` is the sole `Const`; its `Nat` comes from
// nalgebra's `ToTypenum` (mapped `0..=127`), so the same `Const` appears in both
// sizing and indexing bounds.
#[cfg(feature = "nalgebra")]
#[diagnostic::do_not_recommend]
impl<const N: usize> ShapeIndex for ::nalgebra::Const<N>
where
    ::nalgebra::Const<N>: ::nalgebra::ToTypenum,
{
    type Nat = <::nalgebra::Const<N> as ::nalgebra::ToTypenum>::Typenum;
}

// Backend-agnostic fallback: typenum's own const-generic bridge, so the core
// list machinery still compiles with no backend feature enabled.
#[cfg(not(feature = "nalgebra"))]
#[diagnostic::do_not_recommend]
impl<T: typenum::ToUInt> ShapeIndex for T
where
    <T as typenum::ToUInt>::Output: Unsigned,
{
    type Nat = <T as typenum::ToUInt>::Output;
}

/// The type-level natural a const size/index `N` maps to, routed through
/// [`ShapeIndex`] on the backend's `Const` (so a missing generic bound reports
/// the actionable diagnostic). This is the alias the list operations below are
/// driven by; either carrier yields the same `typenum::U{N}`.
#[cfg(feature = "nalgebra")]
pub type Nat<const N: usize> = <::nalgebra::Const<N> as ShapeIndex>::Nat;
/// The type-level natural a const size/index `N` maps to (backend-agnostic
/// fallback, via typenum's `Const`).
#[cfg(not(feature = "nalgebra"))]
pub type Nat<const N: usize> = <typenum::Const<N> as ShapeIndex>::Nat;

/// Type-level indexing: `Self`'s element at position `Index` (an [`Unsigned`]).
pub trait ElemAt<Index> {
    /// The `Unit` entry at this position.
    type Elem;
}
impl<H, Tail> ElemAt<UTerm> for DCons<H, Tail> {
    type Elem = H;
}
impl<H, Tail, Hi, B> ElemAt<UInt<Hi, B>> for DCons<H, Tail>
where
    Hi: Unsigned,
    B: Bit,
    UInt<Hi, B>: Sub<B1>,
    Tail: ElemAt<Sub1<UInt<Hi, B>>>,
{
    type Elem = <Tail as ElemAt<Sub1<UInt<Hi, B>>>>::Elem;
}

/// Type-level `take`: the first `N` (an [`Unsigned`]) entries of a list, in
/// order.
///
/// Taking more entries than the list holds is a compile error — there is no
/// `Take<UInt<..>>` impl for [`DNil`], so the recursion runs
/// off the end and fails to resolve. This is what bounds a block to the matrix.
pub trait Take<N> {
    /// The prefix of length `N`.
    type Out;
}
impl<L> Take<UTerm> for L {
    type Out = DNil;
}
impl<H, Tail, Hi, B> Take<UInt<Hi, B>> for DCons<H, Tail>
where
    Hi: Unsigned,
    B: Bit,
    UInt<Hi, B>: Sub<B1>,
    Tail: Take<Sub1<UInt<Hi, B>>>,
{
    type Out = DCons<H, <Tail as Take<Sub1<UInt<Hi, B>>>>::Out>;
}

/// Type-level `drop`: the list with its first `N` (an [`Unsigned`]) entries
/// removed.
pub trait Drop<N> {
    /// The suffix after skipping `N` entries.
    type Out;
}
impl<L> Drop<UTerm> for L {
    type Out = L;
}
impl<H, Tail, Hi, B> Drop<UInt<Hi, B>> for DCons<H, Tail>
where
    Hi: Unsigned,
    B: Bit,
    UInt<Hi, B>: Sub<B1>,
    Tail: Drop<Sub1<UInt<Hi, B>>>,
{
    type Out = <Tail as Drop<Sub1<UInt<Hi, B>>>>::Out;
}

/// The contiguous sublist of `L` of length `LEN` starting at the type-level
/// offset `Off` (an [`Unsigned`]): `drop Off`, then `take LEN`.
///
/// The offset is a type rather than a const so it can be formed by type-level
/// addition — `Sum<Nat<N>, Nat<M>>` — without `generic_const_exprs`. That is
/// what lets [`unblock_matrix!`](crate::nalgebra::unblock_matrix) place a block at the sum
/// of the preceding blocks' sizes even when those sizes are generic const
/// parameters: a partition of any width/height, not just two.
pub type SlicedAt<L, Off, const LEN: usize> = <<L as Drop<Off>>::Out as Take<Nat<LEN>>>::Out;

/// The contiguous sublist of `L` of length `LEN` starting at the const offset
/// `OFF`: `drop OFF`, then `take LEN`. The const-offset spelling of
/// [`SlicedAt`], used by [`block`](crate::nalgebra::MixedUnitMatrix::block) and friends.
pub type Sliced<L, const OFF: usize, const LEN: usize> = SlicedAt<L, Nat<OFF>, LEN>;

/// Type-level `repeat`: a uniform list of `Self`-many copies of the unit `U`
/// (`Self` an [`Unsigned`]).
///
/// The inverse of [`CollapseUniform`](crate::CollapseUniform): rather than
/// reading the shared unit out of a uniform list, it builds the uniform list
/// of a given length from one unit. This is what lets a gauge be stated as a
/// single row/column unit — [`gauge!`](crate::nalgebra::gauge) — with the list lengths
/// taken from the block's own (known) shape rather than spelled out entry by
/// entry.
pub trait Repeat<U> {
    /// The list `[U; Self]`.
    type Out;
}
impl<U> Repeat<U> for UTerm {
    type Out = DNil;
}
impl<U, Hi, B> Repeat<U> for UInt<Hi, B>
where
    Hi: Unsigned,
    B: Bit,
    UInt<Hi, B>: Sub<B1>,
    Sub1<UInt<Hi, B>>: Repeat<U>,
{
    type Out = DCons<U, <Sub1<UInt<Hi, B>> as Repeat<U>>::Out>;
}

/// The uniform list of `N` copies of the unit `U`.
pub type Repeated<U, const N: usize> = <Nat<N> as Repeat<U>>::Out;

/// Type-level `Uⁿ`: the unit `U` raised to a type-level [`Unsigned`] power `N`,
/// i.e. `N` copies of `U` multiplied together (`Dimensionless` for `N = 0`).
///
/// `N` comes from a matrix's own statically known shape: `R·C` for a whole-matrix
/// [`product`](crate::nalgebra::UniformUnitMatrix::product), the square dimension
/// for a [`determinant`](crate::nalgebra::UniformUnitMatrix::determinant), a const
/// exponent for [`powi`](crate::nalgebra::UniformUnitMatrix::powi).
///
/// Every `N` resolves: a power only adds exponents, so `Uⁿ` is always
/// representable at a static size — unlike a [`UnitSqrt`](whippyunits::UnitSqrt)
/// root, which has an even-exponent gate.
pub trait UnitPow<N> {
    /// The power `Uⁿ`.
    type Output;
}
impl<U, N> UnitPow<N> for U
where
    N: Repeat<U>,
    <N as Repeat<U>>::Out: Product,
{
    type Output = <<N as Repeat<U>>::Out as Product>::Out;
}

/// `Uⁿ`: [`UnitPow`] as a type alias.
pub type PowUnit<U, N> = <U as UnitPow<N>>::Output;
