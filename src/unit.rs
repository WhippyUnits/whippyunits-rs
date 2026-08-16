//! Value-free unit types and their type-level algebra.
//!
//! A [`Unit`] is the dimensional signature of a [`Quantity`]
//! — its [`Scale`] (prefix) and
//! [`Dimension`] — with no storage type, brand, or
//! runtime value. It exists so that unit arithmetic can be performed purely at
//! the type level, without materializing a `Quantity`.
//!
//! This is what downstream crates that reason about units as types (for
//! example a unit-safe linear-algebra layer, where each matrix entry has its
//! own statically-known unit) want: the multiplicative structure of units,
//! divorced from any particular value, storage type, or brand.
//!
//! The operations are exposed as associated-type traits:
//!
//! - [`UnitMul`] — multiply two units (add exponents),
//! - [`UnitDiv`] — divide two units (subtract exponents),
//! - [`UnitInv`] — invert a unit (negate exponents).
//!
//! and the bridge to `Quantity`:
//!
//! - [`HasUnit`] — project a `Quantity` type to its `Unit`.
//!
//! (The reverse bridge is trivial and needs no trait: a `Unit<S, D>` reifies
//! into a `Quantity` simply as `Quantity<Unit<S, D>, T, Brand>`.)
//!
//! The exponent arithmetic is delegated to the same type-level numeral
//! (`whippyunits_core::num::N`) used by `Quantity`'s own arithmetic, so units
//! and quantities agree by construction.
//!
//! Crucially, because these operations are *unary*/*binary over the unit types
//! directly* (rather than over a `Quantity` that also carries a storage type),
//! a generic caller can name results like `<U as UnitInv>::Output` while `U` is
//! still abstract: the obligation is deferred to the concrete instantiation,
//! exactly like `Quantity`'s arithmetic defers when both operands are abstract.

use core::marker::PhantomData;
use core::ops::{Add, Neg, Sub};

use whippyunits_core::num::{BitAll, Halve, N, NumEq};

use crate::quantity::{
    _2, _3, _5, _A, _I, _J, _L, _M, _N, _Pi, _T, _Θ, Dimension, Quantity, Scale,
};

/// The dimensional signature of a quantity: its `Scale` (prefix) and
/// `Dimension`, without any storage type, brand, or value.
#[derive(Clone, PartialEq)]
pub struct Unit<Scale, Dimension>(PhantomData<(Scale, Dimension)>);

/// Projects a `Quantity` type to its constituent scale, dimension, and [`Unit`],
/// discarding the storage type and brand.
pub trait HasUnit {
    /// The quantity's scale (prefix) marker.
    type Scale;
    /// The quantity's dimension marker.
    type Dimension;
    /// The quantity's [`Unit`] (scale + dimension).
    type Unit;
}

impl<S, D, T, Brand> HasUnit for Quantity<Unit<S, D>, T, Brand> {
    type Scale = S;
    type Dimension = D;
    type Unit = Unit<S, D>;
}

/// Type-level unit multiplication (adds the scale and dimension exponents).
pub trait UnitMul<Rhs> {
    /// The product unit.
    type Output;
}

/// Type-level unit division (subtracts the scale and dimension exponents).
pub trait UnitDiv<Rhs> {
    /// The quotient unit.
    type Output;
}

/// Type-level unit inversion (negates the scale and dimension exponents).
pub trait UnitInv {
    /// The reciprocal unit.
    type Output;
}

/// Type-level unit square root: halves every one of the twelve exponents.
///
/// This is a partial operation — it exists only when every exponent
/// (scale and dimension alike) is even, since a fractional exponent has no
/// representation in this integer-exponent algebra. So a `where U: UnitSqrt`
/// bound doubles as a compile-time proof that `U` is a perfect square, and
/// [`Output`](UnitSqrt::Output) is the unique unit `W` with `W · W = U`.
// NOTE: intent-documentation only — an odd exponent fails through `Halve`'s
// numeral-level `Rem`/`PartialDiv` equality, so rustc reports that leaf rather
// than "`UnitSqrt` unimplemented" and this text does not reach the user.
#[diagnostic::on_unimplemented(
    message = "`{Self}` has no unit square root",
    label = "not a perfect square: at least one dimension or scale exponent is odd",
    note = "A unit square root exists only when every one of the twelve exponents is even (each is halved); there is no fractional exponent in this integer-exponent algebra.",
    note = "This gates operations that root a *uniform* value — e.g. a uniform Cholesky factor or the gated uniform generalized eigenproblem; the mixed-metric variants root numerically and need no even exponents."
)]
pub trait UnitSqrt {
    /// The square-root unit `W` such that `W · W == Self`.
    type Output;
}

/// Type-level unit equality: [`Output`](UnitEq::Output) is a `typenum::Bit`,
/// `B1` iff the two units have identical scale and dimension (every one of the
/// twelve exponents agrees).
///
/// Unlike a `where A == B` bound, a `Bit` verdict can be branched on at the
/// type level — which is what lets a caller decide, purely from types, whether
/// a collection of units is uniform (and therefore whether it can collapse to a
/// single shared unit).
pub trait UnitEq<Rhs> {
    /// The `typenum::Bit` `B1` (identical) or `B0` (differ).
    type Output;
}

/// The product of two units.
pub type MulUnit<A, B> = <A as UnitMul<B>>::Output;
/// The quotient of two units.
pub type DivUnit<A, B> = <A as UnitDiv<B>>::Output;
/// The reciprocal of a unit.
pub type InvUnit<U> = <U as UnitInv>::Output;
/// The square root of a unit (defined only when every exponent is even).
pub type SqrtUnit<U> = <U as UnitSqrt>::Output;
/// The `typenum::Bit` verdict of whether two units are identical.
pub type UnitsEqual<A, B> = <A as UnitEq<B>>::Output;

impl<
    const P2_1: i16,
    const P3_1: i16,
    const P5_1: i16,
    const PI_1: i16,
    const M_1: i16,
    const L_1: i16,
    const T_1: i16,
    const I_1: i16,
    const TH_1: i16,
    const AMT_1: i16,
    const J_1: i16,
    const ANG_1: i16,
    const P2_2: i16,
    const P3_2: i16,
    const P5_2: i16,
    const PI_2: i16,
    const M_2: i16,
    const L_2: i16,
    const T_2: i16,
    const I_2: i16,
    const TH_2: i16,
    const AMT_2: i16,
    const J_2: i16,
    const ANG_2: i16,
    const P2_O: i16,
    const P3_O: i16,
    const P5_O: i16,
    const PI_O: i16,
    const M_O: i16,
    const L_O: i16,
    const T_O: i16,
    const I_O: i16,
    const TH_O: i16,
    const AMT_O: i16,
    const J_O: i16,
    const ANG_O: i16,
>
    UnitMul<
        Unit<
            Scale<_2<P2_2>, _3<P3_2>, _5<P5_2>, _Pi<PI_2>>,
            Dimension<_M<M_2>, _L<L_2>, _T<T_2>, _I<I_2>, _Θ<TH_2>, _N<AMT_2>, _J<J_2>, _A<ANG_2>>,
        >,
    >
    for Unit<
        Scale<_2<P2_1>, _3<P3_1>, _5<P5_1>, _Pi<PI_1>>,
        Dimension<_M<M_1>, _L<L_1>, _T<T_1>, _I<I_1>, _Θ<TH_1>, _N<AMT_1>, _J<J_1>, _A<ANG_1>>,
    >
where
    N<P2_1>: Add<N<P2_2>, Output = N<P2_O>>,
    N<P3_1>: Add<N<P3_2>, Output = N<P3_O>>,
    N<P5_1>: Add<N<P5_2>, Output = N<P5_O>>,
    N<PI_1>: Add<N<PI_2>, Output = N<PI_O>>,
    N<M_1>: Add<N<M_2>, Output = N<M_O>>,
    N<L_1>: Add<N<L_2>, Output = N<L_O>>,
    N<T_1>: Add<N<T_2>, Output = N<T_O>>,
    N<I_1>: Add<N<I_2>, Output = N<I_O>>,
    N<TH_1>: Add<N<TH_2>, Output = N<TH_O>>,
    N<AMT_1>: Add<N<AMT_2>, Output = N<AMT_O>>,
    N<J_1>: Add<N<J_2>, Output = N<J_O>>,
    N<ANG_1>: Add<N<ANG_2>, Output = N<ANG_O>>,
{
    type Output = Unit<
        Scale<_2<P2_O>, _3<P3_O>, _5<P5_O>, _Pi<PI_O>>,
        Dimension<_M<M_O>, _L<L_O>, _T<T_O>, _I<I_O>, _Θ<TH_O>, _N<AMT_O>, _J<J_O>, _A<ANG_O>>,
    >;
}

impl<
    const P2_1: i16,
    const P3_1: i16,
    const P5_1: i16,
    const PI_1: i16,
    const M_1: i16,
    const L_1: i16,
    const T_1: i16,
    const I_1: i16,
    const TH_1: i16,
    const AMT_1: i16,
    const J_1: i16,
    const ANG_1: i16,
    const P2_2: i16,
    const P3_2: i16,
    const P5_2: i16,
    const PI_2: i16,
    const M_2: i16,
    const L_2: i16,
    const T_2: i16,
    const I_2: i16,
    const TH_2: i16,
    const AMT_2: i16,
    const J_2: i16,
    const ANG_2: i16,
    const P2_O: i16,
    const P3_O: i16,
    const P5_O: i16,
    const PI_O: i16,
    const M_O: i16,
    const L_O: i16,
    const T_O: i16,
    const I_O: i16,
    const TH_O: i16,
    const AMT_O: i16,
    const J_O: i16,
    const ANG_O: i16,
>
    UnitDiv<
        Unit<
            Scale<_2<P2_2>, _3<P3_2>, _5<P5_2>, _Pi<PI_2>>,
            Dimension<_M<M_2>, _L<L_2>, _T<T_2>, _I<I_2>, _Θ<TH_2>, _N<AMT_2>, _J<J_2>, _A<ANG_2>>,
        >,
    >
    for Unit<
        Scale<_2<P2_1>, _3<P3_1>, _5<P5_1>, _Pi<PI_1>>,
        Dimension<_M<M_1>, _L<L_1>, _T<T_1>, _I<I_1>, _Θ<TH_1>, _N<AMT_1>, _J<J_1>, _A<ANG_1>>,
    >
where
    N<P2_1>: Sub<N<P2_2>, Output = N<P2_O>>,
    N<P3_1>: Sub<N<P3_2>, Output = N<P3_O>>,
    N<P5_1>: Sub<N<P5_2>, Output = N<P5_O>>,
    N<PI_1>: Sub<N<PI_2>, Output = N<PI_O>>,
    N<M_1>: Sub<N<M_2>, Output = N<M_O>>,
    N<L_1>: Sub<N<L_2>, Output = N<L_O>>,
    N<T_1>: Sub<N<T_2>, Output = N<T_O>>,
    N<I_1>: Sub<N<I_2>, Output = N<I_O>>,
    N<TH_1>: Sub<N<TH_2>, Output = N<TH_O>>,
    N<AMT_1>: Sub<N<AMT_2>, Output = N<AMT_O>>,
    N<J_1>: Sub<N<J_2>, Output = N<J_O>>,
    N<ANG_1>: Sub<N<ANG_2>, Output = N<ANG_O>>,
{
    type Output = Unit<
        Scale<_2<P2_O>, _3<P3_O>, _5<P5_O>, _Pi<PI_O>>,
        Dimension<_M<M_O>, _L<L_O>, _T<T_O>, _I<I_O>, _Θ<TH_O>, _N<AMT_O>, _J<J_O>, _A<ANG_O>>,
    >;
}

impl<
    const P2: i16,
    const P3: i16,
    const P5: i16,
    const PI: i16,
    const M: i16,
    const L: i16,
    const T: i16,
    const I: i16,
    const TH: i16,
    const AMT: i16,
    const J: i16,
    const ANG: i16,
    const P2_O: i16,
    const P3_O: i16,
    const P5_O: i16,
    const PI_O: i16,
    const M_O: i16,
    const L_O: i16,
    const T_O: i16,
    const I_O: i16,
    const TH_O: i16,
    const AMT_O: i16,
    const J_O: i16,
    const ANG_O: i16,
> UnitInv
    for Unit<
        Scale<_2<P2>, _3<P3>, _5<P5>, _Pi<PI>>,
        Dimension<_M<M>, _L<L>, _T<T>, _I<I>, _Θ<TH>, _N<AMT>, _J<J>, _A<ANG>>,
    >
where
    N<P2>: Neg<Output = N<P2_O>>,
    N<P3>: Neg<Output = N<P3_O>>,
    N<P5>: Neg<Output = N<P5_O>>,
    N<PI>: Neg<Output = N<PI_O>>,
    N<M>: Neg<Output = N<M_O>>,
    N<L>: Neg<Output = N<L_O>>,
    N<T>: Neg<Output = N<T_O>>,
    N<I>: Neg<Output = N<I_O>>,
    N<TH>: Neg<Output = N<TH_O>>,
    N<AMT>: Neg<Output = N<AMT_O>>,
    N<J>: Neg<Output = N<J_O>>,
    N<ANG>: Neg<Output = N<ANG_O>>,
{
    type Output = Unit<
        Scale<_2<P2_O>, _3<P3_O>, _5<P5_O>, _Pi<PI_O>>,
        Dimension<_M<M_O>, _L<L_O>, _T<T_O>, _I<I_O>, _Θ<TH_O>, _N<AMT_O>, _J<J_O>, _A<ANG_O>>,
    >;
}

impl<
    const P2: i16,
    const P3: i16,
    const P5: i16,
    const PI: i16,
    const M: i16,
    const L: i16,
    const T: i16,
    const I: i16,
    const TH: i16,
    const AMT: i16,
    const J: i16,
    const ANG: i16,
    const P2_O: i16,
    const P3_O: i16,
    const P5_O: i16,
    const PI_O: i16,
    const M_O: i16,
    const L_O: i16,
    const T_O: i16,
    const I_O: i16,
    const TH_O: i16,
    const AMT_O: i16,
    const J_O: i16,
    const ANG_O: i16,
> UnitSqrt
    for Unit<
        Scale<_2<P2>, _3<P3>, _5<P5>, _Pi<PI>>,
        Dimension<_M<M>, _L<L>, _T<T>, _I<I>, _Θ<TH>, _N<AMT>, _J<J>, _A<ANG>>,
    >
where
    N<P2>: Halve<Output = N<P2_O>>,
    N<P3>: Halve<Output = N<P3_O>>,
    N<P5>: Halve<Output = N<P5_O>>,
    N<PI>: Halve<Output = N<PI_O>>,
    N<M>: Halve<Output = N<M_O>>,
    N<L>: Halve<Output = N<L_O>>,
    N<T>: Halve<Output = N<T_O>>,
    N<I>: Halve<Output = N<I_O>>,
    N<TH>: Halve<Output = N<TH_O>>,
    N<AMT>: Halve<Output = N<AMT_O>>,
    N<J>: Halve<Output = N<J_O>>,
    N<ANG>: Halve<Output = N<ANG_O>>,
{
    type Output = Unit<
        Scale<_2<P2_O>, _3<P3_O>, _5<P5_O>, _Pi<PI_O>>,
        Dimension<_M<M_O>, _L<L_O>, _T<T_O>, _I<I_O>, _Θ<TH_O>, _N<AMT_O>, _J<J_O>, _A<ANG_O>>,
    >;
}

impl<
    const P2_1: i16,
    const P3_1: i16,
    const P5_1: i16,
    const PI_1: i16,
    const M_1: i16,
    const L_1: i16,
    const T_1: i16,
    const I_1: i16,
    const TH_1: i16,
    const AMT_1: i16,
    const J_1: i16,
    const ANG_1: i16,
    const P2_2: i16,
    const P3_2: i16,
    const P5_2: i16,
    const PI_2: i16,
    const M_2: i16,
    const L_2: i16,
    const T_2: i16,
    const I_2: i16,
    const TH_2: i16,
    const AMT_2: i16,
    const J_2: i16,
    const ANG_2: i16,
>
    UnitEq<
        Unit<
            Scale<_2<P2_2>, _3<P3_2>, _5<P5_2>, _Pi<PI_2>>,
            Dimension<_M<M_2>, _L<L_2>, _T<T_2>, _I<I_2>, _Θ<TH_2>, _N<AMT_2>, _J<J_2>, _A<ANG_2>>,
        >,
    >
    for Unit<
        Scale<_2<P2_1>, _3<P3_1>, _5<P5_1>, _Pi<PI_1>>,
        Dimension<_M<M_1>, _L<L_1>, _T<T_1>, _I<I_1>, _Θ<TH_1>, _N<AMT_1>, _J<J_1>, _A<ANG_1>>,
    >
where
    N<P2_1>: NumEq<N<P2_2>>,
    N<P3_1>: NumEq<N<P3_2>>,
    N<P5_1>: NumEq<N<P5_2>>,
    N<PI_1>: NumEq<N<PI_2>>,
    N<M_1>: NumEq<N<M_2>>,
    N<L_1>: NumEq<N<L_2>>,
    N<T_1>: NumEq<N<T_2>>,
    N<I_1>: NumEq<N<I_2>>,
    N<TH_1>: NumEq<N<TH_2>>,
    N<AMT_1>: NumEq<N<AMT_2>>,
    N<J_1>: NumEq<N<J_2>>,
    N<ANG_1>: NumEq<N<ANG_2>>,
    UnitEqBits<
        P2_1,
        P3_1,
        P5_1,
        PI_1,
        M_1,
        L_1,
        T_1,
        I_1,
        TH_1,
        AMT_1,
        J_1,
        ANG_1,
        P2_2,
        P3_2,
        P5_2,
        PI_2,
        M_2,
        L_2,
        T_2,
        I_2,
        TH_2,
        AMT_2,
        J_2,
        ANG_2,
    >: BitAll,
{
    type Output = <UnitEqBits<
        P2_1,
        P3_1,
        P5_1,
        PI_1,
        M_1,
        L_1,
        T_1,
        I_1,
        TH_1,
        AMT_1,
        J_1,
        ANG_1,
        P2_2,
        P3_2,
        P5_2,
        PI_2,
        M_2,
        L_2,
        T_2,
        I_2,
        TH_2,
        AMT_2,
        J_2,
        ANG_2,
    > as BitAll>::Output;
}

/// The cons-list of the twelve per-exponent [`NumEq`] verdicts for two units,
/// ready to be AND-folded by [`BitAll`]. Naming it as an alias keeps the
/// [`UnitEq`] impl legible instead of a twelve-deep nested tuple written twice.
#[allow(clippy::type_complexity)]
type UnitEqBits<
    const P2_1: i16,
    const P3_1: i16,
    const P5_1: i16,
    const PI_1: i16,
    const M_1: i16,
    const L_1: i16,
    const T_1: i16,
    const I_1: i16,
    const TH_1: i16,
    const AMT_1: i16,
    const J_1: i16,
    const ANG_1: i16,
    const P2_2: i16,
    const P3_2: i16,
    const P5_2: i16,
    const PI_2: i16,
    const M_2: i16,
    const L_2: i16,
    const T_2: i16,
    const I_2: i16,
    const TH_2: i16,
    const AMT_2: i16,
    const J_2: i16,
    const ANG_2: i16,
> = (
    <N<P2_1> as NumEq<N<P2_2>>>::Output,
    (
        <N<P3_1> as NumEq<N<P3_2>>>::Output,
        (
            <N<P5_1> as NumEq<N<P5_2>>>::Output,
            (
                <N<PI_1> as NumEq<N<PI_2>>>::Output,
                (
                    <N<M_1> as NumEq<N<M_2>>>::Output,
                    (
                        <N<L_1> as NumEq<N<L_2>>>::Output,
                        (
                            <N<T_1> as NumEq<N<T_2>>>::Output,
                            (
                                <N<I_1> as NumEq<N<I_2>>>::Output,
                                (
                                    <N<TH_1> as NumEq<N<TH_2>>>::Output,
                                    (
                                        <N<AMT_1> as NumEq<N<AMT_2>>>::Output,
                                        (
                                            <N<J_1> as NumEq<N<J_2>>>::Output,
                                            (<N<ANG_1> as NumEq<N<ANG_2>>>::Output, ()),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
);

#[cfg(test)]
mod tests {
    use super::UnitEq;
    use typenum::{B0, B1};

    fn assert_eq_bit<A: UnitEq<B, Output = O>, B, O>() {}

    #[test]
    fn identical_units_are_b1() {
        assert_eq_bit::<crate::unit!(m / s), crate::unit!(m / s), B1>();
        assert_eq_bit::<crate::unit!(kg), crate::unit!(kg), B1>();
        assert_eq_bit::<crate::unit!(1), crate::unit!(1), B1>();
        // Same unit reached by different spellings must still be identical.
        assert_eq_bit::<crate::unit!(N), crate::unit!(kg * m / s ^ 2), B1>();
    }

    #[test]
    fn differing_units_are_b0() {
        assert_eq_bit::<crate::unit!(m / s), crate::unit!(m), B0>();
        assert_eq_bit::<crate::unit!(m), crate::unit!(km), B0>(); // scale differs
        assert_eq_bit::<crate::unit!(m), crate::unit!(kg), B0>(); // dimension differs
    }
}
