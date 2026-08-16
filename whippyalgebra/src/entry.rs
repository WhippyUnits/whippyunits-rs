//! Backend-agnostic unit-entry machinery.
//!
//! These type aliases and the [`FromRaw`] trait describe how a matrix entry's
//! unit is derived from a row/column dimension list and how a raw scalar is
//! wrapped back into a `Quantity`. Nothing here depends on any particular
//! linear-algebra backend, so it is always available regardless of which
//! backend feature (e.g. `nalgebra`) is enabled.

use whippyunits::quantity::{
    _2, _3, _5, _A, _I, _J, _L, _M, _N, _Pi, _T, _Θ, Dimension, Quantity, Scale,
};
use whippyunits::{DivUnit, Unit};

use crate::dims::{Dimensionless, Producted};
use crate::index::{ElemAt, Nat};

/// The `Unit` entry at row index `I` of a dimension list.
pub type RowUnitOf<RowDims, const I: usize> = <RowDims as ElemAt<Nat<I>>>::Elem;

/// The `Unit` entry at column index `J` of a dimension list.
pub type ColUnitOf<ColDims, const J: usize> = <ColDims as ElemAt<Nat<J>>>::Elem;

/// The unit of matrix entry `(I, J)` as a bare `Unit`: `RowDims[I] / ColDims[J]`.
pub type EntryUnit<RowDims, ColDims, const I: usize, const J: usize> =
    DivUnit<RowUnitOf<RowDims, I>, ColUnitOf<ColDims, J>>;

/// The unit of matrix entry `(I, J)` as a full `Quantity`, carrying the
/// matrix's storage type `T` and `Brand`.
pub type EntryUnitOf<RowDims, ColDims, T, Brand, const I: usize, const J: usize> =
    Quantity<EntryUnit<RowDims, ColDims, I, J>, T, Brand>;

/// The unit of the determinant of an `n x n` matrix as a bare `Unit`:
/// `∏ RowDims / ∏ ColDims`.
pub type DetUnit<RowDims, ColDims> = DivUnit<Producted<RowDims>, Producted<ColDims>>;

/// The determinant's unit as a full `Quantity`, carrying storage type `T` and
/// `Brand`.
pub type DetUnitOf<RowDims, ColDims, T, Brand> = Quantity<DetUnit<RowDims, ColDims>, T, Brand>;

/// A value that can populate a matrix entry whose unit is `U` (with storage
/// type `T` and brand `Brand`).
///
/// This is the write-side dual of [`FromRaw`], and the basis of a matrix's
/// unit-safe element setters. It is implemented by:
///
/// - any [`Quantity<U, T, Brand>`] — so populating entry `(i, j)` demands a
///   quantity of exactly its unit `RowDims[i] / ColDims[j]`; a wrong-unit
///   quantity has no matching impl and is a compile error; and
/// - a bare scalar `T`, but only when the entry is [`Dimensionless`] — a
///   convenience so a dimensionless cell can be filled with a plain number
///   instead of wrapping it in a trivial quantity.
///
/// The stored scalar is the quantity's value at that unit's scale (its
/// `unsafe_value`), symmetric with how [`FromRaw`] reads one back out.
pub trait IntoEntry<U, T, Brand> {
    /// Consumes `self`, yielding the raw scalar to store at the entry.
    fn into_entry(self) -> T;
}

impl<U, T, Brand> IntoEntry<U, T, Brand> for Quantity<U, T, Brand> {
    fn into_entry(self) -> T {
        self.unsafe_value
    }
}

macro_rules! scalar_dimensionless_entry {
    ($($t:ty),* $(,)?) => {$(
        impl<Brand> IntoEntry<Dimensionless, $t, Brand> for $t {
            fn into_entry(self) -> $t {
                self
            }
        }
    )*};
}

// A bare scalar can only stand in for a *dimensionless* entry. (Enumerated per
// type rather than blanket-`impl<T>` so it can't collide with the `Quantity`
// impl above under coherence's occurs check.)
scalar_dimensionless_entry!(
    f64, f32, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
);

/// Constructs a `Quantity` of a statically-unknown unit from a raw scalar.
///
/// This is the escape hatch that lets a matrix's `get` accessor build the
/// correctly-typed element `Quantity` from the underlying backend scalar. The
/// blanket impl covers every `Quantity` shape, delegating to `Quantity::new`.
pub trait FromRaw<T> {
    /// Wraps `value` as `Self` without any scaling or conversion.
    fn from_raw(value: T) -> Self;

    /// Reinterprets a reference to a raw scalar as a reference to `Self`.
    fn ref_from_raw(value: &T) -> &Self;

    /// Reinterprets a mutable reference to a raw scalar as a mutable reference
    /// to `Self`.
    fn mut_from_raw(value: &mut T) -> &mut Self;
}

impl<
    const P2: i16,
    const P3: i16,
    const P5: i16,
    const PI: i16,
    const MASS: i16,
    const LENGTH: i16,
    const TIME: i16,
    const CURRENT: i16,
    const TEMPERATURE: i16,
    const AMOUNT: i16,
    const LUMINOSITY: i16,
    const ANGLE: i16,
    T,
    Brand,
> FromRaw<T>
    for Quantity<
        Unit<
            Scale<_2<P2>, _3<P3>, _5<P5>, _Pi<PI>>,
            Dimension<
                _M<MASS>,
                _L<LENGTH>,
                _T<TIME>,
                _I<CURRENT>,
                _Θ<TEMPERATURE>,
                _N<AMOUNT>,
                _J<LUMINOSITY>,
                _A<ANGLE>,
            >,
        >,
        T,
        Brand,
    >
{
    fn from_raw(value: T) -> Self {
        Quantity::new(value)
    }

    fn ref_from_raw(value: &T) -> &Self {
        Quantity::from_ref(value)
    }

    fn mut_from_raw(value: &mut T) -> &mut Self {
        Quantity::from_mut(value)
    }
}
