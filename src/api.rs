//! Arithmetic, rescaling, and display traits for the [Quantity] type.
//!
//! This module provides the API implementations for most operations on the
//! whippyunits [`Quantity`] type.
//!
//! The functions in this module are generated via macros to provide type-safe implementations
//! for all combinations of storage types (f32, f64, i8-i128, u8-u128) and quantity dimensions.
//!
//! ## Rescale Functions
//!
//! - [`rescale`](crate::api::rescale()): default function aliases `rescale_f64`
//! - [`rescale_f32`]
//! - [`rescale_f64`]
//! - [`rescale_i8`]
//! - [`rescale_i16`]
//! - [`rescale_i32`]
//! - [`rescale_i64`]
//! - [`rescale_i128`]
//! - [`rescale_u8`]
//! - [`rescale_u16`]
//! - [`rescale_u32`]
//! - [`rescale_u64`]
//! - [`rescale_u128`]
//!
//! All rescale functions work with type inference - specify the target type using the [`unit!`](crate::unit!) macro:
//!
//! ```rust
//! # #[culit::culit(whippyunits::default_declarators::literals)]
//! # fn main() {
//! # use whippyunits::api::rescale;
//! # use whippyunits::qty;
//! let distance: qty!(mm) = rescale(1.0m); // Converts meters to millimeters
//! let distance: qty!(m) = rescale(1000.0mm); // Converts millimeters to meters
//! // let _distance: qty!(s) = rescale(1.0m); // ❌ Compile error: dimension mismatch
//! # }
//! ```
//!
//! ## Arithmetic Operations
//!
//! Arithmetic operations are zero-cost unit-safe wrappers around the underlying numeric type operations:
//! they either compile directly to the underlying numeric type's operation, or else generate a compile error.
//!
//! ### Addition and Subtraction
//!
//! Addition and subtraction require both operands to have the same scale. To add or subtract quantities
//! with different scales, use [`rescale`](crate::api::rescale()) to convert one to match the other:
//!
//! ```rust
//! # #[culit::culit(whippyunits::default_declarators::literals)]
//! # fn main() {
//! # use whippyunits::api::rescale;
//! let distance = rescale(1.0m) + 1.0mm; // ✅ 1001.0 Quantity<mm, f64>
//! let distance = 1.0m + rescale(1.0mm); // ✅ 1.001 Quantity<m, f64>
//! // let _distance = 1.0m + 1.0mm; // 🚫 Compile error (scale mismatch)
//! // let _distance = 1.0m + 1.0s; // 🚫 Compile error (dimension mismatch)
//! # }
//! ```
//!
//! The result has the same dimensions and scale as the operands.
//!
//! ### Multiplication and Division
//!
//! Without an explicit type annotation, multiplication and division won't catch dimensional errors
//! at compile time because the compiler doesn't know what dimension you expect to get back. Use
//! [`unit!`](crate::unit!) to specify the expected result type and enable compile-time checking:
//!
//! ```rust
//! # #[culit::culit(whippyunits::default_declarators::literals)]
//! # fn main() {
//! # use whippyunits::qty;
//! let area = 5.0m * 5.0m; // ⚠️ Correct, but unchecked; will compile regardless of the units
//! let area = 5.0m * 5.0s; // ❌ BUG: compiles fine, but is not an area
//! let area: qty!(m^2) = 5.0m * 5.0m; // ✅ Correct, will compile only if the units are correct
//! // let area: qty!(m^2) = 5.0m * 5.0s; // 🚫 Compile error, as expected
//! # }
//! ```
//!
//! If you want to check the dimensionality without constraining the scale, use
//! [`define_generic_dimension!`](crate::dimension_traits::define_generic_dimension) to create a dimension trait:
//!
//! ```rust
//! # #[culit::culit(whippyunits::default_declarators::literals)]
//! # fn main() {
//! # use whippyunits::dimension_traits::define_generic_dimension;
//! define_generic_dimension!(Area, L2);
//! fn assert_area<A: Area>(value: A) -> A {
//!     value
//! }
//!
//! // Works with any scale - meters, millimeters, etc.
//! let area1 = assert_area(5.0m * 5.0m); // ✅
//! let area2 = assert_area(5.0mm * 5.0mm); // ✅
//! // let _area = assert_area(5.0m * 5.0s); // 🚫 Compile error (wrong dimension)
//! # }
//! ```
//!
//! Multiplication and division combine both dimensions and scales. The result type is *constrained by*
//! the types of the operands, but does not uniquely determine them.
//!
//! For example, `m * mm` produces `m(m²)`, but so do:
//!  - `mm * m`
//!  - `cm * dm`
//!  - `(m.s) * (mm/s)`
//!  - etc.
//!
//! ### Comparison Operators
//!
//! Comparison operators (`<`, `<=`, `>`, `>=`) are scale-strict, just like addition and subtraction.
//! Both operands must have the same scale. To compare quantities with different scales, use
//! [`rescale`](crate::api::rescale()) to convert one to match the other:
//!
//! ```rust
//! # #[culit::culit(whippyunits::default_declarators::literals)]
//! # fn main() {
//! # use whippyunits::api::rescale;
//! assert!(rescale(1.0m) > 500.0mm); // ✅ 1000.0 mm > 500.0 mm
//! assert!(1.0m > rescale(500.0mm)); // ✅ 1.0 m > 0.5 m
//! // assert!(1.0m > 500.0mm); // 🚫 Compile error (scale mismatch)
//! // assert!(1.0m > 1.0s); // 🚫 Compile error (dimension mismatch)
//! # }
//! ```
//!
//! ## Display Traits
//!
//! The [`Display`](std::fmt::Display) and [`Debug`](std::fmt::Debug) traits are implemented for
//! all quantity types, providing human-readable output with proper unit formatting.

use crate::define_aggregate_scale_factor_float;
use crate::define_aggregate_scale_factor_rational;
#[cfg(feature = "alloc")]
use crate::define_display_traits;
#[cfg(feature = "alloc")]
use crate::print::prettyprint::*;
use crate::quantity::*;
use crate::scale_conversion::*;
#[cfg(feature = "alloc")]
use core::fmt;
use whippyunits_core::num::N;

define_aggregate_scale_factor_rational!(
    // params
    (
        scale_p2_from: i16, scale_p3_from: i16, scale_p5_from: i16, scale_pi_from: i16,
        scale_p2_to: i16, scale_p3_to: i16, scale_p5_to: i16, scale_pi_to: i16,
    ),
    // diff expressions
    (
        let diff_scale_p2 = scale_p2_from - scale_p2_to;
        let diff_scale_p3 = scale_p3_from - scale_p3_to;
        let diff_scale_p5 = scale_p5_from - scale_p5_to;
        let diff_scale_pi = scale_pi_from - scale_pi_to;
    ),
    // pow expressions
    (
        let (num2, den2) = pow2(diff_scale_p2 as i32);
        let (num3, den3) = pow3(diff_scale_p3 as i32);
        let (num5, den5) = pow5(diff_scale_p5 as i32);
        let (num_pi, den_pi) = pow_pi(diff_scale_pi as i32);
    ),
    // num and den expressions
    (num2 * num3 * num5 * num_pi),
    (den2 * den3 * den5 * den_pi),
);

define_aggregate_scale_factor_float!(
    // params
    (
        scale_p2_from: i16, scale_p3_from: i16, scale_p5_from: i16, scale_pi_from: i16,
        scale_p2_to: i16, scale_p3_to: i16, scale_p5_to: i16, scale_pi_to: i16,
    ),
    // diff expressions
    (
        let diff_scale_p2 = scale_p2_from - scale_p2_to;
        let diff_scale_p3 = scale_p3_from - scale_p3_to;
        let diff_scale_p5 = scale_p5_from - scale_p5_to;
        let diff_scale_pi = scale_pi_from - scale_pi_to;
    ),
    // pow expressions
    (
        let pow_2 = crate::scale_conversion::pow2_float(diff_scale_p2 as i32);
        let pow_3 = crate::scale_conversion::pow3_float(diff_scale_p3 as i32);
        let pow_5 = crate::scale_conversion::pow5_float(diff_scale_p5 as i32);
        let pow_pi = crate::scale_conversion::pow_pi_float(diff_scale_pi as i32);
    ),
    // final expression
    (pow_2 * pow_3 * pow_5 * pow_pi),
);

#[doc(hidden)]
macro_rules! define_float_rescale {
    ($rescale_fn:ident, $T:ty) => {
        $crate::_define_float_rescale!(
            (
                const MASS_EXPONENT: i16,
                const LENGTH_EXPONENT: i16,
                const TIME_EXPONENT: i16,
                const CURRENT_EXPONENT: i16,
                const TEMPERATURE_EXPONENT: i16,
                const AMOUNT_EXPONENT: i16,
                const LUMINOSITY_EXPONENT: i16,
                const ANGLE_EXPONENT: i16,
                const SCALE_P2_FROM: i16, const SCALE_P2_TO: i16,
                const SCALE_P3_FROM: i16, const SCALE_P3_TO: i16,
                const SCALE_P5_FROM: i16, const SCALE_P5_TO: i16,
                const SCALE_PI_FROM: i16, const SCALE_PI_TO: i16,
                Brand,
            ),
            (
                Quantity<
                    Unit<Scale<_2<SCALE_P2_FROM>, _3<SCALE_P3_FROM>, _5<SCALE_P5_FROM>, _Pi<SCALE_PI_FROM>>,
                    Dimension<_M<MASS_EXPONENT>, _L<LENGTH_EXPONENT>, _T<TIME_EXPONENT>, _I<CURRENT_EXPONENT>, _Θ<TEMPERATURE_EXPONENT>, _N<AMOUNT_EXPONENT>, _J<LUMINOSITY_EXPONENT>, _A<ANGLE_EXPONENT>>>,
                    $T,
                    Brand,
                >
            ),
            (
                Quantity<
                    Unit<Scale<_2<SCALE_P2_TO>, _3<SCALE_P3_TO>, _5<SCALE_P5_TO>, _Pi<SCALE_PI_TO>>,
                    Dimension<_M<MASS_EXPONENT>, _L<LENGTH_EXPONENT>, _T<TIME_EXPONENT>, _I<CURRENT_EXPONENT>, _Θ<TEMPERATURE_EXPONENT>, _N<AMOUNT_EXPONENT>, _J<LUMINOSITY_EXPONENT>, _A<ANGLE_EXPONENT>>>,
                    $T,
                    Brand,
                >
            ),
            (
                SCALE_P2_FROM, SCALE_P3_FROM, SCALE_P5_FROM, SCALE_PI_FROM,
                SCALE_P2_TO, SCALE_P3_TO, SCALE_P5_TO, SCALE_PI_TO,
            ),
            $rescale_fn, $T,
        );
    };
}

#[doc(hidden)]
macro_rules! define_int_rescale {
    ($rescale_fn:ident, $T:ty) => {
        $crate::_define_int_rescale!(
            (
                const MASS_EXPONENT: i16,
                const LENGTH_EXPONENT: i16,
                const TIME_EXPONENT: i16,
                const CURRENT_EXPONENT: i16,
                const TEMPERATURE_EXPONENT: i16,
                const AMOUNT_EXPONENT: i16,
                const LUMINOSITY_EXPONENT: i16,
                const ANGLE_EXPONENT: i16,
                const SCALE_P2_FROM: i16, const SCALE_P2_TO: i16,
                const SCALE_P3_FROM: i16, const SCALE_P3_TO: i16,
                const SCALE_P5_FROM: i16, const SCALE_P5_TO: i16,
                const SCALE_PI_FROM: i16, const SCALE_PI_TO: i16,
                Brand
            ),
            (
                Quantity<
                    Unit<Scale<_2<SCALE_P2_FROM>, _3<SCALE_P3_FROM>, _5<SCALE_P5_FROM>, _Pi<SCALE_PI_FROM>>,
                    Dimension<_M<MASS_EXPONENT>, _L<LENGTH_EXPONENT>, _T<TIME_EXPONENT>, _I<CURRENT_EXPONENT>, _Θ<TEMPERATURE_EXPONENT>, _N<AMOUNT_EXPONENT>, _J<LUMINOSITY_EXPONENT>, _A<ANGLE_EXPONENT>>>,
                    $T,
                    Brand,
                >
            ),
            (
                Quantity<
                    Unit<Scale<_2<SCALE_P2_TO>, _3<SCALE_P3_TO>, _5<SCALE_P5_TO>, _Pi<SCALE_PI_TO>>,
                    Dimension<_M<MASS_EXPONENT>, _L<LENGTH_EXPONENT>, _T<TIME_EXPONENT>, _I<CURRENT_EXPONENT>, _Θ<TEMPERATURE_EXPONENT>, _N<AMOUNT_EXPONENT>, _J<LUMINOSITY_EXPONENT>, _A<ANGLE_EXPONENT>>>,
                    $T,
                    Brand,
                >
            ),
            (
                SCALE_P2_FROM, SCALE_P3_FROM, SCALE_P5_FROM, SCALE_PI_FROM,
                SCALE_P2_TO, SCALE_P3_TO, SCALE_P5_TO, SCALE_PI_TO,
            ),
            $rescale_fn, $T,
        );
    };
}

// Float rescale functions - support f32 and f64 storage types
define_float_rescale!(rescale, f64);
define_float_rescale!(rescale_f64, f64);
define_float_rescale!(rescale_f32, f32);

// Signed integer rescale functions - support i8, i16, i32, i64, i128, isize
define_int_rescale!(rescale_i8, i8);
define_int_rescale!(rescale_i16, i16);
define_int_rescale!(rescale_i32, i32);
define_int_rescale!(rescale_i64, i64);
define_int_rescale!(rescale_i128, i128);
define_int_rescale!(rescale_isize, isize);

// Unsigned integer rescale functions - support u8, u16, u32, u64, u128, usize
define_int_rescale!(rescale_u8, u8);
define_int_rescale!(rescale_u16, u16);
define_int_rescale!(rescale_u32, u32);
define_int_rescale!(rescale_u64, u64);
define_int_rescale!(rescale_u128, u128);
define_int_rescale!(rescale_usize, usize);

#[macro_export]
#[doc(hidden)]
macro_rules! define_arithmetic_signed {
    ($T:ty, $rescale_fn:ident) => {
        $crate::_define_arithmetic_signed!(
        // single dimension, single scale
        (
            const MASS_EXPONENT: i16,
            const LENGTH_EXPONENT: i16,
            const TIME_EXPONENT: i16,
            const CURRENT_EXPONENT: i16,
            const TEMPERATURE_EXPONENT: i16,
            const AMOUNT_EXPONENT: i16,
            const LUMINOSITY_EXPONENT: i16,
            const ANGLE_EXPONENT: i16,
            const SCALE_P2: i16,
            const SCALE_P3: i16,
            const SCALE_P5: i16,
            const SCALE_PI: i16,
            Brand,
        ),
        // multiple dimension, multiple scales
        (
            const MASS_EXPONENT: i16,
            const LENGTH_EXPONENT: i16,
            const TIME_EXPONENT: i16,
            const CURRENT_EXPONENT: i16,
            const TEMPERATURE_EXPONENT: i16,
            const AMOUNT_EXPONENT: i16,
            const LUMINOSITY_EXPONENT: i16,
            const ANGLE_EXPONENT: i16,
            const SCALE_P2: i16,
            const SCALE_P3: i16,
            const SCALE_P5: i16,
            const SCALE_PI: i16,
            const MASS_EXPONENT_1: i16, const MASS_EXPONENT_2: i16,
            const LENGTH_EXPONENT_1: i16, const LENGTH_EXPONENT_2: i16,
            const TIME_EXPONENT_1: i16, const TIME_EXPONENT_2: i16,
            const CURRENT_EXPONENT_1: i16, const CURRENT_EXPONENT_2: i16,
            const TEMPERATURE_EXPONENT_1: i16, const TEMPERATURE_EXPONENT_2: i16,
            const AMOUNT_EXPONENT_1: i16, const AMOUNT_EXPONENT_2: i16,
            const LUMINOSITY_EXPONENT_1: i16, const LUMINOSITY_EXPONENT_2: i16,
            const ANGLE_EXPONENT_1: i16, const ANGLE_EXPONENT_2: i16,
            const SCALE_P2_1: i16, const SCALE_P3_1: i16, const SCALE_P5_1: i16, const SCALE_PI_1: i16,
            const SCALE_P2_2: i16, const SCALE_P3_2: i16, const SCALE_P5_2: i16, const SCALE_PI_2: i16,
            Brand,
        ),
        // inversion parameters
        (
            const INVERSE_MASS_EXPONENT: i16,
            const INVERSE_LENGTH_EXPONENT: i16,
            const INVERSE_TIME_EXPONENT: i16,
            const INVERSE_CURRENT_EXPONENT: i16,
            const INVERSE_TEMPERATURE_EXPONENT: i16,
            const INVERSE_AMOUNT_EXPONENT: i16,
            const INVERSE_LUMINOSITY_EXPONENT: i16,
            const INVERSE_ANGLE_EXPONENT: i16,
            const INVERSE_SCALE_P2: i16,
            const INVERSE_SCALE_P3: i16,
            const INVERSE_SCALE_P5: i16,
            const INVERSE_SCALE_PI: i16,
        ),
        // inversion where clauses
        (
            N<MASS_EXPONENT>: core::ops::Neg<Output = N<INVERSE_MASS_EXPONENT>>,
            N<LENGTH_EXPONENT>: core::ops::Neg<Output = N<INVERSE_LENGTH_EXPONENT>>,
            N<TIME_EXPONENT>: core::ops::Neg<Output = N<INVERSE_TIME_EXPONENT>>,
            N<CURRENT_EXPONENT>: core::ops::Neg<Output = N<INVERSE_CURRENT_EXPONENT>>,
            N<TEMPERATURE_EXPONENT>: core::ops::Neg<Output = N<INVERSE_TEMPERATURE_EXPONENT>>,
            N<AMOUNT_EXPONENT>: core::ops::Neg<Output = N<INVERSE_AMOUNT_EXPONENT>>,
            N<LUMINOSITY_EXPONENT>: core::ops::Neg<Output = N<INVERSE_LUMINOSITY_EXPONENT>>,
            N<ANGLE_EXPONENT>: core::ops::Neg<Output = N<INVERSE_ANGLE_EXPONENT>>,
            N<SCALE_P2>: core::ops::Neg<Output = N<INVERSE_SCALE_P2>>,
            N<SCALE_P3>: core::ops::Neg<Output = N<INVERSE_SCALE_P3>>,
            N<SCALE_P5>: core::ops::Neg<Output = N<INVERSE_SCALE_P5>>,
            N<SCALE_PI>: core::ops::Neg<Output = N<INVERSE_SCALE_PI>>
        ),
        // mul output dimension where clauses
        (
            N<MASS_EXPONENT_1>: core::ops::Add<N<MASS_EXPONENT_2>, Output = N<MASS_EXPONENT>>,
            N<LENGTH_EXPONENT_1>: core::ops::Add<N<LENGTH_EXPONENT_2>, Output = N<LENGTH_EXPONENT>>,
            N<TIME_EXPONENT_1>: core::ops::Add<N<TIME_EXPONENT_2>, Output = N<TIME_EXPONENT>>,
            N<CURRENT_EXPONENT_1>: core::ops::Add<N<CURRENT_EXPONENT_2>, Output = N<CURRENT_EXPONENT>>,
            N<TEMPERATURE_EXPONENT_1>: core::ops::Add<N<TEMPERATURE_EXPONENT_2>, Output = N<TEMPERATURE_EXPONENT>>,
            N<AMOUNT_EXPONENT_1>: core::ops::Add<N<AMOUNT_EXPONENT_2>, Output = N<AMOUNT_EXPONENT>>,
            N<LUMINOSITY_EXPONENT_1>: core::ops::Add<N<LUMINOSITY_EXPONENT_2>, Output = N<LUMINOSITY_EXPONENT>>,
            N<ANGLE_EXPONENT_1>: core::ops::Add<N<ANGLE_EXPONENT_2>, Output = N<ANGLE_EXPONENT>>,
            N<SCALE_P2_1>: core::ops::Add<N<SCALE_P2_2>, Output = N<SCALE_P2>>,
            N<SCALE_P3_1>: core::ops::Add<N<SCALE_P3_2>, Output = N<SCALE_P3>>,
            N<SCALE_P5_1>: core::ops::Add<N<SCALE_P5_2>, Output = N<SCALE_P5>>,
            N<SCALE_PI_1>: core::ops::Add<N<SCALE_PI_2>, Output = N<SCALE_PI>>
        ),
        // div output dimension where clauses
        (
            N<MASS_EXPONENT_1>: core::ops::Sub<N<MASS_EXPONENT_2>, Output = N<MASS_EXPONENT>>,
            N<LENGTH_EXPONENT_1>: core::ops::Sub<N<LENGTH_EXPONENT_2>, Output = N<LENGTH_EXPONENT>>,
            N<TIME_EXPONENT_1>: core::ops::Sub<N<TIME_EXPONENT_2>, Output = N<TIME_EXPONENT>>,
            N<CURRENT_EXPONENT_1>: core::ops::Sub<N<CURRENT_EXPONENT_2>, Output = N<CURRENT_EXPONENT>>,
            N<TEMPERATURE_EXPONENT_1>: core::ops::Sub<N<TEMPERATURE_EXPONENT_2>, Output = N<TEMPERATURE_EXPONENT>>,
            N<AMOUNT_EXPONENT_1>: core::ops::Sub<N<AMOUNT_EXPONENT_2>, Output = N<AMOUNT_EXPONENT>>,
            N<LUMINOSITY_EXPONENT_1>: core::ops::Sub<N<LUMINOSITY_EXPONENT_2>, Output = N<LUMINOSITY_EXPONENT>>,
            N<ANGLE_EXPONENT_1>: core::ops::Sub<N<ANGLE_EXPONENT_2>, Output = N<ANGLE_EXPONENT>>,
            N<SCALE_P2_1>: core::ops::Sub<N<SCALE_P2_2>, Output = N<SCALE_P2>>,
            N<SCALE_P3_1>: core::ops::Sub<N<SCALE_P3_2>, Output = N<SCALE_P3>>,
            N<SCALE_P5_1>: core::ops::Sub<N<SCALE_P5_2>, Output = N<SCALE_P5>>,
            N<SCALE_PI_1>: core::ops::Sub<N<SCALE_PI_2>, Output = N<SCALE_PI>>
        ),
            // other parameters
            $T, rescale_fn
        );
    }
}

#[macro_export]
#[doc(hidden)]
macro_rules! define_arithmetic {
    ($T:ty, $rescale_fn:ident) => {
        $crate::_define_arithmetic!(
        // single dimension, single scale
        (
            const MASS_EXPONENT: i16,
            const LENGTH_EXPONENT: i16,
            const TIME_EXPONENT: i16,
            const CURRENT_EXPONENT: i16,
            const TEMPERATURE_EXPONENT: i16,
            const AMOUNT_EXPONENT: i16,
            const LUMINOSITY_EXPONENT: i16,
            const ANGLE_EXPONENT: i16,
            const SCALE_P2: i16,
            const SCALE_P3: i16,
            const SCALE_P5: i16,
            const SCALE_PI: i16,
            Brand,
        ),
        // multiple dimension, multiple scales
        (
            const MASS_EXPONENT: i16,
            const LENGTH_EXPONENT: i16,
            const TIME_EXPONENT: i16,
            const CURRENT_EXPONENT: i16,
            const TEMPERATURE_EXPONENT: i16,
            const AMOUNT_EXPONENT: i16,
            const LUMINOSITY_EXPONENT: i16,
            const ANGLE_EXPONENT: i16,
            const SCALE_P2: i16,
            const SCALE_P3: i16,
            const SCALE_P5: i16,
            const SCALE_PI: i16,
            const MASS_EXPONENT_1: i16, const MASS_EXPONENT_2: i16,
            const LENGTH_EXPONENT_1: i16, const LENGTH_EXPONENT_2: i16,
            const TIME_EXPONENT_1: i16, const TIME_EXPONENT_2: i16,
            const CURRENT_EXPONENT_1: i16, const CURRENT_EXPONENT_2: i16,
            const TEMPERATURE_EXPONENT_1: i16, const TEMPERATURE_EXPONENT_2: i16,
            const AMOUNT_EXPONENT_1: i16, const AMOUNT_EXPONENT_2: i16,
            const LUMINOSITY_EXPONENT_1: i16, const LUMINOSITY_EXPONENT_2: i16,
            const ANGLE_EXPONENT_1: i16, const ANGLE_EXPONENT_2: i16,
            const SCALE_P2_1: i16, const SCALE_P3_1: i16, const SCALE_P5_1: i16, const SCALE_PI_1: i16,
            const SCALE_P2_2: i16, const SCALE_P3_2: i16, const SCALE_P5_2: i16, const SCALE_PI_2: i16,
            Brand,
        ),
        // inversion parameters
        (
            const INVERSE_MASS_EXPONENT: i16,
            const INVERSE_LENGTH_EXPONENT: i16,
            const INVERSE_TIME_EXPONENT: i16,
            const INVERSE_CURRENT_EXPONENT: i16,
            const INVERSE_TEMPERATURE_EXPONENT: i16,
            const INVERSE_AMOUNT_EXPONENT: i16,
            const INVERSE_LUMINOSITY_EXPONENT: i16,
            const INVERSE_ANGLE_EXPONENT: i16,
            const INVERSE_SCALE_P2: i16,
            const INVERSE_SCALE_P3: i16,
            const INVERSE_SCALE_P5: i16,
            const INVERSE_SCALE_PI: i16,
        ),
        // inversion where clauses
        (
            N<MASS_EXPONENT>: core::ops::Neg<Output = N<INVERSE_MASS_EXPONENT>>,
            N<LENGTH_EXPONENT>: core::ops::Neg<Output = N<INVERSE_LENGTH_EXPONENT>>,
            N<TIME_EXPONENT>: core::ops::Neg<Output = N<INVERSE_TIME_EXPONENT>>,
            N<CURRENT_EXPONENT>: core::ops::Neg<Output = N<INVERSE_CURRENT_EXPONENT>>,
            N<TEMPERATURE_EXPONENT>: core::ops::Neg<Output = N<INVERSE_TEMPERATURE_EXPONENT>>,
            N<AMOUNT_EXPONENT>: core::ops::Neg<Output = N<INVERSE_AMOUNT_EXPONENT>>,
            N<LUMINOSITY_EXPONENT>: core::ops::Neg<Output = N<INVERSE_LUMINOSITY_EXPONENT>>,
            N<ANGLE_EXPONENT>: core::ops::Neg<Output = N<INVERSE_ANGLE_EXPONENT>>,
            N<SCALE_P2>: core::ops::Neg<Output = N<INVERSE_SCALE_P2>>,
            N<SCALE_P3>: core::ops::Neg<Output = N<INVERSE_SCALE_P3>>,
            N<SCALE_P5>: core::ops::Neg<Output = N<INVERSE_SCALE_P5>>,
            N<SCALE_PI>: core::ops::Neg<Output = N<INVERSE_SCALE_PI>>
        ),
        // mul output dimension where clauses
        (
            N<MASS_EXPONENT_1>: core::ops::Add<N<MASS_EXPONENT_2>, Output = N<MASS_EXPONENT>>,
            N<LENGTH_EXPONENT_1>: core::ops::Add<N<LENGTH_EXPONENT_2>, Output = N<LENGTH_EXPONENT>>,
            N<TIME_EXPONENT_1>: core::ops::Add<N<TIME_EXPONENT_2>, Output = N<TIME_EXPONENT>>,
            N<CURRENT_EXPONENT_1>: core::ops::Add<N<CURRENT_EXPONENT_2>, Output = N<CURRENT_EXPONENT>>,
            N<TEMPERATURE_EXPONENT_1>: core::ops::Add<N<TEMPERATURE_EXPONENT_2>, Output = N<TEMPERATURE_EXPONENT>>,
            N<AMOUNT_EXPONENT_1>: core::ops::Add<N<AMOUNT_EXPONENT_2>, Output = N<AMOUNT_EXPONENT>>,
            N<LUMINOSITY_EXPONENT_1>: core::ops::Add<N<LUMINOSITY_EXPONENT_2>, Output = N<LUMINOSITY_EXPONENT>>,
            N<ANGLE_EXPONENT_1>: core::ops::Add<N<ANGLE_EXPONENT_2>, Output = N<ANGLE_EXPONENT>>,
            N<SCALE_P2_1>: core::ops::Add<N<SCALE_P2_2>, Output = N<SCALE_P2>>,
            N<SCALE_P3_1>: core::ops::Add<N<SCALE_P3_2>, Output = N<SCALE_P3>>,
            N<SCALE_P5_1>: core::ops::Add<N<SCALE_P5_2>, Output = N<SCALE_P5>>,
            N<SCALE_PI_1>: core::ops::Add<N<SCALE_PI_2>, Output = N<SCALE_PI>>
        ),
        // div output dimension where clauses
        (
            N<MASS_EXPONENT_1>: core::ops::Sub<N<MASS_EXPONENT_2>, Output = N<MASS_EXPONENT>>,
            N<LENGTH_EXPONENT_1>: core::ops::Sub<N<LENGTH_EXPONENT_2>, Output = N<LENGTH_EXPONENT>>,
            N<TIME_EXPONENT_1>: core::ops::Sub<N<TIME_EXPONENT_2>, Output = N<TIME_EXPONENT>>,
            N<CURRENT_EXPONENT_1>: core::ops::Sub<N<CURRENT_EXPONENT_2>, Output = N<CURRENT_EXPONENT>>,
            N<TEMPERATURE_EXPONENT_1>: core::ops::Sub<N<TEMPERATURE_EXPONENT_2>, Output = N<TEMPERATURE_EXPONENT>>,
            N<AMOUNT_EXPONENT_1>: core::ops::Sub<N<AMOUNT_EXPONENT_2>, Output = N<AMOUNT_EXPONENT>>,
            N<LUMINOSITY_EXPONENT_1>: core::ops::Sub<N<LUMINOSITY_EXPONENT_2>, Output = N<LUMINOSITY_EXPONENT>>,
            N<ANGLE_EXPONENT_1>: core::ops::Sub<N<ANGLE_EXPONENT_2>, Output = N<ANGLE_EXPONENT>>,
            N<SCALE_P2_1>: core::ops::Sub<N<SCALE_P2_2>, Output = N<SCALE_P2>>,
            N<SCALE_P3_1>: core::ops::Sub<N<SCALE_P3_2>, Output = N<SCALE_P3>>,
            N<SCALE_P5_1>: core::ops::Sub<N<SCALE_P5_2>, Output = N<SCALE_P5>>,
            N<SCALE_PI_1>: core::ops::Sub<N<SCALE_PI_2>, Output = N<SCALE_PI>>
        ),
            // other parameters
            $T, rescale_fn
        );
    }
}

// Float arithmetic implementations - signed numeric types (support negation)
define_arithmetic_signed!(f32, rescale_f32);
define_arithmetic_signed!(f64, rescale_f64);

// Signed integer arithmetic implementations (support negation)
define_arithmetic_signed!(i8, rescale_i8);
define_arithmetic_signed!(i16, rescale_i16);
define_arithmetic_signed!(i32, rescale_i32);
define_arithmetic_signed!(i64, rescale_i64);
define_arithmetic_signed!(i128, rescale_i128);
define_arithmetic_signed!(isize, rescale_isize);

// Unsigned integer arithmetic implementations (no negation)
define_arithmetic!(u8, rescale_u8);
define_arithmetic!(u16, rescale_u16);
define_arithmetic!(u32, rescale_u32);
define_arithmetic!(u64, rescale_u64);
define_arithmetic!(u128, rescale_u128);
define_arithmetic!(usize, rescale_usize);

// Display traits for all supported types
#[cfg(feature = "alloc")]
define_display_traits!(
    (
        const MASS_EXPONENT: i16,
        const LENGTH_EXPONENT: i16,
        const TIME_EXPONENT: i16,
        const CURRENT_EXPONENT: i16,
        const TEMPERATURE_EXPONENT: i16,
        const AMOUNT_EXPONENT: i16,
        const LUMINOSITY_EXPONENT: i16,
        const ANGLE_EXPONENT: i16,
        const SCALE_P2: i16,
        const SCALE_P3: i16,
        const SCALE_P5: i16,
        const SCALE_PI: i16,
    ),
    (
        MASS_EXPONENT,
        LENGTH_EXPONENT,
        TIME_EXPONENT,
        CURRENT_EXPONENT,
        TEMPERATURE_EXPONENT,
        AMOUNT_EXPONENT,
        LUMINOSITY_EXPONENT,
        ANGLE_EXPONENT,
    ),
    (
        SCALE_P2,
        SCALE_P3,
        SCALE_P5,
        SCALE_PI,
    )
);

/// A [`Display`](fmt::Display) adapter that renders a quantity as `value unit`,
/// omitting the `Quantity<…, T>` type annotation that the default `Display`
/// impl appends.
///
/// Obtain one via [`UnitDisplayExt::unit_display`]. This is useful when many
/// quantities are printed together (e.g. the cells of a matrix), where the
/// repeated type annotation is noise.
#[cfg(feature = "alloc")]
pub struct UnitDisplay {
    value: f64,
    dimensions: whippyunits_core::dimension_exponents::DynDimensionExponents,
    scale: whippyunits_core::scale_exponents::ScaleExponents,
}

#[cfg(feature = "alloc")]
impl fmt::Display for UnitDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pretty = pretty_print_unit_only(self.value, self.dimensions, self.scale);
        f.write_str(&pretty)
    }
}

/// Extension trait providing [`unit_display`](Self::unit_display), a `value unit`
/// renderer without the bracketed storage-type annotation.
#[cfg(feature = "alloc")]
pub trait UnitDisplayExt {
    /// Returns a [`Display`](fmt::Display) adapter that renders `value unit`.
    fn unit_display(&self) -> UnitDisplay;
}

#[cfg(feature = "alloc")]
impl<
    const MASS_EXPONENT: i16,
    const LENGTH_EXPONENT: i16,
    const TIME_EXPONENT: i16,
    const CURRENT_EXPONENT: i16,
    const TEMPERATURE_EXPONENT: i16,
    const AMOUNT_EXPONENT: i16,
    const LUMINOSITY_EXPONENT: i16,
    const ANGLE_EXPONENT: i16,
    const SCALE_P2: i16,
    const SCALE_P3: i16,
    const SCALE_P5: i16,
    const SCALE_PI: i16,
    T,
    Brand,
> UnitDisplayExt
    for Quantity<
        Unit<
            Scale<_2<SCALE_P2>, _3<SCALE_P3>, _5<SCALE_P5>, _Pi<SCALE_PI>>,
            Dimension<
                _M<MASS_EXPONENT>,
                _L<LENGTH_EXPONENT>,
                _T<TIME_EXPONENT>,
                _I<CURRENT_EXPONENT>,
                _Θ<TEMPERATURE_EXPONENT>,
                _N<AMOUNT_EXPONENT>,
                _J<LUMINOSITY_EXPONENT>,
                _A<ANGLE_EXPONENT>,
            >,
        >,
        T,
        Brand,
    >
where
    T: Copy + num_traits::NumCast,
{
    fn unit_display(&self) -> UnitDisplay {
        let value = <f64 as num_traits::NumCast>::from(self.unsafe_value)
            .expect("unable to convert numeric value to f64 for display");
        UnitDisplay {
            value,
            dimensions: whippyunits_core::dimension_exponents::DynDimensionExponents([
                MASS_EXPONENT,
                LENGTH_EXPONENT,
                TIME_EXPONENT,
                CURRENT_EXPONENT,
                TEMPERATURE_EXPONENT,
                AMOUNT_EXPONENT,
                LUMINOSITY_EXPONENT,
                ANGLE_EXPONENT,
            ]),
            scale: whippyunits_core::scale_exponents::ScaleExponents([
                SCALE_P2, SCALE_P3, SCALE_P5, SCALE_PI,
            ]),
        }
    }
}

/// Renders a quantity type's unit label (e.g. `m/s`, `s⁻²`) from its type-level
/// scale and dimension alone — no value, and no instance required.
///
/// Because [`unit_label`](Self::unit_label) is an associated function reading
/// only const generics, callers that have a `Quantity` *type* but no value (for
/// example, code that wants to print the units in a matrix's row/column margins)
/// can obtain the label with `<Q as UnitLabel>::unit_label()`. Returns an empty
/// string for a dimensionless, unscaled unit.
#[cfg(feature = "alloc")]
pub trait UnitLabel {
    /// Returns this unit's label, or an empty string if dimensionless.
    fn unit_label() -> crate::alloc::String;
}

#[cfg(feature = "alloc")]
impl<
    const MASS_EXPONENT: i16,
    const LENGTH_EXPONENT: i16,
    const TIME_EXPONENT: i16,
    const CURRENT_EXPONENT: i16,
    const TEMPERATURE_EXPONENT: i16,
    const AMOUNT_EXPONENT: i16,
    const LUMINOSITY_EXPONENT: i16,
    const ANGLE_EXPONENT: i16,
    const SCALE_P2: i16,
    const SCALE_P3: i16,
    const SCALE_P5: i16,
    const SCALE_PI: i16,
    T,
    Brand,
> UnitLabel
    for Quantity<
        Unit<
            Scale<_2<SCALE_P2>, _3<SCALE_P3>, _5<SCALE_P5>, _Pi<SCALE_PI>>,
            Dimension<
                _M<MASS_EXPONENT>,
                _L<LENGTH_EXPONENT>,
                _T<TIME_EXPONENT>,
                _I<CURRENT_EXPONENT>,
                _Θ<TEMPERATURE_EXPONENT>,
                _N<AMOUNT_EXPONENT>,
                _J<LUMINOSITY_EXPONENT>,
                _A<ANGLE_EXPONENT>,
            >,
        >,
        T,
        Brand,
    >
{
    fn unit_label() -> crate::alloc::String {
        pretty_print_unit_label(
            whippyunits_core::dimension_exponents::DynDimensionExponents([
                MASS_EXPONENT,
                LENGTH_EXPONENT,
                TIME_EXPONENT,
                CURRENT_EXPONENT,
                TEMPERATURE_EXPONENT,
                AMOUNT_EXPONENT,
                LUMINOSITY_EXPONENT,
                ANGLE_EXPONENT,
            ]),
            whippyunits_core::scale_exponents::ScaleExponents([
                SCALE_P2, SCALE_P3, SCALE_P5, SCALE_PI,
            ]),
        )
    }
}
