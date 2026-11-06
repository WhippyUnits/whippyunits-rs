//! Numeric scale factors in unit expressions (`100 * rot / min`, `0.6 * m`).

use whippyunits::api::{rescale, rescale_f32};
use whippyunits::{qty, quantity, value};

/// Factors on the 2·3·5 grid give the same type as the named unit.
#[test]
fn factors_match_named_units() {
    let _: qty!(km) = <qty!(1000 * m)>::new(1.0);
    let _: qty!(km) = <qty!(10^3 * m)>::new(1.0);
    let _: qty!(km) = <qty!(2^3 * 125 * m)>::new(1.0);
    let _: qty!(mm) = <qty!(0.001 * m)>::new(1.0);
    let _: qty!(ms) = <qty!(1e-3 * s)>::new(1.0);
    let _: qty!(ms) = <qty!(s / 1000)>::new(1.0);
    let _: qty!(min) = <qty!(60 * s)>::new(1.0);
    let _: qty!(1 / s) = <qty!(Hz)>::new(1.0);
}

#[test]
fn factors_rescale_exactly() {
    let per_count: qty!(rot / s, f32) = rescale_f32(<qty!(100 * rot / min, f32)>::new(1.0));
    assert_eq!(value!(per_count, rot / s, f32), 100.0 / 60.0);

    // 3·5⁻¹ is applied as 3 × 0.2, which is not exactly 0.6.
    let m: qty!(m) = rescale(<qty!(0.6 * m)>::new(10.0));
    assert!((value!(m, m) - 6.0).abs() < 1e-12);

    let m: qty!(m) = rescale(<qty!(2.5e3 * m)>::new(2.0));
    assert_eq!(value!(m, m), 5000.0);
}

#[test]
fn factors_in_value_macros() {
    let speed = quantity!(3.0, 100 * rot / min);
    assert_eq!(value!(speed, rot / min), 300.0);
    assert_eq!(value!(quantity!(300.0, rot / min), 100 * rot / min), 3.0);
}
