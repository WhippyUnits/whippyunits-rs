use whippyunits::api::rescale_f32;
use whippyunits::default_declarators::*;
use whippyunits::{qty, quantity, value};

#[test]
fn standard_gravity_round_trips() {
    let a = quantity!(1.0, g_n);
    assert!((value!(a, m / s ^ 2) - 9.80665).abs() < 1e-12);
    assert!((value!(a, g_n) - 1.0).abs() < 1e-12);
    assert!((value!(2.0.standard_gravity(), m / s ^ 2) - 19.6133).abs() < 1e-12);
}

#[test]
fn standard_gravity_is_an_acceleration() {
    let a: qty!(m / s ^ 2, f32) = rescale_f32(quantity!(0.5, g_n, f32));
    assert!((value!(a, m / s ^ 2, f32) - 4.903325).abs() < 1e-6);
    let v: qty!(m / s, f32) = a * quantity!(2.0, s, f32);
    assert!((value!(v, m / s, f32) - 9.80665).abs() < 1e-5);
    assert!((value!(a, g_n, f32) - 0.5).abs() < 1e-6);
}

/// Adding a dimension whose only unit is non-storage leaves every other
/// dimension's base unit where it was (its first listed unit).
#[test]
fn base_unit_is_first_unit_except_acceleration() {
    use whippyunits_core::Dimension;
    for dimension in Dimension::ALL {
        match dimension.name {
            "Acceleration" => assert!(dimension.base_unit().is_none()),
            _ => assert_eq!(
                dimension.base_unit().map(|u| u.name),
                dimension.units.first().map(|u| u.name),
                "{}",
                dimension.name
            ),
        }
    }
}

/// Like any non-storage unit, g_n prints as its storage unit (as `ft` prints
/// as `dm`), and a non-storage unit never names a stored acceleration.
#[test]
fn standard_gravity_prints_as_its_storage_unit() {
    let g = format!("{}", quantity!(1.0, g_n));
    assert!(g.contains("da(m·s⁻²)"), "{g}");
    let a = format!("{}", quantity!(9.80665, m / s ^ 2));
    assert!(a.contains("(m·s⁻²)") && !a.contains("g_n"), "{a}");
}
