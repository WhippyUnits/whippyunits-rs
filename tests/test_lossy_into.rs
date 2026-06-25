use whippyunits::{value, quantity, rescale};

#[test]
fn test_f64_to_f32() {
    let temp_f64 = quantity!(300.0, K);
    let temp_f32 = temp_f64.lossy_into::<f32>();
    assert!((temp_f32.unsafe_value - 300.0_f32).abs() < 1e-3);
}

#[test]
fn test_f32_to_f64() {
    let temp_f32 = quantity!(300.0, K, f32);
    let temp_f64 = temp_f32.lossy_into::<f64>();
    assert!((temp_f64.unsafe_value - 300.0).abs() < 1e-5);
}

#[test]
fn test_identity() {
    let q = quantity!(123.456, m);
    let q2 = q.lossy_into::<f64>();
    assert_eq!(q.unsafe_value, q2.unsafe_value);
}

#[test]
fn test_preserves_scale() {
    let dist_km = quantity!(1.5, km);
    let dist_km_f32 = dist_km.lossy_into::<f32>();
    assert!((dist_km_f32.unsafe_value - 1.5_f32).abs() < 1e-5);
}

#[test]
fn test_rescaled() {
    let dist_km = quantity!(1.5, km);
    let dist_m = rescale!(dist_km, m);
    let dist_m_f32 = dist_m.lossy_into::<f32>();
    assert!((dist_m_f32.unsafe_value - 1500.0_f32).abs() < 1e-5);
}

#[test]
fn test_preserves_compound_dimension() {
    let v_f64 = quantity!(9.8, m/s^2);
    let v_f32 = v_f64.lossy_into::<f32>();
    assert!((value!(v_f32, m/s^2, f32) - 9.8_f32).abs() < 1e-4);
}

#[test]
fn test_chain() {
    let q_f64 = quantity!(42.0, K);
    let q_f32 = q_f64.lossy_into::<f32>();
    let q_f64 = q_f32.lossy_into::<f64>();
    assert!((q_f64.unsafe_value - 42.0_f64).abs() < 1e-4);
}

#[test]
fn test_from_large_value() {
    // 1e30 fits in f64 but loses significant bits in f32.
    let large_f64 = quantity!(1e30_f64, kg);
    let large_f32 = large_f64.lossy_into::<f32>();
    assert!(value!(large_f32, kg, f32).is_finite());
}
