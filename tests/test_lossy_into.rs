use whippyunits::{value, quantity, rescale};

// ---- f32 <=> f64 ----

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
    let large_f64 = quantity!(1e30_f64, kg);
    let large_f32 = large_f64.lossy_into::<f32>();
    assert!(value!(large_f32, kg, f32).is_finite());
}

// ---- float <=> integer ----

#[test]
fn test_f64_to_i32() {
    let q = quantity!(42.0, m);
    let q_i32 = q.lossy_into::<i32>();
    assert_eq!(q_i32.unsafe_value, 42_i32);
}

#[test]
fn test_f64_to_i32_truncates() {
    let q = quantity!(3.7, m);
    let q_i32 = q.lossy_into::<i32>();
    assert_eq!(q_i32.unsafe_value, 3_i32);
}

#[test]
fn test_i32_to_f64() {
    let q = quantity!(42, m, i32);
    let q_f64 = q.lossy_into::<f64>();
    assert!((q_f64.unsafe_value - 42.0).abs() < 1e-10);
}

#[test]
fn test_i32_to_f32() {
    let q = quantity!(1000, kg, i32);
    let q_f32 = q.lossy_into::<f32>();
    assert!((q_f32.unsafe_value - 1000.0_f32).abs() < 1e-3);
}

#[test]
fn test_f32_to_u8() {
    let q = quantity!(200.0, m, f32);
    let q_u8 = q.lossy_into::<u8>();
    assert_eq!(q_u8.unsafe_value, 200_u8);
}

#[test]
fn test_u8_to_f64() {
    let q = quantity!(255, m, u8);
    let q_f64 = q.lossy_into::<f64>();
    assert!((q_f64.unsafe_value - 255.0).abs() < 1e-10);
}

// ---- integer <=> integer ----

#[test]
fn test_i32_to_i64() {
    let q = quantity!(42, m, i32);
    let q_i64 = q.lossy_into::<i64>();
    assert_eq!(q_i64.unsafe_value, 42_i64);
}

#[test]
fn test_i64_to_i32() {
    let q = quantity!(42, m, i64);
    let q_i32 = q.lossy_into::<i32>();
    assert_eq!(q_i32.unsafe_value, 42_i32);
}

#[test]
fn test_u8_to_u64() {
    let q = quantity!(100, m, u8);
    let q_u64 = q.lossy_into::<u64>();
    assert_eq!(q_u64.unsafe_value, 100_u64);
}

#[test]
fn test_i32_to_u32() {
    let q = quantity!(42, m, i32);
    let q_u32 = q.lossy_into::<u32>();
    assert_eq!(q_u32.unsafe_value, 42_u32);
}

#[test]
fn test_u16_to_i128() {
    let q = quantity!(5000, kg, u16);
    let q_i128 = q.lossy_into::<i128>();
    assert_eq!(q_i128.unsafe_value, 5000_i128);
}

// ---- cross-type with compound dimensions ----

#[test]
fn test_cross_type_compound_dimension() {
    let v = quantity!(10, m/s^2, i32);
    let v_f64 = v.lossy_into::<f64>();
    assert!((value!(v_f64, m/s^2) - 10.0).abs() < 1e-10);
}

// ---- cross-type chain ----

#[test]
fn test_cross_type_chain() {
    let q = quantity!(42, m, i32);
    let q = q.lossy_into::<f64>();
    let q = q.lossy_into::<f32>();
    let q = q.lossy_into::<i64>();
    assert_eq!(q.unsafe_value, 42_i64);
}
