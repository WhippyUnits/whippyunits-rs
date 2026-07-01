//! Type Conversions Demo
//!
//! This example demonstrates `lossless_into` and `lossy_into` for converting
//! the storage type of a quantity while preserving its scale, dimension, and brand.

#![cfg_attr(has_generic_const_exprs, feature(generic_const_exprs))]
#![cfg_attr(has_generic_const_exprs, allow(incomplete_features))]

use whippyunits::quantity;

fn main() {
    println!("Type Conversions Demo");
    println!("=====================\n");

    // ============================================================
    // lossless_into — guaranteed no information loss
    // ============================================================
    // Uses Rust's standard `From` trait, so only widening conversions
    // (e.g. f32 → f64, u16 → i32) are available.

    println!("lossless_into (guaranteed no information loss):");

    let distance_f32 = quantity!(6.9, kg, f32);
    let distance_f64 = distance_f32.lossless_into::<f64>();
    println!("  f32 → f64: {} → {}", distance_f32, distance_f64);

    let count_u16 = quantity!(1000, m, u16);
    let count_i32 = count_u16.lossless_into::<i32>();
    println!("  u16 → i32: {} → {}", count_u16, count_i32);

    println!();

    // ============================================================
    // lossy_into — potentially lossy, all primitive types supported
    // ============================================================
    // Uses the `LossyFrom` trait, which is implemented between all
    // primitive numeric types via `as` casts.

    println!("lossy_into (potentially lossy, all types):");

    // f64 → f32: may lose precision
    let distance = quantity!(6.9, kg);
    let distance_f32 = distance.lossy_into::<f32>();
    println!("  f64 → f32: {} → {}", distance, distance_f32);

    // f32 → f64: lossless in practice, but also available here
    let distance_f64 = distance_f32.lossy_into::<f64>();
    println!("  f32 → f64: {} → {}", distance_f32, distance_f64);

    // float → integer: truncates toward zero
    let speed = quantity!(9.8, m/s^2);
    let speed_i32 = speed.lossy_into::<i32>();
    println!("  f64 → i32: {} → {} (truncated)", speed, speed_i32);

    // integer → float
    let mass = quantity!(42, kg, i32);
    let mass_f64 = mass.lossy_into::<f64>();
    println!("  i32 → f64: {} → {}", mass, mass_f64);

    // integer → integer (narrowing)
    let big = quantity!(100, m, i64);
    let small = big.lossy_into::<i8>();
    println!("  i64 → i8:  {} → {}", big, small);

    // signed → unsigned
    let signed = quantity!(42, m, i32);
    let unsigned = signed.lossy_into::<u32>();
    println!("  i32 → u32: {} → {}", signed, unsigned);

    println!();

    // ============================================================
    // Chaining conversions
    // ============================================================

    println!("Chaining conversions:");
    let original = quantity!(42, m, i32);
    let as_f64 = original.lossy_into::<f64>();
    let as_f32 = as_f64.lossy_into::<f32>();
    let back = as_f32.lossy_into::<i32>();
    println!("  i32 → f64 → f32 → i32: {} → {} → {} → {}", original, as_f64, as_f32, back);
}
