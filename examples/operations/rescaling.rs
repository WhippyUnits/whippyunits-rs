//! Rescaling Demo
//!
//! This example demonstrates how to rescale quantities to different units
//! of the same dimension using the rescale function.

use whippyunits::{api::rescale, qty, quantity, rescale};

#[culit::culit(whippyunits::default_declarators::literals)]
fn main() {
    println!("Rescaling Demo");
    println!("==============\n");

    println!("1. Length Rescaling:");
    let distance_m = 1.0m;
    let distance_mm: qty!(mm) = rescale(distance_m);
    let distance_km: qty!(km) = rescale(distance_m);
    println!("   {} = {}", distance_m, distance_mm);
    println!("   {} = {}", distance_m, distance_km);

    let distance_cm = rescale!(1.0m, cm);
    println!("   {} = {} (using rescale! macro)", distance_m, distance_cm);

    println!("\n2. Mass Rescaling:");
    let mass_kg = 1.0kg;
    let mass_g: qty!(g) = rescale(mass_kg);
    let mass_mg: qty!(mg) = rescale(mass_kg);
    println!("   {} = {}", mass_kg, mass_g);
    println!("   {} = {}", mass_kg, mass_mg);

    println!("\n3. Time Rescaling:");
    let time_s = 30.0s;
    let time_min: qty!(min) = rescale(time_s);
    let time_ms: qty!(ms) = rescale(time_s);
    println!("   {} = {}", time_s, time_min);
    println!("   {} = {}", time_s, time_ms);

    println!("\n4. Rescaling for Addition:");
    let distance1 = 1.0m;
    let distance2 = 500.0mm;

    let sum_mm = rescale(distance1) + distance2;
    let sum_m = distance1 + rescale(distance2);

    println!("   {} + {} = {}", distance1, distance2, sum_mm);
    println!("   {} + {} = {}", distance1, distance2, sum_m);

    println!("\n5. Rescaling with Different Numeric Types:");
    let distance_i32 = quantity!(1, m, i32);
    let distance_mm_i32 = rescale!(distance_i32, mm, i32);
    println!("   {} (i32) = {} (i32)", distance_i32, distance_mm_i32);

    println!("\n6. Rescaling Compound Units:");
    let velocity = quantity!(100.0, km / h);
    let velocity_ms: qty!(m / s) = rescale(velocity);
    println!("   {} = {}", velocity, velocity_ms);

    println!("\n7. Bidirectional Rescaling:");
    let original = 5.0m;
    let converted: qty!(mm) = rescale(original);
    let back: qty!(m) = rescale(converted);
    println!("   {} -> {} -> {}", original, converted, back);
    assert!((original.unsafe_value - back.unsafe_value).abs() < 1e-10);
}
