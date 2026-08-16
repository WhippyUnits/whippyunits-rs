//! Type Assertions for Rescale Operations
//!
//! This example shows how to use qty!() to specify target types
//! for rescale operations, ensuring type safety and clarity.

use whippyunits::{qty, rescale};

#[culit::culit(whippyunits::default_declarators::literals)]
fn main() {
    println!("Type Assertions for Rescale Operations");
    println!("======================================\n");

    // You can use qty!() to specify the target type for a rescale operation:
    let distance_mm: qty!(mm) = rescale!(1.0m, mm);
    println!("   Result: {}", distance_mm);

    // dimensionally invalid rescale operation will compile error:
    // let distance_km: qty!(km) = rescale!(1.0m, km); // ❌ Compile error!
}
