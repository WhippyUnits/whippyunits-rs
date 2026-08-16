//! Type Assertions for Safe Multiplication and Division
//!
//! This example shows how to use qty!() to verify that multiplication
//! and division operations produce the expected dimensions, catching
//! errors at compile time instead of runtime.

use whippyunits::qty;

#[culit::culit(whippyunits::default_declarators::literals)]
fn main() {
    println!("Type Assertions for Safe Multiplication and Division");
    println!("====================================================\n");

    let width = 5.0m;
    let height = 4.0m;

    // Without a type assertion, the operation will compile, but the result will be unchecked.
    let _unchecked_area = width * height; // ⚠️ No compile error if units are wrong

    // With a type assertion, the operation will compile only if the units are correct.
    let _area: qty!(m ^ 2) = width * height; // ✅ Compile error if units are wrong
}
