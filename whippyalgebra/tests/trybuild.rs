#![cfg(feature = "nalgebra")]
//! Compile-fail tests: dimensionally inconsistent operations must not compile.

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    // A `pass` case forces trybuild to run `cargo build` rather than `cargo
    // check`, so post-monomorphization `const` assertions (e.g. the zero-cost
    // shape check in `MixedUnitMatrix::new`) are actually evaluated and their
    // compile errors observed by the `compile_fail` cases below.
    t.pass("tests/ui/pass/*.rs");
    t.compile_fail("tests/ui/*.rs");
}
