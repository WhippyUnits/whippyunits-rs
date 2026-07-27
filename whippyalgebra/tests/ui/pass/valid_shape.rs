//! A correctly-shaped construction compiles cleanly, and its shape check is
//! zero-cost: `new` performs a compile-time `const` assertion that the matrix's
//! static row/column counts equal the dimension-vector lengths, emitting no
//! runtime code. See `dim_length_mismatch.rs` for the failing dual.
//!
//! This `pass` case also matters mechanically: trybuild only runs `cargo build`
//! (full monomorphization) when a run contains at least one `pass` test, and the
//! post-monomorphization `const` assertion is invisible to the `cargo check`
//! used for a compile-fail-only run.

use nalgebra::SMatrix;
use whippyalgebra::{dims, nalgebra::MixedUnitMatrix};

type M6x3 = SMatrix<f64, 6, 3>;

// 6 rows, 3 columns — matching the 6x3 matrix exactly.
type RowDims = dims![m, m, m, m, m, m];
type ColDims = dims![1, 1, 1];
type Valid = MixedUnitMatrix<RowDims, ColDims, M6x3>;

fn main() {
    let _ok = Valid::new(M6x3::zeros());
}
