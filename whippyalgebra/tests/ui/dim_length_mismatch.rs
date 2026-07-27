//! `new` checks that a statically-sized matrix's numeric shape matches the
//! dimension-vector lengths, and does so at *compile time* (a zero-cost `const`
//! assertion). A `RowDims` list whose length disagrees with the matrix's row
//! count must therefore fail to compile, never slipping through to a runtime
//! panic.

use nalgebra::SMatrix;
use whippyalgebra::{dims, nalgebra::MixedUnitMatrix};

type M6x3 = SMatrix<f64, 6, 3>;

// The matrix has 6 rows, but `RowDims` lists only 5 entries.
type RowDims = dims![m, m, m, m, m];
type ColDims = dims![1, 1, 1];
type Bad = MixedUnitMatrix<RowDims, ColDims, M6x3>;

fn main() {
    let _bad = Bad::new(M6x3::zeros());
}
