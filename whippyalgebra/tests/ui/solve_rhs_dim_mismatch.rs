//! `solve` requires the right-hand side to live in the same output-space as the
//! matrix: `b`'s row dimension vector must equal the matrix's `RowDims`. Here
//! `A` has row dims [m, m] but `b` has row dims [s, s], so the system is
//! dimensionally ill-posed and `A.solve(&b)` must fail to compile.

use nalgebra::SMatrix;
use whippyalgebra::{dims, nalgebra::MixedUnitMatrix};

type M2 = SMatrix<f64, 2, 2>;
type V2 = SMatrix<f64, 2, 1>;

// A maps input-space [s, s] into output-space [m, m].
type A = MixedUnitMatrix<dims![m, m], dims![s, s], M2>;
// b's rows are [s, s] — the wrong space (they must be [m, m] to match A).
type B = MixedUnitMatrix<dims![s, s], dims![1], V2>;

fn main() {
    let a = A::new(M2::zeros());
    let b = B::new(V2::zeros());

    // A expects a right-hand side with row dims [m, m]; `b` has [s, s].
    let _bad = a.solve(&b);
}
