//! Two matrices can be added only when their row *and* column dimension vectors
//! (and brand) match exactly. Here the row dimension vectors differ, so there is
//! no `Add` impl between the two operands and the sum must fail to compile.

use nalgebra::SMatrix;
use whippyalgebra::{dims, nalgebra::MixedUnitMatrix};

type M2 = SMatrix<f64, 2, 2>;

// Same shape and column units, but different row units.
type Meters = MixedUnitMatrix<dims![m, m], dims![s, s], M2>;
type Seconds = MixedUnitMatrix<dims![s, s], dims![s, s], M2>;

fn main() {
    let a = Meters::new(M2::zeros());
    let b = Seconds::new(M2::zeros());

    // `a` has row dims [m, m]; `b` has row dims [s, s]: no matching `Add` impl.
    let _bad = a + b;
}
