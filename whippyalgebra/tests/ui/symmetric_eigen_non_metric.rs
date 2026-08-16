//! `symmetric_eigen` needs a metric (`ColDims = 1 / RowDims`): a matrix equal
//! to its own transpose *as a type*. Symmetry of the *numbers* is not enough —
//! here the matrix is a square endomorphism `⟨[m, m], [m, m]⟩`, whose transpose
//! would carry different units, so `RowDims: MetricShape<ColDims>` has no impl and
//! `.symmetric_eigen()` must fail to compile.

use nalgebra::SMatrix;
use whippyalgebra::{dims, nalgebra::MixedUnitMatrix};

type M2 = SMatrix<f64, 2, 2>;

fn main() {
    // An endomorphism `⟨[m, m], [m, m]⟩`, not a metric `⟨[m, m], [1/m, 1/m]⟩`.
    let a = MixedUnitMatrix::<dims![m, m], dims![m, m], M2>::new(M2::identity());

    let _bad = a.symmetric_eigen();
}
