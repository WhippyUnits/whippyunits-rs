//! Mixed `schur` shares the unitary frame `Q … Qᵀ` with `symmetric_eigen`, so it
//! carries the same dimensional bound: the input must be a metric
//! (`ColDims = 1 / RowDims`). Here the matrix is a square endomorphism
//! `⟨[m, m], [m, m]⟩`, not a metric, so `RowDims: MetricShape<ColDims>` has no impl
//! and `.schur()` must fail to compile. (For a general endomorphism, erase to a
//! `UniformUnitMatrix` and use its unconditional `schur`.)

use nalgebra::SMatrix;
use whippyalgebra::{dims, nalgebra::MixedUnitMatrix};

type M2 = SMatrix<f64, 2, 2>;

fn main() {
    // An endomorphism `⟨[m, m], [m, m]⟩`, not a metric `⟨[m, m], [1/m, 1/m]⟩`.
    let a = MixedUnitMatrix::<dims![m, m], dims![m, m], M2>::new(M2::identity());

    let _bad = a.schur();
}
