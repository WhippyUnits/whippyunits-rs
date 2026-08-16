//! Exposing the `L`/`U`/`P` factors via `lu` requires a *uniform row space*:
//! partial pivoting permutes `RowDims` at runtime, so a non-uniform row list has
//! no statically-known permuted type (and the pivot would compare incommensurable
//! magnitudes down a column). Rows `[m, s]` disagree, so `RowDims: CollapseUniform`
//! has no impl and `.lu()` must fail to compile. (The pivot-agnostic `solve` needs
//! no such gate.)

use nalgebra::SMatrix;
use whippyalgebra::{dims, nalgebra::MixedUnitMatrix};

type M2 = SMatrix<f64, 2, 2>;

fn main() {
    // Row units are `m` and `s`: not a uniform row space.
    let a = MixedUnitMatrix::<dims![m, s], dims![s, s], M2>::new(M2::identity());

    let _bad = a.lu();
}
