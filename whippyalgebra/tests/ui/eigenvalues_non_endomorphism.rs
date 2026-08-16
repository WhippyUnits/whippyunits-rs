//! `eigenvalues` needs the diagonal to carry a single shared unit (a *uniform
//! endomorphism*): `det(A − λI) = 0` forces every `λ` to share the diagonal
//! unit, so a matrix whose diagonal ratios `RowDims[i] / ColDims[i]` disagree has
//! no well-typed spectrum. Here the diagonal is `[m, s]` — two different units —
//! so `RowDims: UniformDiag<ColDims>` has no impl and `.eigenvalues()` must fail
//! to compile.

use nalgebra::SMatrix;
use whippyalgebra::{dims, nalgebra::MixedUnitMatrix};

type M2 = SMatrix<f64, 2, 2>;

fn main() {
    // Diagonal units are `m` and `s`: not a uniform endomorphism.
    let a = MixedUnitMatrix::<dims![m, s], dims![1, 1], M2>::new(M2::identity());

    let _bad = a.eigenvalues();
}
