//! A *uniform* Cholesky factor lives in `√U`, which is representable only when
//! every exponent of `U` is even. A uniform matrix whose single entry unit is
//! `m` (an odd length exponent) has no such factor, so `.cholesky()` must fail
//! to compile: the `U: UnitSqrt` bound is unsatisfiable because the numeral-level
//! `Halve` gate (typenum `PartialDiv` by 2) rejects the odd exponent.
//!
//! The mixed-metric Cholesky has no such restriction — its columns collapse to
//! dimensionless — so this limitation is specific to the uniform variant.

use nalgebra::SMatrix;
use whippyalgebra::{dims, nalgebra::MixedUnitMatrix, nalgebra::UniformUnitMatrix};
use whippyunits::unit;

type M2 = SMatrix<f64, 2, 2>;

fn main() {
    // Entry unit m (rows [m, m] over dimensionless cols) — odd exponent, no √.
    let m: UniformUnitMatrix<unit!(m), M2> =
        MixedUnitMatrix::<dims![m, m], dims![1, 1], M2>::new(M2::zeros()).into_uniform();

    // `unit!(m)` is not a perfect square, so there is no `UnitSqrt` impl and this
    // does not compile.
    let _bad = m.cholesky();
}
