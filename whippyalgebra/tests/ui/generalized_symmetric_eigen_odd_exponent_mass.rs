//! The `UnitSqrt`-gated generalized *symmetric* eigendecomposition reduces the
//! pencil `K v = λ M v` by rooting the mass metric (`M = L Lᴴ`), so the mass
//! factor lives in `√Um` — representable only when every exponent of `Um` is
//! even, exactly like the [uniform Cholesky](UniformUnitMatrix::cholesky). A
//! mass in `kg` (an odd mass exponent) has no uniform root, so
//! `.generalized_symmetric_eigen()` must fail to compile via the unsatisfiable
//! `Um: UnitSqrt` bound.
//!
//! The eigenvalues-only [`generalized_eigenvalues`] path has no such gate (it
//! never forms a root), so this limitation is specific to the symmetric variant.

use nalgebra::SMatrix;
use whippyalgebra::nalgebra::UniformUnitMatrix;
use whippyunits::unit;

type M2 = SMatrix<f64, 2, 2>;

fn main() {
    // A dimensionless stiffness and a mass in kg (odd exponent, no √).
    let k: UniformUnitMatrix<unit!(1), M2> = UniformUnitMatrix::from_nalgebra(M2::identity());
    let m: UniformUnitMatrix<unit!(kg), M2> = UniformUnitMatrix::from_nalgebra(M2::identity());

    // `unit!(kg)` is not a perfect square, so there is no `UnitSqrt` impl for the
    // mass and this does not compile.
    let _bad = k.generalized_symmetric_eigen(&m);
}
