#![cfg(feature = "nalgebra")]
//! Standalone exercise of `#[generic_matrix(uniform, ..)]`: a function generic
//! over both rectangular axes (`const N`, `const M`) pseudo-inverts a *uniform*
//! matrix and multiplies the result by a uniform column — with no
//! hand-written `where` clause. The attribute supplies the whole nalgebra-level
//! pile (the `decompose` Householder scratch keyed on `DimMinimum<N, M>`); the
//! single-unit algebra (`Ω → 1/Ω`, then `·V = A`) resolves on its own because
//! the units are concrete.
//!
//! Contrast with the mixed `generic_block`/`generic_matrix` tests: a uniform
//! matrix carries no dimension lists, so there is no `ShapeIndex`/`DimList`
//! plumbing here at all.

use whippyalgebra::nalgebra::{Const, OMatrix, UniformUnitMatrix, generic_matrix};
use whippyunits::unit;

type Transfer = unit!(V / A); // Ω
type Volt = unit!(V);
type Amp = unit!(A);

/// `x = G⁺ b`, uniform-typed: `G` in Ω, `b` in V, `x` in A — doubly generic over
/// the observation count `N` and the unknown count `M`.
#[generic_matrix(uniform, rows(N), cols(M), decompose)]
fn solve_uniform<const N: usize, const M: usize>(
    transfer: UniformUnitMatrix<Transfer, OMatrix<f64, Const<N>, Const<M>>>,
    readings: UniformUnitMatrix<Volt, OMatrix<f64, Const<N>, Const<1>>>,
) -> Result<UniformUnitMatrix<Amp, OMatrix<f64, Const<M>, Const<1>>>, &'static str> {
    let pinv = transfer.pseudo_inverse(1e-12)?;
    Ok(pinv * readings)
}

#[test]
fn uniform_pseudo_inverse_solve_is_doubly_generic() {
    // 4×2: two currents from four noisy readings (truth 2 A, 3 A).
    let g4 = UniformUnitMatrix::<Transfer, OMatrix<f64, Const<4>, Const<2>>>::from_row_slice(&[
        1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0,
    ]);
    let b4 = UniformUnitMatrix::<Volt, OMatrix<f64, Const<4>, Const<1>>>::from_row_slice(&[
        2.01, 2.98, 5.02, -0.97,
    ]);
    let x4 = solve_uniform::<4, 2>(g4, b4).expect("full column rank");
    assert!((x4.nalgebra()[(0, 0)] - 2.02).abs() < 1e-6);
    assert!((x4.nalgebra()[(1, 0)] - 2.99).abs() < 1e-6);

    // 5×3 (truth 1 A, 2 A, 3 A): the *same* generic fn at a second shape.
    let g5 = UniformUnitMatrix::<Transfer, OMatrix<f64, Const<5>, Const<3>>>::from_row_slice(&[
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0,
    ]);
    let b5 = UniformUnitMatrix::<Volt, OMatrix<f64, Const<5>, Const<1>>>::from_row_slice(&[
        1.02, 1.99, 3.01, 5.97, 2.03,
    ]);
    let x5 = solve_uniform::<5, 3>(g5, b5).expect("full column rank");
    assert!((x5.nalgebra()[(0, 0)] - 1.0).abs() < 0.05);
    assert!((x5.nalgebra()[(1, 0)] - 2.0).abs() < 0.05);
    assert!((x5.nalgebra()[(2, 0)] - 3.0).abs() < 0.05);
}
