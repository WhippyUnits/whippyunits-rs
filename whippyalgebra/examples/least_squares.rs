//! Unit-safe linear least squares, generic over both matrix axes.
//!
//! We recover `M` unknown branch currents (in `A`) from `N` noisy node-voltage
//! readings (in `V`) related by a rectangular transfer matrix (in `V/A = Ω`),
//! solving the overdetermined system `G x = b` in the least-squares sense.
//!
//! The transfer matrix is *uniform*: every entry is `V/A = Ω`, so it is typed as
//! a single-unit [`UniformUnitMatrix`]. A single-unit matrix *carries* its
//! canonical Euclidean metric, so
//! [`pseudo_inverse`](whippyalgebra::UniformUnitMatrix::pseudo_inverse) is
//! well-posed with no extra ceremony — and, because there are no per-entry
//! dimension lists to thread, the solver stays generic over both `N` and `M`.
//!
//! When the matrix is *genuinely mixed* (heterogeneous units on each axis) it has
//! no built-in metric and one must be supplied; that is a different enough story —
//! and a more interesting one, once the metric is a real inverse-noise covariance
//! rather than the identity — that it lives in the sibling `generalized_least_squares`
//! example.

use whippyalgebra::nalgebra::{
    Const, OMatrix, UniformUnitMatrix, generic_matrix, uniform_unit_matrix,
};
use whippyunits::api::UnitDisplayExt;
use whippyunits::{quantity, unit};

/// The transfer matrix `G` is *uniform*: every entry is `V/A = Ω`.
type Transfer = unit!(V / A);
type Volt = unit!(V);
type Amp = unit!(A);

/// Least-squares solve of `G x = b`, generic over both axes (`N` readings,
/// `M` unknowns). `G` is uniform in `Ω`, so `G⁺` is uniform in `1/Ω = ℧ = A/V`,
/// and `G⁺ · b` (with `b` uniform in `V`) lands uniformly in `A`. The single
/// shared unit *is* the canonical metric, so the pseudo-inverse is sound with the
/// `decompose` bounds alone.
#[generic_matrix(uniform, rows(N), cols(M), decompose)]
fn solve_currents<const N: usize, const M: usize>(
    transfer: UniformUnitMatrix<Transfer, OMatrix<f64, Const<N>, Const<M>>>,
    readings: UniformUnitMatrix<Volt, OMatrix<f64, Const<N>, Const<1>>>,
) -> Result<UniformUnitMatrix<Amp, OMatrix<f64, Const<M>, Const<1>>>, &'static str> {
    // G⁺ : uniform in A/V, shape M × N.
    let pinv = transfer.pseudo_inverse(1e-12)?;
    // G⁺ · b : (A/V)·V = A, shape M × 1.
    Ok(pinv * readings)
}

fn main() {
    // Model: G x = b, entries of G in Ω, b in V, unknown currents x in A. Each
    // system is overdetermined (more readings than unknowns) and mildly noisy, so
    // the pseudo-inverse gives the least-squares fit. The *same* generic solver is
    // instantiated at two distinct rectangular shapes — 4×2 and 5×3.

    // 4 readings, 2 unknown currents (truth x = [2 A, 3 A]) → a 4×2 transfer.
    // Built with `uniform_unit_matrix!`, one unit in the header, shape from the rows.
    let g4 = uniform_unit_matrix![V / A;
        [quantity!(1.0, V / A), quantity!(0.0, V / A)],
        [quantity!(0.0, V / A), quantity!(1.0, V / A)],
        [quantity!(1.0, V / A), quantity!(1.0, V / A)],
        [quantity!(1.0, V / A), quantity!(-1.0, V / A)],
    ];
    let b4 = uniform_unit_matrix![V;
        [quantity!(2.01, V)],  // ≈ 2
        [quantity!(2.98, V)],  // ≈ 3
        [quantity!(5.02, V)],  // ≈ 2 + 3
        [quantity!(-0.97, V)], // ≈ 2 − 3
    ];
    let x4 = solve_currents::<4, 2>(g4, b4).expect("full column rank");
    println!("4×2 solve (2 currents from 4 readings):");
    println!("  i0 = {}   (truth 2 A)", x4.get(0, 0).unit_display());
    println!("  i1 = {}   (truth 3 A)", x4.get(1, 0).unit_display());

    // 5 readings, 3 unknown currents (truth x = [1 A, 2 A, 3 A]) → a 5×3 transfer,
    // the same generic solver at a second shape.
    let g5 = uniform_unit_matrix![V / A;
        [quantity!(1.0, V / A), quantity!(0.0, V / A), quantity!(0.0, V / A)],
        [quantity!(0.0, V / A), quantity!(1.0, V / A), quantity!(0.0, V / A)],
        [quantity!(0.0, V / A), quantity!(0.0, V / A), quantity!(1.0, V / A)],
        [quantity!(1.0, V / A), quantity!(1.0, V / A), quantity!(1.0, V / A)],
        [quantity!(1.0, V / A), quantity!(-1.0, V / A), quantity!(1.0, V / A)],
    ];
    let b5 = uniform_unit_matrix![V;
        [quantity!(1.02, V)],  // ≈ 1
        [quantity!(1.99, V)],  // ≈ 2
        [quantity!(3.01, V)],  // ≈ 3
        [quantity!(5.97, V)],  // ≈ 1 + 2 + 3
        [quantity!(2.03, V)],  // ≈ 1 − 2 + 3
    ];
    let x5 = solve_currents::<5, 3>(g5, b5).expect("full column rank");
    println!("\n5×3 solve (3 currents from 5 readings):");
    println!("  i0 = {}   (truth 1 A)", x5.get(0, 0).unit_display());
    println!("  i1 = {}   (truth 2 A)", x5.get(1, 0).unit_display());
    println!("  i2 = {}   (truth 3 A)", x5.get(2, 0).unit_display());
}
