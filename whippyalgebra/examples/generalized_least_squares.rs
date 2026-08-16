//! Generalized (weighted) least squares on a mixed matrix.

use whippyalgebra::nalgebra::{MixedUnitMatrix, SMatrix, mixed_unit_matrix};
use whippyalgebra::dims;
use whippyunits::api::UnitDisplayExt;
use whippyunits::{qty, quantity};

/// The mixed design matrix `A : ⟨[m, m/s, m], [m, m/s]⟩` (3×2).
type Design = MixedUnitMatrix<dims![m, m / s, m], dims![m, m / s], SMatrix<f64, 3, 2>>;
/// The measurement column `b : ⟨[m, m/s, m], [1]⟩` (3×1).
type Obs = MixedUnitMatrix<dims![m, m / s, m], dims![1], SMatrix<f64, 3, 1>>;
/// The measurement-space metric `g_r : ⟨[1/m, s/m, 1/m], [m, m/s, m]⟩` (3×3) — the
/// inverse noise covariance `Σ⁻¹`.
type MeasMetric = MixedUnitMatrix<dims![1 / m, s / m, 1 / m], dims![m, m / s, m], SMatrix<f64, 3, 3>>;
/// The state-space metric `g_c : ⟨[1/m, s/m], [m, m/s]⟩` (2×2).
type StateMetric = MixedUnitMatrix<dims![1 / m, s / m], dims![m, m / s], SMatrix<f64, 2, 2>>;
/// The estimated state `x̂ : ⟨[m, m/s], [1]⟩` (2×1): position and velocity.
type State = MixedUnitMatrix<dims![m, m / s], dims![1], SMatrix<f64, 2, 1>>;

fn main() {
    // Truth: p = 10 m, v = 3 m/s ⇒ noiseless (b1, b2, b3) = (10, 3, 10 + 2·3 = 16).
    // A : rows are the three sensors, columns are (p, v). `mixed_unit_matrix!`
    // unit-checks every entry against `RowDims[i]/ColDims[j]` — the dimensionless
    // cells (`m/m`, `(m/s)/(m/s)`) take a bare scalar, the rest a typed `quantity!`.
    // Entry A[2,1] is the dead-reckoning time constant τ = 2 s.
    let a = mixed_unit_matrix! {Design;
        [1.0,                   quantity!(0.0, s)], // sensor 1: p          [m/m, s]
        [quantity!(0.0, 1 / s), 1.0],               // sensor 2: v          [1/s, 1]
        [1.0,                   quantity!(2.0, s)], // sensor 3: p + τ·v    [m/m, s]
    };

    // Noisy readings: the rangefinder is spot-on, the Doppler is close, but the
    // dead-reckoning sensor is badly off (reads 12.5 m against a true 16 m). Each
    // row carries its own sensor's unit.
    let b = mixed_unit_matrix! {Obs;
        [quantity!(10.03, m)],
        [quantity!(3.10, m / s)],
        [quantity!(12.5, m)],
    };

    // g_r = Σ⁻¹: precisions 1/σ² = (100, 11.11, 0.25) in each sensor's own unit²,
    // with a small −1 cross-term between the two position sensors (1 & 3). The
    // entry units spell the "inverse variance" reading out loud — 1/m² for the two
    // position sensors, s²/m² for the velocity sensor.
    let g_r = mixed_unit_matrix! {MeasMetric;
        [quantity!(100.0, 1 / m ^ 2), quantity!(0.0, s / m ^ 2),       quantity!(-1.0, 1 / m ^ 2)],
        [quantity!(0.0, s / m ^ 2),   quantity!(11.11, s ^ 2 / m ^ 2), quantity!(0.0, s / m ^ 2)],
        [quantity!(-1.0, 1 / m ^ 2),  quantity!(0.0, s / m ^ 2),       quantity!(0.25, 1 / m ^ 2)],
    };
    // The state-space metric — canonical here (position and velocity at unit
    // scale). It fixes the result type and, for an overdetermined full-rank fit
    // like this one, does not otherwise move the estimate.
    let g_c = mixed_unit_matrix! {StateMetric;
        [quantity!(1.0, 1 / m ^ 2), quantity!(0.0, s / m ^ 2)],
        [quantity!(0.0, s / m ^ 2), quantity!(1.0, s ^ 2 / m ^ 2)],
    };

    // Generalized (weighted) least squares: x̂ = A⁺_{g_r,g_c} · b.
    let pinv = a
        .generalized_pseudo_inverse(&g_r, &g_c, 1e-12)
        .expect("SPD metrics, full column rank");
    let x: State = pinv * b;

    let (p, v): (qty!(m), qty!(m / s)) = (x.get::<0, 0>(), x.get::<1, 0>());

    println!("generalized least squares from 3 heterogeneous sensors");
    println!("(truth: p = 10 m, v = 3 m/s)");
    println!();
    println!("  x̂ (weighted by the inverse noise covariance g_r):");
    println!("    p = {}   v = {}", p.unit_display(), v.unit_display());
    println!();
    println!("  The noisy dead-reckoning sensor is down-weighted by g_r, so the");
    println!("  estimate stays on the truth instead of being dragged toward it.");
}
