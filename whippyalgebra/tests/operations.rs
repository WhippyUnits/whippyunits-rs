#![cfg(feature = "nalgebra")]
//! Operation coverage: the unit metadata transforms correctly under
//! negation, subtraction, scalar-by-quantity multiplication/division,
//! transpose, inverse, the determinant, linear solves, endomorphism powers and
//! exponential, the component-wise (Hadamard) product/quotient, the Kronecker
//! product, and dimensionless scaling.

use whippyalgebra::dims;
use whippyalgebra::nalgebra::{
    MixedUnitMatrix, SMatrix, SVector, mixed_unit_matrix, rescale_matrix,
};
use whippyunits::{qty, quantity};

type M2 = SMatrix<f64, 2, 2>;
type M3 = SMatrix<f64, 3, 3>;

// A 2x2 matrix mapping [s, s] (columns) into [m, m] (rows): every entry is m/s.
type RowDims = dims![m, m];
type ColDims = dims![s, s];
type Mat = MixedUnitMatrix<RowDims, ColDims, M2>;

fn sample() -> Mat {
    Mat::new(M2::new(1.0, 2.0, 3.0, 4.0))
}

#[test]
fn shape_metadata() {
    let m = sample();
    // Unit-invariant size queries forward to the inner matrix.
    assert_eq!(m.shape(), (2, 2));
    assert_eq!(m.nrows(), 2);
    assert_eq!(m.ncols(), 2);
    assert_eq!(m.len(), 4);
    assert!(!m.is_empty());
    assert!(m.is_square());
}

#[test]
fn mixed_unit_matrix_macro_builds_row_major() {
    // Row-major: the first `[…]` is row 0, the second row 1 (matching nalgebra's
    // `Matrix::new` argument order). Dimensioned cells take a `Quantity` of exactly
    // their unit (m/s); the macro would reject a wrong unit at compile time (see the
    // `mixed_unit_matrix_wrong_unit` UI test). The 2×2 `f64` storage shape is read
    // from `RowDims`/`ColDims`.
    let m: Mat = mixed_unit_matrix![RowDims, ColDims;
        [quantity!(1.0, m / s), quantity!(2.0, m / s)], // row 0: entries (0,0), (0,1)
        [quantity!(3.0, m / s), quantity!(4.0, m / s)], // row 1: entries (1,0), (1,1)
    ];
    // Same storage as `sample()` (which uses nalgebra's row-major `new`).
    let e00: qty!(m / s) = m.get::<0, 0>();
    let e10: qty!(m / s) = m.get::<1, 0>();
    let e01: qty!(m / s) = m.get::<0, 1>();
    assert_eq!(e00.unsafe_value, 1.0);
    assert_eq!(e10.unsafe_value, 3.0);
    assert_eq!(e01.unsafe_value, 2.0);
    assert_eq!(m.get::<1, 1>().unsafe_value, 4.0);
}

#[test]
fn mixed_unit_matrix_macro_takes_scalar_for_dimensionless() {
    // A matrix whose row and column dims match has dimensionless entries, so the
    // macro accepts bare scalars (no wrapping quantity needed).
    type D = dims![s, s];
    let m: MixedUnitMatrix<D, D, M2> = mixed_unit_matrix![MixedUnitMatrix<D, D, M2>;
        [1.0, 0.0],
        [0.0, 1.0],
    ];
    let diag: qty!(1) = m.get::<0, 0>();
    assert_eq!(diag.unsafe_value, 1.0);
    assert_eq!(m.get::<0, 1>().unsafe_value, 0.0);
}

#[test]
fn mixed_unit_matrix_macro_accepts_scalar_storage() {
    // `… as f32` backs the matrix with `SMatrix<f32, …>` instead of the default
    // `f64`; the shape is still read from the dim lists. Equal row/col dims make
    // every entry dimensionless, so bare `f32` scalars are accepted.
    type D = dims![s, s];
    type M2f32 = SMatrix<f32, 2, 2>;
    let m: MixedUnitMatrix<D, D, M2f32> = mixed_unit_matrix![D, D as f32;
        [1.0f32, 2.0f32], // row 0
        [3.0f32, 4.0f32], // row 1
    ];
    let e00: qty!(1, f32) = m.get::<0, 0>();
    assert_eq!(e00.unsafe_value, 1.0f32);
    assert_eq!(m.get::<1, 0>().unsafe_value, 3.0f32);
    assert_eq!(m.get::<0, 1>().unsafe_value, 2.0f32);
}

#[test]
fn negation_preserves_units() {
    let neg = -sample();
    // Units unchanged; entry (0,0) is still m/s.
    let e: qty!(m / s) = neg.get::<0, 0>();
    assert_eq!(e.unsafe_value, -1.0);
}

#[test]
fn subtraction_preserves_units() {
    let diff = sample() - sample();
    let e: qty!(m / s) = diff.get::<0, 0>();
    assert_eq!(e.unsafe_value, 0.0);
}

#[test]
fn scalar_quantity_mul_scales_row_units() {
    // Multiplying by a time quantity turns m/s into m (the seconds cancel the
    // per-second): entry unit (q · row_i) / col_j = (s · m) / s = m.
    let scaled = sample() * quantity!(2.0, s);
    let e: qty!(m) = scaled.get::<0, 0>();
    assert_eq!(e.unsafe_value, 2.0);
}

#[test]
fn scalar_quantity_div_scales_row_units() {
    // Dividing by a time quantity turns m/s into m/s^2.
    let scaled = sample() / quantity!(2.0, s);
    let e: qty!(m / s ^ 2) = scaled.get::<0, 0>();
    assert_eq!(e.unsafe_value, 0.5);
}

#[test]
fn transpose_reciprocates_and_swaps() {
    // Mᵀ(i, j) = M(j, i) has unit row_j / col_i. With rows [m, m] and cols
    // [s, s], the transpose's row/col dims are the reciprocals: rows [1/s, 1/s],
    // cols [1/m, 1/m], so entry (0,0) is (1/s)/(1/m) = m/s, holding M(0,0).
    let t = sample().transpose();
    let e: qty!(m / s) = t.get::<0, 0>();
    assert_eq!(e.unsafe_value, 1.0);
    // Off-diagonal confirms the value transpose: t(0,1) = M(1,0) = 3.0.
    let e01: qty!(m / s) = t.get::<0, 1>();
    assert_eq!(e01.unsafe_value, 3.0);
}

#[test]
fn adjoint_reciprocates_like_transpose() {
    // Over a real field the adjoint is the transpose: same reciprocated dims
    // (rows [1/s, 1/s], cols [1/m, 1/m]) and the same value swap.
    let a = sample().adjoint();
    let e: qty!(m / s) = a.get::<0, 0>();
    assert_eq!(e.unsafe_value, 1.0);
    let e01: qty!(m / s) = a.get::<0, 1>();
    assert_eq!(e01.unsafe_value, 3.0);
}

#[test]
fn lu_of_uniform_rows_keeps_mixed_columns_in_u() {
    // A uniform row space [m, m] with mixed columns [s, kg] ⇒ ⟨[m,m],[s,kg]⟩,
    // entries [[1, 2], [3, 4]]. Rows are uniform, so partial-pivot LU is
    // well-typed: L and P collapse to dimensionless ⟨[m,m],[m,m]⟩ and U keeps
    // the mixed columns.
    type RowM = dims![m, m];
    type ColM = dims![s, kg];
    type Mat2 = MixedUnitMatrix<RowM, ColM, M2>;
    let a = Mat2::new(M2::new(1.0, 2.0, 3.0, 4.0));

    let lu = a.lu();
    let _l00: qty!(1) = lu.l.get::<0, 0>(); // dimensionless (m/m), unit diagonal
    let _u00: qty!(m / s) = lu.u.get::<0, 0>();
    let _u01: qty!(m / kg) = lu.u.get::<0, 1>();

    // Pᵀ·L·U reconstructs ⟨[m,m],[s,kg]⟩ exactly.
    let recon: Mat2 = lu.recompose();
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (recon.nalgebra()[(i, j)] - a.nalgebra()[(i, j)]).abs() < 1e-12,
                "entry ({i}, {j}) not reconstructed"
            );
        }
    }
}

#[test]
fn trace_sums_the_uniform_diagonal() {
    // The trace sums the diagonal entries, so it needs them commensurable — the
    // uniform-diagonal gate. A continuous state matrix ⟨[m/s, m/s²], [m, m/s]⟩ has
    // every diagonal ratio 1/s, so the trace lands in 1/s. diag(-2, -5) ⇒ -7.
    type Continuous = MixedUnitMatrix<dims![m / s, m / s ^ 2], dims![m, m / s], M2>;
    let a = Continuous::new(M2::new(-2.0, 0.0, 0.0, -5.0));
    let tr: qty!(1 / s) = a.trace();
    assert_eq!(tr.unsafe_value, -7.0);
}

#[test]
fn inverse_swaps_dims() {
    // A 2x2 endomorphism on [m, m] (identity-shaped dims) so it is invertible
    // and its inverse dims are a clean swap.
    type Endo = MixedUnitMatrix<dims![m, m], dims![m, m], M2>;
    let m = Endo::new(M2::new(4.0, 0.0, 0.0, 2.0));
    let inv = m.try_inverse().expect("invertible");
    // Inverse dims swap rows/cols; here both are [m, m], so entries are still
    // dimensionless, and (0,0) = 1/4.
    let e: qty!(1) = inv.get::<0, 0>();
    assert_eq!(e.unsafe_value, 0.25);
}

#[test]
fn determinant_multiplies_row_over_col_products() {
    // det has unit ∏RowDims / ∏ColDims = (m·m) / (s·s) = m^2 / s^2, regardless
    // of the entries. Value: det[[1, 2], [3, 4]] = 1·4 - 2·3 = -2.
    let d: qty!(m ^ 2 / s ^ 2) = sample().determinant();
    assert_eq!(d.unsafe_value, -2.0);
}

#[test]
fn determinant_of_endomorphism_is_dimensionless() {
    // A [m, m] <- [m, m] endomorphism: ∏rows / ∏cols cancels to dimensionless.
    type Endo = MixedUnitMatrix<dims![m, m], dims![m, m], M2>;
    let m = Endo::new(M2::new(4.0, 0.0, 0.0, 2.0));
    let d: qty!(1) = m.determinant();
    assert_eq!(d.unsafe_value, 8.0);
}

#[test]
fn solve_lands_in_input_space() {
    // A maps input-space [s, s] into output-space [m, m] (entry m/s), diagonal.
    type A = MixedUnitMatrix<dims![m, m], dims![s, s], M2>;
    let a = A::new(M2::new(2.0, 0.0, 0.0, 4.0));

    // A right-hand side b lives in output-space: rows [m, m], one dimensionless
    // column. Sharing RowDims with A is what makes the system well-posed.
    type V2 = SMatrix<f64, 2, 1>;
    type B = MixedUnitMatrix<dims![m, m], dims![1], V2>;
    let b = B::new(V2::new(2.0, 8.0));

    // The solution lands in input-space: rows [s, s], keeping b's [1] column, so
    // each entry has unit s / 1 = s. x = [2/2, 8/4] = [1, 2] seconds.
    let x = a.solve(&b).expect("invertible");
    let x0: qty!(s) = x.get::<0, 0>();
    assert_eq!(x0.unsafe_value, 1.0);
    let x1: qty!(s) = x.get::<1, 0>();
    assert_eq!(x1.unsafe_value, 2.0);
}

#[test]
fn triangular_solves_share_the_solve_signature() {
    // A lower-triangular A : ⟨[m, m], [s, s]⟩ (entry m/s). Forward substitution
    // lands x in input-space [s, s], keeping b's dimensionless column — same
    // signature as the general solve. A = [[2, 0], [1, 4]], b = [2, 9] m.
    type A = MixedUnitMatrix<dims![m, m], dims![s, s], M2>;
    let lower = A::new(M2::new(2.0, 0.0, 1.0, 4.0));
    type V2 = SMatrix<f64, 2, 1>;
    type B = MixedUnitMatrix<dims![m, m], dims![1], V2>;
    let b = B::new(V2::new(2.0, 9.0));

    let x = lower.solve_lower_triangular(&b).expect("nonzero diagonal");
    let x0: qty!(s) = x.get::<0, 0>(); // 2/2 = 1
    let x1: qty!(s) = x.get::<1, 0>(); // (9 - 1·1)/4 = 2
    assert_eq!(x0.unsafe_value, 1.0);
    assert_eq!(x1.unsafe_value, 2.0);

    // Upper-triangular back substitution has the identical unit signature.
    let upper = A::new(M2::new(2.0, 1.0, 0.0, 4.0));
    let y = upper.solve_upper_triangular(&b).expect("nonzero diagonal");
    let _y0: qty!(s) = y.get::<0, 0>();
    assert!((upper.nalgebra() * y.nalgebra() - b.nalgebra()).norm() < 1e-12);
}

#[test]
fn lu_reuse_solve_matches_recompose() {
    // Factor once (uniform rows [m, m]), then solve many RHS off the held
    // factors — no re-factorization. A : ⟨[m, m], [s, s]⟩.
    type A = MixedUnitMatrix<dims![m, m], dims![s, s], M2>;
    let a = A::new(M2::new(2.0, 1.0, 1.0, 3.0));
    let lu = a.lu();

    type V2 = SMatrix<f64, 2, 1>;
    type B = MixedUnitMatrix<dims![m, m], dims![1], V2>;
    let b = B::new(V2::new(5.0, 10.0));

    // Reuse-solve lands in input-space [s, s] (b's [1] column ⇒ unit s), and
    // agrees with the one-shot solve to numerical tolerance.
    let x = lu.solve(&b).expect("nonsingular U");
    let _x0: qty!(s) = x.get::<0, 0>();
    let x_ref = a.solve(&b).expect("invertible");
    assert!((x.nalgebra() - x_ref.nalgebra()).norm() < 1e-12);
    // A·x reconstructs b (⟨[m,m],[1]⟩).
    assert!((a.nalgebra() * x.nalgebra() - b.nalgebra()).norm() < 1e-12);
}

#[test]
fn opaque_lu_reuses_and_solves_a_non_uniform_matrix_ungated() {
    // Rows [m, s] are NOT uniform, so the typed `.lu()` (which would have to name a
    // pivoted row axis) does not even compile here. `lu_opaque` keeps the factors
    // opaque and retains only the invariant ⟨[m, s], [s, s]⟩ type, so the solve —
    // pivot-invariant — is available ungated. A = diag(2, 4).
    type A = MixedUnitMatrix<dims![m, s], dims![s, s], M2>;
    let a = A::new(M2::new(2.0, 0.0, 0.0, 4.0));
    let f = a.lu_opaque();

    // b in output-space ⟨[m, s], [1]⟩; solution in input-space ⟨[s, s], [1]⟩ ⇒ s.
    type V2 = SMatrix<f64, 2, 1>;
    type B = MixedUnitMatrix<dims![m, s], dims![1], V2>;
    let b = B::new(V2::new(4.0, 8.0));

    let x = f.solve(&b).expect("invertible");
    let x0: qty!(s) = x.get::<0, 0>(); // 4/2 = 2
    let x1: qty!(s) = x.get::<1, 0>(); // 8/4 = 2
    assert_eq!(x0.unsafe_value, 2.0);
    assert_eq!(x1.unsafe_value, 2.0);

    // try_inverse swaps dims to ⟨[s, s], [m, s]⟩; entry (0,0) = s/m = 0.5.
    let inv = f.try_inverse().expect("invertible");
    let i00: qty!(s / m) = inv.get::<0, 0>();
    assert_eq!(i00.unsafe_value, 0.5);

    // determinant: ∏row / ∏col = (m·s) / (s·s) = m/s; value 2·4 = 8.
    let d: qty!(m / s) = f.determinant();
    assert_eq!(d.unsafe_value, 8.0);

    // Raw escape hatch: the pivoted factors as bare nalgebra (no units attached).
    let raw = f.nalgebra();
    assert_eq!(raw.u().nrows(), 2);
}

#[test]
fn generalized_col_piv_qr_is_metric_orthonormal_and_solves() {
    // Square M : ⟨[m, m], [s, s]⟩ pivoted against a codomain metric
    // Gr : ⟨[1/m, 1/m], [m, m]⟩ and a domain metric Gc : ⟨[1/s, 1/s], [s, s]⟩. The
    // second metric is what makes the column pivot well-defined — the columns carry
    // s, so their raw norms would be unit-dependent.
    type Gr = MixedUnitMatrix<dims![1 / m, 1 / m], dims![m, m], M2>;
    type Gc = MixedUnitMatrix<dims![1 / s, 1 / s], dims![s, s], M2>;

    let m = sample();
    let g_r = Gr::new(M2::new(2.0, 0.5, 0.5, 2.0));
    let g_c = Gc::new(M2::new(3.0, 0.0, 0.0, 1.0));

    let gcpqr = m.generalized_col_piv_qr(&g_r, &g_c).expect("SPD metrics");

    // Q : ⟨[m, m], [1, 1]⟩ (Gr-orthonormal); R, P dimensionless.
    let _q00: qty!(m) = gcpqr.q.get::<0, 0>();
    let _r00: qty!(1) = gcpqr.r.get(0, 0);
    let q = gcpqr.q.nalgebra();
    assert!(
        ((q.transpose() * g_r.nalgebra() * q) - M2::identity()).norm() < 1e-10,
        "Q not Gr-orthonormal"
    );

    // recompose rebuilds M in its exact type ⟨[m, m], [s, s]⟩.
    let recon: Mat = gcpqr.recompose();
    for i in 0..2 {
        for j in 0..2 {
            assert!((recon.nalgebra()[(i, j)] - m.nalgebra()[(i, j)]).abs() < 1e-10);
        }
    }

    // Square full-rank solve is metric-independent: for b = M·x_true (x_true =
    // [1 s, 1 s]) it recovers x_true exactly. b = [1+2, 3+4] = [3, 7] m.
    type V2 = SMatrix<f64, 2, 1>;
    type B = MixedUnitMatrix<dims![m, m], dims![1], V2>;
    let b = B::new(V2::new(3.0, 7.0));
    let x = gcpqr.solve(&b).expect("full rank");
    let x0: qty!(s) = x.get::<0, 0>();
    let x1: qty!(s) = x.get::<1, 0>();
    assert!((x0.unsafe_value - 1.0).abs() < 1e-10);
    assert!((x1.unsafe_value - 1.0).abs() < 1e-10);
}

#[test]
fn generalized_col_piv_qr_is_rectangular_least_squares() {
    // Tall 3×2 M against Gr (3×3) and Gc (2×2): thin pivot of length min(3, 2) = 2,
    // so Q is 3×2 (Gr-orthonormal) and R is 2×2 dimensionless upper-triangular.
    let a = tall_gen();
    let g_r = gr3();
    let g_c = gc2();

    let gcpqr = a
        .generalized_col_piv_qr(&g_r, &g_c)
        .expect("SPD metrics");
    assert_eq!(gcpqr.q.shape(), (3, 2));
    assert_eq!(gcpqr.r.shape(), (2, 2));
    assert_eq!(gcpqr.p.shape(), (2, 2));
    let _q00: qty!(m) = gcpqr.q.get::<0, 0>();

    // Qᵀ Gr Q = I₂ on the thin pivot axis.
    let q = gcpqr.q.nalgebra();
    assert!(
        ((q.transpose() * g_r.nalgebra() * q) - M2::identity()).norm() < 1e-10,
        "Q not Gr-orthonormal"
    );

    let recon: TallGen = gcpqr.recompose();
    for i in 0..3 {
        for j in 0..2 {
            assert!((recon.nalgebra()[(i, j)] - a.nalgebra()[(i, j)]).abs() < 1e-10);
        }
    }

    // A consistent overdetermined system b = M·x_true (x_true = [1 s, 1 s]) has a
    // zero-residual least-squares solution that recovers x_true. b = [1+2, 3+4,
    // 5+7] = [3, 7, 12] m.
    type V3 = SMatrix<f64, 3, 1>;
    type B = MixedUnitMatrix<dims![m, m, m], dims![1], V3>;
    let b = B::new(V3::new(3.0, 7.0, 12.0));
    let x = gcpqr.solve(&b).expect("full column rank");
    let x0: qty!(s) = x.get::<0, 0>();
    let x1: qty!(s) = x.get::<1, 0>();
    assert!((x0.unsafe_value - 1.0).abs() < 1e-10);
    assert!((x1.unsafe_value - 1.0).abs() < 1e-10);
}

#[test]
fn opaque_full_piv_lu_reuses_and_solves_a_non_uniform_matrix_ungated() {
    // Full pivoting reorders *both* axes, so no typed full-pivot LU exists for a
    // mixed matrix. The opaque wrapper retains only the invariant ⟨[m, s], [s, s]⟩
    // type and solves/inverts/determines ungated. A = diag(2, 4).
    type A = MixedUnitMatrix<dims![m, s], dims![s, s], M2>;
    let a = A::new(M2::new(2.0, 0.0, 0.0, 4.0));
    let f = a.full_piv_lu_opaque();

    type V2 = SMatrix<f64, 2, 1>;
    type B = MixedUnitMatrix<dims![m, s], dims![1], V2>;
    let b = B::new(V2::new(4.0, 8.0));

    let x = f.solve(&b).expect("invertible");
    let x0: qty!(s) = x.get::<0, 0>();
    let x1: qty!(s) = x.get::<1, 0>();
    assert_eq!(x0.unsafe_value, 2.0);
    assert_eq!(x1.unsafe_value, 2.0);

    let inv = f.try_inverse().expect("invertible");
    let i00: qty!(s / m) = inv.get::<0, 0>();
    assert_eq!(i00.unsafe_value, 0.5);

    let d: qty!(m / s) = f.determinant();
    assert_eq!(d.unsafe_value, 8.0);

    let raw = f.nalgebra();
    assert_eq!(raw.u().nrows(), 2);
}

#[test]
fn pow_of_endomorphism_stays_in_space() {
    // A discrete-time transition on state [m, m/s]. The diagonal entries are
    // dimensionless (Dims[i]/Dims[i] = 1), the off-diagonals carry s and 1/s.
    // Every power stays <[m, m/s], [m, m/s]>. Using a diagonal Φ makes the
    // values predictable: diag(1, 2)^3 = diag(1, 8).
    type Dims = dims![m, m / s];
    type Phi = MixedUnitMatrix<Dims, Dims, M2>;
    let phi = Phi::new(M2::new(1.0, 0.0, 0.0, 2.0));

    let p3 = phi.pow(3);
    let e00: qty!(1) = p3.get::<0, 0>();
    assert_eq!(e00.unsafe_value, 1.0);
    let e11: qty!(1) = p3.get::<1, 1>();
    assert_eq!(e11.unsafe_value, 8.0);

    // The zeroth power is the identity, still typed in the same space.
    let p0 = phi.pow(0);
    let i00: qty!(1) = p0.get::<0, 0>();
    assert_eq!(i00.unsafe_value, 1.0);
}

#[test]
fn exp_of_endomorphism_stays_in_space() {
    // The matrix exponential of a state endomorphism stays <[m, m/s], [m, m/s]>,
    // just like pow. A diagonal argument makes the values predictable:
    // exp(diag(0, 1)) = diag(1, e).
    type Dims = dims![m, m / s];
    type Phi = MixedUnitMatrix<Dims, Dims, M2>;
    let a_c = Phi::new(M2::new(0.0, 0.0, 0.0, 1.0));

    let e = a_c.exp();
    let e00: qty!(1) = e.get::<0, 0>();
    assert_eq!(e00.unsafe_value, 1.0);
    let e11: qty!(1) = e.get::<1, 1>();
    assert!((e11.unsafe_value - std::f64::consts::E).abs() < 1e-12);
}

#[test]
fn component_mul_multiplies_dims() {
    // Hadamard product of two m/s matrices: the row and column lists multiply
    // element-wise, so entry (0,0) has unit (m·m)/(s·s) = m^2/s^2.
    let a = sample();
    let b = Mat::new(M2::new(2.0, 0.0, 0.0, 0.0));
    let had = a.component_mul(&b);
    let e: qty!(m ^ 2 / s ^ 2) = had.get::<0, 0>();
    assert_eq!(e.unsafe_value, 2.0); // 1.0 * 2.0
}

#[test]
fn component_div_divides_dims() {
    // Hadamard quotient of two m/s matrices: the lists divide element-wise, so
    // every entry cancels to dimensionless.
    let a = sample();
    let b = Mat::new(M2::new(4.0, 1.0, 1.0, 1.0));
    let had = a.component_div(&b);
    let e: qty!(1) = had.get::<0, 0>();
    assert_eq!(e.unsafe_value, 0.25); // 1.0 / 4.0
}

#[test]
fn component_mul_across_distinct_unit_grids() {
    // No coherence beyond matching shape/brand: `a` is m/s, while `b` maps [s, s]
    // rows over [m, m] cols (entry unit s/m). Their Hadamard product is
    // dimensionless — (m·s)/(s·m) — even though neither operand is.
    let a = sample();
    type Bmat = MixedUnitMatrix<dims![s, s], dims![m, m], M2>;
    let b = Bmat::new(M2::new(3.0, 0.0, 0.0, 0.0));
    let had = a.component_mul(&b);
    let e: qty!(1) = had.get::<0, 0>();
    assert_eq!(e.unsafe_value, 3.0); // 1.0 * 3.0
}

#[test]
fn scale_and_unscale_preserve_units() {
    // Multiplying/dividing by a bare real leaves every entry's unit untouched.
    let scaled = sample().scale(2.0);
    let e: qty!(m / s) = scaled.get::<0, 0>();
    assert_eq!(e.unsafe_value, 2.0); // 1.0 * 2.0

    let back = scaled.unscale(4.0);
    let e: qty!(m / s) = back.get::<0, 0>();
    assert_eq!(e.unsafe_value, 0.5); // 2.0 / 4.0
}

#[test]
fn rescale_matrix_rescales_all_rows_and_columns_at_once() {
    // sample : ⟨[m, m], [s, s]⟩, entries m/s, values [[1, 2], [3, 4]].
    let a = sample();

    // Rescale only the rows to mm (× 1000), keeping the columns in s. Each entry
    // gains rowfactor / colfactor = 1000 / 1, so its unit becomes mm/s and its
    // value is multiplied by 1000 — like `rescale`, but along every row at once.
    type MmRows = dims![mm, mm];
    let mm: MixedUnitMatrix<MmRows, ColDims, M2> = rescale_matrix(&a);
    let e00: qty!(mm / s) = mm.get::<0, 0>();
    assert!((e00.unsafe_value - 1000.0).abs() < 1e-9);
    let e10: qty!(mm / s) = mm.get::<1, 0>();
    assert!((e10.unsafe_value - 3000.0).abs() < 1e-9);

    // Rescale rows to km (× 1/1000) *and* columns to ms (× 1000) simultaneously:
    // the per-entry factor is (1/1000) / 1000 = 1e-6, unit km/ms.
    type KmRows = dims![km, km];
    type MsCols = dims![ms, ms];
    let km: MixedUnitMatrix<KmRows, MsCols, M2> = rescale_matrix(&a);
    let f00: qty!(km / ms) = km.get::<0, 0>();
    assert!((f00.unsafe_value - 1e-6).abs() < 1e-15);
    let f11: qty!(km / ms) = km.get::<1, 1>();
    assert!((f11.unsafe_value - 4e-6).abs() < 1e-15);
}

#[test]
fn rescale_matrix_across_compound_and_dimensionless_entries() {
    // A continuous state matrix ⟨[m/s, m/s²], [m, m/s]⟩ (diagonal dimensionless,
    // off-diagonals s and 1/s). Rescale the time base everywhere: rows to
    // [mm/s, mm/s²] and columns to [mm, mm/s]. The gauge factor cancels on the
    // dimensionless diagonal (mm/mm = 1) and rescales the off-diagonals.
    type Rows = dims![m / s, m / s ^ 2];
    type Cols = dims![m, m / s];
    type Continuous = MixedUnitMatrix<Rows, Cols, M2>;
    let a = Continuous::new(M2::new(1.0, 2.0, 3.0, 4.0));

    type NewRows = dims![mm / s, mm / s ^ 2];
    type NewCols = dims![mm, mm / s];
    let r: MixedUnitMatrix<NewRows, NewCols, M2> = rescale_matrix(&a);

    // Diagonal entry (0,0): (mm/s)/(mm) = 1/s, factor 1000/1000 = 1 ⇒ value 1.
    let d00: qty!(1 / s) = r.get::<0, 0>();
    assert!((d00.unsafe_value - 1.0).abs() < 1e-12);
    // Off-diagonal (0,1): (mm/s)/(mm/s) = 1 (dimensionless), factor 1 ⇒ value 2.
    let o01: qty!(1) = r.get::<0, 1>();
    assert!((o01.unsafe_value - 2.0).abs() < 1e-12);
}

#[test]
fn kronecker_takes_outer_product_of_dim_lists() {
    // A maps [A, K] (columns) into [m, s] (rows); B maps [s, s] into [m, m].
    // Their Kronecker product is 4x4, with row dims the outer product of the
    // row lists and column dims the outer product of the column lists:
    //   rows: [m·m, m·m, s·m, s·m]   cols: [A·s, A·s, K·s, K·s]
    type A = MixedUnitMatrix<dims![m, s], dims![A, K], M2>;
    type B = MixedUnitMatrix<dims![m, m], dims![s, s], M2>;
    let a = A::new(M2::new(1.0, 2.0, 3.0, 4.0));
    let b = B::new(M2::new(10.0, 20.0, 30.0, 40.0));

    let k = a.kronecker(&b);
    assert_eq!(k.shape(), (4, 4));

    // Entry (0,0): row m·m over col A·s => m^2 / (A·s); value A(0,0)·B(0,0).
    let e00: qty!(m ^ 2 / (A * s)) = k.get::<0, 0>();
    assert_eq!(e00.unsafe_value, 10.0); // 1.0 * 10.0

    // Entry (2,0): row index 2 is A's second row (s) times B's first row (m),
    // i.e. s·m over col A·s => m / A; value A(1,0)·B(0,0).
    let e20: qty!(m / A) = k.get::<2, 0>();
    assert_eq!(e20.unsafe_value, 30.0); // 3.0 * 10.0
}

#[test]
fn cholesky_factors_a_metric_with_dimensionless_pivots() {
    // A symmetric metric M : <[1/m, 1/s], [m, s]> — note ColDims = 1/RowDims, the
    // self-transpose (metric) shape the factorization requires. Entry units are
    // 1/m², 1/(m·s), 1/s². The numeric matrix [[4, 1], [1, 4]] is SPD.
    type RowM = dims![1 / m, 1 / s];
    type ColM = dims![m, s];
    type Metric = MixedUnitMatrix<RowM, ColM, M2>;
    let m = Metric::new(M2::new(4.0, 1.0, 1.0, 4.0));

    let l = m.cholesky().expect("positive-definite").l;

    // L : <[1/m, 1/s], [1, 1]> — columns collapse to dimensionless, rows keep
    // M's row units. Numerically lower-triangular: L = [[2, 0], [0.5, √3.75]].
    let l00: qty!(1 / m) = l.get::<0, 0>();
    assert_eq!(l00.unsafe_value, 2.0);
    let l01: qty!(1 / m) = l.get::<0, 1>();
    assert_eq!(l01.unsafe_value, 0.0); // upper triangle zeroed by `unpack`
    let l10: qty!(1 / s) = l.get::<1, 0>();
    assert_eq!(l10.unsafe_value, 0.5);
    let l11: qty!(1 / s) = l.get::<1, 1>();
    assert!((l11.unsafe_value - 3.75_f64.sqrt()).abs() < 1e-12);

    // The round trip L·Lᵀ reconstructs M's *exact* type (the annotation forces
    // the contraction to unify on the dimensionless pivot) and its values.
    let m2: Metric = l * l.transpose();
    let r00: qty!(1 / m ^ 2) = m2.get::<0, 0>();
    assert!((r00.unsafe_value - 4.0).abs() < 1e-12);
    let r01: qty!(1 / (m * s)) = m2.get::<0, 1>();
    assert!((r01.unsafe_value - 1.0).abs() < 1e-12);
    let r11: qty!(1 / s ^ 2) = m2.get::<1, 1>();
    assert!((r11.unsafe_value - 4.0).abs() < 1e-12);
}

#[test]
fn cholesky_held_factor_reuses_solve_inverse_and_determinant() {
    // Same SPD metric M : <[1/m, 1/s], [m, s]>, [[4, 1], [1, 4]]. The held
    // factorization solves/inverts/determines without re-factoring.
    type RowM = dims![1 / m, 1 / s];
    type ColM = dims![m, s];
    type Metric = MixedUnitMatrix<RowM, ColM, M2>;
    let chol = Metric::new(M2::new(4.0, 1.0, 1.0, 4.0))
        .cholesky()
        .expect("positive-definite");

    // Solve M x = b: b in output-space ⟨[1/m, 1/s], [1]⟩, x in input-space
    // ⟨[m, s], [1]⟩. With x_true = [1 m, 1 s], M·x = [5, 5].
    type V2 = SMatrix<f64, 2, 1>;
    let b = MixedUnitMatrix::<RowM, dims![1], V2>::new(V2::new(5.0, 5.0));
    let x = chol.solve(&b);
    let x0: qty!(m) = x.get::<0, 0>();
    let x1: qty!(s) = x.get::<1, 0>();
    assert!((x0.unsafe_value - 1.0).abs() < 1e-12);
    assert!((x1.unsafe_value - 1.0).abs() < 1e-12);

    // Inverse lands in ⟨[m, s], [1/m, 1/s]⟩; entry (0,0) = m² with value 4/15.
    let inv = chol.inverse();
    let i00: qty!(m ^ 2) = inv.get::<0, 0>();
    assert!((i00.unsafe_value - 4.0 / 15.0).abs() < 1e-12);

    // determinant: ∏row / ∏col = 1/(m²·s²), value det = 15.
    let d: qty!(1 / (m ^ 2 * s ^ 2)) = chol.determinant();
    assert!((d.unsafe_value - 15.0).abs() < 1e-12);
}

#[test]
fn eigenvalues_of_an_endomorphism_are_dimensionless() {
    // An endomorphism <[m, m], [m, m]> maps the state space to itself, so its
    // spectrum is a set of pure numbers. Diagonal ⇒ eigenvalues are {2, 3}.
    type Endo = MixedUnitMatrix<dims![m, m], dims![m, m], M2>;
    let a = Endo::new(M2::new(2.0, 0.0, 0.0, 3.0));

    let ev = a.eigenvalues().expect("real spectrum");
    // Every eigenvalue reads back as a dimensionless quantity (unit `1`).
    let mut vals: Vec<f64> = ev.iter().map(|q: qty!(1)| q.unsafe_value).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(vals, vec![2.0, 3.0]);
}

#[test]
fn eigenvalues_of_a_uniform_endomorphism_carry_the_diagonal_unit() {
    // A continuous state matrix `⟨C/t, C⟩` is a *uniform* endomorphism: its
    // diagonal ratios `RowDims[i] / ColDims[i]` all equal `1/s`, so the spectrum
    // (the poles) carries `1/s` rather than being dimensionless. State
    // `[position (m), velocity (m/s)]` ⇒ rows `[m/s, m/s²]`, cols `[m, m/s]`.
    type Continuous = MixedUnitMatrix<dims![m / s, m / s ^ 2], dims![m, m / s], M2>;
    // Numerically diagonal ⇒ eigenvalues are the diagonal {-2, -5} (real).
    let a = Continuous::new(M2::new(-2.0, 0.0, 0.0, -5.0));

    let ev = a.eigenvalues().expect("real spectrum");
    // The eigenvalues read back in `1/s` — a wrong unit here would not compile.
    let mut vals: Vec<f64> = ev.iter().map(|q: qty!(1 / s)| q.unsafe_value).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(vals, vec![-5.0, -2.0]);
}

#[test]
fn generalized_qr_is_metric_orthonormal() {
    // sample M : ⟨[m, m], [s, s]⟩ measured against a codomain metric
    // Gr : ⟨[1/m, 1/m], [m, m]⟩ — the only shape whose norm is well-typed. SPD but
    // non-trivial, so the whitening is genuinely exercised.
    type GrRow = dims![1 / m, 1 / m];
    type GrCol = dims![m, m];
    type Gr = MixedUnitMatrix<GrRow, GrCol, M2>;

    let m = sample();
    let g_r = Gr::new(M2::new(2.0, 0.5, 0.5, 2.0));

    let gqr = m.generalized_qr(&g_r).expect("SPD metric");

    // Q : ⟨[m, m], [1, 1]⟩ (Gr-orthonormal codomain basis); R : ⟨[1, 1], [s, s]⟩,
    // upper-triangular so R(1, 0) is zeroed.
    let _q00: qty!(m) = gqr.q.get::<0, 0>();
    let r10: qty!(1 / s) = gqr.r.get::<1, 0>();
    assert_eq!(r10.unsafe_value, 0.0);

    // Metric-orthonormality: Qᵀ Gr Q = I.
    let q = gqr.q.nalgebra();
    let gram = q.transpose() * g_r.nalgebra() * q;
    assert!(
        (gram - M2::identity()).norm() < 1e-10,
        "Q not Gr-orthonormal"
    );

    // recompose rebuilds M = Q R in M's exact type ⟨[m, m], [s, s]⟩ (QR is
    // one-sided, so it needs no metric back).
    let recon: Mat = gqr.recompose();
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (recon.nalgebra()[(i, j)] - m.nalgebra()[(i, j)]).abs() < 1e-10,
                "entry ({i}, {j}) not reconstructed"
            );
        }
    }
}

#[test]
fn symmetric_eigen_reconstructs_a_metric() {
    // Same metric shape as Cholesky: <[1/m, 1/s], [m, s]> with ColDims = 1/RowDims.
    // Numeric [[4, 1], [1, 4]] is symmetric with eigenvalues {3, 5}.
    type RowM = dims![1 / m, 1 / s];
    type ColM = dims![m, s];
    type Metric = MixedUnitMatrix<RowM, ColM, M2>;
    let m = Metric::new(M2::new(4.0, 1.0, 1.0, 4.0));

    let eig = m.symmetric_eigen();

    // Eigenvalues are dimensionless {3, 5}.
    let mut vals: Vec<f64> = eig
        .eigenvalues
        .iter()
        .map(|x: qty!(1)| x.unsafe_value)
        .collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((vals[0] - 3.0).abs() < 1e-12);
    assert!((vals[1] - 5.0).abs() < 1e-12);

    // `recompose` rebuilds M = Q·diag(Λ)·Qᵀ, landing back in the metric type: the
    // `Metric` annotation makes the unit round trip part of the check, not just
    // the stored values.
    let recon: Metric = eig.recompose();
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (recon.nalgebra()[(i, j)] - m.nalgebra()[(i, j)]).abs() < 1e-10,
                "entry ({i}, {j}) not reconstructed"
            );
        }
    }
}

#[test]
fn schur_triangularizes_a_non_symmetric_metric() {
    // Metric shape <[1/m, 1/s], [m, s]>, but numerically NON-symmetric
    // [[2, 2], [1, 3]] (eigenvalues {1, 4}, real) — the case `symmetric_eigen`
    // cannot express. Schur keeps the same unitary frame (same metric bound) but
    // returns a dimensionless upper-triangular T with the eigenvalues on its
    // diagonal, and reconstructs the metric type exactly.
    type RowM = dims![1 / m, 1 / s];
    type ColM = dims![m, s];
    type Metric = MixedUnitMatrix<RowM, ColM, M2>;
    let m = Metric::new(M2::new(2.0, 2.0, 1.0, 3.0));

    let schur = m.schur();

    // T is dimensionless and (quasi-)upper-triangular: the strictly-lower entry
    // vanishes for a real spectrum.
    let t10: qty!(1) = schur.t.get(1, 0);
    assert!(t10.unsafe_value.abs() < 1e-10);
    // Its diagonal carries the eigenvalues {1, 4}.
    let d0: qty!(1) = schur.t.get(0, 0);
    let d1: qty!(1) = schur.t.get(1, 1);
    let mut diag = [d0.unsafe_value, d1.unsafe_value];
    diag.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((diag[0] - 1.0).abs() < 1e-10);
    assert!((diag[1] - 4.0).abs() < 1e-10);

    // Q·T·Qᵀ reconstructs M's exact metric type <[1/m, 1/s], [m, s]>.
    let recon: Metric = schur.recompose();
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (recon.nalgebra()[(i, j)] - m.nalgebra()[(i, j)]).abs() < 1e-10,
                "entry ({i}, {j}) not reconstructed"
            );
        }
    }
}

#[test]
fn generalized_bidiagonalize_is_metric_orthonormal_with_a_dimensionless_band() {
    // sample M : ⟨[m,m],[s,s]⟩ reduced against a codomain metric
    // Gr : ⟨[1/m,1/m],[m,m]⟩ and a domain metric Gc : ⟨[1/s,1/s],[s,s]⟩ — the SVD
    // reduction step, with metric-orthonormal frames and a dimensionless band.
    type GrRow = dims![1 / m, 1 / m];
    type GrCol = dims![m, m];
    type Gr = MixedUnitMatrix<GrRow, GrCol, M2>;
    type GcRow = dims![1 / s, 1 / s];
    type GcCol = dims![s, s];
    type Gc = MixedUnitMatrix<GcRow, GcCol, M2>;

    let m = sample();
    let g_r = Gr::new(M2::new(2.0, 0.5, 0.5, 2.0));
    let g_c = Gc::new(M2::new(3.0, 0.0, 0.0, 1.0));

    let gbd = m
        .generalized_bidiagonalize(&g_r, &g_c)
        .expect("SPD metrics");

    // Band B dimensionless; U : ⟨[m,m],[1,1]⟩, V : ⟨[s,s],[1,1]⟩.
    let _b00: qty!(1) = gbd.d.get(0, 0);
    let _u00: qty!(m) = gbd.u.get::<0, 0>();
    let _v00: qty!(s) = gbd.v.get::<0, 0>();

    // Metric-orthonormality: Uᵀ Gr U = I and Vᵀ Gc V = I.
    let u = gbd.u.nalgebra();
    assert!(
        ((u.transpose() * g_r.nalgebra() * u) - M2::identity()).norm() < 1e-10,
        "U not Gr-orthonormal"
    );
    let v = gbd.v.nalgebra();
    assert!(
        ((v.transpose() * g_c.nalgebra() * v) - M2::identity()).norm() < 1e-10,
        "V not Gc-orthonormal"
    );

    // recompose rebuilds M = U B Vᵀ Gc in M's exact type ⟨[m,m],[s,s]⟩.
    let recon: Mat = gbd.recompose(&g_c);
    for i in 0..2 {
        for j in 0..2 {
            assert!((recon.nalgebra()[(i, j)] - m.nalgebra()[(i, j)]).abs() < 1e-10);
        }
    }
}

#[test]
fn hessenberg_reduces_a_metric_below_the_subdiagonal() {
    // A 3x3 metric ⟨[1/m,1/s,1/kg],[m,s,kg]⟩, numerically non-symmetric. The
    // similarity frame Q … Qᵀ needs the metric shape; H is dimensionless.
    type RowM = dims![1 / m, 1 / s, 1 / kg];
    type ColM = dims![m, s, kg];
    type Metric3 = MixedUnitMatrix<RowM, ColM, M3>;
    let m = Metric3::new(M3::new(4.0, 1.0, 2.0, 0.0, 3.0, 1.0, 1.0, 2.0, 5.0));

    let hess = m.hessenberg();
    // Upper-Hessenberg: the (2,0) entry below the subdiagonal vanishes.
    let h20: qty!(1) = hess.h.get(2, 0);
    assert!(h20.unsafe_value.abs() < 1e-10);

    // Q·H·Qᵀ reconstructs the metric type exactly.
    let recon: Metric3 = hess.recompose();
    for i in 0..3 {
        for j in 0..3 {
            assert!((recon.nalgebra()[(i, j)] - m.nalgebra()[(i, j)]).abs() < 1e-10);
        }
    }
}

#[test]
fn symmetric_tridiagonalize_bands_a_metric() {
    // A 3x3 symmetric metric. Same similarity frame ⇒ metric bound; T is a
    // dimensionless tridiagonal.
    type RowM = dims![1 / m, 1 / s, 1 / kg];
    type ColM = dims![m, s, kg];
    type Metric3 = MixedUnitMatrix<RowM, ColM, M3>;
    let m = Metric3::new(M3::new(4.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 5.0));

    let tri = m.symmetric_tridiagonalize();
    // Tridiagonal: the (2,0) corner vanishes.
    let t20: qty!(1) = tri.t.get(2, 0);
    assert!(t20.unsafe_value.abs() < 1e-10);

    let recon: Metric3 = tri.recompose();
    for i in 0..3 {
        for j in 0..3 {
            assert!((recon.nalgebra()[(i, j)] - m.nalgebra()[(i, j)]).abs() < 1e-10);
        }
    }
}

#[test]
fn generalized_symmetric_eigen_of_a_metric_pencil_is_dimensionless() {
    // Two metrics of the *same* shape <[1/m, 1/s], [m, s]>: K = [[4, 1], [1, 4]]
    // against M = 2·I (SPD). Because a metric's Cholesky factor has dimensionless
    // columns, this reduces with *no* even-exponent gate — the price being that
    // same-shape metrics force a dimensionless spectrum. K v = λ M v ⇒
    // M⁻¹K = [[2, .5], [.5, 2]] ⇒ {1.5, 2.5}.
    type RowM = dims![1 / m, 1 / s];
    type ColM = dims![m, s];
    type Metric = MixedUnitMatrix<RowM, ColM, M2>;
    let k = Metric::new(M2::new(4.0, 1.0, 1.0, 4.0));
    let m = Metric::new(M2::new(2.0, 0.0, 0.0, 2.0));

    let eig = k
        .generalized_symmetric_eigen(&m)
        .expect("positive-definite mass");

    // Dimensionless generalized eigenvalues.
    let mut vals: Vec<f64> = eig
        .eigenvalues
        .iter()
        .map(|q: qty!(1)| q.unsafe_value)
        .collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((vals[0] - 1.5).abs() < 1e-10);
    assert!((vals[1] - 2.5).abs() < 1e-10);

    // Eigenvectors land in <1/RowDims, [1, 1]> = <[m, s], [1, 1]>: dimensionless
    // columns, rows m and s. A wrong unit annotation would not compile.
    let _v00: qty!(m) = eig.eigenvectors.get::<0, 0>();
    let _v10: qty!(s) = eig.eigenvectors.get::<1, 0>();

    // The eigenvectors are M-orthonormal: Vᵀ M V = I.
    let v = eig.eigenvectors.nalgebra();
    let gram = v.transpose() * m.nalgebra() * v;
    assert!((gram - M2::identity()).norm() < 1e-10);

    // recompose rebuilds K = M V Λ Vᴴ M in K's exact type <[1/m, 1/s], [m, s]>.
    let k_recon = eig.recompose(&m);
    let r00: qty!(1 / m ^ 2) = k_recon.get::<0, 0>();
    let r11: qty!(1 / s ^ 2) = k_recon.get::<1, 1>();
    assert!((r00.unsafe_value - 4.0).abs() < 1e-10);
    assert!((r11.unsafe_value - 4.0).abs() < 1e-10);
}

#[test]
fn generalized_svd_is_metric_orthonormal_and_dimensionless() {
    // sample M : <[m, m], [s, s]>, values [[1, 2], [3, 4]]. Measured against a
    // codomain metric Gr : <[1/m, 1/m], [m, m]> and a domain metric
    // Gc : <[1/s, 1/s], [s, s]> — the only shapes whose norms are well-typed.
    // Both SPD but non-trivial, so the whitening is genuinely exercised.
    type GrRow = dims![1 / m, 1 / m];
    type GrCol = dims![m, m];
    type Gr = MixedUnitMatrix<GrRow, GrCol, M2>;
    type GcRow = dims![1 / s, 1 / s];
    type GcCol = dims![s, s];
    type Gc = MixedUnitMatrix<GcRow, GcCol, M2>;

    let m = sample();
    let g_r = Gr::new(M2::new(2.0, 0.5, 0.5, 2.0));
    let g_c = Gc::new(M2::new(3.0, 0.0, 0.0, 1.0));

    let gsvd = m.generalized_svd(&g_r, &g_c).expect("SPD metrics");

    // Singular values are dimensionless — a well-typed norm cannot carry a unit.
    let _sv: Vec<f64> = gsvd
        .singular_values
        .iter()
        .map(|q: qty!(1)| q.unsafe_value)
        .collect();

    // U : <[m, m], [1, 1]> (Gr-orthonormal codomain basis); V : <[s, s], [1, 1]>.
    let _u00: qty!(m) = gsvd.u.get::<0, 0>();
    let _v00: qty!(s) = gsvd.v.get::<0, 0>();

    // Metric-orthonormality: Uᵀ Gr U = I and Vᵀ Gc V = I.
    let u = gsvd.u.nalgebra();
    let gram_u = u.transpose() * g_r.nalgebra() * u;
    assert!(
        (gram_u - M2::identity()).norm() < 1e-10,
        "U not Gr-orthonormal"
    );
    let v = gsvd.v.nalgebra();
    let gram_v = v.transpose() * g_c.nalgebra() * v;
    assert!(
        (gram_v - M2::identity()).norm() < 1e-10,
        "V not Gc-orthonormal"
    );

    // recompose rebuilds M = U Σ Vᵀ Gc in M's exact type <[m, m], [s, s]>.
    let recon: Mat = gsvd.recompose(&g_c);
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (recon.nalgebra()[(i, j)] - m.nalgebra()[(i, j)]).abs() < 1e-10,
                "entry ({i}, {j}) not reconstructed"
            );
        }
    }
}

// A tall 3×2 map ⟨[m,m,m],[s,s]⟩ (entry m/s), full column rank, shared by the
// three rectangular generalized decompositions below.
type TallGen = MixedUnitMatrix<dims![m, m, m], dims![s, s], SMatrix<f64, 3, 2>>;
type GrRow3 = dims![1 / m, 1 / m, 1 / m];
type GrCol3 = dims![m, m, m];
type Gr3 = MixedUnitMatrix<GrRow3, GrCol3, M3>;
type GcRow2 = dims![1 / s, 1 / s];
type GcCol2 = dims![s, s];
type Gc2 = MixedUnitMatrix<GcRow2, GcCol2, M2>;

fn tall_gen() -> TallGen {
    TallGen::new(SMatrix::<f64, 3, 2>::new(1.0, 2.0, 3.0, 4.0, 5.0, 7.0))
}
fn gr3() -> Gr3 {
    // SPD 3×3 (symmetric, diagonally dominant).
    Gr3::new(M3::new(2.0, 0.5, 0.0, 0.5, 2.0, 0.5, 0.0, 0.5, 2.0))
}
fn gc2() -> Gc2 {
    Gc2::new(M2::new(3.0, 0.0, 0.0, 1.0))
}

#[test]
fn generalized_qr_is_rectangular_through_a_thin_pivot() {
    // Tall 3×2 M against a 3×3 codomain metric Gr: thin pivot of length
    // min(3, 2) = 2, so Q is 3×2 ⟨[m,m,m],[1,1]⟩ (Gr-orthonormal) and R is 2×2
    // ⟨[1,1],[s,s]⟩ upper-triangular.
    let a = tall_gen();
    let g_r = gr3();

    let gqr = a.generalized_qr(&g_r).expect("SPD metric");
    assert_eq!(gqr.q.shape(), (3, 2));
    assert_eq!(gqr.r.shape(), (2, 2));
    let _q00: qty!(m) = gqr.q.get::<0, 0>();
    let r10: qty!(1 / s) = gqr.r.get::<1, 0>();
    assert_eq!(r10.unsafe_value, 0.0);

    // Qᵀ Gr Q = I₂ (the thin Gramian on the pivot axis).
    let q = gqr.q.nalgebra();
    assert!(
        ((q.transpose() * g_r.nalgebra() * q) - M2::identity()).norm() < 1e-10,
        "Q not Gr-orthonormal"
    );

    let recon: TallGen = gqr.recompose();
    for i in 0..3 {
        for j in 0..2 {
            assert!((recon.nalgebra()[(i, j)] - a.nalgebra()[(i, j)]).abs() < 1e-10);
        }
    }
}

#[test]
fn generalized_svd_is_rectangular_through_a_thin_pivot() {
    // Tall 3×2 M against Gr (3×3) and Gc (2×2): U is 3×2, 2 dimensionless singular
    // values, V is 2×2. recompose lands back in the exact 3×2 type.
    let a = tall_gen();
    let g_r = gr3();
    let g_c = gc2();

    let gsvd = a.generalized_svd(&g_r, &g_c).expect("SPD metrics");
    assert_eq!(gsvd.u.shape(), (3, 2));
    assert_eq!(gsvd.v.shape(), (2, 2));
    assert_eq!(gsvd.singular_values.nalgebra().len(), 2);
    let _sv: qty!(1) = gsvd.singular_values.iter().next().unwrap();
    let _u00: qty!(m) = gsvd.u.get::<0, 0>();
    let _v00: qty!(s) = gsvd.v.get::<0, 0>();

    // Uᵀ Gr U = I₂ and Vᵀ Gc V = I₂.
    let u = gsvd.u.nalgebra();
    assert!(
        ((u.transpose() * g_r.nalgebra() * u) - M2::identity()).norm() < 1e-10,
        "U not Gr-orthonormal"
    );
    let v = gsvd.v.nalgebra();
    assert!(
        ((v.transpose() * g_c.nalgebra() * v) - M2::identity()).norm() < 1e-10,
        "V not Gc-orthonormal"
    );

    let recon: TallGen = gsvd.recompose(&g_c);
    for i in 0..3 {
        for j in 0..2 {
            assert!((recon.nalgebra()[(i, j)] - a.nalgebra()[(i, j)]).abs() < 1e-10);
        }
    }
}

#[test]
fn generalized_bidiagonalize_is_rectangular_through_a_thin_pivot() {
    // Tall 3×2 M against Gr (3×3) and Gc (2×2): U is 3×2, the band B is 2×2 and
    // dimensionless, V is 2×2. recompose rebuilds the exact 3×2 type.
    let a = tall_gen();
    let g_r = gr3();
    let g_c = gc2();

    let gbd = a
        .generalized_bidiagonalize(&g_r, &g_c)
        .expect("SPD metrics");
    assert_eq!(gbd.u.shape(), (3, 2));
    assert_eq!(gbd.d.shape(), (2, 2));
    assert_eq!(gbd.v.shape(), (2, 2));
    let _b00: qty!(1) = gbd.d.get(0, 0);
    let _u00: qty!(m) = gbd.u.get::<0, 0>();
    let _v00: qty!(s) = gbd.v.get::<0, 0>();

    let u = gbd.u.nalgebra();
    assert!(
        ((u.transpose() * g_r.nalgebra() * u) - M2::identity()).norm() < 1e-10,
        "U not Gr-orthonormal"
    );
    let v = gbd.v.nalgebra();
    assert!(
        ((v.transpose() * g_c.nalgebra() * v) - M2::identity()).norm() < 1e-10,
        "V not Gc-orthonormal"
    );

    let recon: TallGen = gbd.recompose(&g_c);
    for i in 0..3 {
        for j in 0..2 {
            assert!((recon.nalgebra()[(i, j)] - a.nalgebra()[(i, j)]).abs() < 1e-10);
        }
    }
}

#[test]
fn identity_is_a_dimensionless_diagonal_endomorphism() {
    // I : ⟨[m, s], [m, s]⟩ — the diagonal ones are dimensionless (Dims[i]/Dims[i]
    // = 1) and the off-diagonal zeros unit-agnostic, so one numeric identity
    // serves any space.
    type Endo = MixedUnitMatrix<dims![m, s], dims![m, s], M2>;
    let i = Endo::identity();
    let d00: qty!(1) = i.get::<0, 0>();
    assert_eq!(d00.unsafe_value, 1.0);
    assert_eq!(i.get::<0, 1>().unsafe_value, 0.0);
    assert_eq!(i.get::<1, 1>().unsafe_value, 1.0);

    // It is the right unit for endomorphism `Mul`: A · I = A with I over A's
    // column space [s, s]. The `Mat` annotation keeps A's exact type.
    type ColI = MixedUnitMatrix<dims![s, s], dims![s, s], M2>;
    let a = sample();
    let prod: Mat = a * ColI::identity();
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(prod.nalgebra()[(i, j)], a.nalgebra()[(i, j)]);
        }
    }
}

#[test]
fn from_diagonal_places_index_units_on_the_diagonal() {
    // An index column vector v : ⟨[m, s], [1]⟩ with entries [m: 3, s: 5].
    // from_diagonal builds ⟨[m, s], [1, 1]⟩ with v[i] at cell (i, i).
    type V = MixedUnitMatrix<dims![m, s], dims![1], SVector<f64, 2>>;
    let v = V::new(SVector::<f64, 2>::new(3.0, 5.0));

    type Diag = MixedUnitMatrix<dims![m, s], dims![1, 1], M2>;
    let d: Diag = MixedUnitMatrix::from_diagonal(&v);

    let d00: qty!(m) = d.get::<0, 0>();
    assert_eq!(d00.unsafe_value, 3.0);
    let d11: qty!(s) = d.get::<1, 1>();
    assert_eq!(d11.unsafe_value, 5.0);
    assert_eq!(d.get::<0, 1>().unsafe_value, 0.0);
    assert_eq!(d.get::<1, 0>().unsafe_value, 0.0);
}

#[test]
fn udu_factors_a_metric_with_a_dimensionless_diagonal() {
    // Same metric shape as cholesky/symmetric_eigen: <[1/m, 1/s], [m, s]> with
    // ColDims = 1/RowDims, SPD [[4, 1], [1, 4]]. U D Uᵀ keeps a separate,
    // dimensionless diagonal D, and U lands in the Cholesky-factor shape.
    type RowM = dims![1 / m, 1 / s];
    type ColM = dims![m, s];
    type Metric = MixedUnitMatrix<RowM, ColM, M2>;
    let m = Metric::new(M2::new(4.0, 1.0, 1.0, 4.0));

    let udu = m.udu().expect("factorizable");

    // U : ⟨[1/m, 1/s], [1, 1]⟩ (unit-upper-triangular, dimensionless columns).
    let _u00: qty!(1 / m) = udu.u.get::<0, 0>();
    // D is dimensionless (a metric's pivots are pure numbers).
    let dvals: Vec<f64> = udu.d.iter().map(|x: qty!(1)| x.unsafe_value).collect();
    assert_eq!(dvals.len(), 2);

    // U · diag(D) · Uᵀ reconstructs the metric type exactly.
    let recon: Metric = udu.recompose();
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (recon.nalgebra()[(i, j)] - m.nalgebra()[(i, j)]).abs() < 1e-10,
                "entry ({i}, {j}) not reconstructed"
            );
        }
    }
}

#[test]
fn generalized_pseudo_inverse_is_the_swapped_metric_inverse() {
    // sample : ⟨[m, m], [s, s]⟩ (invertible), against metrics Gr : ⟨[1/m,1/m],[m,m]⟩
    // and Gc : ⟨[1/s,1/s],[s,s]⟩. A⁺ has the clean swapped type ⟨[s, s], [m, m]⟩ —
    // the same as try_inverse — and satisfies A A⁺ A = A for any SPD metrics.
    type Gr = MixedUnitMatrix<dims![1 / m, 1 / m], dims![m, m], M2>;
    type Gc = MixedUnitMatrix<dims![1 / s, 1 / s], dims![s, s], M2>;

    let a = sample();
    let g_r = Gr::new(M2::new(2.0, 0.5, 0.5, 2.0));
    let g_c = Gc::new(M2::new(3.0, 0.0, 0.0, 1.0));

    let pinv = a
        .generalized_pseudo_inverse(&g_r, &g_c, 1e-12)
        .expect("SPD metrics");
    let _p00: qty!(s / m) = pinv.get::<0, 0>();

    // Penrose A A⁺ A = A lands back in A's exact type ⟨[m, m], [s, s]⟩.
    let recon: Mat = a * pinv * a;
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (recon.nalgebra()[(i, j)] - a.nalgebra()[(i, j)]).abs() < 1e-10,
                "entry ({i}, {j}) not reconstructed"
            );
        }
    }
}

#[test]
fn generalized_pseudo_inverse_of_a_tall_matrix_swaps_and_reconstructs() {
    // A genuinely rectangular case: the tall 3×2 map ⟨[m, m, m], [s, s]⟩, full
    // column rank, against Gr (3×3) and Gc (2×2). A⁺ is the wide 2×3
    // ⟨[s, s], [m, m, m]⟩ — the swapped type — and Penrose A A⁺ A = A lands back in
    // A's exact type ⟨[m, m, m], [s, s]⟩ for any SPD metrics.
    let a = tall_gen();
    let g_r = gr3();
    let g_c = gc2();

    let pinv = a
        .generalized_pseudo_inverse(&g_r, &g_c, 1e-12)
        .expect("SPD metrics");
    let _p00: qty!(s / m) = pinv.get::<0, 0>();
    assert_eq!(pinv.shape(), (2, 3));

    let recon: TallGen = a * pinv * a;
    for i in 0..3 {
        for j in 0..2 {
            assert!(
                (recon.nalgebra()[(i, j)] - a.nalgebra()[(i, j)]).abs() < 1e-10,
                "entry ({i}, {j}) not reconstructed"
            );
        }
    }
}

#[test]
fn tr_mul_and_ad_mul_match_transpose_times() {
    // self : ⟨[m, m], [s, s]⟩, so selfᵀ : ⟨[1/s, 1/s], [1/m, 1/m]⟩. tr_mul needs
    // rhs's rows to be 1/RowDims = [1/m, 1/m]; take rhs : ⟨[1/m, 1/m], [1, 1]⟩.
    let a = sample();
    type Rhs = MixedUnitMatrix<dims![1 / m, 1 / m], dims![1, 1], M2>;
    let b = Rhs::new(M2::new(1.0, 2.0, 3.0, 4.0));

    // aᵀ · b : ⟨[1/s, 1/s], [1, 1]⟩ — the same type and values as transpose() * b.
    let c = a.tr_mul(&b);
    let _c00: qty!(1 / s) = c.get::<0, 0>();
    let reference: MixedUnitMatrix<dims![1 / s, 1 / s], dims![1, 1], M2> = a.transpose() * b;
    for i in 0..2 {
        for j in 0..2 {
            assert!((c.nalgebra()[(i, j)] - reference.nalgebra()[(i, j)]).abs() < 1e-12);
            // ad_mul coincides with tr_mul over the reals.
            assert!((a.ad_mul(&b).nalgebra()[(i, j)] - c.nalgebra()[(i, j)]).abs() < 1e-12);
        }
    }
}

#[test]
fn triangle_extraction_is_type_preserving() {
    // sample() : ⟨[m, m], [s, s]⟩, [[1, 2], [3, 4]], every entry m/s. Zeroing a
    // triangle keeps the exact ⟨RowDims, ColDims⟩ type — no uniformity required.
    let upper = sample().upper_triangle(); // [[1, 2], [0, 4]]
    let u10: qty!(m / s) = upper.get::<1, 0>();
    let u01: qty!(m / s) = upper.get::<0, 1>();
    assert_eq!(u10.unsafe_value, 0.0);
    assert_eq!(u01.unsafe_value, 2.0);

    let lower = sample().lower_triangle(); // [[1, 0], [3, 4]]
    let l01: qty!(m / s) = lower.get::<0, 1>();
    let l10: qty!(m / s) = lower.get::<1, 0>();
    assert_eq!(l01.unsafe_value, 0.0);
    assert_eq!(l10.unsafe_value, 3.0);
}
