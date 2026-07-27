#![cfg(feature = "nalgebra")]
//! [`UniformUnitMatrix`]: the homogeneous specialization whose entries all share
//! one unit `U`. This exercises the unambiguous collapse from a uniform
//! [`MixedUnitMatrix`] (`.into_uniform()`), the bulk unit-safe access the mixed type
//! cannot offer (runtime `get`, `iter`, `map`), and the specialized algebra
//! (transpose keeps the unit; the matrix product multiplies entry units).

use whippyalgebra::dims;
use whippyalgebra::nalgebra::{
    MixedUnitMatrix, SMatrix, SVector, UniformUnitMatrix, block_matrix, gauge,
    rescale_uniform_matrix, unblock_matrix, zeros,
};
use whippyunits::{qty, quantity, unit};

type M2 = SMatrix<f64, 2, 2>;

// A mixed matrix whose row list is uniformly `m` and column list uniformly `s`,
// so every entry has the single unit m/s and it can collapse to uniform.
type UniformMixed = MixedUnitMatrix<dims![m, m], dims![s, s], M2>;

fn speeds() -> UniformUnitMatrix<unit!(m / s), M2> {
    // Row-major: (0,0)=1 (0,1)=2 (1,0)=3 (1,1)=4, every entry m/s.
    UniformMixed::new(M2::new(1.0, 2.0, 3.0, 4.0)).into_uniform()
}

#[test]
fn collapse_yields_shared_entry_unit() {
    let m = speeds();
    // Runtime indices are sound now: the unit no longer depends on which entry.
    let a: qty!(m / s) = m.get(0, 0);
    let d: qty!(m / s) = m.get(1, 1);
    assert_eq!(a.unsafe_value, 1.0);
    assert_eq!(d.unsafe_value, 4.0);
    assert_eq!(m.shape(), (2, 2));
}

#[test]
fn iter_walks_every_entry_as_a_quantity() {
    let m = speeds();
    // Column-major order: 1, 3, 2, 4.
    let vals: Vec<f64> = m.iter().map(|q: qty!(m / s)| q.unsafe_value).collect();
    assert_eq!(vals, vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn map_preserves_unit_and_transforms_values() {
    let m = speeds();
    let doubled = m.map(|q: qty!(m / s)| quantity!(q.unsafe_value * 2.0, m / s));
    let a: qty!(m / s) = doubled.get(0, 0);
    assert_eq!(a.unsafe_value, 2.0);
}

#[test]
fn rescale_uniform_matrix_rescales_every_entry_by_one_factor() {
    // speeds() is uniform in m/s, values [[1, 2], [3, 4]]. Rescaling to mm/s is a
    // single gauge change (× 1000) applied to every entry — the whole matrix at
    // once, the direct analog of scalar `rescale`.
    let m = speeds();
    let mm: UniformUnitMatrix<unit!(mm / s), M2> = rescale_uniform_matrix(&m);
    let a: qty!(mm / s) = mm.get(0, 0);
    let d: qty!(mm / s) = mm.get(1, 1);
    assert!((a.unsafe_value - 1000.0).abs() < 1e-9);
    assert!((d.unsafe_value - 4000.0).abs() < 1e-9);

    // Rescaling to km/h converts m/s → km/h by the factor 3.6.
    let kmh: UniformUnitMatrix<unit!(km / h), M2> = rescale_uniform_matrix(&m);
    let v: qty!(km / h) = kmh.get(0, 0);
    assert!((v.unsafe_value - 3.6).abs() < 1e-9);
}

#[test]
fn transpose_keeps_the_unit() {
    // Uniform transpose changes only the shape, not the entry unit.
    let t = speeds().transpose();
    let off: qty!(m / s) = t.get(0, 1); // was (1,0) = 3
    assert_eq!(off.unsafe_value, 3.0);
}

#[test]
fn adjoint_keeps_the_unit() {
    // Over a real field the uniform adjoint is the transpose: shape swaps, unit
    // stays m/s.
    let a = speeds().adjoint();
    let off: qty!(m / s) = a.get(0, 1); // was (1,0) = 3
    assert_eq!(off.unsafe_value, 3.0);
}

#[test]
fn trace_sums_the_diagonal_in_the_entry_unit() {
    // Every diagonal entry already carries the one unit m/s (no homogeneity gate
    // needed), so the trace lands in m/s. diag entries 1 + 4 = 5.
    let tr: qty!(m / s) = speeds().trace();
    assert_eq!(tr.unsafe_value, 5.0);
}

#[test]
fn reductions_land_in_the_entry_unit() {
    // speeds() = [[1, 2], [3, 4]] m/s. Scalar reductions collapse the whole
    // matrix, each staying in the shared unit (or its square).
    let m = speeds();

    let sum: qty!(m / s) = m.sum();
    assert_eq!(sum.unsafe_value, 10.0);

    let mean: qty!(m / s) = m.mean();
    assert_eq!(mean.unsafe_value, 2.5);

    // Frobenius norm √(1+4+9+16) = √30, in m/s; its square 30 in (m/s)².
    let norm: qty!(m / s) = m.norm();
    assert!((norm.unsafe_value - 30.0_f64.sqrt()).abs() < 1e-12);
    let nsq: qty!(m ^ 2 / s ^ 2) = m.norm_squared();
    assert!((nsq.unsafe_value - 30.0).abs() < 1e-12);

    // Order reductions return an actual entry, in m/s.
    let max: qty!(m / s) = m.max();
    let min: qty!(m / s) = m.min();
    assert_eq!(max.unsafe_value, 4.0);
    assert_eq!(min.unsafe_value, 1.0);
    let amax: qty!(m / s) = m.amax();
    assert_eq!(amax.unsafe_value, 4.0);
}

#[test]
fn lu_splits_dimensionless_l_from_unit_u() {
    // speeds() = [[1, 2], [3, 4]] m/s. Partial-pivot LU sends the unit-diagonal
    // L to dimensionless and keeps U in m/s (its diagonal are the pivots); the
    // row permutation P is dimensionless. (Col 0's largest is 3, so P is a swap.)
    let m = speeds();
    let lu = m.lu();

    let _l00: qty!(1) = lu.l.get(0, 0); // unit diagonal ⇒ 1, dimensionless
    let _u00: qty!(m / s) = lu.u.get(0, 0);
    let _p00: qty!(1) = lu.p.get(0, 0);

    // Pᵀ·L·U reconstructs the original, in m/s.
    let recon: UniformUnitMatrix<unit!(m / s), M2> = lu.recompose();
    for i in 0..2 {
        for j in 0..2 {
            assert!((recon.nalgebra()[(i, j)] - m.nalgebra()[(i, j)]).abs() < 1e-12);
        }
    }
}

#[test]
fn bidiagonalize_bands_the_unit_with_dimensionless_frames() {
    let m = speeds();
    let bidiag = m.bidiagonalize();
    let _u00: qty!(1) = bidiag.u.get(0, 0);
    let _d00: qty!(m / s) = bidiag.d.get(0, 0);
    let _v00: qty!(1) = bidiag.v_t.get(0, 0);

    let recon: UniformUnitMatrix<unit!(m / s), M2> = bidiag.recompose();
    for i in 0..2 {
        for j in 0..2 {
            assert!((recon.nalgebra()[(i, j)] - m.nalgebra()[(i, j)]).abs() < 1e-10);
        }
    }
}

#[test]
fn hessenberg_keeps_q_dimensionless_and_h_in_the_unit() {
    let m = speeds();
    let hess = m.hessenberg();
    let _q00: qty!(1) = hess.q.get(0, 0);
    let _h00: qty!(m / s) = hess.h.get(0, 0);

    let recon: UniformUnitMatrix<unit!(m / s), M2> = hess.recompose();
    for i in 0..2 {
        for j in 0..2 {
            assert!((recon.nalgebra()[(i, j)] - m.nalgebra()[(i, j)]).abs() < 1e-10);
        }
    }
}

#[test]
fn symmetric_tridiagonalize_keeps_q_dimensionless_and_t_in_the_unit() {
    // A symmetric uniform matrix (m/s everywhere) so the reduction round-trips.
    let m = UniformMixed::new(M2::new(1.0, 2.0, 2.0, 5.0)).into_uniform();
    let tri = m.symmetric_tridiagonalize();
    let _q00: qty!(1) = tri.q.get(0, 0);
    let _t00: qty!(m / s) = tri.t.get(0, 0);

    let recon: UniformUnitMatrix<unit!(m / s), M2> = tri.recompose();
    for i in 0..2 {
        for j in 0..2 {
            assert!((recon.nalgebra()[(i, j)] - m.nalgebra()[(i, j)]).abs() < 1e-10);
        }
    }
}

#[test]
fn schur_frame_is_dimensionless_and_t_carries_the_unit() {
    // speeds() = [[1, 2], [3, 4]] m/s (real spectrum). Uniform Schur: Q
    // dimensionless orthonormal, T in m/s with the eigenvalues on its diagonal.
    let m = speeds();
    let schur = m.schur();

    let _q00: qty!(1) = schur.q.get(0, 0);
    let _t00: qty!(m / s) = schur.t.get(0, 0);

    // Q·T·Qᵀ reconstructs the original, in m/s.
    let recon: UniformUnitMatrix<unit!(m / s), M2> = schur.recompose();
    for i in 0..2 {
        for j in 0..2 {
            assert!((recon.nalgebra()[(i, j)] - m.nalgebra()[(i, j)]).abs() < 1e-10);
        }
    }
}

#[test]
fn full_piv_lu_reconstructs_through_two_permutations() {
    let m = speeds();
    let lu = m.full_piv_lu();
    let _l00: qty!(1) = lu.l.get(0, 0);
    let _u00: qty!(m / s) = lu.u.get(0, 0);

    // Pᵀ·L·U·Qᵀ reconstructs the original.
    let recon: UniformUnitMatrix<unit!(m / s), M2> = lu.recompose();
    for i in 0..2 {
        for j in 0..2 {
            assert!((recon.nalgebra()[(i, j)] - m.nalgebra()[(i, j)]).abs() < 1e-12);
        }
    }
}

#[test]
fn identity_is_the_dimensionless_uniform_identity() {
    // A uniform identity must be dimensionless: its diagonal ones pin the shared
    // unit to 1 (a unit-carrying diagonal is the mixed ⟨Dims, Dims⟩ identity).
    let i = UniformUnitMatrix::<unit!(1), M2>::identity();
    let d00: qty!(1) = i.get(0, 0);
    assert_eq!(d00.unsafe_value, 1.0);
    assert_eq!(i.get(0, 1).unsafe_value, 0.0);
    assert_eq!(i.get(1, 1).unsafe_value, 1.0);
}

#[test]
fn from_diagonal_keeps_the_uniform_unit() {
    // A uniform column vector in m/s becomes a diagonal matrix uniform in m/s —
    // no gauge choice, unlike the mixed `from_diagonal`.
    let v = UniformUnitMatrix::<unit!(m / s), SVector<f64, 2>>::from_nalgebra(
        SVector::<f64, 2>::new(3.0, 5.0),
    );
    let d = UniformUnitMatrix::<unit!(m / s), M2>::from_diagonal(&v);
    let d00: qty!(m / s) = d.get(0, 0);
    assert_eq!(d00.unsafe_value, 3.0);
    let d11: qty!(m / s) = d.get(1, 1);
    assert_eq!(d11.unsafe_value, 5.0);
    assert_eq!(d.get(0, 1).unsafe_value, 0.0);
}

#[test]
fn col_piv_qr_is_ungated_with_a_unit_r() {
    // Uniform on both margins ⇒ the runtime column pivot is type-invisible. Q and
    // P are dimensionless, R carries m/s.
    let m = speeds();
    let d = m.col_piv_qr();
    let _q00: qty!(1) = d.q.get(0, 0);
    let _r00: qty!(m / s) = d.r.get(0, 0);
    let _p00: qty!(1) = d.p.get(0, 0);

    // Q·R·Pᵀ reconstructs the original, in m/s.
    let recon: UniformUnitMatrix<unit!(m / s), M2> = d.recompose();
    for i in 0..2 {
        for j in 0..2 {
            assert!((recon.nalgebra()[(i, j)] - m.nalgebra()[(i, j)]).abs() < 1e-10);
        }
    }
}

#[test]
fn uniform_col_piv_qr_held_factor_reuses_solve() {
    // speeds() = [[1, 2], [3, 4]] (m/s). The held column-pivoted QR solves
    // M x = b (x = P R⁻¹ Qᴴ b) without re-factoring. b in m ⇒ x in m/(m/s) = s.
    let d = speeds().col_piv_qr();

    // [[1, 2], [3, 4]] · [1, 2] = [5, 11], so x_true = [1 s, 2 s].
    type V2c = SMatrix<f64, 2, 1>;
    let b = UniformUnitMatrix::<unit!(m), V2c>::from_nalgebra(V2c::new(5.0, 11.0));
    let x = d.solve(&b).expect("nonsingular");
    let x0: qty!(s) = x.get(0, 0);
    let x1: qty!(s) = x.get(1, 0);
    assert!((x0.unsafe_value - 1.0).abs() < 1e-10);
    assert!((x1.unsafe_value - 2.0).abs() < 1e-10);
}

#[test]
fn uniform_qr_reconstructs_and_reuses_solve() {
    // speeds() = [[1, 2], [3, 4]] (m/s). Plain (unpivoted) QR: Q dimensionless,
    // R in m/s, so Q·R reconstructs the m/s matrix.
    let m = speeds();
    let qr = m.qr();
    let _q00: qty!(1) = qr.q.get(0, 0);
    let _r00: qty!(m / s) = qr.r.get(0, 0);

    let recon: UniformUnitMatrix<unit!(m / s), M2> = qr.recompose();
    for i in 0..2 {
        for j in 0..2 {
            assert!((recon.nalgebra()[(i, j)] - m.nalgebra()[(i, j)]).abs() < 1e-10);
        }
    }

    // Reuse solve: b in m ⇒ x in m/(m/s) = s. [[1, 2], [3, 4]] · [1, 2] = [5, 11].
    type V2c = SMatrix<f64, 2, 1>;
    let b = UniformUnitMatrix::<unit!(m), V2c>::from_nalgebra(V2c::new(5.0, 11.0));
    let x = qr.solve(&b).expect("nonsingular");
    let x0: qty!(s) = x.get(0, 0);
    let x1: qty!(s) = x.get(1, 0);
    assert!((x0.unsafe_value - 1.0).abs() < 1e-10);
    assert!((x1.unsafe_value - 2.0).abs() < 1e-10);

    // try_inverse reciprocates the unit to s/m (1 / (m/s)).
    let inv = qr.try_inverse().expect("nonsingular");
    let _i00: qty!(s / m) = inv.get(0, 0);
    // M · M⁻¹ ≈ I.
    let prod = m.nalgebra() * inv.nalgebra();
    assert!((prod[(0, 0)] - 1.0).abs() < 1e-10);
    assert!((prod[(0, 1)]).abs() < 1e-10);
}

#[test]
fn uniform_from_element_and_from_fn_are_unit_safe() {
    // from_element fills every cell with one Quantity of the shared unit.
    let filled = UniformUnitMatrix::<unit!(m / s), M2>::from_element(quantity!(3.0, m / s));
    for q in filled.iter() {
        let v: qty!(m / s) = q;
        assert_eq!(v.unsafe_value, 3.0);
    }

    // from_fn builds each cell from (i, j); the closure must return the entry unit.
    let ramp = UniformUnitMatrix::<unit!(m / s), M2>::from_fn(|i, j| {
        quantity!((i * 10 + j) as f64, m / s)
    });
    let e10: qty!(m / s) = ramp.get(1, 0);
    let e01: qty!(m / s) = ramp.get(0, 1);
    assert_eq!(e10.unsafe_value, 10.0);
    assert_eq!(e01.unsafe_value, 1.0);
}

#[test]
fn udu_keeps_a_dimensionless_frame_and_a_unit_diagonal() {
    // A symmetric uniform matrix (m/s everywhere). Unlike uniform Cholesky (which
    // would need √(m/s)), U D Uᵀ keeps a separate diagonal that carries the whole
    // unit, so U is dimensionless and D is in m/s — ungated.
    let m = UniformMixed::new(M2::new(4.0, 1.0, 1.0, 4.0)).into_uniform();
    let udu = m.udu().expect("factorizable");
    let _u00: qty!(1) = udu.u.get(0, 0);
    let _d0: qty!(m / s) = udu.d.get(0, 0);

    let recon: UniformUnitMatrix<unit!(m / s), M2> = udu.recompose();
    for i in 0..2 {
        for j in 0..2 {
            assert!((recon.nalgebra()[(i, j)] - m.nalgebra()[(i, j)]).abs() < 1e-10);
        }
    }
}

#[test]
fn pseudo_inverse_reciprocates_the_unit() {
    // speeds() (m/s, invertible). A⁺ is uniform in s/m and satisfies A A⁺ A = A.
    let a = speeds();
    let pinv = a.pseudo_inverse(1e-12).expect("full rank");
    let _p00: qty!(s / m) = pinv.get(0, 0);

    let recon = a * pinv * a; // (m/s)·(s/m)·(m/s) = m/s
    let r00: qty!(m / s) = recon.get(0, 0);
    for i in 0..2 {
        for j in 0..2 {
            assert!((recon.nalgebra()[(i, j)] - a.nalgebra()[(i, j)]).abs() < 1e-10);
        }
    }
    let _ = r00;
}

#[test]
fn pseudo_inverse_of_a_tall_uniform_reciprocates_and_reshapes() {
    // A non-square uniform case: a tall 3×2 matrix in m/s (full column rank). A⁺
    // is the wide 2×3 uniform matrix in the reciprocal unit s/m, and A A⁺ A = A.
    type Tall = MixedUnitMatrix<dims![m, m, m], dims![s, s], SMatrix<f64, 3, 2>>;
    let a = Tall::new(SMatrix::<f64, 3, 2>::new(1.0, 2.0, 3.0, 4.0, 5.0, 7.0)).into_uniform();

    let pinv = a.pseudo_inverse(1e-12).expect("full column rank");
    let _p00: qty!(s / m) = pinv.get(0, 0);
    assert_eq!(pinv.shape(), (2, 3));

    let recon = a * pinv * a; // (m/s)·(s/m)·(m/s) = m/s
    let _r00: qty!(m / s) = recon.get(0, 0);
    for i in 0..3 {
        for j in 0..2 {
            assert!((recon.get(i, j).unsafe_value - a.get(i, j).unsafe_value).abs() < 1e-10);
        }
    }
}

#[test]
fn tr_mul_and_ad_mul_multiply_the_units() {
    // aᵀ · b with a in m/s and b in s lands in m (uniform transpose keeps the
    // unit, so it multiplies just like the ordinary product).
    let a = speeds();
    type SecMat = MixedUnitMatrix<dims![s, s], dims![1, 1], M2>;
    let b: UniformUnitMatrix<unit!(s), M2> =
        SecMat::new(M2::new(1.0, 2.0, 3.0, 4.0)).into_uniform();

    let c = a.tr_mul(&b);
    let _c00: qty!(m) = c.get(0, 0);
    let reference = a.transpose() * b; // both in m
    for i in 0..2 {
        for j in 0..2 {
            assert!((c.nalgebra()[(i, j)] - reference.nalgebra()[(i, j)]).abs() < 1e-12);
            assert!((a.ad_mul(&b).nalgebra()[(i, j)] - c.nalgebra()[(i, j)]).abs() < 1e-12);
        }
    }
}

#[test]
fn component_mul_and_div_transform_the_unit() {
    // Hadamard product/quotient of a uniform m/s with a uniform s.
    let a = speeds(); // m/s, [[1, 2], [3, 4]]
    type SecMat = MixedUnitMatrix<dims![s, s], dims![1, 1], M2>;
    let b: UniformUnitMatrix<unit!(s), M2> =
        SecMat::new(M2::new(2.0, 2.0, 2.0, 2.0)).into_uniform();

    // (m/s) · s = m, entry-wise.
    let prod = a.component_mul(&b);
    let p00: qty!(m) = prod.get(0, 0);
    assert_eq!(p00.unsafe_value, 2.0);

    // (m/s) / s = m/s², entry-wise.
    let quot = a.component_div(&b);
    let q11: qty!(m / s ^ 2) = quot.get(1, 1);
    assert_eq!(q11.unsafe_value, 2.0); // 4 / 2
}

#[test]
fn inf_and_sup_keep_the_unit() {
    // Element-wise min/max of two m/s matrices stays in m/s.
    let a = speeds(); // [[1, 2], [3, 4]]
    let b = UniformMixed::new(M2::new(4.0, 1.0, 1.0, 5.0)).into_uniform();

    let lo = a.inf(&b);
    let hi = a.sup(&b);
    let lo00: qty!(m / s) = lo.get(0, 0);
    let hi00: qty!(m / s) = hi.get(0, 0);
    assert_eq!(lo00.unsafe_value, 1.0); // min(1, 4)
    assert_eq!(hi00.unsafe_value, 4.0); // max(1, 4)
}

#[test]
fn dot_multiplies_the_two_entry_units() {
    // (m/s) · s = m: a dot against a uniform matrix in s lands in m.
    let a = speeds(); // m/s, [[1, 2], [3, 4]]
    type SecMat = MixedUnitMatrix<dims![s, s], dims![1, 1], M2>;
    let b: UniformUnitMatrix<unit!(s), M2> =
        SecMat::new(M2::new(1.0, 1.0, 1.0, 1.0)).into_uniform();

    let d: qty!(m) = a.dot(&b);
    assert_eq!(d.unsafe_value, 10.0); // Σ a_ij·1 = 1+2+3+4
}

#[test]
fn matrix_product_multiplies_entry_units() {
    // (m/s) · (s) = m: the product's entry unit is the product of the operands'.
    let a = speeds(); // entries m/s
    type SecMat = MixedUnitMatrix<dims![s, s], dims![1, 1], M2>;
    let b: UniformUnitMatrix<unit!(s), M2> =
        SecMat::new(M2::new(1.0, 0.0, 0.0, 1.0)).into_uniform();

    let c = a * b; // UniformUnitMatrix<m, _>
    let e: qty!(m) = c.get(0, 0);
    assert_eq!(e.unsafe_value, 1.0);
}

#[test]
fn add_and_scalar_scaling() {
    let a = speeds();
    let b = speeds();
    let sum = a + b;
    let s00: qty!(m / s) = sum.get(0, 0);
    assert_eq!(s00.unsafe_value, 2.0);

    // Scalar-by-quantity multiplication folds the scalar's unit into the entry
    // unit: (m/s) · s = m.
    let scaled = speeds() * quantity!(3.0, s);
    let p: qty!(m) = scaled.get(0, 0);
    assert_eq!(p.unsafe_value, 3.0);
}

// A 2x3 matrix whose rows are uniformly `m` and whose columns are `[s, K, s]`.
// A column-1 slice (cols `[s]`) is uniform, so a block over it reduces to
// `UniformUnitMatrix`; a slice over cols `[K, s]` is not, so it stays mixed —
// letting one unblock grid exercise both branches of the auto-reduction.
type SplitRows = dims![m, m];
type SplitCols = dims![s, K, s];
type M23 = SMatrix<f64, 2, 3>;

fn split() -> MixedUnitMatrix<SplitRows, SplitCols, M23> {
    // Row-major: row0 = 1,2,3 ; row1 = 4,5,6.
    MixedUnitMatrix::new(M23::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
}

#[test]
fn unblock_auto_reduces_uniform_blocks() {
    let m = split();

    // `reduce_uniform;` (uniform extraction is opt-in): `left` (rows [m,m], cols
    // [s]) is uniform → UniformUnitMatrix<m/s>; `right` (rows [m,m], cols [K,s])
    // is not → mixed.
    unblock_matrix!(reduce_uniform; m => [
        [left(2, 1), right(2, 2)],
    ]);

    // Uniform block: runtime `get` is sound because the unit is shared.
    let l0: qty!(m / s) = left.get(0, 0);
    let l1: qty!(m / s) = left.get(1, 0);
    assert_eq!(l0.unsafe_value, 1.0);
    assert_eq!(l1.unsafe_value, 4.0);

    // Mixed block: compile-time `get`, per-entry units (K in col 0, s in col 1).
    let r00: qty!(m / K) = right.get::<0, 0>();
    let r01: qty!(m / s) = right.get::<0, 1>();
    assert_eq!(r00.unsafe_value, 2.0);
    assert_eq!(r01.unsafe_value, 3.0);
}

#[test]
fn unblock_default_mode_keeps_uniform_blocks_mixed() {
    let m = split();

    // The default (no keyword) is `mixed`: even the uniform `left` stays mixed
    // (gauge retained), so it is indexed with compile-time `get::<_, _>` (a
    // UniformUnitMatrix would take runtime indices instead). `mixed;` is accepted
    // as the explicit spelling of the same thing.
    unblock_matrix!(m => [
        [left(2, 1), right(2, 2)],
    ]);
    let _ = &right;

    let l00: qty!(m / s) = left.get::<0, 0>();
    let l10: qty!(m / s) = left.get::<1, 0>();
    assert_eq!(l00.unsafe_value, 1.0);
    assert_eq!(l10.unsafe_value, 4.0);
}

#[test]
fn unblock_views_auto_reduces_uniform_blocks() {
    let m = split();

    // Same auto-reduction over borrowing storage (opt-in via `views
    // reduce_uniform;`): `left` is a uniform *view*, `right` a mixed one, and
    // both borrow `m` at once.
    unblock_matrix!(views reduce_uniform; m => [
        [left(2, 1), right(2, 2)],
    ]);

    let l0: qty!(m / s) = left.get(0, 0); // uniform view → runtime get
    assert_eq!(l0.unsafe_value, 1.0);
    let r00: qty!(m / K) = right.get::<0, 0>(); // mixed view → compile-time get
    assert_eq!(r00.unsafe_value, 2.0);

    // Parent stays readable alongside the shared views.
    let m02: qty!(m / s) = m.get::<0, 2>();
    assert_eq!(m02.unsafe_value, 3.0);
}

#[test]
fn gauge_places_a_uniform_block_into_a_block_matrix() {
    // A mixed 2x2 (rows [m,m], cols [s,s]) and a *uniform* 2x1 block of unit m/s.
    let a = MixedUnitMatrix::<dims![m, m], dims![s, s], M2>::new(M2::new(1.0, 2.0, 3.0, 4.0));
    let g: UniformUnitMatrix<unit!(m / s), SMatrix<f64, 2, 1>> =
        MixedUnitMatrix::<dims![m, m], dims![s], SMatrix<f64, 2, 1>>::new(
            SMatrix::<f64, 2, 1>::new(10.0, 20.0),
        )
        .into_uniform();

    // Assemble [[a, g], [0, 0]] over rows [m, m, rot], cols [s, s, s]. The uniform
    // block is gauged with row unit m, column unit s — quotient exactly m/s, its
    // lengths taken from the block's 2x1 shape — so it lines up with `a` (shared
    // rows [m, m]) and the zero block below it (shared cols [s]).
    let m: MixedUnitMatrix<dims![m, m, rot], dims![s, s, s], _> = block_matrix![
        [a, gauge!(g => m, s)],
        [
            zeros![dims![rot], dims![s, s]],
            zeros![dims![rot], dims![s]]
        ],
    ];

    // `a` in the top-left reads back as m/s.
    let a00: qty!(m / s) = m.get::<0, 0>();
    assert_eq!(a00.unsafe_value, 1.0);

    // The gauged uniform block sits in the top-right column; its entries carry
    // the entry unit dictated by the layout (rows m / col s = m/s) and its values.
    let g00: qty!(m / s) = m.get::<0, 2>();
    let g10: qty!(m / s) = m.get::<1, 2>();
    assert_eq!(g00.unsafe_value, 10.0);
    assert_eq!(g10.unsafe_value, 20.0);
}

#[test]
fn gauge_copy_and_view_selectors_leave_the_source_usable() {
    // Same uniform 2x1 block of unit m/s, but re-gauged non-destructively: the
    // `copy;` and `view;` selectors both leave `g` intact, unlike the default
    // (consuming) form which would move it in.
    let g: UniformUnitMatrix<unit!(m / s), SMatrix<f64, 2, 1>> =
        MixedUnitMatrix::<dims![m, m], dims![s], SMatrix<f64, 2, 1>>::new(
            SMatrix::<f64, 2, 1>::new(10.0, 20.0),
        )
        .into_uniform();

    // `copy;` re-gauges into an owned mixed block; `g` stays usable afterward.
    let copied: MixedUnitMatrix<dims![m, m], dims![s], _> = gauge!(copy; g => m, s);
    assert_eq!(copied.get::<0, 0>().unsafe_value, 10.0);
    assert_eq!(copied.get::<1, 0>().unsafe_value, 20.0);

    // `view;` re-gauges into a zero-copy borrowing view; `g` is only borrowed and
    // remains readable. Both borrow the same source in a single block layout.
    let viewed: MixedUnitMatrix<dims![m, m], dims![s], _> = gauge!(view; g => m, s);
    assert_eq!(viewed.get::<0, 0>().unsafe_value, 10.0);
    assert_eq!(viewed.get::<1, 0>().unsafe_value, 20.0);

    // `g` survived both non-consuming re-gauges: read it back as the uniform m/s.
    let g_still: qty!(m / s) = g.get(0, 0);
    assert_eq!(g_still.unsafe_value, 10.0);
}

#[test]
fn decompose_then_recompose_round_trips_through_the_gauge() {
    // End to end: tear a mixed 3x3 into blocks with `reduce_uniform;` — the blocks whose
    // row and column sublists are both uniform come back gauge-*erased* as
    // `UniformUnitMatrix`, the rest stay mixed with their gauge intact — then
    // reassemble the *same* matrix with `block_matrix!`, re-entering the erased
    // gauge on the uniform blocks via `gauge!`. The reconstruction is pinned to
    // the original's dimension lists, so the round trip is checked at the type
    // level (units line up entry for entry), not just in the stored values.
    type M3 = SMatrix<f64, 3, 3>;
    // Rows [m, m, s], cols [s, s, K]; row-major 1..=9.
    let original = MixedUnitMatrix::<dims![m, m, s], dims![s, s, K], M3>::new(M3::new(
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
    ));

    // Decompose with uniform extraction opt-in. The [2,1] x [1,2] partition:
    //   tl ⟨[m,m],[s]⟩   → uniform m/s   (both sublists constant → gauge erased)
    //   tr ⟨[m,m],[s,K]⟩ → mixed         (columns [s, K] are not uniform)
    //   bl ⟨[s],[s]⟩     → uniform s/s
    //   br ⟨[s],[s,K]⟩   → mixed
    unblock_matrix!(reduce_uniform; original => [
        [tl(2, 1), tr(2, 2)],
        [bl(1, 1), br(1, 2)],
    ]);

    // The uniform blocks dropped their gauge: single entry unit, runtime index.
    let tl00: qty!(m / s) = tl.get(0, 0);
    assert_eq!(tl00.unsafe_value, 1.0);
    let bl00: qty!(s / s) = bl.get(0, 0);
    assert_eq!(bl00.unsafe_value, 7.0);
    // The mixed blocks kept theirs: per-entry units, compile-time index.
    let tr01: qty!(m / K) = tr.get::<0, 1>();
    assert_eq!(tr01.unsafe_value, 3.0);

    // Recompose. `gauge!` re-enters the erased gauge on the uniform blocks,
    // pinning each to the row/column spaces its neighbors demand (a wrong tag
    // would not compile); the mixed blocks slot straight back in. The `let`
    // annotation pins the whole result to the original's lists.
    let rebuilt: MixedUnitMatrix<dims![m, m, s], dims![s, s, K], _> =
        block_matrix![[gauge!(tl => m, s), tr], [gauge!(bl => s, s), br],];

    assert_eq!(rebuilt.shape(), (3, 3));

    // Every stored value matches the original; the pinned type above already
    // guarantees the reconstructed units match, entry for entry.
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(
                rebuilt.nalgebra()[(i, j)],
                original.nalgebra()[(i, j)],
                "entry ({i}, {j}) differs after the decompose/recompose round trip"
            );
        }
    }

    // Spot-check a reconstructed entry from a re-gauged block, with its unit.
    let r22: qty!(s / K) = rebuilt.get::<2, 2>();
    assert_eq!(r22.unsafe_value, 9.0);
}

#[test]
fn to_uniform_copies_and_keeps_the_gauge_matrix() {
    // A uniform mixed matrix (rows [m,m], cols [s,s]); copy it *as uniform* to use
    // bulk ops, while the original mixed matrix (with its gauge) stays usable.
    let mixed = UniformMixed::new(M2::new(1.0, 2.0, 3.0, 4.0));
    let u = mixed.to_uniform(); // owned UniformUnitMatrix<m/s>, `mixed` untouched

    // The copy exposes the uniform API (runtime get, single unit m/s)...
    let u00: qty!(m / s) = u.get(0, 0);
    assert_eq!(u00.unsafe_value, 1.0);

    // ...and `mixed` is still around with its gauge intact (compile-time,
    // per-entry indexing that a uniform matrix could not offer).
    let m01: qty!(m / s) = mixed.get::<0, 1>();
    assert_eq!(m01.unsafe_value, 2.0);
}

#[test]
fn as_uniform_views_without_erasing_the_original() {
    let mixed = UniformMixed::new(M2::new(1.0, 2.0, 3.0, 4.0));

    // Borrow as a uniform view (zero-copy): reads carry the single unit m/s.
    let v = mixed.as_uniform();
    let v10: qty!(m / s) = v.get(1, 0);
    assert_eq!(v10.unsafe_value, 3.0);

    // The shared borrow leaves `mixed` readable alongside the view.
    let m11: qty!(m / s) = mixed.get::<1, 1>();
    assert_eq!(m11.unsafe_value, 4.0);
}

#[test]
fn uniform_cholesky_roots_an_even_exponent_unit() {
    // A uniform matrix whose single entry unit is m² (an even exponent, so its
    // square root m is representable): rows [m, m] over cols [1/m, 1/m] gives
    // entry m·m = m². Numeric [[4, 2], [2, 4]] is SPD.
    let m: UniformUnitMatrix<unit!(m ^ 2), M2> =
        MixedUnitMatrix::<dims![m, m], dims![1 / m, 1 / m], M2>::new(M2::new(4.0, 2.0, 2.0, 4.0))
            .into_uniform();

    let l = m.cholesky().expect("positive-definite").l;

    // The factor is uniform in √(m²) = m; numerically L = [[2, 0], [1, √3]].
    let l00: qty!(m) = l.get(0, 0);
    assert_eq!(l00.unsafe_value, 2.0);
    let l10: qty!(m) = l.get(1, 0);
    assert_eq!(l10.unsafe_value, 1.0);
    let l11: qty!(m) = l.get(1, 1);
    assert!((l11.unsafe_value - 3.0_f64.sqrt()).abs() < 1e-12);

    // L·Lᵀ multiplies the entry units (m · m) straight back to m², reconstructing
    // M's type and values (the transpose keeps the uniform unit unchanged).
    let m2: UniformUnitMatrix<unit!(m ^ 2), _> = l * l.transpose();
    let r00: qty!(m ^ 2) = m2.get(0, 0);
    assert!((r00.unsafe_value - 4.0).abs() < 1e-12);
    let r01: qty!(m ^ 2) = m2.get(0, 1);
    assert!((r01.unsafe_value - 2.0).abs() < 1e-12);
}

#[test]
fn uniform_cholesky_held_factor_reuses_solve_and_inverse() {
    // Same SPD m² matrix [[4, 2], [2, 4]]. The held factorization reuses the
    // solve/inverse without re-factoring.
    let chol =
        MixedUnitMatrix::<dims![m, m], dims![1 / m, 1 / m], M2>::new(M2::new(4.0, 2.0, 2.0, 4.0))
            .into_uniform()
            .cholesky()
            .expect("positive-definite");

    // Solve M x = b with b in m⁴ ⇒ x in m⁴/m² = m². [[4, 2], [2, 4]] · [1, 1] =
    // [6, 6], so x_true = [1 m², 1 m²].
    type V2c = SMatrix<f64, 2, 1>;
    let b = UniformUnitMatrix::<unit!(m ^ 4), V2c>::from_nalgebra(V2c::new(6.0, 6.0));
    let x = chol.solve(&b);
    let x0: qty!(m ^ 2) = x.get(0, 0);
    let x1: qty!(m ^ 2) = x.get(1, 0);
    assert!((x0.unsafe_value - 1.0).abs() < 1e-12);
    assert!((x1.unsafe_value - 1.0).abs() < 1e-12);

    // Inverse reciprocates the unit to 1/m²; M⁻¹ = [[1/3, -1/6], [-1/6, 1/3]].
    let inv = chol.inverse();
    let i00: qty!(1 / m ^ 2) = inv.get(0, 0);
    assert!((i00.unsafe_value - 1.0 / 3.0).abs() < 1e-12);
}

#[test]
fn uniform_eigenvalues_carry_the_entry_unit() {
    // A uniform "rate" matrix in 1/s (e.g. a continuous state matrix already
    // reduced to uniform form). Numerically diagonal ⇒ eigenvalues {-2, -5}.
    let a: UniformUnitMatrix<unit!(1 / s), M2> = UniformUnitMatrix::from_nalgebra(M2::new(
        -2.0, 0.0, //
        0.0, -5.0,
    ));

    let ev = a.eigenvalues().expect("real spectrum");
    // The spectrum reads back in 1/s — the fully-uniform analogue of the mixed
    // uniform-endomorphism spectrum. A wrong unit here would not compile.
    let mut vals: Vec<f64> = ev.iter().map(|q: qty!(1 / s)| q.unsafe_value).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(vals, vec![-5.0, -2.0]);
}

#[test]
fn uniform_try_inverse_reciprocates_the_unit() {
    // A uniform matrix in s; its inverse is uniform in 1/s so that M⁻¹·M is the
    // dimensionless identity. Numeric diag(2, 4) ⇒ inverse diag(0.5, 0.25).
    let a: UniformUnitMatrix<unit!(s), M2> = UniformUnitMatrix::from_nalgebra(M2::new(
        2.0, 0.0, //
        0.0, 4.0,
    ));

    let inv = a.try_inverse().expect("invertible");
    let i00: qty!(1 / s) = inv.get(0, 0);
    let i11: qty!(1 / s) = inv.get(1, 1);
    assert!((i00.unsafe_value - 0.5).abs() < 1e-12);
    assert!((i11.unsafe_value - 0.25).abs() < 1e-12);
}

#[test]
fn uniform_svd_singular_values_carry_the_entry_unit() {
    // A uniform matrix in s. Unlike the mixed SVD (dimensionless σ), a uniform
    // matrix is its own canonical metric on both sides, so the singular values
    // are dimensioned in the entry unit s — and no even-exponent gate applies.
    // Numeric [[3, 0], [0, -4]] ⇒ singular values {4, 3}.
    let a: UniformUnitMatrix<unit!(s), M2> = UniformUnitMatrix::from_nalgebra(M2::new(
        3.0, 0.0, //
        0.0, -4.0,
    ));

    let svd = a.svd();
    // Singular values read back in s (a wrong unit would not compile).
    let mut sv: Vec<f64> = svd
        .singular_values
        .iter()
        .map(|q: qty!(s)| q.unsafe_value)
        .collect();
    sv.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert!((sv[0] - 4.0).abs() < 1e-12);
    assert!((sv[1] - 3.0).abs() < 1e-12);

    // The singular vectors are dimensionless and orthonormal.
    let _u00: qty!(1) = svd.u.get(0, 0);
    let _vt00: qty!(1) = svd.v_t.get(0, 0);

    // recompose rebuilds M in the entry unit s, matching the stored values.
    let recon: UniformUnitMatrix<unit!(s), M2> = svd.recompose();
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
fn pencil_generalized_eigenvalues_are_squared_frequencies() {
    // Stiffness/mass pencil `K v = λ M v`: the ungated path needs no square root,
    // so an *odd*-exponent mass (`kg`) is fine. K = diag(2, 8) N/m, M = diag(1, 2)
    // kg ⇒ M⁻¹K = diag(2, 4), and λ = ω² lands in (N/m)/kg = 1/s².
    let k: UniformUnitMatrix<unit!(N / m), M2> = UniformUnitMatrix::from_nalgebra(M2::new(
        2.0, 0.0, //
        0.0, 8.0,
    ));
    let m: UniformUnitMatrix<unit!(kg), M2> = UniformUnitMatrix::from_nalgebra(M2::new(
        1.0, 0.0, //
        0.0, 2.0,
    ));

    let omega_sq = k.generalized_eigenvalues(&m).expect("real spectrum");
    let mut vals: Vec<f64> = omega_sq
        .iter()
        .map(|q: qty!(1 / s ^ 2)| q.unsafe_value)
        .collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(vals, vec![2.0, 4.0]);
}

#[test]
fn generalized_symmetric_eigen_gives_real_spectrum_and_m_orthonormal_vectors() {
    // The `UnitSqrt`-gated variant: rooting the mass metric needs even exponents,
    // so use an even-exponent mass `s²`. K = diag(3, 5) (dimensionless), M = I in
    // s² ⇒ λ = {3, 5} in 1/s² and eigenvectors in 1/√(s²) = 1/s.
    let k: UniformUnitMatrix<unit!(1), M2> = UniformUnitMatrix::from_nalgebra(M2::new(
        3.0, 0.0, //
        0.0, 5.0,
    ));
    let m: UniformUnitMatrix<unit!(s ^ 2), M2> = UniformUnitMatrix::from_nalgebra(M2::identity());

    let eig = k
        .generalized_symmetric_eigen(&m)
        .expect("positive-definite mass");

    // Real generalized eigenvalues in ω² = 1/s².
    let mut vals: Vec<f64> = eig
        .eigenvalues
        .iter()
        .map(|q: qty!(1 / s ^ 2)| q.unsafe_value)
        .collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(vals, vec![3.0, 5.0]);

    // The eigenvectors carry 1/√(s²) = 1/s; a wrong unit would not compile.
    let _v00: qty!(1 / s) = eig.eigenvectors.get(0, 0);
    // They are M-orthonormal: Vᴴ M V = I (here M = I in s², so VᵀV = I).
    let v = eig.eigenvectors.nalgebra();
    let gram = v.transpose() * v;
    assert!((gram - M2::identity()).norm() < 1e-10);

    // recompose reconstructs K = M V Λ Vᴴ M back in the stiffness unit (here
    // dimensionless) and the original values diag(3, 5). A wrong unit annotation
    // on the reconstruction would not compile.
    let k_recon = eig.recompose(&m);
    let r00: qty!(1) = k_recon.get(0, 0);
    let r11: qty!(1) = k_recon.get(1, 1);
    let r01: qty!(1) = k_recon.get(0, 1);
    assert!((r00.unsafe_value - 3.0).abs() < 1e-10);
    assert!((r11.unsafe_value - 5.0).abs() < 1e-10);
    assert!(r01.unsafe_value.abs() < 1e-10);
}

#[test]
fn as_uniform_mut_mutates_in_place_and_keeps_the_gauge() {
    let mut mixed = UniformMixed::new(M2::new(1.0, 2.0, 3.0, 4.0));

    {
        // Exclusive mutable uniform view: mutate through the uniform-only API
        // (`set`, `[(i, j)]`, `iter_mut`) that the mixed matrix withholds.
        let mut v = mixed.as_uniform_mut();
        v.set(0, 0, quantity!(10.0, m / s)); // via set
        v[(0, 1)] = quantity!(20.0, m / s); // via IndexMut (unit-checked)
        for q in v.iter_mut() {
            // Double every entry, staying in unit m/s.
            *q = quantity!(q.unsafe_value * 2.0, m / s);
        }
    } // view dropped here → `mixed` usable again, gauge intact

    // `mixed` is still a MixedUnitMatrix<[m,m],[s,s]>: compile-time, per-entry
    // indexing (which a uniform matrix cannot offer) proves the gauge survived,
    // and the values reflect the in-place mutations (set/index, then doubled).
    let e00: qty!(m / s) = mixed.get::<0, 0>();
    let e01: qty!(m / s) = mixed.get::<0, 1>();
    let e10: qty!(m / s) = mixed.get::<1, 0>();
    assert_eq!(e00.unsafe_value, 20.0); // 10 set, then *2
    assert_eq!(e01.unsafe_value, 40.0); // 20 set, then *2
    assert_eq!(e10.unsafe_value, 6.0); // 3 (untouched), then *2
}

#[test]
fn uniform_solve_divides_the_entry_units() {
    // M in s, b in m ⇒ x in m/s. M = diag(2, 4) s, b = [2, 8] m ⇒ x = [1, 2] m/s.
    let m = UniformUnitMatrix::<unit!(s), M2>::from_nalgebra(M2::new(2.0, 0.0, 0.0, 4.0));
    let b = UniformUnitMatrix::<unit!(m), SVector<f64, 2>>::from_nalgebra(SVector::<f64, 2>::new(
        2.0, 8.0,
    ));

    let x = m.solve(&b).expect("invertible");
    let x0: qty!(m / s) = x.get(0, 0);
    let x1: qty!(m / s) = x.get(1, 0);
    assert_eq!(x0.unsafe_value, 1.0);
    assert_eq!(x1.unsafe_value, 2.0);

    // A lower-triangular forward substitution shares the m/s signature.
    let lower = UniformUnitMatrix::<unit!(s), M2>::from_nalgebra(M2::new(2.0, 0.0, 1.0, 4.0));
    let y = lower.solve_lower_triangular(&b).expect("nonzero diagonal");
    let _y0: qty!(m / s) = y.get(0, 0);
    assert!((lower.nalgebra() * y.nalgebra() - b.nalgebra()).norm() < 1e-12);
}

#[test]
fn uniform_lu_reuse_solve_matches_one_shot() {
    // Factor once, then reuse the held L/U/P for the solve (no re-factorization).
    let m = UniformUnitMatrix::<unit!(s), M2>::from_nalgebra(M2::new(2.0, 1.0, 1.0, 3.0));
    let b = UniformUnitMatrix::<unit!(m), SVector<f64, 2>>::from_nalgebra(SVector::<f64, 2>::new(
        5.0, 10.0,
    ));

    let x = m.lu().solve(&b).expect("nonsingular U");
    let _x0: qty!(m / s) = x.get(0, 0);
    let x_ref = m.solve(&b).expect("invertible");
    assert!((x.nalgebra() - x_ref.nalgebra()).norm() < 1e-12);

    // Full-pivot LU reuse agrees too, undoing the extra column permutation.
    let x_full = m.full_piv_lu().solve(&b).expect("nonsingular U");
    assert!((x_full.nalgebra() - x_ref.nalgebra()).norm() < 1e-12);
}

#[test]
fn uniform_add_scalar_and_lp_norm_keep_the_unit() {
    let v = speeds(); // 2×2 in m/s: entries 1, 2, 3, 4

    // add_scalar broadcasts a same-unit quantity, staying in m/s.
    let shifted = v.add_scalar(quantity!(10.0, m / s));
    let s00: qty!(m / s) = shifted.get(0, 0);
    assert_eq!(s00.unsafe_value, 11.0);

    // The L1 norm sums |entries|, in m/s: 1 + 2 + 3 + 4 = 10.
    let l1: qty!(m / s) = v.lp_norm(1);
    assert_eq!(l1.unsafe_value, 10.0);
}

#[test]
fn uniform_normalize_is_dimensionless() {
    // A [3, 4] m/s vector normalizes to the unit direction [0.6, 0.8], and the
    // result is dimensionless — U / U cancels. The `qty!(1)` annotation is the
    // real test that `DivUnit<U, U>` canonicalizes to `Dimensionless`.
    let v = UniformUnitMatrix::<unit!(m / s), SVector<f64, 2>>::from_nalgebra(
        SVector::<f64, 2>::new(3.0, 4.0),
    );
    let dir = v.normalize();
    let d0: qty!(1) = dir.get(0, 0);
    let d1: qty!(1) = dir.get(1, 0);
    assert!((d0.unsafe_value - 0.6).abs() < 1e-12);
    assert!((d1.unsafe_value - 0.8).abs() < 1e-12);

    // The magnitude that was divided out is still recoverable, in m/s.
    let mag: qty!(m / s) = v.norm();
    assert!((mag.unsafe_value - 5.0).abs() < 1e-12);
}

#[test]
fn uniform_cross_multiplies_the_entry_units() {
    // x̂ (m) × ŷ (s) = ẑ, in m·s. This is genuinely bilinear, so the units
    // multiply — cross is the vector-geometry sibling of `dot`.
    type V3 = SVector<f64, 3>;
    let x = UniformUnitMatrix::<unit!(m), V3>::from_nalgebra(V3::new(1.0, 0.0, 0.0));
    let y = UniformUnitMatrix::<unit!(s), V3>::from_nalgebra(V3::new(0.0, 1.0, 0.0));
    let z = x.cross(&y);
    let z0: qty!(m * s) = z.get(0, 0);
    let z1: qty!(m * s) = z.get(1, 0);
    let z2: qty!(m * s) = z.get(2, 0);
    assert_eq!(z0.unsafe_value, 0.0);
    assert_eq!(z1.unsafe_value, 0.0);
    assert_eq!(z2.unsafe_value, 1.0);
}

#[test]
fn uniform_lerp_keeps_the_unit() {
    // Blend two m/s vectors at t = 0.25: (1 − t)·a + t·b, still m/s.
    type V2v = SVector<f64, 2>;
    let a = UniformUnitMatrix::<unit!(m / s), V2v>::from_nalgebra(V2v::new(0.0, 4.0));
    let b = UniformUnitMatrix::<unit!(m / s), V2v>::from_nalgebra(V2v::new(8.0, 0.0));
    let mid = a.lerp(&b, 0.25);
    let m0: qty!(m / s) = mid.get(0, 0);
    let m1: qty!(m / s) = mid.get(1, 0);
    assert!((m0.unsafe_value - 2.0).abs() < 1e-12);
    assert!((m1.unsafe_value - 3.0).abs() < 1e-12);
}

#[test]
fn uniform_distance_keeps_the_unit() {
    // ‖a − b‖ between two m vectors: [3, 0] and [0, 4] are 5 m apart.
    type V2v = SVector<f64, 2>;
    let a = UniformUnitMatrix::<unit!(m), V2v>::from_nalgebra(V2v::new(3.0, 0.0));
    let b = UniformUnitMatrix::<unit!(m), V2v>::from_nalgebra(V2v::new(0.0, -4.0));
    let d: qty!(m) = a.metric_distance(&b);
    assert!((d.unsafe_value - 5.0).abs() < 1e-12);
    // `distance` is the alias.
    let d2: qty!(m) = a.distance(&b);
    assert_eq!(d.unsafe_value, d2.unsafe_value);
}

#[test]
fn uniform_angle_lands_in_radians() {
    // The angle between two orthogonal m vectors is π/2 rad, dimensionless of
    // dimension Angle — `angle` normalizes both, so the m units cancel.
    type V2v = SVector<f64, 2>;
    let x = UniformUnitMatrix::<unit!(m), V2v>::from_nalgebra(V2v::new(1.0, 0.0));
    let y = UniformUnitMatrix::<unit!(m), V2v>::from_nalgebra(V2v::new(0.0, 2.0));
    let theta: qty!(rad) = x.angle(&y);
    assert!((theta.unsafe_value - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
}

#[test]
fn uniform_per_axis_reductions_stay_in_the_unit() {
    // speeds() = [[1, 2], [3, 4]] m/s (row-major).
    let m = speeds();

    // row_sum collapses rows -> 1x2 row vector [1+3, 2+4] = [4, 6], in m/s.
    let rs = m.row_sum();
    let rs0: qty!(m / s) = rs.get(0, 0);
    let rs1: qty!(m / s) = rs.get(0, 1);
    assert_eq!(rs0.unsafe_value, 4.0);
    assert_eq!(rs1.unsafe_value, 6.0);

    // column_sum collapses columns -> 2x1 column vector [1+2, 3+4] = [3, 7].
    let cs = m.column_sum();
    let cs0: qty!(m / s) = cs.get(0, 0);
    let cs1: qty!(m / s) = cs.get(1, 0);
    assert_eq!(cs0.unsafe_value, 3.0);
    assert_eq!(cs1.unsafe_value, 7.0);

    // means: row_mean = [2, 3], column_mean = [1.5, 3.5].
    let rm0: qty!(m / s) = m.row_mean().get(0, 0);
    let cm1: qty!(m / s) = m.column_mean().get(1, 0);
    assert_eq!(rm0.unsafe_value, 2.0);
    assert_eq!(cm1.unsafe_value, 3.5);

    // variance of {1,2,3,4} (population) = 1.25, in m²/s².
    let v: qty!(m ^ 2 / s ^ 2) = m.variance();
    assert!((v.unsafe_value - 1.25).abs() < 1e-12);

    // per-column variance {1,3},{2,4} = [1, 1]; per-row variance = [0.25, 0.25].
    let rv0: qty!(m ^ 2 / s ^ 2) = m.row_variance().get(0, 0);
    let cv0: qty!(m ^ 2 / s ^ 2) = m.column_variance().get(0, 0);
    assert!((rv0.unsafe_value - 1.0).abs() < 1e-12);
    assert!((cv0.unsafe_value - 0.25).abs() < 1e-12);
}

#[test]
fn uniform_zip_map_picks_its_own_output_unit() {
    // Combine two m/s matrices entry-wise into a product in m²/s² — the output
    // unit is whatever the closure returns, unlike component_mul which fixes it.
    let a = speeds();
    let b = speeds();
    let prod = a.zip_map(&b, |x: qty!(m / s), y: qty!(m / s)| {
        quantity!(x.unsafe_value * y.unsafe_value, m ^ 2 / s ^ 2)
    });
    let p00: qty!(m ^ 2 / s ^ 2) = prod.get(0, 0);
    let p11: qty!(m ^ 2 / s ^ 2) = prod.get(1, 1);
    assert_eq!(p00.unsafe_value, 1.0);
    assert_eq!(p11.unsafe_value, 16.0);
}

#[test]
fn uniform_try_normalize_returns_none_below_min_norm() {
    type V2v = SVector<f64, 2>;
    let v = UniformUnitMatrix::<unit!(m / s), V2v>::from_nalgebra(V2v::new(3.0, 4.0));
    // Norm is 5; a min of 100 rejects it.
    assert!(v.try_normalize(100.0).is_none());
    // A min of 0 accepts and yields the dimensionless direction [0.6, 0.8].
    let dir = v.try_normalize(0.0).expect("nonzero");
    let d0: qty!(1) = dir.get(0, 0);
    assert!((d0.unsafe_value - 0.6).abs() < 1e-12);
}

#[test]
fn uniform_bulk_constructors_read_raw_scalars_as_the_unit() {
    // from_row_slice is row-major: [[1, 2], [3, 4]].
    let r = UniformUnitMatrix::<unit!(m), M2>::from_row_slice(&[1.0, 2.0, 3.0, 4.0]);
    let r01: qty!(m) = r.get(0, 1);
    assert_eq!(r01.unsafe_value, 2.0);

    // from_column_slice / from_vec / from_iterator are column-major: [[1, 3], [2, 4]].
    let c = UniformUnitMatrix::<unit!(m), M2>::from_column_slice(&[1.0, 2.0, 3.0, 4.0]);
    let c01: qty!(m) = c.get(0, 1);
    assert_eq!(c01.unsafe_value, 3.0);
    let vv = UniformUnitMatrix::<unit!(m), M2>::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let v10: qty!(m) = vv.get(1, 0);
    assert_eq!(v10.unsafe_value, 2.0);
    let it = UniformUnitMatrix::<unit!(m), M2>::from_iterator([1.0, 2.0, 3.0, 4.0]);
    assert_eq!(it.get(1, 0).unsafe_value, 2.0);

    // from_columns: columns [1, 2] and [3, 4] -> [[1, 3], [2, 4]].
    type Col = SMatrix<f64, 2, 1>;
    let col0 = UniformUnitMatrix::<unit!(m), Col>::from_nalgebra(Col::new(1.0, 2.0));
    let col1 = UniformUnitMatrix::<unit!(m), Col>::from_nalgebra(Col::new(3.0, 4.0));
    let byc = UniformUnitMatrix::<unit!(m), M2>::from_columns(&[col0, col1]);
    let bc01: qty!(m) = byc.get(0, 1);
    assert_eq!(bc01.unsafe_value, 3.0);

    // from_rows: rows [1, 2] and [3, 4] -> [[1, 2], [3, 4]].
    type Rw = SMatrix<f64, 1, 2>;
    let row0 = UniformUnitMatrix::<unit!(m), Rw>::from_nalgebra(Rw::new(1.0, 2.0));
    let row1 = UniformUnitMatrix::<unit!(m), Rw>::from_nalgebra(Rw::new(3.0, 4.0));
    let byr = UniformUnitMatrix::<unit!(m), M2>::from_rows(&[row0, row1]);
    let br10: qty!(m) = byr.get(1, 0);
    assert_eq!(br10.unsafe_value, 3.0);
}

#[test]
fn uniform_swap_and_triangle_preserve_the_unit() {
    // Swaps mutate in place and keep the unit (sound only because it is uniform).
    let mut m = speeds();
    m.swap_rows(0, 1); // [[3, 4], [1, 2]]
    let s00: qty!(m / s) = m.get(0, 0);
    assert_eq!(s00.unsafe_value, 3.0);
    m.swap_columns(0, 1); // [[4, 3], [2, 1]]
    let s00b: qty!(m / s) = m.get(0, 0);
    assert_eq!(s00b.unsafe_value, 4.0);

    // Triangles zero the opposite corner but keep the unit.
    let upper = speeds().upper_triangle(); // [[1, 2], [0, 4]]
    let u10: qty!(m / s) = upper.get(1, 0);
    assert_eq!(u10.unsafe_value, 0.0);
    let lower = speeds().lower_triangle(); // [[1, 0], [3, 4]]
    let l01: qty!(m / s) = lower.get(0, 1);
    assert_eq!(l01.unsafe_value, 0.0);
}

#[test]
fn uniform_determinant_raises_the_unit_to_the_dimension() {
    // speeds() = [[1, 2], [3, 4]] m/s. A 2×2 determinant multiplies one entry
    // per row/column — two factors of m/s — so it lands in (m/s)² = m²/s².
    // Value: 1·4 − 2·3 = −2.
    let d: qty!(m ^ 2 / s ^ 2) = speeds().determinant();
    assert!((d.unsafe_value + 2.0).abs() < 1e-12);
}

#[test]
fn uniform_product_raises_the_unit_to_the_entry_count() {
    // All four entries of speeds() multiply: 1·2·3·4 = 24, in (m/s)⁴ = m⁴/s⁴.
    let p: qty!(m ^ 4 / s ^ 4) = speeds().product();
    assert!((p.unsafe_value - 24.0).abs() < 1e-12);
}

#[test]
fn uniform_powi_raises_the_unit_to_the_const_exponent() {
    // M = [[1, 2], [3, 4]] m/s. M² = [[7, 10], [15, 22]], in (m/s)² = m²/s².
    let m = speeds();
    let sq = m.powi::<2>();
    let e00: qty!(m ^ 2 / s ^ 2) = sq.get(0, 0);
    let e11: qty!(m ^ 2 / s ^ 2) = sq.get(1, 1);
    assert!((e00.unsafe_value - 7.0).abs() < 1e-12);
    assert!((e11.unsafe_value - 22.0).abs() < 1e-12);

    // M⁰ is the dimensionless identity.
    let id = m.powi::<0>();
    let i00: qty!(1) = id.get(0, 0);
    let i01: qty!(1) = id.get(0, 1);
    assert_eq!(i00.unsafe_value, 1.0);
    assert_eq!(i01.unsafe_value, 0.0);

    // M³ lands in (m/s)³ = m³/s³.
    let cube = m.powi::<3>();
    let _c00: qty!(m ^ 3 / s ^ 3) = cube.get(0, 0);
}

#[test]
fn uniform_slerp_is_a_dimensionless_direction() {
    // Slerp normalizes its inputs, so the result is a dimensionless unit vector.
    // Halfway between x̂ and ŷ (both m) is the 45° direction of length 1.
    type V2v = SVector<f64, 2>;
    let x = UniformUnitMatrix::<unit!(m), V2v>::from_nalgebra(V2v::new(1.0, 0.0));
    let y = UniformUnitMatrix::<unit!(m), V2v>::from_nalgebra(V2v::new(0.0, 1.0));
    let half = x.slerp(&y, 0.5);
    let h0: qty!(1) = half.get(0, 0);
    let h1: qty!(1) = half.get(1, 0);
    let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
    assert!((h0.unsafe_value - inv_sqrt2).abs() < 1e-12);
    assert!((h1.unsafe_value - inv_sqrt2).abs() < 1e-12);
}

#[test]
fn uniform_rectangular_qr_svd_and_bidiagonalize_reconstruct() {
    // A tall 3×2 uniform matrix, every entry m/s. All three factor-exposing
    // decompositions use the thin pivot of length min(3, 2) = 2 and rebuild the
    // exact 3×2 m/s type.
    type M32 = SMatrix<f64, 3, 2>;
    type Tall = UniformUnitMatrix<unit!(m / s), M32>;
    let a = Tall::from_nalgebra(M32::new(1.0, 2.0, 3.0, 4.0, 5.0, 7.0));

    // QR: Q dimensionless 3×2, R in m/s 2×2.
    let qr = a.qr();
    assert_eq!(qr.q.shape(), (3, 2));
    assert_eq!(qr.r.shape(), (2, 2));
    let _q00: qty!(1) = qr.q.get(0, 0);
    let _r00: qty!(m / s) = qr.r.get(0, 0);
    let qr_recon: Tall = qr.recompose();

    // SVD: U dimensionless 3×2, 2 singular values in m/s, Vᵀ dimensionless 2×2.
    let svd = a.svd();
    assert_eq!(svd.singular_values.nalgebra().len(), 2);
    let _sv: qty!(m / s) = svd.singular_values.get(0, 0);
    let svd_recon: Tall = svd.recompose();

    // Bidiagonalize: U 3×2, B (m/s) 2×2, Vᵀ 2×2.
    let bidiag = a.bidiagonalize();
    assert_eq!(bidiag.d.shape(), (2, 2));
    let _b00: qty!(m / s) = bidiag.d.get(0, 0);
    let bidiag_recon: Tall = bidiag.recompose();

    for i in 0..3 {
        for j in 0..2 {
            let target = a.nalgebra()[(i, j)];
            assert!((qr_recon.nalgebra()[(i, j)] - target).abs() < 1e-10);
            assert!((svd_recon.nalgebra()[(i, j)] - target).abs() < 1e-10);
            assert!((bidiag_recon.nalgebra()[(i, j)] - target).abs() < 1e-10);
        }
    }
}

#[test]
fn uniform_held_qr_least_squares_solve_and_left_inverse_on_a_tall_matrix() {
    // Tall 3×2 uniform M in m/s, full column rank, rows [1,0] [0,1] [1,1]. A
    // consistent RHS makes the least-squares fit exact: b in m ⇒ x in
    // m / (m/s) = s. x_true = [1 s, 2 s] ⇒ b = [1, 2, 3] m.
    type M32 = SMatrix<f64, 3, 2>;
    type Tall = UniformUnitMatrix<unit!(m / s), M32>;
    let m = Tall::from_nalgebra(M32::new(1.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    let qr = m.qr();

    type B = SMatrix<f64, 3, 1>;
    let b = UniformUnitMatrix::<unit!(m), B>::from_nalgebra(B::new(1.0, 2.0, 3.0));
    let x = qr.solve(&b).expect("full column rank");
    let x0: qty!(s) = x.get(0, 0);
    let x1: qty!(s) = x.get(1, 0);
    assert!((x0.unsafe_value - 1.0).abs() < 1e-12);
    assert!((x1.unsafe_value - 2.0).abs() < 1e-12);

    // Held left inverse M⁺ in s/m = 1/(m/s), 2×3, with M⁺·M = I₂.
    let pinv = qr.try_inverse().expect("full column rank");
    assert_eq!(pinv.shape(), (2, 3));
    let _p00: qty!(s / m) = pinv.get(0, 0);
    let left = pinv.nalgebra() * m.nalgebra();
    assert!(
        (left - SMatrix::<f64, 2, 2>::identity()).norm() < 1e-12,
        "M⁺M ≠ I"
    );
}
