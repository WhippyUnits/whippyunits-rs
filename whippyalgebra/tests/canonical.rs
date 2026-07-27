#![cfg(feature = "nalgebra")]
//! Gauge canonicalization: the `(RowDims, ColDims)` factorization of a matrix's
//! entry-unit grid has one global gauge freedom (scale every row *and* column
//! unit by a common `g`), so distinct types can describe the same physical
//! matrix. `canonical_at_row`/`canonical_at_col` pin that gauge to a normal
//! form (dimensionless anchor), and `with_gauge_at_row`/`with_gauge_at_column`
//! pin it to a chosen unit — either way letting homotypes add.

use whippyalgebra::nalgebra::{MixedUnitMatrix, SMatrix, UnitRowVector, UnitVector};
use whippyalgebra::{Dimensionless, dims};
use whippyunits::{qty, unit};

type M2 = SMatrix<f64, 2, 2>;
type V2 = SMatrix<f64, 2, 1>;
type R2 = SMatrix<f64, 1, 2>;

#[test]
fn canonical_lets_homotypes_add() {
    // Two all-dimensionless endomorphisms with different labels: [m,m]→[m,m]
    // and [s,s]→[s,s]. Both have entry unit `1` everywhere, but their types
    // differ, so they do not add directly. Anchoring row 0 divides each by its
    // own row-0 unit, collapsing both to the shared normal form [1,1]→[1,1].
    type OnMeters = MixedUnitMatrix<dims![m, m], dims![m, m], M2>;
    type OnSeconds = MixedUnitMatrix<dims![s, s], dims![s, s], M2>;

    let a = OnMeters::new(M2::new(1.0, 2.0, 3.0, 4.0));
    let b = OnSeconds::new(M2::new(10.0, 20.0, 30.0, 40.0));

    // The two regauged values are now the *same* type, so `+` type-checks even
    // though `a` and `b` did not.
    let sum = a.canonical_at_row::<0>() + b.canonical_at_row::<0>();

    // Values are untouched by regauging; only the type labels changed.
    let e: qty!(1) = sum.get::<0, 0>();
    assert_eq!(e.unsafe_value, 11.0);
    let e11: qty!(1) = sum.get::<1, 1>();
    assert_eq!(e11.unsafe_value, 44.0);
}

#[test]
fn canonical_preserves_entry_units_and_values() {
    // [m, m] → [s, s]: every entry is m/s. Anchoring row 0 divides both dim
    // vectors by m, giving rows [1, 1] and cols [s/m, s/m] — but the entry units
    // (row_i / col_j) are invariant, so (0,0) is still m/s.
    type Mat = MixedUnitMatrix<dims![m, m], dims![s, s], M2>;
    let m = Mat::new(M2::new(1.0, 2.0, 3.0, 4.0));

    let c = m.canonical_at_row::<0>();
    let e: qty!(m / s) = c.get::<0, 0>();
    assert_eq!(e.unsafe_value, 1.0);
    let e10: qty!(m / s) = c.get::<1, 0>();
    assert_eq!(e10.unsafe_value, 3.0);
}

#[test]
fn canonical_at_col_selects_the_anchor_column() {
    // Columns (the denominator) carry different units [s, m]; rows are [m, m].
    // Anchoring column 1 (unit m) makes that column dimensionless while leaving
    // every entry unit unchanged: (0,0) is row0/col0 = m/s regardless of gauge.
    type Mat = MixedUnitMatrix<dims![m, m], dims![s, m], M2>;
    let m = Mat::new(M2::new(1.0, 2.0, 3.0, 4.0));

    let c = m.canonical_at_col::<1>();
    let e00: qty!(m / s) = c.get::<0, 0>();
    assert_eq!(e00.unsafe_value, 1.0);
    // Entry (0,1) is row0/col1 = m/m = 1 (dimensionless) both before and after.
    let e01: qty!(1) = c.get::<0, 1>();
    assert_eq!(e01.unsafe_value, 2.0);
}

#[test]
fn canonical_at_row_pins_the_numerator() {
    // Rows (the numerator) carry different units [m, s]; columns are [s, s].
    // Anchoring row 0 (unit m) divides both dim vectors by m, sending
    // RowDims[0] → 1 while entry units stay put: (0,0) is row0/col0 = m/s.
    type Mat = MixedUnitMatrix<dims![m, s], dims![s, s], M2>;
    let m = Mat::new(M2::new(1.0, 2.0, 3.0, 4.0));

    let c = m.canonical_at_row::<0>();
    // Entry (0,0) = row0/col0 = m/s, unchanged by the regauge.
    let e00: qty!(m / s) = c.get::<0, 0>();
    assert_eq!(e00.unsafe_value, 1.0);
    // Entry (1,0) = row1/col0 = s/s = 1 (dimensionless), unchanged.
    let e10: qty!(1) = c.get::<1, 0>();
    assert_eq!(e10.unsafe_value, 3.0);
}

#[test]
fn vector_column_unit_splits_input_and_output() {
    // A gain vector is no longer forced to fold all its dimensionality into the
    // rows. Here the single column (input) carries `s` and the rows (output)
    // carry `m`, so each entry is m/s — the split, physically-natural form.
    type Gain = UnitVector<dims![m, m], unit!(s), V2>;
    let g = Gain::new(V2::new(2.0, 6.0));

    let e0: qty!(m / s) = g.get::<0, 0>();
    assert_eq!(e0.unsafe_value, 2.0);
    let e1: qty!(m / s) = g.get::<1, 0>();
    assert_eq!(e1.unsafe_value, 6.0);

    // Regauging onto the column recovers the all-in-the-rows representation
    // (column → dimensionless) without touching the entry units or values.
    let folded = g.canonical_at_col::<0>();
    let f0: qty!(m / s) = folded.get::<0, 0>();
    assert_eq!(f0.unsafe_value, 2.0);
}

#[test]
fn row_vector_splits_across_columns() {
    // The 1×n dual: a single row (output) unit m over column (input) units
    // [s, s], so each entry is m/s — a covector consuming per-second inputs.
    type Row = UnitRowVector<unit!(m), dims![s, s], R2>;
    let r = Row::new(R2::new(2.0, 6.0));

    let e0: qty!(m / s) = r.get::<0, 0>();
    assert_eq!(e0.unsafe_value, 2.0);
    let e1: qty!(m / s) = r.get::<0, 1>();
    assert_eq!(e1.unsafe_value, 6.0);
}

#[test]
fn default_vector_column_is_dimensionless() {
    // A dimensionless column unit gives the plain "index" vector whose entries
    // are exactly the row units.
    type Vel = UnitVector<dims![m, m], Dimensionless, V2>;
    let v = Vel::new(V2::new(1.0, 2.0));
    let e0: qty!(m) = v.get::<0, 0>();
    assert_eq!(e0.unsafe_value, 1.0);
}

#[test]
fn with_gauge_at_row_fixes_homotypes_to_a_chosen_unit() {
    // Same all-dimensionless homotypes, but pinned to the unit `km` at row 0
    // instead of dimensionless: dividing by (row0 / km) sends both to
    // [km,km]→[km,km], the shared normal form for that gauge choice.
    type OnMeters = MixedUnitMatrix<dims![m, m], dims![m, m], M2>;
    type OnSeconds = MixedUnitMatrix<dims![s, s], dims![s, s], M2>;

    let a = OnMeters::new(M2::new(1.0, 2.0, 3.0, 4.0));
    let b = OnSeconds::new(M2::new(10.0, 20.0, 30.0, 40.0));

    let sum = a.with_gauge_at_row::<unit!(km), 0>() + b.with_gauge_at_row::<unit!(km), 0>();

    // Entries remain dimensionless (km/km) and values are untouched.
    let e: qty!(1) = sum.get::<0, 0>();
    assert_eq!(e.unsafe_value, 11.0);
}

#[test]
fn with_gauge_at_column_fixes_homotypes_to_a_chosen_unit() {
    // The column dual: pin column 0 to `s`. OnMeters divides by (m / s) → its
    // lists become [s,s]; OnSeconds divides by (s / s = 1) and is already [s,s].
    // Both land on [s,s]→[s,s], so they add.
    type OnMeters = MixedUnitMatrix<dims![m, m], dims![m, m], M2>;
    type OnSeconds = MixedUnitMatrix<dims![s, s], dims![s, s], M2>;

    let a = OnMeters::new(M2::new(1.0, 2.0, 3.0, 4.0));
    let b = OnSeconds::new(M2::new(10.0, 20.0, 30.0, 40.0));

    let sum = a.with_gauge_at_column::<unit!(s), 0>() + b.with_gauge_at_column::<unit!(s), 0>();

    let e: qty!(1) = sum.get::<0, 0>();
    assert_eq!(e.unsafe_value, 11.0);
}

#[test]
fn row_and_column_anchors_agree_up_to_gauge() {
    // Two homotypes of an all-`m/s` matrix: [m,m]→[s,s] and, scaled by a common
    // gauge, [m·m, m·m]→[m·s, m·s] would be equivalent — but here we simply
    // show row- and column-anchoring both preserve the physical entries.
    type Mat = MixedUnitMatrix<dims![m, m], dims![s, s], M2>;
    let inner = M2::new(1.0, 2.0, 3.0, 4.0);

    let by_col = Mat::new(inner).canonical_at_col::<0>(); // divide by col0 = s
    let by_row = Mat::new(inner).canonical_at_row::<0>(); // divide by row0 = m

    // Different normal forms, but each leaves entry (0,0) = m/s intact.
    let ec: qty!(m / s) = by_col.get::<0, 0>();
    let er: qty!(m / s) = by_row.get::<0, 0>();
    assert_eq!(ec.unsafe_value, 1.0);
    assert_eq!(er.unsafe_value, 1.0);
}
