#![cfg(feature = "nalgebra")]
//! Structural extraction: `column`, `row`, `diagonal`, and `block` slice a
//! piece out of the matrix while transforming the dimension metadata exactly as
//! the entry-unit rule `RowDims[i] / ColDims[j]` demands. `set_block` is the
//! mutating inverse of `block`, assembling a matrix from typed pieces, and
//! `unblock_matrix!` is the reading inverse of `block_matrix!`, and its leading
//! `views` keyword is the zero-copy variant (each block borrows the parent
//! instead of copying).

use whippyalgebra::dims;
use whippyalgebra::nalgebra::{MixedUnitMatrix, SMatrix, block_matrix, unblock_matrix, zeros};
use whippyunits::qty;

type M2 = SMatrix<f64, 2, 2>;

// Rows carry [m, s], columns carry [A, K], so entry (i, j) has unit
// RowDims[i] / ColDims[j]:  (0,0)=m/A  (0,1)=m/K  (1,0)=s/A  (1,1)=s/K.
type RowDims = dims![m, s];
type ColDims = dims![A, K];
type Mat = MixedUnitMatrix<RowDims, ColDims, M2>;

fn sample() -> Mat {
    // Row-major constructor: M(0,0)=1, M(0,1)=2, M(1,0)=3, M(1,1)=4.
    Mat::new(M2::new(1.0, 2.0, 3.0, 4.0))
}

#[test]
fn column_keeps_rows_and_fixes_column_unit() {
    // Column 0 keeps RowDims = [m, s] and pins the single column unit to
    // ColDims[0] = A, so entry i has unit RowDims[i] / A.
    let c = sample().column::<0>();
    let top: qty!(m / A) = c.get::<0, 0>();
    let bot: qty!(s / A) = c.get::<1, 0>();
    assert_eq!(top.unsafe_value, 1.0);
    assert_eq!(bot.unsafe_value, 3.0);
}

#[test]
fn row_keeps_columns_and_fixes_row_unit() {
    // Row 0 keeps ColDims = [A, K] and pins the single row unit to
    // RowDims[0] = m, so entry j has unit m / ColDims[j].
    let r = sample().row::<0>();
    let left: qty!(m / A) = r.get::<0, 0>();
    let right: qty!(m / K) = r.get::<0, 1>();
    assert_eq!(left.unsafe_value, 1.0);
    assert_eq!(right.unsafe_value, 2.0);
}

#[test]
fn diagonal_is_elementwise_quotient() {
    // The i-th diagonal entry has unit RowDims[i] / ColDims[i]:
    // (0) = m/A holding M(0,0)=1, (1) = s/K holding M(1,1)=4.
    let d = sample().diagonal();
    let d0: qty!(m / A) = d.get::<0, 0>();
    let d1: qty!(s / K) = d.get::<1, 0>();
    assert_eq!(d0.unsafe_value, 1.0);
    assert_eq!(d1.unsafe_value, 4.0);
}

#[test]
fn block_slices_dimension_sublists() {
    // A 3x3 matrix over rows [m, s, A] and cols [A, rot, K]. The 2x2 block at
    // (0, 1) keeps the row sublist [m, s] and the column sublist [rot, K], so
    // entry (i, j) keeps its parent unit RowDims[0+i] / ColDims[1+j].
    type M3 = SMatrix<f64, 3, 3>;
    type Mat3 = MixedUnitMatrix<dims![m, s, A], dims![A, rot, K], M3>;

    // Row-major 1..=9: row 0 = [1,2,3], row 1 = [4,5,6], row 2 = [7,8,9].
    let m = Mat3::new(M3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));

    // Block at (0,1) of size 2x2 = rows {0,1} × cols {1,2} = [[2,3],[5,6]].
    let blk = m.block::<0, 1, 2, 2>();
    assert_eq!(blk.shape(), (2, 2));

    let e00: qty!(m / rot) = blk.get::<0, 0>();
    assert_eq!(e00.unsafe_value, 2.0);
    let e01: qty!(m / K) = blk.get::<0, 1>();
    assert_eq!(e01.unsafe_value, 3.0);
    let e10: qty!(s / rot) = blk.get::<1, 0>();
    assert_eq!(e10.unsafe_value, 5.0);
    let e11: qty!(s / K) = blk.get::<1, 1>();
    assert_eq!(e11.unsafe_value, 6.0);
}

#[test]
fn set_block_writes_typed_sublists() {
    // Same 3x3 grid over rows [m, s, A], cols [A, rot, K]. Writing a 2x2 block
    // at (0, 1) requires the block to carry exactly the destination sublists —
    // rows [m, s] and cols [rot, K] — so the units are checked on assembly.
    type M3 = SMatrix<f64, 3, 3>;
    type Mat3 = MixedUnitMatrix<dims![m, s, A], dims![A, rot, K], M3>;
    type Blk = MixedUnitMatrix<dims![m, s], dims![rot, K], M2>;

    let blk = Blk::new(M2::new(2.0, 3.0, 5.0, 6.0));
    let mut m = Mat3::new(M3::zeros());
    m.set_block::<0, 1, 2, 2, _>(&blk);

    // The block landed at rows {0,1} × cols {1,2}; everything else stays zero.
    let e01: qty!(m / rot) = m.get::<0, 1>();
    assert_eq!(e01.unsafe_value, 2.0);
    let e02: qty!(m / K) = m.get::<0, 2>();
    assert_eq!(e02.unsafe_value, 3.0);
    let e11: qty!(s / rot) = m.get::<1, 1>();
    assert_eq!(e11.unsafe_value, 5.0);
    let e12: qty!(s / K) = m.get::<1, 2>();
    assert_eq!(e12.unsafe_value, 6.0);
    let untouched: qty!(m / A) = m.get::<0, 0>();
    assert_eq!(untouched.unsafe_value, 0.0);

    // Round-trip: extracting the same block returns what we wrote.
    let back = m.block::<0, 1, 2, 2>();
    assert_eq!(back.get::<1, 1>().unsafe_value, 6.0);
}

#[test]
fn block_matrix_assembles_from_typed_blocks() {
    // A 3x3 partitioned matrix over rows [m, s, A] and cols [m, s, A], built as a
    // block literal `[[a, b], [0, 0]]`: `a` is ⟨[m,s],[m,s]⟩ (2x2), `b` is
    // ⟨[m,s],[A]⟩ (2x1), and the bottom row is zeros stating their [A]-row space.
    let a = MixedUnitMatrix::<dims![m, s], dims![m, s], M2>::new(M2::new(1.0, 2.0, 3.0, 4.0));
    let b = MixedUnitMatrix::<dims![m, s], dims![A], SMatrix<f64, 2, 1>>::new(
        SMatrix::<f64, 2, 1>::new(5.0, 6.0),
    );

    // hcat unifies the shared row space, vcat the shared column space; a block
    // whose spaces didn't line up here would fail to compile.
    let m: MixedUnitMatrix<dims![m, s, A], dims![m, s, A], _> = block_matrix![
        [a, b],
        [zeros![dims![A], dims![m, s]], zeros![dims![A], dims![A]]],
    ];

    assert_eq!(m.shape(), (3, 3));

    // Top-left `a` and top-right `b` land where written, keeping entry units
    // RowDims[i] / ColDims[j]; the bottom row stays zero.
    let a01: qty!(m / s) = m.get::<0, 1>();
    assert_eq!(a01.unsafe_value, 2.0);
    let b02: qty!(m / A) = m.get::<0, 2>();
    assert_eq!(b02.unsafe_value, 5.0);
    let b12: qty!(s / A) = m.get::<1, 2>();
    assert_eq!(b12.unsafe_value, 6.0);
    let zero20: qty!(A / m) = m.get::<2, 0>();
    assert_eq!(zero20.unsafe_value, 0.0);
}

#[test]
fn unblock_matrix_reads_back_typed_blocks() {
    // The same 3x3 partition `[[a, b], [0, 0]]` over rows/cols [m, s, A], now
    // read back apart with `unblock_matrix!`. The grid layout mirrors the
    // assembly: named cells bind the blocks we want and each `_(h, w)` standin
    // holds an unused slot, stating its own size (concrete literals here, so the
    // offsets are ordinary const expressions) so every offset is recovered.
    let a = MixedUnitMatrix::<dims![m, s], dims![m, s], M2>::new(M2::new(1.0, 2.0, 3.0, 4.0));
    let b = MixedUnitMatrix::<dims![m, s], dims![A], SMatrix<f64, 2, 1>>::new(
        SMatrix::<f64, 2, 1>::new(5.0, 6.0),
    );
    let m: MixedUnitMatrix<dims![m, s, A], dims![m, s, A], _> = block_matrix![
        [a, b],
        [zeros![dims![A], dims![m, s]], zeros![dims![A], dims![A]]],
    ];

    // Rows partition as [2, 1] over [m,s | A]; columns as [2, 1] over [m,s | A].
    unblock_matrix!(m => [
        [top_left(2, 2), top_right(2, 1)],
        [_(1, 2),        _(1, 1)        ],
    ]);

    // `top_left` is ⟨[m,s],[m,s]⟩ and equals `a`; `top_right` is ⟨[m,s],[A]⟩ and
    // equals `b` — unit-checked by these annotated reads.
    let tl00: qty!(m / m) = top_left.get::<0, 0>();
    assert_eq!(tl00.unsafe_value, 1.0);
    let tl11: qty!(s / s) = top_left.get::<1, 1>();
    assert_eq!(tl11.unsafe_value, 4.0);
    let tr00: qty!(m / A) = top_right.get::<0, 0>();
    assert_eq!(tr00.unsafe_value, 5.0);
    let tr10: qty!(s / A) = top_right.get::<1, 0>();
    assert_eq!(tr10.unsafe_value, 6.0);
}

#[test]
fn unblock_matrix_views_reads_borrowed_blocks() {
    // Same partition, read back as zero-copy views. Each named cell borrows `m`
    // via `block_view` instead of copying, so the blocks carry identical units
    // and values but share `m`'s storage.
    let a = MixedUnitMatrix::<dims![m, s], dims![m, s], M2>::new(M2::new(1.0, 2.0, 3.0, 4.0));
    let b = MixedUnitMatrix::<dims![m, s], dims![A], SMatrix<f64, 2, 1>>::new(
        SMatrix::<f64, 2, 1>::new(5.0, 6.0),
    );
    let m: MixedUnitMatrix<dims![m, s, A], dims![m, s, A], _> = block_matrix![
        [a, b],
        [zeros![dims![A], dims![m, s]], zeros![dims![A], dims![A]]],
    ];

    unblock_matrix!(views; m => [
        [top_left(2, 2), top_right(2, 1)],
        [_(1, 2),        _(1, 1)        ],
    ]);

    // `top_left` is ⟨[m,s],[m,s]⟩, `top_right` is ⟨[m,s],[A]⟩ — same units as the
    // owned read, but views into `m`.
    let tl00: qty!(m / m) = top_left.get::<0, 0>();
    assert_eq!(tl00.unsafe_value, 1.0);
    let tr10: qty!(s / A) = top_right.get::<1, 0>();
    assert_eq!(tr10.unsafe_value, 6.0);

    // Because the borrows are *shared*, both views live at once and `m` stays
    // readable alongside them (a view over borrowed storage is still a matrix).
    let m00: qty!(m / m) = m.get::<0, 0>();
    assert_eq!(m00.unsafe_value, 1.0);
    let _keep_alive = (top_left, top_right);
}
