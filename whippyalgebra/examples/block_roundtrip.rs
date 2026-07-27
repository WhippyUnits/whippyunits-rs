//! Demonstrates unblocking and re-blocking a MixedUnitMatrix to manipulate its internal
//! uniform blocks dynamically by-entry.
//!
//! Iteration over an entire MixedUnitMatrix is not possible, because it would be unit-unsafe.
//! Splitting into uniform blocks allows us to iterate over each block entrywise, since
//! each block has a fixed unit.
//!
//!
use whippyalgebra::dims;
use whippyalgebra::nalgebra::{
    MixedUnitMatrix, SMatrix, block_matrix, gauge, mixed_unit_matrix, unblock_matrix,
};
use whippyunits::quantity;

type M3 = SMatrix<f64, 3, 3>;
type Mixed3 = MixedUnitMatrix<dims![m, m, s], dims![s, s, K], M3>;

fn main() {
    // A mixed 3x3 over rows [m, m, s] and columns [s, s, K]: entry (i, j) has
    // unit RowDims[i] / ColDims[j]. Written row-major as inline quantities, each
    // carrying exactly its cell's unit (a wrong unit would be a compile error).
    let original = mixed_unit_matrix![dims![m, m, s], dims![s, s, K];
        [quantity!(1.0, m / s), quantity!(2.0, m / s), quantity!(3.0, m / K)],
        [quantity!(4.0, m / s), quantity!(5.0, m / s), quantity!(6.0, m / K)],
        [7.0, 8.0, quantity!(9.0, s / K)],
    ];
    println!("original: mixed 3x3 over rows [m, m, s], cols [s, s, K]\n");

    // The [2,1] x [2,1] partition both paths use (rows [m,m|s], cols [s,s|K]):
    //   tl ⟨[m,m],[s,s]⟩ -> uniform m/s   (both sublists constant)  <- edited
    //   tr ⟨[m,m],[K]⟩   -> uniform m/K
    //   bl ⟨[s],[s,s]⟩   -> uniform s/s   (dimensionless)
    //   br ⟨[s],[K]⟩     -> uniform s/K
    let rebuilt_a = path_a_reduce_and_regauge(&original);
    let rebuilt_b = path_b_view_and_keep_gauge(&original);

    // Same manipulation on both sides: tl is the uniform m/s block covering the
    // top-left 2x2 (global rows 0..2, cols 0..2); doubling it turns 1,2,4,5 into
    // 2,4,8,10 and leaves everything else untouched.
    assert!(
        same(&rebuilt_a, &rebuilt_b),
        "the two paths must agree entry for entry"
    );
    for i in 0..3 {
        for j in 0..3 {
            let factor = if i < 2 && j < 2 { 2.0 } else { 1.0 };
            let expected = original.nalgebra()[(i, j)] * factor;
            assert_eq!(
                rebuilt_a.nalgebra()[(i, j)],
                expected,
                "mismatch at ({i}, {j})"
            );
        }
    }
    println!("\nboth paths doubled the uniform block identically; results agree exactly");
}

/// Path A: `reduce_uniform` erases the gauge, so after editing the uniform block
/// we must re-enter the gauge with `gauge!` to reblock.
fn path_a_reduce_and_regauge(original: &Mixed3) -> Mixed3 {
    println!("Path A — reduce-unblock, reblock with a gauge spec:");

    // Opt in to uniform extraction: uniform blocks are reduced to a gauge-less UniformUnitMatrix.
    unblock_matrix!(reduce_uniform; original => [
        [tl(2, 2), tr(2, 1)],
        [bl(1, 2), br(1, 1)],
    ]);

    // Manipulate: iterate the uniform block entrywise and double each entry in
    // place. `iter_mut` is sound here precisely because every entry shares m/s.
    let mut tl = tl;
    for q in tl.iter_mut() {
        *q = quantity!(q.unsafe_value * 2.0, m / s);
    }
    println!("  tl: owned UniformUnitMatrix<m/s>; doubled every entry via iter_mut");

    // Reblock: every block was reduced to a gauge-less UniformUnitMatrix, so
    // re-enter each block's gauge by hand (a wrong tag would not compile).
    let rebuilt = block_matrix![
        [gauge!(tl => m, s), gauge!(tr => m, K)],
        [gauge!(bl => s, s), gauge!(br => s, K)],
    ];
    println!("  reblocked with explicit gauge! on every block");
    rebuilt
}

/// Path B: default `mixed` mode keeps every block's gauge; we mutate the uniform
/// block *through a borrowed uniform view*, then reassemble with no gauge spec.
fn path_b_view_and_keep_gauge(original: &Mixed3) -> Mixed3 {
    println!("\nPath B — mixed-unblock, mutate through a borrowed uniform view:");

    // Default mode: every block stays a MixedUnitMatrix carrying its exact
    // RowDims/ColDims — the gauge. (Owned copies here, so we can mutate one.)
    unblock_matrix!(original => [
        [tl(2, 2), tr(2, 1)],
        [bl(1, 2), br(1, 1)],
    ]);

    // Manipulate: `tl` is ⟨[m,m],[s,s]⟩ and happens to be uniform, so borrow a
    // uniform *view* onto it (gauge erased only in the view — `tl` keeps its
    // mixed type) and do the identical iter_mut doubling through the view.
    let mut tl = tl;
    {
        let mut view = tl.as_uniform_mut();
        for q in view.iter_mut() {
            *q = quantity!(q.unsafe_value * 2.0, m / s);
        }
    } // view dropped; `tl` is a MixedUnitMatrix<[m,m],[s,s]> again, gauge intact
    println!("  tl: borrowed UniformUnitMatrix<m/s> view; doubled every entry via iter_mut");

    // Reblock straight from the mixed blocks — the edited `tl` included — with no
    // `gauge!` anywhere, because each block still carries the spaces it belongs to.
    let rebuilt = block_matrix![[tl, tr], [bl, br],];
    println!("  reblocked with no gauge annotation");
    rebuilt
}

/// Entry-wise value equality between two reconstructions (units are guaranteed by
/// the shared `Whole` type, which fixes the dimension lists).
fn same(a: &Mixed3, b: &Mixed3) -> bool {
    (0..3).all(|i| (0..3).all(|j| a.nalgebra()[(i, j)] == b.nalgebra()[(i, j)]))
}
