#![cfg(feature = "nalgebra")]
//! Standalone exercise of the `#[generic_block]` attribute, deliberately free of
//! any inherited context (no `SquareDim`, no LQR-style bounds): a function that
//! is generic over both the block *shapes* (`const N`, `const M`) and the block
//! *dimension lists* (`RowA/RowB/ColA/ColB`) assembles a 2×2 partition with
//! `block_matrix!` and then reads it back apart with `unblock_matrix!`.
//!
//! The point is to check what `#[generic_block]` must supply on its own for the
//! full assemble→slice round-trip to compile in generic code — i.e. without the
//! attribute the caller would have to transcribe the nalgebra allocator/DimAdd
//! obligations (assembly) *and* the `ShapeIndex` shape obligations (slicing) by
//! hand. Only the genuinely problem-semantic dim-list facts (how the lists
//! concatenate and partition) are stated in the `where` clause here.

use whippyalgebra::nalgebra::{
    MixedUnitMatrix, SMatrix, block_matrix, generic_block, unblock_matrix, zeros,
};
use whippyalgebra::dims;

/// Assemble the four blocks into one matrix and immediately slice it back into
/// four blocks, generic over the shapes and the dimension lists. Returns the
/// sum of the reblocked pieces' Frobenius norms (forces every block's type and
/// storage to resolve).
///
/// Naming the sublists (`row_dims`/`col_dims`) lets `#[generic_block]`
/// synthesize the entire partition `where` clause — the `Concat` that assembly
/// needs and the `Take`/`Drop` that slicing needs — so the function body carries
/// no hand-written dim-list bounds at all.
#[generic_block(
    rows(nalgebra::Const<N>, RowA; nalgebra::Const<M>, RowB),
    cols(nalgebra::Const<N>, ColA; nalgebra::Const<M>, ColB),
)]
fn reblock<RowA, RowB, ColA, ColB, const N: usize, const M: usize>(
    tl: MixedUnitMatrix<RowA, ColA, SMatrix<f64, N, N>>,
    tr: MixedUnitMatrix<RowA, ColB, SMatrix<f64, N, M>>,
    bl: MixedUnitMatrix<RowB, ColA, SMatrix<f64, M, N>>,
    br: MixedUnitMatrix<RowB, ColB, SMatrix<f64, M, M>>,
) -> f64 {
    let whole = block_matrix![[tl, tr], [bl, br]];
    unblock_matrix!(whole => [
        [a(N, N), b(N, M)],
        [c(M, N), d(M, M)],
    ]);
    a.nalgebra().norm() + b.nalgebra().norm() + c.nalgebra().norm() + d.nalgebra().norm()
}

#[test]
fn reblock_roundtrip_is_shape_and_unit_generic() {
    // Concrete instantiation: N = 2, M = 1.
    let tl = zeros![dims![m, s], dims![A, K]]; // RowA x ColA (2x2)
    let tr = zeros![dims![m, s], dims![mol]]; //  RowA x ColB (2x1)
    let bl = zeros![dims![kg], dims![A, K]]; //   RowB x ColA (1x2)
    let br = zeros![dims![kg], dims![mol]]; //    RowB x ColB (1x1)

    // All blocks start at zero, so the round-tripped norm is zero — the test is
    // really that this monomorphization type-checks at all.
    let total = reblock(tl, tr, bl, br);
    assert_eq!(total, 0.0);
}

/// Assembles the *same* four blocks into two different grids in one signature —
/// exercising the multi-grid `block(..)` form (and the bare-`N`/`M` size syntax).
/// Grid 1 is the natural `(N+M) × (N+M)` layout; grid 2 is the transposed
/// arrangement, `(M+N) × (M+N)`. Both grids' obligations land in one `where`
/// clause, deduplicated across grids (`Const<N>`/`Const<M>` etc. appear once).
#[generic_block(
    block(rows(N, RowA; M, RowB), cols(N, ColA; M, ColB)),
    block(rows(M, RowB; N, RowA), cols(M, ColB; N, ColA)),
)]
fn two_grids<RowA, RowB, ColA, ColB, const N: usize, const M: usize>(
    tl: MixedUnitMatrix<RowA, ColA, SMatrix<f64, N, N>>,
    tr: MixedUnitMatrix<RowA, ColB, SMatrix<f64, N, M>>,
    bl: MixedUnitMatrix<RowB, ColA, SMatrix<f64, M, N>>,
    br: MixedUnitMatrix<RowB, ColB, SMatrix<f64, M, M>>,
) -> f64 {
    let g1 = block_matrix![[tl, tr], [bl, br]]; // (N+M) x (N+M)
    let g2 = block_matrix![[br, bl], [tr, tl]]; // (M+N) x (M+N)
    g1.nalgebra().norm() + g2.nalgebra().norm()
}

#[test]
fn two_grids_in_one_signature_type_checks() {
    let tl = zeros![dims![m, s], dims![A, K]]; // RowA x ColA (2x2)
    let tr = zeros![dims![m, s], dims![mol]]; //  RowA x ColB (2x1)
    let bl = zeros![dims![kg], dims![A, K]]; //   RowB x ColA (1x2)
    let br = zeros![dims![kg], dims![mol]]; //    RowB x ColB (1x1)

    let total = two_grids(tl, tr, bl, br);
    assert_eq!(total, 0.0);
}

// The `decompose` opt-in is exercised by the surviving generic decompose demo,
// `#[generic_matrix(uniform, … decompose)]` on `solve_currents` (uniform
// `pseudo_inverse`) in the `least_squares` example. After the QR-family gating,
// every *mixed* reduction is either metric-supplied (`generalized_pseudo_inverse`,
// whose per-axis metric a dimension-list-generic block can't name for its `N + M`
// axis — it is therefore spelled at concrete shapes in the
// `generalized_least_squares` example) or an endomorphism spectrum (`eigenvalues`,
// whose result-type inference does not converge through a generic `block_matrix!`
// assembly), so no honest *mixed* decompose canary survives *here* — the flag
// itself is emitted by the shared `decompose_preds` path either way.
