#![cfg(feature = "nalgebra")]
//! Exercises the uniform block/construction surface:
//!
//! - [`uniform_unit_matrix!`] — the single-unit twin of `mixed_unit_matrix!`,
//!   shape counted from the literal;
//! - [`block_matrix!`]`[uniform; …]` — assembling same-unit blocks into one
//!   uniform matrix (via `UniformUnitMatrix::hcat`/`vcat`);
//! - `#[generic_block(uniform, …)]` — the attribute writing that assembly's
//!   nalgebra-level `where` bounds in code generic over the block shapes, with
//!   no `ShapeIndex`/partition plumbing.

use whippyalgebra::nalgebra::{
    SMatrix, UniformUnitMatrix, block_matrix, generic_block, uniform_unit_matrix,
};
use whippyunits::{quantity, unit};

type Ohm = unit!(V / A);

#[test]
fn uniform_unit_matrix_macro_counts_shape() {
    // 2×3, every entry in Ω; shape (2 rows × 3 cols) is counted from the literal.
    let m = uniform_unit_matrix![V / A;
        [quantity!(1.0, V / A), quantity!(2.0, V / A), quantity!(3.0, V / A)],
        [quantity!(4.0, V / A), quantity!(5.0, V / A), quantity!(6.0, V / A)],
    ];
    let _: &SMatrix<f64, 2, 3> = m.nalgebra();
    assert_eq!(m.shape(), (2, 3));
    assert_eq!(m.nalgebra()[(1, 2)], 6.0);
}

#[test]
fn block_matrix_uniform_assembles_concrete() {
    // Four Ω blocks tiling a 3×3 (= (2+1)×(2+1)) uniform matrix.
    let tl = uniform_unit_matrix![V / A;
        [quantity!(1.0, V / A), quantity!(2.0, V / A)],
        [quantity!(3.0, V / A), quantity!(4.0, V / A)],
    ];
    let tr = uniform_unit_matrix![V / A; [quantity!(5.0, V / A)], [quantity!(6.0, V / A)]];
    let bl = uniform_unit_matrix![V / A; [quantity!(7.0, V / A), quantity!(8.0, V / A)]];
    let br = uniform_unit_matrix![V / A; [quantity!(9.0, V / A)]];

    let whole: UniformUnitMatrix<Ohm, SMatrix<f64, 3, 3>> =
        block_matrix![uniform; [tl, tr], [bl, br]];
    assert_eq!(whole.shape(), (3, 3));
    assert_eq!(whole.nalgebra()[(0, 2)], 5.0); // from `tr`
    assert_eq!(whole.nalgebra()[(2, 0)], 7.0); // from `bl`
    assert_eq!(whole.nalgebra()[(2, 2)], 9.0); // from `br`
}

/// Assemble a `(N+M) × (N+M)` uniform block matrix generically — the attribute
/// supplies every `DimAdd`/allocator (and the detected square-op bounds) for the
/// four `hcat`/`vcat`s, so the body needs no hand-written `where` clause.
#[generic_block(uniform, rows(N; M), cols(N; M))]
fn assemble_uniform<const N: usize, const M: usize>(
    tl: UniformUnitMatrix<Ohm, SMatrix<f64, N, N>>,
    tr: UniformUnitMatrix<Ohm, SMatrix<f64, N, M>>,
    bl: UniformUnitMatrix<Ohm, SMatrix<f64, M, N>>,
    br: UniformUnitMatrix<Ohm, SMatrix<f64, M, M>>,
) -> f64 {
    let whole = block_matrix![uniform; [tl, tr], [bl, br]];
    whole.nalgebra().sum()
}

#[test]
fn generic_block_uniform_assembles_generically() {
    let one = || {
        UniformUnitMatrix::<Ohm, SMatrix<f64, 2, 2>>::from_nalgebra(SMatrix::<f64, 2, 2>::repeat(
            1.0,
        ))
    };
    let tr = UniformUnitMatrix::<Ohm, SMatrix<f64, 2, 1>>::from_nalgebra(SMatrix::repeat(1.0));
    let bl = UniformUnitMatrix::<Ohm, SMatrix<f64, 1, 2>>::from_nalgebra(SMatrix::repeat(1.0));
    let br = UniformUnitMatrix::<Ohm, SMatrix<f64, 1, 1>>::from_nalgebra(SMatrix::repeat(1.0));
    // 3×3 of all ones -> sum 9.
    let s = assemble_uniform::<2, 1>(one(), tr, bl, br);
    assert_eq!(s, 9.0);
}
