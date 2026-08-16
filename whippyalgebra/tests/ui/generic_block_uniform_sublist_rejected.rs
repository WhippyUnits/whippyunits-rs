//! `#[generic_block(uniform, ..)]` assembles single-unit blocks, which carry no
//! dimension lists — so naming a per-track sublist (`rows(N, RowA; M, RowB)`) is
//! meaningless and must be rejected at parse time with actionable guidance.
#![allow(unused_imports)]

use nalgebra::SMatrix;
use whippyalgebra::nalgebra::{generic_block, UniformUnitMatrix};

type Ohm = whippyunits::unit!(V / A);

#[generic_block(uniform, rows(N, RowA; M, RowB), cols(N; M))]
fn f<RowA, RowB, const N: usize, const M: usize>(
    _tl: UniformUnitMatrix<Ohm, SMatrix<f64, N, N>>,
) {
}

fn main() {}
