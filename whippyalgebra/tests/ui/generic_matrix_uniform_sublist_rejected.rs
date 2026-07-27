//! `#[generic_matrix(uniform, ..)]` targets a single-unit matrix, which has no
//! dimension lists — so naming a sublist on an axis (`rows(N, State)`, or the
//! `rows(N, [Element])` uniform-axis spelling) is meaningless and must be
//! rejected at parse time with actionable guidance.
#![allow(unused_imports)]

use nalgebra::{Const, OMatrix};
use whippyalgebra::nalgebra::{generic_matrix, UniformUnitMatrix};

type Ohm = whippyunits::unit!(V / A);

#[generic_matrix(uniform, rows(N, State), cols(M))]
fn f<State, const N: usize, const M: usize>(
    _m: UniformUnitMatrix<Ohm, OMatrix<f64, Const<N>, Const<M>>>,
) {
}

fn main() {}
