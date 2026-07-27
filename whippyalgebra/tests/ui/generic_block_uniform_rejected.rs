//! The uniform `[Element]` axis spelling is a `#[generic_matrix]` feature only.
//! `#[generic_block]` must reject it: a partitioned track's disassembly
//! obligation pins `Take<_, Out = Repeated<Element, N>>`, which would force the
//! trait solver to normalize the repeated-list projection at a *generic* length
//! — impossible, since no `Repeat` impl matches the opaque `Nat<N>`. The
//! attribute reports this at parse time with actionable guidance.
#![allow(unused_imports)]

use nalgebra::SMatrix;
use whippyalgebra::{nalgebra::generic_block, nalgebra::MixedUnitMatrix, Repeated};

type Volt = whippyunits::unit!(V);

#[generic_block(
    rows(N, [Volt]; M, RowB),
    cols(N, ColA; M, ColB),
)]
fn reblock_uniform<RowB, ColA, ColB, const N: usize, const M: usize>(
    _tl: MixedUnitMatrix<Repeated<Volt, N>, ColA, SMatrix<f64, N, N>>,
) -> f64 {
    0.0
}

fn main() {}
