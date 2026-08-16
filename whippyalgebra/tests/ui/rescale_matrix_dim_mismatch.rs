//! `rescale_matrix` reexpresses each row/column unit in a *different scale of the
//! same dimension*, exactly like whippyunits' `rescale`. Targeting a different
//! *dimension* (here a row of `m` rescaled to `s`) has no `UnitRescale` impl, so
//! the list has no `RescaleFactors` impl and the call must fail to compile.

use nalgebra::SMatrix;
use whippyalgebra::{
    dims,
    nalgebra::{rescale_matrix, MixedUnitMatrix},
};

type M2 = SMatrix<f64, 2, 2>;

type Source = MixedUnitMatrix<dims![m, m], dims![s, s], M2>;
// Row 0 asks to become seconds — a different dimension than metres.
type Bad = MixedUnitMatrix<dims![s, m], dims![s, s], M2>;

fn main() {
    let a = Source::new(M2::zeros());
    let _bad: Bad = rescale_matrix(&a);
}
