//! A uniform block's gauge tag must reproduce its entry unit. The block below
//! carries unit `m/s`, so gauging it to rows `[m, m]` / cols `[m]` — whose
//! entry-unit quotient is `m/m` (dimensionless), not `m/s` — is a compile
//! error: `Uniform(RowDims) / Uniform(ColDims)` must equal the block's unit.

use nalgebra::SMatrix;
use whippyalgebra::{dims, nalgebra::MixedUnitMatrix, nalgebra::UniformUnitMatrix};
use whippyunits::unit;

fn main() {
    let g: UniformUnitMatrix<unit!(m / s), SMatrix<f64, 2, 1>> =
        MixedUnitMatrix::<dims![m, m], dims![s], SMatrix<f64, 2, 1>>::new(SMatrix::<f64, 2, 1>::new(
            10.0, 20.0,
        ))
        .into_uniform();

    // Wrong gauge: quotient m/m = dimensionless, not the block's unit m/s.
    let _bad = g.gauge::<dims![m, m], dims![m]>();
}
