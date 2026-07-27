//! The `mixed_unit_matrix!` constructor checks each cell against its unit
//! `RowDims[i] / ColDims[j]`. This matrix maps [s, s] (columns) into [m, m]
//! (rows), so every entry must be `m/s`. Supplying a plain seconds quantity for
//! cell (0,0) has the wrong unit and must fail to compile — there is no
//! `IntoEntry<m/s, …>` impl for a `Quantity<s, …>`.

use nalgebra::SMatrix;
use whippyalgebra::{dims, nalgebra::mixed_unit_matrix, nalgebra::MixedUnitMatrix};
use whippyunits::quantity;

type M2 = SMatrix<f64, 2, 2>;
type RowDims = dims![m, m];
type ColDims = dims![s, s];

fn main() {
    // Cell (0,0) must be m/s, but `quantity!(1.0, s)` is seconds.
    let _bad = mixed_unit_matrix![MixedUnitMatrix<RowDims, ColDims, M2>;
        [quantity!(1.0, s), quantity!(3.0, m / s)],
        [quantity!(2.0, m / s), quantity!(4.0, m / s)],
    ];
}
