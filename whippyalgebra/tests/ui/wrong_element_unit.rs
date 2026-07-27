//! The unit-safe accessor must reject an incorrect unit annotation. When a
//! matrix's row and column dimension vectors are identical, every entry is
//! dimensionless (`(m/s) / (m/s)`), so annotating one as seconds must fail to
//! compile.

use nalgebra::SMatrix;
use whippyalgebra::{dims, nalgebra::MixedUnitMatrix};
use whippyunits::qty;

type M2 = SMatrix<f64, 2, 2>;
type Dims = dims![m / s, m / s];
type Mat = MixedUnitMatrix<Dims, Dims, M2>;

fn main() {
    let a = Mat::new(M2::zeros());
    let _wrong: qty!(s) = a.get::<0, 0>();
}
