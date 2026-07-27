#![cfg(feature = "nalgebra")]
//! A branded matrix constructs fine, and its elements come back as `Quantity`
//! values carrying the matrix's brand.

use whippyalgebra::dims;
use whippyalgebra::nalgebra::{MixedUnitMatrix, SMatrix};
use whippyunits::qty;

type V3 = SMatrix<f64, 3, 1>;

struct Frame;

// The brand and storage type now live once on the wrapper, so the dimension
// vectors carry only units.
type RowDims = dims![m, m, m];
type ColDims = dims![1];
type BrandedVec = MixedUnitMatrix<RowDims, ColDims, V3, Frame>;

#[test]
fn branded_vector_constructs_and_reads() {
    let mut raw = V3::zeros();
    raw[(0, 0)] = 2.0;
    let v = BrandedVec::new(raw);

    // Entry (0, 0) = RowDims[0] / ColDims[0] = m / 1 = m, reified with the
    // matrix's storage type (f64) and brand (Frame).
    let e0: qty!(m, f64, Frame) = v.get::<0, 0>();
    assert_eq!(e0.unsafe_value, 2.0);
}
