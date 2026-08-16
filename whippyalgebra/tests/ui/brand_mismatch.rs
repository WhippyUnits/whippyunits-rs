//! Brands are enforced at the *operation* level: the brand is stored once on
//! each matrix, so a well-formed matrix can never internally disagree on brand.
//! But multiplying two matrices whose brands differ must fail to compile — the
//! `Mul` impl requires both operands (and their shared inner dimension) to carry
//! the same brand.

use nalgebra::SMatrix;
use whippyalgebra::{dims, nalgebra::MixedUnitMatrix};

type M6x3 = SMatrix<f64, 6, 3>;
type V3 = SMatrix<f64, 3, 1>;

struct FrameA;
struct FrameB;

type RowDims = dims![m, m, m, m, m, m];
type MidDims = dims![m, m, m];
type ColDims = dims![1];

// A `FrameA` matrix mapping a `FrameA` 3-vector into a `FrameA` 6-space...
type MatA = MixedUnitMatrix<RowDims, MidDims, M6x3, FrameA>;
// ...and a `FrameB` column vector living in the input space.
type VecB = MixedUnitMatrix<MidDims, ColDims, V3, FrameB>;

fn main() {
    let a = MatA::new(M6x3::zeros());
    let x = VecB::new(V3::zeros());

    // `a` is branded `FrameA`, `x` is branded `FrameB`: no shared-brand `Mul`.
    let _bad = a * x;
}
