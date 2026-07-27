//! `set_block` must reject a block whose dimension lists don't match the
//! destination sublists. The 2x2 block at (0, 1) of a matrix over rows
//! [m, s, A] and cols [A, rot, K] must carry rows [m, s]; supplying a block
//! with rows [m, A] is a compile error.

use nalgebra::SMatrix;
use whippyalgebra::{dims, nalgebra::MixedUnitMatrix};

type M2 = SMatrix<f64, 2, 2>;
type M3 = SMatrix<f64, 3, 3>;
type Mat3 = MixedUnitMatrix<dims![m, s, A], dims![A, rot, K], M3>;
type WrongBlk = MixedUnitMatrix<dims![m, A], dims![rot, K], M2>;

fn main() {
    let mut m = Mat3::new(M3::zeros());
    let blk = WrongBlk::new(M2::zeros());
    m.set_block::<0, 1, 2, 2, _>(&blk);
}
