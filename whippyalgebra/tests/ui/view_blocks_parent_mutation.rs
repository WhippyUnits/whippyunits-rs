//! `block_view` borrows the parent matrix for as long as the view lives — that
//! is the whole point of the zero-copy variant. So mutating the parent (here
//! via `set_block`, which needs `&mut self`) while a view is still alive must
//! not compile: the shared view borrow and the mutable write conflict.

use nalgebra::SMatrix;
use whippyalgebra::{dims, nalgebra::MixedUnitMatrix};

type M2 = SMatrix<f64, 2, 2>;

fn main() {
    let mut m = MixedUnitMatrix::<dims![m, s], dims![m, s], M2>::new(M2::new(1.0, 2.0, 3.0, 4.0));

    // `view` borrows `m` immutably.
    let view = m.block_view::<0, 0, 1, 1>();

    // Writing into `m` needs `&mut m` — rejected while `view` is alive.
    let blk =
        MixedUnitMatrix::<dims![m], dims![m], SMatrix<f64, 1, 1>>::new(SMatrix::<f64, 1, 1>::new(9.0));
    m.set_block::<0, 0, 1, 1, _>(&blk);

    // Keep the view borrow live past the mutation.
    let _ = view.get::<0, 0>();
}
