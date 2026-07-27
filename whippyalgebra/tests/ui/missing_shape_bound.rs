//! A function generic over a shape (`const N: usize`) that indexes or slices a
//! matrix must declare that the shape has a type-level representation
//! (`Const<N>: ShapeIndex`). Omitting it is the one obligation the type-nat
//! encoding leaks, so the failure must report *as* `ShapeIndex` — naming the
//! documented trait — rather than dumping typenum's `ToUInt` impls.

use nalgebra::SMatrix;
use whippyalgebra::{dims, nalgebra::MixedUnitMatrix};

// Slices an `N x N` block by a *generic* index without `Const<N>: ShapeIndex`.
fn top_left<const N: usize>(m: &MixedUnitMatrix<dims![m, m], dims![1, 1], SMatrix<f64, 2, 2>>) {
    let _ = m.block::<0, 0, N, N>();
}

fn main() {}
