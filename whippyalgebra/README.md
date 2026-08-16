# WhippyAlgebra

Zero-cost, unit-safe linear algebra, powered by [whippyunits](https://crates.io/crates/whippyunits).

WhippyAlgebra wraps existing linear-algebra libraries in transparent newtype wrappers that add
compile-time unit safety and compile away to the underlying library types. Matrix products,
transposes, determinants, inverses, and decompositions all stay dimensionally coherent — and a
dimensional mistake is a compile error, not a runtime surprise.

## Quick Start

```rust
use whippyalgebra::dims;
use whippyalgebra::nalgebra::mixed_unit_matrix;
use whippyunits::{quantity, qty};

// A state-transition matrix for a [position, velocity] state.
type State = dims![m, m / s];
let phi = mixed_unit_matrix![State, State;
    [1.0, quantity!(0.5, s)],
    [quantity!(0.0, 1 / s), 1.0],
];

let x0 = mixed_unit_matrix![State, dims![1];
    [quantity!(2.0, m)],     // position
    [quantity!(3.0, m / s)], // velocity
];

let x1 = phi * x0;

// x1 = [x0 + v0·Δt, v0] = [3.5 m, 3.0 m/s], read back with enforced units.
let pos: qty!(m) = x1.get::<0, 0>();
let vel: qty!(m / s) = x1.get::<1, 0>();
```

## Matrix types

- **`MixedUnitMatrix`** — carries a *row* dimension list and a *column* dimension list; entry
  `(i, j)` carries the factored unit `RowDims[i] / ColDims[j]`. This is what keeps matrix
  products, transposes, determinants, and inverses dimensionally coherent.
- **`UniformUnitMatrix`** — every entry shares a single unit `U`. Operations that cannot be made
  statically safe on a mixed matrix (e.g. reciprocating the whole matrix, or an SVD with
  dimensioned singular values) live here.

Both are `#[repr(transparent)]` wrappers over the backend's own matrix and expose the same
interface, so there is no runtime cost.

## Backends

| Backend | Feature | Default |
|---|---|---|
| [nalgebra](https://docs.rs/nalgebra) | `nalgebra` | Yes |

The `nalgebra` adapter provides unit-checked wrappers over nalgebra's decompositions (LU,
full-pivot LU, QR, column-pivot QR, SVD, Cholesky, eigen, Schur, bidiagonal, Hessenberg, UDU,
and their generalized variants).

Disable the default feature to depend only on the backend-agnostic unit machinery (`dims`,
`index`, `entry`) while a different backend adapter is used:

```toml
[dependencies]
whippyalgebra = { version = "0.1", default-features = false }
```

## License

Licensed under either of Apache License, Version 2.0 or the MIT license at your option.
