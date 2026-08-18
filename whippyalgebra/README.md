# WhippyAlgebra

Zero-cost, unit-safe linear algebra, powered by [whippyunits](https://crates.io/crates/whippyunits).

WhippyAlgebra wraps existing linear-algebra libraries in transparent newtype wrappers that add
compile-time unit safety and compile away to the underlying library types.

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

Disable the `nalgebra` feature to depend only on the backend-agnostic unit machinery (`dims`,
`index`, `entry`) while a different backend adapter is used:

```toml
[dependencies]
whippyalgebra = { version = "0.1", default-features = false, features = ["std"] }
```

## `no_std` / `no-alloc`

Like [whippyunits](https://crates.io/crates/whippyunits), WhippyAlgebra is `#![no_std]` when the
default `std` feature is off. The features are additive:

| Feature | Default | Enables |
|---|---|---|
| `std` | Yes | Full standard library; implies `alloc`. Adds nalgebra's `matrixmultiply` fast path and the `std`-only matrix exponential (`.exp()`). |
| `alloc` | via `std` | Heap allocation without `std`. Required for matrix `Display` (unit-aware pretty-printing renders each cell to a `String`), dynamically sized matrices (`DMatrix`/`DVector`, `from_dyn`), and the `Vec`/columns/rows uniform constructors. |
| `nalgebra` | Yes | The nalgebra backend adapter. |

With neither `std` nor `alloc`, the statically sized, unit-checked core — construction,
arithmetic, decompositions, element access, and `rescale_matrix` — keeps working; the
allocation-dependent items above are compiled out, so matrices print via `Debug` (delegating to
the inner matrix) rather than the unit-aware `Display`. Floating-point math resolves in every
configuration: the `nalgebra` backend always enables nalgebra's `libm` feature, so `f32`/`f64`
satisfy `ComplexField` even without `std`.

```toml
[dependencies]
# no_std, no-alloc:
whippyalgebra = { version = "0.1", default-features = false, features = ["nalgebra"] }
# no_std + alloc (matrix Display, dynamic matrices):
whippyalgebra = { version = "0.1", default-features = false, features = ["alloc", "nalgebra"] }
```

## Editor support (LSP proxy)

The type signatures WhippyAlgebra produces are deeply nested — a `MixedUnitMatrix` carries two
type-level dimension lists, each entry a `Quantity<Unit<…>, …>` — so the
[`whippyunits-lsp-proxy`](https://github.com/WhippyUnits/whippyunits-rs/tree/main/lsp-proxy)
crate makes them legible in your editor. It sits between rust-analyzer and
your editor and rewrites hovers, inlay hints, and error messages. For WhippyAlgebra, it:

- collapses the type-level cons-lists (`DCons<(m·s⁻¹), DCons<m, DNil>>`) into bracketed array
  notation (`[(m·s⁻¹), m]`);
- elides the redundant default `()` brand, so unbranded matrices read as
  `MixedUnitMatrix<[…], […], Matrix<…>>`;
- for a concrete `MixedUnitMatrix` / `UniformUnitMatrix`, appends a values-free diagram of the
  matrix on hover — the row/column unit labels in the margins and each entry's unit literal
  (`RowDims[i] / ColDims[j]`) in the cells.

See the `whippyunits-lsp-proxy` crate for setup and configuration.

## License

Licensed under either of Apache License, Version 2.0 or the MIT license at your option.
