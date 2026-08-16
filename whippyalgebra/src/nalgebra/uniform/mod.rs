//! A unit-safe matrix all of whose entries share one unit.
//!
//! [`MixedUnitMatrix`] tags each entry `(i, j)` with its
//! own unit `RowDims[i] / ColDims[j]`, so whole-matrix operations that would
//! have to erase units to a bare `&T`/`&[T]` (iteration, slicing, mapping) are
//! deliberately withheld. A uniform matrix is the special case in which
//! every entry carries the single unit `U`.
//!
//! A uniform matrix is parameterized by that one entry unit `U`, not by a
//! separate row and column unit: splitting it would reintroduce the mixed type's
//! gauge freedom (`RowUnit / ColUnit` is invariant under scaling both by a common
//! `g`) without strengthening the contract, which promises only that all entries
//! share one unit.
//!
//! Collapsing the two per-entry lists to a single scalar unit is what makes bulk
//! access sound again: [`iter`](UniformUnitMatrix::iter) yields a `Quantity` per
//! element, [`get`](UniformUnitMatrix::get) takes runtime indices (the unit no
//! longer depends on which entry), and [`map`](UniformUnitMatrix::map) rewrites
//! every value at once — all with the unit fixed statically.
//!
//! The algebra specializes cleanly to one unit:
//!
//! - transpose leaves the unit unchanged (`Mᵀ(i, j) = M(j, i)` still has
//!   unit `U`) — no reciprocal, unlike the mixed case;
//! - matrix product multiplies entry units (`⟨Ua⟩ · ⟨Ub⟩ = ⟨Ua · Ub⟩`),
//!   since every summed term of `C(i,k) = Σⱼ A(i,j)·B(j,k)` has unit `Ua · Ub`;
//! - scalar-by-quantity scaling multiplies/divides `U` by the scalar's unit.

#[allow(unused_imports)] // re-exported siblings + Quantity, for intra-doc links
use crate::nalgebra::*;
#[allow(unused_imports)]
use whippyunits::quantity::Quantity;

use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;

mod construction;
mod ops;

pub use ops::rescale_uniform_matrix;

/// A unit-safe nalgebra matrix whose entries all share the unit `U`.
///
/// - `U` is the single unit carried by every entry.
/// - `M` is the underlying nalgebra matrix type (also fixing the storage type).
/// - `Brand` is the shared brand; it defaults to `()` and is last so it can be
///   omitted.
///
/// This is the homogeneous specialization of
/// [`MixedUnitMatrix`]: the two per-entry dimension
/// lists collapse to a single entry unit `U`. That one shared unit is what
/// re-enables the whole-matrix, runtime-indexed, and bulk operations the mixed
/// type must forgo.
///
/// # Construction
///
/// The declarative [`uniform_unit_matrix!`](crate::nalgebra::uniform_unit_matrix)
/// macro is the normal entry point: it leads with the single entry unit (any
/// `whippyunits::unit!` expression) and counts the shape from the literal rows.
/// To attach a unit to a matrix you already hold, use
/// [`from_nalgebra`](Self::from_nalgebra) — no shape check is needed, since a
/// uniform matrix imposes no per-row/column list lengths. Bulk constructors fill
/// from a value or closure ([`from_element`](Self::from_element) /
/// [`from_fn`](Self::from_fn)) or from raw data
/// ([`from_row_slice`](Self::from_row_slice) /
/// [`from_column_slice`](Self::from_column_slice) / [`from_vec`](Self::from_vec) /
/// [`from_iterator`](Self::from_iterator)), and
/// [`from_rows`](Self::from_rows) / [`from_columns`](Self::from_columns) stack
/// vectors. Special forms have their own constructors:
/// [`identity`](Self::identity) and [`from_diagonal`](Self::from_diagonal).
///
/// ```
/// use whippyalgebra::nalgebra::uniform_unit_matrix;
/// use whippyunits::quantity;
///
/// // One unit in the header; the 2×2 shape is counted from the rows.
/// let m = uniform_unit_matrix![m / s;
///     [quantity!(1.0, m / s), quantity!(2.0, m / s)],
///     [quantity!(3.0, m / s), quantity!(4.0, m / s)],
/// ];
/// # let _ = m;
/// ```
///
/// # Basic operations
///
/// Because all entries share the same unit `U`, the bulk, runtime-indexed access is sound:
/// [`get`](Self::get) / [`get_mut`](Self::get_mut) / [`set`](Self::set) take
/// runtime indices (the unit no longer depends on which entry),
/// [`iter`](Self::iter) / [`iter_mut`](Self::iter_mut) yield a [`Quantity`] per
/// element, and [`map`](Self::map) / [`zip_map`](Self::zip_map) rewrite every
/// value at once. The type is also `Index`/`IndexMut`-able. Vector routines
/// follow: [`dot`](Self::dot) / [`cross`](Self::cross), [`norm`](Self::norm) /
/// [`normalize`](Self::normalize), the reductions [`sum`](Self::sum) /
/// [`mean`](Self::mean) / [`variance`](Self::variance), and
/// [`min`](Self::min) / [`max`](Self::max).
///
/// The algebra specializes cleanly to one unit: `Mul` multiplies entry units
/// (`⟨Ua⟩ · ⟨Ub⟩ = ⟨Ua · Ub⟩`), [`transpose`](Self::transpose) leaves `U`
/// unchanged (no reciprocal, unlike the mixed case), `Add` / `Sub` require the
/// units to match, and [`scale`](Self::scale) / [`unscale`](Self::unscale) fold
/// in a scalar's unit. [`determinant`](Self::determinant) raises `U` to the
/// matrix order (`Uⁿ`). To rescale the entry unit without moving data, see
/// [`rescale_uniform_matrix`].
///
/// ```
/// use whippyalgebra::nalgebra::uniform_unit_matrix;
/// use whippyunits::{quantity, qty};
///
/// let v = uniform_unit_matrix![m;
///     [quantity!(3.0, m)],
///     [quantity!(4.0, m)],
/// ];
///
/// // Runtime indices are sound because every entry shares the unit `U`.
/// let first: qty!(m) = v.get(0, 0);
/// // Bulk ops the mixed type withholds: `iter` yields a `Quantity` per entry,
/// // and both `sum` and the Frobenius `norm` stay in `m`.
/// let count = v.iter().count();
/// let total: qty!(m) = v.sum();
/// let length: qty!(m) = v.norm();
/// # let _ = (first, count, total, length);
/// ```
///
/// # Gauge
///
/// Unlike [`MixedUnitMatrix`], which factors
/// each entry unit into a quotient of a row (output) and column (input) unit, a
/// uniform matrix has no gauge freedom: it stores only the single unit `U` for
/// the whole matrix, not a quotient. Being gaugeless, the two views convert
/// asymmetrically:
///
/// - mixed → uniform erases the gauge (only the quotient `RowUnit / ColUnit`
///   survives), and compiles only when both lists are already uniform:
///   [`into_uniform`](crate::nalgebra::MixedUnitMatrix::into_uniform) consumes,
///   while [`to_uniform`](crate::nalgebra::MixedUnitMatrix::to_uniform) /
///   [`as_uniform`](crate::nalgebra::MixedUnitMatrix::as_uniform) copy or borrow
///   and leave the gauge-carrying mixed matrix intact for later recomposition.
/// - uniform → mixed must invent a gauge — any `RowUnit`, `ColUnit` with
///   `RowUnit / ColUnit = U` — via [`gauge`](Self::gauge) /
///   [`into_mixed`](Self::into_mixed) (or the [`gauge!`](crate::nalgebra::gauge)
///   cell wrapper in a block layout); a quotient that doesn't reproduce `U` is a
///   compile error.
///
/// ```
/// use whippyalgebra::dims;
/// use whippyalgebra::nalgebra::mixed_unit_matrix;
/// use whippyunits::quantity;
///
/// // A mixed velocity vector gauged as `m` output per `s` input...
/// let v = mixed_unit_matrix![dims![m, m], dims![s];
///     [quantity!(1.0, m / s)],
///     [quantity!(2.0, m / s)],
/// ];
/// // ...collapses to a single unit `m/s`, forgetting which gauge it lived in.
/// let u = v.to_uniform();
/// # let _ = u.get(0, 0);
/// ```
///
/// # Block construction
///
/// Uniform blocks that all share one entry unit `U` assemble into a larger
/// uniform matrix with [`block_matrix![uniform; …]`](crate::nalgebra::block_matrix):
/// the leading `uniform;` tag states the intent (and pairs with
/// `#[generic_block(uniform, …)]` in generic code), each grid row is a bracketed
/// list of blocks, and the right-associative [`hcat`](Self::hcat) /
/// [`vcat`](Self::vcat) keep the result uniform in `U` — a cell of a different
/// unit is a compile error. Because there are no per-block row/column gauges to
/// reconcile, none of the mixed layout's [`zeros!`](crate::nalgebra::zeros) /
/// [`gauge!`](crate::nalgebra::gauge) plumbing is needed.
///
/// To place a uniform block into a *mixed*
/// [`block_matrix!`](crate::nalgebra::block_matrix) layout instead — where its
/// neighbors fix which spaces it must line up with — gauge it with the
/// [`gauge!`](crate::nalgebra::gauge) cell wrapper (see [Generics](#generics)).
/// Conversely, [`unblock_matrix!(reduce_uniform; …)`](crate::nalgebra::unblock_matrix)
/// is what reads a uniform sub-block back out of a mixed matrix as a
/// `UniformUnitMatrix`.
///
/// ```
/// use whippyalgebra::nalgebra::{block_matrix, uniform_unit_matrix};
/// use whippyunits::quantity;
///
/// // Four blocks, all uniform in `m`; the assembly stays uniform in `m`.
/// let a = uniform_unit_matrix![m; [quantity!(1.0, m), quantity!(2.0, m)]];
/// let b = uniform_unit_matrix![m; [quantity!(3.0, m)]];
/// let c = uniform_unit_matrix![m; [quantity!(4.0, m), quantity!(5.0, m)]];
/// let d = uniform_unit_matrix![m; [quantity!(6.0, m)]];
///
/// let grid = block_matrix![uniform;
///     [a, b],
///     [c, d],
/// ];
/// # let _ = grid;
/// ```
///
/// # Decompositions
///
/// The standard nalgebra factorizations are exposed as unit-carrying wrappers,
/// each returning a struct that supports factor-once,
/// [`solve`](UniformCholesky::solve)-many reuse and a `recompose` round-trip.
/// Available on a square matrix: [`cholesky`](Self::cholesky) and
/// [`udu`](Self::udu), [`lu`](Self::lu) / [`full_piv_lu`](Self::full_piv_lu),
/// [`schur`](Self::schur), [`hessenberg`](Self::hessenberg),
/// [`symmetric_tridiagonalize`](Self::symmetric_tridiagonalize),
/// [`eigenvalues`](Self::eigenvalues) / [`complex_eigenvalues`](Self::complex_eigenvalues),
/// and [`generalized_symmetric_eigen`](Self::generalized_symmetric_eigen) /
/// [`generalized_eigenvalues`](Self::generalized_eigenvalues). Rectangular
/// matrices support [`qr`](Self::qr), [`bidiagonalize`](Self::bidiagonalize),
/// [`svd`](Self::svd), and [`pseudo_inverse`](Self::pseudo_inverse) — their thin
/// factors contract on a dimensionless pivot of length `min(m, n)`. (The
/// rank-revealing [`col_piv_qr`](Self::col_piv_qr) stays square.)
/// [`try_inverse`](Self::try_inverse) reciprocates the unit to `1 / U`.
///
/// A uniform matrix carries the same unit `U` on both margins, which is a
/// canonical metric — the one convention a genuinely mixed matrix lacks (which is
/// why the mixed twin has no bare `svd`/`qr`/`bidiagonalize`, only their
/// metric-supplied `generalized_*` forms). So a uniform matrix's singular values
/// (`svd`'s `Σ`) and eigenvalues come out honestly dimensioned in `U`, with no
/// even-exponent gate (the root lives in the numbers, never the unit).
///
/// # Generics
///
/// A `UniformUnitMatrix` is usually written over concrete sizes, but its shape
/// can be made generic over const generics threaded into nalgebra's
/// [`Const`]. Generic code needs the nalgebra storage-allocator
/// bounds in its `where` clause; rather than spell these out by hand, annotate
/// the item with the [`generic_matrix!`](crate::nalgebra::generic_matrix)
/// attribute (or [`generic_block!`](crate::nalgebra::generic_block) for block
/// grids) under a leading `uniform` keyword, which targets the single-unit
/// form and synthesizes the obligation set. Add the `decompose` keyword when the
/// body reaches for a reduction-based factorization (`svd` / `pseudo_inverse` /
/// `schur` / … / `col_piv_qr`) to also emit the Householder-reduction workspace
/// bounds.
///
/// To place a uniform block into a mixed [`block_matrix!`](crate::nalgebra::block_matrix)
/// layout — where its neighbors fix which spaces it must line up with — gauge
/// it into a [`MixedUnitMatrix`] with [`gauge`](Self::gauge) /
/// [`into_mixed`](Self::into_mixed) (or [`gauge_dyn`](Self::gauge_dyn) for a
/// runtime shape), most legibly via the [`gauge!`](crate::nalgebra::gauge) cell
/// wrapper.
///
/// ```
/// use whippyalgebra::nalgebra::{generic_matrix, Const, OMatrix, UniformUnitMatrix};
/// use whippyunits::unit;
///
/// type Ohm = unit!(V / A);
/// type Volt = unit!(V);
/// type Amp = unit!(A);
///
/// // Generic over both axes; the `uniform` keyword targets the single-unit form
/// // and `decompose` adds the pseudo-inverse workspace bounds.
/// #[generic_matrix(uniform, rows(N), cols(M), decompose)]
/// fn solve_currents<const N: usize, const M: usize>(
///     transfer: UniformUnitMatrix<Ohm, OMatrix<f64, Const<N>, Const<M>>>,
///     readings: UniformUnitMatrix<Volt, OMatrix<f64, Const<N>, Const<1>>>,
/// ) -> Result<UniformUnitMatrix<Amp, OMatrix<f64, Const<M>, Const<1>>>, &'static str> {
///     // G⁺ is uniform in 1/Ω = A/V; contracting the volt readings lands in A.
///     Ok(transfer.pseudo_inverse(1e-12)? * readings)
/// }
/// # let _ = solve_currents::<4, 2>;
/// ```
pub struct UniformUnitMatrix<U, M, Brand = ()> {
    pub(crate) inner: M,
    _unit: PhantomData<fn() -> (U, Brand)>,
}

// As with `MixedUnitMatrix`, the unit and brand are purely phantom (behind a
// `fn() -> …` pointer, always `Copy`/`Send`/`Sync`), so the value traits depend
// on the underlying matrix `M` alone and are written by hand to avoid a derive
// spuriously constraining `U`/`Brand`.

impl<U, M: Clone, Brand> Clone for UniformUnitMatrix<U, M, Brand> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _unit: PhantomData,
        }
    }
}

impl<U, M: Copy, Brand> Copy for UniformUnitMatrix<U, M, Brand> {}

impl<U, M: PartialEq, Brand> PartialEq for UniformUnitMatrix<U, M, Brand> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<U, M: Eq, Brand> Eq for UniformUnitMatrix<U, M, Brand> {}

impl<U, M: Hash, Brand> Hash for UniformUnitMatrix<U, M, Brand> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

impl<U, M: Default, Brand> Default for UniformUnitMatrix<U, M, Brand> {
    fn default() -> Self {
        Self {
            inner: M::default(),
            _unit: PhantomData,
        }
    }
}

impl<U, M: fmt::Debug, Brand> fmt::Debug for UniformUnitMatrix<U, M, Brand> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("UniformUnitMatrix")
            .field(&self.inner)
            .finish()
    }
}
