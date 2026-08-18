//! The unit-safe matrix newtype wrapping an underlying nalgebra matrix.

#[allow(unused_imports)] // re-exported siblings + Quantity, for intra-doc links
use crate::nalgebra::*;
#[allow(unused_imports)]
use whippyunits::quantity::Quantity;

use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;

use crate::dims::{DCons, DNil};
use crate::index::ShapeIndex;

mod construction;
// The matrix `Display` renders each cell to a `String` and aligns columns via
// intermediate `Vec`s, so it lives behind `alloc` (it also leans on
// whippyunits' `UnitDisplayExt`/`UnitLabel`, which are themselves `alloc`-gated).
// Without `alloc` the matrix still has the derived-style `Debug` from this
// module. See the crate `alloc`/`std` features.
#[cfg(feature = "alloc")]
mod display;
mod ops;

pub use ops::rescale_matrix;

/// A unit-safe nalgebra matrix.
///
/// - `RowDims` is the row (output-space) dimension vector (a list of `Unit`s).
/// - `ColDims` is the column (input-space) dimension vector.
/// - `M` is the underlying nalgebra matrix type (e.g. `SMatrix<f64, R, C>`),
///   which also fixes the storage type shared by every entry.
/// - `Brand` is the single brand shared by every entry; it defaults to `()`
///   (unbranded) and is the last parameter so it can usually be omitted.
///
/// The unit of entry `(i, j)` is `RowDims[i] / ColDims[j]`, reified with the
/// matrix's storage type and brand.
///
/// # Construction
///
/// The declarative [`mixed_unit_matrix!`](crate::nalgebra::mixed_unit_matrix)
/// macro is the normal entry point: it leads with the row and column dimension
/// lists (the shape is read from their lengths) and checks every entry against
/// its cell unit `RowDims[i] / ColDims[j]`. To attach units to a matrix you
/// already hold, use [`new`](Self::new) (statically-sized; the shape/length
/// agreement is a *compile-time* assertion) or [`from_dyn`](Self::from_dyn) (a
/// runtime-shaped `Dyn` matrix, copied into static storage after a runtime
/// check). [`from_nalgebra`](Self::from_nalgebra) wraps without any check.
/// Special forms have their own constructors: [`identity`](Self::identity) (the
/// endomorphism `I : ⟨Dims, Dims⟩`) and
/// [`from_diagonal`](Self::from_diagonal) (a diagonal built from an index
/// column vector). For vectors, see the [`UnitVector`] / [`UnitRowVector`]
/// aliases.
///
/// ```
/// use whippyalgebra::dims;
/// use whippyalgebra::nalgebra::mixed_unit_matrix;
/// use whippyunits::{quantity, qty};
///
/// type State = dims![m, m / s];
///
/// // Each entry is checked against its cell unit `RowDims[i] / ColDims[j]`.
/// let phi = mixed_unit_matrix![State, State;
///     [1.0, quantity!(0.5, s)],
///     [quantity!(0.0, 1 / s), 1.0],
/// ];
/// let x0 = mixed_unit_matrix![State, dims![1];
///     [quantity!(2.0, m)],
///     [quantity!(3.0, m / s)],
/// ];
/// # let _ = (&phi, &x0);
/// # let _: qty!(m) = x0.get::<0, 0>();
/// ```
///
/// # Basic operations
///
/// Element access is unit-checked at the type level:
/// [`get`](Self::get) / [`get_mut`](Self::get_mut) read and write an entry as a
/// dimensioned [`Quantity`], and [`set`](Self::set) / [`with`](Self::with)
/// update one. Whole rows, columns, and the [`diagonal`](Self::diagonal) read
/// out as vectors; [`block`](Self::block) and its variants slice sub-matrices,
/// while [`hcat`](Self::hcat) / [`vcat`](Self::vcat) glue matrices together.
///
/// Arithmetic is delegated to the inner matrix with the dimension vectors
/// combined at the type level: `Mul` contracts a shared inner axis (`A · B`
/// type-checks exactly when `A`'s `ColDims` equal `B`'s `RowDims`, giving
/// `⟨A::RowDims, B::ColDims⟩`), `Add` / `Sub` require both lists to match, and
/// [`transpose`](Self::transpose) swaps them. [`scale`](Self::scale) /
/// [`unscale`](Self::unscale) multiply by a bare scalar,
/// [`component_mul`](Self::component_mul) /
/// [`component_div`](Self::component_div) act entrywise, and
/// [`determinant`](Self::determinant) / [`trace`](Self::trace) /
/// [`try_inverse`](Self::try_inverse) / [`solve`](Self::solve) cover the common
/// square-matrix routines. To rescale a matrix's units without moving data, see
/// [`rescale_matrix`].
///
/// ```
/// use whippyalgebra::dims;
/// use whippyalgebra::nalgebra::mixed_unit_matrix;
/// use whippyunits::{quantity, qty};
///
/// type State = dims![m, m / s];
/// let phi = mixed_unit_matrix![State, State;
///     [1.0, quantity!(0.5, s)],
///     [quantity!(0.0, 1 / s), 1.0],
/// ];
/// let x0 = mixed_unit_matrix![State, dims![1];
///     [quantity!(2.0, m)],
///     [quantity!(3.0, m / s)],
/// ];
///
/// // The product contracts the shared inner axis; entries read back with units.
/// let x1 = phi * x0;
/// let pos: qty!(m) = x1.get::<0, 0>();
/// let vel: qty!(m / s) = x1.get::<1, 0>();
/// # let _ = (pos, vel);
/// ```
///
/// # Gauge
///
/// Matrices with consistent entry units may fail to add or multiply directly,
/// because the row and column units may not match even when the entry units are
/// the same. We call this "gauge mismatch". A column of velocities, for instance, can
/// be typed all-in-the-rows as `⟨[m/s, …], [1]⟩` (a plain index vector) or as a
/// per-second gain reading a time input, `⟨[m, …], [s]⟩`.
///
/// The entry units `RowDims[i] / ColDims[j]` do not pin down the
/// two lists: scaling every row and column unit by one common unit `g` (the
/// "gauge" of the matrix) leaves every entry unit unchanged, because
/// `(g·rowᵢ) / (g·colⱼ) = rowᵢ / colⱼ`.
///
/// To reconcile this, use [`canonical_at_row`](Self::canonical_at_row) /
/// [`canonical_at_col`](Self::canonical_at_col) to fix a row or column unit to
/// dimensionless, or [`with_gauge_at_row`](Self::with_gauge_at_row) /
/// [`with_gauge_at_column`](Self::with_gauge_at_column) to fix a row or column
/// unit to a specific unit.
///
/// Collapsing a mixed matrix that happens to be uniform into a single-unit
/// [`UniformUnitMatrix`] (via
/// [`into_uniform`](Self::into_uniform) and friends) erases the gauge entirely,
/// keeping only the invariant quotient; re-expanding to a mixed matrix one must
/// invent a gauge back (see [`gauge!`](crate::nalgebra::gauge)).
///
/// ```
/// use whippyalgebra::dims;
/// use whippyalgebra::nalgebra::{mixed_unit_matrix, MixedUnitMatrix};
/// use whippyunits::{quantity, qty};
///
/// // A mixed-unit 2×2: rows carry `[m, s]` and columns `[s, m]`, so the
/// // entry-unit grid is `[[m/s, 1], [1, s/m]]`.
/// let mat = mixed_unit_matrix![dims![m, s], dims![s, m];
///     [quantity!(1.0, m / s), quantity!(2.0, 1)],
///     [quantity!(3.0, 1),     quantity!(4.0, s / m)],
/// ];
///
/// // Anchoring row 0 divides both dim lists by `m`; anchoring column 0 divides
/// // them by `s`. Same physical matrix, but the two normal forms are *different
/// // types*
/// let by_row: MixedUnitMatrix<dims![1, s / m], dims![s / m, 1], _> =
///     mat.canonical_at_row::<0>();
/// let by_col: MixedUnitMatrix<dims![m / s, 1], dims![1, m / s], _> =
///     mat.canonical_at_col::<0>();
///
/// // Because `by_row` and `by_col` are different types, `by_row + by_col` would
/// // not type-check. Regauging leaves entry units and values untouched, though:
/// // entry (0,0) is `m/s` and still `1.0` in either normal form.
/// let e_row: qty!(m / s) = by_row.get::<0, 0>();
/// let e_col: qty!(m / s) = by_col.get::<0, 0>();
/// assert_eq!(e_row.unsafe_value, e_col.unsafe_value);
/// ```
///
/// # Block construction
///
/// A larger matrix can be assembled from a grid of typed sub-blocks with
/// [`block_matrix!`](crate::nalgebra::block_matrix): each grid row is a bracketed
/// list of blocks, blocks in a row must agree on their row space and blocks in a
/// column on their column space (enforced by [`hcat`](Self::hcat) /
/// [`vcat`](Self::vcat) unification, so a misaligned block is a compile
/// error), and the result's `RowDims` / `ColDims` are the concatenations of the
/// block lists. All-zero cells are written with [`zeros!`](crate::nalgebra::zeros)
/// so they still state their unit spaces (a bare `0` couldn't), and a uniform
/// block is slotted in via the [`gauge!`](crate::nalgebra::gauge) cell wrapper.
/// Storage is inferred: an all-static grid concatenates to an owned `SMatrix`
/// (via nalgebra's [`DimAdd`](nalgebra::DimAdd)), while a `Dyn` block anywhere
/// yields a `DMatrix`.
///
/// The reading inverse is [`unblock_matrix!`](crate::nalgebra::unblock_matrix),
/// which destructures a matrix back into its typed sub-blocks from a partition
/// literal (each cell stating its `(height, width)`); its default `mixed` mode
/// keeps each block's gauge, while `reduce_uniform` reduces any block with
/// uniform row/column sublists to a [`UniformUnitMatrix`] (a leading `views`
/// keyword binds borrowed views instead of owned copies). A single block can
/// also be read or overwritten
/// directly with [`block`](Self::block) / [`set_block`](Self::set_block) /
/// [`with_block`](Self::with_block). Assembling or slicing in generic code
/// needs the nalgebra `DimAdd` / allocator obligations as `where` bounds; the
/// [`generic_block!`](crate::nalgebra::generic_block) attribute writes them from
/// the grid shape.
///
/// ```
/// use whippyalgebra::dims;
/// use whippyalgebra::nalgebra::{block_matrix, mixed_unit_matrix, zeros};
/// use whippyunits::quantity;
///
/// type State = dims![m, m / s];
/// type Input = dims![m / s^2];
///
/// let a = mixed_unit_matrix![State, State;
///     [1.0, quantity!(0.1, s)],
///     [quantity!(0.0, 1 / s), 1.0],
/// ];
/// let b = mixed_unit_matrix![State, Input;
///     [quantity!(0.005, s^2)],
///     [quantity!(0.1, s)],
/// ];
///
/// // The augmented `[[A, B], [0, 0]]` over `State ⊕ Input` (a Van Loan block).
/// let m = block_matrix![
///     [a,                    b                   ],
///     [zeros![Input, State], zeros![Input, Input]],
/// ];
/// # let _ = m;
/// ```
///
/// # Decompositions
///
/// The standard nalgebra factorizations are exposed as unit-carrying wrappers,
/// each threading the row/column types through its factors and returning a
/// struct that supports factor-once, [`solve`](Cholesky::solve)-many reuse and a
/// `recompose` round-trip. Available on a square metric or endomorphism:
/// [`cholesky`](Self::cholesky) and [`udu`](Self::udu) (both `M = L·Lᵀ`-style,
/// no even-exponent constraint on a mixed metric), [`lu`](Self::lu),
/// [`symmetric_eigen`](Self::symmetric_eigen),
/// [`schur`](Self::schur), [`hessenberg`](Self::hessenberg),
/// [`eigenvalues`](Self::eigenvalues), and [`exp`](Self::exp) /
/// [`pow`](Self::pow).
///
/// The orthogonal-frame factorizations (QR, column-pivoted QR, SVD,
/// bidiagonalization) orthonormalize against a metric — and the pivoted forms also
/// rank the columns by norm — and a genuinely mixed matrix has no canonical
/// metric, so its `Q`/`U`/`V` would be orthonormal (and its pivot ordered) only in
/// the silent identity metric, more footgun than feature. The mixed forms
/// therefore take the metric explicitly:
/// [`generalized_qr`](Self::generalized_qr),
/// [`generalized_col_piv_qr`](Self::generalized_col_piv_qr),
/// [`generalized_svd`](Self::generalized_svd), and
/// [`generalized_bidiagonalize`](Self::generalized_bidiagonalize) produce
/// metric-orthonormal factors (and, for the pivoted variant, a well-defined
/// rank-revealing permutation and a metric least-squares `solve`). The
/// pseudoinverse is gated the same way: its type `⟨ColDims, RowDims⟩` is
/// Penrose-forced and honest, but its Euclidean least-squares values silently
/// assume the identity metric, so the mixed rectangular/rank-deficient case takes
/// the metric explicitly via
/// [`generalized_pseudo_inverse`](Self::generalized_pseudo_inverse) (for a square
/// full-rank matrix it is metric-free — that is just
/// [`try_inverse`](Self::try_inverse)). (A uniform matrix carries one shared unit
/// on both sides, which is a canonical metric, so the bare `qr` / `col_piv_qr` /
/// `svd` / `bidiagonalize` / `pseudo_inverse` live on the uniform twin
/// [`UniformUnitMatrix`].)
///
/// ```
/// use whippyalgebra::dims;
/// use whippyalgebra::nalgebra::{MixedUnitMatrix, SMatrix, mixed_unit_matrix};
/// use whippyunits::{qty, quantity};
///
/// // A mixed 3×2 design `M : ⟨[m, m/s, m], [m, m/s]⟩` — three heterogeneous
/// // sensor rows measuring a (position, velocity) column space.
/// type Rows = dims![m, m / s, m];
/// type Cols = dims![m, m / s];
/// type Design = MixedUnitMatrix<Rows, Cols, SMatrix<f64, 3, 2>>;
///
/// // The two metrics live on two different spaces: the codomain (row) metric
/// // `g_r : ⟨1/Rows, Rows⟩` is 3×3, the domain (column) metric
/// // `g_c : ⟨1/Cols, Cols⟩` is 2×2. A genuinely mixed matrix has no canonical
/// // metric, so both are supplied explicitly (here each is its own identity).
/// type RowMetric = MixedUnitMatrix<dims![1 / m, s / m, 1 / m], Rows, SMatrix<f64, 3, 3>>;
/// type ColMetric = MixedUnitMatrix<dims![1 / m, s / m], Cols, SMatrix<f64, 2, 2>>;
///
/// let a = mixed_unit_matrix! {Design;
///     [1.0,                   quantity!(0.5, s)],
///     [quantity!(0.0, 1 / s), 1.0],
///     [1.0,                   quantity!(2.0, s)],
/// };
///
/// let g_r = mixed_unit_matrix! {RowMetric;
///     [quantity!(1.0, 1 / m ^ 2), quantity!(0.0, s / m ^ 2),     quantity!(0.0, 1 / m ^ 2)],
///     [quantity!(0.0, s / m ^ 2), quantity!(1.0, s ^ 2 / m ^ 2), quantity!(0.0, s / m ^ 2)],
///     [quantity!(0.0, 1 / m ^ 2), quantity!(0.0, s / m ^ 2),     quantity!(1.0, 1 / m ^ 2)],
/// };
/// let g_c = mixed_unit_matrix! {ColMetric;
///     [quantity!(1.0, 1 / m ^ 2), quantity!(0.0, s / m ^ 2)],
///     [quantity!(0.0, s / m ^ 2), quantity!(1.0, s ^ 2 / m ^ 2)],
/// };
///
/// // Weighted SVD in the `g_r`/`g_c` inner products: the spectrum is
/// // dimensionless, `U` is `g_r`-orthonormal in `⟨Rows, [1,1]⟩` and `V` is
/// // `g_c`-orthonormal in `⟨Cols, [1,1]⟩`.
/// let svd = a.generalized_svd(&g_r, &g_c).expect("positive-definite metrics");
///
/// // Reassembling `M = U Σ Vᴴ g_c` needs the column metric back, and lands in
/// // the original `⟨Rows, Cols⟩` — entry `(0,1)` reads back as `m / (m/s) = s`.
/// let m = svd.recompose(&g_c);
/// let m01: qty!(s) = m.get::<0, 1>();
/// # let _ = m01;
/// ```
///
/// # Generics
///
/// A `MixedUnitMatrix` is usually written over concrete sizes, but its axes can
/// be made generic over both the shape (const generics threaded into
/// nalgebra's [`Const`]) and the dimension lists (e.g.
/// [`Repeated<U, N>`](crate::Repeated) for a uniform `[U; N]` axis). Generic
/// code needs the nalgebra storage-allocator and this crate's list bounds in
/// its `where` clause; rather than spell these out by hand, annotate the item
/// with the [`generic_matrix!`](crate::nalgebra::generic_matrix) attribute
/// (single matrices) or [`generic_block!`](crate::nalgebra::generic_block)
/// (assembled/sliced block grids), which synthesize the whole obligation set
/// from the declared shapes. Add the `decompose` keyword when the body reaches
/// for a reduction-based factorization to also emit the Householder-reduction
/// workspace bounds — but note that on a *mixed* matrix every reduction is now
/// either metric-supplied (`generalized_svd` / `generalized_col_piv_qr` / …) or an
/// endomorphism spectrum (`eigenvalues`), and each needs the metric's or diagonal's
/// own list facts, which are only nameable at a concrete shape; a
/// dimension-list-generic body therefore reaches for the metric-free direct
/// operations below (or erases to the uniform twin, whose single-unit `decompose`
/// path is shown on [`UniformUnitMatrix`]).
///
/// ```
/// use whippyalgebra::nalgebra::{generic_matrix, Const, MixedUnitMatrix, OMatrix};
/// use whippyalgebra::Repeated;
/// use whippyunits::unit;
///
/// type State = unit!(m);
///
/// // Generic over the shape `N` and the (uniform) dimension list. The attribute
/// // detects the square shape and synthesizes the whole LU/`try_inverse`
/// // obligation set from it — the body carries no hand-written bound.
/// #[generic_matrix(rows(N, [State]), cols(N, [State]))]
/// fn is_invertible<const N: usize>(
///     m: MixedUnitMatrix<Repeated<State, N>, Repeated<State, N>, OMatrix<f64, Const<N>, Const<N>>>,
/// ) -> bool {
///     // A direct (metric-free) operation: it keeps the row/column units opaque,
///     // so no repeated-list fact is needed.
///     m.try_inverse().is_some()
/// }
/// # let _ = is_invertible::<3>;
/// ```
pub struct MixedUnitMatrix<RowDims, ColDims, M, Brand = ()> {
    pub(crate) inner: M,
    _dims: PhantomData<fn() -> (RowDims, ColDims, Brand)>,
}

// The dimension vectors and brand are purely phantom (carried behind a
// `fn() -> …` pointer, which is always `Copy`/`Send`/`Sync`), so the standard
// value traits depend on the underlying matrix `M` alone. We implement them by
// hand rather than `#[derive]` because a derive would spuriously constrain
// `RowDims`/`ColDims`/`Brand` to also implement each trait.

impl<RowDims, ColDims, M: Clone, Brand> Clone for MixedUnitMatrix<RowDims, ColDims, M, Brand> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _dims: PhantomData,
        }
    }
}

impl<RowDims, ColDims, M: Copy, Brand> Copy for MixedUnitMatrix<RowDims, ColDims, M, Brand> {}

impl<RowDims, ColDims, M: PartialEq, Brand> PartialEq
    for MixedUnitMatrix<RowDims, ColDims, M, Brand>
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<RowDims, ColDims, M: Eq, Brand> Eq for MixedUnitMatrix<RowDims, ColDims, M, Brand> {}

impl<RowDims, ColDims, M: Hash, Brand> Hash for MixedUnitMatrix<RowDims, ColDims, M, Brand> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

impl<RowDims, ColDims, M: Default, Brand> Default for MixedUnitMatrix<RowDims, ColDims, M, Brand> {
    fn default() -> Self {
        Self {
            inner: M::default(),
            _dims: PhantomData,
        }
    }
}

impl<RowDims, ColDims, M: fmt::Debug, Brand> fmt::Debug
    for MixedUnitMatrix<RowDims, ColDims, M, Brand>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("MixedUnitMatrix").field(&self.inner).finish()
    }
}

/// A unit-safe column vector: an `n x 1` [`MixedUnitMatrix`] with row (output)
/// units `RowDims` and a single column (input) unit `ColUnit`. Entry `i` has
/// unit `RowDims[i] / ColUnit`.
///
/// `ColUnit` splits a vector's units between its input (the column) and output
/// (the rows): a gain vector mapping an `m` input to `V` outputs is
/// `RowDims = [V, …]` with `ColUnit = m`. Use
/// [`Dimensionless`](crate::Dimensionless) for `ColUnit` to get the plain index
/// vector whose entries are exactly `RowDims`, or
/// [`MixedUnitMatrix::canonical_at_row`]/[`canonical_at_col`](MixedUnitMatrix::canonical_at_col)
/// to pin a different gauge.
///
/// See [`UnitRowVector`] for the `1 x n` (row) dual.
pub type UnitVector<RowDims, ColUnit, M, Brand = ()> =
    MixedUnitMatrix<RowDims, DCons<ColUnit, DNil>, M, Brand>;

/// A unit-safe row vector: the `1 x n` dual of [`UnitVector`], with a single row
/// (output) unit `RowUnit` and column (input) units `ColDims`. Entry `j` has
/// unit `RowUnit / ColDims[j]`.
///
/// Use [`Dimensionless`](crate::Dimensionless) for `RowUnit` to get the index
/// form whose entries are the reciprocals of `ColDims`;
/// [`canonical_at_row::<0>`](MixedUnitMatrix::canonical_at_row) regauges to it.
///
/// A row vector is the natural type for the transpose of a [`UnitVector`], and
/// for covectors/gradients that consume a vector and return a scalar.
pub type UnitRowVector<RowUnit, ColDims, M, Brand = ()> =
    MixedUnitMatrix<DCons<RowUnit, DNil>, ColDims, M, Brand>;

// Block concatenation. `hcat` glues two matrices side by side and `vcat` stacks
// them; together they assemble a partitioned matrix from typed pieces (see the
// [`block_matrix!`](crate::nalgebra::block_matrix) macro). The dimensional coherence falls
// straight out of the entry-unit rule:
//
//   - `hcat` places blocks in the same rows, so they must share `RowDims`
//     (unification enforces it); the result's columns are the two column lists
//     concatenated.
//   - `vcat` places blocks in the same columns, so they must share `ColDims`;
//     the result's rows are the two row lists concatenated.
//
// A block whose spaces don't line up with its neighbours therefore fails to
// compile. The numeric shape of the result is computed with nalgebra's
// [`DimAdd`](nalgebra::DimAdd), so the storage tracks statically whenever it can:
// concatenating two `Const` dimensions yields another `Const` (via nalgebra's
// `Const`↔`typenum` bridge, valid up to its 127 cap), while any `Dyn` operand
// makes the joined axis `Dyn`. Static blocks therefore assemble into an owned
// `SMatrix`; dynamic ones fall back to a `DMatrix` — automatically, with no mode
// to choose. (The bridge obligation is a plain `where` bound, so it propagates
// through generic code exactly like [`ShapeIndex`](crate::ShapeIndex).)

/// Counts a dimension list to a [`nalgebra::Dim`] — the type-level length as a
/// [`Const`], the value-free analogue of reading
/// [`DimList::LEN`](crate::DimList::LEN) at runtime.
///
/// It is built by walking the list and adding one at each cons via nalgebra's
/// [`DimAdd`](nalgebra::DimAdd) (`Const<k> + Const<1> = Const<k+1>`), so the
/// length lands in type position without `generic_const_exprs`. This is what
/// lets a block's static size (e.g. a [`zeros!`](crate::nalgebra::zeros) cell, or a
/// concatenation result) be an `SMatrix` rather than a runtime-sized `DMatrix`.
/// Like the rest of the `Const`↔`typenum` bridge it caps at 127.
pub trait CountDim {
    /// The list's length as a `Const` dimension.
    type Dim: nalgebra::Dim;
}

impl CountDim for DNil {
    type Dim = nalgebra::Const<0>;
}

impl<H, T> CountDim for DCons<H, T>
where
    T: CountDim,
    <T as CountDim>::Dim: nalgebra::DimAdd<nalgebra::Const<1>>,
{
    type Dim = nalgebra::DimSum<<T as CountDim>::Dim, nalgebra::Const<1>>;
}

/// The length of dimension list `L` as a [`nalgebra::Dim`] (a `Const`).
pub type CountedDim<L> = <L as CountDim>::Dim;

/// A square nalgebra dimension that is well-formed for whippyalgebra's generic
/// square-matrix operations — bundling the obligations a generic `N × N` shape
/// must satisfy into one bound so they need not be transcribed separately:
///
/// - [`ShapeIndex`] — the shape indexes and slices dimension lists (element
///   access, [`block`](MixedUnitMatrix::block) extraction);
/// - `DimMin<Self, Output = Self>` — the self-minimization that
///   [`solve`](MixedUnitMatrix::solve) / [`determinant`](MixedUnitMatrix::determinant)
///   / [`try_inverse`](MixedUnitMatrix::try_inverse) route through nalgebra's LU;
/// - [`DimName`](nalgebra::DimName) — it names a compile-time size.
///
/// Every concrete `nalgebra::Const<N>` (N ≤ 127) satisfies it automatically, so
/// it only needs stating when a function is generic over `const N: usize` and
/// uses `N × N` matrices — where it replaces the three bounds above with
/// `nalgebra::Const<N>: SquareDim`.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a well-formed square dimension for whippyalgebra's matrix ops",
    label = "not usable as an `N × N` matrix dimension here",
    note = "`SquareDim` bundles `ShapeIndex` (indexing/slicing), \
            `DimMin<Self, Output = Self>` (LU-backed solve/determinant/inverse), \
            and `DimName` (static size). Every concrete `Const<N>` (N ≤ 127) has \
            it; it only needs stating for a function generic over `const N: usize`.",
    note = "Fix: add `nalgebra::Const<N>: SquareDim` to the enclosing `where` clause."
)]
pub trait SquareDim:
    ShapeIndex + nalgebra::DimName + nalgebra::DimMin<Self, Output = Self>
{
}

impl<D> SquareDim for D where D: ShapeIndex + nalgebra::DimName + nalgebra::DimMin<D, Output = D> {}
