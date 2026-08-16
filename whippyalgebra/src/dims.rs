//! Type-level dimension vectors.
//!
//! A unit-safe matrix carries two type-level lists describing its row
//! (output-space) and column (input-space) dimensions. The unit of entry
//! `(i, j)` is `row_i / col_j`, so the representable matrices are exactly the
//! "dimensionally homogeneous" ones (rank-1 in log space) — precisely the
//! physically meaningful set.
//!
//! Each list entry is a [`Unit`], carrying only the unit's scale/prefix and
//! dimension (no storage type or brand — those are stored once on the matrix
//! wrapper). `Unit` and its value-free algebra (`UnitMul` / `UnitDiv` /
//! `UnitInv`) live in whippyunits, so unit arithmetic is defined in exactly one
//! place. A `Unit` reifies back into a `Quantity` trivially, as
//! `Quantity<Unit, T, Brand>`.

use core::marker::PhantomData;

use whippyunits::api::aggregate_scale_factor_float;
use whippyunits::quantity::{_2, _3, _5, _A, _I, _J, _L, _M, _N, _Pi, _T, _Θ, Dimension, Scale};
pub use whippyunits::unit::{HasUnit, Unit};
use whippyunits::{MulUnit, UnitDiv, UnitInv, UnitMul};

/// The empty dimension list.
pub struct DNil;

/// A cons cell: `H` is a [`Unit`] describing one entry, `Tail` is the rest of
/// the list.
pub struct DCons<H, Tail>(PhantomData<(H, Tail)>);

/// The dimensionless unit entry.
///
/// Passed as the `ColUnit` of a [`UnitVector`](crate::nalgebra::UnitVector) to recover the
/// plain "index" vector whose entries are exactly its row units.
pub type Dimensionless = whippyunits::unit!(1);

/// The compile-time length of a dimension list.
pub trait DimList {
    /// Number of entries in the list.
    const LEN: usize;
}

impl DimList for DNil {
    const LEN: usize = 0;
}

impl<H, Tail: DimList> DimList for DCons<H, Tail> {
    const LEN: usize = 1 + Tail::LEN;
}

/// Type-level concatenation of two dimension lists.
///
/// This is the primitive for building block-heterogeneous vectors: a state
/// vector with a position block followed by a velocity block is just the
/// concatenation of the two block dimension lists.
pub trait Concat<Other> {
    /// The concatenated list.
    type Out;
}

impl<Other> Concat<Other> for DNil {
    type Out = Other;
}

impl<H, Tail, Other> Concat<Other> for DCons<H, Tail>
where
    Tail: Concat<Other>,
{
    type Out = DCons<H, <Tail as Concat<Other>>::Out>;
}

/// Convenience alias for the concatenation of two blocks `A` and `B`.
pub type Concatenated<A, B> = <A as Concat<B>>::Out;

// ---------------------------------------------------------------------------
// Type-level unit algebra over dimension lists
// ---------------------------------------------------------------------------
//
// Many matrix operations transform the metadata by applying a fixed unit
// function to every entry of a dimension list (e.g. transpose reciprocates each
// entry; scalar-by-quantity multiplication multiplies each row entry by the
// scalar's unit). Rather than special-casing each, we express the transform as
// an [`ApplyUnit`] functor and [`MapUnits`] it over the list.
//
// The functors delegate directly to whippyunits' value-free `Unit` algebra, so
// there are no `Quantity` placeholders to construct: a `Unit` is exactly the
// scale/dimension pair these operations act on.

/// A type-level function from one [`Unit`] to another.
///
/// Implemented by the functor marker types ([`Reciprocal`], [`MulBy`],
/// [`DivBy`]) and consumed by [`MapUnits`].
pub trait ApplyUnit<U> {
    /// The transformed unit.
    type Out;
}

/// Applies an [`ApplyUnit`] functor `F` to every entry of a dimension list.
pub trait MapUnits<F> {
    /// The mapped dimension list.
    type Out;
}

impl<F> MapUnits<F> for DNil {
    type Out = DNil;
}

impl<F, H, Tail> MapUnits<F> for DCons<H, Tail>
where
    F: ApplyUnit<H>,
    Tail: MapUnits<F>,
{
    type Out = DCons<<F as ApplyUnit<H>>::Out, <Tail as MapUnits<F>>::Out>;
}

/// The dimension list `L` with functor `F` applied to every entry.
pub type Mapped<F, L> = <L as MapUnits<F>>::Out;

/// Functor computing the reciprocal `1 / U` of each unit.
///
/// Used by matrix transpose: `Mᵀ(i, j) = M(j, i)` has unit `row_j / col_i`,
/// which forces the transposed lists to be the reciprocals of the originals.
pub struct Reciprocal;

impl<U> ApplyUnit<U> for Reciprocal
where
    U: UnitInv,
{
    type Out = <U as UnitInv>::Output;
}

/// Functor multiplying each unit by `V`.
///
/// Used by scalar-by-quantity multiplication, which scales every row entry's
/// unit by the scalar quantity's unit.
pub struct MulBy<V>(PhantomData<V>);

impl<V, U> ApplyUnit<U> for MulBy<V>
where
    U: UnitMul<V>,
{
    type Out = <U as UnitMul<V>>::Output;
}

/// Functor dividing each unit by `V`.
///
/// Used by scalar-by-quantity division.
pub struct DivBy<V>(PhantomData<V>);

impl<V, U> ApplyUnit<U> for DivBy<V>
where
    U: UnitDiv<V>,
{
    type Out = <U as UnitDiv<V>>::Output;
}

/// Functor collapsing every unit to [`Dimensionless`], preserving list length.
///
/// Used by the [Cholesky factor](crate::nalgebra::MixedUnitMatrix::cholesky): the pivot
/// (column) space of `L` in `M = L Lᵀ` is forced dimensionless (the `Lᵀ`
/// contraction requires `C_L = 1/C_L`), so `L`'s column list is `RowDims`
/// mapped through this functor — an all-dimensionless list of the right length.
pub struct ToDimensionless;

impl<U> ApplyUnit<U> for ToDimensionless {
    type Out = Dimensionless;
}

/// Element-wise type-level division of two equal-length dimension lists: entry
/// `i` of the result is `Self[i] / Other[i]`.
///
/// Used by [`diagonal`](crate::nalgebra::MixedUnitMatrix::diagonal), whose `i`th entry is
/// `M(i, i)` with unit `RowDims[i] / ColDims[i]`. (Lists of unequal length have
/// no `ZipDiv` impl, so a diagonal of mismatched blocks is a compile error.)
pub trait ZipDiv<Other> {
    /// The element-wise quotient list.
    type Out;
}

impl ZipDiv<DNil> for DNil {
    type Out = DNil;
}

impl<H1, T1, H2, T2> ZipDiv<DCons<H2, T2>> for DCons<H1, T1>
where
    H1: UnitDiv<H2>,
    T1: ZipDiv<T2>,
{
    type Out = DCons<<H1 as UnitDiv<H2>>::Output, <T1 as ZipDiv<T2>>::Out>;
}

/// The element-wise quotient `A[i] / B[i]` of two equal-length lists.
pub type ZipDivided<A, B> = <A as ZipDiv<B>>::Out;

/// The dimensionless pivot axis shared by the two factors of a rectangular thin
/// factorization — the `Q`/`R` of a
/// [generalized QR](crate::nalgebra::MixedUnitMatrix::generalized_qr), the `U`/`V`
/// of a [generalized SVD](crate::nalgebra::MixedUnitMatrix::generalized_svd) or
/// [generalized bidiagonalization](crate::nalgebra::MixedUnitMatrix::generalized_bidiagonalize):
/// the all-[`Dimensionless`] list of length `min(len(Self), len(Other))`.
///
/// nalgebra's thin factorizations contract on an axis of length `min(m, n)`
/// whose coordinates carry no unit (see the [decompositions
/// guide](crate::nalgebra::MixedUnitMatrix#decompositions)). Zipping the row list against the
/// column list down to the shorter length and collapsing every surviving entry
/// to [`Dimensionless`] produces exactly that pivot, so `Q : ⟨RowDims, pivot⟩`
/// and `R : ⟨pivot, ColDims⟩` contract on it to rebuild `⟨RowDims, ColDims⟩`.
///
/// For a square pair (equal lengths) it coincides with
/// [`Mapped<ToDimensionless, _>`](Mapped). Unequal lengths stop at the shorter
/// list, so one impl covers both the tall (`m ≥ n`) and wide (`m < n`) cases.
pub trait ZipToDimensionless<Other> {
    /// The dimensionless list of length `min(len(Self), len(Other))`.
    type Out;
}

// A `DNil` on either side caps the length: `min(0, k) = 0`.
impl<Other> ZipToDimensionless<Other> for DNil {
    type Out = DNil;
}

impl<H, T> ZipToDimensionless<DNil> for DCons<H, T> {
    type Out = DNil;
}

impl<H1, T1, H2, T2> ZipToDimensionless<DCons<H2, T2>> for DCons<H1, T1>
where
    T1: ZipToDimensionless<T2>,
{
    type Out = DCons<Dimensionless, <T1 as ZipToDimensionless<T2>>::Out>;
}

/// The dimensionless pivot list of length `min(len(Row), len(Col))` — see
/// [`ZipToDimensionless`].
pub type PivotDims<Row, Col> = <Row as ZipToDimensionless<Col>>::Out;

/// Type-level witness that a matrix is a uniform endomorphism: every diagonal
/// entry `RowDims[i] / ColDims[i]` is the same unit `U`.
///
/// This is the condition under which the eigenvalues of a (count-)square matrix
/// are well-typed but not dimensionless. They all carry the shared diagonal unit
/// `U` (e.g. a continuous state matrix `⟨C/t, C⟩` has `U = 1/t`, so its spectrum
/// is the poles, in `1/time`).
///
/// The dimensionless-endomorphism case (`RowDims = ColDims`) is the special
/// instance `U = `[`Dimensionless`]. Lists of unequal length, or a diagonal
/// whose ratios disagree, have no impl (a compile error).
// NOTE: this `on_unimplemented` documents intent but does *not* currently reach
// the user: a non-uniform diagonal fails via `UnitDiv`'s associated-type
// equality deep in the recursive impl, so rustc reports that leaf (E0271/E0599)
// rather than "`UniformDiag` unimplemented". The taxonomy in the
// `MixedUnitMatrix` "Decompositions" section is the real signpost.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not an endomorphism, so its eigenvalues are not well-typed",
    label = "the diagonal units `RowDims[i] / ColDims[i]` are not all one shared unit",
    note = "Eigenvalues solve `det(A − λI) = 0`, which forces every `λ` to share one diagonal unit; a mixed diagonal has none.",
    note = "If this is a continuous state matrix `⟨C/t, C⟩`, scale to the endomorphism `A * dt : ⟨C, C⟩` first, then take eigenvalues.",
    note = "See the decomposition taxonomy: `whippyalgebra::nalgebra::decompositions`."
)]
pub trait UniformDiag<ColDims> {
    /// The shared diagonal unit `U = RowDims[i] / ColDims[i]`.
    type Unit;
}

// Single entry: the diagonal unit is just this pair's ratio, with nothing to
// agree with.
impl<R, C> UniformDiag<DCons<C, DNil>> for DCons<R, DNil>
where
    R: UnitDiv<C>,
{
    type Unit = <R as UnitDiv<C>>::Output;
}

// Two or more: this head pair's ratio must *equal* the shared unit of the tail
// (enforced by the `Output =` bound), and that common unit is propagated up.
impl<R, RH, RT, C, CH, CT> UniformDiag<DCons<C, DCons<CH, CT>>> for DCons<R, DCons<RH, RT>>
where
    DCons<RH, RT>: UniformDiag<DCons<CH, CT>>,
    R: UnitDiv<C, Output = <DCons<RH, RT> as UniformDiag<DCons<CH, CT>>>::Unit>,
{
    type Unit = <DCons<RH, RT> as UniformDiag<DCons<CH, CT>>>::Unit;
}

/// The shared diagonal unit of a uniform endomorphism `⟨RowDims, ColDims⟩`.
pub type DiagUnit<RowDims, ColDims> = <RowDims as UniformDiag<ColDims>>::Unit;

/// Type-level witness that a matrix `⟨RowDims, ColDims⟩` is a metric: its
/// columns are the reciprocals of its rows (`ColDims = 1 / RowDims`), so the
/// matrix equals its own transpose as a type — a quadratic form / metric
/// tensor.
///
/// This is the shape precondition for the decompositions that only make sense on
/// a self-transpose matrix: [`cholesky`](crate::nalgebra::MixedUnitMatrix::cholesky),
/// [`symmetric_eigen`](crate::nalgebra::MixedUnitMatrix::symmetric_eigen), and the
/// metric [`generalized_symmetric_eigen`](crate::nalgebra::MixedUnitMatrix::generalized_symmetric_eigen).
/// Symmetry of the numbers is not enough: a numerically symmetric matrix of the
/// wrong dimensional shape is rejected, because its transpose would carry
/// different units. The impl holds exactly when `ColDims = 1 / RowDims`; any
/// other pairing has no impl (a compile error).
// NOTE: like `UniformDiag`, this `on_unimplemented` text does not currently
// surface — checking `ColDims = 1 / RowDims` normalizes a projection and rustc
// reports the leaf mismatch (E0271). What this trait *does* buy is a readable
// name in the "required by this bound" note (vs. a raw `MapUnits<…, Out = …>`),
// which points a reader here and at the `MixedUnitMatrix` "Decompositions"
// section.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a metric: its columns are not `1 / rows`",
    label = "expected `ColDims = 1 / RowDims` (a self-transpose matrix)",
    note = "`cholesky`, `symmetric_eigen`, and the metric generalized eigenproblem need a metric `⟨R, 1/R⟩` — symmetry of the *numbers* is not symmetry of the *units*.",
    note = "For a non-metric matrix use `qr` (or `lu`); for eigenvalues of an endomorphism see `eigenvalues`.",
    note = "See the decomposition taxonomy: `whippyalgebra::nalgebra::decompositions`."
)]
pub trait MetricShape<ColDims> {}

// The reciprocal lives in the *impl head* (the `ColDims` slot), not in a
// `where … Out = ColDims` clause. That is deliberate: a non-metric `ColDims`
// then fails to *unify* with `1 / RowDims`, so the impl simply does not apply
// and `RowDims: MetricShape<ColDims>` is genuinely unimplemented — which is what
// lets the `on_unimplemented` message above fire. Phrased as a projection
// equality instead, the compiler would drill into the leaf mismatch and bypass
// the custom diagnostic.
impl<RowDims> MetricShape<Mapped<Reciprocal, RowDims>> for RowDims where
    RowDims: MapUnits<Reciprocal>
{
}

/// Element-wise type-level multiplication of two equal-length dimension lists:
/// entry `i` of the result is `Self[i] * Other[i]`.
///
/// Used by [`component_mul`](crate::nalgebra::MixedUnitMatrix::component_mul): the Hadamard
/// product's entry `(i, j)` has unit `(Ar_i · Br_i) / (Ac_j · Bc_j)`, i.e. the
/// element-wise product of the two operands' row lists over that of their column
/// lists. (Unequal-length lists have no impl, so a shape mismatch is a compile
/// error.)
pub trait ZipMul<Other> {
    /// The element-wise product list.
    type Out;
}

impl ZipMul<DNil> for DNil {
    type Out = DNil;
}

impl<H1, T1, H2, T2> ZipMul<DCons<H2, T2>> for DCons<H1, T1>
where
    H1: UnitMul<H2>,
    T1: ZipMul<T2>,
{
    type Out = DCons<<H1 as UnitMul<H2>>::Output, <T1 as ZipMul<T2>>::Out>;
}

/// The element-wise product `A[i] * B[i]` of two equal-length lists.
pub type ZipMulled<A, B> = <A as ZipMul<B>>::Out;

/// Type-level "outer product" of two dimension lists: for a `Self` of length
/// `m` and `Other` of length `n`, the result has length `m * n`, where entry
/// `i * n + k` is `Self[i] * Other[k]`.
///
/// Used by the [Kronecker product](crate::nalgebra::MixedUnitMatrix::kronecker), whose
/// row index `i1 * rows2 + i2` carries the unit `RowDims_A[i1] * RowDims_B[i2]`
/// (and likewise down the columns), so both dimension lists of the result are
/// the outer product of the operands'. The ordering — the whole of `Other`
/// scaled by each entry of `Self`, in turn — matches nalgebra's block layout.
pub trait CrossMul<Other> {
    /// The outer-product list.
    type Out;
}

impl<Other> CrossMul<Other> for DNil {
    type Out = DNil;
}

impl<H, Tail, Other> CrossMul<Other> for DCons<H, Tail>
where
    Other: MapUnits<MulBy<H>>,
    Tail: CrossMul<Other>,
    Mapped<MulBy<H>, Other>: Concat<<Tail as CrossMul<Other>>::Out>,
{
    type Out = Concatenated<Mapped<MulBy<H>, Other>, <Tail as CrossMul<Other>>::Out>;
}

/// The outer product of two lists: length `m * n`, entry `i * n + k` is
/// `A[i] * B[k]`.
pub type CrossMulled<A, B> = <A as CrossMul<B>>::Out;

/// Type-level product of every entry of a dimension list, folding [`UnitMul`]
/// from [`Dimensionless`] (the empty list is `Dimensionless`).
///
/// Used by [`determinant`](crate::nalgebra::MixedUnitMatrix::determinant): every term of
/// its expansion selects one entry from each row and column, so all terms carry
/// the same unit `∏ RowDims / ∏ ColDims`. The determinant therefore has the
/// single unit `Producted<RowDims> / Producted<ColDims>`.
pub trait Product {
    /// The product of all entries (`Dimensionless` for the empty list).
    type Out;
}

impl Product for DNil {
    type Out = Dimensionless;
}

impl<Head, Tail> Product for DCons<Head, Tail>
where
    Tail: Product,
    Head: UnitMul<<Tail as Product>::Out>,
{
    type Out = MulUnit<Head, <Tail as Product>::Out>;
}

/// The product of every entry of `L` (`Dimensionless` if empty).
pub type Producted<L> = <L as Product>::Out;

// ---------------------------------------------------------------------------
// Rescaling: element-wise scale conversion of a whole dimension list
// ---------------------------------------------------------------------------
//
// whippyunits' `rescale` converts a single quantity to a different *scale* of
// the *same* dimension, multiplying the value by the ratio of the two
// magnitudes. These traits lift that to a dimension list, so a matrix can be
// rescaled along all of its rows (and all of its columns) at once: each list
// entry is rescaled independently, and the runtime factors are materialized
// from the type level.

/// Type-level witness that the unit `Self` can be rescaled to `Target`: both
/// carry the same dimension, differing only in scale (prefix). This is the
/// per-entry precondition of a matrix rescale — a target whose dimension does
/// not match has no impl, which is the compile error you are reading.
///
/// [`FACTOR`](UnitRescale::FACTOR) is the multiplicative conversion factor that
/// takes a value expressed in `Self` to the equal value expressed in `Target`,
/// exactly whippyunits' `rescale` factor for a single quantity.
pub trait UnitRescale<Target> {
    /// The conversion factor `Self -> Target` (multiply a `Self`-valued number
    /// by this to reexpress it in `Target`).
    const FACTOR: f64;
}

// The dimension exponents are shared const generics across both the `Self` and
// `Target` positions, so this impl unifies *only* when the two units have
// identical dimensions; the scale exponents are free, so any scale-to-scale
// conversion within one dimension is admitted. The factor is the product of
// prime powers `∏ p^(from − to)`, i.e. whippyunits' own float scale factor.
impl<
    const P2_FROM: i16,
    const P3_FROM: i16,
    const P5_FROM: i16,
    const PI_FROM: i16,
    const P2_TO: i16,
    const P3_TO: i16,
    const P5_TO: i16,
    const PI_TO: i16,
    const MASS: i16,
    const LEN: i16,
    const TIME: i16,
    const CUR: i16,
    const TEMP: i16,
    const AMT: i16,
    const LUM: i16,
    const ANG: i16,
>
    UnitRescale<
        Unit<
            Scale<_2<P2_TO>, _3<P3_TO>, _5<P5_TO>, _Pi<PI_TO>>,
            Dimension<_M<MASS>, _L<LEN>, _T<TIME>, _I<CUR>, _Θ<TEMP>, _N<AMT>, _J<LUM>, _A<ANG>>,
        >,
    >
    for Unit<
        Scale<_2<P2_FROM>, _3<P3_FROM>, _5<P5_FROM>, _Pi<PI_FROM>>,
        Dimension<_M<MASS>, _L<LEN>, _T<TIME>, _I<CUR>, _Θ<TEMP>, _N<AMT>, _J<LUM>, _A<ANG>>,
    >
{
    const FACTOR: f64 = aggregate_scale_factor_float(
        P2_FROM, P3_FROM, P5_FROM, PI_FROM, P2_TO, P3_TO, P5_TO, PI_TO,
    );
}

/// Element-wise rescale of a whole dimension list to `Target`: entry `i` of
/// `Self` is rescaled to entry `i` of `Target` ([`UnitRescale`]). The two lists
/// must therefore have equal length and matching dimensions element-wise; any
/// length or dimension mismatch has no impl (a compile error).
///
/// It materializes the per-entry factors at runtime via
/// [`write_factors`](RescaleFactors::write_factors) — the bridge a matrix
/// rescale uses to turn the static row/column dimension change into concrete
/// per-row and per-column multipliers.
pub trait RescaleFactors<Target> {
    /// Writes `factor(Self[i] -> Target[i])` into `out[i]` for every entry.
    ///
    /// `out` must be exactly as long as the list (`DimList::LEN`); it is indexed
    /// head-first, one factor per entry.
    fn write_factors(out: &mut [f64]);
}

impl RescaleFactors<DNil> for DNil {
    fn write_factors(_out: &mut [f64]) {}
}

impl<HSelf, TSelf, HTarget, TTarget> RescaleFactors<DCons<HTarget, TTarget>> for DCons<HSelf, TSelf>
where
    HSelf: UnitRescale<HTarget>,
    TSelf: RescaleFactors<TTarget>,
{
    fn write_factors(out: &mut [f64]) {
        out[0] = <HSelf as UnitRescale<HTarget>>::FACTOR;
        <TSelf as RescaleFactors<TTarget>>::write_factors(&mut out[1..]);
    }
}

/// Builds a dimension list from comma-separated whippyunits unit expressions.
///
/// Each entry is a unit literal expression as accepted by `whippyunits::unit!`;
/// only its scale and dimension are kept (as a [`Unit`], via [`HasUnit`]).
/// Storage type and brand are not specified here — they are stored once on the
/// matrix wrapper.
///
/// Compound units containing `/`, `*`, or `^` parse unambiguously because the
/// commas are the only separators (unit expressions never contain commas):
///
/// ```ignore
/// // [L, L/T, L/T^2, V]
/// type Dims = dims![m, m / s, m / s^2, V];
/// ```
///
/// An empty invocation `dims![]` yields [`DNil`]. A trailing comma is allowed.
#[macro_export]
macro_rules! dims {
    () => {
        $crate::dims::DNil
    };

    // Internal: nothing left to munch (e.g. after a trailing comma).
    (@munch []) => {
        $crate::dims::DNil
    };

    // Internal: a comma flushes the accumulated unit expression and continues.
    (@munch [$($cur:tt)+] , $($rest:tt)*) => {
        $crate::dims::DCons<
            $crate::__reexport::whippyunits::unit!($($cur)+),
            $crate::dims!(@munch [] $($rest)*)
        >
    };

    // Internal: end of input flushes the final accumulated unit expression.
    (@munch [$($cur:tt)+]) => {
        $crate::dims::DCons<
            $crate::__reexport::whippyunits::unit!($($cur)+),
            $crate::dims::DNil
        >
    };

    // Internal: munch one more token into the current accumulator.
    (@munch [$($cur:tt)*] $head:tt $($rest:tt)*) => {
        $crate::dims!(@munch [$($cur)* $head] $($rest)*)
    };

    // Entry point: start munching with an empty accumulator.
    ($($all:tt)+) => {
        $crate::dims!(@munch [] $($all)+)
    };
}
