#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use core::marker::PhantomData;

use whippyunits::{DivUnit, UnitDiv};

use super::{CountDim, CountedDim, MixedUnitMatrix, UnitRowVector, UnitVector};
use crate::dims::{
    Concat, Concatenated, DCons, DNil, DimList, Dimensionless, DivBy, MapUnits, Mapped,
    ToDimensionless, ZipDiv, ZipDivided,
};
use crate::entry::{ColUnitOf, EntryUnit, EntryUnitOf, FromRaw, IntoEntry, RowUnitOf};
use crate::index::{Drop, ElemAt, Nat, ShapeIndex, Sliced, SlicedAt, Take, Unsigned};
use crate::nalgebra::reduce::{AutoReduce, Reduced};
use crate::nalgebra::uniform::UniformUnitMatrix;
use crate::uniformity::{CollapseUniform, Uniform};
use nalgebra::Const;

impl<RowDims, ColDims, Brand, M> MixedUnitMatrix<RowDims, ColDims, M, Brand> {
    /// Wraps an underlying nalgebra matrix without checking that its shape
    /// matches the dimension-vector lengths.
    pub fn from_nalgebra(inner: M) -> Self {
        Self {
            inner,
            _dims: PhantomData,
        }
    }

    /// Consumes the wrapper, returning the underlying nalgebra matrix.
    pub fn into_nalgebra(self) -> M {
        self.inner
    }

    /// Borrows the underlying nalgebra matrix.
    pub fn nalgebra(&self) -> &M {
        &self.inner
    }

    /// Consumes a uniform matrix and re-tags it as a [`UniformUnitMatrix`],
    /// erasing the gauge (zero-copy: the storage is moved, not cloned).
    ///
    /// Compiles only if both dimension lists are uniform (every `RowDims` entry
    /// the same `Ru`, every `ColDims` entry the same `Cu`), so every entry has
    /// the single unit `Ru / Cu`; a non-uniform matrix has no
    /// [`CollapseUniform`] impl and is a compile error.
    ///
    /// The collapse keeps only the quotient `Ru / Cu`, dropping the specific
    /// `(RowDims, ColDims)`, so it is one-way: re-expanding must invent a gauge
    /// back (e.g. via [`gauge!`](crate::nalgebra::gauge) inside `block_matrix!`).
    /// To keep the gauge for later recomposition, use
    /// [`to_uniform`](Self::to_uniform) (owned copy) or
    /// [`as_uniform`](Self::as_uniform) (borrowing view) instead, both of which
    /// leave `self` intact.
    pub fn into_uniform(
        self,
    ) -> UniformUnitMatrix<DivUnit<Uniform<RowDims>, Uniform<ColDims>>, M, Brand>
    where
        RowDims: CollapseUniform,
        ColDims: CollapseUniform,
        Uniform<RowDims>: UnitDiv<Uniform<ColDims>>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner)
    }
}

// Non-consuming reinterpretation of a matrix that *happens to be uniform* as a
// `UniformUnitMatrix`. Both erase the gauge (only the shared entry unit
// `Ru / Cu` survives), but — unlike `into_uniform` — they leave `self` intact,
// so the gauge-carrying mixed matrix stays available for later recomposition
// (e.g. reassembling a block layout with its exact `RowDims`/`ColDims`). Use
// these to borrow/copy a block *as uniform* for the bulk unit-safe ops, while
// keeping the mixed record around.
impl<RowDims, ColDims, Brand, T, R, C, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, S>, Brand>
where
    RowDims: CollapseUniform,
    ColDims: CollapseUniform,
    Uniform<RowDims>: UnitDiv<Uniform<ColDims>>,
    T: nalgebra::Scalar,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::RawStorage<T, R, C>,
{
    /// Copies this uniform matrix into an owned [`UniformUnitMatrix`], erasing
    /// the gauge, and leaves `self` untouched — so the mixed matrix and its
    /// gauge stay available for recomposition.
    ///
    /// For a zero-copy borrow instead of a copy see
    /// [`as_uniform`](Self::as_uniform); to consume and reuse the storage see
    /// [`into_uniform`](Self::into_uniform).
    pub fn to_uniform(
        &self,
    ) -> UniformUnitMatrix<
        DivUnit<Uniform<RowDims>, Uniform<ColDims>>,
        nalgebra::OMatrix<T, R, C>,
        Brand,
    >
    where
        S: nalgebra::Storage<T, R, C>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.clone_owned())
    }

    /// Borrows this uniform matrix as a zero-copy [`UniformUnitMatrix`] view,
    /// erasing the gauge; `self` is borrowed (shared) for the view's lifetime and
    /// stays readable meanwhile. Use this to read or run bulk uniform ops on a
    /// block without giving up its gauge for later recomposition.
    ///
    /// For an owned copy see [`to_uniform`](Self::to_uniform).
    pub fn as_uniform(
        &self,
    ) -> UniformUnitMatrix<
        DivUnit<Uniform<RowDims>, Uniform<ColDims>>,
        nalgebra::MatrixView<'_, T, R, C, S::RStride, S::CStride>,
        Brand,
    > {
        UniformUnitMatrix::from_nalgebra(
            self.inner.generic_view((0, 0), self.inner.shape_generic()),
        )
    }

    /// Borrows this uniform matrix as a zero-copy mutable [`UniformUnitMatrix`]
    /// view, erasing the gauge for the duration of the borrow.
    ///
    /// The view mutates entries in place through the uniform API the mixed type
    /// withholds (`get_mut`, `set`, `iter_mut`, and `[(i, j)]` indexing). `self`
    /// is borrowed exclusively, and when the view drops `self` keeps its gauge
    /// (`RowDims`/`ColDims`), so per-element mutation costs nothing needed for
    /// later recomposition.
    pub fn as_uniform_mut(
        &mut self,
    ) -> UniformUnitMatrix<
        DivUnit<Uniform<RowDims>, Uniform<ColDims>>,
        nalgebra::MatrixViewMut<'_, T, R, C, S::RStride, S::CStride>,
        Brand,
    >
    where
        S: nalgebra::RawStorageMut<T, R, C>,
    {
        let shape = self.inner.shape_generic();
        UniformUnitMatrix::from_nalgebra(self.inner.generic_view_mut((0, 0), shape))
    }
}

// Size/shape queries. These are unit-invariant, so they forward directly to the
// underlying nalgebra matrix.
//
// Raw whole-matrix iteration (`iter`/`as_slice`) is deliberately *not* provided
// here: every entry `(i, j)` has its own static unit `RowDims[i] / ColDims[j]`,
// so a runtime scan would have to erase units to a bare `&T`/`&[T]`. Use
// `get`/`get_ref` at compile-time indices for unit-typed access, or drop to
// `nalgebra()` for an explicit unit-erased view. Bulk unit-safe iteration will
// live on the forthcoming `UniformUnitMatrix`, where a single shared entry unit
// makes it sound.
impl<RowDims, ColDims, Brand, T, R, C, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, S>, Brand>
where
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::RawStorage<T, R, C>,
{
    /// The `(rows, cols)` shape of the matrix, matching the `RowDims`/`ColDims`
    /// lengths.
    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    /// The number of rows (equivalently, the length of `RowDims`).
    pub fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    /// The number of columns (equivalently, the length of `ColDims`).
    pub fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    /// The total number of entries (`nrows * ncols`).
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the matrix has no entries.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns `true` if the matrix is square (`nrows == ncols`).
    pub fn is_square(&self) -> bool {
        self.inner.is_square()
    }
}

impl<RowDims, ColDims, M, Brand> MixedUnitMatrix<RowDims, ColDims, M, Brand> {
    /// Regauges to the normal form in which column `J` is dimensionless (see
    /// [`MixedUnitMatrix`]'s gauge discussion).
    ///
    /// Columns sit in the denominator of each entry unit, so dividing both
    /// dimension lists by the column-`J` unit ([`ColUnitOf<ColDims, J>`]) sends
    /// `ColDims[J] → 1` and leaves every entry unit and value unchanged. Since
    /// `+` and `*` require the dimension lists to match exactly, this is how you
    /// line up two matrices that agree only up to gauge: anchoring both on the
    /// same column gives them one shared type, so
    /// `a.canonical_at_col::<J>() + b.canonical_at_col::<J>()` type-checks iff
    /// `a` and `b` have the same entry units.
    ///
    /// See [`canonical_at_row`](Self::canonical_at_row) to pin a row instead, and
    /// [`with_gauge_at_column`](Self::with_gauge_at_column) to pin column `J` to
    /// a chosen unit rather than to dimensionless.
    pub fn canonical_at_col<const J: usize>(
        self,
    ) -> MixedUnitMatrix<
        Mapped<DivBy<ColUnitOf<ColDims, J>>, RowDims>,
        Mapped<DivBy<ColUnitOf<ColDims, J>>, ColDims>,
        M,
        Brand,
    >
    where
        Const<J>: ShapeIndex,
        ColDims: ElemAt<Nat<J>> + MapUnits<DivBy<ColUnitOf<ColDims, J>>>,
        RowDims: MapUnits<DivBy<ColUnitOf<ColDims, J>>>,
    {
        MixedUnitMatrix {
            inner: self.inner,
            _dims: PhantomData,
        }
    }

    /// Regauges to the normal form in which row `I` is dimensionless — the dual
    /// of [`canonical_at_col`](Self::canonical_at_col).
    ///
    /// Rows sit in the numerator of each entry unit, so the divisor is drawn from
    /// `RowDims` ([`RowUnitOf<RowDims, I>`]): dividing both dimension lists by it
    /// sends `RowDims[I] → 1` and leaves entry units and values untouched.
    /// Matrices anchored on the same row share one type, so
    /// `a.canonical_at_row::<I>() + b.canonical_at_row::<I>()` type-checks iff
    /// `a` and `b` have the same entry units. For a row vector (single row),
    /// `I = 0` sends that row to dimensionless. Use
    /// [`with_gauge_at_row`](Self::with_gauge_at_row) to pin row `I` to a chosen
    /// unit rather than to dimensionless.
    pub fn canonical_at_row<const I: usize>(
        self,
    ) -> MixedUnitMatrix<
        Mapped<DivBy<RowUnitOf<RowDims, I>>, RowDims>,
        Mapped<DivBy<RowUnitOf<RowDims, I>>, ColDims>,
        M,
        Brand,
    >
    where
        Const<I>: ShapeIndex,
        RowDims: ElemAt<Nat<I>> + MapUnits<DivBy<RowUnitOf<RowDims, I>>>,
        ColDims: MapUnits<DivBy<RowUnitOf<RowDims, I>>>,
    {
        MixedUnitMatrix {
            inner: self.inner,
            _dims: PhantomData,
        }
    }

    /// Regauges so that row `I` carries exactly the unit `G` — the general form
    /// of [`canonical_at_row`](Self::canonical_at_row) (the `G = 1` case).
    ///
    /// Both dimension lists are divided by `RowDims[I] / G`, so `RowDims[I] → G`
    /// while entry units and values stay put. The target unit comes first so it
    /// reads as "gauge to `G` at row `I`": `m.with_gauge_at_row::<SomeUnit, I>()`.
    pub fn with_gauge_at_row<G, const I: usize>(
        self,
    ) -> MixedUnitMatrix<
        Mapped<DivBy<DivUnit<RowUnitOf<RowDims, I>, G>>, RowDims>,
        Mapped<DivBy<DivUnit<RowUnitOf<RowDims, I>, G>>, ColDims>,
        M,
        Brand,
    >
    where
        Const<I>: ShapeIndex,
        RowDims: ElemAt<Nat<I>>,
        RowUnitOf<RowDims, I>: UnitDiv<G>,
        RowDims: MapUnits<DivBy<DivUnit<RowUnitOf<RowDims, I>, G>>>,
        ColDims: MapUnits<DivBy<DivUnit<RowUnitOf<RowDims, I>, G>>>,
    {
        MixedUnitMatrix {
            inner: self.inner,
            _dims: PhantomData,
        }
    }

    /// Regauges so that column `J` carries exactly the unit `G` — the general
    /// form of [`canonical_at_col`](Self::canonical_at_col) (the `G = 1` case).
    ///
    /// Both dimension lists are divided by `ColDims[J] / G`, so `ColDims[J] → G`
    /// while entry units and values stay put. See
    /// [`with_gauge_at_row`](Self::with_gauge_at_row) for the row dual. The
    /// target unit comes first so it reads as "gauge to `G` at column `J`":
    /// `m.with_gauge_at_column::<SomeUnit, J>()`.
    pub fn with_gauge_at_column<G, const J: usize>(
        self,
    ) -> MixedUnitMatrix<
        Mapped<DivBy<DivUnit<ColUnitOf<ColDims, J>, G>>, RowDims>,
        Mapped<DivBy<DivUnit<ColUnitOf<ColDims, J>, G>>, ColDims>,
        M,
        Brand,
    >
    where
        Const<J>: ShapeIndex,
        ColDims: ElemAt<Nat<J>>,
        ColUnitOf<ColDims, J>: UnitDiv<G>,
        RowDims: MapUnits<DivBy<DivUnit<ColUnitOf<ColDims, J>, G>>>,
        ColDims: MapUnits<DivBy<DivUnit<ColUnitOf<ColDims, J>, G>>>,
    {
        MixedUnitMatrix {
            inner: self.inner,
            _dims: PhantomData,
        }
    }
}

impl<RowDims, ColDims, Brand, T, R, C, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, S>, Brand>
where
    RowDims: DimList,
    ColDims: DimList,
    R: nalgebra::DimName,
    C: nalgebra::DimName,
    S: nalgebra::RawStorage<T, R, C>,
{
    /// Wraps a statically-sized nalgebra matrix, checking that its numeric shape
    /// matches the dimension-list lengths.
    ///
    /// The check is zero-cost: both the numeric shape and the list lengths are
    /// compile-time constants, so a mismatch is a compile error and nothing runs
    /// at runtime.
    ///
    /// For a matrix whose shape is only known at runtime (`Dyn` dimensions), use
    /// [`from_dyn`](Self::from_dyn) instead.
    pub fn new(inner: nalgebra::Matrix<T, R, C, S>) -> Self {
        const {
            assert!(
                RowDims::LEN == R::USIZE,
                "row count does not match RowDims length"
            );
        }
        const {
            assert!(
                ColDims::LEN == C::USIZE,
                "column count does not match ColDims length"
            );
        }
        Self {
            inner,
            _dims: PhantomData,
        }
    }
}

impl<RowDims, ColDims, Brand, T>
    MixedUnitMatrix<
        RowDims,
        ColDims,
        nalgebra::OMatrix<T, CountedDim<RowDims>, CountedDim<ColDims>>,
        Brand,
    >
where
    RowDims: DimList + CountDim,
    ColDims: DimList + CountDim,
    T: nalgebra::Scalar,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<CountedDim<RowDims>, CountedDim<ColDims>>,
{
    /// Materializes a (possibly dynamically-sized) nalgebra matrix into the
    /// canonical static storage, asserting at runtime that its numeric shape
    /// matches the dimension-list lengths.
    ///
    /// This is an interop bridge for when you already hold a `Dyn`-dimensioned
    /// matrix (e.g. a `DMatrix` loaded at runtime) and want to attach a
    /// statically-known unit structure to it; a mixed matrix built any other way
    /// already knows its shape statically, so [`new`](Self::new) checks it for
    /// free. It copies into the canonical `SMatrix` (an `OMatrix` sized by
    /// [`CountedDim`]) — a one-time O(n²) copy,
    /// dominated by the O(n³) routines that typically follow.
    #[track_caller]
    pub fn from_dyn<R, C, S>(inner: nalgebra::Matrix<T, R, C, S>) -> Self
    where
        R: nalgebra::Dim,
        C: nalgebra::Dim,
        S: nalgebra::RawStorage<T, R, C>,
    {
        let (rows, cols) = inner.shape();
        assert_eq!(
            rows,
            RowDims::LEN,
            "row count ({rows}) does not match RowDims length ({})",
            RowDims::LEN
        );
        assert_eq!(
            cols,
            ColDims::LEN,
            "column count ({cols}) does not match ColDims length ({})",
            ColDims::LEN
        );
        let nr = <CountedDim<RowDims> as nalgebra::Dim>::from_usize(RowDims::LEN);
        let nc = <CountedDim<ColDims> as nalgebra::Dim>::from_usize(ColDims::LEN);
        let out = nalgebra::OMatrix::<T, CountedDim<RowDims>, CountedDim<ColDims>>::from_fn_generic(
            nr,
            nc,
            |i, j| inner[(i, j)].clone(),
        );
        MixedUnitMatrix::from_nalgebra(out)
    }
}

// The identity is the one endomorphism whose *values* (1 on the diagonal, 0
// elsewhere) are dimensionless regardless of the space: entry `(i, i)` of
// `⟨Dims, Dims⟩` has unit `Dims[i]/Dims[i] = 1`, so the ones sit at exactly the
// dimensionless cells, and the off-diagonal zeros are unit-agnostic. It is
// therefore well-typed as an endomorphism over *any* dimension list `Dims`; the
// list length is tied to the matrix size via [`CountDim`], so a mismatch is a
// compile error.
impl<Dims, Brand, T, D> MixedUnitMatrix<Dims, Dims, nalgebra::OMatrix<T, D, D>, Brand>
where
    Dims: CountDim<Dim = D>,
    T: nalgebra::ComplexField,
    D: nalgebra::DimName,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
{
    /// The identity endomorphism `I : ⟨Dims, Dims⟩` — ones on the diagonal, zeros
    /// off it — for any dimension list `Dims`.
    ///
    /// This is the right/left unit for endomorphism `Mul`: `A · I = A` for any
    /// `A : ⟨R, Dims⟩` (and `I · A = A` for `A : ⟨Dims, C⟩`). Its diagonal is
    /// dimensionless because `Dims[i]/Dims[i] = 1`, so the same numeric identity
    /// serves every space; pick `Dims` to match the operand's column (or row)
    /// units.
    pub fn identity() -> Self {
        MixedUnitMatrix::from_nalgebra(nalgebra::OMatrix::<T, D, D>::identity_generic(
            D::name(),
            D::name(),
        ))
    }
}

// A diagonal matrix built from an "index" column vector `v : ⟨Dims, [1]⟩` (whose
// entry `i` is exactly `Dims[i]`): the result places `v[i]` at cell `(i, i)` and
// zero elsewhere. Its type is the *diagonal-operator* shape `⟨Dims, [1 … 1]⟩` —
// the same dimensionless-column shape as a Cholesky/`QR` factor — so cell
// `(i, i)` has unit `Dims[i]/1 = Dims[i]`, matching the source entry. (This is
// one gauge choice, folding the units into the rows; that is what makes it the
// natural inverse of reading a matrix's [`diagonal`](MixedUnitMatrix::diagonal)
// out as a plain index vector.)
impl<Dims, Brand, T, D>
    MixedUnitMatrix<Dims, Mapped<ToDimensionless, Dims>, nalgebra::OMatrix<T, D, D>, Brand>
where
    Dims: CountDim<Dim = D> + MapUnits<ToDimensionless>,
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
{
    /// Builds the diagonal matrix `⟨Dims, [1 … 1]⟩` from an index column vector
    /// `diag : ⟨Dims, [1]⟩`, placing `diag[i]` at cell `(i, i)`.
    ///
    /// The result's diagonal carries `Dims` (cell `(i, i)` has unit `Dims[i]`),
    /// its columns are dimensionless, and every off-diagonal cell is zero. It is
    /// the inverse of extracting a matrix's [`diagonal`](MixedUnitMatrix::diagonal)
    /// as a `⟨Dims, [1]⟩` vector (which is how a diagonal endomorphism reads out).
    pub fn from_diagonal<S>(
        diag: &MixedUnitMatrix<
            Dims,
            DCons<Dimensionless, DNil>,
            nalgebra::Matrix<T, D, nalgebra::U1, S>,
            Brand,
        >,
    ) -> Self
    where
        S: nalgebra::storage::RawStorage<T, D, nalgebra::U1>,
    {
        MixedUnitMatrix::from_nalgebra(nalgebra::OMatrix::<T, D, D>::from_diagonal(diag.nalgebra()))
    }
}

impl<RowDims, ColDims, Brand, T, R, C, St>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, St>, Brand>
where
    T: Copy,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    St: nalgebra::RawStorage<T, R, C>,
{
    /// Returns the element at the compile-time index `(I, J)` as a `Quantity`
    /// whose unit is `RowDims[I] / ColDims[J]` (with the matrix's storage type
    /// and brand).
    ///
    /// The indices must be compile-time constants: the returned unit is a type,
    /// so it can only be determined statically. An out-of-range index is a
    /// compile error (there is no [`ElemAt`] impl past the end of a list).
    pub fn get<const I: usize, const J: usize>(
        &self,
    ) -> EntryUnitOf<RowDims, ColDims, T, Brand, I, J>
    where
        Const<I>: ShapeIndex,
        Const<J>: ShapeIndex,
        RowDims: ElemAt<Nat<I>>,
        ColDims: ElemAt<Nat<J>>,
        RowUnitOf<RowDims, I>: UnitDiv<ColUnitOf<ColDims, J>>,
        EntryUnitOf<RowDims, ColDims, T, Brand, I, J>: FromRaw<T>,
    {
        let value = self.inner[(I, J)];
        <EntryUnitOf<RowDims, ColDims, T, Brand, I, J> as FromRaw<T>>::from_raw(value)
    }

    /// Returns a reference to the element at the compile-time index `(I, J)`,
    /// viewed as a `Quantity` whose unit is `RowDims[I] / ColDims[J]`.
    ///
    /// Unlike [`get`](Self::get), this borrows the element in place (no copy),
    /// reinterpreting the underlying scalar reference as a `&Quantity`.
    pub fn get_ref<const I: usize, const J: usize>(
        &self,
    ) -> &EntryUnitOf<RowDims, ColDims, T, Brand, I, J>
    where
        Const<I>: ShapeIndex,
        Const<J>: ShapeIndex,
        RowDims: ElemAt<Nat<I>>,
        ColDims: ElemAt<Nat<J>>,
        RowUnitOf<RowDims, I>: UnitDiv<ColUnitOf<ColDims, J>>,
        EntryUnitOf<RowDims, ColDims, T, Brand, I, J>: FromRaw<T>,
    {
        <EntryUnitOf<RowDims, ColDims, T, Brand, I, J> as FromRaw<T>>::ref_from_raw(
            &self.inner[(I, J)],
        )
    }
}

impl<RowDims, ColDims, Brand, T, R, C, St>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, St>, Brand>
where
    T: Copy,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    St: nalgebra::RawStorageMut<T, R, C>,
{
    /// Returns a mutable reference to the element at the compile-time index
    /// `(I, J)`, viewed as a `Quantity` whose unit is `RowDims[I] / ColDims[J]`.
    ///
    /// This enables unit-safe in-place mutation: assigning through the returned
    /// reference requires a `Quantity` of exactly the element's unit.
    pub fn get_mut<const I: usize, const J: usize>(
        &mut self,
    ) -> &mut EntryUnitOf<RowDims, ColDims, T, Brand, I, J>
    where
        Const<I>: ShapeIndex,
        Const<J>: ShapeIndex,
        RowDims: ElemAt<Nat<I>>,
        ColDims: ElemAt<Nat<J>>,
        RowUnitOf<RowDims, I>: UnitDiv<ColUnitOf<ColDims, J>>,
        EntryUnitOf<RowDims, ColDims, T, Brand, I, J>: FromRaw<T>,
    {
        <EntryUnitOf<RowDims, ColDims, T, Brand, I, J> as FromRaw<T>>::mut_from_raw(
            &mut self.inner[(I, J)],
        )
    }

    /// Writes `entry` into position `(I, J)`, returning `&mut self` for
    /// chaining.
    ///
    /// `entry` must be an [`IntoEntry`] for the element's unit
    /// `RowDims[I] / ColDims[J]`: a `Quantity` of exactly that unit (a
    /// wrong-unit quantity is a *compile error*), or a bare scalar when the
    /// element is dimensionless. This is the write-side dual of [`get`](Self::get)
    /// — no numeric erasure, so the units are checked at the boundary.
    pub fn set<const I: usize, const J: usize>(
        &mut self,
        entry: impl IntoEntry<EntryUnit<RowDims, ColDims, I, J>, T, Brand>,
    ) -> &mut Self
    where
        Const<I>: ShapeIndex,
        Const<J>: ShapeIndex,
        RowDims: ElemAt<Nat<I>>,
        ColDims: ElemAt<Nat<J>>,
        RowUnitOf<RowDims, I>: UnitDiv<ColUnitOf<ColDims, J>>,
    {
        self.inner[(I, J)] = entry.into_entry();
        self
    }

    /// Consuming-builder form of [`set`](Self::set): writes `entry` into
    /// position `(I, J)` and returns the matrix, so a whole matrix can be built
    /// up unit-safely from a zeroed start:
    ///
    /// ```ignore
    /// let a = StateMap::<S, N>::new(SMatrix::zeros())
    ///     .with::<0, 1>(quantity!(0.5, s)) // must be seconds here
    ///     .with::<0, 0>(1.0);              // dimensionless cell takes a scalar
    /// ```
    pub fn with<const I: usize, const J: usize>(
        mut self,
        entry: impl IntoEntry<EntryUnit<RowDims, ColDims, I, J>, T, Brand>,
    ) -> Self
    where
        Const<I>: ShapeIndex,
        Const<J>: ShapeIndex,
        RowDims: ElemAt<Nat<I>>,
        ColDims: ElemAt<Nat<J>>,
        RowUnitOf<RowDims, I>: UnitDiv<ColUnitOf<ColDims, J>>,
    {
        self.inner[(I, J)] = entry.into_entry();
        self
    }
}

// Row / column extraction. Both are dimensionally clean: slicing out a line of
// the matrix keeps the perpendicular dimension list intact and collapses the
// parallel one to the single unit at the extracted index. The index is a
// compile-time constant (like `get`) because the result's unit lists are types.
// The extracted line is returned *owned* (a copy) for a simple, view-free type.
impl<RowDims, ColDims, Brand, T, R, C, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::Scalar,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, R, C>,
{
    /// Extracts column `J` as a [`UnitVector`]: an `R x 1` matrix that keeps the
    /// original row (output) dimensions `RowDims` and whose single column unit is
    /// `ColDims[J]`. Entry `i` therefore has unit `RowDims[i] / ColDims[J]`,
    /// exactly as in the parent matrix.
    ///
    /// `J` is a compile-time index; an out-of-range column is a compile error
    /// (there is no [`ElemAt`] impl past the end of `ColDims`).
    pub fn column<const J: usize>(
        &self,
    ) -> UnitVector<RowDims, ColUnitOf<ColDims, J>, nalgebra::OMatrix<T, R, nalgebra::U1>, Brand>
    where
        Const<J>: ShapeIndex,
        ColDims: ElemAt<Nat<J>>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R>,
    {
        UnitVector::from_nalgebra(self.inner.column(J).into_owned())
    }

    /// Extracts row `I` as a [`UnitRowVector`]: a `1 x C` matrix that keeps the
    /// original column (input) dimensions `ColDims` and whose single row unit is
    /// `RowDims[I]`. Entry `j` therefore has unit `RowDims[I] / ColDims[j]`.
    ///
    /// `I` is a compile-time index; an out-of-range row is a compile error.
    pub fn row<const I: usize>(
        &self,
    ) -> UnitRowVector<RowUnitOf<RowDims, I>, ColDims, nalgebra::OMatrix<T, nalgebra::U1, C>, Brand>
    where
        Const<I>: ShapeIndex,
        RowDims: ElemAt<Nat<I>>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<nalgebra::U1, C>,
    {
        UnitRowVector::from_nalgebra(self.inner.row(I).into_owned())
    }
}

// Diagonal extraction (square matrices only). The `i`th diagonal entry is
// `M(i, i)`, whose unit is `RowDims[i] / ColDims[i]` — the element-wise quotient
// of the two dimension lists ([`ZipDiv`]). The result is a plain column vector:
// its row units are those quotients and its single column unit is dimensionless.
impl<RowDims, ColDims, Brand, T, D, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, D, D, S>, Brand>
where
    T: nalgebra::Scalar,
    D: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, D, D>,
{
    /// Extracts the main diagonal as a [`UnitVector`] whose `i`th entry has unit
    /// `RowDims[i] / ColDims[i]`.
    ///
    /// The row/column dimension lists must have equal length (guaranteed for a
    /// square matrix built via [`new`](Self::new)); the diagonal's row units are
    /// their element-wise quotient and its column unit is
    /// [`Dimensionless`].
    pub fn diagonal(
        &self,
    ) -> UnitVector<
        ZipDivided<RowDims, ColDims>,
        Dimensionless,
        nalgebra::OMatrix<T, D, nalgebra::U1>,
        Brand,
    >
    where
        RowDims: ZipDiv<ColDims>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
    {
        UnitVector::from_nalgebra(self.inner.diagonal())
    }
}

// Rectangular block extraction. A contiguous `RB x CB` block anchored at
// `(R0, C0)` keeps each entry's parent unit `RowDims[R0+i] / ColDims[C0+j]`, so
// the extracted matrix carries the *sublists* `RowDims[R0 .. R0+RB]` and
// `ColDims[C0 .. C0+CB]` — computed at the type level by `Sliced` (`drop`, then
// `take`). All four bounds are compile-time constants; a block running past
// either dimension list fails to resolve (there is no `Take` past the end),
// which is a compile error. The block is returned *owned* (a copy) for a
// view-free result type, mirroring `column`/`row`/`diagonal`.
impl<RowDims, ColDims, Brand, T, R, C, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::Scalar,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, R, C>,
{
    /// Extracts the fixed `RB x CB` block whose top-left corner is `(R0, C0)`,
    /// as an owned matrix carrying the corresponding sub-vectors of the
    /// dimension lists: rows `RowDims[R0 .. R0+RB]` and columns
    /// `ColDims[C0 .. C0+CB]`. Entry `(i, j)` keeps its parent unit
    /// `RowDims[R0+i] / ColDims[C0+j]`.
    ///
    /// All four indices are compile-time constants; a block that runs past the
    /// end of either dimension list is a compile error.
    pub fn block<const R0: usize, const C0: usize, const RB: usize, const CB: usize>(
        &self,
    ) -> MixedUnitMatrix<
        Sliced<RowDims, R0, RB>,
        Sliced<ColDims, C0, CB>,
        nalgebra::OMatrix<T, nalgebra::Const<RB>, nalgebra::Const<CB>>,
        Brand,
    >
    where
        Const<R0>: ShapeIndex,
        Const<C0>: ShapeIndex,
        Const<RB>: ShapeIndex,
        Const<CB>: ShapeIndex,
        RowDims: Drop<Nat<R0>>,
        ColDims: Drop<Nat<C0>>,
        <RowDims as Drop<Nat<R0>>>::Out: Take<Nat<RB>>,
        <ColDims as Drop<Nat<C0>>>::Out: Take<Nat<CB>>,
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<nalgebra::Const<RB>, nalgebra::Const<CB>>,
    {
        self.block_off::<Nat<R0>, Nat<C0>, RB, CB>()
    }

    /// [`block`](Self::block) with the offsets given as type-level naturals
    /// (`RowOff`/`ColOff`, [`Unsigned`]) rather than const generics. The block's
    /// top-left corner is `(RowOff, ColOff)` and its size is the const `RB x CB`.
    ///
    /// Const generics can't be added in argument position without
    /// `generic_const_exprs`, so a block placed at the sum of preceding block
    /// sizes (a third-or-later block in a generic partition) can't spell its
    /// offset as `{ N + M }`; a type-level offset `Sum<Nat<N>, Nat<M>>` can. This
    /// is the primitive [`unblock_matrix!`](crate::nalgebra::unblock_matrix) uses
    /// so a partition of any width/height works even over generic const shapes.
    /// The const-offset [`block`](Self::block) is the wrapper for literal/single
    /// offsets.
    pub fn block_off<RowOff, ColOff, const RB: usize, const CB: usize>(
        &self,
    ) -> MixedUnitMatrix<
        SlicedAt<RowDims, RowOff, RB>,
        SlicedAt<ColDims, ColOff, CB>,
        nalgebra::OMatrix<T, nalgebra::Const<RB>, nalgebra::Const<CB>>,
        Brand,
    >
    where
        RowOff: Unsigned,
        ColOff: Unsigned,
        Const<RB>: ShapeIndex,
        Const<CB>: ShapeIndex,
        RowDims: Drop<RowOff>,
        ColDims: Drop<ColOff>,
        <RowDims as Drop<RowOff>>::Out: Take<Nat<RB>>,
        <ColDims as Drop<ColOff>>::Out: Take<Nat<CB>>,
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<nalgebra::Const<RB>, nalgebra::Const<CB>>,
    {
        MixedUnitMatrix::from_nalgebra(
            self.inner
                .fixed_view::<RB, CB>(RowOff::USIZE, ColOff::USIZE)
                .into_owned(),
        )
    }

    /// [`block`](Self::block), but auto-reducing: the extracted block comes back
    /// as a [`UniformUnitMatrix`] when its row and column sublists are both
    /// uniform (so every entry shares one unit), and as a `MixedUnitMatrix`
    /// otherwise. The representation is chosen at the type level from the
    /// sublists alone — see [`AutoReduce`] — so no annotation is needed at the
    /// call site. This is what [`unblock_matrix!`](crate::nalgebra::unblock_matrix) reads
    /// blocks through, so uniform blocks decode straight to the lighter type.
    pub fn block_auto<const R0: usize, const C0: usize, const RB: usize, const CB: usize>(
        &self,
    ) -> Reduced<
        Sliced<RowDims, R0, RB>,
        Sliced<ColDims, C0, CB>,
        nalgebra::OMatrix<T, nalgebra::Const<RB>, nalgebra::Const<CB>>,
        Brand,
    >
    where
        Const<R0>: ShapeIndex,
        Const<C0>: ShapeIndex,
        Const<RB>: ShapeIndex,
        Const<CB>: ShapeIndex,
        RowDims: Drop<Nat<R0>>,
        ColDims: Drop<Nat<C0>>,
        <RowDims as Drop<Nat<R0>>>::Out: Take<Nat<RB>>,
        <ColDims as Drop<Nat<C0>>>::Out: Take<Nat<CB>>,
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<nalgebra::Const<RB>, nalgebra::Const<CB>>,
        (Sliced<RowDims, R0, RB>, Sliced<ColDims, C0, CB>):
            AutoReduce<nalgebra::OMatrix<T, nalgebra::Const<RB>, nalgebra::Const<CB>>, Brand>,
    {
        self.block_auto_off::<Nat<R0>, Nat<C0>, RB, CB>()
    }

    /// [`block_auto`](Self::block_auto) with type-level offsets — the
    /// auto-reducing analogue of [`block_off`](Self::block_off). See
    /// `block_off` for why the offsets are types.
    pub fn block_auto_off<RowOff, ColOff, const RB: usize, const CB: usize>(
        &self,
    ) -> Reduced<
        SlicedAt<RowDims, RowOff, RB>,
        SlicedAt<ColDims, ColOff, CB>,
        nalgebra::OMatrix<T, nalgebra::Const<RB>, nalgebra::Const<CB>>,
        Brand,
    >
    where
        RowOff: Unsigned,
        ColOff: Unsigned,
        Const<RB>: ShapeIndex,
        Const<CB>: ShapeIndex,
        RowDims: Drop<RowOff>,
        ColDims: Drop<ColOff>,
        <RowDims as Drop<RowOff>>::Out: Take<Nat<RB>>,
        <ColDims as Drop<ColOff>>::Out: Take<Nat<CB>>,
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<nalgebra::Const<RB>, nalgebra::Const<CB>>,
        (SlicedAt<RowDims, RowOff, RB>, SlicedAt<ColDims, ColOff, CB>):
            AutoReduce<nalgebra::OMatrix<T, nalgebra::Const<RB>, nalgebra::Const<CB>>, Brand>,
    {
        <(SlicedAt<RowDims, RowOff, RB>, SlicedAt<ColDims, ColOff, CB>) as AutoReduce<_, Brand>>::wrap(
            self.inner
                .fixed_view::<RB, CB>(RowOff::USIZE, ColOff::USIZE)
                .into_owned(),
        )
    }

    /// Borrows the fixed `RB x CB` block at top-left `(R0, C0)` as a
    /// [`MatrixView`](nalgebra::MatrixView): the zero-copy analogue of
    /// [`block`](Self::block). It carries the same sublists (rows
    /// `RowDims[R0 .. R0+RB]`, columns `ColDims[C0 .. C0+CB]`) and the same unit
    /// on every entry, but shares the parent's storage rather than copying, so
    /// it borrows `self` for as long as the view lives. Because the borrow is
    /// shared, any number of blocks may be viewed at once (see
    /// [`unblock_matrix!(views; …)`](crate::nalgebra::unblock_matrix)); prefer this for
    /// reading large sub-blocks, and [`block`](Self::block) when an owned,
    /// view-free result is needed.
    ///
    /// The result is still a `MixedUnitMatrix`, so it reads, prints, and feeds
    /// into arithmetic exactly like an owned block — a `MatrixView` is just a
    /// `Matrix` over borrowing storage, and every op is bounded on the storage.
    pub fn block_view<const R0: usize, const C0: usize, const RB: usize, const CB: usize>(
        &self,
    ) -> MixedUnitMatrix<
        Sliced<RowDims, R0, RB>,
        Sliced<ColDims, C0, CB>,
        nalgebra::MatrixView<
            '_,
            T,
            nalgebra::Const<RB>,
            nalgebra::Const<CB>,
            S::RStride,
            S::CStride,
        >,
        Brand,
    >
    where
        Const<R0>: ShapeIndex,
        Const<C0>: ShapeIndex,
        Const<RB>: ShapeIndex,
        Const<CB>: ShapeIndex,
        RowDims: Drop<Nat<R0>>,
        ColDims: Drop<Nat<C0>>,
        <RowDims as Drop<Nat<R0>>>::Out: Take<Nat<RB>>,
        <ColDims as Drop<Nat<C0>>>::Out: Take<Nat<CB>>,
    {
        self.block_view_off::<Nat<R0>, Nat<C0>, RB, CB>()
    }

    /// [`block_view`](Self::block_view) with type-level offsets — the borrowing
    /// analogue of [`block_off`](Self::block_off). See `block_off` for why the
    /// offsets are types.
    pub fn block_view_off<RowOff, ColOff, const RB: usize, const CB: usize>(
        &self,
    ) -> MixedUnitMatrix<
        SlicedAt<RowDims, RowOff, RB>,
        SlicedAt<ColDims, ColOff, CB>,
        nalgebra::MatrixView<
            '_,
            T,
            nalgebra::Const<RB>,
            nalgebra::Const<CB>,
            S::RStride,
            S::CStride,
        >,
        Brand,
    >
    where
        RowOff: Unsigned,
        ColOff: Unsigned,
        Const<RB>: ShapeIndex,
        Const<CB>: ShapeIndex,
        RowDims: Drop<RowOff>,
        ColDims: Drop<ColOff>,
        <RowDims as Drop<RowOff>>::Out: Take<Nat<RB>>,
        <ColDims as Drop<ColOff>>::Out: Take<Nat<CB>>,
    {
        MixedUnitMatrix::from_nalgebra(
            self.inner
                .fixed_view::<RB, CB>(RowOff::USIZE, ColOff::USIZE),
        )
    }

    /// [`block_view`](Self::block_view), but auto-reducing: the borrowed block
    /// comes back as a [`UniformUnitMatrix`] over borrowing storage when its row
    /// and column sublists are both uniform, and as a `MixedUnitMatrix`
    /// otherwise (see [`AutoReduce`]). Like `block_view` it shares `self`'s
    /// storage, so any number of views may coexist; this is what
    /// [`unblock_matrix!(views; …)`](crate::nalgebra::unblock_matrix) reads through.
    pub fn block_view_auto<'a, const R0: usize, const C0: usize, const RB: usize, const CB: usize>(
        &'a self,
    ) -> Reduced<
        Sliced<RowDims, R0, RB>,
        Sliced<ColDims, C0, CB>,
        nalgebra::MatrixView<
            'a,
            T,
            nalgebra::Const<RB>,
            nalgebra::Const<CB>,
            S::RStride,
            S::CStride,
        >,
        Brand,
    >
    where
        Const<R0>: ShapeIndex,
        Const<C0>: ShapeIndex,
        Const<RB>: ShapeIndex,
        Const<CB>: ShapeIndex,
        RowDims: Drop<Nat<R0>>,
        ColDims: Drop<Nat<C0>>,
        <RowDims as Drop<Nat<R0>>>::Out: Take<Nat<RB>>,
        <ColDims as Drop<Nat<C0>>>::Out: Take<Nat<CB>>,
        (Sliced<RowDims, R0, RB>, Sliced<ColDims, C0, CB>): AutoReduce<
                nalgebra::MatrixView<
                    'a,
                    T,
                    nalgebra::Const<RB>,
                    nalgebra::Const<CB>,
                    S::RStride,
                    S::CStride,
                >,
                Brand,
            >,
    {
        self.block_view_auto_off::<Nat<R0>, Nat<C0>, RB, CB>()
    }

    /// [`block_view_auto`](Self::block_view_auto) with type-level offsets — the
    /// borrowing, auto-reducing analogue of [`block_off`](Self::block_off). See
    /// `block_off` for why the offsets are types.
    pub fn block_view_auto_off<'a, RowOff, ColOff, const RB: usize, const CB: usize>(
        &'a self,
    ) -> Reduced<
        SlicedAt<RowDims, RowOff, RB>,
        SlicedAt<ColDims, ColOff, CB>,
        nalgebra::MatrixView<
            'a,
            T,
            nalgebra::Const<RB>,
            nalgebra::Const<CB>,
            S::RStride,
            S::CStride,
        >,
        Brand,
    >
    where
        RowOff: Unsigned,
        ColOff: Unsigned,
        Const<RB>: ShapeIndex,
        Const<CB>: ShapeIndex,
        RowDims: Drop<RowOff>,
        ColDims: Drop<ColOff>,
        <RowDims as Drop<RowOff>>::Out: Take<Nat<RB>>,
        <ColDims as Drop<ColOff>>::Out: Take<Nat<CB>>,
        (SlicedAt<RowDims, RowOff, RB>, SlicedAt<ColDims, ColOff, CB>): AutoReduce<
                nalgebra::MatrixView<
                    'a,
                    T,
                    nalgebra::Const<RB>,
                    nalgebra::Const<CB>,
                    S::RStride,
                    S::CStride,
                >,
                Brand,
            >,
    {
        <(SlicedAt<RowDims, RowOff, RB>, SlicedAt<ColDims, ColOff, CB>) as AutoReduce<_, Brand>>::wrap(
            self.inner.fixed_view::<RB, CB>(RowOff::USIZE, ColOff::USIZE),
        )
    }

    /// Writes a fixed `RB x CB` block into this matrix at top-left `(R0, C0)`,
    /// the mutating inverse of [`block`](Self::block). The block's dimension
    /// lists must match the target's sublists there — rows `RowDims[R0 .. R0+RB]`
    /// and columns `ColDims[C0 .. C0+CB]`, spelled by the same `Sliced` types —
    /// so a block whose units don't line up with its destination is a compile
    /// error. This is what lets a partitioned matrix be assembled from typed
    /// pieces without dropping to raw storage (e.g. a Van Loan augmented block
    /// matrix `[[A, B], [0, 0]]`).
    pub fn set_block<const R0: usize, const C0: usize, const RB: usize, const CB: usize, SB>(
        &mut self,
        block: &MixedUnitMatrix<
            Sliced<RowDims, R0, RB>,
            Sliced<ColDims, C0, CB>,
            nalgebra::Matrix<T, nalgebra::Const<RB>, nalgebra::Const<CB>, SB>,
            Brand,
        >,
    ) where
        S: nalgebra::storage::StorageMut<T, R, C>,
        SB: nalgebra::storage::Storage<T, nalgebra::Const<RB>, nalgebra::Const<CB>>,
        Const<R0>: ShapeIndex,
        Const<C0>: ShapeIndex,
        Const<RB>: ShapeIndex,
        Const<CB>: ShapeIndex,
        RowDims: Drop<Nat<R0>>,
        ColDims: Drop<Nat<C0>>,
        <RowDims as Drop<Nat<R0>>>::Out: Take<Nat<RB>>,
        <ColDims as Drop<Nat<C0>>>::Out: Take<Nat<CB>>,
    {
        self.inner
            .fixed_view_mut::<RB, CB>(R0, C0)
            .copy_from(&block.inner);
    }

    /// Consuming form of [`set_block`](Self::set_block): writes `block` at
    /// top-left `(R0, C0)` and returns `self`, so a partitioned matrix can be
    /// built as a single expression — start from a zero matrix and place each
    /// occupied block — rather than mutating in a sequence of statements:
    ///
    /// ```ignore
    /// let m = zeros![Rows, Cols]
    ///     .with_block::<0, 0, N, N, _>(&top_left)
    ///     .with_block::<0, N, N, M, _>(&top_right);
    /// ```
    ///
    /// Unoccupied cells keep their zero (or prior) value. Same unit check as
    /// `set_block`: the block's dimension lists must match the destination
    /// sublists, or it fails to compile.
    #[must_use]
    pub fn with_block<const R0: usize, const C0: usize, const RB: usize, const CB: usize, SB>(
        mut self,
        block: &MixedUnitMatrix<
            Sliced<RowDims, R0, RB>,
            Sliced<ColDims, C0, CB>,
            nalgebra::Matrix<T, nalgebra::Const<RB>, nalgebra::Const<CB>, SB>,
            Brand,
        >,
    ) -> Self
    where
        S: nalgebra::storage::StorageMut<T, R, C>,
        SB: nalgebra::storage::Storage<T, nalgebra::Const<RB>, nalgebra::Const<CB>>,
        Const<R0>: ShapeIndex,
        Const<C0>: ShapeIndex,
        Const<RB>: ShapeIndex,
        Const<CB>: ShapeIndex,
        RowDims: Drop<Nat<R0>>,
        ColDims: Drop<Nat<C0>>,
        <RowDims as Drop<Nat<R0>>>::Out: Take<Nat<RB>>,
        <ColDims as Drop<Nat<C0>>>::Out: Take<Nat<CB>>,
    {
        self.set_block::<R0, C0, RB, CB, SB>(block);
        self
    }
}

impl<RowDims, ColDims, Brand, T, R, C, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::Scalar,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::RawStorage<T, R, C>,
{
    /// Horizontally concatenates `[self | right]`. Both operands must span the
    /// same row space `RowDims` (and the same numeric row dimension `R`); the
    /// result's columns are `ColDims` followed by `right`'s columns, and its
    /// column dimension is `DimSum<C, C2>` — a `Const` when both are `Const`
    /// (≤127), else `Dyn`.
    pub fn hcat<ColDimsR, C2, S2>(
        &self,
        right: &MixedUnitMatrix<RowDims, ColDimsR, nalgebra::Matrix<T, R, C2, S2>, Brand>,
    ) -> MixedUnitMatrix<
        RowDims,
        Concatenated<ColDims, ColDimsR>,
        nalgebra::OMatrix<T, R, nalgebra::DimSum<C, C2>>,
        Brand,
    >
    where
        ColDims: Concat<ColDimsR>,
        C: nalgebra::DimAdd<C2>,
        C2: nalgebra::Dim,
        S2: nalgebra::RawStorage<T, R, C2>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, nalgebra::DimSum<C, C2>>,
    {
        let (r, c1) = self.inner.shape_generic();
        let (r2, c2) = right.inner.shape_generic();
        assert_eq!(r.value(), r2.value(), "hcat: row-count mismatch");
        let res_c = <C as nalgebra::DimAdd<C2>>::add(c1, c2);
        let c1n = c1.value();
        let out = nalgebra::OMatrix::<T, R, nalgebra::DimSum<C, C2>>::from_fn_generic(
            r,
            res_c,
            |i, j| {
                if j < c1n {
                    self.inner[(i, j)].clone()
                } else {
                    right.inner[(i, j - c1n)].clone()
                }
            },
        );
        MixedUnitMatrix::from_nalgebra(out)
    }

    /// Vertically stacks `[self; below]`. Both operands must span the same column
    /// space `ColDims` (and the same numeric column dimension `C`); the result's
    /// rows are `RowDims` followed by `below`'s rows, and its row dimension is
    /// `DimSum<R, R2>` — a `Const` when both are `Const` (≤127), else `Dyn`.
    pub fn vcat<RowDimsB, R2, S2>(
        &self,
        below: &MixedUnitMatrix<RowDimsB, ColDims, nalgebra::Matrix<T, R2, C, S2>, Brand>,
    ) -> MixedUnitMatrix<
        Concatenated<RowDims, RowDimsB>,
        ColDims,
        nalgebra::OMatrix<T, nalgebra::DimSum<R, R2>, C>,
        Brand,
    >
    where
        RowDims: Concat<RowDimsB>,
        R: nalgebra::DimAdd<R2>,
        R2: nalgebra::Dim,
        S2: nalgebra::RawStorage<T, R2, C>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<nalgebra::DimSum<R, R2>, C>,
    {
        let (r1, c) = self.inner.shape_generic();
        let (r2, c2) = below.inner.shape_generic();
        assert_eq!(c.value(), c2.value(), "vcat: column-count mismatch");
        let res_r = <R as nalgebra::DimAdd<R2>>::add(r1, r2);
        let r1n = r1.value();
        let out = nalgebra::OMatrix::<T, nalgebra::DimSum<R, R2>, C>::from_fn_generic(
            res_r,
            c,
            |i, j| {
                if i < r1n {
                    self.inner[(i, j)].clone()
                } else {
                    below.inner[(i - r1n, j)].clone()
                }
            },
        );
        MixedUnitMatrix::from_nalgebra(out)
    }
}

// Triangular extraction zeroes the strict opposite triangle. A zero is a valid
// entry at any unit and every surviving entry keeps its own `RowDims[i]/ColDims[j]`
// unit, so — unlike a reduction — this is fully type-preserving on the mixed
// matrix (no uniformity needed).
impl<RowDims, ColDims, Brand, T, R, C, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, R, C>,
{
    /// The upper-triangular part (including the diagonal), zeroing everything
    /// below it. Keeps the exact `⟨RowDims, ColDims⟩` type.
    pub fn upper_triangle(
        &self,
    ) -> MixedUnitMatrix<RowDims, ColDims, nalgebra::OMatrix<T, R, C>, Brand>
    where
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
    {
        MixedUnitMatrix {
            inner: self.inner.upper_triangle(),
            _dims: PhantomData,
        }
    }

    /// The lower-triangular part (including the diagonal), zeroing everything
    /// above it. Keeps the exact `⟨RowDims, ColDims⟩` type.
    pub fn lower_triangle(
        &self,
    ) -> MixedUnitMatrix<RowDims, ColDims, nalgebra::OMatrix<T, R, C>, Brand>
    where
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
    {
        MixedUnitMatrix {
            inner: self.inner.lower_triangle(),
            _dims: PhantomData,
        }
    }
}
