#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use whippyunits::quantity::Quantity;
use whippyunits::{DivUnit, UnitDiv};

use super::UniformUnitMatrix;
use crate::dims::{DimList, Dimensionless};
use crate::entry::FromRaw;
use crate::index::{Nat, Repeat, Repeated, ShapeIndex};
use crate::nalgebra::matrix::{CountDim, CountedDim, MixedUnitMatrix};
use crate::uniformity::{CollapseUniform, GaugeReproduces, Uniform};

impl<U, M, Brand> UniformUnitMatrix<U, M, Brand> {
    /// Wraps an underlying nalgebra matrix, tagging every entry with the unit
    /// `U`.
    ///
    /// Unlike [`MixedUnitMatrix::new`](crate::nalgebra::MixedUnitMatrix), no shape check is
    /// needed: a uniform matrix imposes no per-row/column list lengths, so any
    /// shape is valid.
    pub fn from_nalgebra(inner: M) -> Self {
        Self {
            inner,
            _unit: PhantomData,
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
}

// Size/shape queries forward directly; they are unit-invariant.
impl<U, Brand, T, R, C, S> UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, S>, Brand>
where
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::RawStorage<T, R, C>,
{
    /// The `(rows, cols)` shape of the matrix.
    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    /// The number of rows.
    pub fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    /// The number of columns.
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

    /// Returns `true` if the matrix is square.
    pub fn is_square(&self) -> bool {
        self.inner.is_square()
    }
}

// The identity is inherently dimensionless in uniform form: its ones sit on
// the diagonal and its zeros elsewhere, so for *every* entry to carry one shared
// unit `U` that `U` must be `1` (a `U ≠ 1` would have to appear on the diagonal
// too). The unit-carrying diagonal lives in the *mixed* identity
// `⟨Dims, Dims⟩`; here the uniform identity is the dimensionless one.
impl<Brand, T, D> UniformUnitMatrix<Dimensionless, nalgebra::OMatrix<T, D, D>, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::DimName,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
{
    /// The dimensionless identity matrix (ones on the diagonal, zeros off it).
    ///
    /// A uniform matrix shares one unit across every entry, and the identity's
    /// values pin that unit to `1`; a unit-carrying diagonal is the *mixed*
    /// [`MixedUnitMatrix::identity`](crate::nalgebra::MixedUnitMatrix::identity) over
    /// `⟨Dims, Dims⟩`.
    pub fn identity() -> Self {
        UniformUnitMatrix::from_nalgebra(nalgebra::OMatrix::<T, D, D>::identity_generic(
            D::name(),
            D::name(),
        ))
    }
}

// A diagonal matrix built from a uniform column vector is itself uniform in the
// same unit `U`: the entries `diag[i]` (each in `U`) land on the diagonal and
// the off-diagonal zeros are unit-agnostic, so the whole matrix carries `U`. No
// gauge choice is needed — unlike the mixed
// [`from_diagonal`](crate::nalgebra::MixedUnitMatrix::from_diagonal), whose diagonal-vs-
// column split is one of the gauges a single shared unit collapses away.
impl<U, Brand, T, D> UniformUnitMatrix<U, nalgebra::OMatrix<T, D, D>, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D, D>,
{
    /// Builds the diagonal matrix from a uniform column vector `diag`, placing
    /// `diag[i]` at cell `(i, i)` and zero elsewhere. The result is uniform in
    /// the same entry unit `U`.
    pub fn from_diagonal<S>(
        diag: &UniformUnitMatrix<U, nalgebra::Matrix<T, D, nalgebra::U1, S>, Brand>,
    ) -> Self
    where
        S: nalgebra::storage::RawStorage<T, D, nalgebra::U1>,
    {
        UniformUnitMatrix::from_nalgebra(nalgebra::OMatrix::<T, D, D>::from_diagonal(
            diag.nalgebra(),
        ))
    }
}

// Shaped constructors for a statically-sized uniform matrix. Both are unit-safe:
// every entry shares the one unit `U`, so a single `Quantity<U>` (or a closure
// producing them) fully determines the matrix — the uniform echo of nalgebra's
// `from_element` / `from_fn`. The mixed matrix has no such pair, since its
// per-cell units differ; it is built cell-typed via
// [`mixed_unit_matrix!`](crate::nalgebra::mixed_unit_matrix).
impl<U, Brand, T, R, C> UniformUnitMatrix<U, nalgebra::OMatrix<T, R, C>, Brand>
where
    T: nalgebra::Scalar,
    R: nalgebra::DimName,
    C: nalgebra::DimName,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
{
    /// Builds a uniform matrix with every entry equal to `value` (unit `U`).
    pub fn from_element(value: Quantity<U, T, Brand>) -> Self {
        UniformUnitMatrix::from_nalgebra(nalgebra::OMatrix::<T, R, C>::from_element_generic(
            R::name(),
            C::name(),
            value.unsafe_value,
        ))
    }

    /// Builds a uniform matrix whose entry `(i, j)` is `f(i, j)` — each a
    /// `Quantity<U>`, so the result is uniform in `U` by construction.
    pub fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> Quantity<U, T, Brand>,
    {
        UniformUnitMatrix::from_nalgebra(nalgebra::OMatrix::<T, R, C>::from_fn_generic(
            R::name(),
            C::name(),
            |i, j| f(i, j).unsafe_value,
        ))
    }

    /// Builds a uniform matrix from raw scalars given in row-major order,
    /// each read as a magnitude in the shared unit `U`.
    ///
    /// Like [`from_nalgebra`](Self::from_nalgebra), this asserts that the bare
    /// numbers are already in `U` — the bulk-entry bridge, honestly unchecked at
    /// the scalar level but fixed to one unit by the type parameter.
    pub fn from_row_slice(data: &[T]) -> Self {
        UniformUnitMatrix::from_nalgebra(nalgebra::OMatrix::<T, R, C>::from_row_slice_generic(
            R::name(),
            C::name(),
            data,
        ))
    }

    /// Builds a uniform matrix from raw scalars given in column-major order
    /// (nalgebra's native layout), each read as a magnitude in `U`. See
    /// [`from_row_slice`](Self::from_row_slice) for the unit contract.
    pub fn from_column_slice(data: &[T]) -> Self {
        UniformUnitMatrix::from_nalgebra(nalgebra::OMatrix::<T, R, C>::from_column_slice_generic(
            R::name(),
            C::name(),
            data,
        ))
    }

    /// Builds a uniform matrix from a column-major `Vec` of raw scalars, each
    /// read as a magnitude in `U`. See [`from_row_slice`](Self::from_row_slice)
    /// for the unit contract.
    pub fn from_vec(data: Vec<T>) -> Self {
        UniformUnitMatrix::from_nalgebra(nalgebra::OMatrix::<T, R, C>::from_vec_generic(
            R::name(),
            C::name(),
            data,
        ))
    }

    /// Builds a uniform matrix from a column-major iterator of raw scalars,
    /// each read as a magnitude in `U`. See [`from_row_slice`](Self::from_row_slice)
    /// for the unit contract.
    pub fn from_iterator<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        UniformUnitMatrix::from_nalgebra(nalgebra::OMatrix::<T, R, C>::from_iterator_generic(
            R::name(),
            C::name(),
            iter,
        ))
    }

    /// Builds a uniform matrix from its columns — each a uniform column vector of
    /// the same unit `U`. Panics if the number of columns disagrees with `C`.
    pub fn from_columns<Sc>(
        columns: &[UniformUnitMatrix<U, nalgebra::Matrix<T, R, nalgebra::U1, Sc>, Brand>],
    ) -> Self
    where
        Sc: nalgebra::storage::Storage<T, R, nalgebra::U1>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, nalgebra::U1>,
    {
        let cols: Vec<_> = columns.iter().map(|c| c.inner.clone_owned()).collect();
        UniformUnitMatrix::from_nalgebra(nalgebra::OMatrix::<T, R, C>::from_columns(&cols))
    }

    /// Builds a uniform matrix from its rows — each a uniform row vector of the
    /// same unit `U`. Panics if the number of rows disagrees with `R`.
    pub fn from_rows<Sr>(
        rows: &[UniformUnitMatrix<U, nalgebra::Matrix<T, nalgebra::U1, C, Sr>, Brand>],
    ) -> Self
    where
        Sr: nalgebra::storage::Storage<T, nalgebra::U1, C>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<nalgebra::U1, C>,
    {
        let rs: Vec<_> = rows.iter().map(|r| r.inner.clone_owned()).collect();
        UniformUnitMatrix::from_nalgebra(nalgebra::OMatrix::<T, R, C>::from_rows(&rs))
    }
}

// Block assembly, uniform. Both operands share the single entry unit `U`, so —
// unlike the mixed [`hcat`](crate::nalgebra::MixedUnitMatrix::hcat)/[`vcat`](crate::nalgebra::MixedUnitMatrix::vcat)
// which concatenate row/column dimension *lists* — there is nothing to join at
// the unit level: the concatenation just widens/stacks the numeric shape and
// the result stays uniform in `U`. These back the uniform arm of
// [`block_matrix!`](crate::nalgebra::block_matrix).
impl<U, Brand, T, R, C, S> UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::Scalar,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::RawStorage<T, R, C>,
{
    /// Horizontally concatenates `[self | right]`. Both operands must span the
    /// same numeric row dimension `R` and share the entry unit `U`; the result's
    /// column dimension is `DimSum<C, C2>` (a `Const` when both are `Const`,
    /// else `Dyn`), still uniform in `U`.
    pub fn hcat<C2, S2>(
        &self,
        right: &UniformUnitMatrix<U, nalgebra::Matrix<T, R, C2, S2>, Brand>,
    ) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, R, nalgebra::DimSum<C, C2>>, Brand>
    where
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
        UniformUnitMatrix::from_nalgebra(out)
    }

    /// Vertically stacks `[self; below]`. Both operands must span the same
    /// numeric column dimension `C` and share the entry unit `U`; the result's
    /// row dimension is `DimSum<R, R2>` (a `Const` when both are `Const`, else
    /// `Dyn`), still uniform in `U`.
    pub fn vcat<R2, S2>(
        &self,
        below: &UniformUnitMatrix<U, nalgebra::Matrix<T, R2, C, S2>, Brand>,
    ) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, nalgebra::DimSum<R, R2>, C>, Brand>
    where
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
        UniformUnitMatrix::from_nalgebra(out)
    }
}

// Gauge assignment: express a uniform matrix as a `MixedUnitMatrix` over
// explicit row/column dimension lists. A uniform matrix carries no row/column
// split of its own (whether declared that way or collapsed from a mixed one), so
// one must be assigned here — and it leaves a genuine gauge freedom: since every
// entry shares the single unit `U`, and entry `(i, j)` of a mixed matrix is
// `RowDims[i] / ColDims[j]`, the target lists must both be *uniform* (constant)
// with `Uniform(RowDims) / Uniform(ColDims) == U`. Only the common scale between
// the two constants is free — that is the gauge — and it is stated explicitly
// here rather than guessed from neighbors, so embedding a uniform block into a
// `block_matrix!` layout is an explicit, checked act.
impl<U, T, R, C, S, Brand> UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, S>, Brand>
where
    R: nalgebra::DimName,
    C: nalgebra::DimName,
    S: nalgebra::RawStorage<T, R, C>,
{
    /// Gauges this uniform matrix into a [`MixedUnitMatrix`] by assigning it the
    /// explicit dimension lists `RowDims` and `ColDims`.
    ///
    /// Both lists must be uniform and reproduce this matrix's entry unit —
    /// `Uniform(RowDims) / Uniform(ColDims) == U` — which is exactly the freedom
    /// a single shared unit leaves open (a uniform block admits only constant
    /// row/column lists; their common scale is the gauge you pick here). A tag
    /// that isn't uniform, or whose quotient isn't `U`, is a compile error. The
    /// shape (list lengths vs. the matrix's rows/columns) is checked at compile
    /// time and zero-cost, exactly as in [`MixedUnitMatrix::new`].
    ///
    /// This is how a uniform block enters a [`block_matrix!`](crate::nalgebra::block_matrix)
    /// layout — most legibly via the [`gauge!`](crate::nalgebra::gauge) cell wrapper — where
    /// its neighbors fix which spaces it must line up with.
    pub fn gauge<RowDims, ColDims>(
        self,
    ) -> MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, S>, Brand>
    where
        RowDims: DimList + CollapseUniform,
        ColDims: DimList + CollapseUniform,
        Uniform<RowDims>: UnitDiv<Uniform<ColDims>>,
        DivUnit<Uniform<RowDims>, Uniform<ColDims>>: GaugeReproduces<U>,
    {
        MixedUnitMatrix::new(self.inner)
    }
}

// Gauge stated as a single row unit and column unit, with the list *lengths*
// taken from the block's own (statically-known) shape. A uniform block over a
// `Const<RB> x Const<CB>` matrix admits only constant row/column lists, so those
// lists are fully determined by one row unit, one column unit, and the shape —
// there is nothing to repeat by hand. This trio mirrors the mixed → uniform
// `into_uniform`/`to_uniform`/`as_uniform` ownership split, and is what the
// [`gauge!`] cell wrapper expands to (bare = `into_mixed`, `copy;` = `to_mixed`,
// `view;` = `as_mixed`).
impl<U, T, const RB: usize, const CB: usize, S, Brand>
    UniformUnitMatrix<U, nalgebra::Matrix<T, nalgebra::Const<RB>, nalgebra::Const<CB>, S>, Brand>
where
    S: nalgebra::RawStorage<T, nalgebra::Const<RB>, nalgebra::Const<CB>>,
{
    /// Gauges this uniform matrix into a [`MixedUnitMatrix`] from a single
    /// `RowUnit` and `ColUnit`, consuming `self` and moving its storage into the
    /// result (zero-copy). The (necessarily constant) row and column dimension
    /// lists are built as `[RowUnit; RB]` and `[ColUnit; CB]` from the matrix's
    /// own `RB x CB` shape.
    ///
    /// This is [`gauge`](Self::gauge) with the redundancy removed: a uniform
    /// block's target lists must be uniform anyway, so only the two units carry
    /// information — the lengths are the shape, which is already known. The one
    /// soundness condition is unchanged: the entry unit `RowUnit / ColUnit` must
    /// reproduce this block's unit `U` (a wrong pair is a compile error), and the
    /// common scale shared by both units is the free gauge.
    ///
    /// For variants that leave `self` intact see [`to_mixed`](Self::to_mixed)
    /// (owned copy) and [`as_mixed`](Self::as_mixed) (borrowing view) — the same
    /// ownership split as [`into_uniform`](MixedUnitMatrix::into_uniform) /
    /// [`to_uniform`](MixedUnitMatrix::to_uniform) /
    /// [`as_uniform`](MixedUnitMatrix::as_uniform) the other way.
    pub fn into_mixed<RowUnit, ColUnit>(
        self,
    ) -> MixedUnitMatrix<
        Repeated<RowUnit, RB>,
        Repeated<ColUnit, CB>,
        nalgebra::Matrix<T, nalgebra::Const<RB>, nalgebra::Const<CB>, S>,
        Brand,
    >
    where
        nalgebra::Const<RB>: ShapeIndex,
        nalgebra::Const<CB>: ShapeIndex,
        Nat<RB>: Repeat<RowUnit>,
        Nat<CB>: Repeat<ColUnit>,
        Repeated<RowUnit, RB>: DimList,
        Repeated<ColUnit, CB>: DimList,
        RowUnit: UnitDiv<ColUnit>,
        DivUnit<RowUnit, ColUnit>: GaugeReproduces<U>,
    {
        MixedUnitMatrix::new(self.inner)
    }

    /// Copies this uniform matrix into an owned [`MixedUnitMatrix`] gauged from a
    /// single `RowUnit` and `ColUnit`, leaving `self` intact — so the uniform
    /// matrix stays usable afterward. The checked gauge is identical to
    /// [`into_mixed`](Self::into_mixed) (`RowUnit / ColUnit` must reproduce `U`);
    /// only the ownership differs, at the cost of one copy.
    ///
    /// For a zero-copy borrow instead of a copy see [`as_mixed`](Self::as_mixed);
    /// to consume and reuse the storage see [`into_mixed`](Self::into_mixed).
    pub fn to_mixed<RowUnit, ColUnit>(
        &self,
    ) -> MixedUnitMatrix<
        Repeated<RowUnit, RB>,
        Repeated<ColUnit, CB>,
        nalgebra::OMatrix<T, nalgebra::Const<RB>, nalgebra::Const<CB>>,
        Brand,
    >
    where
        T: nalgebra::Scalar,
        S: nalgebra::Storage<T, nalgebra::Const<RB>, nalgebra::Const<CB>>,
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<nalgebra::Const<RB>, nalgebra::Const<CB>>,
        nalgebra::Const<RB>: ShapeIndex,
        nalgebra::Const<CB>: ShapeIndex,
        Nat<RB>: Repeat<RowUnit>,
        Nat<CB>: Repeat<ColUnit>,
        Repeated<RowUnit, RB>: DimList,
        Repeated<ColUnit, CB>: DimList,
        RowUnit: UnitDiv<ColUnit>,
        DivUnit<RowUnit, ColUnit>: GaugeReproduces<U>,
    {
        MixedUnitMatrix::new(self.inner.clone_owned())
    }

    /// Borrows this uniform matrix as a zero-copy [`MixedUnitMatrix`] view,
    /// gauged from a single `RowUnit` and `ColUnit`; `self` is borrowed
    /// (shared) for the view's lifetime and stays readable meanwhile. This is the
    /// cheapest way to slot a uniform block into a
    /// [`block_matrix!`](crate::nalgebra::block_matrix) layout — the surrounding
    /// `hcat`/`vcat` materialize the block anyway, so the view neither copies nor
    /// consumes the source. The checked gauge is identical to
    /// [`into_mixed`](Self::into_mixed).
    ///
    /// For an owned copy see [`to_mixed`](Self::to_mixed); to consume and reuse
    /// the storage see [`into_mixed`](Self::into_mixed).
    pub fn as_mixed<RowUnit, ColUnit>(
        &self,
    ) -> MixedUnitMatrix<
        Repeated<RowUnit, RB>,
        Repeated<ColUnit, CB>,
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
        nalgebra::Const<RB>: ShapeIndex,
        nalgebra::Const<CB>: ShapeIndex,
        Nat<RB>: Repeat<RowUnit>,
        Nat<CB>: Repeat<ColUnit>,
        Repeated<RowUnit, RB>: DimList,
        Repeated<ColUnit, CB>: DimList,
        RowUnit: UnitDiv<ColUnit>,
        DivUnit<RowUnit, ColUnit>: GaugeReproduces<U>,
    {
        MixedUnitMatrix::new(self.inner.generic_view((0, 0), self.inner.shape_generic()))
    }
}

// Runtime-checked gauge for dynamically-sized uniform blocks (`Dyn` dimensions,
// e.g. a `DMatrix` produced by concatenation), mirroring
// [`MixedUnitMatrix::from_dyn`]. The unit consistency is still fully compile-time;
// only the list-length-vs-shape agreement is deferred to runtime.
impl<U, T, R, C, S, Brand> UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, S>, Brand>
where
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::RawStorage<T, R, C>,
{
    /// [`gauge`](Self::gauge) for a matrix whose shape is only known at runtime:
    /// the row/column list lengths are asserted against the actual shape at
    /// runtime (as in [`MixedUnitMatrix::from_dyn`]), while the unit-consistency
    /// check remains fully at compile time.
    ///
    /// Like [`from_dyn`](MixedUnitMatrix::from_dyn), it materializes the result
    /// into the canonical static storage: a mixed matrix's shape is a
    /// compile-time fact, so the `Dyn` uniform block is copied into the
    /// `CountedDim`-sized `SMatrix` rather than staying heap-backed.
    #[track_caller]
    pub fn gauge_dyn<RowDims, ColDims>(
        self,
    ) -> MixedUnitMatrix<
        RowDims,
        ColDims,
        nalgebra::OMatrix<T, CountedDim<RowDims>, CountedDim<ColDims>>,
        Brand,
    >
    where
        T: nalgebra::Scalar,
        RowDims: DimList + CollapseUniform + CountDim,
        ColDims: DimList + CollapseUniform + CountDim,
        Uniform<RowDims>: UnitDiv<Uniform<ColDims>>,
        DivUnit<Uniform<RowDims>, Uniform<ColDims>>: GaugeReproduces<U>,
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<CountedDim<RowDims>, CountedDim<ColDims>>,
    {
        MixedUnitMatrix::from_dyn(self.inner)
    }
}

// Unit-typed element access and bulk iteration. Because every entry shares the
// unit `U`, indices may be ordinary *runtime* values (the unit does not depend
// on which entry) — the decisive contrast with the mixed type's
// compile-time-index `get`.
impl<U, Brand, T, R, C, St> UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, St>, Brand>
where
    T: nalgebra::Scalar + Copy,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    St: nalgebra::RawStorage<T, R, C>,
    Quantity<U, T, Brand>: FromRaw<T>,
{
    /// Returns entry `(i, j)` as a `Quantity` of unit `U`.
    ///
    /// Panics if the index is out of bounds (as nalgebra indexing does).
    pub fn get(&self, i: usize, j: usize) -> Quantity<U, T, Brand> {
        <Quantity<U, T, Brand> as FromRaw<T>>::from_raw(self.inner[(i, j)])
    }

    /// Borrows entry `(i, j)` as a `Quantity` reference (zero-copy, via the
    /// `#[repr(transparent)]` layout of `Quantity`).
    pub fn get_ref(&self, i: usize, j: usize) -> &Quantity<U, T, Brand> {
        <Quantity<U, T, Brand> as FromRaw<T>>::ref_from_raw(&self.inner[(i, j)])
    }

    /// Iterates over every entry (in nalgebra's column-major order) as a
    /// `Quantity` of unit `U`.
    ///
    /// This is the bulk-iteration counterpart the mixed matrix cannot offer:
    /// with one shared entry unit there is nothing to erase.
    pub fn iter(&self) -> impl Iterator<Item = Quantity<U, T, Brand>> + '_ {
        self.inner
            .iter()
            .map(|&v| <Quantity<U, T, Brand> as FromRaw<T>>::from_raw(v))
    }
}

impl<U, Brand, T, R, C, St> UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, St>, Brand>
where
    T: nalgebra::Scalar + Copy,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    St: nalgebra::RawStorageMut<T, R, C>,
    Quantity<U, T, Brand>: FromRaw<T>,
{
    /// Mutably borrows entry `(i, j)` as a `Quantity` reference.
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut Quantity<U, T, Brand> {
        <Quantity<U, T, Brand> as FromRaw<T>>::mut_from_raw(&mut self.inner[(i, j)])
    }

    /// Overwrites entry `(i, j)` with a `Quantity` of the matrix's entry unit.
    /// A wrong-unit quantity is a compile error.
    pub fn set(&mut self, i: usize, j: usize, value: Quantity<U, T, Brand>) {
        self.inner[(i, j)] = value.unsafe_value;
    }

    /// Mutably iterates over every entry (in nalgebra's column-major order) as a
    /// `&mut Quantity` of unit `U` — the mutating counterpart of
    /// [`iter`](Self::iter), for in-place per-element updates that stay
    /// unit-checked.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Quantity<U, T, Brand>> + '_ {
        self.inner
            .iter_mut()
            .map(|v| <Quantity<U, T, Brand> as FromRaw<T>>::mut_from_raw(v))
    }

    /// Swaps rows `i` and `j` in place. Sound here — but not on a
    /// [`MixedUnitMatrix`], where each row carries its
    /// own unit and swapping would move an entry into a differently-typed slot —
    /// because every entry already shares the one unit `U`.
    pub fn swap_rows(&mut self, i: usize, j: usize) {
        self.inner.swap_rows(i, j);
    }

    /// Swaps columns `i` and `j` in place. Sound here for the same reason as
    /// [`swap_rows`](Self::swap_rows): the single shared unit `U` is
    /// position-independent.
    pub fn swap_columns(&mut self, i: usize, j: usize) {
        self.inner.swap_columns(i, j);
    }
}

// `matrix[(i, j)]` indexing, unit-typed: sound here (but not on the mixed matrix)
// because every entry shares the unit `U`, so the element type does not depend on
// which entry is indexed. `IndexMut` enables `matrix[(i, j)] = quantity` with a
// compile-time unit check on the assigned `Quantity`.
impl<U, Brand, T, R, C, St> Index<(usize, usize)>
    for UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, St>, Brand>
where
    T: nalgebra::Scalar + Copy,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    St: nalgebra::RawStorage<T, R, C>,
    Quantity<U, T, Brand>: FromRaw<T>,
{
    type Output = Quantity<U, T, Brand>;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        <Quantity<U, T, Brand> as FromRaw<T>>::ref_from_raw(&self.inner[(i, j)])
    }
}

impl<U, Brand, T, R, C, St> IndexMut<(usize, usize)>
    for UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, St>, Brand>
where
    T: nalgebra::Scalar + Copy,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    St: nalgebra::RawStorageMut<T, R, C>,
    Quantity<U, T, Brand>: FromRaw<T>,
{
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        <Quantity<U, T, Brand> as FromRaw<T>>::mut_from_raw(&mut self.inner[(i, j)])
    }
}

// The unit-erased raw slice is offered explicitly (and named as such) as an
// escape hatch: unlike the mixed matrix, a uniform matrix *could* expose a
// unit-typed slice safely thanks to `Quantity`'s transparent layout, but that
// cast is deferred to the forthcoming bytemuck-backed helpers.
impl<U, Brand, T, R, C, St> UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, St>, Brand>
where
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    St: nalgebra::RawStorage<T, R, C> + nalgebra::storage::IsContiguous,
{
    /// The underlying entries as a contiguous raw scalar slice, in column-major
    /// order. Unit-erased: the shared entry unit `U` is not represented on
    /// the returned `&[T]`.
    pub fn as_raw_slice(&self) -> &[T] {
        self.inner.as_slice()
    }
}

// Triangular extraction zeroes the strict opposite triangle. A zero is a valid
// entry at any unit, and every surviving entry keeps its unit `U`, so the result
// stays uniform in `U` — type-preserving, and equally sound on the mixed matrix.
impl<U, Brand, T, R, C, S> UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, R, C>,
{
    /// The upper-triangular part (including the diagonal), zeroing everything
    /// below it. Keeps the entry unit `U`.
    pub fn upper_triangle(&self) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, R, C>, Brand>
    where
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.upper_triangle())
    }

    /// The lower-triangular part (including the diagonal), zeroing everything
    /// above it. Keeps the entry unit `U`.
    pub fn lower_triangle(&self) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, R, C>, Brand>
    where
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.lower_triangle())
    }
}
