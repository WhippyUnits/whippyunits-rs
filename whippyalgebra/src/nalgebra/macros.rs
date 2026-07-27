//! Declarative construction and block-assembly macros for the nalgebra adapter.

#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

/// Builds a [`MixedUnitMatrix`] declaratively and unit-safely, row by row.
///
/// Each `[ … ]` group is one row, written left-to-right — the conventional
/// visual layout, matching how `nalgebra::Matrix::new` takes its arguments even
/// though the storage is column-major. Every entry is checked against its cell's
/// unit `RowDims[i] / ColDims[j]` (via [`set`](MixedUnitMatrix::set) /
/// [`IntoEntry`](crate::IntoEntry)) — pass a `Quantity` of exactly that unit (a
/// wrong unit is a *compile error*) or a bare scalar for a dimensionless cell.
/// No numeric erasure: the units are what's written.
///
/// Three forms are accepted:
///
/// - `RowDims, ColDims; …` — the common case. An `SMatrix<f64, …>` storage whose
///   shape is read straight from the dimension lists' [`LEN`](crate::DimList::LEN),
///   so there is no redundant size to repeat:
///
///   ```ignore
///   // 2×2; the two `Const<2>`s come from `MotorState::LEN`, not a literal.
///   let a = mixed_unit_matrix![MotorState, MotorState;
///       [1.0,                  g], // row 0
///       [quantity!(0.0, 1 / s), e], // row 1  (g: Quantity<s>, e: f64)
///   ];
///   ```
///
/// - `RowDims, ColDims as Scalar; …` — the same, but over an `SMatrix<Scalar, …>`
///   for a non-`f64` element type:
///
///   ```ignore
///   let a = mixed_unit_matrix![RowDims, ColDims as f32;
///       [ /* … f32-backed quantities … */ ],
///   ];
///   ```
///
/// - `MatrixType; …` — an explicit escape hatch when the storage is not a static
///   `SMatrix` at all (e.g. a dynamic dimension):
///
///   ```ignore
///   let a = mixed_unit_matrix![MixedUnitMatrix<RowDims, ColDims, DMatrix<f64>>;
///       [ /* … */ ],
///   ];
///   ```
#[macro_export]
#[doc(hidden)]
macro_rules! __wa_na_mixed_unit_matrix {
    // Internal: build an `SMatrix<$scalar, …>`-backed matrix whose shape is read
    // from the lists' `LEN`. The `@dims` tag keeps this unambiguous with the
    // public arms below (never write it directly).
    (@dims $row:ty, $col:ty, $scalar:ty; $($rows:tt),+ $(,)?) => {{
        let mut __matrix = <
            $crate::nalgebra::MixedUnitMatrix<
                $row,
                $col,
                $crate::nalgebra::__backend::SMatrix<
                    $scalar,
                    { <$row as $crate::DimList>::LEN },
                    { <$col as $crate::DimList>::LEN },
                >,
            > as ::core::default::Default
        >::default();
        $crate::__mixed_unit_matrix_rows!(__matrix, 0usize, $($rows)+);
        __matrix
    }};
    // Dimension-list form with an explicit scalar (`… as f32`).
    ($row:ty, $col:ty as $scalar:ty; $($rows:tt),+ $(,)?) => {
        $crate::nalgebra::mixed_unit_matrix![@dims $row, $col, $scalar; $($rows),+]
    };
    // Dimension-list form, defaulting the scalar to `f64`.
    ($row:ty, $col:ty; $($rows:tt),+ $(,)?) => {
        $crate::nalgebra::mixed_unit_matrix![@dims $row, $col, f64; $($rows),+]
    };
    // Explicit-type form: caller supplies the full matrix type (any storage).
    ($ty:ty; $($rows:tt),+ $(,)?) => {{
        let mut __matrix = <$ty as ::core::default::Default>::default();
        $crate::__mixed_unit_matrix_rows!(__matrix, 0usize, $($rows)+);
        __matrix
    }};
}

#[macro_export]
#[doc(hidden)]
macro_rules! __mixed_unit_matrix_rows {
    ($m:ident, $i:expr,) => {};
    ($m:ident, $i:expr, [$($e:expr),* $(,)?] $($rest:tt)*) => {
        $crate::__mixed_unit_matrix_row!($m, $i, 0usize, $($e,)*);
        $crate::__mixed_unit_matrix_rows!($m, $i + 1usize, $($rest)*);
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! __mixed_unit_matrix_row {
    ($m:ident, $i:expr, $j:expr,) => {};
    ($m:ident, $i:expr, $j:expr, $e:expr, $($rest:tt)*) => {
        $m.set::<{ $i }, { $j }>($e);
        $crate::__mixed_unit_matrix_row!($m, $i, $j + 1usize, $($rest)*);
    };
}

/// Builds a [`UniformUnitMatrix`] declaratively and unit-safely, row by row —
/// the single-unit twin of [`mixed_unit_matrix!`](crate::nalgebra::mixed_unit_matrix).
///
/// A uniform matrix carries one entry unit `U` instead of a row/column
/// dimension-list pair, so the header is a single unit expression (anything
/// [`whippyunits::unit!`] accepts — `V / A`, `m/s`, `1` for dimensionless) in
/// place of `mixed_unit_matrix!`'s `RowDims, ColDims`. The `[ … ]` rows read
/// exactly the same, and every entry is a [`Quantity`](whippyunits::Quantity) of
/// that one unit — a wrong unit is a *compile error* (checked via
/// [`set`](UniformUnitMatrix::set)). The shape is *counted from the literal*
/// (rows × the first row's width), since a uniform matrix has no dimension lists
/// to read a length off.
///
/// Two forms are accepted:
///
/// - `Unit; …` — an `SMatrix<f64, ROWS, COLS>`-backed uniform matrix:
///
///   ```ignore
///   // 4×2, every entry in Ω = V/A.
///   let g = uniform_unit_matrix![V / A;
///       [quantity!(1.0, V / A), quantity!(0.0, V / A)],
///       [quantity!(0.0, V / A), quantity!(1.0, V / A)],
///       [quantity!(1.0, V / A), quantity!(1.0, V / A)],
///       [quantity!(1.0, V / A), quantity!(-1.0, V / A)],
///   ];
///   ```
///
/// - `Unit as Scalar; …` — the same over a non-`f64` element type:
///
///   ```ignore
///   let g = uniform_unit_matrix![m / s as f32; [ /* … f32-backed quantities … */ ]];
///   ```
///
/// For a non-`SMatrix` storage (e.g. a dynamic dimension) build via
/// [`UniformUnitMatrix::from_row_slice`] / [`from_nalgebra`](UniformUnitMatrix::from_nalgebra)
/// directly — there is no single-type escape-hatch arm, because a lone type
/// token is indistinguishable from the unit form.
#[macro_export]
#[doc(hidden)]
macro_rules! __wa_na_uniform_unit_matrix {
    // Internal: build an `SMatrix<$scalar, ROWS, COLS>`-backed uniform matrix of
    // the already-resolved unit *type* `$unit`, its shape counted from the
    // literal (the `@build` tag keeps this arm unambiguous — never write it
    // directly).
    (@build $unit:ty, $scalar:ty;
        [ $($c0:expr),+ $(,)? ]
        $( , [ $($cn:expr),+ $(,)? ] )*
        $(,)?
    ) => {{
        const __UUMM_COLS: usize = $crate::__uniform_unit_matrix_count!($($c0),+);
        const __UUMM_ROWS: usize = 1usize
            $( + $crate::__uniform_unit_matrix_row_one!($($cn),+) )*;
        let mut __matrix = <
            $crate::nalgebra::UniformUnitMatrix<
                $unit,
                $crate::nalgebra::__backend::SMatrix<$scalar, __UUMM_ROWS, __UUMM_COLS>,
            > as ::core::default::Default
        >::default();
        $crate::__uniform_unit_matrix_rows!(
            __matrix, 0usize,
            [ $($c0),+ ] $( [ $($cn),+ ] )*
        );
        __matrix
    }};
    // Public entry: the header is a unit *expression* (like `quantity!`'s —
    // `V / A`, `m/s`, `1`), munched token-by-token until the `;` (or `as
    // Scalar;`) that separates it from the rows, then resolved with `unit!`.
    ($($rest:tt)+) => {
        $crate::__uniform_unit_matrix_split!([] $($rest)+)
    };
}

/// Splits a `uniform_unit_matrix!` invocation into its unit-expression header
/// and its row literal: munch header tokens into the bracketed accumulator until
/// the separating `;` (defaulting the scalar to `f64`) or `as Scalar ;`. Unit
/// expressions never contain `;` or `as`, so the split is unambiguous.
#[macro_export]
#[doc(hidden)]
macro_rules! __uniform_unit_matrix_split {
    // `… as Scalar; rows`
    ([$($u:tt)+] as $scalar:ty; $($rows:tt),+ $(,)?) => {
        $crate::nalgebra::uniform_unit_matrix![
            @build $crate::__reexport::whippyunits::unit!($($u)+), $scalar; $($rows),+
        ]
    };
    // `…; rows` (scalar defaults to `f64`)
    ([$($u:tt)+] ; $($rows:tt),+ $(,)?) => {
        $crate::nalgebra::uniform_unit_matrix![
            @build $crate::__reexport::whippyunits::unit!($($u)+), f64; $($rows),+
        ]
    };
    // Munch one more token into the unit-expression accumulator.
    ([$($u:tt)*] $head:tt $($rest:tt)*) => {
        $crate::__uniform_unit_matrix_split!([$($u)* $head] $($rest)*)
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! __uniform_unit_matrix_count {
    () => { 0usize };
    ($head:expr $(, $tail:expr)* $(,)?) => {
        1usize + $crate::__uniform_unit_matrix_count!($($tail),*)
    };
}

/// Contributes `1` per extra row group, ignoring the entries themselves (they
/// are counted per row by [`__uniform_unit_matrix_count`]).
#[macro_export]
#[doc(hidden)]
macro_rules! __uniform_unit_matrix_row_one {
    ($($e:expr),* $(,)?) => {
        1usize
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! __uniform_unit_matrix_rows {
    ($m:ident, $i:expr,) => {};
    ($m:ident, $i:expr, [ $($e:expr),* $(,)? ] $($rest:tt)*) => {
        $crate::__uniform_unit_matrix_row!($m, $i, 0usize, $($e,)*);
        $crate::__uniform_unit_matrix_rows!($m, $i + 1usize, $($rest)*);
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! __uniform_unit_matrix_row {
    ($m:ident, $i:expr, $j:expr,) => {};
    ($m:ident, $i:expr, $j:expr, $e:expr, $($rest:tt)*) => {
        $m.set($i, $j, $e);
        $crate::__uniform_unit_matrix_row!($m, $i, $j + 1usize, $($rest)*);
    };
}

/// A zero block for [`block_matrix!`](crate::nalgebra::block_matrix), sized from its
/// dimension lists.
///
/// `zeros![RowDims, ColDims]` is a `DimList::LEN`-by-`DimList::LEN` all-zeros
/// matrix over `f64`, typed as `⟨RowDims, ColDims⟩`; `zeros![RowDims, ColDims as
/// T]` chooses a different element type. It exists so an all-zero cell in a
/// block layout can state its unit spaces tersely, rather than as a scalar `0`
/// that carries no dimensional information.
///
/// The storage is a statically-sized `SMatrix` whose dimensions are counted off
/// the lists with [`CountedDim`](crate::nalgebra::CountedDim), so a zero cell composes on
/// the static block-assembly path (≤127 per axis) just like any other block.
#[macro_export]
#[doc(hidden)]
macro_rules! __wa_na_zeros {
    ($row:ty, $col:ty as $scalar:ty) => {
        $crate::nalgebra::MixedUnitMatrix::<
            $row,
            $col,
            $crate::nalgebra::__backend::OMatrix<
                $scalar,
                $crate::nalgebra::CountedDim<$row>,
                $crate::nalgebra::CountedDim<$col>,
            >,
        >::from_nalgebra(
            // Build through the allocator (`zeros_generic`) rather than array
            // `Default`: `[[T; M]; N]: Default` doesn't hold for a *generic* `N`.
            $crate::nalgebra::__backend::OMatrix::<
                $scalar,
                $crate::nalgebra::CountedDim<$row>,
                $crate::nalgebra::CountedDim<$col>,
            >::zeros_generic(
                <$crate::nalgebra::CountedDim<$row> as $crate::nalgebra::__backend::Dim>::from_usize(
                    <$row as $crate::DimList>::LEN,
                ),
                <$crate::nalgebra::CountedDim<$col> as $crate::nalgebra::__backend::Dim>::from_usize(
                    <$col as $crate::DimList>::LEN,
                ),
            ),
        )
    };
    ($row:ty, $col:ty) => {
        $crate::nalgebra::zeros![$row, $col as f64]
    };
}

/// Assembles a partitioned matrix from a grid of typed blocks, reading as a
/// block-matrix literal.
///
/// Each row is a bracketed list of block expressions; rows are separated by
/// commas. Blocks in the same grid row must share a row space and blocks in the
/// same grid column must share a column space — enforced by
/// [`hcat`](crate::nalgebra::MixedUnitMatrix::hcat) / [`vcat`](crate::nalgebra::MixedUnitMatrix::vcat)
/// unification, so a misaligned block is a compile error. The result's row and
/// column dimension lists are the concatenations of the block rows'/columns'
/// lists; its storage tracks the block shapes — see [Storage](#storage) below.
///
/// All-zero cells are written with [`zeros!`](crate::nalgebra::zeros), which states their
/// unit spaces (a bare `0` couldn't):
///
/// ```ignore
/// // Van Loan block: [[A, B], [0, 0]] over Z = State ⊕ Input.
/// let m = block_matrix![
///     [a,                    b                    ],
///     [zeros![Input, State], zeros![Input, Input] ],
/// ];
/// ```
/// # Storage
///
/// The result's storage is inferred from the blocks: statically-sized blocks
/// (`SMatrix`, the usual case — every dimension list has a compile-time length)
/// concatenate to an owned `SMatrix`, tracking the joined shape at the type
/// level via nalgebra's [`DimAdd`](nalgebra::DimAdd) (≤127 per axis); a `Dyn`
/// block anywhere makes the result a `DMatrix`. This is automatic — no mode to
/// choose.
///
/// In generic code the static path needs the `DimAdd`/allocator obligations
/// stated as `where` bounds (they propagate like [`ShapeIndex`](crate::ShapeIndex)).
/// The [`#[generic_block]`](crate::nalgebra::generic_block) attribute writes them from the
/// grid shape so they need not be transcribed by hand.
#[macro_export]
#[doc(hidden)]
macro_rules! __wa_na_block_matrix {
    // Uniform arm: every cell is a `UniformUnitMatrix` of the *same* entry unit
    // `U`, so the assembly stays uniform in `U` (there are no per-block row/
    // column gauges to reconcile — a mismatched cell unit is a compile error).
    // The right-associative `hcat`/`vcat` resolve to
    // [`UniformUnitMatrix`](crate::nalgebra::UniformUnitMatrix)'s own by the cell type; the
    // `uniform;` tag is there to state the intent and to pair with
    // `#[generic_block(uniform, ..)]`, which writes this assembly's `where`
    // bounds in generic code.
    [ uniform; $( [ $($cell:expr),+ $(,)? ] ),+ $(,)? ] => {
        $crate::__block_vcat!( $( $crate::__block_hcat!($($cell),+) ),+ )
    };
    [ $( [ $($cell:expr),+ $(,)? ] ),+ $(,)? ] => {
        $crate::__block_vcat!( $( $crate::__block_hcat!($($cell),+) ),+ )
    };
}

/// Expresses a [`UniformUnitMatrix`](crate::nalgebra::UniformUnitMatrix) as a
/// [`MixedUnitMatrix`](crate::nalgebra::MixedUnitMatrix) by assigning it a
/// row/column gauge — a choice of per-axis units whose quotient is the uniform
/// matrix's single entry unit.
///
/// Only the two units need stating; there is no need to pass lists with repeated
/// entries. The quotient `RowUnit / ColUnit` must equal `u`'s unit `U` (a wrong
/// pair is a compile error).
///
/// # Ownership
///
/// Three variants exist, with different semantics:
///
/// - `gauge!(u => R, C)` — [`into_mixed`](crate::nalgebra::UniformUnitMatrix::into_mixed) (move semantics)
/// - `gauge!(copy; u => R, C)` — [`to_mixed`](crate::nalgebra::UniformUnitMatrix::to_mixed) (copy semantics)
/// - `gauge!(view; u => R, C)` — [`as_mixed`](crate::nalgebra::UniformUnitMatrix::as_mixed) (borrow semantics)
///
/// ```
/// use whippyalgebra::dims;
/// use whippyalgebra::nalgebra::{gauge, uniform_unit_matrix, MixedUnitMatrix};
/// use whippyunits::{qty, quantity};
///
/// let g = uniform_unit_matrix![m / s;
///     [quantity!(10.0, m / s)],
///     [quantity!(20.0, m / s)],
/// ];
///
/// // `copy;` and `view;` both gauge without consuming `g`.
/// let copied: MixedUnitMatrix<dims![m, m], dims![s], _> = gauge!(copy; g => m, s);
/// let viewed: MixedUnitMatrix<dims![m, m], dims![s], _> = gauge!(view; g => m, s);
/// assert_eq!(copied.get::<0, 0>().unsafe_value, 10.0);
/// assert_eq!(viewed.get::<1, 0>().unsafe_value, 20.0);
///
/// // ...so `g` is still readable afterward (a bare `gauge!(g => m, s)` would
/// // have moved it).
/// let still: qty!(m / s) = g.get(0, 0);
/// assert_eq!(still.unsafe_value, 10.0);
/// ```
///
/// # Inside `block_matrix!`
///
/// Introducing a gauge allows a uniform block to be placed in a block matrix layout that builds a mixed matrix.
///
/// ```
/// use whippyalgebra::dims;
/// use whippyalgebra::nalgebra::{
///     block_matrix, gauge, mixed_unit_matrix, uniform_unit_matrix, zeros, MixedUnitMatrix,
/// };
/// use whippyunits::{qty, quantity};
///
/// // A mixed 2×2 block over rows `[m, m]`, cols `[s, s]` (every entry `m/s`)...
/// let a = mixed_unit_matrix![dims![m, m], dims![s, s];
///     [quantity!(1.0, m / s), quantity!(2.0, m / s)],
///     [quantity!(3.0, m / s), quantity!(4.0, m / s)],
/// ];
/// // ...and a *uniform* 2×1 block declared with the single entry unit `m/s`,
/// // carrying no row/column split of its own.
/// let g = uniform_unit_matrix![m / s;
///     [quantity!(10.0, m / s)],
///     [quantity!(20.0, m / s)],
/// ];
///
/// // `gauge!(g => m, s)` gauges `g` as rows `m` / col `s` — quotient `m/s`,
/// // matching its unit — so it lines up with `a` (shared rows `[m, m]`) and the
/// // filler zero row below (shared cols `[s, s, s]`). A wrong pair (say
/// // `=> s, s`) would be a compile error.
/// let m: MixedUnitMatrix<dims![m, m, rot], dims![s, s, s], _> = block_matrix![
///     [a,                               gauge!(g => m, s)           ],
///     [zeros![dims![rot], dims![s, s]], zeros![dims![rot], dims![s]]],
/// ];
///
/// // The gauged block occupies the top-right column, carrying the layout's entry
/// // unit (row `m` / col `s` = `m/s`) and `g`'s original values.
/// let g_top: qty!(m / s) = m.get::<0, 2>();
/// let g_bot: qty!(m / s) = m.get::<1, 2>();
/// assert_eq!(g_top.unsafe_value, 10.0);
/// assert_eq!(g_bot.unsafe_value, 20.0);
///
/// // `a` still reads back at `m/s` from the top-left.
/// let a00: qty!(m / s) = m.get::<0, 0>();
/// assert_eq!(a00.unsafe_value, 1.0);
/// ```
#[macro_export]
#[doc(hidden)]
macro_rules! __wa_na_gauge {
    // Ownership selectors, mirroring the mixed → uniform trio. A leading keyword
    // picks which gauge method the cell expands to; bare (no keyword) consumes
    // the source, matching how a plain block cell is moved in.
    (copy; $src:expr => $($units:tt)+) => {
        $crate::__gauge_split!(to_mixed; ($src), [] $($units)+)
    };
    (view; $src:expr => $($units:tt)+) => {
        $crate::__gauge_split!(as_mixed; ($src), [] $($units)+)
    };
    // Munch the row-unit tokens (bracket-delimited accumulator) until the
    // separating comma; whatever follows is the column unit.
    ($src:expr => $($units:tt)+) => {
        $crate::__gauge_split!(into_mixed; ($src), [] $($units)+)
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! __gauge_split {
    // Comma reached: the bracketed accumulator is the row unit, the rest the
    // column unit. Turn each into a `Unit` type via whippyunits' `unit!` and
    // gauge through the chosen method (`into_mixed`/`to_mixed`/`as_mixed`).
    ($method:ident; ($src:expr), [$($row:tt)+] , $($col:tt)+) => {
        ($src).$method::<
            $crate::__reexport::whippyunits::unit!($($row)+),
            $crate::__reexport::whippyunits::unit!($($col)+)
        >()
    };
    // Otherwise munch one more token into the row accumulator.
    ($method:ident; ($src:expr), [$($row:tt)*] $head:tt $($rest:tt)*) => {
        $crate::__gauge_split!($method; ($src), [$($row)* $head] $($rest)*)
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! __block_hcat {
    ($only:expr) => { $only };
    ($first:expr, $($rest:expr),+) => {
        $first.hcat(&$crate::__block_hcat!($($rest),+))
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! __block_vcat {
    ($only:expr) => { $only };
    ($first:expr, $($rest:expr),+) => {
        $first.vcat(&$crate::__block_vcat!($($rest),+))
    };
}

/// Destructures a matrix into its typed sub-blocks, the reading inverse of
/// [`block_matrix!`](crate::nalgebra::block_matrix). The grid is written as a block-matrix
/// literal where every cell states its own `(height, width)`: a named cell
/// `id(h, w)` binds that block, and a standin `_(h, w)` holds a slot that isn't
/// being read. The standins are redundant with the named cells' sizes, but they
/// make the layout read as an actual partition and let the macro recover every
/// block's `(row, col)` offset:
///
/// ```ignore
/// // ϕ = [[A_d, B_d], [_, _]] over Z = State ⊕ Input; read the top row back out.
/// unblock_matrix!(phi => [
///     [a_d(N, N), b_d(N, M)],
///     [_(M, N),   _(M, M)  ],
/// ]);
/// // a_d : ⟨State, State⟩,  b_d : ⟨State, Input⟩
/// ```
///
/// # Reduction mode
///
/// An optional leading mode keyword picks how each named block is typed:
///
/// - `mixed` (the default, i.e. no keyword): each block is read through
///   [`block`](crate::nalgebra::MixedUnitMatrix::block) and stays a `MixedUnitMatrix`,
///   retaining its gauge (the exact `RowDims`/`ColDims` sublists). This is
///   lossless — the block recomposes later at its original gauge with no
///   re-entry — and it is the only mode that works in generic code, where the
///   uniform-vs-mixed decision can't be resolved while the dimension lists are
///   type parameters. To treat an extracted block as uniform for bulk ops
///   without losing the gauge, call
///   [`to_uniform`](crate::nalgebra::MixedUnitMatrix::to_uniform) /
///   [`as_uniform`](crate::nalgebra::MixedUnitMatrix::as_uniform) on it.
/// - `reduce_uniform`: each block is read through
///   [`block_auto`](crate::nalgebra::MixedUnitMatrix::block_auto), so a block whose row and
///   column sublists are each uniform comes back as a
///   [`UniformUnitMatrix`](crate::nalgebra::UniformUnitMatrix) (a single shared entry unit,
///   gauge erased) and every other block as a `MixedUnitMatrix`. The choice is
///   made from the sublists at the type level, so it requires the source's
///   dimension lists to be concrete; ask for it when you want the erased uniform
///   form. Erasing the gauge here is what later forces you to re-enter it (via
///   [`gauge!`](crate::nalgebra::gauge)) to recompose; the default `mixed` mode
///   carries the gauge along, so recomposition needs no re-tagging.
///
/// ```ignore
/// unblock_matrix!(m => [ /* every block stays MixedUnitMatrix (gauge kept)   */ ]);
/// unblock_matrix!(reduce_uniform; m => [ /* uniform blocks -> UniformUnitMatrix */ ]);
/// ```
///
/// # Owned copies vs. views
///
/// By default each named block is an *owned* copy of `$src`'s storage (read
/// through [`block`](crate::nalgebra::MixedUnitMatrix::block) /
/// [`block_auto`](crate::nalgebra::MixedUnitMatrix::block_auto)). A leading
/// `views` keyword instead binds each block as a zero-copy *view* that shares
/// `$src`'s storage (read through
/// [`block_view`](crate::nalgebra::MixedUnitMatrix::block_view) /
/// [`block_view_auto`](crate::nalgebra::MixedUnitMatrix::block_view_auto)): `$src`
/// is borrowed for as long as any binding lives and stays read-only meanwhile,
/// and because the borrows are shared any number of blocks may be viewed at
/// once. The grid syntax (`id(h, w)` for a read cell, `_(h, w)` for a standin)
/// is unchanged; only the storage differs. `views` precedes the reduction
/// keyword and composes with it — `views`, `views mixed`, `views
/// reduce_uniform`:
///
/// ```ignore
/// // Read the top row of ϕ as views instead of copies; both borrow ϕ at once.
/// unblock_matrix!(views; phi => [
///     [a_d(N, N), b_d(N, M)],
///     [_(M, N),   _(M, M)  ],
/// ]);
/// unblock_matrix!(views reduce_uniform; phi => [ /* borrowed + uniform reduction */ ]);
/// ```
///
/// The sizes are const tokens (`N`, `M`, literals), not `DimList::LEN`s. A
/// block's offset is the sum of the preceding blocks' sizes along that axis;
/// that sum is formed at the type level (`Sum<Nat<N>, Nat<M>>`, via
/// [`block_off`](crate::nalgebra::MixedUnitMatrix::block_off)), so it never needs
/// `generic_const_exprs` and a partition of any width/height works — even
/// over generic const shapes, not just literals. (A generic partition three or
/// more blocks wide does have to prove the offset sums are well-formed — e.g.
/// `Nat<N>: Add<Nat<M>>` and that the sum is `Unsigned` — as ordinary `where`
/// bounds; concrete sizes discharge them automatically.)
#[macro_export]
#[doc(hidden)]
macro_rules! __wa_na_unblock_matrix {
    // Zero-copy views (leading `views`), optionally carrying a reduction mode.
    // Each named cell borrows `$src` through the `block_view_*` extractors
    // instead of copying; `views` precedes the reduction keyword.
    (views; $src:expr => [ $( [ $($cells:tt)* ] ),+ $(,)? ]) => {
        $crate::__unblock_rows!(block_view_off; $src, []; $( [ $($cells)* ] )+);
    };
    (views mixed; $src:expr => [ $( [ $($cells:tt)* ] ),+ $(,)? ]) => {
        $crate::__unblock_rows!(block_view_off; $src, []; $( [ $($cells)* ] )+);
    };
    (views reduce_uniform; $src:expr => [ $( [ $($cells:tt)* ] ),+ $(,)? ]) => {
        $crate::__unblock_rows!(block_view_auto_off; $src, []; $( [ $($cells)* ] )+);
    };
    // Owned copies (default), optionally carrying a reduction mode.
    (mixed; $src:expr => [ $( [ $($cells:tt)* ] ),+ $(,)? ]) => {
        $crate::__unblock_rows!(block_off; $src, []; $( [ $($cells)* ] )+);
    };
    (reduce_uniform; $src:expr => [ $( [ $($cells:tt)* ] ),+ $(,)? ]) => {
        $crate::__unblock_rows!(block_auto_off; $src, []; $( [ $($cells)* ] )+);
    };
    ($src:expr => [ $( [ $($cells:tt)* ] ),+ $(,)? ]) => {
        $crate::__unblock_rows!(block_off; $src, []; $( [ $($cells)* ] )+);
    };
}

// The extraction machinery is shared across `unblock_matrix!`'s ownership
// variants and reduction modes: the leading `$m:ident` is the
// (type-level-offset) method each cell is read through (`block_off`/
// `block_auto_off` for owned copies, `block_view_off`/`block_view_auto_off` for
// borrowed views); it is threaded unchanged down to `__unblock_extract`.
#[macro_export]
#[doc(hidden)]
macro_rules! __unblock_rows {
    // No more rows.
    ($m:ident; $src:expr, $rprec:tt;) => {};
    // Peek the first cell's height as this row's height, emit the row at the
    // accumulated row offset, then recurse with that height appended.
    ($m:ident; $src:expr, [$($rp:tt)*]; [ $head:tt ( $rh:tt , $cb0:tt ) $($crest:tt)* ] $($rest:tt)*) => {
        $crate::__unblock_row!($m; $src, [$($rp)*], []; $head ( $rh , $cb0 ) $($crest)*);
        $crate::__unblock_rows!($m; $src, [$($rp)* $rh]; $($rest)*);
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! __unblock_row {
    // No more cells in this row.
    ($m:ident; $src:expr, $roff:tt, $cprec:tt;) => {};
    // Named cell `id(rb, cb)`: bind the block at (row offset, accumulated col
    // offset), then recurse with this column's width appended.
    ($m:ident; $src:expr, [$($ro:tt)*], [$($cp:tt)*]; $id:ident ( $rb:tt , $cb:tt ) $(, $($rest:tt)*)?) => {
        $crate::__unblock_extract!($m; $src, $id, [$($ro)*], [$($cp)*], $rb, $cb);
        $crate::__unblock_row!($m; $src, [$($ro)*], [$($cp)* $cb]; $($($rest)*)?);
    };
    // Standin cell `_(rb, cb)`: binds nothing, only advances the column offset.
    ($m:ident; $src:expr, [$($ro:tt)*], [$($cp:tt)*]; _ ( $rb:tt , $cb:tt ) $(, $($rest:tt)*)?) => {
        $crate::__unblock_row!($m; $src, [$($ro)*], [$($cp)* $cb]; $($($rest)*)?);
    };
}

// Emits `let $id = $src.$m::<RowOff, ColOff, $rh, $cb>();`, forming each offset
// as a *type-level* natural (via `__nat_sum!`) from its preceding-sizes list.
// Because the offset is a type — `Sum<Nat<N>, Nat<M>>`, not `{ N + M }` — the
// sum is legal even when the sizes are generic const parameters, so a partition
// of any width/height works (not just two). The `$m` methods are the
// type-level-offset extractors (`block_off` & friends).
#[macro_export]
#[doc(hidden)]
macro_rules! __unblock_extract {
    ($m:ident; $src:expr, $id:ident, [$($ro:tt)*], [$($co:tt)*], $rh:tt, $cb:tt) => {
        let $id = $src.$m::<$crate::__nat_sum!($($ro)*), $crate::__nat_sum!($($co)*), $rh, $cb>();
    };
}

// Folds a (possibly empty) list of const-size *tokens* into the type-level
// natural that is their sum: the offset of a block is the sum of the sizes of
// the blocks preceding it along that axis. Empty is zero (`Nat<0>`), a single
// size `s` is `Nat<s>`, and a longer list folds right through typenum's
// `Add`. Each summand is `Nat<token>`, so a literal (`2`) and a generic const
// parameter (`N`) are treated uniformly — the sum stays in type position and
// never needs `generic_const_exprs`.
#[macro_export]
#[doc(hidden)]
macro_rules! __nat_sum {
    () => { $crate::Nat<0> };
    ($only:tt) => { $crate::Nat<$only> };
    ($first:tt $($rest:tt)+) => {
        <$crate::Nat<$first> as ::core::ops::Add<$crate::__nat_sum!($($rest)+)>>::Output
    };
}
