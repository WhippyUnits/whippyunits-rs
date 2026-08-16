//! Procedural macros for [`whippyalgebra`](https://docs.rs/whippyalgebra).
//!
//! Two exports:
//!
//! - [`macro@generic_block`]: writes the obligations for assembling and
//!   slicing one or more partitioned matrices in generic code — the nalgebra
//!   storage-allocator bounds, the `ShapeIndex` shape bounds that block-slicing
//!   needs, and, when an assembly is square, the shape bounds that square
//!   operations (`.exp()`/inverse/solve) require — straight into a function's
//!   `where` clause, given each block grid's shape.
//! - [`macro@generic_matrix`]: writes the shape well-formedness of a set of
//!   matrix sizes (`SquareDim`/`ShapeIndex`, plus `DimList` for named axes — or
//!   the `Nat<_>: Repeat<_>` constructor bound for a uniform `[Element]` axis)
//!   into an item's `where` clause.
//!
//! Both accept bare const-generic parameters for sizes (`rows(N; M)`,
//! `rows(N, State)`) — no need to spell `Const<_>` out.
//!
//! Both also accept an opt-in `decompose` keyword that additionally emits the
//! Householder-reduction workspace pile: the reducer `DimSub<U1>` plus
//! nalgebra's reduced-dimension (`DimMinimum` / `DimDiff<…, U1>`) scratch
//! allocators. This is exactly — and only — what the reduction-based
//! decompositions need beyond a plain square operation:
//!
//! - needs `decompose` (Householder scratch): `svd` / `generalized_svd`,
//!   `pseudo_inverse`, `eigenvalues` / `symmetric_eigen` / `schur` /
//!   `hessenberg` / `symmetric_tridiagonalize` / `bidiagonalize`,
//!   `generalized_symmetric_eigen`, and column-pivoted QR (`col_piv_qr`);
//! - does not need `decompose` — covered by the detected square-op bounds
//!   alone: `solve` / `determinant` / `try_inverse`, plain `lu` / `full_piv_lu`
//!   / `qr`, `cholesky`, and `udu` (they ask only for `DimMin<D, Output = D>`
//!   and basic storage, which squareness detection already supplies).
//!
//! The two mechanisms compose rather than overlap: the square branch of the
//! `decompose` pile deliberately omits `DimMin<D, Output = D>` (supplied by the
//! detected `SquareDim` / square-block bound) and contributes only the reducer
//! and scratch that no shape fact implies. Unlike the square-operation bounds —
//! detected from the shape — whether a body reaches for a reduction is
//! invisible from the signature, so this one must be flagged.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    parenthesized,
    parse::{Parse, ParseStream},
    Ident, Token, Type, WherePredicate,
};

/// The block grid's shape. Each axis (`rows`/`cols`) is a `;`-separated list of
/// tracks, and each track pairs a block size (one per grid row/column) with,
/// optionally, the dimension sublist occupying that track:
/// `rows(N, State; M, Input)`. Naming the sublists unlocks the partition
/// (`Concat`/`Take`/`Drop`) bounds; sizes-only (`rows(N; M)`) emits just the
/// shape/allocator bounds. Whether the assembled matrix is square — and thus
/// admits the square operations (`.exp()`/inverse/solve) at the shape level —
/// is detected from the shape, not flagged. (Squareness is only the shape
/// half; whether the map is dimensionally an endomorphism is a separate unit
/// bound the consuming code carries.)
struct Shape {
    rows: Vec<Type>,
    cols: Vec<Type>,
    row_dims: Option<Vec<Type>>,
    col_dims: Option<Vec<Type>>,
    /// The `decompose` opt-in: also emit the Householder-reduction *workspace*
    /// pile (the `DimSub<U1>` reducer + `DimMinimum`/`DimDiff` scratch) for the
    /// assembled matrix — what `svd`/`pseudo_inverse`/`eigenvalues`/`schur`/… and
    /// `col_piv_qr` need, but not plain `lu`/`qr`/`cholesky`/`solve` (those
    /// ride on the detected square-op bounds). Off by default — whether the body
    /// reduces the assembly is invisible from the shape, so unlike the square-op
    /// bounds it can't be detected and must be flagged.
    decompose: bool,
}

/// Parse one axis group `(size[, dims]; size[, dims]; …)`. Each `;`-separated
/// track is a block size — a bare const-generic param (wrapped in `Const<_>`) or
/// an explicit dim — optionally paired, after a `,`, with the dimension list
/// occupying that track. Sublists are all-or-nothing across the axis (either
/// every track names one, or none do). Returns the sizes and the sublists (when
/// present).
fn parse_tracks(input: ParseStream, axis: &str) -> syn::Result<(Vec<Type>, Option<Vec<Type>>)> {
    let content;
    parenthesized!(content in input);
    let mut sizes: Vec<Type> = Vec::new();
    let mut dims: Vec<Type> = Vec::new();
    while !content.is_empty() {
        let size: Type = content.parse()?;
        sizes.push(norm_dim(size));
        // Optional `, <dims>` for this track. Unlike `generic_matrix`, the uniform
        // `[Element]` spelling is *not* accepted here: a partitioned track's
        // disassembly obligation pins `Take<_, Out = Repeated<Element, N>>`, which
        // forces the trait solver to normalize the repeated-list projection at a
        // generic length — impossible, since no `Repeat` impl matches the opaque
        // `Nat<N>`. (A `generic_matrix` axis only ever *names* that projection, so
        // there it works.) Reject it early with that guidance.
        if content.peek(Token![,]) {
            content.parse::<Token![,]>()?;
            let dim: Type = content.parse()?;
            if matches!(dim, Type::Slice(_)) {
                return Err(syn::Error::new_spanned(
                    dim,
                    "`generic_block` does not support the uniform `[Element]` axis \
                     spelling: a partitioned track's disassembly obligation \
                     (`Take<_, Out = Repeated<Element, N>>`) would require \
                     normalizing the repeated-list projection at a generic length, \
                     which the trait solver cannot do. Use `#[generic_matrix]` for a \
                     uniform axis, or name this track's sublist as an opaque type \
                     parameter.",
                ));
            }
            dims.push(dim);
        }
        // Tracks are separated by `;`; anything else here is a mistake.
        if content.peek(Token![;]) {
            content.parse::<Token![;]>()?;
        } else if !content.is_empty() {
            return Err(content.error(
                "expected `;` between tracks (or `,` before this track's dimension list)",
            ));
        }
    }
    if !dims.is_empty() && dims.len() != sizes.len() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!(
                "`{axis}`: {} of {} tracks name a dimension list; give one for \
                 every track (`size, dims`) or none",
                dims.len(),
                sizes.len()
            ),
        ));
    }
    let dims = if dims.is_empty() { None } else { Some(dims) };
    Ok((sizes, dims))
}

impl Parse for Shape {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut rows: Option<(Vec<Type>, Option<Vec<Type>>)> = None;
        let mut cols: Option<(Vec<Type>, Option<Vec<Type>>)> = None;
        let mut decompose = false;
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            match key.to_string().as_str() {
                "rows" => rows = Some(parse_tracks(input, "rows")?),
                "cols" => cols = Some(parse_tracks(input, "cols")?),
                "decompose" => decompose = true,
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unexpected `{other}`; expected `rows`, `cols`, or `decompose`"),
                    ))
                }
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }
        let (rows, row_dims) = rows
            .ok_or_else(|| syn::Error::new(proc_macro2::Span::call_site(), "missing `rows(...)`"))?;
        let (cols, col_dims) = cols
            .ok_or_else(|| syn::Error::new(proc_macro2::Span::call_site(), "missing `cols(...)`"))?;
        Ok(Shape {
            rows,
            cols,
            row_dims,
            col_dims,
            decompose,
        })
    }
}

/// One or more block grids for a single `#[generic_block]` attribute.
///
/// Two accepted forms:
/// - single (the common case, backwards compatible): the `rows(..)`/
///   `cols(..)`/… keys directly, describing one grid;
/// - multiple: one or more `block(..)` groups, each wrapping the keys of one
///   grid — for a function that assembles (and/or slices) several block matrices,
///   so all their obligations land in one `where` clause.
///
/// A leading `uniform` keyword switches every grid to the
/// [`UniformUnitMatrix`] assembly spec: emit only the nalgebra-level assembly
/// pile (the `DimAdd`s and storage allocators of `hcat`/`vcat`, plus the
/// detected square-op / `decompose` bounds), and drop the mixed-only slicing
/// plumbing — no `ShapeIndex`, no partition (`Concat`/`Take`/`Drop`/`DimList`),
/// no `CountDim` ties. Naming a sublist is then an error.
struct Shapes {
    uniform: bool,
    shapes: Vec<Shape>,
}

impl Parse for Shapes {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // An optional leading `uniform` keyword selects the single-unit assembly.
        let uniform = {
            let fork = input.fork();
            fork.parse::<Ident>().map(|i| i == "uniform").unwrap_or(false)
        };
        if uniform {
            input.parse::<Ident>()?; // consume `uniform`
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }
        // The multi-grid form leads with the `block` keyword; anything else is
        // the single-grid form (a bare `rows(..)`/`cols(..)/…` list).
        let is_block = {
            let fork = input.fork();
            fork.parse::<Ident>().map(|i| i == "block").unwrap_or(false)
        };
        let mut shapes = Vec::new();
        if is_block {
            while !input.is_empty() {
                let kw: Ident = input.parse()?;
                if kw != "block" {
                    return Err(syn::Error::new(
                        kw.span(),
                        "expected `block(...)`; a `block(...)` group cannot be \
                         mixed with bare `rows(...)` keys",
                    ));
                }
                let content;
                parenthesized!(content in input);
                shapes.push(content.parse::<Shape>()?);
                if input.peek(Token![,]) {
                    input.parse::<Token![,]>()?;
                }
            }
        } else {
            shapes.push(input.parse::<Shape>()?);
        }
        if shapes.is_empty() {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "generic_block: expected at least one block grid",
            ));
        }
        // A uniform assembly carries one unit and no dimension lists, so naming a
        // sublist on any track is meaningless — reject it with guidance rather
        // than silently ignore it.
        if uniform {
            for s in &shapes {
                if s.row_dims.is_some() || s.col_dims.is_some() {
                    return Err(syn::Error::new(
                        proc_macro2::Span::call_site(),
                        "generic_block(uniform, ..): a uniform assembly has no \
                         dimension lists — drop the per-track sublists and write \
                         bare sizes, e.g. `uniform, rows(N; M), cols(N; M)`",
                    ));
                }
            }
        }
        Ok(Shapes { uniform, shapes })
    }
}

/// `whippyalgebra::nalgebra::__backend` — the nalgebra crate re-exported by its
/// adapter, so the emitted bounds resolve wherever the attribute is used (the
/// caller needn't have `nalgebra` in scope, and a future backend keeps its own
/// `__backend`).
fn na() -> TokenStream2 {
    quote! { ::whippyalgebra::nalgebra::__backend }
}

/// Normalize a shape dim to a nalgebra dimension type. A bare const-generic
/// parameter — a single-segment path with no arguments, e.g. `N` — is wrapped as
/// `nalgebra::Const<N>`, so callers may write `rows(N, M)` / `sizes((N, N))`
/// instead of spelling `Const<_>` out (and needn't have `Const` in scope). A
/// path that already carries arguments (`Const<N>`, `nalgebra::Const<N>`) or is
/// multi-segment is left untouched. Applied only to the sizes — never to the
/// dimension-list args (`row_dims`/`col_dims`/`dims`), which are genuine types.
fn norm_dim(ty: Type) -> Type {
    if let Type::Path(tp) = &ty
        && tp.qself.is_none() && tp.path.leading_colon.is_none() && tp.path.segments.len() == 1 {
            let seg = &tp.path.segments[0];
            if seg.arguments.is_empty() {
                let ident = &seg.ident;
                let na = na();
                return syn::parse2(quote! { #na::Const<#ident> })
                    .expect("norm_dim: `Const<_>` is a valid type");
            }
        }
    ty
}

/// The dimension sublist occupying one `generic_matrix` axis, in the two
/// spellings the `dims` slot accepts:
///
/// - [`List`](AxisDim::List) — a written-out axis type (`State`, `Readings<N>`,
///   …), taken opaquely: it only needs the generic `DimList` well-formedness
///   fact.
/// - [`Uniform`](AxisDim::Uniform) — the uniform spelling `[Element]` (a slice
///   type), meaning "this axis is `N` copies of the single unit `Element`", i.e.
///   [`Repeated<Element, N>`](../whippyalgebra/type.Repeated.html) with `N` the
///   axis's own size. Naming the element is what lets the attribute synthesize
///   the constructor bound `<Const<N> as ShapeIndex>::Nat: Repeat<Element>` (=
///   `Nat<N>: Repeat<Element>`) that makes the generic uniform axis nameable —
///   the one hand-written bound a `Repeated<_, N>` axis used to force.
///
/// (`generic_block` deliberately does not accept the uniform spelling: its
/// partition obligations would need to normalize the `Repeated<…>` projection at
/// a generic length, which the trait solver cannot do — so it is rejected at
/// parse time. See `parse_tracks`.)
enum AxisDim {
    List(Type),
    Uniform(Type),
}

/// Classify a parsed `generic_matrix` `dims` type: a slice `[Element]` is the
/// uniform spelling (element = `Element`); anything else is an opaque axis list.
fn axis_dim(ty: Type) -> AxisDim {
    match ty {
        Type::Slice(s) => AxisDim::Uniform(*s.elem),
        other => AxisDim::List(other),
    }
}

/// The constructor bound a [`Uniform`](AxisDim::Uniform) axis of `element`
/// repeated `size`-many times leaks — `Nat<size>: Repeat<element>`, spelled
/// through the [`ShapeIndex`] projection so no `Nat` alias need be in scope and
/// the already-normalized `Const<_>` size can be reused verbatim.
fn repeat_pred(size: &Type, element: &Type) -> TokenStream2 {
    let wa = wa();
    quote! { <#size as #wa::ShapeIndex>::Nat: #wa::Repeat<#element> }
}

/// Right-fold a non-empty slice of dims into the nested `DimSum` that nalgebra's
/// `hcat`/`vcat` produce: `[A, B, C]` → `DimSum<A, DimSum<B, C>>`.
fn dimsum(parts: &[Type]) -> TokenStream2 {
    let na = na();
    let mut iter = parts.iter().rev();
    let last = iter.next().expect("dimsum: empty slice");
    let mut acc = quote! { #last };
    for p in iter {
        acc = quote! { #na::DimSum<#p, #acc> };
    }
    acc
}

/// `whippyalgebra`, so emitted list-operation bounds resolve wherever the
/// attribute is used. Names the backend-agnostic items (`ShapeIndex`,
/// `Concatenated`, `Repeat`, …) that live at the crate root.
fn wa() -> TokenStream2 {
    quote! { ::whippyalgebra }
}

/// `whippyalgebra::nalgebra` — the adapter module — for the *backend-specific*
/// traits the attributes emit (`SquareDim`, `CountDim`), which are no longer
/// re-exported at the crate root.
fn wa_na() -> TokenStream2 {
    quote! { ::whippyalgebra::nalgebra }
}

/// The `i`-th block extent (`dims[i]`, e.g. `Const<N>`) as a type-level
/// `Unsigned`, via `ShapeIndex` — the same natural the block offsets and
/// `Take`/`Drop` are keyed on.
fn nat_size(dim: &Type) -> TokenStream2 {
    let wa = wa();
    quote! { <#dim as #wa::ShapeIndex>::Nat }
}

/// The right-folded concatenation of `lists[from..]`:
/// `Concatenated<L_from, Concatenated<…, L_last>>`. `from == last` is just the
/// single list; `from == 0` is the whole assembled list. This mirrors the
/// right-associative `vcat`/`hcat` that `block_matrix!` performs.
fn suffix(lists: &[Type], from: usize) -> TokenStream2 {
    let wa = wa();
    let k = lists.len();
    let last = &lists[k - 1];
    let mut acc = quote! { #last };
    for l in lists[from..k - 1].iter().rev() {
        acc = quote! { #wa::Concatenated<#l, #acc> };
    }
    acc
}

/// The offset of block `i` along an axis: the type-level sum of the preceding
/// block sizes (`Nat<0>` for the first). A sum of two-or-more summands needs
/// each `Add` node proven well-formed, so those bounds are pushed into `extra`.
fn offset(dims: &[Type], i: usize, extra: &mut Vec<TokenStream2>) -> TokenStream2 {
    let wa = wa();
    if i == 0 {
        return quote! { #wa::Nat<0> };
    }
    let mut acc = nat_size(&dims[i - 1]);
    for d in dims[..i - 1].iter().rev() {
        let s = nat_size(d);
        extra.push(quote! { #s: ::core::ops::Add<#acc> });
        acc = quote! { <#s as ::core::ops::Add<#acc>>::Output };
    }
    acc
}

/// The partition obligations for one axis, given its block sizes (`dims`) and
/// the sublists (`lists`) occupying each track. These are the facts a hand
/// `where` clause would otherwise carry:
///
/// - `L_i: DimList` and the assembled `Concatenated<…>: DimList`;
/// - the `Concat` chain that builds the concatenation (assembly, matching
///   `block_matrix!`'s `hcat`/`vcat`);
/// - the `Drop`-then-`Take` that slices each `L_i` back out at its offset
///   (disassembly, matching `unblock_matrix!`), pinned to the original `L_i`.
fn partition_preds(dims: &[Type], lists: &[Type]) -> Vec<TokenStream2> {
    let wa = wa();
    let k = lists.len();
    let mut out: Vec<TokenStream2> = Vec::new();

    let whole = suffix(lists, 0);
    out.push(quote! { #whole: #wa::DimList });
    for l in lists {
        out.push(quote! { #l: #wa::DimList });
    }

    // Assembly: each head concatenates onto the suffix of the tracks after it.
    for (i, l) in lists.iter().enumerate().take(k.saturating_sub(1)) {
        let suf1 = suffix(lists, i + 1);
        out.push(quote! { #l: #wa::Concat<#suf1> });
    }

    // Disassembly: drop to block `i`'s offset, then take its size, landing on
    // exactly `L_i`. `Drop<0>` is the identity, so the first block only needs
    // the `Take`.
    for (i, l) in lists.iter().enumerate() {
        let s_i = nat_size(&dims[i]);
        if i == 0 {
            out.push(quote! { #whole: #wa::Take<#s_i, Out = #l> });
        } else {
            let suf_i = suffix(lists, i);
            let off_i = offset(dims, i, &mut out);
            out.push(quote! { #whole: #wa::Drop<#off_i, Out = #suf_i> });
            out.push(quote! { #suf_i: #wa::Take<#s_i, Out = #l> });
        }
    }
    out
}

/// The Householder-reduction workspace obligations for a matrix of shape
/// `row × col`.
///
/// nalgebra's reduction-based decompositions — `svd`/`generalized_svd`,
/// `pseudo_inverse`, `eigenvalues`/`symmetric_eigen`/`schur`/`hessenberg`/
/// `symmetric_tridiagonalize`/`bidiagonalize`, `generalized_symmetric_eigen`,
/// and `col_piv_qr` — allocate Householder scratch whose sizes are the reduced
/// dimension `DimMinimum<row, col>` and its predecessor `DimDiff<…, U1>`, and
/// require the reducer `DimSub<U1>`. No shape fact implies these — whether a
/// body reaches for a reduction is invisible from the signature — so they are
/// emitted only under the `decompose` opt-in (the un-detectable counterpart of
/// the square-op bounds). The plain factorizations (`lu`/`full_piv_lu`/`qr`/
/// `cholesky`/`udu`) and the direct `solve`/`determinant`/`try_inverse` need
/// none of this: they ask only for `DimMin<D, Output = D>` + basic storage, which
/// the square-op detection already supplies, so they are never gated on
/// `decompose`.
///
/// When the shape is square (`row == col`) the reduced dimension is the axis
/// itself. This branch emits only the reducer `DimSub<U1>` and the four scratch
/// allocators (`Allocator<D, D>`, `Allocator<D>`, `Allocator<DimDiff<D, U1>>`,
/// `Allocator<D, DimDiff<D, U1>>`) — a superset of what any single square
/// reduction needs (Schur/Hessenberg/tridiagonal use three of the four; the
/// extra is harmless). It deliberately omits `DimMin<D, Output = D>`: the
/// reductions that also want it (`svd`/`col_piv_qr`/`generalized_*`) get it from
/// [`SquareDim`] / the detected square-block bound, so the two compose without
/// duplication. Otherwise it is the full rectangular pile keyed on
/// `DimMinimum<row, col>`, matching the `where` clause of the only genuinely
/// rectangular-storage reduction — `UniformUnitMatrix::pseudo_inverse` (and the
/// dimensionless inner pinv that `MixedUnitMatrix::generalized_pseudo_inverse`
/// whitens down to) — token-for-token.
fn decompose_preds(row: &TokenStream2, col: &TokenStream2, square: bool) -> Vec<TokenStream2> {
    let na = na();
    if square {
        let d = row;
        vec![
            quote! { #d: #na::DimSub<#na::U1> },
            quote! {
                #na::DefaultAllocator: #na::allocator::Allocator<#d, #d>
                    + #na::allocator::Allocator<#d>
                    + #na::allocator::Allocator<#na::DimDiff<#d, #na::U1>>
                    + #na::allocator::Allocator<#d, #na::DimDiff<#d, #na::U1>>
            },
        ]
    } else {
        let (r, c) = (row, col);
        let min = quote! { #na::DimMinimum<#r, #c> };
        vec![
            quote! { #r: #na::DimMin<#c> },
            quote! { #min: #na::DimSub<#na::U1> },
            quote! {
                #na::DefaultAllocator: #na::allocator::Allocator<#r, #c>
                    + #na::allocator::Allocator<#c>
                    + #na::allocator::Allocator<#r>
                    + #na::allocator::Allocator<#na::DimDiff<#min, #na::U1>>
                    + #na::allocator::Allocator<#min, #c>
                    + #na::allocator::Allocator<#r, #min>
                    + #na::allocator::Allocator<#min>
                    + #na::allocator::Allocator<#c, #r>
            },
        ]
    }
}

/// The exact obligation set for assembling `rows × cols` via right-associative
/// `hcat`/`vcat` (matching `block_matrix!`), the `ShapeIndex` shape bounds that
/// slicing needs, the partition bounds when the sublists are named, plus the
/// square-operation shape bounds when the assembly is detected square, and — under
/// the `decompose` opt-in — the SVD/eigen/QR workspace pile for the assembly. See
/// the `whippyalgebra` docs for the derivation.
// The obligation builders index `rows`/`cols` by position while also computing
// neighbouring suffix-sums (`w(j + 1)`, `v(i + 1)`), so the loop index is
// load-bearing and `enumerate()` would not simplify them.
#[allow(clippy::needless_range_loop)]
fn predicates(shape: &Shape, uniform: bool) -> Vec<WherePredicate> {
    let na = na();
    let (rows, cols) = (&shape.rows, &shape.cols);
    let (k, l) = (rows.len(), cols.len());
    let mut out: Vec<WherePredicate> = Vec::new();
    let mut push = |ts: TokenStream2| {
        out.push(syn::parse2(ts).expect("generic_block: generated an invalid where-predicate"));
    };

    // Column suffix-sum `W_j = C_j + … + C_{l-1}`; `W_0` is the total width.
    let w = |j: usize| dimsum(&cols[j..]);
    // Row suffix-sum `V_i = R_i + … + R_{k-1}`; `V_0` is the total height.
    let v = |i: usize| dimsum(&rows[i..]);

    // `hcat` widens `C_j` onto `W_{j+1}`; `vcat` stacks `R_i` onto `V_{i+1}`.
    // Those `DimAdd`s form the `DimSum` types below and are needed by the body.
    for j in 0..l.saturating_sub(1) {
        let (cj, wj1) = (&cols[j], w(j + 1));
        push(quote! { #cj: #na::DimAdd<#wj1> });
    }
    for i in 0..k.saturating_sub(1) {
        let (ri, vi1) = (&rows[i], v(i + 1));
        push(quote! { #ri: #na::DimAdd<#vi1> });
    }

    // Each row's `l-1` hcats allocate `R_i × W_j` (j = 0..l-1); the `k-1` vcats
    // allocate `V_i × W_0`.
    for i in 0..k {
        let ri = &rows[i];
        for j in 0..l.saturating_sub(1) {
            let wj = w(j);
            push(quote! { #na::DefaultAllocator: #na::allocator::Allocator<#ri, #wj> });
        }
    }
    if l >= 1 {
        let total_c = w(0);
        for i in 0..k.saturating_sub(1) {
            let vi = v(i);
            push(quote! { #na::DefaultAllocator: #na::allocator::Allocator<#vi, #total_c> });
        }
    }

    // Each distinct grid dimension must also be a usable compile-time
    // index/size (`ShapeIndex`). This is what lets the assembled grid be
    // *sliced back apart* (`unblock_matrix!`): the block offsets are formed as
    // `Nat<_>` (which only names given `Const<_>: ShapeIndex`) and `block_off`
    // requires it on every block size. Unlike the allocator/`DimAdd` bounds it
    // is a pure shape fact — derivable from the grid alone — so `generic_block`
    // owns it too, and an assemble→slice round-trip needs no separate
    // `ShapeIndex`/`SquareDim` on the caller. (Redundant with `SquareDim` when
    // that is also present; a repeated `where` predicate is harmless.)
    //
    // A *uniform* assembly needs none of the slicing plumbing below: it carries
    // no dimension lists (nothing to `Concat`/`Take`/`Drop`) and indexes at
    // runtime, so the assembly `DimAdd`/allocator pile above (plus the detected
    // square-op / `decompose` bounds) is the whole story.
    if !uniform {
        let mut seen: Vec<String> = Vec::new();
        for d in rows.iter().chain(cols.iter()) {
            let key = quote! { #d }.to_string();
            if !seen.contains(&key) {
                seen.push(key);
                push(quote! { #d: ::whippyalgebra::ShapeIndex });
            }
        }

        // Partition bounds: when the sublists occupying each grid track are
        // named, synthesize the `Concat`/`Take`/`Drop`/`DimList` facts that tie
        // those sublists to the shape — so an assemble→slice round-trip over
        // generic lists needs no hand-written partition `where` clause at all.
        if let Some(lists) = &shape.row_dims {
            for p in partition_preds(rows, lists) {
                push(p);
            }
        }
        if let Some(lists) = &shape.col_dims {
            for p in partition_preds(cols, lists) {
                push(p);
            }
        }

        // Cardinality tie: each named sublist has exactly its track's length as
        // a nalgebra `Const`. This is what lets list-sized construction —
        // `zeros![L, K]`, whose storage is `OMatrix<_, CountedDim<L>,
        // CountedDim<K>>` — resolve to the grid's shape in generic code (the Van
        // Loan zero blocks). It is redundant with the partition `Take`/`Drop`
        // above, which already pin each sublist's length, but restates it in the
        // `Dim` form `CountedDim` reads. Deduplicated, so a sublist shared
        // between a row and column track (a square block) states its tie once.
        let mut seen_cd: Vec<String> = Vec::new();
        let mut tie = |dims: &[Type], lists: &[Type], push: &mut dyn FnMut(TokenStream2)| {
            for (d, l) in dims.iter().zip(lists) {
                let key = quote! { #l : #d }.to_string();
                if !seen_cd.contains(&key) {
                    seen_cd.push(key);
                    let wa_na = wa_na();
                    push(quote! { #l: #wa_na::CountDim<Dim = #d> });
                }
            }
        };
        if let Some(lists) = &shape.row_dims {
            tie(rows, lists, &mut push);
        }
        if let Some(lists) = &shape.col_dims {
            tie(cols, lists, &mut push);
        }
    }

    // Square-operation *shape* bounds, *detected* (like `generic_matrix`'s
    // per-dim squareness) rather than flagged: the assembly is square exactly
    // when its row tracks and column tracks are the same sequence of dims, so
    // `Total` is one type on both axes (otherwise the assembled `OMatrix` is
    // rectangular-typed and `.exp()`/inverse/solve wouldn't typecheck). When
    // square, emit the vector workspace (`Allocator<Total>`) and
    // `Total: DimMin<Total, Output = Total>` — what the Van Loan `.exp()`, and
    // equally inverse/solve/determinant, need *of the shape*. This is only the
    // shape half; it does not assert the map is a *dimensional* endomorphism —
    // that comes from the consuming code's unit bound (e.g. `StateDot:
    // MapUnits<MulBy<Sec>, Out = State>`). The square storage itself is already
    // covered by the outermost vcat above.
    let square = rows.len() == cols.len()
        && rows
            .iter()
            .zip(cols)
            .all(|(r, c)| quote! { #r }.to_string() == quote! { #c }.to_string());
    if square {
        let total = w(0);
        push(quote! { #na::DefaultAllocator: #na::allocator::Allocator<#total> });
        push(quote! { #total: #na::DimMin<#total, Output = #total> });
    }

    // Decomposition workspace, under the `decompose` opt-in: the assembled matrix
    // is `V₀ × W₀` (total rows × total cols), and when the tracks match it is the
    // square `Total × Total`, whose `DimMin<Total, Output = Total>` the block above
    // already supplied. Emit the SVD/eigen/QR scratch for that assembled shape.
    if shape.decompose {
        for p in decompose_preds(&v(0), &w(0), square) {
            push(p);
        }
    }

    out
}

/// The union of every grid's obligations, deduplicated across grids: distinct
/// block matrices in one signature routinely share a dimension (`Const<N>:
/// ShapeIndex`, an allocator bound, …), and a `where` clause states each fact
/// once. Order is preserved (first occurrence wins).
fn all_predicates(shapes: &[Shape], uniform: bool) -> Vec<WherePredicate> {
    let mut out: Vec<WherePredicate> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for shape in shapes {
        for p in predicates(shape, uniform) {
            let key = quote! { #p }.to_string();
            if !seen.contains(&key) {
                seen.push(key);
                out.push(p);
            }
        }
    }
    out
}

/// Splice generated predicates into a function's `where` clause.
///
/// Works on both free functions and `impl` methods: it edits the token stream's
/// `Signature`, which both share.
fn splice(mut generics: syn::Generics, preds: Vec<WherePredicate>) -> syn::Generics {
    let wc = generics.make_where_clause();
    for p in preds {
        wc.predicates.push(p);
    }
    generics
}

/// Writes the obligations for assembling a partitioned matrix in generic code
/// into the annotated function's `where` clause, so a generic block assembly
/// (`block_matrix!`) into a static `SMatrix` compiles without transcribing the
/// storage (and matrix-exponential) bounds by hand.
///
/// # Syntax
///
/// ```ignore
/// #[generic_block(rows(N, State; M, Input),   // size, sublist per track (`;`-separated)
///                 cols(N, State; M, Input))]
/// fn discretize<const N: usize, const M: usize, /* … */>(/* … */) -> /* … */
/// where /* only the problem's own, non-partition bounds */
/// { /* block_matrix![[a, b], [zeros, zeros]].exp() … then unblock_matrix! … */ }
/// ```
///
/// For a function that works with several block matrices, wrap each grid in
/// a `block(..)` group; the obligations of every grid are emitted, deduplicated
/// across grids (a shared dimension states its `ShapeIndex`/allocator bound
/// once):
///
/// ```ignore
/// #[generic_block(
///     block(rows(N, RowA; M, RowB), cols(N, ColA; M, ColB)),
///     block(rows(M, RowB; N, RowA), cols(M, ColB; N, ColA)),
/// )]
/// fn two_grids</* … */>(/* … */) -> /* … */ { /* two block_matrix! assemblies */ }
/// ```
///
/// A leading `uniform` keyword targets a `UniformUnitMatrix` assembly
/// (`block_matrix![uniform; …]`): the blocks share one unit and carry no
/// dimension lists, so only the nalgebra-level assembly pile is emitted — the
/// `hcat`/`vcat` `DimAdd`s and storage allocators, plus the detected square-op /
/// `decompose` bounds — with no `ShapeIndex`, partition, or `CountDim`
/// plumbing (naming a sublist is then an error). Bare sizes only:
///
/// ```ignore
/// #[generic_block(uniform, rows(N; M), cols(N; M))]   // uniform (N+M) × (N+M) assembly
/// fn assemble_uniform<const N: usize, const M: usize>(/* … UniformUnitMatrix blocks … */) { /* block_matrix![uniform; …] */ }
/// ```
///
/// - `rows(..)` / `cols(..)` (required): each axis is a `;`-separated list of
///   tracks — one per grid row / column — and each track is a block size
///   optionally paired, after a `,`, with the dimension sublist occupying it
///   (`N, State`). A bare const-generic parameter (`N`) is wrapped in `Const<N>`;
///   an explicit `Const<N>` / `nalgebra::Const<N>` is taken as written. Any
///   `k × l` grid is accepted — the obligation set is derived from the shape, so
///   there is no fixed arity cap. The block sizes must be const generics: they
///   key the storage-allocator bounds to the signature's `SMatrix<f64, N, M>` and
///   supply `unblock_matrix!`'s block sizes (which can't be counted from a list
///   without `generic_const_exprs`).
///
///   Naming the sublists (all-or-nothing per axis) makes the attribute
///   synthesize the whole partition `where` clause (`Concat` for assembly,
///   `Take`/`Drop` for the reblock, the `DimList`s, and the `CountDim` length
///   ties that let `zeros!` size to the grid), so an assemble→slice round-trip
///   over generic lists needs no hand-written dim-list bounds. Give sizes only
///   (`rows(N; M)`) to emit just the shape/allocator bounds (assembly-only, or
///   when you'd rather write the partition facts yourself).
///
/// Squareness is detected, not flagged: when the `rows` and `cols` track
/// sequences are equal the assembly is a square `Total × Total` (the same
/// `DimSum` type on both axes), and the square-operation shape bounds
/// (`Allocator<Total>` and `Total: DimMin<Total>`) are emitted automatically —
/// exactly what `block_matrix![…].exp()` (Van Loan discretization) and equally
/// inverse/solve/determinant on the assembled square require of the shape.
/// This is only the shape half: it does not assert the map is a
/// dimensional endomorphism (that `.exp()` is unit-meaningful) — that follows
/// separately, from the unit bound the consuming code carries (e.g.
/// `StateDot: MapUnits<MulBy<Sec>, Out = State>` in the discretization).
///
/// The emitted predicates are:
///
/// - the assembly obligations of right-associative `hcat`/`vcat` — per-row
///   `Allocator<Rᵢ, Cⱼ+…>` and per-row-suffix `Allocator<Rᵢ+…, TotalC>`, with
///   the `DimAdd`s that form them;
/// - the slicing obligation `dim: ShapeIndex` for each distinct grid
///   dimension, so the same grid can be read back apart with
///   [`unblock_matrix!`](../whippyalgebra/macro.unblock_matrix.html) (whose
///   `Nat<_>` block offsets require it) with no extra `ShapeIndex`/`SquareDim`
///   on the caller;
/// - the partition obligations, when the sublists are named — the
///   `Concat`/`Take`/`Drop`/`DimList` facts that tie each sublist to its track
///   (offsets are type-level sums of the preceding sizes, so any arity works),
///   plus a `CountDim<Dim = _>` per sublist tying its length to the track's
///   shape (redundant with `Take`/`Drop`, but in the form `zeros!` reads);
/// - the square-operation shape bounds `Allocator<Total>` and
///   `Total: DimMin<Total>`, when the assembly is detected square (`rows` track
///   sequence == `cols`);
/// - under the `decompose` opt-in, the Householder-reduction workspace
///   pile for the assembled `Total` matrix (the `DimSub<U1>` reducer and the
///   `DimMinimum`/`DimDiff` scratch allocators). This is what a body needs to
///   `svd`/`pseudo_inverse`/`eigenvalues`/`schur`/… or `col_piv_qr` the
///   assembly — but not what plain `lu`/`qr`/`cholesky`/`solve`/`determinant`/
///   inverse need (those ride on the detected square-op bounds). It is emitted
///   only when flagged — reducing the assembly is a body fact, not a shape one —
///   and composes with the detected square-op bounds: the square branch omits
///   `Total: DimMin<Total>` (already emitted above) and contributes only the
///   reducer and scratch.
///
/// The `decompose` flag is a bare keyword among the keys, e.g.
/// `#[generic_block(rows(N, RowA; M, RowB), cols(N, RowA; M, RowB), decompose)]`.
///
/// All are placed directly in the `where` clause (no helper trait, no
/// supertrait elaboration).
#[proc_macro_attribute]
pub fn generic_block(attr: TokenStream, item: TokenStream) -> TokenStream {
    let shapes = syn::parse_macro_input!(attr as Shapes);
    let preds = all_predicates(&shapes.shapes, shapes.uniform);
    let item2: TokenStream2 = item.into();

    // A free `fn` parses as `ItemFn`; an `impl` method as `ImplItemFn`. Both
    // carry the `where` clause on `sig.generics`.
    if let Ok(mut f) = syn::parse2::<syn::ItemFn>(item2.clone()) {
        f.sig.generics = splice(f.sig.generics, preds);
        return quote! { #f }.into();
    }
    match syn::parse2::<syn::ImplItemFn>(item2) {
        Ok(mut f) => {
            f.sig.generics = splice(f.sig.generics, preds);
            quote! { #f }.into()
        }
        Err(e) => e.to_compile_error().into(),
    }
}

/// One declared matrix: its `(RowDim, ColDim)` size pair, the (optional)
/// dimension sublist on each axis (a written-out list or the uniform `[Element]`
/// spelling), and whether the `decompose` opt-in was set on it (also emit the
/// SVD/eigen/QR workspace pile).
struct MatrixShape {
    row: Type,
    col: Type,
    row_dim: Option<AxisDim>,
    col_dim: Option<AxisDim>,
    decompose: bool,
}

/// What an item's generic matrices look like: the per-matrix `shapes` (size pair,
/// per-axis dims, and `decompose` flag). The distinct `DimList`/`Repeat` facts
/// the dims carry are derived (and deduplicated) in [`matrix_predicates`].
///
/// A leading `uniform` keyword switches to the [`UniformUnitMatrix`] spec: the
/// matrices carry one unit and no dimension lists, so the emitted pile is purely
/// nalgebra-level — `SquareDim` on square axes and the `decompose` scratch, but
/// no `ShapeIndex` on rectangular axes and no `DimList` (naming a sublist
/// is then an error).
struct MatrixSpec {
    uniform: bool,
    shapes: Vec<MatrixShape>,
}

/// Parse one matrix axis `(size[, dims])`: a block size — a bare const-generic
/// param (wrapped in `Const<_>`) or an explicit dim — optionally paired, after a
/// `,`, with the dimension list on that axis. The list is either a written-out
/// axis type (`State`, `Readings<N>`) or the uniform spelling `[Element]`,
/// meaning `N` copies of `Element`. Unlike `generic_block`'s tracks a matrix axis
/// is a single size (no `;`).
fn parse_axis(input: ParseStream) -> syn::Result<(Type, Option<AxisDim>)> {
    let content;
    parenthesized!(content in input);
    let size = norm_dim(content.parse()?);
    let dim = if content.peek(Token![,]) {
        content.parse::<Token![,]>()?;
        Some(axis_dim(content.parse()?))
    } else {
        None
    };
    if !content.is_empty() {
        return Err(content.error("expected `(size)`, `(size, dims)`, or `(size, [Element])`"));
    }
    Ok((size, dim))
}

/// Parse one matrix declaration `rows(..), cols(..)` (either order), optionally
/// followed by `, decompose`, into its size pair, per-axis dims, and `decompose`
/// flag.
fn parse_matrix_decl(input: ParseStream) -> syn::Result<MatrixShape> {
    let mut row: Option<(Type, Option<AxisDim>)> = None;
    let mut col: Option<(Type, Option<AxisDim>)> = None;
    let mut decompose = false;
    while !input.is_empty() {
        let key: Ident = input.parse()?;
        match key.to_string().as_str() {
            "rows" => row = Some(parse_axis(input)?),
            "cols" => col = Some(parse_axis(input)?),
            "decompose" => decompose = true,
            other => {
                return Err(syn::Error::new(
                    key.span(),
                    format!("unexpected `{other}`; expected `rows`, `cols`, or `decompose`"),
                ))
            }
        }
        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
        }
    }
    let (rs, rd) = row
        .ok_or_else(|| syn::Error::new(proc_macro2::Span::call_site(), "missing `rows(...)`"))?;
    let (cs, cd) = col
        .ok_or_else(|| syn::Error::new(proc_macro2::Span::call_site(), "missing `cols(...)`"))?;
    Ok(MatrixShape {
        row: rs,
        col: cs,
        row_dim: rd,
        col_dim: cd,
        decompose,
    })
}

impl Parse for MatrixSpec {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut shapes: Vec<MatrixShape> = Vec::new();
        // An optional leading `uniform` keyword selects the single-unit spec.
        let uniform = {
            let fork = input.fork();
            fork.parse::<Ident>().map(|i| i == "uniform").unwrap_or(false)
        };
        if uniform {
            input.parse::<Ident>()?; // consume `uniform`
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }
        // Multi-matrix form leads with `matrix`; otherwise the whole attribute is
        // one bare `rows(..), cols(..)` declaration.
        let is_matrix = {
            let fork = input.fork();
            fork.parse::<Ident>().map(|i| i == "matrix").unwrap_or(false)
        };
        if is_matrix {
            while !input.is_empty() {
                let kw: Ident = input.parse()?;
                if kw != "matrix" {
                    return Err(syn::Error::new(
                        kw.span(),
                        "expected `matrix(...)`; a `matrix(...)` group cannot be \
                         mixed with a bare `rows(...)` declaration",
                    ));
                }
                let content;
                parenthesized!(content in input);
                shapes.push(parse_matrix_decl(&content)?);
                if input.peek(Token![,]) {
                    input.parse::<Token![,]>()?;
                }
            }
        } else {
            shapes.push(parse_matrix_decl(input)?);
        }
        if shapes.is_empty() {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "generic_matrix: expected at least one matrix (`rows(..), cols(..)`)",
            ));
        }
        // A uniform matrix carries a single unit and no dimension lists, so a
        // named sublist (`rows(N, State)` / `rows(N, [Element])`) is meaningless
        // here — reject it rather than silently ignore it.
        if uniform {
            for s in &shapes {
                if s.row_dim.is_some() || s.col_dim.is_some() {
                    return Err(syn::Error::new(
                        proc_macro2::Span::call_site(),
                        "generic_matrix(uniform, ..): a uniform matrix has no dimension \
                         lists — drop the sublist and write bare sizes, e.g. \
                         `uniform, rows(N), cols(M)`",
                    ));
                }
            }
        }
        Ok(MatrixSpec { uniform, shapes })
    }
}

/// The well-formedness a set of matrix sizes (and dims) needs:
///
/// - from the sizes: a dim used as both axes of some matrix is square and
///   gets [`SquareDim`]; a dim only ever seen on one axis gets [`ShapeIndex`]
///   (subsumed by `SquareDim`, so squares skip it);
/// - from the dims: each dimension list gets [`DimList`] — the generic "this is
///   a valid matrix axis" fact, safe to emit because it is a property of being
///   a matrix axis, not of the problem.
///
/// Deduplicated.
fn matrix_predicates(spec: &MatrixSpec) -> Vec<WherePredicate> {
    let wa = wa();
    let mut out: Vec<WherePredicate> = Vec::new();
    let mut push = |ts: TokenStream2| {
        out.push(syn::parse2(ts).expect("generic_matrix: generated an invalid where-predicate"));
    };

    // First pass: any dim appearing as both axes of a size is square.
    let mut square: Vec<String> = Vec::new();
    for s in &spec.shapes {
        let (r, c) = (&s.row, &s.col);
        if quote! { #r }.to_string() == quote! { #c }.to_string() {
            let key = quote! { #r }.to_string();
            if !square.contains(&key) {
                square.push(key);
                let wa_na = wa_na();
                push(quote! { #r: #wa_na::SquareDim });
            }
        }
    }
    // Second pass: every other dim only needs to index/slice — a *mixed*-only
    // fact. A uniform matrix indexes at runtime and carries no dimension lists,
    // so its rectangular axes need nothing here (a uniform `Const × Const` is
    // well-formed via nalgebra's blanket allocator).
    if !spec.uniform {
        let mut seen = square.clone();
        for s in &spec.shapes {
            for d in [&s.row, &s.col] {
                let key = quote! { #d }.to_string();
                if !seen.contains(&key) {
                    seen.push(key);
                    push(quote! { #d: #wa::ShapeIndex });
                }
            }
        }
    }
    // Dims: each axis's sublist fact, keyed to that axis's size. A written-out
    // list only needs the generic `DimList` well-formedness; a uniform `[E]`
    // axis instead leaks the constructor bound `Nat<size>: Repeat<E>` that makes
    // the generic `Repeated<E, size>` axis nameable (the hand-written bound a
    // `Repeated<_, N>` axis used to force). Both are deduplicated. (In `uniform`
    // mode there are no sublists — the parser rejects them — so this is empty.)
    let mut seen_dim: Vec<String> = Vec::new();
    for s in &spec.shapes {
        for (size, dim) in [(&s.row, &s.row_dim), (&s.col, &s.col_dim)] {
            let pred = match dim {
                Some(AxisDim::List(l)) => quote! { #l: #wa::DimList },
                Some(AxisDim::Uniform(e)) => repeat_pred(size, e),
                None => continue,
            };
            let key = pred.to_string();
            if !seen_dim.contains(&key) {
                seen_dim.push(key);
                push(pred);
            }
        }
    }
    // Decomposition workspace, under the per-matrix `decompose` opt-in. A square
    // matrix already carries `SquareDim` (hence `DimMin<Self, Output = Self>`)
    // from the first pass; the pile adds the reducer and Householder scratch.
    // Deduplicated across matrices that share a shape.
    let mut seen_dec: Vec<String> = Vec::new();
    for s in &spec.shapes {
        if !s.decompose {
            continue;
        }
        let (r, c) = (&s.row, &s.col);
        let (row_ts, col_ts) = (quote! { #r }, quote! { #c });
        let is_square = square.contains(&row_ts.to_string());
        for p in decompose_preds(&row_ts, &col_ts, is_square) {
            let key = p.to_string();
            if !seen_dec.contains(&key) {
                seen_dec.push(key);
                push(p);
            }
        }
    }
    out
}

/// Writes the shape well-formedness of a set of matrices — declared as
/// `rows(size, dims) / cols(size, dims)` — into an item's `where` clause: the
/// "I'm going to have generic matrices of these sizes" plumbing, and nothing
/// else.
///
/// From the sizes it emits, for every distinct dim named:
/// [`SquareDim`](../whippyalgebra/trait.SquareDim.html) when that dim is used as
/// both axes of some matrix (so it's square and LU-ready — what
/// `solve`/`determinant`/`try_inverse` need), and
/// [`ShapeIndex`](../whippyalgebra/trait.ShapeIndex.html) when it is only ever
/// rectangular (enough to index/slice). Square-ness is detected from the sizes
/// — you list the matrix shapes, not which dims are square. A bare const-generic
/// parameter (`N`) is wrapped in `Const<N>`; an explicit `Const<N>` is kept.
///
/// From the dims it emits, for each axis whose sublist is named alongside
/// its size (`rows(N, State)`), one of two facts depending on the spelling:
///
/// - a written-out axis type (`State`, `Readings<N>`, …) gets
///   [`DimList`](../whippyalgebra/trait.DimList.html) — the generic "this is a
///   valid matrix axis" fact;
/// - the uniform spelling `[Element]` (a slice type) declares the axis to be
///   `N` copies of the single unit `Element` —
///   [`Repeated<Element, N>`](../whippyalgebra/type.Repeated.html), with `N` the
///   axis's own size — and emits the constructor bound `Nat<N>:
///   Repeat<Element>` (spelled `<Const<N> as ShapeIndex>::Nat: Repeat<Element>`).
///   That is exactly the bound a generic `Repeated<_, N>` axis leaks — the one a
///   uniform-axis function used to have to hand-write — so naming the element
///   here retires it. (This is a `generic_matrix`-only spelling: a
///   `#[generic_block]` track cannot be uniform, because its disassembly
///   obligations would have to normalize the `Repeated<…>` projection at a
///   generic length, which the trait solver cannot do.)
///
/// The size/dim pairing isn't yet checked against the body, but keeps each
/// declaration self-describing.
///
/// A per-matrix `decompose` opt-in (`rows(..), cols(..), decompose` — or
/// inside a `matrix(..)` group) additionally emits that matrix's
/// Householder-reduction workspace pile: the reducer `DimSub<U1>` and
/// nalgebra's reduced-dimension (`DimMinimum`/`DimDiff<…, U1>`) scratch
/// allocators, keyed on `DimMinimum<Row, Col>` when rectangular and collapsing to
/// the axis when square. This is exactly what the reduction-based decompositions
/// need — `svd`/`generalized_svd`, `pseudo_inverse`, `eigenvalues`/
/// `symmetric_eigen`/`schur`/`hessenberg`/`symmetric_tridiagonalize`/
/// `bidiagonalize`, `generalized_symmetric_eigen`, and `col_piv_qr` — and
/// nothing more: plain `lu`/`full_piv_lu`/`qr`/`cholesky`/`udu` and the direct
/// `solve`/`determinant`/`try_inverse` ride on the square-op bounds `SquareDim`
/// already supplies and must not be flagged. The square branch omits
/// `DimMin<Self, Output = Self>` (already carried by `SquareDim`) and so composes
/// with it. Unlike squareness this can't be detected — whether a body reduces is
/// invisible from the size — so it's flagged, and flagged only on the matrices
/// that are actually reduced (the others stay free of the pile).
///
/// It deliberately emits no unit-algebra bounds (`MapUnits`, the
/// reciprocal/`Dual` involution, …): those are problem-specific, not a property
/// of the size, so they stay in the hand-written `where` clause.
///
/// # Syntax
///
/// A single matrix is a bare `rows(..), cols(..)`:
///
/// ```ignore
/// #[generic_matrix(rows(N, State), cols(M, Input))]   // one N × M matrix
/// impl<const N: usize, const M: usize, /* … */> Foo<N, M> { /* … */ }
/// ```
///
/// A uniform axis names its element in brackets, so the attribute writes the
/// `Nat<_>: Repeat<_>` constructor bound for you (no hand-written `where`):
///
/// ```ignore
/// #[generic_matrix(rows(N, [Reading]), cols(M, [Source]), decompose)]  // [V; N] × [A; M]
/// fn solve_currents_mixed<const N: usize, const M: usize>(/* … */) -> /* … */ { /* generalized_pseudo_inverse … */ }
/// ```
///
/// A leading `uniform` keyword targets `UniformUnitMatrix` instead: those
/// matrices carry one unit and no dimension lists, so the emitted pile is
/// purely nalgebra-level — `SquareDim` on any square axis and, under
/// `decompose`, the reduction scratch — with no `ShapeIndex` on rectangular
/// axes (uniform matrices index at runtime) and no `DimList` (naming a
/// sublist is then an error). Bare sizes only:
///
/// ```ignore
/// #[generic_matrix(uniform, rows(N), cols(M), decompose)]  // uniform Ω, pinv-ready
/// fn solve_currents<const N: usize, const M: usize>(/* … */) -> /* … */ { /* pseudo_inverse … */ }
/// ```
///
/// Several matrices each go in a `matrix(..)` group; sizes and dims are unioned
/// (deduplicated) across them:
///
/// ```ignore
/// #[generic_matrix(
///     matrix(rows(N, State), cols(N, State)),            // square N × N  -> Const<N>: SquareDim
///     matrix(rows(N, State), cols(M, Input), decompose), // rectangular N × M, reduction-ready
///     matrix(rows(M, Input), cols(M, Input)),            // square M × M  -> Const<M>: SquareDim
/// )]                                                      // dims -> State/Input: DimList
/// impl<const N: usize, const M: usize, /* … */> Foo<N, M>
/// where /* only the problem's unit-algebra bounds */
/// { /* … solve / try_inverse / block on N×N, M×M; generalized_pseudo_inverse on N×M … */ }
/// ```
///
/// Works on an `impl` block (the usual place — the sizes serve every method), a
/// free `fn`, or an `impl` method.
#[proc_macro_attribute]
pub fn generic_matrix(attr: TokenStream, item: TokenStream) -> TokenStream {
    let spec = syn::parse_macro_input!(attr as MatrixSpec);
    let preds = matrix_predicates(&spec);
    let item2: TokenStream2 = item.into();

    // An `impl` block carries its `where` clause on `generics`; a free `fn` and
    // an `impl` method carry theirs on `sig.generics`.
    if let Ok(mut i) = syn::parse2::<syn::ItemImpl>(item2.clone()) {
        i.generics = splice(i.generics, preds);
        return quote! { #i }.into();
    }
    if let Ok(mut f) = syn::parse2::<syn::ItemFn>(item2.clone()) {
        f.sig.generics = splice(f.sig.generics, preds);
        return quote! { #f }.into();
    }
    match syn::parse2::<syn::ImplItemFn>(item2) {
        Ok(mut f) => {
            f.sig.generics = splice(f.sig.generics, preds);
            quote! { #f }.into()
        }
        Err(e) => e.to_compile_error().into(),
    }
}
